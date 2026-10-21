"""H3LipSyncWav2Lip - phoneme-level mouth re-synthesis with Wav2Lip.

The in-model route (H3 audio conditioning) cannot hold a face AND follow the
soundtrack at the same time, and expression transfer (LivePortrait) only copies
motion that already exists.  Wav2Lip rewrites the mouth from the audio itself,
frame by frame, and leaves every other pixel alone - so identity is exact and
the lips land on the phonemes.

Runs the reference `inference.py` in its own interpreter (same pattern as
H3LipSyncTransfer), so the ComfyUI process never imports Wav2Lip's old deps.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch

from .nodes_audio_drive import H3_FPS, _shift_audio

DEFAULT_PYTHON = r"G:\lipsync-tools\venv-latentsync\Scripts\python.exe"
DEFAULT_REPO = r"G:\lipsync-tools\Wav2Lip"
DEFAULT_FFMPEG_DIR = r"F:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin"

CHECKPOINTS = ["wav2lip_gan.pth", "wav2lip.pth"]


def _frames_to_uint8(images):
    """IMAGE tensor -> uint8 numpy array [N,H,W,C]."""
    x = images
    if x.ndim == 5:  # [B,T,H,W,C]
        x = x.reshape(-1, *x.shape[-3:])
    if x.ndim != 4:
        raise ValueError("H3LipSyncWav2Lip: expected an IMAGE batch, got shape %s" % (tuple(x.shape),))
    x = x.detach().float().clamp(0.0, 1.0).cpu().numpy()
    return (x * 255.0).round().astype(np.uint8)


def _write_video(frames, path, fps):
    from PIL import Image

    n, h, w, _ = frames.shape
    tmp = tempfile.mkdtemp(prefix="w2l-frames-")
    try:
        for i in range(n):
            Image.fromarray(frames[i]).save(os.path.join(tmp, "%06d.png" % i))
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg, "-v", "error", "-y", "-framerate", "%.6f" % fps,
               "-i", os.path.join(tmp, "%06d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", path]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _frame_count(path):
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        n = 0
        for _ in reader:
            n += 1
    finally:
        reader.close()
    return n


def _read_video(path):
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        frames = [np.asarray(f)[..., :3] for f in reader]
    finally:
        reader.close()
    if not frames:
        raise RuntimeError("H3LipSyncWav2Lip: %r has no frames" % path)
    arr = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _as_audio(audio):
    """Accept a ComfyUI AUDIO dict, a LazyAudioMap, a (waveform, rate) pair, or a tensor."""
    w = sr = None
    if isinstance(audio, dict):
        w, sr = audio.get("waveform"), audio.get("sample_rate")
    else:
        for getter in (lambda k: audio[k], lambda k: getattr(audio, k)):
            try:
                w = getter("waveform")
                sr = getter("sample_rate")
                break
            except Exception:
                w = sr = None
    if w is not None and sr:
        return w, int(sr)
    if isinstance(audio, (tuple, list)) and len(audio) == 2:
        return audio[0], int(audio[1])
    if torch.is_tensor(audio):
        return audio, 16000
    raise ValueError("H3LipSyncWav2Lip: could not read the AUDIO input (%r)." % type(audio).__name__)


def _write_wav(waveform, sr, path):
    w = waveform
    if torch.is_tensor(w):
        w = w.detach().float().cpu().numpy()
    w = np.asarray(w, dtype=np.float32)
    if w.ndim == 3:      # [B, C, T]
        w = w[0]
    if w.ndim == 2:      # [C, T] -> mono
        w = w.mean(axis=0)
    w = np.clip(w, -1.0, 1.0)
    sf.write(path, w, int(sr), subtype="FLOAT")


class H3LipSyncWav2Lip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Frames to re-synthesise. Only the mouth is replaced; "
                               "face, hair and body pixels are kept.",
                }),
                "audio": ("AUDIO", {
                    "tooltip": "The soundtrack to lip-sync to. Returned (optionally shifted) for the mux.",
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.001}),
                "checkpoint": (CHECKPOINTS, {
                    "default": "wav2lip_gan.pth",
                    "tooltip": "wav2lip_gan = sharper picture; wav2lip = slightly tighter lipsync "
                               "and a softer mouth (both measured on this machine).",
                }),
                "pads": ("STRING", {
                    "default": "0 10 0 0",
                    "tooltip": "Extra crop around the detected face: top bottom left right, in pixels.",
                }),
                "resize_factor": ("INT", {
                    "default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Downscale before processing (1 = full resolution). 2 is faster on big frames.",
                }),
                "nosmooth": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Disable the temporal smoothing of the detected face box.",
                }),
            },
            "optional": {
                "audio_offset_frames": ("INT", {
                    "default": 0, "min": -48, "max": 48, "step": 1,
                    "tooltip": "Shifts the returned audio against the frames (same meaning as the "
                               "driving nodes). Leave 0 to lip-sync in place.",
                }),
                "python_exe": ("STRING", {
                    "default": DEFAULT_PYTHON,
                    "tooltip": "Interpreter that can import Wav2Lip's deps (librosa, torch, face_detection).",
                }),
                "wav2lip_dir": ("STRING", {
                    "default": DEFAULT_REPO,
                    "tooltip": "Wav2Lip checkout containing inference.py and checkpoints/.",
                }),
                "ffmpeg_dir": ("STRING", {
                    "default": DEFAULT_FFMPEG_DIR,
                    "tooltip": "Directory prepended to PATH; Wav2Lip shells out to ffmpeg for audio.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "audio", "report")
    FUNCTION = "lipsync"
    CATEGORY = "minimax/h3/audio"

    def lipsync(self, images, audio, fps, checkpoint, pads, resize_factor, nosmooth,
                audio_offset_frames=0, python_exe=DEFAULT_PYTHON, wav2lip_dir=DEFAULT_REPO,
                ffmpeg_dir=DEFAULT_FFMPEG_DIR):
        frames = _frames_to_uint8(images)
        n_in = frames.shape[0]
        if n_in < 2:
            raise ValueError("H3LipSyncWav2Lip: need at least 2 frames, got %d." % n_in)

        if not os.path.exists(os.path.join(wav2lip_dir, "inference.py")):
            raise FileNotFoundError(
                "H3LipSyncWav2Lip: no inference.py under %r - set wav2lip_dir." % wav2lip_dir)
        ckpt_path = os.path.join(wav2lip_dir, "checkpoints", checkpoint)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                "H3LipSyncWav2Lip: missing %r - put the Wav2Lip checkpoint there." % ckpt_path)
        s3fd = os.path.join(wav2lip_dir, "face_detection", "detection", "sfd", "s3fd.pth")
        if not os.path.exists(s3fd):
            raise FileNotFoundError(
                "H3LipSyncWav2Lip: missing the face detector %r (s3fd.pth)." % s3fd)

        w, sr = _as_audio(audio)
        shifts = int(audio_offset_frames or 0)

        work = tempfile.mkdtemp(prefix="h3ls-w2l-")
        face_mp4 = os.path.join(work, "faces.mp4")
        speech_wav = os.path.join(work, "speech.wav")
        out_mp4 = os.path.join(work, "out.mp4")
        try:
            _write_video(frames, face_mp4, fps)
            # Synthesise against the ORIGINAL track: Wav2Lip rewrites the mouth from
            # this audio, so shifting it first would move the lips and the sound
            # together and the knob would do nothing.  The shift below only moves
            # the track that is handed back for the mux.
            _write_wav(w, sr, speech_wav)

            cmd = [python_exe, "inference.py",
                   "--checkpoint_path", os.path.join("checkpoints", checkpoint),
                   "--face", face_mp4,
                   "--audio", speech_wav,
                   "--outfile", out_mp4,
                   "--pads"] + [p for p in str(pads).replace(",", " ").split() if p] + [
                   "--resize_factor", str(int(resize_factor))]
            if nosmooth:
                cmd.append("--nosmooth")

            env = dict(os.environ)
            env.setdefault("PYTHONIOENCODING", "utf-8")
            if ffmpeg_dir and os.path.isdir(ffmpeg_dir):
                env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
            proc = subprocess.run(cmd, cwd=wav2lip_dir, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", env=env, timeout=3600)
            if proc.returncode != 0 or not os.path.exists(out_mp4):
                tail = ((proc.stderr or "") + (proc.stdout or ""))[-1200:]
                raise RuntimeError("H3LipSyncWav2Lip: Wav2Lip failed (exit %d)\n%s"
                                   % (proc.returncode, tail))

            result = _read_video(out_mp4)
            n_out = int(result.shape[0])
            if n_out != n_in:
                # Wav2Lip follows the audio length; it repeats/pads frames when the
                # soundtrack is longer than the picture.  Keep the picture length so
                # downstream graphs still line up 1:1 with the source frames.
                if n_out > n_in:
                    result = result[:n_in]
                    note = ("output %d frames, trimmed back to %d (audio was longer than the "
                            "picture - trim the audio first for an exact tail)" % (n_out, n_in))
                else:
                    pad = result[-1:].repeat(n_in - n_out, 1, 1, 1)
                    result = torch.cat([result, pad], dim=0)
                    note = "output %d frames, last frame repeated to %d" % (n_out, n_in)
            else:
                note = "output matches input length exactly"
        finally:
            shutil.rmtree(work, ignore_errors=True)

        seconds = result.shape[0] / float(fps)
        report = "\n".join([
            "H3 Lip Sync Wav2Lip",
            "input frames     : %d (%.3fs @ %.3f fps)" % (n_in, n_in / float(fps), fps),
            "checkpoint       : %s" % checkpoint,
            "pads / resize    : %s / %d" % (pads, resize_factor),
            "result frames    : %d (%.3fs)" % (result.shape[0], seconds),
            "length note      : %s" % note,
            "audio shift      : %+d frames (%+.3fs, mux track only)" % (shifts, shifts / float(H3_FPS)),
            "mux tip          : frames -> CreateVideo, audio -> CreateVideo.audio",
        ])
        print(report, flush=True)
        out_audio = {"waveform": _shift_audio(w, sr, shifts), "sample_rate": sr}
        return (result, out_audio, report)


NODE_CLASS_MAPPINGS = {"H3LipSyncWav2Lip": H3LipSyncWav2Lip}
