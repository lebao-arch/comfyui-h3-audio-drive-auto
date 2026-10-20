"""H3 Lip Sync Transfer - post-hoc expression transfer.

H3 can keep one subject for a whole shot, or follow the soundtrack, but not
both (see H3AudioDriveLong). This node closes the gap: take the stable render as
the *source* and the audio-driven render as the *driving* video, and let
LivePortrait warp the driving performance (mouth included) onto the stable face.

It shells out to a local LivePortrait checkout because that pipeline owns its own
model graph, config paths and ffmpeg I/O; running it as a subprocess keeps this
package dependency-free and keeps the ComfyUI process out of its CUDA lifetime.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch

from .nodes_audio_drive import H3_FPS, _shift_audio

DEFAULT_PYTHON = r"G:\ComfyUI-aki-v3.2\ComfyUI-aki-v3.2\python\python.exe"
DEFAULT_REPO = r"G:\lipsync-tools\LivePortrait"


def _frames_to_uint8(images):
    """IMAGE tensor -> uint8 numpy array [N,H,W,C]."""
    x = images
    if x.ndim == 5:  # [B,T,H,W,C]
        x = x.reshape(-1, *x.shape[-3:])
    if x.ndim != 4:
        raise ValueError("H3LipSyncTransfer: expected an IMAGE batch, got shape %s" % (tuple(x.shape),))
    x = x.detach().float().clamp(0.0, 1.0).cpu().numpy()
    return (x * 255.0).round().astype(np.uint8)


def _write_video(frames, path, fps):
    from PIL import Image

    n, h, w, _ = frames.shape
    tmp = tempfile.mkdtemp(prefix="h3ls-frames-")
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


def _read_video(path):
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    try:
        frames = [np.asarray(f)[..., :3] for f in reader]
    finally:
        reader.close()
    arr = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _as_audio(audio):
    """Accept a ComfyUI AUDIO dict, a LazyAudioMap, a (waveform, rate) pair, or a tensor."""
    w = sr = None
    if isinstance(audio, dict):
        w, sr = audio.get("waveform"), audio.get("sample_rate")
    else:  # LazyAudioMap and other mapping-like wrappers
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
    raise ValueError("H3LipSyncTransfer: could not read the AUDIO input (%r)." % type(audio).__name__)


class H3LipSyncTransfer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_images": ("IMAGE", {
                    "tooltip": "The stable render (H3AudioDriveLong with chain_frames=true). "
                               "Supplies the face that survives the whole clip.",
                }),
                "driving_images": ("IMAGE", {
                    "tooltip": "The audio-driven render of the same shot (chain_frames=false). "
                               "Supplies the mouth and expression motion.",
                }),
                "audio": ("AUDIO", {"tooltip": "Returned for the mux, shifted by audio_offset_frames."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.001}),
                "animation_region": (["all", "exp", "lip", "pose", "eyes"], {
                    "default": "all",
                    "tooltip": "Which motion regions LivePortrait transfers. 'all' is what was "
                               "measured here; 'lip' only moves the mouth (safer for long takes).",
                }),
                "driving_multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.6, "max": 1.6, "step": 0.05,
                    "tooltip": "Expression gain, matching LivePortrait's driving_multiplier.",
                }),
            },
            "optional": {
                "audio_offset_frames": ("INT", {
                    "default": 0, "min": -48, "max": 48, "step": 1,
                    "tooltip": "Same meaning as in the driving nodes: shifts the returned audio "
                               "against the frames so the lips land on the sound.",
                }),
                "python_exe": ("STRING", {"default": DEFAULT_PYTHON,
                                          "tooltip": "Interpreter that can import LivePortrait's deps."}),
                "liveportrait_dir": ("STRING", {"default": DEFAULT_REPO,
                                                "tooltip": "Local LivePortrait checkout (cwd for inference.py)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("frames", "audio", "report")
    FUNCTION = "transfer"
    CATEGORY = "minimax/h3/audio"

    def transfer(self, source_images, driving_images, audio, fps, animation_region,
                 driving_multiplier, audio_offset_frames=0,
                 python_exe=DEFAULT_PYTHON, liveportrait_dir=DEFAULT_REPO):
        src = _frames_to_uint8(source_images)
        drv = _frames_to_uint8(driving_images)
        n = min(src.shape[0], drv.shape[0])
        if n < 2:
            raise ValueError("H3LipSyncTransfer: need at least 2 frames on both inputs.")
        src, drv = src[:n], drv[:n]

        if not os.path.exists(os.path.join(liveportrait_dir, "inference.py")):
            raise FileNotFoundError(
                "H3LipSyncTransfer: no inference.py under %r - set liveportrait_dir." % liveportrait_dir)

        w, sr = _as_audio(audio)
        work = tempfile.mkdtemp(prefix="h3ls-")
        out_dir = os.path.join(work, "out")
        src_mp4 = os.path.join(work, "source.mp4")
        drv_mp4 = os.path.join(work, "driving.mp4")
        try:
            _write_video(src, src_mp4, fps)
            _write_video(drv, drv_mp4, fps)
            cmd = [python_exe, "inference.py", "-s", src_mp4, "-d", drv_mp4,
                   "-o", out_dir, "--audio_priority", "source",
                   "--animation_region", animation_region,
                   "--driving_multiplier", "%.3f" % driving_multiplier]
            env = dict(os.environ)
            env.setdefault("PYTHONIOENCODING", "utf-8")
            proc = subprocess.run(cmd, cwd=liveportrait_dir, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace",
                                  env=env, timeout=3600)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "")[-1200:]
                raise RuntimeError("H3LipSyncTransfer: LivePortrait failed (exit %d)\n%s"
                                   % (proc.returncode, tail))

            vids = [f for f in os.listdir(out_dir) if f.endswith(".mp4") and "_concat" not in f]
            if not vids:
                raise RuntimeError("H3LipSyncTransfer: LivePortrait produced no video in %r" % out_dir)
            result_path = os.path.join(out_dir, sorted(vids)[0])
            frames = _read_video(result_path)
        finally:
            shutil.rmtree(work, ignore_errors=True)

        shifts = int(audio_offset_frames or 0)
        report = "\n".join([
            "H3 Lip Sync Transfer",
            "source frames    : %d" % n,
            "driving frames   : %d" % drv.shape[0],
            "result frames    : %d (%.3fs @ %.3f fps)" % (frames.shape[0], frames.shape[0] / float(fps), fps),
            "animation region : %s, multiplier %.2f" % (animation_region, driving_multiplier),
            "audio shift      : %+d frames (%+.3fs)" % (shifts, shifts / float(H3_FPS)),
            "mux tip          : connect frames -> CreateVideo and audio -> CreateVideo.audio",
        ])
        print(report, flush=True)
        out_audio = {"waveform": _shift_audio(w, sr, shifts), "sample_rate": sr}
        return (frames, out_audio, report)


NODE_CLASS_MAPPINGS = {"H3LipSyncTransfer": H3LipSyncTransfer}
NODE_DISPLAY_NAME_MAPPINGS = {"H3LipSyncTransfer": "H3 Lip Sync Transfer (stable face + driven mouth)"}
