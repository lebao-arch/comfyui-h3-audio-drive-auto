"""H3 Audio Drive Auto - robust audio<>frame alignment + source-audio lock for MiniMax H3.

Author: built on the DeepSeek Harness session, 2026-10-20.
License: MIT

Why this exists
---------------
MiniMax H3 generates video and audio *jointly* in one AV latent
(video [B,24,T,H/16,W/16] + audio [B,32,2,T40], 40 audio-latent frames/second).
Lip sync therefore cannot be added afterwards: the audio has to be inside the
latent before sampling.

Two things are easy to get wrong, and both silently produce a "mouth does not
match the voice" video:

1.  H3 snaps every length to the 17k+5 frame grid at 24 fps (5, 22, 39, ... 124).
    Feeding 5.0 s (=120 frames) instead of 124/125 changes the audio latent
    length, and the encoder pads/truncates the source track behind your back.
2.  Feeding an audio file longer than the video silently truncates it, so the
    tail of a sentence can vanish while the mouth keeps moving.

This node does the arithmetic explicitly, aligns the track to the exact latent
length, and reports what it trimmed or padded.
"""

import math

import torch

import comfy.nested_tensor

AUDIO_LATENT_FPS = 40
H3_FPS = 24.0
GRID = 17
GRID_OFFSET = 5


# --------------------------------------------------------------------------
# H3 lenght <-> latent helpers (mirrors comfy_extras/nodes_minimax_h3.py)
# --------------------------------------------------------------------------
def align_frame_count(n: int) -> int:
    """Snap a frame count up to the model's 17k+5 grid (5, 22, ..., 124, 141)."""
    n = max(GRID_OFFSET, int(n))
    while (n - GRID_OFFSET) % GRID != 0:
        n += 1
    return n


def video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((frame_count - 5) // 17) * 5 + 2


def frame_count_for_latent_t(latent_t: int) -> int:
    """Inverse of video_latent_t (valid for latent_t >= 2, latent_t % 5 == 2)."""
    if latent_t <= 2:
        return 5
    k = round((latent_t - 2) / 5)
    return 17 * k + 5


def audio_latent_t(duration: float) -> int:
    return max(1, round(duration * AUDIO_LATENT_FPS))


def frames_for_audio(duration: float, extra_seconds: float = 0.0) -> int:
    """Smallest grid-legal frame count that covers a whole audio track."""
    return align_frame_count(int(math.ceil((duration + extra_seconds) * H3_FPS)))


# --------------------------------------------------------------------------
# AV latent helpers
# --------------------------------------------------------------------------
def _split_av_latent(latent):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("H3AudioDriveAuto: expected a ComfyUI LATENT dict.")
    samples = latent["samples"]
    if not getattr(samples, "is_nested", False):
        raise ValueError(
            "H3AudioDriveAuto: expected a joint MiniMax H3 video+audio latent "
            "(connect the LATENT output of MiniMax H3 Image to Video / Reference to Video)."
        )
    parts = list(samples.unbind())
    if len(parts) < 2:
        raise ValueError("H3AudioDriveAuto: AV latent has no audio half.")
    return parts[0], parts[1]


def _fit_1d_last(t: torch.Tensor, target_t: int) -> torch.Tensor:
    cur = t.shape[-1]
    if cur == target_t:
        return t
    if cur > target_t:
        return t[..., :target_t]
    pad = t.new_zeros((*t.shape[:-1], target_t - cur))
    return torch.cat((t, pad), dim=-1)


def _fit_audio_to_latent(encoded: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Match channel/stereo layout and time axis of the model's audio half."""
    if encoded.shape[1:-1] != template.shape[1:-1]:
        raise ValueError(
            "H3AudioDriveAuto: encoded audio layout %s does not match the AV latent %s."
            % (tuple(encoded.shape), tuple(template.shape))
        )
    b = template.shape[0]
    if encoded.shape[0] == 1 and b > 1:
        encoded = encoded.repeat(b, 1, 1, 1)
    elif encoded.shape[0] != b:
        encoded = encoded[:b]
        if encoded.shape[0] != b:
            raise ValueError("H3AudioDriveAuto: cannot match audio batch to latent batch.")
    encoded = _fit_1d_last(encoded, template.shape[-1])
    return encoded.to(device=template.device, dtype=template.dtype)


# --------------------------------------------------------------------------
# Audio waveform helpers
# --------------------------------------------------------------------------
def _waveform(audio):
    if not isinstance(audio, dict):
        raise ValueError("H3AudioDriveAuto: connect an AUDIO input.")
    w = audio.get("waveform")
    sr = audio.get("sample_rate")
    if w is None or sr is None:
        raise ValueError("H3AudioDriveAuto: AUDIO input is missing waveform/sample_rate.")
    if w.ndim != 3:
        raise ValueError("H3AudioDriveAuto: waveform must be [batch, channels, samples].")
    return w, int(sr)


def _resample(w, sr, target_sr, target_channels=None):
    if sr != target_sr:
        import torchaudio

        w = torchaudio.functional.resample(w, sr, target_sr)
        sr = target_sr
    if target_channels is not None and w.shape[1] != target_channels:
        if w.shape[1] == 1:
            w = w.repeat(1, target_channels, 1)
        else:
            w = w[:, :target_channels]
    return w, sr


def _fit_audio_duration(w, sr, seconds, pad_mode):
    """Trim or pad the waveform to exactly `seconds` (mode-aware)."""
    target = int(round(seconds * sr))
    cur = w.shape[-1]
    action = "exact"
    if cur - target > 2:
        action = "trimmed %.3fs" % ((cur - target) / sr)
        w = w[..., :target]
    elif target - cur > 2:
        action = "padded %.3fs" % ((target - cur) / sr)
        w = torch.cat((w, w.new_zeros((*w.shape[:-1], target - cur))), dim=-1)
    else:
        w = w[..., :target] if cur >= target else w
    return w, target, action


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------
def _shift_audio(w, sr, frames, fps=H3_FPS):
    """Shift a waveform in time without changing its length.

    Positive `frames` delays the audio (silence padded at the head, tail cut);
    negative advances it (head trimmed, silence padded at the tail). Length is
    preserved so the mux still matches the rendered video exactly.
    """
    frames = int(frames or 0)
    if frames == 0 or w.shape[-1] == 0:
        return w
    n = int(round(abs(frames) / float(fps) * sr))
    n = min(n, w.shape[-1])
    if n <= 0:
        return w
    pad = torch.zeros_like(w[..., :n])
    if frames > 0:
        return torch.cat([pad, w[..., :-n]], dim=-1)
    return torch.cat([w[..., n:], pad], dim=-1)


class H3AudioDriveAuto:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Voice/dialogue that should drive the mouth. The same clip is "
                               "returned unchanged (duration-aligned) for the final mux.",
                }),
                "audio_vae": ("VAE", {
                    "tooltip": "MiniMax H3 audio VAE (minimax_h3_audio_vae_fp32.safetensors).",
                }),
                "mode": (["lock_source", "remix_source", "native"], {
                    "default": "lock_source",
                    "tooltip": "lock_source: the source audio is frozen into the latent and only "
                               "drives the video. remix: audio is partially denoised "
                               "(keeps timing, lets the model re-voice it). native: leave the "
                               "audio half free so the model invents its own track.",
                }),
                "remix_strength": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Only used when mode=remix_source. 0=fully locked, 1=fully regenerated.",
                }),
                "pad_mode": (["silence", "trim", "reject"], {
                    "default": "silence",
                    "tooltip": "When the audio is shorter than the video: silence-pad, trim the "
                               "video length down to the audio, or raise an error.",
                }),
                "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
                "max_window_seconds": ("FLOAT", {
                    "default": 14.0, "min": 1.0, "max": 60.0, "step": 0.5,
                    "tooltip": "Long-audio planning only. H3 was trained on ~124-362 frames "
                               "(5-15 s); longer tracks should be rendered as windows.",
                }),
                "window_overlap_seconds": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Overlap suggested for a windowed render (continuity guide).",
                }),
                "audio_offset_frames": ("INT", {
                    "default": 0, "min": -48, "max": 48, "step": 1,
                    "tooltip": "Shift the RETURNED audio against the rendered video, in frames at "
                               "24 fps (length preserved). H3 tends to play the mouth ~10 frames "
                               "ahead of the track, so a value of -10 lands the two on the same "
                               "frame; use the offset H3LipSyncScore reports. The latent that "
                               "drives the performance is never shifted.",
                }),
            },
            "optional": {
                "av_latent": ("LATENT", {
                    "tooltip": "AV latent from MiniMax H3 Image/Reference to Video. If connected, "
                               "the audio is aligned to THAT latent. If omitted, a clean empty AV "
                               "latent is built from width/height/audio duration.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("av_latent", "audio_aligned", "length", "report")
    FUNCTION = "drive"
    CATEGORY = "minimax/h3/audio"
    DESCRIPTION = (
        "Aligns an audio track to MiniMax H3's 17k+5 frame grid and freezes it into the joint "
        "AV latent (video denoises, audio is masked at denoise 0). Returns the duration-aligned "
        "audio for an exact final mux, the frame-grid-legal length, and a human-readable report."
    )

    def drive(self, audio, audio_vae, mode, remix_strength, pad_mode,
              width, height, max_window_seconds, window_overlap_seconds,
              audio_offset_frames=0, av_latent=None):
        w, sr = _waveform(audio)
        duration = w.shape[-1] / float(sr)

        # ---- 1. decide the frame count -------------------------------------
        notes = []
        if av_latent is not None:
            video_half, audio_half = _split_av_latent(av_latent)
            latent_t = int(audio_half.shape[-1])
            frames = frame_count_for_latent_t(int(video_half.shape[-1]))
            target_seconds = frames / H3_FPS
            notes.append("aligned to connected AV latent")
        else:
            frames = frames_for_audio(duration)
            target_seconds = frames / H3_FPS
            if pad_mode == "trim" and duration < target_seconds:
                # snap the video down to the nearest legal grid point that still fits
                frames = align_frame_count(max(GRID_OFFSET, int(math.floor(duration * H3_FPS))))
                target_seconds = frames / H3_FPS
            if pad_mode == "reject" and duration > target_seconds + 1e-6:
                raise ValueError(
                    "H3AudioDriveAuto: audio (%.3fs) is longer than the selected grid length "
                    "(%.3fs). Use pad_mode=silence or render windows." % (duration, target_seconds)
                )
            latent_t = video_latent_t(frames)
            notes.append("built a fresh empty AV latent")

        # ---- 2. align the waveform to the latent duration -------------------
        aligned, target_samples, action = _fit_audio_duration(w, sr, target_seconds, pad_mode)
        if action != "exact":
            notes.append("audio " + action)
        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        w_for_vae, _ = _resample(aligned[:1], sr, vae_sr)

        # ---- 3. build / patch the AV latent --------------------------------
        if av_latent is None:
            video_half = torch.zeros(
                [1, 24, latent_t, height // 16, width // 16],
                device=comfy.model_management.intermediate_device(),
            )
            audio_half = torch.zeros(
                [1, 32, 2, audio_latent_t(target_seconds)],
                device=comfy.model_management.intermediate_device(),
            )
            out_latent = {"samples": comfy.nested_tensor.NestedTensor((video_half, audio_half))}
        else:
            out_latent = av_latent.copy()

        if mode == "native":
            audio_out_latent = audio_half
            mask_audio = torch.ones_like(audio_half)
        else:
            encoded = audio_vae.encode(w_for_vae.movedim(1, -1))
            audio_out_latent = _fit_audio_to_latent(encoded, audio_half)
            if mode == "lock_source":
                mask_audio = torch.zeros_like(audio_out_latent)
            else:
                mask_audio = torch.full_like(audio_out_latent, float(remix_strength))

        out_latent["samples"] = comfy.nested_tensor.NestedTensor((video_half, audio_out_latent))
        out_latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video_half), mask_audio)
        )

        # ---- 4. long-audio window plan -------------------------------------
        if duration > max_window_seconds + 1e-6:
            stride = max(1.0, max_window_seconds - window_overlap_seconds)
            n = int(math.ceil((duration - window_overlap_seconds) / stride))
            n = max(1, n)
            plan = "%d windows x %.1fs (stride %.1fs, overlap %.1fs)" % (
                n, max_window_seconds, stride, window_overlap_seconds)
        else:
            plan = "single window (%.1fs)" % duration

        # ---- 5. report ------------------------------------------------------
        report_lines = [
            "H3 Audio Drive Auto",
            "audio            : %.3fs @ %d Hz, %d ch" % (duration, sr, w.shape[1]),
            "mode             : %s" % mode,
            "frame grid       : %d frames (%.3fs @ %.1f fps, latent_t=%d, audio_latent_t=%d)"
            % (frames, target_seconds, H3_FPS, latent_t, audio_out_latent.shape[-1]),
            "alignment        : %s (%d samples @ %d Hz)" % (action, target_samples, sr),
            "noise mask       : video=1.0  audio=%s" % (
                "0.0 (frozen)" if mode == "lock_source" else
                ("1.0 (free)" if mode == "native" else "%.2f" % remix_strength)),
            "window plan      : %s" % plan,
            "audio shift      : %+d frames (%+.3fs) applied to the returned track"
            % (int(audio_offset_frames), int(audio_offset_frames) / float(H3_FPS)),
            "notes            : %s" % ("; ".join(notes) if notes else "none"),
        ]
        if mode != "native":
            report_lines.append(
                "tip              : keep this same clip on ref_audio_0 and name it <Audio 1> "
                "in the prompt, otherwise the model may treat it as background sound."
            )
        report = "\n".join(report_lines)

        out_audio = {"waveform": _shift_audio(aligned, sr, audio_offset_frames),
                     "sample_rate": sr}
        return (out_latent, out_audio, int(frames), report)


NODE_CLASS_MAPPINGS = {"H3AudioDriveAuto": H3AudioDriveAuto}
NODE_DISPLAY_NAME_MAPPINGS = {"H3AudioDriveAuto": "H3 Audio Drive Auto (align + lock)"}
