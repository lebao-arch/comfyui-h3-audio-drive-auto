"""H3 Audio Drive Long - one node, any audio length.

Renders a MiniMax H3 video that follows an arbitrarily long audio track by
splitting it into grid-legal windows, sampling each window with the source audio
frozen into the joint AV latent, chaining the last frame of one window into the
first frame of the next, and returning the concatenated frames plus the original
(trimmed) audio for an exact mux.

This node samples internally: ComfyUI graphs cannot loop, and a windowed render
needs `latent -> sample -> decode -> next first_frame`, which is a cycle in graph
terms. Doing it in one node is the only way to get "connect audio, press run".
"""

import math

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import node_helpers
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise

from .nodes_audio_drive import (
    AUDIO_LATENT_FPS,
    H3_FPS,
    _fit_audio_to_latent,
    _resample,
    _shift_audio,
    _waveform,
    align_frame_count,
    video_latent_t,
)

try:  # progress callback is optional (ComfyUI moves this module around)
    from comfy.utils import ProgressBar
except Exception:  # pragma: no cover
    ProgressBar = None


def _resize_image(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _empty_av(width, height, frames):
    latent_t = video_latent_t(frames)
    audio_t = max(1, round(frames / H3_FPS * AUDIO_LATENT_FPS))
    device = comfy.model_management.intermediate_device()
    video = torch.zeros([1, 24, latent_t, height // 16, width // 16], device=device)
    audio = torch.zeros([1, 32, 2, audio_t], device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _text_cond(clip, video_vae, prompt, first_frame, ref_image, width, height):
    images = []
    keyframes = []
    if ref_image is not None:
        # text-side vision conditioning only (no frame anchor): identity without
        # switching the window to image-to-video semantics.
        rf = ref_image[:, -1] if ref_image.ndim == 5 else ref_image
        images.append(_resize_image(rf[:1], width, height, "center"))
    if first_frame is not None:
        ff = first_frame[:, -1] if first_frame.ndim == 5 else first_frame
        img = _resize_image(ff[:1], width, height, "disabled")
        images.append(img)
        keyframes.append({"resolved_frame_index": 0, "image": img})
    tokens = clip.tokenize(prompt, images=images)
    cond = clip.encode_from_tokens_scheduled(tokens)
    if keyframes:
        for kf in keyframes:
            kf["latent"] = video_vae.encode(kf.pop("image"))
        cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": keyframes})
    return cond


def _sample_av(model, cond, latent, seed, steps, sampler_name, scheduler):
    guider = Guider_Basic(model)
    guider.set_conds(cond)
    sampler = comfy.samplers.sampler_object(sampler_name)
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, steps).cpu()

    patch = guider.model_patcher
    samples = comfy.sample.fix_empty_latent_channels(patch, latent["samples"], None, None)
    noise = Noise_RandomNoise(seed)
    out = guider.sample(
        noise.generate_noise({"samples": samples}),
        samples,
        sampler,
        sigmas,
        denoise_mask=latent.get("noise_mask"),
        callback=lambda *a, **k: None,
        disable_pbar=True,
        seed=seed,
    )
    return out.to(comfy.model_management.intermediate_device())


class H3AudioDriveLong:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MiniMax H3 diffusion model (a patched/EasyCache model is fine)."}),
                "clip": ("CLIP", {"tooltip": "The MiniMax H3 text encoder (CLIPLoader type=minimax)."}),
                "video_vae": ("VAE", {"tooltip": "minimax_h3_video_vae_fp16."}),
                "audio_vae": ("VAE", {"tooltip": "minimax_h3_audio_vae_fp32."}),
                "audio": ("AUDIO", {"tooltip": "Drives every window. The same waveform is returned for the mux."}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 864, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 16}),
                "max_window_seconds": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 15.0, "step": 0.25,
                    "tooltip": "Window length. H3 was trained around 5-15 s; 5 s keeps VRAM "
                               "flat and seams predictable.",
                }),
                "steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "sampler_name": (list(comfy.samplers.SAMPLER_NAMES),),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "seam_fade_frames": ("INT", {
                    "default": 0, "min": 0, "max": 12,
                    "tooltip": "0 = hard cut (the new window is generated from the previous last "
                               "frame, so the seam is normally invisible). >0 cross-fades that many "
                               "frames around the seam.",
                }),
                "chain_frames": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Anchor each window on the previous last frame. Keeps the subject "
                               "stable across seams, but an image anchor makes H3 treat window 2+ "
                               "as image-to-video and they stop following their own audio "
                               "(measured 0.00 vs 0.55 unchained). Off = lips track, subject may "
                               "walk away from you.",
                }),
                "max_windows": ("INT", {
                    "default": 0, "min": 0, "max": 200,
                    "tooltip": "0 = automatic: use as many windows as the audio needs. A positive "
                               "value is a safety cap; the node refuses (with the exact window "
                               "count you need) instead of silently truncating your audio.",
                }),
                "audio_offset_frames": ("INT", {
                    "default": 0, "min": -48, "max": 48, "step": 1,
                    "tooltip": "Shift the RETURNED audio against the rendered frames, in frames at "
                               "24 fps (length preserved). H3 plays the mouth ~10 frames ahead of "
                               "the track, so -10 lands them on the same frame; use the offset "
                               "H3LipSyncScore reports. The conditioning audio is never shifted.",
                }),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip": "Anchors window 1 at frame 0."}),
                "ref_image": ("IMAGE", {
                    "tooltip": "Identity reference seen by the text encoder as <Picture 1> (or "
                               "<Picture 2> when first_frame is also connected). No frame anchor, "
                               "so window 1 stays audio-driven - but window 2+ still stop "
                               "tracking their own audio when this is connected (measured 0.00).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("frames", "audio", "frames_total", "report")
    FUNCTION = "render"
    CATEGORY = "minimax/h3/audio"
    DESCRIPTION = (
        "Render a MiniMax H3 video driven by an audio track of any length: grid-legal windows, "
        "source audio frozen into each AV latent, frame chaining for continuity, original audio "
        "returned for an exact mux."
    )

    # ------------------------------------------------------------------
    def _plan(self, duration, max_window_seconds, max_windows, width=864, height=480):
        win_frames = align_frame_count(int(round(max_window_seconds * H3_FPS)))
        win_frames = max(5, win_frames)
        win_seconds = win_frames / H3_FPS
        plan = []
        t = 0.0
        while t < duration - 1e-6:
            remain = duration - t
            frames = (win_frames if remain > win_seconds
                      else align_frame_count(int(math.ceil(remain * H3_FPS))))
            frames = max(5, min(frames, win_frames))
            start_f = int(round(t * H3_FPS))
            plan.append((start_f, frames))
            t += frames / H3_FPS
        if not plan:
            plan = [(0, 5)]
        needed = len(plan)
        auto = max_windows is None or max_windows <= 0
        limit = needed if auto else max_windows
        total = sum(f for _, f in plan)
        # frames are collected as float16 before the final cat(): keep the plan inside a
        # memory budget instead of failing with a cryptic OOM half an hour in.
        budget = 16.0 * (1024.0 ** 3)
        frames_limit = max(win_frames, int(budget / (width * height * 3 * 2)))
        windows_limit = max(1, frames_limit // win_frames)
        if total > frames_limit:
            raise ValueError(
                "H3AudioDriveLong: %.2fs of audio is %d frames, which is about %.1f GB of "
                "image data - past the %.0f GB budget for one graph. Split the track into "
                "chunks of about %.0fs (<= %d windows) and render them separately, or lower "
                "width/height." % (duration, total, total * width * height * 3 * 2 / (1024.0 ** 3),
                                   budget / (1024.0 ** 3), windows_limit * win_seconds,
                                   windows_limit))
        if needed > limit:
            gb = total * width * height * 3 * 2 / (1024.0 ** 3)
            raise ValueError(
                "H3AudioDriveLong: %.2fs of audio needs %d windows of %.2fs, but "
                "max_windows=%d only covers %.2fs. Set max_windows to %d (or 0 = "
                "automatic). Cost grows linearly: %d windows at 864x480 is roughly "
                "%.0f min here and holds about %.1f GB of image data."
                % (duration, needed, win_seconds, limit, limit * win_seconds, needed,
                   needed, needed * 1.15, total, gb))
        self._last_auto = auto
        total = sum(f for _, f in plan)
        return plan, total

    def render(self, model, clip, video_vae, audio_vae, audio, prompt, width, height,
               max_window_seconds, steps, sampler_name, scheduler, seed,
               seam_fade_frames, max_windows, chain_frames=True, audio_offset_frames=0,
               first_frame=None, ref_image=None):
        w, sr = _waveform(audio)
        duration = w.shape[-1] / float(sr)
        plan, total_frames = self._plan(duration, max_window_seconds, max_windows, width, height)
        total_seconds = total_frames / H3_FPS

        # audio for the mux: exactly as long as the frames we will produce
        want = int(round(total_seconds * sr))
        aligned = w[..., :want] if w.shape[-1] >= want else torch.cat(
            (w, w.new_zeros((*w.shape[:-1], want - w.shape[-1]))), dim=-1)
        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        w32, _ = _resample(aligned[:1], sr, vae_sr)

        bar = ProgressBar(len(plan)) if ProgressBar is not None else None
        collected = []
        carry = first_frame[:1] if first_frame is not None else None
        lines = [
            "H3 Audio Drive Long",
            "audio        : %.2fs @ %d Hz" % (duration, sr),
            "windows      : %d x <= %.2fs  (total %d frames = %.3fs%s)"
            % (len(plan), align_frame_count(int(round(max_window_seconds * H3_FPS))) / H3_FPS,
               total_frames, total_seconds,
               ", max_windows=auto" if getattr(self, "_last_auto", False) else ""),
            "sampling     : %d steps, %s / %s, seed %d" % (steps, sampler_name, scheduler, seed),
        ]

        for idx, (start_f, frames) in enumerate(plan):
            t0 = start_f / H3_FPS
            span = frames / H3_FPS
            a0 = int(round(t0 * vae_sr))
            a1 = a0 + max(1, int(round(span * vae_sr)))
            slice_w = w32[..., a0:a1]

            latent = _empty_av(width, height, frames)
            video_half, audio_half = latent["samples"].unbind()
            encoded = audio_vae.encode(slice_w.movedim(1, -1))
            audio_lock = _fit_audio_to_latent(encoded, audio_half)
            latent["samples"] = comfy.nested_tensor.NestedTensor((video_half, audio_lock))
            latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (torch.ones_like(video_half), torch.zeros_like(audio_lock)))

            cond = _text_cond(clip, video_vae, prompt, carry if chain_frames else None, ref_image, width, height)
            samples = _sample_av(model, cond, latent, seed + idx, steps, sampler_name, scheduler)
            video_latent = samples.unbind()[0]
            decoded = video_vae.decode(video_latent)[..., :3]
            if decoded.ndim == 5:  # video VAEs hand back [B, T, H, W, C]
                decoded = decoded.reshape(-1, *decoded.shape[-3:])

            collected.append(decoded.to(torch.float16).cpu())
            carry = decoded[-1:].to(comfy.model_management.get_torch_device(),
                                    dtype=decoded.dtype)
            if bar is not None:
                bar.update(1)
            lines.append("  window %-2d t=%6.2fs frames=%-3d (%.3fs) rms_src=%.3f rms_lat=%.3f -> %d frames"
                         % (idx + 1, t0, frames, span,
                            float(slice_w.abs().mean()), float(audio_lock.abs().mean()),
                            decoded.shape[0]))
            del latent, samples, video_latent, decoded, cond
            comfy.model_management.soft_empty_cache()

        frames = torch.cat(collected, dim=0)
        del collected

        if seam_fade_frames > 0 and len(plan) > 1:
            frames = self._fade_seams(frames, plan, seam_fade_frames)
            lines.append("seams        : cross-faded %d frames at %d seams"
                         % (seam_fade_frames, len(plan) - 1))

        lines.append("result       : %d frames, %.3fs @ %.0f fps" % (frames.shape[0], total_seconds, H3_FPS))
        lines.append("mux tip      : connect audio -> VHS_VideoCombine.audio (it is already length-matched)")
        if len(plan) > 1:
            lines.append("")
            lines.append("measured limit (H3, this machine, 2 x 5.17s windows):")
            lines.append("  chain_frames=True  -> window 2 scored 0.00 vs its own audio")
            lines.append("                        (0.40 vs window-1 audio: it keeps the old rhythm)")
            lines.append("  chain_frames=False -> window 2 scored 0.55 vs its own audio,")
            lines.append("                        but the subject changed between windows")
            lines.append("  null controls      -> shuffled audio 0.08, white noise 0.01")
            lines.append("  so: chain for a stable subject, unchain for tracking lips - not both.")
        report = "\n".join(lines)
        print(report, flush=True)
        out_audio = _shift_audio(aligned, sr, audio_offset_frames)
        if int(audio_offset_frames or 0):
            print("H3 Audio Drive Long: audio shifted %+d frames (%.3fs)"
                  % (int(audio_offset_frames), int(audio_offset_frames) / float(H3_FPS)),
                  flush=True)
        return (frames, {"waveform": out_audio, "sample_rate": sr}, int(frames.shape[0]), report)

    # ------------------------------------------------------------------
    @staticmethod
    def _fade_seams(frames, plan, fade):
        out = frames.clone()
        pos = 0
        for _, f in plan[:-1]:
            pos += f
            half = min(fade, pos, frames.shape[0] - pos)
            if half <= 0:
                continue
            for i in range(half):
                alpha = (i + 1) / (half + 1.0)
                out[pos - half + i] = (frames[pos - half + i] * (1.0 - alpha * 0.5)
                                       + frames[pos + i] * (alpha * 0.5))
        return out


NODE_CLASS_MAPPINGS = {"H3AudioDriveLong": H3AudioDriveLong}
NODE_DISPLAY_NAME_MAPPINGS = {"H3AudioDriveLong": "H3 Audio Drive Long (any length)"}
