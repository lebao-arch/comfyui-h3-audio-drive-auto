"""H3 Lip Sync Score - offline, dependency-light lip-sync verification.

No lip-sync model, no network, no API key. It measures whether the mouth region
actually moves in step with the audio track, which is exactly the failure mode
you get when somebody replaces the soundtrack *after* MiniMax H3 has already
generated the video: the mouth keeps following the model's original audio while
a new voice plays over it.

Method
------
1. Audio energy envelope, resampled to the video frame rate.
2. Mouth-region articulation curve from the frames themselves:
   per-frame absolute difference plus vertical-edge energy inside the mouth ROI
   of the detected face (OpenCV Haar cascade; falls back to a centre box).
3. Normalised cross-correlation over a lag sweep -> single 0..1 score plus the
   offset in frames (positive = mouth lags behind the audio).

Limitations, stated plainly: this is a *correlation* check on articulation
energy, not a phoneme aligner. It reliably catches "audio and mouth are not
driven by the same source" and "the video is offset by N frames". It cannot
certify that every vowel is correct.
"""

import math

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:  # pragma: no cover - cv2 ships with ComfyUI
    cv2 = None


# --------------------------------------------------------------------------
def _to_numpy_uint8(images):
    arr = images.detach().cpu().numpy() if torch.is_tensor(images) else np.asarray(images)
    while arr.ndim > 4 and arr.shape[0] == 1:  # tolerate [1,B,H,W,C]
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[None, ...]
    arr = np.clip(arr * 255.0 if arr.max() <= 1.5 else arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _envelope(audio, fps, n_frames):
    w = audio["waveform"]
    sr = int(audio["sample_rate"])
    if torch.is_tensor(w):
        w = w.detach().cpu().numpy()
    w = np.asarray(w, dtype=np.float32)
    if w.ndim == 3:
        w = w[0]
    mono = w.mean(axis=0) if w.ndim == 2 else w
    # sample index -> frame index
    hop = sr / float(fps)
    env = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        a = int(i * hop)
        b = min(len(mono), int((i + 1) * hop))
        if b <= a:
            env[i] = 0.0
        else:
            seg = mono[a:b]
            env[i] = float(np.sqrt(np.mean(seg * seg)))
    return env


def _detect_face_box(frames, sample=8):
    """Median frontal-face box across sampled frames; None if cv2/cascade fails."""
    if cv2 is None:
        return None
    candidates = []
    try:
        candidates.append(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:
        pass
    candidates += [
        "haarcascade_frontalface_default.xml",
        "C:/opencv/data/haarcascades/haarcascade_frontalface_default.xml",
    ]
    cascade = None
    for path in candidates:
        try:
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                cascade = c
                break
        except Exception:
            continue
    if cascade is None:
        return None

    n = frames.shape[0]
    idx = np.linspace(0, n - 1, min(sample, n)).astype(int)
    boxes = []
    for i in idx:
        gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4,
                                        minSize=(48, 48))
        if len(faces):
            boxes.append(max(faces, key=lambda b: b[2] * b[3]))
    if not boxes:
        return None
    return np.median(np.array(boxes, dtype=np.float32), axis=0)


def _auto_roi(frames, grid=12):
    """Locate the most-active screen region WITHOUT consulting the audio.

    Block energies come from pure temporal difference, so selecting the block
    introduces no bias toward the correlation we are about to measure: it only
    says 'this is where things move', never 'this is where things move with the
    sound'. That keeps the reported correlation honest.
    """
    g = frames.astype(np.float32).mean(axis=3)          # [N,H,W]
    d = np.abs(np.diff(g, axis=0)).mean(axis=0)         # [H,W]
    h, w = d.shape
    bh, bw = max(8, h // grid), max(8, w // grid)
    rows, cols = h // bh, w // bw
    pooled = d[: rows * bh, : cols * bw].reshape(rows, bh, cols, bw).mean(axis=(1, 3))
    ry, rx = np.unravel_index(int(np.argmax(pooled)), pooled.shape)
    # grow one block in each direction so the whole mouth stays inside the ROI
    y0 = max(0, (ry - 1) * bh)
    y1 = min(h, (ry + 2) * bh)
    x0 = max(0, (rx - 1) * bw)
    x1 = min(w, (rx + 2) * bw)
    return np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)


def _mouth_curves(frames, box):
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    if box is None:
        x0, x1 = int(0.28 * w), int(0.72 * w)
        y0, y1 = int(0.55 * h), int(0.95 * h)
    else:
        bx, by, bw, bh = [float(v) for v in box]
        x0 = int(bx + 0.22 * bw)
        x1 = int(bx + 0.78 * bw)
        y0 = int(by + 0.60 * bh)
        y1 = int(by + 0.95 * bh)
    x0, x1 = max(0, min(x0, w - 2)), max(1, min(x1, w))
    y0, y1 = max(0, min(y0, h - 2)), max(1, min(y1, h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        x0, x1 = int(0.25 * w), int(0.75 * w)
        y0, y1 = int(0.6 * h), int(0.95 * h)

    rois = frames[:, y0:y1, x0:x1, :]
    # articulation = temporal difference (any mouth movement)
    diff = np.abs(np.diff(rois.astype(np.float32), axis=0)).mean(axis=(1, 2, 3))
    motion = np.concatenate([[0.0], diff])
    # vertical edges = lip line / teeth gap becoming visible
    gray = rois.astype(np.float32).mean(axis=3)
    gy = np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2))
    edges = gy.astype(np.float32)
    return motion, edges, (x0, y0, x1, y1)


def _z(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    if s < 1e-9:
        return np.zeros_like(x)
    return (x - x.mean()) / s


def _rcurve(a, b, max_lag):
    """Correlation of a vs b for every lag in [-max_lag, +max_lag]."""
    a, b = _z(a), _z(b)
    lags = np.arange(-max_lag, max_lag + 1)
    rs = np.full(len(lags), -2.0)
    for i, lag in enumerate(lags):
        if lag >= 0:
            x, y = a[: len(a) - lag], b[lag:]
        else:
            x, y = a[-lag:], b[: len(b) + lag]
        if len(x) < 12:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if math.isfinite(r):
            rs[i] = r
    return lags, rs


def _pick_lag(lags, rs, prior=0.06):
    """Smoothed argmax with a mild preference for zero lag.

    Empirical reason for both steps: on a 5 s clip, raw argmax jumps to the
   窗口 edge (the curve is nearly flat there), while a 3-lag moving average plus
    a small |lag| penalty recovers the true offset of 0, 3, 5 and 10 frames
    exactly on the reference render (see _probe/lipsync_lag_lab.py).
    """
    grid = np.nan_to_num(rs, nan=-2.0, posinf=-2.0, neginf=-2.0)
    sm = np.convolve(grid, np.ones(3) / 3.0, mode="same")
    adj = sm - prior * np.abs(lags) / max(1, int(np.abs(lags).max()))
    i = int(np.argmax(adj))
    peak = float(sm[i])
    far = np.abs(lags - lags[i]) >= 4
    contrast = peak - float(sm[far].mean()) if bool(far.any()) else 0.0
    return peak, int(lags[i]), float(np.clip(contrast, 0.0, 1.0))


def _plot(env, motion, score, lag, box_text, conf=0.0, w=900, h=320):
    img = Image.new("RGB", (w, h), (18, 18, 22))
    d = ImageDraw.Draw(img)
    pad = 24
    plot_h = h - 90

    def draw_series(series, color, scale):
        vals = (series - series.min()) / (series.max() - series.min() + 1e-9)
        pts = []
        for i, v in enumerate(vals):
            x = pad + i * (w - 2 * pad) / max(1, len(vals) - 1)
            y = 40 + plot_h - v * plot_h * 0.92
            pts.append((x, y))
        for i in range(len(pts) - 1):
            d.line([pts[i], pts[i + 1]], fill=color, width=2)

    draw_series(env, (90, 200, 255), 1.0)
    draw_series(motion, (255, 150, 90), 1.0)

    verdict = "GOOD" if score >= 0.18 else ("WEAK" if score >= 0.10 else "MISMATCH")
    color = (120, 230, 140) if score >= 0.18 else ((240, 210, 120) if score >= 0.10 else (240, 110, 110))
    d.text((pad, 12), "audio envelope (blue)  vs  mouth articulation (orange)", fill=(200, 200, 210))
    d.text((pad, h - 42),
           "score %.3f [%s]  conf %.2f  offset %+d frames  %s"
           % (score, verdict, conf, lag, box_text), fill=color)
    d.text((pad, h - 24), "positive offset = mouth lags the audio", fill=(120, 120, 130))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


# --------------------------------------------------------------------------
class H3LipSyncScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Decoded video frames."}),
                "audio": ("AUDIO", {"tooltip": "The sound that is supposed to drive the mouth."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "roi_mode": (["auto", "face", "center"], {
                    "default": "auto",
                    "tooltip": "auto: pick the most temporally-active block (no audio involvement, "
                               "so the correlation stays unbiased). face: OpenCV Haar cascade, "
                               "falls back to auto. center: fixed centre box.",
                }),
                "max_lag_frames": ("INT", {"default": 8, "min": 0, "max": 48, "step": 1}),
                "plot": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING", "IMAGE")
    RETURN_NAMES = ("score", "offset_frames", "report", "plot")
    FUNCTION = "score_pair"
    CATEGORY = "minimax/h3/audio"
    DESCRIPTION = (
        "Offline lip-sync verification: correlates the audio energy envelope with mouth-region "
        "articulation in the rendered frames. Returns a 0..1 score, the best-fit offset in "
        "frames, a report and a plot. Catches 'soundtrack replaced after generation' and "
        "'video is N frames out of sync'."
    )

    def score_pair(self, images, audio, fps, roi_mode, max_lag_frames, plot):
        frames = _to_numpy_uint8(images)
        n = frames.shape[0]
        if n < 8:
            report = ("H3LipSyncScore: need at least ~8 frames, got %d. "
                      "Nothing to measure." % n)
            empty = torch.zeros((1, 64, 256, 3))
            return (0.0, 0, report, empty)

        env = _envelope(audio, fps, n)
        if float(np.asarray(env).std()) < 1e-6:
            report = ("H3 Lip Sync Score: the audio is silent (or below -80 dBFS), "
                      "so there is nothing for a mouth to follow. Score 0.0.")
            return (0.0, 0, report, torch.zeros((1, 64, 256, 3)))
        box = None
        if roi_mode == "face":
            box = _detect_face_box(frames)
            roi_src = ("face detected" if box is not None
                       else "face not found - auto ROI fallback")
            if box is None:
                box = _auto_roi(frames)
        elif roi_mode == "auto":
            box = _auto_roi(frames)
            roi_src = "auto: most-active block (audio-blind selection)"
        else:
            roi_src = "centre box (fixed)"
        motion, edges, roi = _mouth_curves(frames, box)

        # Cap the lag sweep: on short clips a wide sweep has no statistical power
        # and the correlation curve is flat at the edges.
        sweep = max(1, min(int(max_lag_frames), max(1, n // 8)))
        lags, r_motion = _rcurve(env, motion, sweep)
        _, r_edges = _rcurve(env, edges, sweep)
        peak, lag, conf = _pick_lag(lags, r_motion)
        score = float(np.clip(peak, 0.0, 1.0))
        curve, used = motion, "frame-difference energy"
        edge_peak = float(np.max(r_edges)) if len(r_edges) else -2.0

        verdict = ("good - the mouth is being driven by this audio"
                   if score >= 0.18 else
                   ("weak - some correlation, check by eye"
                    if score >= 0.10 else
                    "mismatch - this audio is very likely NOT what drove the video"))
        advice = []
        if abs(lag) >= 2 and conf >= 0.10:
            advice.append(
                "shift the audio %+d frames (%.3fs) relative to the video to align them"
                % (-lag, -lag / fps))
        if conf < 0.10:
            advice.append("lag peak is flat (conf %.2f) - treat the offset as a hint, not a fact" % conf)
        if score < 0.18:
            advice.append(
                "typical causes: soundtrack replaced after generation, wrong clip, "
                "or the mouth is off-screen / too small")
        if roi_mode == "auto":
            advice.append("ROI was auto-picked by motion energy; roi_mode=center as a cross-check")

        report = "\n".join([
            "H3 Lip Sync Score",
            "frames           : %d @ %.1f fps (%.2fs)" % (n, fps, n / fps),
            "roi              : %s  box=%s" % (roi_src, roi),
            "signal used      : %s (sweep +/-%d frames)" % (used, sweep),
            "score            : %.3f   confidence %.3f" % (score, conf),
            "diagnostics      : r_edges peak %.3f" % edge_peak,
            "offset           : %+d frames (%+.3fs), positive = mouth lags audio"
            % (lag, lag / fps),
            "verdict          : %s" % verdict,
            "notes            : correlation of articulation energy, not a phoneme aligner",
        ] + (["advice           : " + a for a in advice] if advice else []))

        plot_img = (_plot(env, curve, score, lag, roi_src, conf) if plot
                    else torch.zeros((1, 64, 256, 3)))
        return (score, int(lag), report, plot_img)


NODE_CLASS_MAPPINGS = {"H3LipSyncScore": H3LipSyncScore}
NODE_DISPLAY_NAME_MAPPINGS = {"H3LipSyncScore": "H3 Lip Sync Score (offline)"}
