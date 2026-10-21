# Ready-to-load workflows

Five UI graphs that wire the pack into a complete run. Open them with
**Workflows → Open** in ComfyUI (they are also copied to
`ComfyUI/user/default/workflows/`, so they show up in the sidebar).

All five were executed on this machine before being written here; the numbers
are measured, not estimated.

| file | nodes | what it does | time (3090 Ti, 864x480) |
|---|---|---|---|
| `01_quick_audio_drive.json` | 18 | one 5.2 s shot, audio locked into the AV latent, video + lip-sync plot | 117 s |
| `02_long_any_length.json` | 9 | any track, `max_windows=0`, muxed, scored, saved | 27.19 s audio → 6 windows, 659 frames, **~13 min** |
| `03_identity_transfer.json` | 11 | chained take + unchained take → LivePortrait transfer, 248 frames | 156 s |
| `04_wav2lip.json` | 5 | any clip + its own or a dubbing track → Wav2Lip mouth, 248 frames | 252 s |
| `05_dub_w2l_liveportrait.json` | 7 | dubbing track → Wav2Lip → LivePortrait onto the crisp source face | ~6 min |

`02` ships with `max_windows = 0` (automatic) so it accepts any audio file; a
positive value is only a safety cap. If you set a cap that is too small the node
stops with the exact number of windows your track needs, e.g.
`27.19s of audio needs 6 windows of 5.17s, but max_windows=4 only covers 20.67s`.
The same guard refuses tracks that would exceed ~16 GB of frames and tells you
what chunk length to render instead.

## 01 — quick audio drive

```
UNETLoader ─┬─> BasicGuider ─┐
CLIPLoader ─┘   (cond from i2v)│
MiniMaxH3ImageToVideo ──latent┴─> H3AudioDriveAuto ──> SamplerCustomAdvanced ──> VAEDecode
VAELoader(video) ────────────────────────────────────────────────────────────────┘
VAELoader(audio) ──> H3AudioDriveAuto.audio_vae
LoadAudio ─────────> H3AudioDriveAuto.audio ──> CreateVideo.audio
```

Change before running: `LoadAudio` file, the prompt, and `width`/`height`/`length`
(`length` must satisfy `17k+5`; 124 = 5.17 s at 24 fps).
`audio_offset_frames = -4` is the measured correction for this voice; set it to
whatever `H3LipSyncScore` reports.

Outputs: `output/video/H3_AUTO_UI_*.mp4` and `output/h3_lipsync_ui_*.png`
(the plot prints score / verdict / offset).

## 02 — any length in one node

```
UNETLoader, CLIPLoader, VAELoader x2, LoadAudio
        └────────> H3AudioDriveLong ──frames──> VHS_VideoCombine
                          │                    └─audio──┘
                          └──frames+audio──> H3LipSyncScore ──> SaveImage
```

The node plans windows on the `17k+5` grid, samples each one, stitches them and
returns the audio padded/trimmed to the exact frame count, so the mux cannot
drift (`lag = 0 windows of 20 ms, r = 1.000` measured).

`chain_frames = true` keeps the subject stable across seams; window 2+ then
stops following its own audio (measured 0.00 vs 0.55 unchained) — that is what
`03` exists for. `audio_offset_frames = -10` is the measured correction for the
speech track; with the 27.19 s MP3 the plot reported `+8` after that shift, so
the right value for that track is `-2`.

One caveat on scoring a **music** bed: the scorer correlates picture motion with
loudness, so a GOOD score on non-speech audio only proves the mouth moves in
time with the beat, not that it forms phonemes.

## 03 — identity + lips (the three-step recipe)

```
VHS_LoadVideoPath (chained take)   ──> H3LipSyncTransfer.source_images
VHS_LoadVideoPath (unchained take) ──> H3LipSyncTransfer.driving_images
LoadAudio                           ──> H3LipSyncTransfer.audio
                                       └─> frames/audio ──> VHS_VideoCombine, H3LipSyncScore
```

The two input videos are the ones this pack produced earlier
(`H3_LONGDRIVE_TEST_00001-audio.mp4` = stable face, `H3_LONGDRIVE_NOCHAIN_00001-audio.mp4`
= correct mouth movement). Point them at your own takes if you have them.

`H3LipSyncTransfer` shells out to a LivePortrait checkout, so it needs:

- `python_exe` → the interpreter that has LivePortrait's deps (default `G:\ComfyUI-aki-v3.2\ComfyUI-aki-v3.2\python\python.exe`);
- `liveportrait_dir` → a LivePortrait checkout with `pretrained_weights/` filled (default `G:\lipsync-tools\LivePortrait`).

Both are optional widgets with these defaults; change them on another machine.
Expect ~2.5 min per 10 s at 864x480, and 248 frames in → 248 frames out.

`H3LipSyncTransfer` starts that subprocess with the CUDA-12 toolkit `bin` (and
torch's own `lib`) prepended to `PATH`: onnxruntime 1.23 loads
`cublasLt64_12.dll` while torch 2.13 ships CUDA 13, so a plain inherited
environment dies with `Error 126` before inference starts. If your machine has no
CUDA toolkit installed, the node falls back to the torch directories only.

## 04 — Wav2Lip, phoneme-level mouth

```
VHS_LoadVideoPath ──images──> H3LipSyncWav2Lip ──frames──> VHS_VideoCombine
                  └─audio───────────────────────┘  └─audio──┘
                                 └──> H3LipSyncScore ──> SaveImage
```

Ships with the video's own soundtrack wired in; drop a `LoadAudio` on the
`audio` input to dub with a different voice. Measured on the two-window render:
248 frames in → 248 out, 252 s, score 0.224 GOOD.

Notes: the checkpoint is LRS2-trained (**non-commercial**), and the mouth is
synthesised at 96×96 before pasting back, so a tight close-up looks soft.

## 05 — dubbing + Wav2Lip + LivePortrait

```
VHS_LoadVideoPath (crisp source) ──images──> H3LipSyncTransfer.source_images
LoadAudio (your voice) ──audio──> H3LipSyncWav2Lip ──frames──> H3LipSyncTransfer.driving_images
                     └────────────────────────audio────────────────> H3LipSyncTransfer.audio
H3LipSyncTransfer ──frames/audio──> VHS_VideoCombine + H3LipSyncScore
```

Wav2Lip supplies the phoneme timing, LivePortrait re-paints that performance onto
the untouched (sharp) source frames, so you keep the original texture instead of
Wav2Lip's 96×96 patch. Measured: 248 frames in → 248 out, 10.333 s, about 6 min,
score **0.171 WEAK** (Wav2Lip alone on the same clip: 0.224 GOOD) — the transfer
pass smooths mouth amplitude a little while removing the blur. Raise
`driving_multiplier` (1.2–1.4) if you want more movement, lower it if the face
over-articulates.
