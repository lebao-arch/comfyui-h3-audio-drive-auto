# Ready-to-load workflows

Three UI graphs that wire the pack into a complete run. Open them with
**Workflows → Open** in ComfyUI (they are also copied to
`ComfyUI/user/default/workflows/`, so they show up in the sidebar).

All three were executed on this machine before being written here; the numbers
are measured, not estimated.

| file | nodes | what it does | time (3090 Ti, 864x480) |
|---|---|---|---|
| `01_quick_audio_drive.json` | 18 | one 5.2 s shot, audio locked into the AV latent, video + lip-sync plot | 117 s |
| `02_long_any_length.json` | 9 | any track, `max_windows=0`, muxed, scored, saved | 27.19 s audio → 6 windows, 659 frames, **~13 min** |
| `03_identity_transfer.json` | 11 | chained take + unchained take → LivePortrait transfer, 248 frames | 156 s |

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
