# comfyui-h3-audio-drive-auto

Four nodes for **MiniMax H3 audio-driven video**: put your audio *inside* the
joint AV latent so the mouth is generated against it, render a track of any
length by splitting it into windows, prove afterwards that the mouth actually
moved with that audio, and - when one shot has to keep a single face for its
whole length - graft the driven performance onto the stable render in one node.

| node | one line |
|---|---|
| `H3AudioDriveAuto` | snap the audio to H3's `17k+5` grid and freeze it into the AV latent (video denoises, audio is masked at 0) |
| `H3AudioDriveLong` | any length of audio, one node: legal windows, per-window sampling, chained first frames, original audio out |
| `H3LipSyncScore` | offline check: does the mouth track this audio, and by how many frames is it offset |
| `H3LipSyncTransfer` | source render (stable face) + driving render (tracking mouth) -> LivePortrait expression transfer, one node |

![demo](docs/demo.gif)

*10.3 s, 248 frames, one H3 pass per style plus one transfer pass: the face stays
the same person and the mouth tracks the track (H3LipSyncScore 0.352 / 0.340,
offset -2 / +0). Left/right below is the same frame of the stable render and of
the transferred one.*

Quick start: [`examples/minimal_api.json`](examples/minimal_api.json) is a 16-node
graph that runs end to end (verified: 124 frames, 5.17 s, scored 0.515 with a
-10 frame lead). Point node `5` at your wav, node `1`-`4` at your H3 checkpoints,
then convert it to UI form with *Load* on a saved copy if your ComfyUI build does
not accept the API format directly.

Why this is not a wrapper around something already published: MiniMax H3 samples
video and audio **jointly** (`video [B,24,T,H/16,W/16]` + `audio [B,32,2,T*40]`).
Any "add audio afterwards" workflow — including the CosyVoice dub version that
ships in the wild — only swaps the soundtrack on top of a mouth that is still
following the model's own generated track. Lip sync has to happen *before*
sampling. These nodes make that path robust and checkable.

## H3AudioDriveAuto — align + lock

| input | meaning |
|---|---|
| `audio` | the voice that should drive the mouth |
| `audio_vae` | `minimax_h3_audio_vae_fp32.safetensors` |
| `mode` | `lock_source` (audio frozen, drives video) / `remix_source` (partial denoise) / `native` (model invents audio) |
| `pad_mode` | `silence` / `trim` / `reject` when the audio is shorter than the video |
| `width`/`height` | only used when no `av_latent` is connected |
| `av_latent` *(optional)* | the AV latent from `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` |

| output | meaning |
|---|---|
| `av_latent` | the latent with your audio encoded in and `noise_mask` = 1 for video, 0 for audio |
| `audio_aligned` | your audio, trimmed or silence-padded to exactly `length / 24` seconds |
| `length` | the frame count, already snapped to H3's `17k+5` grid |
| `report` | what it did, in plain text |

What it fixes, concretely:

1. **The 17k+5 grid.** H3 only accepts lengths `5, 22, 39, … 124, 141`. A 5.0 s
   request (120 frames) is not legal; the encoder then pads or truncates your
   track behind your back. This node computes `ceil(duration*24)` and snaps up.
2. **Silent truncation.** An audio longer than the video is cut off with no
   warning in the usual graph. Here the cut (or the silence pad) is reported, and
   the returned audio is exactly as long as the video, so the final mux cannot
   drift.
3. **Wrong mux.** It returns the *aligned* waveform for `VHS_VideoCombine`, not
   the raw file.

Typical wiring:

```
LoadAudio ─┬─> H3AudioDriveAuto.audio
           │        ^ audio_vae <- VAELoader(audio vae)
           │        ^ av_latent  <- MiniMaxH3ReferenceToVideo.LATENT
           └─> (keep the same clip on ref_audio_0, name it <Audio 1> in the prompt)

H3AudioDriveAuto.av_latent ─> SamplerCustomAdvanced.latent_image
H3AudioDriveAuto.audio_aligned ─> VHS_VideoCombine.audio
```

## H3AudioDriveLong — any length, one node

Splits the track into legal windows, samples each one with its own slice of audio
frozen into the AV latent, concatenates the frames and hands back an audio track
that is exactly as long as the video.

| input | meaning |
|---|---|
| `model`/`clip`/`video_vae`/`audio_vae` | the usual H3 loaders |
| `audio`, `prompt`, `width`, `height` | what to render |
| `max_window_seconds` | window size ceiling; snapped down to the `17k+5` grid (5.0 s → 124 frames) |
| `chain_frames` | anchor each window on the previous last frame (see the measured limit below) |
| `ref_image` *(optional)* | identity reference for the text encoder |
| `first_frame` *(optional)* | anchor for window 1 only |

Returns `frames, audio, frames_total, report`.

Measured on a 10.12 s track → 2 windows → 248 frames (10.3333 s):

* **the mux is exact**: cross-correlating the muxed track against the source gives
  lag = 0 windows of 20 ms, r = 1.000 (sample-exact);
* window 1 tracks its audio whether or not you chain (0.27–0.35);
* **the measured limit** — window 2 and later:

| setting | window 2 vs its own audio | the subject |
|---|---|---|
| `chain_frames=True` | **0.00** (null level) | stays the same person, pose continuous |
| `chain_frames=False` | **0.55** | changes person at the seam (verified by eye, frame 124) |
| `chain_frames=False` + `ref_image` | **0.00** | same person, but the take goes static |

Null controls on the same frames: shuffled audio 0.08, white noise 0.01, an
unrelated speaker −0.09. So 0.00 there is a real absence, not a flat curve.

Why: when a window carries an image anchor, H3 treats it as image-to-video and
stops following the audio half of the latent. Without an anchor it follows the
audio but re-invents the subject. The chained window 2 does track *something* —
0.40 against **window 1's** audio — i.e. it keeps the previous take's rhythm.

Practical consequence, stated plainly: **you can have a stable subject or
tracking lips across a seam, not both** — with this model, today. For a long
dialogue shot the honest recipes are (a) chain + post-hoc lip-sync
(LatentSync/Wav2Lip/MuseTalk) on the result, or (b) unchain and cut between
windows like an editor would.



## H3LipSyncScore — offline verification

Input `images` + `audio`; outputs `score` (0..1), `offset_frames`, `report`, `plot`.

* ROI: by default the most temporally-active block, chosen **without looking at
  the audio**, so the reported correlation stays unbiased. `roi_mode=face` uses
  an OpenCV Haar cascade when one is installed, `center` is a fixed box.
* Signal: audio RMS envelope vs mouth-region frame-difference energy, correlated
  over a lag sweep with a 3-tap smoothing and a small `|lag|` prior (raw argmax on
  a 5 s clip jumps to the sweep edge; the smoothed version recovers offsets of 0,
  3, 5 and 10 frames exactly on the reference render).
* Measured, not claimed:
  * matching track on the reference render: **0.27** offline, **0.35** through the
    live API graph (the live number is the one saved as
    `output/h3_lipsync_live_v2_00001_.png`);
  * a different voice against the same video: **0.00**;
  * silence: **0.00** (short-circuited before any correlation);
  * synthetic delays of 3, 5 and 10 frames are all reported exactly.
* Honest limit: this correlates articulation *energy*. It reliably catches
  "this audio did not drive this video" and "the video is N frames late". It is
  not a phoneme aligner and cannot certify individual vowels.

## Recipe C - post-hoc expression transfer (measured, works)

When a shot must keep one face for its whole length *and* track the soundtrack,
the fix is two passes plus a transfer step, not one bigger node:

1. **Identity pass** - `H3AudioDriveLong(chain_frames=true)`. Subject stays put;
   window 2+ lips stop tracking audio (0.00 on the scorer).
2. **Performance pass** - the same graph with `chain_frames=false`. Window 2
   tracks audio (0.55) but the subject drifts (a different person in our test).
3. **Transfer** - feed pass 1 as *source* and pass 2 as *driving* to a portrait
   animation model (LivePortrait, `KwaiVGI/LivePortrait`): the driving
   expression, including the mouth, is warped onto the stable face.

Measured on the 10.33 s / 248-frame two-window render (`H3LipSyncScore`,
auto ROI, 24 fps):

| render | window 1 | window 2 | whole clip |
|---|---|---|---|
| chained only | 0.278 | 0.000 | 0.221 |
| chained + LivePortrait driven by the unchained take | 0.328 | 0.156 | 0.280 |

Both windows now score above the WEAK threshold, the face is the same person in
both, and the residual `-10` frame offset is inherited from H3's own
audio-conditioning timing (the driving take shows the same offset), so align
audio afterwards if you need frame-exact dubbing.

Practical notes from the run (Windows, Python 3.13, torch 2.13, 3090 Ti):

- code: `git clone --depth 1 https://gitcode.com/gh_mirrors/li/LivePortrait.git`
  (the github.com remote stalled with `SSL_read: unexpected eof`; the gitcode
  mirror cloned 213 files in 1.6 s);
- weights: ModelScope `AI-ModelScope/LivePortrait` (`liveportrait/base_models/*`,
  `liveportrait/retargeting_models/*`, `liveportrait/landmark.onnx`,
  `insightface/models/buffalo_l/{det_10g,2d106det}.onnx`) linked into
  `pretrained_weights/` with the repo's expected layout;
- extra deps: `pykalman`, `tyro` (both wheel-installable); everything else was
  already present;
- command:
  `python inference.py -s chained.mp4 -d unchained.mp4 -o out --audio_priority driving`
  writes `<source>--<driving>.mp4` (the animation, 864x480) plus a `_concat`
  3-up preview;
- frames: 248 in, 245 out (the pipeline drops the tail), audio muxed at 10.304 s.

MuseTalk is **not** a working alternative on this machine: the ModelScope
`AI-ModelScope/MuseTalk` checkpoints (`musetalkV15/unet.pth` and the v1.0
`pytorch_model.bin`, different files) both decode to the *masked* input - the
lower half stays flat grey with `cos(bottom, masked_bottom)=0.94` while the
reference bottom sits at `0.07`, independent of timestep, channel order, latent
scaling, fp16/fp32 and audio content. The port matches the official
`train.py:306` call exactly, so the released bytes are the suspect, not the
inference code.

## Frame-exact dubbing (`audio_offset_frames`)

H3 does not put the mouth on the frame the sound lands on: it plays the
performance *ahead* of the track. That lead is real and linear, not scorer bias -
advancing the audio by 4 frames moves the measured lag by exactly +4 frames
(`_probe/sync_signal.py`, frame-difference signal, 4-frame steps):

| audio advanced | measured lag (frames) |
|---|---|
| 0 | -10 |
| -4 | -6 |
| -8 | -2 |
| -10 | +1 |
| -12 | +2 |

Both driving nodes therefore take `audio_offset_frames` (INT, -48..48, frames at
24 fps):

- it shifts **only the returned AUDIO**, length preserved (head silence pad when
  positive, head trim when negative), so the mux still matches the frames;
- the conditioning audio is never shifted - the performance is untouched;
- set it to the offset `H3LipSyncScore` reports for that shot.

Measured with the knob (same seed, so the frames are identical and only the
track moves):

| shot | uncalibrated | after `audio_offset_frames` | scorer | independent |
|---|---|---|---|---|
| single window, 124 frames | -4 | -4 | **-1** | -1 |
| 10.3 s LivePortrait clip | -10 | -10 | **+0** | +1 |

Residual spread across windows is about +-2 frames (window 1 of the long clip
settles at -4 while window 2 sits at +1), and the envelope correlation itself is
weak (r ~ 0.3-0.4), so treat the number as a calibration aid rather than a
ground truth: check one shot with clear plosives before mastering.

## H3LipSyncTransfer - the three-step recipe as one node

`Recipe C` above is now a node:

```
H3AudioDriveLong(chain_frames=true)  -> source_images
H3AudioDriveLong(chain_frames=false) -> driving_images
any AUDIO                            -> audio
                                        H3LipSyncTransfer -> frames, audio, report
```

It writes both frame batches to temporary mp4s, runs your local LivePortrait
checkout (`python inference.py -s source.mp4 -d driving.mp4 -o out
--audio_priority source --animation_region ... --driving_multiplier ...`), reads
the animated result back as frames, and returns the audio shifted by
`audio_offset_frames` so the lips land on the sound. No extra Python packages are
needed inside ComfyUI itself - the subprocess uses the interpreter you point it
at (`python_exe`, default the G-drive ComfyUI python, which already has
torch/insightface/onnxruntime/imageio-ffmpeg/pykalman/tyro).

Measured on the 10.33 s two-window shot (248 frames, auto ROI, 24 fps):

| output | window 1 | window 2 | offset |
|---|---|---|---|
| chained only | 0.278 | 0.000 | -11 |
| LivePortrait by hand | 0.328 | 0.156 | -10 |
| **this node** (`audio_offset_frames=-10`) | **0.352** | **0.340** | **-2 / +0** |

The whole clip keeps 248 frames here (the standalone run dropped 3), both windows
clear the GOOD threshold, and an independent frame-difference estimator agrees at
+1 frame. Cost: roughly 2-3 minutes for a 10 s shot, plus temp disk for the two
mp4s.

## Notes

- `lock_source` still leaves the model in charge of *how* the mouth performs; H3
  is not a phoneme-level lip-sync model. For frame-exact dubbing you still need a
  post-hoc lip-sync model (LatentSync / Wav2Lip / MuseTalk) — none of which are
  bundled with this pack.
- Audio longer than ~15 s should be rendered as windows; `H3AudioDriveLong` does
  that for you and returns a length-matched track for the mux.

MIT.

## Publishing (what is left)

The package is ready to ship: MIT licence, registry-ready pyproject.toml,
verified example graph, and every number in this README was measured, not
estimated. Publishing needs your accounts, so the last mile is manual:

`ash
# 1) repo
git init && git add . && git commit -m "H3 audio drive: 4 nodes, measured"
gh auth login                                  # if gh is not authenticated
gh repo create comfyui-h3-audio-drive-auto --public --source . --push

# 2) ComfyUI Registry (needs PublisherId from registry.comfy.org)
pip install comfy-cli
comfy --skip-prompt --here node publish         # reads [tool.comfy] from pyproject.toml
`

Before publishing, replace PublisherId = "your-publisher-id" in
pyproject.toml with the id you register at <https://registry.comfy.org>.

## Licence

MIT - see [LICENSE](LICENSE).
