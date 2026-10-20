# Examples

## `minimal_api.json`

A 16-node graph that runs end to end with only core ComfyUI nodes plus
`H3AudioDriveAuto`:

```
UNETLoader ─────────────┐
CLIPLoader ──┐          │
VAELoader(v) ─┐         │
VAELoader(a) ─┐         │
LoadAudio ────┴─> MiniMaxH3ImageToVideo ──> H3AudioDriveAuto ──> SamplerCustomAdvanced
                                     (cond + av_latent)        (euler / simple / 12 steps)
                                                                        │
                                       VAEDecode + VAEDecodeAudio ──────┤
                                                                        ▼
                                                                  CreateVideo ──> SaveVideo
```

Verified on this machine: 124 frames, 864x480, 5.167 s, `H3LipSyncScore` 0.515
with a -10 frame lead (feed `audio_offset_frames = -10` to land it).

To use it:

1. replace the four model names with the checkpoints you have
   (`minimax_h3_*` diffusion model, `qwen3vl_32b_minimax_h3*` text encoder,
   `minimax_h3_video_vae_fp16`, `minimax_h3_audio_vae_fp32`);
2. drop your wav into `ComfyUI/input/` and set node `5.LoadAudio.audio`;
3. `length` is already grid-legal (`124` = 5.167 s at 24 fps). For another
   length use `17k+5`: 124, 141, 158, ... (`H3AudioDriveAuto` will snap whatever
   you pass, but the base node wants a legal value).

The output audio of `H3AudioDriveAuto` (`[7, 1]`) is what feeds `CreateVideo`, so
the mux is sample-exact against the frames - that is the point of routing the
track through the node instead of using the model's own generated audio.

For an any-length render, swap `MiniMaxH3ImageToVideo + H3AudioDriveAuto +
sampler` for a single `H3AudioDriveLong`, and feed its `frames` + `audio` into
`VHS_VideoCombine` (or `CreateVideo`).

For the "same face for the whole clip *and* lips that track" case, render twice
(`chain_frames=true` and `false`) and run both through `H3LipSyncTransfer`.
