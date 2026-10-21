"""comfyui-h3-audio-drive-auto

MiniMax H3 audio-driven video, end to end:

  H3AudioDriveAuto    - align an audio track to the 17k+5 frame grid and freeze it
                        into the joint AV latent (video denoises, audio masked at
                        denoise 0). Returns the aligned audio for an exact mux.
  H3AudioDriveLong    - one node for audio of ANY length: grid-legal windows,
                        per-window sampling, frame chaining, original audio out.
  H3LipSyncScore      - offline verification: does the mouth actually move with
                        this audio, and is the video offset by N frames?
  H3LipSyncTransfer   - stable face (chained render) + mouth motion (unchained
                        render) merged by LivePortrait expression transfer.
  H3LipSyncWav2Lip    - phoneme-level mouth re-synthesis on top of any frames.

License: MIT
"""

from .nodes_audio_drive import H3AudioDriveAuto
from .nodes_audio_drive_long import H3AudioDriveLong
from .nodes_lipsync_score import H3LipSyncScore
from .nodes_lipsync_transfer import H3LipSyncTransfer
from .nodes_lipsync_wav2lip import H3LipSyncWav2Lip

NODE_CLASS_MAPPINGS = {
    "H3AudioDriveAuto": H3AudioDriveAuto,
    "H3AudioDriveLong": H3AudioDriveLong,
    "H3LipSyncScore": H3LipSyncScore,
    "H3LipSyncTransfer": H3LipSyncTransfer,
    "H3LipSyncWav2Lip": H3LipSyncWav2Lip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3AudioDriveAuto": "H3 Audio Drive Auto (align + lock)",
    "H3AudioDriveLong": "H3 Audio Drive Long (any length)",
    "H3LipSyncScore": "H3 Lip Sync Score (offline)",
    "H3LipSyncTransfer": "H3 Lip Sync Transfer (stable face + driven mouth)",
    "H3LipSyncWav2Lip": "H3 Lip Sync Wav2Lip (phoneme-level mouth)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
