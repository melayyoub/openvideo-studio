# Custom node patches

Fixes for third-party custom nodes. `ComfyUI/custom_nodes/` is gitignored, so
nothing here is applied automatically — re-run a patch after installing,
updating, or reinstalling the node it targets.

## controlnet_instantx_format2.py — drop-in replacement

```powershell
cp patches\controlnet_instantx_format2.py ComfyUI\custom_nodes\ComfyUI-eesahesNodes\controlnet\controlnet_instantx_format2.py
```

## tts_audio_suite_s3gen_token_bounds.py — in-place fix

Stops ChatterBox TTS from aborting the ComfyUI process with a CUDA
`vectorized gather kernel index out of bounds` assert (reported, misleadingly,
from `s3gen/transformer/embedding.py` in `extend_pe`).

T3 samples speech tokens from an 8194-wide head, `drop_invalid_tokens` only
strips the SoS/EoS markers, and S3Gen's `flow.input_embedding` has 6561 rows —
so a sampled id in 6563..8193 indexes off the end of the embedding table. It is
seed-dependent, so it looks intermittent. The patch drops out-of-vocabulary ids
and clamps the embedding index at both ends.

```powershell
python patches\tts_audio_suite_s3gen_token_bounds.py           # apply
python patches\tts_audio_suite_s3gen_token_bounds.py --check   # report only, exit 1 if unpatched
```

Re-runnable, writes `.orig` backups, and leaves already-fixed files alone (the
node's `chatterbox_official_23lang` engine ships part of this fix upstream).
