# Custom node patches

Fixes for third-party custom nodes. The launcher automatically runs
`repair_custom_nodes.py` and `pixal3d_meta_loading.py` against the selected
ComfyUI checkout. Other standalone patches below have manual instructions.

## Pixal3D memory and Windows compatibility

The startup repairs validate safetensors headers and read one tensor at a
time into owned buffers, transferring each tensor directly to the active
stage's device. Opening a large checkpoint does not require a full-file
memory mapping or a second checkpoint-sized commit reservation.

With Pixal3D's `low_vram` option, checkpoints load one stage at a time and CPU
offload releases the active weights. Later stages reload from disk, avoiding
large GPU-to-CPU copies and keeping inactive stages out of host memory. The
four projection extractors share their identical frozen DINO backbone.

On Windows RTX 50-series GPUs, sparse convolutions use chunked PyTorch matrix
multiplications to avoid a native abort in Triton's masked split-K compiler.
The im2col temporary is limited to 64 MiB per chunk. Disk reloads and this
convolution path can increase generation time.

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

Re-runnable, writes `.orig` backups, preserves each file's line endings, and
leaves already-fixed files alone (the node's `chatterbox_official_23lang` engine
ships part of this fix upstream). Pass `--node-root <path>` to patch a
tts_audio_suite checkout somewhere other than `ComfyUI/custom_nodes/`.

Submitted upstream as [diodiogod/TTS-Audio-Suite#368][pr]; drop this patch once
it lands in a release we install.

[pr]: https://github.com/diodiogod/TTS-Audio-Suite/pull/368
