#!/usr/bin/env python3
"""Bound S3Gen speech-token ids in the ``tts_audio_suite`` custom node.

Symptom
-------
ChatterBox TTS aborts the whole ComfyUI process, mid-generation, with::

    IndexKernelUtils.cu:16: block: [...] Assertion `ind >=0 && ind < ind_dim_size
    && "vectorized gather kernel index out of bounds"` failed.
    Fatal Python error: Aborted

The reported Python frame is ``s3gen/transformer/embedding.py`` in ``extend_pe``,
which is misleading: CUDA reports the assert at the next synchronising call, and
``extend_pe`` just happens to do a host-to-device copy of its cached ``pe``.

Cause
-----
T3's ``speech_head`` spans ``speech_tokens_dict_size`` (8194) logits, so
``torch.multinomial`` can sample any id in ``[0, 8193]``. ``drop_invalid_tokens``
only slices off the SoS/EoS markers (6561/6562) and never bounds-checks, while
S3Gen's ``flow.input_embedding`` is ``nn.Embedding(SPEECH_VOCAB_SIZE=6561, 512)``.
An id in 6563..8193 landing before EoS therefore indexes past the embedding
table. ``torch.clamp(token, min=0)`` at the call site guards only the low end.

Occurrence is seed-dependent, which is why it looks random.

Fix
---
1. ``drop_invalid_tokens`` drops out-of-vocabulary ids (with a warning) instead
   of forwarding them, in both bundled chatterbox engines.
2. ``flow.py`` clamps the embedding index at both ends, as a hard backstop. The
   node's newer ``chatterbox_official_23lang`` engine already ships this; the
   older ``chatterbox`` engine does not.

Usage
-----
    python patches/tts_audio_suite_s3gen_token_bounds.py            # apply
    python patches/tts_audio_suite_s3gen_token_bounds.py --check    # report only

Re-runnable: already-patched files are left untouched. Re-apply after updating
or reinstalling the node, since ``custom_nodes/`` is not tracked by this repo.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

NODE_RELATIVE_PATH = pathlib.PurePosixPath("ComfyUI/custom_nodes/tts_audio_suite")

# --- Fix 1: bound the embedding index at both ends -------------------------
CLAMP_OLD = "self.input_embedding(torch.clamp(token, min=0)) * mask"
CLAMP_NEW = (
    "self.input_embedding(\n"
    "            torch.clamp(token, min=0, max=self.input_embedding.num_embeddings - 1)\n"
    "        ) * mask"
)
# A file is considered fixed once no unbounded clamp remains; the node's
# 23lang engine already ships its own upper bound and must not be rewritten.
clamp_is_fixed = lambda text: CLAMP_OLD not in text  # noqa: E731

# --- Fix 2: drop out-of-vocabulary ids instead of forwarding them -----------
FILTER_ANCHOR = "    x = x[s: e]\n    return x"
filter_is_fixed = lambda text: "invalid = x >= SPEECH_VOCAB_SIZE" in text  # noqa: E731
FILTER_NEW = '''    x = x[s: e]

    # T3's speech head spans `speech_tokens_dict_size` (8194) while S3Gen's
    # input embedding only has SPEECH_VOCAB_SIZE (6561) rows, so a sampled id
    # above the vocab that is not exactly SoS/EoS reaches the embedding gather
    # and trips a device-side "index out of bounds" assert (which surfaces
    # later, at the next CUDA sync, with a misleading traceback).
    invalid = x >= SPEECH_VOCAB_SIZE
    if invalid.any():
        logging.warning(
            "Dropping %d out-of-vocabulary speech token(s) (max id %d, vocab %d)",
            int(invalid.sum()), int(x.max()), SPEECH_VOCAB_SIZE,
        )
        x = x[~invalid]
    return x'''

CLAMP_TARGETS = (
    "engines/chatterbox/models/s3gen/flow.py",
    "engines/chatterbox_official_23lang/models/s3gen/flow.py",
)
FILTER_TARGETS = (
    "engines/chatterbox/models/s3tokenizer/__init__.py",
    "engines/chatterbox_official_23lang/models/s3tokenizer/__init__.py",
)


def read(path: pathlib.Path):
    """Return (text normalised to LF, the file's own newline style)."""
    # newline="" keeps the bytes as authored. pathlib grew that argument
    # only in 3.13 and ComfyUI still ships 3.10, so use open() directly.
    with open(path, "r", encoding="utf-8", newline="") as handle:
        raw = handle.read()
    crlf, lf = chr(13) + chr(10), chr(10)
    return raw.replace(crlf, lf), (crlf if crlf in raw else lf)


def write(path: pathlib.Path, text: str, newline: str) -> None:
    # Write back in the file's original newline style, so the diff stays
    # confined to the lines that actually changed.
    with open(path, "w", encoding="utf-8", newline=newline) as handle:
        handle.write(text)
    # Drop any stale bytecode so the next ComfyUI start cannot reuse it.
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for compiled in cache.glob(f"{path.stem}.*.pyc"):
            compiled.unlink()


def apply_clamp(text: str) -> tuple[str, int]:
    count = text.count(CLAMP_OLD)
    return (text.replace(CLAMP_OLD, CLAMP_NEW) if count else text), count


def apply_filter(text: str) -> tuple[str, int]:
    if FILTER_ANCHOR not in text:
        return text, 0
    text = text.replace(FILTER_ANCHOR, FILTER_NEW, 1)
    if not text.startswith("import logging"):
        text = "import logging\n\n" + text
    return text, 1


def process(path: pathlib.Path, is_fixed, transform, check: bool) -> str:
    """Return one of: 'patched', 'already', 'skipped', 'failed'."""
    if not path.is_file():
        print(f"  SKIP    {path.name}: not present")
        return "skipped"

    text, newline = read(path)
    if is_fixed(text):
        print(f"  OK      {path.name}: already patched")
        return "already"

    updated, count = transform(text)
    if not count:
        print(f"  FAILED  {path.name}: expected code not found (node version changed?)")
        return "failed"

    if check:
        print(f"  NEEDED  {path.name}: {count} site(s) would be patched")
        return "failed"

    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    write(path, updated, newline)
    print(f"  PATCHED {path.name}: {count} site(s) (backup: {backup.name})")
    return "patched"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--project-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent,
        help="Repository root holding ComfyUI/ (default: this file's parent repo).",
    )
    parser.add_argument(
        "--node-root",
        type=pathlib.Path,
        default=None,
        help="Patch this tts_audio_suite checkout directly, ignoring --project-root.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report status without writing; exit 1 if the fix is not applied.",
    )
    args = parser.parse_args()

    node_root = (
        args.node_root
        if args.node_root is not None
        else args.project_root.joinpath(*NODE_RELATIVE_PATH.parts)
    )
    if not node_root.is_dir():
        print(f"tts_audio_suite is not installed at {node_root} - nothing to do.")
        return 0

    print(f"tts_audio_suite: {node_root}")
    results: list[str] = []

    print("drop_invalid_tokens - drop out-of-vocabulary speech tokens:")
    for relative in FILTER_TARGETS:
        results.append(process(node_root / relative, filter_is_fixed, apply_filter, args.check))

    print("flow.py - clamp the input_embedding index at both ends:")
    for relative in CLAMP_TARGETS:
        results.append(process(node_root / relative, clamp_is_fixed, apply_clamp, args.check))

    if "failed" in results:
        print("\nIncomplete: review the entries above.")
        return 1
    if "patched" in results:
        print("\nDone - restart ComfyUI to load the patched modules.")
    else:
        print("\nNothing to do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
