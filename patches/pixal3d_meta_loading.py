#!/usr/bin/env python3
"""Load Pixal3D checkpoints without allocating duplicate initialized weights."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = """    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.load_state_dict(load_file(model_file), strict=False)"""
NEW = """    import torch
    model_class = __getattr__(config['name'])
    state_dict = load_file(model_file)
    try:
        with torch.device('meta'):
            model = model_class(**config['args'], **kwargs)
        if set(model.state_dict()) != set(state_dict):
            raise ValueError('Pixal3D checkpoint keys do not match the model')
        model.load_state_dict(state_dict, strict=False, assign=True)
    except Exception:
        model = model_class(**config['args'], **kwargs)
        model.load_state_dict(state_dict, strict=False)"""


def patch_pixal3d(comfyui: Path) -> Path | None:
    path = (
        comfyui
        / "custom_nodes"
        / "comfyui-pixal3d"
        / "_pixal3d_src"
        / "pixal3d"
        / "models"
        / "__init__.py"
    )
    if not path.is_file():
        return None

    source = path.read_text(encoding="utf-8")
    if NEW in source:
        return None
    if source.count(OLD) != 1:
        raise RuntimeError(f"Could not safely patch Pixal3D model loading in {path}")
    path.write_bytes(source.replace(OLD, NEW, 1).encode("utf-8"))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", type=Path, required=True)
    args = parser.parse_args()
    path = patch_pixal3d(args.comfyui.resolve())
    if path:
        print(f"[pixal3d-meta-loading] Patched {path}")


if __name__ == "__main__":
    main()
