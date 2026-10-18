#!/usr/bin/env python3
"""Load Pixal3D checkpoints without allocating duplicate initialized weights."""

from __future__ import annotations

import argparse
from pathlib import Path


OLD = """    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.load_state_dict(load_file(model_file), strict=False)"""
CURRENT_NEW = """    import torch
    from safetensors import safe_open
    model_class = __getattr__(config['name'])
    try:
        with torch.device('meta'):
            model = model_class(**config['args'], **kwargs)
        state_dict = model.state_dict()
        with safe_open(model_file, framework='pt', device='cpu') as checkpoint:
            if set(state_dict) != set(checkpoint.keys()):
                raise ValueError('Pixal3D checkpoint keys do not match the model')
            for key in checkpoint.keys():
                tensor = checkpoint.get_tensor(key)
                if tensor.dtype != state_dict[key].dtype:
                    tensor = tensor.to(dtype=state_dict[key].dtype)
                state_dict[key] = tensor
        model.load_state_dict(state_dict, strict=False, assign=True)
    except OSError as exc:
        if getattr(exc, 'winerror', None) == 1455:
            raise OSError(
                1455,
                f'Windows ran out of commit memory loading Pixal3D checkpoint {model_file}. '
                'Close memory-heavy applications or increase the Windows paging file, '
                'then restart ComfyUI.',
            ) from exc
        model = model_class(**config['args'], **kwargs)
        model.load_state_dict(load_file(model_file), strict=False)
    except Exception:
        model = model_class(**config['args'], **kwargs)
        model.load_state_dict(load_file(model_file), strict=False)"""
NEW = """    import numpy as np
    import torch
    from safetensors import safe_open
    model_class = __getattr__(config['name'])
    try:
        with torch.device('meta'):
            model = model_class(**config['args'], **kwargs)
        state_dict = model.state_dict()
        with safe_open(model_file, framework='np', device='cpu') as checkpoint:
            if set(state_dict) != set(checkpoint.keys()):
                raise ValueError('Pixal3D checkpoint keys do not match the model')
            for key in checkpoint.keys():
                # Copy the mmap-backed NumPy view before creating a Torch tensor.
                # PyTorch's safetensors storage path can access-violate on Windows.
                tensor = torch.from_numpy(np.array(checkpoint.get_tensor(key), copy=True))
                if tensor.dtype != state_dict[key].dtype:
                    tensor = tensor.to(dtype=state_dict[key].dtype)
                state_dict[key] = tensor
        model.load_state_dict(state_dict, strict=False, assign=True)
    except OSError as exc:
        if getattr(exc, 'winerror', None) == 1455:
            raise OSError(
                1455,
                f'Windows ran out of commit memory loading Pixal3D checkpoint {model_file}. '
                'Close memory-heavy applications or increase the Windows paging file, '
                'then restart ComfyUI.',
            ) from exc
        raise"""
PREVIOUS_NEW = """    import torch
    model_class = __getattr__(config['name'])
    try:
        state_dict = load_file(model_file)
    except OSError as exc:
        if getattr(exc, 'winerror', None) == 1455:
            raise OSError(
                1455,
                f'Windows ran out of commit memory loading Pixal3D checkpoint {model_file}. '
                'Close memory-heavy applications or increase the Windows paging file, '
                'then restart ComfyUI.',
            ) from exc
        raise
    try:
        with torch.device('meta'):
            model = model_class(**config['args'], **kwargs)
        if set(model.state_dict()) != set(state_dict):
            raise ValueError('Pixal3D checkpoint keys do not match the model')
        model.load_state_dict(state_dict, strict=False, assign=True)
    except Exception:
        model = model_class(**config['args'], **kwargs)
        model.load_state_dict(state_dict, strict=False)"""
PREVIOUS_META = """    import torch
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
OLD_FALLBACK = """            except Exception as e:
                _models[k] = models.from_pretrained(v)"""
NEW_FALLBACK = """            except Exception as e:
                if isinstance(e, OSError) and getattr(e, 'winerror', None) == 1455:
                    raise
                _models[k] = models.from_pretrained(v)"""
CURRENT_FALLBACK = """            except Exception:
                # A local snapshot's relative checkpoint paths must never be
                # retried as Hugging Face repo IDs (e.g. "ckpts/foo"). That
                # hides the actual local load failure, including Windows
                # commit/pagefile exhaustion, behind a misleading HTTP 401.
                # The fallback is only valid when the caller supplied a repo.
                if is_local or os.path.isabs(path):
                    raise
                _models[k] = models.from_pretrained(v)"""


def replace_once(path: Path, old: str, new: str) -> bool:
    source = path.read_text(encoding="utf-8")
    if new in source:
        return False
    if source.count(old) != 1:
        raise RuntimeError(f"Could not safely patch Pixal3D model loading in {path}")
    path.write_bytes(source.replace(old, new, 1).encode("utf-8"))
    return True


def patch_pixal3d(comfyui: Path) -> list[Path]:
    pixal3d = (
        comfyui / "custom_nodes" / "comfyui-pixal3d" / "_pixal3d_src" / "pixal3d"
    )
    model_loader = pixal3d / "models" / "__init__.py"
    pipeline_loader = pixal3d / "pipelines" / "base.py"
    if not model_loader.is_file():
        return []

    patched = []
    source = model_loader.read_text(encoding="utf-8")
    if NEW not in source:
        previous = next(
            (candidate for candidate in (OLD, CURRENT_NEW, PREVIOUS_NEW, PREVIOUS_META)
             if source.count(candidate) == 1),
            None,
        )
        if previous is None:
            raise RuntimeError(
                f"Could not safely patch Pixal3D model loading in {model_loader}"
            )
        model_loader.write_bytes(source.replace(previous, NEW, 1).encode("utf-8"))
        patched.append(model_loader)
    if pipeline_loader.is_file():
        pipeline_source = pipeline_loader.read_text(encoding="utf-8")
        if NEW_FALLBACK not in pipeline_source and CURRENT_FALLBACK not in pipeline_source:
            if pipeline_source.count(OLD_FALLBACK) != 1:
                raise RuntimeError(
                    f"Could not safely patch Pixal3D pipeline fallback in {pipeline_loader}"
                )
            pipeline_loader.write_bytes(
                pipeline_source.replace(OLD_FALLBACK, NEW_FALLBACK, 1).encode("utf-8")
            )
            patched.append(pipeline_loader)
    return patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", type=Path, required=True)
    args = parser.parse_args()
    for path in patch_pixal3d(args.comfyui.resolve()):
        print(f"[pixal3d-meta-loading] Patched {path}")


if __name__ == "__main__":
    main()
