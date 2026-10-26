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
PREVIOUS_NUMPY = """    import numpy as np
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
PREVIOUS_PREAD = PREVIOUS_NUMPY.replace(
    "        with safe_open(model_file, framework='np', device='cpu') as checkpoint:",
    "        # Stream tensor bytes instead of committing a full-file mmap on Windows.\n"
    "        with safe_open(model_file, framework='np', device='cpu', backend='pread') as checkpoint:",
).replace("Copy the mmap-backed NumPy view", "Copy the NumPy buffer")
PREVIOUS_STREAMED = """    import torch
    from .checkpoint_reader import load_checkpoint
    model_class = __getattr__(config['name'])
    try:
        with torch.device('meta'):
            model = model_class(**config['args'], **kwargs)
        state_dict = load_checkpoint(model_file, model.state_dict())
        model.load_state_dict(state_dict, strict=True, assign=True)
    except OSError as exc:
        if getattr(exc, 'winerror', None) == 1455 or getattr(exc, 'errno', None) == 1455:
            raise OSError(
                1455,
                f'Windows ran out of commit memory loading Pixal3D checkpoint {model_file}. '
                'Close memory-heavy applications or increase the Windows paging file, '
                'then restart ComfyUI.',
            ) from exc
        raise"""
NEW = PREVIOUS_STREAMED.replace(
    "    model_class = __getattr__(config['name'])",
    "    checkpoint_device = kwargs.pop('_checkpoint_device', 'cpu')\n"
    "    model_class = __getattr__(config['name'])",
).replace(
    "load_checkpoint(model_file, model.state_dict())",
    "load_checkpoint(model_file, model.state_dict(), device=checkpoint_device)",
)
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

MOGE_OLD = "    m = MoGeModel.from_pretrained(MOGE_MODEL_NAME).to(device)"
MOGE_NEW = """    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download

    checkpoint_path = pathlib.Path(MOGE_MODEL_NAME)
    if not checkpoint_path.is_file():
        checkpoint_path = hf_hub_download(MOGE_MODEL_NAME, filename="model.pt")
    # Map checkpoint storage and assign it directly instead of allocating a
    # second ViT-L worth of CPU weights while Windows is low on commit memory.
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True,
    )
    # Keep buffers and constructor scalars on CPU: DINOv2 calls .item() on
    # its drop-path schedule, which cannot be evaluated on the meta device.
    with init_empty_weights(include_buffers=False):
        m = MoGeModel(**checkpoint["model_config"])
    m.load_state_dict(checkpoint["model"], strict=True, assign=True)
    del checkpoint
    m = m.to(device)"""

DINO_OLD = """    # Transformers 5.x's async safetensors copy can access-violate on Windows.
    # Force its supported synchronous path while loading each DINOv3 model.
    import transformers.core_model_loading as _core_model_loading
    _original_env_check = _core_model_loading.is_env_variable_true
    _core_model_loading.is_env_variable_true = lambda name: (
        name == "HF_DEACTIVATE_ASYNC_LOAD" or _original_env_check(name)
    )
    try:
        m = DinoV3ProjFeatureExtractor(**config, backbone=backbone)
    finally:
        _core_model_loading.is_env_variable_true = _original_env_check"""
DINO_NEW = """    if backbone is None:
        from accelerate import init_empty_weights
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        from pixal3d.models.checkpoint_reader import load_checkpoint

        paths = {}
        for filename in ("config.json", "model.safetensors"):
            path = try_to_load_from_cache(config["model_name"], filename)
            if not isinstance(path, str):
                path = hf_hub_download(config["model_name"], filename=filename)
            paths[filename] = path
        model_config = DINOv3ViTConfig.from_json_file(paths["config.json"])
        # Keep computed RoPE buffers real while leaving checkpoint parameters empty.
        with init_empty_weights(include_buffers=False):
            backbone = DINOv3ViTModel(model_config)
        expected = backbone.state_dict()
        # Transformers 5 moved checkpoint layer.* keys under model.layer.*.
        checkpoint_keys = {
            (key.removeprefix("model.") if key.startswith("model.layer.") else key): key
            for key in expected
        }
        # Stream directly to the selected device, avoiding whole-file mappings
        # and a complete CPU copy of the weights under Windows commit pressure.
        state = load_checkpoint(
            paths["model.safetensors"],
            {key: expected[target] for key, target in checkpoint_keys.items()},
            device=mm.get_torch_device(),
        )
        backbone.load_state_dict(
            {checkpoint_keys[key]: value for key, value in state.items()},
            strict=True, assign=True,
        )
        backbone.to(mm.get_torch_device())
    m = DinoV3ProjFeatureExtractor(**config, backbone=backbone)"""


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
    stages = comfyui / "custom_nodes" / "comfyui-pixal3d" / "nodes" / "pixal3d_stages.py"
    if stages.is_file() and replace_once(stages, MOGE_OLD, MOGE_NEW):
        patched.append(stages)
    if stages.is_file() and "def _build_image_cond_model(" in stages.read_text(encoding="utf-8"):
        if replace_once(stages, DINO_OLD, DINO_NEW) and stages not in patched:
            patched.append(stages)
    reader_source = Path(__file__).with_name("pixal3d_checkpoint_reader.py").read_text(encoding="utf-8")
    reader_path = model_loader.with_name("checkpoint_reader.py")
    if not reader_path.is_file() or reader_path.read_text(encoding="utf-8") != reader_source:
        reader_path.write_text(reader_source, encoding="utf-8", newline="")
        patched.append(reader_path)
    source = model_loader.read_text(encoding="utf-8")
    if NEW not in source:
        previous = next(
            (candidate for candidate in (OLD, CURRENT_NEW, PREVIOUS_NEW, PREVIOUS_META, PREVIOUS_NUMPY, PREVIOUS_PREAD, PREVIOUS_STREAMED)
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
        recognized_fallbacks = (
            candidate.replace("models.from_pretrained(v)", loader)
            for candidate in (NEW_FALLBACK, CURRENT_FALLBACK)
            for loader in ("models.from_pretrained(v)", "load_model(v)")
        )
        if not any(candidate in pipeline_source for candidate in recognized_fallbacks):
            old_fallback = next(
                (candidate for candidate in (OLD_FALLBACK, OLD_FALLBACK.replace(
                    "models.from_pretrained(v)", "load_model(v)"
                )) if pipeline_source.count(candidate) == 1),
                None,
            )
            if old_fallback is None:
                raise RuntimeError(
                    f"Could not safely patch Pixal3D pipeline fallback in {pipeline_loader}"
                )
            new_fallback = NEW_FALLBACK
            if "load_model(v)" in old_fallback:
                new_fallback = new_fallback.replace("models.from_pretrained(v)", "load_model(v)")
            pipeline_loader.write_bytes(
                pipeline_source.replace(old_fallback, new_fallback, 1).encode("utf-8")
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
