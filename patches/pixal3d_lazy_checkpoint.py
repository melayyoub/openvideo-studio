"""Payload copied into Pixal3D's models package for stage-wise disk offload."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from torch import nn


class LazyCheckpoint(nn.Module):
    """Materialize one checkpoint on demand; release it on CPU offload.

    Frozen inference weights can be reloaded from the checkpoint instead of
    copied from CUDA into another large CPU allocation. Unused stages consume
    only their JSON metadata, so low-VRAM mode also bounds host commit memory.
    """

    def __init__(self, path: str):
        super().__init__()
        checkpoint = Path(path)
        if not Path(f"{checkpoint}.safetensors").is_file():
            raise FileNotFoundError(f"Missing local Pixal3D checkpoint: {checkpoint}")
        with Path(f"{checkpoint}.json").open(encoding="utf-8") as config_file:
            self._checkpoint_args = json.load(config_file)["args"]
        self._checkpoint_path = str(checkpoint)
        self._runtime_options = {}

    def _materialize(self, device="cpu"):
        loaded = self._modules.get("_loaded_model")
        if loaded is None:
            from . import from_pretrained

            loaded = from_pretrained(self._checkpoint_path, _checkpoint_device=device)
            loaded.train(self.training)
            for name, value in self._runtime_options.items():
                setattr(loaded, name, value)
            self.add_module("_loaded_model", loaded)
        return loaded

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith("_"):
                raise
            loaded = self._modules.get("_loaded_model")
            if loaded is not None:
                return getattr(loaded, name)
            # The sampler queries these dimensions before moving the stage
            # to CUDA. Reading metadata must not allocate the checkpoint.
            if name in ("resolution", "in_channels", "out_channels") and name in self._checkpoint_args:
                return self._checkpoint_args[name]
            return getattr(self._materialize(), name)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "low_vram":
            self._runtime_options[name] = value
            loaded = self._modules.get("_loaded_model")
            if loaded is not None:
                setattr(loaded, name, value)

    def forward(self, *args, **kwargs):
        return self._materialize()(*args, **kwargs)

    def to(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if isinstance(device, (str, torch.device)) and torch.device(device).type == "cpu":
            return self.cpu()
        target = device if isinstance(device, (str, torch.device)) else "cpu"
        self._materialize(target).to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        if isinstance(device, int):
            device = torch.device("cuda", device)
        return self.to("cuda" if device is None else device)

    def cpu(self):
        loaded = self._modules.get("_loaded_model")
        if loaded is not None:
            parameter = next(loaded.parameters(), None)
            device = parameter.device if parameter is not None else None
            if device is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            # Release CUDA storage rather than allocating a CPU copy. The
            # next use reloads the immutable checkpoint and runtime options.
            del self._modules["_loaded_model"]
            del loaded
            del parameter
            gc.collect()
            if device is not None and device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
        return self
