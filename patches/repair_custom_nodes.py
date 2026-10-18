#!/usr/bin/env python3
"""Apply repeatable Windows compatibility repairs for installed ComfyUI nodes."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys


def replace_once(path: Path, old: str, new: str) -> bool:
    if not path.is_file():
        return False
    source = path.read_text(encoding="utf-8")
    if new in source:
        return False
    count = source.count(old)
    if count == 0:
        # Nodes may already contain a manual equivalent fix.
        return False
    if count != 1:
        raise RuntimeError(f"Expected exactly one patch target in {path}; found {count}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8", newline="")
    print(f"[node-repair] Patched {path}")
    return True


def repair_pixal3d(comfyui: Path) -> None:
    requirements = comfyui / "custom_nodes" / "comfyui-pixal3d" / "requirements.txt"
    # pyrender pins PyOpenGL 3.1.0, so the node's >=3.1.7 constraint made the
    # setup's final `pip check` fail immediately after a successful install.
    replace_once(
        requirements,
        "PyOpenGL>=3.1.7",
        "PyOpenGL==3.1.0",
    )
    source = comfyui / "custom_nodes" / "comfyui-pixal3d" / "_pixal3d_src" / "pixal3d"
    features = source / "trainers" / "flow_matching" / "mixins" / "image_conditioned_proj.py"
    replace_once(
        features,
        """        for layer_module in self.model.layer:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        return F.layer_norm(hidden_states, hidden_states.shape[-1:])""",
        """        # Transformers moved DINOv3's encoder blocks under model.model.
        layers = getattr(self.model, \"layer\", None)
        if layers is None:
            layers = self.model.model.layer

        for layer_module in layers:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        # Use DINOv3's learned final normalization, as its forward() does.
        norm = getattr(self.model, \"norm\", None)
        return norm(hidden_states) if norm is not None else F.layer_norm(
            hidden_states, hidden_states.shape[-1:]
        )""",
    )

    stages = comfyui / "custom_nodes" / "comfyui-pixal3d" / "nodes" / "pixal3d_stages.py"
    old_builder = """    m = DinoV3ProjFeatureExtractor(**config)
    m.eval()"""
    new_builder = """    # Transformers 5.x's async safetensors copy can access-violate on Windows.
    # Force its supported synchronous path while loading each DINOv3 model.
    import transformers.core_model_loading as _core_model_loading
    _original_env_check = _core_model_loading.is_env_variable_true
    _core_model_loading.is_env_variable_true = lambda name: (
        name == \"HF_DEACTIVATE_ASYNC_LOAD\" or _original_env_check(name)
    )
    try:
        m = DinoV3ProjFeatureExtractor(**config)
    finally:
        _core_model_loading.is_env_variable_true = _original_env_check
    m.eval()"""
    replace_once(stages, old_builder, new_builder)


def repair_optional_nodes(comfyui: Path) -> None:
    nodes = comfyui / "custom_nodes"
    stereo = nodes / "comfystereo" / "native_viewer" / "constants.py"
    replace_once(
        stereo,
        "except ImportError as e:",
        "except (ImportError, AttributeError) as e:",
    )

    loop_utils = nodes / "comfyui-loop" / "utils" / "loop_path_utils.py"
    replace_once(
        loop_utils,
        """        try:
            shutil.copytree(source, dest, dirs_exist_ok=True)
        except FileNotFoundError:
            print(f\"Error: Cannot find source directory\")""",
        """        if not source.is_dir():
            # The preview-icons companion node is optional.
            return
        try:
            shutil.copytree(source, dest, dirs_exist_ok=True)
        except FileNotFoundError:
            return""",
    )

    # Some kiui wheels ship an empty typing.py even though their consumers rely
    # on star-imported typing and tensor names across multiple node packs.
    site_packages = comfyui / ".venv" / "Lib" / "site-packages"
    kiui_typing = site_packages / "kiui" / "typing.py"
    if kiui_typing.is_file():
        source = kiui_typing.read_text(encoding="utf-8")
        if "from typing import *" not in source:
            with kiui_typing.open("a", encoding="utf-8", newline="") as stream:
                stream.write(
                    "\nfrom typing import *\nfrom numpy import ndarray\nfrom torch import Tensor\n"
                )
            print(f"[node-repair] Restored shared typing exports in {kiui_typing}")

    kiui_op = site_packages / "kiui" / "op.py"
    replace_once(
        kiui_op,
        "from kiui.typing import *",
        "from kiui.typing import *\nfrom typing import Literal, Optional, Sequence, Union\nfrom torch import Tensor\nfrom numpy import ndarray",
    )


def install_missing_optional_dependencies(comfyui: Path) -> None:
    """Install small runtime dependencies explicitly reported missing in logs."""
    modules = {
        "reportlab": "reportlab",
        "imageio_ffmpeg": "imageio-ffmpeg",
        "OpenEXR": "OpenEXR",
        "meshlib": "meshlib",
        "ultralytics": "ultralytics",
    }
    missing = [package for module, package in modules.items() if importlib.util.find_spec(module) is None]
    if not missing:
        return
    python = Path(sys.executable)
    constraints = comfyui / ".venv" / "pip-constraints.txt"
    command = [str(python), "-m", "pip", "install"]
    if constraints.is_file():
        command += ["-c", str(constraints)]
    command += missing
    print(f"[node-repair] Installing missing node dependencies: {', '.join(missing)}")
    subprocess.run(command, check=True)


def repair_pixal3d_dependencies(comfyui: Path) -> None:
    """Run the node's pinned dependency installer when its runtime is incomplete.

    Pixal3D includes CUDA/Torch ABI-specific Windows wheels such as o_voxel.
    They must come from that node's requirements file, not generic PyPI
    dependency guessing. Check in the target ComfyUI interpreter so this also
    works when the repair script itself is invoked by a launcher Python.
    """
    node = comfyui / "custom_nodes" / "comfyui-pixal3d"
    installer = node / "install.py"
    requirements = node / "requirements.txt"
    if not installer.is_file() or not requirements.is_file():
        return

    python = comfyui / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        python = Path(sys.executable)
    modules = [
        "moge", "utils3d", "o_voxel", "flex_gemm", "cumesh",
        "nvdiffrast", "triton", "natten", "flash_attn", "pipeline", "moderngl", "gradio",
    ]
    probe = (
        "import importlib.util, json; "
        f"mods = {modules!r}; "
        "print(json.dumps([m for m in mods if importlib.util.find_spec(m) is None]))"
    )
    result = subprocess.run(
        [str(python), "-c", probe], check=True, capture_output=True, text=True
    )
    import json
    missing = json.loads(result.stdout.strip().splitlines()[-1])
    if not missing:
        return

    print(
        "[node-repair] Pixal3D runtime is missing "
        f"{', '.join(missing)}; installing its pinned Windows requirements."
    )

    # The upstream requirements intentionally use --no-deps because MoGe
    # declares a conflicting utils3d source dependency. Install those node
    # requirements, then resolve the small set of omitted runtime deps.
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", "-r", str(requirements)],
        check=True,
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "pipeline", "moderngl", "gradio", "PyOpenGL==3.1.0"],
        check=True,
    )

    install_args = [str(python), str(installer)]
    if (node / "_pixal3d_src" / "pixal3d").is_dir():
        install_args.append("--skip-clone")
    install_args.append("--skip-deps")
    subprocess.run(
        install_args,
        cwd=str(node),
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--install-dependencies", action="store_true")
    args = parser.parse_args()
    comfyui = args.comfyui.resolve()
    repair_pixal3d(comfyui)
    repair_optional_nodes(comfyui)
    if args.install_dependencies:
        install_missing_optional_dependencies(comfyui)
        repair_pixal3d_dependencies(comfyui)


if __name__ == "__main__":
    main()
