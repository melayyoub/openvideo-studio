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
    replace_once(features, "self.naf_target_size = (512, 512)", "self.naf_target_size = (256, 256)")
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
    replace_once(
        features,
        """                if _naf_fp16:
                    self.naf_model.to(dtype=torch.float16)""",
        """                if _naf_fp16:
                    # Use a 512px conditioning feature map on RTX 50 series;
                    # 1024px triggers an oversized NAF interpolation request.
                    self.naf_target_size = (256, 256)
                    self.naf_model.to(dtype=torch.float16)""",
    )
    replace_once(
        features,
        """                    lr_features_bchw = lr_features_bchw.to(dtype=torch.float16)
                hr_features = self.naf_model(""",
        """                    lr_features_bchw = lr_features_bchw.to(dtype=torch.float16)
                    try:
                        torch.cuda.set_per_process_memory_fraction(0.85, device=image_for_naf.device)
                    except Exception:
                        pass
                hr_features = self.naf_model(""",
    )

    # NAF's CrossAttention used to upsample fp32 feature maps to 1024px and
    # only cast after interpolation. That creates a multi-GiB temporary even
    # when the caller and NAF weights were already moved to fp16. Cast before
    # interpolation so the peak allocation follows the active attention dtype.
    naf_attention = Path.home() / ".cache" / "torch" / "hub" / "valeoai_NAF_main" / "src" / "layers" / "attentions.py"
    replace_once(
        naf_attention,
        """    def _resize(self, x, size, dtype):
        x = F.interpolate(x, size=size, mode=\"nearest-exact\")
        x = rearrange(x, \"b (n d) h w -> b h w n d\", n=self.num_heads)
        return x.to(dtype)""",
        """    def _resize(self, x, size, dtype):
        if x.dtype != dtype:
            x = x.to(dtype)
        x = F.interpolate(x, size=size, mode=\"nearest-exact\")
        return rearrange(x, \"b (n d) h w -> b h w n d\", n=self.num_heads)""",
    )
    replace_once(
        naf_attention,
        """    def forward(self, q, k, v, image=None, return_weights=False, **kwargs):
        hq, wq = q.shape[-2:]""",
        """    def forward(self, q, k, v, image=None, return_weights=False, **kwargs):
        # The NAF image encoder can emit fp32 queries even when its source
        # feature map was cast to fp16. Keep RTX 50 series high-resolution
        # attention at fp16 so the 1024px upsample fits ComfyUI's memory cap.
        if q.is_cuda:
            try:
                if torch.cuda.get_device_capability(q.device)[0] >= 12:
                    q = q.to(torch.float16)
            except Exception:
                pass
        hq, wq = q.shape[-2:]""",
    )
    replace_once(
        features,
        """    grid = queries_ndc.view(B, K, 1, 2)  # (B, K, 1, 2)""",
        """    # grid_sample requires the grid and feature map to share a dtype.
    grid = queries_ndc.to(dtype=fmap.dtype).view(B, K, 1, 2)  # (B, K, 1, 2)""",
    )
    replace_once(
        features,
        """            # --- High-resolution branch (NAF): upsample then sample ---
            if self.use_naf_upsample:
                self._load_naf()""",
        """            # --- High-resolution branch (NAF): upsample then sample ---
            if self.use_naf_upsample:
                # DINO is no longer needed once the compact projected LR
                # features have been sampled. Free its weights before NAF's
                # 1024px feature upsample to avoid overlapping both models.
                self.model.cpu()
                if image.is_cuda:
                    torch.cuda.empty_cache()
                self._load_naf()""",
    )
    replace_once(
        features,
        """                lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2)  # [B, D, h, w]
                hr_features = self.naf_model(
                    image_for_naf, lr_features_bchw, self.naf_target_size
                )  # [B, D, H', W']""",
        """                lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2)  # [B, D, h, w]
                # The 1024px texture NAF can create a 4 GiB feature map in
                # fp32. RTX 50 series ComfyUI runs with a 16 GiB per-process
                # budget, so use fp16 for NAF and its grid sampling there.
                _naf_fp16 = False
                if image_for_naf.is_cuda:
                    try:
                        _naf_fp16 = torch.cuda.get_device_capability(image_for_naf.device)[0] >= 12
                    except Exception:
                        pass
                if _naf_fp16:
                    # A full 1024x1024 NAF feature tensor can exceed the
                    # RTX 5090 ComfyUI process allocator's single-allocation
                    # ceiling. Keep the balanced Pixal3D model/texture size,
                    # while using a 256px conditioning feature map.
                    self.naf_target_size = (256, 256)
                    self.naf_model.to(dtype=torch.float16)
                    image_for_naf = image_for_naf.to(dtype=torch.float16)
                    lr_features_bchw = lr_features_bchw.to(dtype=torch.float16)
                    # ComfyUI can reserve only half of a 32GB RTX 5090 for
                    # this process by default. Pixal3D's single NAF upsample
                    # needs a contiguous 2GB allocation; raise the allocator
                    # ceiling while low-VRAM mode has offloaded other stages.
                    try:
                        torch.cuda.set_per_process_memory_fraction(
                            0.85, device=image_for_naf.device
                        )
                    except Exception:
                        pass
                hr_features = self.naf_model(
                    image_for_naf, lr_features_bchw, self.naf_target_size
                )  # [B, D, H', W']""",
    )
    replace_once(
        features,
        """                z_proj_hr = self.proj_grid(
                    hr_features,
                    camera_angle_x,
                    distance,
                    mesh_scale,
                    transform_matrix,
                    BHWC=False  # hr_features is [B, C, H', W']
                )""",
        """                z_proj_hr = self.proj_grid(
                    hr_features,
                    camera_angle_x,
                    distance,
                    mesh_scale,
                    transform_matrix,
                    BHWC=False  # hr_features is [B, C, H', W']
                )
                if z_proj_hr.dtype != z_proj_lr.dtype:
                    z_proj_hr = z_proj_hr.to(dtype=z_proj_lr.dtype)""",
    )

    # Meta-device construction leaves these plain (non-buffer) positional
    # frequency tensors unmaterialized. Recreate them lazily before moving to
    # CUDA; state_dict assignment cannot supply them because they are not
    # checkpoint weights.
    sparse_rope = source / "modules" / "sparse" / "attention" / "rope.py"
    replace_once(
        sparse_rope,
        """        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)""",
        """        if self.freqs.is_meta:
            freq = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
            self.freqs = self.rope_freq[0] / (self.rope_freq[1] ** freq)
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)""",
    )

    dense_rope = source / "modules" / "attention" / "rope.py"
    replace_once(
        dense_rope,
        """        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)""",
        """        if self.freqs.is_meta:
            freq = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
            self.freqs = self.rope_freq[0] / (self.rope_freq[1] ** freq)
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)""",
    )

    abs_pos = source / "modules" / "transformer" / "blocks.py"
    replace_once(
        abs_pos,
        """        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)""",
        """        if self.freqs.is_meta:
            freq = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
            self.freqs = 1.0 / (10000 ** freq)
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)""",
    )

    sparse_config = source / "modules" / "sparse" / "config.py"
    replace_once(
        sparse_config,
        "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'flash_attn_4']:",
        "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'flash_attn_4', 'sdpa']:",
    )
    replace_once(
        sparse_config,
        "Literal['xformers', 'flash_attn', 'flash_attn_3', 'flash_attn_4']",
        "Literal['xformers', 'flash_attn', 'flash_attn_3', 'flash_attn_4', 'sdpa']",
    )

    sparse_attention = source / "modules" / "sparse" / "attention" / "full_attn.py"
    replace_once(
        sparse_attention,
        """    elif config.ATTN == 'flash_attn':""",
        """    elif config.ATTN == 'sdpa':
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        q_parts = torch.split(q, q_seqlen, dim=0)
        k_parts = torch.split(k, kv_seqlen, dim=0)
        v_parts = torch.split(v, kv_seqlen, dim=0)
        out = torch.cat([
            sdpa(qi.permute(1, 0, 2).unsqueeze(0),
                 ki.permute(1, 0, 2).unsqueeze(0),
                 vi.permute(1, 0, 2).unsqueeze(0))
            .squeeze(0).permute(1, 0, 2)
            for qi, ki, vi in zip(q_parts, k_parts, v_parts)
        ], dim=0)
    elif config.ATTN == 'flash_attn':""",
    )

    stages = comfyui / "custom_nodes" / "comfyui-pixal3d" / "nodes" / "pixal3d_stages.py"
    replace_once(
        stages,
        """    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )""",
        """    _glb_args = dict(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    try:
        glb = o_voxel.postprocess.to_glb(**_glb_args)
    except RuntimeError as _error:
        # CuMesh's optional remesh can exhaust CUDA memory on unusually dense
        # shapes. Retry that mesh with the non-remesh UV path; keep the fast
        # path for normal assets.
        if not remesh or "out of memory" not in str(_error).lower():
            raise
        log.warning("[Pixal3D] CuMesh remesh ran out of memory; retrying GLB export without remesh")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _glb_args["remesh"] = False
        glb = o_voxel.postprocess.to_glb(**_glb_args)""",
    )
    replace_once(
        stages,
        """    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        preprocessed,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params=ss_params,
        shape_slat_sampler_params=shape_params,
        tex_slat_sampler_params=tex_params,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )""",
        """    # Some inputs/seeds can yield no active sparse voxels. Retry only that
    # known sampling failure with the next seeds so one bad draw cannot abort
    # a complete asset batch.
    for _attempt in range(3):
        _trial_seed = seed + _attempt
        torch.manual_seed(_trial_seed)
        try:
            mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
                preprocessed,
                camera_params=camera_params,
                seed=_trial_seed,
                sparse_structure_sampler_params=ss_params,
                shape_slat_sampler_params=shape_params,
                tex_slat_sampler_params=tex_params,
                preprocess_image=False,
                return_latent=True,
                pipeline_type=pipeline_type,
                max_num_tokens=max_num_tokens,
            )
            break
        except RuntimeError as _error:
            if "input.numel() == 0" not in str(_error) or _attempt == 2:
                raise
            log.warning(
                f"[Pixal3D] Empty sparse shape for seed {_trial_seed}; "
                f"retrying with seed {_trial_seed + 1}"
            )""",
    )
    replace_once(
        stages,
        'os.environ.setdefault("ATTN_BACKEND", "flash_attn")',
        '''# The bundled FlashAttention wheel has no sm_120 kernel (RTX 50 series).
# PyTorch SDPA selects a supported kernel for the active device at runtime.
if "ATTN_BACKEND" not in os.environ:
    try:
        _sm = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    except Exception:
        _sm = (0, 0)
    os.environ["ATTN_BACKEND"] = "sdpa" if _sm >= (12, 0) else "flash_attn"''',
    )
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

    # Setup and launch entry points call this repair with the selected
    # ComfyUI venv's interpreter. Prefer it so installs also work with named
    # environments such as `.venv-cuda`; only fall back to the default venv
    # when invoked from an unrelated Python.
    python = Path(sys.executable).resolve()
    try:
        python.relative_to(comfyui.resolve())
    except ValueError:
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
    pixal3d_loader_patch = Path(__file__).with_name("pixal3d_meta_loading.py")
    if pixal3d_loader_patch.is_file():
        subprocess.run(
            [sys.executable, str(pixal3d_loader_patch), "--comfyui", str(comfyui)],
            check=True,
        )
    repair_optional_nodes(comfyui)
    if args.install_dependencies:
        install_missing_optional_dependencies(comfyui)
        repair_pixal3d_dependencies(comfyui)


if __name__ == "__main__":
    main()
