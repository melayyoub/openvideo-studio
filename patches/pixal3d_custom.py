"""Apply local Pixal3D customizations to the upstream source checkout."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> bool:
    if not path.is_file():
        return False
    source = path.read_text(encoding="utf-8")
    if new in source:
        return False
    if source.count(old) != 1:
        return False
    path.write_text(source.replace(old, new, 1), encoding="utf-8", newline="")
    print(f"[pixal3d-custom] Patched {path}")
    return True


def patch_pixal3d(comfyui: Path) -> list[Path]:
    node = comfyui / "custom_nodes" / "comfyui-pixal3d"
    source = node / "_pixal3d_src"
    changed: list[Path] = []

    if replace_once(
        node / "install.py",
        'PIXAL3D_REF = "d42dcaad99ba07d35a02fa62a23e1cd6f2f61da1"',
        'PIXAL3D_REF = "f7cf38429b0bd264f1995f0f8743a88b1c728b94"',
    ):
        changed.append(node / "install.py")

    inference = source / "inference.py"
    if replace_once(
        inference,
        """    mesh = mesh_list[0]

    # Extract GLB
    print("[Inference] Extracting GLB...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,""",
        """    mesh = mesh_list[0]
    pbr_attr_layout = pipeline.pbr_attr_layout
    del mesh_list, shape_slat, tex_slat, pipeline
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Extract GLB
    print("[Inference] Extracting GLB...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pbr_attr_layout,""",
    ):
        changed.append(inference)

    birefnet = source / "pixal3d" / "pipelines" / "rembg" / "BiRefNet.py"
    if replace_once(
        birefnet,
        """    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model = AutoModelForImageSegmentation.from_pretrained(""",
        """    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        if model_name in ("briaai/RMBG-2.0", "briaai/RMBG-1.4"):
            model_name = "ZhengPeng7/BiRefNet"
        self.model = AutoModelForImageSegmentation.from_pretrained(""",
    ):
        changed.append(birefnet)

    features = (
        source
        / "pixal3d"
        / "trainers"
        / "flow_matching"
        / "mixins"
        / "image_conditioned_proj.py"
    )
    if replace_once(
        features,
        "grid = queries_ndc.view(B, K, 1, 2)  # (B, K, 1, 2)",
        "grid = queries_ndc.to(dtype=fmap.dtype).view(B, K, 1, 2)  # (B, K, 1, 2)",
    ):
        changed.append(features)
    if replace_once(
        features,
        """        # Use DINOv3's learned final normalization, as its forward() does.
        norm = getattr(self.model, "norm", None)
        return norm(hidden_states) if norm is not None else F.layer_norm(
            hidden_states, hidden_states.shape[-1:]
        )""",
        """        norm = getattr(self.model, "norm", None)
        if norm is None and hasattr(self.model, "model"):
            norm = getattr(self.model.model, "norm", None)
        return norm(hidden_states) if norm is not None else F.layer_norm(
            hidden_states, hidden_states.shape[-1:]
        )""",
    ):
        changed.append(features)

    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", type=Path, required=True)
    args = parser.parse_args()
    patch_pixal3d(args.comfyui.resolve())


if __name__ == "__main__":
    main()
