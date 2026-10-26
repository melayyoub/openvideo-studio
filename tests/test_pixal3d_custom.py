from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from patches.pixal3d_custom import patch_pixal3d


class Pixal3DCustomPatchTests(unittest.TestCase):
    def test_applies_custom_changes_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "custom_nodes" / "comfyui-pixal3d" / "_pixal3d_src"
            files = {
                "inference.py": """    mesh = mesh_list[0]

    # Extract GLB
    print("[Inference] Extracting GLB...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,""",
                "pixal3d/pipelines/rembg/BiRefNet.py": """    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model = AutoModelForImageSegmentation.from_pretrained(""",
                "pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py": """grid = queries_ndc.view(B, K, 1, 2)  # (B, K, 1, 2)

        # Use DINOv3's learned final normalization, as its forward() does.
        norm = getattr(self.model, "norm", None)
        return norm(hidden_states) if norm is not None else F.layer_norm(
            hidden_states, hidden_states.shape[-1:]
        )""",
            }
            for relative_path, contents in files.items():
                path = source / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")

            changed = patch_pixal3d(root)
            self.assertEqual(len(changed), 4)
            self.assertEqual(patch_pixal3d(root), [])

            inference = (source / "inference.py").read_text(encoding="utf-8")
            self.assertIn("del mesh_list, shape_slat, tex_slat, pipeline", inference)
            birefnet = (source / "pixal3d/pipelines/rembg/BiRefNet.py").read_text(encoding="utf-8")
            self.assertIn("briaai/RMBG-2.0", birefnet)
            features = (source / "pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py").read_text(encoding="utf-8")
            self.assertIn("queries_ndc.to(dtype=fmap.dtype)", features)
            self.assertIn("self.model.model, \"norm\"", features)


if __name__ == "__main__":
    unittest.main()
