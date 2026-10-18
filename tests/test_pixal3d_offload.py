"""Regression checks for checkpoint offload and repeatable pipeline repairs."""

from __future__ import annotations

import json
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import patches
from patches import pixal3d_meta_loading as meta

try:
    import torch
    from safetensors.torch import load_file, save_file

    from patches.pixal3d_lazy_checkpoint import LazyCheckpoint
    from patches import pixal3d_safe_sparse_conv as safe_conv
    from patches.pixal3d_checkpoint_reader import load_checkpoint
except ImportError:
    torch = None


class PipelineFallbackTests(unittest.TestCase):
    def test_previous_numpy_loader_upgrades_to_streaming_and_stays_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "custom_nodes/comfyui-pixal3d/_pixal3d_src/pixal3d"
            (source / "models").mkdir(parents=True)
            model_loader = source / "models/__init__.py"
            model_loader.write_text(meta.PREVIOUS_NUMPY, encoding="utf-8")
            self.assertIn(model_loader, meta.patch_pixal3d(root))
            self.assertIn("load_checkpoint(model_file", model_loader.read_text(encoding="utf-8"))
            self.assertEqual(meta.patch_pixal3d(root), [])

    def test_repeatable_repairs_preserve_local_snapshot_guards_with_lazy_loading(self):
        for loader in ("models.from_pretrained(v)", "load_model(v)"):
            for fallback in (meta.OLD_FALLBACK, meta.NEW_FALLBACK, meta.CURRENT_FALLBACK):
                with self.subTest(loader=loader, fallback=fallback), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = root / "custom_nodes/comfyui-pixal3d/_pixal3d_src/pixal3d"
                    (source / "models").mkdir(parents=True)
                    (source / "pipelines").mkdir()
                    (source / "models/__init__.py").write_text(meta.NEW, encoding="utf-8")
                    pipeline = source / "pipelines/base.py"
                    original = fallback.replace("models.from_pretrained(v)", loader)
                    pipeline.write_text(original, encoding="utf-8")
                    meta.patch_pixal3d(root)
                    updated = pipeline.read_text(encoding="utf-8")
                    if fallback != meta.OLD_FALLBACK:
                        self.assertEqual(updated, original)
                    else:
                        self.assertIn("getattr(e, 'winerror', None) == 1455", updated)
                    self.assertIn(loader, updated)
                    self.assertEqual(meta.patch_pixal3d(root), [])


@unittest.skipIf(torch is None, "Requires the ComfyUI torch/safetensors runtime")
class LazyCheckpointTests(unittest.TestCase):
    def test_offload_releases_weights_without_cpu_copy_and_reloads_exact_checkpoint(self):
        class CopyForbidden(torch.nn.Conv1d):
            def cpu(self):
                raise AssertionError("Disk offload must not copy the model to CPU")

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stage"
            config = {"resolution": 4, "in_channels": 3, "out_channels": 2}
            Path(f"{checkpoint}.json").write_text(json.dumps({"args": config}), encoding="utf-8")
            original = CopyForbidden(3, 2, 1).eval()
            save_file(original.state_dict(), f"{checkpoint}.safetensors")

            def load(path, **kwargs):
                loaded = CopyForbidden(3, 2, 1)
                loaded.load_state_dict(load_file(f"{path}.safetensors"))
                return loaded

            with mock.patch.object(patches, "from_pretrained", side_effect=load, create=True) as loader:
                stage = LazyCheckpoint(str(checkpoint)).eval()
                stage.low_vram = True
                self.assertEqual(stage.resolution, 4)
                self.assertEqual(stage.in_channels, 3)
                stage.to("cpu")
                loader.assert_not_called()
                image = torch.randn(1, 3, 4)
                with torch.no_grad():
                    expected = original(image)
                    first = stage(image)
                    self.assertFalse(stage._loaded_model.training)
                    self.assertTrue(stage._loaded_model.low_vram)
                    torch.testing.assert_close(first, expected, rtol=0, atol=0)
                    self.assertGreater(len(list(stage.parameters())), 0)
                    self.assertIs(stage.cpu(), stage)
                    self.assertEqual(list(stage.parameters()), [])
                    stage.low_vram = False
                    torch.testing.assert_close(stage(image), expected, rtol=0, atol=0)
                    self.assertFalse(stage._loaded_model.low_vram)
                    self.assertEqual(loader.call_count, 2)

    def test_missing_checkpoint_fails_before_pipeline_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                LazyCheckpoint(str(Path(directory) / "missing"))


@unittest.skipIf(torch is None, "Requires the ComfyUI torch runtime")
class CheckpointReaderTests(unittest.TestCase):
    def test_owned_streamed_tensors_match_safetensors_including_empty_scalar_and_bfloat16(self):
        tensors = {
            "weights": torch.randn(3, 4), "bf16": torch.randn(3).bfloat16(),
            "scalar": torch.tensor(42), "empty": torch.empty(0, 2),
            "mask": torch.tensor([True, False]),
        }
        expected = {key: value.to("meta") for key, value in tensors.items()}
        expected["weights"] = expected["weights"].bfloat16()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            save_file(tensors, str(path))
            with mock.patch("safetensors.safe_open", side_effect=AssertionError("Must not mmap")):
                actual = load_checkpoint(path, expected)
            for key, value in tensors.items():
                self.assertEqual(actual[key].device.type, "cpu")
                torch.testing.assert_close(actual[key], value.to(expected[key].dtype), rtol=0, atol=0)

    def test_invalid_ranges_shapes_and_duplicate_keys_are_rejected(self):
        headers = [
            '{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}',
            '{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}',
            '{"w":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}',
            '{"w":{},"w":{}}',
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.safetensors"
            for header in headers:
                raw = header.encode()
                path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(5))
                with self.subTest(header=header), self.assertRaises(ValueError):
                    load_checkpoint(path, {"w": torch.empty(1, device="meta")})


@unittest.skipIf(torch is None, "Requires the ComfyUI torch runtime")
class SparseConvolutionTests(unittest.TestCase):
    def test_chunked_sparse_convolution_matches_dense_with_holes_and_bias(self):
        coords = [(0, 0, 0), (0, 1, 2), (1, 1, 1), (2, 1, 2), (2, 2, 2)]
        lookup = {coord: index for index, coord in enumerate(coords)}
        neighbors = torch.tensor([
            [lookup.get(tuple(c + d for c, d in zip(coord, delta)), 0xFFFFFFFF)
             for delta in itertools.product(range(-1, 2), repeat=3)]
            for coord in coords
        ], dtype=torch.int64)
        feats = torch.randn(len(coords), 3)
        weight = torch.randn(2, 3, 3, 3, 3)
        for bias in (None, torch.randn(2)):
            dense = torch.zeros(1, 3, 3, 3, 3)
            for index, coord in enumerate(coords):
                dense[(0, slice(None), *coord)] = feats[index]
            reference = torch.nn.functional.conv3d(
                dense, weight.permute(0, 4, 1, 2, 3), bias, padding=1,
            )
            expected = torch.stack([reference[(0, slice(None), *coord)] for coord in coords])
            # Force multiple small chunks and verify absent neighbors stay zero.
            with mock.patch.object(safe_conv, "MAX_IM2COL_BYTES", 2 * 27 * 3 * 4):
                actual = safe_conv.explicit_submanifold_conv(feats, neighbors, weight, bias)
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
