from __future__ import annotations

import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from patches import repair_custom_nodes


class CustomNodeRepairTests(unittest.TestCase):
    def test_pixal3d_installer_requires_matching_native_wheel_stack(self):
        supported = {"missing": ["natten"], "python": [3, 10], "platform": "win32",
                     "machine": "AMD64", "torch": "2.11.0+cu128",
                     "cuda": "12.8", "hip": None}
        cases = [supported, {**supported, "python": [3, 12], "cuda": None, "hip": "7.2"},
                 {**supported, "torch": "2.10.0+cu128"},
                 {**supported, "platform": "linux"}]
        for runtime in cases:
            with self.subTest(runtime=runtime), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                node = root / "custom_nodes" / "comfyui-pixal3d"
                node.mkdir(parents=True)
                (node / "install.py").touch()
                (node / "requirements.txt").touch()
                with mock.patch.object(repair_custom_nodes, "comfyui_python", return_value=Path("python")), \
                     mock.patch.object(repair_custom_nodes.subprocess, "run") as run:
                    run.return_value.stdout = json.dumps(runtime)
                    repair_custom_nodes.repair_pixal3d_dependencies(root)
                self.assertEqual(run.call_count, 4 if runtime is supported else 1)

    def ollama_node(self, directory, client, platform="nt"):
        comfyui = Path(directory) / "ComfyUI"
        node = comfyui / "custom_nodes" / "comfyui-ollama-describer" / "nodes.py"
        node.parent.mkdir(parents=True)
        node.write_text(
            """from ollama import Client, ListResponse
import os

class OllamaImageDescriber:
    def ollama_image_describe(self, **inputs):
        client = Client()
        full_response = client.generate(**inputs)
        return (full_response['response'],)
""", encoding="utf-8")
        repair_custom_nodes.repair_ollama(comfyui)
        patched = node.read_bytes()
        repair_custom_nodes.repair_ollama(comfyui)
        self.assertEqual(node.read_bytes(), patched)
        namespace = {}
        ollama = types.SimpleNamespace(
            Client=mock.Mock(return_value=client), ListResponse=object,
            ResponseError=OllamaResponseError,
        )
        with mock.patch.dict("sys.modules", {"ollama": ollama}):
            exec(compile(patched, str(node), "exec"), namespace)
        namespace["os"] = types.SimpleNamespace(name=platform)
        return namespace["OllamaImageDescriber"]()

    def test_ollama_native_crash_retries_same_request_on_cpu_once(self):
        calls = []

        def generate(**request):
            calls.append(copy.deepcopy(request))
            if len(calls) == 1:
                raise OllamaResponseError(NATIVE_CRASH, 500)
            return {"response": "The image is red."}

        with tempfile.TemporaryDirectory() as directory:
            node = self.ollama_node(directory, types.SimpleNamespace(generate=generate))
            request = {
                "model": "llava:7b-v1.6", "system": "Keep one paragraph.",
                "prompt": "Describe the image.", "images": ["encoded-image"],
                "keep_alive": 5, "stream": False, "format": None,
                "options": {"num_ctx": 2048, "num_predict": 200, "seed": 42},
            }
            original = copy.deepcopy(request)
            self.assertEqual(node.ollama_image_describe(**request), ("The image is red.",))
        self.assertEqual(calls[0], original)
        original["options"]["num_gpu"] = 0
        self.assertEqual(calls[1:], [original])

    def test_ollama_other_errors_are_not_retried(self):
        for platform, status, message, options in (
            ("posix", 500, NATIVE_CRASH, {}),
            ("nt", 404, NATIVE_CRASH, {}),
            ("nt", 500, "context length exceeded", {}),
            ("nt", 500, "llama-server process has terminated: exit status 1", {}),
            ("nt", 500, NATIVE_CRASH, {"num_gpu": 0}),
        ):
            with self.subTest(platform=platform, status=status, message=message, options=options):
                client = mock.Mock()
                error = OllamaResponseError(message, status)
                client.generate.side_effect = error
                with tempfile.TemporaryDirectory() as directory:
                    node = self.ollama_node(directory, client, platform)
                    with self.assertRaises(OllamaResponseError) as raised:
                        node.ollama_image_describe(options=options)
                self.assertIs(raised.exception, error)
                self.assertEqual(client.generate.call_count, 1)

    def test_ollama_cpu_failure_is_propagated_without_a_third_request(self):
        client = mock.Mock()
        cpu_error = OllamaResponseError("CPU load failed", 500)
        client.generate.side_effect = [OllamaResponseError(NATIVE_CRASH, 500), cpu_error]
        with tempfile.TemporaryDirectory() as directory:
            node = self.ollama_node(directory, client)
            with self.assertRaises(OllamaResponseError) as raised:
                node.ollama_image_describe(options={})
        self.assertIs(raised.exception, cpu_error)
        self.assertEqual(client.generate.call_count, 2)

    def test_ollama_success_keeps_gpu_request(self):
        client = mock.Mock()
        client.generate.return_value = {"response": "Caption"}
        with tempfile.TemporaryDirectory() as directory:
            node = self.ollama_node(directory, client)
            self.assertEqual(node.ollama_image_describe(options={}), ("Caption",))
        client.generate.assert_called_once_with(options={})

    def test_kiui_repair_targets_the_selected_named_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            comfyui = Path(directory) / "ComfyUI"
            selected = comfyui / ".venv-cuda"
            kiui = selected / "Lib" / "site-packages" / "kiui"
            kiui.mkdir(parents=True)
            (kiui / "typing.py").write_text("# empty wheel module\n", encoding="utf-8")
            (kiui / "op.py").write_text(
                "from kiui.typing import *\n", encoding="utf-8"
            )

            with mock.patch.object(
                repair_custom_nodes.sys,
                "executable",
                str(selected / "Scripts" / "python.exe"),
            ):
                repair_custom_nodes.repair_optional_nodes(comfyui)

            self.assertIn("from typing import *", (kiui / "typing.py").read_text())
            self.assertIn("Literal, Optional, Sequence, Union", (kiui / "op.py").read_text())


NATIVE_CRASH = "llama-server process has terminated: exit status 0xe06d7363: NTSTATUS 0xe06d7363"


class OllamaResponseError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


if __name__ == "__main__":
    unittest.main()
