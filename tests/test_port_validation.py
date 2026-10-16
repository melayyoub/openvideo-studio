"""Boundary and CLI integration tests for the public port contract."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import wan2_cli  # noqa: E402
from wan2_cli import (  # noqa: E402
    REQUIREMENTS_STAMP,
    backend_arguments,
    ensure_comfyui_requirements,
    gfx_for_adapter,
    resolve_checkout_path,
    venv_site_packages,
)
from wan2_cli_args import (  # noqa: E402
    MAX_PORT,
    MIN_PORT,
    model_repository,
    model_revision,
    port_number,
)


class PortNumberTests(unittest.TestCase):
    def test_accepts_inclusive_boundaries(self) -> None:
        self.assertEqual(port_number(str(MIN_PORT)), MIN_PORT)
        self.assertEqual(port_number(str(MAX_PORT)), MAX_PORT)

    def test_rejects_values_outside_boundaries_and_non_integer_input(self) -> None:
        for value in ("0", "65536", "not-a-port"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    argparse.ArgumentTypeError, "1 through 65535"
                ):
                    port_number(value)


class CheckoutPathTests(unittest.TestCase):
    def test_default_checkout_is_scoped_to_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.assertEqual(resolve_checkout_path("", root), root / "ComfyUI")

    def test_relative_override_is_scoped_to_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.assertEqual(
                resolve_checkout_path("checkouts/custom", root),
                root / "checkouts" / "custom",
            )

    def test_absolute_override_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkout = Path(temporary_directory).resolve()
            self.assertEqual(resolve_checkout_path(str(checkout), Path.cwd()), checkout)


class RocmArchitectureTests(unittest.TestCase):
    def test_maps_supported_radeon_cards_to_their_wheel_index(self) -> None:
        for name, gfx in (
            ("AMD Radeon RX 7900 XT", "gfx110X-dgpu"),
            ("AMD Radeon RX 9070 XT", "gfx120X-all"),
            ("AMD Radeon RX 6800", "gfx103X-dgpu"),
            ("AMD Radeon RX 5700 XT", "gfx101X-dgpu"),
            ("AMD Radeon Pro W7900", "gfx110X-dgpu"),
        ):
            with self.subTest(adapter=name):
                self.assertEqual(gfx_for_adapter(name), gfx)

    def test_rejects_adapters_without_a_rocm_build(self) -> None:
        for name in ("NVIDIA GeForce RTX 5090", "AMD Radeon RX 580", "Intel Iris Xe"):
            with self.subTest(adapter=name):
                self.assertIsNone(gfx_for_adapter(name))


class SitePackagesDiscoveryTests(unittest.TestCase):
    def test_finds_site_packages_in_both_venv_layouts(self) -> None:
        for layout in ("Lib/site-packages", "lib/python3.12/site-packages"):
            with self.subTest(layout=layout):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    venv = Path(temporary_directory)
                    (venv / "pyvenv.cfg").write_text("", encoding="utf-8")
                    expected = venv / layout
                    expected.mkdir(parents=True)
                    self.assertEqual(venv_site_packages(venv), expected)

    def test_ignores_directories_that_are_not_virtual_environments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            plain = Path(temporary_directory)
            (plain / "Lib" / "site-packages").mkdir(parents=True)
            self.assertIsNone(venv_site_packages(plain))


class BackendArgumentTests(unittest.TestCase):
    def test_rocm_attention_default_survives_other_extra_arguments(self) -> None:
        self.assertEqual(
            backend_arguments(["--reserve-vram", "6"], rocm=True),
            [
                "--use-pytorch-cross-attention",
                "--enable-manager",
                "--reserve-vram",
                "6",
            ],
        )

    def test_an_explicit_attention_choice_replaces_the_rocm_default(self) -> None:
        self.assertEqual(
            backend_arguments(["--use-split-cross-attention"], rocm=True),
            ["--enable-manager", "--use-split-cross-attention"],
        )

    def test_no_attention_default_is_added_off_rocm(self) -> None:
        self.assertEqual(backend_arguments([], rocm=False), ["--enable-manager"])
        self.assertEqual(
            backend_arguments(["--reserve-vram", "6"], rocm=False),
            ["--enable-manager", "--reserve-vram", "6"],
        )

    def test_the_manager_is_enabled_on_every_backend(self) -> None:
        for rocm in (True, False):
            self.assertIn("--enable-manager", backend_arguments([], rocm=rocm))

    def test_an_explicit_manager_flag_is_not_duplicated(self) -> None:
        self.assertEqual(
            backend_arguments(["--enable-manager-legacy-ui"], rocm=False),
            ["--enable-manager-legacy-ui"],
        )
        self.assertEqual(
            backend_arguments(["--enable-manager"], rocm=False),
            ["--enable-manager"],
        )

    def test_pinned_memory_is_disabled_on_low_memory_hosts(self) -> None:
        self.assertEqual(
            backend_arguments([], rocm=False, low_memory=True),
            ["--disable-pinned-memory", "--enable-manager"],
        )
        self.assertEqual(backend_arguments([], rocm=False), ["--enable-manager"])

    def test_an_explicit_pinned_memory_flag_is_not_duplicated(self) -> None:
        self.assertEqual(
            backend_arguments(["--disable-pinned-memory"], rocm=False, low_memory=True),
            ["--enable-manager", "--disable-pinned-memory"],
        )

    def test_pinned_memory_risk_follows_installed_ram(self) -> None:
        gib = 1024**3
        small = {"total_ram": 32 * gib, "available_ram": 8 * gib,
                 "commit_limit": 96 * gib, "available_commit": 64 * gib}
        large = dict(small, total_ram=64 * gib)
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OVS_PINNED_MEMORY", None)
            os.environ.pop("CUSTOM_WAN_PINNED_MEMORY", None)
            self.assertTrue(wan2_cli.pinned_memory_is_risky(small))
            self.assertFalse(wan2_cli.pinned_memory_is_risky(large))
            self.assertFalse(wan2_cli.pinned_memory_is_risky(None))
            os.environ["OVS_PINNED_MEMORY"] = "1"
            self.assertFalse(wan2_cli.pinned_memory_is_risky(small))

    def test_memory_warnings_flag_low_commit_and_ram(self) -> None:
        gib = 1024**3
        healthy = {"total_ram": 32 * gib, "available_ram": 8 * gib,
                   "commit_limit": 128 * gib, "available_commit": 64 * gib}
        self.assertEqual(wan2_cli.memory_warnings(healthy), [])
        self.assertEqual(wan2_cli.memory_warnings(None), [])
        starved = dict(healthy, available_ram=gib // 2, available_commit=20 * gib)
        self.assertEqual(len(wan2_cli.memory_warnings(starved)), 2)

    def test_cuda_malloc_is_disabled_on_blackwell(self) -> None:
        self.assertEqual(
            backend_arguments([], rocm=False, blackwell=True),
            ["--disable-cuda-malloc", "--enable-manager"],
        )

    def test_cuda_malloc_default_is_not_added_off_blackwell(self) -> None:
        self.assertEqual(
            backend_arguments([], rocm=False, blackwell=False),
            ["--enable-manager"],
        )

    def test_an_explicit_cuda_malloc_choice_replaces_the_blackwell_default(self) -> None:
        self.assertEqual(
            backend_arguments(["--cuda-malloc"], rocm=False, blackwell=True),
            ["--enable-manager", "--cuda-malloc"],
        )
        self.assertEqual(
            backend_arguments(["--disable-cuda-malloc"], rocm=False, blackwell=True),
            ["--enable-manager", "--disable-cuda-malloc"],
        )


class RequirementsDriftTests(unittest.TestCase):
    """A stale environment is the failure that broke startup twice."""

    def build_layout(self, base: Path, requirements: str) -> tuple[Path, Path]:
        comfyui = base / "ComfyUI"
        (comfyui / "custom_nodes").mkdir(parents=True)
        (comfyui / "requirements.txt").write_text(requirements, encoding="utf-8")
        venv = comfyui / ".venv"
        (venv / "Lib" / "site-packages").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("", encoding="utf-8")
        (venv / "pip-constraints.txt").write_text("torch==2.11.0\n", encoding="utf-8")
        return comfyui, venv / "Scripts" / "python.exe"

    def install_calls(self, comfyui: Path, python_path: Path) -> list[list[str]]:
        calls: list[list[str]] = []
        original = wan2_cli.run_checked
        wan2_cli.run_checked = calls.append
        try:
            ensure_comfyui_requirements(python_path, comfyui)
        finally:
            wan2_cli.run_checked = original
        return calls

    def test_reinstalls_and_stamps_when_requirements_are_unseen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            comfyui, python_path = self.build_layout(
                Path(temporary_directory), "comfy-kitchen==0.2.31\n"
            )
            calls = self.install_calls(comfyui, python_path)
            self.assertEqual(len(calls), 1, calls)
            self.assertIn("-r", calls[0])
            self.assertIn("-c", calls[0])
            self.assertTrue((python_path.parent.parent / REQUIREMENTS_STAMP).exists())

    def test_skips_reinstall_when_the_stamp_still_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            comfyui, python_path = self.build_layout(
                Path(temporary_directory), "comfy-kitchen==0.2.31\n"
            )
            self.install_calls(comfyui, python_path)
            self.assertEqual(self.install_calls(comfyui, python_path), [])

    def test_reinstalls_after_the_checkout_changes_its_pins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            comfyui, python_path = self.build_layout(
                Path(temporary_directory), "comfy-kitchen==0.2.31\n"
            )
            self.install_calls(comfyui, python_path)
            (comfyui / "requirements.txt").write_text(
                "comfy-kitchen==0.2.32\n", encoding="utf-8"
            )
            self.assertEqual(len(self.install_calls(comfyui, python_path)), 1)


class ModelSourceValidationTests(unittest.TestCase):
    def test_accepts_portable_repository_and_revision_values(self) -> None:
        self.assertEqual(model_repository("owner/model-name"), "owner/model-name")
        self.assertEqual(model_revision("refs/reviewed-v1"), "refs/reviewed-v1")

    def test_rejects_url_or_code_shaped_model_values(self) -> None:
        for value in (
            "https://example.invalid/model",
            "owner/model/extra",
            "owner model",
        ):
            with self.subTest(repository=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    model_repository(value)
        for value in ("../main", "main'''", "/main", "main/"):
            with self.subTest(revision=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    model_revision(value)


class LauncherPortTests(unittest.TestCase):
    def run_cli(
        self,
        script: str,
        *arguments: str,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(REPOSITORY_ROOT / script), *arguments],
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            check=False,
            text=True,
        )

    def assert_port_error(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("1 through 65535", result.stderr)

    def test_lightweight_launcher_accepts_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            comfyui = base / "ComfyUI"
            comfyui.mkdir()
            (comfyui / "main.py").write_text("", encoding="utf-8")

            for value in (str(MIN_PORT), str(MAX_PORT)):
                with self.subTest(value=value):
                    result = self.run_cli(
                        "wan2_cli.py",
                        "start",
                        "--path",
                        str(base),
                        "--device",
                        "cpu",
                        "--port",
                        value,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )

    def test_lightweight_launcher_rejects_out_of_range_cli_ports(self) -> None:
        for value in ("0", "65536"):
            with self.subTest(value=value):
                result = self.run_cli("wan2_cli.py", "start", "--port", value)
                self.assert_port_error(result)

    def test_lightweight_launcher_validates_environment_default(self) -> None:
        environment = os.environ.copy()
        environment["CUSTOM_WAN_COMFYUI_PORT"] = "65536"
        result = self.run_cli("wan2_cli.py", "start", environment=environment)
        self.assert_port_error(result)

    def test_installer_launcher_accepts_boundaries(self) -> None:
        for value in (str(MIN_PORT), str(MAX_PORT)):
            with self.subTest(value=value):
                result = self.run_cli(
                    "wan2_installer.py",
                    "start",
                    "--dry-run",
                    "--port",
                    value,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_installer_launcher_rejects_out_of_range_ports(self) -> None:
        for value in ("0", "65536"):
            with self.subTest(value=value):
                result = self.run_cli(
                    "wan2_installer.py",
                    "start",
                    "--dry-run",
                    "--port",
                    value,
                )
                self.assert_port_error(result)


if __name__ == "__main__":
    unittest.main()
