from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from patches import repair_custom_nodes


class CustomNodeRepairTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
