#!/usr/bin/env python3
"""Use an isolated environment's Python when comfy-env's Pixi binary is absent."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


OLD = "if is_pixi:"
NEW = (
    'if is_pixi and (Path.home() / ".pixi" / "bin" / '
    '("pixi.exe" if sys.platform == "win32" else "pixi")).is_file():'
)
FILES = (
    Path("isolation/metadata.py"),
    Path("isolation/workers/subprocess.py"),
)


def patch_comfy_env(python: Path) -> list[Path]:
    package_path = subprocess.check_output(
        [
            str(python),
            "-c",
            "import importlib.util; s=importlib.util.find_spec('comfy_env'); "
            "print(s.origin if s else '')",
        ],
        text=True,
    ).strip()
    if not package_path:
        return []
    site_packages = Path(package_path).resolve().parent

    patched = []
    for relative in FILES:
        path = site_packages / relative
        source = path.read_text(encoding="utf-8")
        if NEW in source:
            continue
        if source.count(OLD) != 1:
            raise RuntimeError(f"Could not safely patch Pixi selection in {path}")
        path.write_bytes(source.replace(OLD, NEW, 1).encode("utf-8"))
        patched.append(path)
    return patched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    for path in patch_comfy_env(args.python.resolve()):
        print(f"[comfy-env-pixi-fallback] Patched {path}")


if __name__ == "__main__":
    main()
