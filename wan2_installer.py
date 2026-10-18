#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wan2_installer.py — cross-platform ComfyUI + Wan 2.2 installer backend.

Called by install.sh on Linux/macOS; usable directly on any platform.

Layout (ALWAYS under <base>/ComfyUI):
  <base>/ComfyUI               # ComfyUI repo
  <base>/ComfyUI/.venv         # virtual environment
  <base>/ComfyUI/models/**     # models
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Optional, List

from wan2_cli_args import model_repository, model_revision, port_number

APP_NAME = "wan2_installer"
DEFAULT_DIR = Path(os.getenv("CUSTOM_WAN_PROJECT_PATH", Path.cwd()))
COMFY_DIR = "ComfyUI"  # enforced subfolder for repo
VENV_DIR = ".venv"
DEFAULT_PORT = 8188
# Keep in sync with install.ps1, which clones the same repository.
COMFYUI_GIT_URL = "https://github.com/Comfy-Org/ComfyUI.git"

CONFIG_DIR = Path(__file__).resolve().parent / "config"
MODEL_MANIFEST = json.loads((CONFIG_DIR / "models.json").read_text(encoding="utf-8"))
if MODEL_MANIFEST.get("schema_version") != 1:
    raise RuntimeError("Unsupported model manifest schema version")
WAN_CONFIG = MODEL_MANIFEST["wan"]
WAN_REPO = WAN_CONFIG["repository"]
WAN_REVISION = WAN_CONFIG["revision"]

NODE_MANIFEST = json.loads((CONFIG_DIR / "nodes.json").read_text(encoding="utf-8"))
if NODE_MANIFEST.get("schema_version") != 1:
    raise RuntimeError("Unsupported node manifest schema version")
NODE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
NODE_REPOSITORY_PATTERN = re.compile(
    r"^https://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.git$"
)
NODE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")


# ------------------------------ helpers ---------------------------------------


def log(msg: str) -> None:
    print(f"[{APP_NAME}] {msg}")


def is_windows() -> bool:
    return platform.system().lower().startswith("win")


def run(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    check: bool = True,
    dry: bool = False,
) -> int:
    shown_cwd = f" (cwd={cwd})" if cwd else ""
    log(("DRYRUN: " if dry else "") + " ".join(map(str, cmd)) + shown_cwd)
    if dry:
        return 0
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")
    return result.returncode


def py_exec(venv_bin: Path) -> str:
    return str(venv_bin / ("python.exe" if is_windows() else "python"))


def base_python_command(py_ver: Optional[str] = None) -> List[str]:
    if is_windows() and py_ver:
        return ["py", f"-{py_ver}"]
    base_executable = getattr(sys, "_base_executable", None) or sys.executable
    return [str(Path(base_executable).resolve())]


def ensure_tools(tools: List[str]) -> None:
    for tool in tools:
        if shutil.which(tool) is None:
            log(f"WARNING: '{tool}' not found in PATH. Please install it.")


def ensure_base_dir(base: Path, dry: bool = False) -> None:
    if not base.exists():
        log(f"Creating {base}")
        if not dry:
            base.mkdir(parents=True, exist_ok=True)


def comfy_root(base: Path) -> Path:
    return base / COMFY_DIR


# --------------------------- clone ComfyUI (subfolder) ------------------------


def clone_comfy(base: Path, dry: bool = False) -> None:
    """
    Ensure ComfyUI exists at <base>/ComfyUI ONLY.
    - If <base>/ComfyUI doesn't exist -> clone there.
    - If exists and is a git repo -> git pull.
    - If exists, empty dir -> clone into '.' with cwd=comfy.
    - If exists and non-empty & not a repo -> raise a clear error.
    """
    comfy = comfy_root(base)
    if comfy.exists():
        if (comfy / ".git").exists():
            log("ComfyUI already cloned — pulling latest.")
            run(["git", "pull", "--ff-only"], cwd=comfy, dry=dry)
            return
        # not a repo: check if empty
        try:
            is_empty = next(comfy.iterdir(), None) is None
        except PermissionError:
            is_empty = False
        if is_empty:
            log("Empty ComfyUI/ dir — cloning into it.")
            run(["git", "clone", COMFYUI_GIT_URL, "."], cwd=comfy, dry=dry)
            return
        raise RuntimeError(
            f"'{comfy}' exists and is not a git repo. Delete it or choose a different --path."
        )
    # folder not present: clone into subfolder
    run(["git", "clone", COMFYUI_GIT_URL, str(comfy)], dry=dry)


# ------------------------------ venv handling ---------------------------------


def recreate_venv(base: Path, py_ver: Optional[str], dry: bool = False) -> Path:
    comfy = comfy_root(base)
    venv = comfy / VENV_DIR
    if venv.exists():
        log(f"Removing existing venv at {venv}")
        if not dry:
            if venv.name != VENV_DIR or venv.parent.resolve() != comfy.resolve():
                raise RuntimeError(
                    f"Refusing to remove an unexpected venv path: {venv}"
                )
            shutil.rmtree(venv)
    if not comfy.exists():
        log(f"Creating {comfy}")
        if not dry:
            comfy.mkdir(parents=True, exist_ok=True)
    run(base_python_command(py_ver) + ["-m", "venv", str(venv)], cwd=comfy, dry=dry)
    return venv


def ensure_venv(base: Path, py_ver: Optional[str], dry: bool = False) -> Path:
    comfy = comfy_root(base)
    venv = comfy / VENV_DIR
    bin_dir = venv / ("Scripts" if is_windows() else "bin")
    if bin_dir.exists():
        log("Venv exists — reusing.")
        return venv
    if not comfy.exists():
        log(f"Creating {comfy}")
        if not dry:
            comfy.mkdir(parents=True, exist_ok=True)
    run(base_python_command(py_ver) + ["-m", "venv", str(venv)], cwd=comfy, dry=dry)
    return venv


def pip(
    venv: Path, args: List[str], cwd: Optional[Path] = None, dry: bool = False
) -> int:
    pip_cmd = [
        py_exec(venv_bin=venv / ("Scripts" if is_windows() else "bin")),
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--retries",
        "10",
        "--resume-retries",
        "10",
        "--timeout",
        "120",
    ] + args
    for attempt in range(1, 4):
        code = run(pip_cmd, cwd=cwd, check=False, dry=dry)
        if code == 0:
            return 0
        if attempt < 3:
            delay = 5 * (2 ** (attempt - 1))
            log(f"pip attempt {attempt}/3 failed; retrying in {delay} seconds")
            time.sleep(delay)
    raise RuntimeError(f"pip failed after 3 attempts: {' '.join(pip_cmd)}")


def install_torch(venv: Path, cuda: str, dry: bool = False) -> None:
    idx_map = {
        "cu128": "https://download.pytorch.org/whl/cu128",
        "cu121": "https://download.pytorch.org/whl/cu121",
        "cu118": "https://download.pytorch.org/whl/cu118",
        "cpu": "https://download.pytorch.org/whl/cpu",
    }
    if cuda not in idx_map:
        raise ValueError("--cuda must be one of cu128|cu121|cu118|cpu")
    python = py_exec(venv_bin=venv / ("Scripts" if is_windows() else "bin"))
    run(
        [python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        dry=dry,
    )
    pip(
        venv,
        ["torch", "torchvision", "torchaudio", "--index-url", idx_map[cuda]],
        dry=dry,
    )


# ------------------------------ requirements ----------------------------------


def ensure_model_dirs(base: Path, dry: bool = False) -> None:
    comfy = comfy_root(base)
    for p in [
        comfy / "models" / "checkpoints",
        comfy / "models" / "diffusion_models",
        comfy / "models" / "vae",
        comfy / "models" / "text_encoders",
        comfy / "models" / "loras",
        comfy / "models" / "latent_upscale_models",
        comfy / "custom_nodes",
    ]:
        if not p.exists():
            log(f"Creating {p}")
            if not dry:
                p.mkdir(parents=True, exist_ok=True)


def install_comfy_requirements(
    venv: Path,
    base: Path,
    extra_requirements: Optional[Path] = None,
    dry: bool = False,
) -> None:
    """
    Install ComfyUI's supported requirements with normal dependency solving.
    A caller may explicitly add a reviewed requirements file; generated local
    environment snapshots are never consumed implicitly.
    """
    comfy = comfy_root(base)

    pip(venv, ["-r", "requirements.txt"], cwd=comfy, dry=dry)

    if extra_requirements is not None:
        reviewed_requirements = extra_requirements.expanduser().resolve()
        if not dry and not reviewed_requirements.is_file():
            raise FileNotFoundError(
                f"Extra requirements file does not exist: {reviewed_requirements}"
            )
        pip(venv, ["-r", str(reviewed_requirements)], cwd=comfy, dry=dry)


def install_required_nodes(venv: Path, base: Path, dry: bool = False) -> None:
    """
    Install the curated custom-node stack from config/nodes.json at pinned
    commits. Each node's requirements.txt is installed into the project venv.
    """
    custom_nodes = comfy_root(base) / "custom_nodes"
    if not dry:
        custom_nodes.mkdir(parents=True, exist_ok=True)

    for node in NODE_MANIFEST["nodes"]:
        name = str(node["name"])
        repository = str(node["repository"])
        revision = str(node["revision"])
        if not NODE_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Unsafe node name in manifest: {name}")
        if not NODE_REPOSITORY_PATTERN.fullmatch(repository):
            raise ValueError(f"Unsupported node repository in manifest: {repository}")
        if not NODE_REVISION_PATTERN.fullmatch(revision):
            raise ValueError(f"Node revision must be a pinned commit: {revision}")

        path = custom_nodes / name
        if (path / ".git").exists():
            log(f"[nodes] {name} present — fetching pinned revision.")
            run(["git", "-C", str(path), "fetch", "origin"], dry=dry, check=False)
        else:
            log(f"[nodes] Installing {name}...")
            run(["git", "clone", repository, str(path)], dry=dry)
        run(["git", "-C", str(path), "checkout", "--detach", revision], dry=dry)

        requirements = path / "requirements.txt"
        if dry or requirements.is_file():
            pip(venv, ["-r", str(requirements)], dry=dry)


def install_manager(venv: Path, base: Path, dry: bool = False) -> None:
    comfy = comfy_root(base)
    npath = comfy / "custom_nodes" / "ComfyUI-Manager"
    if npath.exists():
        log("ComfyUI-Manager already present — skipping.")
        return
    npath.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "https://github.com/Comfy-Org/ComfyUI-Manager.git",
            str(npath),
        ],
        dry=dry,
    )


# ------------------------------- Hugging Face ----------------------------------


def install_hf_cli(venv: Path, dry: bool = False) -> None:
    # only the library is needed; we use the Python API (no external CLI launcher)
    pip(venv, ["huggingface_hub>=0.23"], dry=dry)


def hf_login(venv: Path, token: Optional[str], dry: bool = False) -> None:
    # No interactive login needed if HF_TOKEN is present.
    if token:
        pybin = py_exec(venv_bin=venv / ("Scripts" if is_windows() else "bin"))
        run(
            [
                pybin,
                "-c",
                "import os; from huggingface_hub import whoami; "
                "t=os.getenv('HF_TOKEN'); "
                "print('HF whoami:', whoami(token=t) if t else 'no token')",
            ],
            dry=dry,
            check=False,
        )
    else:
        log("No HF_TOKEN provided; if models are gated, set HF_TOKEN and rerun.")


def hf_download(
    venv: Path,
    repo_id: str,
    revision: str,
    files: list[str],
    dest: Path,
    dry: bool = False,
) -> None:
    pybin = py_exec(venv_bin=venv / ("Scripts" if is_windows() else "bin"))
    code = f"""
import hashlib, os, pathlib
from huggingface_hub import hf_hub_download
dest = r'''{dest}'''
pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
repo = r'''{repo_id}'''
revision = r'''{revision}'''
files = {files!r}
tok = os.getenv('HF_TOKEN')
for f, expected_sha256 in files:
    path = hf_hub_download(repo_id=repo, revision=revision, filename=f, local_dir=dest,
                           token=tok)
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise SystemExit(f'sha256 mismatch for {{f}}: {{digest.hexdigest()}}')
    print(path, 'sha256 verified')
"""
    run([pybin, "-c", code], dry=dry)


ALLOWED_MODEL_DESTINATIONS = {
    "checkpoints",
    "diffusion_models",
    "vae",
    "text_encoders",
    "loras",
    "latent_upscale_models",
}


def download_models(
    venv: Path,
    base: Path,
    which: str,
    wan_repository: str,
    wan_revision: str,
    dry: bool = False,
) -> None:
    """
    Download every manifest artifact matching the selection across all model
    families. The repository/revision overrides apply to the Wan family only;
    other families always use their pinned manifest source.
    """
    models_root = comfy_root(base) / "models"
    selected_groups = {"5b", "14b", "i2v", "ltx", "ltx2"} if which == "all" else {which}

    install_hf_cli(venv, dry=dry)
    hf_login(venv, token=os.environ.get("HF_TOKEN"), dry=dry)

    for family_name, family in MODEL_MANIFEST.items():
        if family_name == "schema_version":
            continue
        repository = str(family["repository"])
        revision = str(family["revision"])
        if family_name == "wan":
            repository, revision = wan_repository, wan_revision

        selected_by_destination: dict[str, list[tuple[str, str]]] = {}
        for artifact in family["artifacts"]:
            if set(artifact["groups"]).isdisjoint(selected_groups):
                continue
            destination = str(artifact["destination"])
            if destination not in ALLOWED_MODEL_DESTINATIONS:
                raise ValueError(f"Unsupported model destination: {destination}")
            path = str(artifact["path"])
            if ".." in path or path.startswith("/"):
                raise ValueError(f"Unsafe model path in manifest: {path}")
            sha256 = str(artifact.get("sha256", ""))
            if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
                raise ValueError(f"Model artifact is missing a valid sha256: {path}")
            selected_by_destination.setdefault(destination, []).append((path, sha256))

        for destination, files in selected_by_destination.items():
            hf_download(
                venv, repository, revision, files, models_root / destination, dry=dry
            )


# -------------------------------- React loader --------------------------------


def make_react(
    base: Path, app_name: str, comfy_url: str, dry: bool = False
) -> Path | None:
    target = base / app_name
    if target.exists():
        log(f"React app '{app_name}' already exists — skipping scaffold.")
        return target
    ensure_tools(["node", "npm"])
    run(
        ["npm", "create", "vite@latest", app_name, "--", "--template", "react-ts"],
        cwd=base,
        dry=dry,
    )
    run(["npm", "i"], cwd=target, dry=dry)
    run(["npm", "i", "axios"], cwd=target, dry=dry)
    # .env
    env_path = target / ".env"
    content = f"VITE_COMFY_BASE_URL={comfy_url}\n"
    if not dry:
        env_path.write_text(content, encoding="utf-8")
    log(f"Wrote {env_path}")
    # Minimal status UI
    app_tsx = target / "src" / "App.tsx"
    loader = dedent("""
        import { useEffect, useState } from "react";
        const API = import.meta.env.VITE_COMFY_BASE_URL ?? "http://127.0.0.1:8188";
        type Stats = Record<string, any> | null;

        export default function App() {
          const [ready, setReady] = useState(false);
          const [stats, setStats] = useState<Stats>(null);
          useEffect(() => {
            const id = setInterval(async () => {
              try {
                const r = await fetch(`${API}/system_stats`);
                if (r.ok) { setStats(await r.json()); setReady(true); clearInterval(id); }
              } catch {}
            }, 1000);
            return () => clearInterval(id);
          }, []);
          return (
            <div style={{fontFamily:"system-ui",padding:24}}>
              <h1>ComfyUI Loader</h1>
              {!ready ? <p>Starting ComfyUI…</p> : <pre>{JSON.stringify(stats, null, 2)}</pre>}
            </div>
          );
        }
    """).strip()
    if not dry:
        app_tsx.write_text(loader, encoding="utf-8")
    log(f"Wrote loader UI to {app_tsx}")
    return target


# --------------------------------- run ComfyUI --------------------------------
def start_comfy(
    venv: Path, base: Path, port: int, listen_all: bool, dry: bool = False
) -> None:
    # Ensure environment variables for PyTorch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    comfy = comfy_root(base)
    pybin = py_exec(venv_bin=(venv / ("Scripts" if is_windows() else "bin")))

    args = [pybin, "main.py", "--port", str(port)]
    if listen_all:
        args += ["--listen"]

    log(f"Starting ComfyUI on port {port} with GPU only...")
    run(args, cwd=comfy, dry=dry, check=False)
    log("ComfyUI process exited.")


# ------------------------------------- CLI ------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog=APP_NAME, description="ComfyUI + Wan 2.2 setup CLI (Subfolder Layout)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Base directory (default: {DEFAULT_DIR})",
    )
    common.add_argument(
        "--dry-run", action="store_true", help="Print commands, don’t execute"
    )
    common.add_argument(
        "--verbose", action="store_true", help="(Reserved) Verbose logging"
    )

    p_install = sub.add_parser(
        "install",
        parents=[common],
        help="Full install into <base>/ComfyUI (fresh venv by default).",
    )
    p_install.add_argument(
        "--python",
        dest="pyver",
        default=None,
        help="Python launcher version (Windows), e.g. 3.11",
    )
    p_install.add_argument(
        "--cuda",
        choices=["cu128", "cu121", "cu118", "cpu"],
        required=True,
        help="Torch build",
    )
    p_install.add_argument(
        "--with-manager", action="store_true", help="Install ComfyUI-Manager"
    )
    p_install.add_argument(
        "--skip-nodes",
        action="store_true",
        help="Skip the curated custom-node stack from config/nodes.json",
    )
    p_install.add_argument(
        "--models",
        choices=["5b", "14b", "i2v", "ltx", "ltx2", "all"],
        default=None,
        help="Model set (Wan 2.2 groups, LTX-Video, or all)",
    )
    p_install.add_argument(
        "--hf-token", default=None, help="Hugging Face token (optional)"
    )
    p_install.add_argument(
        "--model-repository",
        type=model_repository,
        default=os.getenv("CUSTOM_WAN_MODEL_REPOSITORY", WAN_REPO),
        help="Hugging Face model repository (owner/repository)",
    )
    p_install.add_argument(
        "--model-revision",
        type=model_revision,
        default=os.getenv("CUSTOM_WAN_MODEL_REVISION", WAN_REVISION),
        help="Hugging Face branch, tag, or immutable commit",
    )
    p_install.add_argument(
        "--start", action="store_true", help="Start ComfyUI when done"
    )
    p_install.add_argument(
        "--port",
        type=port_number,
        default=DEFAULT_PORT,
        help=f"TCP port (1-65535; default: {DEFAULT_PORT})",
    )
    p_install.add_argument("--listen-all", action="store_true", help="Bind 0.0.0.0")
    p_install.add_argument(
        "--reuse-venv",
        action="store_true",
        help="Reuse existing .venv instead of recreating",
    )
    p_install.add_argument(
        "--extra-requirements",
        type=Path,
        default=None,
        help="Optional reviewed requirements file; no local snapshot is loaded by default",
    )

    p_models = sub.add_parser(
        "models", parents=[common], help="Download/refresh Wan 2.2 models (uses venv)."
    )
    p_models.add_argument(
        "--models", choices=["5b", "14b", "i2v", "ltx", "ltx2", "all"], required=True
    )
    p_models.add_argument(
        "--hf-token", default=None, help="Hugging Face token (optional)"
    )
    p_models.add_argument(
        "--model-repository",
        type=model_repository,
        default=os.getenv("CUSTOM_WAN_MODEL_REPOSITORY", WAN_REPO),
        help="Hugging Face model repository (owner/repository)",
    )
    p_models.add_argument(
        "--model-revision",
        type=model_revision,
        default=os.getenv("CUSTOM_WAN_MODEL_REVISION", WAN_REVISION),
        help="Hugging Face branch, tag, or immutable commit",
    )

    p_react = sub.add_parser(
        "react", parents=[common], help="Create a React loader app beside ComfyUI."
    )
    p_react.add_argument("--name", default="comfy-loader", help="App folder name")
    p_react.add_argument(
        "--url",
        default=os.getenv("CUSTOM_WAN_COMFYUI_URL", f"http://127.0.0.1:{DEFAULT_PORT}"),
        help="Comfy base URL",
    )

    p_start = sub.add_parser(
        "start", parents=[common], help="Run ComfyUI from the local venv."
    )
    p_start.add_argument(
        "--port",
        type=port_number,
        default=DEFAULT_PORT,
        help=f"TCP port (1-65535; default: {DEFAULT_PORT})",
    )
    p_start.add_argument("--listen-all", action="store_true")

    args = parser.parse_args()
    dry = args.dry_run

    if args.cmd in ("install", "models") and args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    ensure_tools(["git", "ffmpeg"])

    base: Path = args.path.expanduser().resolve()
    ensure_base_dir(base, dry=dry)

    if args.cmd == "install":
        # 1) Ensure ComfyUI code exists in subfolder
        clone_comfy(base, dry=dry)
        ensure_model_dirs(base, dry=dry)

        # 2) Venv (fresh by default)
        if args.reuse_venv:
            venv = ensure_venv(base, args.pyver, dry=dry)
        else:
            venv = recreate_venv(base, args.pyver, dry=dry)

        # 3) Python deps
        install_torch(venv, args.cuda, dry=dry)
        install_comfy_requirements(
            venv,
            base,
            extra_requirements=args.extra_requirements,
            dry=dry,
        )
        if args.with_manager:
            install_manager(venv, base, dry=dry)
        if not args.skip_nodes:
            install_required_nodes(venv, base, dry=dry)

        python = py_exec(venv_bin=venv / ("Scripts" if is_windows() else "bin"))
        if is_windows():
            repair_script = Path(__file__).resolve().parent / "patches" / "repair_custom_nodes.py"
            if repair_script.is_file():
                run(
                    [
                        python,
                        str(repair_script),
                        "--comfyui",
                        str(comfy_root(base)),
                        "--install-dependencies",
                    ],
                    dry=dry,
                )
        run([python, "-m", "pip", "check"], cwd=comfy_root(base), dry=dry)

        # 4) Models (optional)
        if args.models:
            download_models(
                venv,
                base,
                args.models,
                args.model_repository,
                args.model_revision,
                dry=dry,
            )

        # 5) Done / maybe start
        act = (
            str(venv / "Scripts" / "Activate.ps1")
            if is_windows()
            else f"source {venv}/bin/activate"
        )
        log(f"Done. To activate venv manually: {act}")
        if args.start:
            start_comfy(venv, base, args.port, args.listen_all, dry=dry)

    elif args.cmd == "models":
        venv = ensure_venv(base, py_ver=None, dry=dry)
        download_models(
            venv,
            base,
            args.models,
            args.model_repository,
            args.model_revision,
            dry=dry,
        )
        log("Models downloaded.")

    elif args.cmd == "react":
        target = make_react(base, args.name, args.url, dry=dry)
        if target:
            log(f"React app created at: {target}")
            log("Run it with: npm run dev (inside the app directory)")

    elif args.cmd == "start":
        venv = comfy_root(base) / VENV_DIR
        start_comfy(venv, base, args.port, args.listen_all, dry=dry)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user.")
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
