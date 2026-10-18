#!/usr/bin/env python3
"""Small launcher shim for the platform-managed ComfyUI checkout."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from wan2_cli_args import port_number


def env_setting(name: str, default: str = "") -> str:
    """Read OVS_<name>, with the legacy CUSTOM_WAN_<name> as fallback."""
    for prefix in ("OVS_", "CUSTOM_WAN_"):
        value = os.getenv(prefix + name, "")
        if value:
            return value
    return default


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def rocm_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None))
    except Exception:
        return False


def cuda_is_blackwell() -> bool:
    """True on RTX 50-series and newer (compute capability 10.0+, sm_120 etc).

    cudaMallocAsync's stream-ordered allocator, combined with the extra CUDA
    streams async weight offload creates, has produced host-side memory
    corruption (a hard access-violation crash, not a catchable exception)
    during tiled VAE decode on an RTX 5090 -- a young allocator meeting a
    freshly-supported architecture. RTX 30/40-series (capability 8.x) don't
    hit this; disabling cudaMallocAsync there would only cost them its real
    performance benefit for no reason, so this stays scoped to Blackwell.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability(0)[0] >= 10
    except Exception:
        return False


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def windows_memory_status() -> dict[str, int] | None:
    """Physical RAM and commit (RAM + page file) totals on Windows, else None."""
    if os.name != "nt":
        return None
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    try:
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
    except Exception:
        return None
    return {
        "total_ram": status.ullTotalPhys,
        "available_ram": status.ullAvailPhys,
        "commit_limit": status.ullTotalPageFile,
        "available_commit": status.ullAvailPageFile,
    }


# Below this much RAM, Windows hosts start ComfyUI without pinned memory.
# ComfyUI page-locks up to 40% of RAM on Windows for faster offload. Locked
# pages can never be swapped out, so on a 32 GB machine running a large image
# model next to an isolated custom-node worker (TRELLIS2) the system ran out of
# lockable memory: allocations of a few MB failed ("DefaultCPUAllocator: not
# enough memory", WinError 1450) and the process or its worker died without a
# Python traceback. Unpinned offload is slower but pageable, so it survives.
LOW_MEMORY_HOST_BYTES = 48 * 1024**3
LOW_COMMIT_WARNING_BYTES = 32 * 1024**3
LOW_RAM_WARNING_BYTES = 4 * 1024**3


def pinned_memory_is_risky(status: dict[str, int] | None) -> bool:
    if status is None or env_setting("PINNED_MEMORY").strip() == "1":
        return False
    return status["total_ram"] < LOW_MEMORY_HOST_BYTES


def memory_warnings(status: dict[str, int] | None) -> list[str]:
    if status is None:
        return []
    gib = 1024**3
    warnings = []
    if status["available_commit"] < LOW_COMMIT_WARNING_BYTES:
        warnings.append(
            f"only {status['available_commit'] / gib:.1f} GB of "
            f"{status['commit_limit'] / gib:.1f} GB commit (RAM + page file) is "
            "free. Large video and image models can crash ComfyUI without a "
            "traceback; close other heavy programs or enlarge the page file."
        )
    if status["available_ram"] < LOW_RAM_WARNING_BYTES:
        warnings.append(
            f"only {status['available_ram'] / gib:.1f} GB of "
            f"{status['total_ram'] / gib:.1f} GB RAM is free."
        )
    return warnings


# Device-visibility variables that CUDA and ROCm read when they initialize.
GPU_VISIBILITY_VARIABLES = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def hides_every_gpu(value: str) -> bool:
    """True when a *_VISIBLE_DEVICES value leaves no device visible.

    CUDA stops reading the list at the first invalid index, so an empty value
    or one that starts with -1 hides every GPU.
    """
    first = value.split(",")[0].strip()
    return first == "" or first == "-1"


def reveal_hidden_gpus(environ: dict[str, str] | os._Environ) -> list[str]:
    """Drop visibility variables that hide every GPU; return the names dropped."""
    dropped = [
        name
        for name in GPU_VISIBILITY_VARIABLES
        if name in environ and hides_every_gpu(environ[name])
    ]
    for name in dropped:
        del environ[name]
    return dropped


PIP_NETWORK_OPTIONS = ["--retries", "10", "--timeout", "120"]

# ComfyUI's mutually exclusive attention backends. Any of these in COMFYUI_ARGS
# means the user picked one, so no default should be added on top.
ATTENTION_ARGUMENTS = frozenset(
    {
        "--use-pytorch-cross-attention",
        "--use-split-cross-attention",
        "--use-quad-cross-attention",
        "--use-sage-attention",
        "--use-flash-attention",
        "--use-ck-attention",
    }
)

BACKEND_PROBES = {
    "directml": (
        "import sys, torch_directml; "
        "sys.exit(0 if torch_directml.is_available() else 1)"
    ),
    "cuda": "import sys, torch; sys.exit(0 if torch.version.cuda else 1)",
    # Build presence only. Whether the GPU is actually reachable is checked
    # separately, so a driver problem reports itself instead of triggering a
    # multi-gigabyte reprovision on every launch.
    "rocm": "import sys, torch; sys.exit(0 if torch.version.hip else 1)",
}


def run_checked(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"[wan2_cli] > {printable}")
    if subprocess.call(command) != 0:
        raise SystemExit(f"Command failed: {printable}")


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def python_provides_backend(python_path: Path, backend: str) -> bool:
    probe = [str(python_path), "-c", BACKEND_PROBES[backend]]
    return subprocess.call(probe) == 0


def installed_versions(python_path: Path, packages: list[str]) -> list[str]:
    code = (
        "import importlib.metadata, sys\n"
        "for name in sys.argv[1:]:\n"
        "    print(name + '==' + importlib.metadata.version(name))\n"
    )
    result = subprocess.run(
        [str(python_path), "-c", code, *packages],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Unable to read installed package versions: " + result.stderr.strip()
        )
    return result.stdout.split()


INTEGRATED_ADAPTER_MARKERS = ("(tm) graphics", "uhd graphics", "iris", "microsoft basic render")


def pick_directml_device(python_path: Path) -> int:
    """Pick the DirectML adapter index, preferring a discrete AMD GPU.

    DirectML's default adapter can be an integrated GPU, and ComfyUI reports
    a blank name for the implicit default. CUSTOM_WAN_DIRECTML_DEVICE
    overrides the selection.
    """
    override = env_setting("DIRECTML_DEVICE").strip()
    if override:
        return int(override)
    code = (
        "import torch_directml\n"
        "for i in range(torch_directml.device_count()):\n"
        "    print(torch_directml.device_name(i))\n"
    )
    result = subprocess.run(
        [str(python_path), "-c", code], capture_output=True, text=True
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not names:
        return 0
    for index, name in enumerate(names):
        print(f"[wan2_cli] DirectML adapter {index}: {name}")

    def integrated(name: str) -> bool:
        return any(marker in name.lower() for marker in INTEGRATED_ADAPTER_MARKERS)

    selected = 0
    discrete_amd = [
        i
        for i, name in enumerate(names)
        if ("amd" in name.lower() or "radeon" in name.lower())
        and not integrated(name)
    ]
    discrete_any = [i for i, name in enumerate(names) if not integrated(name)]
    if discrete_amd:
        selected = discrete_amd[0]
    elif discrete_any:
        selected = discrete_any[0]
    print(
        f"[wan2_cli] Using DirectML adapter {selected} ({names[selected]}); "
        "set CUSTOM_WAN_DIRECTML_DEVICE to override."
    )
    return selected


REQUIREMENTS_STAMP = "ovs-requirements.sha256"

# Either of these in COMFYUI_ARGS already turns the manager on, and repeating
# the flag would be redundant.
MANAGER_ARGUMENTS = frozenset({"--enable-manager", "--enable-manager-legacy-ui"})

# An explicit choice either way in COMFYUI_ARGS wins over the Blackwell default.
CUDA_MALLOC_ARGUMENTS = frozenset({"--cuda-malloc", "--disable-cuda-malloc"})
# An explicit choice in COMFYUI_ARGS wins over the low-memory default.
PINNED_MEMORY_ARGUMENTS = frozenset({"--disable-pinned-memory"})


def torch_stack_constraints(python_path: Path, venv_dir: Path) -> Path | None:
    """Constraints file pinning an environment's torch stack, written if absent.

    Environments the installer built have no constraints file, so one is
    derived from what is installed. Without it an unpinned requirement can
    replace the torch build the environment exists to provide.
    """
    constraints_path = venv_dir / "pip-constraints.txt"
    if constraints_path.exists():
        return constraints_path
    code = (
        "import importlib.metadata as metadata, sys\n"
        "for name in sys.argv[1:]:\n"
        "    try:\n"
        "        print(name + '==' + metadata.version(name))\n"
        "    except metadata.PackageNotFoundError:\n"
        "        pass\n"
    )
    result = subprocess.run(
        [
            str(python_path),
            "-c",
            code,
            "torch",
            "torchvision",
            "torchaudio",
            "torch-directml",
        ],
        capture_output=True,
        text=True,
    )
    pins = result.stdout.split()
    if not pins:
        return None
    constraints_path.write_text("\n".join(pins) + "\n", encoding="utf-8")
    return constraints_path


def ensure_comfyui_requirements(python_path: Path, comfyui_dir: Path) -> None:
    """Reinstall ComfyUI's requirements when the checkout's have changed.

    Updating the checkout moves `requirements.txt`, but nothing re-runs pip for
    an environment that already exists: the backend probe only checks torch, so
    the drift stays invisible until ComfyUI fails at import against the older
    pins. Stamping the file's digest into the environment makes the check cheap
    enough to run on every launch.
    """
    venv_dir = python_path.parent.parent
    requirement_files = [
        path
        for path in (
            comfyui_dir / "requirements.txt",
            # ComfyUI keeps the manager pinned separately, and skipping it
            # leaves --enable-manager silently switched off.
            comfyui_dir / "manager_requirements.txt",
        )
        if path.exists()
    ]
    if not requirement_files or venv_site_packages(venv_dir) is None:
        return
    digest_source = hashlib.sha256()
    for path in requirement_files:
        digest_source.update(path.read_bytes())
    digest = digest_source.hexdigest()
    stamp = venv_dir / REQUIREMENTS_STAMP
    if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == digest:
        return

    print(f"[wan2_cli] ComfyUI requirements changed; updating {venv_dir}")
    pip = [str(python_path), "-m", "pip", "install", *PIP_NETWORK_OPTIONS]
    constraints = torch_stack_constraints(python_path, venv_dir)
    if constraints is not None:
        pip.extend(["-c", str(constraints)])
    for path in requirement_files:
        pip.extend(["-r", str(path)])
    run_checked(pip)
    stamp.write_text(digest + "\n", encoding="utf-8")


# comfy-kitchen 0.2.28 and newer declare their custom ops with PEP 585
# annotations (`list[int]`) that torch's schema inference only accepts from 2.5
# on, and torch-directml 0.2.5 pins torch 2.4.1, so those builds cannot import
# there at all. 0.2.27 is the newest build that can, and the shim below fills
# in what ComfyUI expects from the newer ones.
DIRECTML_KITCHEN_PIN = "comfy-kitchen==0.2.27"
KITCHEN_COMPAT_SOURCE = (
    Path(__file__).resolve().parent / "compat" / "comfy_kitchen_torch24.py"
)
KITCHEN_COMPAT_MODULE = "ovs_comfy_kitchen_torch24"


def venv_site_packages(venv_dir: Path) -> Path | None:
    if not (venv_dir / "pyvenv.cfg").exists():
        return None
    for pattern in ("[Ll]ib/site-packages", "lib/python*/site-packages"):
        for path in sorted(venv_dir.glob(pattern)):
            if path.is_dir():
                return path
    return None


def ensure_directml_kitchen_compat(python_path: Path) -> None:
    """Hold comfy-kitchen at the last DirectML-compatible build and shim it.

    This runs on every DirectML launch rather than only during provisioning:
    installing a custom node's requirements can pull the newer comfy-kitchen
    back in, and ComfyUI then fails to start at all.
    """
    venv_dir = python_path.parent.parent
    site_packages = venv_site_packages(venv_dir)
    if site_packages is None:
        return
    if not KITCHEN_COMPAT_SOURCE.exists():
        raise SystemExit(
            f"The comfy-kitchen compatibility shim is missing: {KITCHEN_COMPAT_SOURCE}"
        )

    pinned = DIRECTML_KITCHEN_PIN.split("==")[1]
    installed = [
        path.name.removesuffix(".dist-info").split("-")[-1]
        for path in site_packages.glob("comfy_kitchen-*.dist-info")
    ]
    if installed != [pinned]:
        print(f"[wan2_cli] Holding comfy-kitchen at {pinned} for torch-directml")
        pip = [str(python_path), "-m", "pip", "install", *PIP_NETWORK_OPTIONS]
        constraints = venv_dir / "pip-constraints.txt"
        if constraints.exists():
            pip.extend(["-c", str(constraints)])
        run_checked([*pip, "--no-deps", DIRECTML_KITCHEN_PIN])

    module_path = site_packages / f"{KITCHEN_COMPAT_MODULE}.py"
    if (
        not module_path.exists()
        or module_path.read_bytes() != KITCHEN_COMPAT_SOURCE.read_bytes()
    ):
        shutil.copyfile(KITCHEN_COMPAT_SOURCE, module_path)
    # A .pth line starting with "import" is executed by site, which installs the
    # shim's import hook before ComfyUI ever imports comfy_kitchen.
    pth_path = site_packages / f"{KITCHEN_COMPAT_MODULE}.pth"
    pth_line = f"import {KITCHEN_COMPAT_MODULE}\n"
    if not pth_path.exists() or pth_path.read_text(encoding="utf-8") != pth_line:
        pth_path.write_text(pth_line, encoding="utf-8")


# pytorch.org ships ROCm builds for Linux only; on Windows they come from AMD's
# per-architecture wheel indexes, which are published for CPython 3.11-3.13.
ROCM_WINDOWS_INDEX = "https://rocm.nightlies.amd.com/v2/{gfx}/"
ROCM_WINDOWS_PYTHON_RANGE = ((3, 11), (3, 13))
ROCM_RADEON_SERIES_GFX = (
    (9000, "gfx120X-all"),
    (7000, "gfx110X-dgpu"),
    (6000, "gfx103X-dgpu"),
    (5000, "gfx101X-dgpu"),
)


def windows_adapter_names() -> list[str]:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | ForEach-Object Name",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gfx_for_adapter(name: str) -> str | None:
    """Map an adapter name to AMD's ROCm wheel index, or None if unsupported."""
    match = re.search(r"\bradeon\s+(?:rx|pro\s+w)\s*(\d{4})\b", name.lower())
    if not match:
        return None
    model = int(match.group(1))
    for floor, gfx in ROCM_RADEON_SERIES_GFX:
        if model >= floor:
            return gfx
    return None


def detect_rocm_gfx() -> str:
    for name in windows_adapter_names():
        gfx = gfx_for_adapter(name)
        if gfx:
            print(f"[wan2_cli] {name} uses the {gfx} ROCm wheel index")
            return gfx
    raise SystemExit(
        "No ROCm-capable Radeon GPU was recognised. Set OVS_ROCM_GFX to the "
        "architecture your card uses (for example gfx110X-dgpu for RX 7000, "
        "gfx120X-all for RX 9000); the published names are listed at "
        "https://rocm.nightlies.amd.com/v2/"
    )


def rocm_index_url() -> str:
    override = env_setting("ROCM_INDEX").strip()
    if override:
        return override
    if os.name != "nt":
        rocm_build = env_setting("TORCH_ROCM").strip() or "rocm6.4"
        return f"https://download.pytorch.org/whl/{rocm_build}"
    gfx = env_setting("ROCM_GFX").strip() or detect_rocm_gfx()
    return ROCM_WINDOWS_INDEX.format(gfx=gfx)


def windows_interpreters() -> dict[tuple[int, int], str]:
    """Map CPython version to interpreter path, as reported by the py launcher."""
    try:
        result = subprocess.run(["py", "-0p"], capture_output=True, text=True)
    except OSError:
        return {}
    if result.returncode != 0:
        return {}
    found: dict[tuple[int, int], str] = {}
    for line in result.stdout.splitlines():
        match = re.match(r"\s*-V:(\d+)\.(\d+)\S*\s+(\S.*?)\s*$", line)
        if match:
            version = (int(match.group(1)), int(match.group(2)))
            found.setdefault(version, match.group(3))
    return found


def base_interpreter(backend: str) -> str:
    """Return the interpreter a backend's venv should be built from."""
    if backend != "rocm" or os.name != "nt":
        return sys.executable
    override = env_setting("ROCM_PYTHON").strip()
    if override:
        return override
    low, high = ROCM_WINDOWS_PYTHON_RANGE
    if low <= sys.version_info[:2] <= high:
        return sys.executable
    for version, path in sorted(windows_interpreters().items(), reverse=True):
        if low <= version <= high:
            print(
                f"[wan2_cli] Building the ROCm environment with Python "
                f"{version[0]}.{version[1]} ({path})"
            )
            return path
    raise SystemExit(
        "The ROCm backend needs CPython 3.11-3.13 on Windows; AMD publishes no "
        f"ROCm wheels for {sys.version_info[0]}.{sys.version_info[1]}. Install a "
        "supported version, or point OVS_ROCM_PYTHON at one."
    )


def ensure_backend_venv(backend: str, comfyui_dir: Path) -> Path:
    """Return a Python that provides `backend`, provisioning a venv on demand.

    CUDA and DirectML torch builds cannot share one environment, so each
    backend gets a sibling venv next to the primary one. Selecting a device
    the current environment does not provide creates and provisions that
    sibling once; later selections reuse it.
    """
    venv_dir = comfyui_dir / f".venv-{backend}"
    python_path = venv_python_path(venv_dir)
    if python_path.exists() and python_provides_backend(python_path, backend):
        return python_path

    print(f"[wan2_cli] Provisioning the {backend} environment at {venv_dir}")
    if not python_path.exists():
        run_checked([base_interpreter(backend), "-m", "venv", str(venv_dir)])
    pip = [str(python_path), "-m", "pip", "install", *PIP_NETWORK_OPTIONS]
    run_checked([*pip, "--upgrade", "pip", "setuptools<82", "wheel"])
    if backend == "directml":
        # Install torch-directml on its own first so the resolver honors its
        # exact torch/torchvision pins instead of backtracking to an older
        # plugin build to satisfy unrelated newer packages.
        run_checked([*pip, "--upgrade", "torch-directml", "onnxruntime-directml"])
        pins = installed_versions(
            python_path, ["torch", "torchvision", "torch-directml"]
        )
    elif backend == "rocm":
        # The ROCm index resolves torch's own dependencies, so it is used
        # alone. Adding PyPI as an extra index makes pip prefer the plain
        # PyPI torch instead, because its version outranks the ROCm build's.
        run_checked(
            [
                *pip,
                "--upgrade",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                rocm_index_url(),
            ]
        )
        pins = installed_versions(python_path, ["torch", "torchvision", "torchaudio"])
    else:
        cuda_build = env_setting("TORCH_CUDA").strip() or "cu128"
        run_checked(
            [
                *pip,
                "--upgrade",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                f"https://download.pytorch.org/whl/{cuda_build}",
            ]
        )
        run_checked([*pip, "--upgrade", "onnxruntime-gpu"])
        pins = installed_versions(python_path, ["torch", "torchvision", "torchaudio"])

    # Every later install is constrained to the pinned torch stack. Without
    # this, an unpinned custom-node requirement (for example a package that
    # needs a newer torch) silently replaces the backend's torch build and
    # breaks it.
    constraints_path = venv_dir / "pip-constraints.txt"
    constraints_path.write_text("\n".join(pins) + "\n", encoding="utf-8")
    constrained_pip = [*pip, "-c", str(constraints_path)]
    if backend == "directml":
        # Pin torchaudio to the torch version explicitly: a bare "torchaudio"
        # is "already satisfied" by a stale mismatched install and stays put.
        torch_pin = next(pin for pin in pins if pin.startswith("torch=="))
        run_checked([*constrained_pip, "torchaudio" + torch_pin.removeprefix("torch")])
        pins = installed_versions(
            python_path, ["torch", "torchvision", "torchaudio", "torch-directml"]
        )
        constraints_path.write_text("\n".join(pins) + "\n", encoding="utf-8")

    ensure_comfyui_requirements(python_path, comfyui_dir)
    for node_requirements in sorted(
        (comfyui_dir / "custom_nodes").glob("*/requirements.txt")
    ):
        if subprocess.call([*constrained_pip, "-r", str(node_requirements)]) != 0:
            print(
                "[wan2_cli] WARNING: node requirements skipped (incompatible "
                f"with the pinned torch stack): {node_requirements}"
            )
    if backend == "directml":
        ensure_directml_kitchen_compat(python_path)
    if not python_provides_backend(python_path, backend):
        raise SystemExit(
            f"Provisioning finished, but the {backend} backend still does not "
            f"work in {python_path}. Delete {venv_dir} and select the device "
            "again to rebuild it."
        )
    return python_path


def backend_arguments(
    extra_args: list[str],
    rocm: bool,
    blackwell: bool = False,
    low_memory: bool = False,
) -> list[str]:
    """Merge COMFYUI_ARGS with the arguments a backend contributes.

    Backend defaults are added alongside COMFYUI_ARGS rather than instead of
    it; setting any extra argument used to drop them silently. An explicit
    flag in COMFYUI_ARGS always wins over the matching default.
    """
    arguments = []
    if rocm and not ATTENTION_ARGUMENTS.intersection(extra_args):
        arguments.append("--use-pytorch-cross-attention")
    if blackwell and not CUDA_MALLOC_ARGUMENTS.intersection(extra_args):
        # cudaMallocAsync's stream-ordered allocator, combined with async
        # weight offload's extra CUDA streams, produced a hard access
        # violation (uncatchable memory corruption, not a Python exception)
        # during tiled VAE decode on an RTX 5090. Scoped to Blackwell only --
        # RTX 30/40-series don't hit this and would just lose the allocator's
        # real performance benefit for nothing.
        arguments.append("--disable-cuda-malloc")
    if low_memory and not PINNED_MEMORY_ARGUMENTS.intersection(extra_args):
        # See LOW_MEMORY_HOST_BYTES.
        arguments.append("--disable-pinned-memory")
    if not MANAGER_ARGUMENTS.intersection(extra_args):
        arguments.append("--enable-manager")
    arguments.extend(extra_args)
    return arguments


def resolve_checkout_path(configured: str, root: Path) -> Path:
    if not configured:
        return (root / "ComfyUI").resolve()
    path = Path(configured).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the local ComfyUI backend.")
    parser.add_argument("command", choices=["start"])
    parser.add_argument("--path", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--host", default=env_setting("COMFYUI_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port",
        type=port_number,
        default=env_setting("COMFYUI_PORT", "8188"),
        help="TCP port (1-65535; default: 8188)",
    )
    parser.add_argument("--listen-all", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu", "rocm", "directml"],
        default=env_setting("COMFYUI_DEVICE", "auto").strip().lower() or "auto",
    )
    args = parser.parse_args()

    if args.device in ("gpu", "rocm"):
        # An explicit GPU request wins over a stray CUDA_VISIBLE_DEVICES=-1
        # (often left in the user environment by another tool), which would
        # otherwise make torch report no GPU in every environment we try.
        for name in reveal_hidden_gpus(os.environ):
            print(
                f"[wan2_cli] Ignoring {name}, which hides every GPU, because "
                f"--device {args.device} was requested."
            )

    root = Path(args.path).expanduser().resolve()
    configured_checkout = (
        env_setting("COMFYUI_CHECKOUT").strip()
        or os.getenv("CUSTOM_WAN_DOCKER_COMFYUI_CHECKOUT", "").strip()
    )
    comfyui_dir = resolve_checkout_path(configured_checkout, root)
    main_py = comfyui_dir / "main.py"
    if not main_py.exists():
        raise SystemExit(f"ComfyUI main.py was not found at {main_py}")

    hf_home = root / "hf_cache"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))

    directml_ready = False
    try:
        importlib.import_module("torch_directml")

        directml_ready = True
    except Exception:
        directml_ready = False
    gpu_ready = cuda_available()
    rocm_ready = rocm_available()

    launch_python = Path(sys.executable)
    device_arguments: list[str] = []
    using_directml = False
    if args.device == "directml":
        if not directml_ready:
            launch_python = ensure_backend_venv("directml", comfyui_dir)
        using_directml = True
        device_arguments = ["--directml", str(pick_directml_device(launch_python))]
    elif args.device == "gpu":
        if not gpu_ready:
            launch_python = ensure_backend_venv("cuda", comfyui_dir)
            cuda_probe = (
                "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
            )
            if subprocess.call([str(launch_python), "-c", cuda_probe]) != 0:
                raise SystemExit(
                    f"gpu was requested, but CUDA is unavailable in {launch_python}. "
                    "Check the NVIDIA driver, or set CUSTOM_WAN_TORCH_CUDA to a "
                    "matching build and delete that environment to provision it "
                    "again."
                )
    elif args.device == "rocm":
        if not rocm_ready:
            launch_python = ensure_backend_venv("rocm", comfyui_dir)
            rocm_probe = (
                "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
            )
            if subprocess.call([str(launch_python), "-c", rocm_probe]) != 0:
                raise SystemExit(
                    f"rocm was requested, but no GPU is visible to {launch_python}. "
                    "Check the AMD driver, or set OVS_ROCM_GFX to the architecture "
                    "your card uses and delete that environment to provision it "
                    "again."
                )
            rocm_ready = True
    elif args.device == "cpu":
        device_arguments = ["--cpu"]
    elif gpu_ready:  # auto: CUDA needs no device arguments
        pass
    elif directml_ready:
        using_directml = True
        device_arguments = ["--directml", str(pick_directml_device(launch_python))]
    else:
        device_arguments = ["--cpu"]

    # After the device is settled, so the environment about to run ComfyUI is
    # the one brought up to date. The DirectML pin has to be reapplied last:
    # ComfyUI's requirements would otherwise leave a comfy-kitchen that cannot
    # import on torch 2.4.1.
    ensure_comfyui_requirements(launch_python, comfyui_dir)
    if using_directml:
        ensure_directml_kitchen_compat(launch_python)
    pixi_fallback = Path(__file__).resolve().parent / "patches" / "comfy_env_pixi_fallback.py"
    run_checked(
        [str(launch_python), str(pixi_fallback), "--python", str(launch_python)]
    )
    node_repair = Path(__file__).resolve().parent / "patches" / "repair_custom_nodes.py"
    run_checked(
        [
            str(launch_python),
            str(node_repair),
            "--comfyui",
            str(comfyui_dir),
            "--install-dependencies",
        ]
    )
    pixal3d_patch = Path(__file__).resolve().parent / "patches" / "pixal3d_meta_loading.py"
    run_checked(
        [
            sys.executable,
            str(pixal3d_patch),
            "--comfyui",
            str(comfyui_dir),
        ]
    )

    # NOT auto-enabled: a synthetic alloc-heavy benchmark on an RTX 5090 showed
    # no measurable difference (11.7ms vs 10.9ms/step) between cudaMallocAsync
    # and the native allocator, so there's nothing confirmed here to trade
    # performance for. cuda_is_blackwell() stays available so a real, verified
    # need can gate this again later -- see CUDA_MALLOC_ARGUMENTS for the
    # explicit opt-in via COMFYUI_ARGS in the meantime.
    blackwell_ready = False

    if rocm_ready:
        # ROCm gates its fused attention kernels off on RDNA3 behind this flag,
        # leaving SDPA on the math backend. Measured on a 7900 XT, the fused
        # path is ~10x faster and needs ~30x less peak memory for an identical
        # result, and the math fallback runs a 20 GB card out of memory on
        # video workloads. An explicit setting in the environment still wins.
        os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

    memory_status = windows_memory_status()
    for warning in memory_warnings(memory_status):
        print(f"[wan2_cli] Warning: {warning}")
    low_memory = pinned_memory_is_risky(memory_status)
    if low_memory:
        print(
            "[wan2_cli] Pinned memory is off on this host (under 48 GB RAM); "
            "set OVS_PINNED_MEMORY=1 to keep it."
        )

    command = [str(launch_python), str(main_py), "--port", str(args.port)]
    if args.listen_all:
        command.extend(["--listen", "0.0.0.0"])
    elif args.host:
        command.extend(["--listen", str(args.host)])
    command.extend(device_arguments)
    command.extend(
        backend_arguments(
            shlex.split(env_setting("COMFYUI_ARGS").strip()),
            rocm_ready,
            blackwell_ready,
            low_memory,
        )
    )
    raise SystemExit(subprocess.call(command, cwd=str(comfyui_dir)))


if __name__ == "__main__":
    main()
