"""Record hardware/software identity for every experiment (spec section 4).

Anything that cannot be read is recorded as "NOT AVAILABLE", never guessed.
"""
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

NA = "NOT AVAILABLE"
PACKAGES = ["torch", "torchvision", "ultralytics", "numpy", "pandas", "opencv-python", "onnx",
            "onnxruntime", "tensorrt", "fastapi", "pydantic", "pillow", "pyyaml", "shapely", "scikit-learn"]


def _pkg_version(name: str) -> str:
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        return NA


def _run(cmd: List[str], timeout: int = 10) -> str:
    if shutil.which(cmd[0]) is None:
        return NA
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             timeout=timeout, universal_newlines=True)
        return out.stdout.strip() or NA
    except Exception:
        return NA


def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="ignore").replace("\x00", "").strip() or NA
    except Exception:
        return NA


def _torch_info() -> Dict[str, Any]:
    try:
        import torch
        info = {"torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_version": getattr(torch.version, "cuda", None) or NA,
                "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else NA}
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / 1e9, 2)
        return info
    except Exception as exc:
        return {"torch_cuda_available": NA, "note": "torch not importable: {}".format(exc)}


def _memory_gb() -> Any:
    meminfo = _read("/proc/meminfo")
    if meminfo != NA:
        for line in meminfo.splitlines():
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 1e6, 1)
    return NA


def collect_environment() -> Dict[str, Any]:
    jetson_model = _read("/proc/device-tree/model")
    is_jetson = jetson_model != NA or Path("/etc/nv_tegra_release").exists()
    env = {
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": __import__("os").cpu_count(),
        "ram_total_gb": _memory_gb(),
        "is_jetson": is_jetson,
        "jetson_model": jetson_model if is_jetson else NA,
        "l4t_release": _read("/etc/nv_tegra_release") if is_jetson else NA,
        "nvpmodel": _run(["nvpmodel", "-q"]) if is_jetson else NA,
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"]),
        "packages": {p: _pkg_version(p) for p in PACKAGES},
    }
    env.update(_torch_info())
    env["notes"] = ("Power mode, clocks, fan/cooling and ambient temperature are not auto-detected "
                    "reliably; record them manually for edge benchmarks.")
    return env


def save_environment(path) -> Dict[str, Any]:
    env = collect_environment()
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(str(p), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    return env
