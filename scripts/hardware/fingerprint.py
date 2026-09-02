#!/usr/bin/env python3
"""Hardware + runtime fingerprint for the ShinrAI hardware map (research/hardware-map/).

Prints one JSON object describing the machine a benchmark ran on: CPU model, core
counts, ISA flags that matter for GEMM (AVX-512 VNNI, AMX, NEON dot-product, I8MM,
SME), RAM, cgroup CPU quota (containers), GPU, OS, Python and ONNX Runtime build.

The record is meant to be published next to benchmark results, so it carries no
hostname, no IP, no username. Identify the machine with --platform-id instead.

    python scripts/hardware/fingerprint.py --platform-id mac-mini-m4-pro
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
from datetime import UTC, datetime
from pathlib import Path

# ISA flags that change GEMM throughput. Everything else is noise for the map.
X86_FLAGS = (
    "sse4_2", "avx", "avx2", "fma", "f16c",
    "avx512f", "avx512_vnni", "avx512_bf16", "avx512_fp16", "avx_vnni",
    "amx_tile", "amx_int8", "amx_bf16",
)
ARM_FLAGS = (
    "asimd", "asimdhp", "asimddp", "fphp", "bf16", "i8mm",
    "sve", "sve2", "sveint8mm", "sme", "sme2",
)
APPLE_FEATURES = (
    "AdvSIMD", "FEAT_DotProd", "FEAT_I8MM", "FEAT_BF16", "FEAT_FP16",
    "FEAT_SME", "FEAT_SME2", "FEAT_SVE",
)


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    if shutil.which(cmd[0]) is None:
        return ""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip()


def _sysctl(key: str) -> str:
    return _run(["sysctl", "-n", key])


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _int(value: str) -> int | None:
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return None


# ---- CPU ----------------------------------------------------------------------


def _cpu_darwin() -> dict:
    info: dict = {
        "model": _sysctl("machdep.cpu.brand_string"),
        "physical_cores": _int(_sysctl("hw.physicalcpu")),
        "logical_cores": _int(_sysctl("hw.logicalcpu")),
    }
    perf = _int(_sysctl("hw.perflevel0.physicalcpu"))
    eff = _int(_sysctl("hw.perflevel1.physicalcpu"))
    if perf is not None:
        info["performance_cores"] = perf
    if eff is not None:
        info["efficiency_cores"] = eff
    flags = []
    for feat in APPLE_FEATURES:
        if _sysctl(f"hw.optional.arm.{feat}") == "1":
            flags.append(feat)
    info["isa"] = flags
    freq = _int(_sysctl("hw.cpufrequency_max"))
    if freq:
        info["max_mhz"] = round(freq / 1e6)
    return info


def _lscpu_map() -> dict[str, str]:
    out = _run(["lscpu"])
    result: dict[str, str] = {}
    for line in out.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result


def _cpu_linux() -> dict:
    cpuinfo = _read("/proc/cpuinfo")
    lscpu = _lscpu_map()
    info: dict = {"logical_cores": os.cpu_count()}

    model = lscpu.get("Model name")
    if not model:
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
        model = match.group(1).strip() if match else None
    if not model:
        # ARM boards often expose only "CPU part"; lscpu translates it, /proc does not.
        match = re.search(r"^Hardware\s*:\s*(.+)$", cpuinfo, re.M)
        model = match.group(1).strip() if match else platform.processor() or None
    info["model"] = model

    board = _read("/proc/device-tree/model").replace("\x00", "").strip()
    if board:
        info["board"] = board  # e.g. "Raspberry Pi 4 Model B Rev 1.5"

    cores_per_socket = _int(lscpu.get("Core(s) per socket", ""))
    sockets = _int(lscpu.get("Socket(s)", ""))
    if cores_per_socket and sockets:
        info["physical_cores"] = cores_per_socket * sockets
    else:
        pairs = set(re.findall(r"^physical id\s*:\s*(\d+)\n(?:.*\n)*?core id\s*:\s*(\d+)", cpuinfo, re.M))
        info["physical_cores"] = len(pairs) or os.cpu_count()

    flag_line = ""
    match = re.search(r"^(?:flags|Features)\s*:\s*(.+)$", cpuinfo, re.M)
    if match:
        flag_line = match.group(1)
    elif lscpu.get("Flags"):
        flag_line = lscpu["Flags"]
    present = set(flag_line.split())
    wanted = X86_FLAGS if platform.machine() in ("x86_64", "AMD64") else ARM_FLAGS
    info["isa"] = [f for f in wanted if f in present]

    max_mhz = lscpu.get("CPU max MHz") or lscpu.get("CPU MHz")
    if max_mhz:
        try:
            info["max_mhz"] = round(float(max_mhz.replace(",", ".")))
        except ValueError:
            pass

    try:
        info["affinity_cores"] = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        pass

    quota = _cgroup_cpu_quota()
    if quota is not None:
        info["cgroup_cpu_limit"] = quota

    throttled = _run(["vcgencmd", "get_throttled"])  # Raspberry Pi only
    if throttled:
        info["rpi_throttled"] = throttled.replace("throttled=", "")
    return info


def _cgroup_cpu_quota() -> float | None:
    """CPU limit imposed by cgroup v2 (`cpu.max`) or v1 (`cpu.cfs_quota_us`)."""
    v2 = _read("/sys/fs/cgroup/cpu.max").split()
    if len(v2) == 2 and v2[0] != "max":
        try:
            return round(int(v2[0]) / int(v2[1]), 2)
        except ValueError:
            return None
    quota = _int(_read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period = _int(_read("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if quota and period and quota > 0:
        return round(quota / period, 2)
    return None


def cpu_info() -> dict:
    system = platform.system()
    if system == "Darwin":
        info = _cpu_darwin()
    elif system == "Linux":
        info = _cpu_linux()
    else:
        info = {"model": platform.processor() or None, "logical_cores": os.cpu_count(), "isa": []}
    info["arch"] = platform.machine()
    return info


# ---- memory / GPU / OS / runtime -------------------------------------------------


def memory_info() -> dict:
    info: dict = {}
    system = platform.system()
    if system == "Darwin":
        total = _int(_sysctl("hw.memsize"))
    else:
        match = re.search(r"^MemTotal:\s*(\d+)\s*kB", _read("/proc/meminfo"), re.M)
        total = int(match.group(1)) * 1024 if match else None
    if total:
        info["total_gb"] = round(total / 2**30, 1)
    limit = _read("/sys/fs/cgroup/memory.max").strip()
    if limit and limit != "max" and limit.isdigit():
        info["cgroup_limit_gb"] = round(int(limit) / 2**30, 2)
    return info


def gpu_info() -> list[dict]:
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader",
    ])
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            gpus.append({
                "name": parts[0],
                "memory": parts[1],
                "driver": parts[2] if len(parts) > 2 else None,
                "compute_capability": parts[3] if len(parts) > 3 else None,
            })
    tegra = _read("/etc/nv_tegra_release").strip()
    if tegra:
        gpus.append({"name": "NVIDIA Jetson (integrated)", "tegra_release": tegra.splitlines()[0][:80]})
    return gpus


def os_info() -> dict:
    system = platform.system()
    info: dict = {"system": system, "release": platform.release()}
    if system == "Darwin":
        info["name"] = f"macOS {_run(['sw_vers', '-productVersion'])}"
    elif system == "Linux":
        match = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', _read("/etc/os-release"), re.M)
        info["name"] = match.group(1) if match else "Linux"
        cgroup = _read("/proc/1/cgroup") + _read("/proc/self/cgroup")
        info["container"] = bool(
            Path("/.dockerenv").exists() or "kubepods" in cgroup or "docker" in cgroup
        )
    return info


def runtime_info() -> dict:
    info: dict = {"python": platform.python_version()}
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["onnxruntime_providers"] = ort.get_available_providers()
        try:
            build = ort.get_build_info()
            match = re.search(r"git-commit-id=([0-9a-f]+)", build)
            info["onnxruntime_commit"] = match.group(1) if match else build[:80]
        except Exception:  # noqa: BLE001 — optional detail
            pass
    except ImportError:
        info["onnxruntime"] = None
    for module in ("numpy", "transformers", "tokenizers"):
        try:
            info[module] = __import__(module).__version__
        except ImportError:
            info[module] = None
    return info


def fingerprint(platform_id: str | None = None, note: str | None = None) -> dict:
    record = {
        "fingerprint_version": "1.0",
        "platform_id": platform_id,
        "captured_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cpu": cpu_info(),
        "memory": memory_info(),
        "gpus": gpu_info(),
        "os": os_info(),
        "runtime": runtime_info(),
    }
    if note:
        record["note"] = note
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform-id", default=None, help="slug naming this machine class, e.g. pi4-4gb")
    parser.add_argument("--note", default=None, help="free text: cooling, power mode, VM/bare metal")
    args = parser.parse_args()
    json.dump(fingerprint(args.platform_id, args.note), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
