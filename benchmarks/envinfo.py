import json
import platform
import subprocess
import time
import uuid

import torch


def _run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def gpu_info():
    name = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    cuda = None
    out = _run(["nvidia-smi"])
    if out:
        for line in out.splitlines():
            if "CUDA Version" in line:
                cuda = line.split("CUDA Version:")[-1].strip().split()[0]
    mem = _run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    return {
        "gpu_name": name,
        "gpu_memory_total_mb": float(mem) if mem else None,
        "driver_version": driver,
        "cuda_version": cuda,
    }


def system_info():
    cpu = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    ram_total = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    ram_total = int(line.split()[1]) / 1024.0
                    break
    except Exception:
        pass
    storage = _run(["findmnt", "-no", "FSTYPE", "/"])
    return {
        "cpu_model": cpu,
        "system_ram_total_mb": ram_total,
        "root_fs_type": storage,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }


def base_metadata(engine, model, quantization=None, model_revision=None, experiment_id=None):
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiment_id": experiment_id or str(uuid.uuid4()),
        "engine": engine,
        "model": model,
        "quantization": quantization,
        "model_revision": model_revision,
    }
    meta.update(gpu_info())
    meta.update(system_info())
    return meta
