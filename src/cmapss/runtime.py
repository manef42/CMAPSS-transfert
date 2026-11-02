"""CPU/CUDA determinism, epoch-boundary faithful resume and atomic artifacts."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import numpy as np
import torch


def seed_all(seed, deterministic=False, threads=1):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save_checkpoint(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_checkpoint(path):
    # Contains Python/NumPy RNG state: only load your own trusted checkpoints.
    return torch.load(path, map_location="cpu", weights_only=False)


def fingerprints(paths):
    out = {}
    for p in paths:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        out[str(p)] = h.hexdigest()
    return out


def environment(device):
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        git, dirty = None, None
    packages = {}
    for name in ["torch", "numpy", "pandas", "PyYAML", "matplotlib"]:
        packages[name] = importlib.metadata.version(name)
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "git_commit": git, "git_dirty": dirty, "device": str(device),
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
