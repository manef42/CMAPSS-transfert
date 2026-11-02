"""Explicit local synthetic fixtures, not a simulator or performance benchmark."""
from pathlib import Path
import numpy as np


def generate(directory, dataset="FD001", seed=7, engines=5):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for partition in ["train", "test"]:
        rows = []
        for uid in range(1, engines + 1):
            n = 7 + uid if partition == "train" else 3 + uid
            for cycle in range(1, n + 1):
                features = rng.normal(0, .02, 24) + cycle * .01
                features[0] = 1.
                rows.append([uid, cycle, *features])
        path = directory / f"{partition}_{dataset}.txt"
        if path.exists():
            raise FileExistsError(path)
        np.savetxt(path, rows, fmt="%.8f", delimiter="   ")
    path = directory / f"RUL_{dataset}.txt"
    if path.exists():
        raise FileExistsError(path)
    np.savetxt(path, np.arange(1, engines + 1) + 2, fmt="%d")
