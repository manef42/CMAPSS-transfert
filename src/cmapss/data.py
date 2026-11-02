"""Strict, read-only C-MAPSS IO; engine-local windows, never future-fitted scaling."""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [f"setting_{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit_id", "cycle"] + FEATURES


def read_frames(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing C-MAPSS file: {path}")
    try:
        a = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except ValueError as e:
        raise ValueError(f"{path}: invalid numeric/column data: {e}") from e
    if a.shape[0] == 0 or a.shape[1] != 26:
        raise ValueError(f"{path}: expected nonempty 26 columns, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{path}: non-finite values")
    if not np.equal(a[:, :2], np.floor(a[:, :2])).all() or (a[:, :2] < 1).any() or (a[:, :2] >= np.iinfo(np.int64).max).any():
        raise ValueError(f"{path}: unit_id and cycle must be positive integers")
    df = pd.DataFrame(a, columns=COLUMNS).astype({"unit_id": "int64", "cycle": "int64"})
    if df.duplicated(["unit_id", "cycle"]).any():
        raise ValueError(f"{path}: duplicate (unit_id, cycle)")
    if not df[["unit_id", "cycle"]].equals(df.sort_values(["unit_id", "cycle"])[["unit_id", "cycle"]]):
        raise ValueError(f"{path}: rows must be ordered by unit_id then cycle; not silently sorted")
    for uid, g in df.groupby("unit_id", sort=True):
        if not np.array_equal(g.cycle.to_numpy(), np.arange(1, len(g) + 1)):
            raise ValueError(f"{path}: engine {uid}: cycles must start at 1 and be continuous")
    return df


def train_labels(df):
    return (df.groupby("unit_id").cycle.transform("max") - df.cycle).to_numpy(dtype=float)


def terminal_labels(path, df):
    ids = np.sort(df.unit_id.unique())
    # Original NASA files associate RUL line i with engine i, not with sensor row i.
    if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise ValueError("Terminal RUL requires original contiguous engine IDs 1..N")
    a = np.loadtxt(path, dtype=float, ndmin=2)
    if a.shape != (len(ids), 1) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f"{path}: expected {len(ids)} finite nonnegative one-column RUL values")
    return {int(uid): float(value) for uid, value in zip(ids, a[:, 0])}


def transform_target(raw, cap=125.0, scale=125.0):
    if scale <= 0 or (cap is not None and cap <= 0):
        raise ValueError("target_scale and non-null rul_cap must be positive")
    transformed = np.asarray(raw, dtype=float)
    if cap is not None:
        transformed = np.minimum(transformed, cap)
    return transformed, transformed / scale


def split_engines(df, fraction=.2, seed=42):
    ids = np.sort(df.unit_id.unique())
    if len(ids) < 2 or not 0 < fraction < 1:
        raise ValueError("Split needs >=2 engines and 0 < validation_fraction < 1")
    ids = np.random.default_rng(seed).permutation(ids)
    n = max(1, min(len(ids) - 1, int(np.ceil(len(ids) * fraction))))
    return {"train": sorted(map(int, ids[n:])), "validation": sorted(map(int, ids[:n]))}


@dataclass
class Preprocessor:
    features: list
    mean: list
    scale: list
    low_variance: list
    threshold: float = 1e-3
    policy: str = "center; scale=1 for source std<threshold; otherwise source std"

    @classmethod
    def fit(cls, source_train, threshold=1e-3):
        if threshold <= 0:
            raise ValueError("low_variance_threshold must be positive")
        a = source_train[FEATURES].to_numpy(dtype=float)
        mean, std = a.mean(0), a.std(0)
        low = std < threshold
        return cls(FEATURES.copy(), mean.tolist(), np.where(low, 1., std).tolist(), low.tolist(), threshold)

    def transform(self, df):
        a = (df[self.features].to_numpy(dtype=float) - self.mean) / self.scale
        if not np.isfinite(a).all() or np.max(np.abs(a), initial=0) > np.finfo(np.float32).max:
            raise ValueError("Non-finite/float32-overflow source-transformed inputs; inspect domain shift")
        return a.astype(np.float32)

    def diagnostic(self, df):
        a = self.transform(df).astype(float)
        raw = df[self.features].to_numpy(dtype=float)
        return pd.DataFrame({"feature": self.features, "raw_std": raw.std(0),
            "source_mean": self.mean, "source_scale": self.scale, "low_variance": self.low_variance,
            "transformed_mean": a.mean(0), "transformed_std": a.std(0),
            "transformed_abs_max": np.abs(a).max(0),
            "fraction_abs_gt_10": (np.abs(a) > 10).mean(0)})

    def to_dict(self):
        return asdict(self)


class Windows:
    """Numpy-backed, DataLoader-compatible lazy engine/end index.

    Unlabelled instances never calculate RUL and return x only. A window ends at
    position end (inclusive); short histories repeat the first observation left.
    all/stride policy always includes the last available cycle. first end=0.
    """
    def __init__(self, df, preprocessing, length=30, stride=1, cap=125., scale=125.,
                 labelled=True, terminal=False, terminal_rul=None):
        if length < 1 or stride < 1:
            raise ValueError("window length and stride must be >=1")
        self.length, self.cap, self.scale, self.labelled = length, cap, scale, labelled
        self.arrays, self.raw, self.cycles, self.index = {}, {}, {}, []
        for uid, g in df.groupby("unit_id", sort=True):
            uid = int(uid)
            self.arrays[uid] = preprocessing.transform(g)
            self.cycles[uid] = g.cycle.to_numpy()
            if labelled:
                offset = 0. if terminal_rul is None else terminal_rul[uid]
                self.raw[uid] = g.cycle.max() - g.cycle.to_numpy() + offset
            ends = [len(g) - 1] if terminal else sorted(set(range(0, len(g), stride)) | {len(g) - 1})
            self.index.extend((uid, e) for e in ends)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        uid, end = self.index[i]
        a = self.arrays[uid][max(0, end - self.length + 1):end + 1]
        if len(a) < self.length:
            a = np.concatenate([np.repeat(a[:1], self.length - len(a), axis=0), a])
        x = a.copy()
        if not self.labelled:
            return x
        raw = float(self.raw[uid][end])
        target, normalized = transform_target(raw, self.cap, self.scale)
        return {"x": x, "y": np.float32(normalized), "raw": raw,
                "target": float(target), "unit_id": uid, "cycle": int(self.cycles[uid][end])}

    def sampling_weights(self):
        ids = np.array([u for u, _ in self.index])
        counts = dict(zip(*np.unique(ids, return_counts=True)))
        return np.array([1. / counts[u] for u in ids], dtype=float)
