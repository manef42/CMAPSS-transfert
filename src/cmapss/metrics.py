"""Cycle-space metrics, no prediction clipping; overflow is reported explicitly."""
import numpy as np


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError("Metric arrays must have equal nonempty shapes")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs must be finite")
    e = prediction - truth
    with np.errstate(over="ignore", invalid="ignore"):
        score = np.expm1(np.where(e < 0, -e / 13., e / 10.))
        total = score.sum(dtype=np.float64)
        rmse = np.sqrt(np.mean(e * e))
    def finite(value):
        return float(value) if np.isfinite(value) else None
    return {"rmse_cycles": finite(rmse), "mae_cycles": finite(np.abs(e).mean()),
            "bias_cycles": finite(e.mean()), "nasa_sum": finite(total),
            "nasa_mean": finite(total / e.size), "nasa_overflow": bool(not np.isfinite(total)),
            "rmse_overflow": bool(not np.isfinite(rmse)), "count": int(e.size),
            "prediction_policy": "raw_linear_no_clipping", "error": "prediction_minus_truth"}


def engine_balanced_rmse(frame):
    mse = frame.assign(squared=(frame.prediction_cycles-frame.evaluation_target)**2)
    return float(np.sqrt(mse.groupby("unit_id").squared.mean().mean()))
