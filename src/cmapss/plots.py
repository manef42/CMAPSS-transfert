"""Deterministic, cycle-space PNG diagnostics; no cherry-picked engines."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .25, "savefig.dpi": 200})


def _save(fig, out, name):
    path = Path(out) / "figures"
    path.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(path / (name + ".png"), bbox_inches="tight")
    plt.close(fig)


def training_figures(out):
    h = pd.read_csv(Path(out) / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(h.epoch + 1, h.rul_loss, label="Source RUL loss")
    axes[0].set(xlabel="Epoch", ylabel="MSE (normalized RUL)", title="Training objective")
    axes[1].plot(h.epoch + 1, h.validation_engine_rmse_cycles, label="Engine-balanced validation")
    axes[1].set(xlabel="Epoch", ylabel="RMSE (cycles)", title="Source validation, fixed windows")
    for ax in axes:
        ax.legend()
    _save(fig, out, "learning")
    if "domain_loss" in h:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for col in ["rul_loss", "domain_loss"]:
            axes[0].plot(h.epoch + 1, h[col], label=col)
        axes[0].set(xlabel="Epoch", ylabel="Mean loss", title="DANN objectives")
        axes[1].plot(h.epoch + 1, h.domain_balanced_accuracy, label="Balanced domain accuracy")
        axes[1].axhline(.5, color="k", linestyle="--", label="Chance")
        axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1), title="Domain diagnostic only")
        axes[2].plot(h.epoch + 1, h["lambda"], label="Mean GRL coefficient")
        axes[2].set(xlabel="Epoch", ylabel="Lambda", title="Adversarial schedule")
        for ax in axes:
            ax.legend()
        _save(fig, out, "dann")


def evaluation_figures(out):
    p = pd.read_csv(Path(out) / "predictions.csv")
    truth, pred, err = p.evaluation_target, p.prediction_cycles, p.signed_error
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(truth, pred, s=18, alpha=.7, label="Terminal engines")
    lo, hi = min(truth.min(), pred.min()), max(truth.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", label="Perfect prediction")
    ax.set(xlabel="Evaluation RUL (cycles)", ylabel="Predicted RUL (cycles)", title="Terminal RUL predictions")
    ax.legend()
    _save(fig, out, "predicted_vs_true")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(err, bins="auto", alpha=.8, label="Terminal errors")
    ax.axvline(0, color="k", linestyle="--", label="Zero error")
    ax.set(xlabel="Prediction minus evaluation RUL (cycles)", ylabel="Engines", title="Signed terminal errors")
    ax.legend()
    _save(fig, out, "signed_errors")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(truth, err, s=18, alpha=.7, label="Terminal engines")
    ax.axhline(0, color="k", linestyle="--", label="Zero error")
    ax.set(xlabel="Evaluation RUL (cycles)", ylabel="Signed error (cycles)", title="Error versus RUL")
    ax.legend()
    _save(fig, out, "error_vs_rul")
    path = Path(out) / "trajectories.csv"
    if path.exists():
        trajectories = pd.read_csv(path)
        units = sorted(trajectories.unit_id.unique())
        fig, axes = plt.subplots(len(units), 1, figsize=(9, 3 * len(units)), squeeze=False)
        for ax, uid in zip(axes[:, 0], units):
            g = trajectories[trajectories.unit_id == uid]
            ax.plot(g.last_cycle_observed, g.rul_true_raw, ":", label="Raw historical RUL")
            ax.plot(g.last_cycle_observed, g.evaluation_target, label="Evaluation target")
            ax.plot(g.last_cycle_observed, g.prediction_cycles, label="Prediction")
            ax.set(xlabel="Observed cycle", ylabel="RUL (cycles)", title=f"Engine {uid} (smallest IDs rule)")
            ax.legend()
        _save(fig, out, "trajectories")
