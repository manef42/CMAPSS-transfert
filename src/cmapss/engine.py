"""Separated source training, fixed-budget adaptation, and evaluation protocols."""
import copy
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
import yaml
from .data import (Preprocessor, Windows, read_frames, train_labels, terminal_labels, split_engines)
from .metrics import metrics, engine_balanced_rmse
from .model import RULModel
from .runtime import (seed_all, seed_worker, rng_state, restore_rng, write_json,
                      save_checkpoint, load_checkpoint, environment, fingerprints)

DEFAULTS = {
    "seed": 42, "data_dir": "src", "source": "FD001", "target": "FD002",
    "validation_fraction": .2, "window_size": 30, "training_stride": 1,
    "validation_stride": 10, "rul_cap": 125., "target_scale": 125.,
    "low_variance_threshold": .001, "batch_size": 128, "target_batch_size": 128,
    "sampling": "engine_balanced", "epochs": 100, "adaptation_epochs": 20,
    "steps_per_epoch": None, "patience": 15, "min_delta": 0.,
    "learning_rate": .001, "adaptation_learning_rate": .0001,
    "optimizer": "AdamW", "weight_decay": .0001, "gradient_clip": 5.,
    "device": "auto", "deterministic": True, "num_workers": 0, "threads": 1,
    "lambda_max": 1., "lambda_schedule": "logistic", "lambda_gamma": 10.,
    "target_units": None, "trajectory_count": 5,
    "model": {"input_size": 24, "hidden_size": 64, "num_layers": 2, "dropout": .2, "head_size": 32},
}


def resolve_config(path=None, overrides=None):
    c = copy.deepcopy(DEFAULTS)
    supplied = yaml.safe_load(Path(path).read_text()) or {} if path else {}
    supplied.update(overrides or {})
    unknown = set(supplied) - set(c)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    model = c["model"] | supplied.pop("model", {})
    c.update(supplied)
    c["model"] = model
    positive = ["window_size", "training_stride", "validation_stride", "batch_size", "target_batch_size",
                "epochs", "adaptation_epochs", "patience", "target_scale", "gradient_clip", "threads",
                "learning_rate", "adaptation_learning_rate", "low_variance_threshold"]
    if any(c[k] <= 0 for k in positive):
        raise ValueError(f"Positive configuration required: {positive}")
    if c["rul_cap"] is not None and c["rul_cap"] <= 0:
        raise ValueError("rul_cap must be positive or null")
    if c["model"]["input_size"] != 24:
        raise ValueError("This protocol retains all 24 variables")
    if c["optimizer"] not in ("Adam", "AdamW") or c["sampling"] not in ("engine_balanced", "windows"):
        raise ValueError("Unknown optimizer/sampling policy")
    if c["lambda_schedule"] not in ("logistic", "constant") or c["lambda_max"] < 0:
        raise ValueError("Invalid GRL schedule")
    if c["steps_per_epoch"] is not None and c["steps_per_epoch"] < 1:
        raise ValueError("steps_per_epoch must be >=1")
    return c


def device_for(c):
    name = ("cuda" if torch.cuda.is_available() else "cpu") if c["device"] == "auto" else c["device"]
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def loader(ds, c, generator, training=False, target=False):
    sampler = None
    if training:
        sampler = (WeightedRandomSampler(ds.sampling_weights(), len(ds), replacement=True, generator=generator)
                   if c["sampling"] == "engine_balanced" else RandomSampler(ds, generator=generator))
    return DataLoader(ds, batch_size=c["target_batch_size"] if target else c["batch_size"],
                      sampler=sampler, num_workers=c["num_workers"], generator=generator,
                      worker_init_fn=seed_worker, pin_memory=device_for(c).type == "cuda", drop_last=False)


def make_windows(df, prep, c, **kwargs):
    return Windows(df, prep, length=c["window_size"], cap=c["rul_cap"], scale=c["target_scale"], **kwargs)


def supervised_step(model, optimizer, batch, device, clip):
    optimizer.zero_grad(set_to_none=True)
    pred = model.predict_rul(batch["x"].to(device, non_blocking=True))
    loss = nn.functional.mse_loss(pred, batch["y"].to(device, non_blocking=True))
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite supervised loss")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def dann_step(model, optimizer, source_x, source_y, target_x, coefficient, clip=5.):
    """No target-label argument. GRL alone scales encoder adversarial gradients."""
    optimizer.zero_grad(set_to_none=True)
    rul, source_logits = model.forward_dann(source_x, coefficient)
    # Target RUL head is not executed, nor are target labels constructed.
    target_logits = model.domain_head(model_encode_reverse(model, target_x, coefficient)).squeeze(-1)
    rul_loss = nn.functional.mse_loss(rul, source_y)
    domain = .5 * (nn.functional.binary_cross_entropy_with_logits(source_logits, torch.zeros_like(source_logits))
                  + nn.functional.binary_cross_entropy_with_logits(target_logits, torch.ones_like(target_logits)))
    total = rul_loss + domain
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite DANN loss")
    total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    accuracy = .5 * ((source_logits < 0).float().mean() + (target_logits >= 0).float().mean())
    return {"rul_loss": float(rul_loss.detach()), "domain_loss": float(domain.detach()),
            "domain_balanced_accuracy": float(accuracy.detach()), "lambda": float(coefficient)}


def model_encode_reverse(model, x, coefficient):
    from .model import GradientReversal
    return GradientReversal.apply(model.encode(x), coefficient)


@torch.no_grad()
def predict(model, data_loader, device, scale, dataset):
    model.eval()
    rows = []
    for batch in data_loader:
        pred = model.predict_rul(batch["x"].to(device, non_blocking=True)).cpu().numpy() * scale
        for j, value in enumerate(pred):
            rows.append({"dataset": dataset, "unit_id": int(batch["unit_id"][j]),
                "last_cycle_observed": int(batch["cycle"][j]), "rul_true_raw": float(batch["raw"][j]),
                "evaluation_target": float(batch["target"][j]), "prediction_cycles": float(value)})
    frame = pd.DataFrame(rows)
    frame["signed_error"] = frame.prediction_cycles - frame.evaluation_target
    frame["absolute_error"] = frame.signed_error.abs()
    return frame


def coefficient(c, step, total):
    progress = step / max(1, total - 1)
    if c["lambda_schedule"] == "constant":
        return float(c["lambda_max"])
    return float(c["lambda_max"] * (2. / (1. + math.exp(-c["lambda_gamma"] * progress)) - 1.))


def _next(iterator, data_loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(data_loader)
        return next(iterator), iterator


def fit(config, output, source_checkpoint=None, resume=None):
    """Resume only at completed epoch boundaries, same config/hardware/data.

    DANN: budget fixed before training, longer loader determines steps unless
    explicitly configured. Shorter loader is restarted, never silently zipped.
    """
    c, out = copy.deepcopy(config), Path(output)
    adapting = source_checkpoint is not None
    mode = "dann" if adapting else "supervised"
    prior = load_checkpoint(resume) if resume else None
    if prior and (prior["config"] != c or prior["mode"] != mode):
        raise ValueError("Resume requires identical resolved configuration and mode")
    if prior:
        if not out.is_dir() or prior["run_dir"] != str(out.resolve()):
            raise ValueError("Resume must use its original run directory")
        if Path(resume).resolve() != (out / "checkpoints/last.pt").resolve():
            raise ValueError("Resume only from last.pt; branching history is not implicit")
        if pd.read_csv(out / "history.csv").epoch.iloc[-1] != prior["epoch"]:
            raise ValueError("History/checkpoint epoch mismatch; recover interrupted artifact write explicitly")
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "checkpoints").mkdir()
    seed_all(c["seed"], c["deterministic"], c["threads"])
    device = device_for(c)
    source_path = Path(c["data_dir"]) / f"train_{c['source']}.txt"
    source = read_frames(source_path)
    initialization = load_checkpoint(source_checkpoint) if adapting else None
    base = prior or initialization
    if initialization:
        for key in ["source", "window_size", "rul_cap", "target_scale", "model", "validation_fraction"]:
            if c[key] != initialization["config"][key]:
                raise ValueError(f"DANN must preserve source {key}")
    split = base["split"] if base else split_engines(source, c["validation_fraction"], c["seed"])
    ids = set(map(int, source.unit_id.unique()))
    if set(split["train"]) & set(split["validation"]) or set(split["train"]) | set(split["validation"]) != ids:
        raise ValueError("Saved split does not match source engines")
    train_df = source[source.unit_id.isin(split["train"])]
    val_df = source[source.unit_id.isin(split["validation"])]
    prep = Preprocessor(**base["preprocessing"]) if base else Preprocessor.fit(train_df, c["low_variance_threshold"])
    source_ds = make_windows(train_df, prep, c, stride=c["training_stride"])
    val_ds = make_windows(val_df, prep, c, stride=c["validation_stride"])
    generator = torch.Generator().manual_seed(c["seed"])
    source_loader = loader(source_ds, c, generator, training=True)
    val_loader = loader(val_ds, c, generator)
    paths = [source_path]
    target_loader = None
    split = copy.deepcopy(split)
    if adapting:
        target_path = Path(c["data_dir"]) / f"train_{c['target']}.txt"
        target = read_frames(target_path)
        target_ids = sorted(map(int, target.unit_id.unique())) if c["target_units"] is None else sorted(c["target_units"])
        if not target_ids or not set(target_ids).issubset(set(target.unit_id.unique())):
            raise ValueError("target_units must be nonempty and present in target training file")
        target = target[target.unit_id.isin(target_ids)]
        split["target_training_unlabelled"] = target_ids
        # No terminal RUL file or target RUL construction anywhere in this branch.
        target_ds = make_windows(target, prep, c, stride=c["training_stride"], labelled=False)
        target_loader = loader(target_ds, c, generator, training=True, target=True)
        prep.diagnostic(target).to_csv(out / "dispersion_target.csv", index=False)
        paths.append(target_path)
    hashes = fingerprints(paths)
    if prior and hashes != prior["data_sha256"]:
        raise ValueError("Training input fingerprint changed since checkpoint")
    if initialization and hashes[str(source_path)] != initialization["data_sha256"].get(str(source_path)):
        raise ValueError("Source training data must match source checkpoint fingerprint/path")
    model = RULModel(**c["model"]).to(device)
    if base:
        model.load_state_dict(base["model"])
    if adapting and not prior:
        model.reset_domain_head()
    lr = c["adaptation_learning_rate"] if adapting else c["learning_rate"]
    optimizer = getattr(torch.optim, c["optimizer"])(model.parameters(), lr=lr, weight_decay=c["weight_decay"])
    start, global_step, best, stale = 0, 0, float("inf"), 0
    history = []
    if prior:
        model.load_state_dict(prior["model"])
        optimizer.load_state_dict(prior["optimizer"])
        start, global_step = prior["epoch"] + 1, prior["global_step"]
        best, stale = prior["best_metric"], prior["early_stopping"]["stale"]
        history = prior["history"]
        generator.set_state(prior["loader_rng"])
        restore_rng(prior["rng"])
    budget = c["adaptation_epochs"] if adapting else c["epochs"]
    steps = c["steps_per_epoch"] or (max(len(source_loader), len(target_loader)) if adapting else len(source_loader))
    if not adapting and c["steps_per_epoch"] is not None:
        raise ValueError("steps_per_epoch override applies only to DANN")
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    write_json(out / "split_manifest.json", split)
    write_json(out / "preprocessing.json", prep.to_dict())
    write_json(out / "environment.json", environment(device))
    write_json(out / "data_sha256.json", hashes)
    prep.diagnostic(train_df).to_csv(out / "dispersion_source_train.csv", index=False)
    prep.diagnostic(val_df).to_csv(out / "dispersion_source_validation.csv", index=False)
    started = time.perf_counter()
    elapsed_prior = prior.get("elapsed_seconds", 0.) if prior else 0.
    for epoch in range(start, budget):
        if not adapting and stale >= c["patience"]:
            break
        model.train()
        records = []
        if adapting:
            source_it, target_it = iter(source_loader), iter(target_loader)
            for _ in range(steps):
                sb, source_it = _next(source_it, source_loader)
                tx, target_it = _next(target_it, target_loader)
                records.append(dann_step(model, optimizer, sb["x"].to(device, non_blocking=True),
                    sb["y"].to(device, non_blocking=True), tx.to(device, non_blocking=True),
                    coefficient(c, global_step, budget * steps), c["gradient_clip"]))
                global_step += 1
        else:
            for batch in source_loader:
                records.append({"rul_loss": supervised_step(model, optimizer, batch, device, c["gradient_clip"])})
                global_step += 1
        validation = predict(model, val_loader, device, c["target_scale"], c["source"])
        score = engine_balanced_rmse(validation)
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite source validation RMSE")
        improved = score < best - c["min_delta"]
        best, stale = (score, 0) if improved else (best, stale + 1)
        row = {"epoch": epoch, "global_step": global_step, "validation_engine_rmse_cycles": score}
        row.update(pd.DataFrame(records).mean().to_dict())
        history.append(row)
        state = {"schema_version": 1, "mode": mode, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": None, "amp_scaler": None,
            "rng": rng_state(), "loader_rng": generator.get_state(), "epoch": epoch,
            "global_step": global_step, "config": c, "preprocessing": prep.to_dict(), "split": split,
            "best_metric": best, "early_stopping": {"stale": stale, "patience": c["patience"]},
            "history": history, "run_dir": str(out.resolve()), "data_sha256": hashes,
            "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
            "selection": "last_fixed_budget" if adapting else "source_engine_balanced_rmse"}
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        save_checkpoint(out / "checkpoints/last.pt", state)
        if improved and not adapting:
            save_checkpoint(out / "checkpoints/best.pt", state)
            validation.to_csv(out / "validation_predictions.csv", index=False)
    write_json(out / "run_summary.json", {"mode": mode, "completed_epochs": len(history),
        "global_step": global_step, "steps_per_epoch": steps,
        "selection": "last_fixed_budget" if adapting else "source_engine_balanced_rmse",
        "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
        "target_labels_used": False, "source_free": False})
    from .plots import training_figures
    training_figures(out)
    return out / ("checkpoints/last.pt" if adapting else "checkpoints/best.pt")


def evaluate(checkpoint, data_dir, dataset, output, trajectories=True, device="auto"):
    state = load_checkpoint(checkpoint)
    c = copy.deepcopy(state["config"])
    c["device"] = device
    seed_all(c["seed"], c["deterministic"], c["threads"])
    dev = device_for(c)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    prep = Preprocessor(**state["preprocessing"])
    # Direct transfer only reads target TEST and terminal RUL; no fit or optimizer.
    test_path, rul_path = Path(data_dir) / f"test_{dataset}.txt", Path(data_dir) / f"RUL_{dataset}.txt"
    test = read_frames(test_path)
    terminal = terminal_labels(rul_path, test)
    model = RULModel(**c["model"]).to(dev)
    model.load_state_dict(state["model"])
    generator = torch.Generator().manual_seed(c["seed"])
    started = time.perf_counter()
    def prediction(ds):
        return predict(model, loader(ds, c, generator), dev, c["target_scale"], dataset)
    frame = prediction(make_windows(test, prep, c, terminal=True, terminal_rul=terminal))
    frame.to_csv(out / "predictions.csv", index=False)
    result = metrics(frame.evaluation_target, frame.prediction_cycles)
    result.update({"dataset": dataset, "evaluation": "one_terminal_prediction_per_engine",
                   "rul_cap": c["rul_cap"], "target_scale": c["target_scale"],
                   "checkpoint": str(checkpoint), "checkpoint_epoch": state["epoch"]})
    write_json(out / "metrics.json", result)
    if trajectories:
        # Historical target labels exist exclusively in evaluation, never adaptation.
        chosen = sorted(terminal)[:c["trajectory_count"]]
        detail = test[test.unit_id.isin(chosen)]
        if chosen:
            prediction(make_windows(detail, prep, c, stride=1, terminal_rul=terminal)).to_csv(
                out / "trajectories.csv", index=False)
    prep.diagnostic(test).to_csv(out / "dispersion_test.csv", index=False)
    write_json(out / "terminal_rul_mapping.json", terminal)
    write_json(out / "preprocessing.json", prep.to_dict())
    write_json(out / "split_manifest.json", state["split"])
    write_json(out / "environment.json", environment(dev))
    write_json(out / "data_sha256.json", fingerprints([test_path, rul_path, checkpoint]))
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    write_json(out / "run_summary.json", {"elapsed_seconds": time.perf_counter() - started,
        "weights_updated": False, "preprocessing_fitted": False})
    from .plots import evaluation_figures
    evaluation_figures(out)
    return result
