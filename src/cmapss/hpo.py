"""Sequential Rustuna HPO; source-only, fixed split/targets/validation windows."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import pandas as pd
import yaml

FIELDS = ("learning_rate", "weight_decay", "batch_size", "hidden_size", "num_layers", "dropout")


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def validate(spec, base):
    required = {"base_config", "n_trials", "trial_epochs", "trial_patience", "search_seed", "space"}
    if set(spec) != required or set(spec["space"]) != set(FIELDS):
        raise ValueError("HPO config must specify exactly the documented keys and six search dimensions")
    for k in ("n_trials", "trial_epochs", "trial_patience"):
        if type(spec[k]) is not int or spec[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    if type(spec["search_seed"]) is not int or spec["search_seed"] < 0:
        raise ValueError("search_seed must be a nonnegative integer")
    if base["source"] != "FD001" or base["steps_per_epoch"] is not None:
        raise ValueError("This HPO command is exclusively supervised FD001")
    for name in ("learning_rate", "weight_decay"):
        bounds = spec["space"][name]
        if len(bounds) != 2 or not all(math.isfinite(x) and x > 0 for x in bounds) or bounds[0] >= bounds[1]:
            raise ValueError(f"{name}: expected positive increasing log bounds")
        if not bounds[0] <= base[name] <= bounds[1]:
            raise ValueError(f"Baseline {name} is outside search bounds")
    for name in ("batch_size", "hidden_size", "num_layers", "dropout"):
        values = spec["space"][name]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{name}: expected distinct nonempty choices")
        if name == "dropout":
            if not all(math.isfinite(x) and 0 <= x < 1 for x in values):
                raise ValueError("dropout choices must lie in [0,1)")
        elif not all(type(x) is int and x > 0 for x in values):
            raise ValueError(f"{name}: expected positive integer choices")
        baseline = base[name] if name == "batch_size" else base["model"][name]
        if baseline not in values:
            raise ValueError(f"Baseline {name} must belong to search choices")


def trial_config(base, spec, trial):
    c = copy.deepcopy(base)
    c["epochs"], c["patience"] = spec["trial_epochs"], spec["trial_patience"]
    # Trial zero evaluates the existing reference at the same search budget.
    if trial.number != 0:
        for name in FIELDS:
            values = spec["space"][name]
            value = (trial.suggest_float(name, *values, log=True) if name in ("learning_rate", "weight_decay")
                     else trial.suggest_categorical(name, values))
            if name in ("hidden_size", "num_layers", "dropout"):
                c["model"][name] = value
            else:
                c[name] = value
    return c


def run(spec, base, output, resume=False, train_fn=None, load_fn=None, backend=None):
    validate(spec, base)
    if sys.version_info < (3, 11):
        raise RuntimeError("Rustuna HPO requires Python >=3.11")
    if backend is None:
        try:
            import rustuna as backend
        except ImportError as e:
            raise RuntimeError("Install optional HPO dependency: python -m pip install -r requirements-hpo.txt") from e
    if train_fn is None or load_fn is None:
        from .engine import fit
        from .runtime import load_checkpoint
        train_fn, load_fn = fit, load_checkpoint
    source_path = Path(base["data_dir"]) / "train_FD001.txt"
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    out = Path(output)
    if resume and not out.is_dir():
        raise FileNotFoundError("HPO resume requires its original output directory")
    version = importlib.metadata.version("rustuna") if backend.__name__ == "rustuna" else "test-double"
    manifest = {"schema": 1, "spec": spec, "base_config": base,
                "source_sha256": digest, "rustuna_version": version,
                "python": sys.version, "output": str(out.resolve()),
                "selection": "minimum_source_engine_balanced_validation_rmse_cycles",
                "sampler": "TPESampler, re-seeded search_seed+trial_number for boundary resume",
                "test_used": False, "target_used": False}
    if resume:
        if json.loads((out / "search_manifest.json").read_text()) != manifest:
            raise ValueError("Resume requires identical config, source, directory, Python and Rustuna version")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "search_manifest.json", manifest)
    storage = backend.storages.SQLite3Storage(str(out / "study.db"), create_database=not resume)
    if not resume:
        backend.create_study(study_name="source_fd001", storage=storage, direction="minimize",
            sampler=backend.samplers.TPESampler(seed=spec["search_seed"]))
    started = time.perf_counter()

    def export(study):
        rows = []
        for t in study.trials:
            p = out / f"trial_{t.number:05d}" / "result.json"
            row = {"trial": t.number, "state": str(t.state)}
            if p.exists():
                row.update(json.loads(p.read_text()))
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / "trials.csv", index=False)
        complete = [r for r in rows if "value_cycles" in r and "COMPLETE" in r["state"]]
        if complete:
            best = min(complete, key=lambda r: (r["value_cycles"], r["trial"]))
            selected = yaml.safe_load((out / f"trial_{best['trial']:05d}" / "resolved_config.yaml").read_text())
            # Restore the final training budget; never continue a short trial implicitly.
            selected["epochs"], selected["patience"] = base["epochs"], base["patience"]
            (out / "best_config.yaml").write_text(yaml.safe_dump(selected, sort_keys=False))
            write_json(out / "best_trial.json", best)
        write_json(out / "search_summary.json", {"attempted_trials": len(rows), "completed_trials": len(complete),
            "requested_trials": spec["n_trials"], "segment_seconds": time.perf_counter()-started,
            "final_training_executed": False, "test_used": False, "target_used": False})

    def objective(trial):
        c = trial_config(base, spec, trial)
        directory = out / f"trial_{trial.number:05d}"
        checkpoint = Path(train_fn(c, directory))
        state = load_fn(checkpoint)
        if state["mode"] != "supervised" or state["config"] != c:
            raise ValueError("Trial checkpoint does not match the supervised configuration")
        if state["data_sha256"].get(str(source_path)) != digest:
            raise ValueError("Source data changed during the study")
        protocol = {"split": state["split"], "preprocessing": state["preprocessing"]}
        expected = out / "source_protocol.json"
        if expected.exists() and json.loads(expected.read_text()) != protocol:
            raise ValueError("Source split/preprocessing changed across trials")
        if not expected.exists():
            write_json(expected, protocol)
        value = float(state["best_metric"])
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite HPO objective")
        params = {k: c[k] if k in c else c["model"][k] for k in FIELDS}
        write_json(directory / "result.json", {"value_cycles": value, "checkpoint": str(checkpoint),
            "best_epoch_zero_based": state["epoch"], "is_reference": trial.number == 0, **params})
        return value

    while True:
        previous = backend.load_study(study_name="source_fd001", storage=storage)
        study = backend.load_study(study_name="source_fd001", storage=storage,
            sampler=backend.samplers.TPESampler(seed=spec["search_seed"] + len(previous.trials)))
        if any("RUNNING" in str(t.state) or "WAITING" in str(t.state) for t in study.trials):
            raise RuntimeError("Unfinished trial: do not resume this study automatically; inspect its state first")
        if len(study.trials) >= spec["n_trials"]:
            export(study)
            break
        try:
            study.optimize(objective, n_trials=1)
        finally:
            export(study)
    return out / "best_config.yaml"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rustuna source-only LSTM search; no test evaluation")
    parser.add_argument("--config", default="configs/hpo_fd001.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.config)
    spec = yaml.safe_load(path.read_text())
    from .engine import resolve_config
    base = resolve_config(path.parent / spec["base_config"])
    print(run(spec, base, args.output, args.resume))


if __name__ == "__main__":
    main()
