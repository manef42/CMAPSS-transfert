"""Pure orchestration tests use an explicit test double, not Rustuna validation."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
import yaml
from cmapss.hpo import FIELDS, run, trial_config, validate


def setup(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train_FD001.txt").write_text("source fixture fingerprint")
    base = {"source": "FD001", "data_dir": str(data), "steps_per_epoch": None,
        "seed": 42, "validation_fraction": .2, "window_size": 30, "validation_stride": 10,
        "rul_cap": 125, "target_scale": 125, "epochs": 100, "patience": 15,
        "learning_rate": .001, "weight_decay": .0001, "batch_size": 128,
        "model": {"hidden_size": 64, "num_layers": 2, "dropout": .2}}
    spec = {"base_config": "source_fd001.yaml", "n_trials": 3, "trial_epochs": 2,
        "trial_patience": 1, "search_seed": 123, "space": {
            "learning_rate": [.0001, .003], "weight_decay": [.000001, .001],
            "batch_size": [64, 128], "hidden_size": [32, 64, 128],
            "num_layers": [1, 2], "dropout": [0., .2, .4]}}
    return spec, base


class Trial:
    def __init__(self, number):
        self.number, self.state = number, "RUNNING"
    def suggest_float(self, name, low, high, log):
        assert log
        return (low * high)**.5
    def suggest_categorical(self, name, values):
        return values[0]


class FakeBackend:
    __name__ = "test_double"
    def __init__(self):
        self.studies = {}
        self.samplers = SimpleNamespace(TPESampler=lambda seed: seed)
        self.storages = SimpleNamespace(SQLite3Storage=lambda path, create_database: path)
    def create_study(self, study_name, storage, direction, sampler):
        assert direction == "minimize"
        backend = self
        class Study:
            trials = None
            def __init__(self):
                self.trials = []
            def optimize(self, objective, n_trials):
                assert n_trials == 1
                t = Trial(len(self.trials))
                self.trials.append(t)
                try:
                    objective(t)
                except BaseException:
                    t.state = "FAIL"
                    raise
                t.state = "COMPLETE"
        backend.studies[storage] = Study()
        return backend.studies[storage]
    def load_study(self, study_name, storage, sampler=None):
        return self.studies[storage]


def fake_train(base):
    states, calls = {}, []
    digest = hashlib.sha256((Path(base["data_dir"]) / "train_FD001.txt").read_bytes()).hexdigest()
    def train(c, directory):
        calls.append(copy.deepcopy(c))
        directory.mkdir()
        (directory / "resolved_config.yaml").write_text(yaml.safe_dump(c))
        path = directory / "best.pt"
        states[path] = {"mode": "supervised", "config": c, "split": {"train": [1], "validation": [2]},
            "preprocessing": {"features": ["x"], "mean": [0]}, "epoch": 1,
            "best_metric": 10. if len(calls) == 2 else 20.,
            "data_sha256": {str(Path(base["data_dir"]) / "train_FD001.txt"): digest}}
        return path
    return train, states.__getitem__, calls


def test_trial_config_only_changes_allowed_fields(tmp_path):
    spec, base = setup(tmp_path)
    original = copy.deepcopy(base)
    validate(spec, base)
    c = trial_config(base, spec, Trial(1))
    for key in ("seed", "validation_fraction", "window_size", "validation_stride", "rul_cap", "target_scale"):
        assert c[key] == base[key]
    assert base == original
    baseline = trial_config(base, spec, Trial(0))
    assert baseline["model"] == base["model"]
    assert baseline["learning_rate"] == base["learning_rate"]


@pytest.mark.parametrize("change", ["source", "budget", "space"])
def test_invalid_protocol_is_rejected(tmp_path, change):
    spec, base = setup(tmp_path)
    if change == "source":
        base["source"] = "FD002"
    elif change == "budget":
        spec["n_trials"] = 0
    else:
        spec["space"]["rul_cap"] = [100, 125]
    with pytest.raises(ValueError):
        validate(spec, base)


def test_exports_selection_resume_and_no_holdout_inputs(tmp_path):
    spec, base = setup(tmp_path)
    backend = FakeBackend()
    train, load, calls = fake_train(base)
    out = tmp_path / "search"
    selected = run(spec, base, out, train_fn=train, load_fn=load, backend=backend)
    best = yaml.safe_load(selected.read_text())
    assert best["epochs"] == 100 and best["patience"] == 15
    assert json.loads((out / "best_trial.json").read_text())["trial"] == 1
    assert len(pd.read_csv(out / "trials.csv")) == 3
    assert len(calls) == 3 and all(c["epochs"] == 2 for c in calls)
    assert not list(Path(base["data_dir"]).glob("*FD002*"))
    run(spec, base, out, resume=True, train_fn=train, load_fn=load, backend=backend)
    assert len(calls) == 3
    with pytest.raises(FileExistsError):
        run(spec, base, out, train_fn=train, load_fn=load, backend=backend)
    changed = copy.deepcopy(base)
    changed["seed"] = 43
    with pytest.raises(ValueError, match="identical"):
        run(spec, changed, out, resume=True, train_fn=train, load_fn=load, backend=backend)


def test_source_protocol_mismatch_fails_closed(tmp_path):
    spec, base = setup(tmp_path)
    train, load, _ = fake_train(base)
    n = 0
    def corrupted(path):
        nonlocal n
        n += 1
        state = load(path)
        if n == 2:
            state["split"] = {"train": [2], "validation": [1]}
        return state
    with pytest.raises(ValueError, match="changed across trials"):
        run(spec, base, tmp_path / "bad", train_fn=train, load_fn=corrupted, backend=FakeBackend())


def test_real_rustuna_sqlite_boundary_resume(tmp_path):
    pytest.importorskip("rustuna")
    spec, base = setup(tmp_path)
    train, load, calls = fake_train(base)
    out = tmp_path / "real_backend"
    run(spec, base, out, train_fn=train, load_fn=load)
    run(spec, base, out, resume=True, train_fn=train, load_fn=load)
    assert len(calls) == 3 and (out / "study.db").is_file()


def test_real_supervised_search_without_test_or_target(tmp_path, monkeypatch):
    pytest.importorskip("rustuna")
    pytest.importorskip("torch")
    import cmapss.engine as engine
    from cmapss.synthetic import generate
    data = tmp_path / "real_data"
    generate(data)
    (data / "test_FD001.txt").unlink()
    (data / "RUL_FD001.txt").unlink()
    base = engine.resolve_config(overrides={"data_dir": str(data), "device": "cpu", "epochs": 2,
        "batch_size": 8, "window_size": 5, "model": {"hidden_size": 4, "head_size": 3,
        "num_layers": 1, "dropout": 0.}})
    spec = {"base_config": "unused", "n_trials": 2, "trial_epochs": 1, "trial_patience": 1,
        "search_seed": 123, "space": {"learning_rate": [.0001, .003], "weight_decay": [.000001, .001],
        "batch_size": [8], "hidden_size": [4], "num_layers": [1], "dropout": [0.]}}
    monkeypatch.setattr(engine, "evaluate", lambda *a, **k: pytest.fail("Official evaluation called"))
    selected = run(spec, base, tmp_path / "real_search")
    assert selected.is_file()
    assert yaml.safe_load(selected.read_text())["epochs"] == 2
