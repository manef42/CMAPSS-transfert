import numpy as np
import pandas as pd
import pytest
from cmapss.data import (COLUMNS, FEATURES, read_frames, train_labels, terminal_labels,
                         transform_target, split_engines, Preprocessor, Windows)
from cmapss.metrics import metrics, engine_balanced_rmse
from cmapss.synthetic import generate


def frame(ids=(1, 2), lengths=(4, 6)):
    return pd.DataFrame([[uid, t, *([uid * 10 + t] * 24)]
        for uid, n in zip(ids, lengths) for t in range(1, n + 1)], columns=COLUMNS)


def test_spaces_and_train_rul(tmp_path):
    df = frame()
    p = tmp_path / "train.txt"
    np.savetxt(p, df, delimiter="    ")
    loaded = read_frames(p)
    np.testing.assert_array_equal(train_labels(loaded), [3, 2, 1, 0, 5, 4, 3, 2, 1, 0])
    assert loaded.shape == (10, 26)


@pytest.mark.parametrize("kind", ["columns", "nan", "fraction", "duplicate", "order", "gap", "start"])
def test_reject_corruption(tmp_path, kind):
    a = frame().to_numpy(dtype=float)
    if kind == "columns":
        a = a[:, :-1]
    elif kind == "nan":
        a[0, 5] = np.nan
    elif kind == "fraction":
        a[0, 0] = 1.5
    elif kind == "duplicate":
        a[1] = a[0]
    elif kind == "order":
        a[[0, 1]] = a[[1, 0]]
    elif kind == "gap":
        a = np.delete(a, 1, axis=0)
    else:
        a[:4, 1] += 1
    p = tmp_path / "bad.txt"
    np.savetxt(p, a)
    with pytest.raises(ValueError):
        read_frames(p)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_frames(tmp_path / "missing.txt")


def test_cap_is_not_scaling():
    raw = np.array([0., 70., 180.])
    transformed, normalized = transform_target(raw, 125, 250)
    np.testing.assert_array_equal(transformed, [0, 70, 125])
    np.testing.assert_array_equal(normalized, transformed / 250)
    np.testing.assert_array_equal(transform_target(raw, None, 125)[0], raw)


def test_terminal_association(tmp_path):
    df = frame(lengths=(3, 6))
    p = tmp_path / "rul.txt"
    p.write_text("17\n29\n")
    mapping = terminal_labels(p, df)
    assert mapping == {1: 17., 2: 29.}
    ds = Windows(df, Preprocessor.fit(df), length=5, terminal=True, terminal_rul=mapping)
    assert [ds[i]["raw"] for i in range(2)] == [17., 29.]
    p.write_text("1\n")
    with pytest.raises(ValueError):
        terminal_labels(p, df)
    p.write_text("17\n29\n")
    with pytest.raises(ValueError):
        terminal_labels(p, frame(ids=(2, 7)))


def test_engine_split_and_train_only_preprocessing():
    df = frame(ids=(1, 2, 3, 4), lengths=(4, 4, 4, 4))
    split = split_engines(df, .25, 3)
    assert split == split_engines(df, .25, 3)
    assert not set(split["train"]) & set(split["validation"])
    train = df[df.unit_id.isin(split["train"])]
    prep = Preprocessor.fit(train)
    np.testing.assert_allclose(prep.mean, train[FEATURES].mean())
    assert not np.allclose(prep.mean, df[FEATURES].mean())
    before = prep.to_dict()
    prep.transform(df.assign(s1=1e6))
    assert prep.to_dict() == before


def test_constant_variables_retained():
    df = frame()
    df["setting_1"] = 3.
    df["s21"] = np.linspace(0, 1e-9, len(df))
    prep = Preprocessor.fit(df)
    assert len(prep.features) == 24
    assert prep.scale[0] == 1. and prep.low_variance[0]
    assert prep.scale[-1] == 1.
    shifted = df.copy()
    shifted["setting_1"] = 5.
    np.testing.assert_allclose(prep.transform(shifted)[:, 0], 2.)
    assert np.isfinite(prep.transform(df)).all()


def test_windows_alignment_padding_and_no_engine_mixing():
    df = frame()
    prep = Preprocessor.fit(df)
    ds = Windows(df, prep, length=3, stride=2, cap=None, scale=2)
    assert ds.index == [(1, 0), (1, 2), (1, 3), (2, 0), (2, 2), (2, 4), (2, 5)]
    assert ds[1]["raw"] == 1. and ds[1]["y"] == .5
    np.testing.assert_allclose(ds[1]["x"], prep.transform(df.iloc[:3]))
    np.testing.assert_allclose(ds[0]["x"], np.repeat(prep.transform(df.iloc[:1]), 3, axis=0))
    np.testing.assert_allclose(ds[3]["x"], np.repeat(prep.transform(df.iloc[4:5]), 3, axis=0))
    terminal = Windows(df, prep, length=8, terminal=True)
    assert len(terminal) == 2 and terminal[0]["x"].shape == (8, 24)
    np.testing.assert_allclose(terminal[0]["x"][:5], np.repeat(prep.transform(df.iloc[:1]), 5, axis=0))
    assert sum(ds.sampling_weights()[:3]) == pytest.approx(1.)
    assert sum(ds.sampling_weights()[3:]) == pytest.approx(1.)


def test_unlabelled_dataset_has_no_labels():
    df = frame()
    ds = Windows(df, Preprocessor.fit(df), labelled=False)
    assert ds.raw == {} and isinstance(ds[0], np.ndarray)


def test_nasa_sign_and_overflow():
    m = metrics([10., 10.], [-3., 20.])
    assert m["nasa_sum"] == pytest.approx(2 * np.expm1(1.))
    assert m["bias_cycles"] == -1.5
    assert m["mae_cycles"] == 11.5
    assert m["prediction_policy"] == "raw_linear_no_clipping"
    m = metrics([0.], [1e10])
    assert m["nasa_sum"] is None and m["nasa_overflow"]
    with pytest.raises(ValueError):
        metrics([np.nan], [1])


def test_engine_metric_formula():
    df = pd.DataFrame({"unit_id": [1, 1, 2], "prediction_cycles": [2, 2, 4], "evaluation_target": [0, 0, 0]})
    assert engine_balanced_rmse(df) == pytest.approx(np.sqrt((4 + 16) / 2))


def test_synthetic_inspect_cli(tmp_path, capsys):
    from cmapss.cli import main
    generate(tmp_path)
    main(["inspect", "--data-dir", str(tmp_path), "--partition", "test"])
    assert '"terminal_mapping_verified": true' in capsys.readouterr().out
    assert not (tmp_path / "train_FD002.txt").exists()


def test_plot_exports(tmp_path):
    from cmapss.plots import training_figures, evaluation_figures
    pd.DataFrame({"epoch": [0, 1], "rul_loss": [.2, .1],
        "validation_engine_rmse_cycles": [40, 30], "domain_loss": [.7, .6],
        "domain_balanced_accuracy": [.5, .6], "lambda": [0, .9]}).to_csv(tmp_path / "history.csv", index=False)
    pd.DataFrame({"unit_id": [1, 2], "evaluation_target": [10, 20], "prediction_cycles": [8, 23],
                  "signed_error": [-2, 3]}).to_csv(tmp_path / "predictions.csv", index=False)
    training_figures(tmp_path)
    evaluation_figures(tmp_path)
    assert len(list((tmp_path / "figures").glob("*.png"))) == 5
