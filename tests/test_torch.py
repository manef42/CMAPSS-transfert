import copy
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
torch = pytest.importorskip("torch", reason="PyTorch required for integration/gradient tests")
from cmapss.model import RULModel, GradientReversal
from cmapss.engine import resolve_config, fit, evaluate, dann_step, supervised_step
from cmapss.runtime import load_checkpoint
from cmapss.synthetic import generate


def tiny(data_dir):
    return resolve_config(overrides={"data_dir": str(data_dir), "epochs": 2, "adaptation_epochs": 2,
        "batch_size": 8, "target_batch_size": 3, "window_size": 5, "validation_stride": 2,
        "device": "cpu", "steps_per_epoch": None, "model": {
            "hidden_size": 4, "head_size": 3, "num_layers": 1, "dropout": 0.}})


def model():
    return RULModel(hidden_size=4, head_size=3, num_layers=1, dropout=0.)


def test_shapes_and_no_domain_inference():
    m = model()
    x = torch.randn(3, 5, 24)
    assert m.encode(x).shape == (3, 4)
    assert m.predict_rul(x).shape == (3,)
    assert [z.shape for z in m.forward_dann(x, 1.)] == [(3,), (3,)]
    handle = m.domain_head.register_forward_hook(lambda *args: pytest.fail("Domain head used in inference"))
    m.predict_rul(x)
    handle.remove()


@pytest.mark.parametrize("coefficient", [0., .3, 1., 2.])
def test_grl_identity_sign_amplitude(coefficient):
    x = torch.tensor([1., 3.], requires_grad=True)
    y = GradientReversal.apply(x, coefficient)
    torch.testing.assert_close(y, x)
    (y * torch.tensor([2., -4.])).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([-2., 4.]) * coefficient)


def test_domain_gradients_and_lambda_zero():
    torch.manual_seed(1)
    m = model()
    x = torch.randn(3, 5, 24)
    for coefficient in [0., .5]:
        m.zero_grad(set_to_none=True)
        _, logits = m.forward_dann(x, coefficient)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(3))
        loss.backward()
        encoder_grad = [p.grad.clone() for p in m.lstm.parameters()]
        head_grad = [p.grad.clone() for p in m.domain_head.parameters()]
        if coefficient == 0:
            assert all(torch.count_nonzero(g) == 0 for g in encoder_grad)
        else:
            m.zero_grad(set_to_none=True)
            direct = m.domain_head(m.encode(x)).squeeze(-1)
            torch.nn.functional.binary_cross_entropy_with_logits(direct, torch.ones(3)).backward()
            for actual, p in zip(encoder_grad, m.lstm.parameters()):
                torch.testing.assert_close(actual, -.5 * p.grad)
            for actual, p in zip(head_grad, m.domain_head.parameters()):
                torch.testing.assert_close(actual, p.grad)


def test_supervised_and_unequal_dann_steps():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    m = model()
    opt = torch.optim.Adam(m.parameters(), lr=.02)
    batch = {"x": torch.randn(4, 5, 24), "y": torch.ones(4) * .1}
    first = supervised_step(m, opt, batch, torch.device("cpu"), 5.)
    for _ in range(20):
        last = supervised_step(m, opt, batch, torch.device("cpu"), 5.)
    assert last < first
    stats = dann_step(m, opt, batch["x"], batch["y"], torch.randn(2, 5, 24), .4)
    assert np.isfinite(list(stats.values())).all()
    assert "target_y" not in inspect.signature(dann_step).parameters


def test_unequal_domain_loss_exact():
    torch.manual_seed(3)
    m = model().eval()
    x, tx, y = torch.randn(4, 5, 24), torch.randn(2, 5, 24), torch.ones(4)
    with torch.no_grad():
        _, sl = m.forward_dann(x, .5)
        _, tl = m.forward_dann(tx, .5)
        expected = .5 * (torch.nn.functional.binary_cross_entropy_with_logits(sl, torch.zeros(4)) +
                         torch.nn.functional.binary_cross_entropy_with_logits(tl, torch.ones(2)))
    result = dann_step(m, torch.optim.SGD(m.parameters(), lr=0.), x, y, tx, .5)
    assert result["domain_loss"] == pytest.approx(float(expected))


def test_checkpoint_identical_predictions(tmp_path):
    m = model().eval()
    x = torch.randn(2, 5, 24)
    path = tmp_path / "model.pt"
    torch.save(m.state_dict(), path)
    restored = model().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(m(x), restored(x), rtol=0, atol=0)


def test_fd001_cli_smoke(tmp_path):
    from cmapss.cli import main
    main(["smoke", "--output", str(tmp_path / "smoke")])
    out = tmp_path / "smoke"
    assert not (out / "data/train_FD002.txt").exists()
    assert len(pd.read_csv(out / "evaluation/predictions.csv")) == 5
    assert (out / "evaluation/figures/trajectories.png").is_file()
    assert load_checkpoint(out / "train/checkpoints/best.pt")["mode"] == "supervised"


def test_direct_transfer_needs_no_target_train(tmp_path):
    data = tmp_path / "data"
    generate(data)
    cfg = tiny(data)
    checkpoint = fit(cfg, tmp_path / "source")
    generate(data, "FD002", seed=8)
    (data / "train_FD002.txt").unlink()
    result = evaluate(checkpoint, data, "FD002", tmp_path / "direct", device="cpu")
    assert result["count"] == 5


def test_dann_target_evaluation_labels_cannot_affect_training(tmp_path, monkeypatch):
    import cmapss.engine as engine
    data = tmp_path / "data"
    generate(data)
    generate(data, "FD002", seed=8)
    cfg = tiny(data)
    checkpoint = fit(cfg, tmp_path / "source")
    # Tripwire: no evaluation RUL loader may be used during adaptation.
    monkeypatch.setattr(engine, "terminal_labels", lambda *a: pytest.fail("Target labels accessed"))
    cfg["steps_per_epoch"] = 2
    cfg["target_units"] = [1, 2]
    first = fit(cfg, tmp_path / "adapt_a", source_checkpoint=checkpoint)
    (data / "RUL_FD002.txt").write_text("invalid changed evaluation labels, must never be read")
    second = fit(cfg, tmp_path / "adapt_b", source_checkpoint=checkpoint)
    a, b = load_checkpoint(first), load_checkpoint(second)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert a["history"] == b["history"]
    assert a["selection"] == b["selection"] == "last_fixed_budget"
    assert a["epoch"] == b["epoch"] == cfg["adaptation_epochs"] - 1
    assert not (tmp_path / "adapt_a/checkpoints/best.pt").exists()
    assert a["split"]["target_training_unlabelled"] == [1, 2]


def test_epoch_boundary_resume_is_identical(tmp_path, monkeypatch):
    import cmapss.engine as engine
    data = tmp_path / "data"
    generate(data)
    cfg = tiny(data)
    uninterrupted = fit(cfg, tmp_path / "full")
    original = engine.save_checkpoint
    class Interrupted(Exception):
        pass
    def interrupt(path, state):
        original(path, state)
        if Path(path).name == "best.pt" and state["epoch"] == 0:
            raise Interrupted()
    monkeypatch.setattr(engine, "save_checkpoint", interrupt)
    with pytest.raises(Interrupted):
        fit(cfg, tmp_path / "resumed")
    monkeypatch.setattr(engine, "save_checkpoint", original)
    fit(cfg, tmp_path / "resumed", resume=tmp_path / "resumed/checkpoints/last.pt")
    a = load_checkpoint(tmp_path / "full/checkpoints/last.pt")
    b = load_checkpoint(tmp_path / "resumed/checkpoints/last.pt")
    assert a["history"] == b["history"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
