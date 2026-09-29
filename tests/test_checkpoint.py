import os

import torch

from src.checkpoint import CheckpointManager, load_checkpoint, save_checkpoint


def make_model():
    return torch.nn.Linear(3, 2)


def test_save_and_load_restores_weights_and_optimizer(tmp_path):
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.123)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, optimizer, step=42, metrics={"loss": 1.0})

    restored = make_model()
    restored_opt = torch.optim.Adam(restored.parameters(), lr=0.001)
    payload = load_checkpoint(path, restored, restored_opt)

    assert payload["step"] == 42
    assert payload["metrics"]["loss"] == 1.0
    assert restored_opt.param_groups[0]["lr"] == 0.123
    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.equal(a, b)
    assert not os.path.exists(path + ".tmp")


def test_manager_keeps_only_last_n(tmp_path):
    model = make_model()
    manager = CheckpointManager(str(tmp_path), keep_last=2)
    for step in (100, 200, 300):
        manager.save(model, None, step)

    names = sorted(os.listdir(tmp_path))
    assert names == ["checkpoint_000000200.pt", "checkpoint_000000300.pt"]
    assert manager.latest_path().endswith("checkpoint_000000300.pt")


def test_manager_tracks_best_score(tmp_path):
    model = make_model()
    manager = CheckpointManager(str(tmp_path), keep_last=1)
    manager.save(model, None, 100, score=10.0)
    with torch.no_grad():
        model.weight.fill_(5.0)
    manager.save(model, None, 200, score=5.0)  # worse: must not replace best

    best = make_model()
    payload = manager.load_best(best)
    assert payload["step"] == 100
    assert not torch.all(best.weight == 5.0)

    # A new manager on the same directory remembers the best score.
    assert CheckpointManager(str(tmp_path)).best_score == 10.0


def test_load_latest_when_empty(tmp_path):
    assert CheckpointManager(str(tmp_path)).load_latest(make_model()) is None
