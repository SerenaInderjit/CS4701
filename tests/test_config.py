import numpy as np
import torch

from src.training.config import load_config, save_config
from src.training.config import seed_everything


def test_load_default_config():
    config = load_config("configs/ppo.yaml")

    assert config["updates"] == 1000
    assert config["rollout_size"] == 2048
    assert config["epsilon"] == 0.2
    assert config["checkpoint_dir"] == "checkpoints"


def test_save_and_load_round_trip(tmp_path):
    config = {"updates": 5, "device": "cpu", "nested": {"a": 1}}
    path = str(tmp_path / "config.yaml")

    save_config(config, path)

    assert load_config(path) == config


def test_seed_everything_is_reproducible():
    seed_everything(42)
    a = (torch.rand(3), np.random.rand(3))
    seed_everything(42)
    b = (torch.rand(3), np.random.rand(3))

    assert torch.equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
