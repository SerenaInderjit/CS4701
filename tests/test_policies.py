import numpy as np
import torch

from src.algorithms.cnn import ConvolutionalNeuralNetwork
from src.policies.baselines import ConstantPolicy, RandomPolicy
from src.policies.ppo_policy import PPOPolicy
from src.training.checkpoint import save_checkpoint


def test_random_policy_is_seedable_and_in_range():
    a, b = RandomPolicy(seed=1), RandomPolicy(seed=1)
    actions = [a.act(None) for _ in range(50)]
    assert actions == [b.act(None) for _ in range(50)]
    assert all(0 <= x < 7 for x in actions)


def test_constant_policy():
    assert ConstantPolicy(action=3).act(None) == 3


def test_ppo_policy_act_returns_valid_action():
    policy = PPOPolicy(ConvolutionalNeuralNetwork(output_dim=7))
    observation = np.zeros((4, 84, 84), dtype=np.uint8)

    action = policy.act(observation)

    assert 0 <= action < 7


def test_ppo_policy_loads_checkpoint(tmp_path):
    model = ConvolutionalNeuralNetwork(output_dim=7)
    with torch.no_grad():
        model.output.weight.fill_(0.25)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, None, step=1)

    restored = ConvolutionalNeuralNetwork(output_dim=7)
    PPOPolicy(restored, checkpoint_path=path)

    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.equal(a, b)
