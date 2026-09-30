import numpy as np
import torch

from src.algorithms.ppo import PPO


def make_agent(**overrides):
    kwargs = {
        "observation_shape": (4, 84, 84),
        "num_actions": 7,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "epochs": 2,
        "minibatch_size": 8,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return PPO(**kwargs)


def test_act_returns_action_and_extras():
    agent = make_agent()
    observation = np.zeros((4, 84, 84), dtype=np.uint8)

    action, extras = agent.act(observation)

    assert 0 <= action < 7
    assert isinstance(extras["log_prob"], float)
    assert isinstance(extras["value"], float)


def test_value_of_returns_float():
    agent = make_agent()
    observation = np.zeros((4, 84, 84), dtype=np.uint8)

    value = agent.value_of(observation)

    assert isinstance(value, float)


def test_compute_gae_with_gamma_one_lambda_one():
    # With gamma=lambda=1 and zero values, advantage at t is the sum of
    # remaining rewards: A = [3, 2, 1], returns = advantages + values.
    agent = make_agent()
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    dones = np.array([False, False, True])
    values = np.zeros(3, dtype=np.float32)

    advantages, returns = agent.compute_gae(rewards, dones, values, last_value=0.0)

    np.testing.assert_allclose(advantages, [3.0, 2.0, 1.0])
    np.testing.assert_allclose(returns, [3.0, 2.0, 1.0])


def test_compute_gae_done_cuts_bootstrapping():
    # A done step must not bootstrap from last_value, however large.
    agent = make_agent(gamma=0.99)
    rewards = np.array([1.0], dtype=np.float32)
    dones = np.array([True])
    values = np.zeros(1, dtype=np.float32)

    advantages, returns = agent.compute_gae(rewards, dones, values, last_value=100.0)

    np.testing.assert_allclose(advantages, [1.0])
    np.testing.assert_allclose(returns, [1.0])


def test_compute_gae_bootstraps_final_step():
    # The final step of an unfinished rollout bootstraps from last_value.
    agent = make_agent(gamma=0.5, gae_lambda=1.0)
    rewards = np.array([0.0], dtype=np.float32)
    dones = np.array([False])
    values = np.array([0.0], dtype=np.float32)

    advantages, _ = agent.compute_gae(rewards, dones, values, last_value=2.0)

    # delta = 0 + 0.5 * 2.0 - 0 = 1.0
    np.testing.assert_allclose(advantages, [1.0])


def test_update_returns_mean_losses():
    agent = make_agent()
    n = 16
    observations = np.random.randint(0, 256, size=(n, 4, 84, 84), dtype=np.uint8)
    actions = np.random.randint(0, 7, size=n)
    old_log_probs = np.full(n, -1.0, dtype=np.float32)
    advantages = np.random.randn(n).astype(np.float32)
    returns = np.random.randn(n).astype(np.float32)

    metrics = agent.update(observations, actions, old_log_probs, advantages, returns)

    assert set(metrics) == {"policy_loss", "value_loss"}
    assert metrics["policy_loss"] > 0
    assert metrics["value_loss"] > 0


def test_update_changes_policy():
    agent = make_agent()
    # A fixed batch of observations and actions; the update must move the
    # policy's logits for those inputs.
    n = 16
    observations = np.random.randint(0, 256, size=(n, 4, 84, 84), dtype=np.uint8)
    actions = np.random.randint(0, 7, size=n)
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32) / 255.0
    before = agent.policy_model(obs_tensor).detach().clone()

    old_log_probs = np.full(n, -1.0, dtype=np.float32)
    advantages = np.random.randn(n).astype(np.float32)
    returns = np.zeros(n, dtype=np.float32)
    agent.update(observations, actions, old_log_probs, advantages, returns)

    after = agent.policy_model(obs_tensor).detach()
    assert not torch.allclose(before, after)
