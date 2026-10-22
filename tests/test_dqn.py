import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.algorithms.dqn import DQN


def make_agent(**overrides):
    kwargs = {
        "observation_shape": (4, 40, 40),
        "num_actions": 7,
        "gamma": 0.99,
        "batch_size": 8,
        "target_update_freq": 1000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 100,
        "hidden_dim": 32,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return DQN(**kwargs)


def make_batch(n=8, shape=(4, 40, 40)):
    return {
        "observations": np.random.randint(0, 256, size=(n, *shape), dtype=np.uint8),
        "next_observations": np.random.randint(0, 256, size=(n, *shape), dtype=np.uint8),
        "actions": np.random.randint(0, 7, size=n),
        "rewards": np.random.randn(n).astype(np.float32),
        "dones": np.zeros(n, dtype=bool),
    }


def state_dicts_equal(a, b):
    for key in a:
        for name in a[key]:
            left, right = a[key][name], b[key][name]
            if not torch.allclose(left.float(), right.float()):
                return False
    return True


def test_act_returns_a_valid_action():
    agent = make_agent()
    observation = np.zeros((4, 40, 40), dtype=np.uint8)

    action = agent.act(observation)

    assert 0 <= action < 7


def test_epsilon_one_means_always_random():
    agent = make_agent(epsilon_start=1.0, epsilon_end=1.0, epsilon_decay_steps=100)
    observation = np.zeros((4, 40, 40), dtype=np.uint8)

    actions = {agent.act(observation) for _ in range(200)}

    # A greedy policy would return the same action every time.
    assert len(actions) > 1


def test_epsilon_zero_means_greedy():
    agent = make_agent(epsilon_start=0.0, epsilon_end=0.0)
    observation = np.random.randint(0, 256, size=(4, 40, 40), dtype=np.uint8)

    expected = int(np.argmax(agent.q_values(observation)))
    actions = {agent.act(observation) for _ in range(20)}

    assert actions == {expected}


def test_epsilon_decays_linearly_to_the_end_value():
    agent = make_agent(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=10)

    for _ in range(5):
        agent.anneal_epsilon()
    assert agent.epsilon == pytest.approx(0.55)

    for _ in range(5):
        agent.anneal_epsilon()
    assert agent.epsilon == pytest.approx(0.1)

    # It stays at the end value once the schedule is over.
    agent.anneal_epsilon()
    assert agent.epsilon == pytest.approx(0.1)


def test_q_values_have_one_entry_per_action():
    agent = make_agent()
    observation = np.zeros((4, 40, 40), dtype=np.uint8)

    q_values = agent.q_values(observation)

    assert q_values.shape == (7,)


def test_train_step_returns_diagnostics():
    agent = make_agent()

    metrics = agent.train_step(make_batch())

    assert set(metrics) == {"loss", "td_error", "q_mean", "grad_norm"}
    assert metrics["loss"] >= 0.0
    assert metrics["td_error"] >= 0.0


def test_train_step_changes_the_online_weights():
    agent = make_agent()
    before = {k: v.clone() for k, v in agent.model.state_dict().items()}

    agent.train_step(make_batch())

    after = agent.model.state_dict()
    assert any(not torch.allclose(before[k].float(), after[k].float()) for k in before)


def test_target_network_is_not_updated_by_a_gradient_step():
    agent = make_agent(target_update_freq=1000)
    target_before = {k: v.clone() for k, v in agent.target_model.state_dict().items()}

    agent.train_step(make_batch())

    after = agent.target_model.state_dict()
    assert all(torch.allclose(target_before[k].float(), after[k].float()) for k in target_before)


def test_target_network_syncs_after_the_configured_number_of_updates():
    agent = make_agent(target_update_freq=3)
    target_before = {k: v.clone() for k, v in agent.target_model.state_dict().items()}

    agent.train_step(make_batch())
    agent.train_step(make_batch())
    assert agent.updates == 2
    after_two = agent.target_model.state_dict()
    assert all(torch.allclose(target_before[k].float(), after_two[k].float()) for k in target_before)

    agent.train_step(make_batch())
    assert agent.updates == 3
    after_three = agent.target_model.state_dict()
    assert any(not torch.allclose(target_before[k].float(), after_three[k].float()) for k in target_before)


def test_sync_target_copies_the_online_weights():
    agent = make_agent()
    agent.train_step(make_batch())

    agent.sync_target()

    assert state_dicts_equal(
        {"m": agent.model.state_dict()}, {"m": agent.target_model.state_dict()}
    )


def chosen_q_values(agent, batch):
    """Q-value of the taken action, evaluated with the *current* weights."""
    observations = torch.as_tensor(batch["observations"], dtype=torch.float32)
    actions = torch.as_tensor(batch["actions"], dtype=torch.long)
    with torch.no_grad():
        q_values = agent.model(observations)
        return q_values.gather(1, actions.unsqueeze(1)).squeeze(1)


def test_td_target_uses_the_target_network_and_the_reward():
    # With a constant target network the max Q-value is known exactly, so the
    # Huber loss can be recomputed by hand.
    agent = make_agent(gamma=0.9)
    with torch.no_grad():
        agent.target_model.output.weight.zero_()
        agent.target_model.output.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    batch = make_batch(n=4)
    batch["rewards"] = np.array([1.0, -1.0, 0.5, 2.0], dtype=np.float32)
    batch["dones"] = np.array([False, False, False, False])

    # Snapshot the online Q-values before the gradient step changes them.
    chosen = chosen_q_values(agent, batch)
    expected_targets = torch.tensor(batch["rewards"], dtype=torch.float32) + 0.9 * 6.0
    expected_loss = F.smooth_l1_loss(chosen, expected_targets, beta=1.0)

    metrics = agent.train_step(batch)

    assert metrics["loss"] == pytest.approx(expected_loss.item(), rel=1e-5)


def test_a_done_transition_does_not_bootstrap():
    agent = make_agent(gamma=0.9)
    with torch.no_grad():
        agent.target_model.output.weight.zero_()
        agent.target_model.output.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    batch = make_batch(n=4)
    batch["rewards"] = np.array([1.0, -1.0, 0.5, 2.0], dtype=np.float32)
    batch["dones"] = np.array([True, True, True, True])

    chosen = chosen_q_values(agent, batch)
    # A terminal transition has no next state to bootstrap from.
    expected_targets = torch.tensor(batch["rewards"], dtype=torch.float32)
    expected_loss = F.smooth_l1_loss(chosen, expected_targets, beta=1.0)

    metrics = agent.train_step(batch)

    assert metrics["loss"] == pytest.approx(expected_loss.item(), rel=1e-5)


def test_a_done_transition_does_not_bootstrap_but_a_live_one_does():
    agent = make_agent(gamma=0.9)
    with torch.no_grad():
        agent.target_model.output.weight.zero_()
        agent.target_model.output.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

    batch = make_batch(n=4)
    batch["rewards"] = np.array([1.0, -1.0, 0.5, 2.0], dtype=np.float32)
    batch["dones"] = np.array([True, False, True, False])

    chosen = chosen_q_values(agent, batch)
    bootstrap = torch.tensor([0.0, 0.9 * 6.0, 0.0, 0.9 * 6.0])
    expected_targets = torch.tensor(batch["rewards"], dtype=torch.float32) + bootstrap
    expected_loss = F.smooth_l1_loss(chosen, expected_targets, beta=1.0)

    metrics = agent.train_step(batch)

    assert metrics["loss"] == pytest.approx(expected_loss.item(), rel=1e-5)


def test_state_dicts_round_trip():
    agent = make_agent()
    agent.train_step(make_batch())
    agent.anneal_epsilon()

    other = make_agent()
    other.load_state_dicts(agent.state_dicts())

    assert other.epsilon == agent.epsilon
    assert other.epsilon_steps == agent.epsilon_steps
    assert other.updates == agent.updates
    assert state_dicts_equal(
        {"m": agent.model.state_dict()}, {"m": other.model.state_dict()}
    )
    assert state_dicts_equal(
        {"m": agent.target_model.state_dict()}, {"m": other.target_model.state_dict()}
    )


def test_set_learning_rate_updates_the_optimizer():
    agent = make_agent()

    agent.set_learning_rate(1e-3)

    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
