import numpy as np
import pytest

from src.policies.baselines import ConstantPolicy
from src.training.logger import Logger
from src.training.rollout_buffer import RolloutBuffer
from src.training.runner import Runner
from tests.fakes import FakeEnvironment


def make_buffer(capacity=5):
    return RolloutBuffer(capacity, observation_shape=(2, 3, 3))


def test_add_and_get():
    buffer = make_buffer()
    buffer.add(np.ones((2, 3, 3)), action=3, reward=1.5, done=False, log_prob=-0.5, value=2.0)
    buffer.add(np.zeros((2, 3, 3)), action=1, reward=-1.0, done=True)

    data = buffer.get()
    assert len(buffer) == 2
    assert data["observations"].shape == (2, 2, 3, 3)
    assert data["actions"].tolist() == [3, 1]
    assert data["rewards"].tolist() == [1.5, -1.0]
    assert data["dones"].tolist() == [False, True]
    assert data["log_probs"][0] == pytest.approx(-0.5)
    assert data["values"][0] == pytest.approx(2.0)


def test_full_and_clear():
    buffer = make_buffer(capacity=2)
    buffer.add(np.zeros((2, 3, 3)), 0, 0.0, False)
    buffer.add(np.zeros((2, 3, 3)), 0, 0.0, False)
    assert buffer.is_full
    with pytest.raises(ValueError):
        buffer.add(np.zeros((2, 3, 3)), 0, 0.0, False)
    buffer.clear()
    assert len(buffer) == 0 and not buffer.is_full


def test_collect_rollout_stores_aligned_transitions_across_episodes():
    environment = FakeEnvironment(episode_length=4)
    logger = Logger()
    runner = Runner(environment, ConstantPolicy(1), logger)
    buffer = RolloutBuffer(10, environment.observation_shape)

    runner.collect_rollout(10, buffer)
    data = buffer.get()

    # Episodes of length 4: dones at steps 3 and 7; observation is the step counter.
    assert data["dones"].tolist() == [False, False, False, True] * 2 + [False, False]
    assert data["observations"][:, 0, 0, 0].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    assert (data["actions"] == 1).all()
    assert data["rewards"].tolist() == [5.0] * 10
    # After 10 steps we're 2 steps into a third episode.
    assert data["last_observation"][0, 0, 0] == 2
    assert len(logger.episodes) == 2


def test_rollout_continues_episode_across_calls():
    environment = FakeEnvironment(episode_length=6)
    runner = Runner(environment, ConstantPolicy(1), Logger())
    buffer = RolloutBuffer(4, environment.observation_shape)

    runner.collect_rollout(4, buffer)
    buffer.clear()
    runner.collect_rollout(4, buffer)

    # Second rollout picks up mid-episode (obs 4, 5, then a fresh episode 0, 1).
    assert buffer.get()["observations"][:, 0, 0, 0].tolist() == [4, 5, 0, 1]
    assert environment.num_resets == 2


def test_collect_rollout_rejects_overflow():
    environment = FakeEnvironment()
    runner = Runner(environment, ConstantPolicy(1), Logger())
    with pytest.raises(ValueError):
        runner.collect_rollout(5, RolloutBuffer(3, environment.observation_shape))


def test_policy_extras_are_stored():
    class ExtrasPolicy:
        def act(self, observation):
            return 2, {"log_prob": -0.25, "value": 1.5}

    environment = FakeEnvironment()
    runner = Runner(environment, ExtrasPolicy(), Logger())
    buffer = RolloutBuffer(3, environment.observation_shape)
    runner.collect_rollout(3, buffer)

    data = buffer.get()
    assert data["actions"].tolist() == [2, 2, 2]
    assert data["log_probs"].tolist() == [-0.25] * 3
    assert data["values"].tolist() == [1.5] * 3
