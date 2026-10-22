import numpy as np
import pytest

from src.training.replay_buffer import ReplayBuffer


def make_buffer(capacity=10, shape=(4, 8, 8)):
    return ReplayBuffer(capacity=capacity, observation_shape=shape)


def test_starts_empty():
    buffer = make_buffer()
    assert len(buffer) == 0
    assert not buffer.is_ready(1)


def test_add_stores_the_transition():
    buffer = make_buffer()
    obs = np.ones((4, 8, 8), dtype=np.uint8)
    next_obs = np.full((4, 8, 8), 2, dtype=np.uint8)

    buffer.add(obs, action=3, reward=1.5, done=False, next_observation=next_obs)

    assert len(buffer) == 1
    assert buffer.actions[0] == 3
    assert buffer.rewards[0] == pytest.approx(1.5)
    assert not buffer.dones[0]
    np.testing.assert_array_equal(buffer.observations[0], obs)
    np.testing.assert_array_equal(buffer.next_observations[0], next_obs)


def test_buffer_overwrites_the_oldest_entry_when_full():
    buffer = make_buffer(capacity=3)
    for i in range(5):
        obs = np.full((4, 8, 8), i, dtype=np.uint8)
        buffer.add(obs, action=i, reward=float(i), done=False, next_observation=obs)

    assert len(buffer) == 3
    # The three most recent writes survive; the first two were overwritten.
    stored = buffer.chronological()["observations"][:, 0, 0, 0]
    assert sorted(stored.tolist()) == [2, 3, 4]


def test_stored_entries_are_ordered_oldest_first():
    buffer = make_buffer(capacity=3)
    for i in range(5):
        obs = np.full((4, 8, 8), i, dtype=np.uint8)
        buffer.add(obs, action=i, reward=float(i), done=False, next_observation=obs)

    stored = buffer.chronological()["observations"][:, 0, 0, 0]
    assert stored.tolist() == [2, 3, 4]


def test_is_ready_only_once_there_are_enough_transitions():
    buffer = make_buffer(capacity=10)
    for i in range(3):
        obs = np.zeros((4, 8, 8), dtype=np.uint8)
        buffer.add(obs, action=0, reward=0.0, done=False, next_observation=obs)

    assert not buffer.is_ready(4)
    assert buffer.is_ready(3)


def test_sample_returns_the_requested_batch():
    buffer = make_buffer(capacity=10)
    for i in range(10):
        obs = np.full((4, 8, 8), i, dtype=np.uint8)
        buffer.add(obs, action=i % 7, reward=float(i), done=i % 2 == 0, next_observation=obs)

    batch = buffer.sample(4)

    assert set(batch) == {"observations", "next_observations", "actions", "rewards", "dones"}
    assert batch["observations"].shape == (4, 4, 8, 8)
    assert batch["next_observations"].shape == (4, 4, 8, 8)
    assert batch["actions"].shape == (4,)
    assert batch["rewards"].shape == (4,)
    assert batch["dones"].shape == (4,)


def test_sample_only_returns_stored_transitions():
    buffer = make_buffer(capacity=10)
    for i in range(1, 5):
        obs = np.full((4, 8, 8), i, dtype=np.uint8)
        buffer.add(obs, action=i, reward=float(i), done=False, next_observation=obs)

    batch = buffer.sample(4)

    # Only stored rewards may come back, never an uninitialised one (0.0).
    assert set(batch["rewards"].tolist()) <= {1.0, 2.0, 3.0, 4.0}


def test_sample_does_not_return_overwritten_transitions():
    buffer = make_buffer(capacity=4)
    for i in range(4):
        obs = np.full((4, 8, 8), i, dtype=np.uint8)
        buffer.add(obs, action=i, reward=float(i), done=False, next_observation=obs)

    # Overwrite everything with new values.
    for i in range(4, 8):
        obs = np.full((4, 8, 8), i, dtype=np.uint8)
        buffer.add(obs, action=i, reward=float(i), done=False, next_observation=obs)

    batch = buffer.sample(4)

    assert set(batch["rewards"].tolist()) <= {4.0, 5.0, 6.0, 7.0}


def test_sample_returns_copies_that_survive_further_adds():
    buffer = make_buffer(capacity=4)
    for i in range(4):
        obs = np.full((4, 8, 8), i, dtype=np.uint8)
        buffer.add(obs, action=0, reward=float(i), done=False, next_observation=obs)

    batch = buffer.sample(4)
    np.full((4, 8, 8), 99, dtype=np.uint8)
    buffer.add(np.zeros((4, 8, 8), dtype=np.uint8), action=0, reward=99.0, done=False,
               next_observation=np.zeros((4, 8, 8), dtype=np.uint8))

    assert 99.0 not in batch["rewards"].tolist()


def test_sample_raises_when_the_buffer_is_too_small():
    buffer = make_buffer(capacity=10)
    obs = np.zeros((4, 8, 8), dtype=np.uint8)
    buffer.add(obs, action=0, reward=0.0, done=False, next_observation=obs)

    with pytest.raises(ValueError):
        buffer.sample(8)


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0, observation_shape=(4, 8, 8))
