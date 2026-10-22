"""Circular transition buffer for off-policy learning (DQN).

This is deliberately *not* the same thing as `RolloutBuffer`
(src/training/rollout_buffer.py). A rollout buffer is on-policy: it is filled
sequentially, read once, and cleared after every update. A replay buffer is
off-policy: it keeps transitions around, samples them in random order, and
reuses each one many times.

Why DQN needs this:
  * Consecutive frames of Mario are extremely correlated. Training on them in
    order makes the gradient steps bounce around; shuffling decorrelates them.
  * Every transition collected is used for several gradient steps, so the
    agent learns from each (expensive) emulator step more than once.

Memory: a transition stores two stacked observations of `observation_shape`
uint8, i.e. 2 * prod(shape) bytes. At the default (4, 84, 84) that is
~56 KB per transition, so capacity=30000 needs ~1.7 GB and capacity=100000
needs ~5.6 GB. Size the capacity against the RAM you actually have.
"""
from typing import Dict, Tuple

import numpy as np


class ReplayBuffer:
    """Fixed-capacity circular buffer of (s, a, r, s', done) transitions.

    `add` overwrites the oldest entry once the buffer is full, so the buffer
    always holds the most recent `capacity` transitions. `sample` draws
    uniformly at random from everything currently stored.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, ...],
        observation_dtype=np.uint8,
    ):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.observation_shape = tuple(observation_shape)
        self.observations = np.zeros((capacity, *self.observation_shape), dtype=observation_dtype)
        self.next_observations = np.zeros((capacity, *self.observation_shape), dtype=observation_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        # `head` is the slot holding the oldest transition, `size` how many are
        # stored. Entries live in the circular window [head, head + size).
        self.head = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """True once there are enough transitions for a full mini-batch."""
        return self.size >= batch_size

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_observation: np.ndarray,
    ) -> None:
        """Store one transition, overwriting the oldest one when full."""
        if self.size < self.capacity:
            idx = (self.head + self.size) % self.capacity
            self.size += 1
        else:
            # Full: overwrite the oldest entry and move the window forward.
            idx = self.head
            self.head = (self.head + 1) % self.capacity

        self.observations[idx] = observation
        self.next_observations[idx] = next_observation
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

    def _indices(self) -> np.ndarray:
        """Physical slot indices of the stored transitions, oldest first."""
        offsets = np.arange(self.size)
        return (self.head + offsets) % self.capacity

    def chronological(self) -> Dict[str, np.ndarray]:
        """Every stored transition, ordered oldest first.

        Handy for debugging and for tests: the raw arrays are a ring, so their
        physical order does not match the order the transitions were added.
        """
        idx = self._indices()
        return {
            "observations": self.observations[idx].copy(),
            "next_observations": self.next_observations[idx].copy(),
            "actions": self.actions[idx].copy(),
            "rewards": self.rewards[idx].copy(),
            "dones": self.dones[idx].copy(),
        }

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Draw `batch_size` transitions uniformly at random.

        Returns copies, so the caller can keep them past the next `add`.
        """
        if not self.is_ready(batch_size):
            raise ValueError(
                f"Not enough transitions ({self.size}) for a batch of {batch_size}"
            )
        stored = self._indices()
        picks = np.random.randint(0, self.size, size=batch_size)
        idx = stored[picks]
        return {
            "observations": self.observations[idx].copy(),
            "next_observations": self.next_observations[idx].copy(),
            "actions": self.actions[idx].copy(),
            "rewards": self.rewards[idx].copy(),
            "dones": self.dones[idx].copy(),
        }
