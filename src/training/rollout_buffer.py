from typing import Dict, Optional, Tuple

import numpy as np


class RolloutBuffer:
    """Fixed-capacity, time-ordered storage for on-policy trajectories (PPO).

    Transition t is stored as (observation_t, action_t, reward_t, done_t, ...):
      - observation_t: what the policy saw before acting
      - reward_t / done_t: the result of taking action_t
      - done_t=True means the episode ended on that step, so the next stored
        observation belongs to a fresh episode (cut GAE bootstrapping there).

    `last_observation` is the observation the policy would see next after the
    final stored step. PPO uses it to bootstrap the value of the truncated
    trajectory. `log_prob` and `value` are optional and default to 0 so the
    buffer also works for policies that don't produce them (random, DQN, ...).

    Multiple rollouts can be appended across episode boundaries; the buffer does
    not need to start or end on an episode boundary.
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
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.last_observation: Optional[np.ndarray] = None
        self.size = 0

    def __len__(self) -> int:
        return self.size

    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity

    def add(
        self,
        observation,
        action: int,
        reward: float,
        done: bool,
        log_prob: float = 0.0,
        value: float = 0.0,
    ) -> None:
        if self.is_full:
            raise ValueError("RolloutBuffer is full; call clear() before adding more")
        i = self.size
        self.observations[i] = observation
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.size += 1

    def set_last_observation(self, observation) -> None:
        self.last_observation = np.array(observation, dtype=self.observations.dtype)

    def clear(self) -> None:
        self.size = 0
        self.last_observation = None

    def get(self) -> Dict[str, np.ndarray]:
        """Return the stored transitions as a dict of arrays.

        Returns views into the underlying buffers (no copies). The caller must
        not modify the returned arrays; they become invalid after clear().
        """
        n = self.size
        data = {
            "observations": self.observations[:n],
            "actions": self.actions[:n],
            "rewards": self.rewards[:n],
            "dones": self.dones[:n],
            "log_probs": self.log_probs[:n],
            "values": self.values[:n],
        }
        if self.last_observation is not None:
            data["last_observation"] = self.last_observation
        return data
