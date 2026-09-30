import random
from typing import Optional


class RandomPolicy:
    """Uniformly random actions. The Milestone #0 dummy policy, moved out of tests."""

    def __init__(self, num_actions: int = 7, seed: Optional[int] = None):
        # SIMPLE_MOVEMENT has 7 actions
        self.num_actions = num_actions
        self._rng = random.Random(seed)

    def act(self, observation) -> int:
        return self._rng.randrange(self.num_actions)


class ConstantPolicy:
    """Always plays the same action. In SIMPLE_MOVEMENT, 1 = 'right'."""

    def __init__(self, action: int = 1):
        self.action = action

    def act(self, observation) -> int:
        return self.action
