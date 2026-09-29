from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RewardConfig:
    """Tunable weights for reward shaping (candidates for Milestone #4 tuning)."""

    x_progress_weight: float = 1.0    # per pixel of rightward movement
    time_penalty_weight: float = 0.1  # per game-clock tick that elapsed
    death_penalty: float = -15.0      # episode ended without reaching the flag
    flag_bonus: float = 15.0          # reached the flag (level complete)
    reward_clip: float = 15.0         # final reward is clipped to [-clip, clip]


class RewardShaper:
    """Replaces the built-in reward with a shaped one.

    reward = x_weight * delta_x
             - time_weight * (clock ticks elapsed)
             + flag_bonus     (on level completion)
             + death_penalty  (episode ended without the flag)

    The shaper is stateful (it remembers the previous x_pos / clock), so call
    `reset()` at the start of every episode. MarioEnvironment does this.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self._prev_x: Optional[float] = None
        self._prev_time: Optional[float] = None

    def reset(self) -> None:
        self._prev_x = None
        self._prev_time = None

    def __call__(self, info: Dict[str, Any], episode_ended: bool) -> float:
        cfg = self.config

        x_pos = info.get("x_pos", self._prev_x)
        clock = info.get("time", self._prev_time)

        # The first step of an episode has no previous values to diff against.
        delta_x = 0.0 if self._prev_x is None or x_pos is None else x_pos - self._prev_x
        # The clock only counts down; ignore any increase.
        ticks = 0.0 if self._prev_time is None or clock is None else max(self._prev_time - clock, 0)

        self._prev_x = x_pos
        self._prev_time = clock

        reward = cfg.x_progress_weight * delta_x - cfg.time_penalty_weight * ticks

        if episode_ended:
            if info.get("flag_get", False):
                reward += cfg.flag_bonus
            else:
                # In gym-super-mario-bros an episode ends on every death
                # (or when time runs out), so "ended without flag" == failure.
                reward += cfg.death_penalty

        return float(max(-cfg.reward_clip, min(cfg.reward_clip, reward)))
