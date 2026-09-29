import numpy as np


class FakeEnvironment:
    """Emulator-free stand-in that follows the MarioEnvironment dict interface.

    Mario moves `x_step` pixels right per step; each episode lasts
    `episode_length` steps. The observation is filled with the step counter so
    tests can check exactly which observation was stored for which step.
    """

    def __init__(self, episode_length=10, observation_shape=(4, 8, 8), flag_at_end=False, x_step=5):
        self.episode_length = episode_length
        self.observation_shape = observation_shape
        self.flag_at_end = flag_at_end
        self.x_step = x_step
        self.t = 0
        self.num_resets = 0

    def _obs(self):
        return np.full(self.observation_shape, self.t % 256, dtype=np.uint8)

    def reset(self):
        self.t = 0
        self.num_resets += 1
        return {"observation": self._obs(), "episode_ended": False, "info": {}}

    def step(self, action):
        self.t += 1
        ended = self.t >= self.episode_length
        info = {
            "x_pos": 40 + self.x_step * self.t,
            "time": 400 - self.t,
            "flag_get": bool(ended and self.flag_at_end),
            "life": 2,
        }
        return {
            "observation": self._obs(),
            "reward": float(self.x_step),
            "episode_ended": ended,
            "info": info,
        }

    def render(self):
        pass

    def close(self):
        pass
