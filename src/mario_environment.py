from collections import deque
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from src.preprocessing import ObservationPreprocessor
from src.reward import RewardConfig, RewardShaper


class MarioEnvironment:
    """Thin wrapper around gym-super-mario-bros with a dict-based interface.

    With the defaults (frame_skip=1, no preprocessor, no reward shaper) this
    behaves exactly like the Milestone #0 environment: raw frames and the
    built-in reward. Use `make_training_environment()` for the RL setup.

    Args:
        frame_skip: repeat each action this many emulator frames, summing the
            raw reward. With frame_skip > 1 the returned frame is the pixel-wise
            max of the last two frames, which removes NES sprite flicker.
        preprocessor: optional ObservationPreprocessor (grayscale/resize/stack).
        reward_shaper: optional RewardShaper replacing the built-in reward.
            The original reward is still returned as `raw_reward`.
    """

    def __init__(
        self,
        env_name: str = "SuperMarioBros-1-1-v0",
        frame_skip: int = 1,
        preprocessor: Optional[ObservationPreprocessor] = None,
        reward_shaper: Optional[RewardShaper] = None,
    ):
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")

        env = gym_super_mario_bros.make(env_name)
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.frame_skip = frame_skip
        self.preprocessor = preprocessor
        self.reward_shaper = reward_shaper

    @property
    def num_actions(self) -> int:
        return len(SIMPLE_MOVEMENT)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        if self.preprocessor is not None:
            return self.preprocessor.observation_shape
        return tuple(self.env.observation_space.shape)

    def _process(self, frame: np.ndarray, first: bool) -> Any:
        if self.preprocessor is None:
            return frame
        return self.preprocessor.reset(frame) if first else self.preprocessor.step(frame)

    def reset(self) -> Dict[str, Any]:
        frame = np.array(self.env.reset())  # copy: nes-py reuses its screen buffer
        if self.reward_shaper is not None:
            self.reward_shaper.reset()

        return {
            "observation": self._process(frame, first=True),
            "episode_ended": False,
            "info": {},
        }

    def step(self, action: int) -> Dict[str, Any]:
        raw_reward = 0.0
        episode_ended = False
        info: Dict[Any, Any] = {}
        frames = deque(maxlen=2)

        for _ in range(self.frame_skip):
            observation, reward, episode_ended, info = cast(
                Tuple[Any, float, bool, Dict[Any, Any]],
                self.env.step(action),
            )
            raw_reward += reward
            frames.append(np.array(observation))  # copy: nes-py reuses its screen buffer
            if episode_ended:
                break

        frame = frames[0] if len(frames) == 1 else np.maximum(frames[0], frames[1])
        episode_ended = bool(episode_ended)

        if self.reward_shaper is not None:
            reward = self.reward_shaper(info, episode_ended)
        else:
            reward = raw_reward

        return {
            "observation": self._process(frame, first=False),
            "reward": reward,
            "raw_reward": raw_reward,
            "episode_ended": episode_ended,
            "info": info,
        }

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()


def make_training_environment(
    env_name: str = "SuperMarioBros-1-1-v0",
    frame_skip: int = 4,
    num_stack: int = 4,
    frame_size: Tuple[int, int] = (84, 84),
    reward_config: Optional[RewardConfig] = None,
) -> MarioEnvironment:
    """The standard RL setup: frame skip + grayscale/resize/stack + shaped reward."""
    return MarioEnvironment(
        env_name=env_name,
        frame_skip=frame_skip,
        preprocessor=ObservationPreprocessor(frame_size=frame_size, num_stack=num_stack),
        reward_shaper=RewardShaper(reward_config),
    )
