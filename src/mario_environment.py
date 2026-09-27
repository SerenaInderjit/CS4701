from typing import Dict, Any, Tuple, cast
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


class MarioEnvironment:
    def __init__(self, env_name: str = "SuperMarioBros-v0"):
        env = gym_super_mario_bros.make(env_name)
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)

    def reset(self) -> Dict[str, Any]:
        observation = self.env.reset()

        return {
            "observation": observation,
            "episode_ended": False,
            "info": {},
        }

    def step(self, action: int) -> Dict[str, Any]:
        observation, reward, episode_ended, info = cast(
            Tuple[Any, float, bool, Dict[Any, Any]],
            self.env.step(action),
        )

        return {
            "observation": observation,
            "reward": reward,
            "episode_ended": episode_ended,
            "info": info,
        }    

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()

    