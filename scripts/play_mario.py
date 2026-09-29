"""Watch a baseline policy play Mario.

Usage: ./run.sh play  (or: python scripts/play_mario.py --policy right --episodes 3)
"""
import argparse

from src.logger import Logger
from src.mario_environment import MarioEnvironment
from src.policies import ConstantPolicy, RandomPolicy
from src.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "right"], default="random")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=2000, help="max steps per episode")
    args = parser.parse_args()

    environment = MarioEnvironment()
    if args.policy == "random":
        policy = RandomPolicy(num_actions=environment.num_actions)
    else:
        policy = ConstantPolicy(action=1)

    logger = Logger()
    runner = Runner(environment, policy, logger, render=True)
    try:
        for _ in range(args.episodes):
            runner.run_episode(timeout=args.timeout)
    finally:
        environment.close()

    for episode in logger.episodes:
        print(
            f"episode {episode['episode']}: distance={episode['max_x_pos']} "
            f"steps={episode['length']} reward={episode['total_reward']:.1f}"
        )


if __name__ == "__main__":
    main()
