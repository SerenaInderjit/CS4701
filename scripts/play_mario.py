"""Watch a policy play Mario.

Usage: ./run.sh play  (or: python scripts/play_mario.py --policy right --episodes 3)

--policy is 'random', 'right', or 'ppo'. 'ppo' loads a trained checkpoint
(produced by scripts/train.py) and needs the preprocessed training
observations, so it runs in a make_training_environment instead of the raw
one the baselines use.

--fps controls playback speed (agent steps per second). Default 15 is
approximately real-time for frame_skip=4 on the NES (60 FPS / 4).
"""
import argparse
import time

from src.algorithms.cnn import ConvolutionalNeuralNetwork
from src.environment.mario_environment import MarioEnvironment, make_training_environment
from src.policies.baselines import ConstantPolicy, RandomPolicy
from src.policies.ppo_policy import PPOPolicy
from src.training.logger import Logger
from src.training.runner import Runner


def make_policy(name, num_actions, checkpoint=None, device="cpu"):
    if name == "ppo":
        model = ConvolutionalNeuralNetwork(output_dim=num_actions)
        return PPOPolicy(model, checkpoint_path=checkpoint, device=device)
    if name == "random":
        return RandomPolicy(num_actions=num_actions)
    return ConstantPolicy(action=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "right", "ppo"], default="random")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="trained policy checkpoint (required for --policy ppo)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=2000, help="max steps per episode")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fps", type=int, default=15,
                        help="agent steps per second (15 ≈ real-time for frame_skip=4)")
    args = parser.parse_args()

    if args.policy == "ppo" and args.checkpoint is None:
        parser.error("--policy ppo requires --checkpoint")

    # PPO acts on preprocessed (4, 84, 84) stacks, so it needs the training env.
    if args.policy == "ppo":
        environment = make_training_environment()
    else:
        environment = MarioEnvironment()

    policy = make_policy(args.policy, environment.num_actions, args.checkpoint, args.device)

    logger = Logger()
    runner = Runner(environment, policy, logger, render=True)
    step_delay = 1.0 / args.fps if args.fps > 0 else 0.0

    try:
        for _ in range(args.episodes):
            runner.run_episode(timeout=args.timeout)
            if step_delay > 0:
                time.sleep(step_delay)
    finally:
        environment.close()

    for episode in logger.episodes:
        print(
            f"episode {episode['episode']}: distance={episode['max_x_pos']} "
            f"steps={episode['length']} reward={episode['total_reward']:.1f}"
        )


if __name__ == "__main__":
    main()
