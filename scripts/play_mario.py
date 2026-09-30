"""Watch a policy play Mario.

Usage: ./run.sh play  (or: python scripts/play_mario.py --policy right --episodes 3)

--policy is 'random', 'right', or 'ppo'. 'ppo' loads a trained checkpoint
(produced by scripts/train.py) and needs the preprocessed training
observations, so it runs in a make_training_environment instead of the raw
one the baselines use.
"""
import argparse

from src.algorithms.cnn import ConvolutionalNeuralNetwork
from src.environment.mario_environment import MarioEnvironment, make_training_environment
from src.policies.baselines import ConstantPolicy, RandomPolicy
from src.policies.ppo_policy import PPOPolicy
from src.training.logger import Logger
from src.training.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "right", "ppo"], default="random")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="trained policy checkpoint (required for --policy ppo)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=2000, help="max steps per episode")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.policy == "ppo":
        if args.checkpoint is None:
            parser.error("--policy ppo requires --checkpoint")
        # PPO acts on preprocessed (4, 84, 84) stacks, so it needs the training env.
        environment = make_training_environment()
        model = ConvolutionalNeuralNetwork(output_dim=environment.num_actions)
        policy = PPOPolicy(model, checkpoint_path=args.checkpoint, device=args.device)
    else:
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
