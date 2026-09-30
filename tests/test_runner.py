from src.environment.mario_environment import MarioEnvironment
from src.policies.baselines import RandomPolicy
from src.training.logger import Logger
from src.training.runner import Runner


def test_runner():
    environment = MarioEnvironment()
    policy = RandomPolicy(num_actions=environment.num_actions)
    logger = Logger()

    runner = Runner(environment, policy, logger, render=True)

    try:
        result = runner.run_episode(timeout=100)

        assert "observation" in result
        assert "reward" in result
        assert "episode_ended" in result
        assert "info" in result

        assert len(logger.records) > 0
        assert len(logger.episodes) == 1

        for record in logger.records:
            assert 0 <= record["action"] < environment.num_actions
            assert isinstance(record["reward"], (int, float))
            assert isinstance(record["info"], dict)
    finally:
        environment.close()
