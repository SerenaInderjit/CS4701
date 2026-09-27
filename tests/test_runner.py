from src.mario_environment import MarioEnvironment
from src.runner import Runner
from src.logger import Logger
from random import randrange


class RandomPolicy:
    def act(self, observation):
        return randrange(5)


def test_runner():
    environment = MarioEnvironment()
    policy = RandomPolicy()
    logger = Logger()

    runner = Runner(environment, policy, logger, render=True)

    try:
        result = runner.run_episode(timeout=100)

        assert "observation" in result
        assert "reward" in result
        assert "episode_ended" in result
        assert "info" in result

        assert len(logger.records) > 0

        for record in logger.records:
            assert 0 <= record["action"] < 5
            assert isinstance(record["reward"], (int, float))
            assert isinstance(record["info"], dict)
    finally:
        environment.close()