from src.environment.mario_environment import MarioEnvironment

def test_mario_environment():
    env = MarioEnvironment()

    try:
        reset_result = env.reset()

        assert "observation" in reset_result
        assert reset_result["observation"] is not None

        step_result = env.step(0)

        assert "observation" in step_result
        assert step_result["observation"] is not None

        assert "reward" in step_result
        assert isinstance(step_result["reward"], (int, float))

        assert "episode_ended" in step_result
        assert isinstance(step_result["episode_ended"], bool)

        assert "info" in step_result
        assert isinstance(step_result["info"], dict)

        env.render()
    finally:
        env.close()