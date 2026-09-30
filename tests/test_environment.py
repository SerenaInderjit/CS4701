from src.environment.mario_environment import MarioEnvironment, make_training_environment

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


def test_training_environment():
    env = make_training_environment(frame_skip=4, num_stack=4)

    try:
        reset_result = env.reset()
        assert reset_result["observation"].shape == (4, 84, 84)
        assert reset_result["observation"].shape == env.observation_shape

        for _ in range(15):  # Mario needs a few frames to start moving
            step_result = env.step(1)  # 'right'
        assert step_result["observation"].shape == (4, 84, 84)
        assert step_result["observation"].dtype.name == "uint8"
        assert "raw_reward" in step_result
        assert isinstance(step_result["reward"], float)
        assert step_result["info"]["x_pos"] > 40
    finally:
        env.close()


def test_frame_skip_advances_multiple_frames():
    skip1, skip4 = MarioEnvironment(frame_skip=1), MarioEnvironment(frame_skip=4)

    try:
        skip1.reset()
        skip4.reset()
        for _ in range(15):  # same number of agent steps; skip4 covers 4x the frames
            x1 = skip1.step(1)["info"]["x_pos"]
            x4 = skip4.step(1)["info"]["x_pos"]
        assert x4 > x1
    finally:
        skip1.close()
        skip4.close()
        