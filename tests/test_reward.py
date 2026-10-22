from src.environment.reward import RewardConfig, RewardShaper


def info(x, time=400, flag=False):
    return {"x_pos": x, "time": time, "flag_get": flag}


def test_first_step_has_no_progress_reward():
    shaper = RewardShaper()
    assert shaper(info(40), False) == 0.0


def test_rightward_movement_is_rewarded_and_leftward_penalised():
    shaper = RewardShaper(RewardConfig(time_penalty_weight=0.0))
    shaper(info(40), False)
    assert shaper(info(46), False) == 6.0
    assert shaper(info(43), False) == -3.0


def test_time_penalty():
    shaper = RewardShaper(RewardConfig(time_penalty_weight=0.5))
    shaper(info(40, time=400), False)
    assert shaper(info(40, time=398), False) == -1.0


def test_death_penalty_when_ended_without_flag():
    shaper = RewardShaper(RewardConfig(time_penalty_weight=0.0))
    shaper(info(40), False)
    assert shaper(info(40), True) == -15.0


def test_flag_bonus_on_completion():
    shaper = RewardShaper(RewardConfig(time_penalty_weight=0.0))
    shaper(info(40), False)
    assert shaper(info(40, flag=True), True) == 15.0


def test_reward_is_clipped():
    shaper = RewardShaper(RewardConfig(reward_clip=5.0))
    shaper(info(40), False)
    assert shaper(info(400), False) == 5.0


def test_reset_clears_state():
    shaper = RewardShaper(RewardConfig(time_penalty_weight=0.0))
    shaper(info(500), False)
    shaper.reset()
    assert shaper(info(40), False) == 0.0


def dqn_config(**overrides):
    """Reward config used by the DQN runs: dense part clipped, terminals added after."""
    kwargs = dict(
        x_progress_weight=0.05,
        time_penalty_weight=0.0,
        death_penalty=-1.0,
        flag_bonus=1.0,
        reward_clip=2.0,
        clip_dense=True,
        dense_clip=1.0,
    )
    kwargs.update(overrides)
    return RewardConfig(**kwargs)


def test_dense_clipping_bounds_the_progress_reward():
    shaper = RewardShaper(dqn_config())
    shaper(info(40), False)
    # 360 px of progress * 0.05 = 18, clipped to the dense clip of 1.0
    assert shaper(info(400), False) == 1.0


def test_dense_clipping_leaves_the_flag_bonus_intact():
    shaper = RewardShaper(dqn_config())
    shaper(info(40), False)
    # The dense part is clipped to 1.0, then the flag bonus is added on top.
    assert shaper(info(400, flag=True), True) == 2.0


def test_dense_clipping_leaves_the_death_penalty_intact():
    shaper = RewardShaper(dqn_config())
    shaper(info(40), False)
    assert shaper(info(40), True) == -1.0


def test_clip_dense_is_off_by_default():
    # PPO keeps the original behaviour: the total reward is clipped once, at
    # the end, so the terminal bonus is scaled by the same clip.
    shaper = RewardShaper(RewardConfig(time_penalty_weight=0.0, reward_clip=5.0))
    shaper(info(40), False)
    assert shaper(info(400, flag=True), True) == 5.0
