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
