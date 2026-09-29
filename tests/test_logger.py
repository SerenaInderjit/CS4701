import json

from src.logger import Logger


def test_logger():
    logger = Logger()

    logger.log_step(
        action=1,
        reward=10.0,
        info={"x_pos": 50},
    )

    assert len(logger.records) == 1
    assert logger.records[0]["action"] == 1
    assert logger.records[0]["reward"] == 10.0
    assert logger.records[0]["info"]["x_pos"] == 50


def test_episode_statistics():
    logger = Logger()
    logger.log_step(1, 1.0, {"x_pos": 50})
    logger.log_step(1, 2.0, {"x_pos": 90})
    logger.log_step(1, -1.0, {"x_pos": 80, "flag_get": True})
    stats = logger.end_episode()

    assert stats["length"] == 3
    assert stats["total_reward"] == 2.0
    assert stats["max_x_pos"] == 90
    assert stats["final_x_pos"] == 80
    assert stats["flag_get"] is True

    logger.log_step(1, 0.0, {"x_pos": 10})
    second = logger.end_episode()
    assert second["length"] == 1
    assert second["flag_get"] is False
    assert len(logger.episodes) == 2


def test_end_episode_with_no_steps_returns_none():
    assert Logger().end_episode() is None


def test_summary():
    logger = Logger()
    for x, flag in [(100, False), (300, True)]:
        logger.log_step(1, 1.0, {"x_pos": x, "flag_get": flag})
        logger.end_episode()

    summary = logger.summary()
    assert summary["num_episodes"] == 2
    assert summary["mean_distance"] == 200
    assert summary["best_distance"] == 300
    assert summary["completion_rate"] == 0.5
    assert Logger().summary() == {"num_episodes": 0}


def test_keep_step_records_false_clears_after_episode():
    logger = Logger(keep_step_records=False)
    logger.log_step(1, 1.0, {"x_pos": 10})
    logger.end_episode()
    assert logger.records == []
    assert len(logger.episodes) == 1


def test_save_episodes(tmp_path):
    logger = Logger()
    logger.log_step(1, 1.0, {"x_pos": 10})
    logger.end_episode()
    path = tmp_path / "episodes.json"
    logger.save_episodes(str(path))
    assert json.loads(path.read_text())[0]["max_x_pos"] == 10
