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