import logging


class Logger:
    def __init__(self):
        self.records = []
        self.logger = logging.getLogger(__name__)

    def log_step(self, action, reward, info):
        self.records.append({
            "action": action,
            "reward": reward,
            "info": info,
        })

    def info(self, message):
        self.logger.info(message)

    def error(self, message, exc_info=False):
        self.logger.error(message, exc_info=exc_info)