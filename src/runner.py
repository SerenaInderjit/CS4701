class Runner:
    def __init__(self, environment, policy, logger, render=False):
        self.environment = environment
        self.policy = policy
        self.logger = logger
        self.render = render

    def run_episode(self, timeout=None):
        result = self.environment.reset()
        steps = 0

        while not result["episode_ended"]:
            observation = result["observation"]
            action = self.policy.act(observation)

            result = self.environment.step(action)
            if self.render:
                self.environment.render()

            self.logger.log_step(
                action=action,
                reward=result["reward"],
                info=result["info"],
            )

            steps += 1

            if timeout is not None and steps >= timeout:
                break

        return result