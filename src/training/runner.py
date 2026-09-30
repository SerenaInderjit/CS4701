class Runner:
    """Runs a policy in an environment, logging steps and collecting rollouts.

    A policy's `act(observation)` may return either an action, or a tuple
    `(action, extras)` where `extras` is a dict that may contain `log_prob`
    and/or `value` (used by PPO). Extras are stored in the rollout buffer.
    """

    def __init__(self, environment, policy, logger, render=False):
        self.environment = environment
        self.policy = policy
        self.logger = logger
        self.render = render
        self._result = None  # environment state carried across collect_rollout calls

    @staticmethod
    def _unpack_action(output):
        if isinstance(output, tuple):
            action, extras = output
            return action, dict(extras)
        return output, {}

    def run_episode(self, timeout=None):
        """Play one full episode (or until `timeout` steps) and return the last result."""
        result = self.environment.reset()
        steps = 0

        while not result["episode_ended"]:
            observation = result["observation"]
            action, _ = self._unpack_action(self.policy.act(observation))

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

        self.logger.end_episode()
        self._result = None  # this episode is finished; don't resume it in collect_rollout
        return result

    def collect_rollout(self, num_steps, buffer):
        """Collect `num_steps` transitions into `buffer`, across episode boundaries.

        Episodes are reset automatically when they end. The in-progress episode
        is carried over to the next call, so a PPO loop can do
        `collect_rollout -> update -> buffer.clear() -> collect_rollout ...`
        without breaking episodes at rollout boundaries.
        """
        if buffer.size + num_steps > buffer.capacity:
            raise ValueError("Not enough room left in the rollout buffer")

        if self._result is None or self._result["episode_ended"]:
            self._result = self.environment.reset()

        for _ in range(num_steps):
            observation = self._result["observation"]
            action, extras = self._unpack_action(self.policy.act(observation))

            result = self.environment.step(action)
            if self.render:
                self.environment.render()

            self.logger.log_step(
                action=action,
                reward=result["reward"],
                info=result["info"],
            )
            buffer.add(
                observation,
                action,
                result["reward"],
                result["episode_ended"],
                **extras,
            )

            if result["episode_ended"]:
                self.logger.end_episode()
                result = self.environment.reset()
            self._result = result

        buffer.set_last_observation(self._result["observation"])
        return buffer
