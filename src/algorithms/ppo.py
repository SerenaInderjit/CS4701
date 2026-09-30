from typing import Dict, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F

from src.algorithms.cnn import ConvolutionalNeuralNetwork


class PPO:
    """Proximal Policy Optimization agent.

    Two CNNs (policy + value) trained with clipped surrogate objectives on
    minibatches of a rollout, with Generalized Advantage Estimation.

    `act()` returns (action, {"log_prob": ..., "value": ...}) so the Runner
    can store everything the update needs in the rollout buffer.
    """

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        num_actions: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epsilon: float = 0.2,
        epochs: int = 4,
        minibatch_size: int = 64,
        device: str = "cpu",
    ):
        # The CNN is fixed for 4-channel stacked 84x84 input; observation_shape
        # is accepted so callers can pass env.observation_shape uniformly.
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.device = torch.device(device)

        self.policy_model = ConvolutionalNeuralNetwork(output_dim=num_actions).to(self.device)
        self.value_model = ConvolutionalNeuralNetwork(output_dim=1).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=learning_rate)

    def _to_tensor(self, observation) -> torch.Tensor:
        return torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

    def evaluate(self, observation) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """Return (action distribution, state value) for one observation."""
        tensor = self._to_tensor(observation)
        with torch.no_grad():
            logits = self.policy_model(tensor)
            value = self.value_model(tensor).squeeze()
        return torch.distributions.Categorical(logits=logits), value

    def act(self, observation) -> Tuple[int, Dict[str, float]]:
        """Sample an action; extras carry log_prob and value for the rollout buffer."""
        distribution, value = self.evaluate(observation)
        action = distribution.sample()
        extras = {
            "log_prob": distribution.log_prob(action).item(),
            "value": value.item(),
        }
        return int(action.item()), extras

    def value_of(self, observation) -> float:
        """Value of a single observation, used to bootstrap the final GAE step."""
        tensor = self._to_tensor(observation)
        with torch.no_grad():
            return self.value_model(tensor).squeeze().item()

    def compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation over a finished rollout.

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

        `last_value` bootstraps V(s_T) for the final step, so truncated
        rollouts still get credit for the state they end in.
        """
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        observations,
        actions,
        old_log_probs,
        advantages,
        returns,
    ) -> Dict[str, float]:
        """Run `epochs` of minibatch updates over the rollout. Returns mean losses."""
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Advantage normalization: standard PPO practice, keeps the clipped
        # objective well-scaled across updates.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        policy_losses, value_losses = [], []
        for _ in range(self.epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.minibatch_size):
                idx = indices[start:start + self.minibatch_size]
                policy_losses.append(self.update_policy(obs[idx], act[idx], old_log_probs[idx], advantages[idx]))
                value_losses.append(self.update_value(obs[idx], returns[idx]))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
        }

    def update_policy(self, observations, actions, old_log_probs, advantages):
        # New policy distribution.
        logits = self.policy_model(observations)
        distribution = torch.distributions.Categorical(logits=logits)
        new_log_probs = distribution.log_prob(actions)

        # P_theta(X_t = x | S_t = s) / P_theta_k(X_t = x | S_t = s)
        ratio = torch.exp(new_log_probs - old_log_probs)

        clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)

        policy_loss = -objective.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss.item()

    def update_value(self, observations, returns):
        # estimate of E[Yt | St]
        expected_returns = self.value_model(observations).squeeze(-1)
        # l2 regression on expected retuns
        value_loss = F.mse_loss(expected_returns, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        return value_loss.item()

    def state_dicts(self) -> Dict[str, Any]:
        """All model/optimizer state, so a checkpoint can restore the full agent."""
        return {
            "policy_model": self.policy_model.state_dict(),
            "value_model": self.value_model.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
        }

    def load_state_dicts(self, state: Dict[str, Any]) -> None:
        """Restore from state_dicts(); missing keys are skipped (older checkpoints)."""
        for name, obj in (
            ("policy_model", self.policy_model),
            ("value_model", self.value_model),
            ("policy_optimizer", self.policy_optimizer),
            ("value_optimizer", self.value_optimizer),
        ):
            if name in state:
                obj.load_state_dict(state[name])
