from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.algorithms.cnn import ConvolutionalNeuralNetwork


class PPO:
    """Proximal Policy Optimization agent (Schulman et al., 2017).

    Two CNNs: a policy pi_theta(a|s) and a value function V_phi(s), trained
    with clipped surrogate objectives on minibatches of a rollout.

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
        conv_channels: Tuple[int, int, int] = (64, 128, 128),
        hidden_dim: int = 1024,
    ):
        # The CNN is fixed for 4-channel stacked input; observation_shape
        # is accepted so callers can pass env.observation_shape uniformly.
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.device = torch.device(device)

        frame_size = observation_shape[-1] if len(observation_shape) >= 2 else 84
        self.policy_model = ConvolutionalNeuralNetwork(
            output_dim=num_actions, conv_channels=conv_channels, hidden_dim=hidden_dim, frame_size=frame_size
        ).to(self.device)
        self.value_model = ConvolutionalNeuralNetwork(
            output_dim=1, conv_channels=conv_channels, hidden_dim=hidden_dim, frame_size=frame_size
        ).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=learning_rate)

    def _to_tensor(self, observation) -> torch.Tensor:
        return torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

    def evaluate(self, observation) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """Return (pi_theta(·|s), V_phi(s)) for one observation."""
        tensor = self._to_tensor(observation)
        with torch.inference_mode():
            logits = self.policy_model(tensor)
            value = self.value_model(tensor).squeeze()
        return torch.distributions.Categorical(logits=logits), value

    def act(self, observation) -> Tuple[int, Dict[str, float]]:
        """Sample a_t ~ pi_theta(·|s_t); extras carry log pi_theta(a_t|s_t) and V_phi(s_t)."""
        distribution, value = self.evaluate(observation)
        action = distribution.sample()
        extras = {
            "log_prob": distribution.log_prob(action).item(),
            "value": value.item(),
        }
        return int(action.item()), extras

    def value_of(self, observation) -> float:
        """V_phi(s), used to bootstrap the final GAE step."""
        tensor = self._to_tensor(observation)
        with torch.inference_mode():
            return self.value_model(tensor).squeeze().item()

    def compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation over a finished rollout.

        delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

        returns_t = A_t + V(s_t)  (regression target for V_phi)

        `last_value` is V(s_T) for the final step, so a rollout truncated at
        an episode boundary bootstraps from the state it ends in.
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
        """One PPO update: `epochs` passes over the rollout in minibatches.

        For each minibatch B:
            L^CLIP = -mean(min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t))
            L^VF   = mean((V_phi(s_t) - R_t)^2)
        where r_t = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t).
        """
        # from_numpy shares memory with the numpy array (no copy) when possible
        obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=self.device)
        act = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(old_log_probs), dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(np.asarray(advantages), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(np.asarray(returns), dtype=torch.float32, device=self.device)

        # Advantage normalization keeps the clipped objective well-scaled.
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
        """L^CLIP on one minibatch: clipped surrogate policy gradient."""
        logits = self.policy_model(observations)
        distribution = torch.distributions.Categorical(logits=logits)
        new_log_probs = distribution.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)

        policy_loss = -objective.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss.item()

    def update_value(self, observations, returns):
        """L^VF on one minibatch: MSE between V_phi(s) and the GAE returns."""
        predicted = self.value_model(observations).squeeze(-1)
        value_loss = F.mse_loss(predicted, returns)
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
