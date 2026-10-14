from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.algorithms.cnn import ConvolutionalNeuralNetwork


class RunningStats:
    """Running mean/variance (parallel Welford), used to normalize value targets."""

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean, batch_var, n = x.mean(), x.var(), x.size
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        m2 = self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + 1e-8)

    def state_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.mean, self.var, self.count = state["mean"], state["var"], state["count"]


def _init_weights(model: ConvolutionalNeuralNetwork, output_gain: float) -> None:
    """Orthogonal init (gain sqrt(2) for hidden layers). A small output gain on the
    policy head starts it near-uniform; gain 1 on the value head."""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=float(np.sqrt(2)))
            nn.init.zeros_(module.bias)
    nn.init.orthogonal_(model.output.weight, gain=output_gain)
    nn.init.zeros_(model.output.bias)


class PPO:
    """Proximal Policy Optimization agent (Schulman et al., 2017).

    Two CNNs: a policy pi_theta(a|s) and a value function V_phi(s), trained
    with clipped surrogate objectives on minibatches of a rollout.

    The critic is trained on *normalized* returns (running mean/std), but
    `act()` / `value_of()` return values in raw reward units, so GAE and the
    rollout buffer are unaffected. This keeps the value loss well-scaled no
    matter what the reward scale is.

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
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        frame_size = observation_shape[-1] if len(observation_shape) >= 2 else 84
        self.policy_model = ConvolutionalNeuralNetwork(
            output_dim=num_actions, conv_channels=conv_channels, hidden_dim=hidden_dim, frame_size=frame_size
        )
        self.value_model = ConvolutionalNeuralNetwork(
            output_dim=1, conv_channels=conv_channels, hidden_dim=hidden_dim, frame_size=frame_size
        )
        _init_weights(self.policy_model, output_gain=0.01)
        _init_weights(self.value_model, output_gain=1.0)
        self.policy_model.to(self.device)
        self.value_model.to(self.device)

        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=learning_rate, eps=1e-5)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=learning_rate, eps=1e-5)
        self.return_stats = RunningStats()

    def set_learning_rate(self, lr: float) -> None:
        for opt in (self.policy_optimizer, self.value_optimizer):
            for group in opt.param_groups:
                group["lr"] = lr

    def _to_tensor(self, observation) -> torch.Tensor:
        return torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _denormalize(self, normalized_value: float) -> float:
        return normalized_value * self.return_stats.std + self.return_stats.mean

    def evaluate(self, observation) -> Tuple[torch.distributions.Categorical, float]:
        """Return (pi_theta(·|s), V_phi(s) in raw reward units) for one observation."""
        tensor = self._to_tensor(observation)
        with torch.inference_mode():
            logits = self.policy_model(tensor)
            value = self.value_model(tensor).squeeze().item()
        return torch.distributions.Categorical(logits=logits), self._denormalize(value)

    def act(self, observation) -> Tuple[int, Dict[str, float]]:
        """Sample a_t ~ pi_theta(·|s_t); extras carry log pi_theta(a_t|s_t) and V_phi(s_t)."""
        distribution, value = self.evaluate(observation)
        action = distribution.sample()
        extras = {
            "log_prob": distribution.log_prob(action).item(),
            "value": float(value),
        }
        return int(action.item()), extras

    def value_of(self, observation) -> float:
        """V_phi(s) in raw reward units, used to bootstrap the final GAE step."""
        tensor = self._to_tensor(observation)
        with torch.inference_mode():
            value = self.value_model(tensor).squeeze().item()
        return float(self._denormalize(value))

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

        Policy: L^CLIP - ent_coef * entropy.
        Value:  MSE between V_phi(s) and the *normalized* GAE returns.
        """
        obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=self.device)
        act = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(old_log_probs), dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(np.asarray(advantages), dtype=torch.float32, device=self.device)
        returns_raw = np.asarray(returns, dtype=np.float32)

        # Explained variance of the critic on this rollout (in raw units, before
        # advantage normalization): returns - values == raw advantages.
        var_returns = float(returns_raw.var())
        explained_variance = (
            0.0 if var_returns < 1e-8
            else float(1.0 - advantages.var().item() / var_returns)
        )

        # Normalize value targets with running return statistics.
        self.return_stats.update(returns_raw)
        returns = torch.as_tensor(
            (returns_raw - self.return_stats.mean) / self.return_stats.std,
            dtype=torch.float32, device=self.device,
        )

        # Advantage normalization keeps the clipped objective well-scaled.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}
        for _ in range(self.epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.minibatch_size):
                idx = indices[start:start + self.minibatch_size]
                p_loss, ent, kl, clip_frac = self.update_policy(
                    obs[idx], act[idx], old_log_probs[idx], advantages[idx]
                )
                stats["policy_loss"].append(p_loss)
                stats["entropy"].append(ent)
                stats["approx_kl"].append(kl)
                stats["clip_fraction"].append(clip_frac)
                stats["value_loss"].append(self.update_value(obs[idx], returns[idx]))

        result = {k: float(np.mean(v)) for k, v in stats.items()}
        result["explained_variance"] = explained_variance
        return result

    def update_policy(self, observations, actions, old_log_probs, advantages):
        """L^CLIP minus an entropy bonus on one minibatch.

        Returns (surrogate policy loss, mean entropy, approx KL, clip fraction)."""
        logits = self.policy_model(observations)
        distribution = torch.distributions.Categorical(logits=logits)
        new_log_probs = distribution.log_prob(actions)

        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)

        policy_loss = -objective.mean()
        entropy = distribution.entropy().mean()
        loss = policy_loss - self.ent_coef * entropy

        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
            clip_fraction = ((ratio - 1.0).abs() > self.epsilon).float().mean().item()
        return policy_loss.item(), entropy.item(), approx_kl, clip_fraction

    def update_value(self, observations, returns):
        """L^VF on one minibatch: MSE between V_phi(s) and the normalized GAE returns."""
        predicted = self.value_model(observations).squeeze(-1)
        value_loss = F.mse_loss(predicted, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        return value_loss.item()

    def state_dicts(self) -> Dict[str, Any]:
        """All model/optimizer state, so a checkpoint can restore the full agent."""
        return {
            "policy_model": self.policy_model.state_dict(),
            "value_model": self.value_model.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "return_stats": self.return_stats.state_dict(),
        }

    def load_state_dicts(self, state: Dict[str, Any]) -> None:
        """Restore from state_dicts(); missing keys are skipped (older checkpoints)."""
        for name, obj in (
            ("policy_model", self.policy_model),
            ("value_model", self.value_model),
            ("policy_optimizer", self.policy_optimizer),
            ("value_optimizer", self.value_optimizer),
            ("return_stats", self.return_stats),
        ):
            if name in state:
                obj.load_state_dict(state[name])
