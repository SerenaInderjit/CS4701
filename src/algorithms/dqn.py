"""Deep Q-Network agent (Mnih et al., 2015) with a target network."""
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.algorithms.cnn import ConvolutionalNeuralNetwork


def _init_weights(model: ConvolutionalNeuralNetwork) -> None:
    """Orthogonal initialisation, gain sqrt(2) for the hidden layers.

    The output layer keeps gain 1.0: DQN has no policy head to keep near
    uniform, and Q-values are compared against each other, so they should
    start spread out rather than identical.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=float(np.sqrt(2)))
            nn.init.zeros_(module.bias)
    nn.init.orthogonal_(model.output.weight, gain=1.0)
    nn.init.zeros_(model.output.bias)


class DQN:
    """Deep Q-Network agent with a target network and epsilon-greedy exploration.

    The online network `Q(s, a; theta)` maps a stacked observation to one raw
    value per action. These are Q-values, not probabilities, so there is no
    softmax: the greedy action is simply the argmax.

    The target network is a frozen copy of the online one, resynced every
    `target_update_freq` gradient steps. The TD target is computed with it:

        y    = r + gamma * (1 - done) * max_a' Q_target(s', a')
        loss = Huber(Q_online(s, a), y)

    Plain DQN: the action that maximises the target is also the action the
    target network evaluates, so the max operator biases Q-values upwards.
    (Double DQN would pick the action with the online net and evaluate it with
    the target net; that is not implemented here.)

    Exploration is epsilon-greedy. `anneal_epsilon` decays epsilon linearly
    from `epsilon_start` to `epsilon_end` over `epsilon_decay_steps`
    environment steps; the Trainer calls it once per environment step.
    """

    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        num_actions: int,
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50000,
        conv_channels: Tuple[int, int, int] = (64, 128, 128),
        hidden_dim: int = 512,
        device: str = "cpu",
        max_grad_norm: float = 10.0,
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = max(1, target_update_freq)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.epsilon = epsilon_start
        self.epsilon_steps = 0
        self.updates = 0

        frame_size = observation_shape[-1] if len(observation_shape) >= 2 else 84
        self.model = ConvolutionalNeuralNetwork(
            output_dim=num_actions,
            conv_channels=conv_channels,
            hidden_dim=hidden_dim,
            frame_size=frame_size,
        )
        self.target_model = ConvolutionalNeuralNetwork(
            output_dim=num_actions,
            conv_channels=conv_channels,
            hidden_dim=hidden_dim,
            frame_size=frame_size,
        )
        _init_weights(self.model)
        _init_weights(self.target_model)
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)

    # ------------------------------------------------------------------ acting

    def set_learning_rate(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def q_values(self, observation) -> np.ndarray:
        """Q-value of every action for one observation."""
        tensor = torch.as_tensor(
            np.asarray(observation), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.inference_mode():
            return self.model(tensor).squeeze(0).cpu().numpy()

    def act(self, observation) -> int:
        """Epsilon-greedy action: random with probability epsilon, else argmax Q."""
        if self.epsilon > 0.0 and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        return int(np.argmax(self.q_values(observation)))

    def anneal_epsilon(self) -> None:
        """Move epsilon one environment step along its linear schedule."""
        self.epsilon_steps += 1
        fraction = min(1.0, self.epsilon_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    # ---------------------------------------------------------------- learning

    def sync_target(self) -> None:
        """Copy the online weights onto the target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """One gradient step on a mini-batch of transitions.

        `batch` is the dict returned by ReplayBuffer.sample().
        """
        observations = torch.as_tensor(
            batch["observations"], dtype=torch.float32, device=self.device
        )
        next_observations = torch.as_tensor(
            batch["next_observations"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_q = self.target_model(next_observations)
            next_q_max = next_q.max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q_max

        q_values = self.model(observations)
        chosen = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        td_error = targets - chosen
        # Huber loss (beta=1): linear for small errors, quadratic for large
        # ones, so a few big TD errors cannot dominate the gradient.
        loss = F.smooth_l1_loss(chosen, targets, beta=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.target_update_freq == 0:
            self.sync_target()

        return {
            "loss": float(loss.item()),
            "td_error": float(td_error.abs().mean().item()),
            "q_mean": float(chosen.mean().item()),
            "grad_norm": float(grad_norm),
        }

    # ------------------------------------------------------------- checkpointing

    def state_dicts(self) -> Dict[str, Any]:
        """Full agent state: both networks, the optimizer and the counters."""
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "epsilon_steps": self.epsilon_steps,
            "updates": self.updates,
        }

    def load_state_dicts(self, state: Dict[str, Any]) -> None:
        """Restore from state_dicts(); missing keys are skipped (older checkpoints)."""
        if not state:
            return
        if "model" in state:
            self.model.load_state_dict(state["model"])
        if "target_model" in state:
            self.target_model.load_state_dict(state["target_model"])
        elif "model" in state:
            self.target_model.load_state_dict(state["model"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.epsilon = state.get("epsilon", self.epsilon)
        self.epsilon_steps = state.get("epsilon_steps", 0)
        self.updates = state.get("updates", 0)
