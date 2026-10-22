"""Turns a trained DQN checkpoint into a policy the Runner/evaluate() can use."""
from typing import Optional

import numpy as np
import torch

from src.training.checkpoint import load_checkpoint


class DQNPolicy:
    """Wraps a trained DQN Q-network so it can play through the Runner/evaluate().

    `act` takes a preprocessed observation (4, H, W) uint8 stack, exactly as
    produced by make_training_environment, and returns an action.

    epsilon=0.0 plays the greedy action (argmax over the Q-values). A small
    epsilon (0.05 is the standard DQN evaluation protocol) leaves a little
    exploration in, which is the "sampled" counterpart of PPO's sampled
    evaluation.
    """

    def __init__(
        self,
        model,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        epsilon: float = 0.0,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.epsilon = epsilon
        self.num_actions = self.model.output.out_features

        if checkpoint_path is not None:
            load_checkpoint(checkpoint_path, self.model)

        self.model.eval()

    def act(self, observation) -> int:
        """Select an action from the Q-values for the given observation."""
        if self.epsilon > 0.0 and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        tensor = torch.as_tensor(
            np.asarray(observation), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(tensor).squeeze(0)
        return int(torch.argmax(q_values).item())
