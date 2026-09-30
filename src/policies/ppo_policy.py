import torch

from src.training.checkpoint import load_checkpoint


class PPOPolicy:
    """Wraps a trained policy CNN so it can play through the Runner/evaluate().

    `act` takes a preprocessed observation (4, 84, 84) uint8 stack, exactly as
    produced by make_training_environment, and returns a sampled action.
    """

    def __init__(self, model, checkpoint_path=None, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)

        if checkpoint_path is not None:
            load_checkpoint(
                checkpoint_path,
                self.model
            )

        self.model.eval()

    def act(self, observation) -> int:
        """Select an action from the policy given an observation."""
        observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(observation)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            return int(action.item())
