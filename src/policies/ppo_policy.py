import torch

from src.training.checkpoint import load_checkpoint


class PPOPolicy:
    """Wraps a trained policy CNN so it can play through the Runner/evaluate().

    `act` takes a preprocessed observation (4, H, W) uint8 stack, exactly as
    produced by make_training_environment, and returns an action.

    deterministic=False samples from the policy (what training did);
    deterministic=True takes the most likely action (greedy).
    """

    def __init__(self, model, checkpoint_path=None, device="cpu", deterministic=False):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.deterministic = deterministic

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
            if self.deterministic:
                return int(torch.argmax(logits, dim=-1).item())
            distribution = torch.distributions.Categorical(logits=logits)
            return int(distribution.sample().item())
