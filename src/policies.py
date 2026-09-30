import random
from typing import Optional
import torch


class RandomPolicy:
    """Uniformly random actions. The Milestone #0 dummy policy, moved out of tests."""

    def __init__(self, num_actions: int = 7, seed: Optional[int] = None):
        # SIMPLE_MOVEMENT has 7 actions
        self.num_actions = num_actions
        self._rng = random.Random(seed)

    def act(self, observation) -> int:
        return self._rng.randrange(self.num_actions)


class ConstantPolicy:
    """Always plays the same action. In SIMPLE_MOVEMENT, 1 = 'right'."""

    def __init__(self, action: int = 1):
        self.action = action

    def act(self, observation) -> int:
        return self.action
    
class PPOPolicy:
    def __init__(self, model, checkpoint_path=None, device="cpu"): 
        self.device = torch.device(device) 
        self.model = model.to(self.device) 
        if checkpoint_path is not None: 
            checkpoint = torch.load(checkpoint_path, map_location=self.device) 
            self.model.load_state_dict(checkpoint) 
            self.model.eval() 

    def act(self, observation) -> int: 
        """Select an action from the policy given an observation.""" 
        observation = torch.as_tensor( observation, dtype=torch.float32, device=self.device, )
        with torch.no_grad(): 
            probabilities = self.model(observation) 
            distribution = torch.distributions.Categorical(probabilities) 
            action = distribution.sample() 
            return int(action.item())