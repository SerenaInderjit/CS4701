import torch
import torch.nn.functional as F
from src.cnn import ConvolutionalNeuralNetwork


class PPO:
    def __init__(self, observation_shape, num_actions, learning_rate=3e-4, gamma=0.99, epsilon=0.2 ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_model = ConvolutionalNeuralNetwork(output_dim=num_actions)
        self.value_model = ConvolutionalNeuralNetwork(output_dim=1)
        self.policy_optimizer = torch.optim.Adam( self.policy_model.parameters(), lr=learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=learning_rate)

    def update_policy(self, observations, actions, old_log_probs, advantages):
        
        # New policy distribution.
        logits = self.policy_model(observations)
        distribution = torch.distributions.Categorical(logits=logits)
        new_log_probs = distribution.log_prob(actions)

        # P_theta(X_t = x | S_t = s) / P_theta_k(X_t = x | S_t = s)
        ratio = torch.exp(new_log_probs - old_log_probs)

        clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon,)
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
    