import torch
import torch.nn as nn

PIXEL_VALUE_MAX = 255.0

class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(self, output_dim: int, conv_channels=(32, 64, 64), hidden_dim=512, frame_size=84):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(4, conv_channels[0], kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d( conv_channels[0], conv_channels[1], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=1),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 4, frame_size, frame_size)
            feature_size = self.features(dummy).flatten(1).shape[1]

        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(feature_size, hidden_dim), nn.ReLU())
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, observation):
        observation = observation.float() / PIXEL_VALUE_MAX
        features = self.features(observation)
        hidden = self.shared(features)
        return self.output(hidden)