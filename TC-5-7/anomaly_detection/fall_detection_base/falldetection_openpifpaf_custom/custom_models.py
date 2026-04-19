import torch
from torch import nn


class FallDetector(nn.Module):

    def __init__(self, feature_dim=32):
        super(FallDetector, self).__init__()

        self.feature_extractor = nn.Sequential(nn.Linear(19*3, feature_dim), nn.LeakyReLU(0.1), nn.Linear(feature_dim, feature_dim))
        self.norm = nn.LayerNorm(feature_dim)
        self.detector = nn.LSTM(feature_dim, feature_dim, 1, batch_first=True,)
        self.classifier = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())

    def forward(self, frames):
        frames = frames.flatten(-2, -1)
        x = self.feature_extractor(frames)
        x, (hn, cn) = self.detector(x)
        x = self.classifier(x[:, -1]).squeeze(-1)
        return x