"""
Custom CNN feature extractor for the v8 generalized rendezvous env.

Architecture:
  global_cnn:  Conv2d stack over the (4, coarse, coarse) global view.
  local_cnn:   Shared Conv2d stack over each robot slot's (5, crop, crop) crop.
  per-robot:   Concatenate local CNN features with normalized xy, MLP-embed.
  pool:        Mean-pool per-robot embeddings using the presence mask.
  fuse:        Concat global features + pooled robot features -> features_dim.

The presence mask makes pooling permutation-invariant and robot-count-agnostic,
so a single trained policy handles 2..max_robots without retraining.
"""

import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MultiRobotCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 512,
                 robot_embed_dim: int = 256):
        super().__init__(observation_space, features_dim)

        g_shape = observation_space["global"].shape
        l_shape = observation_space["local"].shape
        self.max_robots = l_shape[0]

        self.global_cnn = nn.Sequential(
            nn.Conv2d(g_shape[0], 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.local_cnn = nn.Sequential(
            nn.Conv2d(l_shape[1], 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

        with torch.no_grad():
            g_dim = self.global_cnn(torch.zeros(1, *g_shape)).shape[1]
            l_dim = self.local_cnn(torch.zeros(1, l_shape[1], l_shape[2], l_shape[3])).shape[1]

        self.robot_embed = nn.Sequential(
            nn.Linear(l_dim + 2, robot_embed_dim), nn.ReLU(),
            nn.Linear(robot_embed_dim, robot_embed_dim), nn.ReLU(),
            nn.Linear(robot_embed_dim, robot_embed_dim), nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(g_dim + robot_embed_dim, features_dim), nn.ReLU(),
            nn.Linear(features_dim, features_dim), nn.ReLU(),
        )

    def forward(self, obs):
        g = obs["global"].float()
        l = obs["local"].float()
        xy = obs["robot_xy"].float()
        mask = obs["presence"].float()

        B, R = l.shape[:2]
        g_feat = self.global_cnn(g)

        l_flat = l.reshape(B * R, *l.shape[2:])
        l_feat = self.local_cnn(l_flat).reshape(B, R, -1)

        per_robot = torch.cat([l_feat, xy], dim=-1)
        per_robot = self.robot_embed(per_robot)

        m_exp = mask.unsqueeze(-1)
        denom = m_exp.sum(dim=1).clamp(min=1.0)
        pooled = (per_robot * m_exp).sum(dim=1) / denom

        return self.fuse(torch.cat([g_feat, pooled], dim=-1))
