#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn

from agent_ppo_strong.conf.conf import Config


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.norm(x + out)


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_ppo_strong"
        self.device = device
        input_dim = Config.DIM_OF_OBSERVATION
        hidden_dim = Config.HIDDEN_DIM

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            ResidualMLPBlock(hidden_dim),
            ResidualMLPBlock(hidden_dim),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, Config.ACTOR_HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(Config.ACTOR_HIDDEN_DIM, Config.ACTION_NUM),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, Config.CRITIC_HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(Config.CRITIC_HIDDEN_DIM, Config.VALUE_NUM),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = 0.01 if m is self.actor[-1] else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, obs, inference=False):
        hidden = self.encoder(obs)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
