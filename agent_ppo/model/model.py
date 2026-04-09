#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features, gain=1.0):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=gain)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_ppo_v3"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        self.backbone = nn.Sequential(
            make_fc_layer(input_dim, 256, gain=1.0),
            nn.Tanh(),
            make_fc_layer(256, 256, gain=1.0),
            nn.Tanh(),
            make_fc_layer(256, 128, gain=1.0),
            nn.Tanh(),
        )
        self.actor_head = nn.Sequential(
            make_fc_layer(128, 64, gain=1.0),
            nn.Tanh(),
            make_fc_layer(64, action_num, gain=0.01),
        )
        self.critic_head = nn.Sequential(
            make_fc_layer(128, 64, gain=1.0),
            nn.Tanh(),
            make_fc_layer(64, value_num, gain=1.0),
        )

    def forward(self, obs, inference=False):
        hidden = self.backbone(obs)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
