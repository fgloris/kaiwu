#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

import torch
import torch.nn as nn
import numpy as np

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_lite_v2"
        self.device = device

        input_dim = Config.DIM_OF_OBSERVATION
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        self.backbone = nn.Sequential(
            make_fc_layer(input_dim, 256),
            nn.ReLU(),
            make_fc_layer(256, 512),
            nn.ReLU(),
        )

        self.actor_head = nn.Sequential(
            make_fc_layer(512, 128),
            nn.ReLU(),
            make_fc_layer(128, action_num),
        )

        self.critic_head = nn.Sequential(
            make_fc_layer(512, 128),
            nn.ReLU(),
            make_fc_layer(128, value_num),
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
