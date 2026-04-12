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
        self.model_name = "gorge_chase_cnn_v1"
        self.device = device

        vector_dim = Config.VECTOR_FEATURE_LEN
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        # 向量分支
        self.vector_encoder = nn.Sequential(
            make_fc_layer(vector_dim, 128),
            nn.ReLU(),
            make_fc_layer(128, 128),
            nn.ReLU(),
        )

        # 地图分支: [B,1,21,21]->[B,2,36,36]
        self.map_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 36 -> 18
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 18 -> 9
            nn.Flatten(),
        )

        self.map_fc = nn.Sequential(
            make_fc_layer(64 * 9 * 9, 128),
            nn.ReLU(),
        )

        # 融合层
        self.fusion = nn.Sequential(
            make_fc_layer(128 + 128, 256),
            nn.ReLU(),
            make_fc_layer(256, 256),
            nn.ReLU(),
        )

        self.actor_head = nn.Sequential(
            make_fc_layer(256, 256),
            nn.ReLU(),
            make_fc_layer(256, 128),
            nn.ReLU(),
            make_fc_layer(128, action_num),
        )

        self.critic_head = nn.Sequential(
            make_fc_layer(256, 256),
            nn.ReLU(),
            make_fc_layer(256, 128),
            nn.ReLU(),
            make_fc_layer(128, value_num),
        )

    def forward(self, vector_obs, map_obs, inference=False):
        vector_hidden = self.vector_encoder(vector_obs)
        map_hidden = self.map_encoder(map_obs)
        map_hidden = self.map_fc(map_hidden)

        hidden = torch.cat([vector_hidden, map_hidden], dim=1)
        hidden = self.fusion(hidden)

        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
