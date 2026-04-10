#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Structured neural network model for Gorge Chase PPO V4.
峡谷追猎 PPO V4 结构化神经网络模型。
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            make_fc_layer(in_dim, hidden_dim),
            nn.ReLU(),
            make_fc_layer(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_v4_structured"
        self.device = device

        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        self.hero_dim = 8
        self.monster_dim = 6
        self.treasure_dim = 5
        self.buff_dim = 4
        self.space_dim = 20
        self.global_dim = 6

        self.hero_encoder = MLPBlock(self.hero_dim, 32, 32)
        self.monster_encoder = MLPBlock(self.monster_dim, 24, 24)
        self.treasure_encoder = MLPBlock(self.treasure_dim, 20, 20)
        self.buff_encoder = MLPBlock(self.buff_dim, 16, 16)
        self.space_encoder = MLPBlock(self.space_dim, 64, 48)
        self.global_encoder = MLPBlock(self.global_dim, 24, 24)

        fused_dim = 32 + 24 * 2 + 20 * 2 + 16 * 2 + 48 + 24
        self.backbone = nn.Sequential(
            make_fc_layer(fused_dim, 256),
            nn.ReLU(),
            make_fc_layer(256, 256),
            nn.ReLU(),
        )

        self.actor_head = nn.Sequential(
            make_fc_layer(256, 128),
            nn.ReLU(),
            make_fc_layer(128, action_num),
        )

        self.critic_head = nn.Sequential(
            make_fc_layer(256, 128),
            nn.ReLU(),
            make_fc_layer(128, value_num),
        )

    def _split_obs(self, obs):
        # [8, 6, 6, 5, 5, 4, 4, 20, 6]
        idx = 0
        hero = obs[:, idx:idx + 8]
        idx += 8
        monster_1 = obs[:, idx:idx + 6]
        idx += 6
        monster_2 = obs[:, idx:idx + 6]
        idx += 6
        treasure_1 = obs[:, idx:idx + 5]
        idx += 5
        treasure_2 = obs[:, idx:idx + 5]
        idx += 5
        buff_1 = obs[:, idx:idx + 4]
        idx += 4
        buff_2 = obs[:, idx:idx + 4]
        idx += 4
        space = obs[:, idx:idx + 20]
        idx += 20
        global_state = obs[:, idx:idx + 6]
        return hero, monster_1, monster_2, treasure_1, treasure_2, buff_1, buff_2, space, global_state

    def forward(self, obs, inference=False):
        hero, monster_1, monster_2, treasure_1, treasure_2, buff_1, buff_2, space, global_state = self._split_obs(obs)

        hero_h = self.hero_encoder(hero)
        monster_1_h = self.monster_encoder(monster_1)
        monster_2_h = self.monster_encoder(monster_2)
        treasure_1_h = self.treasure_encoder(treasure_1)
        treasure_2_h = self.treasure_encoder(treasure_2)
        buff_1_h = self.buff_encoder(buff_1)
        buff_2_h = self.buff_encoder(buff_2)
        space_h = self.space_encoder(space)
        global_h = self.global_encoder(global_state)

        fused = torch.cat([
            hero_h,
            monster_1_h,
            monster_2_h,
            treasure_1_h,
            treasure_2_h,
            buff_1_h,
            buff_2_h,
            space_h,
            global_h,
        ], dim=1)

        hidden = self.backbone(fused)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
