#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Structured neural network model for Gorge Chase PPO V6.
保持你当前的 MLP + CNN + 小注意力结构，只把维度与容量做稳健升级。
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


class AttnPool(nn.Module):
    def __init__(self, entity_dim, query_dim, hidden_dim):
        super().__init__()
        self.entity_proj = make_fc_layer(entity_dim, hidden_dim)
        self.query_proj = make_fc_layer(query_dim, hidden_dim)
        self.score = make_fc_layer(hidden_dim, 1)

    def forward(self, entities, query, mask=None):
        e = self.entity_proj(entities)
        q = self.query_proj(query).unsqueeze(1)
        h = torch.tanh(e + q)
        logits = self.score(h).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, -1e9)
        weight = torch.softmax(logits, dim=1)
        pooled = torch.sum(entities * weight.unsqueeze(-1), dim=1)
        return pooled


class MapCNN(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.fc = nn.Sequential(
            make_fc_layer(32 * 3 * 3, 128),
            nn.ReLU(),
            make_fc_layer(128, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc(h)


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_v6_mlp_cnn_memory"
        self.device = device

        self.hero_encoder = MLPBlock(Config.HERO_DIM, 48, 48)
        self.monster_encoder = MLPBlock(Config.MONSTER_DIM, 32, 32)
        self.treasure_encoder = MLPBlock(Config.TREASURE_DIM, 24, 24)
        self.buff_encoder = MLPBlock(Config.BUFF_DIM, 20, 20)
        self.move_encoder = MLPBlock(Config.MOVE_SAFETY_DIM, 32, 32)
        self.flash_encoder = MLPBlock(Config.FLASH_SAFETY_DIM, 32, 32)
        self.global_encoder = MLPBlock(Config.GLOBAL_DIM, 40, 40)

        self.monster_pool = AttnPool(entity_dim=32, query_dim=88, hidden_dim=32)
        self.treasure_pool = AttnPool(entity_dim=24, query_dim=88, hidden_dim=24)
        self.buff_pool = AttnPool(entity_dim=20, query_dim=88, hidden_dim=20)
        self.map_encoder = MapCNN(in_ch=Config.MAP_CHANNEL)

        fused_dim = 48 + 32 + 24 + 20 + 32 + 32 + 40 + 64
        self.backbone = nn.Sequential(
            make_fc_layer(fused_dim, 384),
            nn.ReLU(),
            make_fc_layer(384, 256),
            nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            make_fc_layer(256, 128),
            nn.ReLU(),
            make_fc_layer(128, Config.ACTION_NUM),
        )
        self.critic_head = nn.Sequential(
            make_fc_layer(256, 128),
            nn.ReLU(),
            make_fc_layer(128, Config.VALUE_NUM),
        )

    def _split_obs(self, obs):
        idx = 0
        hero = obs[:, idx:idx + Config.HERO_DIM]
        idx += Config.HERO_DIM
        monster_1 = obs[:, idx:idx + Config.MONSTER_DIM]
        idx += Config.MONSTER_DIM
        monster_2 = obs[:, idx:idx + Config.MONSTER_DIM]
        idx += Config.MONSTER_DIM
        treasure_1 = obs[:, idx:idx + Config.TREASURE_DIM]
        idx += Config.TREASURE_DIM
        treasure_2 = obs[:, idx:idx + Config.TREASURE_DIM]
        idx += Config.TREASURE_DIM
        buff_1 = obs[:, idx:idx + Config.BUFF_DIM]
        idx += Config.BUFF_DIM
        buff_2 = obs[:, idx:idx + Config.BUFF_DIM]
        idx += Config.BUFF_DIM
        move_safety = obs[:, idx:idx + Config.MOVE_SAFETY_DIM]
        idx += Config.MOVE_SAFETY_DIM
        flash_safety = obs[:, idx:idx + Config.FLASH_SAFETY_DIM]
        idx += Config.FLASH_SAFETY_DIM
        global_state = obs[:, idx:idx + Config.GLOBAL_DIM]
        idx += Config.GLOBAL_DIM
        map_flat = obs[:, idx:idx + Config.MAP_PATCH_FLAT_DIM]
        return (
            hero,
            monster_1,
            monster_2,
            treasure_1,
            treasure_2,
            buff_1,
            buff_2,
            move_safety,
            flash_safety,
            global_state,
            map_flat,
        )

    def forward(self, obs, inference=False):
        (
            hero,
            monster_1,
            monster_2,
            treasure_1,
            treasure_2,
            buff_1,
            buff_2,
            move_safety,
            flash_safety,
            global_state,
            map_flat,
        ) = self._split_obs(obs)

        hero_h = self.hero_encoder(hero)
        global_h = self.global_encoder(global_state)
        query = torch.cat([hero_h, global_h], dim=1)

        m1_h = self.monster_encoder(monster_1)
        m2_h = self.monster_encoder(monster_2)
        monsters_h = torch.stack([m1_h, m2_h], dim=1)
        monster_mask = torch.stack([monster_1[:, 0], monster_2[:, 0]], dim=1)
        monster_pool = self.monster_pool(monsters_h, query, monster_mask)

        t1_h = self.treasure_encoder(treasure_1)
        t2_h = self.treasure_encoder(treasure_2)
        treasures_h = torch.stack([t1_h, t2_h], dim=1)
        treasure_mask = torch.stack([treasure_1[:, 0], treasure_2[:, 0]], dim=1)
        treasure_pool = self.treasure_pool(treasures_h, query, treasure_mask)

        b1_h = self.buff_encoder(buff_1)
        b2_h = self.buff_encoder(buff_2)
        buffs_h = torch.stack([b1_h, b2_h], dim=1)
        buff_mask = torch.stack([buff_1[:, 0], buff_2[:, 0]], dim=1)
        buff_pool = self.buff_pool(buffs_h, query, buff_mask)

        move_h = self.move_encoder(move_safety)
        flash_h = self.flash_encoder(flash_safety)

        map_tensor = map_flat.reshape(-1, Config.MAP_CHANNEL, Config.MAP_PATCH_SIZE, Config.MAP_PATCH_SIZE)
        map_h = self.map_encoder(map_tensor)

        fused = torch.cat([
            hero_h,
            monster_pool,
            treasure_pool,
            buff_pool,
            move_h,
            flash_h,
            global_h,
            map_h,
        ], dim=1)
        hidden = self.backbone(fused)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
