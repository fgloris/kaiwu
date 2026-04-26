#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:
    # 非地图向量特征维度
    VECTOR_FEATURES = [
        4,   # hero
        7,   # monster1
        7,   # monster2
        8,   # ray collision
        8,   # boundery cluster
        4,  # nearest treasure
        8,  # 2 buffs
        17,  # legal action mask
        4,   # progress
        2,   # situation
    ]
    VECTOR_FEATURE_LEN = sum(VECTOR_FEATURES)

    # 局部地图大小：完整 21x21
    MAP_CHANNEL = 3
    MAP_SIZE = 21

    # 兼容 SampleData 里的 obs 维度定义
    # 这里不再表示真实 flatten 后长度，只给 definition 用
    DIM_OF_OBSERVATION = VECTOR_FEATURE_LEN + MAP_CHANNEL * MAP_SIZE * MAP_SIZE

    ACTION_NUM = 17
    THROUGH_MONSTER_FLASH_ACTION = 16
    FLASH_SURVIVAL_BONUS_THRESHOLD = 15
    FLASH_SURVIVAL_BONUS = 4.0
    FLASH_SURVIVAL_PENALTY = -4.0
    VALUE_NUM = 1

    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0002
    LR_SCHEDULE_ENABLE = False
    LR_WARMUP_EPISODES = 800
    LR_COSINE_EPISODES = 1500
    MIN_LEARNING_RATE = 0.00001
    BETA_START = 0.001
    CLIP_PARAM = 0.18
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 1.0
