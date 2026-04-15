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
        12,  # nearest 2 treasures
        12,  # 2 buffs
        16,  # legal action mask
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

    ACTION_NUM = 16
    VALUE_NUM = 1

    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.002
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
