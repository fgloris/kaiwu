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
        10,   # treasures
        10,   # buffs
        16,  # legal action
        4,   # progress
    ]
    VECTOR_FEATURE_LEN = sum(VECTOR_FEATURES)

    # 局部地图大小：完整 36x36
    MAP_CHANNEL = 2
    MAP_SIZE = 36

    # 兼容 SampleData 里的 obs 维度定义
    # 这里不再表示真实 flatten 后长度，只给 definition 用
    DIM_OF_OBSERVATION = VECTOR_FEATURE_LEN + MAP_CHANNEL * MAP_SIZE * MAP_SIZE

    ACTION_NUM = 16
    VALUE_NUM = 1

    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0005
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
