#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO V4.
峡谷追猎 PPO V4 配置。
"""


class Config:
    # Feature split for structured encoder
    # hero(8) + monster1(6) + monster2(6) + treasure1(5) + treasure2(5)
    # + buff1(4) + buff2(4) + space(20) + global(6) = 64
    FEATURES = [8, 6, 6, 5, 5, 4, 4, 20, 6]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # 16 actions: 8 move + 8 flash
    ACTION_NUM = 16
    VALUE_NUM = 1

    # PPO hyperparameters
    # 根据高分经验，适度提高 gamma，更重视长程存活与后期决策质量
    GAMMA = 0.995
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
