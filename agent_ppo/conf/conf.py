#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

class Config:

    # Feature dimensions / 特征维度（共40维）
    FEATURES = [
        4,   # 鲁班自身
        6,   # 怪物1
        6,   # 怪物2
        2,   # 宝箱特征（最近一个）
        2,   # buff特征（最近一个）
        16,  # 局部地图
        16,   # 合法动作mask
        2,   # 进度特征
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：8个移动方向
    ACTION_NUM = 16

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1

    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 3e-4
    BETA_START = 0.005
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
    GRAD_CLIP_RANGE = 0.5

    PPO_EPOCH = 4
    MINIBATCH_SIZE = 128
    MAX_BATCH_SIZE = 4096
    ADV_NORM_EPS = 1e-8
