#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

class Config:
    # Feature layout
    # hero_self: 8
    # monsters: 2 x 6 = 12
    # treasures: 4 x 6 = 24
    # buffs: 2 x 6 = 12
    # map_local: 25
    # legal_action: 16
    # progress: 4
    # last_action_one_hot: 16
    FEATURES = [8, 12, 24, 12, 25, 16, 4, 16]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    ACTION_NUM = 16
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
