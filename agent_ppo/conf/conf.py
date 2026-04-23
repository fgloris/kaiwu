#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for Gorge Chase PPO V6.
这版重点围绕：卡墙治理、怪物逼近惩罚、score 对齐奖励、宽敞区域偏好、2k 截断 bootstrap。
"""


class Config:
    # hero(13) + m1(6) + m2(6) + t1(5) + t2(5) + b1(4) + b2(4)
    # + move_safety(8) + flash_safety(8) + global(13) = 72
    HERO_DIM = 13
    MONSTER_DIM = 6
    TREASURE_DIM = 5
    BUFF_DIM = 4
    MOVE_SAFETY_DIM = 8
    FLASH_SAFETY_DIM = 8
    GLOBAL_DIM = 13

    SCALAR_FEATURE_SPLIT = [
        HERO_DIM,
        MONSTER_DIM,
        MONSTER_DIM,
        TREASURE_DIM,
        TREASURE_DIM,
        BUFF_DIM,
        BUFF_DIM,
        MOVE_SAFETY_DIM,
        FLASH_SAFETY_DIM,
        GLOBAL_DIM,
    ]
    SCALAR_FEATURE_DIM = sum(SCALAR_FEATURE_SPLIT)

    MAP_CHANNEL = 4
    MAP_PATCH_SIZE = 9
    MAP_PATCH_FLAT_DIM = MAP_CHANNEL * MAP_PATCH_SIZE * MAP_PATCH_SIZE

    FEATURE_LEN = SCALAR_FEATURE_DIM + MAP_PATCH_FLAT_DIM
    DIM_OF_OBSERVATION = FEATURE_LEN

    ACTION_NUM = 16
    VALUE_NUM = 1

    GAMMA = 0.995
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5

    TRAIN_MAPS = [1, 2, 3, 4, 5, 6, 7, 8]
    VAL_MAPS = [9, 10]
    VAL_EVERY_EPISODES = 10

    STALL_WINDOW = 10
    LOOP_WINDOW_LONG = 20
    STALL_DIST_THRESHOLD = 5.0
    BACKTRACK_NEAR_DIST = 14.0
    BACKTRACK_FAR_DIST = 20.0
    FLASH_EVAL_WINDOW = 4
