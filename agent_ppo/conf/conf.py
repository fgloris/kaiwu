#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for Gorge Chase PPO V6.
在你当前 V5 代码基础上补强：目标记忆、探索/反绕圈、截断 bootstrap、train/val split。
"""


class Config:
    # ---------------- feature dims ----------------
    # hero(10) + monster1(6) + monster2(6) + treasure1(5) + treasure2(5)
    # + buff1(4) + buff2(4) + move_safety(8) + flash_safety(8) + global(11)
    # = 67
    HERO_DIM = 10
    MONSTER_DIM = 6
    TREASURE_DIM = 5
    BUFF_DIM = 4
    MOVE_SAFETY_DIM = 8
    FLASH_SAFETY_DIM = 8
    GLOBAL_DIM = 11

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

    # Local map patch for CNN
    MAP_CHANNEL = 4
    MAP_PATCH_SIZE = 9
    MAP_PATCH_FLAT_DIM = MAP_CHANNEL * MAP_PATCH_SIZE * MAP_PATCH_SIZE

    FEATURE_LEN = SCALAR_FEATURE_DIM + MAP_PATCH_FLAT_DIM
    DIM_OF_OBSERVATION = FEATURE_LEN

    ACTION_NUM = 16
    VALUE_NUM = 1

    # ---------------- PPO ----------------
    GAMMA = 0.995
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5

    # ---------------- workflow ----------------
    TRAIN_MAPS = [1, 2, 3, 4, 5, 6, 7, 8]
    VAL_MAPS = [9, 10]
    VAL_EVERY_EPISODES = 10

    # ---------------- heuristics ----------------
    STALL_WINDOW = 10
    STALL_DIST_THRESHOLD = 5.0
