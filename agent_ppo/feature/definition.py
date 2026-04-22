#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Data definitions and GAE computation for Gorge Chase PPO.
支持 terminated / truncated 区分：terminated 终止时 done=1，截断 bootstrap 时 done=0。
"""

import numpy as np
from common_python.utils.common_func import create_cls
from agent_ppo.conf.conf import Config


ObsData = create_cls("ObsData", feature=None, legal_action=None)
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)


def sample_process(list_sample_data):
    """Fill next_value for intermediate transitions and compute GAE."""
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    if list_sample_data:
        _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    gae = np.zeros((Config.VALUE_NUM,), dtype=np.float32)
    gamma = Config.GAMMA
    lamda = Config.LAMDA

    for sample in reversed(list_sample_data):
        done = float(np.array(sample.done).reshape(-1)[0])
        mask = 1.0 - done

        reward = np.array(sample.reward, dtype=np.float32).reshape(-1)[: Config.VALUE_NUM]
        value = np.array(sample.value, dtype=np.float32).reshape(-1)[: Config.VALUE_NUM]
        next_value = np.array(sample.next_value, dtype=np.float32).reshape(-1)[: Config.VALUE_NUM]

        delta = reward + gamma * next_value * mask - value
        gae = delta + gamma * lamda * mask * gae

        sample.advantage = gae.astype(np.float32)
        sample.reward_sum = (gae + value).astype(np.float32)
