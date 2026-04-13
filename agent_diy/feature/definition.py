#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
from common_python.utils.common_func import create_cls
from agent_diy.conf.conf import Config

ObsData = create_cls(
    "ObsData",
    vector_feature=None,
    map_feature=None,
    legal_action=None,
)

ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    value=None,
)

SampleData = create_cls(
    "SampleData",
    vector_obs=Config.VECTOR_FEATURE_LEN,
    map_obs=Config.MAP_CHANNEL * Config.MAP_SIZE * Config.MAP_SIZE,
    legal_action=Config.ACTION_DIM,
    act=1,
    reward=1,
    reward_sum=1,
    done=1,
    value=1,
    next_value=1,
    advantage=1,
    prob=Config.ACTION_DIM,
)


def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs):
    return score


def sample_process(list_sample_data):
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value
    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    gae = 0.0
    gamma = getattr(Config, 'GAMMA', 0.95)
    lamda = getattr(Config, 'LAMDA', 0.95)
    for sample in reversed(list_sample_data):
        delta = -sample.value + sample.reward + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        sample.reward_sum = gae + sample.value
