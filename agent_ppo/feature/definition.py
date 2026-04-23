#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Data definitions and GAE computation for Gorge Chase PPO.
修正 2k 截断 bootstrap：terminated 才 done=1；truncated 可继续 bootstrap。
同时去掉对平台某些环境下不可用的 attached 远端依赖，避免 learner 导入阶段崩溃。
"""

import numpy as np
from common_python.utils.common_func import create_cls
from agent_ppo.conf.conf import Config


def attached(func):
    """兼容装饰器。

    某些平台镜像里 common_func.attached 会在导入时依赖
    kaiwudrl.interface.base_agent_kaiwudrl_remote，导致 learner 直接 import 失败。
    这里保留同名 no-op 装饰器，满足调用习惯，同时避免导入阶段报错。
    """
    return func


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


@attached
def sample_process(list_sample_data):
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value
    if list_sample_data:
        _calc_gae(list_sample_data)
    return list_sample_data


@attached
def SampleData2NumpyData(g_data):
    if hasattr(g_data, "npdata"):
        return g_data.npdata
    return np.array([g_data], dtype=object)


@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)


def _calc_gae(list_sample_data):
    gae = np.zeros((Config.VALUE_NUM,), dtype=np.float32)
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        reward = np.array(sample.reward, dtype=np.float32).reshape(-1)[: Config.VALUE_NUM]
        value = np.array(sample.value, dtype=np.float32).reshape(-1)[: Config.VALUE_NUM]
        next_value = np.array(sample.next_value, dtype=np.float32).reshape(-1)[: Config.VALUE_NUM]
        done = float(np.array(sample.done, dtype=np.float32).reshape(-1)[0])
        mask = 1.0 - done

        delta = reward + gamma * next_value * mask - value
        gae = delta + gamma * lamda * mask * gae
        sample.advantage = gae.astype(np.float32)
        sample.reward_sum = (gae + value).astype(np.float32)
