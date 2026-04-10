#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

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
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value
    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        done = float(sample.done[0]) if hasattr(sample.done, "__len__") else float(sample.done)
        not_done = 1.0 - done
        delta = sample.reward + gamma * not_done * sample.next_value - sample.value
        gae = delta + gamma * lamda * not_done * gae
        sample.advantage = gae
        sample.reward_sum = gae + sample.value
