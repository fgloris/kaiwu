#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO V5.
峡谷追猎 PPO V5 训练工作流。

增强点：
1. 课程学习：从资源更多/压力更低的阶段，逐步过渡到高压泛化阶段。
2. 保持随机地图训练，训练分布更贴近正式评测。
3. 监控中额外保留 episode 基本指标，便于看阶段表现。
"""

import copy
import os
import random
import time

import numpy as np
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    base_usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if base_usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        base_usr_conf=base_usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, base_usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.base_usr_conf = base_usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def _sample_range(self, lo, hi):
        if lo == hi:
            return lo
        return random.randint(lo, hi)

    def _build_curriculum_usr_conf(self):
        """Build per-episode env config.

        按 episode 阶段生成训练环境配置。
        """
        usr_conf = copy.deepcopy(self.base_usr_conf)
        env_conf = usr_conf.setdefault("env_conf", {})

        # Keep map setting from train_env_conf, only ensure random training.
        env_conf["map_random"] = True

        ep = self.episode_cnt
        if ep < 150:
            # warmup_stable
            env_conf["treasure_count"] = self._sample_range(9, 10)
            env_conf["buff_count"] = 2
            env_conf["monster_interval"] = self._sample_range(220, 300)
            env_conf["monster_speedup"] = self._sample_range(360, 460)
            env_conf["max_step"] = 2000
        elif ep < 500:
            # mid_pressure
            env_conf["treasure_count"] = self._sample_range(8, 10)
            env_conf["buff_count"] = self._sample_range(1, 2)
            env_conf["monster_interval"] = self._sample_range(160, 280)
            env_conf["monster_speedup"] = self._sample_range(240, 420)
            env_conf["max_step"] = 2000
        elif ep < 900:
            # late_speedup_survival
            env_conf["treasure_count"] = self._sample_range(7, 10)
            env_conf["buff_count"] = self._sample_range(1, 2)
            env_conf["monster_interval"] = self._sample_range(120, 220)
            env_conf["monster_speedup"] = self._sample_range(180, 320)
            env_conf["max_step"] = 2000
        else:
            # hard_generalization
            env_conf["treasure_count"] = self._sample_range(6, 10)
            env_conf["buff_count"] = self._sample_range(0, 2)
            env_conf["monster_interval"] = self._sample_range(120, 320)
            env_conf["monster_speedup"] = self._sample_range(140, 420)
            env_conf["max_step"] = 2000

        # Keep reasonable cooldown defaults unless user explicitly changed them.
        env_conf["buff_cooldown"] = int(env_conf.get("buff_cooldown", 200))
        env_conf["talent_cooldown"] = int(env_conf.get("talent_cooldown", 100))
        return usr_conf

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            self.episode_cnt += 1
            usr_conf = self._build_curriculum_usr_conf()
            env_obs = self.env.reset(usr_conf)

            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            done = False
            step = 0
            total_reward = 0.0

            self.logger.info(
                f"Episode {self.episode_cnt} start | "
                f"treasure:{usr_conf['env_conf']['treasure_count']} "
                f"buff:{usr_conf['env_conf']['buff_count']} "
                f"monster_interval:{usr_conf['env_conf']['monster_interval']} "
                f"monster_speedup:{usr_conf['env_conf']['monster_speedup']}"
            )

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                _obs_data, _remain_info = self.agent.observation_process(env_obs)
                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])

                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0.0)
                    if terminated:
                        final_reward[0] = -12.0
                        result_str = "FAIL"
                    else:
                        final_reward[0] = 12.0
                        result_str = "WIN"

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{total_reward:.3f}"
                    )

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        env_info = env_obs["observation"]["env_info"]
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "treasures_collected": env_info.get("treasures_collected", 0),
                            "flash_count": env_info.get("flash_count", 0),
                            "total_score": round(float(env_info.get("total_score", 0.0)), 2),
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = _obs_data
                remain_info = _remain_info
