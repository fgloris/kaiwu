#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Training workflow for Gorge Chase PPO V6.
结合高分经验补上：train/val split、2k 截断 bootstrap、训练分布更贴近比赛高压阶段。
"""

import copy
import os
import random
import time
from collections import defaultdict

import numpy as np
from agent_ppo.conf.conf import Config
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

    episode_runner = EpisodeRunner(env=env, agent=agent, base_usr_conf=base_usr_conf, logger=logger, monitor=monitor)

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
        self.last_val_stats = None

    def _sample_range(self, lo, hi):
        return lo if lo == hi else random.randint(lo, hi)

    def _build_train_usr_conf(self):
        usr_conf = copy.deepcopy(self.base_usr_conf)
        env_conf = usr_conf.setdefault("env_conf", {})
        env_conf["map"] = list(Config.TRAIN_MAPS)
        env_conf["map_random"] = True

        ep = self.episode_cnt
        if ep < 150:
            env_conf["treasure_count"] = self._sample_range(9, 10)
            env_conf["buff_count"] = 2
            env_conf["monster_interval"] = self._sample_range(220, 300)
            env_conf["monster_speedup"] = self._sample_range(360, 460)
            env_conf["max_step"] = 2000
        elif ep < 500:
            env_conf["treasure_count"] = self._sample_range(8, 10)
            env_conf["buff_count"] = self._sample_range(1, 2)
            env_conf["monster_interval"] = self._sample_range(160, 280)
            env_conf["monster_speedup"] = self._sample_range(240, 420)
            env_conf["max_step"] = 2000
        elif ep < 900:
            env_conf["treasure_count"] = self._sample_range(7, 10)
            env_conf["buff_count"] = self._sample_range(1, 2)
            env_conf["monster_interval"] = self._sample_range(120, 220)
            env_conf["monster_speedup"] = self._sample_range(180, 320)
            env_conf["max_step"] = 2000
        else:
            env_conf["treasure_count"] = self._sample_range(6, 10)
            env_conf["buff_count"] = self._sample_range(0, 2)
            env_conf["monster_interval"] = self._sample_range(120, 320)
            env_conf["monster_speedup"] = self._sample_range(140, 420)
            env_conf["max_step"] = 2000

        env_conf["buff_cooldown"] = int(env_conf.get("buff_cooldown", 200))
        env_conf["talent_cooldown"] = int(env_conf.get("talent_cooldown", 100))
        return usr_conf

    def _build_val_usr_conf(self):
        usr_conf = copy.deepcopy(self.base_usr_conf)
        env_conf = usr_conf.setdefault("env_conf", {})
        env_conf["map"] = list(Config.VAL_MAPS)
        env_conf["map_random"] = True
        env_conf["treasure_count"] = 10
        env_conf["buff_count"] = 2
        env_conf["buff_cooldown"] = int(env_conf.get("buff_cooldown", 200))
        env_conf["talent_cooldown"] = int(env_conf.get("talent_cooldown", 100))
        env_conf["monster_interval"] = 300
        env_conf["monster_speedup"] = 500
        env_conf["max_step"] = 2000
        return usr_conf

    def _make_frame(self, obs_data, act_data, reward, done_flag):
        return SampleData(
            obs=np.array(obs_data.feature, dtype=np.float32),
            legal_action=np.array(obs_data.legal_action, dtype=np.float32),
            act=np.array([act_data.action[0]], dtype=np.float32),
            reward=np.array(reward, dtype=np.float32),
            done=np.array([float(done_flag)], dtype=np.float32),
            reward_sum=np.zeros(1, dtype=np.float32),
            value=np.array(act_data.value, dtype=np.float32).reshape(-1)[:1],
            next_value=np.zeros(1, dtype=np.float32),
            advantage=np.zeros(1, dtype=np.float32),
            prob=np.array(act_data.prob, dtype=np.float32),
        )

    def _play_episode(self, training=True):
        usr_conf = self._build_train_usr_conf() if training else self._build_val_usr_conf()
        env_obs = self.env.reset(usr_conf)
        if handle_disaster_recovery(env_obs, self.logger):
            return None, None

        self.agent.reset(env_obs)
        self.agent.load_model(id="latest")
        obs_data, _ = self.agent.observation_process(env_obs)

        collector = []
        done = False
        step = 0
        total_reward = 0.0
        reward_component_sums = defaultdict(float)

        while not done:
            if training:
                act_data = self.agent.predict([obs_data])[0]
                act = self.agent.action_process(act_data, is_stochastic=True)
            else:
                act_data = self.agent.predict([obs_data])[0]
                act = self.agent.action_process(act_data, is_stochastic=False)

            env_reward, env_obs = self.env.step(act)
            if handle_disaster_recovery(env_obs, self.logger):
                return None, None

            terminated = bool(env_obs["terminated"])
            truncated = bool(env_obs["truncated"])
            step += 1
            done = terminated or truncated

            next_obs_data, next_remain_info = self.agent.observation_process(env_obs)
            reward = np.array(next_remain_info.get("reward", [0.0]), dtype=np.float32)
            total_reward += float(reward[0])
            reward_components = next_remain_info.get("reward_components", {}) or {}
            for key, value in reward_components.items():
                if key == "total_reward":
                    continue
                reward_component_sums[key] += float(value)

            if training:
                collector.append(self._make_frame(obs_data, act_data, reward, done_flag=terminated))

            if done:
                env_info = env_obs["observation"]["env_info"]
                total_score = float(env_info.get("total_score", 0.0))
                step_score = float(env_info.get("step_score", 0.0))
                treasure_score = float(env_info.get("treasure_score", 0.0))
                result_str = "FAIL" if terminated else "WIN"

                # 终局奖励只保留较小常数，避免压过中间过程的 reward 主体。
                final_reward = 0.0
                if terminated:
                    final_reward = -2.0
                else:
                    final_reward = 1.0

                reward_component_sums["terminal_reward"] += float(final_reward)

                if training and collector:
                    collector[-1].reward = collector[-1].reward + np.array([final_reward], dtype=np.float32)
                    if truncated and not terminated and step >= int(usr_conf["env_conf"].get("max_step", 2000)):
                        bootstrap_value = self.agent.estimate_value(next_obs_data)
                        collector[-1].next_value = np.array([bootstrap_value], dtype=np.float32)
                        collector[-1].done = np.array([0.0], dtype=np.float32)
                    else:
                        collector[-1].next_value = np.zeros(1, dtype=np.float32)
                        collector[-1].done = np.array([1.0], dtype=np.float32)

                self.logger.info(
                    f"[GAMEOVER] episode:{self.episode_cnt} mode:{'train' if training else 'val'} steps:{step} "
                    f"result:{result_str} sim_score:{total_score:.1f} total_reward:{total_reward:.3f}"
                )

                stats = {
                    "mode": "train" if training else "val",
                    "steps": step,
                    "reward": round(total_reward + final_reward, 4),
                    "total_score": round(total_score, 2),
                    "step_score": round(step_score, 2),
                    "treasure_score": round(treasure_score, 2),
                    "treasures_collected": int(env_info.get("treasures_collected", 0)),
                    "flash_count": int(env_info.get("flash_count", 0)),
                    "win": 0 if terminated else 1,
                    "reward_components": {k: round(v, 4) for k, v in reward_component_sums.items()},
                }
                return stats, collector

            obs_data = next_obs_data

        return None, collector

    def _report_monitor(self, train_stats):
        if not self.monitor or train_stats is None:
            return
        now = time.time()
        if now - self.last_report_monitor_time < 60:
            return
        data = {
            "train_reward": train_stats["reward"],
            "train_total_score": train_stats["total_score"],
            "train_step_score": train_stats["step_score"],
            "train_treasure_score": train_stats["treasure_score"],
            "train_steps": train_stats["steps"],
            "train_treasures_collected": train_stats["treasures_collected"],
            "train_flash_count": train_stats["flash_count"],
            "train_win_rate": train_stats["win"],
            "episode_cnt": self.episode_cnt,
        }
        for key, value in train_stats.get("reward_components", {}).items():
            data[f"train_{key}"] = value

        if self.last_val_stats is not None:
            data.update({
                "val_total_score": self.last_val_stats["total_score"],
                "val_step_score": self.last_val_stats["step_score"],
                "val_treasure_score": self.last_val_stats["treasure_score"],
                "val_steps": self.last_val_stats["steps"],
                "val_win_rate": self.last_val_stats["win"],
                "generalization_gap_total": round(train_stats["total_score"] - self.last_val_stats["total_score"], 4),
            })
            for key, value in self.last_val_stats.get("reward_components", {}).items():
                data[f"val_{key}"] = value
        self.monitor.put_data({os.getpid(): data})
        self.last_report_monitor_time = now

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            self.episode_cnt += 1
            train_stats, collector = self._play_episode(training=True)
            if train_stats is None:
                continue

            if self.episode_cnt % Config.VAL_EVERY_EPISODES == 0:
                val_stats, _ = self._play_episode(training=False)
                if val_stats is not None:
                    self.last_val_stats = val_stats
                    self.logger.info(
                        f"[VAL] episode:{self.episode_cnt} total:{val_stats['total_score']} "
                        f"step:{val_stats['step_score']} treasure:{val_stats['treasure_score']} win:{val_stats['win']}"
                    )

            self._report_monitor(train_stats)

            if collector:
                collector = sample_process(collector)
                yield collector
