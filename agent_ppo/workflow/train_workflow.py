#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
峡谷追猎 PPO 训练工作流。
"""

import os
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

    # Read user config / 读取用户配置
    train_usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if train_usr_conf is None:
        logger.error("train_usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    val_usr_conf = read_usr_conf("agent_ppo/conf/val_env_conf.toml", logger)
    if val_usr_conf is None:
        logger.error("val_usr_conf is None, please check agent_ppo/conf/val_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        train_usr_conf=train_usr_conf,
        val_usr_conf=val_usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 300:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, train_usr_conf, val_usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.train_usr_conf = train_usr_conf
        self.val_usr_conf = val_usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

        self.val_every_n_episode = 40
        self.val_episode_num = 10

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        执行单局对局并 yield 训练样本。
        """
        while True:
            # Periodically fetch training metrics / 定期获取训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 20:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # Reset env / 重置环境
            env_obs = self.env.reset(self.train_usr_conf)

            # Disaster recovery / 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent & load latest model / 重置 Agent 并加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation / 初始观测处理
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            reward_vec_keys = [
                "r_score_gain_sum",
                "r_survival_gain_sum",
                "r_monster_dist_sum",
                "r_treasure_gain_sum",
                "r_treasure_dist_sum",
                "r_buff_gain_sum",
                "r_buff_dist_sum",
                "r_flash_sum",
                "r_wall_penalty_sum",
            ]
            episode_reward_vec_sum = np.zeros(len(reward_vec_keys), dtype=np.float32)

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Predict action / Agent 推理（随机采样）
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                # Step env / 与环境交互
                env_reward, env_obs = self.env.step(act)

                # Disaster recovery / 容灾处理
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                # Next observation / 处理下一步观测
                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                # Step reward / 每步即时奖励
                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                reward_vector = np.array(_remain_info.get("reward_vector", [0.0] * len(reward_vec_keys)), dtype=np.float32)
                if reward_vector.shape[0] != len(reward_vec_keys):
                    aligned_reward_vector = np.zeros(len(reward_vec_keys), dtype=np.float32)
                    copy_n = min(len(reward_vec_keys), reward_vector.shape[0])
                    aligned_reward_vector[:copy_n] = reward_vector[:copy_n]
                    reward_vector = aligned_reward_vector
                episode_reward_vec_sum += reward_vector
                total_reward += float(reward[0])

                # Terminal reward / 终局奖励
                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)

                    if terminated:
                        final_reward[0] = -10.0
                        result_str = "FAIL"
                    else:
                        final_reward[0] = 10.0 + 0.01 * total_score
                        result_str = "WIN"

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"total_reward:{total_reward:.3f}"
                    )

                # Build sample frame / 构造样本帧
                frame = SampleData(
                    vector_obs = np.array(obs_data.vector_feature, dtype=np.float32),
                    map_obs = np.array(obs_data.map_feature, dtype=np.float32),
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

                # Episode end / 对局结束
                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    # Monitor report / 监控上报
                    now = time.time()
                    train_total_score = float(env_info.get("total_score", 0.0))
                    train_treasure_score = float(env_info.get("treasure_score", 0.0))
                    train_step_score = float(env_info.get("step_score", 0.0))

                    if now - self.last_report_monitor_time >= 20 and self.monitor:
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,

                            "train_total_score": round(train_total_score, 4),
                            "train_treasure_score": round(train_treasure_score, 4),
                            "train_step_score": round(train_step_score, 4),
                        }
                        for i, key in enumerate(reward_vec_keys):
                            monitor_data[key] = round(float(episode_reward_vec_sum[i]), 4)
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if self.episode_cnt % self.val_every_n_episode == 0:
                        self.logger.info(f"[VAL] start validation at episode {self.episode_cnt}")
                        self.run_validation()

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                # Update state / 状态更新
                obs_data = _obs_data
                remain_info = _remain_info

    def run_one_eval_episode(self):
        env_obs = self.env.reset(self.val_usr_conf)

        if handle_disaster_recovery(env_obs, self.logger):
            return None

        self.agent.reset(env_obs)
        self.agent.load_model(id="latest")

        done = False
        step = 0
        while not done:
            act = self.agent.exploit(env_obs)
            env_reward, env_obs = self.env.step(act)

            if handle_disaster_recovery(env_obs, self.logger):
                return None

            terminated = env_obs["terminated"]
            truncated = env_obs["truncated"]
            done = terminated or truncated
            step += 1

        env_info = env_obs["observation"]["env_info"]

        result = {
            "steps": step,
            "total_score": float(env_info.get("total_score", 0.0)),
            "treasure_score": float(env_info.get("treasure_score", 0.0)),
            "step_score": float(env_info.get("step_score", 0.0)),
            "treasures_collected": int(env_info.get("treasures_collected", 0)),
            "collected_buff": int(env_info.get("collected_buff", 0)),
            "flash_count": int(env_info.get("flash_count", 0)),
            "terminated": bool(env_obs["terminated"]),
            "truncated": bool(env_obs["truncated"]),
        }
        return result
    
    def run_validation(self):
        results = []

        for _ in range(self.val_episode_num):
            r = self.run_one_eval_episode()
            if r is not None:
                results.append(r)

        if not results:
            self.logger.info("[VAL] no valid results")
            return

        avg_total_score = np.mean([x["total_score"] for x in results])
        avg_treasure_score = np.mean([x["treasure_score"] for x in results])
        avg_step_score = np.mean([x["step_score"] for x in results])

        avg_steps = np.mean([x["steps"] for x in results])
        avg_treasures = np.mean([x["treasures_collected"] for x in results])
        avg_buffs = np.mean([x["collected_buff"] for x in results])
        avg_flash = np.mean([x["flash_count"] for x in results])
        term_rate = np.mean([1.0 if x["terminated"] else 0.0 for x in results])

        self.logger.info(
            f"[VAL] episodes:{len(results)} "
            f"avg_total_score:{avg_total_score:.2f} "
            f"avg_treasure_score:{avg_treasure_score:.2f} "
            f"avg_step_score:{avg_step_score:.2f} "
            f"avg_steps:{avg_steps:.2f} "
            f"avg_treasures:{avg_treasures:.2f} "
            f"avg_buffs:{avg_buffs:.2f} "
            f"avg_flash:{avg_flash:.2f} "
            f"terminated_rate:{term_rate:.2%}"
        )

        if self.monitor:
            monitor_data = {
                "eval_total_score": round(float(avg_total_score), 4),
                "eval_treasure_score": round(float(avg_treasure_score), 4),
                "eval_step_score": round(float(avg_step_score), 4),
            }
            self.monitor.put_data({os.getpid(): monitor_data})