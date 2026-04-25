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

import copy
import math
import os
import time

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

        self.monitor_report_interval = 2.0
        self.val_every_n_episode = 40
        self.val_episode_num = 10
        self.train_score_window = []
        self.current_learning_rate = float(Config.INIT_LEARNING_RATE_START)
        
    def _append_train_score_window(self, total_score, treasure_score, step_score):
        self.train_score_window.append({
            "total_score": float(total_score),
            "treasure_score": float(treasure_score),
            "step_score": float(step_score),
        })

    def _report_train_monitor_if_needed(self, now, reward_value, step, episode_reward_vec_sum, reward_vec_keys, treasure_mode_ratio, force=False):
        if not self.monitor:
            return
        if not force and now - self.last_report_monitor_time < self.monitor_report_interval:
            return
        if not self.train_score_window:
            return

        total_scores = np.asarray([x["total_score"] for x in self.train_score_window], dtype=np.float32)
        treasure_scores = np.asarray([x["treasure_score"] for x in self.train_score_window], dtype=np.float32)
        step_scores = np.asarray([x["step_score"] for x in self.train_score_window], dtype=np.float32)

        monitor_data = {
            "reward": round(float(reward_value), 4),
            "episode_steps": int(step),
            "episode_cnt": int(self.episode_cnt),
            "treasure_mode_ratio": round(float(treasure_mode_ratio), 4),
            "learning_rate": round(float(self.current_learning_rate), 8),
            "peak_learning_rate": round(float(Config.INIT_LEARNING_RATE_START), 8),
            "min_learning_rate": round(float(Config.MIN_LEARNING_RATE), 8),
            "lr_warmup_episodes": int(Config.LR_WARMUP_EPISODES),
            "lr_cosine_end_episode": int(Config.LR_WARMUP_EPISODES + Config.LR_COSINE_EPISODES),
            "train_avg_total_score": round(float(np.mean(total_scores)), 4),
            "train_avg_treasure_score": round(float(np.mean(treasure_scores)), 4),
            "train_avg_step_score": round(float(np.mean(step_scores)), 4),
            "train_min_total_score": round(float(np.min(total_scores)), 4),
            "train_min_treasure_score": round(float(np.min(treasure_scores)), 4),
            "train_min_step_score": round(float(np.min(step_scores)), 4),
        }
        for i, key in enumerate(reward_vec_keys):
            monitor_data[key] = round(float(episode_reward_vec_sum[i]), 4)

        self.monitor.put_data({os.getpid(): monitor_data})
        self.last_report_monitor_time = now
        self.train_score_window.clear()

    def _calc_learning_rate_by_episode(self, episode_cnt):
        peak_lr = float(Config.INIT_LEARNING_RATE_START)
        min_lr = float(Config.MIN_LEARNING_RATE)
        if not bool(Config.LR_SCHEDULE_ENABLE):
            return peak_lr

        warmup_episodes = max(1, int(Config.LR_WARMUP_EPISODES))
        cosine_episodes = max(1, int(Config.LR_COSINE_EPISODES))
        episode_cnt = max(0, int(episode_cnt))

        if episode_cnt <= warmup_episodes:
            progress = episode_cnt / float(warmup_episodes)
            return min_lr + (peak_lr - min_lr) * progress

        progress = min(1.0, (episode_cnt - warmup_episodes) / float(cosine_episodes))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (peak_lr - min_lr) * cosine

    def _set_learning_rate(self, lr):
        optimizer = getattr(self.agent, "optimizer", None)
        if optimizer is None:
            return False

        lr = float(lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        self.current_learning_rate = lr
        return True

    def _update_learning_rate_by_episode(self):
        lr = self._calc_learning_rate_by_episode(self.episode_cnt)
        ok = self._set_learning_rate(lr)
        if ok and (
            self.episode_cnt <= 3
            or self.episode_cnt == int(Config.LR_WARMUP_EPISODES)
            or self.episode_cnt % 1000 == 0
        ):
            self.logger.info(f"[LR] episode:{self.episode_cnt} learning_rate:{lr:.8f}")
        return lr

    def _make_eval_conf(self, map_ids):
        eval_conf = copy.deepcopy(self.val_usr_conf)
        if isinstance(eval_conf, dict):
            if "env_conf" in eval_conf and isinstance(eval_conf["env_conf"], dict):
                eval_conf["env_conf"]["map"] = list(map_ids)
                eval_conf["env_conf"]["map_random"] = False
            else:
                eval_conf["map"] = list(map_ids)
                eval_conf["map_random"] = False
        return eval_conf

    def _set_learning_rate_scale(self, scale):
        optimizer = getattr(self.agent, "optimizer", None)
        if optimizer is None:
            return False

        lr = float(Config.INIT_LEARNING_RATE_START) * float(scale)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        self.current_learning_rate = lr

        return True

    def _get_life_step_terminal_bonus(self, step):
        step = int(step)
        if step < 100:
            return -20.0
        if step < 200:
            return -10.0
        if step < 300:
            return 0.0
        if step < 400:
            return 10.0
        return 20.0

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

            collector = []
            self.episode_cnt += 1

            # Reset env / 重置环境
            env_obs = self.env.reset(self.train_usr_conf)

            # Disaster recovery / 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent & load latest model / 重置 Agent 并加载最新模型
            if hasattr(self.agent, "preprocessor") and hasattr(self.agent.preprocessor, "set_curriculum_episode"):
                self.agent.preprocessor.set_curriculum_episode(self.episode_cnt)
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")
            self._update_learning_rate_by_episode()

            # Initial observation / 初始观测处理
            obs_data, remain_info = self.agent.observation_process(env_obs)
            done = False
            step = 0
            total_reward = 0.0
            reward_vec_keys = [
                "r_score_gain_sum",
                "r_survival_gain_sum",
                "r_monster_los_break_sum",
                "r_flash_sum",
                "r_wall_penalty_sum",
                "r_abb_penalty_sum",
                "r_danger_penalty_sum",
                "r_treasure_dist_sum",
                "r_buff_dist_sum",
                "r_buff_pick_sum",
                "r_monster_dist_sum",
            ]
            episode_reward_vec_sum = np.zeros(len(reward_vec_keys), dtype=np.float32)
            treasure_mode_steps = 0

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Predict action / Agent 推理（随机采样）
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                if int(getattr(obs_data, "policy_mode", Config.ESCAPE_POLICY_MODE)) == int(Config.TREASURE_POLICY_MODE):
                    treasure_mode_steps += 1
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
                    life_step_bonus = self._get_life_step_terminal_bonus(step)

                    if terminated:
                        final_reward[0] = -10.0
                        result_str = "FAIL"
                    else:
                        final_reward[0] = 10.0 + 0.01 * total_score
                        result_str = "WIN"
                    final_reward[0] += life_step_bonus

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"life_step_bonus:{life_step_bonus:.1f} "
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
                    policy_mode=np.array([obs_data.policy_mode], dtype=np.float32),
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

                    self._append_train_score_window(
                        train_total_score,
                        train_treasure_score,
                        train_step_score,
                    )
                    self._report_train_monitor_if_needed(
                        now=now,
                        reward_value=total_reward + float(final_reward[0]),
                        step=step,
                        episode_reward_vec_sum=episode_reward_vec_sum,
                        reward_vec_keys=reward_vec_keys,
                        treasure_mode_ratio=(float(treasure_mode_steps) / float(max(step, 1))),
                    )

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

    def run_one_eval_episode(self, eval_conf=None):
        if eval_conf is None:
            eval_conf = self.val_usr_conf
        env_obs = self.env.reset(eval_conf)

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
    
    def _run_validation_group(self, map_ids, metric_prefix):
        eval_conf = self._make_eval_conf(map_ids)
        results = []

        for _ in range(self.val_episode_num):
            r = self.run_one_eval_episode(eval_conf=eval_conf)
            if r is not None:
                results.append(r)

        if not results:
            self.logger.info(f"[VAL-{metric_prefix}] no valid results for maps {map_ids}")
            return None

        total_scores = np.asarray([x["total_score"] for x in results], dtype=np.float32)
        treasure_scores = np.asarray([x["treasure_score"] for x in results], dtype=np.float32)
        step_scores = np.asarray([x["step_score"] for x in results], dtype=np.float32)

        avg_total_score = np.mean(total_scores)
        avg_treasure_score = np.mean(treasure_scores)
        avg_step_score = np.mean(step_scores)
        min_total_score = np.min(total_scores)
        min_treasure_score = np.min(treasure_scores)
        min_step_score = np.min(step_scores)

        avg_steps = np.mean([x["steps"] for x in results])
        avg_treasures = np.mean([x["treasures_collected"] for x in results])
        avg_buffs = np.mean([x["collected_buff"] for x in results])
        avg_flash = np.mean([x["flash_count"] for x in results])
        term_rate = np.mean([1.0 if x["terminated"] else 0.0 for x in results])

        self.logger.info(
            f"[VAL-{metric_prefix}] maps:{map_ids} episodes:{len(results)} "
            f"avg_total_score:{avg_total_score:.2f} "
            f"avg_treasure_score:{avg_treasure_score:.2f} "
            f"avg_step_score:{avg_step_score:.2f} "
            f"min_total_score:{min_total_score:.2f} "
            f"min_treasure_score:{min_treasure_score:.2f} "
            f"min_step_score:{min_step_score:.2f} "
            f"avg_steps:{avg_steps:.2f} "
            f"avg_treasures:{avg_treasures:.2f} "
            f"avg_buffs:{avg_buffs:.2f} "
            f"avg_flash:{avg_flash:.2f} "
            f"terminated_rate:{term_rate:.2%}"
        )

        return {
            f"{metric_prefix}_total_score": round(float(avg_total_score), 4),
            f"{metric_prefix}_treasure_score": round(float(avg_treasure_score), 4),
            f"{metric_prefix}_step_score": round(float(avg_step_score), 4),
            f"{metric_prefix}_min_total_score": round(float(min_total_score), 4),
            f"{metric_prefix}_min_treasure_score": round(float(min_treasure_score), 4),
            f"{metric_prefix}_min_step_score": round(float(min_step_score), 4),
        }

    def run_validation(self):
        monitor_data = {}

        eval_12 = self._run_validation_group([1, 2], "eval_12")
        if eval_12 is not None:
            monitor_data.update(eval_12)

        eval_910 = self._run_validation_group([9, 10], "eval_910")
        if eval_910 is not None:
            monitor_data.update(eval_910)

        if self.monitor and monitor_data:
            self.monitor.put_data({os.getpid(): monitor_data})
