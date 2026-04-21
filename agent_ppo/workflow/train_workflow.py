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

        self.map_rebalance_start_episode = 1200
        self.map_12_probability_after_rebalance = 0.40
        self.early_death_penalty_start_episode = 800
        self.early_death_penalty_upgrade_episode = 1600
        self.early_death_step_score_threshold = 100.0
        self.early_death_step_score_threshold_after_upgrade = 200.0
        self.default_fail_reward = -10.0
        self.early_death_fail_reward = -30.0
        self.lr_stage_1_episode = self.map_rebalance_start_episode
        self.lr_stage_2_episode = self.early_death_penalty_upgrade_episode
        self.lr_stage_1_multiplier = 0.50
        self.lr_stage_2_multiplier = 0.20
        self.base_learning_rates = self._get_current_learning_rates()
        self.current_lr_multiplier = 1.0
        self._logged_stage_1 = False
        self._logged_stage_2 = False
        
    def _append_train_score_window(self, total_score, treasure_score, step_score):
        self.train_score_window.append({
            "total_score": float(total_score),
            "treasure_score": float(treasure_score),
            "step_score": float(step_score),
        })

    def _report_train_monitor_if_needed(self, now, reward_value, step, episode_reward_vec_sum, reward_vec_keys, force=False):
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

    def _get_env_conf(self, conf):
        if isinstance(conf, dict) and isinstance(conf.get("env_conf"), dict):
            return conf["env_conf"]
        return conf if isinstance(conf, dict) else None

    def _get_current_learning_rates(self):
        optimizer = getattr(self.agent, "optimizer", None)
        if optimizer is None:
            return []
        return [float(group.get("lr", 0.0)) for group in optimizer.param_groups]

    def _set_learning_rate_multiplier(self, multiplier):
        optimizer = getattr(self.agent, "optimizer", None)
        if optimizer is None or not self.base_learning_rates:
            return False
        for group, base_lr in zip(optimizer.param_groups, self.base_learning_rates):
            group["lr"] = float(base_lr) * float(multiplier)
        self.current_lr_multiplier = float(multiplier)
        return True

    def _adjust_learning_rate_if_needed(self):
        if self.episode_cnt > self.lr_stage_2_episode:
            target_multiplier = self.lr_stage_2_multiplier
        elif self.episode_cnt > self.lr_stage_1_episode:
            target_multiplier = self.lr_stage_1_multiplier
        else:
            target_multiplier = 1.0

        if abs(float(target_multiplier) - float(self.current_lr_multiplier)) < 1e-8:
            return

        if self._set_learning_rate_multiplier(target_multiplier):
            current_lrs = self._get_current_learning_rates()
            self.logger.warning(
                f"[TRAIN-STAGE] episode:{self.episode_cnt} set learning rate multiplier "
                f"to {target_multiplier:.0%}, lr={current_lrs}"
            )
        else:
            self.logger.warning(
                f"[TRAIN-STAGE] episode:{self.episode_cnt} failed to set learning rate: "
                "agent optimizer is unavailable"
            )

    def _make_train_conf_for_episode(self, episode_cnt):
        train_conf = copy.deepcopy(self.train_usr_conf)
        if episode_cnt <= self.map_rebalance_start_episode:
            return train_conf

        env_conf = self._get_env_conf(train_conf)
        if env_conf is None:
            return train_conf

        configured_maps = list(env_conf.get("map", []))
        if not configured_maps:
            configured_maps = list(range(1, 11))

        map_12 = [m for m in configured_maps if int(m) in (1, 2)]
        other_maps = [m for m in configured_maps if int(m) not in (1, 2)]

        if map_12 and other_maps:
            if random.random() < self.map_12_probability_after_rebalance:
                selected_map = random.choice(map_12)
            else:
                selected_map = random.choice(other_maps)
        else:
            selected_map = random.choice(configured_maps)

        env_conf["map"] = [int(selected_map)]
        env_conf["map_random"] = False
        return train_conf

    def _get_terminated_final_reward(self, step_score):
        threshold = None
        if self.episode_cnt > self.early_death_penalty_upgrade_episode:
            threshold = self.early_death_step_score_threshold_after_upgrade
        elif self.episode_cnt > self.early_death_penalty_start_episode:
            threshold = self.early_death_step_score_threshold

        if threshold is not None and float(step_score) < float(threshold):
            return self.early_death_fail_reward, threshold
        return self.default_fail_reward, threshold

    def _log_stage_change_if_needed(self):
        if (
            not self._logged_stage_1
            and self.episode_cnt > self.map_rebalance_start_episode
        ):
            self.logger.warning(
                f"[TRAIN-STAGE] episode:{self.episode_cnt} entered post-{self.map_rebalance_start_episode} stage: "
                f"map 1/2 probability={self.map_12_probability_after_rebalance:.0%}, "
                f"early death penalty={self.early_death_fail_reward} when step_score < "
                f"{self.early_death_step_score_threshold:.0f}"
            )
            self._logged_stage_1 = True

        if (
            not self._logged_stage_2
            and self.episode_cnt > self.early_death_penalty_upgrade_episode
        ):
            self.logger.warning(
                f"[TRAIN-STAGE] episode:{self.episode_cnt} entered post-{self.early_death_penalty_upgrade_episode} stage: "
                f"early death penalty={self.early_death_fail_reward} when step_score < "
                f"{self.early_death_step_score_threshold_after_upgrade:.0f}"
            )
            self._logged_stage_2 = True

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
            self._adjust_learning_rate_if_needed()
            self._log_stage_change_if_needed()

            train_conf_this_episode = self._make_train_conf_for_episode(self.episode_cnt)

            # Reset env / 重置环境
            env_obs = self.env.reset(train_conf_this_episode)

            # Disaster recovery / 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent & load latest model / 重置 Agent 并加载最新模型
            if hasattr(self.agent, "preprocessor") and hasattr(self.agent.preprocessor, "set_curriculum_episode"):
                self.agent.preprocessor.set_curriculum_episode(self.episode_cnt)
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation / 初始观测处理
            obs_data, remain_info = self.agent.observation_process(env_obs)
            done = False
            step = 0
            total_reward = 0.0
            reward_vec_keys = [
                "r_treasure_score_gain_sum",
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
                    step_score = float(env_info.get("step_score", 0.0))

                    if terminated:
                        fail_reward, early_death_threshold = self._get_terminated_final_reward(step_score)
                        final_reward[0] = fail_reward
                        result_str = "FAIL"
                        if early_death_threshold is not None and step_score < early_death_threshold:
                            result_str = "EARLY_FAIL"
                    else:
                        final_reward[0] = 10.0 + 0.01 * total_score
                        result_str = "WIN"

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"step_score:{step_score:.1f} "
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
