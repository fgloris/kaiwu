#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
from agent_ppo.feature.definition import SampleData, sample_process
from common_python.utils.common_func import get_training_metrics, read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(env=env, agent=agent, usr_conf=usr_conf, logger=logger, monitor=monitor)

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")
            obs_data, _ = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)
                env_reward, env_obs = self.env.step(act)

                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = bool(env_obs["terminated"])
                truncated = bool(env_obs["truncated"])
                done = terminated or truncated
                step += 1

                next_obs_data, next_remain_info = self.agent.observation_process(env_obs)
                reward = np.array(next_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])

                final_reward = np.zeros(1, dtype=np.float32)
                env_info = env_obs["observation"].get("env_info", {})
                total_score = float(env_info.get("total_score", 0.0))
                treasures_collected = int(env_info.get("treasures_collected", 0))
                collected_buff = int(env_info.get("collected_buff", 0))
                flash_count = int(env_info.get("flash_count", 0))

                if done:
                    if terminated:
                        final_reward[0] = -1.0
                        result_str = "FAIL"
                    else:
                        final_reward[0] = 1.0
                        result_str = "WIN"

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} result:{result_str} "
                        f"score:{total_score:.1f} treasure:{treasures_collected} buff:{collected_buff} "
                        f"flash:{flash_count} total_reward:{total_reward + float(final_reward[0]):.3f}"
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
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "episode_score": round(total_score, 2),
                            "treasures_collected": treasures_collected,
                            "buff_collected": collected_buff,
                            "flash_count": flash_count,
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = next_obs_data
