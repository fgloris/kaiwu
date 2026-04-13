#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import time
import numpy as np
from tools.train_env_conf_validate import read_usr_conf
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env, agent = envs[0], agents[0]

    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    last_save_model_time = time.time()
    last_metrics_time = 0.0
    episode_cnt = 0
    report_interval = 2.0
    last_report_time = 0.0

    planner_stat_window = []
    score_window = []

    while True:
        now = time.time()
        if now - last_metrics_time >= 20.0:
            training_metrics = get_training_metrics()
            last_metrics_time = now
            if training_metrics is not None and logger is not None:
                logger.info(f"training_metrics is {training_metrics}")

        env_obs = env.reset(usr_conf)
        if handle_disaster_recovery(env_obs, logger):
            continue

        agent.reset(env_obs)
        agent.load_model(id="latest")
        agent.save_model(id="init")
        obs_data, remain_info = agent.observation_process(env_obs)

        episode_cnt += 1
        step = 0
        done = False
        episode_return = 0.0
        fallback_count = 0

        if logger is not None:
            logger.info(f"Episode {episode_cnt} start")

        while not done:
            act_data = agent.predict([obs_data])[0]
            act = agent.action_process(act_data)
            env_reward, next_env_obs = env.step(act)

            if handle_disaster_recovery(next_env_obs, logger):
                done = True
                break

            step += 1
            terminated = bool(next_env_obs.get("terminated", False))
            truncated = bool(next_env_obs.get("truncated", False))
            done = terminated or truncated

            next_obs_data, next_remain_info = agent.observation_process(next_env_obs)
            reward_scalar = _extract_reward_scalar(env_reward, next_env_obs)
            episode_return += reward_scalar

            plan_info = next_remain_info.get("plan_info", {}) or {}
            fallback_count += int(plan_info.get("used_fallback", 0.0) > 0.5)
            planner_stat_window.append({
                "target_score": float(plan_info.get("target_score", 0.0)),
                "path_len": float(plan_info.get("path_len", 0.0)),
                "frontier_count": float(plan_info.get("frontier_count", 0.0)),
                "cluster_count": float(plan_info.get("cluster_count", 0.0)),
                "used_fallback": float(plan_info.get("used_fallback", 0.0)),
            })

            if monitor is not None:
                now = time.time()
                if now - last_report_time >= report_interval and planner_stat_window:
                    env_info = next_env_obs.get("observation", {}).get("env_info", {})
                    monitor_data = _build_monitor_data(
                        episode_cnt=episode_cnt,
                        step=step,
                        episode_return=episode_return,
                        env_info=env_info,
                        planner_stat_window=planner_stat_window,
                        score_window=score_window,
                    )
                    monitor.put_data({os.getpid(): monitor_data})
                    planner_stat_window.clear()
                    score_window.clear()
                    last_report_time = now

            env_info = next_env_obs.get("observation", {}).get("env_info", {})
            score_window.append({
                "total_score": float(env_info.get("total_score", 0.0)),
                "treasure_score": float(env_info.get("treasure_score", 0.0)),
                "step_score": float(env_info.get("step_score", 0.0)),
            })

            obs_data, remain_info = next_obs_data, next_remain_info

        env_info = env_obs.get("observation", {}).get("env_info", {}) if not done else next_env_obs.get("observation", {}).get("env_info", {})
        if logger is not None:
            logger.info(
                f"[GAMEOVER] episode:{episode_cnt} steps:{step} total_score:{float(env_info.get('total_score', 0.0)):.1f} "
                f"treasure_score:{float(env_info.get('treasure_score', 0.0)):.1f} "
                f"step_score:{float(env_info.get('step_score', 0.0)):.1f} "
                f"fallback_rate:{(fallback_count / max(step,1)):.3f}"
            )

        if monitor is not None and (planner_stat_window or score_window):
            monitor_data = _build_monitor_data(
                episode_cnt=episode_cnt,
                step=step,
                episode_return=episode_return,
                env_info=env_info,
                planner_stat_window=planner_stat_window,
                score_window=score_window,
                episode_done=1.0,
            )
            monitor.put_data({os.getpid(): monitor_data})
            planner_stat_window.clear()
            score_window.clear()
            last_report_time = time.time()

        now = time.time()
        if now - last_save_model_time >= 300.0:
            agent.save_model()
            last_save_model_time = now


def _build_monitor_data(episode_cnt, step, episode_return, env_info, planner_stat_window, score_window, episode_done=0.0):
    if planner_stat_window:
        target_score = float(np.mean([x["target_score"] for x in planner_stat_window]))
        path_len = float(np.mean([x["path_len"] for x in planner_stat_window]))
        frontier_count = float(np.mean([x["frontier_count"] for x in planner_stat_window]))
        cluster_count = float(np.mean([x["cluster_count"] for x in planner_stat_window]))
        fallback_rate = float(np.mean([x["used_fallback"] for x in planner_stat_window]))
    else:
        target_score = path_len = frontier_count = cluster_count = fallback_rate = 0.0

    if score_window:
        avg_total = float(np.mean([x["total_score"] for x in score_window]))
        avg_treasure = float(np.mean([x["treasure_score"] for x in score_window]))
        avg_step = float(np.mean([x["step_score"] for x in score_window]))
    else:
        avg_total = float(env_info.get("total_score", 0.0))
        avg_treasure = float(env_info.get("treasure_score", 0.0))
        avg_step = float(env_info.get("step_score", 0.0))

    return {
        "reward": round(float(episode_return), 4),
        "episode_steps": int(step),
        "episode_cnt": int(episode_cnt),
        "episode_done": float(episode_done),
        "total_score": round(float(env_info.get("total_score", 0.0)), 4),
        "treasure_score": round(float(env_info.get("treasure_score", 0.0)), 4),
        "step_score": round(float(env_info.get("step_score", 0.0)), 4),
        "avg_total_score": round(avg_total, 4),
        "avg_treasure_score": round(avg_treasure, 4),
        "avg_step_score": round(avg_step, 4),
        "target_score": round(target_score, 4),
        "path_len": round(path_len, 4),
        "frontier_count": round(frontier_count, 4),
        "cluster_count": round(cluster_count, 4),
        "fallback_rate": round(fallback_rate, 4),
    }

def _extract_reward_scalar(env_reward, next_env_obs):
    if env_reward is None:
        return 0.0

    if isinstance(env_reward, (int, float, np.integer, np.floating)):
        return float(env_reward)

    if isinstance(env_reward, dict):
        for key in ("reward", "total_reward", "score", "total_score", "step_score", "value"):
            val = env_reward.get(key, None)
            if isinstance(val, (int, float, np.integer, np.floating)):
                return float(val)

        total = 0.0
        found = False
        for val in env_reward.values():
            if isinstance(val, (int, float, np.integer, np.floating)):
                total += float(val)
                found = True
        if found:
            return total

    env_info = (next_env_obs or {}).get("observation", {}).get("env_info", {})
    for key in ("step_score", "total_score", "score"):
        val = env_info.get(key, None)
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)

    return 0.0