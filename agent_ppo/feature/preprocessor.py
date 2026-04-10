#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

import numpy as np

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.last_total_score = 0.0
        self.last_treasure_collected = 0
        self.last_collected_buff = 0
        self.last_flash_count = 0
        self.last_nearest_treasure_dist_norm = 0
        self.last_nearest_buff_dist_norm = 0

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        # Hero self features (4D) / 英雄自身特征
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION)

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # Monster features (5D x 2) / 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                # 当 is_in_view = 0 时，仅有 hero_relative_direction 有效
                # {'hero_l2_distance': 0, 'hero_relative_direction': 2, 'monster_id': 14, 'monster_interval': 300, 'pos': {'x': 105, 'z': 56}, 'speed': 1, 'is_in_view': 1}
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                dir_norm = _norm(m.get("hero_relative_direction", 0), 8.0)

                if is_in_view:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)
                    # Euclidean distance / 欧式距离
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                    
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm, dir_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(6, dtype=np.float32))

        # Organ features
        organs = frame_state.get("organs", [])

        treasure_feat = np.array([1.0, 0.0], dtype=np.float32)
        buff_feat = np.array([1.0, 0.0], dtype=np.float32)

        treasure_feat = np.zeros(4, dtype=np.float32)  # 前2个宝箱：每个2维
        buff_feat = np.zeros(4, dtype=np.float32)      # 前2个buff：每个2维

        treasures = []
        buffs = []

        for organ in organs:
            if organ.get("status", 0) != 1:
                continue
            sub_type = organ.get("sub_type", 0)
            if sub_type == 1:
                treasures.append(organ)
            elif sub_type == 2:
                buffs.append(organ)

        treasures.sort(key=lambda x: x.get("hero_l2_distance", MAX_DIST_BUCKET))
        buffs.sort(key=lambda x: x.get("hero_l2_distance", MAX_DIST_BUCKET))

        for i, organ in enumerate(treasures[:2]):
            organ_pos = organ["pos"]
            treasure_feat[i * 4 : i * 4 + 4] = np.array(
                [
                    _norm(organ_pos["x"], MAP_SIZE),
                    _norm(organ_pos["z"], MAP_SIZE),
                    _norm(organ.get("hero_l2_distance", MAX_DIST_BUCKET), MAX_DIST_BUCKET),
                    _norm(organ.get("hero_relative_direction", 0), 8.0),
                ],
                dtype=np.float32,
            )

        for i, organ in enumerate(buffs[:2]):
            buff_feat[i * 4 : i * 4 + 4] = np.array(
                [
                    _norm(organ_pos["x"], MAP_SIZE),
                    _norm(organ_pos["z"], MAP_SIZE),
                    _norm(organ.get("hero_l2_distance", MAX_DIST_BUCKET), MAX_DIST_BUCKET),
                    _norm(organ.get("hero_relative_direction", 0), 8.0),
                ],
                dtype=np.float32,
            )

        # Local map features (16D) / 局部地图特征
        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        # Legal action mask (8D) / 合法动作掩码
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(16, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 16}
                legal_action = [1 if j in valid_set else 0 for j in range(16)]

        if sum(legal_action) == 0:
            legal_action = [1] * 16

        # Progress features (2D) / 进度特征
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])

        step_norm = _norm(self.step_no, self.max_step)
        progress_treasure_collect = _norm(int(hero.get("treasure_collected_count", 0)), 10)
        progress_feat = np.array([step_norm, progress_treasure_collect], dtype=np.float32)

        # Nearest treasure / buff distance shaping
        cur_nearest_treasure_dist_norm = 1.0
        if len(treasures) > 0:
            cur_nearest_treasure_dist_norm = _norm(
                treasures[0].get("hero_l2_distance", MAX_DIST_BUCKET),
                MAX_DIST_BUCKET,
            )

        cur_nearest_buff_dist_norm = 1.0
        if len(buffs) > 0:
            cur_nearest_buff_dist_norm = _norm(
                buffs[0].get("hero_l2_distance", MAX_DIST_BUCKET),
                MAX_DIST_BUCKET,
            )

        treasure_dist_delta = self.last_nearest_treasure_dist_norm - cur_nearest_treasure_dist_norm
        buff_dist_delta = self.last_nearest_buff_dist_norm - cur_nearest_buff_dist_norm

        treasure_dist_reward = 0.0
        buff_dist_reward = 0.0

        # 只有在目标还存在时才做距离塑形
        if len(treasures) > 0:
            treasure_dist_reward = treasure_dist_delta

        if len(buffs) > 0:
            buff_dist_reward = buff_dist_delta

        self.last_nearest_treasure_dist_norm = cur_nearest_treasure_dist_norm
        self.last_nearest_buff_dist_norm = cur_nearest_buff_dist_norm

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                treasure_feat,
                buff_feat,
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        )

        # Step reward / 即时奖励
        survive_reward = 0.01
        dist_shaping = 0.1 * (cur_min_dist_norm - self.last_min_monster_dist_norm)

        self.last_min_monster_dist_norm = cur_min_dist_norm

        # score-based reward / 基于比赛分数增量的奖励
        env_info = env_obs["observation"].get("env_info", {})
        cur_total_score = float(env_info.get("total_score", 0.0))
        score_gain = cur_total_score - self.last_total_score
        self.last_total_score = cur_total_score

        # 3) 宝箱收集奖励
        cur_treasure_collected = int(hero.get("treasure_collected_count", 0))
        treasure_gain = cur_treasure_collected - self.last_treasure_collected
        self.last_treasure_collected = cur_treasure_collected

        treasure_reward = max(0, treasure_gain)

        # 4) buff收集奖励
        cur_collected_buff = int(env_info.get("collected_buff", 0))
        buff_gain = cur_collected_buff - self.last_collected_buff
        self.last_collected_buff = cur_collected_buff

        buff_reward = max(0, buff_gain)

        # 5) 闪现释放奖励
        flash_count = env_info.get("flash_count", 0)
        flash_gain = flash_count - self.last_flash_count
        self.last_flash_count = flash_count

        flash_reward = max(0, flash_gain)

        # final step reward scalar
        reward_scalar = (
            1.0 * score_gain
            + survive_reward
            + 0.1 * dist_shaping
            + 0.5 * treasure_reward
            + 0.3 * buff_reward
            + 0.3 * flash_reward
            + 0.08 * treasure_dist_reward
            + 0.04 * buff_dist_reward
        )
        reward = [reward_scalar]

        return feature, legal_action, reward
