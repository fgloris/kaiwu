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

# 角度转向量特征
DIR8_TO_VEC = {
    0: (1.0, 0.0),
    1: (1/np.sqrt(2), -1/np.sqrt(2)),
    2: (0.0, -1.0),
    3: (-1/np.sqrt(2), -1/np.sqrt(2)),
    4: (-1.0, 0.0),
    5: (-1/np.sqrt(2), 1/np.sqrt(2)),
    6: (0.0, 1.0),
    7: (1/np.sqrt(2), 1/np.sqrt(2)),
}

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

        self.last_monster_dist_norm_1 = -1.0
        self.last_monster_dist_norm_2 = -1.0

        self.last_total_score = 0.0
        self.last_treasure_collected = 0
        self.last_collected_buff = 0
        self.last_flash_count = 0

        self.last_treasure_dist_norm_1 = 0.0
        self.last_treasure_dist_norm_2 = 0.0
        self.last_buff_dist_norm_1 = 0.0
        self.last_buff_dist_norm_2 = 0.0

        self.prev_hero_pos = None

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

        # 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []

        for i in range(2):
            if i < len(monsters):
                m = monsters[i]

                # 视野外时，hero_relative_direction 和 hero_l2_distance 仍然可用
                is_in_view = float(m.get("is_in_view", 0))
                m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED) if is_in_view else 0.0

                rel_x = 0.0
                rel_z = 0.0

                # 先给默认值：视野外时只保留粗信息
                dir_idx = int(m.get("hero_relative_direction", 0)) % 8
                dir_x, dir_z = DIR8_TO_VEC[dir_idx]

                dist_norm = _norm(m.get("hero_l2_distance", MAX_DIST_BUCKET), MAX_DIST_BUCKET)

                if is_in_view:
                    m_pos = m["pos"]
                    dx = float(m_pos["x"] - hero_pos["x"])
                    dz = float(m_pos["z"] - hero_pos["z"])

                    # 精细相对位置：保留正负号
                    rel_x = float(np.clip(dx / MAP_SIZE, -1.0, 1.0))
                    rel_z = float(np.clip(dz / MAP_SIZE, -1.0, 1.0))

                    raw_dist = np.sqrt(dx * dx + dz * dz)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)

                    # 视野内时，用连续方向覆盖离散方向
                    if raw_dist > 1e-6:
                        dir_x = dx / raw_dist
                        dir_z = dz / raw_dist

                monster_feats.append(
                    np.array(
                        [is_in_view, m_speed_norm, rel_x, rel_z, dist_norm, dir_x, dir_z],
                        dtype=np.float32,
                    )
                )
            else:
                monster_feats.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))

        # buff和宝箱特征
        organs = frame_state.get("organs", [])

        # 前2个宝箱 / buff：每个5维 [rel_x, rel_z, dist_norm, dir_x, dir_z]
        treasure_feat = np.zeros(10, dtype=np.float32)
        buff_feat = np.zeros(10, dtype=np.float32)

        treasures = []
        buffs = []

        for organ in organs:
            if organ.get("status", 0) != 1:
                continue

            sub_type = organ.get("sub_type", 0)
            organ_pos = organ["pos"]

            dx = float(organ_pos["x"] - hero_pos["x"])
            dz = float(organ_pos["z"] - hero_pos["z"])
            raw_dist = np.sqrt(dx * dx + dz * dz)

            # 存起来，方便排序和后续直接用
            organ["raw_dist"] = raw_dist
            organ["_dx"] = dx
            organ["_dz"] = dz

            if sub_type == 1:
                treasures.append(organ)
            elif sub_type == 2:
                buffs.append(organ)

        treasures.sort(key=lambda x: x.get("raw_dist", 1e9))
        buffs.sort(key=lambda x: x.get("raw_dist", 1e9))

        for i, organ in enumerate(treasures[:2]):
            dx = organ["_dx"]
            dz = organ["_dz"]
            raw_dist = organ.get("raw_dist", MAP_SIZE * 1.41)

            rel_x = float(np.clip(dx / MAP_SIZE, -1.0, 1.0))
            rel_z = float(np.clip(dz / MAP_SIZE, -1.0, 1.0))
            dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)

            # 优先用连续方向；太近时退化到离散方向
            if raw_dist > 1e-6:
                dir_x = dx / raw_dist
                dir_z = dz / raw_dist
            else:
                dir_idx = int(organ.get("hero_relative_direction", 0)) % 8
                dir_x, dir_z = DIR8_TO_VEC[dir_idx]

            treasure_feat[i * 5 : i * 5 + 5] = np.array(
                [rel_x, rel_z, dist_norm, dir_x, dir_z],
                dtype=np.float32,
            )

        for i, organ in enumerate(buffs[:2]):
            dx = organ["_dx"]
            dz = organ["_dz"]
            raw_dist = organ.get("raw_dist", MAP_SIZE * 1.41)

            rel_x = float(np.clip(dx / MAP_SIZE, -1.0, 1.0))
            rel_z = float(np.clip(dz / MAP_SIZE, -1.0, 1.0))
            dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)

            if raw_dist > 1e-6:
                dir_x = dx / raw_dist
                dir_z = dz / raw_dist
            else:
                dir_idx = int(organ.get("hero_relative_direction", 0)) % 8
                dir_x, dir_z = DIR8_TO_VEC[dir_idx]

            buff_feat[i * 5 : i * 5 + 5] = np.array(
                [rel_x, rel_z, dist_norm, dir_x, dir_z],
                dtype=np.float32,
            )

        # 局部地图特征 (16D)
        map_feat = np.zeros((21, 21), dtype=np.float32)
        if map_info is not None:
            h = min(21, len(map_info))
            w = min(21, len(map_info[0]))
            for i in range(h):
                for j in range(w):
                    map_feat[i, j] = float(map_info[i][j] != 0)

        # 合法动作掩码 (8D)
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

        # 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        progress_treasure_collect = _norm(int(hero.get("treasure_collected_count", 0)), 10)
        monster_interval = env_info.get("monster_interval", 300)
        assert monster_interval > 0, f"monster insterval < 0! value:{monster_interval}"
        time_before_second_mounster = _norm(max(0, monster_interval - self.step_no), self.max_step)
        has_monster_speedup = 0.0 if env_info.get("monster_speed", 0) <= 1 else 1.0
        progress_feat = np.array([step_norm, progress_treasure_collect, time_before_second_mounster, has_monster_speedup], dtype=np.float32)

        # Concatenate features / 拼接特征
        vector_feat = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                treasure_feat,
                buff_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        )

        reward_feats = {
            "monster_feats": monster_feats,
            "monster_feats_available": len(monsters),
            "treasure_feats": treasure_feat,
            "treasure_feats_available": len(treasures),
            "buff_feats": buff_feat,
            "buff_feats_available": len(buffs),
            "progress_feats": progress_feat,
            "hero_pos": (int(hero_pos["x"]), int(hero_pos["z"])),
            "prev_hero_pos": self.prev_hero_pos,
            "last_action": int(last_action),
        }

        self.prev_hero_pos = (int(hero_pos["x"]), int(hero_pos["z"]))

        return vector_feat, map_feat, reward_feats, legal_action
    
    def calculate_reward(self, env_obs, reward_feats):
        # 基于比赛分数增量的奖励
        env_info = env_obs["observation"].get("env_info", {})
        cur_total_score = float(env_info.get("total_score", 0.0))
        score_gain = cur_total_score - self.last_total_score
        self.last_total_score = cur_total_score
        
        # 怪物 dist shaping
        # 防止首帧的错误 reward

        monster_dist_reward = 0.0
        if self.last_monster_dist_norm_1 >= 0  and self.last_monster_dist_norm_2 >= 0:
            monster_dist_reward = \
                ( reward_feats['monster_feats'][0][4] - self.last_monster_dist_norm_1) + \
                0.2 * (reward_feats['monster_feats'][1][4] - self.last_monster_dist_norm_2)
            
        self.last_monster_dist_norm_1 = reward_feats['monster_feats'][0][4]
        self.last_monster_dist_norm_2 = reward_feats['monster_feats'][1][4]

        # buff和宝箱 distance shaping
        # 靠近奖励但远离不惩罚

        treasure_dist_norm_1 = 0.0
        treasure_dist_norm_2 = 0.0
        if reward_feats['treasure_feats_available'] > 0: 
            treasure_dist_norm_1 = reward_feats['treasure_feats'][2]
            
        if reward_feats['treasure_feats_available'] > 1:
            treasure_dist_norm_2 = reward_feats['treasure_feats'][7]

        treasure_dist_reward = max(0.0, (self.last_treasure_dist_norm_1 - treasure_dist_norm_1) + 
                            0.2 * (self.last_treasure_dist_norm_2 - treasure_dist_norm_2))
        
        self.last_treasure_dist_norm_1 = treasure_dist_norm_1
        self.last_treasure_dist_norm_2 = treasure_dist_norm_2

        buff_dist_norm_1 = 0.0
        buff_dist_norm_2 = 0.0
        if reward_feats['buff_feats_available'] > 0: 
            buff_dist_norm_1 = reward_feats['buff_feats'][2]
            
        if reward_feats['buff_feats_available'] > 1:
            buff_dist_norm_2 = reward_feats['buff_feats'][7]

        buff_dist_reward = max(0.0, (self.last_buff_dist_norm_1 - buff_dist_norm_1) + 
                        0.2 * (self.last_buff_dist_norm_2 - buff_dist_norm_2))
        
        self.last_buff_dist_norm_1 = buff_dist_norm_1
        self.last_buff_dist_norm_2 = buff_dist_norm_2

        # 宝箱收集奖励
        cur_treasure_collected = int(env_obs["observation"]["frame_state"]["heroes"].get("treasure_collected_count", 0))
        treasure_gain = cur_treasure_collected - self.last_treasure_collected
        self.last_treasure_collected = cur_treasure_collected

        treasure_reward = max(0, treasure_gain)

        # buff收集奖励
        cur_collected_buff = int(env_info.get("collected_buff", 0))
        buff_gain = cur_collected_buff - self.last_collected_buff
        self.last_collected_buff = cur_collected_buff

        buff_reward = max(0, buff_gain)

        # 闪现释放奖励
        flash_reward = 0.0
        flash_count = env_info.get("flash_count", 0)
        if (flash_count - self.last_flash_count) > 0:
            flash_reward = 0.5 * monster_dist_reward + 0.5 * treasure_dist_reward + 0.1 * buff_dist_reward
        self.last_flash_count = flash_count

        # 撞墙惩罚
        wall_penalty = 0.0
        prev_hero_pos = reward_feats.get("prev_hero_pos")
        cur_hero_pos = reward_feats.get("hero_pos")

        if prev_hero_pos is not None and cur_hero_pos is not None:
            dx = cur_hero_pos[0] - prev_hero_pos[0]
            dz = cur_hero_pos[1] - prev_hero_pos[1]
            moved = (dx != 0) or (dz != 0)
            if not moved:
                wall_penalty = -0.2

        if reward_feats["progress_feats"][2] > 0: # time before second monseter
            # 早期：鼓励探索和拿宝箱
            treasure_phase_weight = 1.20
            survive_phase_weight = 0.85
        elif reward_feats["progress_feats"][3] == 0: # has monster speedup
            # 中期：逐步平衡
            treasure_phase_weight = 1.00
            survive_phase_weight = 1.00
        else:
            # 后期：怪物加速后，生存优先
            treasure_phase_weight = 0.75
            survive_phase_weight = 1.50

        # final step reward vector
        dist_shaping_norm_weight = 12.8

        reward_vector = [
            0.30 * score_gain,
            survive_phase_weight * 0.01,
            0.35 * dist_shaping_norm_weight * monster_dist_reward,
            0.50 * treasure_phase_weight * treasure_reward,
            0.35 * treasure_phase_weight * dist_shaping_norm_weight * treasure_dist_reward,
            0.20 * buff_reward,
            0.05 * dist_shaping_norm_weight * buff_dist_reward,
            0.25 * flash_reward,
            1.00 * wall_penalty,
        ]

        return reward_vector, sum(reward_vector)
