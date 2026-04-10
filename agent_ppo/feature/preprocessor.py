#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math
import numpy as np

MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_DIST_BUCKET = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
MAX_MONSTER_INTERVAL = 2000.0
DIR_TO_VEC = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (math.sqrt(0.5), math.sqrt(0.5)),
    3: (0.0, 1.0),
    4: (-math.sqrt(0.5), math.sqrt(0.5)),
    5: (-1.0, 0.0),
    6: (-math.sqrt(0.5), -math.sqrt(0.5)),
    7: (0.0, -1.0),
    8: (math.sqrt(0.5), -math.sqrt(0.5)),
}


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _get_pos(entity):
    pos = entity.get("pos", {}) if isinstance(entity, dict) else {}
    return float(pos.get("x", 0.0)), float(pos.get("z", 0.0))


def _dir_vec(direction):
    return DIR_TO_VEC.get(int(direction), (0.0, 0.0))


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.last_total_score = 0.0
        self.last_treasure_collected = 0
        self.last_collected_buff = 0
        self.last_flash_count = 0

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})
        map_info = observation.get("map_info", [])
        legal_act_raw = observation.get("legal_act", observation.get("legal_action", []))

        self.step_no = int(observation.get("step_no", env_info.get("step_no", 0)))
        self.max_step = int(env_info.get("max_step", self.max_step))

        hero = frame_state.get("heroes", {}) or {}
        hero_x, hero_z = _get_pos(hero)
        hero_pos = {"x": hero_x, "z": hero_z}

        flash_cd_value = hero.get("flash_cooldown", env_info.get("flash_cooldown", 0))
        buff_remain_value = hero.get("buff_remaining_time", hero.get("speed_up_remaining_time", 0))
        flash_ready = 1.0 if any(bool(x) for x in list(legal_act_raw)[8:16]) else 0.0 if isinstance(legal_act_raw, (list, tuple)) and len(legal_act_raw) >= 16 else 0.0
        hero_feat = np.array(
            [
                _norm(hero_x, MAP_SIZE),
                _norm(hero_z, MAP_SIZE),
                flash_ready,
                _norm(flash_cd_value, MAX_FLASH_CD),
                _norm(buff_remain_value, MAX_BUFF_DURATION),
                _norm(env_info.get("treasures_collected", hero.get("treasure_collected_count", 0)), max(env_info.get("total_treasure", 10), 1)),
                _norm(env_info.get("collected_buff", 0), max(env_info.get("total_buff", 2), 1)),
                _norm(env_info.get("monster_interval", 300), MAX_MONSTER_INTERVAL),
            ],
            dtype=np.float32,
        )

        monsters = frame_state.get("monsters", []) or []
        monster_feats = []
        min_monster_dist = MAX_DIST_BUCKET
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

        nearest_treasure = None
        nearest_buff = None

        for organ in organs:
            if organ.get("status", 0) != 1:
                continue

            sub_type = organ.get("sub_type", 0)
            dist_bucket = organ.get("hero_l2_distance", MAX_DIST_BUCKET)

            if sub_type == 1:
                if nearest_treasure is None or dist_bucket < nearest_treasure["hero_l2_distance"]:
                    nearest_treasure = organ
            elif sub_type == 2:
                if nearest_buff is None or dist_bucket < nearest_buff["hero_l2_distance"]:
                    nearest_buff = organ

        if nearest_treasure is not None:
            treasure_feat = np.array(
                [
                    _norm(nearest_treasure.get("hero_l2_distance", MAX_DIST_BUCKET), MAX_DIST_BUCKET),
                    _norm(nearest_treasure.get("hero_relative_direction", 0), 8.0),
                ],
                dtype=np.float32,
            )

        if nearest_buff is not None:
            buff_feat = np.array(
                [
                    _norm(nearest_buff.get("hero_l2_distance", MAX_DIST_BUCKET), MAX_DIST_BUCKET),
                    _norm(nearest_buff.get("hero_relative_direction", 0), 8.0),
                ],
                dtype=np.float32,
            )

        treasure_feats, min_treasure_dist = self._encode_targets(treasures, hero_pos, topk=4)
        buff_feats, _ = self._encode_targets(buffs, hero_pos, topk=2)

        map_feat = np.zeros(25, dtype=np.float32)
        if isinstance(map_info, list) and len(map_info) > 0 and len(map_info[0]) > 0:
            h = len(map_info)
            w = len(map_info[0])
            center_r = h // 2
            center_c = w // 2
            flat_idx = 0
            for row in range(center_r - 2, center_r + 3):
                for col in range(center_c - 2, center_c + 3):
                    if 0 <= row < h and 0 <= col < w:
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
        #survival_ratio = step_norm * (0.5 + 0.5 * cur_min_dist_norm)
        progress_feat = np.array([step_norm, progress_treasure_collect], dtype=np.float32)

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
                last_action_feat,
            ]
        ).astype(np.float32)

        # Step reward / 即时奖励
        survive_reward = 0.01
        dist_shaping = 0.1 * (cur_min_dist_norm - self.last_min_monster_dist_norm)

    def _build_legal_action(self, legal_act_raw):
        legal_action = [1] * 16
        if isinstance(legal_act_raw, (list, tuple)) and len(legal_act_raw) > 0:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(16, len(legal_act_raw))):
                    legal_action[j] = int(bool(legal_act_raw[j]))
            else:
                valid_set = {int(a) for a in legal_act_raw if 0 <= int(a) < 16}
                legal_action = [1 if j in valid_set else 0 for j in range(16)]
        if sum(legal_action) == 0:
            legal_action = [1] * 16
        return legal_action

    def _build_reward(self, env_info, min_monster_dist, min_treasure_dist, last_action):
        cur_total_score = float(env_info.get("total_score", 0.0))
        cur_step_score = float(env_info.get("step_score", 0.0))
        cur_treasure_score = float(env_info.get("treasure_score", 0.0))
        cur_treasures_collected = int(env_info.get("treasures_collected", 0))
        cur_collected_buff = int(env_info.get("collected_buff", 0))
        cur_flash_count = int(env_info.get("flash_count", 0))

        score_gain = cur_total_score - self.last_total_score
        step_gain = cur_step_score - self.last_step_score
        treasure_score_gain = cur_treasure_score - self.last_treasure_score
        treasure_cnt_gain = cur_treasures_collected - self.last_treasures_collected
        buff_gain = cur_collected_buff - self.last_collected_buff
        flash_gain = cur_flash_count - self.last_flash_count

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
        reward_scalar = 1.0 * score_gain + survive_reward + 0.1 * dist_shaping + 0.5 * treasure_reward + 0.3 * buff_reward + 0.3 * flash_reward
        reward = [reward_scalar]

        reward_scalar = (
            0.01 * score_gain
            + 0.60 * treasure_cnt_gain
            + 0.08 * buff_gain
            + 0.02 * dist_gain
            + 0.015 * treasure_approach
            + flash_escape_bonus
        )

        reward_scalar = float(np.clip(reward_scalar, -2.0, 3.0))

        self.last_total_score = cur_total_score
        self.last_step_score = cur_step_score
        self.last_treasure_score = cur_treasure_score
        self.last_treasures_collected = cur_treasures_collected
        self.last_collected_buff = cur_collected_buff
        self.last_flash_count = cur_flash_count
        self.last_min_monster_dist = min_monster_dist
        self.last_min_treasure_dist = min_treasure_dist
        return reward_scalar
