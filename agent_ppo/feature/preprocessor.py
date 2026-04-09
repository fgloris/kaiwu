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
        self.last_step_score = 0.0
        self.last_treasure_score = 0.0
        self.last_treasures_collected = 0
        self.last_collected_buff = 0
        self.last_flash_count = 0
        self.last_min_monster_dist = 5.0
        self.last_min_treasure_dist = 5.0

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
                m = monsters[i] or {}
                mx, mz = _get_pos(m)
                dist_bucket = float(m.get("hero_l2_distance", MAX_DIST_BUCKET))
                dir_id = int(m.get("hero_relative_direction", 0))
                dx_vec, dz_vec = _dir_vec(dir_id)
                if mx == 0.0 and mz == 0.0 and dir_id != 0:
                    dx = dx_vec * (dist_bucket + 1.0) / 6.0
                    dz = dz_vec * (dist_bucket + 1.0) / 6.0
                else:
                    dx = (mx - hero_x) / MAP_SIZE
                    dz = (mz - hero_z) / MAP_SIZE
                min_monster_dist = min(min_monster_dist, dist_bucket)
                monster_feats.append(
                    np.array(
                        [
                            1.0,
                            dx,
                            dz,
                            _norm(m.get("speed", 1), MAX_MONSTER_SPEED),
                            _norm(dist_bucket, MAX_DIST_BUCKET),
                            dir_id / 8.0,
                        ],
                        dtype=np.float32,
                    )
                )
            else:
                monster_feats.append(np.zeros(6, dtype=np.float32))

        organs = frame_state.get("organs", []) or []
        treasures = [o for o in organs if int(o.get("sub_type", 0)) == 1 and int(o.get("status", 0)) == 1]
        buffs = [o for o in organs if int(o.get("sub_type", 0)) == 2 and int(o.get("status", 0)) == 1]
        treasures.sort(key=lambda x: (x.get("hero_l2_distance", 99), x.get("config_id", 0)))
        buffs.sort(key=lambda x: (x.get("hero_l2_distance", 99), x.get("config_id", 0)))

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

        legal_action = self._build_legal_action(legal_act_raw)

        progress_feat = np.array(
            [
                _norm(self.step_no, max(self.max_step, 1)),
                _norm(max(self.max_step - self.step_no, 0), max(self.max_step, 1)),
                _norm(len(treasures), max(env_info.get("total_treasure", 10), 1)),
                _norm(env_info.get("flash_count", 0), 20.0),
            ],
            dtype=np.float32,
        )

        last_action_feat = np.zeros(16, dtype=np.float32)
        if 0 <= int(last_action) < 16:
            last_action_feat[int(last_action)] = 1.0

        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                treasure_feats,
                buff_feats,
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
                last_action_feat,
            ]
        ).astype(np.float32)

        reward = [self._build_reward(env_info, min_monster_dist, min_treasure_dist, last_action)]
        return feature, legal_action, reward

    def _encode_targets(self, entities, hero_pos, topk):
        feats = []
        min_dist = MAX_DIST_BUCKET
        hero_x = hero_pos["x"]
        hero_z = hero_pos["z"]
        for i in range(topk):
            if i < len(entities):
                e = entities[i] or {}
                ex, ez = _get_pos(e)
                dist_bucket = float(e.get("hero_l2_distance", MAX_DIST_BUCKET))
                dir_id = int(e.get("hero_relative_direction", 0))
                dx_vec, dz_vec = _dir_vec(dir_id)
                if ex == 0.0 and ez == 0.0 and dir_id != 0:
                    dx = dx_vec * (dist_bucket + 1.0) / 6.0
                    dz = dz_vec * (dist_bucket + 1.0) / 6.0
                else:
                    dx = (ex - hero_x) / MAP_SIZE
                    dz = (ez - hero_z) / MAP_SIZE
                min_dist = min(min_dist, dist_bucket)
                feats.append(
                    np.array(
                        [
                            1.0,
                            dx,
                            dz,
                            _norm(dist_bucket, MAX_DIST_BUCKET),
                            dir_id / 8.0,
                            float(e.get("status", 1)),
                        ],
                        dtype=np.float32,
                    )
                )
            else:
                feats.append(np.zeros(6, dtype=np.float32))
        return np.concatenate(feats).astype(np.float32), min_dist

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

        dist_gain = min_monster_dist - self.last_min_monster_dist
        treasure_approach = self.last_min_treasure_dist - min_treasure_dist if min_treasure_dist < MAX_DIST_BUCKET else 0.0

        flash_escape_bonus = 0.0
        if flash_gain > 0 and self.last_min_monster_dist <= 1.0:
            flash_escape_bonus = 0.15
        elif flash_gain > 0 and self.last_min_monster_dist >= 3.0:
            flash_escape_bonus = -0.03

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
