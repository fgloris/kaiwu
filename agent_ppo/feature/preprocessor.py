#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO V5.
峡谷追猎 PPO V5 特征预处理与奖励设计。

V5 重点：
1. 标量决策特征交给 MLP，局部地图交给 CNN。
2. 重点增强“前两个宝箱 / 前两个 buff / 危险-机会切换 / 闪现时机”。
3. 奖励分前后期：前期更鼓励资源推进，后期更鼓励空间与保命。
4. 特征不堆原始字段，尽量回答：
   - 我现在危险不危险
   - 我现在有没有机会拿资源
   - 走和闪哪一个更值
   - 我周围是不是容易被卡死
"""

from collections import deque

import numpy as np

from agent_ppo.conf.conf import Config

MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
MAX_DIST_BUCKET = 5.0
MAX_TREASURE_COUNT = 10.0
MAX_BUFF_COUNT = 2.0

ACTION_DIRS = [
    (1, 0),    # 0 右
    (1, -1),   # 1 右上
    (0, -1),   # 2 上
    (-1, -1),  # 3 左上
    (-1, 0),   # 4 左
    (-1, 1),   # 5 左下
    (0, 1),    # 6 下
    (1, 1),    # 7 右下
]

REL_DIRS = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (0.7071, -0.7071),
    3: (0.0, -1.0),
    4: (-0.7071, -0.7071),
    5: (-1.0, 0.0),
    6: (-0.7071, 0.7071),
    7: (0.0, 1.0),
    8: (0.7071, 0.7071),
}

DIST_BUCKET_CENTER = {
    0: 15.0,
    1: 45.0,
    2: 75.0,
    3: 105.0,
    4: 135.0,
    5: 165.0,
}


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _clip_signed(v, denom):
    if abs(denom) < 1e-6:
        return 0.0
    return float(np.clip(v / denom, -1.0, 1.0))


def _l2(x1, z1, x2, z2):
    return float(np.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2))


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000

        self.last_min_monster_dist_norm = None
        self.last_second_monster_dist_norm = None
        self.last_best_treasure_dist = None
        self.last_best_treasure_priority = 0.0
        self.last_total_score = 0.0
        self.last_step_score = 0.0
        self.last_treasure_score = 0.0
        self.last_treasure_collected = 0
        self.last_collected_buff = 0
        self.last_flash_count = 0
        self.last_pos = None
        self.last_corridor_score = 0.0
        self.last_action = -1
        self.recent_positions = deque(maxlen=12)

    def _get_hero(self, frame_state):
        hero = frame_state.get("heroes", {})
        if isinstance(hero, list):
            return hero[0] if hero else {}
        return hero if isinstance(hero, dict) else {}

    def _get_map_center(self, map_info):
        if map_info is None or len(map_info) == 0 or len(map_info[0]) == 0:
            return 0, 0
        return len(map_info) // 2, len(map_info[0]) // 2

    def _get_local_cell(self, map_info, dx, dz):
        if map_info is None or len(map_info) == 0 or len(map_info[0]) == 0:
            return 0.0
        cr, cc = self._get_map_center(map_info)
        r = cr + dz
        c = cc + dx
        if r < 0 or r >= len(map_info) or c < 0 or c >= len(map_info[0]):
            return 0.0
        return 1.0 if map_info[r][c] != 0 else 0.0

    def _is_passable_step(self, map_info, dx, dz):
        target_ok = self._get_local_cell(map_info, dx, dz) > 0.5
        if not target_ok:
            return False
        if dx != 0 and dz != 0:
            edge1_ok = self._get_local_cell(map_info, dx, 0) > 0.5
            edge2_ok = self._get_local_cell(map_info, 0, dz) > 0.5
            return edge1_ok or edge2_ok
        return True

    def _open_length(self, map_info, dx, dz, max_len=6):
        if map_info is None or len(map_info) == 0 or len(map_info[0]) == 0:
            return 0.0
        cr, cc = self._get_map_center(map_info)
        open_len = 0
        for step in range(1, max_len + 1):
            r = cr + dz * step
            c = cc + dx * step
            if r < 0 or r >= len(map_info) or c < 0 or c >= len(map_info[0]):
                break
            if map_info[r][c] == 0:
                break
            open_len += 1
        return float(open_len)

    def _flash_landing_offset(self, map_info, dx, dz):
        max_dist = 8 if dx != 0 and dz != 0 else 10
        for step in range(max_dist, 0, -1):
            tx = dx * step
            tz = dz * step
            if self._get_local_cell(map_info, tx, tz) > 0.5:
                return tx, tz, True
        return 0, 0, False

    def _estimate_monster_position(self, monster, hero_x, hero_z):
        if float(monster.get("is_in_view", 0)) > 0.5 and isinstance(monster.get("pos"), dict):
            pos = monster["pos"]
            return float(pos.get("x", hero_x)), float(pos.get("z", hero_z))

        rel_dir = int(monster.get("hero_relative_direction", 0))
        dist_bucket = int(monster.get("hero_l2_distance", MAX_DIST_BUCKET))
        ux, uz = REL_DIRS.get(rel_dir, (0.0, 0.0))
        approx_dist = DIST_BUCKET_CENTER.get(dist_bucket, 165.0)
        mx = float(np.clip(hero_x + ux * approx_dist, 0.0, MAP_SIZE - 1.0))
        mz = float(np.clip(hero_z + uz * approx_dist, 0.0, MAP_SIZE - 1.0))
        return mx, mz

    def _monster_feature(self, monster, hero_x, hero_z):
        exists = 1.0
        in_view = float(monster.get("is_in_view", 0))
        mx, mz = self._estimate_monster_position(monster, hero_x, hero_z)
        dx = mx - hero_x
        dz = mz - hero_z
        dist = _l2(hero_x, hero_z, mx, mz)
        speed_norm = _norm(monster.get("speed", 1), MAX_MONSTER_SPEED)
        feat = np.array([
            exists,
            in_view,
            _clip_signed(dx, 40.0),
            _clip_signed(dz, 40.0),
            min(dist / 180.0, 1.0),
            speed_norm,
        ], dtype=np.float32)
        return feat, (mx, mz), dist

    def _extract_entities(self, frame_state):
        treasures, buffs = [], []
        organs = frame_state.get("organs", [])
        if not isinstance(organs, list):
            return treasures, buffs
        for organ in organs:
            if not isinstance(organ, dict):
                continue
            if int(organ.get("status", 1)) != 1:
                continue
            sub_type = int(organ.get("sub_type", 0))
            if sub_type == 1:
                treasures.append(organ)
            elif sub_type == 2:
                buffs.append(organ)
        return treasures, buffs

    def _entity_position(self, ent, hero_x, hero_z):
        pos = ent.get("pos", {})
        if isinstance(pos, dict) and "x" in pos and "z" in pos:
            return float(pos.get("x", hero_x)), float(pos.get("z", hero_z))
        rel_dir = int(ent.get("hero_relative_direction", 0))
        dist_bucket = int(ent.get("hero_l2_distance", MAX_DIST_BUCKET))
        ux, uz = REL_DIRS.get(rel_dir, (0.0, 0.0))
        approx_dist = DIST_BUCKET_CENTER.get(dist_bucket, 165.0)
        tx = float(np.clip(hero_x + ux * approx_dist, 0.0, MAP_SIZE - 1.0))
        tz = float(np.clip(hero_z + uz * approx_dist, 0.0, MAP_SIZE - 1.0))
        return tx, tz

    def _treasure_priority(self, hero_x, hero_z, tx, tz, monster_positions, high_pressure):
        hero_dist = _l2(hero_x, hero_z, tx, tz)
        if monster_positions:
            min_monster_to_target = min(_l2(mx, mz, tx, tz) for mx, mz in monster_positions)
        else:
            min_monster_to_target = 180.0

        hero_term = 1.0 - min(hero_dist / 65.0, 1.0)
        safety_term = min(min_monster_to_target / 70.0, 1.0)
        pressure_scale = 0.85 if high_pressure > 0.5 else 1.0
        priority = pressure_scale * (0.55 * hero_term + 0.45 * safety_term)
        return float(np.clip(priority, 0.0, 1.0)), hero_dist, min_monster_to_target

    def _pack_treasure_features(self, treasures, hero_x, hero_z, monster_positions, high_pressure):
        items = []
        for tr in treasures:
            tx, tz = self._entity_position(tr, hero_x, hero_z)
            priority, hero_dist, _ = self._treasure_priority(hero_x, hero_z, tx, tz, monster_positions, high_pressure)
            items.append((priority, hero_dist, tx, tz))

        items.sort(key=lambda x: (-x[0], x[1]))

        feats = []
        best_dist = None
        best_priority = 0.0
        for i in range(2):
            if i < len(items):
                priority, dist, tx, tz = items[i]
                feats.extend([
                    1.0,
                    _clip_signed(tx - hero_x, 35.0),
                    _clip_signed(tz - hero_z, 35.0),
                    min(dist / 80.0, 1.0),
                    priority,
                ])
                if i == 0:
                    best_dist = dist
                    best_priority = priority
            else:
                feats.extend([0.0, 0.0, 0.0, 1.0, 0.0])
        return np.array(feats, dtype=np.float32), best_dist, best_priority

    def _pack_buff_features(self, buffs, hero_x, hero_z):
        items = []
        for bf in buffs:
            bx, bz = self._entity_position(bf, hero_x, hero_z)
            dist = _l2(hero_x, hero_z, bx, bz)
            items.append((dist, bx, bz))
        items.sort(key=lambda x: x[0])

        feats = []
        for i in range(2):
            if i < len(items):
                dist, bx, bz = items[i]
                feats.extend([
                    1.0,
                    _clip_signed(bx - hero_x, 35.0),
                    _clip_signed(bz - hero_z, 35.0),
                    min(dist / 80.0, 1.0),
                ])
            else:
                feats.extend([0.0, 0.0, 0.0, 1.0])
        return np.array(feats, dtype=np.float32)

    def _move_safety(self, map_info, hero_x, hero_z, monster_positions):
        scores = []
        open_lengths = []
        for dx, dz in ACTION_DIRS:
            if not self._is_passable_step(map_info, dx, dz):
                scores.append(0.0)
                open_lengths.append(0.0)
                continue
            nx = float(np.clip(hero_x + dx, 0.0, MAP_SIZE - 1.0))
            nz = float(np.clip(hero_z + dz, 0.0, MAP_SIZE - 1.0))
            min_dist = min((_l2(nx, nz, mx, mz) for mx, mz in monster_positions), default=60.0)
            open_len = self._open_length(map_info, dx, dz, max_len=6)
            score = 0.7 * np.clip(min_dist / 35.0, 0.0, 1.0) + 0.3 * (open_len / 6.0)
            scores.append(float(np.clip(score, 0.0, 1.0)))
            open_lengths.append(open_len)
        return np.array(scores, dtype=np.float32), open_lengths

    def _flash_safety(self, map_info, hero_x, hero_z, monster_positions):
        scores = []
        for dx, dz in ACTION_DIRS:
            fx, fz, ok = self._flash_landing_offset(map_info, dx, dz)
            if not ok:
                scores.append(0.0)
                continue
            nx = float(np.clip(hero_x + fx, 0.0, MAP_SIZE - 1.0))
            nz = float(np.clip(hero_z + fz, 0.0, MAP_SIZE - 1.0))
            min_dist = min((_l2(nx, nz, mx, mz) for mx, mz in monster_positions), default=80.0)
            open_len = self._open_length(map_info, np.sign(fx), np.sign(fz), max_len=6)
            distance_bonus = min(abs(fx) + abs(fz), 12.0) / 12.0
            score = 0.60 * np.clip(min_dist / 45.0, 0.0, 1.0) + 0.20 * (open_len / 6.0) + 0.20 * distance_bonus
            scores.append(float(np.clip(score, 0.0, 1.0)))
        return np.array(scores, dtype=np.float32)

    def _corridor_score(self, open_lengths):
        if not open_lengths:
            return 0.0
        vals = sorted(open_lengths, reverse=True)
        topk = vals[:3] if len(vals) >= 3 else vals
        return float(np.clip(np.mean(topk) / 6.0, 0.0, 1.0)) if topk else 0.0

    def _trap_pressure(self, hero_x, hero_z, monster_positions):
        if len(monster_positions) < 2:
            return 0.0
        (m1x, m1z), (m2x, m2z) = monster_positions[:2]
        v1 = np.array([m1x - hero_x, m1z - hero_z], dtype=np.float32)
        v2 = np.array([m2x - hero_x, m2z - hero_z], dtype=np.float32)
        d1 = float(np.linalg.norm(v1))
        d2 = float(np.linalg.norm(v2))
        if d1 < 1e-6 or d2 < 1e-6:
            return 1.0
        dot = float(np.dot(v1 / d1, v2 / d2))
        opposite = (1.0 - dot) * 0.5
        close1 = 1.0 - min(d1 / 45.0, 1.0)
        close2 = 1.0 - min(d2 / 45.0, 1.0)
        return float(np.clip(opposite * close1 * close2, 0.0, 1.0))

    def _build_local_patch(self, map_info, hero_x, hero_z, treasures, buffs, monster_positions, treasure_items=None):
        patch_size = Config.MAP_PATCH_SIZE
        radius = patch_size // 2
        ch0 = np.zeros((patch_size, patch_size), dtype=np.float32)
        ch1 = np.zeros((patch_size, patch_size), dtype=np.float32)
        ch2 = np.zeros((patch_size, patch_size), dtype=np.float32)

        # Passable channel
        for rz, dz in enumerate(range(-radius, radius + 1)):
            for cx, dx in enumerate(range(-radius, radius + 1)):
                ch0[rz, cx] = self._get_local_cell(map_info, dx, dz)

        # Resource channel
        for tr in treasures:
            tx, tz = self._entity_position(tr, hero_x, hero_z)
            ldx = int(round(tx - hero_x))
            ldz = int(round(tz - hero_z))
            if -radius <= ldx <= radius and -radius <= ldz <= radius:
                prio, _, _ = self._treasure_priority(hero_x, hero_z, tx, tz, monster_positions, 0.0)
                ch1[ldz + radius, ldx + radius] = max(ch1[ldz + radius, ldx + radius], 0.6 + 0.4 * prio)

        for bf in buffs:
            bx, bz = self._entity_position(bf, hero_x, hero_z)
            ldx = int(round(bx - hero_x))
            ldz = int(round(bz - hero_z))
            if -radius <= ldx <= radius and -radius <= ldz <= radius:
                ch1[ldz + radius, ldx + radius] = max(ch1[ldz + radius, ldx + radius], 0.45)

        # Monster threat channel
        for rz, dz in enumerate(range(-radius, radius + 1)):
            for cx, dx in enumerate(range(-radius, radius + 1)):
                world_x = hero_x + dx
                world_z = hero_z + dz
                threat = 0.0
                for mx, mz in monster_positions:
                    dist = _l2(world_x, world_z, mx, mz)
                    threat = max(threat, max(0.0, 1.0 - dist / 6.0))
                ch2[rz, cx] = threat

        patch = np.stack([ch0, ch1, ch2], axis=0)
        return patch.reshape(-1).astype(np.float32)

    def _legal_action_16(self, legal_act_raw):
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(16, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if 0 <= int(a) < 16}
                legal_action = [1 if j in valid_set else 0 for j in range(16)]
        if sum(legal_action) == 0:
            legal_action = [1] * 16
        return legal_action

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation.get("map_info", None)
        legal_act_raw = observation.get("legal_action", observation.get("legal_act", []))

        self.step_no = int(observation.get("step_no", 0))
        self.max_step = int(env_info.get("max_step", self.max_step))

        hero = self._get_hero(frame_state)
        hero_pos = hero.get("pos", {"x": 0, "z": 0})
        hero_x = float(hero_pos.get("x", 0.0))
        hero_z = float(hero_pos.get("z", 0.0))

        if self.last_pos is not None:
            move_delta = _l2(hero_x, hero_z, self.last_pos[0], self.last_pos[1])
        else:
            move_delta = 1.0
        stuck_flag = 1.0 if move_delta < 0.2 else 0.0

        cur_grid = (int(round(hero_x)), int(round(hero_z)))
        repeat_ratio = 0.0
        if self.recent_positions:
            repeat_cnt = sum(1 for p in self.recent_positions if p == cur_grid)
            repeat_ratio = float(np.clip(repeat_cnt / len(self.recent_positions), 0.0, 1.0))
        self.recent_positions.append(cur_grid)

        flash_cd = float(hero.get("flash_cooldown", env_info.get("flash_cooldown", 0)))
        buff_remaining = float(hero.get("buff_remaining_time", 0))
        buff_active = 1.0 if buff_remaining > 0 else 0.0
        flash_ready = 1.0 if flash_cd <= 0 else 0.0
        last_action_flash = 1.0 if last_action >= 8 else 0.0

        treasures_collected = float(hero.get("treasure_collected_count", env_info.get("treasures_collected", 0)))
        treasure_progress = _norm(treasures_collected, MAX_TREASURE_COUNT)
        collected_buff = float(env_info.get("collected_buff", 0))
        buff_progress = _norm(collected_buff, MAX_BUFF_COUNT)

        hero_feat = np.array([
            flash_ready,
            _norm(flash_cd, MAX_FLASH_CD),
            _norm(buff_remaining, MAX_BUFF_DURATION),
            buff_active,
            last_action_flash,
            stuck_flag,
            repeat_ratio,
        ], dtype=np.float32)

        monsters = frame_state.get("monsters", [])
        if not isinstance(monsters, list):
            monsters = []

        monster_feats = []
        monster_positions = []
        monster_dists = []
        for i in range(2):
            if i < len(monsters) and isinstance(monsters[i], dict):
                feat, pos, dist = self._monster_feature(monsters[i], hero_x, hero_z)
                monster_feats.append(feat)
                monster_positions.append(pos)
                monster_dists.append(dist)
            else:
                monster_feats.append(np.zeros(Config.MONSTER_DIM, dtype=np.float32))
                monster_dists.append(None)

        min_monster_dist = min((d for d in monster_dists if d is not None), default=180.0)
        second_monster_dist = 180.0
        valid_sorted = sorted([d for d in monster_dists if d is not None])
        if len(valid_sorted) >= 2:
            second_monster_dist = valid_sorted[1]
        elif len(valid_sorted) == 1:
            second_monster_dist = 180.0

        monster_speed_cfg = float(env_info.get("monster_speed", 500))
        any_speedup = any(float(m.get("speed", 1)) >= 2.0 for m in monsters if isinstance(m, dict))
        pre_speedup_urgency = 1.0
        if monster_speed_cfg > 1 and not any_speedup:
            pre_speedup_urgency = float(np.clip(1.0 - (monster_speed_cfg - self.step_no) / 220.0, 0.0, 1.0))

        high_pressure = 1.0 if (
            second_monster_dist < 55.0
            or any_speedup
            or self.step_no >= 0.60 * self.max_step
        ) else 0.0

        treasures, buffs = self._extract_entities(frame_state)
        treasure_feat, best_treasure_dist, best_treasure_priority = self._pack_treasure_features(
            treasures, hero_x, hero_z, monster_positions, high_pressure
        )
        buff_feat = self._pack_buff_features(buffs, hero_x, hero_z)

        move_safety_feat, move_open_lengths = self._move_safety(map_info, hero_x, hero_z, monster_positions)
        flash_safety_feat = self._flash_safety(map_info, hero_x, hero_z, monster_positions)
        corridor_score = self._corridor_score(move_open_lengths)
        trap_pressure = self._trap_pressure(hero_x, hero_z, monster_positions)
        flash_advantage = float(np.clip(np.max(flash_safety_feat) - np.max(move_safety_feat), -1.0, 1.0))

        global_feat = np.array([
            _norm(self.step_no, self.max_step),
            treasure_progress,
            buff_progress,
            min(min_monster_dist / 180.0, 1.0),
            min(second_monster_dist / 180.0, 1.0),
            high_pressure,
            pre_speedup_urgency,
            trap_pressure,
            corridor_score,
            _clip_signed(flash_advantage, 1.0),
        ], dtype=np.float32)

        map_patch_feat = self._build_local_patch(
            map_info=map_info,
            hero_x=hero_x,
            hero_z=hero_z,
            treasures=treasures,
            buffs=buffs,
            monster_positions=monster_positions,
        )

        feature = np.concatenate([
            hero_feat,
            monster_feats[0],
            monster_feats[1],
            treasure_feat[:5],
            treasure_feat[5:10],
            buff_feat[:4],
            buff_feat[4:8],
            move_safety_feat,
            flash_safety_feat,
            global_feat,
            map_patch_feat,
        ]).astype(np.float32)
        assert len(feature) == Config.DIM_OF_OBSERVATION, (
            f"Unexpected feature dim: {len(feature)} != {Config.DIM_OF_OBSERVATION}"
        )

        legal_action = self._legal_action_16(legal_act_raw)

        # ------------------------ reward shaping ------------------------
        cur_min_monster_dist_norm = min(min_monster_dist / 180.0, 1.0)
        cur_second_monster_dist_norm = min(second_monster_dist / 180.0, 1.0)

        survive_reward = 0.015 if high_pressure > 0.5 else 0.01

        step_score = float(env_info.get("step_score", 0.0))
        treasure_score = float(env_info.get("treasure_score", 0.0))
        total_score = float(env_info.get("total_score", step_score + treasure_score))
        step_score_gain = max(step_score - self.last_step_score, 0.0)
        treasure_score_gain = max(treasure_score - self.last_treasure_score, 0.0)
        total_score_gain = max(total_score - self.last_total_score, 0.0)

        step_score_reward = 0.012 * min(step_score_gain / 1.5, 1.0)
        treasure_score_reward = 0.0
        if treasure_score_gain > 0:
            treasure_score_reward = 2.2 * min(treasure_score_gain / 100.0, 2.0)

        cur_treasure_count = int(hero.get("treasure_collected_count", env_info.get("treasures_collected", 0)))
        treasure_count_delta = max(cur_treasure_count - self.last_treasure_collected, 0)
        treasure_count_reward = 0.8 * treasure_count_delta

        cur_buff_count = int(env_info.get("collected_buff", 0))
        buff_delta = max(cur_buff_count - self.last_collected_buff, 0)
        buff_reward = 0.45 * buff_delta

        if self.last_min_monster_dist_norm is None:
            dist_shaping = 0.0
        else:
            dist_delta = cur_min_monster_dist_norm - self.last_min_monster_dist_norm
            dist_weight = 0.14 if high_pressure < 0.5 else 0.26
            dist_shaping = dist_weight * dist_delta

        second_monster_penalty = 0.0
        if cur_second_monster_dist_norm < 0.20:
            second_monster_penalty = -0.05 * (0.20 - cur_second_monster_dist_norm) / 0.20

        treasure_approach_reward = 0.0
        if (
            best_treasure_dist is not None
            and self.last_best_treasure_dist is not None
            and self.last_min_monster_dist_norm is not None
        ):
            approach = self.last_best_treasure_dist - best_treasure_dist
            monster_closer = self.last_min_monster_dist_norm - cur_min_monster_dist_norm
            if approach > 0 and monster_closer < 0.04:
                phase_scale = 1.0 if high_pressure < 0.5 else 0.45
                treasure_approach_reward = 0.07 * phase_scale * best_treasure_priority * float(np.clip(approach / 5.0, 0.0, 1.0))

        priority_tracking_reward = 0.0
        if (
            self.last_best_treasure_priority is not None
            and best_treasure_priority > 0
            and high_pressure < 0.5
        ):
            priority_tracking_reward = 0.03 * max(best_treasure_priority - 0.55, 0.0)

        speedup_buffer_reward = 0.0
        if pre_speedup_urgency > 0.55 and self.last_min_monster_dist_norm is not None:
            dist_delta = cur_min_monster_dist_norm - self.last_min_monster_dist_norm
            if dist_delta > 0:
                speedup_buffer_reward = 0.06 * pre_speedup_urgency * dist_delta

        corridor_reward = 0.02 * corridor_score * (0.5 + 0.5 * high_pressure)
        deadend_penalty = 0.0
        if corridor_score < 0.28 and cur_min_monster_dist_norm < 0.22:
            deadend_penalty = -0.08 * (0.28 - corridor_score) / 0.28

        trap_penalty = -0.10 * trap_pressure * (0.5 + 0.5 * high_pressure)

        danger_penalty = 0.0
        danger_threshold = 0.22 if high_pressure < 0.5 else 0.28
        if cur_min_monster_dist_norm < danger_threshold:
            danger_penalty = -0.12 * (danger_threshold - cur_min_monster_dist_norm) / danger_threshold

        invalid_move_penalty = 0.0
        if last_action != -1 and last_action < 8 and move_delta < 0.2:
            invalid_move_penalty = -0.03

        repeat_penalty = 0.0
        if repeat_ratio > 0.55:
            repeat_penalty = -0.03 * (repeat_ratio - 0.55) / 0.45

        flash_reward = 0.0
        if last_action >= 8 and self.last_min_monster_dist_norm is not None:
            dist_gain = cur_min_monster_dist_norm - self.last_min_monster_dist_norm
            corridor_gain = corridor_score - self.last_corridor_score
            if dist_gain > 0.04 or corridor_gain > 0.08:
                flash_reward = 0.22 + 0.35 * max(dist_gain, 0.0) + 0.12 * max(corridor_gain, 0.0)
            else:
                flash_reward = -0.06

        reward_value = (
            survive_reward
            + step_score_reward
            + treasure_score_reward
            + treasure_count_reward
            + buff_reward
            + dist_shaping
            + second_monster_penalty
            + treasure_approach_reward
            + priority_tracking_reward
            + speedup_buffer_reward
            + corridor_reward
            + deadend_penalty
            + trap_penalty
            + danger_penalty
            + invalid_move_penalty
            + repeat_penalty
            + flash_reward
        )

        self.last_min_monster_dist_norm = cur_min_monster_dist_norm
        self.last_second_monster_dist_norm = cur_second_monster_dist_norm
        self.last_best_treasure_dist = best_treasure_dist
        self.last_best_treasure_priority = best_treasure_priority
        self.last_total_score = total_score
        self.last_step_score = step_score
        self.last_treasure_score = treasure_score
        self.last_treasure_collected = cur_treasure_count
        self.last_collected_buff = cur_buff_count
        self.last_flash_count = int(env_info.get("flash_count", 0))
        self.last_pos = (hero_x, hero_z)
        self.last_corridor_score = corridor_score
        self.last_action = last_action

        reward = [float(reward_value)]
        return feature, legal_action, reward
