#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO (V4).
峡谷追猎 PPO 特征预处理与奖励设计（V4）。

V4 核心思路：
1. 不再堆原始字段，而是围绕“危险 / 机会 / 空间 / 闪现收益”重构特征；
2. 加入前两个宝箱、前两个 buff，并给出宝箱优先级 / 风险收益比特征；
3. 奖励按前后期切换：前期更鼓励拿资源，后期更鼓励保命和拉空间；
4. 让闪现学成“危险时的脱险技能”，而不是可有可无。

特征共 64 维：
- hero_self      : 8
- monster_1      : 6
- monster_2      : 6
- treasure_1     : 5
- treasure_2     : 5
- buff_1         : 4
- buff_2         : 4
- space_feature  : 20
- global_feature : 6
"""

import numpy as np

MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_DIST_BUCKET = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
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
        self.last_best_treasure_dist = None
        self.last_best_treasure_priority = None
        self.last_total_score = 0.0
        self.last_step_score = 0.0
        self.last_treasure_score = 0.0
        self.last_treasure_collected = 0
        self.last_collected_buff = 0
        self.last_flash_count = 0
        self.last_pos = None
        self.last_action = -1

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
            min(dist / 180.0, 1.0),
            _clip_signed(dx, 30.0),
            _clip_signed(dz, 30.0),
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

    def _treasure_priority(self, hero_x, hero_z, tx, tz, monster_positions):
        hero_dist = _l2(hero_x, hero_z, tx, tz)
        if monster_positions:
            min_monster_to_target = min(_l2(mx, mz, tx, tz) for mx, mz in monster_positions)
        else:
            min_monster_to_target = 180.0

        # 越近越好，怪物离目标越远越好。
        hero_term = 1.0 - min(hero_dist / 60.0, 1.0)
        safety_term = min(min_monster_to_target / 60.0, 1.0)
        priority = 0.55 * hero_term + 0.45 * safety_term
        return float(np.clip(priority, 0.0, 1.0)), hero_dist, min_monster_to_target

    def _pack_treasure_features(self, treasures, hero_x, hero_z, monster_positions):
        items = []
        for tr in treasures:
            tx, tz = self._entity_position(tr, hero_x, hero_z)
            priority, hero_dist, _ = self._treasure_priority(hero_x, hero_z, tx, tz, monster_positions)
            items.append((priority, hero_dist, tx, tz))

        # 先按优先级，再按距离排序
        items.sort(key=lambda x: (-x[0], x[1]))

        feats = []
        best_dist = None
        best_priority = 0.0
        for i in range(2):
            if i < len(items):
                priority, hero_dist, tx, tz = items[i]
                dx = tx - hero_x
                dz = tz - hero_z
                feats.extend([
                    1.0,
                    min(hero_dist / 60.0, 1.0),
                    _clip_signed(dx, 25.0),
                    _clip_signed(dz, 25.0),
                    priority,
                ])
                if i == 0:
                    best_dist = hero_dist
                    best_priority = priority
            else:
                feats.extend([0.0, 1.0, 0.0, 0.0, 0.0])
        return np.array(feats, dtype=np.float32), best_dist, best_priority

    def _pack_buff_features(self, buffs, hero_x, hero_z):
        items = []
        for bf in buffs:
            bx, bz = self._entity_position(bf, hero_x, hero_z)
            hero_dist = _l2(hero_x, hero_z, bx, bz)
            items.append((hero_dist, bx, bz))
        items.sort(key=lambda x: x[0])

        feats = []
        for i in range(2):
            if i < len(items):
                hero_dist, bx, bz = items[i]
                dx = bx - hero_x
                dz = bz - hero_z
                feats.extend([
                    1.0,
                    min(hero_dist / 50.0, 1.0),
                    _clip_signed(dx, 25.0),
                    _clip_signed(dz, 25.0),
                ])
            else:
                feats.extend([0.0, 1.0, 0.0, 0.0])
        return np.array(feats, dtype=np.float32)

    def _landing_safety(self, land_x, land_z, monster_positions, land_open_len, high_pressure=False):
        if monster_positions:
            min_dist = min(_l2(land_x, land_z, mx, mz) for mx, mz in monster_positions)
        else:
            min_dist = 180.0
        dist_score = min(min_dist / (36.0 if high_pressure else 45.0), 1.0)
        open_score = min(land_open_len / 6.0, 1.0)
        score = 0.72 * dist_score + 0.28 * open_score
        return float(np.clip(score, 0.0, 1.0)), min_dist

    def _move_and_flash_space_features(self, map_info, hero_x, hero_z, hero_pos, legal_action, monster_positions, high_pressure):
        move_scores = []
        flash_scores = []
        cardinal_opens = []
        diagonal_opens = []

        for idx, (dx, dz) in enumerate(ACTION_DIRS):
            passable = self._is_passable_step(map_info, dx, dz)
            open_len = self._open_length(map_info, dx, dz)
            if dx == 0 or dz == 0:
                cardinal_opens.append(open_len)
            else:
                diagonal_opens.append(open_len)

            if passable:
                next_x = float(np.clip(hero_x + dx, 0.0, MAP_SIZE - 1.0))
                next_z = float(np.clip(hero_z + dz, 0.0, MAP_SIZE - 1.0))
                move_score, _ = self._landing_safety(next_x, next_z, monster_positions, open_len, high_pressure)
            else:
                move_score = 0.0
            move_scores.append(move_score)

            flash_act = 8 + idx
            if flash_act >= len(legal_action) or legal_action[flash_act] == 0:
                flash_scores.append(0.0)
                continue

            flash_dist = 10 if dx == 0 or dz == 0 else 8
            land_step = 0
            for step in range(flash_dist, 0, -1):
                if self._is_passable_step(map_info, dx * step, dz * step):
                    land_step = step
                    break
            if land_step == 0:
                flash_scores.append(0.0)
                continue

            land_x = float(np.clip(hero_x + dx * land_step, 0.0, MAP_SIZE - 1.0))
            land_z = float(np.clip(hero_z + dz * land_step, 0.0, MAP_SIZE - 1.0))
            land_open_len = self._open_length(map_info, dx, dz)
            flash_score, _ = self._landing_safety(land_x, land_z, monster_positions, land_open_len + 1.5, high_pressure)
            # 适度鼓励有位移的闪现落点
            flash_scores.append(float(np.clip(0.85 * flash_score + 0.15 * min(land_step / float(flash_dist), 1.0), 0.0, 1.0)))

        cardinal_mean = float(np.mean(cardinal_opens)) if cardinal_opens else 0.0
        diagonal_mean = float(np.mean(diagonal_opens)) if diagonal_opens else 0.0
        center_open = float(np.mean([self._get_local_cell(map_info, dx, dz) for dx, dz in ACTION_DIRS]))
        deadend_risk = float(np.clip(1.0 - max(move_scores) - 0.25 * center_open, 0.0, 1.0))

        space_feat = np.array(
            move_scores + flash_scores + [
                min(cardinal_mean / 6.0, 1.0),
                min(diagonal_mean / 6.0, 1.0),
                center_open,
                deadend_risk,
            ],
            dtype=np.float32,
        )
        return space_feat, move_scores, flash_scores, deadend_risk

    def _legal_action_16(self, legal_act_raw):
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
        cur_pos = (hero_x, hero_z)

        legal_action = self._legal_action_16(legal_act_raw)

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
                monster_feats.append(np.zeros(6, dtype=np.float32))
                monster_positions.append(None)
                monster_dists.append(None)

        valid_dists = [d for d in monster_dists if d is not None]
        nearest_monster_dist = min(valid_dists) if valid_dists else 180.0
        second_monster_dist = monster_dists[1] if monster_dists[1] is not None else 180.0
        any_monster_speedup = any(float(m.get("speed", 1)) > 1.0 for m in monsters if isinstance(m, dict))
        two_monsters_alive = 1.0 if sum(1 for m in monsters if isinstance(m, dict)) >= 2 else 0.0
        step_norm = _norm(self.step_no, self.max_step)
        high_pressure = bool(any_monster_speedup or two_monsters_alive > 0.5 or step_norm > 0.55)

        flash_cd_norm = _norm(hero.get("flash_cooldown", 0), MAX_FLASH_CD)
        flash_ready = 1.0 if float(hero.get("flash_cooldown", 0)) <= 1e-6 and legal_action[8] == 1 else 0.0
        buff_remain_norm = _norm(hero.get("buff_remaining_time", 0), MAX_BUFF_DURATION)
        buff_active = 1.0 if float(hero.get("buff_remaining_time", 0)) > 0 else 0.0
        last_action_is_flash = 1.0 if last_action is not None and int(last_action) >= 8 else 0.0
        last_action_dir = float((int(last_action) % 8) / 7.0) if last_action is not None and int(last_action) >= 0 else 0.0

        hero_feat = np.array([
            _norm(hero_x, MAP_SIZE),
            _norm(hero_z, MAP_SIZE),
            flash_cd_norm,
            flash_ready,
            buff_remain_norm,
            buff_active,
            last_action_is_flash,
            last_action_dir,
        ], dtype=np.float32)

        treasures, buffs = self._extract_entities(frame_state)
        valid_monster_positions = [p for p in monster_positions if p is not None]
        treasure_feat, best_treasure_dist, best_treasure_priority = self._pack_treasure_features(
            treasures, hero_x, hero_z, valid_monster_positions
        )
        buff_feat = self._pack_buff_features(buffs, hero_x, hero_z)

        space_feat, move_scores, flash_scores, deadend_risk = self._move_and_flash_space_features(
            map_info, hero_x, hero_z, hero_pos, legal_action, valid_monster_positions, high_pressure
        )
        best_move = max(move_scores) if move_scores else 0.0
        best_flash = max(flash_scores) if flash_scores else 0.0
        flash_advantage = float(np.clip(best_flash - best_move, -1.0, 1.0))

        treasure_progress = _norm(int(hero.get("treasure_collected_count", 0)), MAX_TREASURE_COUNT)
        second_pressure = 1.0 - min(second_monster_dist / 120.0, 1.0)
        global_feat = np.array([
            step_norm,
            treasure_progress,
            min(nearest_monster_dist / 180.0, 1.0),
            second_pressure,
            1.0 if high_pressure else 0.0,
            flash_advantage,
        ], dtype=np.float32)

        feature = np.concatenate([
            hero_feat,
            monster_feats[0],
            monster_feats[1],
            treasure_feat[:5],
            treasure_feat[5:10],
            buff_feat[:4],
            buff_feat[4:8],
            space_feat,
            global_feat,
        ]).astype(np.float32)

        assert len(feature) == 64, f"Unexpected feature dim {len(feature)}"

        # =========================== reward V4 ============================
        cur_total_score = float(env_info.get("total_score", 0.0))
        cur_step_score = float(env_info.get("step_score", 0.0))
        cur_treasure_score = float(env_info.get("treasure_score", 0.0))
        cur_collected_buff = int(env_info.get("collected_buff", 0))
        cur_flash_count = int(env_info.get("flash_count", 0))
        cur_treasure_collected = int(hero.get("treasure_collected_count", 0))

        total_score_gain = cur_total_score - self.last_total_score
        step_score_gain = cur_step_score - self.last_step_score
        treasure_score_gain = cur_treasure_score - self.last_treasure_score
        treasure_gain = max(cur_treasure_collected - self.last_treasure_collected, 0)
        buff_gain = max(cur_collected_buff - self.last_collected_buff, 0)
        flash_gain = max(cur_flash_count - self.last_flash_count, 0)

        cur_min_monster_dist_norm = min(nearest_monster_dist / 180.0, 1.0)
        if self.last_min_monster_dist_norm is None:
            monster_dist_delta = 0.0
        else:
            monster_dist_delta = cur_min_monster_dist_norm - self.last_min_monster_dist_norm

        # 前期偏资源，后期偏生存
        resource_phase = 1.0 - (0.65 if high_pressure else 0.25)
        survival_phase = 0.65 if high_pressure else 0.35

        survive_reward = 0.012 if not high_pressure else 0.020
        step_reward = 0.015 * max(step_score_gain / 1.5, 0.0)
        treasure_score_reward = 1.60 * max(treasure_score_gain / 100.0, 0.0)
        treasure_count_reward = 0.40 * treasure_gain
        buff_reward = 0.28 * buff_gain
        monster_dist_reward = (0.16 if high_pressure else 0.10) * monster_dist_delta

        treasure_approach_reward = 0.0
        if (
            best_treasure_dist is not None
            and self.last_best_treasure_dist is not None
            and self.last_min_monster_dist_norm is not None
        ):
            treasure_dist_delta = self.last_best_treasure_dist - best_treasure_dist
            if treasure_dist_delta > 0 and best_treasure_priority >= 0.45 and cur_min_monster_dist_norm >= 0.18:
                treasure_approach_reward = (0.05 + 0.05 * best_treasure_priority) * min(treasure_dist_delta / 6.0, 1.0)

        priority_tracking_reward = 0.0
        if self.last_best_treasure_priority is not None:
            priority_tracking_reward = 0.04 * (best_treasure_priority - self.last_best_treasure_priority)

        corridor_reward = 0.0
        if cur_min_monster_dist_norm < 0.28 or high_pressure:
            corridor_reward = 0.05 * (space_feat[16] + 0.5 * space_feat[17])

        deadend_penalty = 0.0
        if cur_min_monster_dist_norm < 0.30 or high_pressure:
            deadend_penalty = -0.08 * deadend_risk

        sandwich_penalty = 0.0
        if two_monsters_alive > 0.5:
            sandwich_penalty = -0.08 * max(0.0, 0.45 - cur_min_monster_dist_norm) - 0.05 * second_pressure

        invalid_move_penalty = 0.0
        if self.last_pos is not None:
            move_dist = _l2(self.last_pos[0], self.last_pos[1], hero_x, hero_z)
            if move_dist < 0.1 and (last_action is not None and 0 <= int(last_action) < 8):
                invalid_move_penalty = -0.05

        flash_escape_reward = 0.0
        if last_action is not None and int(last_action) >= 8:
            if monster_dist_delta > 0.06:
                flash_escape_reward += 0.30 + 0.20 * min(monster_dist_delta / 0.25, 1.0)
            if best_flash > best_move + 0.12:
                flash_escape_reward += 0.08
            if monster_dist_delta < 0.01 and best_treasure_priority < 0.45:
                flash_escape_reward -= 0.12

        danger_penalty = 0.0
        if cur_min_monster_dist_norm < (0.18 if high_pressure else 0.12):
            danger_penalty = -0.12 * ((0.18 if high_pressure else 0.12) - cur_min_monster_dist_norm) / (0.18 if high_pressure else 0.12)

        reward_scalar = (
            survive_reward
            + step_reward
            + treasure_score_reward
            + treasure_count_reward
            + buff_reward
            + resource_phase * treasure_approach_reward
            + 0.5 * resource_phase * priority_tracking_reward
            + survival_phase * monster_dist_reward
            + corridor_reward
            + deadend_penalty
            + sandwich_penalty
            + invalid_move_penalty
            + flash_escape_reward
            + danger_penalty
            + 0.08 * max(total_score_gain, 0.0)
            + 0.06 * flash_gain
        )

        self.last_min_monster_dist_norm = cur_min_monster_dist_norm
        self.last_best_treasure_dist = best_treasure_dist
        self.last_best_treasure_priority = best_treasure_priority
        self.last_total_score = cur_total_score
        self.last_step_score = cur_step_score
        self.last_treasure_score = cur_treasure_score
        self.last_treasure_collected = cur_treasure_collected
        self.last_collected_buff = cur_collected_buff
        self.last_flash_count = cur_flash_count
        self.last_pos = cur_pos
        self.last_action = last_action

        reward = [float(reward_scalar)]
        return feature, legal_action, reward
