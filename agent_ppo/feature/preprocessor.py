#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature preprocessor and reward design for Gorge Chase PPO V7.2.
第二轮在 V7 基础上继续做两件事：
1. 把 reward 系数进一步往 score 主导方向压实；
2. 补上针对“第二只怪在 10 步前位置生成”的反回环特征与惩罚。
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
LOCAL_RADIUS = 10

ACTION_DIRS = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
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
        self.last_total_score = 0.0
        self.last_step_score = 0.0
        self.last_treasure_score = 0.0
        self.last_treasure_collected = 0
        self.last_collected_buff = 0
        self.last_flash_count = 0
        self.last_pos = None
        self.last_corridor_score = 0.0
        self.last_trap_pressure = 0.0
        self.last_action = -1

        self.locked_treasure_key = None
        self.locked_treasure_age = 0
        self.last_locked_treasure_dist = None
        self.last_locked_treasure_priority = 0.0

        self.recent_positions = deque(maxlen=24)
        self.recent_exact_positions = deque(maxlen=32)
        self.visited_count = {}
        self.seen_treasure_once = set()
        self.treasure_memory = {}
        self.buff_memory = {}

        self.pending_flash_eval = None
        self.last_reward_components = {}

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

    def _get_rel_cell(self, map_info, rx, rz):
        return self._get_local_cell(map_info, rx, rz)

    def _is_passable_step(self, map_info, dx, dz):
        target_ok = self._get_local_cell(map_info, dx, dz) > 0.5
        if not target_ok:
            return False
        if dx != 0 and dz != 0:
            edge1_ok = self._get_local_cell(map_info, dx, 0) > 0.5
            edge2_ok = self._get_local_cell(map_info, 0, dz) > 0.5
            return edge1_ok or edge2_ok
        return True

    def _is_passable_from(self, map_info, cur_x, cur_z, dx, dz):
        nx = cur_x + dx
        nz = cur_z + dz
        target_ok = self._get_rel_cell(map_info, nx, nz) > 0.5
        if not target_ok:
            return False
        if dx != 0 and dz != 0:
            edge1_ok = self._get_rel_cell(map_info, cur_x + dx, cur_z) > 0.5
            edge2_ok = self._get_rel_cell(map_info, cur_x, cur_z + dz) > 0.5
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

    def _entity_key(self, ent, prefix, hero_x, hero_z):
        cid = ent.get("config_id", None)
        if cid is not None:
            return f"{prefix}_{int(cid)}"
        x, z = self._entity_position(ent, hero_x, hero_z)
        return f"{prefix}_{int(round(x))}_{int(round(z))}"

    def _refresh_memory(self, visible_treasures, visible_buffs, env_info, hero_x, hero_z):
        remaining_ids = set(int(x) for x in env_info.get("treasure_id", []) if x is not None)
        newly_seen_treasure = 0

        for tr in visible_treasures:
            key = self._entity_key(tr, "treasure", hero_x, hero_z)
            tx, tz = self._entity_position(tr, hero_x, hero_z)
            self.treasure_memory[key] = (tx, tz)
            if key not in self.seen_treasure_once:
                self.seen_treasure_once.add(key)
                newly_seen_treasure += 1

        if remaining_ids:
            remove_keys = []
            for key in list(self.treasure_memory.keys()):
                suffix = key.split("treasure_", 1)[1] if key.startswith("treasure_") else ""
                if suffix.isdigit() and int(suffix) not in remaining_ids:
                    remove_keys.append(key)
            for key in remove_keys:
                self.treasure_memory.pop(key, None)

        remove_keys = []
        for key, (tx, tz) in self.treasure_memory.items():
            if _l2(hero_x, hero_z, tx, tz) <= 1.5:
                remove_keys.append(key)
        for key in remove_keys:
            self.treasure_memory.pop(key, None)
            if key == self.locked_treasure_key:
                self.locked_treasure_key = None
                self.locked_treasure_age = 0

        for bf in visible_buffs:
            key = self._entity_key(bf, "buff", hero_x, hero_z)
            bx, bz = self._entity_position(bf, hero_x, hero_z)
            self.buff_memory[key] = (bx, bz)

        return newly_seen_treasure

    def _memory_entities(self, visible_entities, memory_dict, prefix, hero_x, hero_z):
        visible_keys = set()
        merged = list(visible_entities)
        for ent in visible_entities:
            visible_keys.add(self._entity_key(ent, prefix, hero_x, hero_z))
        for key, (x, z) in memory_dict.items():
            if key in visible_keys:
                continue
            merged.append({"pos": {"x": x, "z": z}, "status": 1, "memory": 1, "memory_key": key})
        return merged

    def _treasure_priority(self, hero_x, hero_z, tx, tz, monster_positions, high_pressure):
        hero_dist = _l2(hero_x, hero_z, tx, tz)
        if monster_positions:
            min_monster_to_target = min(_l2(mx, mz, tx, tz) for mx, mz in monster_positions)
        else:
            min_monster_to_target = 180.0
        hero_term = 1.0 - min(hero_dist / 65.0, 1.0)
        safety_term = min(min_monster_to_target / 70.0, 1.0)
        pressure_scale = 0.82 if high_pressure > 0.5 else 1.0
        priority = pressure_scale * (0.58 * hero_term + 0.42 * safety_term)
        return float(np.clip(priority, 0.0, 1.0)), hero_dist, min_monster_to_target

    def _build_treasure_items(self, treasures, hero_x, hero_z, monster_positions, high_pressure):
        items = []
        for tr in treasures:
            key = tr.get("memory_key") or self._entity_key(tr, "treasure", hero_x, hero_z)
            tx, tz = self._entity_position(tr, hero_x, hero_z)
            priority, hero_dist, monster_dist = self._treasure_priority(
                hero_x, hero_z, tx, tz, monster_positions, high_pressure
            )
            items.append({
                "key": key,
                "tx": tx,
                "tz": tz,
                "priority": priority,
                "hero_dist": hero_dist,
                "monster_dist": monster_dist,
            })
        items.sort(key=lambda x: (-x["priority"], x["hero_dist"]))
        return items

    def _select_locked_treasure(self, items, high_pressure):
        if not items:
            self.locked_treasure_key = None
            self.locked_treasure_age = 0
            return None

        best = items[0]
        item_by_key = {item["key"]: item for item in items}

        if self.locked_treasure_key is None or self.locked_treasure_key not in item_by_key:
            self.locked_treasure_key = best["key"]
            self.locked_treasure_age = 0
            return best

        cur = item_by_key[self.locked_treasure_key]
        priority_slack = 0.22 if high_pressure < 0.5 else 0.10
        current_is_too_far = cur["hero_dist"] > best["hero_dist"] + 16.0
        current_is_too_unsafe = high_pressure > 0.5 and cur["monster_dist"] + 8.0 < best["monster_dist"]
        current_is_much_worse = cur["priority"] + priority_slack < best["priority"]

        if current_is_much_worse and (current_is_too_far or current_is_too_unsafe):
            self.locked_treasure_key = best["key"]
            self.locked_treasure_age = 0
            return best

        self.locked_treasure_age += 1
        return cur

    def _pack_treasure_features(self, items, locked_item):
        if locked_item is not None:
            locked_key = locked_item["key"]
            display_items = sorted(
                items,
                key=lambda item: (0 if item["key"] == locked_key else 1, -item["priority"], item["hero_dist"]),
            )
        else:
            display_items = list(items)

        feats = []
        locked_dist = None
        locked_priority = 0.0
        for i in range(2):
            if i < len(display_items):
                item = display_items[i]
                priority_feat = item["priority"]
                if locked_item is not None and item["key"] == locked_item["key"]:
                    priority_feat = min(priority_feat + 0.10, 1.0)
                feats.extend([
                    1.0,
                    _clip_signed(item["tx"] - item.get("hero_x", 0.0), 35.0),
                    _clip_signed(item["tz"] - item.get("hero_z", 0.0), 35.0),
                    min(item["hero_dist"] / 80.0, 1.0),
                    priority_feat,
                ])
            else:
                feats.extend([0.0, 0.0, 0.0, 1.0, 0.0])

        if locked_item is not None:
            locked_dist = locked_item["hero_dist"]
            locked_priority = locked_item["priority"]
        return np.array(feats, dtype=np.float32), locked_dist, locked_priority

    def _pack_treasure_features_from_items(self, items, locked_item, hero_x, hero_z):
        if locked_item is not None:
            locked_key = locked_item["key"]
            display_items = sorted(
                items,
                key=lambda item: (0 if item["key"] == locked_key else 1, -item["priority"], item["hero_dist"]),
            )
        else:
            display_items = list(items)

        feats = []
        locked_dist = None
        locked_priority = 0.0
        locked_monster_dist = 180.0
        for i in range(2):
            if i < len(display_items):
                item = display_items[i]
                priority_feat = item["priority"]
                if locked_item is not None and item["key"] == locked_item["key"]:
                    priority_feat = min(priority_feat + 0.10, 1.0)
                feats.extend([
                    1.0,
                    _clip_signed(item["tx"] - hero_x, 35.0),
                    _clip_signed(item["tz"] - hero_z, 35.0),
                    min(item["hero_dist"] / 80.0, 1.0),
                    priority_feat,
                ])
            else:
                feats.extend([0.0, 0.0, 0.0, 1.0, 0.0])

        if locked_item is not None:
            locked_dist = locked_item["hero_dist"]
            locked_priority = locked_item["priority"]
            locked_monster_dist = locked_item["monster_dist"]
        return np.array(feats, dtype=np.float32), locked_dist, locked_priority, locked_monster_dist

    def _pack_buff_features(self, buffs, hero_x, hero_z):
        items = []
        for bf in buffs:
            bx, bz = self._entity_position(bf, hero_x, hero_z)
            dist = _l2(hero_x, hero_z, bx, bz)
            items.append((dist, bx, bz))
        items.sort(key=lambda x: x[0])

        feats = []
        best_dist = None
        for i in range(2):
            if i < len(items):
                dist, bx, bz = items[i]
                feats.extend([
                    1.0,
                    _clip_signed(bx - hero_x, 35.0),
                    _clip_signed(bz - hero_z, 35.0),
                    min(dist / 80.0, 1.0),
                ])
                if i == 0:
                    best_dist = dist
            else:
                feats.extend([0.0, 0.0, 0.0, 1.0])
        return np.array(feats, dtype=np.float32), best_dist

    def _bfs_connectivity(self, map_info, start_dx, start_dz, max_nodes=72):
        if self._get_rel_cell(map_info, start_dx, start_dz) <= 0.5:
            return 0.0, 0.0, 0.0

        queue = deque([(start_dx, start_dz, 0)])
        visited = {(start_dx, start_dz)}
        max_depth = 0
        max_branch = 0

        while queue and len(visited) < max_nodes:
            cur_x, cur_z, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            local_branch = 0
            for dx, dz in ACTION_DIRS:
                nx = cur_x + dx
                nz = cur_z + dz
                if abs(nx) > LOCAL_RADIUS or abs(nz) > LOCAL_RADIUS:
                    continue
                if (nx, nz) in visited:
                    continue
                if not self._is_passable_from(map_info, cur_x, cur_z, dx, dz):
                    continue
                visited.add((nx, nz))
                queue.append((nx, nz, depth + 1))
                local_branch += 1
            max_branch = max(max_branch, local_branch)

        area_norm = min(len(visited) / 36.0, 1.0)
        depth_norm = min(max_depth / 10.0, 1.0)
        branch_norm = min(max_branch / 4.0, 1.0)
        return float(area_norm), float(depth_norm), float(branch_norm)

    def _move_safety(self, map_info, hero_x, hero_z, monster_positions):
        scores = []
        stats = []
        for dx, dz in ACTION_DIRS:
            if not self._is_passable_step(map_info, dx, dz):
                scores.append(0.0)
                stats.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                continue

            nx = int(round(np.clip(hero_x + dx, 0.0, MAP_SIZE - 1.0)))
            nz = int(round(np.clip(hero_z + dz, 0.0, MAP_SIZE - 1.0)))
            min_dist = min((_l2(nx, nz, mx, mz) for mx, mz in monster_positions), default=60.0)
            monster_term = np.clip(min_dist / 35.0, 0.0, 1.0)
            area_norm, depth_norm, branch_norm = self._bfs_connectivity(map_info, dx, dz)
            direct_open = self._open_length(map_info, dx, dz, max_len=6) / 6.0
            novelty = 1.0 - min(self.visited_count.get((nx, nz), 0) / 3.0, 1.0)
            trail_recent = self._trail_risk_at(nx, nz, 8, 12, Config.BACKTRACK_NEAR_DIST)
            trail_long = self._trail_risk_at(nx, nz, 16, 24, Config.BACKTRACK_FAR_DIST)
            score = (
                0.35 * monster_term
                + 0.24 * area_norm
                + 0.17 * depth_norm
                + 0.08 * branch_norm
                + 0.08 * direct_open
                + 0.06 * novelty
                - 0.10 * trail_recent
                - 0.04 * trail_long
            )
            scores.append(float(np.clip(score, 0.0, 1.0)))
            stats.append((area_norm, depth_norm, branch_norm, direct_open, trail_recent, trail_long))
        return np.array(scores, dtype=np.float32), stats

    def _flash_safety(self, map_info, hero_x, hero_z, monster_positions):
        scores = []
        stats = []
        for dx, dz in ACTION_DIRS:
            fx, fz, ok = self._flash_landing_offset(map_info, dx, dz)
            if not ok:
                scores.append(0.0)
                stats.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                continue

            nx = int(round(np.clip(hero_x + fx, 0.0, MAP_SIZE - 1.0)))
            nz = int(round(np.clip(hero_z + fz, 0.0, MAP_SIZE - 1.0)))
            min_dist = min((_l2(nx, nz, mx, mz) for mx, mz in monster_positions), default=80.0)
            monster_term = np.clip(min_dist / 45.0, 0.0, 1.0)
            area_norm, depth_norm, branch_norm = self._bfs_connectivity(map_info, fx, fz)
            direct_open = self._open_length(map_info, int(np.sign(fx)), int(np.sign(fz)), max_len=6) / 6.0
            distance_bonus = min(abs(fx) + abs(fz), 12.0) / 12.0
            novelty = 1.0 - min(self.visited_count.get((nx, nz), 0) / 3.0, 1.0)
            trail_recent = self._trail_risk_at(nx, nz, 8, 12, Config.BACKTRACK_NEAR_DIST)
            trail_long = self._trail_risk_at(nx, nz, 16, 24, Config.BACKTRACK_FAR_DIST)
            score = (
                0.28 * monster_term
                + 0.22 * area_norm
                + 0.16 * depth_norm
                + 0.08 * branch_norm
                + 0.10 * direct_open
                + 0.10 * distance_bonus
                + 0.06 * novelty
                - 0.10 * trail_recent
                - 0.05 * trail_long
            )
            scores.append(float(np.clip(score, 0.0, 1.0)))
            stats.append((area_norm, depth_norm, branch_norm, direct_open, trail_recent, trail_long))
        return np.array(scores, dtype=np.float32), stats

    def _corridor_score(self, move_stats):
        if not move_stats:
            return 0.0
        values = []
        for area_norm, depth_norm, branch_norm, direct_open, trail_recent, trail_long in move_stats:
            combined = 0.42 * area_norm + 0.28 * depth_norm + 0.12 * branch_norm + 0.18 * direct_open - 0.10 * trail_recent - 0.05 * trail_long
            values.append(combined)
        values.sort(reverse=True)
        topk = values[:3]
        return float(np.clip(np.mean(topk), 0.0, 1.0)) if topk else 0.0

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

    def _trail_risk_at(self, world_x, world_z, lag_lo, lag_hi, norm_dist):
        pts = list(self.recent_exact_positions)
        if len(pts) <= lag_lo:
            return 0.0
        risk = 0.0
        max_lag = min(lag_hi, len(pts) - 1)
        for lag in range(lag_lo, max_lag + 1):
            px, pz = pts[-1 - lag]
            dist = _l2(world_x, world_z, px, pz)
            risk = max(risk, max(0.0, 1.0 - dist / norm_dist))
        return float(np.clip(risk, 0.0, 1.0))

    def _movement_alignment(self, move_vec, target_vec):
        move_norm = float(np.linalg.norm(move_vec))
        target_norm = float(np.linalg.norm(target_vec))
        if move_norm < 1e-6 or target_norm < 1e-6:
            return 0.0
        cosine = float(np.dot(move_vec / move_norm, target_vec / target_norm))
        return float(np.clip(max(cosine, 0.0), 0.0, 1.0))

    def _build_local_patch(self, map_info, hero_x, hero_z, treasures, buffs, monster_positions):
        patch_size = Config.MAP_PATCH_SIZE
        radius = patch_size // 2
        ch0 = np.zeros((patch_size, patch_size), dtype=np.float32)
        ch1 = np.zeros((patch_size, patch_size), dtype=np.float32)
        ch2 = np.zeros((patch_size, patch_size), dtype=np.float32)
        ch3 = np.zeros((patch_size, patch_size), dtype=np.float32)

        for rz, dz in enumerate(range(-radius, radius + 1)):
            for cx, dx in enumerate(range(-radius, radius + 1)):
                passable = self._get_local_cell(map_info, dx, dz)
                ch0[rz, cx] = passable
                wx = int(round(hero_x + dx))
                wz = int(round(hero_z + dz))
                visits = self.visited_count.get((wx, wz), 0)
                novelty = 1.0 - min(visits / 3.0, 1.0)
                ch3[rz, cx] = novelty * passable

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

        for rz, dz in enumerate(range(-radius, radius + 1)):
            for cx, dx in enumerate(range(-radius, radius + 1)):
                world_x = hero_x + dx
                world_z = hero_z + dz
                threat = 0.0
                for mx, mz in monster_positions:
                    dist = _l2(world_x, world_z, mx, mz)
                    threat = max(threat, max(0.0, 1.0 - dist / 6.0))
                ch2[rz, cx] = threat

        patch = np.stack([ch0, ch1, ch2, ch3], axis=0)
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

    def _start_flash_eval(
        self,
        base_dist,
        base_corridor,
        base_score,
        base_trap,
        base_treasure_score,
        base_buff_count,
        cur_dist,
        cur_corridor,
        cur_score,
        cur_trap,
        cur_treasure_score,
        cur_buff_count,
    ):
        self.pending_flash_eval = {
            "frames_left": max(int(Config.FLASH_EVAL_WINDOW) - 1, 0),
            "base_dist": float(base_dist),
            "base_corridor": float(base_corridor),
            "base_score": float(base_score),
            "base_trap": float(base_trap),
            "base_treasure_score": float(base_treasure_score),
            "base_buff_count": int(base_buff_count),
            "best_dist": float(cur_dist),
            "best_corridor": float(cur_corridor),
            "best_score": float(cur_score),
            "best_trap_reduction": max(float(base_trap) - float(cur_trap), 0.0),
            "best_treasure_score": float(cur_treasure_score),
            "best_buff_count": int(cur_buff_count),
        }

    def _update_flash_eval(self, cur_dist, cur_corridor, cur_score, cur_trap, cur_treasure_score, cur_buff_count):
        if self.pending_flash_eval is None:
            return 0.0

        pending = self.pending_flash_eval
        pending["best_dist"] = max(pending["best_dist"], float(cur_dist))
        pending["best_corridor"] = max(pending["best_corridor"], float(cur_corridor))
        pending["best_score"] = max(pending["best_score"], float(cur_score))
        pending["best_trap_reduction"] = max(pending["best_trap_reduction"], float(pending["base_trap"]) - float(cur_trap))
        pending["best_treasure_score"] = max(pending["best_treasure_score"], float(cur_treasure_score))
        pending["best_buff_count"] = max(pending["best_buff_count"], int(cur_buff_count))
        pending["frames_left"] -= 1

        should_finalize = (
            pending["frames_left"] <= 0
            or cur_treasure_score > pending["base_treasure_score"]
            or cur_buff_count > pending["base_buff_count"]
        )
        if not should_finalize:
            return 0.0

        dist_gain = max(pending["best_dist"] - pending["base_dist"], 0.0)
        corridor_gain = max(pending["best_corridor"] - pending["base_corridor"], 0.0)
        score_gain = max(pending["best_score"] - pending["base_score"], 0.0)
        trap_gain = max(pending["best_trap_reduction"], 0.0)
        treasure_gain = max(pending["best_treasure_score"] - pending["base_treasure_score"], 0.0)
        buff_gain = max(pending["best_buff_count"] - pending["base_buff_count"], 0)

        good_flash = (
            treasure_gain > 0
            or buff_gain > 0
            or dist_gain > 0.03
            or corridor_gain > 0.08
            or trap_gain > 0.06
        )
        if good_flash:
            delayed_flash_reward = (
                0.08
                + 0.22 * dist_gain
                + 0.08 * corridor_gain
                + 0.06 * trap_gain
                + 0.003 * min(score_gain, 20.0)
            )
        else:
            delayed_flash_reward = -0.04

        self.pending_flash_eval = None
        return float(delayed_flash_reward)

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
        stuck_flag = 1.0 if move_delta < 0.20 else 0.0

        cur_grid = (int(round(hero_x)), int(round(hero_z)))
        prev_visit = self.visited_count.get(cur_grid, 0)
        explore_novelty = 1.0 - min(prev_visit / 4.0, 1.0)

        repeat_ratio = 0.0
        if self.recent_positions:
            repeat_cnt = sum(1 for p in self.recent_positions if p == cur_grid)
            repeat_ratio = float(np.clip(repeat_cnt / len(self.recent_positions), 0.0, 1.0))
        self.recent_positions.append(cur_grid)
        self.recent_exact_positions.append((hero_x, hero_z))

        move_vec = np.array([
            hero_x - (self.last_pos[0] if self.last_pos is not None else hero_x),
            hero_z - (self.last_pos[1] if self.last_pos is not None else hero_z),
        ], dtype=np.float32)
        recent_unique_ratio = 1.0
        if self.recent_positions:
            recent_unique_ratio = len(set(self.recent_positions)) / len(self.recent_positions)
        loop_density = float(np.clip(1.0 - recent_unique_ratio, 0.0, 1.0))

        past_dx = 0.0
        past_dz = 0.0
        anti_stall_penalty = 0.0
        dist_10 = None
        dist_20 = None
        backtrack_risk_10 = 0.0
        return_alignment_10 = 0.0
        backtrack_risk_20 = 0.0
        trail_pressure_10 = self._trail_risk_at(hero_x, hero_z, 8, 12, Config.BACKTRACK_NEAR_DIST)
        trail_pressure_20 = self._trail_risk_at(hero_x, hero_z, 16, 24, Config.BACKTRACK_FAR_DIST)
        if len(self.recent_exact_positions) > Config.STALL_WINDOW:
            old_x, old_z = list(self.recent_exact_positions)[-1 - Config.STALL_WINDOW]
            dist_10 = _l2(hero_x, hero_z, old_x, old_z)
            past_dx = _clip_signed(hero_x - old_x, 20.0)
            past_dz = _clip_signed(hero_z - old_z, 20.0)
            backtrack_risk_10 = float(np.clip(1.0 - dist_10 / Config.BACKTRACK_NEAR_DIST, 0.0, 1.0))
            return_alignment_10 = self._movement_alignment(move_vec, np.array([old_x - hero_x, old_z - hero_z], dtype=np.float32))
            if dist_10 < Config.STALL_DIST_THRESHOLD:
                anti_stall_penalty = -0.045 * (1.0 - dist_10 / Config.STALL_DIST_THRESHOLD)
        if len(self.recent_exact_positions) > Config.LOOP_WINDOW_LONG:
            old2_x, old2_z = list(self.recent_exact_positions)[-1 - Config.LOOP_WINDOW_LONG]
            dist_20 = _l2(hero_x, hero_z, old2_x, old2_z)
            backtrack_risk_20 = float(np.clip(1.0 - dist_20 / Config.BACKTRACK_FAR_DIST, 0.0, 1.0))

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
            past_dx,
            past_dz,
            explore_novelty,
            backtrack_risk_10,
            return_alignment_10,
            loop_density,
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
        valid_sorted = sorted([d for d in monster_dists if d is not None])
        second_monster_dist = valid_sorted[1] if len(valid_sorted) >= 2 else 180.0

        any_speedup = any(float(m.get("speed", 1)) >= 2.0 for m in monsters if isinstance(m, dict))
        monster_speedup_step = float(env_info.get("monster_speed", 500))
        pre_speedup_urgency = 1.0
        if monster_speedup_step > 1 and not any_speedup:
            pre_speedup_urgency = float(np.clip(1.0 - (monster_speedup_step - self.step_no) / 220.0, 0.0, 1.0))

        high_pressure = 1.0 if (second_monster_dist < 55.0 or any_speedup or self.step_no >= 0.60 * self.max_step) else 0.0

        visible_treasures, visible_buffs = self._extract_entities(frame_state)
        newly_seen_treasure = self._refresh_memory(visible_treasures, visible_buffs, env_info, hero_x, hero_z)
        treasures = self._memory_entities(visible_treasures, self.treasure_memory, "treasure", hero_x, hero_z)
        buffs = self._memory_entities(visible_buffs, self.buff_memory, "buff", hero_x, hero_z)

        treasure_items = self._build_treasure_items(treasures, hero_x, hero_z, monster_positions, high_pressure)
        locked_treasure = self._select_locked_treasure(treasure_items, high_pressure)
        treasure_feat, locked_treasure_dist, locked_treasure_priority, locked_treasure_monster_dist = self._pack_treasure_features_from_items(
            treasure_items, locked_treasure, hero_x, hero_z
        )
        buff_feat, best_buff_dist = self._pack_buff_features(buffs, hero_x, hero_z)

        move_safety_feat, move_stats = self._move_safety(map_info, hero_x, hero_z, monster_positions)
        flash_safety_feat, _ = self._flash_safety(map_info, hero_x, hero_z, monster_positions)
        corridor_score = self._corridor_score(move_stats)
        trap_pressure = self._trap_pressure(hero_x, hero_z, monster_positions)
        flash_advantage = float(np.clip(np.max(flash_safety_feat) - np.max(move_safety_feat), -1.0, 1.0))
        mean_move_safety = float(np.mean(move_safety_feat))

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
            mean_move_safety,
            trail_pressure_10,
            backtrack_risk_20,
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
        assert len(feature) == Config.DIM_OF_OBSERVATION, f"Unexpected feature dim: {len(feature)} != {Config.DIM_OF_OBSERVATION}"

        legal_action = self._legal_action_16(legal_act_raw)

        cur_min_monster_dist_norm = min(min_monster_dist / 180.0, 1.0)
        cur_second_monster_dist_norm = min(second_monster_dist / 180.0, 1.0)

        step_score = float(env_info.get("step_score", 0.0))
        treasure_score = float(env_info.get("treasure_score", 0.0))
        total_score = float(env_info.get("total_score", step_score + treasure_score))

        step_score_gain = max(step_score - self.last_step_score, 0.0)
        treasure_score_gain = max(treasure_score - self.last_treasure_score, 0.0)

        score_scale = 0.018
        step_score_reward = score_scale * step_score_gain
        treasure_score_reward = score_scale * treasure_score_gain
        survive_reward = 0.0008 + 0.0008 * high_pressure

        cur_treasure_count = int(hero.get("treasure_collected_count", env_info.get("treasures_collected", 0)))
        cur_buff_count = int(env_info.get("collected_buff", 0))
        buff_delta = max(cur_buff_count - self.last_collected_buff, 0)
        buff_reward = 0.10 * buff_delta

        monster_margin_reward = 0.0
        if self.last_min_monster_dist_norm is not None:
            dist_delta = cur_min_monster_dist_norm - self.last_min_monster_dist_norm
            if dist_delta > 0:
                monster_margin_reward = 0.04 * min(dist_delta / 0.08, 1.0) * (0.7 + 0.3 * high_pressure)
            elif dist_delta < 0:
                close_scale = 1.0 + 1.2 * max(0.22 - cur_min_monster_dist_norm, 0.0) / 0.22
                monster_margin_reward = -0.06 * min((-dist_delta) / 0.08, 1.0) * close_scale * (0.8 + 0.2 * high_pressure)
        else:
            dist_delta = 0.0

        second_monster_penalty = 0.0
        if self.last_second_monster_dist_norm is not None:
            second_delta = cur_second_monster_dist_norm - self.last_second_monster_dist_norm
            if second_delta < 0 and cur_second_monster_dist_norm < 0.20:
                second_monster_penalty = -0.03 * min((-second_delta) / 0.08, 1.0)

        treasure_track_reward = 0.0
        risky_treasure_penalty = 0.0
        if locked_treasure_dist is not None and self.last_locked_treasure_dist is not None:
            approach = self.last_locked_treasure_dist - locked_treasure_dist
            safe_enough = cur_min_monster_dist_norm > (0.12 if high_pressure < 0.5 else 0.16)
            if approach > 0 and safe_enough:
                phase_scale = 1.0 if high_pressure < 0.5 else 0.35
                treasure_track_reward = 0.020 * phase_scale * locked_treasure_priority * float(np.clip(approach / 5.0, 0.0, 1.0))
            elif approach > 0 and locked_treasure_monster_dist < 14.0 and cur_min_monster_dist_norm < 0.20:
                risky_treasure_penalty = -0.025 * min(approach / 5.0, 1.0)

        map_shaping_reward = 0.0
        if cur_min_monster_dist_norm > (0.12 if high_pressure < 0.5 else 0.16):
            if prev_visit == 0:
                map_shaping_reward += 0.006 if high_pressure < 0.5 else 0.003
            elif prev_visit <= 2 and high_pressure < 0.5:
                map_shaping_reward += 0.002
        map_shaping_reward += 0.008 * corridor_score * (0.7 + 0.3 * high_pressure)
        map_shaping_reward += 0.004 * mean_move_safety
        if newly_seen_treasure > 0 and high_pressure < 0.5:
            map_shaping_reward += 0.008 * min(newly_seen_treasure, 2)

        danger_penalty = 0.0
        danger_threshold = 0.12 if high_pressure < 0.5 else 0.16
        if cur_min_monster_dist_norm < danger_threshold:
            danger_penalty -= 0.05 * (danger_threshold - cur_min_monster_dist_norm) / danger_threshold
        if corridor_score < 0.26 and cur_min_monster_dist_norm < 0.24:
            danger_penalty -= 0.03 * (0.26 - corridor_score) / 0.26
        danger_penalty -= 0.035 * trap_pressure * (0.4 + 0.6 * high_pressure)

        anti_loop_penalty = 0.0
        if last_action != -1 and last_action < 8 and move_delta < 0.20:
            anti_loop_penalty -= 0.05
            if repeat_ratio > 0.45:
                anti_loop_penalty -= 0.02
        if repeat_ratio > 0.60:
            anti_loop_penalty -= 0.020 * (repeat_ratio - 0.60) / 0.40
        anti_loop_penalty += anti_stall_penalty

        anti_backtrack_penalty = 0.0
        if backtrack_risk_10 > 0.0:
            anti_backtrack_penalty -= 0.030 * backtrack_risk_10 * (0.55 + 0.45 * loop_density)
            anti_backtrack_penalty -= 0.028 * return_alignment_10 * (0.6 + 0.4 * high_pressure)
        if backtrack_risk_20 > 0.0 and repeat_ratio > 0.35:
            anti_backtrack_penalty -= 0.016 * backtrack_risk_20 * (0.5 + 0.5 * repeat_ratio)
        if trail_pressure_10 > 0.55 and cur_min_monster_dist_norm < 0.28:
            anti_backtrack_penalty -= 0.018 * trail_pressure_10 * (0.5 + 0.5 * high_pressure)

        delayed_flash_reward = self._update_flash_eval(
            cur_dist=cur_min_monster_dist_norm,
            cur_corridor=corridor_score,
            cur_score=total_score,
            cur_trap=trap_pressure,
            cur_treasure_score=treasure_score,
            cur_buff_count=cur_buff_count,
        )

        if last_action >= 8:
            self._start_flash_eval(
                base_dist=self.last_min_monster_dist_norm if self.last_min_monster_dist_norm is not None else cur_min_monster_dist_norm,
                base_corridor=self.last_corridor_score,
                base_score=self.last_total_score,
                base_trap=self.last_trap_pressure,
                base_treasure_score=self.last_treasure_score,
                base_buff_count=self.last_collected_buff,
                cur_dist=cur_min_monster_dist_norm,
                cur_corridor=corridor_score,
                cur_score=total_score,
                cur_trap=trap_pressure,
                cur_treasure_score=treasure_score,
                cur_buff_count=cur_buff_count,
            )

        reward_components = {
            "step_score_reward": float(step_score_reward),
            "treasure_score_reward": float(treasure_score_reward),
            "survive_reward": float(survive_reward),
            "buff_reward": float(buff_reward),
            "monster_margin_reward": float(monster_margin_reward),
            "second_monster_penalty": float(second_monster_penalty),
            "treasure_track_reward": float(treasure_track_reward),
            "risky_treasure_penalty": float(risky_treasure_penalty),
            "map_shaping_reward": float(map_shaping_reward),
            "danger_penalty": float(danger_penalty),
            "anti_loop_penalty": float(anti_loop_penalty),
            "anti_backtrack_penalty": float(anti_backtrack_penalty),
            "delayed_flash_reward": float(delayed_flash_reward),
        }
        reward_value = float(sum(reward_components.values()))
        reward_components["total_reward"] = reward_value
        self.last_reward_components = reward_components

        self.last_min_monster_dist_norm = cur_min_monster_dist_norm
        self.last_second_monster_dist_norm = cur_second_monster_dist_norm
        self.last_locked_treasure_dist = locked_treasure_dist
        self.last_locked_treasure_priority = locked_treasure_priority
        self.last_total_score = total_score
        self.last_step_score = step_score
        self.last_treasure_score = treasure_score
        self.last_treasure_collected = cur_treasure_count
        self.last_collected_buff = cur_buff_count
        self.last_flash_count = int(env_info.get("flash_count", 0))
        self.last_pos = (hero_x, hero_z)
        self.last_corridor_score = corridor_score
        self.last_trap_pressure = trap_pressure
        self.last_action = last_action
        self.visited_count[cur_grid] = prev_visit + 1

        reward = [reward_value]
        return feature, legal_action, reward
