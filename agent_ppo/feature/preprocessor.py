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

import json
import heapq
import numpy as np
from collections import deque

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
MAP_SIZE_INT = 128
LOCAL_MAP_SIZE = 21
LOCAL_MAP_HALF = 10
VIEW_MAP_SIZE = 21

# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 200.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0

# 官方 monster / organ relative direction 映射
# 0=重叠/无效，1=东，2=东北，3=北，4=西北，5=西，6=西南，7=南，8=东南
DIR9_TO_VEC = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),                          # 东
    2: (1 / np.sqrt(2), -1 / np.sqrt(2)),   # 东北
    3: (0.0, -1.0),                         # 北
    4: (-1 / np.sqrt(2), -1 / np.sqrt(2)),  # 西北
    5: (-1.0, 0.0),                         # 西
    6: (-1 / np.sqrt(2), 1 / np.sqrt(2)),   # 西南
    7: (0.0, 1.0),                          # 南
    8: (1 / np.sqrt(2), 1 / np.sqrt(2)),    # 东南
}

# 8方向（与 DIR9_TO_VEC 去掉 0 后保持一致）：东、东北、北、西北、西、西南、南、东南
DIR8 = [
    (1, 0),    # 东
    (1, -1),   # 东北
    (0, -1),   # 北
    (-1, -1),  # 西北
    (-1, 0),   # 西
    (-1, 1),   # 西南
    (0, 1),    # 南
    (1, 1),    # 东南
]

# 24 个扫描角：0, 15, 30, ..., 345
SCAN_ANGLES_DEG = list(range(0, 360, 15))

# 视野外怪物预测：每 N 帧重算一次 A* 路径
MONSTER_ASTAR_REPLAN_INTERVAL = 1

# Curriculum / 课程训练切换
CURRICULUM_STAGE2_EPISODE = 2000
SCORE_GAIN_WEIGHT_STAGE1 = 0.50
SCORE_GAIN_WEIGHT_STAGE2 = 0.70
SURVIVAL_WEIGHT_STAGE1 = 0.03
SURVIVAL_WEIGHT_STAGE2 = 0.08

def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0

def _clip_window(x0, x1, y0, y1, size=MAP_SIZE_INT):
    x0 = max(0, x0)
    x1 = min(size, x1)
    y0 = max(0, y0)
    y1 = min(size, y1)
    return x0, x1, y0, y1

def _bucketize_left(x, num_bins, x_min=0.0, x_max=1.0):
    """
    将连续值桶化，并映射到所在桶的左端点。
    例如:
        x in [0.0, 0.2) -> 0.0
        x in [0.2, 0.4) -> 0.2

    参数:
        x: float 或 numpy array
        num_bins: 桶数，例如 5
        x_min, x_max: 取值范围
    """

    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, x_min, x_max)

    step = (x_max - x_min) / float(num_bins)
    # 处理 x == x_max 的边界
    idx = np.floor((x - x_min) / step).astype(np.int32)
    idx = np.clip(idx, 0, num_bins - 1)

    return x_min + idx * step

def _distance_bucket_to_radius(dist_bucket):
    """
    将 hero_l2_distance 桶编号(0~5)估算成一个代表距离。
    桶定义：
        0=[0,30), 1=[30,60), 2=[60,90), 3=[90,120), 4=[120,150), 5=[150,180)
    这里取各桶中点作为估计半径，更稳一些。
    """
    dist_bucket = int(np.clip(dist_bucket, 0, 5))
    bucket_mid = {
        0: 15.0,
        1: 45.0,
        2: 75.0,
        3: 105.0,
        4: 135.0,
        5: 165.0,
    }
    return bucket_mid[dist_bucket]

def _estimate_monster_pos(hero_x, hero_z, monster):
    """
    返回怪物估计位置 (mx, mz)，整数网格坐标。
    规则：
    - 视野内：直接用精确 pos
    - 视野外：用 hero_relative_direction + hero_l2_distance 估算
    """
    is_in_view = int(monster.get("is_in_view", 0))

    if is_in_view and ("pos" in monster) and (monster["pos"] is not None):
        mx = int(monster["pos"]["x"])
        mz = int(monster["pos"]["z"])
        return mx, mz

    dir_idx = int(monster.get("hero_relative_direction", 0))
    dir_x, dir_z = DIR9_TO_VEC.get(dir_idx, (0.0, 0.0))

    dist_bucket = int(monster.get("hero_l2_distance", 5))
    est_radius = _distance_bucket_to_radius(dist_bucket)

    mx = int(round(hero_x + dir_x * est_radius))
    mz = int(round(hero_z + dir_z * est_radius))
    return mx, mz

def _paint_square(mask, center_i, center_j, radius=1, value=1.0):
    h, w = mask.shape
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ii = center_i + di
            jj = center_j + dj
            if 0 <= ii < h and 0 <= jj < w:
                mask[ii, jj] = value

def _paint_path(layer, path, gx0, gy0, path_value=0.35, radius=0):
    """
    将全局路径 path=[(x,z), ...] 画到局部 layer 上。
    - 中间路径点使用较低强度
    - 路径起点/终点可使用更高强度
    """
    if path is None or len(path) == 0:
        return

    h, w = layer.shape
    for idx, (px, pz) in enumerate(path):
        li = int(px - gx0)
        lj = int(pz - gy0)
        if not (0 <= li < h and 0 <= lj < w):
            continue

        value = path_value * ()
        if radius <= 0:
            layer[li, lj] = value
        else:
            _paint_square(layer, li, lj, radius=radius, value=value)

def _paint_recent_positions_on_passable(layer, positions, gx0, gy0):
    """
    将最近若干帧自身轨迹画到 passable layer 上。
    采用递增强度，越新的位置越明显。
    """
    if positions is None:
        return

    positions = list(positions)
    if not positions:
        return

    total = len(positions)
    h, w = layer.shape
    for idx, (px, pz) in enumerate(positions):
        li = int(px - gx0)
        lj = int(pz - gy0)
        if not (0 <= li < h and 0 <= lj < w):
            continue
        # 仅在可通行位置上覆盖，避免把障碍误画亮
        if layer[li, lj] <= 0.0:
            continue
        if total == 1:
            value = 0.6
        else:
            value = 0.8 - 0.6 * (idx / float(total - 1))
        layer[li, lj] = value

class Preprocessor:
    def __init__(self, logger=None):
        self.logger = logger
        self.total_train_steps = 0
        self.curriculum_episode = 0
        self.reset()

    def set_curriculum_episode(self, episode_cnt):
        """Set current training episode for stage-aware curriculum and rewards."""
        self.curriculum_episode = int(max(0, episode_cnt))

    def reset(self):
        self.step_no = 0
        self.max_step = 200

        self.last_monster_dist_norm_1 = -1.0
        self.last_monster_dist_norm_2 = -1.0
        self.last_monster_invisible_1 = False
        self.last_monster_invisible_2 = False

        self.last_total_score = 0.0
        self.last_flash_count = 0
        self.last_is_dangerous = False

        self.pos_history = deque(maxlen=8)
        self.abb_safe_score = 4.0

        self.last_treasure_dist_norm = -1.0
        self.last_buff_dist_norm = -1.0

        # 全局物件记忆
        self.treasure_memory = {}
        self.buff_memory = {}
        self.last_collected_buff = 0
        self.buff_refresh_time = 200

        # ========= 两层全局记忆 =========
        # 第一层：可通行地图：1=可走, 0=不能走/未知
        self.passable_map = np.zeros((MAP_SIZE_INT, MAP_SIZE_INT), dtype=np.uint8)
        # 第二层：可见性地图：1=已知, 0=未知
        self.visibility_map = np.zeros((MAP_SIZE_INT, MAP_SIZE_INT), dtype=np.uint8)
        self.monster_predicted_paths = [[], []]
        self.monster_replan_counters = [0, 0]

        # 怪物视野外预测位置缓存
        self.last_seen_monster_pos = [None, None]
        self.predicted_monster_pos = [None, None]
        self.monster_prediction_error_sum = 0.0
        self.monster_prediction_error_count = 0
        self.monster_prediction_error_avg = 0.0

    def update_global_maps(self, hero_x, hero_y, map_info):
        """
        将 21x21 局部视野拼接到全局 memory。
        约定：
            passable_map: 1=可走, 0=障碍/未知
            visibility_map: 1=已知, 0=未知
        返回：
            local global window: (x0, x1, y0, y1)
        """
        h = min(LOCAL_MAP_SIZE, len(map_info))
        w = min(LOCAL_MAP_SIZE, len(map_info[0]))

        x0 = hero_x - LOCAL_MAP_HALF
        y0 = hero_y - LOCAL_MAP_HALF
        x1 = x0 + h
        y1 = y0 + w

        gx0, gx1, gy0, gy1 = _clip_window(x0, x1, y0, y1, MAP_SIZE_INT)
        newly_discovered_passable_count = 0

        for i in range(h):
            for j in range(w):
                gx = x0 + j
                gy = y0 + i
                if not (0 <= gx < MAP_SIZE_INT and 0 <= gy < MAP_SIZE_INT):
                    continue

                # 文档定义：1=可通行，0=障碍
                visible_val = 1
                passable_val = 1 if int(map_info[i][j]) != 0 else 0

                if int(self.visibility_map[gx, gy]) == 0 and int(map_info[i][j]) > 0:
                    newly_discovered_passable_count += 1

                self.visibility_map[gx, gy] = visible_val
                self.passable_map[gx, gy] = passable_val

        return gx0, gx1, gy0, gy1, newly_discovered_passable_count


    def _is_global_passable(self, x, z):
        """Check whether global memory marks (x, z) as passable.

        检查全局记忆中的 (x, z) 是否可通行。
        """
        x = int(x)
        z = int(z)
        if not (0 <= x < MAP_SIZE_INT and 0 <= z < MAP_SIZE_INT):
            return False
        return bool(self.passable_map[x, z] > 0)

    def _flash_landing_offset(self, hero_x, hero_z, dx, dz):
        """Find the farthest valid flash landing cell in the given direction.

        在给定方向上，从远到近寻找最远的合法闪现落点。
        闪现允许穿墙，因此只检查落点本身：
        - 不能出界
        - 不能落在“已知墙”上
        - 未知区域视为可尝试，由官方 legal_action 再做最终兜底
        返回: (offset_x, offset_z, ok)
        """
        hero_x = int(hero_x)
        hero_z = int(hero_z)
        dx = int(dx)
        dz = int(dz)

        diagonal = (dx != 0 and dz != 0)
        max_dist = 8 if diagonal else 10

        for step in range(max_dist, 0, -1):
            nx = hero_x + dx * step
            nz = hero_z + dz * step
            if not (0 <= nx < MAP_SIZE_INT and 0 <= nz < MAP_SIZE_INT):
                continue
            if self._is_known_wall(nx, nz):
                continue
            return dx * step, dz * step, True
        return 0, 0, False
    
    def _is_known_wall(self, x, z):
        """
        已知墙：visibility=1 且 passable=0
        """
        if not (0 <= x < MAP_SIZE_INT and 0 <= z < MAP_SIZE_INT):
            return True  # 出界直接视为墙，更保守
        return bool(self.visibility_map[x, z] > 0 and self.passable_map[x, z] == 0)

    def _parse_legal_action_raw(self, legal_act_raw):
        """Parse env legal_action into a 16D binary mask."""
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

    def _build_processed_legal_action(self, hero_x, hero_z, legal_action_mask):
        """Build a model-facing 16D legal-action feature and a hard 16D mask.

        - 0~7: 移动。若下一格确定撞墙/出界，则直接置 0。
        - 8~15: 闪现。允许穿墙，但落点不能是墙；分数按可闪出的距离线性归一化。
        返回:
            legal_action_feat: float32[16]，给模型作为输入
            legal_action_mask: int[16]，给 PPO action masking 使用
        """
        hero_x = int(hero_x)
        hero_z = int(hero_z)

        processed_mask = [int(v) for v in legal_action_mask]
        processed_feat = np.zeros(16, dtype=np.float32)

        for i, (dx, dz) in enumerate(DIR8):
            if processed_mask[i] <= 0:
                continue

            nx = hero_x + dx
            nz = hero_z + dz
            blocked = (not (0 <= nx < MAP_SIZE_INT and 0 <= nz < MAP_SIZE_INT)) or self._is_known_wall(nx, nz)
            if blocked:
                processed_mask[i] = 0
                processed_feat[i] = 0.0
            else:
                processed_feat[i] = 1.0

        for i, (dx, dz) in enumerate(DIR8, start=8):
            if processed_mask[i] <= 0:
                continue

            off_x, off_z, ok = self._flash_landing_offset(hero_x, hero_z, dx, dz)
            if not ok:
                processed_mask[i] = 0
                processed_feat[i] = 0.0
                continue

            dist = float(np.hypot(off_x, off_z))
            max_dist = 8.0 if (dx != 0 and dz != 0) else 10.0
            processed_feat[i] = _norm(dist, max_dist)

        if sum(processed_mask) == 0:
            processed_mask = [1] * 16
            processed_feat[:] = 1.0

        return processed_feat, processed_mask

    def _is_unknown(self, x, z):
        """
        未知区域：visibility=0
        """
        if not (0 <= x < MAP_SIZE_INT and 0 <= z < MAP_SIZE_INT):
            return False
        return bool(self.visibility_map[x, z] == 0)


    def _compute_near_wall_penalty(self, hero_x, hero_z, search_radius=2):
        """
        在 hero 周围 (2*search_radius+1)x(2*search_radius+1) 小窗口内，
        计算到最近“已知墙”的欧氏距离，并返回靠墙惩罚。
        惩罚规则：
        - min_dist <= 1.0 -> 1.0
        - 1.0 < min_dist <= 2.0 -> 0.2
        - 其它 -> 0.0
        """
        hero_x = int(hero_x)
        hero_z = int(hero_z)

        min_dist = None
        for dz in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                if dx == 0 and dz == 0:
                    continue

                x = hero_x + dx
                z = hero_z + dz

                if not self._is_known_wall(x, z):
                    continue

                dist = float(np.hypot(dx, dz))
                if min_dist is None or dist < min_dist:
                    min_dist = dist

        if min_dist is None:
            return 0.0

        if min_dist <= 1.0 + 1e-6:
            return -1.0
        elif min_dist <= 3.0 + 1e-6:
            return -np.exp(-np.log(5.0) * (min_dist - 1.0))
        return 0.0

    def _ray_collision_score(self, start_x, start_z, angle_deg, max_len=VIEW_MAP_SIZE/2, step_size=1.0):
        """
        从 (start_x, start_z) 朝 angle_deg 方向发射一条射线。
        - 若在已知区域内撞到墙，则 score = dist / max_len，越晚撞墙分越高
        - 若遇到未知区域，则返回 1.0（未知区域不判危险）
        - 若一直到 max_len 都没撞墙，则返回 1.0
        - 若射线走出地图边界，则按“撞边界”处理，也返回 dist / max_len
        """
        theta = np.deg2rad(angle_deg)
        dx = np.cos(theta)
        dz = -np.sin(theta)   # 与 DIR9_TO_VEC 的 z 方向保持一致：北是负 z

        dist = step_size
        while dist <= max_len:
            x = int(round(start_x + dx * dist))
            z = int(round(start_z + dz * dist))

            # 出界：按在该距离处碰壁处理
            if not (0 <= x < MAP_SIZE_INT and 0 <= z < MAP_SIZE_INT):
                return float(np.clip(dist / max_len, 0.0, 1.0))

            # 未知区域：不继续往前判，直接认为安全
            if self._is_unknown(x, z):
                return 1.0

            # 已知墙：按撞墙距离线性给分
            if self._is_known_wall(x, z):
                return float(np.clip(dist / max_len, 0.0, 1.0))

            dist += step_size
        return 1.0

    def _angle_diff_deg(self, a, b):
        """
        返回两个角度的最小夹角，范围 [0, 180]
        """
        d = abs(a - b) % 360
        return min(d, 360 - d)

    def _dir8_to_angle_deg(self, dx, dz):
        """
        DIR8 -> 角度
        约定：
        东=0, 东北=45, 北=90, 西北=135, 西=180, 西南=225, 南=270, 东南=315
        注意这里 dz 轴向下为正，所以北对应 dz=-1
        """
        angle = np.degrees(np.arctan2(-dz, dx))
        if angle < 0:
            angle += 360
        return float(angle)
    
    def _compute_global_rays(self, start_x, start_z, max_len=18, step_size=1.0):
        """
        从同一个起点统一发射全局 rays（0,15,30,...,345），每根只算一次。
        返回：
            [
                {"angle": 0.0, "score": 1.0},
                {"angle": 15.0, "score": 1.0},
                ...
            ]
        """
        start_x = int(start_x)
        start_z = int(start_z)

        rays = []
        for angle_deg in SCAN_ANGLES_DEG:
            ray_score = self._ray_collision_score(
                start_x=start_x,
                start_z=start_z,
                angle_deg=angle_deg,
                max_len=max_len,
                step_size=step_size,
            )
            rays.append({
                "angle": float(angle_deg),
                "score": float(ray_score),
            })
        return rays
    
    def _score_ray_collision_direction_from_rays(self, move_angle, rays, angle_window=30.0):
        """
        用统一的一组全局 rays，对某个 move_angle 打分。
        只看 ±angle_window 内的 rays，按角度差线性加权。
        """
        weighted_sum = 0.0
        weight_total = 0.0
        matched_rays = []

        for ray in rays:
            ray_angle = float(ray["angle"])
            ray_score = float(ray["score"])

            diff = self._angle_diff_deg(move_angle, ray_angle)
            if diff > angle_window:
                continue

            weight = 1.0 - diff / angle_window

            weighted_sum += weight * ray_score
            weight_total += weight

            matched_rays.append({
                "angle": ray_angle,
                "score": ray_score,
            })

        if weight_total <= 1e-6:
            move_score = 0.0
        else:
            move_score = float(np.clip(weighted_sum / weight_total, 0.0, 1.0))

        return move_score, matched_rays

    def _ray_collision_direction_scores(self, hero_x, hero_z, return_debug=False):
        """
        基于一组全局 rays 的 8 方向 ray collision 分数。
        1. 从 hero 当前位置统一发射一组全局 rays（0,15,...,345），每根只算一次
        2. 对每个动作方向：
        - 若下一步不可走，score=0
        - 否则用该方向去匹配全局 rays 中 ±30° 内的那些 ray
        - 按角度差加权聚合成该方向的 move score
        """
        hero_x = int(hero_x)
        hero_z = int(hero_z)

        # 先统一计算一组全局 rays
        global_rays = self._compute_global_rays(
            start_x=hero_x,
            start_z=hero_z,
            max_len=18,
            step_size=1.0,
        )

        ray_collision_scores = []
        debug_infos = []

        for dx, dz in DIR8:
            nx = hero_x + dx
            nz = hero_z + dz

            action_debug = {
                "score": 0.0,
                "rays": [],
            }

            # 下一步本身不能走，则该方向直接为 0
            if not self._is_global_passable(nx, nz):
                ray_collision_scores.append(0.0)
                debug_infos.append(action_debug)
                continue

            move_angle = self._dir8_to_angle_deg(dx, dz)
            ray_collision_score, matched_rays = self._score_ray_collision_direction_from_rays(
                move_angle=move_angle,
                rays=global_rays,
                angle_window=30.0,
            )

            action_debug["score"] = ray_collision_score
            action_debug["rays"] = matched_rays

            ray_collision_scores.append(ray_collision_score)
            debug_infos.append(action_debug)

        ray_collision_scores = np.asarray(ray_collision_scores, dtype=np.float32)

        if return_debug:
            return ray_collision_scores, debug_infos, global_rays
        return ray_collision_scores


    def _extract_local_passable_patch(self, map_info):
        """
        从当前 21x21 视野中提取局部可通行二值图。
        约定：非 0 为可通行，0 为障碍。
        """
        local_passable = np.zeros((LOCAL_MAP_SIZE, LOCAL_MAP_SIZE), dtype=np.uint8)
        if map_info is None:
            return local_passable

        h = min(LOCAL_MAP_SIZE, len(map_info))
        w = min(LOCAL_MAP_SIZE, len(map_info[0])) if h > 0 else 0
        for i in range(h):
            for j in range(w):
                local_passable[i, j] = 1 if int(map_info[i][j]) != 0 else 0
        return local_passable

    def _mask_monster_danger_zone_local(self, local_passable, monsters, hero_pos, radius=3):
        """
        在局部 21x21 邻域内，如果怪物在视野内，
        则将其周围 7x7 区域置 0，再用于 boundary cluster 计算。
        """
        masked = np.array(local_passable, copy=True)
        hero_x = int(hero_pos["x"])
        hero_z = int(hero_pos["z"])

        for monster in monsters[:2]:
            if int(monster.get("is_in_view", 0)) <= 0:
                continue
            pos = monster.get("pos", {}) or {}
            mx = int(pos.get("x", hero_x))
            mz = int(pos.get("z", hero_z))

            lx = mx - hero_x + LOCAL_MAP_HALF
            ly = mz - hero_z + LOCAL_MAP_HALF
            if not (0 <= lx < LOCAL_MAP_SIZE and 0 <= ly < LOCAL_MAP_SIZE):
                continue

            x0 = max(0, lx - radius)
            x1 = min(LOCAL_MAP_SIZE, lx + radius + 1)
            y0 = max(0, ly - radius)
            y1 = min(LOCAL_MAP_SIZE, ly + radius + 1)
            masked[y0:y1, x0:x1] = 0

        return masked

    def _get_boundary_passable_points(self, local_passable):
        """
        提取 21x21 局部图边框上的所有可通行点。
        坐标采用 (x, y) = (列, 行)。
        """
        pts = []
        h, w = local_passable.shape

        for x in range(w):
            if local_passable[0, x] > 0:
                pts.append((x, 0))
            if h > 1 and local_passable[h - 1, x] > 0:
                pts.append((x, h - 1))

        for y in range(1, h - 1):
            if local_passable[y, 0] > 0:
                pts.append((0, y))
            if w > 1 and local_passable[y, w - 1] > 0:
                pts.append((w - 1, y))

        return pts

    def _cluster_boundary_points(self, boundary_pts):
        """
        对边框可通行点按 8 邻接进行聚类。
        boundary_pts 中坐标采用 (x, y)。
        """
        if not boundary_pts:
            return []

        pts_set = set(boundary_pts)
        visited = set()
        clusters = []

        for seed in boundary_pts:
            if seed in visited:
                continue

            queue = deque([seed])
            visited.add(seed)
            cluster = []

            while queue:
                x, y = queue.popleft()
                cluster.append((x, y))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nxt = (x + dx, y + dy)
                        if nxt in pts_set and nxt not in visited:
                            visited.add(nxt)
                            queue.append(nxt)

            clusters.append(cluster)

        return clusters

    def _compute_local_connected_mask(self, local_passable, start_x=LOCAL_MAP_HALF, start_y=LOCAL_MAP_HALF):
        """
        在 21x21 局部图中，从 agent 中心位置出发做 8 邻接 BFS，
        返回与 agent 连通的可通行 mask。
        坐标采用 (x, y) = (列, 行)。
        """
        h, w = local_passable.shape
        connected = np.zeros((h, w), dtype=np.uint8)

        if not (0 <= start_x < w and 0 <= start_y < h):
            return connected
        if local_passable[start_y, start_x] == 0:
            return connected

        queue = deque([(start_x, start_y)])
        connected[start_y, start_x] = 1

        while queue:
            x, y = queue.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    if connected[ny, nx] > 0:
                        continue
                    if local_passable[ny, nx] == 0:
                        continue
                    connected[ny, nx] = 1
                    queue.append((nx, ny))

        return connected

    def _compute_boundary_cluster_direction_scores(self, local_passable):
        """
        基于 21x21 局部图：
        1. 提取边框可通行点
        2. 对边框点做 8 邻接聚类
        3. 仅保留与 agent 中心连通的簇
        4. 将连通簇中心对 8 个动作方向做余弦相似度投影，得到 8 维方向分数
        5. 若连通的边界开口簇数量 <= 1，则记为危险局势
        """
        boundary_pts = self._get_boundary_passable_points(local_passable)
        clusters = self._cluster_boundary_points(boundary_pts)
        connected_mask = self._compute_local_connected_mask(local_passable)

        agent_x = float(LOCAL_MAP_HALF)
        agent_y = float(LOCAL_MAP_HALF)
        dir_scores = np.zeros(len(DIR8), dtype=np.float32)

        connected_clusters = []
        total_connected_weight = 0.0

        for cluster in clusters:
            is_connected = any(connected_mask[y, x] > 0 for x, y in cluster)
            if not is_connected:
                continue

            xs = np.asarray([p[0] for p in cluster], dtype=np.float32)
            ys = np.asarray([p[1] for p in cluster], dtype=np.float32)
            cx = float(xs.mean())
            cy = float(ys.mean())

            vx = cx - agent_x
            vy = cy - agent_y
            dist = float(np.hypot(vx, vy))
            if dist <= 1e-6:
                continue

            uvx = vx / dist
            uvy = vy / dist
            size_weight = float(np.clip(len(cluster) / 3.0, 0.0, 1.0))
            if size_weight <= 1e-6:
                continue

            total_connected_weight += size_weight
            connected_clusters.append({
                "center": (cx, cy),
                "size": len(cluster),
                "dist": dist,
            })

            for i, (dx, dz) in enumerate(DIR8):
                dir_vec = np.asarray([float(dx), float(dz)], dtype=np.float32)
                dir_norm = float(np.linalg.norm(dir_vec))
                if dir_norm <= 1e-6:
                    continue
                align = float((uvx * dir_vec[0] + uvy * dir_vec[1]) / dir_norm)
                if align > 0.0:
                    dir_scores[i] += align * size_weight

        if total_connected_weight > 1e-6:
            dir_scores /= total_connected_weight

        dir_scores = np.clip(dir_scores, 0.0, 1.0).astype(np.float32)
        is_dangerous = len(connected_clusters) <= 1
        return dir_scores, {
            "boundary_pts": boundary_pts,
            "clusters": clusters,
            "connected_clusters": connected_clusters,
            "connected_opening_count": len(connected_clusters),
            "is_dangerous": is_dangerous,
            "masked_local_passable": np.array(local_passable, copy=True),
        }

    def _select_reachable_opposite_boundary_cluster(self, connected_clusters, monster_feat, angle_cos_threshold=0.0):
        """
        从所有“能到达”的 boundary clusters（即 connected_clusters）中，
        选择一个与“怪物方向相反”最一致的 cluster。

        monster_feat: [is_in_view, speed, rel_x, rel_z, dist_norm, dir_x, dir_z]
        返回：
            {
                "center": (cx, cy),
                "size": size,
                "dist": agent 到该 cluster center 的局部距离,
                "align": 与“远离怪物方向”的余弦相似度,
            }
            或 None
        """
        if not connected_clusters:
            return None

        mdx = float(monster_feat[5])
        mdz = float(monster_feat[6])
        mnorm = float(np.hypot(mdx, mdz))
        if mnorm <= 1e-6:
            return None

        # 远离怪物方向 = -(hero->monster)
        away_x = -mdx / mnorm
        away_z = -mdz / mnorm

        agent_x = float(LOCAL_MAP_HALF)
        agent_y = float(LOCAL_MAP_HALF)

        best = None
        best_align = -1e9

        for c in connected_clusters:
            cx, cy = c["center"]
            vx = float(cx - agent_x)
            vy = float(cy - agent_y)
            dist = float(np.hypot(vx, vy))
            if dist <= 1e-6:
                continue

            uvx = vx / dist
            uvy = vy / dist
            align = float(uvx * away_x + uvy * away_z)

            if align > best_align:
                best_align = align
                best = {
                    "center": (float(cx), float(cy)),
                    "size": int(c["size"]),
                    "dist": float(dist),
                    "align": float(align),
                }

        if best is None:
            return None
        if best["align"] < angle_cos_threshold:
            return None
        return best

    def _action_to_dir_vec(self, action_idx):
        """
        将动作索引映射成 8 方向单位向量。
        - 0~7: 普通移动
        - 8~15: 闪现，方向仍按 action-8 映射
        返回: (ux, uz) 或 None
        """
        if action_idx is None:
            return None

        action_idx = int(action_idx)
        if 0 <= action_idx < 8:
            dx, dz = DIR8[action_idx]
        elif 8 <= action_idx < 16:
            dx, dz = DIR8[action_idx - 8]
        else:
            return None

        norm = float(np.hypot(dx, dz))
        if norm <= 1e-6:
            return None
        return float(dx / norm), float(dz / norm)

    def _astar_next_step_towards(self, start, goal):
        """
        在已知可通行区域上，用 A* 从 start 朝 goal 导航。
        未知区域视作阻碍；若存在路径，返回从 start 出发的下一步位置。
        """
        if start is None or goal is None:
            return None

        sx, sz = int(start[0]), int(start[1])
        gx, gz = int(goal[0]), int(goal[1])

        if not (0 <= sx < MAP_SIZE_INT and 0 <= sz < MAP_SIZE_INT):
            return None
        if not (0 <= gx < MAP_SIZE_INT and 0 <= gz < MAP_SIZE_INT):
            return None
        if not self._is_global_passable(sx, sz):
            return None
        if not self._is_global_passable(gx, gz):
            return None
        if (sx, sz) == (gx, gz):
            return (sx, sz)

        def heuristic(x, z):
            return max(abs(x - gx), abs(z - gz))

        open_heap = []
        heapq.heappush(open_heap, (heuristic(sx, sz), 0.0, (sx, sz)))
        came_from = {}
        g_score = {(sx, sz): 0.0}
        closed = set()

        while open_heap:
            _, cur_g, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)

            if cur == (gx, gz):
                break

            cx, cz = cur
            for dx, dz in DIR8:
                nx = cx + dx
                nz = cz + dz
                nxt = (nx, nz)
                if nxt in closed:
                    continue
                if not self._is_global_passable(nx, nz):
                    continue

                step_cost = 1.41421356 if (dx != 0 and dz != 0) else 1.0
                new_g = cur_g + step_cost
                if new_g + 1e-6 < g_score.get(nxt, float('inf')):
                    g_score[nxt] = new_g
                    came_from[nxt] = cur
                    heapq.heappush(open_heap, (new_g + heuristic(nx, nz), new_g, nxt))

        if (gx, gz) not in came_from and (gx, gz) != (sx, sz):
            return None

        cur = (gx, gz)
        while came_from.get(cur) != (sx, sz):
            parent = came_from.get(cur)
            if parent is None:
                return None
            cur = parent
        return cur

    def _update_predicted_monster_pos(self, idx, monster, hero_pos, env_info):
        """
        视野内：
        - 直接使用真实位置
        - 每帧基于真实位置到 hero 重新计算一次 A* 路径，并缓存到 monster layer
        视野外：
        - 从上一预测位置/最后可见位置出发
        - 每 N 帧重算一次 A*，中间沿缓存路径每帧推进若干步
        """
        is_in_view = int(monster.get("is_in_view", 0)) > 0
        hero_xy = (int(hero_pos["x"]), int(hero_pos["z"]))

        if is_in_view and ("pos" in monster) and (monster["pos"] is not None):
            mx = int(monster["pos"]["x"])
            mz = int(monster["pos"]["z"])
            self.last_seen_monster_pos[idx] = (mx, mz)
            self.predicted_monster_pos[idx] = (mx, mz)

            path = self._astar_path((mx, mz), hero_xy)
            self.monster_predicted_paths[idx] = path if path else [(mx, mz)]
            self.monster_replan_counters[idx] = max(1, int(MONSTER_ASTAR_REPLAN_INTERVAL))
            return mx, mz

        start_pos = self.predicted_monster_pos[idx]
        if start_pos is None:
            start_pos = self.last_seen_monster_pos[idx]

        step_count = monster.get("speed", 1)

        predicted = self._maybe_replan_monster_path(
            idx=idx,
            start_pos=start_pos,
            hero_pos=hero_xy,
            step_count=step_count,
        )

        if predicted is not None:
            self.predicted_monster_pos[idx] = predicted
            return int(predicted[0]), int(predicted[1])

        est_mx, est_mz = _estimate_monster_pos(hero_pos["x"], hero_pos["z"], monster)
        self.predicted_monster_pos[idx] = (int(est_mx), int(est_mz))
        self.monster_predicted_paths[idx] = [(int(est_mx), int(est_mz))]
        self.monster_replan_counters[idx] = 0
        return int(est_mx), int(est_mz)

    def _update_organ_memory(self, env_info, organs, hero_pos):
        self.buff_refresh_time = int(env_info.get("buff_refresh_time", self.buff_refresh_time))
        remaining_treasures = {int(x) for x in env_info.get("treasure_id", [])}

        for organ in organs:
            config_id = int(organ.get("config_id", -1))
            if config_id < 0:
                continue
            sub_type = int(organ.get("sub_type", 0))
            pos = organ.get("pos", {}) or {}
            x = int(pos.get("x", 0))
            z = int(pos.get("z", 0))
            status = int(organ.get("status", 0))

            if sub_type == 1:
                mem = self.treasure_memory.setdefault(config_id, {"pos": (x, z), "available": False})
                mem["pos"] = (x, z)
                mem["available"] = config_id in remaining_treasures
            elif sub_type == 2:
                mem = self.buff_memory.setdefault(
                    config_id,
                    {"pos": (x, z), "available": False, "respawn_step": -1},
                )
                mem["pos"] = (x, z)
                if status == 1:
                    mem["available"] = True
                    mem["respawn_step"] = -1
                else:
                    mem["available"] = False
                    if mem.get("respawn_step", -1) < 0:
                        mem["respawn_step"] = self.step_no + self.buff_refresh_time

        for config_id, mem in self.treasure_memory.items():
            mem["available"] = config_id in remaining_treasures

        collected_buff = int(env_info.get("collected_buff", self.last_collected_buff))
        buff_delta = max(0, collected_buff - self.last_collected_buff)
        if buff_delta > 0 and self.buff_memory:
            hero_x = int(hero_pos["x"])
            hero_z = int(hero_pos["z"])
            candidates = []
            for bid, mem in self.buff_memory.items():
                if not mem.get("available", False):
                    continue
                bx, bz = mem["pos"]
                dist = float(np.hypot(bx - hero_x, bz - hero_z))
                candidates.append((dist, bid))
            candidates.sort()
            for _, bid in candidates[:buff_delta]:
                self.buff_memory[bid]["available"] = False
                self.buff_memory[bid]["respawn_step"] = self.step_no + self.buff_refresh_time

        self.last_collected_buff = collected_buff

        for mem in self.buff_memory.values():
            respawn_step = int(mem.get("respawn_step", -1))
            if respawn_step >= 0 and self.step_no >= respawn_step:
                mem["available"] = True
                mem["respawn_step"] = -1

    def _build_target_features(self, hero_pos, memory, topk=2, prefer_available_only=False):
        hero_x = int(hero_pos["x"])
        hero_z = int(hero_pos["z"])
        items = []
        for obj_id, mem in memory.items():
            available = bool(mem.get("available", False))
            if prefer_available_only and not available:
                continue
            x, z = mem["pos"]
            dx = float(x - hero_x)
            dz = float(z - hero_z)
            dist = float(np.hypot(dx, dz))
            items.append({
                "id": int(obj_id),
                "pos": (int(x), int(z)),
                "available": available,
                "dist": dist,
                "dist_norm": _norm(dist, MAP_SIZE * 1.41),
                "feat": np.array([
                    float(np.clip(dx / MAP_SIZE, -1.0, 1.0)),
                    float(np.clip(dz / MAP_SIZE, -1.0, 1.0)),
                    _norm(dist, MAP_SIZE * 1.41),
                    1.0 if available else 0.0,
                ], dtype=np.float32),
            })

        items.sort(key=lambda x: (0 if x["available"] else 1, x["dist"]))

        feat_list = []
        for item in items[:topk]:
            feat_list.append(item["feat"])
        while len(feat_list) < topk:
            feat_list.append(np.zeros(6, dtype=np.float32))

        best_dist_norm = -1.0
        for item in items:
            if item["available"]:
                best_dist_norm = float(item["dist_norm"])
                break

        return np.concatenate(feat_list, axis=0), items, best_dist_norm

    def _compute_positive_dist_shaping(self, cur_dist_norm, last_attr_name):
        if cur_dist_norm < 0:
            setattr(self, last_attr_name, -1.0)
            return 0.0
        last_dist_norm = float(getattr(self, last_attr_name, -1.0))
        reward = 0.0
        if last_dist_norm >= 0.0:
            reward = max(0.0, last_dist_norm - cur_dist_norm)
        setattr(self, last_attr_name, float(cur_dist_norm))
        return reward

    def _build_history_position_feat(self, hero_pos):
        cur_x = int(hero_pos["x"])
        cur_z = int(hero_pos["z"])

        history = list(self.pos_history)

        if len(history) >= 4:
            p4_x, p4_z = history[-4]
        elif len(history) > 0:
            p4_x, p4_z = history[0]
        else:
            p4_x, p4_z = cur_x, cur_z

        if len(history) >= 8:
            p8_x, p8_z = history[-8]
        elif len(history) > 0:
            p8_x, p8_z = history[0]
        else:
            p8_x, p8_z = cur_x, cur_z

        return np.array([
            _norm(p4_x, MAP_SIZE),
            _norm(p4_z, MAP_SIZE),
            _norm(p8_x, MAP_SIZE),
            _norm(p8_z, MAP_SIZE),
        ], dtype=np.float32)

    def _compute_abb_score(self, cur_hero_pos):
        """
        abb_score = d(p0,p-1)/1 + d(p0,p-2)/2 + ... + d(p0,p-8)/8
        分数越小，说明越可能在局部反复徘徊。
        """
        if cur_hero_pos is None or len(self.pos_history) == 0:
            return 0.0

        cur_x = float(cur_hero_pos[0])
        cur_z = float(cur_hero_pos[1])

        abb_score = 0.0
        history = list(self.pos_history)
        max_steps = min(8, len(history))

        for step in range(1, max_steps + 1):
            px, pz = history[-step]
            dist = float(np.hypot(cur_x - float(px), cur_z - float(pz)))
            abb_score += dist / float(step)

        return abb_score

    def _compute_abb_penalty(self, cur_hero_pos):
        abb_score = self._compute_abb_score(cur_hero_pos)
        penalty = -max(0.0, 1.0 - abb_score / max(self.abb_safe_score, 1e-6))
        return abb_score, penalty

    def _astar_path(self, start, goal):
        """
        在已知且可通行区域上做 8 邻接 A*。
        未知区域视作阻碍。
        返回完整路径 [start, ..., goal]；若失败返回 []。
        """
        import heapq

        if start is None or goal is None:
            return []

        sx, sz = int(start[0]), int(start[1])
        gx, gz = int(goal[0]), int(goal[1])

        if not (0 <= sx < MAP_SIZE_INT and 0 <= sz < MAP_SIZE_INT):
            return []
        if not (0 <= gx < MAP_SIZE_INT and 0 <= gz < MAP_SIZE_INT):
            return []
        if not (self.visibility_map[sx, sz] > 0 and self.passable_map[sx, sz] > 0):
            return []
        if not (self.visibility_map[gx, gz] > 0 and self.passable_map[gx, gz] > 0):
            return []

        def heuristic(x, z):
            return float(np.hypot(x - gx, z - gz))

        open_heap = []
        heapq.heappush(open_heap, (heuristic(sx, sz), 0.0, (sx, sz)))
        came_from = {}
        g_score = {(sx, sz): 0.0}
        visited = set()

        while open_heap:
            _, cur_g, (x, z) = heapq.heappop(open_heap)
            if (x, z) in visited:
                continue
            visited.add((x, z))

            if (x, z) == (gx, gz):
                path = [(x, z)]
                while (x, z) in came_from:
                    x, z = came_from[(x, z)]
                    path.append((x, z))
                path.reverse()
                return path

            for dx, dz in DIR8:
                nx, nz = x + dx, z + dz
                if not (0 <= nx < MAP_SIZE_INT and 0 <= nz < MAP_SIZE_INT):
                    continue
                if not (self.visibility_map[nx, nz] > 0 and self.passable_map[nx, nz] > 0):
                    continue

                step_cost = 1.41421356 if (dx != 0 and dz != 0) else 1.0
                ng = cur_g + step_cost
                if ng + 1e-6 >= g_score.get((nx, nz), float('inf')):
                    continue
                g_score[(nx, nz)] = ng
                came_from[(nx, nz)] = (x, z)
                heapq.heappush(open_heap, (ng + heuristic(nx, nz), ng, (nx, nz)))

        return []

    def _maybe_replan_monster_path(self, idx, start_pos, hero_pos, step_count):
        """
        每 N 帧重算一次 A*；中间沿已有路径每帧推进 step_count 步。
        """
        if start_pos is None or hero_pos is None:
            self.monster_predicted_paths[idx] = []
            self.monster_replan_counters[idx] = 0
            return None

        need_replan = (
            len(self.monster_predicted_paths[idx]) <= 1 or
            self.monster_replan_counters[idx] <= 0
        )

        if need_replan:
            path = self._astar_path(start_pos, hero_pos)
            self.monster_predicted_paths[idx] = path if path else [start_pos]
            self.monster_replan_counters[idx] = max(1, int(MONSTER_ASTAR_REPLAN_INTERVAL))

        path = self.monster_predicted_paths[idx]
        cur_pos = path[0] if path else start_pos

        steps_to_take = max(1, int(step_count))
        for _ in range(steps_to_take):
            if len(path) > 1:
                cur_pos = path[1]
                path = path[1:]
            else:
                break

        self.monster_predicted_paths[idx] = path if path else [cur_pos]
        self.monster_replan_counters[idx] -= 1
        return cur_pos

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]
        self.logger.warning(f"legal_action: {legal_act_raw}")

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)
        newly_discovered_passable_count = 0

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

        monster_reappeared = [False, False]
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]

                prev_pred_pos = self.predicted_monster_pos[i]
                pred_pos = self._update_predicted_monster_pos(i, m, hero_pos, env_info)

                is_in_view = float(m.get("is_in_view", 0))
                prev_invisible = self.last_monster_invisible_1 if i == 0 else self.last_monster_invisible_2
                monster_reappeared[i] = bool(is_in_view > 0.5 and prev_invisible)

                if monster_reappeared[i] and prev_pred_pos is not None and ("pos" in m) and (m["pos"] is not None):
                    true_mx = int(m["pos"]["x"])
                    true_mz = int(m["pos"]["z"])
                    pred_mx = int(prev_pred_pos[0])
                    pred_mz = int(prev_pred_pos[1])
                    pred_error = float(np.hypot(true_mx - pred_mx, true_mz - pred_mz))
                    self.monster_prediction_error_sum += pred_error
                    self.monster_prediction_error_count += 1
                    self.monster_prediction_error_avg = self.monster_prediction_error_sum / max(1, self.monster_prediction_error_count)

                m_speed_norm = _norm(m.get("speed", 1), 2) if is_in_view else 0.0

                rel_x = 0.0
                rel_z = 0.0
                dir_x = 0.0
                dir_z = 0.0
                dist_norm = _norm(m.get("hero_l2_distance", MAX_DIST_BUCKET), MAX_DIST_BUCKET)

                if is_in_view:
                    m_pos = m["pos"]
                    dx = float(m_pos["x"] - hero_pos["x"])
                    dz = float(m_pos["z"] - hero_pos["z"])
                    rel_x = float(np.clip(dx / MAP_SIZE, -1.0, 1.0))
                    rel_z = float(np.clip(dz / MAP_SIZE, -1.0, 1.0))

                    raw_dist = np.sqrt(dx * dx + dz * dz)
                    dist_norm = _bucketize_left(_norm(raw_dist, MAP_SIZE * 1.41), 10)
                    if raw_dist > 1e-6:
                        dir_x = dx / raw_dist
                        dir_z = dz / raw_dist
                else:
                    # 视野外的 monster feature 仍保持“模糊”表征：
                    # 只使用环境给的相对方向 + 距离桶估计，
                    # 不把 A* 预测位置直接注入 vector feature。
                    est_mx, est_mz = _estimate_monster_pos(hero_pos["x"], hero_pos["z"], m)
                    dx = float(est_mx - hero_pos["x"])
                    dz = float(est_mz - hero_pos["z"])
                    rel_x = float(np.clip(dx / MAP_SIZE, -1.0, 1.0))
                    rel_z = float(np.clip(dz / MAP_SIZE, -1.0, 1.0))
                    raw_dist = np.sqrt(dx * dx + dz * dz)
                    dist_norm = _bucketize_left(_norm(raw_dist, MAP_SIZE * 1.41), 10)
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
                monster_reappeared[i] = False
                monster_feats.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))

        organs = frame_state.get("organs", [])
        self._update_organ_memory(env_info, organs, hero_pos)
        treasure_feat, treasure_items, nearest_treasure_dist_norm = self._build_target_features(
            hero_pos, self.treasure_memory, topk=2, prefer_available_only=True
        )
        buff_feat, buff_items, nearest_buff_dist_norm = self._build_target_features(
            hero_pos, self.buff_memory, topk=2, prefer_available_only=False
        )

        if map_info is not None:
            x0, x1, y0, y1, newly_discovered_passable_count = self.update_global_maps(hero_pos['x'], hero_pos['z'], map_info)

        map_feat = np.zeros((3, VIEW_MAP_SIZE, VIEW_MAP_SIZE), dtype=np.float32)

        crop_size = VIEW_MAP_SIZE
        half = crop_size // 2  # 18

        gx0 = int(hero_pos['x'] - half)
        gy0 = int(hero_pos['z'] - half)
        gx1 = gx0 + crop_size
        gy1 = gy0 + crop_size

        for i in range(crop_size):
            for j in range(crop_size):
                gx = gx0 + i
                gy = gy0 + j
                if 0 <= gx < MAP_SIZE_INT and 0 <= gy < MAP_SIZE_INT:
                    map_feat[0, i, j] = float(self.passable_map[gx, gy])

        # 在 passable layer 上叠加自身最近轨迹
        recent_self_path = list(self.pos_history) + [(int(hero_pos["x"]), int(hero_pos["z"]))]
        _paint_recent_positions_on_passable(
            map_feat[0],
            recent_self_path,
            gx0=gx0,
            gy0=gy0,
        )
        
        # _log_gray_map_as_binary(self.logger, map_feat[0], title=f"map:{self.step_no}")
        # _log_gray_map_as_binary(self.logger, map_feat[1], title=f"vis:{self.step_no}")

        # 第二层：monster layer
        # - 先把怪物追击 hero 的 A* 路径画出来
        # - 再把当前真实/预测怪物位置画得更亮
        for i, m in enumerate(monsters[:2]):
            path = self.monster_predicted_paths[i] if i < len(self.monster_predicted_paths) else []
            if path is not None and len(path) > 0:
                _paint_path(
                    map_feat[1],
                    path,
                    gx0=gx0,
                    gy0=gy0,
                    path_value=0.35,
                    radius=0,
                )

            if int(m.get("is_in_view", 0)) > 0 and ("pos" in m) and (m["pos"] is not None):
                mx = int(m["pos"]["x"])
                mz = int(m["pos"]["z"])
            elif self.predicted_monster_pos[i] is not None:
                mx, mz = self.predicted_monster_pos[i]
            else:
                mx, mz = _estimate_monster_pos(hero_pos["x"], hero_pos["z"], m)

            if not (0 <= mx < MAP_SIZE_INT and 0 <= mz < MAP_SIZE_INT):
                continue

            center_i = mx - gx0
            center_j = mz - gy0
            _paint_square(map_feat[1], center_i, center_j, radius=1, value=1.0)

        # 第三层：当前视野内的 treasure / buff mask
        for organ in organs:
            if int(organ.get("status", 0)) != 1:
                continue
            pos = organ.get("pos", {}) or {}
            ox = int(pos.get("x", -1))
            oz = int(pos.get("z", -1))
            if not (0 <= ox < MAP_SIZE_INT and 0 <= oz < MAP_SIZE_INT):
                continue
            if self.visibility_map[ox, oz] == 0:
                continue

            center_i = ox - gx0
            center_j = oz - gy0
            sub_type = int(organ.get("sub_type", 0))
            if sub_type == 1:
                _paint_square(map_feat[2], center_i, center_j, radius=1, value=1.0)
            elif sub_type == 2:
                value = max(float(map_feat[2, center_i, center_j]) if 0 <= center_i < VIEW_MAP_SIZE and 0 <= center_j < VIEW_MAP_SIZE else 0.0, 0.4)
                _paint_square(map_feat[2], center_i, center_j, radius=1, value=value)

        ray_collision_feat = self._ray_collision_direction_scores(
            hero_pos["x"],
            hero_pos["z"],
            return_debug=False,
        )

        # _log_passable_map_and_ray_collision(
        #     self.logger,
        #     map_feat[0],
        #     global_rays,
        #     ray_collision_feat,
        #     step_no=self.step_no,
        #     title="ray_collision_debug",
        # )

        # 合法动作特征 + 合法动作掩码
        raw_legal_action = self._parse_legal_action_raw(legal_act_raw)
        legal_action_feat, legal_action = self._build_processed_legal_action(
            hero_pos["x"],
            hero_pos["z"],
            raw_legal_action,
        )

        # 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        progress_treasure_collect = _norm(int(hero.get("treasure_collected_count", 0)), 10)
        monster_interval = env_info.get("monster_interval", 300)
        time_before_second_mounster = _norm(max(0, monster_interval - self.step_no), self.max_step)
        
        monster_speedup_time = env_info.get("monster_speed_boost_step", 0)
        #if self.logger is not None:
        #    self.logger.warning(f"env info: {env_info}, monster speedup time value:{monster_speedup_time}")
        time_before_mounster_speedup = _norm(max(0, monster_speedup_time - self.step_no), self.max_step)
        progress_feat = np.array([step_norm, progress_treasure_collect, time_before_second_mounster, time_before_mounster_speedup], dtype=np.float32)

        # 基于当前 21x21 绝对已知区域，先将视野内怪物周围 7x7 置 0，
        # 再提取边缘连通簇的 8 方向余弦投影特征
        local_passable_21 = self._extract_local_passable_patch(map_info)
        local_passable_21_masked = self._mask_monster_danger_zone_local(
            local_passable_21,
            monsters,
            hero_pos,
            radius=3,
        )
        boundary_cluster_feat, boundary_cluster_info = self._compute_boundary_cluster_direction_scores(
            local_passable_21_masked
        )
        connected_opening_count = _norm(boundary_cluster_info["connected_opening_count"], 5)
        is_dangerous = _norm(int(boundary_cluster_info["is_dangerous"]), 1)
        
        situation_feat = np.array([connected_opening_count, is_dangerous], dtype=np.float32)

        # Concatenate features / 拼接特征
        # 新增一组 8 维边缘连通簇方向特征，放在 ray collision 特征之后
        vector_feat = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                ray_collision_feat,
                boundary_cluster_feat,
                treasure_feat,
                buff_feat,
                legal_action_feat,
                progress_feat,
                situation_feat,
            ]
        )

        reward_feats = {
            "monster_feats": monster_feats,
            "monster_feats_available": len(monsters),
            "progress_feats": progress_feat,
            "hero_pos": (int(hero_pos["x"]), int(hero_pos["z"])),
            "newly_discovered_passable_count": int(newly_discovered_passable_count),
            "connected_opening_count": int(boundary_cluster_info["connected_opening_count"]),
            "is_dangerous": bool(boundary_cluster_info["is_dangerous"]),
            "nearest_treasure_dist_norm": float(nearest_treasure_dist_norm),
            "nearest_buff_dist_norm": float(nearest_buff_dist_norm),
            "monster_reappeared": monster_reappeared,
        }

        self.pos_history.append((int(hero_pos["x"]), int(hero_pos["z"])))

        return vector_feat, map_feat, reward_feats, legal_action
    
    def calculate_reward(self, env_obs, reward_feats):
        self.total_train_steps += 1
        # 1.若怪物在视野外，让模型跑得更远一点。                            --> 通过 monster dist shaping? 这个足以做到吗？
        # 2.若怪物在视野内且附近有弯道，让模型尽快将其拉脱视野。             --> 加一点视野脱离奖励？monster dist shaping?
        # 3.尽量不要撞墙。1.不要撞侧面的墙。2.不要走进死胡同。              --> 计算路径方向？
        # 4.不要原地打转。                                               --> ABB惩罚？好像做不到。方向一致性惩罚？

        # 基于比赛分数增量的奖励
        env_info = env_obs["observation"].get("env_info", {})
        cur_total_score = float(env_info.get("total_score", 0.0))
        score_gain = cur_total_score - self.last_total_score
        self.last_total_score = cur_total_score
        
        # 怪物 dist shaping
        second_exists = bool(reward_feats['progress_feats'][2] < 1e-6)

        monster_dist_reward = 0.0
        cur_hero_pos = reward_feats.get("hero_pos")

        m1 = reward_feats['monster_feats'][0]
        m2 = reward_feats['monster_feats'][1]
        r1 = 0.0
        r2 = 0.0

        monster_reappeared = reward_feats.get("monster_reappeared", [False, False])

        # monster 1
        if monster_reappeared[0]:
            r1 = 0.0
        elif self.last_monster_dist_norm_1 >= 0:
            cur_dist_1 = float(m1[4])   # dist_norm
            r1 = cur_dist_1 - self.last_monster_dist_norm_1

        # monster 2
        if second_exists:
            if monster_reappeared[1]:
                r2 = 0.0
            elif self.last_monster_dist_norm_2 >= 0:
                cur_dist_2 = float(m2[4])   # dist_norm
                r2 = cur_dist_2 - self.last_monster_dist_norm_2

        self.last_monster_dist_norm_1 = float(m1[4])
        if second_exists:
            self.last_monster_dist_norm_2 = float(m2[4])
        
        monster_dist_reward = r1 + r2

        # 稀疏奖励：如果 monster 从视野内变成视野外给奖励，从视野内变成视野外给轻惩罚
        # 稠密奖励：在视野外时持续获得奖励
        los_break_reward = 0.0

        cur_invisible_1 = bool(m1[0] < 1e-6)
        cur_invisible_2 = bool(m2[0] < 1e-6)

        if cur_invisible_1:
            los_break_reward += 0.01
        if (not self.last_monster_invisible_1) and cur_invisible_1:
            los_break_reward += 1.0
        if self.last_monster_invisible_1 and (not cur_invisible_1):
            los_break_reward -= 0.5

        if second_exists:
            if cur_invisible_2:
                los_break_reward += 0.01
            if (not self.last_monster_invisible_2) and cur_invisible_2:
                los_break_reward += 1.0
            if self.last_monster_invisible_2 and (not cur_invisible_2):
                los_break_reward -= 1.0
        
        self.last_monster_invisible_1 = cur_invisible_1
        if second_exists:
            self.last_monster_invisible_2 = cur_invisible_2
        else:
            self.last_monster_invisible_2 = False
        
        # 局势相关 reward settings
        cur_is_dangerous = bool(reward_feats.get("is_dangerous", False))
        danger_to_safe_reward = 1.0 if (self.last_is_dangerous and (not cur_is_dangerous)) else 0.0
        danger_penalty = -1.0 if cur_is_dangerous else 0.0
        self.last_is_dangerous = cur_is_dangerous

        # 闪现释放奖励：仅当“危险 -> 不危险”时给大分
        flash_reward = 0.0
        flash_count = env_info.get("flash_count", 0)
        if (flash_count - self.last_flash_count) > 0:
            if danger_to_safe_reward > 0.0:
                flash_reward = 3.0
            else:
                flash_reward = -0.5
        self.last_flash_count = flash_count

        # 靠墙惩罚：只在 hero 周围 5x5 小窗口内查最近已知墙，减少计算量
        near_wall_penalty = 0.0
        if cur_hero_pos is not None:
            near_wall_penalty = self._compute_near_wall_penalty(
                cur_hero_pos[0],
                cur_hero_pos[1],
                search_radius=3,
            )

        abb_score, abb_penalty = self._compute_abb_penalty(cur_hero_pos)

        # 探索奖励
        newly_discovered_passable_count = reward_feats.get("newly_discovered_passable_count", 0)
        if self.step_no <= 1:
            explore_reward = 0.0
        else: explore_reward = _norm(newly_discovered_passable_count, 40.0)

        # 生存奖励
        survive_phase_weight = 1.00 + (self.step_no / 200)
        if abb_score < self.abb_safe_score:
            los_break_reward = min(los_break_reward, 0.0)
        abb_score = _norm(abb_score, self.abb_safe_score)

        treasure_dist_reward = self._compute_positive_dist_shaping(
            reward_feats.get("nearest_treasure_dist_norm", -1.0),
            "last_treasure_dist_norm",
        )
        buff_dist_reward = 0.4 * self._compute_positive_dist_shaping(
            reward_feats.get("nearest_buff_dist_norm", -1.0),
            "last_buff_dist_norm",
        )

        # buff 奖励
        monster_goingto_speedup = bool(reward_feats['progress_feats'][3] < 100)

        collected_buff = int(env_info.get("collected_buff", self.last_collected_buff))
        buff_delta = float(max(0, collected_buff - self.last_collected_buff))
        self.last_collected_buff = collected_buff
        buff_pick_reward = buff_delta * (40.0 if monster_goingto_speedup else 20.0)

        # final step: reward vector
        dist_shaping_norm_weight = 12.8

        exploration_rate = 1.0
        if cur_invisible_1 and cur_invisible_2:
            exploration_rate = 2.0
        else:
            exploration_rate = 0.1

        exploration_rate *= (2.0 if (not cur_is_dangerous) else 0.1)

        in_stage2 = self.curriculum_episode >= CURRICULUM_STAGE2_EPISODE
        score_gain_weight = SCORE_GAIN_WEIGHT_STAGE2 if in_stage2 else SCORE_GAIN_WEIGHT_STAGE1
        survival_weight = SURVIVAL_WEIGHT_STAGE2 if in_stage2 else SURVIVAL_WEIGHT_STAGE1

        reward_vector = [
            score_gain_weight * score_gain,
            survival_weight * survive_phase_weight * abb_score,
            0.50 * los_break_reward,
            0.25 * flash_reward,
            0.30 * near_wall_penalty,
            0.50 * abb_penalty,
            0.10 * exploration_rate * explore_reward,
            0.30 * danger_penalty,
            0.30 * dist_shaping_norm_weight * treasure_dist_reward,
            0.30 * dist_shaping_norm_weight * buff_dist_reward,
            survival_weight * buff_pick_reward,
            abs(1.50 * dist_shaping_norm_weight * monster_dist_reward),
            self.monster_prediction_error_avg,
        ]

        return reward_vector, sum(reward_vector[:-2]) + 1.50 * dist_shaping_norm_weight * monster_dist_reward