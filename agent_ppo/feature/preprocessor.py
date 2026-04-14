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
import numpy as np
from collections import deque

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
MAP_SIZE_INT = 128
LOCAL_MAP_SIZE = 21
LOCAL_MAP_HALF = 10
VIEW_MAP_SIZE = 21

# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
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

class Preprocessor:
    def __init__(self, logger=None):
        self.logger = logger
        self.total_train_steps = 0
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200

        self.last_monster_dist_norm_1 = -1.0
        self.last_monster_dist_norm_2 = -1.0
        self.last_monster_invisible_1 = False
        self.last_monster_invisible_2 = False

        self.last_offview_guidance_dist_1 = -1.0
        self.last_offview_guidance_dist_2 = -1.0
        self.last_offview_guidance_target_1 = None
        self.last_offview_guidance_target_2 = None

        self.last_total_score = 0.0
        self.last_flash_count = 0

        self.prev_hero_pos = None

        # ========= 两层全局记忆 =========
        # 第一层：可通行地图：1=可走, 0=不能走/未知
        self.passable_map = np.zeros((MAP_SIZE_INT, MAP_SIZE_INT), dtype=np.uint8)
        # 第二层：可见性地图：1=已知, 0=未知
        self.visibility_map = np.zeros((MAP_SIZE_INT, MAP_SIZE_INT), dtype=np.uint8)

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
        return dir_scores, {
            "boundary_pts": boundary_pts,
            "clusters": clusters,
            "connected_clusters": connected_clusters,
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

    def _compute_offview_guidance_reward(
        self,
        monster_feat,
        connected_clusters,
        last_dist_attr_name,
        last_target_attr_name,
        last_action,
        angle_cos_threshold=0.0,
    ):
        """
        当怪物不在视野内时：
        - 从所有可到达的 boundary clusters 中，选与“怪物方向相反”最一致的 cluster
        - 不再使用 hero 到 cluster center 的距离减小量
        - 改为：上一帧动作方向 与 当前目标 cluster 方向 的余弦相似度
        """
        is_in_view = bool(monster_feat[0] > 0.5)
        if is_in_view:
            setattr(self, last_dist_attr_name, -1.0)
            setattr(self, last_target_attr_name, None)
            return 0.0, None

        target = self._select_reachable_opposite_boundary_cluster(
            connected_clusters=connected_clusters,
            monster_feat=monster_feat,
            angle_cos_threshold=angle_cos_threshold,
        )
        if target is None:
            setattr(self, last_dist_attr_name, -1.0)
            setattr(self, last_target_attr_name, None)
            return 0.0, None

        action_vec = self._action_to_dir_vec(last_action)
        cur_target = tuple(round(v, 3) for v in target["center"])
        if action_vec is None:
            setattr(self, last_dist_attr_name, -1.0)
            setattr(self, last_target_attr_name, cur_target)
            return 0.0, target

        agent_x = float(LOCAL_MAP_HALF)
        agent_y = float(LOCAL_MAP_HALF)
        cx, cy = target["center"]
        vx = float(cx - agent_x)
        vy = float(cy - agent_y)
        vnorm = float(np.hypot(vx, vy))
        if vnorm <= 1e-6:
            setattr(self, last_dist_attr_name, -1.0)
            setattr(self, last_target_attr_name, cur_target)
            return 0.0, target

        target_dir_x = vx / vnorm
        target_dir_y = vy / vnorm
        act_x, act_y = action_vec
        cos_sim = float(act_x * target_dir_x + act_y * target_dir_y)

        reward = max(0.0, cos_sim)

        setattr(self, last_dist_attr_name, -1.0)
        setattr(self, last_target_attr_name, cur_target)
        return reward, target


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
                dir_idx = int(m.get("hero_relative_direction", 0))
                dir_x, dir_z = DIR9_TO_VEC.get(dir_idx, (0.0, 0.0))

                dist_norm = _norm(m.get("hero_l2_distance", MAX_DIST_BUCKET), MAX_DIST_BUCKET)

                if is_in_view:
                    m_pos = m["pos"]
                    dx = float(m_pos["x"] - hero_pos["x"])
                    dz = float(m_pos["z"] - hero_pos["z"])

                    # 精细相对位置：保留正负号
                    rel_x = float(np.clip(dx / MAP_SIZE, -1.0, 1.0))
                    rel_z = float(np.clip(dz / MAP_SIZE, -1.0, 1.0))

                    raw_dist = np.sqrt(dx * dx + dz * dz)
                    dist_norm = _bucketize_left(_norm(raw_dist, MAP_SIZE * 1.41), 10)

                    # 视野内时，用连续方向覆盖离散方向
                    if raw_dist > 1e-6:
                        dir_x = dx / raw_dist
                        dir_z = dz / raw_dist
                else:
                    est_mx, est_mz = _estimate_monster_pos(hero_pos["x"], hero_pos["z"], m)
                    dx = float(est_mx - hero_pos["x"])
                    dz = float(est_mz - hero_pos["z"])
                    rel_x = float(np.clip(dx / MAP_SIZE, -1.0, 1.0))
                    rel_z = float(np.clip(dz / MAP_SIZE, -1.0, 1.0))

                monster_feats.append(
                    np.array(
                        [is_in_view, m_speed_norm, rel_x, rel_z, dist_norm, dir_x, dir_z],
                        dtype=np.float32,
                    )
                )
            else:
                monster_feats.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))

        if map_info is not None:
            x0, x1, y0, y1, newly_discovered_passable_count = self.update_global_maps(hero_pos['x'], hero_pos['z'], map_info)

        map_feat = np.zeros((2, VIEW_MAP_SIZE, VIEW_MAP_SIZE), dtype=np.float32)

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
        
        # _log_gray_map_as_binary(self.logger, map_feat[0], title=f"map:{self.step_no}")
        # _log_gray_map_as_binary(self.logger, map_feat[1], title=f"vis:{self.step_no}")

        # 第二层：monster mask
        # 规则：
        # - 视野内：用精确位置
        # - 视野外但怪物存在：用粗方向 + 桶距离估计位置
        for m in monsters[:2]:
            mx, mz = _estimate_monster_pos(hero_pos["x"], hero_pos["z"], m)

            if not (0 <= mx < MAP_SIZE_INT and 0 <= mz < MAP_SIZE_INT):
                continue

            center_i = mx - gx0
            center_j = mz - gy0
            _paint_square(map_feat[1], center_i, center_j, radius=1, value=1.0)

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

        # 基于当前 21x21 绝对已知区域，提取边缘连通簇的 8 方向余弦投影特征
        local_passable_21 = self._extract_local_passable_patch(map_info)
        boundary_cluster_feat, _boundary_cluster_debug = self._compute_boundary_cluster_direction_scores(
            local_passable_21
        )

        # Concatenate features / 拼接特征
        # 新增一组 8 维边缘连通簇方向特征，放在 ray collision 特征之后
        vector_feat = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                ray_collision_feat,
                boundary_cluster_feat,
                legal_action_feat,
                progress_feat,
            ]
        )

        reward_feats = {
            "monster_feats": monster_feats,
            "monster_feats_available": len(monsters),
            "progress_feats": progress_feat,
            "hero_pos": (int(hero_pos["x"]), int(hero_pos["z"])),
            "prev_hero_pos": self.prev_hero_pos,
            "last_action": int(last_action),
            "newly_discovered_passable_count": int(newly_discovered_passable_count),
            "connected_boundary_clusters": _boundary_cluster_debug["connected_clusters"],
        }

        self.prev_hero_pos = (int(hero_pos["x"]), int(hero_pos["z"]))

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

        # monster 1
        if self.last_monster_dist_norm_1 >= 0:
            cur_dist_1 = float(m1[4])   # dist_norm
            r1 = cur_dist_1 - self.last_monster_dist_norm_1

        # monster 2
        if second_exists:
            if self.last_monster_dist_norm_2 >= 0:
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
        
        # 闪现释放奖励
        flash_reward = 0.0
        flash_count = env_info.get("flash_count", 0)
        if (flash_count - self.last_flash_count) > 0:
            flash_reward = los_break_reward + 0.5 * monster_dist_reward
        self.last_flash_count = flash_count

        connected_boundary_clusters = reward_feats.get("connected_boundary_clusters", [])
        last_action = reward_feats.get("last_action", -1)
        offview_guidance_reward_1, _ = self._compute_offview_guidance_reward(
            monster_feat=m1,
            connected_clusters=connected_boundary_clusters,
            last_dist_attr_name="last_offview_guidance_dist_1",
            last_target_attr_name="last_offview_guidance_target_1",
            last_action=last_action,
            angle_cos_threshold=0.0,
        )

        offview_guidance_reward_2 = 0.0
        if second_exists:
            offview_guidance_reward_2, _ = self._compute_offview_guidance_reward(
                monster_feat=m2,
                connected_clusters=connected_boundary_clusters,
                last_dist_attr_name="last_offview_guidance_dist_2",
                last_target_attr_name="last_offview_guidance_target_2",
                last_action=last_action,
                angle_cos_threshold=0.0,
            )
        else:
            self.last_offview_guidance_dist_2 = -1.0
            self.last_offview_guidance_target_2 = None

        offview_guidance_reward = offview_guidance_reward_1 + offview_guidance_reward_2

        # 靠墙惩罚：只在 hero 周围 5x5 小窗口内查最近已知墙，减少计算量
        near_wall_penalty = 0.0
        if cur_hero_pos is not None:
            near_wall_penalty = self._compute_near_wall_penalty(
                cur_hero_pos[0],
                cur_hero_pos[1],
                search_radius=3,
            )

        # 探索奖励
        newly_discovered_passable_count = reward_feats.get("newly_discovered_passable_count", 0)
        if self.step_no <= 1:
            explore_reward = 0.0
        else: explore_reward = _norm(newly_discovered_passable_count, 40.0)

        survive_phase_weight = 1.00 + (self.step_no / 200)

        # final step reward vector
        dist_shaping_norm_weight = 12.8

        reward_vector = [
            0.0, #0.20 * score_gain,
            0.08 * survive_phase_weight,
            0.0, # 3.50 * dist_shaping_norm_weight * monster_dist_reward,
            0.50 * los_break_reward,
            0.0, # 0.25 * flash_reward,
            0.0, # 0.25 * offview_guidance_reward,
            0.20 * near_wall_penalty,
            0.08 * explore_reward
        ]

        return reward_vector, sum(reward_vector)
