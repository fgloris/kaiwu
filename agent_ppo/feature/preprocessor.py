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

def _log_gray_map_as_binary(logger, gray_map, title="map36"):
    """
    将 21x21 灰度图压成单个 01 字符串，并一次 warning 输出。
    规则：>0 的都记为 1，因此 0.5 也会记成 1。
    """
    arr = np.asarray(gray_map)
    assert arr.shape == (VIEW_MAP_SIZE, VIEW_MAP_SIZE), f"expect ({VIEW_MAP_SIZE},{VIEW_MAP_SIZE}), got {arr.shape}"

    s = "".join("1" if v > 0 else "0" for v in arr.reshape(-1))
    logger.warning(f"[{title}]{s}")

def _log_passable_map_and_ray_collision(logger, gray_map, global_rays, ray_collision_scores, step_no=None, title="ray_collision_debug"):
    """
    精简日志：
    1. map: 21x21 passable map 压平
    2. rays: 全局 rays 的 [angle, score]
    3. ray_collision_scores: 8个动作方向分数
    """
    if logger is None:
        return

    arr = np.asarray(gray_map)
    assert arr.shape == (VIEW_MAP_SIZE, VIEW_MAP_SIZE), \
        f"expect ({VIEW_MAP_SIZE},{VIEW_MAP_SIZE}), got {arr.shape}"

    map_bits = "".join("1" if v > 0 else "0" for v in arr.reshape(-1))

    rays = []
    for ray in global_rays:
        rays.append([
            int(round(float(ray["angle"]))),
            round(float(ray["score"]), 4),
        ])

    ray_collision_scores_list = [round(float(x), 4) for x in ray_collision_scores]

    payload = {
        "step": None if step_no is None else int(step_no),
        "map": map_bits,
        "rays": rays,
        "ray_collision_scores": ray_collision_scores_list,
    }

    logger.warning(f"[{title}]{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}")

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

        self.last_total_score = 0.0
        self.last_flash_count = 0
        self.last_safety_score = 0.0
        self.last_trap_score = 0.0
        self.last_progress_score = 0.0

        self.prev_hero_pos = None
        self.recent_positions = deque(maxlen=24)
        self.last_visit_step = np.full((MAP_SIZE_INT, MAP_SIZE_INT), -100000, dtype=np.int32)

        # ========= 全局记忆 =========
        self.passable_map = np.zeros((MAP_SIZE_INT, MAP_SIZE_INT), dtype=np.uint8)
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

        for i in range(h):
            for j in range(w):
                gx = x0 + j
                gy = y0 + i
                if not (0 <= gx < MAP_SIZE_INT and 0 <= gy < MAP_SIZE_INT):
                    continue

                # 文档定义：1=可通行，0=障碍
                visible_val = 1
                passable_val = 1 if int(map_info[i][j]) != 0 else 0

                self.visibility_map[gx, gy] = visible_val
                self.passable_map[gx, gy] = passable_val

        return gx0, gx1, gy0, gy1


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
            if self._is_global_passable(nx, nz):
                return dx * step, dz * step, True
        return 0, 0, False
    
    def _is_known_wall(self, x, z):
        """
        已知墙：visibility=1 且 passable=0
        """
        if not (0 <= x < MAP_SIZE_INT and 0 <= z < MAP_SIZE_INT):
            return True  # 出界直接视为墙，更保守
        return bool(self.visibility_map[x, z] > 0 and self.passable_map[x, z] == 0)

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
        elif min_dist <= 1.414:
            return -0.514
        elif min_dist <= 2.0 + 1e-6:
            return -0.2
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

    def _count_frontier_neighbors(self, x, z):
        frontier = 0
        for dx, dz in DIR8:
            nx = x + dx
            nz = z + dz
            if not (0 <= nx < MAP_SIZE_INT and 0 <= nz < MAP_SIZE_INT):
                continue
            if self.visibility_map[nx, nz] == 0:
                frontier += 1
        return frontier

    def _cell_open_degree(self, x, z):
        cnt = 0
        for dx, dz in DIR8:
            if self._is_global_passable(x + dx, z + dz):
                cnt += 1
        return cnt

    def _bfs_local_area(self, start_x, start_z, radius=12):
        if not self._is_global_passable(start_x, start_z):
            return 0, 0, 0
        x0 = max(0, int(start_x - radius))
        x1 = min(MAP_SIZE_INT - 1, int(start_x + radius))
        z0 = max(0, int(start_z - radius))
        z1 = min(MAP_SIZE_INT - 1, int(start_z + radius))

        q = deque([(int(start_x), int(start_z), 0)])
        visited = {(int(start_x), int(start_z))}
        area = 0
        frontier = 0
        max_depth = 0
        while q:
            x, z, d = q.popleft()
            area += 1
            max_depth = max(max_depth, d)
            frontier += min(self._count_frontier_neighbors(x, z), 2)
            for dx, dz in DIR8:
                nx = x + dx
                nz = z + dz
                if not (x0 <= nx <= x1 and z0 <= nz <= z1):
                    continue
                if (nx, nz) in visited:
                    continue
                if not self._is_global_passable(nx, nz):
                    continue
                visited.add((nx, nz))
                q.append((nx, nz, d + 1))
        return area, frontier, max_depth

    def _monster_danger_score(self, x, z, monster_feats):
        danger = 0.0
        for m in monster_feats[:2]:
            dist_norm = float(m[4])
            in_view = float(m[0])
            dir_x = float(m[5])
            dir_z = float(m[6])
            rel_x = float(m[2]) * MAP_SIZE
            rel_z = float(m[3]) * MAP_SIZE
            approx_dist = max(1.0, float(np.hypot(rel_x, rel_z)))
            proximity = 1.0 - np.clip(dist_norm, 0.0, 1.0)
            base = 0.45 * proximity + 0.55 * np.clip((30.0 / approx_dist), 0.0, 1.0)
            if in_view > 0.5:
                base += 0.25
            if abs(dir_x) + abs(dir_z) < 1e-6:
                base *= 0.6
            danger = max(danger, base)
        return float(np.clip(danger, 0.0, 1.5))

    def _compute_directional_escape_scores(self, hero_x, hero_z, monster_feats, legal_action, ray_collision_feat, boundary_cluster_feat):
        move_scores = np.zeros(8, dtype=np.float32)
        flash_scores = np.zeros(8, dtype=np.float32)
        move_debug = []
        current_area, current_frontier, current_depth = self._bfs_local_area(hero_x, hero_z, radius=12)
        current_degree = self._cell_open_degree(hero_x, hero_z)
        current_frontier_local = self._count_frontier_neighbors(hero_x, hero_z)
        current_safety = 0.0

        for i, (dx, dz) in enumerate(DIR8):
            nx = hero_x + dx
            nz = hero_z + dz
            if legal_action[i] <= 0 or (not self._is_global_passable(nx, nz)):
                move_scores[i] = 0.0
                move_debug.append({"ok": False})
                continue

            area, frontier, depth = self._bfs_local_area(nx, nz, radius=12)
            degree = self._cell_open_degree(nx, nz)
            open_score = float(ray_collision_feat[i])
            cluster_score = float(boundary_cluster_feat[i])
            revisit_age = self.step_no - int(self.last_visit_step[nx, nz])
            revisit_penalty = 1.0 - np.clip(revisit_age / 18.0, 0.0, 1.0)
            frontier_local = self._count_frontier_neighbors(nx, nz)
            area_gain = np.clip((area - current_area) / 48.0, -1.0, 1.0)
            depth_gain = np.clip((depth - current_depth) / 12.0, -1.0, 1.0)
            degree_term = np.clip((degree - 1) / 4.0, 0.0, 1.0)
            frontier_term = np.clip((frontier_local + 0.35 * frontier) / 8.0, 0.0, 1.0)
            trap_risk = np.clip((2.0 - degree) / 2.0, 0.0, 1.0)

            score = (
                0.28 * open_score
                + 0.16 * cluster_score
                + 0.24 * np.clip(area / 80.0, 0.0, 1.0)
                + 0.12 * degree_term
                + 0.18 * frontier_term
                + 0.10 * np.clip(area_gain + depth_gain, -1.0, 1.0)
                - 0.22 * revisit_penalty
                - 0.16 * trap_risk
            )
            score = float(np.clip((score + 0.18) / 1.08, 0.0, 1.0))
            move_scores[i] = score
            move_debug.append({
                "ok": True,
                "area": area,
                "frontier": frontier,
                "depth": depth,
                "degree": degree,
                "trap_risk": trap_risk,
                "revisit_penalty": revisit_penalty,
                "score": score,
            })

        current_safety = float(
            np.clip(
                0.35 * np.clip(current_area / 80.0, 0.0, 1.0)
                + 0.25 * np.clip(current_degree / 4.0, 0.0, 1.0)
                + 0.20 * np.clip((current_frontier_local + current_frontier * 0.2) / 8.0, 0.0, 1.0)
                + 0.20 * np.max(move_scores),
                0.0,
                1.0,
            )
        )
        trap_score = float(np.clip(1.0 - (0.55 * current_safety + 0.25 * np.clip(current_degree / 4.0, 0.0, 1.0) + 0.20 * np.clip(current_area / 80.0, 0.0, 1.0)), 0.0, 1.0))

        danger = self._monster_danger_score(hero_x, hero_z, monster_feats)

        for i, (dx, dz) in enumerate(DIR8):
            flash_legal = legal_action[8 + i] > 0
            fx, fz, ok = self._flash_landing_offset(hero_x, hero_z, dx, dz)
            if (not flash_legal) or (not ok):
                flash_scores[i] = 0.0
                continue
            tx = hero_x + fx
            tz = hero_z + fz
            area, frontier, depth = self._bfs_local_area(tx, tz, radius=12)
            degree = self._cell_open_degree(tx, tz)
            revisit_age = self.step_no - int(self.last_visit_step[tx, tz])
            revisit_penalty = 1.0 - np.clip(revisit_age / 18.0, 0.0, 1.0)
            frontier_term = np.clip((self._count_frontier_neighbors(tx, tz) + 0.35 * frontier) / 8.0, 0.0, 1.0)
            flash_gain = np.clip((area - current_area) / 48.0, -1.0, 1.0)
            score = (
                0.36 * np.clip(area / 80.0, 0.0, 1.0)
                + 0.18 * np.clip(degree / 4.0, 0.0, 1.0)
                + 0.14 * frontier_term
                + 0.24 * np.clip(flash_gain + depth / 18.0, -1.0, 1.0)
                - 0.22 * revisit_penalty
            )
            score = float(np.clip((score + 0.10) / 0.92, 0.0, 1.0))
            if danger < 0.40 and score < 0.72:
                score *= 0.2
            flash_scores[i] = score

        return move_scores, flash_scores, {
            "current_safety": current_safety,
            "trap_score": trap_score,
            "danger": danger,
            "current_area": current_area,
            "current_degree": current_degree,
            "current_frontier": current_frontier_local,
            "move_debug": move_debug,
        }

    def _build_local_maps(self, hero_x, hero_z, monsters):
        map_feat = np.zeros((4, VIEW_MAP_SIZE, VIEW_MAP_SIZE), dtype=np.float32)
        crop_size = VIEW_MAP_SIZE
        half = crop_size // 2
        gx0 = int(hero_x - half)
        gz0 = int(hero_z - half)

        for i in range(crop_size):
            for j in range(crop_size):
                gx = gx0 + i
                gz = gz0 + j
                if not (0 <= gx < MAP_SIZE_INT and 0 <= gz < MAP_SIZE_INT):
                    continue
                passable = float(self.passable_map[gx, gz])
                visible = float(self.visibility_map[gx, gz])
                map_feat[0, i, j] = passable
                if visible > 0.5 and passable > 0.5:
                    age = self.step_no - int(self.last_visit_step[gx, gz])
                    visit_decay = np.clip(1.0 - age / 24.0, 0.0, 1.0)
                    frontier = 1.0 if self._count_frontier_neighbors(gx, gz) > 0 else 0.0
                    map_feat[2, i, j] = visit_decay
                    map_feat[3, i, j] = frontier
        for m in monsters[:2]:
            mx, mz = _estimate_monster_pos(hero_x, hero_z, m)
            center_i = mx - gx0
            center_j = mz - gz0
            _paint_square(map_feat[1], center_i, center_j, radius=1, value=1.0)
        return map_feat

    def _apply_flash_gate(self, legal_action, danger, trap_score, move_scores, flash_scores):
        gated = list(legal_action)
        best_move = float(np.max(move_scores)) if len(move_scores) else 0.0
        best_flash = float(np.max(flash_scores)) if len(flash_scores) else 0.0
        allow_flash = (danger > 0.60) or (trap_score > 0.62) or (best_flash > best_move + 0.12)
        if not allow_flash:
            for i in range(8, 16):
                gated[i] = 0
        else:
            for i in range(8):
                if flash_scores[i] < max(0.50, best_flash - 0.18):
                    gated[8 + i] = 0
        if sum(gated) == 0:
            gated = list(legal_action)
        return gated


    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, local map, reward features and legal mask."""
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x = int(hero_pos["x"])
        hero_z = int(hero_pos["z"])

        hero_feat = np.array([
            _norm(hero_x, MAP_SIZE),
            _norm(hero_z, MAP_SIZE),
            _norm(hero["flash_cooldown"], MAX_FLASH_CD),
            _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION),
        ], dtype=np.float32)

        monsters = frame_state.get("monsters", [])
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED) if is_in_view else 0.0
                dir_idx = int(m.get("hero_relative_direction", 0))
                dir_x, dir_z = DIR9_TO_VEC.get(dir_idx, (0.0, 0.0))
                dist_norm = _norm(m.get("hero_l2_distance", MAX_DIST_BUCKET), MAX_DIST_BUCKET)
                rel_x = 0.0
                rel_z = 0.0
                if is_in_view:
                    m_pos = m["pos"]
                    dx = float(m_pos["x"] - hero_x)
                    dz = float(m_pos["z"] - hero_z)
                    rel_x = float(np.clip(dx / MAP_SIZE, -1.0, 1.0))
                    rel_z = float(np.clip(dz / MAP_SIZE, -1.0, 1.0))
                    raw_dist = float(np.hypot(dx, dz))
                    dist_norm = _bucketize_left(_norm(raw_dist, MAP_SIZE * 1.41), 10)
                    if raw_dist > 1e-6:
                        dir_x = dx / raw_dist
                        dir_z = dz / raw_dist
                else:
                    est_mx, est_mz = _estimate_monster_pos(hero_x, hero_z, m)
                    dx = float(est_mx - hero_x)
                    dz = float(est_mz - hero_z)
                    rel_x = float(np.clip(dx / MAP_SIZE, -1.0, 1.0))
                    rel_z = float(np.clip(dz / MAP_SIZE, -1.0, 1.0))
                monster_feats.append(np.array([is_in_view, m_speed_norm, rel_x, rel_z, dist_norm, dir_x, dir_z], dtype=np.float32))
            else:
                monster_feats.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))

        if map_info is not None:
            self.update_global_maps(hero_x, hero_z, map_info)

        if 0 <= hero_x < MAP_SIZE_INT and 0 <= hero_z < MAP_SIZE_INT:
            self.last_visit_step[hero_x, hero_z] = self.step_no
            self.recent_positions.append((hero_x, hero_z))

        map_feat = self._build_local_maps(hero_x, hero_z, monsters)
        ray_collision_feat = self._ray_collision_direction_scores(hero_x, hero_z, return_debug=False)

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

        local_passable_21 = self._extract_local_passable_patch(map_info)
        boundary_cluster_feat, _ = self._compute_boundary_cluster_direction_scores(local_passable_21)

        move_prior_feat, flash_prior_feat, escape_debug = self._compute_directional_escape_scores(
            hero_x, hero_z, monster_feats, legal_action, ray_collision_feat, boundary_cluster_feat
        )
        legal_action = self._apply_flash_gate(
            legal_action,
            danger=escape_debug["danger"],
            trap_score=escape_debug["trap_score"],
            move_scores=move_prior_feat,
            flash_scores=flash_prior_feat,
        )

        step_norm = _norm(self.step_no, self.max_step)
        time_ratio = _norm(max(0, env_info.get("monster_interval", 300) - self.step_no), self.max_step)
        speedup_cfg = env_info.get("monster_speed_boost_step", env_info.get("monster_speedup", self.max_step))
        speedup_ratio = _norm(max(0, speedup_cfg - self.step_no), self.max_step)
        progress_feat = np.array([
            step_norm,
            _norm(int(hero.get("treasure_collected_count", 0)), 10),
            float(np.clip(escape_debug["danger"], 0.0, 1.0)),
            float(np.clip(escape_debug["trap_score"], 0.0, 1.0)),
            time_ratio,
            speedup_ratio,
        ], dtype=np.float32)

        vector_feat = np.concatenate([
            hero_feat,
            monster_feats[0],
            monster_feats[1],
            ray_collision_feat,
            boundary_cluster_feat,
            move_prior_feat,
            flash_prior_feat,
            progress_feat,
        ])

        reward_feats = {
            "monster_feats": monster_feats,
            "hero_pos": (hero_x, hero_z),
            "prev_hero_pos": self.prev_hero_pos,
            "last_action": int(last_action),
            "move_prior": move_prior_feat,
            "flash_prior": flash_prior_feat,
            "danger": float(escape_debug["danger"]),
            "trap_score": float(escape_debug["trap_score"]),
            "current_safety": float(escape_debug["current_safety"]),
            "best_move_score": float(np.max(move_prior_feat)),
            "best_flash_score": float(np.max(flash_prior_feat)),
            "current_area": float(escape_debug["current_area"]),
            "current_degree": float(escape_debug["current_degree"]),
        }

        self.prev_hero_pos = (hero_x, hero_z)
        return vector_feat, map_feat, reward_feats, legal_action

    def calculate_reward(self, env_obs, reward_feats):
        self.total_train_steps += 1
        env_info = env_obs["observation"].get("env_info", {})
        cur_total_score = float(env_info.get("total_score", 0.0))
        score_gain = cur_total_score - self.last_total_score
        self.last_total_score = cur_total_score

        m1 = reward_feats["monster_feats"][0]
        m2 = reward_feats["monster_feats"][1]
        cur_dist_1 = float(m1[4])
        cur_dist_2 = float(m2[4])
        r1 = 0.0 if self.last_monster_dist_norm_1 < 0 else (cur_dist_1 - self.last_monster_dist_norm_1)
        r2 = 0.0 if self.last_monster_dist_norm_2 < 0 else (cur_dist_2 - self.last_monster_dist_norm_2)
        self.last_monster_dist_norm_1 = cur_dist_1
        self.last_monster_dist_norm_2 = cur_dist_2
        monster_dist_reward = 0.7 * r1 + 0.4 * r2

        cur_invisible_1 = bool(m1[0] < 1e-6)
        cur_invisible_2 = bool(m2[0] < 1e-6)
        los_break_reward = 0.0
        if (not self.last_monster_invisible_1) and cur_invisible_1:
            los_break_reward += 0.35
        if self.last_monster_invisible_1 and (not cur_invisible_1):
            los_break_reward -= 0.18
        if (not self.last_monster_invisible_2) and cur_invisible_2:
            los_break_reward += 0.18
        if self.last_monster_invisible_2 and (not cur_invisible_2):
            los_break_reward -= 0.10
        self.last_monster_invisible_1 = cur_invisible_1
        self.last_monster_invisible_2 = cur_invisible_2

        cur_hero_pos = reward_feats.get("hero_pos")
        prev_hero_pos = reward_feats.get("prev_hero_pos")
        moved = 0.0
        if (cur_hero_pos is not None) and (prev_hero_pos is not None):
            moved = float(np.hypot(cur_hero_pos[0] - prev_hero_pos[0], cur_hero_pos[1] - prev_hero_pos[1]))
        stall_penalty = -1.0 if moved < 0.5 else 0.0

        revisit_penalty = 0.0
        if cur_hero_pos is not None and len(self.recent_positions) >= 4:
            repeats = sum(1 for p in list(self.recent_positions)[:-1] if p == cur_hero_pos)
            revisit_penalty = -float(np.clip(repeats / 3.0, 0.0, 1.0))

        safety_score = float(reward_feats.get("current_safety", 0.0))
        trap_score = float(reward_feats.get("trap_score", 0.0))
        danger = float(reward_feats.get("danger", 0.0))

        safety_delta = safety_score - self.last_safety_score
        trap_delta = self.last_trap_score - trap_score
        progress_score = 0.6 * safety_score + 0.4 * reward_feats.get("best_move_score", 0.0)
        progress_delta = progress_score - self.last_progress_score
        self.last_safety_score = safety_score
        self.last_trap_score = trap_score
        self.last_progress_score = progress_score

        wall_penalty = -float(np.clip(1.0 - reward_feats.get("best_move_score", 0.0), 0.0, 1.0)) * 0.5
        if cur_hero_pos is not None:
            wall_penalty += 0.6 * self._compute_near_wall_penalty(cur_hero_pos[0], cur_hero_pos[1], search_radius=2)

        flash_count = int(env_info.get("flash_count", 0))
        used_flash = (flash_count - self.last_flash_count) > 0
        self.last_flash_count = flash_count
        flash_reward = 0.0
        if used_flash:
            flash_quality = float(reward_feats.get("best_flash_score", 0.0))
            best_move = float(reward_feats.get("best_move_score", 0.0))
            if danger > 0.55 or trap_score > 0.60:
                flash_reward += 0.45 * max(0.0, flash_quality - best_move + 0.05)
                flash_reward += 0.25 * max(0.0, trap_delta)
                flash_reward += 0.20 * max(0.0, los_break_reward)
            else:
                flash_reward -= 0.35
            if flash_quality < best_move + 0.03:
                flash_reward -= 0.20
        else:
            if danger > 0.75 and reward_feats.get("best_flash_score", 0.0) > reward_feats.get("best_move_score", 0.0) + 0.18:
                flash_reward -= 0.06

        survive_reward = 0.015
        danger_penalty = -0.12 * danger * max(0.0, trap_score - 0.45)

        reward_vector = [
            0.25 * score_gain,
            survive_reward,
            0.90 * progress_delta,
            0.55 * safety_delta,
            0.60 * trap_delta,
            0.12 * monster_dist_reward,
            0.40 * los_break_reward,
            0.45 * flash_reward,
            0.22 * stall_penalty,
            0.18 * revisit_penalty,
            0.20 * wall_penalty,
            danger_penalty,
        ]
        return reward_vector, float(sum(reward_vector))
