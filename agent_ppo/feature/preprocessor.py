#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
# 旧版本
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

import json
import numpy as np
from collections import deque
from agent_ppo.feature.curriculum import REWARD_CONFIG

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
MAP_SIZE_INT = 128
LOCAL_MAP_SIZE = 21
LOCAL_MAP_HALF = 10
VIEW_MAP_SIZE = 21

# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 2.0
# Monster distance bucket / 怪物距离桶
MONSTER_DIST_BUCKET_MAX = 7.0
MONSTER_DIST_BUCKET_EDGES = (0.0, 4.0, 10.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0)
OLD_MONSTER_DIST_BUCKET_MAX = 5
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


def _get_reward_config():
    return REWARD_CONFIG

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

def _monster_dist_bucket_norm_from_raw(dist):
    dist = float(np.clip(dist, MONSTER_DIST_BUCKET_EDGES[0], MONSTER_DIST_BUCKET_EDGES[-1]))
    bucket_idx = len(MONSTER_DIST_BUCKET_EDGES) - 2
    for i in range(len(MONSTER_DIST_BUCKET_EDGES) - 1):
        left = MONSTER_DIST_BUCKET_EDGES[i]
        right = MONSTER_DIST_BUCKET_EDGES[i + 1]
        if i == len(MONSTER_DIST_BUCKET_EDGES) - 2:
            if left <= dist <= right:
                bucket_idx = i
                break
        elif left <= dist < right:
            bucket_idx = i
            break
    return float(bucket_idx) / MONSTER_DIST_BUCKET_MAX

def _monster_dist_bucket_norm_from_env_bucket(old_bucket):
    old_bucket = int(np.clip(old_bucket, 0, OLD_MONSTER_DIST_BUCKET_MAX))
    new_bucket = old_bucket + 2
    return float(new_bucket) / MONSTER_DIST_BUCKET_MAX

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

        self.last_treasure_score = 0.0
        self.last_flash_count = 0
        self.last_is_dangerous = False
        self.last_hero_pos = None
        self.last_connected_opening_count = 0
        self.last_monster_to_agent_vecs = [None, None]

        self.pos_history = deque(maxlen=8)
        self.abb_safe_score = 4.0

        self.last_treasure_dist_norm = -1.0
        self.last_buff_dist_norm = -1.0

        # 全局物件记忆：只保留 buff；宝箱被捡走后不会刷新，直接使用当前帧 organs
        self.buff_memory = {}
        self.last_collected_buff = 0
        self.buff_refresh_time = 200

        # 视野外怪物速度先验
        self.last_seen_monster_speed = [1, 1]

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

        return bool(self.visibility_map[x, z] > 0 and self.passable_map[x, z] == 0)

    def _did_segment_cross_known_wall(self, start_pos, end_pos):
        if start_pos is None or end_pos is None:
            return False

        x0, z0 = int(start_pos[0]), int(start_pos[1])
        x1, z1 = int(end_pos[0]), int(end_pos[1])
        dx = x1 - x0
        dz = z1 - z0
        steps = int(max(abs(dx), abs(dz)) * 2)
        if steps <= 1:
            return False

        for i in range(1, steps):
            t = i / float(steps)
            x = int(round(x0 + dx * t))
            z = int(round(z0 + dz * t))
            if (x, z) == (x0, z0) or (x, z) == (x1, z1):
                continue
            if self._is_known_wall(x, z):
                return True
        return False

    def _monster_to_agent_vector(self, monster_feat):
        rel_x = float(monster_feat[2])
        rel_z = float(monster_feat[3])
        vec = np.asarray([-rel_x, -rel_z], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            return None
        return vec / norm

    def _did_cross_monster_by_angle(self, cur_monster_vecs, angle_threshold=150.0):
        for last_vec, cur_vec in zip(self.last_monster_to_agent_vecs, cur_monster_vecs):
            if last_vec is None or cur_vec is None:
                continue
            cos_angle = float(np.clip(np.dot(last_vec, cur_vec), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos_angle)))
            if not angle > angle_threshold:
                return False
        return True

    def _nearest_monster_grid_distance(self, monster_feats, active_monster_count):
        dists = []
        for monster_feat in monster_feats[:active_monster_count]:
            rel_x = float(monster_feat[2]) * MAP_SIZE
            rel_z = float(monster_feat[3]) * MAP_SIZE
            dists.append(float(np.hypot(rel_x, rel_z)))
        return min(dists) if dists else None

    def _is_monster_near(self, monster_feats, active_monster_count, env_info):
        nearest_dist = self._nearest_monster_grid_distance(monster_feats, active_monster_count)
        monster_speedup_step = int(env_info.get("monster_speed_boost_step", 0))
        is_speedup = monster_speedup_step > 0 and self.step_no >= monster_speedup_step
        near_threshold = 8.0 if is_speedup else 4.0
        return nearest_dist is not None and nearest_dist <= near_threshold

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
        return dir_scores, {
            "boundary_pts": boundary_pts,
            "clusters": clusters,
            "connected_clusters": connected_clusters,
            "connected_opening_count": len(connected_clusters),
            "masked_local_passable": np.array(local_passable, copy=True),
        }

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

    def _is_in_current_view(self, obj_pos, hero_pos):
        """
        判断某个全局坐标是否落在当前 hero 为中心的 21x21 视野内。
        视野范围：x/z 各自相对 hero 在 [-10, 10] 内。
        """
        x, z = int(obj_pos[0]), int(obj_pos[1])
        hero_x = int(hero_pos["x"])
        hero_z = int(hero_pos["z"])
        return abs(x - hero_x) <= LOCAL_MAP_HALF and abs(z - hero_z) <= LOCAL_MAP_HALF

    def _update_organ_memory(self, env_info, organs, hero_pos):
        """
        organ memory 更新规则：

        1. 当前 organs 里出现的新 id：
        - 加入 memory
        - available = status
        - buff 的 respawn_step = -1

        2. 当前 organs 里出现的旧 id：
        - 刷新位置
        - available = status
        - buff 的 respawn_step = -1

        3. 对所有 memory 中“位置在当前 21x21 视野内，但本帧没出现在 organs 里”的物体：
        - 若 available 原本为 True，则置 False
        - 若是 buff，同时写 respawn_step = step_no + buff_refresh_time

        4. 对所有 buff：
        - 若 respawn_step 到时，则 available = True, respawn_step = -1
        """
        self.buff_refresh_time = int(env_info.get("buff_refresh_time", self.buff_refresh_time))

        # 当前帧在视野内真正出现过的 id
        visible_buff_ids = set()

        # 1) 先处理当前 organs：出现了就记为 available=True
        for organ in organs:
            config_id = int(organ.get("config_id", -1))
            if config_id < 0:
                continue

            sub_type = int(organ.get("sub_type", 0))  # 1=treasure, 2=buff
            pos = organ.get("pos", {}) or {}
            x = int(pos.get("x", 0))
            z = int(pos.get("z", 0))
            available = bool(organ.get("status", 1) == 1)

            if sub_type == 2:
                visible_buff_ids.add(config_id)

                mem = self.buff_memory.setdefault(
                    config_id,
                    {"pos": (x, z), "available": True, "respawn_step": -1}
                )
                mem["pos"] = (x, z)
                mem["available"] = available
                mem["respawn_step"] = -1

        # 2) buff: mark missing visible buffs unavailable and start respawn tracking.
        for bid, mem in self.buff_memory.items():
            if not self._is_in_current_view(mem["pos"], hero_pos):
                continue
            if bid in visible_buff_ids:
                continue
            if mem.get("available", False):
                mem["available"] = False
                mem["respawn_step"] = self.step_no + self.buff_refresh_time

        # 4) buff 到刷新时间则重新可用
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
            dir_x = dx / dist if dist > 1e-6 else 0.0
            dir_z = dz / dist if dist > 1e-6 else 0.0
            items.append({
                "id": int(obj_id),
                "pos": (int(x), int(z)),
                "available": available,
                "dist": dist,
                "dist_norm": _norm(dist, MAP_SIZE * 1.41),
                "feat": np.array([
                    np.clip(dir_x, -1.0, 1.0),
                    np.clip(dir_z, -1.0, 1.0),
                    _norm(dist, MAP_SIZE * 1.41),
                    1.0 if available else 0.0,
                ], dtype=np.float32),
            })

        items.sort(key=lambda x: (0 if x["available"] else 1, x["dist"]))

        feat_list = []
        for item in items[:topk]:
            feat_list.append(item["feat"])
        while len(feat_list) < topk:
            feat_list.append(np.zeros(4, dtype=np.float32))

        best_dist_norm = -1.0
        for item in items:
            if item["available"]:
                best_dist_norm = float(item["dist_norm"])
                break

        return np.concatenate(feat_list, axis=0), items, best_dist_norm

    def _build_current_treasure_features(self, hero_pos, organs, topk=2):
        hero_x = int(hero_pos["x"])
        hero_z = int(hero_pos["z"])
        items = []

        for organ in organs:
            if int(organ.get("sub_type", 0)) != 1:
                continue
            if int(organ.get("status", 0)) != 1:
                continue

            config_id = int(organ.get("config_id", -1))
            pos = organ.get("pos", {}) or {}
            x = int(pos.get("x", -1))
            z = int(pos.get("z", -1))
            if not (0 <= x < MAP_SIZE_INT and 0 <= z < MAP_SIZE_INT):
                continue

            dx = float(x - hero_x)
            dz = float(z - hero_z)
            dist = float(np.hypot(dx, dz))
            dir_x = dx / dist if dist > 1e-6 else 0.0
            dir_z = dz / dist if dist > 1e-6 else 0.0
            dist_norm = _norm(dist, MAP_SIZE * 1.41)
            items.append({
                "id": config_id,
                "pos": (x, z),
                "available": True,
                "dist": dist,
                "dist_norm": dist_norm,
                "in_view": self._is_in_current_view((x, z), hero_pos),
                "feat": np.array([
                    np.clip(dir_x, -1.0, 1.0),
                    np.clip(dir_z, -1.0, 1.0),
                    dist_norm,
                    1.0,
                ], dtype=np.float32),
            })

        items.sort(key=lambda x: x["dist"])

        feat_list = []
        for item in items[:topk]:
            feat_list.append(item["feat"])
        while len(feat_list) < topk:
            feat_list.append(np.zeros(4, dtype=np.float32))

        best_dist_norm = float(items[0]["dist_norm"]) if items else -1.0
        return np.concatenate(feat_list, axis=0), items, best_dist_norm

    def _should_cut_treasure_by_monster_angle(self, hero_pos, treasure_items, monsters):
        if not treasure_items:
            return False

        nearest_treasure = treasure_items[0]
        if not nearest_treasure.get("in_view", False):
            return False

        hero_x = float(hero_pos["x"])
        hero_z = float(hero_pos["z"])
        treasure_pos = nearest_treasure["pos"]
        treasure_vec = np.asarray(
            [float(treasure_pos[0]) - hero_x, float(treasure_pos[1]) - hero_z],
            dtype=np.float32,
        )
        treasure_norm = float(np.linalg.norm(treasure_vec))
        if treasure_norm <= 1e-6:
            return False
        treasure_vec = treasure_vec / treasure_norm

        nearest_monster_vec = None
        nearest_monster_dist = None
        for monster in monsters[:2]:
            if int(monster.get("is_in_view", 0)) <= 0:
                continue
            pos = monster.get("pos", {}) or {}
            mx = float(pos.get("x", hero_x))
            mz = float(pos.get("z", hero_z))
            vec = np.asarray([mx - hero_x, mz - hero_z], dtype=np.float32)
            dist = float(np.linalg.norm(vec))
            if dist <= 1e-6:
                continue
            if nearest_monster_dist is None or dist < nearest_monster_dist:
                nearest_monster_dist = dist
                nearest_monster_vec = vec / dist

        if nearest_monster_vec is None:
            return False

        cos_angle = float(np.clip(np.dot(nearest_monster_vec, treasure_vec), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_angle)))
        return angle < 45.0

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


    def _is_reachable_in_known_map(self, start, goal):
        """
        仅在当前 21x21 视野内，用“hero 到目标的直线是否被阻挡”判断可达性。
        不使用 A*，也不绕路。
        """
        sx, sz = int(start[0]), int(start[1])
        gx, gz = int(goal[0]), int(goal[1])

        if not (0 <= sx < MAP_SIZE_INT and 0 <= sz < MAP_SIZE_INT):
            return False
        if not (0 <= gx < MAP_SIZE_INT and 0 <= gz < MAP_SIZE_INT):
            return False

        if not self._is_global_passable(sx, sz):
            return False
        if not self._is_global_passable(gx, gz):
            return False

        # 只判断 hero 当前 21x21 视野内的目标
        if abs(gx - sx) > LOCAL_MAP_HALF or abs(gz - sz) > LOCAL_MAP_HALF:
            return False

        if (sx, sz) == (gx, gz):
            return True

        dx = gx - sx
        dz = gz - sz
        steps = max(abs(dx), abs(dz))
        if steps <= 0:
            return True

        for t in range(1, steps + 1):
            alpha = t / float(steps)
            x = int(round(sx + dx * alpha))
            z = int(round(sz + dz * alpha))

            if not (0 <= x < MAP_SIZE_INT and 0 <= z < MAP_SIZE_INT):
                return False
            if self.visibility_map[x, z] == 0:
                return False
            if self._is_known_wall(x, z):
                return False

        return True


    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]
        # self.logger.warning(f"legal_action: {legal_act_raw}")

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

        if map_info is not None:
            x0, x1, y0, y1, newly_discovered_passable_count = self.update_global_maps(hero_pos['x'], hero_pos['z'], map_info)

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []

        for i in range(2):
            if i < len(monsters):
                m = monsters[i]

                # 视野外时，hero_relative_direction 和 hero_l2_distance 仍然可用
                is_in_view = float(m.get("is_in_view", 0))
                if is_in_view:
                    self.last_seen_monster_speed[i] = max(1, int(m.get("speed", 1)))
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)
                else:
                    m_speed_norm = _norm(self.last_seen_monster_speed[i], MAX_MONSTER_SPEED)

                rel_x = 0.0
                rel_z = 0.0

                # 先给默认值：视野外时只保留粗信息
                dir_idx = int(m.get("hero_relative_direction", 0))
                dir_x, dir_z = DIR9_TO_VEC.get(dir_idx, (0.0, 0.0))

                dist_norm = _monster_dist_bucket_norm_from_env_bucket(
                    m.get("hero_l2_distance", OLD_MONSTER_DIST_BUCKET_MAX)
                )

                if is_in_view:
                    m_pos = m["pos"]
                    dx = float(m_pos["x"] - hero_pos["x"])
                    dz = float(m_pos["z"] - hero_pos["z"])

                    # 精细相对位置：保留正负号
                    rel_x = float(np.clip(dx / MAP_SIZE, -1.0, 1.0))
                    rel_z = float(np.clip(dz / MAP_SIZE, -1.0, 1.0))

                    raw_dist = np.sqrt(dx * dx + dz * dz)
                    dist_norm = _monster_dist_bucket_norm_from_raw(raw_dist)

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

        organs = frame_state.get("organs", [])
        self._update_organ_memory(env_info, organs, hero_pos)
        treasure_feat, treasure_items, nearest_treasure_dist_norm = self._build_current_treasure_features(
            hero_pos, organs, topk=2
        )
        buff_feat, buff_items, nearest_buff_dist_norm = self._build_target_features(
            hero_pos, self.buff_memory, topk=2, prefer_available_only=False
        )
        cut_treasure_by_monster_angle = self._should_cut_treasure_by_monster_angle(
            hero_pos, treasure_items, monsters
        )

        # 地图特征
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
            if not self._is_reachable_in_known_map((int(hero_pos["x"]), int(hero_pos["z"])),(ox, oz)):
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
            local_passable_21
        )
        connected_opening_count_raw = int(boundary_cluster_info["connected_opening_count"])
        connected_opening_count = _norm(connected_opening_count_raw, 5)
        active_monster_count = min(2, max(0, len(monsters)))
        is_monster_near = self._is_monster_near(monster_feats, active_monster_count, env_info)
        is_dangerous = bool(is_monster_near or connected_opening_count_raw <= 1)
        
        situation_feat = np.array([connected_opening_count, float(is_dangerous)], dtype=np.float32)

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
            "last_action": int(last_action),
            "newly_discovered_passable_count": int(newly_discovered_passable_count),
            "connected_boundary_clusters": boundary_cluster_info["connected_clusters"],
            "connected_opening_count": connected_opening_count_raw,
            "is_dangerous": is_dangerous,
            "cut_treasure_by_monster_angle": bool(cut_treasure_by_monster_angle),
            "nearest_treasure_dist_norm": float(nearest_treasure_dist_norm),
            "nearest_buff_dist_norm": float(nearest_buff_dist_norm),
        }

        self.pos_history.append((int(hero_pos["x"]), int(hero_pos["z"])))

        return vector_feat, map_feat, reward_feats, legal_action
    
    def calculate_reward(self, env_obs, reward_feats):
        self.total_train_steps += 1
        # 1.若怪物在视野外，让模型跑得更远一点。                            --> 通过 monster dist shaping? 这个足以做到吗？
        # 2.若怪物在视野内且附近有弯道，让模型尽快将其拉脱视野。             --> 加一点视野脱离奖励？monster dist shaping?
        # 3.尽量不要撞墙。1.不要撞侧面的墙。2.不要走进死胡同。              --> 计算路径方向？
        # 4.不要原地打转。                                               --> ABB惩罚？好像做不到。方向一致性惩罚？

        # 基于宝箱分数增量的奖励
        env_info = env_obs["observation"].get("env_info", {})
        cur_treasure_score = float(env_info.get("treasure_score", 0.0))
        treasure_score_gain = cur_treasure_score - self.last_treasure_score
        self.last_treasure_score = cur_treasure_score
        
        # 怪物 dist shaping
        second_exists = bool(reward_feats['progress_feats'][2] < 1e-6)

        monster_dist_reward = 0.0
        cur_hero_pos = reward_feats.get("hero_pos")

        m1 = reward_feats['monster_feats'][0]
        m2 = reward_feats['monster_feats'][1]
        active_monster_count = min(2, max(0, int(reward_feats.get("monster_feats_available", 0))))
        is_monster_near = self._is_monster_near(
            reward_feats["monster_feats"],
            active_monster_count,
            env_info,
        )

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
            los_break_reward += 0.8
        if self.last_monster_invisible_1 and (not cur_invisible_1):
            los_break_reward -= 0.8

        if second_exists:
            if cur_invisible_2:
                los_break_reward += 0.01
            if (not self.last_monster_invisible_2) and cur_invisible_2:
                los_break_reward += 0.8
            if self.last_monster_invisible_2 and (not cur_invisible_2):
                los_break_reward -= 0.8
        
        self.last_monster_invisible_1 = cur_invisible_1
        if second_exists:
            self.last_monster_invisible_2 = cur_invisible_2
        else:
            self.last_monster_invisible_2 = False
        
        # 局势相关 reward settings
        cur_is_dangerous = bool(reward_feats.get("is_dangerous", False))
        cur_opening_count = int(reward_feats.get("connected_opening_count", 0))
        last_opening_count = int(self.last_connected_opening_count)
        danger_penalty = -1.0 if cur_is_dangerous else 0.0


        # 靠墙惩罚：只在 hero 周围 5x5 小窗口内查最近已知墙，减少计算量
        near_wall_penalty = 0.0
        if cur_hero_pos is not None:
            near_wall_penalty = self._compute_near_wall_penalty(
                cur_hero_pos[0],
                cur_hero_pos[1],
                search_radius=3,
            )

        # abb 惩罚项
        abb_score, abb_penalty = self._compute_abb_penalty(cur_hero_pos)
        
        # 闪现只奖励受困追逃局势下的有效逃生，非受困乱闪会被惩罚。
        flash_reward = 0.0

        flash_count = int(env_info.get("flash_count", self.last_flash_count))
        used_flash = (flash_count - self.last_flash_count) > 0
        danger_decreased = self.last_is_dangerous and cur_is_dangerous > 1
        flash_move_dist = None
        if used_flash and self.last_hero_pos is not None and cur_hero_pos is not None:
            flash_move_dist = float(np.hypot(
                float(cur_hero_pos[0]) - float(self.last_hero_pos[0]),
                float(cur_hero_pos[1]) - float(self.last_hero_pos[1]),
            ))
        flash_hit_wall = flash_move_dist is not None and flash_move_dist <= 5.0
        crossed_wall = used_flash and self._did_segment_cross_known_wall(self.last_hero_pos, cur_hero_pos)
        cur_monster_to_agent_vecs = [
            self._monster_to_agent_vector(m1),
            self._monster_to_agent_vector(m2) if second_exists else None,
        ]
        crossed_monster = used_flash and self.last_is_dangerous and self._did_cross_monster_by_angle(
            cur_monster_to_agent_vecs,
            angle_threshold=150.0,
        )

        flash_reward = 0.0
        if used_flash:
            if flash_hit_wall:
                flash_reward = -4.0
            elif danger_decreased and crossed_wall:
                flash_reward = 8.0
            elif crossed_monster:
                flash_reward = 8.0
            else:
                flash_reward = -1.0

        self.last_flash_count = flash_count
        self.last_is_dangerous = cur_is_dangerous
        self.last_hero_pos = cur_hero_pos
        self.last_connected_opening_count = cur_opening_count
        self.last_monster_to_agent_vecs = cur_monster_to_agent_vecs

        # buff 奖励
        monster_goingto_speedup = bool(reward_feats['progress_feats'][3] < 100)

        collected_buff = int(env_info.get("collected_buff", self.last_collected_buff))
        buff_delta = float(max(0, collected_buff - self.last_collected_buff))
        self.last_collected_buff = collected_buff
        buff_pick_reward = buff_delta * (40.0 if monster_goingto_speedup else 20.0)

        # 生存奖励
        survive_reward = 1.00 + (self.step_no / 200)
        if abb_score < self.abb_safe_score:
            survive_reward = 0.0
            los_break_reward = min(los_break_reward, 0.0)

        # treasure dist reward
        treasure_dist_reward = self._compute_positive_dist_shaping(
            reward_feats.get("nearest_treasure_dist_norm", -1.0),
            "last_treasure_dist_norm",
        )

        # buff dist reward
        buff_dist_reward = 0.4 * self._compute_positive_dist_shaping(
            reward_feats.get("nearest_buff_dist_norm", -1.0),
            "last_buff_dist_norm",
        )

        # treasure score gain is ignored while a monster is too close or blocks the path to the treasure.
        if is_monster_near or bool(reward_feats.get("cut_treasure_by_monster_angle", False)):
            treasure_score_gain = -5.0
            treasure_dist_reward = -5.0

        # final step reward vector
        dist_shaping_norm_weight = 12.8

        # ============== 最终奖励向量 ==============

        reward_config = _get_reward_config()
        reward_vector = [
            reward_config.treasure_score_gain * treasure_score_gain,
            reward_config.survival * reward_config.survival_multiplier * survive_reward,
            reward_config.los_break * los_break_reward,
            reward_config.flash * flash_reward,
            reward_config.wall_penalty * near_wall_penalty,
            reward_config.abb_penalty * abb_penalty,
            reward_config.danger_penalty * danger_penalty,
            reward_config.treasure_dist * dist_shaping_norm_weight * treasure_dist_reward,
            reward_config.buff_dist * dist_shaping_norm_weight * buff_dist_reward,
            reward_config.survival_multiplier * reward_config.buff_pick * buff_pick_reward,
            abs(reward_config.monster_dist * dist_shaping_norm_weight * monster_dist_reward),
        ]

        return reward_vector, sum(reward_vector[:-1]) + 1.50 * dist_shaping_norm_weight * monster_dist_reward
