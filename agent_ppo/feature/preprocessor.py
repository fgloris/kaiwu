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
VIEW_MAP_SIZE = 36

# Anti-box-behavior / ABB 停留惩罚参数
ABB_WINDOW = 8          # 统计最近多少帧位置
ABB_RADIUS = 2           # 若最近位置都落在半径 2 的小框内，则视为停留
ABB_MAX_PENALTY = 0.25   # 最大 ABB 惩罚幅度

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
    将 36x36 灰度图压成单个 01 字符串，并一次 warning 输出。
    规则：>0 的都记为 1，因此 0.5 也会记成 1。
    """
    arr = np.asarray(gray_map)
    assert arr.shape == (VIEW_MAP_SIZE, VIEW_MAP_SIZE), f"expect ({VIEW_MAP_SIZE},{VIEW_MAP_SIZE}), got {arr.shape}"

    s = "".join("1" if v > 0 else "0" for v in arr.reshape(-1))
    logger.warning(f"[{title}]{s}")

def _log_passable_map_and_rays(logger, gray_map, global_rays, move_scores, step_no=None, title="move_debug"):
    """
    精简日志：
    1. map: 36x36 passable map 压平
    2. rays: 全局 rays 的 [angle, score]
    3. move_scores: 8个动作方向分数
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

    move_scores_list = [round(float(x), 4) for x in move_scores]

    payload = {
        "step": None if step_no is None else int(step_no),
        "map": map_bits,
        "rays": rays,
        "move_scores": move_scores_list,
    }

    logger.warning(f"[{title}]{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}")

class Preprocessor:
    def __init__(self, logger=None):
        self.logger = logger
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200

        self.last_monster_dist_norm_1 = -1.0
        self.last_monster_dist_norm_2 = -1.0
        self.last_monster_blocked_1 = False
        self.last_monster_blocked_2 = False

        self.last_total_score = 0.0
        self.last_flash_count = 0

        self.prev_hero_pos = None
        self.recent_positions = deque(maxlen=ABB_WINDOW)

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

    def _open_length(self, hero_x, hero_z, dx, dz, max_len=int(VIEW_MAP_SIZE*0.707)):
        """
        从当前位置沿 (dx, dz) 方向，统计连续可通行格数。
        """
        open_len = 0
        while open_len <= max_len:
            nx = round(hero_x + dx * open_len)
            nz = round(hero_z + dz * open_len)
            if not self._is_global_passable(nx, nz):
                break
            open_len += 1
        return open_len

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
    
    def _score_move_from_rays(self, move_angle, rays, angle_window=30.0):
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

    def _move_safety(self, hero_x, hero_z, monster_positions=None, return_debug=False):
        """
        基于一组全局 rays 的 8 方向 move safety。
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

        move_scores = []
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
                move_scores.append(0.0)
                debug_infos.append(action_debug)
                continue

            move_angle = self._dir8_to_angle_deg(dx, dz)
            move_score, matched_rays = self._score_move_from_rays(
                move_angle=move_angle,
                rays=global_rays,
                angle_window=30.0,
            )

            action_debug["score"] = move_score
            action_debug["rays"] = matched_rays

            move_scores.append(move_score)
            debug_infos.append(action_debug)

        move_scores = np.asarray(move_scores, dtype=np.float32)

        if return_debug:
            return move_scores, debug_infos, global_rays
        return move_scores

    def _compute_abb_penalty(self, cur_hero_pos):
        """ABB penalty: punish staying inside a tiny axis-aligned box for too long.

        ABB 惩罚：若最近若干帧一直停留在一个很小的轴对齐包围盒内，则给予惩罚。
        """
        if cur_hero_pos is None:
            return 0.0

        self.recent_positions.append((int(cur_hero_pos[0]), int(cur_hero_pos[1])))

        if len(self.recent_positions) < ABB_WINDOW:
            return 0.0

        xs = [p[0] for p in self.recent_positions]
        zs = [p[1] for p in self.recent_positions]

        span_x = max(xs) - min(xs)
        span_z = max(zs) - min(zs)

        if span_x <= 2 * ABB_RADIUS and span_z <= 2 * ABB_RADIUS:
            tightness = 1.0 - max(span_x, span_z) / float(max(1, 2 * ABB_RADIUS))
            return -ABB_MAX_PENALTY * float(np.clip(tightness, 0.0, 1.0))

        return 0.0

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
            x0, x1, y0, y1 = self.update_global_maps(hero_pos['x'], hero_pos['z'], map_info)

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
                    map_feat[1, i, j] = float(self.visibility_map[gx, gy])
        
        # _log_gray_map_as_binary(self.logger, map_feat[0], title=f"map:{self.step_no}")
        # _log_gray_map_as_binary(self.logger, map_feat[1], title=f"vis:{self.step_no}")

        # 第三层：monster mask
        # 规则：
        # - 视野内：用精确位置
        # - 视野外但怪物存在：用粗方向 + 桶距离估计位置
        # - 落在 36x36 crop 内则置 1
        for m in monsters[:2]:
            mx, mz = _estimate_monster_pos(hero_pos["x"], hero_pos["z"], m)

            if not (0 <= mx < MAP_SIZE_INT and 0 <= mz < MAP_SIZE_INT):
                continue

            center_i = mx - gx0
            center_j = mz - gy0
            _paint_square(map_feat[2], center_i, center_j, radius=1, value=1.0)

        # 基于全局记忆 + 怪物估计位置，计算 move / flash safety
        # 基于全局记忆 + 怪物估计位置，计算 move safety
        monster_positions = []
        for m in monsters[:2]:
            mx, mz = _estimate_monster_pos(hero_pos["x"], hero_pos["z"], m)
            if 0 <= mx < MAP_SIZE_INT and 0 <= mz < MAP_SIZE_INT:
                monster_positions.append((mx, mz))

        move_safety_feat, _move_debug_infos, _global_rays = self._move_safety(
            hero_pos["x"],
            hero_pos["z"],
            monster_positions,
            return_debug=True,
        )

        # _log_passable_map_and_rays(
        #     self.logger,
        #     map_feat[0],
        #     global_rays,
        #     move_safety_feat,
        #     step_no=self.step_no,
        #     title="move_debug",
        # )

        # 合法动作掩码 (16D)，仅用于 action masking，不再直接拼入 observation 向量
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
        time_before_second_mounster = _norm(max(0, monster_interval - self.step_no), self.max_step)
        
        monster_speedup_time = env_info.get("monster_speed_boost_step", 0)
        self.logger.warning(f"env info: {env_info}, monster speedup time value:{monster_speedup_time}")
        time_before_mounster_speedup = _norm(max(0, monster_speedup_time - self.step_no), self.max_step)
        progress_feat = np.array([step_norm, progress_treasure_collect, time_before_second_mounster, time_before_mounster_speedup], dtype=np.float32)

        # Concatenate features / 拼接特征
        # 这里用 move/flash safety 替换 legal_action 作为 observation 输入，
        # 因此向量总维度保持不变：原先 16 维 legal_action -> 8+8 维 safety
        vector_feat = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                move_safety_feat,
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
        }

        self.prev_hero_pos = (int(hero_pos["x"]), int(hero_pos["z"]))

        return vector_feat, map_feat, reward_feats, legal_action
    
    def calculate_reward(self, env_obs, reward_feats):
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

        cur_blocked_1 = bool(m1[0] > 1e-6)
        cur_blocked_2 = bool(m2[0] > 1e-6)

        if cur_blocked_1:
            los_break_reward += 0.01
        if (not self.last_monster_blocked_1) and cur_blocked_1:
            los_break_reward += 1.0
        if self.last_monster_blocked_1 and (not cur_blocked_1):
            los_break_reward -= 0.5

        if second_exists:
            if cur_blocked_2:
                los_break_reward += 0.01
            if (not self.last_monster_blocked_2) and cur_blocked_2:
                los_break_reward += 1.0
            if self.last_monster_blocked_2 and (not cur_blocked_2):
                los_break_reward -= 0.5
        
        self.last_monster_blocked_1 = cur_blocked_1
        if second_exists:
            self.last_monster_blocked_2 = cur_blocked_2
        
        # 闪现释放奖励
        flash_reward = 0.0
        flash_count = env_info.get("flash_count", 0)
        if (flash_count - self.last_flash_count) > 0:
            flash_reward = los_break_reward + 0.5 * monster_dist_reward
        self.last_flash_count = flash_count

        # ABB 停留惩罚：若最近若干帧一直困在一个小范围内，则惩罚
        cur_hero_pos = reward_feats.get("hero_pos")
        abb_penalty = self._compute_abb_penalty(cur_hero_pos)

        survive_phase_weight = 1.00

        # final step reward vector
        dist_shaping_norm_weight = 12.8

        reward_vector = [
            0.30 * score_gain,
            0.02 * survive_phase_weight,
            3.50 * dist_shaping_norm_weight * monster_dist_reward,
            0.50 * los_break_reward,
            0.25 * flash_reward,
            1.00 * abb_penalty,
        ]

        return reward_vector, sum(reward_vector)
