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

import numpy as np

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
MAP_SIZE_INT = 128
LOCAL_MAP_SIZE = 21
LOCAL_MAP_HALF = 10

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

def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0

def _clip_window(x0, x1, y0, y1, size=MAP_SIZE_INT):
    x0 = max(0, x0)
    x1 = max(0, x1)
    y0 = min(size, y0)
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

def _expand_passable_edges_in_unknown(passable_crop, visible_crop, fill_value=0.5):
    """
    对 25x25 已知区域的四条边做向外延展，只在不可见区域(visible==0)填充 fill_value。
    
    参数:
        passable_crop: shape [36, 36], 当前 map_feat[0]
        visible_crop:  shape [36, 36], 当前 map_feat[1]
    返回:
        修改后的 passable_crop（原地修改并返回）
    """

    h, w = passable_crop.shape
    assert h == 36 and w == 36
    start = (36 - LOCAL_MAP_SIZE) // 2   # 5
    end = start + LOCAL_MAP_SIZE         # 30

    # 已知区域在 [5:30, 5:30]
    top = start
    bottom = end - 1
    left = start
    right = end - 1

    # 1) 左边缘向左扩展
    for r in range(start, end):
        if passable_crop[r, left] > 0.0:
            for c in range(left - 1, -1, -1):
                if visible_crop[r, c] <= 0.0:
                    passable_crop[r, c] = max(passable_crop[r, c], fill_value)

    # 2) 右边缘向右扩展
    for r in range(start, end):
        if passable_crop[r, right] > 0.0:
            for c in range(right + 1, w):
                if visible_crop[r, c] <= 0.0:
                    passable_crop[r, c] = max(passable_crop[r, c], fill_value)

    # 3) 上边缘向上扩展
    for c in range(start, end):
        if passable_crop[top, c] > 0.0:
            for r in range(top - 1, -1, -1):
                if visible_crop[r, c] <= 0.0:
                    passable_crop[r, c] = max(passable_crop[r, c], fill_value)

    # 4) 下边缘向下扩展
    for c in range(start, end):
        if passable_crop[bottom, c] > 0.0:
            for r in range(bottom + 1, h):
                if visible_crop[r, c] <= 0.0:
                    passable_crop[r, c] = max(passable_crop[r, c], fill_value)

    return passable_crop

def _log_gray_map_as_binary(logger, gray_map, title="map36"):
    """
    将 36x36 灰度图压成单个 01 字符串，并一次 warning 输出。
    规则：>0 的都记为 1，因此 0.5 也会记成 1。
    """
    arr = np.asarray(gray_map)
    assert arr.shape == (36, 36), f"expect (36,36), got {arr.shape}"

    s = "".join("1" if v > 0 else "0" for v in arr.reshape(-1))
    logger.warning(f"[{title}]{s}")

class Preprocessor:
    def __init__(self, logger=None):
        self.logger = logger
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200

        self.last_monster_dist_norm_1 = -1.0
        self.last_monster_dist_norm_2 = -1.0

        self.last_total_score = 0.0
        self.last_treasure_collected = 0
        self.last_collected_buff = 0
        self.last_flash_count = 0

        self.last_treasure_dist_norm_1 = 0.0
        self.last_treasure_dist_norm_2 = 0.0
        self.last_buff_dist_norm_1 = 0.0
        self.last_buff_dist_norm_2 = 0.0

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
                    dist_norm = _bucketize_left(_norm(raw_dist, MAP_SIZE * 1.41), 9)

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

        # buff和宝箱特征
        organs = frame_state.get("organs", [])

        # 前2个宝箱 / buff：每个5维 [rel_x, rel_z, dist_norm, dir_x, dir_z]
        treasure_feat = np.zeros(10, dtype=np.float32)
        buff_feat = np.zeros(10, dtype=np.float32)

        treasures = []
        buffs = []

        for organ in organs:
            if organ.get("status", 0) != 1:
                continue

            sub_type = organ.get("sub_type", 0)
            organ_pos = organ["pos"]

            dx = float(organ_pos["x"] - hero_pos["x"])
            dz = float(organ_pos["z"] - hero_pos["z"])
            raw_dist = np.sqrt(dx * dx + dz * dz)

            # 存起来，方便排序和后续直接用
            organ["raw_dist"] = raw_dist
            organ["_dx"] = dx
            organ["_dz"] = dz

            if sub_type == 1:
                treasures.append(organ)
            elif sub_type == 2:
                buffs.append(organ)

        treasures.sort(key=lambda x: x.get("raw_dist", 1e9))
        buffs.sort(key=lambda x: x.get("raw_dist", 1e9))

        for i, organ in enumerate(treasures[:2]):
            dx = organ["_dx"]
            dz = organ["_dz"]
            raw_dist = organ.get("raw_dist", MAP_SIZE * 1.41)

            rel_x = float(np.clip(dx / MAP_SIZE, -1.0, 1.0))
            rel_z = float(np.clip(dz / MAP_SIZE, -1.0, 1.0))
            dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)

            # 优先用连续方向；太近时退化到离散方向
            if raw_dist > 1e-6:
                dir_x = dx / raw_dist
                dir_z = dz / raw_dist
            else:
                dir_idx = int(m.get("hero_relative_direction", 0))
                dir_x, dir_z = DIR9_TO_VEC.get(dir_idx, (0.0, 0.0))

            treasure_feat[i * 5 : i * 5 + 5] = np.array(
                [rel_x, rel_z, dist_norm, dir_x, dir_z],
                dtype=np.float32,
            )

        for i, organ in enumerate(buffs[:2]):
            dx = organ["_dx"]
            dz = organ["_dz"]
            raw_dist = organ.get("raw_dist", MAP_SIZE * 1.41)

            rel_x = float(np.clip(dx / MAP_SIZE, -1.0, 1.0))
            rel_z = float(np.clip(dz / MAP_SIZE, -1.0, 1.0))
            dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)

            if raw_dist > 1e-6:
                dir_x = dx / raw_dist
                dir_z = dz / raw_dist
            else:
                dir_idx = int(m.get("hero_relative_direction", 0))
                dir_x, dir_z = DIR9_TO_VEC.get(dir_idx, (0.0, 0.0))

            buff_feat[i * 5 : i * 5 + 5] = np.array(
                [rel_x, rel_z, dist_norm, dir_x, dir_z],
                dtype=np.float32,
            )

        if map_info is not None:
            x0, x1, y0, y1 = self.update_global_maps(hero_pos['x'], hero_pos['z'], map_info)

        map_feat = np.zeros((3, 36, 36), dtype=np.float32)

        crop_size = 36
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
        
        _log_gray_map_as_binary(self.logger, map_feat[0], title=f"before:{self.step_no}")
        _log_gray_map_as_binary(self.logger, map_feat[1], title=f"vis:{self.step_no}")
        map_feat[0] = _expand_passable_edges_in_unknown(map_feat[0], map_feat[1], fill_value=0.5)
        _log_gray_map_as_binary(self.logger, map_feat[0], title=f"after:{self.step_no}")

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

        # 合法动作掩码 (8D)
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
        assert monster_interval > 0, f"monster insterval < 0! value:{monster_interval}"
        time_before_second_mounster = _norm(max(0, monster_interval - self.step_no), self.max_step)
        has_monster_speedup = 0.0 if env_info.get("monster_speed", 0) <= 1 else 1.0
        progress_feat = np.array([step_norm, progress_treasure_collect, time_before_second_mounster, has_monster_speedup], dtype=np.float32)

        # Concatenate features / 拼接特征
        vector_feat = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                treasure_feat,
                buff_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        )

        reward_feats = {
            "monster_feats": monster_feats,
            "monster_feats_available": len(monsters),
            "treasure_feats": treasure_feat,
            "treasure_feats_available": len(treasures),
            "buff_feats": buff_feat,
            "buff_feats_available": len(buffs),
            "progress_feats": progress_feat,
            "hero_pos": (int(hero_pos["x"]), int(hero_pos["z"])),
            "prev_hero_pos": self.prev_hero_pos,
            "last_action": int(last_action),
        }

        self.prev_hero_pos = (int(hero_pos["x"]), int(hero_pos["z"]))

        return vector_feat, map_feat, reward_feats, legal_action
    
    def calculate_reward(self, env_obs, reward_feats):
        # 基于比赛分数增量的奖励
        env_info = env_obs["observation"].get("env_info", {})
        cur_total_score = float(env_info.get("total_score", 0.0))
        score_gain = cur_total_score - self.last_total_score
        self.last_total_score = cur_total_score
        
        # 怪物 dist shaping
        # 防止首帧的错误 reward

        monster_dist_reward = 0.0
        if self.last_monster_dist_norm_1 >= 0  and self.last_monster_dist_norm_2 >= 0:
            monster_dist_reward = \
                ( reward_feats['monster_feats'][0][4] - self.last_monster_dist_norm_1) + \
                0.2 * (reward_feats['monster_feats'][1][4] - self.last_monster_dist_norm_2)
            
        self.last_monster_dist_norm_1 = reward_feats['monster_feats'][0][4]
        self.last_monster_dist_norm_2 = reward_feats['monster_feats'][1][4]

        # buff和宝箱 distance shaping
        # 靠近奖励但远离不惩罚

        treasure_dist_norm_1 = 0.0
        treasure_dist_norm_2 = 0.0
        if reward_feats['treasure_feats_available'] > 0: 
            treasure_dist_norm_1 = reward_feats['treasure_feats'][2]
            
        if reward_feats['treasure_feats_available'] > 1:
            treasure_dist_norm_2 = reward_feats['treasure_feats'][7]

        treasure_dist_reward = max(0.0, (self.last_treasure_dist_norm_1 - treasure_dist_norm_1) + 
                            0.2 * (self.last_treasure_dist_norm_2 - treasure_dist_norm_2))
        
        self.last_treasure_dist_norm_1 = treasure_dist_norm_1
        self.last_treasure_dist_norm_2 = treasure_dist_norm_2

        buff_dist_norm_1 = 0.0
        buff_dist_norm_2 = 0.0
        if reward_feats['buff_feats_available'] > 0: 
            buff_dist_norm_1 = reward_feats['buff_feats'][2]
            
        if reward_feats['buff_feats_available'] > 1:
            buff_dist_norm_2 = reward_feats['buff_feats'][7]

        buff_dist_reward = max(0.0, (self.last_buff_dist_norm_1 - buff_dist_norm_1) + 
                        0.2 * (self.last_buff_dist_norm_2 - buff_dist_norm_2))
        
        self.last_buff_dist_norm_1 = buff_dist_norm_1
        self.last_buff_dist_norm_2 = buff_dist_norm_2

        # 宝箱收集奖励
        cur_treasure_collected = int(env_obs["observation"]["frame_state"]["heroes"].get("treasure_collected_count", 0))
        treasure_gain = cur_treasure_collected - self.last_treasure_collected
        self.last_treasure_collected = cur_treasure_collected

        treasure_reward = max(0, treasure_gain)

        # buff收集奖励
        cur_collected_buff = int(env_info.get("collected_buff", 0))
        buff_gain = cur_collected_buff - self.last_collected_buff
        self.last_collected_buff = cur_collected_buff

        buff_reward = max(0, buff_gain)

        # 闪现释放奖励
        flash_reward = 0.0
        flash_count = env_info.get("flash_count", 0)
        if (flash_count - self.last_flash_count) > 0:
            flash_reward = 0.5 * monster_dist_reward + 0.5 * treasure_dist_reward + 0.1 * buff_dist_reward
        self.last_flash_count = flash_count

        # 撞墙惩罚
        wall_penalty = 0.0
        prev_hero_pos = reward_feats.get("prev_hero_pos")
        cur_hero_pos = reward_feats.get("hero_pos")

        if prev_hero_pos is not None and cur_hero_pos is not None:
            dx = cur_hero_pos[0] - prev_hero_pos[0]
            dz = cur_hero_pos[1] - prev_hero_pos[1]
            moved = (dx != 0) or (dz != 0)
            if not moved:
                wall_penalty = -0.2

        if reward_feats["progress_feats"][2] > 0: # time before second monseter
            # 早期：鼓励探索和拿宝箱
            treasure_phase_weight = 1.20
            survive_phase_weight = 0.85
        elif reward_feats["progress_feats"][3] == 0: # has monster speedup
            # 中期：逐步平衡
            treasure_phase_weight = 1.00
            survive_phase_weight = 1.00
        else:
            # 后期：怪物加速后，生存优先
            treasure_phase_weight = 0.75
            survive_phase_weight = 1.50

        # final step reward vector
        dist_shaping_norm_weight = 12.8

        reward_vector = [
            0.30 * score_gain,
            survive_phase_weight * 0.02,
            0.35 * dist_shaping_norm_weight * monster_dist_reward,
            5.00 * treasure_phase_weight * treasure_reward,
            0.25 * treasure_phase_weight * dist_shaping_norm_weight * treasure_dist_reward,
            0.35 * buff_reward,
            0.05 * dist_shaping_norm_weight * buff_dist_reward,
            0.25 * flash_reward,
            1.00 * wall_penalty,
        ]

        return reward_vector, sum(reward_vector)
