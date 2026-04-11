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

from collections import deque
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

# 局部细化时，给视野窗口额外扩一个边，减轻边界伪影
THIN_MARGIN = 3
# thinning 最大迭代次数；不宜太大，避免每步开销上升
THIN_MAX_ITER = 8

# 拓扑距离归一化
MAX_TOPO_DIST = 1000

# 角度转向量特征
DIR8_TO_VEC = {
    0: (1.0, 0.0),
    1: (1 / np.sqrt(2), -1 / np.sqrt(2)),
    2: (0.0, -1.0),
    3: (-1 / np.sqrt(2), -1 / np.sqrt(2)),
    4: (-1.0, 0.0),
    5: (-1 / np.sqrt(2), 1 / np.sqrt(2)),
    6: (0.0, 1.0),
    7: (1 / np.sqrt(2), 1 / np.sqrt(2)),
}


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _clip_window(x0, x1, y0, y1, size=MAP_SIZE_INT):
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(size, x1)
    y1 = min(size, y1)
    return x0, x1, y0, y1


def _neighbors(x, y, img01):
    # P2, P3, ..., P9
    return [
        img01[x - 1, y],     # P2
        img01[x - 1, y + 1], # P3
        img01[x,     y + 1], # P4
        img01[x + 1, y + 1], # P5
        img01[x + 1, y],     # P6
        img01[x + 1, y - 1], # P7
        img01[x,     y - 1], # P8
        img01[x - 1, y - 1], # P9
    ]


def _transitions(neigh):
    seq = neigh + [neigh[0]]
    return sum((seq[i] == 0 and seq[i + 1] == 1) for i in range(8))


def zhang_suen_thinning(binary01, max_iter=None):
    """
    输入: 0/1 二值图
    输出: 0/1 二值图
    作用: 在尽量保持连通性的前提下，对亮区域做“拓扑保持削薄”
    """
    img = (binary01 > 0).astype(np.uint8).copy()
    h, w = img.shape

    if h < 3 or w < 3:
        return img

    changed = True
    it = 0

    while changed:
        changed = False
        to_delete = []

        # step 1
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if img[x, y] != 1:
                    continue
                n = _neighbors(x, y, img)
                s = sum(n)
                t = _transitions(n)
                if (
                    2 <= s <= 6 and
                    t == 1 and
                    n[0] * n[2] * n[4] == 0 and
                    n[2] * n[4] * n[6] == 0
                ):
                    to_delete.append((x, y))

        if to_delete:
            changed = True
            for x, y in to_delete:
                img[x, y] = 0

        to_delete = []

        # step 2
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if img[x, y] != 1:
                    continue
                n = _neighbors(x, y, img)
                s = sum(n)
                t = _transitions(n)
                if (
                    2 <= s <= 6 and
                    t == 1 and
                    n[0] * n[2] * n[6] == 0 and
                    n[0] * n[4] * n[6] == 0
                ):
                    to_delete.append((x, y))

        if to_delete:
            changed = True
            for x, y in to_delete:
                img[x, y] = 0

        it += 1
        if max_iter is not None and it >= max_iter:
            break

    return img


def uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def uf_union(parent, size, a, b):
    ra = uf_find(parent, a)
    rb = uf_find(parent, b)
    if ra == rb:
        return
    if size[ra] < size[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    size[ra] += size[rb]


def build_largest_connected_component(thin_01):
    """
    在 thin_01 上做 8 邻接连通分量，只保留最大连通分量。
    输入: 0/1
    输出: 0/1
    """
    h, w = thin_01.shape
    n = h * w

    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)

    def idx(r, c):
        return r * w + c

    # 只看已扫描过的邻居，避免重复 union
    prev_neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

    for r in range(h):
        for c in range(w):
            if thin_01[r, c] == 0:
                continue
            cur = idx(r, c)
            for dr, dc in prev_neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and thin_01[nr, nc] == 1:
                    uf_union(parent, size, cur, idx(nr, nc))

    comp_count = {}
    for r in range(h):
        for c in range(w):
            if thin_01[r, c] == 0:
                continue
            root = uf_find(parent, idx(r, c))
            comp_count[root] = comp_count.get(root, 0) + 1

    largest_cc_mask = np.zeros_like(thin_01, dtype=np.uint8)
    if len(comp_count) == 0:
        return largest_cc_mask

    largest_root = max(comp_count, key=comp_count.get)

    for r in range(h):
        for c in range(w):
            if thin_01[r, c] == 0:
                continue
            root = uf_find(parent, idx(r, c))
            if root == largest_root:
                largest_cc_mask[r, c] = 1

    return largest_cc_mask


def bfs_distance_on_component(src, dst, component_mask):
    """
    在 component_mask(0/1) 上做 8 邻接 BFS。
    返回 src -> dst 的步长；若不可达则返回 None。
    """
    h, w = component_mask.shape
    sx, sy = src
    tx, ty = dst

    if not (0 <= sx < h and 0 <= sy < w and 0 <= tx < h and 0 <= ty < w):
        return None
    if component_mask[sx, sy] == 0 or component_mask[tx, ty] == 0:
        return None

    dist = np.full((h, w), -1, dtype=np.int32)
    q = deque()
    q.append((sx, sy))
    dist[sx, sy] = 0

    dirs8 = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    while q:
        x, y = q.popleft()
        if (x, y) == (tx, ty):
            return int(dist[x, y])

        for dx, dy in dirs8:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < h and 0 <= ny < w):
                continue
            if component_mask[nx, ny] == 0:
                continue
            if dist[nx, ny] != -1:
                continue
            dist[nx, ny] = dist[x, y] + 1
            q.append((nx, ny))

    return None


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

        # ========= 新增：三层全局记忆 =========
        # 第一层：可通行地图：1=可走, 0=不能走/未知
        self.passable_map = np.zeros((MAP_SIZE_INT, MAP_SIZE_INT), dtype=np.uint8)
        # 第二层：可见性地图：1=已知, 0=未知
        self.visibility_map = np.zeros((MAP_SIZE_INT, MAP_SIZE_INT), dtype=np.uint8)
        # 第三层：细化地图（局部更新）
        self.thin_map = np.zeros((MAP_SIZE_INT, MAP_SIZE_INT), dtype=np.uint8)

        # 最大连通分量（从 thin_map 提取）
        self.largest_cc_map = np.zeros((MAP_SIZE_INT, MAP_SIZE_INT), dtype=np.uint8)

        # 每个“已知且可走”的像素映射到 largest_cc_map 上的最近骨架点
        # 采用欧式最近匹配
        self.proj_x = np.full((MAP_SIZE_INT, MAP_SIZE_INT), -1, dtype=np.int16)
        self.proj_y = np.full((MAP_SIZE_INT, MAP_SIZE_INT), -1, dtype=np.int16)

    # ------------------------------------------------------------------
    # 地图记忆与局部细化
    # ------------------------------------------------------------------
    def _update_global_maps(self, hero_x, hero_y, map_info):
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

    def _update_thin_map_local(self, x0, x1, y0, y1):
        """
        只对局部区域更新细化图：
        1. 取一个带 margin 的局部窗口
        2. 先将 passable_map 与 thin_map 在该窗口内做 OR
        3. 只对该窗口做 thinning
        4. 将结果写回 thin_map
        """
        rx0, rx1, ry0, ry1 = _clip_window(
            x0 - THIN_MARGIN, x1 + THIN_MARGIN, y0 - THIN_MARGIN, y1 + THIN_MARGIN, MAP_SIZE_INT
        )

        region_passable = self.passable_map[rx0:rx1, ry0:ry1]
        region_thin_old = self.thin_map[rx0:rx1, ry0:ry1]

        # 按你的要求：局部视野和细化地图先做 OR
        region_seed = np.logical_or(region_passable > 0, region_thin_old > 0).astype(np.uint8)

        if region_seed.size == 0 or region_seed.sum() == 0:
            self.thin_map[rx0:rx1, ry0:ry1] = 0
            return rx0, rx1, ry0, ry1

        region_thin_new = zhang_suen_thinning(region_seed, max_iter=THIN_MAX_ITER)

        # 细化图不能跑出可通行区域
        region_thin_new = np.logical_and(region_thin_new > 0, region_passable > 0).astype(np.uint8)

        self.thin_map[rx0:rx1, ry0:ry1] = region_thin_new
        return rx0, rx1, ry0, ry1

    def _refresh_largest_component(self):
        """
        从全局 thin_map 提取最大连通分量。
        128x128 很小，这里直接全图重算，稳且简单。
        """
        self.largest_cc_map = build_largest_connected_component(self.thin_map)

        # 兜底：若 thin_map 当前为空，则退化为已知可走图的最大连通分量
        if self.largest_cc_map.sum() == 0 and self.passable_map.sum() > 0:
            self.largest_cc_map = build_largest_connected_component(self.passable_map)

    def _update_projection_local(self, x0, x1, y0, y1):
        """
        更新局部窗口中所有的点到最大连通分量的欧式最近投影。
        """
        if self.largest_cc_map.sum() == 0:
            self.proj_x[x0:x1, y0:y1] = -1
            self.proj_y[x0:x1, y0:y1] = -1
            return

        cc_points = np.argwhere(self.largest_cc_map == 1)  # [N,2], each row is (x,y)
        if len(cc_points) == 0:
            self.proj_x[x0:x1, y0:y1] = -1
            self.proj_y[x0:x1, y0:y1] = -1
            return

        for gx in range(x0, x1):
            for gy in range(y0, y1):
                #if self.visibility_map[gx, gy] == 0 or self.passable_map[gx, gy] == 0:
                #    self.proj_x[gx, gy] = -1
                #    self.proj_y[gx, gy] = -1
                #    continue

                diff = cc_points - np.array([gx, gy], dtype=np.int32)
                d2 = diff[:, 0].astype(np.float32) ** 2 + diff[:, 1].astype(np.float32) ** 2
                idx = int(np.argmin(d2))
                px, py = cc_points[idx]

                self.proj_x[gx, gy] = int(px)
                self.proj_y[gx, gy] = int(py)

    def _update_topology_memory(self, hero_x, hero_y, map_info):
        """
        每步更新：
        1. 拼接 passable_map / visibility_map
        2. 局部更新 thin_map
        3. 全图刷新最大连通分量
        4. 局部刷新投影映射
        """
        x0, x1, y0, y1 = self._update_global_maps(hero_x, hero_y, map_info)
        rx0, rx1, ry0, ry1 = self._update_thin_map_local(x0, x1, y0, y1)
        self._refresh_largest_component()
        self._update_projection_local(rx0, rx1, ry0, ry1)

    # ------------------------------------------------------------------
    # 对外方法：拓扑距离
    # ------------------------------------------------------------------
    def topo_distance(self, p1, p2):
        """
        根据 (x1, y1), (x2, y2) 计算拓扑距离。
        这里按你要求使用：
            d(p,q) = d_bfs(proj(p), proj(q))
        其中 proj 是到最大连通分量上的欧式最近投影。
        若点未知/不可走/无投影/不可达，则返回 None。
        """
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])

        assert self.visibility_map[x1, y1] == 1, "x1,y1 not visible!"
        assert self.visibility_map[x2, y2] == 1, "x2,y2 not visible!"
        assert self.passable_map[x1, y1] == 1, "x1,y1 not passable!"
        assert self.passable_map[x2, y2] == 1, "x2,y2 not passable!"

        sx, sy = int(self.proj_x[x1, y1]), int(self.proj_y[x1, y1])
        tx, ty = int(self.proj_x[x2, y2]), int(self.proj_y[x2, y2])

        return bfs_distance_on_component((sx, sy), (tx, ty), self.largest_cc_map)

    # ------------------------------------------------------------------
    # 主流程：特征处理
    # ------------------------------------------------------------------
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
        hero_x = int(hero_pos["x"])
        hero_y = int(hero_pos["z"])

        # ========= 新增：更新全局拓扑记忆 =========
        if map_info is not None:
            self._update_topology_memory(hero_x, hero_y, map_info)

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
                dir_idx = int(m.get("hero_relative_direction", 0)) % 8
                dir_x, dir_z = DIR8_TO_VEC[dir_idx]

                dist_norm = _norm(m.get("hero_l2_distance", MAX_DIST_BUCKET), MAX_DIST_BUCKET)
                topo_dist_norm = MAX_TOPO_DIST

                if is_in_view:
                    m_pos = m["pos"]
                    dx = float(m_pos["x"] - hero_pos["x"])
                    dz = float(m_pos["z"] - hero_pos["z"])

                    # 精细相对位置：保留正负号
                    rel_x = float(np.clip(dx / MAP_SIZE, -1.0, 1.0))
                    rel_z = float(np.clip(dz / MAP_SIZE, -1.0, 1.0))

                    raw_dist = np.sqrt(dx * dx + dz * dz)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)

                    # 拓扑距离计算
                    topo_dist = self.topo_distance((m_pos["x"], m_pos["z"]), (hero_pos["x"], hero_pos["z"]))
                    assert topo_dist is not None, "topo_dist is None in monster!"
                    topo_dist_norm = _norm(topo_dist, MAX_TOPO_DIST)

                    # 视野内时，用连续方向覆盖离散方向
                    if raw_dist > 1e-6:
                        dir_x = dx / raw_dist
                        dir_z = dz / raw_dist

                monster_feats.append(
                    np.array(
                        [is_in_view, m_speed_norm, rel_x, rel_z, dist_norm, topo_dist_norm, dir_x, dir_z],
                        dtype=np.float32,
                    )
                )
            else:
                monster_feats.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32))

        # buff和宝箱特征
        organs = frame_state.get("organs", [])
        self.logger.warning(f"len of organs:{len(organs)}, organs:{organs}")

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
            topo_dist = self.topo_distance((organ_pos["x"], organ_pos["z"]), (hero_pos["x"], hero_pos["z"]))
            assert topo_dist is not None, "topo_dist is None in organ!"

            # 存起来，方便排序和后续直接用
            organ["raw_dist"] = raw_dist
            organ["topo_dist"] = topo_dist
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
                dir_idx = int(organ.get("hero_relative_direction", 0)) % 8
                dir_x, dir_z = DIR8_TO_VEC[dir_idx]

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
                dir_idx = int(organ.get("hero_relative_direction", 0)) % 8
                dir_x, dir_z = DIR8_TO_VEC[dir_idx]

            buff_feat[i * 5 : i * 5 + 5] = np.array(
                [rel_x, rel_z, dist_norm, dir_x, dir_z],
                dtype=np.float32,
            )

        # 局部地图特征 (16D)
        map_feat = np.zeros((21, 21), dtype=np.float32)
        if map_info is not None:
            h = min(21, len(map_info))
            w = min(21, len(map_info[0]))
            for i in range(h):
                for j in range(w):
                    map_feat[i, j] = float(map_info[i][j] != 0)

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
        time_before_second_mounster = _norm(max(0, monster_interval - self.step_no), self.max_step)
        
        monster_speed = env_info.get("monster_speed", 0)
        self.logger.warning(f"monster speed value:{monster_speed}")
        has_monster_speedup = 0.0 if monster_speed <= 1 else 1.0
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
            "hero_pos": (hero_x, hero_y),
            "prev_hero_pos": self.prev_hero_pos,
            "last_action": int(last_action),
        }

        self.prev_hero_pos = (hero_x, hero_y)

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
                ( reward_feats['monster_feats'][0][5] - self.last_monster_dist_norm_1) + \
                0.2 * (reward_feats['monster_feats'][1][5] - self.last_monster_dist_norm_2)
            
        self.last_monster_dist_norm_1 = reward_feats['monster_feats'][0][5]
        self.last_monster_dist_norm_2 = reward_feats['monster_feats'][1][5]

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

        if reward_feats["progress_feats"][2] > 0 and reward_feats["progress_feats"][3] == 0: # time before second monseter and has monster speedup
            # 早期：鼓励探索和拿宝箱
            treasure_phase_weight = 1.50
            survive_phase_weight = 0.85
        else:
            # 后期：怪物加速后，生存优先
            treasure_phase_weight = 0.85
            survive_phase_weight = 1.50

        # final step reward vector
        dist_shaping_norm_weight = 12.8

        reward_vector = [
            1.0 * score_gain,
            survive_phase_weight * 0.01,
            0.50 * dist_shaping_norm_weight * monster_dist_reward,
            5.00 * treasure_phase_weight * treasure_reward,
            0.25 * treasure_phase_weight * dist_shaping_norm_weight * treasure_dist_reward,
            0.35 * buff_reward,
            0.05 * dist_shaping_norm_weight * buff_dist_reward,
            0.25 * flash_reward,
            1.00 * wall_penalty,
        ]

        return reward_vector, sum(reward_vector)
