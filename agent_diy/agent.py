#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import heapq
from collections import deque

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData

DIR8 = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]

DIR9_TO_VEC = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (1.0, -1.0),
    3: (0.0, -1.0),
    4: (-1.0, -1.0),
    5: (-1.0, 0.0),
    6: (-1.0, 1.0),
    7: (0.0, 1.0),
    8: (1.0, 1.0),
}

BUCKET_CENTERS = [15.0, 45.0, 75.0, 105.0, 135.0, 165.0]


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.device = device
        self.logger = logger
        self.monitor = monitor
        self._reset_episode_state()
        super().__init__(agent_type, device, logger, monitor)

    def _reset_episode_state(self):
        s = Config.MAP_SIZE
        self.passable_map = np.zeros((s, s), dtype=np.uint8)
        self.visibility_map = np.zeros((s, s), dtype=np.uint8)
        self.visit_decay_map = np.zeros((s, s), dtype=np.float32)
        self.last_action = -1
        self.last_target = None
        self.cached_path = []
        self.step_no = 0
        self.last_plan_info = {
            "target_x": -1,
            "target_z": -1,
            "target_score": 0.0,
            "path_len": 0,
            "frontier_count": 0,
            "cluster_count": 0,
            "used_fallback": 0.0,
            "best_move_idx": -1,
        }

    def reset(self, env_obs=None):
        self._reset_episode_state()

    def observation_process(self, env_obs):
        env_obs = self._normalize_env_obs(env_obs)
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation.get("env_info", {})
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x, hero_z = int(hero_pos["x"]), int(hero_pos["z"])
        self.step_no = int(observation.get("step_no", env_info.get("step_no", 0)))

        map_info = observation.get("map_info")
        if map_info is not None:
            self._update_global_maps(hero_x, hero_z, map_info)
        self._update_visit_map(hero_x, hero_z)

        feature = {
            "hero": {
                "x": hero_x,
                "z": hero_z,
                "flash_count": int(env_info.get("flash_count", 0)),
                "step_no": self.step_no,
            },
            "monsters": self._estimate_monsters(frame_state, hero_x, hero_z),
        }
        vector_feature = self._build_vector_feature(feature)
        map_feature = np.zeros((1, 1, 1), dtype=np.float32)
        legal_action = self._parse_legal_action(observation)
        obs_data = ObsData(
            vector_feature=list(vector_feature),
            map_feature=list(map_feature.reshape(-1)),
            legal_action=legal_action,
        )
        remain_info = {
            "reward": [0.0],
            "reward_vector": [0.0] * 6,
            "passable_map": self.passable_map,
            "visibility_map": self.visibility_map,
            "visit_decay_map": self.visit_decay_map,
            "plan_info": dict(self.last_plan_info),
            "planner_feature": feature,
        }
        return obs_data, remain_info

    def predict(self, list_obs_data):
        rst = []
        for obs_data in list_obs_data:
            planner_feature = self._recover_planner_feature(obs_data.vector_feature)
            legal_action = getattr(obs_data, "legal_action", None)
            action = self._plan_action(planner_feature, legal_action)
            prob = self._build_prob(action, legal_action)
            rst.append(ActData(
                action=[int(action)],
                d_action=[int(action)],
                prob=list(prob),
                value=[0.0],
            ))
        return rst

    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        return None

    def save_model(self, path=None, id="1"):
        if self.logger is not None:
            self.logger.info("DIY planner agent has no trainable model; skip save_model")

    def load_model(self, path=None, id="1"):
        if self.logger is not None:
            self.logger.info("DIY planner agent has no trainable model; skip load_model")

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _normalize_env_obs(self, obs):
        if isinstance(obs, dict) and "observation" in obs:
            return obs
        return {
            "observation": obs,
            "extra_info": {},
            "terminated": False,
            "truncated": False,
        }

    def _build_vector_feature(self, feature):
        hero = feature["hero"]
        monsters = feature["monsters"]
        vals = [
            float(hero["x"]) / max(1, Config.MAP_SIZE - 1),
            float(hero["z"]) / max(1, Config.MAP_SIZE - 1),
            float(hero.get("flash_count", 0)) / 3.0,
            float(self.step_no) / 2000.0,
        ]
        for i in range(2):
            if i < len(monsters):
                mx, mz = monsters[i]
                vals.extend([
                    float(mx) / max(1, Config.MAP_SIZE - 1),
                    float(mz) / max(1, Config.MAP_SIZE - 1),
                    float(mx - hero["x"]) / Config.MAP_SIZE,
                    float(mz - hero["z"]) / Config.MAP_SIZE,
                ])
            else:
                vals.extend([0.0, 0.0, 0.0, 0.0])
        while len(vals) < Config.VECTOR_FEATURE_LEN:
            vals.append(0.0)
        return np.asarray(vals[:Config.VECTOR_FEATURE_LEN], dtype=np.float32)

    def _recover_planner_feature(self, vector_feature):
        vec = list(vector_feature)
        hero_x = int(round(vec[0] * max(1, Config.MAP_SIZE - 1))) if len(vec) > 0 else 0
        hero_z = int(round(vec[1] * max(1, Config.MAP_SIZE - 1))) if len(vec) > 1 else 0
        flash_count = int(round((vec[2] if len(vec) > 2 else 0.0) * 3.0))
        monsters = []
        idx = 4
        for _ in range(2):
            if idx + 3 < len(vec):
                mx = int(round(vec[idx] * max(1, Config.MAP_SIZE - 1)))
                mz = int(round(vec[idx + 1] * max(1, Config.MAP_SIZE - 1)))
                if mx != 0 or mz != 0:
                    monsters.append((mx, mz))
            idx += 4
        return {
            "hero": {"x": hero_x, "z": hero_z, "flash_count": flash_count, "step_no": self.step_no},
            "monsters": monsters,
        }

    def _build_prob(self, action, legal_action):
        prob = np.zeros(Config.ACTION_DIM, dtype=np.float32)
        if legal_action is None:
            prob[int(action)] = 1.0
            return prob
        legal = np.asarray(legal_action, dtype=np.float32)
        if legal.shape[0] != Config.ACTION_DIM:
            tmp = np.zeros(Config.ACTION_DIM, dtype=np.float32)
            n = min(Config.ACTION_DIM, legal.shape[0])
            tmp[:n] = legal[:n]
            legal = tmp
        if int(action) < Config.ACTION_DIM and legal[int(action)] > 0:
            prob[int(action)] = 1.0
        else:
            idxs = np.where(legal > 0)[0]
            prob[int(idxs[0]) if len(idxs) else 0] = 1.0
        return prob

    def _parse_legal_action(self, observation):
        raw = observation.get("legal_action") or observation.get("legal_act")
        legal = [1] * Config.ACTION_DIM
        if isinstance(raw, list) and raw:
            if isinstance(raw[0], bool):
                for i in range(min(Config.ACTION_DIM, len(raw))):
                    legal[i] = 1 if raw[i] else 0
            else:
                legal = [0] * Config.ACTION_DIM
                for a in raw:
                    a = int(a)
                    if 0 <= a < Config.ACTION_DIM:
                        legal[a] = 1
        return legal

    def _update_global_maps(self, hero_x, hero_z, map_info):
        h = min(Config.LOCAL_MAP_SIZE, len(map_info))
        w = min(Config.LOCAL_MAP_SIZE, len(map_info[0])) if h > 0 else 0
        x0 = hero_x - Config.LOCAL_HALF
        z0 = hero_z - Config.LOCAL_HALF
        for i in range(h):
            for j in range(w):
                gx = x0 + j
                gz = z0 + i
                if not (0 <= gx < Config.MAP_SIZE and 0 <= gz < Config.MAP_SIZE):
                    continue
                self.visibility_map[gx, gz] = 1
                self.passable_map[gx, gz] = 1 if int(map_info[i][j]) != 0 else 0

    def _update_visit_map(self, hero_x, hero_z):
        self.visit_decay_map *= Config.VISIT_DECAY
        self.visit_decay_map[self.visit_decay_map < 1e-3] = 0.0
        if 0 <= hero_x < Config.MAP_SIZE and 0 <= hero_z < Config.MAP_SIZE:
            self.visit_decay_map[hero_x, hero_z] = 1.0

    def _estimate_monsters(self, frame_state, hero_x, hero_z):
        monsters = []
        for m in frame_state.get("monsters", [])[:2]:
            if int(m.get("is_in_view", 0)) == 1 and "pos" in m:
                mx = int(m["pos"]["x"])
                mz = int(m["pos"]["z"])
            else:
                dist_bucket = int(m.get("hero_l2_distance", 5))
                dist = BUCKET_CENTERS[max(0, min(dist_bucket, len(BUCKET_CENTERS) - 1))]
                vx, vz = DIR9_TO_VEC.get(int(m.get("hero_relative_direction", 0)), (0.0, 0.0))
                norm = max((vx * vx + vz * vz) ** 0.5, 1e-6)
                mx = int(round(hero_x + dist * vx / norm))
                mz = int(round(hero_z + dist * vz / norm))
            mx = max(0, min(Config.MAP_SIZE - 1, mx))
            mz = max(0, min(Config.MAP_SIZE - 1, mz))
            monsters.append((mx, mz))
        return monsters

    def _plan_action(self, feature, legal_act):
        hero_x = int(feature["hero"]["x"])
        hero_z = int(feature["hero"]["z"])
        monsters = feature["monsters"]

        reachable_mask = self._compute_reachable_mask(hero_x, hero_z)
        clusters = self._get_frontier_clusters(reachable_mask)
        target, target_score = self._select_target((hero_x, hero_z), monsters, clusters)

        path = []
        path_len = 0
        if target is not None:
            path = self._astar((hero_x, hero_z), target)
            if path:
                self.last_target = target
                self.cached_path = path
                path_len = max(0, len(path) - 1)

        if len(path) >= 2:
            next_cell = path[1]
            move_idx = self._delta_to_move(next_cell[0] - hero_x, next_cell[1] - hero_z)
            self.last_plan_info = {
                "target_x": int(target[0]) if target is not None else -1,
                "target_z": int(target[1]) if target is not None else -1,
                "target_score": float(target_score),
                "path_len": int(path_len),
                "frontier_count": int(sum(len(c) for c in clusters)),
                "cluster_count": int(len(clusters)),
                "used_fallback": 0.0,
                "best_move_idx": int(move_idx) if move_idx is not None else -1,
            }
            if move_idx is not None and move_idx < 8 and legal_act[move_idx]:
                return move_idx

        move_idx = self._fallback_direction((hero_x, hero_z), monsters, legal_act, reachable_mask)
        self.last_plan_info = {
            "target_x": int(target[0]) if target is not None else -1,
            "target_z": int(target[1]) if target is not None else -1,
            "target_score": float(target_score),
            "path_len": int(path_len),
            "frontier_count": int(sum(len(c) for c in clusters)),
            "cluster_count": int(len(clusters)),
            "used_fallback": 1.0,
            "best_move_idx": int(move_idx),
        }
        return move_idx

    def _compute_reachable_mask(self, hero_x, hero_z):
        reachable = np.zeros_like(self.passable_map, dtype=np.uint8)
        if not self._is_known_passable(hero_x, hero_z):
            return reachable
        q = deque([(hero_x, hero_z)])
        reachable[hero_x, hero_z] = 1
        while q:
            x, z = q.popleft()
            for dx, dz in DIR8:
                nx, nz = x + dx, z + dz
                if not self._is_known_passable(nx, nz):
                    continue
                if reachable[nx, nz]:
                    continue
                reachable[nx, nz] = 1
                q.append((nx, nz))
        return reachable

    def _get_frontier_clusters(self, reachable_mask):
        frontier = []
        for x in range(Config.MAP_SIZE):
            for z in range(Config.MAP_SIZE):
                if not reachable_mask[x, z]:
                    continue
                if self._is_frontier(x, z):
                    frontier.append((x, z))

        frontier_set = set(frontier)
        visited = set()
        clusters = []
        for p in frontier:
            if p in visited:
                continue
            q = deque([p])
            visited.add(p)
            cluster = []
            while q:
                cur = q.popleft()
                cluster.append(cur)
                cx, cz = cur
                for dx, dz in DIR8:
                    npnt = (cx + dx, cz + dz)
                    if npnt in frontier_set and npnt not in visited:
                        visited.add(npnt)
                        q.append(npnt)
            clusters.append(cluster)
        return clusters

    def _is_frontier(self, x, z):
        if not self._is_known_passable(x, z):
            return False
        for dx, dz in DIR8:
            nx, nz = x + dx, z + dz
            if 0 <= nx < Config.MAP_SIZE and 0 <= nz < Config.MAP_SIZE and self.visibility_map[nx, nz] == 0:
                return True
        return False

    def _select_target(self, hero, monsters, clusters):
        if not clusters:
            return None, 0.0
        best_p, best_score = None, -1e18
        for cluster in clusters:
            rx, rz = self._representative_point(cluster)
            score = self._score_target(hero, (rx, rz), monsters, cluster)
            if score > best_score:
                best_score = score
                best_p = (rx, rz)
        return best_p, best_score

    def _representative_point(self, cluster):
        xs = [p[0] for p in cluster]
        zs = [p[1] for p in cluster]
        cx = int(round(sum(xs) / len(xs)))
        cz = int(round(sum(zs) / len(zs)))
        best = cluster[0]
        best_d = 1e18
        for p in cluster:
            d = (p[0] - cx) ** 2 + (p[1] - cz) ** 2
            if d < best_d:
                best_d = d
                best = p
        return best

    def _score_target(self, hero, target, monsters, cluster):
        hx, hz = hero
        tx, tz = target
        dist_path = abs(tx - hx) + abs(tz - hz)
        openness = self._local_open_score(tx, tz)
        revisit_pen = float(self.visit_decay_map[tx, tz])
        monster_dist = min([((tx - mx) ** 2 + (tz - mz) ** 2) ** 0.5 for mx, mz in monsters], default=50.0)
        away_score = 0.0
        if monsters:
            avg_mx = sum(m[0] for m in monsters) / len(monsters)
            avg_mz = sum(m[1] for m in monsters) / len(monsters)
            away_vec = np.array([tx - hx, tz - hz], dtype=np.float32)
            danger_vec = np.array([avg_mx - hx, avg_mz - hz], dtype=np.float32)
            na = np.linalg.norm(away_vec)
            nd = np.linalg.norm(danger_vec)
            if na > 1e-6 and nd > 1e-6:
                away_score = float(np.dot(away_vec / na, -danger_vec / nd))
        cluster_bonus = len(cluster) * 0.15
        return 1.2 * openness + 0.08 * monster_dist + 1.5 * away_score + cluster_bonus - 0.04 * dist_path - 1.0 * revisit_pen

    def _local_open_score(self, x, z):
        score = 0.0
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                nx, nz = x + dx, z + dz
                if self._is_known_passable(nx, nz):
                    score += 1.0
        return score

    def _astar(self, start, goal):
        if start == goal:
            return [start]
        sx, sz = start
        gx, gz = goal
        if not self._is_known_passable(gx, gz):
            return []
        open_heap = []
        heapq.heappush(open_heap, (0.0, start))
        parent = {start: None}
        g_cost = {start: 0.0}
        expanded = 0
        while open_heap and expanded < Config.ASTAR_MAX_EXPAND:
            _, cur = heapq.heappop(open_heap)
            expanded += 1
            if cur == goal:
                break
            cx, cz = cur
            for dx, dz in DIR8:
                nx, nz = cx + dx, cz + dz
                nxt = (nx, nz)
                if not self._is_known_passable(nx, nz):
                    continue
                step_cost = 1.414 if dx != 0 and dz != 0 else 1.0
                new_cost = g_cost[cur] + step_cost + 0.2 * float(self.visit_decay_map[nx, nz])
                if nxt not in g_cost or new_cost < g_cost[nxt]:
                    g_cost[nxt] = new_cost
                    parent[nxt] = cur
                    h = abs(nx - gx) + abs(nz - gz)
                    heapq.heappush(open_heap, (new_cost + h, nxt))
        if goal not in parent:
            return []
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def _fallback_direction(self, hero, monsters, legal_act, reachable_mask):
        hx, hz = hero
        best_idx = 0
        best_score = -1e18
        for i, (dx, dz) in enumerate(DIR8):
            if not legal_act[i]:
                continue
            nx, nz = hx + dx, hz + dz
            if not self._is_known_passable(nx, nz):
                continue
            score = 1.2 * self._ray_open_length(hx, hz, dx, dz)
            score += 0.8 * self._local_open_score(nx, nz)
            score -= 2.0 * float(self.visit_decay_map[nx, nz])
            if monsters:
                min_dist = min([((nx - mx) ** 2 + (nz - mz) ** 2) ** 0.5 for mx, mz in monsters])
                score += 0.15 * min_dist
            if reachable_mask[nx, nz]:
                score += 0.5
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _ray_open_length(self, x, z, dx, dz):
        length = 0.0
        cx, cz = x, z
        for _ in range(Config.RAY_MAX_LEN):
            cx += dx
            cz += dz
            if not self._is_known_passable(cx, cz):
                break
            length += 1.0
        return length

    def _delta_to_move(self, dx, dz):
        for i, (mx, mz) in enumerate(DIR8):
            if dx == mx and dz == mz:
                return i
        return None

    def _is_known_passable(self, x, z):
        if not (0 <= x < Config.MAP_SIZE and 0 <= z < Config.MAP_SIZE):
            return False
        return self.visibility_map[x, z] == 1 and self.passable_map[x, z] == 1
