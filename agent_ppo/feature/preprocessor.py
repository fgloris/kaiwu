#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from agent_ppo_strong.conf.conf import Config


TARGET_KEYWORDS = (
    "treasure", "treasures", "box", "boxes", "chest", "chests",
    "buff", "buffs", "goal", "goals", "target", "targets",
)


def _norm(v: float, v_max: float, v_min: float = 0.0) -> float:
    v = float(np.clip(v, v_min, v_max))
    denom = max(v_max - v_min, 1e-6)
    return (v - v_min) / denom


def _safe_get(dct: Dict[str, Any], *keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _xy(pos: Dict[str, Any]) -> tuple[float, float]:
    if not isinstance(pos, dict):
        return 0.0, 0.0
    x = pos.get("x", pos.get("X", 0.0))
    z = pos.get("z", pos.get("y", pos.get("Z", 0.0)))
    return float(x), float(z)


def _is_entity_list(value: Any) -> bool:
    return isinstance(value, list) and (len(value) == 0 or isinstance(value[0], dict))


def _iter_entity_lists(obj: Any, depth: int = 0) -> Iterable[tuple[str, list]]:
    if depth > 3:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if _is_entity_list(v):
                yield lk, v
            elif isinstance(v, (dict, list)):
                yield from _iter_entity_lists(v, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                yield from _iter_entity_lists(item, depth + 1)


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 1.0
        self.last_total_score = 0.0
        self.last_target_dist_norm = 1.0
        self.non_zero_score_steps = 0
        self.total_score_gain = 0.0

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})
        map_info = observation.get("map_info")
        legal_act_raw = observation.get("legal_action", [])

        self.step_no = int(observation.get("step_no", 0))
        self.max_step = int(env_info.get("max_step", self.max_step))

        hero = frame_state.get("heroes", {})
        hero_pos = hero.get("pos", {})
        hero_x, hero_z = _xy(hero_pos)
        hero_feat = self._build_hero_feature(hero, hero_x, hero_z)

        monster_feats, cur_min_monster_dist_norm = self._build_monster_features(frame_state, hero_x, hero_z)
        target_feats, nearest_target_dist_norm = self._build_target_features(observation, frame_state, env_info, hero_x, hero_z)
        map_feat = self._build_map_feature(map_info)
        legal_action = self._build_legal_action(legal_act_raw)
        progress_feat = self._build_progress_feature(env_info)
        last_action_feat = self._build_last_action_feature(last_action)

        feature = np.concatenate(
            [
                hero_feat,
                monster_feats,
                target_feats,
                map_feat,
                np.asarray(legal_action, dtype=np.float32),
                progress_feat,
                last_action_feat,
            ]
        ).astype(np.float32)

        reward = [
            self._compute_reward(
                env_info=env_info,
                cur_min_monster_dist_norm=cur_min_monster_dist_norm,
                nearest_target_dist_norm=nearest_target_dist_norm,
            )
        ]
        return feature, legal_action, reward

    def _build_hero_feature(self, hero, hero_x, hero_z):
        flash_cd = float(hero.get("flash_cooldown", 0.0))
        buff_remain = float(hero.get("buff_remaining_time", 0.0))
        move_speed = float(hero.get("speed", hero.get("move_speed", 0.0)))
        hero_hp = float(hero.get("hp", hero.get("health", 0.0)))
        norm_hp = _norm(hero_hp, 100.0) if hero_hp > 0 else 0.0
        return np.array(
            [
                _norm(hero_x, Config.MAP_SIZE),
                _norm(hero_z, Config.MAP_SIZE),
                _norm(flash_cd, Config.MAX_FLASH_CD),
                _norm(buff_remain, Config.MAX_BUFF_DURATION),
                _norm(move_speed, 10.0),
                norm_hp,
                math.sin(hero_x / Config.MAP_SIZE * math.pi),
                math.cos(hero_x / Config.MAP_SIZE * math.pi),
                math.sin(hero_z / Config.MAP_SIZE * math.pi),
                math.cos(hero_z / Config.MAP_SIZE * math.pi),
            ],
            dtype=np.float32,
        )

    def _build_monster_features(self, frame_state, hero_x, hero_z):
        monsters = frame_state.get("monsters", []) or []
        feats: List[np.ndarray] = []
        cur_min_dist_norm = 1.0
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 1))
                mx, mz = _xy(m.get("pos", {}))
                speed = float(m.get("speed", 0.0))
                dx = mx - hero_x
                dz = mz - hero_z
                dist = math.sqrt(dx * dx + dz * dz)
                dist_norm = _norm(dist, Config.MAP_SIZE * 1.42)
                cur_min_dist_norm = min(cur_min_dist_norm, dist_norm)
                feats.append(
                    np.array(
                        [
                            is_in_view,
                            _norm(mx, Config.MAP_SIZE),
                            _norm(mz, Config.MAP_SIZE),
                            np.clip(dx / Config.MAP_SIZE, -1.0, 1.0),
                            np.clip(dz / Config.MAP_SIZE, -1.0, 1.0),
                            _norm(speed, Config.MAX_MONSTER_SPEED),
                            dist_norm,
                            1.0 - dist_norm,
                        ],
                        dtype=np.float32,
                    )
                )
            else:
                feats.append(np.zeros(8, dtype=np.float32))
        return np.concatenate(feats, dtype=np.float32), cur_min_dist_norm

    def _build_target_features(self, observation, frame_state, env_info, hero_x, hero_z):
        candidates: List[tuple[float, np.ndarray]] = []
        visited_ids = set()
        for key, entity_list in _iter_entity_lists({"observation": observation, "frame_state": frame_state, "env_info": env_info}):
            if not any(token in key for token in TARGET_KEYWORDS):
                continue
            for ent in entity_list:
                if not isinstance(ent, dict):
                    continue
                ent_id = ent.get("id", ent.get("ID", id(ent)))
                if ent_id in visited_ids:
                    continue
                visited_ids.add(ent_id)
                pos = ent.get("pos", ent.get("position", {}))
                tx, tz = _xy(pos)
                if tx == 0.0 and tz == 0.0 and not pos:
                    continue
                dx = tx - hero_x
                dz = tz - hero_z
                dist = math.sqrt(dx * dx + dz * dz)
                dist_norm = _norm(dist, Config.MAP_SIZE * 1.42)
                active = float(ent.get("status", ent.get("active", ent.get("is_alive", 1))))
                visible = float(ent.get("is_in_view", ent.get("visible", 1)))
                is_buff = 1.0 if "buff" in key else 0.0
                feat = np.array(
                    [
                        visible,
                        np.clip(dx / Config.MAP_SIZE, -1.0, 1.0),
                        np.clip(dz / Config.MAP_SIZE, -1.0, 1.0),
                        dist_norm,
                        np.clip(active, 0.0, 1.0),
                        is_buff,
                    ],
                    dtype=np.float32,
                )
                candidates.append((dist_norm, feat))
        candidates.sort(key=lambda x: x[0])
        nearest = 1.0
        out = []
        for i in range(4):
            if i < len(candidates):
                nearest = min(nearest, candidates[i][0])
                out.append(candidates[i][1])
            else:
                out.append(np.zeros(6, dtype=np.float32))
        return np.concatenate(out, dtype=np.float32), nearest

    def _build_map_feature(self, map_info):
        feat = np.zeros((5, 5), dtype=np.float32)
        if isinstance(map_info, list) and len(map_info) > 0 and isinstance(map_info[0], list):
            h = len(map_info)
            w = len(map_info[0])
            cy = h // 2
            cx = w // 2
            idx_y = 0
            for y in range(cy - 2, cy + 3):
                idx_x = 0
                for x in range(cx - 2, cx + 3):
                    if 0 <= y < h and 0 <= x < w:
                        val = float(map_info[y][x])
                        feat[idx_y, idx_x] = np.tanh(val / 5.0)
                    idx_x += 1
                idx_y += 1
        return feat.reshape(-1)

    def _build_legal_action(self, legal_act_raw):
        legal_action = [1] * Config.ACTION_NUM
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(Config.ACTION_NUM, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if 0 <= int(a) < Config.ACTION_NUM}
                legal_action = [1 if j in valid_set else 0 for j in range(Config.ACTION_NUM)]
        if sum(legal_action) == 0:
            legal_action = [1] * Config.ACTION_NUM
        return legal_action

    def _build_progress_feature(self, env_info):
        total_score = float(env_info.get("total_score", 0.0))
        score_gain = total_score - self.last_total_score
        return np.array(
            [
                _norm(self.step_no, max(self.max_step, 1)),
                _norm(total_score, Config.MAX_SCORE),
                np.clip(score_gain / 5.0, -1.0, 1.0),
                float(self.non_zero_score_steps > 0),
            ],
            dtype=np.float32,
        )

    def _build_last_action_feature(self, last_action):
        feat = np.zeros(Config.LAST_ACTION_DIM, dtype=np.float32)
        if isinstance(last_action, int) and 0 <= last_action < Config.LAST_ACTION_DIM:
            feat[last_action] = 1.0
        return feat

    def _compute_reward(self, env_info, cur_min_monster_dist_norm, nearest_target_dist_norm):
        total_score = float(env_info.get("total_score", 0.0))
        score_gain = total_score - self.last_total_score
        self.last_total_score = total_score
        self.total_score_gain += score_gain
        if abs(score_gain) > 1e-8:
            self.non_zero_score_steps += 1

        score_reward = np.clip(score_gain, -Config.SCORE_GAIN_CLIP, Config.SCORE_GAIN_CLIP) * Config.SCORE_GAIN_SCALE

        monster_shaping = (cur_min_monster_dist_norm - self.last_min_monster_dist_norm) * Config.MONSTER_SHAPING_SCALE
        self.last_min_monster_dist_norm = cur_min_monster_dist_norm

        target_progress = 0.0
        if nearest_target_dist_norm < 0.999:
            target_progress = (self.last_target_dist_norm - nearest_target_dist_norm) * Config.TARGET_PROGRESS_SCALE
            self.last_target_dist_norm = nearest_target_dist_norm
        else:
            self.last_target_dist_norm = 1.0

        stagnation_penalty = -Config.STAGNATION_PENALTY if abs(score_gain) < 1e-8 else 0.0
        reward = (
            score_reward
            + Config.SURVIVE_REWARD
            + monster_shaping
            + target_progress
            + stagnation_penalty
        )
        return float(np.clip(reward, -5.0, 5.0))
