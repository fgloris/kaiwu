#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_PATH = r"D:\code\fwwb\kaiwu\map\2.png"
OUT_IMAGE_PATH = r"D:\code\fwwb\kaiwu\map\2-vis_boundery_cluster-11.png"

MAP_SIZE = 128
LOCAL_MAP_SIZE = 15
LOCAL_MAP_HALF = 10


def _get_boundary_passable_points(local_passable):
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


def _cluster_boundary_points(boundary_pts):
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


def _compute_local_connected_mask(local_passable, start_x=LOCAL_MAP_HALF, start_y=LOCAL_MAP_HALF):
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


def compute_connected_boundary_cluster_count(local_passable):
    boundary_pts = _get_boundary_passable_points(local_passable)
    clusters = _cluster_boundary_points(boundary_pts)
    connected_mask = _compute_local_connected_mask(local_passable)

    connected_count = 0
    for cluster in clusters:
        if any(connected_mask[y, x] > 0 for x, y in cluster):
            connected_count += 1
    return connected_count


def extract_local_patch(passable_map, center_x, center_y):
    patch = np.zeros((LOCAL_MAP_SIZE, LOCAL_MAP_SIZE), dtype=np.uint8)
    src_x0 = center_x - LOCAL_MAP_HALF
    src_y0 = center_y - LOCAL_MAP_HALF

    for y in range(LOCAL_MAP_SIZE):
        gy = src_y0 + y
        if not (0 <= gy < passable_map.shape[0]):
            continue
        for x in range(LOCAL_MAP_SIZE):
            gx = src_x0 + x
            if 0 <= gx < passable_map.shape[1]:
                patch[y, x] = passable_map[gy, gx]

    return patch


def load_binary_map(path):
    image = Image.open(path).convert("L").resize((MAP_SIZE, MAP_SIZE), Image.Resampling.NEAREST)
    arr = np.asarray(image, dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


def color_for_count(count):
    if count == 1:
        return (255, 0, 0)
    if count == 2:
        return (0, 0, 255)
    if count >= 3:
        return (0, 255, 0)
    return (0, 0, 0)


def main():
    passable_map = load_binary_map(IMAGE_PATH)
    output = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)

    for y in range(MAP_SIZE):
        for x in range(MAP_SIZE):
            if passable_map[y, x] == 0:
                continue

            local_patch = extract_local_patch(passable_map, x, y)
            count = compute_connected_boundary_cluster_count(local_patch)
            output[y, x] = color_for_count(count)

    Image.fromarray(output, mode="RGB").save(OUT_IMAGE_PATH)


if __name__ == "__main__":
    main()
