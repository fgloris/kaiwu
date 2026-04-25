#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


IMAGE_PATH = r"D:\code\fwwb\kaiwu\map\1.png"

MAP_SIZE = 128
LOCAL_MAP_SIZE = 21
LOCAL_MAP_HALF = 10
CLUSTER_COLORS = np.asarray(
    [
        (255, 230, 0),
        (0, 220, 255),
        (255, 80, 220),
        (255, 140, 0),
        (80, 255, 80),
        (180, 120, 255),
    ],
    dtype=np.float32,
) / 255.0


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


def compute_connected_boundary_clusters(local_passable):
    boundary_pts = _get_boundary_passable_points(local_passable)
    clusters = _cluster_boundary_points(boundary_pts)
    connected_mask = _compute_local_connected_mask(local_passable)

    connected_clusters = []
    for cluster in clusters:
        if any(connected_mask[y, x] > 0 for x, y in cluster):
            connected_clusters.append(cluster)
    return connected_clusters


def load_binary_map(path):
    image = Image.open(path).convert("L")
    try:
        resample = Image.Resampling.NEAREST
    except AttributeError:
        resample = Image.NEAREST

    if image.size != (MAP_SIZE, MAP_SIZE):
        image = image.resize((MAP_SIZE, MAP_SIZE), resample=resample)

    arr = np.asarray(image, dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


def clusters_to_global_points(clusters, center_x, center_y):
    offsets_x0 = center_x - LOCAL_MAP_HALF
    offsets_y0 = center_y - LOCAL_MAP_HALF
    points = []
    colors = []

    for cluster_idx, cluster in enumerate(clusters):
        color = CLUSTER_COLORS[cluster_idx % len(CLUSTER_COLORS)]
        for local_x, local_y in cluster:
            global_x = offsets_x0 + local_x
            global_y = offsets_y0 + local_y
            if 0 <= global_x < MAP_SIZE and 0 <= global_y < MAP_SIZE:
                points.append((global_x, global_y))
                colors.append(color)

    if not points:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.float32)


def main():
    passable_map = load_binary_map(IMAGE_PATH)
    display_map = np.where(passable_map > 0, 255, 25).astype(np.uint8)
    cache = {}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(display_map, cmap="gray", vmin=0, vmax=255, origin="upper")
    ax.set_xlim(-0.5, MAP_SIZE - 0.5)
    ax.set_ylim(MAP_SIZE - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title("Move mouse over the map")

    view_rect = Rectangle(
        (-0.5, -0.5),
        LOCAL_MAP_SIZE,
        LOCAL_MAP_SIZE,
        fill=False,
        edgecolor="white",
        linewidth=1.5,
        visible=False,
    )
    ax.add_patch(view_rect)

    center_point = ax.scatter([], [], s=28, c="red", marker="x", linewidths=1.5)
    cluster_points = ax.scatter([], [], s=24, c=[], marker="s", edgecolors="black", linewidths=0.25)

    def set_empty_overlay():
        view_rect.set_visible(False)
        center_point.set_offsets(np.empty((0, 2), dtype=np.float32))
        cluster_points.set_offsets(np.empty((0, 2), dtype=np.float32))
        cluster_points.set_facecolors(np.empty((0, 3), dtype=np.float32))

    def get_clusters_for_cell(x, y):
        key = (x, y)
        if key not in cache:
            local_patch = extract_local_patch(passable_map, x, y)
            cache[key] = compute_connected_boundary_clusters(local_patch)
        return cache[key]

    def on_mouse_move(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            set_empty_overlay()
            fig.canvas.draw_idle()
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
            set_empty_overlay()
            fig.canvas.draw_idle()
            return

        view_rect.set_xy((x - LOCAL_MAP_HALF - 0.5, y - LOCAL_MAP_HALF - 0.5))
        view_rect.set_visible(True)
        center_point.set_offsets(np.asarray([[x, y]], dtype=np.float32))

        if passable_map[y, x] == 0:
            cluster_points.set_offsets(np.empty((0, 2), dtype=np.float32))
            cluster_points.set_facecolors(np.empty((0, 3), dtype=np.float32))
            ax.set_title(f"x={x}, y={y} | wall")
            fig.canvas.draw_idle()
            return

        clusters = get_clusters_for_cell(x, y)
        points, colors = clusters_to_global_points(clusters, x, y)
        cluster_points.set_offsets(points)
        cluster_points.set_facecolors(colors)
        ax.set_title(f"x={x}, y={y} | reachable boundary clusters: {len(clusters)}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    plt.show()


if __name__ == "__main__":
    main()
