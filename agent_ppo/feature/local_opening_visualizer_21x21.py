#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TARGET_SIZE = 128
LOCAL_SIZE = 21
LOCAL_HALF = LOCAL_SIZE // 2


def load_and_binarize_image(image_path: str, threshold: int = 128, invert: bool = False) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BILINEAR)
    gray = np.asarray(img, dtype=np.uint8)
    binary = (gray >= int(threshold)).astype(np.uint8)
    if invert:
        binary = 1 - binary
    return binary


def extract_local_patch(binary_map: np.ndarray, cx: int, cy: int, local_size: int = LOCAL_SIZE) -> np.ndarray:
    half = local_size // 2
    h, w = binary_map.shape
    patch = np.zeros((local_size, local_size), dtype=np.uint8)

    x0 = cx - half
    y0 = cy - half
    x1 = cx + half + 1
    y1 = cy + half + 1

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    patch[dst_y0:dst_y1, dst_x0:dst_x1] = binary_map[src_y0:src_y1, src_x0:src_x1]
    return patch


def get_boundary_passable_points(local_passable: np.ndarray):
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


def cluster_boundary_points(boundary_pts):
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


def compute_local_connected_mask(local_passable: np.ndarray, start_x: int = LOCAL_HALF, start_y: int = LOCAL_HALF):
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


def compute_opening_info(local_passable: np.ndarray):
    boundary_pts = get_boundary_passable_points(local_passable)
    clusters = cluster_boundary_points(boundary_pts)
    connected_mask = compute_local_connected_mask(local_passable)

    all_clusters = []
    connected_clusters_only = []

    for cluster in clusters:
        is_connected = any(connected_mask[y, x] > 0 for x, y in cluster)
        xs = np.asarray([p[0] for p in cluster], dtype=np.float32)
        ys = np.asarray([p[1] for p in cluster], dtype=np.float32)
        cx = float(xs.mean()) if len(xs) > 0 else 0.0
        cy = float(ys.mean()) if len(ys) > 0 else 0.0

        entry = {
            "points": cluster,
            "center": (cx, cy),
            "size": len(cluster),
            "connected": is_connected,
        }
        all_clusters.append(entry)
        if is_connected:
            connected_clusters_only.append(entry)

    return {
        "boundary_pts": boundary_pts,
        "clusters": all_clusters,
        "connected_mask": connected_mask,
        "connected_opening_count": len(connected_clusters_only),
        "is_dangerous": len(connected_clusters_only) <= 1,
    }


class LocalOpeningVisualizer:
    def __init__(self, binary_map: np.ndarray):
        self.binary_map = binary_map
        self.h, self.w = binary_map.shape
        self._last_hover = None

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.imshow(self.binary_map, origin="upper", cmap="gray", vmin=0.0, vmax=1.0)

        self.agent_scatter = self.ax.scatter([], [], s=100, marker="o")
        self.local_rect = plt.Rectangle((0, 0), LOCAL_SIZE, LOCAL_SIZE, fill=False, linewidth=1.5)
        self.ax.add_patch(self.local_rect)

        self.boundary_scatter = self.ax.scatter([], [], s=35, marker="s")
        self.cluster_center_scatter = self.ax.scatter([], [], s=90, marker="x")
        self.connected_circle_scatter = self.ax.scatter([], [], s=180, marker="o", facecolors="none", linewidths=1.5)
        self.cluster_texts = []

        self.info_text = self.ax.text(
            0.01, 0.99, "",
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round", alpha=0.85),
        )

        self.ax.set_title("21x21 local opening visualizer | Hover = agent position")
        self.ax.set_xlim(-0.5, self.w - 0.5)
        self.ax.set_ylim(self.h - 0.5, -0.5)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linewidth=0.25)

        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)

    def _clear_texts(self):
        for t in self.cluster_texts:
            try:
                t.remove()
            except Exception:
                pass
        self.cluster_texts = []

    def clear_overlay(self):
        self.agent_scatter.set_offsets(np.empty((0, 2)))
        self.boundary_scatter.set_offsets(np.empty((0, 2)))
        self.cluster_center_scatter.set_offsets(np.empty((0, 2)))
        self.connected_circle_scatter.set_offsets(np.empty((0, 2)))
        self.local_rect.set_xy((-100, -100))
        self.local_rect.set_width(LOCAL_SIZE)
        self.local_rect.set_height(LOCAL_SIZE)
        self.info_text.set_text("Move mouse onto a white/passable pixel")
        self._clear_texts()

    def on_move(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.clear_overlay()
            self._last_hover = None
            self.fig.canvas.draw_idle()
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if not (0 <= x < self.w and 0 <= y < self.h):
            self.clear_overlay()
            self._last_hover = None
            self.fig.canvas.draw_idle()
            return

        if self.binary_map[y, x] == 0:
            self.clear_overlay()
            self.info_text.set_text(f"({x}, {y}) is wall/blocked")
            self._last_hover = None
            self.fig.canvas.draw_idle()
            return

        self._last_hover = (x, y)
        self.update_at(x, y)

    def update_at(self, x: int, y: int):
        patch = extract_local_patch(self.binary_map, x, y, LOCAL_SIZE)
        info = compute_opening_info(patch)

        self.agent_scatter.set_offsets(np.array([[x, y]], dtype=np.float64))
        self.local_rect.set_xy((x - LOCAL_HALF - 0.5, y - LOCAL_HALF - 0.5))
        self._clear_texts()

        if info["boundary_pts"]:
            arr = np.array([[x + (px - LOCAL_HALF), y + (py - LOCAL_HALF)] for px, py in info["boundary_pts"]], dtype=np.float64)
            self.boundary_scatter.set_offsets(arr)
        else:
            self.boundary_scatter.set_offsets(np.empty((0, 2)))

        centers, connected_centers = [], []
        for idx, c in enumerate(info["clusters"]):
            cx, cy = c["center"]
            gx = x + (cx - LOCAL_HALF)
            gy = y + (cy - LOCAL_HALF)
            centers.append([gx, gy])
            if c["connected"]:
                connected_centers.append([gx, gy])

            label = "B"
            if c["connected"]:
                label = "A"
            self.cluster_texts.append(self.ax.text(gx + 0.25, gy + 0.25, label, fontsize=8))

        self.cluster_center_scatter.set_offsets(np.array(centers, dtype=np.float64) if centers else np.empty((0, 2)))
        self.connected_circle_scatter.set_offsets(np.array(connected_centers, dtype=np.float64) if connected_centers else np.empty((0, 2)))

        lines = [
            f"agent=({x}, {y})",
            f"local_view={LOCAL_SIZE}x{LOCAL_SIZE}",
            f"boundary_pts={len(info['boundary_pts'])}",
            f"clusters={len(info['clusters'])}",
            f"connected_clusters={sum(1 for c in info['clusters'] if c['connected'])}",
            f"connected_opening_count={info['connected_opening_count']}",
            f"dangerous={info['is_dangerous']}",
            "Legend: * means connected to the agent center",
        ]
        self.info_text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def show(self):
        self.clear_overlay()
        plt.tight_layout()
        plt.show()


def build_argparser():
    parser = argparse.ArgumentParser(description="Interactive local-opening visualizer (21x21 local view).")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--threshold", type=int, default=128, help="Binarization threshold, default=128")
    parser.add_argument("--invert", action="store_true", help="Invert black/white semantics after thresholding")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    binary_map = load_and_binarize_image(
        str(args.image),
        threshold=args.threshold,
        invert=args.invert,
    )

    vis = LocalOpeningVisualizer(binary_map)
    vis.show()


if __name__ == "__main__":
    main()
