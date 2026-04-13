
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TARGET_SIZE = 128
LOCAL_SIZE = 21
LOCAL_HALF = LOCAL_SIZE // 2


def load_and_binarize_image(image_path: str, threshold: int = 128, invert: bool = False) -> np.ndarray:
    """
    读取图像，中心裁剪最大正方形后 resize 到 128x128，再二值化。
    返回:
        binary_map: np.ndarray, shape [128, 128], uint8
        约定：1=可通行(亮点)，0=墙/障碍(黑点)
    """
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
    """
    以 (cx, cy) 为中心，提取固定 local_size x local_size 的局部 patch。
    超出边界部分按 0（墙）填充。
    """
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


def compute_min_distance_to_wall(local_patch: np.ndarray, agent_x: int = LOCAL_HALF, agent_y: int = LOCAL_HALF):
    """
    计算 agent 到局部 patch 内最近墙壁（黑点）的欧氏距离。
    约定：1=可通行，0=墙。
    返回:
        min_dist: float
        nearest_wall: (x, y) or None, 局部 patch 坐标
    """
    if local_patch[agent_y, agent_x] == 0:
        return 0.0, (agent_x, agent_y)

    wall_points = np.argwhere(local_patch == 0)  # each row is (y, x)
    if wall_points.size == 0:
        return float("inf"), None

    dx = wall_points[:, 1].astype(np.float64) - float(agent_x)
    dy = wall_points[:, 0].astype(np.float64) - float(agent_y)
    dists = np.hypot(dx, dy)

    idx = int(np.argmin(dists))
    nearest_y = int(wall_points[idx, 0])
    nearest_x = int(wall_points[idx, 1])
    min_dist = float(dists[idx])

    return min_dist, (nearest_x, nearest_y)


class MinWallDistanceVisualizer:
    def __init__(self, binary_map: np.ndarray):
        self.binary_map = binary_map
        self.h, self.w = binary_map.shape

        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.ax.imshow(self.binary_map, origin="upper", cmap="gray", vmin=0.0, vmax=1.0)

        self.agent_scatter = self.ax.scatter([], [], s=80, marker="o")
        self.local_rect = plt.Rectangle((0, 0), LOCAL_SIZE, LOCAL_SIZE, fill=False, linewidth=1.5)
        self.ax.add_patch(self.local_rect)

        self.wall_scatter = self.ax.scatter([], [], s=60, marker="x")
        self.link_line, = self.ax.plot([], [], linewidth=1.5)

        self.info_text = self.ax.text(
            0.01, 0.99, "",
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round", alpha=0.85),
        )

        self.ax.set_title("Hover on a white/passable pixel")
        self.ax.set_xlim(-0.5, self.w - 0.5)
        self.ax.set_ylim(self.h - 0.5, -0.5)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linewidth=0.25)

        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)

    def clear_overlay(self):
        self.agent_scatter.set_offsets(np.empty((0, 2)))
        self.wall_scatter.set_offsets(np.empty((0, 2)))
        self.link_line.set_data([], [])
        self.info_text.set_text("Move mouse onto a white/passable pixel")
        self.local_rect.set_xy((-100, -100))
        self.local_rect.set_width(LOCAL_SIZE)
        self.local_rect.set_height(LOCAL_SIZE)

    def on_move(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.clear_overlay()
            self.fig.canvas.draw_idle()
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if not (0 <= x < self.w and 0 <= y < self.h):
            self.clear_overlay()
            self.fig.canvas.draw_idle()
            return

        if self.binary_map[y, x] == 0:
            self.clear_overlay()
            self.info_text.set_text(f"({x}, {y}) is wall/blocked")
            self.fig.canvas.draw_idle()
            return

        patch = extract_local_patch(self.binary_map, x, y, LOCAL_SIZE)
        min_dist, nearest_wall_local = compute_min_distance_to_wall(patch, LOCAL_HALF, LOCAL_HALF)

        self.agent_scatter.set_offsets(np.array([[x, y]], dtype=np.float64))
        self.local_rect.set_xy((x - LOCAL_HALF - 0.5, y - LOCAL_HALF - 0.5))

        if nearest_wall_local is not None:
            wx_local, wy_local = nearest_wall_local
            wx = x + (wx_local - LOCAL_HALF)
            wy = y + (wy_local - LOCAL_HALF)
            self.wall_scatter.set_offsets(np.array([[wx, wy]], dtype=np.float64))
            self.link_line.set_data([x, wx], [y, wy])
        else:
            self.wall_scatter.set_offsets(np.empty((0, 2)))
            self.link_line.set_data([], [])

        shown_dist = "inf" if math.isinf(min_dist) else f"{min_dist:.3f}"
        self.info_text.set_text(
            f"agent=({x}, {y})\n"
            f"local={LOCAL_SIZE}x{LOCAL_SIZE}\n"
            f"min_dist_to_wall={shown_dist}"
        )
        self.fig.canvas.draw_idle()

    def show(self):
        self.clear_overlay()
        plt.tight_layout()
        plt.show()


def build_argparser():
    parser = argparse.ArgumentParser(description="Interactive minimum-distance-to-wall visualizer.")
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
        str(image_path),
        threshold=args.threshold,
        invert=args.invert,
    )

    vis = MinWallDistanceVisualizer(binary_map)
    vis.show()


if __name__ == "__main__":
    main()
