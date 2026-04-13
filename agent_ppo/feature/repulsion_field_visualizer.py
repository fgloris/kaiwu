#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

VIEW_SIZE = 128
LOCAL_SIZE = 21
LOCAL_RADIUS = LOCAL_SIZE // 2
DIR_COLOR = "tab:red"


def center_crop_square_then_resize(img: np.ndarray, out_size: int = VIEW_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = img[y0:y0 + side, x0:x0 + side]
    if side == out_size:
        return cropped.copy()
    return cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_AREA)


def load_binary_map(path: str, threshold: int = 128, invert: bool = False) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")

    img = center_crop_square_then_resize(img, VIEW_SIZE)

    if invert:
        binary = (img < threshold).astype(np.uint8)
    else:
        binary = (img >= threshold).astype(np.uint8)
    return binary


def get_edge_black_points(binary_map: np.ndarray, cx: int, cy: int, radius: int = LOCAL_RADIUS):
    h, w = binary_map.shape
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)

    pts = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            if binary_map[y, x] != 0:
                continue
            has_white_neighbor = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and binary_map[ny, nx] > 0:
                        has_white_neighbor = True
                        break
                if has_white_neighbor:
                    break
            if has_white_neighbor:
                pts.append((x, y))
    return pts


def compute_repulsion(binary_map: np.ndarray, cx: int, cy: int, radius: float = 10.0):
    edge_pts = get_edge_black_points(binary_map, cx, cy, radius=LOCAL_RADIUS)

    force = np.zeros(2, dtype=np.float64)
    used_pts = []
    for x, y in edge_pts:
        dx = cx - x
        dy = cy - y
        dist = float(np.hypot(dx, dy))
        if dist < 1e-6 or dist > radius:
            continue

        weight = 1.0 / (dist * dist + 1e-6)
        vec = np.array([dx / dist, dy / dist], dtype=np.float64)
        force += weight * vec
        used_pts.append((x, y, dist, weight))

    return force, used_pts


class RepulsionFieldViewer:
    def __init__(self, binary_map: np.ndarray, radius: float = 10.0):
        self.binary_map = binary_map
        self.radius = float(radius)
        self.h, self.w = binary_map.shape

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.imshow(self.binary_map, cmap="gray", origin="lower", vmin=0, vmax=1)
        self.ax.set_title("Hover on a white pixel")
        self.ax.set_xlim(-0.5, self.w - 0.5)
        self.ax.set_ylim(-0.5, self.h - 0.5)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linewidth=0.2, alpha=0.4)

        self.agent_plot, = self.ax.plot([], [], marker="o", markersize=6, color="tab:blue")
        self.edge_plot, = self.ax.plot([], [], linestyle="", marker="s", markersize=4, color="tab:orange", alpha=0.9)
        self.arrow = None
        self.local_rect = Rectangle((0, 0), LOCAL_SIZE, LOCAL_SIZE, fill=False, edgecolor="tab:green", linewidth=1.2)
        self.ax.add_patch(self.local_rect)
        self.info_text = self.ax.text(1, self.h - 3, "", fontsize=10, va="top", ha="left",
                                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)

    def on_move(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        cx = int(round(event.xdata))
        cy = int(round(event.ydata))
        if not (0 <= cx < self.w and 0 <= cy < self.h):
            return
        if self.binary_map[cy, cx] == 0:
            return

        force, used_pts = compute_repulsion(self.binary_map, cx, cy, radius=self.radius)
        mag = float(np.linalg.norm(force))

        self.agent_plot.set_data([cx], [cy])

        if used_pts:
            ex = [p[0] for p in used_pts]
            ey = [p[1] for p in used_pts]
            self.edge_plot.set_data(ex, ey)
        else:
            self.edge_plot.set_data([], [])

        self.local_rect.set_xy((cx - LOCAL_RADIUS - 0.5, cy - LOCAL_RADIUS - 0.5))

        if self.arrow is not None:
            self.arrow.remove()
            self.arrow = None

        if mag > 1e-6:
            scale = min(8.0, 2.0 + 2.5 * mag)
            fx, fy = force / mag
            self.arrow = self.ax.arrow(
                cx, cy,
                fx * scale, fy * scale,
                width=0.15,
                head_width=1.2,
                head_length=1.4,
                length_includes_head=True,
                color=DIR_COLOR,
                alpha=0.9,
            )
            force_text = f"force=({force[0]:.3f}, {force[1]:.3f}) |mag|={mag:.3f}"
        else:
            force_text = "force=(0.000, 0.000) |mag|=0.000"

        self.info_text.set_text(
            f"agent=({cx}, {cy})\n"
            f"used_edges={len(used_pts)}\n"
            f"{force_text}"
        )
        self.fig.canvas.draw_idle()

    def show(self):
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Repulsion field visualizer")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()

    binary_map = load_binary_map(args.image_path, threshold=args.threshold, invert=args.invert)
    viewer = RepulsionFieldViewer(binary_map, radius=args.radius)
    viewer.show()


if __name__ == "__main__":
    main()
