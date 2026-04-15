#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import heapq
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

TARGET_SIZE = 128
DIR8 = [
    (1, 0), (1, -1), (0, -1), (-1, -1),
    (-1, 0), (-1, 1), (0, 1), (1, 1),
]


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


def astar_path(passable: np.ndarray, visible: np.ndarray, start, goal):
    if start is None or goal is None:
        return None

    h, w = passable.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None

    def traversable(x, y):
        return visible[y, x] > 0 and passable[y, x] > 0

    if not traversable(sx, sy) or not traversable(gx, gy):
        return None

    def heuristic(x, y):
        return float(np.hypot(x - gx, y - gy))

    open_heap = []
    heapq.heappush(open_heap, (heuristic(sx, sy), 0.0, sx, sy))
    came_from = {}
    g_score = {(sx, sy): 0.0}
    closed = set()

    while open_heap:
        _, cur_g, x, y = heapq.heappop(open_heap)
        if (x, y) in closed:
            continue
        closed.add((x, y))

        if (x, y) == (gx, gy):
            path = [(x, y)]
            while (x, y) in came_from:
                x, y = came_from[(x, y)]
                path.append((x, y))
            path.reverse()
            return path

        for dx, dy in DIR8:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if not traversable(nx, ny):
                continue

            # 斜走时避免穿墙角
            if dx != 0 and dy != 0:
                if not traversable(x + dx, y) or not traversable(x, y + dy):
                    continue
                step_cost = np.sqrt(2.0)
            else:
                step_cost = 1.0

            tentative_g = cur_g + step_cost
            if tentative_g >= g_score.get((nx, ny), float("inf")):
                continue

            g_score[(nx, ny)] = tentative_g
            came_from[(nx, ny)] = (x, y)
            heapq.heappush(open_heap, (tentative_g + heuristic(nx, ny), tentative_g, nx, ny))

    return None


class AStarVisibilityVisualizer:
    def __init__(self, passable_map: np.ndarray, brush_radius: int = 2):
        self.passable = passable_map.astype(np.uint8)
        self.visible = np.zeros_like(self.passable, dtype=np.uint8)
        self.agent = None
        self.monster = None
        self.path = None
        self.brush_radius = int(max(0, brush_radius))
        self.is_dragging = False

        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.base_img = None
        self.visible_img = None
        self.path_line = None
        self.agent_scatter = None
        self.monster_scatter = None
        self.info_text = None

        self._init_plot()
        self._connect_events()
        self._refresh()

    def _init_plot(self):
        self.base_img = self.ax.imshow(
            self.passable,
            origin="upper",
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        visible_overlay = np.zeros_like(self.passable, dtype=np.float32)
        cmap = ListedColormap([
            (0.0, 0.0, 0.0, 0.0),
            (0.1, 0.7, 1.0, 0.35),
        ])
        self.visible_img = self.ax.imshow(
            visible_overlay,
            origin="upper",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        self.path_line, = self.ax.plot([], [], linewidth=2.0)
        self.agent_scatter = self.ax.scatter([], [], s=90, marker="o", label="agent")
        self.monster_scatter = self.ax.scatter([], [], s=100, marker="X", label="monster")
        self.info_text = self.ax.text(
            0.01,
            0.99,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", alpha=0.85),
        )

        self.ax.set_title("Left click: set agent | Right click: set monster | Left drag: paint visibility | key c: clear visibility | key r: reset all")
        self.ax.set_xlim(-0.5, self.passable.shape[1] - 0.5)
        self.ax.set_ylim(self.passable.shape[0] - 0.5, -0.5)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linewidth=0.2)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _event_xy(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        h, w = self.passable.shape
        if not (0 <= x < w and 0 <= y < h):
            return None
        return x, y

    def _paint_visibility(self, x, y):
        h, w = self.visible.shape
        r = self.brush_radius
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        self.visible[y0:y1, x0:x1] = 1

    def _recompute_path(self):
        self.path = astar_path(self.passable, self.visible, self.monster, self.agent)

    def _refresh(self):
        self.visible_img.set_data(self.visible.astype(np.float32))

        if self.path is None or len(self.path) == 0:
            self.path_line.set_data([], [])
        else:
            xs = [p[0] for p in self.path]
            ys = [p[1] for p in self.path]
            self.path_line.set_data(xs, ys)

        if self.agent is None:
            self.agent_scatter.set_offsets(np.empty((0, 2)))
        else:
            self.agent_scatter.set_offsets(np.array([self.agent], dtype=np.float64))

        if self.monster is None:
            self.monster_scatter.set_offsets(np.empty((0, 2)))
        else:
            self.monster_scatter.set_offsets(np.array([self.monster], dtype=np.float64))

        visible_count = int(self.visible.sum())
        passable_visible_count = int(((self.visible > 0) & (self.passable > 0)).sum())
        lines = [
            f"agent={self.agent}",
            f"monster={self.monster}",
            f"visible_pixels={visible_count}",
            f"visible_passable_pixels={passable_visible_count}",
            f"path_len={(len(self.path) if self.path is not None else 0)}",
            f"brush_radius={self.brush_radius}",
        ]
        self.info_text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        xy = self._event_xy(event)
        if xy is None:
            return
        x, y = xy

        if event.button == 1:
            self.agent = (x, y)
            self.is_dragging = True
            self._paint_visibility(x, y)
            self._recompute_path()
            self._refresh()
        elif event.button == 3:
            self.monster = (x, y)
            self._recompute_path()
            self._refresh()

    def on_release(self, event):
        if event.button == 1:
            self.is_dragging = False

    def on_motion(self, event):
        if not self.is_dragging:
            return
        xy = self._event_xy(event)
        if xy is None:
            return
        x, y = xy
        self._paint_visibility(x, y)
        self._recompute_path()
        self._refresh()

    def on_key(self, event):
        if event.key == "c":
            self.visible[:] = 0
            self._recompute_path()
            self._refresh()
        elif event.key == "r":
            self.visible[:] = 0
            self.agent = None
            self.monster = None
            self.path = None
            self._refresh()
        elif event.key == "[":
            self.brush_radius = max(0, self.brush_radius - 1)
            self._refresh()
        elif event.key == "]":
            self.brush_radius += 1
            self._refresh()

    def show(self):
        plt.tight_layout()
        plt.show()


def build_argparser():
    parser = argparse.ArgumentParser(description="A* visibility visualizer for predicted monster path.")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--threshold", type=int, default=128, help="Binarization threshold, default=128")
    parser.add_argument("--invert", action="store_true", help="Invert black/white semantics after thresholding")
    parser.add_argument("--brush-radius", type=int, default=2, help="Visibility paint brush radius, default=2")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    passable = load_and_binarize_image(
        str(image_path),
        threshold=args.threshold,
        invert=args.invert,
    )

    vis = AStarVisibilityVisualizer(passable, brush_radius=args.brush_radius)
    vis.show()


if __name__ == "__main__":
    main()
