#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

VIEW_SIZE = 128
LOCAL_SIZE = 21
LOCAL_RADIUS = LOCAL_SIZE // 2
TEXT_BOX = dict(boxstyle="round", facecolor="white", alpha=0.82)

# 8 邻接，用于边缘点聚类 / 连通性
NEIGHBOR8 = [
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
]

# 与训练代码一致：x 向右，y 向上
DIR8 = [
    (1, 0),    # E
    (1, -1),   # NE
    (0, -1),   # N
    (-1, -1),  # NW
    (-1, 0),   # W
    (-1, 1),   # SW
    (0, 1),    # S
    (1, 1),    # SE
]
DIR8_NAMES = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
DIR8_UNIT = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in DIR8]


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


def build_local_patch(binary_map: np.ndarray, cx: int, cy: int, size: int = LOCAL_SIZE):
    """
    以 (cx, cy) 为中心，裁出 size x size 的局部 patch。
    white=1 视为可通行；出界部分为未知/不可走。

    返回:
        passable21: [size,size] uint8
        known21:    [size,size] uint8
    """
    h, w = binary_map.shape
    half = size // 2
    passable21 = np.zeros((size, size), dtype=np.uint8)
    known21 = np.zeros((size, size), dtype=np.uint8)

    for j in range(size):
        for i in range(size):
            gx = cx + (i - half)
            gy = cy + (j - half)
            if 0 <= gx < w and 0 <= gy < h:
                known21[j, i] = 1
                passable21[j, i] = 1 if binary_map[gy, gx] > 0 else 0

    return passable21, known21


def extract_border_passable_points(passable21: np.ndarray, known21: np.ndarray):
    """
    提取 21x21 patch 边框上所有已知且可通行的点。
    返回 local 坐标列表 [(x, y), ...]，其中 x 向右，y 向上。
    """
    size = passable21.shape[0]
    pts = []

    # 上下边
    for x in range(size):
        for y in (0, size - 1):
            if known21[y, x] and passable21[y, x]:
                pts.append((x, y))

    # 左右边（去掉四个角，避免重复）
    for y in range(1, size - 1):
        for x in (0, size - 1):
            if known21[y, x] and passable21[y, x]:
                pts.append((x, y))

    return pts


def cluster_border_points(border_pts):
    """
    对边框上的可通行点做 8 邻接聚类。
    这里默认一个簇代表一段连续的道路截面。
    """
    pt_set = set(border_pts)
    visited = set()
    clusters = []

    for p in border_pts:
        if p in visited:
            continue

        q = deque([p])
        visited.add(p)
        cluster = []

        while q:
            x, y = q.popleft()
            cluster.append((x, y))
            for dx, dy in NEIGHBOR8:
                np_ = (x + dx, y + dy)
                if np_ in pt_set and np_ not in visited:
                    visited.add(np_)
                    q.append(np_)

        clusters.append(cluster)

    return clusters


def cluster_centers_local(clusters):
    centers = []
    for cluster in clusters:
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        centers.append((float(np.mean(xs)), float(np.mean(ys))))
    return centers


def compute_connected_mask(passable21: np.ndarray, known21: np.ndarray, start_x: int, start_y: int):
    """
    在 21x21 局部 patch 内，从 agent 位置出发，计算可通行连通域。
    返回 [size,size] uint8 mask，1 表示与 agent 连通。
    """
    h, w = passable21.shape
    connected = np.zeros((h, w), dtype=np.uint8)
    if not (0 <= start_x < w and 0 <= start_y < h):
        return connected
    if not (known21[start_y, start_x] and passable21[start_y, start_x]):
        return connected

    q = deque([(start_x, start_y)])
    connected[start_y, start_x] = 1
    while q:
        x, y = q.popleft()
        for dx, dy in NEIGHBOR8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if connected[ny, nx]:
                    continue
                if not known21[ny, nx]:
                    continue
                if not passable21[ny, nx]:
                    continue
                connected[ny, nx] = 1
                q.append((nx, ny))
    return connected


def find_connected_clusters(clusters_local, connected_mask: np.ndarray):
    """
    若某个边缘可通行点簇中存在至少一个点与 agent 连通，则该簇视为连通。
    返回 bool 列表。
    """
    flags = []
    for cluster in clusters_local:
        ok = any(connected_mask[y, x] > 0 for x, y in cluster)
        flags.append(ok)
    return flags


def compute_dir_scores_from_connected_clusters(
    centers_local,
    clusters_local,
    connected_flags,
    agent_local_x: float = LOCAL_RADIUS,
    agent_local_y: float = LOCAL_RADIUS,
):
    """
    只对“与 agent 连通”的簇做 8 方向余弦相似度聚合。

    每个簇：
    - 方向向量 = center - agent
    - 权重 = min(cluster_size / 3, 1)
    - 对每个方向加 max(0, cos_sim) * weight

    最后按所有连通簇总权重归一化到 [0,1]。
    """
    scores = np.zeros(8, dtype=np.float64)
    total_weight = 0.0
    contrib_desc = []

    for idx, (center, cluster, ok) in enumerate(zip(centers_local, clusters_local, connected_flags)):
        if not ok:
            continue

        vx = float(center[0] - agent_local_x)
        vy = float(center[1] - agent_local_y)
        norm = float(np.hypot(vx, vy))
        if norm < 1e-6:
            continue

        vec = np.array([vx / norm, vy / norm], dtype=np.float64)
        weight = min(len(cluster) / 3.0, 1.0)
        total_weight += weight * weight

        cluster_scores = []
        for i, d in enumerate(DIR8_UNIT):
            cos_sim = float(np.dot(vec, d))
            contrib = max(0.0, cos_sim) * weight
            scores[i] += contrib
            cluster_scores.append(contrib)

        contrib_desc.append((idx, weight, cluster_scores))

    if total_weight > 1e-6:
        scores /= total_weight
    scores = np.clip(scores, 0.0, 1.0)
    return scores, contrib_desc


def local_to_global_points(local_pts, cx: int, cy: int, size: int = LOCAL_SIZE):
    half = size // 2
    out = []
    for x, y in local_pts:
        gx = cx + (x - half)
        gy = cy + (y - half)
        out.append((gx, gy))
    return out


def local_to_global_centers(local_centers, cx: int, cy: int, size: int = LOCAL_SIZE):
    half = size // 2
    out = []
    for x, y in local_centers:
        gx = cx + (x - half)
        gy = cy + (y - half)
        out.append((gx, gy))
    return out


class BorderClusterViewer:
    def __init__(self, binary_map: np.ndarray):
        self.binary_map = binary_map
        self.h, self.w = binary_map.shape

        self.fig, self.ax = plt.subplots(figsize=(10.5, 10.5))
        self.ax.imshow(self.binary_map, cmap="gray", origin="lower", vmin=0, vmax=1)
        self.ax.set_title("Hover on a white pixel")
        self.ax.set_xlim(-0.5, self.w - 0.5)
        self.ax.set_ylim(-0.5, self.h - 0.5)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linewidth=0.2, alpha=0.35)

        self.agent_plot, = self.ax.plot([], [], marker="o", markersize=6, color="tab:blue")
        self.local_rect = Rectangle(
            (0, 0), LOCAL_SIZE, LOCAL_SIZE,
            fill=False, edgecolor="tab:green", linewidth=1.2
        )
        self.ax.add_patch(self.local_rect)

        self.border_plot, = self.ax.plot([], [], linestyle="None", marker="s", markersize=5,
                                         color="gold", markeredgecolor="black")
        self.center_plot, = self.ax.plot([], [], linestyle="None", marker="x", markersize=10,
                                         color="tab:red", markeredgewidth=2.0)
        self.connected_center_plot, = self.ax.plot([], [], linestyle="None", marker="o", markersize=16,
                                                   markerfacecolor="none", markeredgecolor="tab:cyan",
                                                   markeredgewidth=2.0)
        self.center_texts = []
        self.dir_arrow_artists = []
        self.dir_text_artists = []
        self.info_text = self.ax.text(1, self.h - 3, "", fontsize=10, va="top", ha="left", bbox=TEXT_BOX)

        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)

    def clear_center_texts(self):
        for artist in self.center_texts:
            try:
                artist.remove()
            except Exception:
                pass
        self.center_texts = []

    def clear_dir_artists(self):
        for artist in self.dir_arrow_artists + self.dir_text_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.dir_arrow_artists = []
        self.dir_text_artists = []

    def draw_dir_scores(self, cx: int, cy: int, dir_scores: np.ndarray):
        self.clear_dir_artists()

        base_len = 2.0
        gain_len = 3.5
        text_len = 6.3
        for i, ((dx, dy), score) in enumerate(zip(DIR8, dir_scores)):
            d = np.array([dx, dy], dtype=np.float64)
            d = d / np.linalg.norm(d)
            arrow_len = base_len + gain_len * float(score)

            arrow = self.ax.arrow(
                cx, cy,
                d[0] * arrow_len, d[1] * arrow_len,
                head_width=0.75,
                head_length=0.95,
                width=0.08,
                length_includes_head=True,
                color="tab:purple",
                alpha=0.35 + 0.55 * float(score),
                zorder=5,
            )
            self.dir_arrow_artists.append(arrow)

            txt = self.ax.text(
                cx + d[0] * text_len,
                cy + d[1] * text_len,
                f"{float(score):.2f}",
                fontsize=10,
                color="tab:purple",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.78),
                zorder=6,
            )
            self.dir_text_artists.append(txt)

    def on_move(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        cx = int(round(event.xdata))
        cy = int(round(event.ydata))
        if not (0 <= cx < self.w and 0 <= cy < self.h):
            return
        if self.binary_map[cy, cx] == 0:
            return

        passable21, known21 = build_local_patch(self.binary_map, cx, cy, size=LOCAL_SIZE)
        border_local_pts = extract_border_passable_points(passable21, known21)
        clusters_local = cluster_border_points(border_local_pts)
        centers_local = cluster_centers_local(clusters_local)
        connected_mask = compute_connected_mask(passable21, known21, LOCAL_RADIUS, LOCAL_RADIUS)
        connected_flags = find_connected_clusters(clusters_local, connected_mask)
        dir_scores, cluster_contribs = compute_dir_scores_from_connected_clusters(
            centers_local,
            clusters_local,
            connected_flags,
            agent_local_x=LOCAL_RADIUS,
            agent_local_y=LOCAL_RADIUS,
        )

        border_global_pts = local_to_global_points(border_local_pts, cx, cy, size=LOCAL_SIZE)
        centers_global = local_to_global_centers(centers_local, cx, cy, size=LOCAL_SIZE)

        self.agent_plot.set_data([cx], [cy])
        self.local_rect.set_xy((cx - LOCAL_RADIUS - 0.5, cy - LOCAL_RADIUS - 0.5))

        if border_global_pts:
            bx = [p[0] for p in border_global_pts]
            by = [p[1] for p in border_global_pts]
            self.border_plot.set_data(bx, by)
        else:
            self.border_plot.set_data([], [])

        if centers_global:
            cxs = [p[0] for p in centers_global]
            cys = [p[1] for p in centers_global]
            self.center_plot.set_data(cxs, cys)

            ccx = [p[0] for p, ok in zip(centers_global, connected_flags) if ok]
            ccy = [p[1] for p, ok in zip(centers_global, connected_flags) if ok]
            self.connected_center_plot.set_data(ccx, ccy)
        else:
            self.center_plot.set_data([], [])
            self.connected_center_plot.set_data([], [])

        self.clear_center_texts()
        for idx, ((gx, gy), cluster, ok) in enumerate(zip(centers_global, clusters_local, connected_flags)):
            txt = self.ax.text(
                gx + 0.8,
                gy + 0.8,
                f"C{idx}:{len(cluster)}{'*' if ok else ''}",
                fontsize=9,
                color="tab:red" if ok else "dimgray",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75),
            )
            self.center_texts.append(txt)

        self.draw_dir_scores(cx, cy, dir_scores)

        center_desc = [
            f"C{idx}=({cg[0]:.1f},{cg[1]:.1f}),n={len(cl)},conn={int(ok)}"
            for idx, (cg, cl, ok) in enumerate(zip(centers_global, clusters_local, connected_flags))
        ]
        score_desc = ", ".join(f"{name}={float(sc):.2f}" for name, sc in zip(DIR8_NAMES, dir_scores))
        self.info_text.set_text(
            f"agent=({cx}, {cy})\n"
            f"border_passable={len(border_local_pts)}\n"
            f"clusters={len(clusters_local)}\n"
            f"connected_clusters={sum(int(x) for x in connected_flags)}\n"
            f"dir_scores: {score_desc}\n"
            + ("\n".join(center_desc[:7]) if center_desc else "no border cluster")
        )
        self.fig.canvas.draw_idle()

    def show(self):
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Border passable-point clustering visualizer with 8-dir cosine scores")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()

    binary_map = load_binary_map(args.image_path, threshold=args.threshold, invert=args.invert)
    viewer = BorderClusterViewer(binary_map)
    viewer.show()


if __name__ == "__main__":
    main()
