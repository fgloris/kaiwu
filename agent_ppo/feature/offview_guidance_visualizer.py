
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TARGET_SIZE = 128
LOCAL_SIZE = 21
LOCAL_HALF = LOCAL_SIZE // 2

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


def mask_monster_danger_zone_local(local_passable: np.ndarray, monster_local, radius: int = 3) -> np.ndarray:
    masked = np.array(local_passable, copy=True)
    if monster_local is None:
        return masked

    mx, my = monster_local
    if not (0 <= mx < LOCAL_SIZE and 0 <= my < LOCAL_SIZE):
        return masked

    x0 = max(0, mx - radius)
    x1 = min(LOCAL_SIZE, mx + radius + 1)
    y0 = max(0, my - radius)
    y1 = min(LOCAL_SIZE, my + radius + 1)
    masked[y0:y1, x0:x1] = 0
    return masked


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

        queue = [seed]
        visited.add(seed)
        cluster = []

        while queue:
            x, y = queue.pop(0)
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

    queue = [(start_x, start_y)]
    connected[start_y, start_x] = 1

    while queue:
        x, y = queue.pop(0)
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


def select_reachable_opposite_boundary_cluster(connected_clusters, monster_vec, angle_cos_threshold=0.0):
    if not connected_clusters:
        return None

    mdx, mdz = monster_vec
    mnorm = float(np.hypot(mdx, mdz))
    if mnorm <= 1e-6:
        return None

    away_x = -mdx / mnorm
    away_z = -mdz / mnorm

    best = None
    best_align = -1e9
    for c in connected_clusters:
        cx, cy = c["center"]
        vx = float(cx - LOCAL_HALF)
        vy = float(cy - LOCAL_HALF)
        dist = float(np.hypot(vx, vy))
        if dist <= 1e-6:
            continue
        align = float((vx / dist) * away_x + (vy / dist) * away_z)
        if align > best_align:
            best_align = align
            best = {
                "center": (float(cx), float(cy)),
                "size": int(c["size"]),
                "dist": float(dist),
                "align": float(align),
            }

    if best is None or best["align"] < angle_cos_threshold:
        return None
    return best


def action_to_dir_vec(action_idx):
    if action_idx is None:
        return None

    action_idx = int(action_idx)
    if 0 <= action_idx < 8:
        dx, dz = DIR8[action_idx]
    elif 8 <= action_idx < 16:
        dx, dz = DIR8[action_idx - 8]
    else:
        return None

    norm = float(np.hypot(dx, dz))
    if norm <= 1e-6:
        return None
    return float(dx / norm), float(dz / norm)


def compute_offview_guidance_info(local_passable, monster_vec, monster_local, last_action: int, angle_cos_threshold=0.0):
    masked_local_passable = mask_monster_danger_zone_local(local_passable, monster_local, radius=3)

    boundary_pts = get_boundary_passable_points(masked_local_passable)
    clusters = cluster_boundary_points(boundary_pts)
    connected_mask = compute_local_connected_mask(masked_local_passable)

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

    target = select_reachable_opposite_boundary_cluster(
        connected_clusters_only,
        monster_vec=monster_vec,
        angle_cos_threshold=angle_cos_threshold,
    )

    action_vec = action_to_dir_vec(last_action)
    cos_sim = None
    reward = 0.0

    if target is not None and action_vec is not None:
        cx, cy = target["center"]
        vx = float(cx - LOCAL_HALF)
        vy = float(cy - LOCAL_HALF)
        vnorm = float(np.hypot(vx, vy))
        if vnorm > 1e-6:
            tx = vx / vnorm
            ty = vy / vnorm
            ax, ay = action_vec
            cos_sim = float(ax * tx + ay * ty)
            reward = max(0.0, cos_sim)

    return {
        "masked_local_passable": masked_local_passable,
        "boundary_pts": boundary_pts,
        "clusters": all_clusters,
        "connected_mask": connected_mask,
        "connected_opening_count": len(connected_clusters_only),
        "is_dangerous": len(connected_clusters_only) <= 1,
        "target": target,
        "action_vec": action_vec,
        "cos_sim": cos_sim,
        "reward": reward,
    }



class OffviewGuidanceVisualizer:
    def __init__(self, binary_map: np.ndarray, last_action: int = 4, angle_cos_threshold: float = 0.0):
        self.binary_map = binary_map
        self.h, self.w = binary_map.shape
        self.last_action = last_action
        self.angle_cos_threshold = angle_cos_threshold
        self._last_hover = None

        self.monster_world = None  # absolute image coords
        self.monster_local = None  # relative to current patch

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.imshow(self.binary_map, origin="upper", cmap="gray", vmin=0.0, vmax=1.0)

        self.agent_scatter = self.ax.scatter([], [], s=100, marker="o")
        self.monster_scatter = self.ax.scatter([], [], s=100, marker="X")
        self.local_rect = plt.Rectangle((0, 0), LOCAL_SIZE, LOCAL_SIZE, fill=False, linewidth=1.5)
        self.ax.add_patch(self.local_rect)
        self.monster_danger_rect = plt.Rectangle((0, 0), 7, 7, fill=True, alpha=0.25)
        self.ax.add_patch(self.monster_danger_rect)

        self.boundary_scatter = self.ax.scatter([], [], s=35, marker="s")
        self.cluster_center_scatter = self.ax.scatter([], [], s=90, marker="x")
        self.connected_circle_scatter = self.ax.scatter([], [], s=180, marker="o", facecolors="none", linewidths=1.5)
        self.target_scatter = self.ax.scatter([], [], s=180, marker="*")

        self.monster_arrow = None
        self.away_arrow = None
        self.action_arrow = None
        self.target_arrow = None
        self.cluster_texts = []

        self.info_text = self.ax.text(
            0.01, 0.99, "",
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round", alpha=0.85),
        )

        self.ax.set_title("Hover=agent | Right click=set monster | keys: - / = action")
        self.ax.set_xlim(-0.5, self.w - 0.5)
        self.ax.set_ylim(self.h - 0.5, -0.5)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linewidth=0.25)

        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _remove_arrows_and_texts(self):
        for name in ["monster_arrow", "away_arrow", "action_arrow", "target_arrow"]:
            obj = getattr(self, name)
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
                setattr(self, name, None)
        for t in self.cluster_texts:
            try:
                t.remove()
            except Exception:
                pass
        self.cluster_texts = []

    def _draw_arrow(self, x, y, dx, dy, scale=2.0):
        return self.ax.arrow(
            x, y, dx * scale, dy * scale,
            head_width=0.28, head_length=0.28, length_includes_head=True
        )

    def clear_overlay(self):
        self.agent_scatter.set_offsets(np.empty((0, 2)))
        self.monster_scatter.set_offsets(np.array([self.monster_world], dtype=np.float64) if self.monster_world is not None else np.empty((0, 2)))
        self.boundary_scatter.set_offsets(np.empty((0, 2)))
        self.cluster_center_scatter.set_offsets(np.empty((0, 2)))
        self.connected_circle_scatter.set_offsets(np.empty((0, 2)))
        self.target_scatter.set_offsets(np.empty((0, 2)))
        self.local_rect.set_xy((-100, -100))
        self.local_rect.set_width(LOCAL_SIZE)
        self.local_rect.set_height(LOCAL_SIZE)
        self.monster_danger_rect.set_xy((-100, -100))
        self.monster_danger_rect.set_width(7)
        self.monster_danger_rect.set_height(7)
        self.info_text.set_text("Move mouse onto a white/passable pixel | Right click to set monster")
        self._remove_arrows_and_texts()

    def refresh_last_hover(self):
        if self._last_hover is None:
            self.clear_overlay()
            self.fig.canvas.draw_idle()
            return
        self.update_at(*self._last_hover)

    def on_key(self, event):
        if event.key == "-":
            self.last_action -= 1
            if self.last_action < 0:
                self.last_action = 15
            self.refresh_last_hover()
        elif event.key == "=":
            self.last_action += 1
            if self.last_action > 15:
                self.last_action = 0
            self.refresh_last_hover()

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button != 3:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if not (0 <= x < self.w and 0 <= y < self.h):
            return

        self.monster_world = (x, y)
        self.refresh_last_hover()

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
            self.info_text.set_text(f"({x}, {y}) is wall/blocked | Right click to set monster")
            self._last_hover = None
            self.fig.canvas.draw_idle()
            return

        self._last_hover = (x, y)
        self.update_at(x, y)

    def update_at(self, x: int, y: int):
        patch = extract_local_patch(self.binary_map, x, y, LOCAL_SIZE)

        if self.monster_world is None:
            monster_vec = None
            self.monster_local = None
        else:
            mx, my = self.monster_world
            self.monster_local = (mx - x + LOCAL_HALF, my - y + LOCAL_HALF)
            monster_vec = (mx - x, my - y)

        self.agent_scatter.set_offsets(np.array([[x, y]], dtype=np.float64))
        self.monster_scatter.set_offsets(np.array([self.monster_world], dtype=np.float64) if self.monster_world is not None else np.empty((0, 2)))
        self.local_rect.set_xy((x - LOCAL_HALF - 0.5, y - LOCAL_HALF - 0.5))
        self._remove_arrows_and_texts()
        if self.monster_world is not None:
            mx, my = self.monster_world
            self.monster_danger_rect.set_xy((mx - 3 - 0.5, my - 3 - 0.5))
        else:
            self.monster_danger_rect.set_xy((-100, -100))

        if monster_vec is None or (abs(monster_vec[0]) < 1e-6 and abs(monster_vec[1]) < 1e-6):
            self.boundary_scatter.set_offsets(np.empty((0, 2)))
            self.cluster_center_scatter.set_offsets(np.empty((0, 2)))
            self.connected_circle_scatter.set_offsets(np.empty((0, 2)))
            self.target_scatter.set_offsets(np.empty((0, 2)))
            self.info_text.set_text(
                f"agent=({x}, {y})\n"
                f"last_action={self.last_action}\n"
                f"monster=not set\n"
                f"Right click anywhere to set monster position"
            )
            self.fig.canvas.draw_idle()
            return

        info = compute_offview_guidance_info(
            patch,
            monster_vec=monster_vec,
            monster_local=self.monster_local,
            last_action=self.last_action,
            angle_cos_threshold=self.angle_cos_threshold,
        )

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

            label = f"C{idx}:{c['size']}"
            if c["connected"]:
                label += "*"
            self.cluster_texts.append(self.ax.text(gx + 0.25, gy + 0.25, label, fontsize=8))

        self.cluster_center_scatter.set_offsets(np.array(centers, dtype=np.float64) if centers else np.empty((0, 2)))
        self.connected_circle_scatter.set_offsets(np.array(connected_centers, dtype=np.float64) if connected_centers else np.empty((0, 2)))

        target = info["target"]
        if target is not None:
            tx, ty = target["center"]
            gx = x + (tx - LOCAL_HALF)
            gy = y + (ty - LOCAL_HALF)
            self.target_scatter.set_offsets(np.array([[gx, gy]], dtype=np.float64))
            vx = tx - LOCAL_HALF
            vy = ty - LOCAL_HALF
            vnorm = float(np.hypot(vx, vy))
            if vnorm > 1e-6:
                self.target_arrow = self._draw_arrow(x, y, vx / vnorm, vy / vnorm, scale=2.5)
        else:
            self.target_scatter.set_offsets(np.empty((0, 2)))

        mvx, mvy = monster_vec
        mnorm = float(np.hypot(mvx, mvy))
        if mnorm > 1e-6:
            ux, uy = mvx / mnorm, mvy / mnorm
            self.monster_arrow = self._draw_arrow(x, y, ux, uy, scale=2.0)
            self.away_arrow = self._draw_arrow(x, y, -ux, -uy, scale=2.0)

        action_vec = info["action_vec"]
        if action_vec is not None:
            ax, ay = action_vec
            self.action_arrow = self._draw_arrow(x, y, ax, ay, scale=1.6)

        shown_cos = "None" if info["cos_sim"] is None else f"{info['cos_sim']:.3f}"
        shown_reward = f"{info['reward']:.3f}"

        lines = [
            f"agent=({x}, {y})",
            f"monster=({self.monster_world[0]}, {self.monster_world[1]})",
            f"monster_vec=({monster_vec[0]}, {monster_vec[1]})",
            f"last_action={self.last_action}",
            f"boundary_pts={len(info['boundary_pts'])}",
            f"clusters={len(info['clusters'])}",
            f"connected_clusters={sum(1 for c in info['clusters'] if c['connected'])}",
            f"connected_opening_count={info['connected_opening_count']}",
            f"dangerous={info['is_dangerous']}",
        ]
        if target is not None:
            lines.append(f"target_center=({target['center'][0]:.2f}, {target['center'][1]:.2f})")
            lines.append(f"target_align={target['align']:.3f}")
        else:
            lines.append("target=None")
        lines.append(f"cos_sim={shown_cos}")
        lines.append(f"reward=max(0, cos)={shown_reward}")
        lines.append("Right click: set monster | Hover: set agent | -/=: change action")
        self.info_text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def show(self):
        self.clear_overlay()
        plt.tight_layout()
        plt.show()


def build_argparser():
    parser = argparse.ArgumentParser(description="Interactive offview-guidance visualizer with right-click monster position.")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--threshold", type=int, default=128, help="Binarization threshold, default=128")
    parser.add_argument("--invert", action="store_true", help="Invert black/white semantics after thresholding")
    parser.add_argument("--last-action", type=int, default=4, help="Last action index 0~15, default=4 (W)")
    parser.add_argument("--angle-cos-threshold", type=float, default=0.0, help="Cluster align threshold, default=0.0")
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

    vis = OffviewGuidanceVisualizer(
        binary_map,
        last_action=args.last_action,
        angle_cos_threshold=args.angle_cos_threshold,
    )
    vis.show()


if __name__ == "__main__":
    main()
