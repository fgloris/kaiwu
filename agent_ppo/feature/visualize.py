#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt


VIEW_MAP_SIZE = 36
DIR8 = [
    (1, 0),    # 东
    (1, -1),   # 东北
    (0, -1),   # 北
    (-1, -1),  # 西北
    (-1, 0),   # 西
    (-1, 1),   # 西南
    (0, 1),    # 南
    (1, 1),    # 东南
]
DIR8_NAMES = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def parse_move_debug_line(line: str):
    tag = "[move_debug]"
    idx = line.find(tag)
    if idx < 0:
        raise ValueError("line does not contain [move_debug]")
    payload = line[idx + len(tag):].strip()
    return json.loads(payload)


def bits_to_map(bits: str, size: int = VIEW_MAP_SIZE):
    arr = np.array([1 if ch == "1" else 0 for ch in bits], dtype=np.float32)
    if arr.size != size * size:
        raise ValueError(f"bits len mismatch: got {arr.size}, expect {size * size}")
    return arr.reshape(size, size)


def angle_to_vec(angle_deg: float):
    theta = np.deg2rad(angle_deg)
    dx = np.cos(theta)
    dz = -np.sin(theta)   # 与训练代码保持一致
    return dx, dz


def visualize_move_debug_data(data):
    passable_map = bits_to_map(data["map"], size=VIEW_MAP_SIZE)
    rays = data["rays"]
    move_scores = data["move_scores"]
    step = data.get("step", None)

    fig, ax = plt.subplots(figsize=(10, 10))

    # 地图
    ax.imshow(passable_map.T, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)

    # hero 位置：局部图中心
    hero_x = (VIEW_MAP_SIZE - 1) / 2.0
    hero_y = (VIEW_MAP_SIZE - 1) / 2.0
    ax.scatter([hero_x], [hero_y], s=180, marker="*")

    # 先画统一的全局 rays
    for angle_deg, ray_score in rays:
        ux, uy = angle_to_vec(float(angle_deg))
        ray_len = 2.0 + 3.0 * float(ray_score)

        ax.plot(
            [hero_x, hero_x + ux * ray_len],
            [hero_y, hero_y + uy * ray_len],
            linewidth=1.2,
            alpha=0.25 + 0.55 * float(ray_score),
        )

        ax.text(
            hero_x + ux * (ray_len + 0.6),
            hero_y + uy * (ray_len + 0.6),
            f"{int(angle_deg)}:{float(ray_score):.1f}",
            fontsize=7
        )

    # 再画 8 个动作方向和 score
    for i, ((dx, dz), score) in enumerate(zip(DIR8, move_scores)):
        ax.arrow(
            hero_x, hero_y,
            dx * 1.4, dz * 1.4,
            head_width=0.22,
            head_length=0.22,
            length_includes_head=True
        )

        ax.text(
            hero_x + dx * 2.1,
            hero_y + dz * 2.1,
            f"{DIR8_NAMES[i]}:{float(score):.2f}",
            fontsize=10
        )

    ax.set_title(f"move_debug step={step}")
    ax.set_xlim(-1, VIEW_MAP_SIZE)
    ax.set_ylim(-1, VIEW_MAP_SIZE)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.show()


def visualize_move_debug_line(line: str):
    data = parse_move_debug_line(line)
    visualize_move_debug_data(data)


def visualize_move_debug_file(log_path: str, index: int = -1):
    """
    从日志文件中提取所有 [move_debug] 行，并可视化第 index 条。
    index=-1 表示最后一条。
    """
    lines = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "[move_debug]" in line:
                lines.append(line.rstrip("\n"))

    if not lines:
        raise ValueError("no [move_debug] lines found")

    visualize_move_debug_line(lines[index])


if __name__ == "__main__":
    visualize_move_debug_line(
        r''''''
    )