#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt


VIEW_MAP_SIZE = 36

DIR8_NAMES = [
    "E", "NE", "N", "NW", "W", "SW", "S", "SE"
]


def parse_move_debug_line(line: str):
    tag = "[move_debug]"
    idx = line.find(tag)
    if idx < 0:
        raise ValueError("line does not contain [move_debug]")

    payload = line[idx + len(tag):].strip()
    data = json.loads(payload)
    return data


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
    size = int(data["map_size"])
    passable_map = bits_to_map(data["passable_map_bits"], size=size)
    actions = data["actions"]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(passable_map.T, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)

    center = (size - 1) / 2.0
    hero_x = center
    hero_y = center

    # 英雄位置
    ax.scatter([hero_x], [hero_y], s=120, marker="*")

    # 画每个动作方向
    for action in actions:
        action_idx = int(action["action_idx"])
        dx, dz = action["dir"]
        score = float(action["score"])
        rays = action["rays"]

        # 下一步落点（相对 36x36 局部图）
        next_x = hero_x + dx
        next_y = hero_y + dz

        # 主方向箭头
        ax.arrow(
            hero_x, hero_y, dx * 1.2, dz * 1.2,
            head_width=0.25, head_length=0.25, length_includes_head=True
        )

        ax.text(
            next_x + 0.2,
            next_y + 0.2,
            f"{DIR8_NAMES[action_idx]}:{score:.2f}",
            fontsize=9
        )

        # 各条 ray
        for ray in rays:
            angle = float(ray["scan_angle"])
            weight = float(ray["weight"])
            ray_score = float(ray["ray_score"])

            ux, uy = angle_to_vec(angle)

            # 用长度和线宽体现 score / weight
            ray_len = 2.5 + 3.0 * weight + 2.0 * ray_score

            ax.plot(
                [next_x, next_x + ux * ray_len],
                [next_y, next_y + uy * ray_len],
                linewidth=1.0 + 2.0 * weight,
                alpha=0.25 + 0.55 * max(weight, 0.05),
            )

            ax.text(
                next_x + ux * ray_len,
                next_y + uy * ray_len,
                f"{int(angle)}|s{ray_score:.1f}|w{weight:.2f}",
                fontsize=7
            )

    step_no = data.get("step_no", None)
    ax.set_title(f"move_debug step={step_no}")
    ax.set_xlim(-1, size)
    ax.set_ylim(-1, size)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.show()


def visualize_move_debug_line(line: str):
    data = parse_move_debug_line(line)
    visualize_move_debug_data(data)

if __name__ == "__main__":
    # 看日志文件最后一条
    # 例如：
    # python visualize_move_debug.py
    # 然后把下面路径改成你的日志文件
    visualize_move_debug_line(
        ""
    )