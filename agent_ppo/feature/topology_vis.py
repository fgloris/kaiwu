#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize topology debug payload dumped by:
    _log_passable_map_and_topology(...)

支持两种输入：
1. 一整条 logger.warning 日志行，例如 [move_topology]{...}
2. 纯 JSON payload

示例：
    python topology_visualizer.py --log-file debug.log --tag move_topology
    python topology_visualizer.py --json '[move_topology]{"step":1,"map":"...","move_scores":[...]}'
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


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


def parse_tagged_line(line: str, tag: str = "move_topology"):
    key = f"[{tag}]"
    idx = line.find(key)
    if idx < 0:
        raise ValueError(f"line does not contain {key}")
    payload = line[idx + len(key):].strip()
    return json.loads(payload)


def bits_to_map(bits: str, size: int = VIEW_MAP_SIZE):
    arr = np.array([1 if ch == "1" else 0 for ch in bits], dtype=np.float32)
    if arr.size != size * size:
        raise ValueError(f"bits len mismatch: got {arr.size}, expect {size * size}")
    return arr.reshape(size, size)


def visualize_topology_data(data):
    passable_map = bits_to_map(data["map"], size=VIEW_MAP_SIZE)
    move_scores = data["move_scores"]
    step = data.get("step", None)

    fig, ax = plt.subplots(figsize=(10, 10))

    # 严格对齐参考代码坐标系：
    # - 用 passable_map.T
    # - origin="lower"
    ax.imshow(passable_map.T, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)

    hero_x = (VIEW_MAP_SIZE - 1) / 2.0
    hero_y = (VIEW_MAP_SIZE - 1) / 2.0
    ax.scatter([hero_x], [hero_y], s=180, marker="*")

    for i, ((dx, dz), score) in enumerate(zip(DIR8, move_scores)):
        ax.arrow(
            hero_x, hero_y,
            dx * 1.4, dz * 1.4,
            head_width=0.22,
            head_length=0.22,
            length_includes_head=True,
        )

        ax.text(
            hero_x + dx * 2.1,
            hero_y + dz * 2.1,
            f"{DIR8_NAMES[i]}:{float(score):.2f}",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, linewidth=0.5),
        )

    ax.set_title(f"move_topology step={step}")
    ax.set_xlim(-1, VIEW_MAP_SIZE)
    ax.set_ylim(-1, VIEW_MAP_SIZE)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.show()


def visualize_topology_line(line: str, tag: str = "move_topology"):
    data = parse_tagged_line(line, tag=tag)
    visualize_topology_data(data)


def visualize_topology_file(log_path: str, index: int = -1, tag: str = "move_topology"):
    """
    从日志文件中提取所有 [tag] 行，并可视化第 index 条。
    index=-1 表示最后一条。
    """
    lines = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if f"[{tag}]" in line:
                lines.append(line.rstrip("\n"))

    if not lines:
        raise ValueError(f"no [{tag}] lines found")

    visualize_topology_line(lines[index], tag=tag)

def main():

    payload = json.loads(
        r'''{"step":103,"map":"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001111000000000001111110000000000000001111100000000001111110000000000000001111110000000000111110000000000000001111111100000000111110000000000000001111111110000000011111111111000000001111111111100000001111111111000000001111111111110000000011111111000000001111111111111100000000111111000000000111111111111110000000000001000000000011111111111111000000000000000000000001111111111111100000000000000000000000011111111111110000000000000000000000001111111111111100000000000000000000000111111111111100000000000000000000000001111111111110000000000000000000000000111111111111000000000000000000000000011111111111110011000000000000000000001111111111111111000000000000000000000111111111111111000000000000000000000011111111111111000000000000000000000001111111111111000000000000000000000000111111111111000000000000000000000000001111111110000000000000000000000000000000100000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000011100000000000000000000000000000000011110000000000000000","move_scores":[0.4459,0.4209,0.4621,0.4977,0.4397,0.4208,0.4571,0.5013]}'''
    )
    visualize_topology_data(payload)


if __name__ == "__main__":
    main()
