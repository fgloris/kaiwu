#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import matplotlib.pyplot as plt


# 直接改这里
LOG_PATH = "log.txt"

# 选择要看的图类型：
# "THIN_MAP" 或 "THIN_CC_MAP"
KEY = "THIN_CC_MAP"


def decode_binary_map_zero_rle(encoded_str):
    """
    解码形如:
        HxW|z10;1z4;1z5;
    的 zero-RLE 字符串，返回 2D 0/1 numpy array。
    """
    shape_str, payload = encoded_str.strip().split("|", 1)
    h_str, w_str = shape_str.split("x")
    h, w = int(h_str), int(w_str)

    data = []
    i = 0
    n = len(payload)

    while i < n:
        ch = payload[i]
        if ch == "1":
            data.append(1)
            i += 1
        elif ch == "z":
            j = i + 1
            while j < n and payload[j] != ";":
                j += 1
            if j >= n:
                raise ValueError("invalid zero-rle payload: missing ';'")
            count = int(payload[i + 1:j])
            data.extend([0] * count)
            i = j + 1
        else:
            raise ValueError(f"invalid zero-rle payload char: {ch}")

    expected = h * w
    if len(data) != expected:
        raise ValueError(f"decoded length mismatch: got {len(data)}, expected {expected}")

    return np.array(data, dtype=np.uint8).reshape(h, w)


def load_maps_from_log(log_path, key="THIN_CC_MAP"):
    """
    读取日志中形如：
        THIN_CC_MAP step=12 hero=(52,71) data=128x128|z10;1z4;...
    的行
    """
    results = []

    pattern = re.compile(
        rf"{re.escape(key)}\s+step=(\d+)(?:\s+hero=\(([-\d]+),([-\d]+)\))?\s+data=([0-9]+x[0-9]+\|[1z0-9;]+)"
    )

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            m = pattern.search(line)
            if not m:
                continue

            step = int(m.group(1))
            hero_x = m.group(2)
            hero_y = m.group(3)
            encoded = m.group(4)

            hero = None
            if hero_x is not None and hero_y is not None:
                hero = (int(hero_x), int(hero_y))

            try:
                arr = decode_binary_map_zero_rle(encoded)
                results.append({
                    "line_no": line_no,
                    "step": step,
                    "hero": hero,
                    "array": arr,
                    "raw_line": line.rstrip("\n"),
                })
            except Exception as e:
                print(f"[WARN] line {line_no} decode failed: {e}")

    return results


def visualize_frames(frames, key="THIN_CC_MAP"):
    if not frames:
        print(f"No frames found for key={key}")
        return

    idx = 0
    fig, ax = plt.subplots(figsize=(7, 7))

    def redraw():
        ax.clear()
        item = frames[idx]
        arr = item["array"]

        ax.imshow(arr, cmap="gray", interpolation="nearest", vmin=0, vmax=1)

        title = f"{key} | frame {idx + 1}/{len(frames)} | step={item['step']} | line={item['line_no']}"
        if item["hero"] is not None:
            title += f" | hero={item['hero']}"
        ax.set_title(title)
        ax.axis("off")

        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal idx
        if event.key in ["right", "d", "n", "down"]:
            idx = min(idx + 1, len(frames) - 1)
            redraw()
        elif event.key in ["left", "a", "p", "up"]:
            idx = max(idx - 1, 0)
            redraw()
        elif event.key == "home":
            idx = 0
            redraw()
        elif event.key == "end":
            idx = len(frames) - 1
            redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.tight_layout()
    plt.show()


def main():
    frames = load_maps_from_log(LOG_PATH, key=KEY)
    print(f"Loaded {len(frames)} frames for key={KEY}")
    visualize_frames(frames, key=KEY)


if __name__ == "__main__":
    main()