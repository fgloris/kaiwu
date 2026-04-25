#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_PATH = r"D:\code\fwwb\kaiwu\map\1.png"
OUT_IMAGE_PATH = r"D:\code\fwwb\kaiwu\map\1-r.png"

MAP_SIZE = 128
SCAN_ANGLES_DEG = list(range(0, 360, 15))
RAY_MAX_LEN = 18.0
RAY_STEP_SIZE = 1.0


def load_binary_map(path):
    image = Image.open(path).convert("L")
    try:
        resample = Image.Resampling.NEAREST
    except AttributeError:
        resample = Image.NEAREST

    if image.size != (MAP_SIZE, MAP_SIZE):
        image = image.resize((MAP_SIZE, MAP_SIZE), resample=resample)

    arr = np.asarray(image, dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


def ray_collision_score(passable_map, start_x, start_y, angle_deg, max_len=RAY_MAX_LEN, step_size=RAY_STEP_SIZE):
    theta = np.deg2rad(angle_deg)
    dx = np.cos(theta)
    dy = -np.sin(theta)

    dist = step_size
    while dist <= max_len:
        x = int(round(start_x + dx * dist))
        y = int(round(start_y + dy * dist))

        if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
            return float(np.clip(dist / max_len, 0.0, 1.0))

        if passable_map[y, x] == 0:
            return float(np.clip(dist / max_len, 0.0, 1.0))

        dist += step_size

    return 1.0


def total_ray_collision_score(passable_map, x, y):
    total_score = 0.0
    for angle_deg in SCAN_ANGLES_DEG:
        total_score += ray_collision_score(passable_map, x, y, angle_deg)
    return total_score


def color_for_normalized_score(score_norm):
    score_norm = float(np.clip(score_norm, 0.0, 1.0))
    hue = 0.66 * score_norm
    saturation = 0.95
    value = 1.0
    return hsv_to_rgb_uint8(hue, saturation, value)


def hsv_to_rgb_uint8(h, s, v):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return np.asarray([r, g, b], dtype=np.float32).clip(0.0, 1.0) * 255.0


def main():
    passable_map = load_binary_map(IMAGE_PATH)
    max_total_score = float(len(SCAN_ANGLES_DEG))
    output = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)

    for y in range(MAP_SIZE):
        for x in range(MAP_SIZE):
            if passable_map[y, x] == 0:
                continue

            total_score = total_ray_collision_score(passable_map, x, y)
            output[y, x] = color_for_normalized_score(total_score / max_total_score)

    Image.fromarray(output, mode="RGB").save(OUT_IMAGE_PATH)
    print(f"Saved: {OUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()
