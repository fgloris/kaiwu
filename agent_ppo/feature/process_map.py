#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from pathlib import Path

from PIL import Image, ImageFilter


IMAGE_PATH = r"D:\code\fwwb\kaiwu\map\7.png"
OUT_IMAGE_PATH = r"D:\code\fwwb\kaiwu\map\7.png"
THRESHOLD = 142
OUTPUT_SIZE = 128
OPEN_KERNEL_SIZE = 3


def center_crop_square(image):
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def opening(image, kernel_size=OPEN_KERNEL_SIZE):
    if kernel_size <= 1:
        return image
    if kernel_size % 2 == 0:
        raise ValueError("OPEN_KERNEL_SIZE must be odd.")

    eroded = image.filter(ImageFilter.MinFilter(kernel_size))
    return eroded.filter(ImageFilter.MaxFilter(kernel_size))


def main():
    image = Image.open(IMAGE_PATH)

    image = image.convert("L")
    image = image.point(lambda pixel: 255 if pixel >= THRESHOLD else 0)
    image = opening(image)
    image = center_crop_square(image)

    try:
        resample = Image.Resampling.NEAREST
    except AttributeError:
        resample = Image.NEAREST

    image = image.resize((OUTPUT_SIZE, OUTPUT_SIZE), resample=resample)
    image.convert("L").save(OUT_IMAGE_PATH)


if __name__ == "__main__":
    main()
