import cv2
import numpy as np

# 这里改成你的图片路径
image_path = r"D:\code\fwwb\kaiwu\code\output_5_largest_cc.png"

# 读取灰度图
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"无法读取图片: {image_path}")

# 二值化
# 大于127的像素设为255，否则设为0
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 总像素点数
total_pixels = binary.size

# 白点数量（值为255）
white_pixels = np.count_nonzero(binary == 255)

# 黑点数量（可选）
black_pixels = np.count_nonzero(binary == 0)

print(f"总点数: {total_pixels}")
print(f"白点数: {white_pixels}")
print(f"黑点数: {black_pixels}")

# 如果你想看看二值化结果，可以取消注释
# cv2.imshow("binary", binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()