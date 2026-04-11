import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


# =========================
# 基础图像处理
# =========================

def center_crop_to_square(img):
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side]


def binarize_by_brightness(img_bgr, thresh=128):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return gray, binary


# =========================
# Zhang-Suen thinning
# 输入/输出均为 0/255
# =========================

def _neighbors(x, y, img01):
    return [
        img01[x - 1, y],     # P2
        img01[x - 1, y + 1], # P3
        img01[x,     y + 1], # P4
        img01[x + 1, y + 1], # P5
        img01[x + 1, y],     # P6
        img01[x + 1, y - 1], # P7
        img01[x,     y - 1], # P8
        img01[x - 1, y - 1], # P9
    ]


def _transitions(neigh):
    seq = neigh + [neigh[0]]
    return sum((seq[i] == 0 and seq[i + 1] == 1) for i in range(8))


def zhang_suen_thinning(binary, max_iter=None):
    img = (binary > 0).astype(np.uint8).copy()
    h, w = img.shape

    changed = True
    it = 0

    while changed:
        changed = False
        to_delete = []

        # step 1
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if img[x, y] != 1:
                    continue
                n = _neighbors(x, y, img)
                s = sum(n)
                t = _transitions(n)
                if (
                    2 <= s <= 6 and
                    t == 1 and
                    n[0] * n[2] * n[4] == 0 and
                    n[2] * n[4] * n[6] == 0
                ):
                    to_delete.append((x, y))

        if to_delete:
            changed = True
            for x, y in to_delete:
                img[x, y] = 0

        to_delete = []

        # step 2
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if img[x, y] != 1:
                    continue
                n = _neighbors(x, y, img)
                s = sum(n)
                t = _transitions(n)
                if (
                    2 <= s <= 6 and
                    t == 1 and
                    n[0] * n[2] * n[6] == 0 and
                    n[0] * n[4] * n[6] == 0
                ):
                    to_delete.append((x, y))

        if to_delete:
            changed = True
            for x, y in to_delete:
                img[x, y] = 0

        it += 1
        if max_iter is not None and it >= max_iter:
            break

    return (img * 255).astype(np.uint8)


# =========================
# 并查集
# =========================

def uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def uf_union(parent, size, a, b):
    ra = uf_find(parent, a)
    rb = uf_find(parent, b)
    if ra == rb:
        return
    if size[ra] < size[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    size[ra] += size[rb]


# =========================
# 最大连通分量
# 在 thin_01 上做 8 邻接连通
# 输入: 0/1
# 输出: largest_cc_mask(0/1)
# =========================

def build_largest_connected_component(thin_01):
    h, w = thin_01.shape
    n = h * w

    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)

    def idx(r, c):
        return r * w + c

    # 只看已扫描过的邻居，避免重复 union
    prev_neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

    for r in range(h):
        for c in range(w):
            if thin_01[r, c] == 0:
                continue
            cur = idx(r, c)
            for dr, dc in prev_neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and thin_01[nr, nc] == 1:
                    uf_union(parent, size, cur, idx(nr, nc))

    # 统计每个 root 的亮点数量
    comp_count = {}
    for r in range(h):
        for c in range(w):
            if thin_01[r, c] == 0:
                continue
            root = uf_find(parent, idx(r, c))
            comp_count[root] = comp_count.get(root, 0) + 1

    largest_cc_mask = np.zeros_like(thin_01, dtype=np.uint8)

    if len(comp_count) == 0:
        return largest_cc_mask

    largest_root = max(comp_count, key=comp_count.get)

    for r in range(h):
        for c in range(w):
            if thin_01[r, c] == 0:
                continue
            root = uf_find(parent, idx(r, c))
            if root == largest_root:
                largest_cc_mask[r, c] = 1

    return largest_cc_mask


# =========================
# 从最大连通分量做多源 BFS
# 将 binary_01 中所有亮点映射到 largest_cc_mask 上
#
# 只在 binary_01 == 1 的区域中传播
# 返回:
#   owner_r, owner_c: 每个亮点对应的骨架点坐标
#   dist_map: 到 owner 的 BFS 层数
# =========================

def build_projection_map_1(binary_01, largest_cc_mask):
    h, w = binary_01.shape

    owner_r = np.full((h, w), -1, dtype=np.int32)
    owner_c = np.full((h, w), -1, dtype=np.int32)
    dist_map = np.full((h, w), -1, dtype=np.int32)

    q = deque()

    # 多源初始化：所有最大连通分量点都是源
    src_points = np.argwhere(largest_cc_mask == 1)
    for r, c in src_points:
        owner_r[r, c] = r
        owner_c[r, c] = c
        dist_map[r, c] = 0
        q.append((r, c))

    # 在原始亮区 binary_01 内传播
    # 这里用 8 邻接，和骨架图一致
    dirs8 = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    while q:
        r, c = q.popleft()
        for dr, dc in dirs8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if binary_01[nr, nc] == 0:
                continue
            if dist_map[nr, nc] != -1:
                continue

            owner_r[nr, nc] = owner_r[r, c]
            owner_c[nr, nc] = owner_c[r, c]
            dist_map[nr, nc] = dist_map[r, c] + 1
            q.append((nr, nc))

    return owner_r, owner_c, dist_map

def build_projection_map(binary_01, largest_cc_mask):
    """
    用最小欧式距离做投影：
    对 binary_01 中每个亮点，找到 largest_cc_mask 上欧式距离最近的点。
    
    返回:
        owner_r, owner_c: 每个亮点对应的骨架点坐标
        dist_map: 对应的欧式距离（float）
    """
    h, w = binary_01.shape

    owner_r = np.full((h, w), -1, dtype=np.int32)
    owner_c = np.full((h, w), -1, dtype=np.int32)
    dist_map = np.full((h, w), np.inf, dtype=np.float32)

    cc_points = np.argwhere(largest_cc_mask == 1)   # [N,2], each row is (r,c)
    fg_points = np.argwhere(binary_01 == 1)         # [M,2]

    if len(cc_points) == 0 or len(fg_points) == 0:
        return owner_r, owner_c, dist_map

    # 逐个前景点找最近骨架点
    # 图只有 128x128，这样写最简单直接
    for r, c in fg_points:
        diff = cc_points - np.array([r, c], dtype=np.int32)   # [N,2]
        d2 = diff[:, 0].astype(np.float32) ** 2 + diff[:, 1].astype(np.float32) ** 2
        idx = int(np.argmin(d2))

        pr, pc = cc_points[idx]
        owner_r[r, c] = int(pr)
        owner_c[r, c] = int(pc)
        dist_map[r, c] = float(np.sqrt(d2[idx]))

    return owner_r, owner_c, dist_map

# =========================
# 任意亮点 p=(r,c) -> proj(p)
# 若该点不可映射，返回 None
# =========================

def project_point(r, c, binary_01, owner_r, owner_c):
    h, w = binary_01.shape
    if not (0 <= r < h and 0 <= c < w):
        return None
    if binary_01[r, c] == 0:
        return None
    pr = owner_r[r, c]
    pc = owner_c[r, c]
    if pr < 0 or pc < 0:
        return None
    return (int(pr), int(pc))


# =========================
# 在最大连通分量上做 BFS
# 计算 d(p,q) = dbfs(proj(p), proj(q))
# p,q 都是 binary_01 上的亮点坐标
# =========================

def bfs_distance_on_component(src, dst, largest_cc_mask):
    if src is None or dst is None:
        return None

    h, w = largest_cc_mask.shape
    sr, sc = src
    tr, tc = dst

    if largest_cc_mask[sr, sc] == 0 or largest_cc_mask[tr, tc] == 0:
        return None

    dist = np.full((h, w), -1, dtype=np.int32)
    q = deque()
    q.append((sr, sc))
    dist[sr, sc] = 0

    dirs8 = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    while q:
        r, c = q.popleft()
        if (r, c) == (tr, tc):
            return int(dist[r, c])

        for dr, dc in dirs8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if largest_cc_mask[nr, nc] == 0:
                continue
            if dist[nr, nc] != -1:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))

    return None


# =========================
# 构建底图:
# binary_01 白色
# largest_cc_mask 红色覆盖
# selected_proj 绿色高亮
# 输出 RGB [H,W,3], float in [0,1]
# =========================

def build_overlay_image(binary_01, largest_cc_mask, selected_proj=None):
    h, w = binary_01.shape
    canvas = np.zeros((h, w, 3), dtype=np.float32)

    # binary: 白色
    canvas[binary_01 == 1] = [1.0, 1.0, 1.0]

    # 最大连通分量: 红色覆盖
    red_mask = (largest_cc_mask == 1)
    canvas[red_mask] = [1.0, 0.0, 0.0]

    # 被点击点映射到的骨架节点: 绿色
    if selected_proj is not None:
        r, c = selected_proj
        if 0 <= r < h and 0 <= c < w:
            canvas[r, c] = [0.0, 1.0, 0.0]

    return canvas


# =========================
# 交互可视化
# 鼠标移动时显示当前像素
# 点击 binary_01 的亮点时，将其 proj 点变绿
# =========================

def interactive_visualization(binary_01, largest_cc_mask, owner_r, owner_c):
    fig, ax = plt.subplots(figsize=(8, 8))
    selected_proj = None

    img_show = build_overlay_image(binary_01, largest_cc_mask, selected_proj)
    im = ax.imshow(img_show, interpolation="nearest")
    ax.set_title("White: binary foreground | Red: largest thinned CC | Click white pixel to show proj in green")
    status_text = ax.text(
        0.01, 1.01, "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom"
    )

    def on_move(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            status_text.set_text("")
            fig.canvas.draw_idle()
            return

        c = int(round(event.xdata))
        r = int(round(event.ydata))

        h, w = binary_01.shape
        if not (0 <= r < h and 0 <= c < w):
            status_text.set_text("")
            fig.canvas.draw_idle()
            return

        if binary_01[r, c] == 1:
            proj = project_point(r, c, binary_01, owner_r, owner_c)
            status_text.set_text(f"hover pixel=(row={r}, col={c}), bright=1, proj={proj}")
        else:
            status_text.set_text(f"hover pixel=(row={r}, col={c}), bright=0")

        fig.canvas.draw_idle()

    def on_click(event):
        nonlocal selected_proj

        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        c = int(round(event.xdata))
        r = int(round(event.ydata))

        h, w = binary_01.shape
        if not (0 <= r < h and 0 <= c < w):
            return

        if binary_01[r, c] == 0:
            print(f"clicked pixel=(row={r}, col={c}) is not a bright pixel")
            return

        proj = project_point(r, c, binary_01, owner_r, owner_c)
        print(f"clicked pixel=(row={r}, col={c}), proj={proj}")

        selected_proj = proj
        new_img = build_overlay_image(binary_01, largest_cc_mask, selected_proj)
        im.set_data(new_img)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.tight_layout()
    plt.show()


# =========================
# main
# =========================

def main():
    # 直接改这里
    image_path = r"D:\code\fwwb\kaiwu\code\test_map.png"

    # 参数
    resize_hw = 128
    brightness_thresh = 128
    thinning_iter = 10

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    # 1) 中心裁剪
    cropped = center_crop_to_square(img)

    # 2) resize 到 128x128
    resized = cv2.resize(cropped, (resize_hw, resize_hw), interpolation=cv2.INTER_AREA)

    # 3) 灰度 + 二值化
    gray, binary = binarize_by_brightness(resized, thresh=brightness_thresh)
    binary_01 = (binary > 0).astype(np.uint8)

    # 4) 拓扑保持细化
    thinned = zhang_suen_thinning(binary, max_iter=thinning_iter)
    thin_01 = (thinned > 0).astype(np.uint8)

    # 5) 最大连通分量
    largest_cc_mask = build_largest_connected_component(thin_01)

    # 如果细化后空了，给个兜底
    if largest_cc_mask.sum() == 0:
        print("Warning: largest connected component on thinned image is empty. Fallback to binary foreground.")
        largest_cc_mask = build_largest_connected_component(binary_01)

    # 6) 原始亮点 -> 主骨架点 映射
    owner_r, owner_c, dist_map = build_projection_map(binary_01, largest_cc_mask)

    # 7) 保存一些中间结果，便于你看
    #cv2.imwrite("output_1_resized.png", resized)
    #cv2.imwrite("output_2_gray.png", gray)
    #cv2.imwrite("output_3_binary.png", binary)
    #cv2.imwrite("output_4_thinned.png", thinned)
    #cv2.imwrite("output_5_largest_cc.png", (largest_cc_mask * 255).astype(np.uint8))

    # 8) 交互可视化
    interactive_visualization(binary_01, largest_cc_mask, owner_r, owner_c)

    #(79,85) #(78,98)
    proj_p = project_point(65,83, binary_01, owner_r, owner_c)
    proj_q = project_point(60,86, binary_01, owner_r, owner_c)
    d = bfs_distance_on_component(proj_p, proj_q, largest_cc_mask)
    print(d)
if __name__ == "__main__":
    main()