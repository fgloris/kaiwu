from collections import deque
import numpy as np

MAP_SIZE = 128.0
MAP_SIZE_INT = 128

def clip_window(x0, x1, y0, y1, size=MAP_SIZE_INT):
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(size, x1)
    y1 = min(size, y1)
    return x0, x1, y0, y1


def _neighbors(x, y, img01):
    # P2, P3, ..., P9
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


def zhang_suen_thinning(binary01, max_iter=None):
    """
    输入: 0/1 二值图
    输出: 0/1 二值图
    作用: 在尽量保持连通性的前提下，对亮区域做“拓扑保持削薄”
    """
    img = (binary01 > 0).astype(np.uint8).copy()
    h, w = img.shape

    if h < 3 or w < 3:
        return img

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

    return img


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


def build_largest_connected_component(thin_01):
    """
    在 thin_01 上做 8 邻接连通分量，只保留最大连通分量。
    输入: 0/1
    输出: 0/1
    """
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


def bfs_distance_on_component(src, dst, component_mask):
    """
    在 component_mask(0/1) 上做 8 邻接 BFS。
    返回 src -> dst 的步长；若不可达则返回 None。
    """
    h, w = component_mask.shape
    sx, sy = src
    tx, ty = dst

    if not (0 <= sx < h and 0 <= sy < w and 0 <= tx < h and 0 <= ty < w):
        return None
    if component_mask[sx, sy] == 0 or component_mask[tx, ty] == 0:
        return None

    dist = np.full((h, w), -1, dtype=np.int32)
    q = deque()
    q.append((sx, sy))
    dist[sx, sy] = 0

    dirs8 = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    while q:
        x, y = q.popleft()
        if (x, y) == (tx, ty):
            return int(dist[x, y])

        for dx, dy in dirs8:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < h and 0 <= ny < w):
                continue
            if component_mask[nx, ny] == 0:
                continue
            if dist[nx, ny] != -1:
                continue
            dist[nx, ny] = dist[x, y] + 1
            q.append((nx, ny))

    return None