"""Hash ảnh để phát hiện ảnh trùng — chỉ dùng Pillow + NumPy.

- md5: trùng tuyệt đối từng byte.
- dHash 64-bit: trùng "gần" (ảnh giống nhau nhưng khác kích thước/nén/chỉnh sáng nhẹ).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def dhash(img: Image.Image, size: int = 8) -> int:
    """Difference hash: so sánh độ sáng các pixel kề nhau trên ảnh thu nhỏ 9x8."""
    small = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(small, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):          # NumPy >= 2.0
        return np.bitwise_count(x)
    # fallback cho NumPy cũ
    x = x.astype(np.uint64)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def near_duplicate_pairs(hashes: list[int], max_distance: int) -> list[tuple[int, int, int]]:
    """Trả về các cặp (i, j, khoảng_cách_Hamming) với khoảng cách <= max_distance."""
    h = np.array(hashes, dtype=np.uint64)
    pairs = []
    for i in range(len(h) - 1):
        d = _popcount(np.bitwise_xor(h[i + 1:], h[i]))
        for k in np.nonzero(d <= max_distance)[0]:
            j = i + 1 + int(k)
            pairs.append((i, j, int(d[k])))
    return pairs


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)
