"""Dataset đọc từ data/splits/{train,val,test}.csv (tạo bởi split.py)."""
from __future__ import annotations

import csv
import io
from pathlib import Path

from PIL import Image

try:
    from torch.utils.data import Dataset
except ImportError:          # cho phép import module khi chưa cài torch
    Dataset = object


class SplitDataset(Dataset):
    """Trả về (tensor ảnh, nhãn, chỉ số dòng) — chỉ số dùng để ghép dự đoán với tên file.

    cache_in_memory=True: đọc trước toàn bộ file vào RAM (~60 MB cho bộ này),
    tránh đọc chậm từ Google Drive trong mỗi epoch.
    """

    def __init__(self, csv_path, data_root, transform=None, cache_in_memory=False):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.rel_paths = [r["rel_path"] for r in rows]
        self.labels = [int(r["label_idx"]) for r in rows]
        self.paths = [Path(data_root) / p for p in self.rel_paths]
        self.transform = transform
        self._cache = [p.read_bytes() for p in self.paths] if cache_in_memory else None

    def __len__(self):
        return len(self.paths)

    def load_image(self, i: int) -> Image.Image:
        data = self._cache[i] if self._cache is not None else self.paths[i].read_bytes()
        return Image.open(io.BytesIO(data)).convert("RGB")

    def __getitem__(self, i: int):
        img = self.load_image(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], i
