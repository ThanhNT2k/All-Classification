"""Tìm ảnh trong thư mục dữ liệu và xác định nhãn — không giả định cấu trúc cố định.

Nhãn lấy từ tên thư mục cha gần nhất khớp một trong 4 lớp (không phân biệt hoa thường).
Nếu dưới thư mục gốc có nhiều "phiên bản" (ví dụ Original/ và Segmented/),
tên thư mục cấp 1 được ghi lại ở trường `variant`.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Thứ tự quan trọng: "early pre-b" phải ra Early, không ra Pre
_CLASS_PATTERNS = [
    ("Benign", re.compile(r"benign|hematogone", re.I)),
    ("Early", re.compile(r"early", re.I)),
    ("Pro", re.compile(r"(^|[^a-z])pro($|[^a-z])|pro-?b", re.I)),
    ("Pre", re.compile(r"(^|[^a-z])pre($|[^a-z])|pre-?b", re.I)),
]


def class_from_name(name: str) -> str | None:
    for label, pattern in _CLASS_PATTERNS:
        if pattern.search(name):
            return label
    return None


@dataclass
class ImageRecord:
    path: Path          # đường dẫn tuyệt đối
    rel_path: str       # đường dẫn tương đối so với data_root, dạng posix
    label: str | None   # None = không xác định được lớp
    variant: str        # thư mục cấp 1 (hoặc "" nếu ảnh nằm thẳng trong thư mục lớp)
    stem: str           # tên file không đuôi, dùng để ghép cùng ảnh giữa các variant


def discover(data_root: str | Path) -> list[ImageRecord]:
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {root}")
    records = []
    # os.walk thay cho rglob + is_file: không stat từng file (rất chậm trên OneDrive/ổ mạng)
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in filenames:
            if Path(name).suffix.lower() in IMAGE_EXTS:
                paths.append(Path(dirpath) / name)
    for p in sorted(paths):
        rel_parts = p.relative_to(root).parts
        label = None
        for folder in reversed(rel_parts[:-1]):
            label = class_from_name(folder)
            if label:
                break
        first = rel_parts[0] if len(rel_parts) > 1 else ""
        variant = "" if class_from_name(first) else first
        records.append(ImageRecord(
            path=p,
            rel_path=p.relative_to(root).as_posix(),
            label=label,
            variant=variant,
            stem=p.stem,
        ))
    return records
