"""Sinh bộ dữ liệu GIẢ LẬP để kiểm thử audit/split (không phải dữ liệu thật).

Cố tình cài các tình huống khó: 2 phiên bản ảnh, ảnh trùng tuyệt đối,
ảnh gần trùng (thu nhỏ + chỉnh sáng), ảnh trùng nhưng khác nhãn, file hỏng,
thư mục không rõ lớp.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

CLASSES = ["Benign", "Early", "Pre", "Pro"]


def _random_image(rng, size=96):
    a = rng.integers(0, 255, (size // 8, size // 8, 3), dtype=np.uint8)
    return Image.fromarray(a).resize((size, size), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(1))


def make(root: Path, per_class=(12, 20, 18, 15), seed=0) -> dict:
    rng = np.random.default_rng(seed)
    expected = {"exact_pairs": 0, "near_pairs": 0, "cross_class": 0}
    for cls, n in zip(CLASSES, per_class):
        for k in range(n):
            img = _random_image(rng)
            for variant in ("Original", "Segmented"):
                d = root / variant / cls
                d.mkdir(parents=True, exist_ok=True)
                out = img if variant == "Original" else img.convert("L").convert("RGB")
                out.save(d / f"{cls}_{k:03d}.png")
        # trùng tuyệt đối: copy y nguyên file
        src = root / "Original" / cls / f"{cls}_000.png"
        (root / "Original" / cls / f"{cls}_000_copy.png").write_bytes(src.read_bytes())
        expected["exact_pairs"] += 1
        # gần trùng: thu nhỏ + tăng sáng nhẹ
        im = Image.open(root / "Original" / cls / f"{cls}_001.png")
        im = ImageEnhance.Brightness(im.resize((80, 80))).enhance(1.05)
        im.save(root / "Original" / cls / f"{cls}_001_resized.jpg", quality=92)
        expected["near_pairs"] += 1
    # trùng nhưng khác nhãn: ảnh Pre_002 bị copy nhầm sang thư mục Pro
    (root / "Original" / "Pro" / "Pre_002_mislabeled.png").write_bytes(
        (root / "Original" / "Pre" / "Pre_002.png").read_bytes())
    expected["cross_class"] = 1
    # file hỏng + thư mục không rõ lớp
    (root / "Original" / "Early" / "broken.jpg").write_bytes(b"not an image")
    (root / "Original" / "misc").mkdir(exist_ok=True)
    _random_image(rng).save(root / "Original" / "misc" / "unknown.png")
    return expected
