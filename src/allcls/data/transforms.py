"""Tiền xử lý + augmentation, điều khiển hoàn toàn bằng config.

Các tùy chọn phục vụ thí nghiệm kiểm tra "lối tắt màu":
  data.color_mode  : rgb | gray        (gray = bỏ thông tin màu)
  data.stain_norm  : none | reinhard   (chuẩn hóa màu nhuộm)
  augment.color_jitter : độ mạnh nhiễu màu khi huấn luyện (0 = tắt)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _rgb_to_lab(img: Image.Image) -> np.ndarray:
    import cv2
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)


def lab_stats(img: Image.Image):
    lab = _rgb_to_lab(img).reshape(-1, 3)
    return lab.mean(0), lab.std(0)


class ReinhardNormalizer:
    """Chuẩn hóa màu nhuộm (Reinhard et al., 2001): đưa trung bình và độ lệch chuẩn
    của từng kênh LAB về giá trị đích chung, tính từ tập TRAIN.
    """

    def __init__(self, target_mean, target_std):
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)

    def __call__(self, img: Image.Image) -> Image.Image:
        import cv2
        lab = _rgb_to_lab(img)
        flat = lab.reshape(-1, 3)
        mean, std = flat.mean(0), flat.std(0) + 1e-6
        lab = (lab - mean) / std * self.target_std + self.target_mean
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

    def to_dict(self) -> dict:
        return {"method": "reinhard", "target_mean": self.target_mean.tolist(),
                "target_std": self.target_std.tolist()}


def fit_reinhard(images) -> ReinhardNormalizer:
    """Giá trị đích = trung bình mean/std LAB của các ảnh (chỉ dùng ảnh train)."""
    means, stds = zip(*(lab_stats(im) for im in images))
    return ReinhardNormalizer(np.mean(means, axis=0), np.mean(stds, axis=0))


class RandomRotate90:
    """Xoay ngẫu nhiên 0/90/180/270 độ — ảnh hiển vi không có hướng "đúng"."""

    _OPS = (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)

    def __call__(self, img: Image.Image) -> Image.Image:
        import torch
        op = self._OPS[int(torch.randint(0, 4, (1,)))]
        return img if op is None else img.transpose(op)


def build_transforms(cfg: dict, train: bool, stain_normalizer=None):
    from torchvision import transforms as T

    d, a = cfg["data"], cfg.get("augment", {})
    size = int(d.get("image_size", 224))
    gray = d.get("color_mode", "rgb") == "gray"
    ops = []
    if stain_normalizer is not None:
        ops.append(stain_normalizer)
    if gray:
        ops.append(T.Grayscale(num_output_channels=3))   # 3 kênh để dùng được trọng số ImageNet
    ops.append(T.Resize((size, size)))
    if train:
        if a.get("hflip", True):
            ops.append(T.RandomHorizontalFlip())
        if a.get("vflip", True):
            ops.append(T.RandomVerticalFlip())
        if a.get("rotate90", True):
            ops.append(RandomRotate90())
        cj = float(a.get("color_jitter", 0.0))
        if cj > 0:
            ops.append(T.ColorJitter(brightness=cj, contrast=cj,
                                     saturation=0.0 if gray else cj,
                                     hue=0.0 if gray else min(cj / 4, 0.5)))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)
