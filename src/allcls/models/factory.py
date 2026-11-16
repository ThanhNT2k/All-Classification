"""Tạo mô hình theo config: "simple_cnn" = CNN tự xây dựng, tên khác = mô hình của thư viện timm."""
from __future__ import annotations


def create_model(cfg: dict, num_classes: int):
    m = cfg["model"]
    if m["name"] == "simple_cnn":
        from .simple_cnn import SimpleCNN
        return SimpleCNN(num_classes, drop_rate=float(m.get("drop_rate", 0.3)))
    import timm
    return timm.create_model(m["name"], pretrained=bool(m.get("pretrained", True)),
                             num_classes=num_classes, drop_rate=float(m.get("drop_rate", 0.0)))
