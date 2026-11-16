"""Đọc file cấu hình YAML, hỗ trợ kế thừa qua khóa `base:`.

Ví dụ configs/resnet50_gray.yaml:
    base: base.yaml
    data:
      color_mode: gray
→ lấy toàn bộ base.yaml rồi ghi đè data.color_mode. Tên thí nghiệm = tên file.
"""
from __future__ import annotations

import copy
from pathlib import Path


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    import yaml

    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    own_name = cfg.pop("name", None)
    base = cfg.pop("base", None)
    if base:
        parent = load_config(path.parent / base)
        parent.pop("name", None)
        cfg = deep_merge(parent, cfg)
    cfg["name"] = own_name or path.stem
    return cfg
