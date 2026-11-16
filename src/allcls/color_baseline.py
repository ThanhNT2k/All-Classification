"""Baseline "CHỈ DÙNG MÀU": phân loại bằng histogram màu, không nhìn hình dạng tế bào.

Mục đích: đo xem riêng thông tin màu đã phân biệt được các lớp đến mức nào.
Nếu baseline này đạt kết quả cao (đặc biệt lớp Pro), mô hình CNN có thể đang
dựa vào màu nhuộm thay vì hình thái — cần đối chiếu với thí nghiệm ảnh xám.

Bước 1 (chỉ cần Pillow + NumPy): trích đặc trưng -> file .npz
Bước 2 (cần scikit-learn): hồi quy logistic, đánh giá bằng evaluate.compute_metrics
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from .evaluate import compute_metrics, metrics_markdown, save_predictions, write_json

FEATURE_NAMES = ([f"H{i}" for i in range(18)] + [f"S{i}" for i in range(16)]
                 + [f"V{i}" for i in range(16)]
                 + ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"])


def color_features(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3) / 255
    hsv = np.asarray(img.convert("RGB").convert("HSV")).reshape(-1, 3)
    h = np.histogram(hsv[:, 0], bins=18, range=(0, 256))[0]
    s = np.histogram(hsv[:, 1], bins=16, range=(0, 256))[0]
    v = np.histogram(hsv[:, 2], bins=16, range=(0, 256))[0]
    n = len(hsv)
    return np.concatenate([h / n, s / n, v / n, rgb.mean(0), rgb.std(0)]).astype(np.float32)


def extract(data_root, splits_dir, out_npz, verbose=True):
    data = {}
    for split in ("train", "val", "test"):
        with open(Path(splits_dir) / f"{split}.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        X = np.stack([color_features(Image.open(Path(data_root) / r["rel_path"])) for r in rows])
        data[f"X_{split}"] = X
        data[f"y_{split}"] = np.array([int(r["label_idx"]) for r in rows])
        data[f"paths_{split}"] = np.array([r["rel_path"] for r in rows])
        if verbose:
            print(f"  {split}: {X.shape}")
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **data)
    return out_npz


FEATURE_GROUPS = {
    "Trung bình + độ lệch RGB (6)": [n for n in FEATURE_NAMES if "_" in n],
    "Histogram sắc độ H (18)": [n for n in FEATURE_NAMES if n.startswith("H")],
    "Histogram độ bão hòa S (16)": [n for n in FEATURE_NAMES if n.startswith("S")],
    "Histogram độ sáng V (16) ≈ ảnh xám": [n for n in FEATURE_NAMES if n.startswith("V")],
    "Tất cả (56)": FEATURE_NAMES,
}


def _classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"))


def ablation(d) -> str:
    """Bảng: từng nhóm đặc trưng màu riêng lẻ đạt bao nhiêu trên tập test."""
    from . import CLASSES
    L = ["## Từng nhóm đặc trưng màu (tập test)", "",
         "| Đặc trưng | Accuracy | Macro F1 | " + " | ".join(f"Recall {c}" for c in CLASSES) + " |",
         "|---|---:|---:|" + "---:|" * len(CLASSES)]
    for name, cols in FEATURE_GROUPS.items():
        idx = [FEATURE_NAMES.index(c) for c in cols]
        clf = _classifier().fit(d["X_train"][:, idx], d["y_train"])
        m = compute_metrics(d["y_test"], clf.predict(d["X_test"][:, idx]), n_boot=0)
        L.append(f"| {name} | {m['accuracy']:.1%} | {m['macro_f1']:.1%} | "
                 + " | ".join(f"{m['per_class'][c]['recall']:.0%}" for c in CLASSES) + " |")
    return "\n".join(L) + "\n"


def fit_eval(npz_path, out_dir, n_boot=1000):
    d = np.load(npz_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clf = _classifier().fit(d["X_train"], d["y_train"])
    result, md = {}, ["# Baseline chỉ dùng màu (histogram HSV + thống kê RGB, hồi quy logistic)", ""]
    for split in ("val", "test"):
        probs = clf.predict_proba(d[f"X_{split}"])
        pred = probs.argmax(1)
        save_predictions(out / f"predictions_{split}.csv", d[f"paths_{split}"].tolist(),
                         d[f"y_{split}"], pred, probs)
        m = compute_metrics(d[f"y_{split}"], pred, probs, n_boot=n_boot)
        result[split] = m
        md.append(metrics_markdown(m, split.upper()))
    md.append(ablation(d))
    n_params = int(clf[-1].coef_.size + clf[-1].intercept_.size)
    result["efficiency"] = {"params": n_params}
    write_json(result, out / "metrics.json")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "DONE").write_text("color_baseline", encoding="utf-8")
    return result
