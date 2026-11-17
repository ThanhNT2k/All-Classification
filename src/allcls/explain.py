"""Bước 5 — Giải thích mô hình: Grad-CAM, phân tích ảnh sai, kiểm tra mô hình nhìn vào đâu.

Với mỗi lần chạy (thư mục có config.yaml + best.pt), tạo <run>/explain/:
  errors_gradcam_*.png   mọi ảnh test bị sai: ảnh gốc | Grad-CAM lớp dự đoán | Grad-CAM lớp thật
  correct_gradcam.png    ảnh đúng mỗi lớp (tự tin nhất và kém tự tin nhất) kèm Grad-CAM
  mask_check.png         mặt nạ tế bào ước lượng — XEM BẰNG MẮT trước khi tin các con số bên dưới
  cells_erased_check.png ví dụ ảnh sau khi xóa tế bào
  background_erased_check.png  ví dụ ảnh sau khi xóa nền (chỉ còn bạch cầu)
  per_image.csv          số liệu từng ảnh
  summary.md / .json     bảng tổng hợp

Ba phép đo định lượng (ngoài hình Grad-CAM):
  1. Grad-CAM trong tế bào: bao nhiêu phần "sự chú ý" rơi vào vùng tế bào nhuộm màu, so với
     diện tích vùng đó (tỷ số > 1 = tập trung vào tế bào hơn mức ngẫu nhiên).
  2. Phép thử xóa vùng: che 20% ô lưới Grad-CAM cao nhất bằng màu nền, so với che 20% ô ngẫu
     nhiên. Xác suất giảm mạnh hơn hẳn = Grad-CAM chỉ đúng vùng mô hình thực sự dùng.
  3. Xóa tế bào: tô toàn bộ tế bào nhuộm màu bằng màu nền rồi dự đoán lại. Nếu mô hình vẫn
     đoán đúng thì nó không cần hình thái tế bào (dựa vào nền / tông màu nhuộm = lối tắt).
  4. Xóa nền: giữ bạch cầu, tô phần còn lại bằng màu nền phẳng — chỉ tế bào có đủ để đoán đúng không.
  5. Ảnh trắng đen: đổi ảnh test sang trắng đen rồi dự đoán — mô hình phụ thuộc màu đến đâu.

Grad-CAM (Selvaraju et al., 2017) tự cài đặt, lấy ở bản đồ đặc trưng không gian cuối cùng
(model.forward_features): trọng số kênh = trung bình gradient của logit theo kênh đó,
CAM = ReLU(tổng có trọng số các kênh), chuẩn hóa về [0, 1] theo từng ảnh.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from . import CLASSES
from .evaluate import load_predictions, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ERASE_FRACTION = 0.2        # phép thử xóa vùng: tỷ lệ ô lưới bị che
SHORTCUT_FLAG = 0.5         # xóa tế bào mà vẫn đúng >= 50% (ngẫu nhiên 4 lớp = 25%) -> cảnh báo


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---------- Grad-CAM ----------

def grad_cam(model, x, targets=None):
    """x: (B,3,H,W) đã chuẩn hóa. targets: lớp cần giải thích (mặc định = lớp dự đoán).
    Trả về (cam (B,h,w) trong [0,1] ở độ phân giải bản đồ đặc trưng, xác suất (B,C))."""
    import torch

    model.eval()
    with torch.enable_grad():
        x = x.clone().requires_grad_(True)
        feats = model.forward_features(x)
        if feats.ndim != 4:
            raise ValueError("Grad-CAM cần bản đồ đặc trưng 4 chiều (mô hình CNN)")
        logits = model.forward_head(feats)
        if targets is None:
            targets = logits.argmax(1)
        score = logits.gather(1, targets.view(-1, 1)).sum()
        grads = torch.autograd.grad(score, feats)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats.detach()).sum(1))
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return cam.cpu().numpy().astype(np.float32), torch.softmax(logits.detach().float(), 1).cpu().numpy()


def upsample(cam: np.ndarray, size: int) -> np.ndarray:
    import cv2
    return np.clip(cv2.resize(cam, (size, size), interpolation=cv2.INTER_LINEAR), 0, 1)


def overlay(rgb: np.ndarray, cam_up: np.ndarray) -> np.ndarray:
    """Phủ bản đồ nhiệt lên ảnh; vùng CAM thấp gần như giữ nguyên ảnh gốc."""
    import matplotlib
    heat = matplotlib.colormaps["jet"](cam_up)[..., :3]
    a = (0.65 * cam_up)[..., None]
    return np.clip((1 - a) * rgb / 255.0 + a * heat, 0, 1)


# ---------- mặt nạ tế bào (ước lượng) ----------

def cell_mask(rgb: np.ndarray):
    """Vùng bạch cầu nhuộm tím/hồng tím: kênh a* của LAB (trục lục–đỏ tím). Hồng cầu trong bộ
    này có màu xám xanh (a* gần 0) nên không lọt vào. Ngưỡng Otsu riêng từng ảnh nên thích
    nghi được với lớp Pro nhuộm nhạt hơn. Trả về mảng bool, hoặc None nếu không tách được.
    Chỉ là ước lượng — kiểm tra bằng mask_check.png."""
    import cv2

    a = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 1].astype(np.int16) - 128
    score = cv2.GaussianBlur(np.clip(a * 3, 0, 255).astype(np.uint8), (5, 5), 0)
    thr, m = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_area = max(8, int(0.001 * m.size))
    keep = np.zeros(m.shape, bool)
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] >= min_area:
            keep |= lab == j
    frac = keep.mean()
    if thr < 18 or not 0.002 <= frac <= 0.6:          # thr 18 = a* 6: dưới mức này chỉ là nhiễu
        return None
    return keep


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def background_color(rgb: np.ndarray, mask) -> np.ndarray:
    px = rgb[~mask] if mask is not None and (~mask).any() else rgb.reshape(-1, 3)
    return np.median(px, axis=0).astype(np.uint8)


def erase(rgb: np.ndarray, region: np.ndarray, color: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[region] = color
    return out


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Ảnh trắng đen (độ sáng), vẫn 3 kênh để đưa vào mô hình."""
    import cv2
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.stack([g, g, g], axis=-1)


def grid_region(cells: np.ndarray, grid_hw, size: int) -> np.ndarray:
    """Các ô lưới (chỉ số phẳng) -> mặt nạ bool kích thước ảnh."""
    import cv2
    g = np.zeros(grid_hw[0] * grid_hw[1], np.uint8)
    g[cells] = 1
    return cv2.resize(g.reshape(grid_hw), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


# ---------- nạp mô hình ----------

def load_run(run_dir, device):
    import torch
    import yaml

    from .data.transforms import ReinhardNormalizer, build_transforms
    from .models.factory import create_model

    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    cfg["model"]["pretrained"] = False               # trọng số lấy từ best.pt, không tải ImageNet
    model = create_model(cfg, len(CLASSES))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.to(device).eval()
    stain = None
    if cfg["data"].get("stain_norm", "none") == "reinhard":
        t = json.loads((run_dir / "stain_target.json").read_text(encoding="utf-8"))
        stain = ReinhardNormalizer(t["target_mean"], t["target_std"])
    return cfg, model, build_transforms(cfg, train=False, stain_normalizer=stain)


def _batches(n, bs):
    for s in range(0, n, bs):
        yield list(range(s, min(n, s + bs)))


def _probs(model, tf, images, device, bs):
    """Xác suất cho danh sách ảnh RGB (numpy)."""
    import torch
    from PIL import Image
    out = []
    with torch.no_grad():
        for b in _batches(len(images), bs):
            x = torch.stack([tf(Image.fromarray(images[i])) for i in b]).to(device)
            out.append(torch.softmax(model(x).float(), 1).cpu().numpy())
    return np.concatenate(out)


# ---------- phân tích một lần chạy ----------

def explain_run(run_dir, split: str = "test", data_root=None, splits_dir=None, out_dir=None,
                max_errors: int = 48, per_class: int = 3, batch_size: int = 32, seed: int = 0,
                device=None) -> dict:
    import torch
    from PIL import Image

    from .data.dataset import SplitDataset

    run_dir = Path(run_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg, model, tf = load_run(run_dir, device)
    d = cfg["data"]
    size = int(d.get("image_size", 224))
    ds = SplitDataset(_resolve(splits_dir or d["splits"]) / f"{split}.csv", _resolve(data_root or d["root"]))
    n = len(ds)
    labels = np.asarray(ds.labels)
    images = [np.asarray(ds.load_image(i).resize((size, size), Image.Resampling.BILINEAR)) for i in range(n)]
    out = Path(out_dir) if out_dir else run_dir / "explain"
    out.mkdir(parents=True, exist_ok=True)

    # 1) dự đoán + Grad-CAM cho lớp dự đoán và lớp thật
    cams_pred, cams_true, probs = [], [], []
    for b in _batches(n, batch_size):
        x = torch.stack([tf(Image.fromarray(images[i])) for i in b]).to(device)
        c, p = grad_cam(model, x)
        ct, _ = grad_cam(model, x, torch.as_tensor(labels[b], device=device))
        cams_pred.append(c), cams_true.append(ct), probs.append(p)
    cams_pred, cams_true, probs = map(np.concatenate, (cams_pred, cams_true, probs))
    pred = probs.argmax(1)
    correct = pred == labels
    grid_hw = cams_pred.shape[1:]

    # đối chiếu với dự đoán đã lưu lúc huấn luyện: khác nhiều = nạp sai trọng số / tiền xử lý
    saved_agree = None
    pf = run_dir / f"predictions_{split}.csv"
    if pf.exists():
        rel, _, yp, _ = load_predictions(pf)
        saved = dict(zip(rel, yp))
        common = [i for i in range(n) if ds.rel_paths[i] in saved]
        if common:
            saved_agree = [int(sum(saved[ds.rel_paths[i]] == pred[i] for i in common)), len(common)]

    # 2) mặt nạ tế bào + các phép thử che/xóa
    rng = np.random.default_rng(seed)
    k = max(1, math.ceil(ERASE_FRACTION * grid_hw[0] * grid_hw[1]))
    masks, top_imgs, rand_imgs, cell_imgs, bgless_imgs = [], [], [], [], []
    for i in range(n):
        m = cell_mask(images[i])
        masks.append(m)
        bg = background_color(images[i], None if m is None else dilate(m, 4))
        flat = cams_pred[i].ravel()
        top_imgs.append(erase(images[i], grid_region(np.argsort(-flat)[:k], grid_hw, size), bg))
        rand_imgs.append(erase(images[i], grid_region(rng.choice(flat.size, k, replace=False), grid_hw, size), bg))
        cell_imgs.append(images[i] if m is None else erase(images[i], dilate(m, 4), bg))
        bgless_imgs.append(images[i] if m is None else erase(images[i], ~dilate(m, 4), bg))
    p_top = _probs(model, tf, top_imgs, device, batch_size)
    p_rand = _probs(model, tf, rand_imgs, device, batch_size)
    p_cell = _probs(model, tf, cell_imgs, device, batch_size)
    p_bgless = _probs(model, tf, bgless_imgs, device, batch_size)
    p_gray = _probs(model, tf, [to_gray(im) for im in images], device, batch_size)

    rows = []
    ar = np.arange(n)
    for i in range(n):
        m = masks[i]
        r = {"rel_path": ds.rel_paths[i], "label": CLASSES[labels[i]], "pred": CLASSES[pred[i]],
             "correct": int(correct[i]), "p_pred": float(probs[i, pred[i]]), "p_true": float(probs[i, labels[i]]),
             "true_rank": int((probs[i] > probs[i, labels[i]]).sum()) + 1,
             "cam_empty": int(cams_pred[i].max() <= 0),
             "mask_ok": int(m is not None), "cell_area": None, "cam_in_cell": None, "peak_in_cell": None,
             "p_pred_erase_top": float(p_top[i, pred[i]]), "p_pred_erase_random": float(p_rand[i, pred[i]]),
             "flip_erase_top": int(p_top[i].argmax() != pred[i]),
             "flip_erase_random": int(p_rand[i].argmax() != pred[i]),
             "pred_cells_erased": None, "p_true_cells_erased": None,
             "pred_background_erased": None, "p_true_background_erased": None,
             "pred_gray": CLASSES[p_gray[i].argmax()], "p_true_gray": float(p_gray[i, labels[i]])}
        if m is not None:
            md = dilate(m, 4)
            cu = upsample(cams_pred[i], size)
            r["cell_area"] = float(md.mean())
            if cu.sum() > 0:
                r["cam_in_cell"] = float(cu[md].sum() / cu.sum())
                r["peak_in_cell"] = int(md[np.unravel_index(cu.argmax(), cu.shape)])
            r["pred_cells_erased"] = CLASSES[p_cell[i].argmax()]
            r["p_true_cells_erased"] = float(p_cell[i, labels[i]])
            r["pred_background_erased"] = CLASSES[p_bgless[i].argmax()]
            r["p_true_background_erased"] = float(p_bgless[i, labels[i]])
        rows.append(r)
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 3) hình
    conf = probs[ar, pred]
    err_idx = [int(i) for i in np.argsort(-conf) if not correct[i]]          # sai mà tự tin nhất trước
    err_figs = _plot_errors(out, images, cams_pred, cams_true, rows, err_idx[:max_errors], size, cfg["name"])
    picks = _pick_correct(labels, correct, conf, per_class)
    _plot_correct(out / "correct_gradcam.png", images, cams_pred, rows, picks, size, cfg["name"])
    show = [i for c in range(len(CLASSES)) for i in np.flatnonzero(labels == c)[:6]]
    _plot_masks(out / "mask_check.png", images, masks, rows, show, len(CLASSES), "Mặt nạ tế bào ước lượng (viền vàng)")
    _plot_masks(out / "cells_erased_check.png", cell_imgs, [None] * n, rows, show, len(CLASSES),
                "Ảnh sau khi xóa tế bào (đầu vào của phép thử 3)", use_erased="pred_cells_erased")
    _plot_masks(out / "background_erased_check.png", bgless_imgs, [None] * n, rows, show, len(CLASSES),
                "Ảnh sau khi xóa nền, chỉ còn bạch cầu (đầu vào của phép thử 4)", use_erased="pred_background_erased")

    summary = _summarize(rows, cfg["name"], run_dir.name, split, saved_agree, grid_hw, len(err_idx), err_figs)
    write_json(summary, out / "summary.json")
    (out / "summary.md").write_text(_summary_md(summary, rows, err_idx), encoding="utf-8")
    return {"name": cfg["name"], "run_dir": run_dir, "out": out, "summary": summary, "rows": rows,
            "cams_pred": cams_pred, "probs": probs, "pred": pred, "labels": labels,
            "rel_paths": ds.rel_paths, "images": images, "size": size}


def _pick_correct(labels, correct, conf, per_class):
    """Mỗi lớp: per_class ảnh đúng tự tin nhất + per_class ảnh đúng kém tự tin nhất."""
    picks = {}
    for c in range(len(CLASSES)):
        idx = np.flatnonzero((labels == c) & correct)
        idx = idx[np.argsort(-conf[idx])]
        hi = list(idx[:per_class])
        lo = [i for i in idx[::-1][:per_class] if i not in hi]
        picks[c] = [int(i) for i in hi + lo[::-1]]
    return picks


# ---------- tổng hợp số liệu ----------

def _mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def _group_stats(rs):
    ok = [r for r in rs if r["mask_ok"] and r["cam_in_cell"] is not None]
    area, inside = _mean([r["cell_area"] for r in ok]), _mean([r["cam_in_cell"] for r in ok])
    er = [r for r in rs if r["mask_ok"]]
    return {
        "n": len(rs), "n_mask": len(ok),
        "cell_area": area, "cam_in_cell": inside,
        "focus_ratio": inside / area if area and inside is not None else None,
        "peak_in_cell": _mean([r["peak_in_cell"] for r in ok]),
        "p_pred": _mean([r["p_pred"] for r in rs]),
        "p_pred_erase_top": _mean([r["p_pred_erase_top"] for r in rs]),
        "p_pred_erase_random": _mean([r["p_pred_erase_random"] for r in rs]),
        "flip_erase_top": _mean([r["flip_erase_top"] for r in rs]),
        "flip_erase_random": _mean([r["flip_erase_random"] for r in rs]),
        "accuracy": _mean([r["correct"] for r in rs]),
        "n_erased": len(er),
        "accuracy_before_erase": _mean([r["correct"] for r in er]),
        "accuracy_cells_erased": _mean([int(r["pred_cells_erased"] == r["label"]) for r in er]),
        "p_true_before_erase": _mean([r["p_true"] for r in er]),
        "p_true_cells_erased": _mean([r["p_true_cells_erased"] for r in er]),
        "accuracy_background_erased": _mean([int(r["pred_background_erased"] == r["label"]) for r in er]),
        "p_true_background_erased": _mean([r["p_true_background_erased"] for r in er]),
        "accuracy_gray": _mean([int(r["pred_gray"] == r["label"]) for r in rs]),
        "p_true_gray": _mean([r["p_true_gray"] for r in rs]),
        "p_true": _mean([r["p_true"] for r in rs]),
    }


def _summarize(rows, name, run_name, split, saved_agree, grid_hw, n_err, err_figs):
    groups = {"all": rows, "correct": [r for r in rows if r["correct"]],
              "wrong": [r for r in rows if not r["correct"]]}
    for c in CLASSES:
        groups[c] = [r for r in rows if r["label"] == c]
    confusions = {}
    for r in rows:
        if not r["correct"]:
            key = f"{r['label']} → {r['pred']}"
            confusions[key] = confusions.get(key, 0) + 1
    wrong = groups["wrong"]
    erased = [r["pred_cells_erased"] for r in rows if r["mask_ok"]]
    return {
        "name": name, "run": run_name, "split": split, "n": len(rows), "n_errors": n_err,
        "cam_grid": list(grid_hw), "saved_predictions_agree": saved_agree,
        "n_mask_failed": sum(1 - r["mask_ok"] for r in rows),
        "n_cam_empty": sum(r["cam_empty"] for r in rows),
        "confusions": dict(sorted(confusions.items(), key=lambda kv: -kv[1])),
        "errors_high_confidence": sum(r["p_pred"] >= 0.9 for r in wrong),
        "errors_true_is_second": sum(r["true_rank"] == 2 for r in wrong),
        "pred_distribution_cells_erased": {c: erased.count(c) for c in CLASSES},
        "pred_distribution_background_erased": {
            c: sum(r["pred_background_erased"] == c for r in rows if r["mask_ok"]) for c in CLASSES},
        "pred_distribution_gray": {c: sum(r["pred_gray"] == c for r in rows) for c in CLASSES},
        "groups": {g: _group_stats(rs) for g, rs in groups.items()},
        "error_figures": err_figs,
    }


def _pct(v, digits=1):
    return "—" if v is None else f"{v:.{digits}%}"


def _num(v):
    return "—" if v is None else f"{v:.2f}"


GROUP_TITLES = {"all": "Tất cả", "correct": "Ảnh đoán đúng", "wrong": "Ảnh đoán sai"}


def _summary_md(s, rows, err_idx) -> str:
    g = s["groups"]
    L = [f"# Giải thích mô hình — {s['name']}", "",
         f"Lần chạy `{s['run']}` · tập {s['split']} · {s['n']} ảnh · sai {s['n_errors']} ảnh · "
         f"lưới Grad-CAM {s['cam_grid'][0]}×{s['cam_grid'][1]}", ""]
    if s["saved_predictions_agree"]:
        a, b = s["saved_predictions_agree"]
        L += [f"Đối chiếu với dự đoán lưu lúc huấn luyện: trùng {a}/{b} ảnh"
              + ("." if a == b else " — **KHÁC NHAU: kiểm tra lại trọng số / tiền xử lý trước khi dùng kết quả.**"), ""]
    if s["n_mask_failed"]:
        L += [f"Không tách được mặt nạ tế bào ở {s['n_mask_failed']} ảnh (bỏ khỏi phép đo 1 và 3).", ""]
    if s["n_cam_empty"]:
        L += [f"Grad-CAM rỗng (mọi vùng đóng góp âm) ở {s['n_cam_empty']} ảnh.", ""]

    L += ["## 1. Ảnh bị phân loại sai", ""]
    if not s["n_errors"]:
        L += ["Không có ảnh sai trên tập này.", ""]
    else:
        L += [f"- Sai với độ tự tin ≥ 90%: **{s['errors_high_confidence']}**/{s['n_errors']}",
              f"- Lớp thật là lựa chọn thứ hai của mô hình: **{s['errors_true_is_second']}**/{s['n_errors']}",
              "", "| Thật → Dự đoán | Số ảnh |", "|---|---:|"]
        L += [f"| {k} | {v} |" for k, v in s["confusions"].items()]
        L += ["", "| Ảnh | Thật | Dự đoán | p(dự đoán) | p(thật) | CAM trong tế bào | Diện tích tế bào |",
              "|---|---|---|---:|---:|---:|---:|"]
        for i in err_idx:
            r = rows[i]
            L.append(f"| {Path(r['rel_path']).name} | {r['label']} | {r['pred']} | {_pct(r['p_pred'])} | "
                     f"{_pct(r['p_true'])} | {_pct(r['cam_in_cell'])} | {_pct(r['cell_area'])} |")
        L += ["", "Hình: " + ", ".join(f"`{f}`" for f in s["error_figures"])
              + " (ảnh gốc | Grad-CAM lớp dự đoán | Grad-CAM lớp thật).", ""]

    L += ["## 2. Mô hình nhìn vào đâu (Grad-CAM so với vùng tế bào)", "",
          "Tỷ số tập trung = (phần Grad-CAM nằm trong vùng tế bào) / (diện tích vùng tế bào). "
          "> 1: chú ý dồn vào tế bào; ≈ 1: rải đều không phân biệt; < 1: nhìn nền nhiều hơn tế bào.", "",
          "| Nhóm | Số ảnh | Diện tích tế bào | CAM trong tế bào | Tỷ số tập trung | Đỉnh CAM nằm trong tế bào |",
          "|---|---:|---:|---:|---:|---:|"]
    for key in ("all", "correct", "wrong", *CLASSES):
        x = g[key]
        L.append(f"| {GROUP_TITLES.get(key, 'Lớp ' + key)} | {x['n_mask']} | {_pct(x['cell_area'])} | "
                 f"{_pct(x['cam_in_cell'])} | {_num(x['focus_ratio'])} | {_pct(x['peak_in_cell'])} |")
    L += ["", "Hình: `correct_gradcam.png` (ảnh đúng mỗi lớp), `mask_check.png` (kiểm tra mặt nạ).", ""]

    L += [f"## 3. Grad-CAM có chỉ đúng vùng mô hình dùng không (che {ERASE_FRACTION:.0%} ô lưới)", "",
          "Che vùng Grad-CAM cao nhất phải làm xác suất lớp dự đoán giảm mạnh hơn che vùng ngẫu nhiên; "
          "nếu hai cột gần nhau thì bản đồ nhiệt không đáng tin, hoặc mô hình dựa vào đặc điểm toàn ảnh (màu).", "",
          "| Nhóm | p ban đầu | p sau che vùng CAM cao | p sau che ngẫu nhiên | Đổi dự đoán (CAM cao) | Đổi dự đoán (ngẫu nhiên) |",
          "|---|---:|---:|---:|---:|---:|"]
    for key in ("all", *CLASSES):
        x = g[key]
        L.append(f"| {GROUP_TITLES.get(key, 'Lớp ' + key)} | {_pct(x['p_pred'])} | {_pct(x['p_pred_erase_top'])} | "
                 f"{_pct(x['p_pred_erase_random'])} | {_pct(x['flip_erase_top'])} | {_pct(x['flip_erase_random'])} |")

    L += ["", "## 4. Xóa tế bào: mô hình có cần hình thái tế bào không", "",
          "Tô toàn bộ vùng tế bào nhuộm màu bằng màu nền của chính ảnh đó rồi dự đoán lại "
          "(hồng cầu và nền giữ nguyên; xem `cells_erased_check.png`). Ngẫu nhiên 4 lớp = 25%.", "",
          "| Lớp thật | Số ảnh | Accuracy ban đầu | Accuracy sau khi xóa tế bào | p(lớp thật) ban đầu | p(lớp thật) sau khi xóa |",
          "|---|---:|---:|---:|---:|---:|"]
    for key in ("all", *CLASSES):
        x = g[key]
        L.append(f"| {GROUP_TITLES.get(key, key)} | {x['n_erased']} | {_pct(x['accuracy_before_erase'])} | "
                 f"{_pct(x['accuracy_cells_erased'])} | {_pct(x['p_true_before_erase'])} | {_pct(x['p_true_cells_erased'])} |")
    dist = s["pred_distribution_cells_erased"]
    L += ["", "Dự đoán trên ảnh đã xóa tế bào: " + " · ".join(f"{c} {v}" for c, v in dist.items()), ""]
    flagged = [c for c in CLASSES if (g[c]["accuracy_cells_erased"] or 0) >= SHORTCUT_FLAG and g[c]["n_erased"]]
    if flagged:
        L += [f"⚠️ Sau khi xóa tế bào, mô hình vẫn đoán đúng ≥ {SHORTCUT_FLAG:.0%} ảnh lớp: **{', '.join(flagged)}**. "
              "Với các lớp này mô hình nhận ra lớp mà không cần tế bào — dấu hiệu dựa vào nền/tông màu nhuộm. "
              "Loại trừ hai cách giải thích khác trước khi kết luận: (a) mặt nạ bỏ sót tế bào (xem "
              "`cells_erased_check.png`); (b) mọi ảnh đã xóa đều bị dồn về một lớp (xem dòng phân bố ở trên) — "
              "khi đó lớp ấy đúng 100% chỉ vì là lớp \"mặc định khi không thấy tế bào\".", ""]
    L += ["## 5. Xóa nền: chỉ bạch cầu có đủ để mô hình đoán đúng không", "",
          "Ngược với mục 4: giữ nguyên bạch cầu, tô toàn bộ phần còn lại (nền, hồng cầu) bằng một màu nền phẳng "
          "(xem `background_erased_check.png`). Vẫn đúng = hình thái/màu của bạch cầu tự nó đủ thông tin; "
          "giảm mạnh = mô hình cần nền để đoán.", "",
          "| Lớp thật | Số ảnh | Accuracy ban đầu | Accuracy sau khi xóa nền | p(lớp thật) ban đầu | p(lớp thật) sau khi xóa nền |",
          "|---|---:|---:|---:|---:|---:|"]
    for key in ("all", *CLASSES):
        x = g[key]
        L.append(f"| {GROUP_TITLES.get(key, key)} | {x['n_erased']} | {_pct(x['accuracy_before_erase'])} | "
                 f"{_pct(x['accuracy_background_erased'])} | {_pct(x['p_true_before_erase'])} | {_pct(x['p_true_background_erased'])} |")
    dist = s["pred_distribution_background_erased"]
    L += ["", "Dự đoán trên ảnh đã xóa nền: " + " · ".join(f"{c} {v}" for c, v in dist.items()), "",
          "## 6. Ảnh trắng đen: mô hình phụ thuộc vào màu đến đâu", "",
          "Đổi ảnh test sang trắng đen (chỉ giữ độ sáng) rồi dự đoán bằng chính mô hình đó, không huấn luyện lại. "
          "Giảm mạnh = mô hình dựa vào màu. (Mô hình huấn luyện bằng ảnh xám thì đầu vào không đổi.)", "",
          "| Lớp thật | Số ảnh | Accuracy ban đầu | Accuracy với ảnh trắng đen | p(lớp thật) ban đầu | p(lớp thật) với ảnh trắng đen |",
          "|---|---:|---:|---:|---:|---:|"]
    for key in ("all", *CLASSES):
        x = g[key]
        L.append(f"| {GROUP_TITLES.get(key, key)} | {x['n']} | {_pct(x['accuracy'])} | {_pct(x['accuracy_gray'])} | "
                 f"{_pct(x['p_true'])} | {_pct(x['p_true_gray'])} |")
    dist = s["pred_distribution_gray"]
    L += ["", "Dự đoán trên ảnh trắng đen: " + " · ".join(f"{c} {v}" for c, v in dist.items()), ""]
    L += ["## Giới hạn", "",
          "- Mặt nạ tế bào là ước lượng bằng ngưỡng màu (Otsu trên kênh a* của LAB), không phải nhãn phân vùng.",
          f"- Grad-CAM có độ phân giải thấp ({s['cam_grid'][0]}×{s['cam_grid'][1]} ô), không chỉ ra được chi tiết bên trong một tế bào.",
          "- Ảnh đã xóa tế bào nằm ngoài phân phối huấn luyện; kết quả mục 4 là bằng chứng gián tiếp.",
          "- Grad-CAM cho biết vùng ảnh hưởng tới dự đoán, không cho biết đặc điểm gì ở vùng đó (màu hay hình dạng).", ""]
    return "\n".join(L)


# ---------- vẽ hình ----------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _show(ax, img, title="", color="black"):
    ax.imshow(img)
    ax.set_title(title, fontsize=7.5, color=color)
    ax.axis("off")


def _plot_errors(out, images, cams_pred, cams_true, rows, idx, size, name, per_page=8):
    plt = _plt()
    files = []
    for page, s in enumerate(range(0, len(idx), per_page), 1):
        chunk = idx[s:s + per_page]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(7.2, 2.5 * len(chunk) + 0.4), squeeze=False)
        for ax_row, i in zip(axes, chunk):
            r = rows[i]
            _show(ax_row[0], images[i], f"{Path(r['rel_path']).name}\nthật: {r['label']}")
            _show(ax_row[1], overlay(images[i], upsample(cams_pred[i], size)),
                  f"vì sao đoán {r['pred']}\np = {r['p_pred']:.1%}", "firebrick")
            _show(ax_row[2], overlay(images[i], upsample(cams_true[i], size)),
                  f"bằng chứng cho {r['label']} (lớp thật)\np = {r['p_true']:.1%}", "darkgreen")
        fig.suptitle(f"{name} — ảnh sai, xếp theo độ tự tin giảm dần (trang {page})", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        f = f"errors_gradcam_{page}.png"
        fig.savefig(out / f, dpi=130)
        plt.close(fig)
        files.append(f)
    return files


def _plot_correct(path, images, cams_pred, rows, picks, size, name):
    plt = _plt()
    ncol = max(1, max(len(v) for v in picks.values()))
    fig, axes = plt.subplots(2 * len(CLASSES), ncol, figsize=(2.1 * ncol, 4.5 * len(CLASSES)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for c, idx in picks.items():
        for j, i in enumerate(idx):
            r = rows[i]
            _show(axes[2 * c][j], images[i], f"{r['label']} · p = {r['p_pred']:.1%}")
            ratio = "" if r["cam_in_cell"] is None else f"CAM trong tế bào {r['cam_in_cell']:.0%}"
            _show(axes[2 * c + 1][j], overlay(images[i], upsample(cams_pred[i], size)), ratio)
    fig.suptitle(f"{name} — ảnh đoán đúng: bên trái tự tin nhất, bên phải kém tự tin nhất", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_masks(path, images, masks, rows, idx, nrow, title, use_erased=False):
    plt = _plt()
    ncol = max(1, math.ceil(len(idx) / nrow))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 2.05 * nrow + 0.3), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        r = rows[i]
        note = "" if r["mask_ok"] else " (không tách được)"
        _show(ax, images[i], f"{r['label']}{note}" + (f" → {r[use_erased]}" if use_erased and r["mask_ok"] else ""))
        if masks[i] is not None:
            ax.contour(dilate(masks[i], 4), levels=[0.5], colors="yellow", linewidths=0.8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------- nhiều mô hình ----------

def compare_runs(results: list[dict], out_dir, max_images: int = 12) -> Path:
    """Bảng tổng hợp + hình đặt Grad-CAM của các mô hình cạnh nhau trên cùng ảnh."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = results[0]
    index = [{p: i for i, p in enumerate(r["rel_paths"])} for r in results]
    shared = [p for p in base["rel_paths"] if all(p in ix for ix in index)]
    n_wrong = {p: sum(int(r["pred"][ix[p]] != r["labels"][ix[p]]) for r, ix in zip(results, index)) for p in shared}
    hard = sorted((p for p in shared if n_wrong[p]), key=lambda p: -n_wrong[p])[:max_images]
    easy = []
    for c in range(len(CLASSES)):                    # thêm 1 ảnh mọi mô hình đều đúng cho mỗi lớp
        easy += [p for p in shared if not n_wrong[p] and base["labels"][index[0][p]] == c][:1]
    chosen = hard + easy
    if chosen:
        fig, axes = plt.subplots(len(chosen), len(results) + 1,
                                 figsize=(2.0 * (len(results) + 1), 2.25 * len(chosen) + 0.4), squeeze=False)
        for ax_row, p in zip(axes, chosen):
            i0 = index[0][p]
            _show(ax_row[0], base["images"][i0], f"{Path(p).name}\nthật: {CLASSES[base['labels'][i0]]}")
            for ax, r, ix in zip(ax_row[1:], results, index):
                i = ix[p]
                ok = r["pred"][i] == r["labels"][i]
                _show(ax, overlay(r["images"][i], upsample(r["cams_pred"][i], r["size"])),
                      f"{r['name']}\n{CLASSES[r['pred'][i]]} {r['probs'][i, r['pred'][i]]:.0%}",
                      "darkgreen" if ok else "firebrick")
        fig.suptitle("Grad-CAM lớp dự đoán — ảnh có mô hình đoán sai (trên), ảnh mọi mô hình đúng (dưới)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        fig.savefig(out_dir / "explain_compare_gradcam.png", dpi=120)
        plt.close(fig)

    L = ["# Giải thích mô hình — tổng hợp", "",
         "Cách đọc từng cột: xem `explain/summary.md` trong thư mục mỗi lần chạy.", "",
         "| Mô hình | Sai | Tỷ số tập trung vào tế bào | Đỉnh CAM trong tế bào | p sau che vùng CAM cao / ngẫu nhiên "
         "| Accuracy sau khi xóa tế bào | Lớp vẫn đúng ≥ 50% khi xóa tế bào "
         "| Accuracy sau khi xóa nền (chỉ còn tế bào) | Accuracy với ảnh trắng đen |",
         "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in results:
        s = r["summary"]
        a = s["groups"]["all"]
        flagged = [c for c in CLASSES if (s["groups"][c]["accuracy_cells_erased"] or 0) >= SHORTCUT_FLAG]
        L.append(f"| {r['name']} | {s['n_errors']}/{s['n']} | {_num(a['focus_ratio'])} | {_pct(a['peak_in_cell'])} | "
                 f"{_pct(a['p_pred_erase_top'])} / {_pct(a['p_pred_erase_random'])} | "
                 f"{_pct(a['accuracy_cells_erased'])} | {', '.join(flagged) or '—'} | {_pct(a['accuracy_background_erased'])} | "
                 f"{_pct(a['accuracy_gray'])} |")
    if chosen:
        n_all = sum(1 for p in shared if n_wrong[p] == len(results))
        L += ["", f"Ảnh có ít nhất một mô hình sai: {sum(1 for v in n_wrong.values() if v)}/{len(shared)}; "
                  f"mọi mô hình cùng sai: {n_all}.", "", "![](explain_compare_gradcam.png)"]
    L += ["", "Chi tiết: " + " · ".join(f"`{r['run_dir'].name}/explain/summary.md`" for r in results), ""]
    path = out_dir / "EXPLAIN.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
