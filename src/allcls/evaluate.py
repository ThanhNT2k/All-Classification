"""Đo lường kết quả phân loại. Không phụ thuộc PyTorch — dùng lại được cho mọi mô hình.

Ngoài các chỉ số phân loại 4 lớp, báo riêng góc độ SÀNG LỌC:
"ác tính" = Early + Pre + Pro, "lành" = Benign.
  - độ nhạy  = tỷ lệ ảnh ác tính được dự đoán là một lớp ác tính bất kỳ
  - độ đặc hiệu = tỷ lệ ảnh Benign được dự đoán là Benign
Khoảng tin cậy 95% bằng bootstrap theo ảnh (lưu ý: ảnh, không phải bệnh nhân).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from . import CLASSES

BENIGN = CLASSES.index("Benign")


def _confusion(y_true, y_pred, k):
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def _macro_f1_from_cm(cm):
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def _screening(y_true, y_pred):
    t, p = y_true != BENIGN, y_pred != BENIGN
    tp, fn = int((t & p).sum()), int((t & ~p).sum())
    tn, fp = int((~t & ~p).sum()), int((~t & p).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return sens, spec, {"tp": tp, "fn": fn, "tn": tn, "fp": fp}


def _bootstrap_ci(y_true, y_pred, k, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc, f1, sens, spec = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        acc.append((yt == yp).mean())
        f1.append(_macro_f1_from_cm(_confusion(yt, yp, k)))
        s, sp, _ = _screening(yt, yp)
        sens.append(s)
        spec.append(sp)

    def ci(v):
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None

    return {"accuracy": ci(acc), "macro_f1": ci(f1),
            "malignant_sensitivity": ci(sens), "benign_specificity": ci(spec)}


def compute_metrics(y_true, y_pred, probs=None, n_boot: int = 1000, seed: int = 0) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = len(CLASSES)
    cm = _confusion(y_true, y_pred, k)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    predicted = cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    sens, spec, counts = _screening(y_true, y_pred)
    present = support > 0
    out = {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_precision": float(precision[present].mean()),
        "macro_recall": float(recall[present].mean()),
        "macro_f1": _macro_f1_from_cm(cm),
        "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(support[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": cm.tolist(),
        "screening": {"malignant_sensitivity": sens, "benign_specificity": spec, **counts},
    }
    if probs is not None:
        probs = np.asarray(probs, dtype=float)
        out["auc"] = _auc(y_true, probs)
    if n_boot:
        out["ci95"] = _bootstrap_ci(y_true, y_pred, k, n_boot, seed)
    return _clean(out)


def _auc(y_true, probs):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    res = {}
    mal_true = (y_true != BENIGN).astype(int)
    if 0 < mal_true.sum() < len(mal_true):
        res["malignant_vs_benign"] = float(roc_auc_score(mal_true, 1 - probs[:, BENIGN]))
    if len(np.unique(y_true)) == len(CLASSES):
        res["macro_ovr"] = float(roc_auc_score(y_true, probs, multi_class="ovr",
                                               labels=list(range(len(CLASSES)))))
    return res


def _clean(x):
    """NaN -> None để ghi JSON hợp lệ."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


# ---------- lưu kết quả ----------

def save_predictions(path, rel_paths, y_true, y_pred, probs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "label", "pred", "correct"] + [f"p_{c}" for c in CLASSES])
        for rp, t, p, pr in zip(rel_paths, y_true, y_pred, probs):
            w.writerow([rp, CLASSES[t], CLASSES[p], int(t == p)] + [f"{v:.5f}" for v in pr])


def load_predictions(path):
    rel, yt, yp, pr = [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel.append(row["rel_path"])
            yt.append(CLASSES.index(row["label"]))
            yp.append(CLASSES.index(row["pred"]))
            pr.append([float(row[f"p_{c}"]) for c in CLASSES])
    return rel, np.array(yt), np.array(yp), np.array(pr)


def plot_confusion(cm, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(v, ci=None):
    if v is None:
        return "—"
    s = f"{v:.1%}"
    if ci:
        s += f" ({ci[0]:.1%}–{ci[1]:.1%})"
    return s


def metrics_markdown(m: dict, title: str) -> str:
    ci = m.get("ci95", {})
    sc = m["screening"]
    L = [f"### {title} (n = {m['n']})", "",
         "| Chỉ số | Giá trị (KTC 95%) |", "|---|---|",
         f"| Accuracy | {_fmt(m['accuracy'], ci.get('accuracy'))} |",
         f"| Balanced accuracy | {_fmt(m['balanced_accuracy'])} |",
         f"| Macro precision | {_fmt(m.get('macro_precision'))} |",
         f"| Macro recall | {_fmt(m.get('macro_recall'))} |",
         f"| Macro F1 | {_fmt(m['macro_f1'], ci.get('macro_f1'))} |",
         f"| Độ nhạy ác tính (Early+Pre+Pro) | {_fmt(sc['malignant_sensitivity'], ci.get('malignant_sensitivity'))} |",
         f"| Độ đặc hiệu (Benign) | {_fmt(sc['benign_specificity'], ci.get('benign_specificity'))} |"]
    if m.get("auc"):
        for k, v in m["auc"].items():
            L.append(f"| AUC {k} | {v:.3f} |")
    L += ["", "| Lớp | Precision | Recall | F1 | Số ảnh |", "|---|---:|---:|---:|---:|"]
    for c, d in m["per_class"].items():
        L.append(f"| {c} | {d['precision']:.1%} | {d['recall']:.1%} | {d['f1']:.1%} | {d['support']} |")
    L += ["", f"Ca ác tính bị bỏ sót (dự đoán Benign): **{sc['fn']}** · "
              f"Benign bị báo nhầm ác tính: **{sc['fp']}**", ""]
    return "\n".join(L)


def write_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
