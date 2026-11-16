"""So sánh các mô hình trên tập TEST — tạo COMPARISON.md, comparison.csv và biểu đồ.

Gồm:
  Bảng 1  Chất lượng phân loại: accuracy, balanced accuracy, macro precision/recall/F1,
          AUC, độ nhạy ác tính, độ đặc hiệu (kèm KTC 95% hoặc trung bình ± độ lệch chuẩn
          nếu chạy nhiều seed)
  Bảng 2  F1 từng lớp
  Bảng 3  Hiệu năng tính toán: số tham số, dung lượng, thời gian huấn luyện, tốc độ suy luận
  Bảng 4  Kiểm định McNemar từng cặp mô hình (p hiệu chỉnh Holm) — chênh lệch có ý nghĩa không
  Bảng 5  Kiểm tra lối tắt màu (ResNet50 các biến thể)
  Hình    accuracy/F1 có thanh sai số, đường ROC (ác tính vs Benign), accuracy theo số tham số
"""
from __future__ import annotations

import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np

from . import CLASSES
from .evaluate import BENIGN, load_predictions
from .experiments import COLOR_BASELINE, EXPERIMENTS, REFERENCE_FOR_COLOR

PRIMARY_SEED = 42

QUALITY = [
    ("Accuracy", lambda m: m["accuracy"], "accuracy"),
    ("Balanced acc.", lambda m: m["balanced_accuracy"], None),
    ("Macro precision", lambda m: m.get("macro_precision"), None),
    ("Macro recall", lambda m: m.get("macro_recall"), None),
    ("Macro F1", lambda m: m["macro_f1"], "macro_f1"),
    ("AUC (OvR)", lambda m: (m.get("auc") or {}).get("macro_ovr"), None),
    ("Độ nhạy ác tính", lambda m: m["screening"]["malignant_sensitivity"], "malignant_sensitivity"),
    ("Độ đặc hiệu", lambda m: m["screening"]["benign_specificity"], "benign_specificity"),
]


# ---------------- thu thập kết quả ----------------

def find_runs(runs_root: Path, name: str) -> dict[int, Path]:
    """Lần chạy hoàn chỉnh (có file DONE) mới nhất của từng seed."""
    pat = re.compile(rf"^{re.escape(name)}_s(\d+)_(\d{{8}}_\d{{6}})$")
    found = {}
    for p in sorted(Path(runs_root).glob(f"{name}_s*")):
        mt = pat.match(p.name)
        if mt and (p / "DONE").exists():
            found[int(mt.group(1))] = p          # sorted -> giữ lần mới nhất
    return found


def collect(runs_root, color_baseline_dir=None) -> list[dict]:
    entries = []
    if color_baseline_dir and (Path(color_baseline_dir) / "metrics.json").exists():
        name, code, title, group = COLOR_BASELINE
        entries.append({"name": name, "code": code, "title": title, "group": group,
                        "runs": {PRIMARY_SEED: Path(color_baseline_dir)}})
    for name, code, title, group in EXPERIMENTS:
        runs = find_runs(runs_root, name)
        entries.append({"name": name, "code": code, "title": title, "group": group, "runs": runs})
    for e in entries:
        e["metrics"] = {s: json.loads((d / "metrics.json").read_text(encoding="utf-8"))
                        for s, d in e["runs"].items()}
        e["primary"] = (PRIMARY_SEED if PRIMARY_SEED in e["runs"]
                        else (min(e["runs"]) if e["runs"] else None))
    return entries


# ---------------- thống kê ----------------

def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Kiểm định McNemar chính xác (nhị thức) trên cùng một tập ảnh."""
    b = int((correct_a & ~correct_b).sum())      # A đúng, B sai
    c = int((~correct_a & correct_b).sum())      # A sai, B đúng
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)
    return {"a_only": b, "b_only": c, "p": p}


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def _aligned_correct(entries, split="test"):
    """Ghép dự đoán của các mô hình theo tên file (seed chính)."""
    data = {}
    for e in entries:
        if e["primary"] is None:
            continue
        f = e["runs"][e["primary"]] / f"predictions_{split}.csv"
        if f.exists():
            rel, yt, yp, pr = load_predictions(f)
            data[e["name"]] = dict(zip(rel, zip(yt, yp, pr)))
    if not data:
        return {}, []
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    out = {}
    for name, d in data.items():
        yt = np.array([d[r][0] for r in common])
        yp = np.array([d[r][1] for r in common])
        pr = np.array([d[r][2] for r in common])
        out[name] = {"y_true": yt, "y_pred": yp, "probs": pr, "correct": yt == yp}
    return out, common


# ---------------- định dạng ----------------

def _vals(e, fn):
    v = [fn(m["test"]) for m in e["metrics"].values()]
    return [x for x in v if x is not None]


def _cell(e, fn, ci_key, fmt="{:.1%}"):
    v = _vals(e, fn)
    if not v:
        return "—"
    if len(v) > 1:
        return f"{fmt.format(np.mean(v))} ± {fmt.format(np.std(v, ddof=1))}"
    s = fmt.format(v[0])
    ci = (e["metrics"][e["primary"]]["test"].get("ci95") or {}).get(ci_key) if ci_key else None
    return s + (f" ({ci[0]:.1%}–{ci[1]:.1%})" if ci else "")


def _eff(e, key, fmt):
    v = [m.get("efficiency", {}).get(key) for m in e["metrics"].values()]
    v = [x for x in v if x is not None]
    return fmt(np.mean(v)) if v else "—"


def _p(p):
    return "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


# ---------------- báo cáo ----------------

def build_report(runs_root, color_baseline_dir=None, out_dir=None) -> str:
    runs_root = Path(runs_root)
    out_dir = Path(out_dir or runs_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = collect(runs_root, color_baseline_dir)
    done = [e for e in entries if e["runs"]]
    models = [e for e in done if e["group"] in ("baseline", "model")]
    colors = [e for e in done if e["group"] == "color"]
    ref = next((e for e in done if e["name"] == REFERENCE_FOR_COLOR), None)
    n_seeds = max((len(e["runs"]) for e in done), default=0)

    L = ["# So sánh các mô hình (tập TEST)", ""]
    missing = [f"{e['code']} {e['title']}" for e in entries if not e["runs"]]
    if missing:
        L += ["> Chưa có kết quả: " + ", ".join(missing), ""]
    if n_seeds > 1:
        L += ["> Mô hình chạy nhiều seed: hiển thị **trung bình ± độ lệch chuẩn** giữa các seed. "
              "Mô hình 1 seed: giá trị kèm KTC 95% (bootstrap).", ""]
    else:
        L += ["> Giá trị kèm khoảng tin cậy 95% (bootstrap theo ảnh).", ""]

    def quality_table(rows, title):
        T = [f"## {title}", "", "| Mã | Mô hình | Seed | " + " | ".join(q[0] for q in QUALITY) + " |",
             "|---|---|---|" + "---|" * len(QUALITY)]
        for e in rows:
            T.append(f"| {e['code']} | {e['title']} | {len(e['runs'])} | "
                     + " | ".join(_cell(e, fn, ck, "{:.3f}" if q.startswith("AUC") else "{:.1%}")
                                  for q, fn, ck in QUALITY) + " |")
        return T + [""]

    L += quality_table(models, "Bảng 1. Chất lượng phân loại")

    L += ["## Bảng 2. F1 từng lớp", "", "| Mô hình | " + " | ".join(CLASSES) + " |",
          "|---|" + "---|" * len(CLASSES)]
    for e in models + colors:
        L.append(f"| {e['code']} {e['title']} | "
                 + " | ".join(_cell(e, lambda m, c=c: m["per_class"][c]["f1"], None) for c in CLASSES) + " |")
    L.append("")

    L += ["## Bảng 3. Hiệu năng tính toán", "",
          "| Mô hình | Tham số (triệu) | Dung lượng (MB) | Huấn luyện (phút) | Giây/epoch | "
          "Độ trễ 1 ảnh (ms) | Thông lượng (ảnh/s) | Thiết bị đo |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for e in [e for e in models if e["group"] == "model"]:
        dev = next((m.get("efficiency", {}).get("benchmark_device") for m in e["metrics"].values()), None)
        L.append(f"| {e['code']} {e['title']} | {_eff(e, 'params', lambda x: f'{x / 1e6:.2f}')} | "
                 f"{_eff(e, 'model_size_mb', lambda x: f'{x:.1f}')} | {_eff(e, 'train_minutes', lambda x: f'{x:.1f}')} | "
                 f"{_eff(e, 'sec_per_epoch', lambda x: f'{x:.1f}')} | {_eff(e, 'latency_ms_batch1', lambda x: f'{x:.1f}')} | "
                 f"{_eff(e, 'throughput_img_per_s', lambda x: f'{x:.0f}')} | {dev or '—'} |")
    L.append("")

    # ----- McNemar -----
    aligned, common = _aligned_correct(models + colors)
    pairs = [(a["name"], b["name"]) for a, b in combinations(models, 2)]
    if ref:
        pairs += [(ref["name"], c["name"]) for c in colors]
    pairs = [(a, b) for a, b in pairs if a in aligned and b in aligned]
    tests = [mcnemar(aligned[a]["correct"], aligned[b]["correct"]) for a, b in pairs]
    adj = holm([t["p"] for t in tests]) if tests else []
    title_of = {e["name"]: f"{e['code']} {e['title']}" for e in entries}
    mc_rows = []
    if tests:
        L += ["## Bảng 4. Kiểm định McNemar từng cặp (cùng " + str(len(common)) + " ảnh test, seed chính)", "",
              "| Mô hình A | Mô hình B | Acc A | Acc B | Chỉ A đúng | Chỉ B đúng | p | p (Holm) | Kết luận |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for (a, b), t, pa in zip(pairs, tests, adj):
            acc_a, acc_b = aligned[a]["correct"].mean(), aligned[b]["correct"].mean()
            verdict = ("khác biệt có ý nghĩa" if pa < 0.05 else "không khác biệt có ý nghĩa")
            L.append(f"| {title_of[a]} | {title_of[b]} | {acc_a:.1%} | {acc_b:.1%} | {t['a_only']} | "
                     f"{t['b_only']} | {_p(t['p'])} | {_p(pa)} | {verdict} |")
            mc_rows.append({"a": a, "b": b, **t, "p_holm": pa})
        L += ["", "Ngưỡng α = 0,05 sau hiệu chỉnh Holm cho " + str(len(tests)) + " phép so sánh. "
              "\"Không khác biệt có ý nghĩa\" nghĩa là chênh lệch accuracy có thể chỉ do ngẫu nhiên.", ""]

    if colors and ref:
        L += quality_table([ref] + colors, "Bảng 5. Kiểm tra lối tắt màu (ResNet50)")
        L += ["Đọc bảng: nếu ảnh xám (C1) giảm mạnh, đặc biệt ở lớp Pro (Bảng 2), mô hình đang dựa vào màu nhuộm. "
              "Nếu C2/C3 giữ được accuracy, mô hình vẫn học được đặc trưng ngoài màu.", ""]

    figs = _plots(models, colors, aligned, out_dir)
    if figs:
        L += ["## Hình", ""] + [f"![{f}]({f})" for f in figs] + [""]

    _write_csv(models + colors, out_dir / "comparison.csv")
    if mc_rows:
        (out_dir / "mcnemar.json").write_text(json.dumps(mc_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    text = "\n".join(L)
    (out_dir / "COMPARISON.md").write_text(text, encoding="utf-8")
    return text


def _write_csv(entries, path):
    cols = ["code", "model", "n_seeds"]
    for q, _, _ in QUALITY:
        cols += [f"{q} (mean)", f"{q} (std)"]
    cols += [f"F1 {c}" for c in CLASSES] + ["params", "model_size_mb", "train_minutes",
                                             "latency_ms_batch1", "throughput_img_per_s"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in entries:
            row = [e["code"], e["title"], len(e["runs"])]
            for _, fn, _ in QUALITY:
                v = _vals(e, fn)
                row += [round(float(np.mean(v)), 4) if v else "",
                        round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else ""]
            for c in CLASSES:
                v = _vals(e, lambda m, c=c: m["per_class"][c]["f1"])
                row.append(round(float(np.mean(v)), 4) if v else "")
            for k in ("params", "model_size_mb", "train_minutes", "latency_ms_batch1", "throughput_img_per_s"):
                v = [m.get("efficiency", {}).get(k) for m in e["metrics"].values()]
                v = [x for x in v if x is not None]
                row.append(round(float(np.mean(v)), 3) if v else "")
            w.writerow(row)


def _roc(y_bin, score):
    order = np.argsort(-score, kind="mergesort")
    y, sc = y_bin[order], score[order]
    cut = np.r_[np.where(np.diff(sc))[0], len(sc) - 1]   # chỉ lấy ngưỡng tại điểm đổi giá trị (xử lý điểm trùng)
    tps, fps = np.cumsum(y)[cut], np.cumsum(1 - y)[cut]
    tpr = np.concatenate([[0], tps / max(y.sum(), 1)])
    fpr = np.concatenate([[0], fps / max((1 - y).sum(), 1)])
    integrate = getattr(np, "trapezoid", None) or np.trapz      # NumPy 2 đổi tên hàm
    return fpr, tpr, float(integrate(tpr, fpr))


def _plots(models, colors, aligned, out_dir) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    rows = models + colors
    if rows:
        labels = [e["code"] for e in rows]
        fig, ax = plt.subplots(figsize=(1.2 * len(rows) + 2, 3.6))
        x = np.arange(len(rows))
        for k, (key, fn, ck, off) in enumerate([("Accuracy", QUALITY[0][1], "accuracy", -0.2),
                                                ("Macro F1", QUALITY[4][1], "macro_f1", 0.2)]):
            means, lo, hi = [], [], []
            for e in rows:
                v = _vals(e, fn)
                mval = float(np.mean(v))
                if len(v) > 1:
                    sd = float(np.std(v, ddof=1))
                    lo.append(sd)
                    hi.append(sd)
                else:
                    ci = (e["metrics"][e["primary"]]["test"].get("ci95") or {}).get(ck)
                    lo.append(mval - ci[0] if ci else 0)
                    hi.append(ci[1] - mval if ci else 0)
                means.append(mval)
            ax.bar(x + off, means, 0.38, yerr=[lo, hi], capsize=3, label=key)
        ax.set_xticks(x, labels)
        allv = [v for e in rows for v in _vals(e, QUALITY[0][1])]
        ax.set_ylim(max(0.0, min(allv) - 0.1), 1.02)
        ax.set_ylabel("Tập test")
        ax.set_title("Accuracy và Macro F1 (thanh sai số: KTC 95% hoặc ±1 SD)", fontsize=10, pad=24)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_accuracy_f1.png", dpi=150)
        plt.close(fig)
        files.append("fig_accuracy_f1.png")

    roc_rows = [e for e in rows if e["name"] in aligned]
    if roc_rows:
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        for e in roc_rows:
            a = aligned[e["name"]]
            fpr, tpr, auc = _roc((a["y_true"] != BENIGN).astype(int), 1 - a["probs"][:, BENIGN])
            ax.plot(fpr, tpr, label=f"{e['code']} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("1 − Độ đặc hiệu")
        ax.set_ylabel("Độ nhạy")
        ax.set_title("ROC: ác tính (Early+Pre+Pro) vs Benign", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_roc_malignant.png", dpi=150)
        plt.close(fig)
        files.append("fig_roc_malignant.png")

    eff = [e for e in models if _vals(e, QUALITY[0][1])
           and any(m.get("efficiency", {}).get("params") for m in e["metrics"].values())
           and e["group"] == "model"]
    if eff:
        from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        xs = []
        for e in eff:
            params = np.mean([m["efficiency"]["params"] for m in e["metrics"].values()]) / 1e6
            acc = float(np.mean(_vals(e, QUALITY[0][1])))
            xs.append(params)
            ax.scatter(params, acc, s=40, label=f"{e['code']} {e['title']} ({params:.1f}M)")
            ax.annotate(e["code"], (params, acc), fontsize=8, xytext=(5, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
        ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100) if min(xs) / 1.6 <= t <= max(xs) * 1.6]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlabel("Số tham số (triệu, thang log)")
        ax.set_ylabel("Accuracy (test)")
        ax.set_title("Accuracy theo kích thước mô hình", fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_accuracy_vs_params.png", dpi=150)
        plt.close(fig)
        files.append("fig_accuracy_vs_params.png")
    return files
