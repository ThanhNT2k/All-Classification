"""Gom số liệu và tạo hình kết quả cho Chương 3 của báo cáo.

  python scripts/make_report_results.py --runs results/runs --out "../bao cao/images"

Tạo training_curves.png, confusion_matrices.png, roc_curves.png và in các bảng số liệu (kèm report_numbers.json).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from allcls import CLASSES  # noqa: E402
from allcls.compare import find_runs  # noqa: E402
from allcls.evaluate import load_predictions  # noqa: E402

MODELS = [("simple_cnn", "CNN cơ bản"), ("resnet50_rgb", "ResNet50"),
          ("efficientnet_b0_rgb", "EfficientNet-B0"), ("mobilenetv2_rgb", "MobileNetV2")]


def metrics(run_dir):
    return json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))


def history(run_dir):
    with open(Path(run_dir) / "history.csv", encoding="utf-8") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def plot_curves(path, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(runs), figsize=(3.3 * len(runs), 5.4), squeeze=False)
    for j, (title, d) in enumerate(runs):
        h = history(d)
        ep = [r["epoch"] for r in h]
        best = metrics(d)["best_epoch"]
        axes[0][j].plot(ep, [r["train_loss"] for r in h], label="train")
        axes[0][j].plot(ep, [r["val_loss"] for r in h], label="validation")
        axes[0][j].set_title(title, fontsize=10)
        axes[0][j].set_yscale("log")
        axes[1][j].plot(ep, [r["train_acc"] for r in h], label="train")
        axes[1][j].plot(ep, [r["val_acc"] for r in h], label="validation")
        axes[1][j].set_ylim(0.5, 1.005)
        axes[1][j].set_xlabel("epoch")
        for ax in (axes[0][j], axes[1][j]):
            ax.axvline(best, color="gray", ls="--", lw=0.8)
            ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("loss (thang log)")
    axes[1][0].set_ylabel("accuracy")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusions(path, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(runs), figsize=(3.3 * len(runs), 3.6), squeeze=False)
    for ax, (title, d) in zip(axes[0], runs):
        cm = np.asarray(metrics(d)["test"]["confusion_matrix"])
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for k in range(4):
                ax.text(k, i, str(cm[i, k]), ha="center", va="center", fontsize=9,
                        color="white" if norm[i, k] > 0.6 else "black")
        ax.set_xticks(range(4), CLASSES, fontsize=8)
        ax.set_yticks(range(4), CLASSES, fontsize=8)
        ax.set_xlabel("Dự đoán")
        ax.set_title(title, fontsize=10)
    axes[0][0].set_ylabel("Thực tế")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(path, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, axes = plt.subplots(1, len(runs), figsize=(3.3 * len(runs), 3.5), squeeze=False)
    for ax, (title, d) in zip(axes[0], runs):
        _, yt, _, pr = load_predictions(Path(d) / "predictions_test.csv")
        for c, name in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(yt == c, pr[:, c])
            ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yt == c, pr[:, c]):.4f})")
        ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=0.8)
        ax.set_xlim(-0.01, 0.2)                    # phóng to góc trên trái, nơi các đường khác nhau
        ax.set_ylim(0.8, 1.005)
        ax.set_xlabel("Tỉ lệ dương giả")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Tỉ lệ dương thật")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ms(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else None, "n": len(v),
            "values": [float(x) for x in v]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    primary = [(t, find_runs(a.runs, n)[42]) for n, t in MODELS]
    plot_curves(out / "training_curves_repo.png", primary)
    plot_confusions(out / "confusion_matrices_repo.png", primary)
    plot_roc(out / "roc_curves_repo.png", primary)

    R = {"models": {}, "fractions": {}, "strategies": {}}
    for n, t in MODELS:
        runs = find_runs(a.runs, n)
        M = {s: metrics(d) for s, d in sorted(runs.items())}
        m42 = M[42]
        R["models"][t] = {
            "seeds": sorted(M),
            **{k: ms([m["test"][k] for m in M.values()]) for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")},
            "auc_macro": ms([m["test"]["auc"]["macro_ovr"] for m in M.values()]),
            "malignant_sensitivity": ms([m["test"]["screening"]["malignant_sensitivity"] for m in M.values()]),
            "missed_malignant": [m["test"]["screening"]["fn"] for m in M.values()],
            "recall": {c: ms([m["test"]["per_class"][c]["recall"] for m in M.values()]) for c in CLASSES},
            "val_macro_f1": ms([m["val"]["macro_f1"] for m in M.values()]),
            "best_epoch": [m["best_epoch"] for m in M.values()], "epochs_run": [m["epochs_run"] for m in M.values()],
            "efficiency_seed42": m42["efficiency"], "ci95_seed42": m42["test"]["ci95"],
            "confusion_seed42": m42["test"]["confusion_matrix"],
        }
        fr = {"100": m42["test"]["macro_f1"]}
        for pct in (75, 50):
            d = find_runs(a.runs, f"{n}_frac{pct}").get(42)
            if d:
                mm = metrics(d)
                fr[str(pct)] = mm["test"]["macro_f1"]
                fr[f"n_train_{pct}"] = json.loads((d / "env.json").read_text(encoding="utf-8"))["n_images"]["train"]
        R["fractions"][t] = fr
        if n != "simple_cnn":
            st = {"full": {"f1": m42["test"]["macro_f1"], "acc": m42["test"]["accuracy"], "epochs": m42["epochs_run"]}}
            for key in ("scratch", "featext", "partial"):
                d = find_runs(a.runs, f"{n}_{key}").get(42)
                if d:
                    mm = metrics(d)
                    st[key] = {"f1": mm["test"]["macro_f1"], "acc": mm["test"]["accuracy"], "epochs": mm["epochs_run"],
                               "trainable": mm["efficiency"].get("trainable_params")}
            R["strategies"][t] = st
    (out.parent / "report_numbers.json").write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")

    pc = lambda x: f"{100 * x:.1f}"   # noqa: E731
    for t, r in R["models"].items():
        print(f"\n{t}: seeds {r['seeds']} best_epoch {r['best_epoch']} epochs {r['epochs_run']}")
        for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            print(f"  {k}: {pc(r[k]['mean'])} ± {pc(r[k]['std'] or 0)}   {[pc(v) for v in r[k]['values']]}")
        print(f"  auc: {r['auc_macro']['mean']:.4f} ± {(r['auc_macro']['std'] or 0):.4f}; bỏ sót ác tính {r['missed_malignant']}")
        print("  recall:", {c: f"{pc(v['mean'])} ± {pc(v['std'] or 0)}" for c, v in r["recall"].items()})
        print("  eff:", r["efficiency_seed42"])
    print("\nfractions:", json.dumps(R["fractions"], ensure_ascii=False))
    print("strategies:", json.dumps(R["strategies"], ensure_ascii=False))
