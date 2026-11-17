"""Tạo hình Grad-CAM cho báo cáo từ các lần chạy đã có (4 mô hình của báo cáo).

  python scripts/make_report_figures.py --runs results/runs --out "../bao cao/images"

Tạo: gradcam.png (ảnh đoán đúng), gradcam_errors.png (ảnh đoán sai), gradcam_occlusion.png (các phép thử che/xóa).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from allcls import CLASSES  # noqa: E402
from allcls.compare import find_runs  # noqa: E402
from allcls.explain import (_plt, _show, background_color, cell_mask, dilate, erase,  # noqa: E402
                            explain_run, overlay, to_gray, upsample)

MODELS = [("simple_cnn", "CNN cơ bản"), ("resnet50_rgb", "ResNet50"),
          ("efficientnet_b0_rgb", "EfficientNet-B0"), ("mobilenetv2_rgb", "MobileNetV2")]


def cam_grid(path, results, titles, rel_paths):
    plt = _plt()
    base = results[0]
    index = [{p: i for i, p in enumerate(r["rel_paths"])} for r in results]
    fig, axes = plt.subplots(len(rel_paths), len(results) + 1,
                             figsize=(2.15 * (len(results) + 1), 2.4 * len(rel_paths)), squeeze=False)
    for ax_row, p in zip(axes, rel_paths):
        i0 = index[0][p]
        _show(ax_row[0], base["images"][i0], f"Ảnh gốc\nlớp thật: {CLASSES[base['labels'][i0]]}")
        for ax, r, ix, title in zip(ax_row[1:], results, index, titles):
            i = ix[p]
            ok = r["pred"][i] == r["labels"][i]
            _show(ax, overlay(r["images"][i], upsample(r["cams_pred"][i], r["size"])),
                  f"{title}\n{CLASSES[r['pred'][i]]} ({r['probs'][i, r['pred'][i]]:.0%})",
                  "darkgreen" if ok else "firebrick")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def occlusion_grid(path, r, rel_paths):
    """Ảnh gốc | xóa bạch cầu | xóa nền | trắng đen, kèm dự đoán của một mô hình."""
    plt = _plt()
    index = {p: i for i, p in enumerate(r["rel_paths"])}
    cols = [("Ảnh gốc", "pred"), ("Xóa bạch cầu", "pred_cells_erased"),
            ("Xóa nền", "pred_background_erased"), ("Trắng đen", "pred_gray")]
    fig, axes = plt.subplots(len(rel_paths), 4, figsize=(8.6, 2.4 * len(rel_paths)), squeeze=False)
    for ax_row, p in zip(axes, rel_paths):
        i = index[p]
        img, row = r["images"][i], r["rows"][i]
        md = dilate(cell_mask(img), 4)
        bg = background_color(img, md)
        variants = [img, erase(img, md, bg), erase(img, ~md, bg), to_gray(img)]
        for ax, v, (name, key) in zip(ax_row, variants, cols):
            ok = row[key] == row["label"]
            _show(ax, v, f"{name} (thật: {row['label']})\ndự đoán: {row[key]}", "darkgreen" if ok else "firebrick")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = [explain_run(find_runs(a.runs, name)[a.seed]) for name, _ in MODELS]
    titles = [t for _, t in MODELS]
    base = results[0]
    index = [{p: i for i, p in enumerate(r["rel_paths"])} for r in results]
    n_wrong = {p: sum(int(r["pred"][ix[p]] != r["labels"][ix[p]]) for r, ix in zip(results, index))
               for p in base["rel_paths"]}

    # ảnh đúng: mỗi lớp 1 ảnh mọi mô hình đều đúng, mặt nạ tách được, chọn ảnh ở giữa theo độ tự tin
    correct = []
    for c in range(len(CLASSES)):
        cand = [i for i, p in enumerate(base["rel_paths"])
                if base["labels"][i] == c and not n_wrong[p] and base["rows"][i]["mask_ok"]]
        cand.sort(key=lambda i: np.mean([r["probs"][ix[base["rel_paths"][i]]].max() for r, ix in zip(results, index)]))
        correct.append(base["rel_paths"][cand[len(cand) // 2]])
    wrong = sorted((p for p in base["rel_paths"] if n_wrong[p]), key=lambda p: (-n_wrong[p], p))

    cam_grid(out / "gradcam.png", results, titles, correct)
    cam_grid(out / "gradcam_errors.png", results, titles, wrong[:6])
    occlusion_grid(out / "gradcam_occlusion.png", results[1], correct)
    print("Ảnh đúng:", correct)
    print(f"Ảnh có ít nhất một trong 4 mô hình sai: {len(wrong)}; cả 4 cùng sai: {sum(n_wrong[p] == 4 for p in wrong)}")
    for p in wrong:
        i0 = index[0][p]
        print(" ", Path(p).name, "thật", CLASSES[base["labels"][i0]], "|",
              " · ".join(f"{t}: {CLASSES[r['pred'][ix[p]]]}" for t, r, ix in zip(titles, results, index)))
