"""Bước 5 — Grad-CAM + phân tích ảnh sai + kiểm tra mô hình nhìn vào đâu.

  python scripts/explain.py --runs results/runs                 # mọi thí nghiệm đã chạy xong (seed 42)
  python scripts/explain.py --runs results/runs --only resnet50_rgb resnet50_gray
  python scripts/explain.py --run results/runs/resnet50_rgb_s42_20261101_120000

Kết quả: <lần chạy>/explain/summary.md + hình; nhiều lần chạy thì thêm <runs>/EXPLAIN.md.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from allcls.compare import find_runs  # noqa: E402
from allcls.experiments import EXPERIMENTS  # noqa: E402
from allcls.explain import compare_runs, explain_run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--runs", default=None, help="thư mục chứa các lần chạy (mặc định: results/runs)")
    g.add_argument("--run", default=None, help="một thư mục lần chạy cụ thể")
    ap.add_argument("--only", nargs="*", default=None, help="chỉ các config này (dùng với --runs)")
    ap.add_argument("--seed", type=int, default=42, help="seed của lần chạy cần giải thích")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--data-root", default=None, help="mặc định: lấy từ config của lần chạy")
    ap.add_argument("--splits", default=None)
    ap.add_argument("--max-errors", type=int, default=48, help="số ảnh sai tối đa đưa vào hình")
    a = ap.parse_args()

    if a.run:
        run_dirs, runs_root = [Path(a.run)], None
    else:
        runs_root = Path(a.runs or ROOT / "results" / "runs")
        run_dirs = []
        for name, code, _, _ in EXPERIMENTS:
            if a.only and name not in a.only:
                continue
            d = find_runs(runs_root, name).get(a.seed)
            if d is None:
                print(f"Bỏ qua {code} {name}: chưa có lần chạy hoàn chỉnh với seed {a.seed}")
            elif not (d / "best.pt").exists():
                print(f"Bỏ qua {code} {name}: thiếu best.pt trong {d}")
            else:
                run_dirs.append(d)
    if not run_dirs:
        sys.exit("Không có lần chạy nào để giải thích.")

    results = []
    for d in run_dirs:
        print(f"\n===== {d.name} =====", flush=True)
        r = explain_run(d, split=a.split, data_root=a.data_root, splits_dir=a.splits, max_errors=a.max_errors)
        s = r["summary"]
        al = s["groups"]["all"]
        print(f"sai {s['n_errors']}/{s['n']} | tỷ số tập trung vào tế bào {al['focus_ratio']} | "
              f"accuracy sau khi xóa tế bào {al['accuracy_cells_erased']} | {r['out'] / 'summary.md'}")
        results.append(r)
    if runs_root is not None and len(results) > 1:
        print(f"\nTổng hợp: {compare_runs(results, runs_root)}")
