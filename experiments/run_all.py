"""Chạy các thí nghiệm rồi tạo báo cáo so sánh (results/runs/COMPARISON.md).

  python experiments/run_all.py --out DIR --skip-done                      # tất cả, seed 42
  python experiments/run_all.py --out DIR --skip-done --group model        # chỉ 4 mô hình
  python experiments/run_all.py --out DIR --skip-done --seeds 42 43 44 --group model
  python experiments/run_all.py --out DIR --compare-only --color-baseline DIR2

--skip-done: bỏ qua thí nghiệm (theo từng seed) đã có kết quả hoàn chỉnh — dùng khi Colab bị
ngắt giữa chừng, chạy lại sẽ tiếp tục phần còn thiếu.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from allcls.compare import build_report, find_runs  # noqa: E402
from allcls.experiments import EXPERIMENTS  # noqa: E402
from allcls.train import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--group", choices=["model", "color", "all"], default="all")
    ap.add_argument("--only", nargs="*", default=None, help="chỉ chạy các config này")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--color-baseline", default=None, help="thư mục kết quả color_baseline")
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.compare_only:
        for name, code, title, group in EXPERIMENTS:
            if a.group != "all" and group != a.group:
                continue
            if a.only and name not in a.only:
                continue
            for seed in a.seeds:
                done = find_runs(out, name).get(seed)
                if a.skip_done and done:
                    print(f"Bỏ qua {code} {name} seed {seed}: đã có ({done.name})")
                    continue
                print(f"\n===== {code} {title} — seed {seed} =====", flush=True)
                run(ROOT / "configs" / f"{name}.yaml", out, seed=seed)
    print(build_report(out, a.color_baseline))
