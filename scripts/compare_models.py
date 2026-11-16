"""Tạo báo cáo so sánh từ các lần chạy đã có.

  python scripts/compare_models.py --runs results/runs --color-baseline results/color_baseline
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from allcls.compare import build_report  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--color-baseline", default=None)
    ap.add_argument("--out", default=None, help="mặc định: cùng thư mục --runs")
    a = ap.parse_args()
    print(build_report(a.runs, a.color_baseline, a.out))
