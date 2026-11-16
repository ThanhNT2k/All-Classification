"""Chạy:
  python scripts/color_baseline.py                      # trích đặc trưng + huấn luyện + đánh giá
  python scripts/color_baseline.py --extract-only       # chỉ trích đặc trưng (không cần sklearn)
  python scripts/color_baseline.py --from-npz FILE.npz  # bỏ qua bước trích đặc trưng
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from allcls.color_baseline import extract, fit_eval  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--splits", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--out", default=str(ROOT / "results" / "color_baseline"))
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--from-npz", default=None)
    a = ap.parse_args()
    npz = a.from_npz or extract(a.data_root, a.splits, Path(a.out) / "color_features.npz")
    if not a.extract_only:
        res = fit_eval(npz, a.out)
        print((Path(a.out) / "report.md").read_text(encoding="utf-8"))
