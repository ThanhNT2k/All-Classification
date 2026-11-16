"""Chạy:  python scripts/split_data.py --data-root data/raw [--variant Original]"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from allcls.data.split import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chia train/val/test theo nhóm, chống rò rỉ")
    ap.add_argument("--data-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--variant", default=None, help="thư mục phiên bản ảnh, ví dụ Original")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patient-map", default=None, help="CSV cột key,patient_id (nếu có)")
    ap.add_argument("--keep-conflicts", action="store_true", help="giữ nhóm trùng mà khác nhãn")
    ap.add_argument("--cache", default=str(ROOT / "results" / "cache" / "scan_cache.csv"))
    ap.add_argument("--rescan", action="store_true", help="bỏ cache, quét lại toàn bộ")
    a = ap.parse_args()
    if a.rescan and Path(a.cache).exists():
        Path(a.cache).unlink()
    run(a.data_root, a.out, a.variant, tuple(a.ratios), a.seed, a.patient_map, a.keep_conflicts, a.cache)
