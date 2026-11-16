"""Chạy:  python scripts/audit_data.py --data-root data/raw"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from allcls.data.audit import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Kiểm kê bộ dữ liệu ảnh ALL")
    ap.add_argument("--data-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "audit"))
    ap.add_argument("--cache", default=str(ROOT / "results" / "cache" / "scan_cache.csv"))
    ap.add_argument("--rescan", action="store_true", help="bỏ cache, quét lại toàn bộ")
    a = ap.parse_args()
    if a.rescan and Path(a.cache).exists():
        Path(a.cache).unlink()
    run(a.data_root, a.out, a.cache)
