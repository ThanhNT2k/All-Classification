"""Chạy:  python scripts/train.py --config configs/resnet50_rgb.yaml [--out DIR] [--smoke]"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from allcls.train import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--smoke", action="store_true", help="chạy thử 1 epoch, 3 batch để bắt lỗi")
    a = ap.parse_args()
    run(a.config, a.out, smoke=a.smoke)
