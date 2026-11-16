"""Đóng gói code + dữ liệu (ảnh Original + file chia tập) thành 1 file zip để tải lên Google Drive.

  python scripts/make_colab_bundle.py                       # -> colab_bundle.zip
  python scripts/make_colab_bundle.py --include-segmented   # thêm cả ảnh Segmented
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ["configs", "scripts", "src", "tests", "experiments", "notebooks", "data/splits"]
INCLUDE_FILES = ["README.md", "requirements.txt"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "colab_bundle.zip"))
    ap.add_argument("--include-segmented", action="store_true")
    a = ap.parse_args()
    files = [ROOT / f for f in INCLUDE_FILES]
    for d in INCLUDE_DIRS:
        files += [p for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    variants = ["Original"] + (["Segmented"] if a.include_segmented else [])
    images = [p for v in variants for p in sorted((ROOT / "data" / "raw" / v).rglob("*"))
              if p.suffix.lower() in IMAGE_EXTS]
    with zipfile.ZipFile(a.out, "w") as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix(), compress_type=zipfile.ZIP_DEFLATED)
        for p in images:   # ảnh JPEG đã nén sẵn -> lưu nguyên, nhanh hơn
            z.write(p, p.relative_to(ROOT).as_posix(), compress_type=zipfile.ZIP_STORED)
    size = Path(a.out).stat().st_size / 1e6
    print(f"Đã tạo {a.out}: {len(files)} file code/cấu hình + {len(images)} ảnh, {size:.0f} MB")
