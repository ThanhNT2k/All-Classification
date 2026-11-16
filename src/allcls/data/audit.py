"""Kiểm kê bộ dữ liệu ảnh: số ảnh mỗi lớp, kích thước, file hỏng, ảnh trùng.

Mọi con số trong phần "Dữ liệu" của báo cáo nên lấy từ output của file này.

Output (trong results/audit/<thời_gian>/):
  per_image.csv  — mỗi ảnh một dòng: nhãn, kích thước, md5, dhash, nhóm trùng
  audit.json     — số liệu tổng hợp (máy đọc)
  summary.md     — tóm tắt tiếng Việt (người đọc, chép vào báo cáo)
"""
from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from .discover import ImageRecord, discover
from .hashing import UnionFind, dhash, md5_bytes, near_duplicate_pairs

NEAR_DUP_DISTANCE = 4   # Hamming <= 4/64 bit coi là gần trùng


SCAN_FIELDS = ["rel_path", "label", "variant", "stem", "width", "height",
               "mode", "md5", "dhash", "error"]


def _scan_one(r: ImageRecord) -> dict:
    row = {"rel_path": r.rel_path, "label": r.label or "", "variant": r.variant,
           "stem": r.stem, "width": "", "height": "", "mode": "",
           "md5": "", "dhash": "", "error": ""}
    try:
        data = r.path.read_bytes()            # đọc file đúng 1 lần
        row["md5"] = md5_bytes(data)
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            row["width"], row["height"], row["mode"] = im.width, im.height, im.mode
            row["dhash"] = f"{dhash(im):016x}"
    except Exception as e:  # file hỏng / không phải ảnh
        msg = re.sub(r" at 0x[0-9a-f]+", "", str(e))   # bỏ địa chỉ bộ nhớ để kết quả ổn định
        row["error"] = f"{type(e).__name__}: {msg}"
    return row


def scan(records: list[ImageRecord], cache_path: str | Path | None = None,
         verbose: bool = True) -> list[dict]:
    """Mở từng ảnh, lấy thông tin + hash. Ảnh hỏng được đánh dấu, không làm dừng chương trình.

    cache_path: file CSV lưu kết quả đã quét (khóa = rel_path). Ghi dần từng dòng,
    nên nếu bị ngắt giữa chừng, lần chạy sau sẽ tiếp tục từ chỗ dừng.
    Xóa file cache (hoặc dùng --rescan) khi dữ liệu trong data/raw thay đổi.
    """
    cached = {}
    if cache_path and Path(cache_path).exists():
        with open(cache_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cached[row["rel_path"]] = row
    todo = [r for r in records if r.rel_path not in cached]
    if verbose:
        print(f"  {len(records) - len(todo)} ảnh lấy từ cache, cần quét {len(todo)} ảnh")

    fh = writer = None
    if cache_path and todo:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        new_file = not Path(cache_path).exists()
        fh = open(cache_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=SCAN_FIELDS)
        if new_file:
            writer.writeheader()
    try:
        for n, r in enumerate(todo, 1):
            row = _scan_one(r)
            cached[r.rel_path] = row
            if writer:
                writer.writerow(row)
                if n % 50 == 0:
                    fh.flush()
            if verbose and n % 500 == 0:
                print(f"  đã quét {n}/{len(todo)} ảnh")
    finally:
        if fh:
            fh.close()

    rows = []
    for r in records:                       # luôn theo thứ tự của records
        row = dict(cached[r.rel_path])
        row["label"], row["variant"], row["stem"] = r.label or "", r.variant, r.stem
        for k in ("width", "height"):
            row[k] = int(row[k]) if str(row[k]).isdigit() else row[k]
        rows.append(row)
    return rows


def reference_variant(variants) -> str:
    """Phiên bản dùng để dò ảnh gần trùng: ưu tiên ảnh gốc ("Original").

    Ảnh đã phân đoạn (nền đen chiếm phần lớn) làm dHash rất giống nhau giữa các
    ảnh không liên quan -> báo trùng giả. Các phiên bản khác đi theo nhóm của
    ảnh gốc nhờ trùng tên file (xem split.build_groups).
    """
    variants = sorted(set(variants))
    originals = [v for v in variants if "orig" in v.lower()]
    return originals[0] if originals else variants[0]


def find_duplicate_groups(rows: list[dict], max_distance: int = NEAR_DUP_DISTANCE,
                          near_variants=None):
    """Gom ảnh trùng (md5 giống hệt hoặc dHash gần nhau) thành nhóm.

    near_variants: tập phiên bản được dò gần trùng (None = tất cả); các phiên bản
    còn lại chỉ dò trùng tuyệt đối (md5).
    Chỉ so sánh trong cùng `variant`: ảnh Original và ảnh Segmented của cùng
    một mẫu đương nhiên giống nhau, việc ghép chúng do split.py xử lý theo `stem`.
    Trả về (danh sách cặp trùng, mảng group_id theo chỉ số dòng).
    """
    uf = UnionFind(len(rows))
    pairs = []
    by_variant = defaultdict(list)
    for i, row in enumerate(rows):
        if not row["error"]:
            by_variant[row["variant"]].append(i)

    for idxs in by_variant.values():
        by_md5 = defaultdict(list)
        for i in idxs:
            by_md5[rows[i]["md5"]].append(i)
        for same in by_md5.values():
            for j in same[1:]:
                uf.union(same[0], j)
                pairs.append((same[0], j, 0, "exact"))
        if near_variants is not None and rows[idxs[0]]["variant"] not in near_variants:
            continue                      # chỉ dò trùng tuyệt đối cho phiên bản này
        hashes = [int(rows[i]["dhash"], 16) for i in idxs]
        for a, b, d in near_duplicate_pairs(hashes, max_distance):
            i, j = idxs[a], idxs[b]
            if rows[i]["md5"] != rows[j]["md5"]:
                uf.union(i, j)
                pairs.append((i, j, d, "near"))
    group_ids = [uf.find(i) for i in range(len(rows))]
    return pairs, group_ids


def summarize(rows, pairs, group_ids, data_root: Path) -> dict:
    ok = [r for r in rows if not r["error"]]
    counts = defaultdict(Counter)
    for r in ok:
        counts[r["variant"] or "(gốc)"][r["label"] or "(không rõ lớp)"] += 1

    sizes = Counter((r["width"], r["height"]) for r in ok)
    modes = Counter(r["mode"] for r in ok)

    groups = defaultdict(list)
    for i, g in enumerate(group_ids):
        groups[g].append(i)
    dup_groups = [m for m in groups.values() if len(m) > 1]
    cross_class = [m for m in dup_groups if len({rows[i]["label"] for i in m}) > 1]

    by_variant = {}
    for v in sorted({r["variant"] for r in ok}):
        vp = [p for p in pairs if rows[p[0]]["variant"] == v]
        vg = [m for m in dup_groups if rows[m[0]]["variant"] == v]
        by_variant[v or "(gốc)"] = {
            "exact_pairs": sum(p[3] == "exact" for p in vp),
            "near_pairs": sum(p[3] == "near" for p in vp),
            "near_checked": any(p[3] == "near" for p in vp) or v == reference_variant([r["variant"] for r in ok]),
            "groups": len(vg),
            "images_in_groups": sum(len(m) for m in vg),
            "cross_class_groups": sum(len({rows[i]["label"] for i in m}) > 1 for m in vg),
        }

    return {
        "data_root": str(data_root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_files": len(rows),
        "readable_images": len(ok),
        "unreadable": [{"rel_path": r["rel_path"], "error": r["error"]} for r in rows if r["error"]],
        "unlabeled": [r["rel_path"] for r in ok if not r["label"]],
        "counts_by_variant_and_class": {v: dict(sorted(c.items())) for v, c in counts.items()},
        "image_sizes_top10": [{"width": w, "height": h, "count": n} for (w, h), n in sizes.most_common(10)],
        "distinct_sizes": len(sizes),
        "color_modes": dict(modes),
        "duplicate_pairs": {"exact": sum(p[3] == "exact" for p in pairs),
                            "near": sum(p[3] == "near" for p in pairs)},
        "duplicates_by_variant": by_variant,
        "duplicate_groups": len(dup_groups),
        "images_in_duplicate_groups": sum(len(m) for m in dup_groups),
        "cross_class_duplicate_groups": [
            [{"rel_path": rows[i]["rel_path"], "label": rows[i]["label"]} for i in m]
            for m in cross_class
        ],
        "near_dup_max_hamming": NEAR_DUP_DISTANCE,
    }


def write_summary_md(s: dict, path: Path) -> None:
    L = ["# Kiểm kê dữ liệu", "",
         f"- Thư mục: `{s['data_root']}`",
         f"- Thời điểm chạy: {s['generated_at']}",
         f"- Tổng số file ảnh: **{s['total_files']}**, đọc được: **{s['readable_images']}**, "
         f"hỏng: **{len(s['unreadable'])}**, không xác định lớp: **{len(s['unlabeled'])}**", "",
         "## Số ảnh theo lớp", ""]
    for variant, c in s["counts_by_variant_and_class"].items():
        total = sum(c.values())
        L += [f"**{variant}** — {total} ảnh", "", "| Lớp | Số ảnh | Tỷ lệ |", "|---|---:|---:|"]
        L += [f"| {k} | {v} | {v / total:.1%} |" for k, v in c.items()]
        L.append("")
    L += ["## Kích thước và màu", "",
          f"- Số kích thước khác nhau: {s['distinct_sizes']}",
          "- Phổ biến nhất: " + ", ".join(f"{x['width']}×{x['height']} ({x['count']})"
                                          for x in s["image_sizes_top10"][:5]),
          f"- Chế độ màu: {s['color_modes']}", "",
          "## Ảnh trùng lặp (so sánh trong cùng phiên bản)", "",
          "| Phiên bản | Cặp trùng tuyệt đối | Cặp gần trùng | Số nhóm | Ảnh trong nhóm | Nhóm khác nhãn |",
          "|---|---:|---:|---:|---:|---:|"]
    for v, d in s["duplicates_by_variant"].items():
        near = str(d["near_pairs"]) if d["near_checked"] else "không dò"
        L.append(f"| {v} | {d['exact_pairs']} | {near} | {d['groups']} | "
                 f"{d['images_in_groups']} | {d['cross_class_groups']} |")
    L += ["",
          f"- Gần trùng: dHash, Hamming ≤ {s['near_dup_max_hamming']}/64 bit; chỉ dò trên phiên bản ảnh gốc "
          "(ảnh phân đoạn có nền đen làm dHash báo trùng giả).",
          f"- **Tổng nhóm trùng nhưng KHÁC nhãn: {len(s['cross_class_duplicate_groups'])}**"
          + (" → cần kiểm tra thủ công, có thể là lỗi nhãn" if s["cross_class_duplicate_groups"] else ""),
          "",
          "> Ảnh trùng/gần trùng phải nằm cùng một tập (train/val/test). "
          "`split.py` đảm bảo điều này.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def run(data_root: str | Path, out_dir: str | Path,
        cache_path: str | Path | None = None) -> Path:
    data_root = Path(data_root).resolve()
    out = Path(out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    records = discover(data_root)
    if not records:
        raise SystemExit(f"Không có ảnh nào trong {data_root}")
    print(f"Tìm thấy {len(records)} file ảnh, đang quét...")
    rows = scan(records, cache_path)
    ref = reference_variant(r["variant"] for r in rows if not r["error"])
    pairs, group_ids = find_duplicate_groups(rows, near_variants={ref})
    for row, g in zip(rows, group_ids):
        row["dup_group"] = g

    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    s = summarize(rows, pairs, group_ids, data_root)
    (out / "audit.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(s, out / "summary.md")
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"Đã lưu kết quả vào: {out}")
    return out
