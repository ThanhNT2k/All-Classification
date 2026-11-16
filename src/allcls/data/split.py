"""Chia train/val/test PHÂN TẦNG THEO NHÓM để chống rò rỉ dữ liệu.

Một "nhóm" là tập ảnh bắt buộc nằm chung một tập:
  1. ảnh trùng tuyệt đối hoặc gần trùng (dHash) trong cùng phiên bản;
  2. cùng một ảnh ở các phiên bản khác nhau (ví dụ Original/x.jpg và Segmented/x.jpg);
  3. cùng bệnh nhân — CHỈ khi có file --patient-map (bộ Kaggle có thể không có mã BN).

Không có mã bệnh nhân thì (1)+(2) chỉ GIẢM rò rỉ, không loại bỏ hoàn toàn:
các ảnh khác nhau của cùng một người vẫn có thể rơi vào train và test.
Cần ghi điều này vào phần Hạn chế của báo cáo.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .. import CLASSES
from .audit import NEAR_DUP_DISTANCE, find_duplicate_groups, reference_variant, scan
from .discover import discover
from .hashing import UnionFind, near_duplicate_pairs

SPLITS = ("train", "val", "test")


def _pick_variant(variants: set[str], wanted: str | None) -> str:
    if wanted is not None:
        if wanted not in variants:
            raise SystemExit(f"Không có phiên bản '{wanted}'. Các phiên bản hiện có: {sorted(variants)}")
        return wanted
    if len(variants) == 1:
        return next(iter(variants))
    originals = [v for v in variants if "orig" in v.lower()]
    if len(originals) == 1:
        return originals[0]
    raise SystemExit(f"Có nhiều phiên bản ảnh {sorted(variants)}. Chọn một bằng --variant")


def _load_patient_map(path: str | Path) -> dict[str, str]:
    """CSV gồm 2 cột: key (rel_path hoặc tên file không đuôi), patient_id."""
    m = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m[row["key"]] = row["patient_id"]
    return m


def build_groups(rows, patient_map=None):
    ref = reference_variant(r["variant"] for r in rows)
    _, dup_ids = find_duplicate_groups(rows, near_variants={ref})
    uf = UnionFind(len(rows))
    for i, g in enumerate(dup_ids):
        uf.union(i, g)
    by_stem = defaultdict(list)                       # cùng ảnh, khác phiên bản
    for i, r in enumerate(rows):
        by_stem[(r["label"], r["stem"])].append(i)
    for idxs in by_stem.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)
    if patient_map:                                   # cùng bệnh nhân
        by_patient = defaultdict(list)
        for i, r in enumerate(rows):
            pid = patient_map.get(r["rel_path"]) or patient_map.get(r["stem"])
            if pid:
                by_patient[pid].append(i)
        for idxs in by_patient.values():
            for j in idxs[1:]:
                uf.union(idxs[0], j)
    return [uf.find(i) for i in range(len(rows))]


def assign_splits(groups: dict[int, list[int]], group_label: dict[int, str],
                  ratios: tuple[float, float, float], seed: int) -> dict[int, str]:
    """Gán từng nhóm vào tập đang "thiếu" nhiều nhất so với mục tiêu, làm riêng cho từng lớp."""
    rng = random.Random(seed)
    assignment = {}
    for label in CLASSES:
        gids = [g for g in groups if group_label[g] == label]
        rng.shuffle(gids)
        gids.sort(key=lambda g: len(groups[g]), reverse=True)   # nhóm lớn xếp trước (sort ổn định)
        total = sum(len(groups[g]) for g in gids)
        target = {s: r * total for s, r in zip(SPLITS, ratios)}
        current = dict.fromkeys(SPLITS, 0)
        for g in gids:
            s = max(SPLITS, key=lambda s: (target[s] - current[s], -SPLITS.index(s)))
            assignment[g] = s
            current[s] += len(groups[g])
    return assignment


def verify(rows, split_of_row, variant) -> dict:
    """Kiểm tra độc lập: không có ảnh trùng/gần trùng nào nằm ở hai tập khác nhau.

    Trùng tuyệt đối: kiểm trên phiên bản được chọn. Gần trùng: kiểm trên phiên bản
    tham chiếu (ảnh gốc), vì các phiên bản khác đi cùng tập với ảnh gốc theo tên file.
    """
    idx = [i for i, r in enumerate(rows) if r["variant"] == variant and split_of_row.get(i)]
    md5_splits = defaultdict(set)
    for i in idx:
        md5_splits[rows[i]["md5"]].add(split_of_row[i])
    exact_leaks = sum(len(s) > 1 for s in md5_splits.values())
    ref = reference_variant(r["variant"] for r in rows)
    idx = [i for i, r in enumerate(rows) if r["variant"] == ref and split_of_row.get(i)]
    hashes = [int(rows[i]["dhash"], 16) for i in idx]
    near_leaks = [(rows[idx[a]]["rel_path"], rows[idx[b]]["rel_path"], d)
                  for a, b, d in near_duplicate_pairs(hashes, NEAR_DUP_DISTANCE)
                  if split_of_row[idx[a]] != split_of_row[idx[b]]]
    return {"exact_duplicate_leaks": exact_leaks,
            "near_duplicate_leaks": len(near_leaks),
            "near_duplicate_leak_examples": near_leaks[:10],
            "passed": exact_leaks == 0 and not near_leaks}


def run(data_root, out_dir, variant=None, ratios=(0.70, 0.15, 0.15), seed=42,
        patient_map_path=None, keep_conflicts=False, cache_path=None) -> dict:
    if abs(sum(ratios) - 1) > 1e-6:
        raise SystemExit("Tổng tỷ lệ train/val/test phải bằng 1")
    data_root = Path(data_root).resolve()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = discover(data_root)
    print(f"Tìm thấy {len(records)} file ảnh, đang quét...")
    all_rows = scan(records, cache_path)
    rows = [r for r in all_rows if not r["error"] and r["label"]]
    skipped = len(all_rows) - len(rows)
    variant = _pick_variant({r["variant"] for r in rows}, variant)
    patient_map = _load_patient_map(patient_map_path) if patient_map_path else None

    group_of = build_groups(rows, patient_map)
    groups = defaultdict(list)
    for i, g in enumerate(group_of):
        groups[g].append(i)

    # Nhãn của nhóm = nhãn đa số; nhóm có nhãn mâu thuẫn bị loại (trừ khi --keep-conflicts)
    group_label, conflicts = {}, []
    for g, members in groups.items():
        c = Counter(rows[i]["label"] for i in members)
        group_label[g] = c.most_common(1)[0][0]
        if len(c) > 1:
            conflicts.append([rows[i]["rel_path"] for i in members])
    if not keep_conflicts:
        for g in [g for g, m in groups.items() if len({rows[i]["label"] for i in m}) > 1]:
            del groups[g]

    assignment = assign_splits(groups, group_label, ratios, seed)
    split_of_row = {i: assignment[g] for g, members in groups.items() for i in members}

    # Ghi file: train/val/test.csv cho phiên bản được chọn + bảng gán cho mọi phiên bản
    label_idx = {c: k for k, c in enumerate(CLASSES)}
    fields = ["rel_path", "label", "label_idx", "group"]
    writers = {}
    files = {s: open(out / f"{s}.csv", "w", newline="", encoding="utf-8") for s in SPLITS}
    try:
        for s, f in files.items():
            writers[s] = csv.DictWriter(f, fieldnames=fields)
            writers[s].writeheader()
        for i, r in enumerate(rows):
            if r["variant"] == variant and i in split_of_row:
                writers[split_of_row[i]].writerow({
                    "rel_path": r["rel_path"], "label": r["label"],
                    "label_idx": label_idx[r["label"]], "group": group_of[i]})
    finally:
        for f in files.values():
            f.close()
    with open(out / "split_all_variants.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "variant", "label", "group", "split"])
        for i, r in enumerate(rows):
            w.writerow([r["rel_path"], r["variant"], r["label"], group_of[i], split_of_row.get(i, "excluded")])

    dist = {s: Counter() for s in SPLITS}
    for i, s in split_of_row.items():
        if rows[i]["variant"] == variant:
            dist[s][rows[i]["label"]] += 1
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root), "variant": variant, "seed": seed,
        "ratios": dict(zip(SPLITS, ratios)),
        "grouping": "duplicates + same-stem-across-variants" + (" + patient_id" if patient_map else ""),
        "skipped_unreadable_or_unlabeled": skipped,
        "label_conflict_groups": len(conflicts),
        "label_conflicts_excluded": not keep_conflicts,
        "n_groups": len(groups),
        # nhóm có >= 2 ảnh TRONG phiên bản được chọn (không tính cặp Original/Segmented)
        "n_multi_image_groups": sum(
            sum(rows[i]["variant"] == variant for i in m) > 1 for m in groups.values()),
        "distribution": {s: {c: dist[s].get(c, 0) for c in CLASSES} | {"total": sum(dist[s].values())}
                         for s in SPLITS},
        "leakage_check": verify(rows, split_of_row, variant),
    }
    (out / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(report, out / "split_report.md")
    print((out / "split_report.md").read_text(encoding="utf-8"))
    if not report["leakage_check"]["passed"]:
        raise SystemExit("KIỂM TRA RÒ RỈ THẤT BẠI — xem split_report.json")
    return report


def _write_md(r: dict, path: Path) -> None:
    L = ["# Báo cáo chia tập dữ liệu", "",
         f"- Phiên bản ảnh: `{r['variant'] or '(gốc)'}` · seed: {r['seed']} · "
         f"tỷ lệ: {r['ratios']['train']:.0%}/{r['ratios']['val']:.0%}/{r['ratios']['test']:.0%}",
         f"- Gom nhóm theo: {r['grouping']}",
         f"- Số nhóm: {r['n_groups']} (trong đó {r['n_multi_image_groups']} nhóm có ≥ 2 ảnh)",
         f"- Bỏ qua (hỏng/không rõ lớp): {r['skipped_unreadable_or_unlabeled']} · "
         f"nhóm mâu thuẫn nhãn: {r['label_conflict_groups']}"
         + (" (đã loại)" if r["label_conflicts_excluded"] else " (giữ theo nhãn đa số)"), "",
         "| Tập | " + " | ".join(CLASSES) + " | Tổng |",
         "|---|" + "---:|" * (len(CLASSES) + 1)]
    for s in SPLITS:
        d = r["distribution"][s]
        L.append(f"| {s} | " + " | ".join(str(d[c]) for c in CLASSES) + f" | {d['total']} |")
    lc = r["leakage_check"]
    L += ["", "## Kiểm tra rò rỉ", "",
          f"- Ảnh trùng tuyệt đối nằm ở 2 tập: {lc['exact_duplicate_leaks']}",
          f"- Cặp gần trùng nằm ở 2 tập: {lc['near_duplicate_leaks']}",
          f"- Kết quả: **{'ĐẠT' if lc['passed'] else 'KHÔNG ĐẠT'}**", ""]
    if "patient_id" not in r["grouping"]:
        L.append("> Chưa gom theo bệnh nhân (không có mã BN). Ảnh khác nhau của cùng một người "
                 "vẫn có thể nằm ở cả train và test → nêu trong phần Hạn chế.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
