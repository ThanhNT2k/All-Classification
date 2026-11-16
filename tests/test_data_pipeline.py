"""Chạy:  python -m unittest discover -s tests -v"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from allcls.data import audit, split  # noqa: E402
from allcls.data.discover import class_from_name  # noqa: E402
from make_fake_dataset import make  # noqa: E402


class TestClassNames(unittest.TestCase):
    def test_mapping(self):
        cases = {"Benign": "Benign", "[Malignant] early Pre-B": "Early", "Early": "Early",
                 "Pre": "Pre", "Pre-B": "Pre", "Pro": "Pro", "Pro-B": "Pro",
                 "Original": None, "Segmented": None, "processed": None, "misc": None}
        for name, want in cases.items():
            self.assertEqual(class_from_name(name), want, name)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name) / "raw"
        cls.expected = make(cls.data)
        cls.audit_dir = audit.run(cls.data, Path(cls.tmp.name) / "audit")
        cls.summary = json.loads((cls.audit_dir / "audit.json").read_text(encoding="utf-8"))
        cls.split_dir = Path(cls.tmp.name) / "splits"
        cls.report = split.run(cls.data, cls.split_dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _read(self, name):
        with open(self.split_dir / name, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_audit_counts(self):
        s = self.summary
        self.assertEqual(len(s["unreadable"]), 1)
        self.assertEqual(len(s["unlabeled"]), 1)
        self.assertEqual(s["counts_by_variant_and_class"]["Segmented"],
                         {"Benign": 12, "Early": 20, "Pre": 18, "Pro": 15})
        self.assertEqual(s["duplicate_pairs"]["exact"], self.expected["exact_pairs"] + 1)  # +1 ảnh nhãn sai
        self.assertGreaterEqual(s["duplicate_pairs"]["near"], self.expected["near_pairs"])
        self.assertEqual(len(s["cross_class_duplicate_groups"]), self.expected["cross_class"])
        # ảnh phân đoạn không được dò gần trùng (tránh báo trùng giả do nền đen)
        self.assertTrue(s["duplicates_by_variant"]["Original"]["near_checked"])
        self.assertFalse(s["duplicates_by_variant"]["Segmented"]["near_checked"])

    def test_no_leakage(self):
        self.assertTrue(self.report["leakage_check"]["passed"])

    def test_duplicates_same_split(self):
        rows = {r["rel_path"]: s for s in ("train", "val", "test") for r in self._read(f"{s}.csv")}
        for cls in ("Benign", "Early", "Pre", "Pro"):
            base = f"Original/{cls}/{cls}_00"
            self.assertEqual(rows[base + "0.png"], rows[base + "0_copy.png"])
            self.assertEqual(rows[base + "1.png"], rows[base + "1_resized.jpg"])

    def test_variants_same_split(self):
        alls = {r["rel_path"]: r["split"] for r in self._read("split_all_variants.csv")}
        for path, s in alls.items():
            if path.startswith("Original/") and "_copy" not in path and "_resized" not in path:
                twin = path.replace("Original/", "Segmented/", 1)
                if twin in alls:
                    self.assertEqual(s, alls[twin], path)

    def test_conflict_excluded(self):
        self.assertEqual(self.report["label_conflict_groups"], 1)
        alls = {r["rel_path"]: r["split"] for r in self._read("split_all_variants.csv")}
        self.assertEqual(alls["Original/Pro/Pre_002_mislabeled.png"], "excluded")
        self.assertEqual(alls["Original/Pre/Pre_002.png"], "excluded")

    def test_ratios_and_coverage(self):
        d = self.report["distribution"]
        total = sum(d[s]["total"] for s in ("train", "val", "test"))
        self.assertGreater(total, 0)
        self.assertAlmostEqual(d["train"]["total"] / total, 0.70, delta=0.08)
        for s in ("train", "val", "test"):
            for c in ("Benign", "Early", "Pre", "Pro"):
                self.assertGreater(d[s][c], 0, f"{s}/{c} rỗng")

    def test_deterministic(self):
        again = split.run(self.data, Path(self.tmp.name) / "splits2")
        self.assertEqual(again["distribution"], self.report["distribution"])


if __name__ == "__main__":
    unittest.main()
