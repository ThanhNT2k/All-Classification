"""Kiểm thử phần không cần GPU: chỉ số đánh giá, file cấu hình, chuẩn hóa màu."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from allcls.evaluate import compute_metrics  # noqa: E402

HAS = {m: importlib.util.find_spec(m) is not None for m in ("yaml", "cv2", "sklearn")}


class TestMetrics(unittest.TestCase):
    def test_perfect(self):
        y = np.array([0, 1, 2, 3, 1, 2])
        m = compute_metrics(y, y, n_boot=100)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["macro_f1"], 1.0)
        self.assertEqual(m["screening"]["malignant_sensitivity"], 1.0)
        self.assertEqual(m["ci95"]["accuracy"], [1.0, 1.0])

    def test_screening_counts(self):
        # Benign=0. 2 ca ác tính bị đoán Benign (bỏ sót), 1 Benign bị đoán Pro.
        y_true = np.array([0, 0, 0, 1, 1, 2, 3, 3])
        y_pred = np.array([0, 0, 3, 0, 1, 2, 3, 0])
        m = compute_metrics(y_true, y_pred, n_boot=0)
        sc = m["screening"]
        self.assertEqual((sc["fn"], sc["fp"], sc["tp"], sc["tn"]), (2, 1, 3, 2))
        self.assertAlmostEqual(sc["malignant_sensitivity"], 3 / 5)
        self.assertAlmostEqual(sc["benign_specificity"], 2 / 3)
        # nhầm giữa các lớp ác tính (Early->Pre) vẫn tính là "phát hiện ác tính"
        m2 = compute_metrics(np.array([1, 2]), np.array([2, 1]), n_boot=0)
        self.assertEqual(m2["accuracy"], 0.0)
        self.assertEqual(m2["screening"]["malignant_sensitivity"], 1.0)

    @unittest.skipUnless(HAS["sklearn"], "cần scikit-learn")
    def test_against_sklearn(self):
        from sklearn.metrics import f1_score
        rng = np.random.default_rng(1)
        yt, yp = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
        m = compute_metrics(yt, yp, n_boot=0)
        self.assertAlmostEqual(m["macro_f1"], f1_score(yt, yp, average="macro"), places=10)

    def test_ci_contains_point(self):
        rng = np.random.default_rng(2)
        yt = rng.integers(0, 4, 400)
        yp = np.where(rng.random(400) < 0.8, yt, rng.integers(0, 4, 400))
        m = compute_metrics(yt, yp, n_boot=500)
        lo, hi = m["ci95"]["accuracy"]
        self.assertLessEqual(lo, m["accuracy"])
        self.assertGreaterEqual(hi, m["accuracy"])


@unittest.skipUnless(HAS["yaml"], "cần pyyaml")
class TestConfig(unittest.TestCase):
    def test_inheritance(self):
        from allcls.config import load_config
        cfg = load_config(ROOT / "configs" / "resnet50_gray.yaml")
        self.assertEqual(cfg["name"], "resnet50_gray")
        self.assertEqual(cfg["data"]["color_mode"], "gray")
        self.assertEqual(cfg["model"]["name"], "resnet50")       # kế thừa từ base
        cfg = load_config(ROOT / "configs" / "simple_cnn.yaml")
        self.assertEqual(cfg["train"]["epochs"], 60)             # ghi đè
        self.assertEqual(cfg["train"]["batch_size"], 32)         # kế thừa
        self.assertFalse(cfg["model"]["pretrained"])

    def test_registry_matches_configs(self):
        from allcls.experiments import EXPERIMENTS
        for name, *_ in EXPERIMENTS:
            self.assertTrue((ROOT / "configs" / f"{name}.yaml").exists(), name)

    def test_all_configs_load(self):
        from allcls.config import load_config
        for p in (ROOT / "configs").glob("*.yaml"):
            cfg = load_config(p)
            self.assertIn("model", cfg, p.name)


@unittest.skipUnless(HAS["cv2"], "cần opencv")
class TestStainNorm(unittest.TestCase):
    def test_reinhard_matches_target(self):
        from PIL import Image

        from allcls.data.transforms import fit_reinhard, lab_stats
        rng = np.random.default_rng(0)
        def img(tint):
            a = np.clip(rng.normal(150, 30, (64, 64, 3)) + tint, 0, 255).astype(np.uint8)
            return Image.fromarray(a)
        norm = fit_reinhard([img([0, 0, 0]) for _ in range(5)])
        shifted = img([60, -40, 40])                     # ảnh lệch màu mạnh
        before = np.abs(lab_stats(shifted)[0] - norm.target_mean).max()
        after = np.abs(lab_stats(norm(shifted))[0] - norm.target_mean).max()
        self.assertLess(after, 2.0)
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
