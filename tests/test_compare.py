"""Kiểm thử báo cáo so sánh bằng các lần chạy GIẢ (không cần GPU)."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from allcls.compare import _roc, build_report, find_runs, holm, mcnemar  # noqa: E402
from allcls.evaluate import compute_metrics, save_predictions, write_json  # noqa: E402

HAS_MPL = importlib.util.find_spec("matplotlib") is not None
N = 300
RNG = np.random.default_rng(0)
Y = RNG.integers(0, 4, N)
PATHS = [f"Original/x/img_{i:04d}.jpg" for i in range(N)]


def fake_run(root: Path, dirname: str, acc: float, seed: int, params=None):
    rng = np.random.default_rng(seed + int(acc * 1000))
    pred = np.where(rng.random(N) < acc, Y, (Y + rng.integers(1, 4, N)) % 4)
    probs = np.full((N, 4), 0.02)
    probs[np.arange(N), pred] = 0.94
    d = root / dirname
    d.mkdir(parents=True)
    save_predictions(d / "predictions_test.csv", PATHS, Y, pred, probs)
    m = {"test": compute_metrics(Y, pred, probs, n_boot=200), "seed": seed}
    if params:
        m["efficiency"] = {"params": params, "model_size_mb": params * 4e-6, "train_minutes": 5.0,
                           "sec_per_epoch": 20.0, "latency_ms_batch1": 8.0, "throughput_img_per_s": 900.0,
                           "benchmark_device": "fake"}
    write_json(m, d / "metrics.json")
    (d / "DONE").write_text("x")
    return d


class TestStats(unittest.TestCase):
    def test_mcnemar_exact(self):
        a = np.array([True] * 10 + [False] * 2 + [True] * 50)
        b = np.array([False] * 10 + [True] * 2 + [True] * 50)
        t = mcnemar(a, b)
        self.assertEqual((t["a_only"], t["b_only"]), (10, 2))
        self.assertAlmostEqual(t["p"], 2 * (1 + 12 + 66) / 4096)      # = 0,0386
        self.assertEqual(mcnemar(a, a)["p"], 1.0)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "cần scikit-learn")
    def test_roc_auc_with_ties(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 400)
        score = np.round(y * 0.3 + rng.random(400), 1)        # nhiều điểm trùng
        self.assertAlmostEqual(_roc(y, score)[2], roc_auc_score(y, score), places=10)

    def test_holm(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


class TestReport(unittest.TestCase):
    def test_full_report(self):
        with tempfile.TemporaryDirectory() as t:
            runs = Path(t) / "runs"
            fake_run(runs, "simple_cnn_s42_20260101_000000", 0.80, 42, params=1.2e6)
            fake_run(runs, "resnet50_rgb_s42_20260101_000000", 0.97, 42, params=23.5e6)
            fake_run(runs, "resnet50_rgb_s43_20260101_000000", 0.96, 43, params=23.5e6)
            fake_run(runs, "efficientnet_b0_rgb_s42_20260101_000000", 0.96, 42, params=4.0e6)
            fake_run(runs, "mobilenetv3_rgb_s42_20260101_000000", 0.95, 42, params=4.2e6)
            fake_run(runs, "resnet50_gray_s42_20260101_000000", 0.70, 42, params=23.5e6)
            # lần chạy chưa xong (không có DONE) phải bị bỏ qua
            (runs / "resnet50_strong_jitter_s42_20260101_000000").mkdir()
            cb = fake_run(Path(t), "color_baseline", 0.98, 42, params=228)

            self.assertEqual(sorted(find_runs(runs, "resnet50_rgb")), [42, 43])
            self.assertEqual(find_runs(runs, "resnet50_strong_jitter"), {})

            text = build_report(runs, cb)
            for s in ("Bảng 1", "Bảng 3", "Bảng 4", "Bảng 5", "CNN tự xây dựng", "± ",
                      "Chưa có kết quả", "C2 ResNet50 – nhiễu màu mạnh"):
                self.assertIn(s, text)
            mc = json.loads((runs / "mcnemar.json").read_text(encoding="utf-8"))
            pair = {(r["a"], r["b"]): r for r in mc}
            self.assertLess(pair[("simple_cnn", "resnet50_rgb")]["p_holm"], 0.05)   # 80% vs 97%
            self.assertIn(("resnet50_rgb", "resnet50_gray"), pair)                  # so sánh màu
            # 5 mô hình nhóm chính (E0..E4) -> 10 cặp, + 1 cặp ResNet50 vs ảnh xám
            self.assertEqual(len(mc), 11)
            rows = (runs / "comparison.csv").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 1 + 6)
            if HAS_MPL:
                for f in ("fig_accuracy_f1.png", "fig_roc_malignant.png", "fig_accuracy_vs_params.png"):
                    self.assertTrue((runs / f).exists(), f)


if __name__ == "__main__":
    unittest.main()
