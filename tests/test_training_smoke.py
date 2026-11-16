"""Chạy thử toàn bộ vòng huấn luyện trên dữ liệu GIẢ (1 epoch, mô hình nhỏ, không tải trọng số).

Chỉ chạy khi có torch + timm (ví dụ trên Colab). Mục đích: bắt lỗi code trước khi
tốn thời gian GPU cho thí nghiệm thật.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

NEEDED = ("torch", "torchvision", "timm", "yaml", "cv2", "matplotlib")
MISSING = [m for m in NEEDED if importlib.util.find_spec(m) is None]


@unittest.skipIf(MISSING, f"thiếu thư viện: {MISSING}")
class TestTrainingSmoke(unittest.TestCase):
    def test_variants(self):
        from allcls.data import split
        from allcls.train import run
        from make_fake_dataset import make

        with tempfile.TemporaryDirectory() as t:
            raw, splits = Path(t) / "raw", Path(t) / "splits"
            make(raw)
            split.run(raw, splits, variant="Original")
            base = {
                "data": {"root": str(raw), "splits": str(splits), "image_size": 64,
                         "color_mode": "rgb", "stain_norm": "none",
                         "cache_in_memory": True, "num_workers": 0},
                "augment": {"hflip": True, "vflip": True, "rotate90": True, "color_jitter": 0.4},
                "model": {"name": "resnet18", "pretrained": False, "drop_rate": 0.1},
                "train": {"epochs": 2, "batch_size": 8, "lr": 1e-3, "weight_decay": 1e-4,
                          "class_weights": True, "early_stopping_patience": 5, "amp": True, "seed": 0},
                "eval": {"bootstrap": 20},
            }
            variants = {
                "rgb": {},
                "gray": {"data": {"color_mode": "gray"}},
                "stain": {"data": {"stain_norm": "reinhard"}},
            }
            from allcls.config import deep_merge
            for name, over in variants.items():
                cfg = deep_merge(base, over)
                cfg["name"] = f"smoke_{name}"
                run_dir = run(cfg, Path(t) / "runs", seed=7)
                self.assertIn("_s7_", run_dir.name)
                for f in ("best.pt", "metrics.json", "report.md", "history.csv", "DONE",
                          "predictions_test.csv", "confusion_test.png", "env.json"):
                    self.assertTrue((run_dir / f).exists(), f"{name}: thiếu {f}")
                m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(m["test"]["n"], 10)
                self.assertIn("ci95", m["test"])
                eff = m["efficiency"]
                for k in ("params", "model_size_mb", "train_minutes", "latency_ms_batch1", "throughput_img_per_s"):
                    self.assertGreater(eff[k], 0, k)
                if name == "stain":
                    self.assertTrue((run_dir / "stain_target.json").exists())


if __name__ == "__main__":
    unittest.main()
