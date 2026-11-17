"""Kiểm thử bước 5 (Grad-CAM) trên dữ liệu GIẢ. Chỉ chạy khi có torch + timm."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

NEEDED = ("torch", "torchvision", "timm", "yaml", "cv2", "matplotlib")
MISSING = [m for m in NEEDED if importlib.util.find_spec(m) is None]


def _cell_image(with_cell=True):
    """Nền sáng + 1 "tế bào" tím đậm ở góc trên trái."""
    img = np.full((96, 96, 3), (225, 215, 220), np.uint8)
    if with_cell:
        yy, xx = np.mgrid[:96, :96]
        img[(yy - 24) ** 2 + (xx - 24) ** 2 < 14 ** 2] = (90, 40, 150)
    return img


@unittest.skipIf(importlib.util.find_spec("cv2") is None, "thiếu cv2")
class TestCellMask(unittest.TestCase):
    def test_finds_stained_cell(self):
        from allcls.explain import background_color, cell_mask, dilate, erase
        img = _cell_image()
        m = cell_mask(img)
        self.assertIsNotNone(m)
        self.assertTrue(m[24, 24])
        self.assertFalse(m[70, 70])
        self.assertAlmostEqual(m.mean(), np.pi * 14 ** 2 / 96 ** 2, delta=0.02)
        erased = erase(img, dilate(m, 4), background_color(img, dilate(m, 4)))
        self.assertIsNone(cell_mask(erased))                 # xóa xong không còn tế bào
        np.testing.assert_array_equal(erased[24, 24], img[70, 70])

    def test_no_cell(self):
        from allcls.explain import cell_mask
        self.assertIsNone(cell_mask(_cell_image(with_cell=False)))


@unittest.skipIf(MISSING, f"thiếu thư viện: {MISSING}")
class TestGradCam(unittest.TestCase):
    def test_all_architectures(self):
        """Grad-CAM chạy được với cả 4 kiến trúc của đề tài và trả về bản đồ hợp lệ."""
        import torch
        from allcls.explain import grad_cam
        from allcls.models.factory import create_model

        torch.manual_seed(0)
        x = torch.randn(2, 3, 64, 64)
        for name in ("simple_cnn", "resnet50", "efficientnet_b0", "mobilenetv3_large_100"):
            model = create_model({"model": {"name": name, "pretrained": False}}, 4).eval()
            cam, probs = grad_cam(model, x)
            with torch.no_grad():
                ref = torch.softmax(model(x), 1).numpy()
            np.testing.assert_allclose(probs, ref, atol=1e-5, err_msg=name)   # 2 nửa = forward gốc
            self.assertEqual(cam.ndim, 3, name)
            self.assertGreaterEqual(cam.min(), 0, name)
            self.assertLessEqual(cam.max(), 1 + 1e-6, name)
            cam2, _ = grad_cam(model, x, torch.tensor([3, 0]))
            self.assertEqual(cam2.shape, cam.shape, name)

    def test_localizes_evidence(self):
        """Mô hình chỉ phản ứng với ô trên-trái -> Grad-CAM phải sáng ở đúng ô đó."""
        import torch
        from torch import nn
        from allcls.explain import grad_cam

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AvgPool2d(8)
                self.fc = nn.Linear(3, 4)

            def forward_features(self, x):
                return torch.relu(self.pool(x))

            def forward_head(self, f):
                return self.fc(f.mean((2, 3)))

        model = Toy()
        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.zero_()
            model.fc.weight[2, 0] = 1.0
        x = torch.zeros(1, 3, 32, 32)
        x[0, 0, :8, :8] = 5.0
        cam, probs = grad_cam(model, x)
        self.assertEqual(int(probs.argmax()), 2)
        self.assertEqual(np.unravel_index(cam[0].argmax(), cam[0].shape), (0, 0))
        self.assertAlmostEqual(float(cam[0].sum()), 1.0, places=5)

    def test_explain_run_end_to_end(self):
        from allcls.data import split
        from allcls.config import deep_merge
        from allcls.explain import compare_runs, explain_run
        from allcls.train import run
        from make_fake_dataset import make

        with tempfile.TemporaryDirectory() as t:
            raw, splits = Path(t) / "raw", Path(t) / "splits"
            make(raw)
            split.run(raw, splits, variant="Original")
            base = {
                "data": {"root": str(raw), "splits": str(splits), "image_size": 64, "color_mode": "rgb",
                         "stain_norm": "none", "cache_in_memory": True, "num_workers": 0},
                "augment": {"color_jitter": 0.1},
                "model": {"name": "resnet18", "pretrained": False, "drop_rate": 0.1},
                "train": {"epochs": 1, "batch_size": 8, "lr": 1e-3, "amp": False, "seed": 0},
                "eval": {"bootstrap": 10},
            }
            variants = {"a_rgb": {}, "b_stain_gray": {"data": {"stain_norm": "reinhard", "color_mode": "gray"}},
                        "c_cnn": {"model": {"name": "simple_cnn"}}}
            results = []
            for name, over in variants.items():
                cfg = deep_merge(base, over)
                cfg["name"] = name
                run_dir = run(cfg, Path(t) / "runs", seed=1)
                r = explain_run(run_dir, max_errors=4, per_class=1)
                results.append(r)
                s = r["summary"]
                self.assertEqual(s["n"], 10)
                self.assertEqual(s["saved_predictions_agree"], [10, 10], name)   # tái lập đúng dự đoán
                for f in ("summary.md", "summary.json", "per_image.csv", "correct_gradcam.png",
                          "mask_check.png", "cells_erased_check.png", "background_erased_check.png"):
                    self.assertTrue((r["out"] / f).exists(), f"{name}: thiếu {f}")
                if s["n_errors"]:
                    self.assertTrue((r["out"] / "errors_gradcam_1.png").exists())
                self.assertEqual(len(r["rows"]), 10)
            path = compare_runs(results, Path(t) / "runs")
            self.assertIn("a_rgb", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
