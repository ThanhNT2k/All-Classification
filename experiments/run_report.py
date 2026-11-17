"""Các thí nghiệm bổ sung cho báo cáo (ngoài run_all.py):
  - kịch bản dữ liệu: huấn luyện với 75% và 50% tập train (val/test giữ nguyên)
  - chiến lược học chuyển giao: từ đầu / trích đặc trưng / tinh chỉnh một phần (tinh chỉnh toàn bộ = run_all.py)

  python experiments/run_report.py --data-root ../data --num-workers 0
Chạy lại sẽ bỏ qua các thí nghiệm đã xong.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from allcls.compare import find_runs  # noqa: E402
from allcls.config import deep_merge, load_config  # noqa: E402
from allcls.train import run  # noqa: E402

MODELS = ["simple_cnn", "resnet50_rgb", "efficientnet_b0_rgb", "mobilenetv2_rgb"]
PRETRAINED = MODELS[1:]


def experiments():
    for frac in (0.75, 0.50):
        for m in MODELS:
            yield f"{m}_frac{int(frac * 100)}", m, {"data": {"train_fraction": frac}}
    for m in PRETRAINED:
        yield f"{m}_scratch", m, {"model": {"pretrained": False},
                                  "train": {"epochs": 60, "lr": 1e-3, "early_stopping_patience": 12}}
        yield f"{m}_featext", m, {"train": {"strategy": "feature_extraction", "lr": 1e-3}}
        yield f"{m}_partial", m, {"train": {"strategy": "partial"}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    for name, base, over in experiments():
        if find_runs(a.out, name).get(a.seed):
            print(f"Bỏ qua {name}: đã có")
            continue
        cfg = deep_merge(load_config(ROOT / "configs" / f"{base}.yaml"), over)
        cfg["name"] = name
        if a.data_root:
            cfg["data"]["root"] = str(Path(a.data_root).resolve())
        if a.num_workers is not None:
            cfg["data"]["num_workers"] = a.num_workers
        print(f"\n===== {name} =====", flush=True)
        run(cfg, a.out, seed=a.seed)
