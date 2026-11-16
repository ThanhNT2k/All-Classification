"""Huấn luyện một thí nghiệm theo file cấu hình.

Mỗi lần chạy tạo thư mục results/runs/<tên>_<thời gian>/ gồm:
  config.yaml        cấu hình đầy đủ đã dùng
  env.json           phiên bản thư viện, GPU, mã băm các file chia tập
  history.csv        loss / accuracy / macro-F1 theo epoch
  best.pt            trọng số tốt nhất (theo macro-F1 trên val)
  predictions_{val,test}.csv, metrics.json, report.md, confusion_{val,test}.png
  DONE               đánh dấu đã chạy xong (run_all.py dùng để bỏ qua khi chạy lại)
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import CLASSES
from .config import load_config
from .evaluate import (compute_metrics, metrics_markdown, plot_confusion,
                       save_predictions, write_json)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _grad_scaler(enabled: bool):
    import torch
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):          # torch cũ
        return torch.cuda.amp.GradScaler(enabled=enabled)


def predict(model, loader, device, amp: bool, max_batches: int | None = None):
    import torch
    model.eval()
    probs, labels, idxs = [], [], []
    with torch.no_grad():
        for b, (x, y, i) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            labels.append(y.numpy())
            idxs.append(i.numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(idxs)


def _plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train acc")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val acc")
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], label="val macro-F1")
    ax[1].set_title("Accuracy / F1")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def benchmark_inference(model, device, image_size: int, amp: bool) -> dict:
    """Đo tốc độ suy luận: độ trễ 1 ảnh (batch=1) và thông lượng (batch=64)."""
    import torch
    model.eval()

    def timed(batch, reps):
        x = torch.randn(batch, 3, image_size, image_size, device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
            for _ in range(5):                       # khởi động
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    lat = timed(1, 30)
    thr_batch = 64 if device.type == "cuda" else 8
    per_batch = timed(thr_batch, 5)
    return {"latency_ms_batch1": round(lat * 1000, 2),
            "throughput_img_per_s": round(thr_batch / per_batch, 1),
            "benchmark_device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}


def run(config, out_root="results/runs", smoke: bool = False, seed: int | None = None) -> Path:
    """config: đường dẫn file YAML hoặc dict. smoke=True: chạy thử vài batch để bắt lỗi.
    seed: ghi đè train.seed (dùng khi lặp lại thí nghiệm với nhiều seed)."""
    import torch
    import yaml
    from torch import nn
    from torch.utils.data import DataLoader

    from .data.dataset import SplitDataset
    from .data.transforms import build_transforms, fit_reinhard
    from .models.factory import create_model

    cfg = load_config(config) if not isinstance(config, dict) else json.loads(json.dumps(config))
    cfg.setdefault("name", "run")
    d, tr = cfg["data"], cfg["train"]
    if seed is not None:
        tr["seed"] = int(seed)
    seed = int(tr.get("seed", 42))
    if smoke:
        tr["epochs"] = 1
        cfg.setdefault("eval", {})["bootstrap"] = 50
    limit = 3 if smoke else None

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(tr.get("amp", True)) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_root, splits = _resolve(d["root"]), _resolve(d["splits"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _resolve(out_root) / f"{cfg['name']}{'_smoke' if smoke else ''}_s{seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                         encoding="utf-8")

    # ----- dữ liệu -----
    cache = bool(d.get("cache_in_memory", False))
    stain = None
    if d.get("stain_norm", "none") == "reinhard":
        ref = SplitDataset(splits / "train.csv", data_root, cache_in_memory=False)
        rng = random.Random(0)
        sample = rng.sample(range(len(ref)), min(500, len(ref)))
        stain = fit_reinhard(ref.load_image(i) for i in sample)      # chỉ dùng ảnh TRAIN
        write_json(stain.to_dict(), run_dir / "stain_target.json")

    ds = {s: SplitDataset(splits / f"{s}.csv", data_root,
                          build_transforms(cfg, train=(s == "train"), stain_normalizer=stain),
                          cache_in_memory=cache)
          for s in ("train", "val", "test")}
    nw = int(d.get("num_workers", 2))
    bs = int(tr.get("batch_size", 32))
    loader_kw = dict(num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    g = torch.Generator()
    g.manual_seed(seed)
    loaders = {"train": DataLoader(ds["train"], batch_size=bs, shuffle=True, drop_last=True,
                                   generator=g, **loader_kw)}
    for s in ("val", "test"):
        loaders[s] = DataLoader(ds[s], batch_size=bs * 2, shuffle=False, **loader_kw)

    write_json({
        "started": stamp, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__, "python": platform.python_version(),
        "timm": __import__("timm").__version__,
        "split_sha256": {s: _sha256(splits / f"{s}.csv") for s in ("train", "val", "test")},
        "n_images": {s: len(ds[s]) for s in ds},
        "smoke": smoke,
    }, run_dir / "env.json")

    # ----- mô hình, loss, tối ưu -----
    model = create_model(cfg, len(CLASSES)).to(device)
    counts = np.bincount(ds["train"].labels, minlength=len(CLASSES))
    weight = None
    if tr.get("class_weights", True):
        weight = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)),
                              dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=float(tr.get("label_smoothing", 0)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]),
                            weight_decay=float(tr.get("weight_decay", 0)))
    epochs = int(tr["epochs"])
    steps_per_epoch = min(len(loaders["train"]), limit or 10**9)
    total, warmup = epochs * steps_per_epoch, steps_per_epoch      # warmup 1 epoch rồi cosine
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    scaler = _grad_scaler(amp)

    # ----- vòng huấn luyện -----
    n_params = sum(p.numel() for p in model.parameters())
    train_start = time.time()
    best_f1, best_epoch, bad = -1.0, 0, 0
    patience = int(tr.get("early_stopping_patience", 7))
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot_loss, correct, seen = 0.0, 0, 0
        for b, (x, y, _) in enumerate(loaders["train"]):
            if limit is not None and b >= limit:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        probs, labels, _ = predict(model, loaders["val"], device, amp, limit)
        val_loss = float(nn.functional.cross_entropy(torch.from_numpy(np.log(probs + 1e-9)),
                                                     torch.from_numpy(labels).long()))
        m = compute_metrics(labels, probs.argmax(1), n_boot=0)
        row = {"epoch": epoch, "train_loss": tot_loss / max(seen, 1), "train_acc": correct / max(seen, 1),
               "val_loss": val_loss, "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"],
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        history.append(row)
        improved = m["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, bad = m["macro_f1"], epoch, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad += 1
        print(f"[{cfg['name']}] epoch {epoch:2d}/{epochs} | loss {row['train_loss']:.4f} "
              f"acc {row['train_acc']:.3f} | val loss {val_loss:.4f} acc {m['accuracy']:.3f} "
              f"F1 {m['macro_f1']:.3f}{' *' if improved else ''} | {row['seconds']}s", flush=True)
        if bad >= patience:
            print(f"Dừng sớm: macro-F1 val không tăng sau {patience} epoch.")
            break

    train_seconds = time.time() - train_start
    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    _plot_history(history, run_dir / "history.png")

    # ----- đánh giá cuối với trọng số tốt nhất -----
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    n_boot = int(cfg.get("eval", {}).get("bootstrap", 1000))
    results = {"best_epoch": best_epoch, "epochs_run": len(history), "seed": seed,
               "model": cfg["model"]["name"],
               "efficiency": {
                   "params": int(n_params),
                   "model_size_mb": round((run_dir / "best.pt").stat().st_size / 1e6, 1),
                   "train_minutes": round(train_seconds / 60, 2),
                   "sec_per_epoch": round(train_seconds / max(len(history), 1), 1),
                   **benchmark_inference(model, device, int(d.get("image_size", 224)), amp),
               }}
    md = [f"# {cfg['name']}", "", f"Epoch tốt nhất: {best_epoch}/{len(history)} (theo macro-F1 trên val)", ""]
    for s in ("val", "test"):
        probs, labels, idx = predict(model, loaders[s], device, amp, limit)
        pred = probs.argmax(1)
        save_predictions(run_dir / f"predictions_{s}.csv",
                         [ds[s].rel_paths[i] for i in idx], labels, pred, probs)
        m = compute_metrics(labels, pred, probs, n_boot=n_boot)
        results[s] = m
        plot_confusion(m["confusion_matrix"], run_dir / f"confusion_{s}.png", f"{cfg['name']} — {s}")
        md.append(metrics_markdown(m, s.upper()))
    write_json(results, run_dir / "metrics.json")
    (run_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    (run_dir / "DONE").write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
    t = results["test"]
    print(f"[{cfg['name']}] TEST acc {t['accuracy']:.3f} | macro-F1 {t['macro_f1']:.3f} | "
          f"độ nhạy ác tính {t['screening']['malignant_sensitivity']} | lưu tại {run_dir}")
    return run_dir
