"""CNN tự xây dựng — mốc so sánh với các mô hình pretrained.

Kiến trúc (đầu vào 3×224×224):
  4 khối [Conv3×3–BN–ReLU] ×2 → MaxPool 2×2, số kênh 32 → 64 → 128 → 256
  → Global Average Pooling → Dropout → Linear(256 → số lớp)
Khoảng 1,2 triệu tham số, huấn luyện từ đầu.
"""
import torch
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, channels=(32, 64, 128, 256), drop_rate: float = 0.3):
        super().__init__()
        layers, c_in = [], 3
        for c in channels:
            layers.append(_block(c_in, c))
            c_in = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(c_in, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    # Tách 2 nửa giống quy ước của timm để Grad-CAM (explain.py) dùng chung cho mọi mô hình.
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward_head(self, feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.pool(feats).flatten(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))
