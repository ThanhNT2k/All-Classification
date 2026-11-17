"""Danh sách thí nghiệm của đề tài (dùng chung cho run_all.py và compare.py)."""

# (tên config, mã, tên hiển thị, nhóm)
EXPERIMENTS = [
    ("simple_cnn", "E1", "CNN tự xây dựng", "model"),
    ("resnet50_rgb", "E2", "ResNet50", "model"),
    ("efficientnet_b0_rgb", "E3", "EfficientNet-B0", "model"),
    ("mobilenetv3_rgb", "E4", "MobileNetV3-Large", "model"),
    ("mobilenetv2_rgb", "E5", "MobileNetV2", "model"),
    ("resnet50_gray", "C1", "ResNet50 – ảnh xám", "color"),
    ("resnet50_strong_jitter", "C2", "ResNet50 – nhiễu màu mạnh", "color"),
    ("resnet50_stainnorm", "C3", "ResNet50 – chuẩn hóa màu Reinhard", "color"),
]
COLOR_BASELINE = ("color_baseline", "E0", "Chỉ dùng màu (hồi quy logistic)", "baseline")
REFERENCE_FOR_COLOR = "resnet50_rgb"   # các thí nghiệm màu so với mô hình này
