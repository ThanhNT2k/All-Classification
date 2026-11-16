# Phân loại ảnh phết máu ngoại vi — Bạch cầu cấp dòng lympho (ALL)

Phân loại ảnh phết máu thành 4 lớp: **Benign** (hematogone), **Early**, **Pre**, **Pro**.
Dữ liệu: [Kaggle – Acute Lymphoblastic Leukemia (ALL) image dataset](https://www.kaggle.com/datasets/mehradaria/leukemia/data).

> Công cụ nghiên cứu, hỗ trợ sàng lọc — không dùng để chẩn đoán.

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Bước 1 (kiểm kê + chia tập) chỉ cần `pillow` và `numpy`.

## Bước 1 — Chuẩn bị dữ liệu

1. Tải bộ dữ liệu từ Kaggle, giải nén **nguyên trạng** vào `data/raw/`.
   Không đổi tên hay sắp xếp lại thư mục — code tự nhận lớp từ tên thư mục.
2. Kiểm kê:
   ```bash
   python scripts/audit_data.py
   ```
   → `results/audit/<thời_gian>/summary.md`: số ảnh mỗi lớp, kích thước, file hỏng,
   ảnh trùng/gần trùng, ảnh trùng nhưng khác nhãn. **Số liệu phần "Dữ liệu" của báo cáo lấy từ đây.**
3. Chia tập:
   ```bash
   python scripts/split_data.py                       # tự chọn thư mục "Original" nếu có
   python scripts/split_data.py --variant Original    # hoặc chỉ định rõ
   ```
   → `data/splits/train.csv`, `val.csv`, `test.csv`, `split_report.md`.

### Vì sao chia theo nhóm

Bộ dữ liệu có nhiều ảnh từ cùng một bệnh nhân. Chia ngẫu nhiên theo ảnh làm ảnh gần như giống nhau
nằm ở cả train và test → accuracy bị thổi phồng. `split.py` gom thành một nhóm và đặt chung một tập:

| Gom theo | Có sẵn? |
|---|---|
| Ảnh trùng tuyệt đối (md5) và gần trùng (dHash, Hamming ≤ 4) | ✅ tự động |
| Cùng một ảnh ở các phiên bản khác nhau (Original/Segmented) | ✅ tự động |
| Cùng bệnh nhân | ⚠️ chỉ khi có file `--patient-map` (CSV `key,patient_id`) |

Sau khi chia, script kiểm tra lại độc lập và **dừng với lỗi** nếu còn ảnh trùng nằm ở hai tập.
Nếu không có mã bệnh nhân, ghi rõ trong phần Hạn chế của báo cáo.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
```

Chạy trên bộ dữ liệu **giả lập** có cài sẵn ảnh trùng, ảnh gần trùng, ảnh sai nhãn, file hỏng.

## Cấu trúc

```
configs/          cấu hình thí nghiệm (YAML)
data/raw/         dữ liệu gốc từ Kaggle (không đưa lên git)
data/splits/      train/val/test.csv + báo cáo chia tập
scripts/          lệnh chạy: audit_data.py, split_data.py
src/allcls/data/  discover (tìm ảnh + nhãn), hashing, audit, split, dataset, transforms
src/allcls/       models/, train.py, evaluate.py, explain.py (Grad-CAM)
experiments/      chạy nhiều cấu hình
results/          kết quả từng lần chạy (không đưa lên git)
notebooks/        EDA, phân tích lỗi
tests/            kiểm thử
```

## Bước 2–4 — Huấn luyện trên Google Colab

1. Tạo gói: `python scripts/make_colab_bundle.py` → `colab_bundle.zip` (~72 MB, code + ảnh Original + file chia tập).
2. Tải `colab_bundle.zip` lên Google Drive vào `MyDrive/ALL/`.
3. Mở `notebooks/colab_train.ipynb` trên Colab (File → Upload notebook), chọn GPU T4, chạy lần lượt.
   Ô 4 là kiểm thử: huấn luyện thử trên dữ liệu giả, phải ra `OK` rồi mới chạy thí nghiệm thật.

| Mã | Config | Mô hình / mục đích |
|---|---|---|
| E0 | `scripts/color_baseline.py` | Chỉ dùng màu (histogram), không nhìn hình dạng tế bào — mốc kiểm tra lối tắt |
| E1 | `simple_cnn` | CNN tự xây dựng, huấn luyện từ đầu (~1,2 triệu tham số) |
| E2 | `resnet50_rgb` | ResNet50, pretrained ImageNet |
| E3 | `efficientnet_b0_rgb` | EfficientNet-B0, pretrained ImageNet |
| E4 | `mobilenetv3_rgb` | MobileNetV3-Large, pretrained ImageNet |
| C1 | `resnet50_gray` | ResNet50, ảnh xám — đo mức phụ thuộc vào màu |
| C2 | `resnet50_strong_jitter` | ResNet50, nhiễu màu mạnh |
| C3 | `resnet50_stainnorm` | ResNet50, chuẩn hóa màu nhuộm Reinhard |

Mỗi lần chạy lưu trong `results/runs/<tên>_<thời gian>/`: config, phiên bản thư viện, mã băm file
chia tập, lịch sử huấn luyện, trọng số tốt nhất, dự đoán từng ảnh, chỉ số kèm khoảng tin cậy 95%
(bootstrap), ma trận nhầm lẫn, số tham số, thời gian huấn luyện, tốc độ suy luận.

### So sánh mô hình — `results/runs/COMPARISON.md`

`python scripts/compare_models.py --runs results/runs --color-baseline results/color_baseline`

- **Bảng 1** Chất lượng: accuracy, balanced accuracy, macro precision/recall/F1, AUC, độ nhạy ác tính, độ đặc hiệu
- **Bảng 2** F1 từng lớp
- **Bảng 3** Hiệu năng tính toán: số tham số, dung lượng, thời gian huấn luyện, độ trễ, thông lượng
- **Bảng 4** Kiểm định McNemar từng cặp mô hình (p hiệu chỉnh Holm): chênh lệch accuracy có ý nghĩa thống kê không
- **Bảng 5** Kiểm tra lối tắt màu (ResNet50 và các biến thể C1–C3)
- **Hình** accuracy/F1 kèm thanh sai số, đường ROC, accuracy theo số tham số; `comparison.csv` cho Excel

Chạy 1 seed: giá trị kèm KTC 95% (bootstrap). Chạy nhiều seed (`--seeds 42 43 44`): trung bình ± độ lệch chuẩn.

## Lộ trình

- [x] Bước 1: kiểm kê dữ liệu, chia tập chống rò rỉ (đã chạy trên dữ liệu thật)
- [x] Bước 2: Dataset + augmentation (ảnh xám, nhiễu màu, chuẩn hóa màu nhuộm)
- [x] Bước 3: Code huấn luyện E1–E4, C1–C3 + baseline màu E0 (E0 đã chạy; còn lại chờ chạy trên Colab)
- [x] So sánh mô hình: bảng, kiểm định McNemar, biểu đồ
- [x] Bước 4: Đánh giá — accuracy, macro F1, độ nhạy ác tính, AUC, KTC 95% bootstrap
- [ ] Bước 5: Giải thích — Grad-CAM, phân tích ảnh bị phân loại sai

## Kết quả kiểm kê dữ liệu thật (21/10/2026)

Chi tiết: `results/audit/20261021_070144/` và `data/splits/split_report.md`.

- 3.256 ảnh mỗi phiên bản (Original, Segmented), 224×224 RGB, không file hỏng.
  Benign 504 · Early 985 · Pre 963 · Pro 804.
- **Ảnh trùng trong Original:** 40 cặp trùng tuyệt đối + 49 cặp gần trùng → 85 nhóm (172 ảnh),
  tất cả cùng lớp. Các cặp thường có số thứ tự liền nhau (chụp liên tiếp cùng vi trường).
  Ảnh minh họa: `results/audit/20261021_065938/dup_examples.jpg`.
- **Ảnh Segmented (tách tế bào ra nền đen, có chủ đích) — lớp Pro khác hẳn các lớp khác:**
  phần ảnh còn lại có trung vị 0,9% diện tích (Benign/Early/Pre: 7–20%); 698/804 ảnh < 2%.
  Kiểm tra trực quan 184 ảnh thấp nhất (kéo sáng ×8): chỉ còn mảnh viền nhỏ, các tế bào
  thấy rõ trên ảnh Original không được giữ lại (`pro_segmented_check.jpg`).
  → Chênh lệch giữa các lớp có thể thành "lối tắt": mô hình học "ảnh gần đen = Pro"
  thay vì học hình thái tế bào. Thí nghiệm chính dùng **Original**; Segmented nếu dùng
  thì để so sánh, kèm kiểm tra lối tắt (ví dụ: chỉ dùng tỷ lệ pixel sáng để đoán lớp).
- **Giải thích (đã kiểm chứng trên 80 ảnh/lớp):** theo mô tả Kaggle, Segmented được tạo bằng
  ngưỡng màu HSV. Trên ảnh Original, tế bào nhuộm lớp Pro nhạt hơn rõ rệt: độ bão hòa S trung vị
  82 (Benign/Early/Pre: 130–136, thang 0–255), sắc độ H 192 (các lớp khác 176–178).
  Ngưỡng HSV chung nên bỏ sót phần lớn tế bào Pro.
- **Hệ quả cho cả ảnh Original:** lớp Pro khác các lớp kia về MÀU NHUỘM, không chỉ hình thái
  (có thể do khác đợt nhuộm/chụp). Mô hình có thể học màu thay vì hình thái.
  → Thí nghiệm bắt buộc: so sánh ảnh màu vs ảnh xám, và có/không chuẩn hóa màu nhuộm
  (stain normalization) + color jitter mạnh khi huấn luyện.
- **Baseline chỉ dùng màu (E0) đạt 98,8% accuracy trên test** (macro F1 98,6%), chỉ bằng
  histogram màu + hồi quy logistic. Chỉ 6 con số (màu trung bình, độ lệch chuẩn RGB) đã cho 85,8%
  và nhận đúng 100% ảnh Pro. → Độ chính xác cao trên bộ này chưa chứng minh mô hình học hình thái.
- Chia tập (seed 42, theo nhóm): train 2.279 · val 490 · test 487 ảnh; kiểm tra rò rỉ: ĐẠT.

## Lưu ý khi viết báo cáo

- Lớp "Benign" là bệnh nhân *nghi ngờ* ALL có hematogone, không phải người khỏe mạnh.
- Early / Pre / Pro là phân nhóm B-ALL, không phải mức độ nặng.
- Dữ liệu từ một trung tâm duy nhất; chưa kiểm chứng trên ảnh từ phòng xét nghiệm khác.
