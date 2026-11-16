# Báo cáo chia tập dữ liệu

- Phiên bản ảnh: `Original` · seed: 42 · tỷ lệ: 70%/15%/15%
- Gom nhóm theo: duplicates + same-stem-across-variants
- Số nhóm: 3169 (trong đó 85 nhóm có ≥ 2 ảnh)
- Bỏ qua (hỏng/không rõ lớp): 0 · nhóm mâu thuẫn nhãn: 0 (đã loại)

| Tập | Benign | Early | Pre | Pro | Tổng |
|---|---:|---:|---:|---:|---:|
| train | 353 | 689 | 674 | 563 | 2279 |
| val | 76 | 148 | 145 | 121 | 490 |
| test | 75 | 148 | 144 | 120 | 487 |

## Kiểm tra rò rỉ

- Ảnh trùng tuyệt đối nằm ở 2 tập: 0
- Cặp gần trùng nằm ở 2 tập: 0
- Kết quả: **ĐẠT**

> Chưa gom theo bệnh nhân (không có mã BN). Ảnh khác nhau của cùng một người vẫn có thể nằm ở cả train và test → nêu trong phần Hạn chế.
