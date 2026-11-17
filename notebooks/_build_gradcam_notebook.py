"""Sinh notebooks/ALL_GradCAM_4Models_Geometry.ipynb.  Chạy:  python notebooks/_build_gradcam_notebook.py"""
import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md(r"""
# Giải thích 4 mô hình ALL bằng Grad-CAM (pipeline Geometry)

Notebook này dùng **kết quả của `ALL_Train_4Models_Geometry.ipynb`**: trọng số `best.pt` của từng lần chạy trong
`training_results/geometry_v1/runs/<mô hình>_<view>_s42/`, cùng manifest và cấu hình của notebook pipeline.
Không huấn luyện lại gì; chỉ suy luận trên **tập test**.

| Phần | Nội dung |
|---|---|
| Grad-CAM | Bản đồ nhiệt cho lớp dự đoán (và lớp thật với ảnh sai) của cả 4 mô hình, view Original và Segmented |
| Ảnh sai | Danh sách và hình mọi ảnh test bị ít nhất một mô hình đoán sai |
| 1. Tập trung vào tế bào | Phần Grad-CAM nằm trong vùng bạch cầu so với diện tích vùng đó (tỉ số > 1 = dồn vào tế bào) |
| 2. Che vùng nóng | Che 20% ô Grad-CAM cao nhất so với che 20% ô ngẫu nhiên: bản đồ nhiệt có đáng tin không |
| 3. Xóa bạch cầu | Tô mọi bạch cầu bằng màu nền rồi dự đoán lại: mô hình có cần tế bào không |
| 4. Xóa nền | Giữ bạch cầu, tô phần còn lại bằng màu phẳng: chỉ tế bào có đủ không |
| 5. Ảnh trắng đen | Đổi ảnh test sang trắng đen rồi dự đoán: mô hình phụ thuộc màu đến đâu |

Phép thử 3 và 4 chỉ chạy trên view **Original** (ảnh Segmented vốn đã bỏ nền).
Kết quả lưu ở `training_results/geometry_v1/explain/`; chạy lại sẽ bỏ qua phần đã xong.
Thời gian trên T4: khoảng 1 phút mỗi (mô hình, view), tổng khoảng 10 phút.
""")

code(r"""
# 1. Cấu hình, kết nối Drive, giải nén dữ liệu (giống notebook huấn luyện)
import os, json, math, zipfile
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torchvision
from torchvision import models as tvm
from PIL import Image
import cv2
import matplotlib
import matplotlib.pyplot as plt

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:                      # chạy ngoài Colab: đặt biến môi trường ALL_PROJECT và ALL_DATA_ROOT
    IN_COLAB = False

PROJECT = Path('/content/drive/MyDrive/ALL_project') if IN_COLAB else Path(os.environ.get('ALL_PROJECT', '.'))
PIPELINE_DIR = PROJECT / 'ALL_preprocessing_separate' / 'auto_group_phash4_v1'
RESULTS = PROJECT / 'training_results' / 'geometry_v1'
EXPLAIN = RESULTS / 'explain'
ZIP_PATH = PROJECT / 'data-ALL.zip'
EXTRACT_DIR = Path('/content/ALL_data')
DATA_ROOT = EXTRACT_DIR / 'data-ALL' if IN_COLAB else Path(os.environ.get('ALL_DATA_ROOT', 'data'))

MODELS_TO_RUN = ['custom_cnn', 'resnet50', 'efficientnet_b0', 'mobilenet_v2']
VIEWS_TO_RUN = ['original', 'segmented']
SEED = 42
BATCH_SIZE = 32
ERASE_FRACTION = 0.2        # phép thử 2: tỉ lệ ô lưới bị che
OCCLUSION_MODEL = 'resnet50'  # mô hình dùng cho hình minh họa các phép thử che/xóa
LIMIT = None                # đặt số nhỏ (ví dụ 40) để chạy thử nhanh; None = toàn bộ tập test

if IN_COLAB and not all((DATA_ROOT / v).is_dir() for v in ['Original', 'Segmented']):
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXTRACT_DIR)
for f in ['main_pipeline_config.json', 'manifest_with_splits.csv']:
    if not (PIPELINE_DIR / f).is_file():
        raise FileNotFoundError(f'Thiếu {PIPELINE_DIR / f} — hãy chạy notebook pipeline trước.')
EXPLAIN.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| device', device)
print('Kết quả huấn luyện:', RESULTS); print('Lưu giải thích tại:', EXPLAIN)
""")

code(r"""
# 2. Dữ liệu test và mô hình (kiến trúc giống notebook huấn luyện; trọng số lấy từ best.pt)
PIPE_CFG = json.loads((PIPELINE_DIR / 'main_pipeline_config.json').read_text(encoding='utf-8'))
CLASSES = sorted(PIPE_CFG['classes'], key=PIPE_CFG['classes'].get)
SIZE = PIPE_CFG['input_size'][1]
MANIFEST = pd.read_csv(PIPELINE_DIR / PIPE_CFG['manifest_file'], keep_default_na=False)
CODES = {'custom_cnn': 'CNN tự xây', 'resnet50': 'ResNet50', 'efficientnet_b0': 'EfficientNet-B0', 'mobilenet_v2': 'MobileNetV2'}

def test_rows(view):
    # Cùng thứ tự với test loader của notebook huấn luyện (không xáo trộn).
    r = MANIFEST[(MANIFEST['view'] == view) & (MANIFEST['split'] == 'test')].reset_index(drop=True)
    return r if LIMIT is None else r.iloc[np.linspace(0, len(r) - 1, min(LIMIT, len(r))).astype(int)].reset_index(drop=True)

def load_images(rows):
    out = []
    for p in rows.relative_path:
        with Image.open(DATA_ROOT / p) as im:
            im = im.convert('RGB')
            if im.size != (SIZE, SIZE):
                raise ValueError(f'Cần ảnh {SIZE}×{SIZE}, nhận {im.size}: {p}')
            out.append(np.asarray(im).copy())
    return out

class CustomCNN(nn.Module):
    def __init__(self, n_classes=4, widths=(32, 64, 128, 256, 384), dropout=0.4):
        super().__init__()
        layers, c_in = [], 3
        for w in widths:
            layers += [nn.Conv2d(c_in, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes))
    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

def build_model(name, n_classes=4):
    if name == 'custom_cnn':
        m = CustomCNN(n_classes)
    elif name == 'resnet50':
        m = tvm.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif name == 'efficientnet_b0':
        m = tvm.efficientnet_b0(weights=None); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif name == 'mobilenet_v2':
        m = tvm.mobilenet_v2(weights=None); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    else:
        raise ValueError(name)
    return m

# Tách mô hình thành 2 nửa tại bản đồ đặc trưng không gian cuối cùng (nơi lấy Grad-CAM).
def forward_features(m, name, x):
    if name == 'resnet50':
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        return m.layer4(m.layer3(m.layer2(m.layer1(x))))          # đầu ra conv5_x, 7×7
    return m.features(x)                                          # custom_cnn / efficientnet_b0 / mobilenet_v2, 7×7

def forward_head(m, name, f):
    if name == 'custom_cnn':
        return m.classifier(m.pool(f))
    if name == 'resnet50':
        return m.fc(torch.flatten(m.avgpool(f), 1))
    if name == 'efficientnet_b0':
        return m.classifier(torch.flatten(m.avgpool(f), 1))
    return m.classifier(torch.flatten(nn.functional.adaptive_avg_pool2d(f, 1), 1))   # mobilenet_v2

def run_dir(name, view):
    return RESULTS / 'runs' / f'{name}_{view}_s{SEED}'

def load_run(name, view):
    m = build_model(name)
    m.load_state_dict(torch.load(run_dir(name, view) / 'best.pt', map_location='cpu'))
    return m.to(device).eval()

def to_batch(images, name):
    st = PIPE_CFG['normalization'][name]
    x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div(255)
    mean = torch.tensor(st['mean']).view(1, 3, 1, 1); std = torch.tensor(st['std']).view(1, 3, 1, 1)
    return ((x - mean) / std).to(device)

# Kiểm tra: hai nửa ghép lại phải cho đúng đầu ra của forward gốc
for n in MODELS_TO_RUN:
    m = build_model(n).eval(); x = torch.randn(2, 3, SIZE, SIZE)
    with torch.no_grad():
        f = forward_features(m, n, x)
        assert torch.allclose(forward_head(m, n, f), m(x), atol=1e-5), n
    print(f'{n:16s} bản đồ đặc trưng {tuple(f.shape[1:])}')
print('Lớp:', CLASSES, '| ảnh test mỗi view:', {v: len(test_rows(v)) for v in VIEWS_TO_RUN})
missing = [run_dir(n, v).name for n in MODELS_TO_RUN for v in VIEWS_TO_RUN if not (run_dir(n, v) / 'best.pt').exists()]
print('Thiếu best.pt (sẽ bỏ qua):', missing or 'không')
""")

md(r"""
## Grad-CAM và các phép che/xóa
Grad-CAM (Selvaraju et al., 2017): trọng số mỗi kênh = trung bình gradient của điểm số lớp theo kênh đó;
bản đồ nhiệt = ReLU(tổng có trọng số các kênh), chuẩn hóa về [0, 1] theo từng ảnh.

Mặt nạ bạch cầu là **ước lượng**: view Original dùng ngưỡng Otsu trên kênh a\* của LAB (bạch cầu nhuộm tím có a\* cao,
hồng cầu và nền gần 0); view Segmented dùng vùng khác màu đen. Xem hình kiểm tra mặt nạ ở ô 7 trước khi tin các con số.
""")

code(r"""
# 3. Grad-CAM, mặt nạ bạch cầu, các phép che/xóa
def grad_cam(m, name, x, targets=None):
    # Trả (cam [B,h,w] trong [0,1], xác suất [B,C]). targets=None: giải thích lớp dự đoán.
    with torch.enable_grad():
        x = x.clone().requires_grad_(True)
        feats = forward_features(m, name, x)
        logits = forward_head(m, name, feats)
        if targets is None:
            targets = logits.argmax(1)
        grads = torch.autograd.grad(logits.gather(1, targets.view(-1, 1)).sum(), feats)[0]
    cam = torch.relu((grads.mean(dim=(2, 3), keepdim=True) * feats.detach()).sum(1))
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return cam.cpu().numpy().astype(np.float32), torch.softmax(logits.detach().float(), 1).cpu().numpy()

@torch.no_grad()
def predict_images(m, name, images):
    out = []
    for s in range(0, len(images), BATCH_SIZE):
        out.append(torch.softmax(m(to_batch(images[s:s + BATCH_SIZE], name)).float(), 1).cpu().numpy())
    return np.concatenate(out)

def upsample(cam):
    return np.clip(cv2.resize(cam, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR), 0, 1)

def overlay(rgb, cam_up):
    heat = matplotlib.colormaps['jet'](cam_up)[..., :3]
    a = (0.65 * cam_up)[..., None]
    return np.clip((1 - a) * rgb / 255.0 + a * heat, 0, 1)

def dilate(mask, px=4):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)

def cell_mask(rgb, view):
    # Vùng bạch cầu (bool) hoặc None nếu không tách được.
    if view == 'segmented':
        m = (rgb.max(2) > 10).astype(np.uint8) * 255
        thr = 255
    else:
        a = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 1].astype(np.int16) - 128
        score = cv2.GaussianBlur(np.clip(a * 3, 0, 255).astype(np.uint8), (5, 5), 0)
        thr, m = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(m.shape, bool)
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] >= max(8, int(0.001 * m.size)):
            keep |= lab == j
    if thr < 18 or not 0.002 <= keep.mean() <= 0.6:      # thr 18 = a* 6: dưới mức này chỉ là nhiễu
        return None
    return keep

def background_color(rgb, mask, view):
    if view == 'segmented':
        return np.zeros(3, np.uint8)
    px = rgb[~mask] if mask is not None and (~mask).any() else rgb.reshape(-1, 3)
    return np.median(px, axis=0).astype(np.uint8)

def erase(rgb, region, color):
    out = rgb.copy(); out[region] = color
    return out

def grid_region(cells, grid_hw):
    g = np.zeros(grid_hw[0] * grid_hw[1], np.uint8); g[cells] = 1
    return cv2.resize(g.reshape(grid_hw), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)

def to_gray(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.stack([g, g, g], axis=-1)

def variants(rgb, view):
    # Ảnh sau khi xóa bạch cầu / xóa nền / đổi trắng đen (None nếu không tách được mặt nạ).
    m = cell_mask(rgb, view)
    md = None if m is None else dilate(m)
    bg = background_color(rgb, md, view)
    return dict(mask=md, bg=bg, cells_erased=None if md is None else erase(rgb, md, bg),
                background_erased=None if md is None else erase(rgb, ~md, bg), gray=to_gray(rgb))
""")

code(r"""
# 4. Phân tích một lần chạy (mô hình, view) -> per_image.csv + cams.npz
def analyze_run(name, view):
    out = EXPLAIN / 'runs' / f'{name}_{view}_s{SEED}'
    if (out / 'per_image.csv').exists() and (out / 'cams.npz').exists():
        print(f'[{out.name}] đã xong — bỏ qua'); return
    if not (run_dir(name, view) / 'best.pt').exists():
        print(f'[{out.name}] thiếu best.pt — bỏ qua'); return
    out.mkdir(parents=True, exist_ok=True)
    rows = test_rows(view); images = load_images(rows); y = rows.target.to_numpy()
    m = load_run(name, view)

    cams_pred, cams_true, probs = [], [], []
    for s in range(0, len(images), BATCH_SIZE):
        x = to_batch(images[s:s + BATCH_SIZE], name)
        c, p = grad_cam(m, name, x)
        ct, _ = grad_cam(m, name, x, torch.as_tensor(y[s:s + BATCH_SIZE], device=device))
        cams_pred.append(c); cams_true.append(ct); probs.append(p)
    cams_pred, cams_true, probs = map(np.concatenate, (cams_pred, cams_true, probs))
    pred = probs.argmax(1); grid_hw = cams_pred.shape[1:]

    # đối chiếu với dự đoán đã lưu lúc huấn luyện: khác nhiều = nạp sai trọng số / tiền xử lý
    agree = ''
    f = run_dir(name, view) / 'test_predictions.npz'
    if f.exists() and LIMIT is None:
        saved = np.load(f)['prob'].argmax(1)
        if len(saved) == len(pred):
            agree = f'{int((saved == pred).sum())}/{len(pred)}'

    rng = np.random.default_rng(0)
    k = max(1, math.ceil(ERASE_FRACTION * grid_hw[0] * grid_hw[1]))
    V = [variants(im, view) for im in images]
    top = [erase(im, grid_region(np.argsort(-c.ravel())[:k], grid_hw), v['bg']) for im, c, v in zip(images, cams_pred, V)]
    rnd = [erase(im, grid_region(rng.choice(c.size, k, replace=False), grid_hw), v['bg']) for im, c, v in zip(images, cams_pred, V)]
    p_top, p_rnd = predict_images(m, name, top), predict_images(m, name, rnd)
    p_gray = predict_images(m, name, [v['gray'] for v in V])
    do_erase = view == 'original'
    if do_erase:
        p_cell = predict_images(m, name, [im if v['mask'] is None else v['cells_erased'] for im, v in zip(images, V)])
        p_bg = predict_images(m, name, [im if v['mask'] is None else v['background_erased'] for im, v in zip(images, V)])

    recs = []
    for i in range(len(images)):
        md = V[i]['mask']; cu = upsample(cams_pred[i])
        r = dict(sample_id=rows.sample_id[i], relative_path=rows.relative_path[i], label=CLASSES[y[i]], pred=CLASSES[pred[i]],
                 correct=int(pred[i] == y[i]), p_pred=probs[i, pred[i]], p_true=probs[i, y[i]],
                 true_rank=int((probs[i] > probs[i, y[i]]).sum()) + 1, mask_ok=int(md is not None),
                 cell_area=np.nan, cam_in_cell=np.nan, peak_in_cell=np.nan,
                 p_pred_erase_top=p_top[i, pred[i]], p_pred_erase_random=p_rnd[i, pred[i]],
                 flip_erase_top=int(p_top[i].argmax() != pred[i]), flip_erase_random=int(p_rnd[i].argmax() != pred[i]),
                 pred_gray=CLASSES[p_gray[i].argmax()], p_true_gray=p_gray[i, y[i]],
                 pred_cells_erased='', p_true_cells_erased=np.nan, pred_background_erased='', p_true_background_erased=np.nan)
        if md is not None:
            r['cell_area'] = md.mean()
            if cu.sum() > 0:
                r['cam_in_cell'] = cu[md].sum() / cu.sum()
                r['peak_in_cell'] = int(md[np.unravel_index(cu.argmax(), cu.shape)])
            if do_erase:
                r.update(pred_cells_erased=CLASSES[p_cell[i].argmax()], p_true_cells_erased=p_cell[i, y[i]],
                         pred_background_erased=CLASSES[p_bg[i].argmax()], p_true_background_erased=p_bg[i, y[i]])
        recs.append(r)
    df = pd.DataFrame(recs)
    np.savez_compressed(out / 'cams.npz', cams_pred=cams_pred, cams_true=cams_true, probs=probs)
    df.to_csv(out / 'per_image.csv', index=False)
    print(f"[{out.name}] sai {int((1 - df.correct).sum())}/{len(df)} | trùng dự đoán đã lưu: {agree or 'không đối chiếu'} | "
          f"không tách được mặt nạ: {int((1 - df.mask_ok).sum())}")
    del m; torch.cuda.empty_cache()
""")

md("## Chạy cho 4 mô hình × 2 view")

code(r"""
# 5. Chạy toàn bộ (bỏ qua phần đã xong)
for name in MODELS_TO_RUN:
    for view in VIEWS_TO_RUN:
        analyze_run(name, view)

def load_result(name, view):
    d = EXPLAIN / 'runs' / f'{name}_{view}_s{SEED}'
    if not (d / 'per_image.csv').exists():
        return None
    df = pd.read_csv(d / 'per_image.csv', keep_default_na=False, na_values=[''])
    df[['pred_cells_erased', 'pred_background_erased']] = df[['pred_cells_erased', 'pred_background_erased']].fillna('')
    z = np.load(d / 'cams.npz')
    return dict(df=df, cams_pred=z['cams_pred'], cams_true=z['cams_true'], probs=z['probs'])

RES = {(n, v): load_result(n, v) for n in MODELS_TO_RUN for v in VIEWS_TO_RUN}
RES = {k: r for k, r in RES.items() if r is not None}
print('Đã có kết quả cho:', [f'{n}_{v}' for n, v in RES])
""")

md(r"""
## Bảng tổng hợp
- **Tỉ số tập trung** = (phần Grad-CAM trong vùng bạch cầu) / (diện tích vùng bạch cầu). > 1: dồn vào tế bào; ≈ 1: rải đều.
- **Đổi dự đoán khi che vùng nóng** phải cao hơn hẳn **khi che ngẫu nhiên** thì bản đồ nhiệt mới phản ánh vùng mô hình dùng.
- **Xóa bạch cầu mà vẫn đúng** = mô hình nhận ra lớp từ nền/hồng cầu/màu nhuộm, không cần hình thái tế bào. Đoán ngẫu nhiên 4 lớp = 25%.
- **Xóa nền mà giảm mạnh** = mô hình cần nền để đoán. Lưu ý ảnh sau khi xóa nằm ngoài phân phối huấn luyện.
""")

code(r"""
# 6. Bảng tổng hợp + bảng theo lớp + danh sách ảnh sai
def acc(df, col):
    d = df[df[col] != '']
    return (d[col] == d.label).mean() if len(d) else np.nan

def summarize(df):
    ok = df[df.mask_ok == 1]
    area, inside = ok.cell_area.mean(), ok.cam_in_cell.mean()
    return {'Số ảnh': len(df), 'Số ảnh sai': int((1 - df.correct).sum()), 'Accuracy': df.correct.mean(),
            'Tỉ số tập trung': inside / area if area else np.nan, 'Đỉnh CAM trong tế bào': ok.peak_in_cell.mean(),
            'Đổi dự đoán: che vùng nóng': df.flip_erase_top.mean(), 'Đổi dự đoán: che ngẫu nhiên': df.flip_erase_random.mean(),
            'Acc xóa bạch cầu': acc(df, 'pred_cells_erased'), 'Acc xóa nền': acc(df, 'pred_background_erased'),
            'Acc trắng đen': acc(df, 'pred_gray')}

PCT = ['Accuracy', 'Đỉnh CAM trong tế bào', 'Đổi dự đoán: che vùng nóng', 'Đổi dự đoán: che ngẫu nhiên',
       'Acc xóa bạch cầu', 'Acc xóa nền', 'Acc trắng đen']
overview = pd.DataFrame({(CODES[n], v): summarize(r['df']) for (n, v), r in RES.items()}).T
overview.index.names = ['Mô hình', 'View']
overview.to_csv(EXPLAIN / 'summary.csv')
display(overview.style.format({c: '{:.1%}' for c in PCT} | {'Tỉ số tập trung': '{:.2f}', 'Số ảnh': '{:.0f}', 'Số ảnh sai': '{:.0f}'}, na_rep='—'))

by_class = []
for (n, v), r in RES.items():
    for c in CLASSES:
        d = r['df'][r['df'].label == c]
        by_class.append({'Mô hình': CODES[n], 'View': v, 'Lớp': c, 'Số ảnh': len(d), 'Ảnh gốc': d.correct.mean(),
                         'Xóa bạch cầu': acc(d, 'pred_cells_erased'), 'Xóa nền': acc(d, 'pred_background_erased'),
                         'Trắng đen': acc(d, 'pred_gray')})
by_class = pd.DataFrame(by_class)
by_class.to_csv(EXPLAIN / 'occlusion_by_class.csv', index=False)
print('Accuracy theo lớp thật khi thay đổi ảnh đầu vào:')
display(by_class.set_index(['View', 'Mô hình', 'Lớp']).sort_index().style.format(
    {c: '{:.1%}' for c in ['Ảnh gốc', 'Xóa bạch cầu', 'Xóa nền', 'Trắng đen']}, na_rep='—'))

errors = pd.concat([r['df'][r['df'].correct == 0].assign(model=CODES[n], view=v) for (n, v), r in RES.items()], ignore_index=True)
errors = errors[['view', 'model', 'sample_id', 'label', 'pred', 'p_pred', 'p_true', 'true_rank', 'cam_in_cell', 'cell_area']]
errors.to_csv(EXPLAIN / 'errors.csv', index=False)
print(f'Ảnh sai: {len(errors)} lượt (mô hình, ảnh). Sai kiểu ác tính → Benign: {int(((errors.label != CLASSES[0]) & (errors.pred == CLASSES[0])).sum())}')
display(errors.sort_values(['view', 'sample_id', 'model']).style.format(
    {'p_pred': '{:.1%}', 'p_true': '{:.1%}', 'cam_in_cell': '{:.1%}', 'cell_area': '{:.1%}'}, na_rep='—'))
""")

md(r"""
## Hình
Các hình dưới đây lưu trong `explain/` để đưa vào báo cáo. Tiêu đề xanh = dự đoán đúng, đỏ = sai.
""")

code(r"""
# 7. Kiểm tra mặt nạ bạch cầu bằng mắt (view Original): viền vàng phải ôm bạch cầu, không dính hồng cầu
rows = test_rows('original')
pick = [i for c in range(len(CLASSES)) for i in np.flatnonzero(rows.target.to_numpy() == c)[:6]]
imgs = load_images(rows.iloc[pick])
fig, axes = plt.subplots(len(CLASSES), 6, figsize=(12, 2.1 * len(CLASSES)), squeeze=False)
for ax in axes.ravel():
    ax.axis('off')
for ax, i, im in zip(axes.ravel(), pick, imgs):
    m = cell_mask(im, 'original')
    ax.imshow(im); ax.set_title(CLASSES[rows.target[i]] + ('' if m is not None else ' (không tách được)'), fontsize=8)
    if m is not None:
        ax.contour(dilate(m), levels=[0.5], colors='yellow', linewidths=0.8)
fig.tight_layout(); fig.savefig(EXPLAIN / 'fig_mask_check.png', dpi=130); plt.show()
""")

code(r"""
# 8. Grad-CAM của 4 mô hình trên cùng ảnh: ảnh đúng (mỗi lớp 1 ảnh) và ảnh sai
def show(ax, img, title='', color='black'):
    ax.imshow(img); ax.set_title(title, fontsize=8, color=color); ax.axis('off')

def cam_grid(view, idx, path, title):
    names = [n for n in MODELS_TO_RUN if (n, view) in RES]
    if not names or not len(idx):
        print(f'[{view}] không có gì để vẽ: {title}'); return
    rows = test_rows(view); imgs = load_images(rows.iloc[idx])
    fig, axes = plt.subplots(len(idx), len(names) + 1, figsize=(2.15 * (len(names) + 1), 2.65 * len(idx) + 0.3), squeeze=False)
    for ax_row, i, im in zip(axes, idx, imgs):
        show(ax_row[0], im, f'{rows.sample_id[i]}\nlớp thật: {rows.label[i]}')
        for ax, n in zip(ax_row[1:], names):
            r = RES[(n, view)]; d = r['df'].iloc[i]
            show(ax, overlay(im, upsample(r['cams_pred'][i])), f'{CODES[n]}\n{d.pred} ({d.p_pred:.0%})',
                 'darkgreen' if d.correct else 'firebrick')
    fig.suptitle(title, fontsize=10); fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=150); plt.show()

PICKS = {}
for view in VIEWS_TO_RUN:
    names = [n for n in MODELS_TO_RUN if (n, view) in RES]
    if not names:
        continue
    n_wrong = sum((1 - RES[(n, view)]['df'].correct.to_numpy()) for n in names)
    conf = np.mean([RES[(n, view)]['df'].p_pred.to_numpy() for n in names], axis=0)
    mask_ok = RES[(names[0], view)]['df'].mask_ok.to_numpy() == 1
    y = test_rows(view).target.to_numpy()
    correct = []
    for c in range(len(CLASSES)):                # ảnh mọi mô hình đều đúng, độ tự tin ở giữa (không chọn ảnh đẹp nhất)
        cand = np.flatnonzero((y == c) & (n_wrong == 0) & mask_ok)
        if len(cand):
            correct.append(int(cand[np.argsort(conf[cand])[len(cand) // 2]]))
    wrong = [int(i) for i in np.argsort(-n_wrong, kind='stable') if n_wrong[i] > 0]
    PICKS[view] = dict(correct=correct, wrong=wrong)
    print(f'[{view}] ảnh bị ít nhất một mô hình đoán sai: {len(wrong)}/{len(y)}; mọi mô hình cùng sai: {int((n_wrong == len(names)).sum())}')
    cam_grid(view, correct, EXPLAIN / f'fig_gradcam_correct_{view}.png', f'Grad-CAM lớp dự đoán — ảnh đúng · {view}')
    cam_grid(view, wrong[:8], EXPLAIN / f'fig_gradcam_errors_{view}.png', f'Grad-CAM lớp dự đoán — ảnh sai · {view}')
""")

code(r"""
# 9. Ảnh sai của từng mô hình: vì sao đoán lớp sai (trái) và bằng chứng cho lớp thật (phải)
for (n, view), r in RES.items():
    df = r['df']; idx = df[df.correct == 0].sort_values('p_pred', ascending=False).index.tolist()[:8]
    if not idx:
        print(f'[{n}_{view}] không có ảnh sai'); continue
    imgs = load_images(test_rows(view).iloc[idx])
    fig, axes = plt.subplots(len(idx), 3, figsize=(7.2, 2.5 * len(idx) + 0.4), squeeze=False)
    for ax_row, i, im in zip(axes, idx, imgs):
        d = df.iloc[i]
        show(ax_row[0], im, f'{d.sample_id}\nthật: {d.label}')
        show(ax_row[1], overlay(im, upsample(r['cams_pred'][i])), f'vì sao đoán {d.pred}\np = {d.p_pred:.1%}', 'firebrick')
        show(ax_row[2], overlay(im, upsample(r['cams_true'][i])), f'bằng chứng cho {d.label} (lớp thật)\np = {d.p_true:.1%}', 'darkgreen')
    fig.suptitle(f'{CODES[n]} · {view} — ảnh sai, tự tin nhất trước', fontsize=10); fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(EXPLAIN / f'fig_errors_{n}_{view}.png', dpi=130); plt.show()
""")

code(r"""
# 10. Minh họa các phép thử: ảnh gốc | xóa bạch cầu | xóa nền | trắng đen, kèm dự đoán của OCCLUSION_MODEL (view Original)
key = (OCCLUSION_MODEL, 'original')
if key in RES and PICKS.get('original', {}).get('correct'):
    idx = PICKS['original']['correct']; df = RES[key]['df']
    imgs = load_images(test_rows('original').iloc[idx])
    cols = [('Ảnh gốc', 'pred', None), ('Xóa bạch cầu', 'pred_cells_erased', 'cells_erased'),
            ('Xóa nền', 'pred_background_erased', 'background_erased'), ('Trắng đen', 'pred_gray', 'gray')]
    fig, axes = plt.subplots(len(idx), 4, figsize=(8.6, 2.4 * len(idx) + 0.3), squeeze=False)
    for ax_row, i, im in zip(axes, idx, imgs):
        v = variants(im, 'original'); d = df.iloc[i]
        for ax, (title, col, k) in zip(ax_row, cols):
            show(ax, im if k is None else v[k], f'{title} (thật: {d.label})\ndự đoán: {d[col]}',
                 'darkgreen' if d[col] == d.label else 'firebrick')
    fig.suptitle(f'Các phép thử che/xóa — dự đoán của {CODES[OCCLUSION_MODEL]}', fontsize=10); fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(EXPLAIN / 'fig_occlusion_examples.png', dpi=150); plt.show()

# Biểu đồ cột: accuracy theo kiểu ảnh đầu vào, view Original
ov = overview.xs('original', level='View')[['Accuracy', 'Acc xóa bạch cầu', 'Acc xóa nền', 'Acc trắng đen']] if 'original' in overview.index.get_level_values('View') else None
if ov is not None:
    ax = ov.rename(columns={'Accuracy': 'Ảnh gốc', 'Acc xóa bạch cầu': 'Xóa bạch cầu', 'Acc xóa nền': 'Xóa nền', 'Acc trắng đen': 'Trắng đen'}).plot.bar(figsize=(8.5, 3.6), rot=0, width=0.8)
    ax.axhline(0.25, color='gray', ls='--', lw=0.8); ax.text(-0.45, 0.265, 'đoán ngẫu nhiên', fontsize=8, color='gray')
    ax.set_ylim(0, 1.02); ax.set_ylabel('Accuracy trên tập test'); ax.set_xlabel(''); ax.grid(axis='y', alpha=.3); ax.legend(fontsize=8, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
    plt.tight_layout(); plt.savefig(EXPLAIN / 'fig_occlusion_accuracy.png', dpi=150); plt.show()
print('Đã lưu vào', EXPLAIN, ':', sorted(p.name for p in EXPLAIN.glob('*.*')))
""")

md(r"""
**Lưu ý khi đọc kết quả**
- Mặt nạ bạch cầu là ước lượng bằng ngưỡng màu, không phải nhãn phân vùng; ảnh không tách được mặt nạ bị loại khỏi phép đo 1, 3, 4.
- Bản đồ Grad-CAM chỉ có 7×7 ô (mỗi ô ứng với 32×32 điểm ảnh): cho biết mô hình dựa vào vùng nào, không cho biết chi tiết bên trong một tế bào, cũng không cho biết mô hình dùng màu hay hình dạng ở vùng đó.
- Ảnh sau khi xóa bạch cầu / xóa nền không giống ảnh huấn luyện, nên hai phép thử này là bằng chứng gián tiếp. Màu nền phẳng lấy từ chính ảnh đó nên vẫn mang tông màu của lớp.
- Kết quả là của một seed.
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "colab": {"provenance": [], "gpuType": "T4"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent / "ALL_GradCAM_4Models_Geometry.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Đã ghi", out, "-", len(CELLS), "ô")
