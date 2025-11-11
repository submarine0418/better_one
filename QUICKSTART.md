# 🚀 快速入門指南

## 📦 核心文件（必讀）

### 必須文件
1. **`color_correction.py`** - 色偏校正模組
2. **`matlab_style_enhancement.py`** - MATLAB 風格增強（包含大氣光估算）
3. **`parameter_predictor.py`** - 參數預測網路
4. **`train_matlab_style.py`** - 訓練腳本
5. **`inference_matlab_style.py`** - 推理腳本

### 文檔
- **`README.md`** - 詳細使用說明
- **`ARCHITECTURE.md`** - 系統架構
- **`SUMMARY.md`** - 完整總結

---

## ⚡ 三步驟快速開始

### 步驟 1: 安裝依賴
```bash
pip install torch torchvision opencv-python numpy scikit-image scipy tqdm
```

### 步驟 2: 訓練模型
```bash
python train_matlab_style.py \
    --input ./raw_images \
    --reference ./reference_images \
    --output ./output \
    --epochs 50 \
    --batch-size 4 \
    --device cuda
```

### 步驟 3: 推理增強
```bash
# 單張圖像
python inference_matlab_style.py \
    --input test.jpg \
    --output enhanced.png \
    --model ./output/best_model.pth \
    --device cuda

# 批量處理
python inference_matlab_style.py \
    --input ./test_images \
    --output ./enhanced \
    --model ./output/best_model.pth \
    --device cuda
```

---

## 📁 目錄結構建議

```
your_project/
├── raw_images/              # 原始水下圖像
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
│
├── reference_images/        # 參考圖像（增強後的標準）
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
│
├── color_correction.py
├── matlab_style_enhancement.py
├── parameter_predictor.py
├── train_matlab_style.py
├── inference_matlab_style.py
│
└── output/                  # 訓練輸出
    ├── best_model.pth      # 最佳模型
    ├── final_model.pth     # 最終模型
    └── checkpoint_*.pth    # 檢查點
```

---

## 🎯 核心概念

### 系統流程
```
原始圖像
    ↓
1. 色偏校正 (預處理)
    ↓
2. 大氣光估算 (預處理)
    ↓
3. 參數預測 (VGG-16)
    ├─ omega (去霧強度)
    ├─ guided_radius (引導濾波半徑)
    ├─ L_low (色彩拉伸下界)
    └─ L_high (色彩拉伸上界)
    ↓
4. MATLAB 風格增強
    ├─ 透射率計算
    ├─ 梯度約束
    ├─ 引導濾波
    ├─ 影像恢復
    └─ 色彩拉伸
    ↓
增強圖像
```

### 訓練 vs 推理
| 階段 | 色偏校正 | 大氣光估算 | 參數預測 | 增強流程 |
|------|---------|-----------|---------|---------|
| 訓練 | 預處理（數據載入時） | 預處理（數據載入時） | ✅ 訓練 | ✅ 可微分 |
| 推理 | ✅ 執行 | ✅ 執行 | ✅ 執行 | ✅ 執行 |

---

## 💡 常見問題

### Q1: 為什麼色偏校正和大氣光估算不參與訓練？
**A**: 因為它們包含不可微分的操作（條件分支、遞迴），但 MATLAB 方法已經很有效，作為預處理更穩定。

### Q2: 訓練時 GPU 記憶體不足怎麼辦？
**A**: 降低 `batch_size`（例如從 4 改為 2），或使用 `--no-amp` 禁用混合精度。

### Q3: 可以只使用部分模組嗎？
**A**: 可以！例如只做色偏校正：
```python
from color_correction import ColorCorrection
corrector = ColorCorrection()
img_corrected, color_type = corrector(img)
```

### Q4: 訓練需要多久？
**A**: 依數據集大小和 GPU 性能而定。以 890 張圖像、batch_size=4、NVIDIA RTX 3080 為例，每個 epoch 約 5-8 分鐘。

### Q5: 如何調整參數範圍？
**A**: 修改 `parameter_predictor.py` 中的 `param_ranges`：
```python
self.param_ranges = {
    'omega': (0.3, 0.9),           # 修改這裡
    'guided_radius': (5.0, 30.0),  # 修改這裡
    'L_low': (2.0, 15.0),          # 修改這裡
    'L_high': (85.0, 98.0),        # 修改這裡
}
```

---

## 🔍 檢查模組是否正常

### 測試色偏校正
```python
from color_correction import ColorCorrection
import cv2
import numpy as np

corrector = ColorCorrection()

# 創建綠色偏移測試圖像
img = np.zeros((256, 256, 3), dtype=np.float32)
img[:, :, 1] = 0.8  # 綠色通道強
img[:, :, 0] = 0.3
img[:, :, 2] = 0.3

corrected, color_type = corrector(img)
print(f"色偏類型: {color_type}")  # 應該輸出 "greenish"
```

### 測試大氣光估算
```python
from matlab_style_enhancement import AtmosphericLightEstimator
import numpy as np

estimator = AtmosphericLightEstimator()

# 測試圖像
img = np.random.rand(256, 256, 3).astype(np.float32)
A = estimator(img)
print(f"大氣光: R={A[0]:.4f}, G={A[1]:.4f}, B={A[2]:.4f}")
```

### 測試參數預測
```python
from parameter_predictor import MATLABParameterPredictor
import torch

model = MATLABParameterPredictor(pretrained=False)
model.eval()

img = torch.rand(1, 3, 224, 224)
features = torch.rand(1, 79)

with torch.no_grad():
    params = model(img, features)

for name, value in params.items():
    print(f"{name}: {value.item():.4f}")
```

---

## 📊 預期輸出範例

### 訓練輸出
```
載入數據集...
找到 890 張圖像
訓練樣本: 756
驗證樣本: 134

開始訓練
================================================================================
Epoch 1/50
100%|████████| 189/189 [05:23<00:00, loss=0.0234, lr=0.000010]

Train Loss: 0.023456
  L1: 0.012345, L2: 0.008901, Perceptual: 0.002210
Val Loss: 0.021234
  L1: 0.011234, L2: 0.007890, Perceptual: 0.002110

✓ 新的最佳模型! Val Loss: 0.021234
```

### 推理輸出
```
載入模型: best_model.pth  (device=cuda)
✓ 模型與增強模組已載入

[1/60] 處理: image001.jpg
------------------------------------------------------------
步驟 1/4: 色偏校正與大氣光估算...
  檢測到的色偏類型: blueish
  大氣光值: R=0.6234, G=0.6123, B=0.7456

步驟 2/4: 提取統計特徵...

步驟 3/4: 預測增強參數...

步驟 4/4: 應用增強...

✓ 儲存: enhanced/image001_enhanced.png
```

---

## 🎓 學習路徑

### 初學者
1. 閱讀 `README.md`
2. 運行快速開始的三個步驟
3. 查看推理輸出的中間結果

### 進階使用者
1. 閱讀 `ARCHITECTURE.md`
2. 理解可微分 vs 不可微分設計
3. 調整參數範圍或網路結構
4. 使用自己的損失函數

### 研究者
1. 閱讀完整源代碼
2. 分析訓練過程
3. 比較與 MATLAB 原始方法
4. 發表改進方案

---

## 🔗 相關資源

### 文檔
- `README.md` - 完整使用說明
- `ARCHITECTURE.md` - 詳細架構
- `SUMMARY.md` - 總結文檔

### 模組
- `color_correction.py` - 色偏校正
- `matlab_style_enhancement.py` - MATLAB 增強
- `parameter_predictor.py` - 參數預測
- `train_matlab_style.py` - 訓練
- `inference_matlab_style.py` - 推理

---

## ✅ 檢查清單

開始前確認：
- [ ] 已安裝所有依賴
- [ ] 準備好訓練數據（原始 + 參考圖像）
- [ ] GPU 可用（或準備使用 CPU）
- [ ] 有足夠的硬碟空間（模型約 100MB）

訓練時檢查：
- [ ] 訓練損失持續下降
- [ ] 驗證損失不過擬合
- [ ] 保存了最佳模型

推理時檢查：
- [ ] 色偏校正正確
- [ ] 大氣光值合理
- [ ] 參數在預期範圍
- [ ] 增強結果視覺良好

---

**祝您使用順利！** 🎉

有任何問題請查看完整文檔或聯繫支援。
