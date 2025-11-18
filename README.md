# MATLAB 風格水下影像增強系統（混合方案）

## 📋 系統架構

```
預處理階段 (不可微分，在數據載入時完成)
├─ color_correction.py          → 色偏校正 (LAB 空間分析)
└─ matlab_style_enhancement.py  → 大氣光估算 (四叉樹分割)

深度學習階段 (可微分，參與訓練)
├─ matlab_style_enhancement.py  → MATLAB 風格增強
│   ├─ 初始透射率計算
│   ├─ 梯度約束
│   ├─ 引導濾波
│   ├─ 影像恢復
│   └─ 色彩拉伸 ✨
│
└─ parameter_predictor.py       → VGG-16 參數預測器
    ├─ omega (去霧強度) [0.3, 0.9]
    ├─ guided_radius (引導濾波半徑) [5, 30]
    ├─ L_low (色彩拉伸下界) [2, 15]
    └─ L_high (色彩拉伸上界) [85, 98]

訓練與推理
├─ train_matlab_style.py        → 完整訓練腳本
└─ inference_matlab_style.py    → 推理腳本
```

---

## 📁 檔案說明

### 1. `color_correction.py`
**功能**: 色偏校正模組

**核心類別**:
- `ColorCorrection`: 主要的色偏校正器

**流程**:
```python
輸入 RGB 圖像
    ↓
LAB 色彩空間分析
    ↓
計算色偏因子 (CCF)
    ↓
判斷色偏類型 (greenish/blueish/yellowish/reddish/no_cast)
    ↓
應用對應的校正方法
    ↓
輸出校正後的圖像
```

**使用範例**:
```python
from color_correction import ColorCorrection

corrector = ColorCorrection()
img_corrected, color_type = corrector(img)  # img: (H,W,3) RGB [0,1]
print(f"色偏類型: {color_type}")
```

---

### 2. `matlab_style_enhancement.py`
**功能**: 大氣光估算 + MATLAB 風格增強

**核心類別**:
- `AtmosphericLightEstimator`: 四叉樹大氣光估算（不可微分）
- `GuidedFilter`: 引導濾波（可微分）
- `MATLABStyleEnhancement`: 完整增強流程（可微分）

**增強流程**:
```python
輸入圖像（已色偏校正）+ 大氣光 + 參數
    ↓
Step 1: 計算初始透射率
    t = 1 - omega * min(I/A)
    ↓
Step 2: 應用梯度約束
    weight = exp(-|gradient|)
    ↓
Step 3: 引導濾波細化
    t_refined = GuidedFilter(I_gray, t)
    ↓
Step 4: 恢復影像
    J = (I - A) / t + A
    ↓
Step 5: 色彩拉伸
    拉伸到 [L_low, L_high] 百分位
    ↓
輸出增強圖像
```

**使用範例**:
```python
from matlab_style_enhancement import AtmosphericLightEstimator, MATLABStyleEnhancement
import torch

# 大氣光估算
estimator = AtmosphericLightEstimator()
A = estimator(img)  # img: (H,W,3) numpy array

# 增強
enhancer = MATLABStyleEnhancement()
enhanced, intermediate = enhancer(
    img_tensor,     # (B, 3, H, W)
    params,         # dict of tensors
    atmos_tensor    # (B, 3)
)
```

---

### 3. `parameter_predictor.py`
**功能**: VGG-16 參數預測網路

**核心類別**:
- `MATLABParameterPredictor`: 預測所有增強參數

**網路架構**:
```
輸入圖像 (224×224×3)
    ↓
VGG-16 特徵提取 (conv1-conv4_3)
    ↓
雙池化 (Avg + Max)
    ↓
融合統計特徵 (79維)
    ↓
全連接層 + BatchNorm + Dropout
    ↓
注意力機制
    ↓
4個參數預測頭
    ↓
輸出參數:
- omega: [0.3, 0.9]
- guided_radius: [5, 30]
- L_low: [2, 15]
- L_high: [85, 98]
```

**使用範例**:
```python
from parameter_predictor import MATLABParameterPredictor

model = MATLABParameterPredictor(pretrained=True)
params = model(img_vgg, features)  # dict of tensors
```

---

### 4. `train_matlab_style.py`
**功能**: 完整訓練腳本

**訓練流程**:
```
數據載入
    ├─ 色偏校正 (預處理)
    ├─ 大氣光估算 (預處理)
    └─ 提取統計特徵
    ↓
前向傳播
    ├─ VGG-16 預測參數
    ├─ MATLAB 風格增強
    └─ 計算損失 (L1 + L2 + Perceptual)
    ↓
反向傳播
    └─ 只更新參數預測網路
```

**使用方法**:
```bash
python train_matlab_style.py \
    --input /path/to/raw/images \
    --reference /path/to/reference/images \
    --output ./output_matlab \
    --epochs 50 \
    --batch-size 4 \
    --device cuda
```

**訓練特點**:
- ✅ 混合精度訓練 (AMP)
- ✅ 梯度裁剪
- ✅ 學習率調度 (Cosine Annealing)
- ✅ 早停機制 (15 epochs)
- ✅ 定期保存檢查點

---

### 5. `inference_matlab_style.py`
**功能**: 推理腳本

**推理流程**:
```
讀取圖像
    ↓
步驟 1: 色偏校正
    ↓
步驟 2: 大氣光估算
    ↓
步驟 3: 提取統計特徵
    ↓
步驟 4: 預測參數
    ├─ omega
    ├─ guided_radius
    ├─ L_low
    └─ L_high
    ↓
步驟 5: MATLAB 風格增強
    ├─ 透射率計算
    ├─ 梯度約束
    ├─ 引導濾波
    ├─ 影像恢復
    └─ 色彩拉伸
    ↓
輸出增強圖像
```

**使用方法**:
```bash
# 單張圖像
python inference_matlab_style.py \
    --input image.jpg \
    --output enhanced.png \
    --model best_model.pth \
    --device cuda

# 批量處理
python inference_matlab_style.py \
    --input input_folder/ \
    --output output_folder/ \
    --model best_model.pth \
    --device cuda
```

---

## 🚀 快速開始

### 安裝依賴
```bash
pip install torch torchvision opencv-python numpy scikit-image scipy tqdm
```

### 訓練模型
```bash
python train_matlab_style.py \
    --input ./raw_images \
    --reference ./reference_images \
    --output ./output \
    --epochs 50 \
    --batch-size 4
```

### 推理增強
```bash
python inference_matlab_style.py \
    --input ./test_images \
    --output ./enhanced_images \
    --model ./output/best_model.pth
```

---

## 📊 完整數據流

```
【訓練階段】
原始圖像
    ↓
色偏校正 (CPU, 不可微分)
    ↓
大氣光估算 (CPU, 不可微分)
    ↓
提取統計特徵
    ↓
┌─────────────────────┬──────────────────────┐
│  VGG 特徵提取       │  統計特徵 (79維)      │
│  (GPU, 可微分)      │                      │
└─────────────────────┴──────────────────────┘
    ↓
參數預測 (GPU, 可微分)
    ↓
MATLAB 風格增強 (GPU, 可微分)
    ├─ 透射率計算
    ├─ 梯度約束
    ├─ 引導濾波
    ├─ 影像恢復
    └─ 色彩拉伸
    ↓
損失計算 & 反向傳播
    ↓
更新參數預測網路

【推理階段】
與訓練流程相同，但不計算梯度
```

---

## 🎯 核心優勢

### ✅ 完全遵循 MATLAB 流程
- 色偏校正：LAB 空間分析，4種色偏類型
- 大氣光估算：四叉樹分割 + Q 值評估
- 透射率細化：梯度約束 + 引導濾波
- 色彩拉伸：百分位拉伸

### ✅ 深度學習自適應
- 自動預測最佳參數
- 端到端可微分訓練
- VGG-16 遷移學習

### ✅ 高效實用
- 混合精度訓練
- 批量處理支持
- GPU 加速

---

## 📝 參數範圍說明

| 參數 | 範圍 | 說明 | MATLAB 原值 |
|------|------|------|-------------|
| **omega** | [0.3, 0.9] | 去霧強度，越大去霧越強 | 0.5 (固定) |
| **guided_radius** | [5, 30] | 引導濾波窗口半徑 | 15 (固定) |
| **L_low** | [2, 15] | 色彩拉伸下界百分位 | 15 (固定) |
| **L_high** | [85, 98] | 色彩拉伸上界百分位 | 95 (固定) |

---

## 🔧 進階使用

### 只做色偏校正
```python
from color_correction import ColorCorrection

corrector = ColorCorrection()
img_corrected, color_type = corrector(img)
```

### 只做大氣光估算
```python
from matlab_style_enhancement import AtmosphericLightEstimator

estimator = AtmosphericLightEstimator()
atmospheric_light = estimator(img)
```

### 自訂參數增強
```python
from matlab_style_enhancement import MATLABStyleEnhancement
import torch

enhancer = MATLABStyleEnhancement()

# 自訂參數
params = {
    'omega': torch.tensor([[0.6]]),
    'guided_radius': torch.tensor([[20.0]]),
    'L_low': torch.tensor([[10.0]]),
    'L_high': torch.tensor([[90.0]])
}

enhanced, _ = enhancer(img_tensor, params, atmos_tensor)
```

---

## ⚠️ 注意事項

1. **色偏校正和大氣光估算不參與訓練**
   - 在數據載入時完成
   - 作為常數傳入增強模組
   - 確保與 MATLAB 完全一致

2. **引導濾波半徑為動態參數**
   - 每張圖像可能不同
   - 需要逐張處理
   - 會影響訓練速度

3. **色彩拉伸參與訓練**
   - 使用可微分的百分位近似
   - L_low, L_high 由網路預測
   - 比固定參數更靈活

4. **GPU 記憶體需求**
   - 建議 8GB+ VRAM
   - batch_size=4 約需 6GB
   - 可降低 batch_size 或使用 CPU

---

## 📞 聯絡資訊

如有問題或建議，歡迎聯繫！

---

## 📄 授權

MIT License

---

**最後更新**: 2025-01-07
