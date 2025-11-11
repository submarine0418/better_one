"""
輕量級 CNN 色彩校正模組 (Lightweight CNN Color Correction Module)
可微分的深度學習色彩校正方法，接口與 LAB 方法完全一致
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2


class LightweightColorCorrectionNet(nn.Module):
    """
    輕量級 CNN 色彩校正網路
    使用深度可分離卷積減少參數量，同時保持較好的校正效果
    """
    
    def __init__(self, base_channels=16):
        super().__init__()
        
        # 第一層：淺層特徵提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 第二層：中層特徵處理
        self.conv2 = nn.Sequential(
            # 深度可分離卷積
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, 
                     groups=base_channels, bias=False),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # 第三層：細節保留
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 輸出層：色彩校正映射
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=1, bias=True),
            nn.Sigmoid()  # 輸出範圍 [0, 1]
        )
        
        # 殘差權重（可學習）
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: (B, 3, H, W) 輸入圖像 [0, 1]
        
        Returns:
            corrected: (B, 3, H, W) 校正後圖像 [0, 1]
        """
        identity = x  # 保存原圖用於殘差連接
        
        # 特徵提取與處理
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 生成校正映射
        correction = self.conv_out(x)
        
        # 殘差連接：混合校正結果和原圖
        # 使用可學習的權重平衡
        alpha = torch.sigmoid(self.residual_weight)  # 確保在 [0, 1]
        corrected = alpha * correction + (1 - alpha) * identity
        
        return torch.clamp(corrected, 0, 1)


class ColorCorrectionCNN:
    """
    CNN 色彩校正器（與 LAB 方法接口一致）
    
    使用方式：
        corrector = ColorCorrectionCNN(model_path='model.pth', device='cuda')
        corrected_img, color_type = corrector(img)
    """
    
    def __init__(self, model_path=None, device='cuda', base_channels=16):
        """
        初始化 CNN 色彩校正器
        
        Args:
            model_path: 預訓練模型路徑（.pth 文件）
            device: 'cuda' 或 'cpu'
            base_channels: 網路基礎通道數（16 或 32）
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 初始化網路
        self.net = LightweightColorCorrectionNet(base_channels=base_channels).to(self.device)
        
        # 載入預訓練模型
        if model_path is not None and Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.net.load_state_dict(checkpoint)
                print(f"✓ 成功載入預訓練模型: {model_path}")
            except Exception as e:
                print(f"⚠ 載入模型失敗，使用隨機初始化: {e}")
        else:
            print("⚠ 未指定模型路徑或文件不存在，使用隨機初始化")
        
        self.net.eval()  # 推理模式
        
        # 色偏類型檢測（保留與 LAB 方法一致的分類）
        self.color_types = ['greenish', 'blueish', 'yellowish', 'reddish', 'no_cast']
    
    def __call__(self, img):
        """
        主要接口函數（與 LAB 方法完全一致）
        
        Args:
            img: numpy array, shape (H, W, 3), RGB, [0, 255] uint8 或 [0, 1] float
        
        Returns:
            corrected_img: numpy array, shape (H, W, 3), RGB, [0, 1] float
            color_type: str, 色偏類型（用於兼容性）
        """
        # 確保輸入是 float [0, 1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
        
        # 保存原始形狀
        original_shape = img.shape
        
        # 檢測色偏類型（用於日誌記錄）
        color_type = self._detect_color_cast(img)
        
        # 轉換為 PyTorch tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # CNN 色彩校正
        with torch.no_grad():
            corrected_tensor = self.net(img_tensor)
        
        # 轉換回 numpy
        corrected = corrected_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        corrected = np.clip(corrected, 0.0, 1.0).astype(np.float32)
        
        # 確保輸出形狀一致
        assert corrected.shape == original_shape, f"形狀不匹配: {corrected.shape} vs {original_shape}"
        
        return corrected, color_type
    
    def _detect_color_cast(self, img):
        """
        檢測色偏類型（簡化版，用於日誌記錄）
        
        Args:
            img: numpy array (H, W, 3), RGB, [0, 1] float
        
        Returns:
            color_type: str
        """
        # 計算 RGB 通道均值
        mean_r = np.mean(img[:, :, 0])
        mean_g = np.mean(img[:, :, 1])
        mean_b = np.mean(img[:, :, 2])
        
        threshold = 0.05
        
        if mean_g > mean_r + threshold and mean_g > mean_b + threshold:
            return 'greenish'
        elif mean_b > mean_r + threshold and mean_b > mean_g + threshold:
            return 'blueish'
        elif mean_r > mean_b + threshold and mean_g > mean_b + threshold:
            return 'yellowish'
        elif mean_r > mean_g + threshold and mean_r > mean_b + threshold:
            return 'reddish'
        else:
            return 'no_cast'
    
    def to(self, device):
        """
        移動模型到指定設備
        
        Args:
            device: 'cuda' 或 'cpu'
        """
        self.device = torch.device(device)
        self.net.to(self.device)
        return self


# ============================================
# 訓練相關函數
# ============================================

class ColorCorrectionTrainer:
    """
    CNN 色彩校正訓練器
    """
    
    def __init__(self, model, device='cuda', lr=1e-3):
        """
        初始化訓練器
        
        Args:
            model: LightweightColorCorrectionNet 實例
            device: 'cuda' 或 'cpu'
            lr: 學習率
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 優化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 學習率調度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
    
    def train_step(self, input_img, target_img):
        """
        單步訓練
        
        Args:
            input_img: (B, 3, H, W) 輸入圖像 [0, 1]
            target_img: (B, 3, H, W) 目標圖像 [0, 1]
        
        Returns:
            loss: float
        """
        self.model.train()
        
        input_img = input_img.to(self.device)
        target_img = target_img.to(self.device)
        
        # 前向傳播
        output = self.model(input_img)
        
        # 損失函數：L1 + Perceptual Loss (簡化版)
        l1_loss = F.l1_loss(output, target_img)
        l2_loss = F.mse_loss(output, target_img)
        
        # 色彩一致性損失
        color_loss = self._color_consistency_loss(output, target_img)
        
        # 總損失
        total_loss = 0.3 * l1_loss + 0.3 * l2_loss + 0.4 * color_loss
        
        # 反向傳播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return total_loss.item()
    
    def _color_consistency_loss(self, pred, target):
        """
        色彩一致性損失（通道間平衡）
        
        Args:
            pred: (B, 3, H, W)
            target: (B, 3, H, W)
        
        Returns:
            loss: scalar
        """
        # 計算每個通道的均值
        pred_means = pred.mean(dim=[2, 3])  # (B, 3)
        target_means = target.mean(dim=[2, 3])  # (B, 3)
        
        # 通道間均值差異
        mean_diff = F.mse_loss(pred_means, target_means)
        
        return mean_diff
    
    def save_checkpoint(self, save_path, epoch, loss):
        """
        保存檢查點
        
        Args:
            save_path: 保存路徑
            epoch: 當前 epoch
            loss: 當前損失
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, save_path)
        print(f"✓ 檢查點已保存: {save_path}")


# ============================================
# 使用範例
# ============================================

if __name__ == '__main__':
    print("=" * 80)
    print("輕量級 CNN 色彩校正模組")
    print("=" * 80)
    
    # ============================================
    # 推理使用範例（與 LAB 方法接口一致）
    # ============================================
    print("\n【推理使用範例】")
    print("-" * 80)
    
    # 初始化（如果有預訓練模型）
    corrector = ColorCorrectionCNN(
        model_path='D:/research/better_one/color_correction_output/best_color_correction_model.pth',  # 可選
        device='cuda',
        base_channels=16
    )
    
    # 載入測試圖像
    img = np.random.rand(480, 640, 3).astype(np.float32)  # 模擬圖像
    
    # 色彩校正（接口與 LAB 方法完全一致）
    corrected_img, color_type = corrector(img)
    
    print(f"輸入圖像形狀: {img.shape}")
    print(f"輸出圖像形狀: {corrected_img.shape}")
    print(f"檢測到的色偏類型: {color_type}")
    
    # ============================================
    # 訓練使用範例
    # ============================================
    print("\n【訓練使用範例】")
    print("-" * 80)
    
    # 創建模型
    model = LightweightColorCorrectionNet(base_channels=16)
    
    # 創建訓練器
    trainer = ColorCorrectionTrainer(model, device='cuda', lr=1e-3)
    
    # 模擬訓練數據
    input_batch = torch.rand(4, 3, 256, 256)  # (B, C, H, W)
    target_batch = torch.rand(4, 3, 256, 256)
    
    # 訓練一步
    loss = trainer.train_step(input_batch, target_batch)
    print(f"訓練損失: {loss:.6f}")
    
    # 保存模型
    trainer.save_checkpoint('color_correction_model.pth', epoch=1, loss=loss)
    
    print("\n" + "=" * 80)
    print("✓ 模組測試完成")
    print("=" * 80)
