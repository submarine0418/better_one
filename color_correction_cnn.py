"""
輕量級 CNN 色彩校正模組 (Lightweight CNN Color Correction Module)
使用 scikit-image 在 LAB 色彩空間進行可微分的深度學習色彩校正
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from skimage import color as skcolor


class LightweightColorCorrectionNet(nn.Module):
    """
    輕量級 CNN 色彩校正網路（在 LAB 空間）
    輸入/輸出：LAB 色彩空間（經過正規化）
    """
    
    def __init__(self, base_channels=16):
        super().__init__()
        
        # 第一層：淺層特徵提取
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 第二層：中層特徵處理（深度可分離卷積）
        self.conv2 = nn.Sequential(
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
        self.conv_out = nn.Conv2d(base_channels, 3, kernel_size=1, bias=True)
        
        # 殘差權重（可學習）
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        """
        前向傳播（在 LAB 空間）
        
        Args:
            x: (B, 3, H, W) LAB 圖像，已正規化到 [0, 1]
               L: [0, 1] (對應原始 [0, 100])
               a: [0, 1] (對應原始 [-128, 127]，中心 0.5)
               b: [0, 1] (對應原始 [-128, 127]，中心 0.5)
        
        Returns:
            corrected: (B, 3, H, W) LAB 圖像 [0, 1]
        """
        identity = x  # 保存原圖用於殘差連接
        
        # 特徵提取與處理
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 生成校正映射（不使用 sigmoid，因為 LAB 範圍不同）
        correction = self.conv_out(x)
        
        # 殘差連接
        alpha = torch.sigmoid(self.residual_weight)
        corrected = identity + alpha * correction
        
        # 確保在有效範圍內
        return torch.clamp(corrected, 0.0, 1.0)


class ColorCorrectionCNN:
    """
    CNN 色彩校正接口（使用 scikit-image 在 LAB 空間）
    """
    
    def __init__(self, model_path=None, device='cuda', base_channels=16):
        """
        初始化
        
        Args:
            model_path: 預訓練模型路徑
            device: 'cuda' 或 'cpu'
            base_channels: 網路基礎通道數
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 初始化網路
        self.net = LightweightColorCorrectionNet(base_channels=base_channels).to(self.device)
        
        # 載入預訓練模型
        if model_path is not None and Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    self.net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.net.load_state_dict(checkpoint)
                print(f"✓ 成功載入預訓練模型: {model_path}")
            except Exception as e:
                print(f"⚠ 載入模型失敗，使用隨機初始化: {e}")
        else:
            if model_path:
                print(f"⚠ 模型文件不存在: {model_path}")
            print("⚠ 使用隨機初始化")
        
        self.net.eval()  # 推理模式
        
        # 色偏類型
        self.color_types = ['greenish', 'blueish', 'yellowish', 'reddish', 'whitish', 'no_cast']
    
    def __call__(self, img):
        """
        色彩校正（接口與 LAB 方法完全一致）
        
        Args:
            img: numpy array (H, W, 3), RGB, [0, 1] float 或 uint8
        
        Returns:
            corrected_img: numpy array (H, W, 3), RGB, [0, 1] float
            color_type: str
        """
        # 確保輸入是 float [0, 1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = np.clip(img.astype(np.float32), 0.0, 1.0)
        
        # 檢測色偏類型
        color_type = self._detect_color_cast(img)
        
        # RGB → LAB (使用 scikit-image)
        lab = skcolor.rgb2lab(img)  # L: [0, 100], a/b: [-128, 127]
        
        # 正規化 LAB 到 [0, 1]
        lab_normalized = self._normalize_lab(lab)
        
        # 轉換為 PyTorch tensor (HWC → CHW)
        lab_tensor = torch.from_numpy(lab_normalized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        # CNN 色彩校正
        with torch.no_grad():
            corrected_lab_tensor = self.net(lab_tensor)
        
        # 轉換回 numpy (CHW → HWC)
        corrected_lab_normalized = corrected_lab_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # 反正規化 LAB
        corrected_lab = self._denormalize_lab(corrected_lab_normalized)
        
        # LAB → RGB (使用 scikit-image)
        corrected_rgb = skcolor.lab2rgb(corrected_lab)
        
        # 確保範圍 [0, 1]
        corrected_rgb = np.clip(corrected_rgb, 0.0, 1.0).astype(np.float32)
        
        return corrected_rgb, color_type
    
    def _normalize_lab(self, lab):
        """
        正規化 LAB 到 [0, 1]
        
        Args:
            lab: (H, W, 3) numpy array
                 L: [0, 100]
                 a: [-128, 127]
                 b: [-128, 127]
        
        Returns:
            lab_normalized: (H, W, 3) [0, 1]
        """
        lab_normalized = np.zeros_like(lab, dtype=np.float32)
        
        # L: [0, 100] → [0, 1]
        lab_normalized[:, :, 0] = lab[:, :, 0] / 100.0
        
        # a: [-128, 127] → [0, 1]
        lab_normalized[:, :, 1] = (lab[:, :, 1] + 128.0) / 255.0
        
        # b: [-128, 127] → [0, 1]
        lab_normalized[:, :, 2] = (lab[:, :, 2] + 128.0) / 255.0
        
        return lab_normalized
    
    def _denormalize_lab(self, lab_normalized):
        """
        反正規化 LAB 從 [0, 1] 回到原始範圍
        
        Args:
            lab_normalized: (H, W, 3) [0, 1]
        
        Returns:
            lab: (H, W, 3)
                 L: [0, 100]
                 a: [-128, 127]
                 b: [-128, 127]
        """
        lab = np.zeros_like(lab_normalized, dtype=np.float32)
        
        # L: [0, 1] → [0, 100]
        lab[:, :, 0] = lab_normalized[:, :, 0] * 100.0
        
        # a: [0, 1] → [-128, 127]
        lab[:, :, 1] = lab_normalized[:, :, 1] * 255.0 - 128.0
        
        # b: [0, 1] → [-128, 127]
        lab[:, :, 2] = lab_normalized[:, :, 2] * 255.0 - 128.0
        
        return lab
    
    def _detect_color_cast(self, img):
        """
        檢測色偏類型（在 LAB 空間）
        
        Args:
            img: numpy array (H, W, 3), RGB, [0, 1]
        
        Returns:
            color_type: str
        """
        # RGB → LAB
        lab = skcolor.rgb2lab(img)
        
        # 提取 a, b 通道的均值
        mean_a = float(np.mean(lab[:, :, 1]))  # [-128, 127]
        mean_b = float(np.mean(lab[:, :, 2]))  # [-128, 127]
        
        # 檢查過白/過曝
        mean_L = float(np.mean(lab[:, :, 0]))  # [0, 100]
        if mean_L > 85:
            overexposed_pixels = np.sum(lab[:, :, 0] > 95)
            overexposed_ratio = overexposed_pixels / lab[:, :, 0].size
            if overexposed_ratio > 0.3:
                return 'whitish'
        
        # 判斷色偏類型（基於 a, b 通道）
        threshold = 3.0  # LAB 空間的閾值
        
        if abs(mean_a) < threshold and abs(mean_b) < threshold:
            return 'no_cast'
        
        # 判斷主要色偏方向
        if abs(mean_a) >= abs(mean_b):
            # a 軸主導
            return 'reddish' if mean_a > 0 else 'greenish'
        else:
            # b 軸主導
            return 'yellowish' if mean_b > 0 else 'blueish'
    
    def to(self, device):
        """移動模型到指定設備"""
        self.device = torch.device(device)
        self.net.to(self.device)
        return self


# ============================================
# 訓練相關函數（在 LAB 空間）
# ============================================

class ColorCorrectionTrainer:
    """
    CNN 色彩校正訓練器（在 LAB 空間）
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        # 學習率調度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        # 損失函數
        self.l1_loss = nn.L1Loss()
    
    def train_step(self, input_batch, target_batch):
        """
        單步訓練（輸入是 RGB，內部轉換為 LAB）
        
        Args:
            input_batch: (B, 3, H, W) RGB tensor [0, 1]
            target_batch: (B, 3, H, W) RGB tensor [0, 1]
        
        Returns:
            loss: float
        """
        self.model.train()
        
        # 確保數據在正確的設備上
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        
        # RGB → LAB (batch 處理)
        input_lab = self._batch_rgb_to_lab(input_batch)
        target_lab = self._batch_rgb_to_lab(target_batch)
        
        # 前向傳播
        output_lab = self.model(input_lab)
        
        # 損失函數
        # L1 損失
        l1_loss = self.l1_loss(output_lab, target_lab)
        ssim_loss = 1 - self.ssim(output_lab, target_lab, window=self.window, window_size=self.window_size, channel=3)
        # LAB 空間的色彩一致性損失
        color_loss = self._lab_color_consistency_loss(output_lab, target_lab)
        
        # 總損失
        total_loss = 0.3 * l1_loss + 0.3 * color_loss + 0.4 * ssim_loss
        
        # 反向傳播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return total_loss.item()
    
    def _batch_rgb_to_lab(self, rgb_batch):
        """
        批次 RGB → LAB 轉換（使用 scikit-image）
        
        Args:
            rgb_batch: (B, 3, H, W) RGB tensor [0, 1]
        
        Returns:
            lab_batch: (B, 3, H, W) LAB tensor [0, 1] (正規化)
        """
        B, C, H, W = rgb_batch.shape
        lab_batch = torch.zeros_like(rgb_batch)
        
        for i in range(B):
            # (C, H, W) → (H, W, C)
            rgb_img = rgb_batch[i].permute(1, 2, 0).cpu().numpy()
            
            # RGB → LAB (scikit-image)
            lab_img = skcolor.rgb2lab(rgb_img)
            
            # 正規化
            lab_normalized = np.zeros_like(lab_img, dtype=np.float32)
            lab_normalized[:, :, 0] = lab_img[:, :, 0] / 100.0
            lab_normalized[:, :, 1] = (lab_img[:, :, 1] + 128.0) / 255.0
            lab_normalized[:, :, 2] = (lab_img[:, :, 2] + 128.0) / 255.0
            
            # (H, W, C) → (C, H, W)
            lab_tensor = torch.from_numpy(lab_normalized).permute(2, 0, 1)
            lab_batch[i] = lab_tensor
        
        return lab_batch.to(self.device)
    
    def _lab_color_consistency_loss(self, pred_lab, target_lab):
        """
        LAB 空間的色彩一致性損失
        
        Args:
            pred_lab: (B, 3, H, W) LAB [0, 1]
            target_lab: (B, 3, H, W) LAB [0, 1]
        
        Returns:
            loss: scalar
        """
        # 分別處理 L 和 ab 通道
        # L 通道（亮度）
        loss_L = F.l1_loss(pred_lab[:, 0:1, :, :], target_lab[:, 0:1, :, :])
        
        # a, b 通道（色彩）- 給予更高權重
        loss_ab = F.l1_loss(pred_lab[:, 1:3, :, :], target_lab[:, 1:3, :, :])
        
        # 綜合損失
        return loss_L + 1.5 * loss_ab
    
    def save_checkpoint(self, save_path, epoch, loss):
        """保存檢查點"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, save_path)
        print(f"✓ 檢查點已保存: {save_path}")
    
    def load_checkpoint(self, load_path):
        """載入檢查點"""
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✓ 已載入檢查點: {load_path}")
        return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """
        Calculate SSIM
        
        Args:
            img1, img2: Input images (B, C, H, W)
            window: Gaussian window
            window_size: Window size
            channel: Number of channels
            size_average: Whether to average the loss
        
        Returns:
            SSIM value
        """
        # Constants (to avoid division by zero)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Ensure window is on the correct device
        window = window.to(img1.device)
        
        # Calculate mean μ
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Calculate variance σ² and covariance σ₁₂
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
# ============================================
# 使用範例
# ============================================

if __name__ == '__main__':
    print("=" * 80)
    print("輕量級 CNN 色彩校正模組（scikit-image LAB 空間）")
    print("=" * 80)
    
    # ============================================
    # 推理使用範例
    # ============================================
    print("\n【推理使用範例】")
    print("-" * 80)
    
    # 初始化
    corrector = ColorCorrectionCNN(
        model_path=r'D:\research\better_one\color_correction_output\best_color_correction_model.pth',
        device='cuda',
        base_channels=16
    )
    
    # 載入測試圖像
    img = np.random.rand(480, 640, 3).astype(np.float32)
    
    # 色彩校正
    corrected_img, color_type = corrector(img)
    
    print(f"✓ 輸入圖像形狀: {img.shape}")
    print(f"✓ 輸出圖像形狀: {corrected_img.shape}")
    print(f"✓ 檢測到的色偏類型: {color_type}")
    print(f"✓ 輸出範圍: [{corrected_img.min():.3f}, {corrected_img.max():.3f}]")
    
    # ============================================
    # 驗證 LAB 轉換
    # ============================================
    print("\n【驗證 LAB 轉換】")
    print("-" * 80)
    
    # 測試 RGB → LAB → RGB 是否一致
    test_img = np.array([[[1.0, 0.5, 0.0]]], dtype=np.float32)  # 橙色
    lab = skcolor.rgb2lab(test_img)
    rgb_back = skcolor.lab2rgb(lab)
    
    print(f"原始 RGB: {test_img[0, 0]}")
    print(f"LAB: {lab[0, 0]}")
    print(f"轉回 RGB: {rgb_back[0, 0]}")
    print(f"誤差: {np.abs(test_img - rgb_back).max():.6f}")
    
    print("\n" + "=" * 80)
    print("✓ 測試完成")
    print("=" * 80)