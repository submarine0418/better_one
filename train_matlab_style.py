"""
MATLAB 風格水下影像增強訓練腳本（混合方案）
預處理：色偏校正 + 大氣光估算
訓練：參數預測網路
增強：透射率計算 + 引導濾波 + 影像恢復 + 色彩拉伸
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
# 導入自定義模組
from color_correction import ColorCorrection
from matlab_style_enhancement import AtmosphericLightEstimator, MATLABStyleEnhancement
from parameter_predictor import MATLABParameterPredictor

# 修改這兩行導入，加入底層網路類別
from color_correction_cnn import ColorCorrectionCNN, LightweightColorCorrectionNet
from airlight_CNN import make_atmospheric_light_cnn, LightweightAtmosphericLightNet

# ============================================
# GPU 色彩轉換工具 (移到這裡)
# ============================================

class RGB2Lab_GPU(nn.Module):
    """
    PyTorch GPU 版 RGB -> LAB (符合您模型的正規化標準)
    輸入: (B, 3, H, W) RGB [0, 1]
    輸出: (B, 3, H, W) LAB Normalized (L:0-1, a:0-1, b:0-1)
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('matrix_srgb2xyz', torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=torch.float32))
        self.register_buffer('d65', torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32))

    def f_func(self, t):
        mask = t > 0.008856
        return torch.where(mask, torch.pow(t, 1/3), 7.787 * t + 16/116)

    def forward(self, rgb):
        # 1. Inverse sRGB Gamma
        mask = rgb > 0.04045
        rgb_linear = torch.where(mask, torch.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
        
        # 2. RGB -> XYZ
        xyz = torch.matmul(rgb_linear.permute(0, 2, 3, 1), self.matrix_srgb2xyz.t())
        xyz = xyz / self.d65
        
        # 3. XYZ -> LAB
        xyz_f = self.f_func(xyz)
        L = 116 * xyz_f[..., 1] - 16
        a = 500 * (xyz_f[..., 0] - xyz_f[..., 1])
        b = 200 * (xyz_f[..., 1] - xyz_f[..., 2])
        
        # 4. Normalize (符合您原本 Dataset 的邏輯)
        # L: 0-100 -> 0-1
        L_norm = L / 100.0
        # a, b: -128~127 -> 0-1
        a_norm = (a + 128.0) / 255.0
        b_norm = (b + 128.0) / 255.0
        
        return torch.stack([L_norm, a_norm, b_norm], dim=-1).permute(0, 3, 1, 2)

class Lab2RGB_GPU(nn.Module):
    """
    PyTorch GPU 版 LAB -> RGB
    輸入: (B, 3, H, W) LAB Normalized
    輸出: (B, 3, H, W) RGB [0, 1]
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('matrix_xyz2srgb', torch.tensor([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252]
        ], dtype=torch.float32))
        self.register_buffer('d65', torch.tensor([0.95047, 1.00000, 1.08883], dtype=torch.float32))

    def forward(self, lab_norm):
        # 1. Denormalize
        L = lab_norm[:, 0, :, :] * 100.0
        a = lab_norm[:, 1, :, :] * 255.0 - 128.0
        b = lab_norm[:, 2, :, :] * 255.0 - 128.0
        
        # 2. LAB -> XYZ
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        fx3, fy3, fz3 = fx**3, fy**3, fz**3
        
        x = torch.where(fx3 > 0.008856, fx3, (fx - 16/116) / 7.787)
        y = torch.where(L > 8, fy3, L / 903.3)
        z = torch.where(fz3 > 0.008856, fz3, (fz - 16/116) / 7.787)
        
        xyz = torch.stack([x, y, z], dim=-1) * self.d65
        
        # 3. XYZ -> RGB
        rgb_linear = torch.matmul(xyz, self.matrix_xyz2srgb.t())
        
        # 4. Gamma Correction
        mask = rgb_linear > 0.0031308
        rgb = torch.where(mask, 1.055 * torch.pow(rgb_linear, 1/2.4) - 0.055, 12.92 * rgb_linear)
        
        return torch.clamp(rgb.permute(0, 3, 1, 2), 0, 1)

# ============================================
# 數據集（帶預處理）
# ============================================

class MATLABStyleDataset(Dataset):
    """
    色偏校正 + 大氣光估算
    """
    
    def __init__(self, image_folder, reference_folder, target_size=224, 
                 augment=True):
        """
        Args:
            image_folder: 輸入圖像資料夾
            reference_folder: 參考圖像資料夾
            target_size: 目標大小
            augment: 是否數據增強
        """
        self.image_folder = Path(image_folder)
        self.reference_folder = Path(reference_folder)
        self.target_size = target_size
        self.augment = augment
        
        # 找到所有圖像
        self.image_paths = list(self.image_folder.glob('*.*')) # 簡化寫法
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_folder}")
        
        print(f"找到 {len(self.image_paths)} 張圖像")
        
        # VGG normalization
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def load_image(self, path, target_size):
    
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_size, target_size), 
                        interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        return img
    
    def augment_pair(self, img, ref):
        
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            ref = np.fliplr(ref).copy()
        
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
            ref = np.flipud(ref).copy()
        
        return img, ref
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 1. 只讀取原始圖像 (Raw Image)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_size, self.target_size))
        img = img.astype(np.float32) / 255.0
        
        # 2. 讀取參考圖像
        ref_path = self.reference_folder / img_path.name
        if ref_path.exists():
            ref = cv2.imread(str(ref_path))
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
            ref = cv2.resize(ref, (self.target_size, self.target_size))
            ref = ref.astype(np.float32) / 255.0
        else:
            ref = img.copy() # Fallback
            
        # 3. 數據增強
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy(); ref = np.fliplr(ref).copy()
            if np.random.rand() > 0.5:
                img = np.flipud(img).copy(); ref = np.flipud(ref).copy()
        
        # 4. 轉 Tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        ref_tensor = torch.from_numpy(ref).permute(2, 0, 1).float()
        
        
        return {
            'image': img_tensor,      # Raw RGB
            'reference': ref_tensor,  # GT RGB
            'path': str(img_path)
        }


# ============================================
# Loss Functions
# ============================================
class SSIMLoss(nn.Module):
    """
    結構相似性損失 (Structural Similarity Index Loss)
    
    SSIM 比較兩張圖像的:
    1. 亮度 (Luminance)
    2. 對比度 (Contrast)  
    3. 結構 (Structure)
    
    return 1 - SSIM 
    """
    def __init__(self, window_size=11, size_average=True, channel=3):
        """
        Args:
            window_size: (default: 11)
            size_average: whether to average the loss (default: True)
            channel: number of image channels (default: 3 for RGB)
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        
        # Create Gaussian window
        self.window = self.create_window(window_size, channel)
    
    def gaussian(self, window_size, sigma=1.5):
        """Create 1D Gaussian kernel"""
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / float(2*sigma**2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        """Create 2D Gaussian window"""
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
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
    
    def forward(self, img1, img2):
        """
        Args:
            img1, img2: (B, C, H, W) tensors, 值域 [0, 1]
        
        Returns:
            loss: 1 - SSIM 
        """
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        
        ssim_value = self.ssim(img1, img2, window, self.window_size, channel, self.size_average)
        
        # Return loss (1 - SSIM)
        return 1 - ssim_value

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self, device='cuda'):
        super().__init__()
        from torchvision import models
        from torchvision.models import VGG16_Weights
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
    
    def forward(self, pred, target):
        pred_feat = self.vgg(pred)
        target_feat = self.vgg(target)
        return nn.functional.mse_loss(pred_feat, target_feat)



        
class FullSSIMCombinedLoss(nn.Module):
    def __init__(self, ssim_weight=0.8, l1_weight=0.2, ab_reg_weight=0.05): # 新增權重參數
        super().__init__()
        self.ssim_module = SSIMLoss()
        self.l1_module = nn.L1Loss()
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.ab_reg_weight = ab_reg_weight # 設定權重 (例如 0.05)
        
        # 載入 GPU 版 LAB 轉換器 (用於計算 Loss)
        self.rgb2lab = RGB2Lab_GPU() 

    def forward(self, enhanced, reference):
        # 1. 原有的 Loss
        ssim_loss = self.ssim_module(enhanced, reference)
        l1_loss = self.l1_module(enhanced, reference)
        
        # 2. 新增：AB 通道約束 (防止色偏)
        # 先把增強後的圖轉成 LAB
        lab_enhanced = self.rgb2lab(enhanced) 
        
        # 取出 a, b 通道 (L是第0個, a是第1個, b是第2個)
        # lab_enhanced shape: [Batch, 3, H, W]
        a_channel = lab_enhanced[:, 1, :, :]
        b_channel = lab_enhanced[:, 2, :, :]
        
        # 計算 AB 的平均絕對值 (L1) 或 平方和 (L2)
        # 這裡假設我們希望色彩不要太誇張，所以懲罰過大的絕對值
        # 或者，更進階一點：懲罰 "增強圖" 與 "參考圖" 之間 AB 通道的差異
        
        # 方法 A: 單純限制飽和度 (防止變成霓虹燈)
        # ab_reg_loss = torch.mean(torch.abs(a_channel)) + torch.mean(torch.abs(b_channel))
        
        # 方法 B (推薦): 確保色調跟 Reference 接近
        lab_ref = self.rgb2lab(reference)
        ab_diff = torch.abs(lab_enhanced[:, 1:] - lab_ref[:, 1:]) # 只看 a, b 通道差異
        ab_reg_loss = torch.mean(ab_diff)

        # 總 Loss
        total_loss = (self.ssim_weight * ssim_loss + 
                      self.l1_weight * l1_loss + 
                      self.ab_reg_weight * ab_reg_loss)
                      
        return total_loss, {'ssim': ssim_loss.item(), 'l1': l1_loss.item(), 'ab_reg': ab_reg_loss.item()}

# ============================================
# Trainer
# ============================================

class MATLABStyleTrainer:
    def __init__(self, device='cuda', use_amp=True):
        """
        Args:
            device: 'cuda' or 'cpu'
            use_amp: 是否使用混合精度訓練
        """
        self.device = device
        self.use_amp = use_amp and (device == 'cuda')
        
        # 1. 載入預處理模型 (Color Correction)
        self.color_net = LightweightColorCorrectionNet(base_channels=16).to(device)
        ckpt_color = torch.load(r"D:\research\better_one\color_correction_cnn_model\best_color_correction_model.pth", map_location=device)
        self.color_net.load_state_dict(ckpt_color['model_state_dict'])
        self.color_net.eval() # 凍結
        for p in self.color_net.parameters(): p.requires_grad = False
        
        # 2. 載入預處理模型 (Airlight)
        self.airlight_net = LightweightAtmosphericLightNet(base_channels=16).to(device)
        ckpt_air = torch.load(r"D:\research\better_one\airlight_cnn_model\best_atmospheric_light_model.pth", map_location=device)
        self.airlight_net.load_state_dict(ckpt_air['model_state_dict'])
        self.airlight_net.eval() # 凍結
        for p in self.airlight_net.parameters(): p.requires_grad = False
        
        # 3. GPU 轉換工具
        self.rgb2lab = RGB2Lab_GPU().to(device)
        self.lab2rgb = Lab2RGB_GPU().to(device)
        self.vgg_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 初始化模型
        self.param_predictor = MATLABParameterPredictor(
            pretrained=True, 
            hidden_dim=256, 
            use_features=False  
        ).to(device)
        
        self.enhancement = MATLABStyleEnhancement().to(device)
        
        # 修改這裡：移除 device=device
        self.criterion = FullSSIMCombinedLoss().to(device) 
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.param_predictor.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Mixed precision
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Records
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.param_predictor.train()
        self.enhancement.eval()  # Enhancement module not trained
        
        total_loss = 0
        # 修改這裡：加入 'ab_reg': 0
        loss_components = {'ssim': 0, 'l1': 0, 'ab_reg': 0} 
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 1. 獲取原始圖像 (Raw RGB)
            raw_images = batch['image'].to(self.device)
            references = batch['reference'].to(self.device)
            
            # 2. 在 GPU 上進行預處理 (極快)
            with torch.no_grad():
                # RGB -> LAB
                lab_images = self.rgb2lab(raw_images)
                
                # 色彩校正 (LAB -> LAB)
                corrected_lab = self.color_net(lab_images)
                
                # LAB -> RGB (得到校正後的 RGB)
                corrected_rgb = self.lab2rgb(corrected_lab)
                
                # 大氣光估算 (輸入校正後的 RGB)
                atmos_light = self.airlight_net(corrected_rgb)
                
                # VGG Normalize (給參數預測網路用)
                images_vgg = self.vgg_norm(corrected_rgb)

            # 3. 準備 Features (因為 use_features=False，我們給一個全 0 的 Tensor 即可)
            # 尺寸: (Batch_Size, 79)
            features = torch.zeros(raw_images.size(0), 79, device=self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    # Predict parameters (使用校正後的圖)
                    params = self.param_predictor(images_vgg, features)
                    
                    # Apply enhancement (注意：這裡要傳入 corrected_rgb，因為原本 Dataset 回傳的就是校正後的圖)
                    enhanced, _ = self.enhancement(corrected_rgb, params, atmos_light)
                    
                    # Calculate loss
                    loss, components = self.criterion(enhanced, references)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.param_predictor.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                params = self.param_predictor(images_vgg, features)
                enhanced, _ = self.enhancement(corrected_rgb, params, atmos_light)
                loss, components = self.criterion(enhanced, references)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.param_predictor.parameters(), 1.0)
                self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v
            
            # update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
        # print(f"Batch images device: {raw_images.device}") 
        self.train_losses.append(avg_loss)
        return avg_loss, avg_components
    
    def validate(self, dataloader):
        """驗證"""
        self.param_predictor.eval()
        self.enhancement.eval()
        
        total_loss = 0
        # 修改這裡：加入 'ab_reg': 0
        loss_components = {'ssim': 0, 'l1': 0, 'ab_reg': 0}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                # 1. 獲取數據
                raw_images = batch['image'].to(self.device)
                references = batch['reference'].to(self.device)
                
                # 2. GPU 預處理 (與訓練時相同)
                lab_images = self.rgb2lab(raw_images)
                corrected_lab = self.color_net(lab_images)
                corrected_rgb = self.lab2rgb(corrected_lab)
                atmos_light = self.airlight_net(corrected_rgb)
                images_vgg = self.vgg_norm(corrected_rgb)
                features = torch.zeros(raw_images.size(0), 79, device=self.device)
                
                # 3. 推理
                params = self.param_predictor(images_vgg, features)
                enhanced, _ = self.enhancement(corrected_rgb, params, atmos_light)
                loss, components = self.criterion(enhanced, references)
                
                total_loss += loss.item()
                for k, v in components.items():
                    loss_components[k] += v
        
        avg_loss = total_loss / len(dataloader)
        avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
        
        self.val_losses.append(avg_loss)
        return avg_loss, avg_components
    
    def save(self, path, epoch=None, metrics=None):
        """Save checkpoint"""
        checkpoint = {
            'model_state_dict': self.param_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")
    
    def load(self, path):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.param_predictor.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"✓ Checkpoint loaded: {path}")


# ============================================
# Main training function
# ============================================

def train_matlab_style(image_folder, reference_folder, output_folder,
                       epochs=100, batch_size=4, device='cuda',
                       use_amp=True, resume=None):
    """
    主訓練函數
    
    Args:
        image_folder: 輸入圖像資料夾
        reference_folder: 參考圖像資料夾
        output_folder: 輸出資料夾
        epochs: 訓練輪數
        batch_size: 批次大小
        device: 'cuda' or 'cpu'
        use_amp: 是否使用混合精度
        resume: 恢復訓練的檢查點路徑
    """
    
    # 設定設備
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
        use_amp = False
    
    if device == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 創建輸出目錄
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Enable Autograd anomaly detection (for debugging, remove after locating issues)
    # try:
    #     # torch.autograd.set_detect_anomaly(True)
    # except Exception:
    #     # If running in an environment without torch.autograd, ignore
    #     pass

    print("=" * 80)
    print(" Underwater Image Enhancement Training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Mixed Precision: {use_amp}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print("=" * 80)
    
    try:
        # 載入數據集
        print("\n載入數據集...")
        train_dataset = MATLABStyleDataset(
            image_folder, reference_folder,
            target_size=224, augment=True
        )
        
        val_dataset = MATLABStyleDataset(
            image_folder, reference_folder,
            target_size=224, augment=False
        )
        
        # 分割數據集
        total_size = len(train_dataset)
        train_size = int(0.85 * total_size)
        val_size = total_size - train_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        # Windows requires num_workers=0 to avoid serialization issues
        import platform
        num_workers = 0 if platform.system() == 'Windows' else 2
        
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
        
        print(f"Training samples: {len(train_subset)}")
        print(f"Validation samples: {len(val_subset)}")
        
        # Initialize trainer
        print("\nInitializing trainer...")
        trainer = MATLABStyleTrainer(device=device, use_amp=use_amp)
        
        # Resume training
        start_epoch = 0
        if resume and Path(resume).exists():
            print(f"Resuming training: {resume}")
            trainer.load(resume)
            start_epoch = len(trainer.train_losses)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        print("\n" + "=" * 80)
        print("Starting Training")
        print("=" * 80)
        
        for epoch in range(start_epoch, epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*80}")
            
            # 訓練
            train_loss, train_components = trainer.train_epoch(train_loader, epoch + 1)
            print(f"\nTrain Loss: {train_loss:.6f}")
            print(f"  SSIM: {train_components['ssim']:.6f}, "
                  f"L1: {train_components['l1']:.6f}, "
                  f"AB_Reg: {train_components.get('ab_reg', 0):.6f}" # <--- 新增這行
                )
            
            # 驗證
            val_loss, val_components = trainer.validate(val_loader)
            print(f"Val Loss: {val_loss:.6f}")
            print(f"  SSIM: {val_components['ssim']:.6f}, "
                  f"L1: {val_components['l1']:.6f}, "
                  f"AB_Reg: {val_components.get('ab_reg', 0):.6f}" # <--- 新增這行
                  )
            
            # 更新學習率
            trainer.scheduler.step()
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                trainer.save(
                    f"{output_folder}/best_model.pth",
                    epoch=epoch + 1,
                    metrics={'val_loss': val_loss}
                )
                print(f"✓ New best model! Val Loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{max_patience}")
            
            # Periodic saving
            if (epoch + 1) % 10 == 0:
                trainer.save(f"{output_folder}/checkpoint_epoch_{epoch + 1}.pth")
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Clear GPU memory
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # Save final model
        trainer.save(f"{output_folder}/final_model.pth")
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best Val Loss: {best_val_loss:.6f}")
        print(f"Model saved at: {output_folder}")
        print("=" * 80)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted")
        print("Saving current state...")
        trainer.save(f"{output_folder}/interrupted_checkpoint.pth")
    
    except Exception as e:
        print(f"\n⚠ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if device == 'cuda':
            torch.cuda.empty_cache()


# ============================================
# 主程式
# ============================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='underwater image enhancement training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', required=True, help='input image folder')
    parser.add_argument('--reference', required=True, help='reference image folder')
    parser.add_argument('--output', default='./output_matlab', help='output folder')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='device')
    parser.add_argument('--no-amp', action='store_true', help='disable mixed precision training')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume training from')
    
    args = parser.parse_args()
    
    train_matlab_style(
        image_folder=args.input,
        reference_folder=args.reference,
        output_folder=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        use_amp=not args.no_amp,
        resume=args.resume
    )