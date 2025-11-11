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
from color_correction_cnn import ColorCorrectionCNN

# ============================================
# 數據集（帶預處理）
# ============================================

class MATLABStyleDataset(Dataset):
    """
    色偏校正 + 大氣光估算
    """
    
    def __init__(self, image_folder, reference_folder, target_size=224, 
                 augment=True, use_features=False):
        """
        Args:
            image_folder: 輸入圖像資料夾
            reference_folder: 參考圖像資料夾
            target_size: 目標大小
            augment: 是否數據增強
            use_features: 是否使用統計特徵
        """
        self.image_folder = Path(image_folder)
        self.reference_folder = Path(reference_folder)
        self.target_size = target_size
        self.augment = augment
        self.use_features = use_features
        
        # 色偏校正器
        self.color_corrector = ColorCorrectionCNN()
        
        # 大氣光估算器
        self.atmos_estimator = AtmosphericLightEstimator(min_size=1)
        
        # 找到所有圖像
        self.image_paths = (
            list(self.image_folder.glob('*.jpg')) + 
            list(self.image_folder.glob('*.png')) +
            list(self.image_folder.glob('*.jpeg'))
        )
        
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
        """載入並調整圖像大小"""
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_size, target_size), 
                        interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        return img
    
    def augment_pair(self, img, ref):
        """數據增強（水平/垂直翻轉）"""
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            ref = np.fliplr(ref).copy()
        
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
            ref = np.flipud(ref).copy()
        
        return img, ref
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 載入原始圖像（不調整大小，用於色偏校正和大氣光估算）
        img_original_cv = cv2.imread(str(img_path))
        if img_original_cv is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img_original_cv = cv2.cvtColor(img_original_cv, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
      
        
        # 1. 色偏校正
        img_corrected, color_type = self.color_corrector(img_original_cv)
        
        # 2. 大氣光估算
        atmospheric_light = self.atmos_estimator(img_corrected)

        # 調整校正後的圖像到目標大小
        img_corrected_resized = cv2.resize(
            (img_corrected * 255).astype(np.uint8), 
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32) / 255.0
        
        # 載入參考圖像
        ref_path = self.reference_folder / img_path.name
        if ref_path.exists():
            ref = self.load_image(ref_path, self.target_size)
        else:
            # 如果沒有參考圖，使用原圖
            ref = cv2.resize(img_original_cv, (self.target_size, self.target_size),
                           interpolation=cv2.INTER_LINEAR)
        
        # ========================================
        # 數據增強
        # ========================================
        if self.augment:
            img_corrected_resized, ref = self.augment_pair(img_corrected_resized, ref)
        
        # ========================================
        # 轉換為 Tensor
        # ========================================
        
        # 1. 用於增強的圖像（原始尺度）
        img_tensor = torch.from_numpy(img_corrected_resized).permute(2, 0, 1).float()
        
        # 2. 用於 VGG 的圖像（VGG 歸一化）
        img_vgg = self.normalize(img_tensor.clone())
        
        # 3. 參考圖像 
        ref_tensor = torch.from_numpy(ref).permute(2, 0, 1).float()
        
        # 4. 大氣光
        atmos_tensor = torch.from_numpy(atmospheric_light).float()
        
        # 5. 統計特徵
        # if self.use_features:
        #     features = extract_statistical_features(img_corrected_resized)
        #     feature_tensor = torch.from_numpy(features).float()
        # else:
        feature_tensor = torch.zeros(79).float()
        
        return {
            'image': img_tensor,              # (3, H, W) 用於增強，[0, 1]
            'image_vgg': img_vgg,             # (3, H, W) 用於 VGG，已歸一化
            'reference': ref_tensor,          # (3, H, W) 參考圖像
            'atmospheric_light': atmos_tensor,  # (3,) 大氣光
            'features': feature_tensor,       # (79,) 統計特徵
            'color_type': color_type,         # str 色偏類型
            'path': str(img_path)
        }


# ============================================
# 損失函數
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
            size_average: 是否取平均 (default: True)
            channel: 圖像通道數 (default: 3 for RGB)
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        
        # 創建高斯窗口
        self.window = self.create_window(window_size, channel)
    
    def gaussian(self, window_size, sigma=1.5):
        """創建 1D 高斯核"""
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / float(2*sigma**2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        """創建 2D 高斯窗口"""
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """
        計算 SSIM
        
        Args:
            img1, img2: 輸入圖像 (B, C, H, W)
            window: 高斯窗口
            window_size: 窗口大小
            channel: 通道數
            size_average: 是否取平均
        
        Returns:
            SSIM 值 (0-1, 越大越相似)
        """
        # 常數 (避免除零)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 確保 window 在正確的設備上
        window = window.to(img1.device)
        
        # 計算均值 μ
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # 計算方差 σ² 和協方差 σ₁₂
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        # SSIM 公式
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
            loss: 1 - SSIM (越小越好)
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
        
        # 返回 loss (1 - SSIM)
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
    """
     SSIM + L1 
    """
    def __init__(self, ssim_weight=0.8, l1_weight=0.2, device='cuda'):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        # self.perceptual_weight = perceptual_weight
        
        self.ssim_loss = SSIMLoss()
        self.l1_loss = nn.L1Loss()
        # self.perceptual_loss = PerceptualLoss(device)
    
    def forward(self, enhanced, reference):
        ssim = self.ssim_loss(enhanced, reference)
        l1 = self.l1_loss(enhanced, reference)
        # perceptual = self.perceptual_loss(enhanced, reference)
        
        total_loss = (self.ssim_weight * ssim + 
                     self.l1_weight * l1  
                     )
        
        return total_loss, {
            'ssim': ssim.item(),
            'l1': l1.item(),
            
        }

# ============================================
# 訓練器
# ============================================

class MATLABStyleTrainer:
    """ 風格增強訓練器"""
    
    def __init__(self, device='cuda', use_amp=True):
        """
        Args:
            device: 'cuda' or 'cpu'
            use_amp: 是否使用混合精度訓練
        """
        self.device = device
        self.use_amp = use_amp and (device == 'cuda')
        
        # 初始化模型
        self.param_predictor = MATLABParameterPredictor(
            pretrained=True, 
            hidden_dim=256, 
            use_features=False  
        ).to(device)
        
        self.enhancement = MATLABStyleEnhancement().to(device)
        self.criterion = FullSSIMCombinedLoss(device=device)
        
        # 優化器
        self.optimizer = optim.AdamW(
            self.param_predictor.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        # 學習率調度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # 混合精度
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # 記錄
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader, epoch):
        """訓練一個 epoch"""
        self.param_predictor.train()
        self.enhancement.eval()  # 增強模組不訓練
        
        total_loss = 0
        loss_components = {'ssim': 0, 'l1': 0}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 取得數據
            images = batch['image'].to(self.device)               # (B, 3, H, W)
            images_vgg = batch['image_vgg'].to(self.device)       # (B, 3, H, W)
            references = batch['reference'].to(self.device)       # (B, 3, H, W)
            atmos_light = batch['atmospheric_light'].to(self.device)  # (B, 3)
            features = batch['features'].to(self.device)          # (B, 79)
            
            self.optimizer.zero_grad()
            
            # 混合精度訓練
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    # 預測參數
                    params = self.param_predictor(images_vgg, features)
                    
                    # 應用增強
                    enhanced, _ = self.enhancement(images, params, atmos_light)
                    
                    # 計算損失
                    loss, components = self.criterion(enhanced, references)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.param_predictor.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 標準訓練
                params = self.param_predictor(images_vgg, features)
                enhanced, _ = self.enhancement(images, params, atmos_light)
                loss, components = self.criterion(enhanced, references)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.param_predictor.parameters(), 1.0)
                self.optimizer.step()
            
            # 更新統計
            total_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v
            
            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
        print(f"Batch images device: {images.device}")  # 必須是 cuda:0
        self.train_losses.append(avg_loss)
        return avg_loss, avg_components
    
    def validate(self, dataloader):
        """驗證"""
        self.param_predictor.eval()
        self.enhancement.eval()
        
        total_loss = 0
        loss_components = {'ssim': 0, 'l1': 0}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                images = batch['image'].to(self.device)
                images_vgg = batch['image_vgg'].to(self.device)
                references = batch['reference'].to(self.device)
                atmos_light = batch['atmospheric_light'].to(self.device)
                features = batch['features'].to(self.device)
                
                params = self.param_predictor(images_vgg, features)
                enhanced, _ = self.enhancement(images, params, atmos_light)
                loss, components = self.criterion(enhanced, references)
                
                total_loss += loss.item()
                for k, v in components.items():
                    loss_components[k] += v
        
        avg_loss = total_loss / len(dataloader)
        avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
        
        self.val_losses.append(avg_loss)
        return avg_loss, avg_components
    
    def save(self, path, epoch=None, metrics=None):
        """保存檢查點"""
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
        """載入檢查點"""
        checkpoint = torch.load(path, map_location=self.device)
        self.param_predictor.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"✓ Checkpoint loaded: {path}")


# ============================================
# 主訓練函數
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

    # 啟用 Autograd anomaly detection（調試用，當定位到問題後可移除）
    # try:
    #     # torch.autograd.set_detect_anomaly(True)
    # except Exception:
    #     # 如果跑在沒有 torch.autograd 的環境，忽略
    #     pass

    print("=" * 80)
    print("MATLAB 風格水下影像增強訓練")
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
        
        # 創建數據載入器
        # Windows 系統需要 num_workers=0 避免序列化問題
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
        
        print(f"訓練樣本: {len(train_subset)}")
        print(f"驗證樣本: {len(val_subset)}")
        
        # 初始化訓練器
        print("\n初始化訓練器...")
        trainer = MATLABStyleTrainer(device=device, use_amp=use_amp)
        
        # 恢復訓練
        start_epoch = 0
        if resume and Path(resume).exists():
            print(f"恢復訓練: {resume}")
            trainer.load(resume)
            start_epoch = len(trainer.train_losses)
        
        # 訓練循環
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        print("\n" + "=" * 80)
        print("開始訓練")
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
                #   f"Perceptual: {train_components['perceptual']:.6f}"
                )
            
            # 驗證
            val_loss, val_components = trainer.validate(val_loader)
            print(f"Val Loss: {val_loss:.6f}")
            print(f"  SSIM: {val_components['ssim']:.6f}, "
                  f"L1: {val_components['l1']:.6f}, "
                #   f"Perceptual: {val_components['perceptual']:.6f}"
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
                print(f"✓ 新的最佳模型! Val Loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{max_patience}")
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                trainer.save(f"{output_folder}/checkpoint_epoch_{epoch + 1}.pth")
            
            # 早停
            if patience_counter >= max_patience:
                print(f"\n早停於 epoch {epoch + 1}")
                break
            
            # 清理 GPU 記憶體
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        # 保存最終模型
        trainer.save(f"{output_folder}/final_model.pth")
        print("\n" + "=" * 80)
        print("訓練完成!")
        print(f"Best Val Loss: {best_val_loss:.6f}")
        print(f"模型保存於: {output_folder}")
        print("=" * 80)
    
    except KeyboardInterrupt:
        print("\n\n訓練中斷")
        print("保存當前狀態...")
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
        description='MATLAB 風格水下影像增強訓練',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', required=True, help='輸入圖像資料夾')
    parser.add_argument('--reference', required=True, help='參考圖像資料夾')
    parser.add_argument('--output', default='./output_matlab', help='輸出資料夾')
    parser.add_argument('--epochs', type=int, default=50, help='訓練輪數')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='設備')
    parser.add_argument('--no-amp', action='store_true', help='禁用混合精度訓練')
    parser.add_argument('--resume', type=str, default=None, help='恢復訓練的檢查點')
    
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