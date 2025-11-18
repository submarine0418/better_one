"""
CNN 色彩校正訓練腳本（LAB 色彩空間版本）
使用配對的原始圖像和參考圖像訓練輕量級 CNN 色彩校正模型
在 LAB 色彩空間進行訓練
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json
from skimage import color as skcolor

from color_correction_cnn import LightweightColorCorrectionNet, ColorCorrectionTrainer


class ColorCorrectionDataset(Dataset):
    """
    色彩校正數據集（LAB 色彩空間）
    載入配對的原始圖像（需要校正）和參考圖像（已校正）
    內部自動轉換為 LAB 色彩空間
    """
    
    def __init__(self, input_dir, reference_dir, img_size=256, augment=True):
        """
        Args:
            input_dir: 原始圖像目錄（未校正）
            reference_dir: 參考圖像目錄（已校正）
            img_size: 訓練圖像大小
            augment: 是否進行數據增強
        """
        self.input_dir = Path(input_dir)
        self.reference_dir = Path(reference_dir)
        self.img_size = img_size
        self.augment = augment
        
        # 收集圖像路徑
        self.image_files = self._collect_image_pairs()
        
        print(f"找到 {len(self.image_files)} 組配對圖像")
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
        
    def _collect_image_pairs(self):
        """收集配對的圖像"""
        image_pairs = []
        
        # 支持的圖像格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 遍歷輸入目錄
        for input_path in self.input_dir.rglob('*'):
            if input_path.suffix.lower() in valid_extensions:
                # 構建對應的參考圖像路徑
                relative_path = input_path.relative_to(self.input_dir)
                reference_path = self.reference_dir / relative_path
                
                # 檢查參考圖像是否存在
                if reference_path.exists():
                    image_pairs.append((str(input_path), str(reference_path)))
                else:
                    print(f"⚠ 警告: 找不到配對的參考圖像: {reference_path}")
        
        return image_pairs
    
    def __len__(self):
        return len(self.image_files)
    
    def _rgb_to_lab_normalized(self, rgb_img):
      
        # RGB → LAB (scikit-image)
        lab = skcolor.rgb2lab(rgb_img)  # L: [0, 100], a/b: [-128, 127]
        
        # 正規化到 [0, 1]
        lab_normalized = np.zeros_like(lab, dtype=np.float32)
        lab_normalized[:, :, 0] = lab[:, :, 0] / 100.0          # L: [0, 100] → [0, 1]
        lab_normalized[:, :, 1] = (lab[:, :, 1] + 128.0) / 255.0  # a: [-128, 127] → [0, 1]
        lab_normalized[:, :, 2] = (lab[:, :, 2] + 128.0) / 255.0  # b: [-128, 127] → [0, 1]
        
        return lab_normalized
    
    def __getitem__(self, idx):
        input_path, reference_path = self.image_files[idx]
        
        # 載入圖像
        input_img = cv2.imread(input_path)
        reference_img = cv2.imread(reference_path)
        
        # BGR to RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        
        # 調整大小
        input_img = cv2.resize(input_img, (self.img_size, self.img_size))
        reference_img = cv2.resize(reference_img, (self.img_size, self.img_size))
        
        # 歸一化到 [0, 1]
        input_img = input_img.astype(np.float32) / 255.0
        reference_img = reference_img.astype(np.float32) / 255.0
        
        # 數據增強（在 RGB 空間）
        if self.augment:
            input_img, reference_img = self._augment(input_img, reference_img)
        
        # RGB → LAB (正規化到 [0, 1])
        input_lab = self._rgb_to_lab_normalized(input_img)
        reference_lab = self._rgb_to_lab_normalized(reference_img)
        
        # 轉換為 tensor (H, W, C) -> (C, H, W)
        input_tensor = torch.from_numpy(input_lab).permute(2, 0, 1)
        reference_tensor = torch.from_numpy(reference_lab).permute(2, 0, 1)
        
        return input_tensor, reference_tensor
    
    def _augment(self, input_img, reference_img):
        """數據增強（保持配對一致性）"""
        # 隨機水平翻轉
        if np.random.rand() > 0.5:
            input_img = np.fliplr(input_img).copy()
            reference_img = np.fliplr(reference_img).copy()
        
        # 隨機垂直翻轉
        if np.random.rand() > 0.5:
            input_img = np.flipud(input_img).copy()
            reference_img = np.flipud(reference_img).copy()
        
        # 隨機旋轉 90 度
        if np.random.rand() > 0.5:
            k = np.random.choice([1, 2, 3])
            input_img = np.rot90(input_img, k).copy()
            reference_img = np.rot90(reference_img, k).copy()
        
        return input_img, reference_img


def train_color_correction_model(args):
   
    
    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用設備: {device}")
    
    # 創建輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # 載入數據集
    # ============================================
    print("\n載入數據集...")
    
    # 訓練集
    train_dataset = ColorCorrectionDataset(
        input_dir=args.input,
        reference_dir=args.reference,
        img_size=args.img_size,
        augment=True
    )
    
    # 驗證集（使用相同數據但不增強）
    val_dataset = ColorCorrectionDataset(
        input_dir=args.input,
        reference_dir=args.reference,
        img_size=args.img_size,
        augment=False
    )
    
    # 劃分訓練/驗證集
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    val_dataset, _ = torch.utils.data.random_split(
        val_dataset, [val_size, len(val_dataset) - val_size]
    )
    
    print(f"訓練樣本: {len(train_dataset)}")
    print(f"驗證樣本: {len(val_dataset)}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # ============================================
    # 創建模型和訓練器
    # ============================================
    print("\n初始化模型...")
    
    model = LightweightColorCorrectionNet(base_channels=args.base_channels)
    trainer = ColorCorrectionTrainer(model, device=device, lr=args.lr)
    
    # 統計參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數量: {total_params:,}")
    print(f"可訓練參數: {trainable_params:,}")
    
    trainer.window_size = 11
    trainer.window = create_window(trainer.window_size, 3).to(device)
    
    # ============================================
    # 訓練循環
    # ============================================
    print(f"\n開始訓練 {args.epochs} 個 epochs...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # 訓練階段
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (input_labs, target_labs) in enumerate(pbar):
            # 注意：此時輸入已經是 LAB 色彩空間 [0, 1]
            input_labs = input_labs.to(device)
            target_labs = target_labs.to(device)
            
            # 訓練步驟（直接在 LAB 空間訓練）
            trainer.model.train()
            trainer.optimizer.zero_grad()
            
            # 前向傳播
            output_labs = trainer.model(input_labs)
            
            # 損失函數（LAB 空間）
            # L1 損失
            l1_loss = trainer.l1_loss(output_labs, target_labs)
            ssim_loss = 1 - trainer.ssim(output_labs, target_labs, window=trainer.window, window_size=trainer.window_size, channel=3)
            # LAB 空間的色彩一致性損失
            color_loss = trainer._lab_color_consistency_loss(output_labs, target_labs)
            
            # 總損失
            loss = 0.3 * l1_loss + 0.0 * color_loss+ 0.7 * ssim_loss
            
            # 反向傳播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
            trainer.optimizer.step()
            
            train_loss += loss.item()
            
            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{train_loss/(batch_idx+1):.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for input_labs, target_labs in tqdm(val_loader, desc='Validation'):
                input_labs = input_labs.to(device)
                target_labs = target_labs.to(device)
                
                output = model(input_labs)
                
                # 驗證損失（使用相同的損失組合）
                l1_loss = F.l1_loss(output, target_labs)
                color_loss = trainer._lab_color_consistency_loss(output, target_labs)
                ssim_loss = 1 - trainer.ssim(output, target_labs, window=trainer.window, window_size=trainer.window_size, channel=3)
                loss = 0.2 * l1_loss + 0.8 * color_loss+ 0.4 * ssim_loss
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 學習率調度
        trainer.scheduler.step(avg_val_loss)
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        # 記錄歷史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)
        
        # 打印統計
        print(f"\nTrain Loss: {avg_train_loss:.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = output_dir / 'best_color_correction_model.pth'
            trainer.save_checkpoint(best_model_path, epoch, avg_val_loss)
            print(f"✓ 新的最佳模型! Val Loss: {best_val_loss:.6f}")
        
        # 定期保存檢查點
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            trainer.save_checkpoint(checkpoint_path, epoch, avg_val_loss)
    
    # ============================================
    # 保存最終模型和訓練歷史
    # ============================================
    final_model_path = output_dir / 'final_color_correction_model.pth'
    trainer.save_checkpoint(final_model_path, args.epochs, avg_val_loss)
    
    # 保存訓練歷史
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ 訓練歷史已保存: {history_path}")
    
    print("\n" + "=" * 80)
    print("✓ 訓練完成!")
    print(f"最佳驗證損失: {best_val_loss:.6f}")
    print(f"模型已保存至: {output_dir}")
    print("=" * 80)


def create_window(window_size, channel):
    """Create Gaussian window for SSIM"""
    def gaussian(window_size, sigma=1.5):
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    _1D_window = gaussian(window_size).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def main():
    parser = argparse.ArgumentParser(description='訓練 CNN 色彩校正模型（LAB 色彩空間）')
    
    # 數據相關
    parser.add_argument('--input', type=str, required=True,
                        help='原始圖像目錄（未校正）')
    parser.add_argument('--reference', type=str, required=True,
                        help='參考圖像目錄（已校正/目標）')
    parser.add_argument('--output', type=str, default='./color_correction_output',
                        help='輸出目錄')
    
    # 訓練超參數
    parser.add_argument('--epochs', type=int, default=100,
                        help='訓練輪數')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='學習率')
    parser.add_argument('--img-size', type=int, default=256,
                        help='訓練圖像大小')
    
    # 模型相關
    parser.add_argument('--base-channels', type=int, default=16,
                        help='網路基礎通道數（16 或 32）')
    
    # 其他
    parser.add_argument('--num-workers', type=int, default=4,
                        help='數據載入線程數')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='保存檢查點的間隔（epochs）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='訓練設備（cuda 或 cpu）')
    
    args = parser.parse_args()
    
    # 開始訓練
    train_color_correction_model(args)


if __name__ == '__main__':
    main()