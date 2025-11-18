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

from airlight_CNN import LightweightAtmosphericLightNet, AtmosphericLightTrainer


class AtmosphericLightDataset(Dataset):
    """
    大氣光估計數據集
    載入圖像和對應的大氣光標籤
    """
    
    def __init__(self, image_dir, label_file, img_size=224, augment=True):
        """
        Args:
            image_dir: 圖像目錄
            label_file: 標籤文件（格式: image_name R G B）
            img_size: 訓練圖像大小
            augment: 是否進行數據增強
        """
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.augment = augment
        
        # 讀取標籤
        self.samples = self._load_labels(label_file)
        
        print(f"找到 {len(self.samples)} 個樣本")
    
    def _load_labels(self, label_file):
        """
        讀取標籤文件
        
        格式範例:
        image001.jpg 0.8 0.85 0.9
        image002.jpg 0.75 0.8 0.88
        """
        samples = []
        
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) != 4:
                    print(f"⚠ 警告: 第 {line_num} 行格式錯誤，跳過: {line}")
                    continue
                
                img_name, r, g, b = parts
                img_path = self.image_dir / img_name
                
                if not img_path.exists():
                    print(f"⚠ 警告: 圖像不存在，跳過: {img_path}")
                    continue
                
                try:
                    atmos_light = [float(r), float(g), float(b)]
                    samples.append({
                        'path': str(img_path),
                        'atmospheric_light': atmos_light
                    })
                except ValueError:
                    print(f"⚠ 警告: 第 {line_num} 行數值錯誤，跳過: {line}")
                    continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 載入圖像
        img = cv2.imread(sample['path'])
        if img is None:
            raise ValueError(f"無法讀取圖像: {sample['path']}")
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 調整大小
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 歸一化到 [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # 數據增強
        if self.augment:
            img = self._augment(img)
        
        # 轉換為 tensor (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        atmos_tensor = torch.tensor(sample['atmospheric_light'], dtype=torch.float32)
        
        return img_tensor, atmos_tensor
    
    def _augment(self, img):
        """數據增強"""
        # 隨機水平翻轉
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
        
        # 隨機垂直翻轉
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
        
        # 隨機旋轉 90 度
        if np.random.rand() > 0.5:
            k = np.random.choice([1, 2, 3])
            img = np.rot90(img, k).copy()
        
        return img


def train_atmospheric_light_model(args):
    """
    訓練大氣光估計模型
    """
    print("=" * 80)
    print("開始訓練 CNN 大氣光估計模型")
    print("=" * 80)
    
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
    train_dataset = AtmosphericLightDataset(
        image_dir=args.images,
        label_file=args.labels,
        img_size=args.img_size,
        augment=True
    )
    
    # 驗證集（使用相同數據但不增強）
    val_dataset = AtmosphericLightDataset(
        image_dir=args.images,
        label_file=args.labels,
        img_size=args.img_size,
        augment=False
    )
    
    # 劃分訓練/驗證集
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset, _ = torch.utils.data.random_split(
        val_dataset, [val_size, len(val_dataset) - val_size],
        generator=torch.Generator().manual_seed(42)
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
    
    model = LightweightAtmosphericLightNet(base_channels=args.base_channels)
    trainer = AtmosphericLightTrainer(model, device=device, lr=args.lr)
    
    # 統計參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數量: {total_params:,}")
    print(f"可訓練參數: {trainable_params:,}")
    
    # ============================================
    # 訓練循環
    # ============================================
    print(f"\n開始訓練 {args.epochs} 個 epochs...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = args.patience
    
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
        for batch_idx, (input_imgs, target_atmos) in enumerate(pbar):
            loss = trainer.train_step(input_imgs, target_atmos)
            train_loss += loss
            
            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss:.6f}',
                'avg_loss': f'{train_loss/(batch_idx+1):.6f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for input_imgs, target_atmos in tqdm(val_loader, desc='Validation'):
                input_imgs = input_imgs.to(device)
                target_atmos = target_atmos.to(device)
                
                output = model(input_imgs)
                loss = trainer.criterion(output, target_atmos)
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
            patience_counter = 0
            best_model_path = output_dir / 'best_atmospheric_light_model.pth'
            trainer.save_checkpoint(best_model_path, epoch, avg_val_loss)
            print(f"✓ 新的最佳模型! Val Loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{max_patience}")
        
        # 早停
        if patience_counter >= max_patience:
            print(f"\n早停於 epoch {epoch}")
            break
        
        # 定期保存檢查點
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            trainer.save_checkpoint(checkpoint_path, epoch, avg_val_loss)
    
    # ============================================
    # 保存最終模型和訓練歷史
    # ============================================
    final_model_path = output_dir / 'final_atmospheric_light_model.pth'
    trainer.save_checkpoint(final_model_path, epoch, avg_val_loss)
    
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


def generate_sample_labels(args):
    """
    生成範例標籤文件（使用傳統方法估計大氣光）
    """
    print("=" * 80)
    print("生成範例標籤文件")
    print("=" * 80)
    
    from matlab_style_enhancement import AtmosphericLightEstimator
    
    image_dir = Path(args.images)
    output_file = Path(args.output) / 'atmospheric_light_labels.txt'
    
    # 收集圖像
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    for ext in valid_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))
    
    print(f"\n找到 {len(image_files)} 張圖像")
    
    if len(image_files) == 0:
        print("⚠ 錯誤: 沒有找到圖像文件")
        return
    
    # 使用傳統方法估計大氣光
    estimator = AtmosphericLightEstimator(min_size=1)
    
    labels = []
    for img_path in tqdm(image_files, desc='估計大氣光'):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠ 跳過無法讀取的圖像: {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        atmos = estimator(img_rgb)
        
        labels.append(f"{img_path.name} {atmos[0]:.6f} {atmos[1]:.6f} {atmos[2]:.6f}\n")
    
    # 保存標籤
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(labels)
    
    print(f"\n✓ 標籤文件已保存: {output_file}")
    print(f"總共 {len(labels)} 個標籤")
    print("\n範例:")
    for label in labels[:5]:
        print(f"  {label.strip()}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='訓練 CNN 大氣光估計模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:

1. 訓練模型（已有標籤文件）:
   python train_atmospheric_light.py train \\
       --images ./data/underwater_images \\
       --labels ./data/atmospheric_light_labels.txt \\
       --output ./output_atmos \\
       --epochs 50 --batch-size 16

2. 生成範例標籤文件（使用傳統方法）:
   python train_atmospheric_light.py generate \\
       --images ./data/underwater_images \\
       --output ./data

標籤文件格式:
   image001.jpg 0.8 0.85 0.9
   image002.jpg 0.75 0.8 0.88
   ...
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # ============================================
    # 訓練命令
    # ============================================
    train_parser = subparsers.add_parser('train', help='訓練模型')
    
    # 數據相關
    train_parser.add_argument('--images', type=str, required=True,
                             help='圖像目錄')
    train_parser.add_argument('--labels', type=str, required=True,
                             help='標籤文件（格式: image_name R G B）')
    train_parser.add_argument('--output', type=str, default='./atmospheric_light_output',
                             help='輸出目錄')
    
    # 訓練超參數
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='訓練輪數')
    train_parser.add_argument('--batch-size', type=int, default=16,
                             help='批次大小')
    train_parser.add_argument('--lr', type=float, default=1e-3,
                             help='學習率')
    train_parser.add_argument('--img-size', type=int, default=224,
                             help='訓練圖像大小')
    train_parser.add_argument('--patience', type=int, default=10,
                             help='早停耐心值')
    
    # 模型相關
    train_parser.add_argument('--base-channels', type=int, default=16,
                             help='網路基礎通道數（16 或 32）')
    
    # 其他
    train_parser.add_argument('--num-workers', type=int, default=4,
                             help='數據載入線程數')
    train_parser.add_argument('--save-interval', type=int, default=10,
                             help='保存檢查點的間隔（epochs）')
    
    # ============================================
    # 生成標籤命令
    # ============================================
    generate_parser = subparsers.add_parser('generate', help='生成範例標籤文件')
    generate_parser.add_argument('--images', type=str, required=True,
                                help='圖像目錄')
    generate_parser.add_argument('--output', type=str, default='.',
                                help='輸出目錄')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_atmospheric_light_model(args)
    elif args.command == 'generate':
        generate_sample_labels(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()