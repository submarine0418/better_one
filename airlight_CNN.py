import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2


class LightweightAtmosphericLightNet(nn.Module):
    """
    輕量級 CNN 大氣光估計網路
    使用輕量級卷積架構提取全局特徵，預測 RGB 大氣光值
    """
    
    def __init__(self, base_channels=16):
        super().__init__()
        
        # Backbone: 輕量級特徵提取器
        self.backbone = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # 第二層: 下採樣 /2
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, 
                     stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # 第三層: 下採樣 /4
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, 
                     stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Prediction Head: 預測 RGB 大氣光值
        feat_dim = base_channels * 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 3),
            nn.Sigmoid()  # 輸出範圍 [0, 1]
        )
    
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: (B, 3, H, W) 輸入圖像 [0, 1]
        
        Returns:
            atmospheric_light: (B, 3) 大氣光值 [0, 1]
        """
        x = self.backbone(x)
        out = self.head(x)
        return out


class AtmosphericLightCNN:
    
    def __init__(self, model_path=None, device='cuda', base_channels=16):
      
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 初始化網路
        self.net = LightweightAtmosphericLightNet(base_channels=base_channels).to(self.device)
        
        # 載入預訓練模型
        if model_path is not None and Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.net.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.net.load_state_dict(checkpoint['state_dict'])
                else:
                    self.net.load_state_dict(checkpoint)
                print(f"✓ 成功載入預訓練模型: {model_path}")
            except Exception as e:
                print(f"⚠ 載入模型失敗，使用隨機初始化: {e}")
        else:
            print("⚠ 未指定模型路徑或文件不存在，使用隨機初始化")
        
        self.net.eval()  # 推理模式
    
    def __call__(self, img):
        """
        估計大氣光（接口與傳統方法完全一致）
        
        Args:
            img: numpy array (H, W, 3), RGB, 支援 uint8 [0, 255] 或 float32 [0, 1]
        
        Returns:
            atmospheric_light: numpy array (3,), RGB, float32 [0, 1]
        """
        # 確保輸入是 float [0, 1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
            img = np.clip(img, 0.0, 1.0)
        
        # 檢查輸入形狀
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"輸入圖像必須是 (H, W, 3)，但得到 {img.shape}")
        
        # 轉換為 PyTorch tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # CNN 大氣光估計
        with torch.no_grad():
            atmos_tensor = self.net(img_tensor)
        
        # 轉換回 numpy
        atmospheric_light = atmos_tensor.squeeze(0).cpu().numpy()
        atmospheric_light = np.clip(atmospheric_light, 0.0, 1.0).astype(np.float32)
        
        return atmospheric_light
    
    def estimate_batch(self, imgs):
        """
        批次估計大氣光
        
        Args:
            imgs: numpy array (B, H, W, 3) 或 list of images
        
        Returns:
            atmospheric_lights: numpy array (B, 3), float32 [0, 1]
        """
        # 標準化輸入
        if isinstance(imgs, np.ndarray) and imgs.ndim == 4:
            arr = imgs
        else:
            arr = np.stack([np.asarray(im) for im in imgs], axis=0)
        
        # 歸一化到 [0, 1]
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)
        
        # 轉置: (B, H, W, 3) → (B, 3, H, W)
        arr = arr.transpose(0, 3, 1, 2)
        batch_tensor = torch.from_numpy(arr).float().to(self.device)
        
        # 批次推理
        with torch.no_grad():
            preds = self.net(batch_tensor).cpu().numpy()
        
        preds = np.clip(preds.astype(np.float32), 0.0, 1.0)
        return preds
    
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

class AtmosphericLightTrainer:
    """
    CNN 大氣光估計訓練器
    """
    
    def __init__(self, model, device='cuda', lr=1e-3):
        """
        初始化訓練器
        
        Args:
            model: LightweightAtmosphericLightNet 實例
            device: 'cuda' 或 'cpu'
            lr: 學習率
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 優化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        # 學習率調度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 損失函數
        self.criterion = nn.MSELoss()
    
    def train_step(self, input_img, target_atmos):
        """
        單步訓練
        
        Args:
            input_img: (B, 3, H, W) 輸入圖像 [0, 1]
            target_atmos: (B, 3) 目標大氣光值 [0, 1]
        
        Returns:
            loss: float
        """
        self.model.train()
        
        input_img = input_img.to(self.device)
        target_atmos = target_atmos.to(self.device)
        
        # 前向傳播
        output = self.model(input_img)
        
        # 損失函數：MSE Loss
        loss = self.criterion(output, target_atmos)
        
        # 反向傳播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
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
# 工廠函數（向後兼容）
# ============================================

def make_atmospheric_light_cnn(model_path=None, device='cpu', base_channels=16):
    """
    創建大氣光估計器實例（工廠函數，向後兼容）
    
    Args:
        model_path: 預訓練模型路徑 (可選)
        device: 'cuda' or 'cpu'
        base_channels: 基礎通道數
    
    Returns:
        AtmosphericLightCNN 實例
    """
    estimator = AtmosphericLightCNN(
        model_path=model_path,
        device=device,
        base_channels=base_channels
    )
    return estimator


# ============================================
# 使用範例
# ============================================

if __name__ == '__main__':
    print("=" * 80)
    print("輕量級 CNN 大氣光估計模組")
    print("=" * 80)
    
    # ============================================
    # 推理使用範例（與傳統方法接口一致）
    # ============================================
    print("\n【推理使用範例】")
    print("-" * 80)
    
    # 方式 1: 直接初始化（如果有預訓練模型）
    estimator = AtmosphericLightCNN(
        model_path=None,  # 可指定預訓練模型路徑
        device='cuda',
        base_channels=16
    )
    
    # 方式 2: 使用工廠函數（向後兼容）
    # estimator = make_atmospheric_light_cnn(
    #     model_path=None,
    #     device='cuda',
    #     base_channels=16
    # )
    
    # ============================================
    # 測試 1: 單張圖像 (uint8)
    # ============================================
    print("\n測試 1: 單張圖像 (uint8)")
    img_uint8 = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    
    atmospheric_light = estimator(img_uint8)
    
    print(f"輸入圖像形狀: {img_uint8.shape}, dtype: {img_uint8.dtype}")
    print(f"輸出大氣光形狀: {atmospheric_light.shape}, dtype: {atmospheric_light.dtype}")
    print(f"大氣光值: R={atmospheric_light[0]:.4f}, G={atmospheric_light[1]:.4f}, B={atmospheric_light[2]:.4f}")
    
    # ============================================
    # 測試 2: 單張圖像 (float32)
    # ============================================
    print("\n測試 2: 單張圖像 (float32)")
    img_float = np.random.rand(480, 640, 3).astype(np.float32)
    
    atmospheric_light = estimator(img_float)
    
    print(f"輸入圖像形狀: {img_float.shape}, dtype: {img_float.dtype}")
    print(f"輸出大氣光形狀: {atmospheric_light.shape}, dtype: {atmospheric_light.dtype}")
    print(f"大氣光值: R={atmospheric_light[0]:.4f}, G={atmospheric_light[1]:.4f}, B={atmospheric_light[2]:.4f}")
    
    # ============================================
    # 測試 3: 批次處理
    # ============================================
    print("\n測試 3: 批次處理")
    batch_imgs = np.random.rand(4, 256, 256, 3).astype(np.float32)
    
    batch_atmos = estimator.estimate_batch(batch_imgs)
    
    print(f"輸入批次形狀: {batch_imgs.shape}")
    print(f"輸出批次形狀: {batch_atmos.shape}")
    print("批次大氣光值:")
    for i, atmos in enumerate(batch_atmos):
        print(f"  [{i}] R={atmos[0]:.4f}, G={atmos[1]:.4f}, B={atmos[2]:.4f}")
    
    # ============================================
    # 測試 4: 實際圖像範例（如果有圖像文件）
    # ============================================
    print("\n測試 4: 實際圖像範例")
    try:
        # 嘗試讀取測試圖像
        test_img_path = "test_underwater.jpg"  # 替換為實際路徑
        if Path(test_img_path).exists():
            img = cv2.imread(test_img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 使用 CNN 估計大氣光
            atmos = estimator(img_rgb)
            
            print(f"圖像路徑: {test_img_path}")
            print(f"圖像形狀: {img_rgb.shape}")
            print(f"估計的大氣光: R={atmos[0]:.4f}, G={atmos[1]:.4f}, B={atmos[2]:.4f}")
        else:
            print(f"測試圖像不存在: {test_img_path}")
    except Exception as e:
        print(f"測試圖像讀取失敗: {e}")
    
    print("\n" + "=" * 80)
    print("✓ 所有測試完成!")
    print("=" * 80)
    
    # ============================================
    # 訓練使用範例（說明）
    # ============================================
    print("\n【訓練使用範例】")
    print("-" * 80)
    print("""
# 1. 創建訓練器
net = LightweightAtmosphericLightNet(base_channels=16)
trainer = AtmosphericLightTrainer(net, device='cuda', lr=1e-3)

# 2. 準備數據（假設）
# input_batch: (B, 3, H, W) 輸入圖像
# target_batch: (B, 3) 目標大氣光值

# 3. 訓練循環
for epoch in range(num_epochs):
    for input_batch, target_batch in dataloader:
        loss = trainer.train_step(input_batch, target_batch)
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    # 4. 保存檢查點
    trainer.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, loss)

# 5. 使用訓練好的模型
estimator = AtmosphericLightCNN(
    model_path='checkpoint_epoch_50.pth',
    device='cuda'
)
atmospheric_light = estimator(img)
    """)
    print("-" * 80)