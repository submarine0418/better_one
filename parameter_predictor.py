"""
參數預測網路 (Parameter Predictor Network)
使用 VGG-16 預測 MATLAB 風格增強所需的參數
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights


class MATLABParameterPredictor(nn.Module):
    """
    預測 MATLAB 風格增強的參數
    
    預測參數:
        - omega: 去霧強度 [0.3, 0.9]
        - guided_radius: 引導濾波半徑 [5, 30]
        - L_low: 色彩拉伸下界 [2, 15]
        - L_high: 色彩拉伸上界 [85, 98]
        -guided_eps: 引導濾波 epsilon (固定值，不預測)
    """
    
    def __init__(self, pretrained=True, hidden_dim=256, use_features=True):
        """
        Args:
            pretrained: 是否使用預訓練權重
            hidden_dim: 隱藏層維度
            use_features: 是否使用統計特徵
        """
        super().__init__()
        
        self.use_features = use_features
        
        # ========================================
        # VGG-16 特徵提取器
        # ========================================
        if pretrained:
            weights = VGG16_Weights.IMAGENET1K_V1
        else:
            weights = None
        vgg16 = models.vgg16(weights=weights)
        
        # 使用到 conv4_3
        self.vgg_features = vgg16.features[:23]
        
        # 凍結早期層
        for i, param in enumerate(self.vgg_features.parameters()):
            if i < 16:  # 凍結前幾個卷積塊
                param.requires_grad = False
        
        # ========================================
        # 全局池化
        # ========================================
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # ========================================
        # 特徵融合
        # ========================================
        vgg_out_dim = 512
        feature_dim = 79 if use_features else 0
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(vgg_out_dim * 2 + feature_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # ========================================
        # 注意力機制
        # ========================================
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        
        # ========================================
        # 參數預測頭
        # ========================================
        self.param_heads = nn.ModuleDict({
            'omega': self._make_param_head(hidden_dim, 1),
            'guided_radius': self._make_param_head(hidden_dim, 1),
            'L_low': self._make_param_head(hidden_dim, 1),
            'L_high': self._make_param_head(hidden_dim, 1),
            'eps': self._make_param_head(hidden_dim, 1),  
        })
        
        # ========================================
        # 參數範圍定義
        # ========================================
        self.param_ranges = {
            'omega': (0.25, 0.99),           # 去霧強度
            'guided_radius': (5.0, 15.0),  # 引導濾波半徑
            'L_low': (0.0, 15.0),          # 色彩拉伸下界
            'L_high': (95.0,100.0),        # 色彩拉伸上界
            'eps': (1e-12, 1),           # 引導濾波 epsilon
        }
    
    def _make_param_head(self, in_dim, out_dim):
        """
        創建參數預測頭
        
        Args:
            in_dim: 輸入維度
            out_dim: 輸出維度
        
        Returns:
            head: nn.Sequential
        """
        return nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_dim // 2, out_dim)
        )
    
    def forward(self, img_tensor, feature_tensor=None):
        """
        前向傳播
        
        Args:
            img_tensor: (B, 3, H, W) 已色偏校正的圖像
            feature_tensor: (B, 79) 統計特徵（可選）
        
        Returns:
            params: dict, 包含預測的參數
                - omega: (B, 1)
                - guided_radius: (B, 1)
                - L_low: (B, 1)
                - L_high: (B, 1)
                - eps: (B, 1)
        """
        B = img_tensor.size(0)
        
        # ========================================
        # 1. 提取 VGG 特徵
        # ========================================
        vgg_feat = self.vgg_features(img_tensor)  # (B, 512, H', W')
        
        # 雙池化獲取更豐富的特徵
        avg_feat = self.avgpool(vgg_feat).view(B, -1)  # (B, 512)
        max_feat = self.maxpool(vgg_feat).view(B, -1)  # (B, 512)
        pooled_feat = torch.cat([avg_feat, max_feat], dim=1)  # (B, 1024)
        
        # ========================================
        # 2. 融合統計特徵（如果有）
        # ========================================
        if self.use_features and feature_tensor is not None:
            if isinstance(feature_tensor, list):
                feature_tensor = torch.stack(feature_tensor)
            feature_tensor = feature_tensor.float().to(img_tensor.device)
            combined = torch.cat([pooled_feat, feature_tensor], dim=1)  # (B, 1024+79)
        else:
            combined = pooled_feat  # (B, 1024)
        
        # ========================================
        # 3. 特徵融合
        # ========================================
        fused = self.feature_fusion(combined)  # (B, hidden_dim)
        
        # ========================================
        # 4. 應用注意力
        # ========================================
        attention_weights = self.attention(fused)
        fused = fused * attention_weights
        
        # ========================================
        # 5. 預測參數（帶範圍限制）
        # ========================================
        params = {}
        for name, head in self.param_heads.items():
            raw_output = head(fused)  # (B, 1)
            
            # 使用 sigmoid 映射到 [0, 1]，然後縮放到目標範圍
            min_val, max_val = self.param_ranges[name]
            params[name] = torch.sigmoid(raw_output) * (max_val - min_val) + min_val
        
        return params


# ============================================
# 統計特徵提取器（與訓練腳本保持一致）
# ============================================

# def extract_statistical_features(img):
#     """
#     提取統計特徵
    
#     Args:
#         img: numpy array (H, W, 3), RGB, [0, 1] float
    
#     Returns:
#         features: numpy array (79,)
#     """
#     import numpy as np
    
#     # 確保是 float [0, 1]
#     if img.dtype == np.uint8:
#         img = img.astype(np.float32) / 255.0
#     else:
#         img = img.astype(np.float32)
    
#     features = []
    
#     # 每個通道的統計特徵
#     for c in range(3):
#         channel = img[:, :, c]
#         features.extend([
#             float(np.mean(channel)),
#             float(np.std(channel)),
#             float(np.min(channel)),
#             float(np.max(channel)),
#             float(np.median(channel))
#         ])
    
#     # 全圖統計特徵
#     features.extend([
#         float(np.mean(img)),
#         float(np.std(img)),
#         float(np.mean(img ** 2)),  # 二階矩
#     ])
    
#     # Pad 到 79 維
#     while len(features) < 79:
#         features.append(0.0)
    
#     return np.array(features[:79], dtype=np.float32)


# ============================================
# Wrapper 類（與舊代碼兼容）
# ============================================

# class _VGGFeaturesWrapper:
#     """與舊代碼的兼容接口"""
#     @staticmethod
#     def extract_all_features(img):
#         return extract_statistical_features(img)


# 導出
# vgg_features = _VGGFeaturesWrapper()


