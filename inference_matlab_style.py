"""
MATLAB 風格水下影像增強推理腳本
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import torchvision.transforms as T

# 導入模組
from color_correction import ColorCorrection
from matlab_style_enhancement import AtmosphericLightEstimator, MATLABStyleEnhancement
from parameter_predictor import MATLABParameterPredictor, extract_statistical_features
from color_correction_cnn import ColorCorrectionCNN

class MATLABStylePredictor:
    """MATLAB 風格增強預測器"""
    
    def __init__(self, model_path, device='cuda', input_size=224):
        """
        Args:
            model_path: 模型檔案路徑
            device: 'cuda' or 'cpu'
            input_size: VGG 輸入大小
        """
        self.device = device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        self.input_size = input_size
        
        print(f"載入模型: {model_path}  (device={self.device})")
        
        # 載入參數預測器
        self.param_predictor = MATLABParameterPredictor(
            pretrained=False, 
            hidden_dim=256, 
            use_features=False
        )
        
        ckpt = torch.load(model_path, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)
        self.param_predictor.load_state_dict(state)
        self.param_predictor.to(self.device).eval()
        
        # 載入增強模組
        self.enhancement = MATLABStyleEnhancement().to(self.device).eval()
        
        # 預處理模組
        # self.color_corrector = ColorCorrection()
        self.color_corrector = ColorCorrectionCNN(model_path='D:\research\better_one\color_correction_output\best_color_correction_model.pth', device='cuda')
        self.atmos_estimator = AtmosphericLightEstimator(min_size=1)
        
        # VGG 歸一化
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        print("✓ 模型與增強模組已載入")
    
    def preprocess(self, img):
        """
        預處理：色偏校正 + 大氣光估算
        
        Args:
            img: (H, W, 3) RGB [0, 1]
        
        Returns:
            img_corrected: (H, W, 3) RGB [0, 1]
            atmospheric_light: (3,) numpy array
            color_type: str
        """
        # 色偏校正
        img_corrected, color_type = self.color_corrector(img)
        
        # 大氣光估算
        atmospheric_light = self.atmos_estimator(img_corrected)
        
        return img_corrected, atmospheric_light, color_type
    
    def predict_parameters(self, img, features):
        """
        預測參數
        
        Args:
            img: (H, W, 3) RGB [0, 1]
            features: (79,) numpy array
        
        Returns:
            params: dict
        """
        # 調整大小到 VGG 輸入
        img_resized = cv2.resize(
            (img * 255).astype(np.uint8), 
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float32) / 255.0
        
        # 轉換為 Tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        img_vgg = self.normalize(img_tensor).unsqueeze(0).to(self.device)
        
        feat_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        # 預測
        with torch.no_grad():
            params_tensor = self.param_predictor(img_vgg, feat_tensor)
        
        # 轉換為字典
        params = {}
        for k, v in params_tensor.items():
            params[k] = v.cpu()
        
        return params
    
    def enhance_image(self, img, atmospheric_light, params):
        """
        增強圖像
        
        Args:
            img: (H, W, 3) RGB [0, 1]
            atmospheric_light: (3,) numpy array
            params: dict of tensors
        
        Returns:
            enhanced: (H, W, 3) RGB [0, 1]
        """
        # 轉換為 Tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        atmos_tensor = torch.from_numpy(atmospheric_light).unsqueeze(0).to(self.device)
        
        # 將參數移到正確的設備
        params_device = {k: v.to(self.device) for k, v in params.items()}
        
        # 增強
        with torch.no_grad():
            enhanced_tensor, _ = self.enhancement(
                img_tensor, 
                params_device, 
                atmos_tensor
            )
        
        # 轉換回 numpy
        enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced = np.clip(enhanced, 0.0, 1.0)
        
        return enhanced
    
    def process_single_image(self, input_path, output_path=None, show_params=True):
        """
        處理單張圖像
        
        Args:
            input_path: 輸入圖像路徑
            output_path: 輸出圖像路徑
            show_params: 是否顯示參數
        
        Returns:
            enhanced: (H, W, 3) RGB [0, 1]
            params: dict
        """
        input_path = Path(input_path)
        
        # 讀取圖像
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"無法讀取影像: {input_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 預處理
        print("步驟 1/4: 色偏校正與大氣光估算...")
        img_corrected, atmospheric_light, color_type = self.preprocess(img)
        print(f"  檢測到的色偏類型: {color_type}")
        print(f"  大氣光值: R={atmospheric_light[0]:.4f}, "
              f"G={atmospheric_light[1]:.4f}, B={atmospheric_light[2]:.4f}")
        
        # 提取特徵
        print("步驟 2/4: 提取統計特徵...")
        features = extract_statistical_features(img_corrected)
        
        # 預測參數
        print("步驟 3/4: 預測增強參數...")
        params = self.predict_parameters(img_corrected, features)
        
        if show_params:
            print("  預測的參數:")
            for k, v in params.items():
                print(f"    {k}: {v.item():.4f}")
        
        # 增強圖像
        print("步驟 4/4: 應用增強...")
        enhanced = self.enhance_image(img_corrected, atmospheric_light, params)
        
        # 保存結果
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_enhanced.png"
        else:
            output_path = Path(output_path)
            if output_path.suffix == '':
                output_path.mkdir(parents=True, exist_ok=True)
                output_path = output_path / f"{input_path.stem}_enhanced.png"
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
        
        enhanced_uint8 = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), enhanced_bgr)
        print(f"✓ 儲存: {output_path}")
        
        return enhanced, params
    
    def process_folder(self, input_folder, output_folder , show_params=True):
        """
        批量處理資料夾
        
        Args:
            input_folder: 輸入資料夾
            output_folder: 輸出資料夾
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 找到所有圖像
        image_files = (
            list(input_path.glob('*.png')) + 
            list(input_path.glob('*.jpg')) + 
            list(input_path.glob('*.jpeg'))
        )
        
        if not image_files:
            print("找不到任何影像檔案！")
            return
        
        print(f"\n找到 {len(image_files)} 張圖像")
        
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 處理: {img_path.name}")
            print("-" * 60)
            try:
                out_file = output_path / f"{img_path.stem}_enhanced.png"
                self.process_single_image(
                    str(img_path), 
                    str(out_file), 
                    show_params=show_params
                )
            except Exception as e:
                print(f"  ✗ 失敗: {e}")
        

# ============================================
# 主程式
# ============================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MATLAB 風格水下影像增強推理'
    )
    
    parser.add_argument('--input', type=str, required=True, 
                       help='輸入影像或資料夾')
    parser.add_argument('--output', type=str, required=True, 
                       help='輸出檔案或資料夾')
    parser.add_argument('--model', type=str, required=True, 
                       help='模型檔案（checkpoint）')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'], help='設備')
    
    args = parser.parse_args()
    
    # 初始化預測器
    predictor = MATLABStylePredictor(
        model_path=args.model, 
        device=args.device
    )
    
    # 處理
    input_path = Path(args.input)
    if input_path.is_file():
        predictor.process_single_image(args.input, args.output)
    elif input_path.is_dir():
        predictor.process_folder(args.input, args.output)
    else:
        raise ValueError("輸入路徑不存在或不是檔案/資料夾")
