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

import logging
from datetime import datetime
logger = logging.getLogger("MATLABStylePredictor")
from airlight_CNN import AtmosphericLightCNN
# 導入模組
from color_correction import ColorCorrection
from matlab_style_enhancement import AtmosphericLightEstimator, MATLABStyleEnhancement
from parameter_predictor import MATLABParameterPredictor
from color_correction_cnn import ColorCorrectionCNN

class MATLABStylePredictor:
    """MATLAB 風格增強預測器"""
    
    def __init__(self, model_path, device='cuda', input_size=224):
      
        self.device = device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        self.input_size = input_size
        
        logger.info(f"Loading model: {model_path}  (device={self.device})")
        
        
        self.param_predictor = MATLABParameterPredictor(
            pretrained=False, 
            hidden_dim=256, 
            use_features=False
        )
        
        ckpt = torch.load(model_path, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)
        self.param_predictor.load_state_dict(state)
        self.param_predictor.to(self.device).eval()
        self.enhancement = MATLABStyleEnhancement().to(self.device).eval()
        
        # self.color_corrector = ColorCorrection()
        self.color_corrector = ColorCorrectionCNN(model_path='D:\research\better_one\color_correction_cnn_model\best_color_correction_model.pth', device='cuda')
        # self.atmos_estimator = AtmosphericLightEstimator(min_size=1)
        logger.info(f"Loading airlight model: {model_path}  (device={self.device})")
        self.atmos_estimator = AtmosphericLightCNN(
            model_path="D:\research\better_one\airlight_cnn_model\best_atmospheric_light_model.pth",
            device=self.device,
            base_channels=16
        )
        
        # VGG 歸一化
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        logger.info("✓ 模型與增強模組已載入")
    
    def preprocess(self, img):
     
        img_corrected, color_type = self.color_corrector(img)
        
        # 大氣光估算
        atmospheric_light = self.atmos_estimator(img_corrected)
        
        return img_corrected, atmospheric_light, color_type
    
    def predict_parameters(self, img, features):
       
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
      
        input_path = Path(input_path)
        
        # 讀取圖像
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"無法讀取影像: {input_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 預處理
        
        img_corrected, atmospheric_light, color_type = self.preprocess(img)
        logger.info(f"  color_type: {color_type}")
        logger.info(
            f"  atmospheric_light: R={atmospheric_light[0]:.4f}, "
            f"G={atmospheric_light[1]:.4f}, B={atmospheric_light[2]:.4f}"
        )
        
        features = np.zeros(79, dtype=np.float32)
        
        # 預測參數
        
        params = self.predict_parameters(img_corrected, features)
        
        if show_params:
            logger.info("  parameters:")
            for k, v in params.items():
                logger.info(f"    {k}: {v.item():.4f}")
        
        # Enhance image
        
        enhanced = self.enhance_image(img_corrected, atmospheric_light, params)
        
        # Save result
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
        logger.info(f"✓ 儲存: {output_path}")
        
        return enhanced, params
    
    def process_folder(self, input_folder, output_folder , show_params=True):
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
    
        image_files = (
            list(input_path.glob('*.png')) + 
            list(input_path.glob('*.jpg')) + 
            list(input_path.glob('*.jpeg'))
        )
        
        if not image_files:
            logger.warning("No image files found!")
            return
        
        logger.info(f"\nFound {len(image_files)} images")
        
        for i, img_path in enumerate(image_files, 1):
            logger.info(f"\n[{i}/{len(image_files)}] 處理: {img_path.name}")
            logger.info("-" * 60)
            try:
                 out_file = output_path / f"{img_path.stem}_enhanced.png"
                 self.process_single_image(
                     str(img_path), 
                     str(out_file), 
                     show_params=show_params
                 )
            except Exception as e:
                 logger.error(f"  ✗  Failed: {e}", exc_info=True)
        

# ============================================
# 主程式
# ============================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MATLAB style underwater image enhancement inference'
    )
    
    parser.add_argument('--input', type=str, required=True, 
                       help='input file or folder')
    parser.add_argument('--output', type=str, required=True, 
                       help='output file or folder')
    parser.add_argument('--model', type=str, required=True, 
                       help='model (checkpoint)')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'], help='device')
    
    args = parser.parse_args()
    
    # Set up logging: output to console and a log file in the output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"run_{timestamp}.log"

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(str(log_path), encoding='utf-8')
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    # 移除預設 handlers（以避免重複）
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Logging to {log_path}")
    
    predictor = MATLABStylePredictor(
        model_path=args.model, 
        device=args.device
    )
    
    input_path = Path(args.input)
    if input_path.is_file():
        predictor.process_single_image(args.input, args.output)
    elif input_path.is_dir():
        predictor.process_folder(args.input, args.output)
    else:
        raise ValueError("Input path does not exist or is not a file/folder")
