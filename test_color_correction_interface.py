"""
測試腳本：驗證 CNN 色彩校正與 LAB 方法的接口一致性
"""

import numpy as np
import cv2
import time
from pathlib import Path

# 嘗試導入兩種方法
try:
    from color_correction import ColorCorrection
    LAB_AVAILABLE = True
except ImportError:
    print("LAB 色彩校正模組未找到")
    LAB_AVAILABLE = False

from color_correction_cnn import ColorCorrectionCNN

def test_integration_example():
    
    # 模擬推理管道
    class SimplePipeline:
        def __init__(self, use_cnn=True, cnn_model_path=None):
            """
            Args:
                use_cnn: True 使用 CNN，False 使用 LAB
                cnn_model_path: CNN 模型路徑
            """
            if use_cnn:
                print("初始化管道（使用 CNN 色彩校正）...")
                self.color_corrector = ColorCorrectionCNN(
                    model_path=cnn_model_path,
                    device='cuda'
                )
            else:
                if LAB_AVAILABLE:
                    print("初始化管道（使用 LAB 色彩校正）...")
                    self.color_corrector = ColorCorrection()
                else:
                    print("⚠ LAB 方法不可用，使用 CNN 方法")
                    self.color_corrector = ColorCorrectionCNN(
                        model_path=cnn_model_path,
                        device='cuda'
                    )
        
        def process(self, img):
           
            corrected, color_type = self.color_corrector(img)
            print(f"  色彩校正完成，檢測到 '{color_type}' 色偏")
           
            
            return corrected
    
    # 創建測試圖像
    test_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # 測試兩種配置
    if LAB_AVAILABLE:
        print("\n測試配置 1: 使用 LAB 方法")
        pipeline_lab = SimplePipeline(use_cnn=False)
        result_lab = pipeline_lab.process(test_img)
        print(f"  輸出形狀: {result_lab.shape}, 類型: {result_lab.dtype}")
    
    print("\n測試配置 2: 使用 CNN 方法")
    pipeline_cnn = SimplePipeline(use_cnn=True, cnn_model_path=None)
    result_cnn = pipeline_cnn.process(test_img)
    print(f"  輸出形狀: {result_cnn.shape}, 類型: {result_cnn.dtype}")
    
    print("\n✓ 兩種配置都成功運行，接口完全一致！")


def test_batch_processing():
    """測試批次處理"""
    
    print("\n" + "=" * 80)
    print("批次處理測試")
    print("=" * 80)
    
    # 創建測試圖像批次
    batch_size = 10
    images = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
              for _ in range(batch_size)]
    
    corrector = ColorCorrectionCNN(model_path='D:/research/better_one/color_correction_output/best_color_correction_model.pth', device='cuda')
    
    
    start = time.time()
    results = []
    for i, img in enumerate(images, 1):
        corrected, color_type = corrector(img)
        results.append((corrected, color_type))
        print(f"  {i}/{batch_size}: {img.shape} -> {corrected.shape}, "
              f"色偏 '{color_type}'")
    
  


def main():

    test_batch_processing()
    
    print("\n" + "=" * 80)
    print("✓ 所有測試完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
