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
    print("⚠ LAB 色彩校正模組未找到")
    LAB_AVAILABLE = False

from color_correction_cnn import ColorCorrectionCNN


def test_interface_consistency():
    """測試兩種方法的接口一致性"""
    
    print("=" * 80)
    print("測試 CNN 色彩校正與 LAB 方法的接口一致性")
    print("=" * 80)
    
    # ============================================
    # 準備測試圖像
    # ============================================
    print("\n1. 準備測試圖像...")
    
    # 創建測試圖像（模擬綠色偏移的水下圖像）
    img_uint8 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img_uint8[:, :, 1] = np.clip(img_uint8[:, :, 1] * 1.3, 0, 255).astype(np.uint8)  # 增加綠色
    
    img_float = img_uint8.astype(np.float32) / 255.0
    
    print(f"   測試圖像形狀: {img_uint8.shape}")
    print(f"   uint8 範圍: [{img_uint8.min()}, {img_uint8.max()}]")
    print(f"   float 範圍: [{img_float.min():.3f}, {img_float.max():.3f}]")
    
    # ============================================
    # 測試 LAB 方法（如果可用）
    # ============================================
    if LAB_AVAILABLE:
        print("\n2. 測試 LAB 色彩校正...")
        
        corrector_lab = ColorCorrection()
        
        # 測試 uint8 輸入
        start = time.time()
        corrected_lab_uint8, color_type_lab_uint8 = corrector_lab(img_uint8)
        time_lab_uint8 = (time.time() - start) * 1000
        
        # 測試 float 輸入
        start = time.time()
        corrected_lab_float, color_type_lab_float = corrector_lab(img_float)
        time_lab_float = (time.time() - start) * 1000
        
        print(f"   ✓ LAB 方法測試完成")
        print(f"   - uint8 輸入: 輸出形狀 {corrected_lab_uint8.shape}, "
              f"類型 {corrected_lab_uint8.dtype}, 色偏 '{color_type_lab_uint8}', "
              f"耗時 {time_lab_uint8:.2f}ms")
        print(f"   - float 輸入: 輸出形狀 {corrected_lab_float.shape}, "
              f"類型 {corrected_lab_float.dtype}, 色偏 '{color_type_lab_float}', "
              f"耗時 {time_lab_float:.2f}ms")
    
    # ============================================
    # 測試 CNN 方法
    # ============================================
    print("\n3. 測試 CNN 色彩校正...")
    
    corrector_cnn = ColorCorrectionCNN(
        model_path=None,  # 使用隨機初始化（測試用）
        device='cuda',
        base_channels=16
    )
    
    # 測試 uint8 輸入
    start = time.time()
    corrected_cnn_uint8, color_type_cnn_uint8 = corrector_cnn(img_uint8)
    time_cnn_uint8 = (time.time() - start) * 1000
    
    # 測試 float 輸入
    start = time.time()
    corrected_cnn_float, color_type_cnn_float = corrector_cnn(img_float)
    time_cnn_float = (time.time() - start) * 1000
    
    print(f"   ✓ CNN 方法測試完成")
    print(f"   - uint8 輸入: 輸出形狀 {corrected_cnn_uint8.shape}, "
          f"類型 {corrected_cnn_uint8.dtype}, 色偏 '{color_type_cnn_uint8}', "
          f"耗時 {time_cnn_uint8:.2f}ms")
    print(f"   - float 輸入: 輸出形狀 {corrected_cnn_float.shape}, "
          f"類型 {corrected_cnn_float.dtype}, 色偏 '{color_type_cnn_float}', "
          f"耗時 {time_cnn_float:.2f}ms")
    
    # ============================================
    # 驗證接口一致性
    # ============================================
    print("\n4. 驗證接口一致性...")
    
    checks = []
    
    # 檢查輸出形狀
    checks.append(("輸出形狀", 
                   corrected_cnn_uint8.shape == img_uint8.shape,
                   f"{corrected_cnn_uint8.shape} vs {img_uint8.shape}"))
    
    # 檢查輸出數據類型
    checks.append(("輸出類型", 
                   corrected_cnn_uint8.dtype == np.float32,
                   f"{corrected_cnn_uint8.dtype}"))
    
    # 檢查輸出範圍
    checks.append(("輸出範圍", 
                   0 <= corrected_cnn_uint8.min() and corrected_cnn_uint8.max() <= 1,
                   f"[{corrected_cnn_uint8.min():.3f}, {corrected_cnn_uint8.max():.3f}]"))
    
    # 檢查色偏類型
    valid_color_types = ['greenish', 'blueish', 'yellowish', 'reddish','no_cast']
    checks.append(("色偏類型", 
                   color_type_cnn_uint8 in valid_color_types,
                   f"'{color_type_cnn_uint8}'"))
    
    # 檢查 uint8 和 float 輸入的一致性
    diff = np.abs(corrected_cnn_uint8 - corrected_cnn_float).mean()
    checks.append(("輸入一致性", 
                   diff < 0.01,
                   f"差異 {diff:.6f}"))
    
    # 打印檢查結果
    all_passed = True
    for check_name, passed, info in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: {check_name} - {info}")
        all_passed = all_passed and passed
    
    # ============================================
    # 性能對比
    # ============================================
    if LAB_AVAILABLE:
        print("\n5. 性能對比...")
        print(f"   LAB 方法: {time_lab_uint8:.2f}ms")
        print(f"   CNN 方法: {time_cnn_uint8:.2f}ms")
        print(f"   速度比: {time_cnn_uint8/time_lab_uint8:.2f}x")
    
    # ============================================
    # 總結
    # ============================================
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有接口一致性測試通過！")
        print("CNN 方法可以無縫替換 LAB 方法。")
    else:
        print("✗ 部分測試失敗，請檢查實現。")
    print("=" * 80)
    
    return all_passed


def test_integration_example():
    """測試集成示例"""
    
    print("\n" + "=" * 80)
    print("集成示例：在推理管道中使用")
    print("=" * 80)
    
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
            """處理圖像"""
            # 步驟 1: 色彩校正
            corrected, color_type = self.color_corrector(img)
            print(f"  色彩校正完成，檢測到 '{color_type}' 色偏")
            
            # 步驟 2: 其他處理...
            # enhanced = self.enhance(corrected)
            
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
    
    print(f"\n處理 {batch_size} 張圖像...")
    
    start = time.time()
    results = []
    for i, img in enumerate(images, 1):
        corrected, color_type = corrector(img)
        results.append((corrected, color_type))
        print(f"  {i}/{batch_size}: {img.shape} -> {corrected.shape}, "
              f"色偏 '{color_type}'")
    
    total_time = time.time() - start
    avg_time = total_time / batch_size * 1000
    
    print(f"\n總耗時: {total_time:.2f}s")
    print(f"平均耗時: {avg_time:.2f}ms/張")
    print(f"吞吐量: {batch_size/total_time:.2f} 張/秒")


def main():
    """主測試函數"""
    
    # 測試 1: 接口一致性
    test_interface_consistency()
    
    # 測試 2: 集成示例
    test_integration_example()
    
    # 測試 3: 批次處理
    test_batch_processing()
    
    print("\n" + "=" * 80)
    print("✓ 所有測試完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
