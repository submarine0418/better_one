
import numpy as np
from skimage import color
import cv2


class ColorCorrection:
    """
    水下影像色偏校正
    根據 LAB 色彩空間分析判斷色偏類型並進行校正
    """
    
    def __init__(self):
        """初始化色偏校正器"""
        self.color_types = {
            'greenish': self._correct_greenish,
            'blueish': self._correct_blueish,
            'yellowish': self._correct_yellowish,
            'reddish': self._correct_reddish,
            'no_cast': self._no_correction
        }
    
    def _no_correction(self, img):
        """無色偏，不做校正"""
        return img
    
    def __call__(self, img):
       
        # 確保輸入是 float [0, 1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
        
        # 判斷色偏類型
        color_type = self.detect_color_cast(img)
        
        # 應用對應的校正方法
        corrected_img = self.color_types[color_type](img.copy())
        
        # 確保輸出在 [0, 1] 範圍
        corrected_img = np.clip(corrected_img, 0.0, 1.0)
        
        return corrected_img, color_type
    
    def detect_color_cast(self, img):
        """
        檢測色偏類型
        
        Args:
            img: (H, W, 3) RGB float32 [0, 1]
        
        Returns:
            color_type: str
        """
        # 轉換到 LAB 色彩空間
        lab = color.rgb2lab(img)
        
        # 分離通道
        L = lab[:, :, 0]  # 亮度 [0, 100]
        A = lab[:, :, 1]  # 綠-紅 [-128, 127]
        B = lab[:, :, 2]  # 藍-黃 [-128, 127]
        
        # 計算 A 和 B 通道的均值
        mean_A = np.mean(A)
        mean_B = np.mean(B)
        median_L = np.median(L)
        
        # 計算色偏強度 M
        M = np.sqrt(mean_A**2 + mean_B**2)
        
        # 計算每個像素與均值的距離
        distances_A = np.abs(A - mean_A)
        distances_B = np.abs(B - mean_B)
        
        Da = np.mean(distances_A)
        Db = np.mean(distances_B)
        
        # 計算距離強度 D
        D = np.sqrt(Da**2 + Db**2)
        
        # 色偏因子 CCF (Color Cast Factor)
        CCF = M / (D + 1e-8)
        
        # 判斷色偏類型
        if CCF >= 1 and median_L >= 25:
            # 計算 A/B 比例
            ratio = np.abs(mean_A / (mean_B + 1e-8))
            
            if mean_A < 0 and ratio >= 1:
                return 'greenish'
            elif mean_B < 0 and ratio < 1:
                return 'blueish'
            elif mean_B > 0 and ratio < 1:
                return 'yellowish'
            elif mean_A > 0 and ratio >= 1:
                return 'reddish'
        
        return 'no_cast'
    
    def _correct_greenish(self, img):
        """綠偏"""
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        
        mean_R = np.mean(R)
        mean_G = np.mean(G)
        mean_B = np.mean(B)
        
        newR = R + (1 - mean_R / (mean_G + 1e-8)) * (1 - R) * G
        newmean_R = np.mean(newR)
        final_R = (newR / (newmean_R + 1e-8)) * 0.5
        
        newB = B + (1 - mean_B / (mean_G + 1e-8)) * (1 - B) * G
        newmean_B = np.mean(newB)
        final_B = (newB / (newmean_B + 1e-8)) * 0.5
        
        corrected = np.stack([final_R, G, final_B], axis=2)
        return corrected
    
    def _correct_blueish(self, img):
        """藍偏"""
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        
        mean_R = np.mean(R)
        mean_G = np.mean(G)
        mean_B = np.mean(B)
        
        newG = G + (1 - mean_G / (mean_B + 1e-8)) * (1 - G) * B
        newmean_G = np.mean(newG)
        final_G = (newG / (newmean_G + 1e-8)) * 0.5
        
        newR = R + (1 - mean_R / (mean_G + 1e-8)) * (1 - R) * final_G
        newmean_R = np.mean(newR)
        final_R = (newR / (newmean_R + 1e-8)) * 0.5
        
        corrected = np.stack([final_R, final_G, B], axis=2)
        return corrected
    
    def _correct_yellowish(self, img):
        """黃偏"""
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        
        mean_R = np.mean(R)
        mean_G = np.mean(G)
        mean_B = np.mean(B)
        
        newG = G + (1 - mean_G / (mean_R + 1e-8)) * (1 - G) * R
        newmean_G = np.mean(newG)
        final_G = (newG / (newmean_G + 1e-8)) * 0.5
        
        newB = B + (1 - mean_B / (mean_R + 1e-8)) * (1 - B) * R
        newmean_B = np.mean(newB)
        final_B = (newB / (newmean_B + 1e-8)) * 0.5
        
        corrected = np.stack([R, final_G, final_B], axis=2)
        return corrected
    
    def _correct_reddish(self, img):
        """紅偏"""
        return img


if __name__ == '__main__':
    print("色偏校正模組已載入")