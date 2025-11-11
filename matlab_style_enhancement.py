"""
MATLAB 風格水下影像增強模組 (可微分版本)
包含: 大氣光估算、初始透射率、梯度約束、引導濾波、影像恢復、色彩拉伸
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================
# 1. 大氣光估算 (不可微分，預處理階段)
# ============================================

class AtmosphericLightEstimator:
    """
    使用四叉樹分割的大氣光估算器
    完全複製 MATLAB 的 find_brightest_region 函數
    """
    
    def __init__(self, min_size=1):
        """
        Args:
            min_size: 最小分割區域大小
        """
        self.min_size = min_size
    
    def __call__(self, img):
        """
        估算圖像的大氣光值
        
        Args:
            img: numpy array, shape (H, W, 3), RGB, [0, 1] float
        
        Returns:
            atmospheric_light: numpy array, shape (3,), RGB 大氣光值
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # 確保是 float [0, 1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        
        # 使用四叉樹找最亮區域
        max_rgb, _ = self._find_brightest_region(img)
        
        return max_rgb
    
    def _find_brightest_region(self, img):
        """
        四叉樹遞迴分割找最亮區域
        
        Args:
            img: (H, W, 3) RGB [0, 1]
        
        Returns:
            max_rgb: (3,) 最亮區域的 RGB 值
            max_q: float, 最大 Q 值
        """
        n_rows, n_cols, _ = img.shape
        
        max_q = -np.inf
        max_rgb = np.array([0.0, 0.0, 0.0])
        
        # 使用 stack 模擬遞迴
        stack = []
        stack.append({
            'block': img,
            'n_rows': n_rows,
            'n_cols': n_cols
        })
        
        while stack:
            current = stack.pop()
            block = current['block']
            n_rows = current['n_rows']
            n_cols = current['n_cols']
            
            # 到達最小區塊
            if n_rows <= self.min_size or n_cols <= self.min_size:
                # 計算此區塊的 Q 值和最亮像素
                q = self._compute_q(block)
                brightest_rgb = self._get_brightest_pixel(block)
                
                # 更新全域最大值
                if q > max_q:
                    max_q = q
                    max_rgb = brightest_rgb
            else:
                # 繼續分割
                mid_row = n_rows // 2
                mid_col = n_cols // 2
                
                # 分割為四個區域
                block1 = block[:mid_row, :mid_col, :]
                block2 = block[:mid_row, mid_col:, :]
                block3 = block[mid_row:, :mid_col, :]
                block4 = block[mid_row:, mid_col:, :]
                
                # 計算四個區域的 Q 值
                q1 = self._compute_q(block1)
                q2 = self._compute_q(block2)
                q3 = self._compute_q(block3)
                q4 = self._compute_q(block4)
                
                # 找到 Q 值最大的區域
                q_values = [q1, q2, q3, q4]
                blocks = [block1, block2, block3, block4]
                sizes = [
                    (mid_row, mid_col),
                    (mid_row, n_cols - mid_col),
                    (n_rows - mid_row, mid_col),
                    (n_rows - mid_row, n_cols - mid_col)
                ]
                
                max_idx = np.argmax(q_values)
                
                # 只把 Q 值最大的區域放回 stack
                stack.append({
                    'block': blocks[max_idx],
                    'n_rows': sizes[max_idx][0],
                    'n_cols': sizes[max_idx][1]
                })
        
        return max_rgb, max_q
    
    def _compute_q(self, block):
        """
        計算區塊的 Q 值
        Q = term1 + term2 - term3 - term4
        
        Args:
            block: (H, W, 3) RGB [0, 1]
        
        Returns:
            q: float
        """
        n_rows, n_cols, _ = block.shape
        n = n_rows * n_cols
        
        I_r = block[:, :, 0]
        I_g = block[:, :, 1]
        I_b = block[:, :, 2]
        
        # Term 1: 亮度平均
        term1 = (np.sum(I_r) + np.sum(I_g) + np.sum(I_b)) / (3 * n)
        
        # Term 2: 色彩對比項 (藍綠強於紅)
        term2 = (np.sum(I_b) + np.sum(I_g) - 2 * np.sum(I_r)) / n
        
        # Term 3: 色彩變異項
        mean_r = np.mean(I_r)
        mean_g = np.mean(I_g)
        mean_b = np.mean(I_b)
        
        var_r = np.sum((I_r - mean_r)**2) / n
        var_g = np.sum((I_g - mean_g)**2) / n
        var_b = np.sum((I_b - mean_b)**2) / n
        
        term3 = (var_r + var_g + var_b) / 3
        
        # Term 4: 邊緣數量
        gray_img = 0.299 * I_r + 0.587 * I_g + 0.114 * I_b
        
        # 簡化的邊緣檢測
        if gray_img.shape[0] > 3 and gray_img.shape[1] > 3:
            # Sobel 邊緣檢測
            from scipy import ndimage
            edge_x = ndimage.sobel(gray_img, axis=0)
            edge_y = ndimage.sobel(gray_img, axis=1)
            edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
            edge_density = np.sum(edge_magnitude > 0.1) / n
        else:
            edge_density = 0.0
        
        term4 = edge_density
        
        # 合成 Q 值
        q = term1 + term2 - term3 - term4
        
        return q
    
    def _get_brightest_pixel(self, block):
        """
        找到區塊中最亮的像素
        
        Args:
            block: (H, W, 3) RGB [0, 1]
        
        Returns:
            brightest_rgb: (3,) 最亮像素的 RGB 值
        """
        # 計算每個像素的亮度（RGB 和）
        pixel_sum = np.sum(block, axis=2)
        
        # 找到最大亮度的索引
        max_idx = np.argmax(pixel_sum)
        row, col = np.unravel_index(max_idx, pixel_sum.shape)
        
        # 返回最亮像素的 RGB 值
        brightest_rgb = block[row, col, :]
        
        return brightest_rgb
    
    def estimate_batch(self, img_batch):
        """
        批次處理多張圖像
        
        Args:
            img_batch: numpy array (B, H, W, 3) 或 torch.Tensor (B, 3, H, W)
        
        Returns:
            atmospheric_lights: numpy array (B, 3)
        """
        if isinstance(img_batch, torch.Tensor):
            # (B, 3, H, W) -> (B, H, W, 3)
            img_batch = img_batch.permute(0, 2, 3, 1).cpu().numpy()
        
        batch_size = img_batch.shape[0]
        atmospheric_lights = np.zeros((batch_size, 3), dtype=np.float32)
        
        for i in range(batch_size):
            atmospheric_lights[i] = self(img_batch[i])
        
        return atmospheric_lights


# ============================================
# 2. 引導濾波 (可微分)
# ============================================

class GuidedFilter(nn.Module):
    """
    可微分的引導濾波
    完全複製 MATLAB 的 guidedFilter 函數邏輯
    """
    
    def __init__(self, radius=15, eps=5e-1):
        """
        Args:
            radius: 濾波窗口半徑
            eps: 正則化參數
        """
        super().__init__()
        self.r = radius
        self.eps = eps
    
    def forward(self, guide, input_map):
        """
        引導濾波
        
        Args:
            guide: (B, 1, H, W) 引導圖（灰度）
            input_map: (B, 1, H, W) 輸入圖（透射圖）
        
        Returns:
            output: (B, 1, H, W) 濾波後的圖
        """
        # 使用 box filter 計算均值
        mean_I = self.box_filter(guide)
        mean_p = self.box_filter(input_map)
        mean_Ip = self.box_filter(guide * input_map)
        
        # 協方差
        cov_Ip = mean_Ip - mean_I * mean_p
        
        # 方差
        mean_II = self.box_filter(guide * guide)
        var_I = mean_II - mean_I * mean_I
        
        # 計算線性係數
        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        
        # 再次平均
        mean_a = self.box_filter(a)
        mean_b = self.box_filter(b)
        
        # 最終輸出
        output = mean_a * guide + mean_b
        
        return output
    
    def box_filter(self, x):
        """
        Box filter (均值濾波)
        
        Args:
            x: (B, C, H, W)
        
        Returns:
            filtered: (B, C, H, W)
        """
        kernel_size = 2 * self.r + 1
        
        # 使用 avg_pool2d 實作
        # 需要先 pad
        pad_size = self.r
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        # 平均池化
        filtered = F.avg_pool2d(
            x_padded,
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )
        
        return filtered


# ============================================
# 3. MATLAB 風格完整增強模組 (可微分)
# ============================================

class MATLABStyleEnhancement(nn.Module):
    """
    MATLAB 風格的水下影像增強（完整流程）
    包含: 初始透射率、梯度約束、引導濾波、影像恢復、色彩拉伸
    """
    
    def __init__(self):
        """
        初始化增強模組
        引導濾波的 radius 和 eps 將由參數決定
        """
        super().__init__()
    
    def forward(self, img, params, atmospheric_light):
        """
        完整的 MATLAB 風格增強流程
        
        Args:
            img: (B, 3, H, W) 輸入圖像 [0, 1]，已經過色偏校正
            params: dict, 包含預測的參數
                - omega: (B, 1) 去霧強度
                - guided_radius: (B, 1) 引導濾波半徑
                - L_low: (B, 1) 色彩拉伸下界百分位
                - L_high: (B, 1) 色彩拉伸上界百分位
            atmospheric_light: (B, 3) 預計算的大氣光值
        
        Returns:
            enhanced: (B, 3, H, W) 增強後的圖像 [0, 1]
            intermediate: dict, 中間結果（用於分析）
        """
        B, C, H, W = img.shape
        
        # ========================================
        # Step 1: 計算初始透射率
        # ========================================
        t_initial = self.compute_initial_transmission(
            img, atmospheric_light, params['omega']
        )
        
        # ========================================
        # Step 2: 應用梯度約束
        # ========================================
        t_gradient = self.apply_gradient_constraint(t_initial)
        
        # ========================================
        # Step 3: 引導濾波細化
        # ========================================
        # 由於每張圖的 radius 可能不同，需要逐張處理
        t_final = self.guided_filter_batch(
            img, t_gradient, params['guided_radius']
        )
        
        # 限制透射率範圍
        t_final = torch.clamp(t_final, 0.1, 1.0)
        
        # ========================================
        # Step 4: 恢復影像
        # ========================================
        J_restored = self.restore_image(img, atmospheric_light, t_final)
        
        # ========================================
        # Step 5: 色彩拉伸
        # ========================================
        J_enhanced = self.color_stretch_batch(
            J_restored, params['L_low'], params['L_high']
        )
        
        # 收集中間結果
        intermediate = {
            't_initial': t_initial,
            't_gradient': t_gradient,
            't_final': t_final,
            'J_restored': J_restored,
        }
        
        return J_enhanced, intermediate
    
    def compute_initial_transmission(self, img, atmospheric_light, omega):
        """
        計算初始透射率
        t = 1 - omega * min(I / A)
        
        Args:
            img: (B, 3, H, W)
            atmospheric_light: (B, 3)
            omega: (B, 1)
        
        Returns:
            t_initial: (B, 1, H, W)
        """
        B = img.size(0)
        
        # 將大氣光擴展到圖像尺寸
        A = atmospheric_light.view(B, 3, 1, 1)  # (B, 3, 1, 1)
        
        # 正規化圖像
        img_normalized = img / (A + 1e-8)
        
        # 計算暗通道（最小值）
        dark_channel = torch.min(img_normalized, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        
        # 計算透射率
        omega = omega.view(B, 1, 1, 1)
        t_initial = 1 - omega * dark_channel
        
        return t_initial
    
    def apply_gradient_constraint(self, t):
        """
        應用梯度約束（完全無 in-place，無切片賦值）
        """
        B, C, H, W = t.shape

        # 計算 X 方向權重
        diff_x = torch.abs(t[:, :, :, 1:] - t[:, :, :, :-1])  # (B,1,H,W-1)
        weight_x = torch.exp(-diff_x)

        # 計算 Y 方向權重
        diff_y = torch.abs(t[:, :, 1:, :] - t[:, :, :-1, :])  # (B,1,H-1,W)
        weight_y = torch.exp(-diff_y)

        # === X 方向：構建新 t_x ===
        left_part = t[:, :, :, :1]  # 第一列不變
        right_part = t[:, :, :, :-1] * weight_x  # 右邊 = 左邊 * 權重
        t_x = torch.cat([left_part, right_part], dim=3)  # (B,1,H,W)

        # === Y 方向：對 t_x 應用 Y 權重 ===
        top_part = t_x[:, :, :1, :]  # 第一行不變
        bottom_part = t_x[:, :, :-1, :] * weight_y  # 下邊 = 上邊 * 權重
        t_out = torch.cat([top_part, bottom_part], dim=2)  # (B,1,H,W)

        return t_out
    
    
    
    def guided_filter_batch(self, img, t, guided_radius):
        """
        批次引導濾波（處理不同 radius）
        
        Args:
            img: (B, 3, H, W)
            t: (B, 1, H, W)
            guided_radius: (B, 1) 每張圖的濾波半徑
        
        Returns:
            t_filtered: (B, 1, H, W)
        """
        B = img.size(0)
        t_filtered_list = []
        
        # 轉換為灰度图
        img_gray = self.rgb_to_gray(img)
        
        # 逐張處理（因為 radius 可能不同）
        for b in range(B):
            radius = int(guided_radius[b].item())
            radius = max(1, min(radius, 50))  # 限制範圍 [1, 50]
            
            # 創建引導濾波器
            guided_filter = GuidedFilter(radius=radius, eps=5e-1)
            
            # 單張處理
            # clone 切片以避免 view 導致的原地版本衝突
            guide_single = img_gray[b:b+1]  # 已是 view
            t_single = t[b:b+1]
            
            t_filtered_single = guided_filter(guide_single, t_single)
            t_filtered_list.append(t_filtered_single)
        
        # 拼接回 batch
        t_filtered = torch.cat(t_filtered_list, dim=0)
        
        return t_filtered
    
    def restore_image(self, img, atmospheric_light, transmission):
        B = img.size(0)
    
        A = atmospheric_light.view(B, 3, 1, 1)
        t = transmission.expand(B, 3, -1, -1).clone()  # 保留 clone，避免版本問題
        
        restored = (img - A) / (t + 1e-8) + A
        return torch.clamp(restored, 0.0, 1.0)
    
    def color_stretch_batch(self, img, L_low, L_high):
        B, C, H, W = img.shape
        stretched = img.clone()  # 從原圖開始
        
        for b in range(B):
            low_pct = L_low[b].item() / 100.0
            high_pct = L_high[b].item() / 100.0
            
            for c in range(C):
                channel = img[b, c]  # (H, W)
                flat = channel.flatten()
                n = flat.shape[0]
                
                # 計算百分位
                low_idx = int(low_pct * n)
                high_idx = int(high_pct * n)
                low_idx = max(0, min(low_idx, n-1))
                high_idx = max(0, min(high_idx, n-1))
                
                sorted_vals, _ = torch.sort(flat)
                p_low = sorted_vals[low_idx]
                p_high = sorted_vals[high_idx]
                
                # 拉伸（無 in-place）
                clipped = torch.clamp(channel, p_low, p_high)
                stretched[b, c] = (clipped - p_low) / (p_high - p_low + 1e-8)
        
        return torch.clamp(stretched, 0.0, 1.0)
    
    def rgb_to_gray(self, img):
        """
        RGB 轉灰度
        
        Args:
            img: (B, 3, H, W)
        
        Returns:
            gray: (B, 1, H, W)
        """
        # 使用標準權重: 0.299*R + 0.587*G + 0.114*B
        weights = torch.tensor([0.299, 0.587, 0.114], device=img.device).view(1, 3, 1, 1)
        gray = (img * weights).sum(dim=1, keepdim=True)
        
        return gray


# ============================================
# 4. 使用範例與說明
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("MATLAB 風格增強模組")
    print("=" * 60)
    print("""
使用方式:

1. 預處理階段 (不可微分):
   ----------------------------------------
   from color_correction import ColorCorrection
   from matlab_style_enhancement import AtmosphericLightEstimator
   
   # 色偏校正
   corrector = ColorCorrection()
   img_corrected, color_type = corrector(img)
   
   # 大氣光估算
   estimator = AtmosphericLightEstimator(min_size=1)
   atmospheric_light = estimator(img_corrected)

2. 深度學習階段 (可微分):
   ----------------------------------------
   from matlab_style_enhancement import MATLABStyleEnhancement
   
   # 初始化增強模組
   enhancer = MATLABStyleEnhancement()
   
   # 參數預測（由神經網路輸出）
   params = {
       'omega': torch.tensor([[0.5]]),           # 去霧強度
       'guided_radius': torch.tensor([[15.0]]),  # 引導濾波半徑
       'L_low': torch.tensor([[15.0]]),          # 色彩拉伸下界
       'L_high': torch.tensor([[95.0]]),         # 色彩拉伸上界
   }
   
   # 增強圖像
   enhanced, intermediate = enhancer(
       img_tensor, 
       params, 
       atmospheric_light_tensor
   )

3. 完整訓練流程:
   ----------------------------------------
   # 數據載入時：色偏校正 + 大氣光估算
   # 前向傳播時：參數預測 + 可微分增強
   # 損失計算：compare(enhanced, reference)
   # 反向傳播：只更新參數預測網路
    """)
    
    print("\n✓ 模組載入完成")
