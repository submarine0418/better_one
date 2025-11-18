# test_gpu_training.py
import torch
import torch.nn as nn
import time

print("="*80)
print("測試 GPU 訓練")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用設備: {device}")

# 創建簡單模型
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 3, 3, padding=1)
).to(device)

print(f"模型參數量: {sum(p.numel() for p in model.parameters())}")

# 創建測試數據
print("\n創建測試數據...")
batch_size = 2
input_data = torch.randn(batch_size, 3, 224, 224).to(device)
target_data = torch.randn(batch_size, 3, 224, 224).to(device)

print(f"Input device: {input_data.device}")
print(f"GPU 記憶體: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

# 訓練循環
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("\n開始訓練...")
for step in range(5):
    start = time.time()
    
    # 前向傳播
    output = model(input_data)
    loss = criterion(output, target_data)
    
    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
   
  