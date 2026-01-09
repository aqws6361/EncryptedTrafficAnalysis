import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import time
import os

# --- 1. 設定裝置 (自動偵測 GPU) ---
def get_device():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ 偵測到 GPU: {device_name}")
        return torch.device("cuda")
    else:
        print("⚠️ 未偵測到 GPU，將使用 CPU 訓練")
        return torch.device("cpu")

device = get_device()

# --- 2. 定義 Transformer 模型架構 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MalwareDetectorTransformer(nn.Module):
    def __init__(self, input_size, d_model, num_classes, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(MalwareDetectorTransformer, self).__init__()
        
        # 1. Feature Embedding: Project 2D features to d_model
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 3. Classifier
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.embedding(x)  # -> (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Global Average Pooling
        x = x.mean(dim=1)  # -> (batch, d_model)
        x = self.dropout(x)
        out = self.fc(x)
        return out

if __name__ == "__main__":
    # --- 自動定位路徑 ---
    current_dir = os.path.dirname(os.path.abspath(__file__)) # src/
    project_root = os.path.dirname(current_dir)              # root/
    
    data_dir = os.path.join(project_root, "data")
    model_dir = os.path.join(project_root, "model")
    
    # 確保 model 資料夾存在
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"建立資料夾: {model_dir}")

    # 定義完整檔案路徑
    x_path = os.path.join(data_dir, "X_data.npy")
    y_path = os.path.join(data_dir, "y_data.npy")
    model_save_path = os.path.join(model_dir, "iot_malware_model.pth")

    print(f"\n[Step 1] 正在檢查數據路徑...")
    print(f"   預期路徑 X: {x_path}")
    print(f"   預期路徑 y: {y_path}")

    # --- 3. 載入資料 ---
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"\n❌ 錯誤: 在上述路徑找不到 .npy 檔案！")
        print(f"💡 請先執行 'python 1_data_prep.py' 來產生數據，並確保它儲存到 data 資料夾。")
        exit()

    try:
        X = np.load(x_path)
        y = np.load(y_path)
        print(f"✅ 資料載入成功! X shape: {X.shape}, y shape: {y.shape}")
    except Exception as e:
        print(f"❌ 讀取錯誤: {e}")
        exit()

    # 切分訓練集 (80%) 與測試集 (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 轉 Tensor
    X_train_tensor = torch.from_numpy(X_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).long().to(device)
    X_test_tensor = torch.from_numpy(X_test).to(device)
    y_test_tensor = torch.from_numpy(y_test).long().to(device)

    # 建立 DataLoader
    BATCH_SIZE = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 4. 初始化模型 (Transformer 設定) ---
    INPUT_SIZE = 2
    D_MODEL = 64        # Embedding Dimension
    NUM_CLASSES = 2
    LEARNING_RATE = 0.001
    EPOCHS = 15         # Transformer 可能需要多一點 epochs
    
    # 初始化 Transformer
    model = MalwareDetectorTransformer(INPUT_SIZE, D_MODEL, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n[Step 2] 開始訓練 (Epochs: {EPOCHS})...")
    print("-" * 50)

    # --- 5. 訓練迴圈 ---
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Accuracy: {epoch_acc:.2f}%")

    training_time = time.time() - start_time
    print("-" * 50)
    print(f"訓練完成! 總耗時: {training_time:.2f} 秒")

    # --- 6. 測試 ---
    print("\n[Step 3] 評估測試集效能...")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        total = y_test_tensor.size(0)
        correct = (predicted == y_test_tensor).sum().item()
        test_acc = 100 * correct / total
        print(f"🎯 測試集準確率: {test_acc:.2f}%")

    # --- 7. 存檔 ---
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ 模型已儲存為 '{model_save_path}'")