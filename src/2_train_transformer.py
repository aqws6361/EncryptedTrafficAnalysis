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
        print(f"[GPU] 偵測到 GPU: {device_name}")
        return torch.device("cuda")
    else:
        print("[CPU] 未偵測到 GPU，將使用 CPU 訓練")
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
    # [MODIFY] Change model save name to distinguish from LSTM
    model_save_path = os.path.join(model_dir, "iot_malware_model_transformer.pth")

    print(f"\n[Step 1] 正在檢查數據路徑...")
    print(f"   預期路徑 X: {x_path}")
    print(f"   預期路徑 y: {y_path}")

    # --- 3. 載入資料 ---
    x_train_path = os.path.join(data_dir, "X_train.npy")
    y_train_path = os.path.join(data_dir, "y_train.npy")
    x_val_path = os.path.join(data_dir, "X_val.npy")
    y_val_path = os.path.join(data_dir, "y_val.npy")
    x_test_path = os.path.join(data_dir, "X_test.npy")
    y_test_path = os.path.join(data_dir, "y_test.npy")

    if not all(os.path.exists(p) for p in [x_train_path, y_train_path, x_val_path, y_val_path, x_test_path, y_test_path]):
        print(f"\n[ERROR] 錯誤: 找不到預切分的訓練/驗證/測試 .npy 檔案！")
        print(f"[TIP] 請先執行 'python 1_data_prep.py' 來產生並切分數據。")
        exit()

    try:
        X_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        X_val = np.load(x_val_path)
        y_val = np.load(y_val_path)
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        print(f"[SUCCESS] 資料載入成功!")
        print(f"   X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"   X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        print(f"   X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    except Exception as e:
        print(f"[ERROR] 讀取錯誤: {e}")
        exit()

    # 轉 Tensor
    X_train_tensor = torch.from_numpy(X_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).long().to(device)
    X_val_tensor = torch.from_numpy(X_val).to(device)
    y_val_tensor = torch.from_numpy(y_val).long().to(device)
    X_test_tensor = torch.from_numpy(X_test).to(device)
    y_test_tensor = torch.from_numpy(y_test).long().to(device)

    # 建立 DataLoader
    BATCH_SIZE = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

    print(f"\n[Step 2] 開始訓練 Transformer (Epochs: {EPOCHS})...")
    print("-" * 50)

    # --- 5. 訓練迴圈 ---
    start_time = time.time()
    best_val_acc = 0.0

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
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # --- 驗證集評估 ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"   [Improve] 偵測到驗證集表現提升，已更新最佳模型儲存！ (Best Val Acc: {best_val_acc:.2f}%)")

    training_time = time.time() - start_time
    print("-" * 50)
    print(f"Transformer 訓練完成! 總耗時: {training_time:.2f} 秒")
    print(f"[Best] 最佳驗證集準確率 (Best Val Acc): {best_val_acc:.2f}%")

    # --- 6. 測試 ---
    print("\n[Step 3] 載入最佳模型評估測試集...")
    best_model = MalwareDetectorTransformer(INPUT_SIZE, D_MODEL, NUM_CLASSES).to(device)
    best_model.load_state_dict(torch.load(model_save_path))
    best_model.eval()
    with torch.no_grad():
        outputs = best_model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        total = y_test_tensor.size(0)
        correct = (predicted == y_test_tensor).sum().item()
        test_acc = 100 * correct / total
        print(f"[TEST] Transformer 最佳模型測試集準確率: {test_acc:.2f}%")
