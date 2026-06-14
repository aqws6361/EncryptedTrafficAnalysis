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

# --- 2. 定義 LSTM 模型架構 ---
class MalwareDetectorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MalwareDetectorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True mean input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0)) 
        
        # Use the output of the last time step
        # out shape: (batch, seq_len, hidden_size)
        out = out[:, -1, :] 
        
        # Classifier
        out = self.fc(out)
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

    # 定義完整檔案路徑
    x_path = os.path.join(data_dir, "X_data.npy")
    y_path = os.path.join(data_dir, "y_data.npy")
    # [MODIFY] LSTM model save path
    model_save_path = os.path.join(model_dir, "iot_malware_model_lstm.pth")

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

    # --- 4. 初始化模型 (LSTM 設定) ---
    INPUT_SIZE = 2
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    NUM_CLASSES = 2
    LEARNING_RATE = 0.001
    EPOCHS = 15
    
    # 初始化 LSTM
    model = MalwareDetectorLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n[Step 2] 開始訓練 LSTM (Epochs: {EPOCHS})...")
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
    print(f"LSTM 訓練完成! 總耗時: {training_time:.2f} 秒")
    print(f"[Best] 最佳驗證集準確率 (Best Val Acc): {best_val_acc:.2f}%")

    # --- 6. 測試 ---
    print("\n[Step 3] 載入最佳模型評估測試集...")
    best_model = MalwareDetectorLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    best_model.load_state_dict(torch.load(model_save_path))
    best_model.eval()
    with torch.no_grad():
        outputs = best_model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        total = y_test_tensor.size(0)
        correct = (predicted == y_test_tensor).sum().item()
        test_acc = 100 * correct / total
        print(f"[TEST] LSTM 最佳模型測試集準確率: {test_acc:.2f}%")
