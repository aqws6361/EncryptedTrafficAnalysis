import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os
import platform

# --- 自動設定中文字體 (解決 Matplotlib 中文亂碼問題) ---
def set_chinese_font():
    system_name = platform.system()
    if system_name == "Windows":
        # Windows 使用微軟正黑體
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    elif system_name == "Darwin":
        # Mac 使用黑體
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        # Linux (Colab/Ubuntu) 嘗試常見中文字體
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Droid Sans Fallback']
    
    # 解決負號 '-' 顯示為方塊的問題
    plt.rcParams['axes.unicode_minus'] = False

# 呼叫設定函式
set_chinese_font()

# --- 1. 定義模型 (需與訓練時一致) ---
class MalwareDetectorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MalwareDetectorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1, :, :]
        out = self.fc(out)
        return out

def evaluate_performance():
    # --- 關鍵修正：自動定位路徑 ---
    # 1. 取得目前這支程式 (3_evaluate.py) 所在的資料夾路徑 (例如 .../src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 取得專案根目錄 (src 的上一層)
    project_root = os.path.dirname(current_dir)
    
    # 3. 組合出正確的檔案路徑
    # 假設您的結構是:
    # root/
    #   ├── data/
    #   │   ├── X_data.npy
    #   │   └── y_data.npy
    #   ├── model/ (或 models)
    #   │   └── iot_malware_model.pth
    #   └── src/
    #       └── 3_evaluate.py
    
    x_path = os.path.join(project_root, "data", "X_data.npy")
    y_path = os.path.join(project_root, "data", "y_data.npy")
    model_path = os.path.join(project_root, "model", "iot_malware_model.pth") 

    print(f"DEBUG: 預期資料路徑: {x_path}")
    print(f"DEBUG: 預期模型路徑: {model_path}")

    # 設定參數
    INPUT_SIZE = 2
    HIDDEN_SIZE = 64
    NUM_CLASSES = 2
    
    # 檢查檔案
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"❌ 找不到數據檔案，請確認檔案是否在 {x_path}")
        return

    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案，請確認檔案是否在 {model_path}")
        return

    # --- 2. 載入資料 ---
    print("正在載入測試數據...")
    # 修正：使用完整路徑載入
    X = np.load(x_path)
    y = np.load(y_path)
    
    # 切分測試集 (跟訓練時一樣 Random State 才能確保是同一份測試集)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 轉 Tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test_tensor = torch.from_numpy(X_test).to(device)
    y_test_tensor = torch.from_numpy(y_test).long().to(device)

    # --- 3. 載入模型 ---
    print(f"正在載入模型...")
    model = MalwareDetectorLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
    # 修正：使用完整路徑載入
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 4. 進行預測 ---
    print("正在進行推論...")
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    # 轉回 CPU Numpy
    y_true = y_test_tensor.cpu().numpy()
    y_pred = predicted.cpu().numpy()

    # --- 5. 產生報表 ---
    print("\n" + "="*40)
    print("📊 分類詳細報表 (Classification Report)")
    print("="*40)
    
    target_names = ['Benign (良性)', 'Attack (惡意)']
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)

    # --- 6. 繪製混淆矩陣 ---
    print("正在繪製混淆矩陣...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names)
    plt.xlabel('Predicted Label (預測)')
    plt.ylabel('True Label (真實)')
    plt.title('Confusion Matrix - IoT Malware Detection')
    
    # 存檔 (存到專案根目錄，比較好找)
    save_path = os.path.join(project_root, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"✅ 混淆矩陣已儲存為: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_performance()