import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import os
import platform

# --- 自動設定中文字體 ---
def set_chinese_font():
    system_name = platform.system()
    if system_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    elif system_name == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Droid Sans Fallback']
    plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

# --- 1. 定義模型 (需與訓練時一致) ---
# Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
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
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out

# LSTM
class MalwareDetectorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MalwareDetectorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(next(self.parameters()).device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(next(self.parameters()).device)
        out, _ = self.lstm(x, (h0, c0)) 
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

def evaluate_performance():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    x_path = os.path.join(project_root, "data", "X_data.npy")
    y_path = os.path.join(project_root, "data", "y_data.npy")
    
    # Model Paths
    model_path_transformer = os.path.join(project_root, "model", "iot_malware_model_transformer.pth") 
    model_path_lstm = os.path.join(project_root, "model", "iot_malware_model_lstm.pth")
    
    print(f"DEBUG: 預期資料路徑: {x_path}")

    # Check Data
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"❌ 找不到數據檔案")
        return

    # Load Data
    print("正在載入測試數據...")
    X = np.load(x_path)
    y = np.load(y_path)
    
    # Ensure same split as training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test_tensor = torch.from_numpy(X_test).to(device)
    y_test_tensor = torch.from_numpy(y_test).long().to(device)
    
    y_true = y_test_tensor.cpu().numpy()
    target_names = ['Benign (良性)', 'Attack (惡意)']
    
    models = []
    
    # --- Load Transformer ---
    if os.path.exists(model_path_transformer):
        print("載入 Transformer 模型...")
        model_t = MalwareDetectorTransformer(2, 64, 2).to(device)
        model_t.load_state_dict(torch.load(model_path_transformer, map_location=device))
        model_t.eval()
        models.append(("Transformer", model_t))
    else:
        print(f"⚠️ 找不到 Transformer 模型: {model_path_transformer}")

    # --- Load LSTM ---
    if os.path.exists(model_path_lstm):
        print("載入 LSTM 模型...")
        model_l = MalwareDetectorLSTM(2, 64, 2, 2).to(device)
        model_l.load_state_dict(torch.load(model_path_lstm, map_location=device))
        model_l.eval()
        models.append(("LSTM", model_l))
    else:
        print(f"⚠️ 找不到 LSTM 模型: {model_path_lstm}")
        
    if not models:
        print("❌ 沒有可用的模型進行評估")
        return

    # --- Evaluate Each Model ---
    for name, model in models:
        print(f"\n{'='*20} 評估模型: {name} {'='*20}")
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        y_pred = predicted.cpu().numpy()
        
        # Report
        print(f"📊 {name} 分類詳細報表:")
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {name}')
        
        save_path = os.path.join(project_root, f'confusion_matrix_{name.lower()}.png')
        plt.savefig(save_path)
        print(f"✅ {name} 混淆矩陣已儲存為: {save_path}")
        plt.close()

if __name__ == "__main__":
    evaluate_performance()