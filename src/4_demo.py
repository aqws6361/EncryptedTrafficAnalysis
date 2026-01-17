import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from scapy.all import PcapReader, IP
import matplotlib.pyplot as plt
import os
import tempfile
import platform
import pandas as pd

# --- 解決 Matplotlib 中文顯示問題 ---
def set_chinese_font():
    system_name = platform.system()
    if system_name == "Windows":
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    elif system_name == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

# --- 1. 定義模型架構 ---

# Transformer Components
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

# LSTM Components
class MalwareDetectorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MalwareDetectorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0)) 
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# --- 2. 載入模型函式 ---
# @st.cache_resource 
def load_models():
    INPUT_SIZE = 2
    NUM_CLASSES = 2
    
    # Transformer Hyperparams
    D_MODEL = 64
    
    # LSTM Hyperparams
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Models
    model_t = MalwareDetectorTransformer(INPUT_SIZE, D_MODEL, NUM_CLASSES).to(device)
    model_l = MalwareDetectorLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.dirname(current_dir)
    
    # Define paths
    path_t = os.path.join(project_root, "model", "iot_malware_model_transformer.pth")
    path_l = os.path.join(project_root, "model", "iot_malware_model_lstm.pth")
    
    models_loaded = {}
    
    # Load Transformer
    if os.path.exists(path_t):
        try:
            model_t.load_state_dict(torch.load(path_t, map_location=device))
            model_t.eval()
            models_loaded['Transformer'] = model_t
        except Exception as e:
            st.error(f"Transformer 模型載入失敗: {e}")
    
    # Load LSTM
    if os.path.exists(path_l):
        try:
            model_l.load_state_dict(torch.load(path_l, map_location=device))
            model_l.eval()
            models_loaded['LSTM'] = model_l
        except Exception as e:
            st.error(f"LSTM 模型載入失敗: {e}")
            
    return models_loaded, device

# --- 3. 封包處理函式 ---
def preprocess_pcap(pcap_path, seq_len=50, max_packets=2000):
    packet_sizes = []
    arrival_times = []
    
    try:
        with PcapReader(pcap_path) as packets:
            for i, pkt in enumerate(packets):
                if i >= max_packets: break 
                if IP in pkt:
                    packet_sizes.append(len(pkt))
                    arrival_times.append(float(pkt.time))
    except Exception as e:
        st.error(f"解析 PCAP 失敗: {e}")
        return None

    if len(packet_sizes) < seq_len:
        st.warning(f"封包數量不足 (至少需要 {seq_len} 個)，無法進行分析")
        return None

    iat = [0.0]
    for i in range(1, len(arrival_times)):
        iat.append(arrival_times[i] - arrival_times[i-1])

    X_data = []
    num_sequences = len(packet_sizes) // seq_len
    
    for i in range(num_sequences):
        start = i * seq_len
        end = start + seq_len
        
        seq_s = packet_sizes[start:end]
        seq_t = iat[start:end]
        
        features = []
        for s, t in zip(seq_s, seq_t):
            features.append([s / 1500.0, t])
            
        X_data.append(features)
        
    return np.array(X_data, dtype=np.float32), packet_sizes, arrival_times

# --- 4. Streamlit UI 主程式 ---
st.set_page_config(page_title="IoT 加密流量偵測系統 - 模型比較", page_icon="🛡️", layout="wide")

st.title("🛡️ IoT Encrypted Traffic Detection System")
st.markdown("### Deep Learning Model Comparison: Transformer vs. LSTM")
st.markdown("---")

# 側邊欄
with st.sidebar:
    st.header("System Status")
    
    if st.button("🔄 重新載入模型"):
        st.cache_resource.clear()
        
    models, device = load_models()
    
    st.markdown("#### 模型狀態")
    if 'Transformer' in models:
        st.success("✅ Transformer: Ready")
    else:
        st.error("❌ Transformer: Not Found")
        
    if 'LSTM' in models:
        st.success("✅ LSTM: Ready")
    else:
        st.error("❌ LSTM: Not Found")
        
    st.caption(f"運算裝置: {device}")
    
    st.markdown("---")
    st.header("Settings")
    max_analyze_packets = st.slider("最大分析封包數", 1000, 50000, 5000, 1000)

# 輸入模式
input_method = st.radio("請選擇資料來源：", ("上傳檔案 (.pcap)", "輸入本機路徑 (Local Path)"), horizontal=True)

target_path = None
temp_file_obj = None

if input_method == "上傳檔案 (.pcap)":
    uploaded_file = st.file_uploader("請上傳 PCAP 封包檔", type=["pcap", "pcapng"])
    if uploaded_file:
        temp_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".pcap")
        temp_file_obj.write(uploaded_file.read())
        temp_file_obj.close() 
        target_path = temp_file_obj.name
        st.success(f"已接收檔案: {uploaded_file.name}")

else: 
    local_path = st.text_input("請輸入檔案完整路徑", placeholder=r"例如: C:\Users\Admin\Desktop\碩士論文\testData\DDoS-PSHACK_Flood10.pcap")
    local_path = local_path.strip('"').strip("'")
    if local_path:
        if os.path.exists(local_path):
            target_path = local_path
            st.success(f"已鎖定檔案: {os.path.basename(local_path)}")
        else:
            st.error("❌ 找不到檔案，請確認路徑是否正確")

# 開始分析
if target_path and models:
    if st.button("🚀 開始分析比較", type="primary"):
        with st.spinner(f"正在分析前 {max_analyze_packets} 個封包特徵..."):
            processed_data = preprocess_pcap(target_path, max_packets=max_analyze_packets)
            
            if processed_data:
                X_input, raw_sizes, raw_times = processed_data
                X_tensor = torch.from_numpy(X_input).to(device)
                
                results = {}
                
                # --- Run Inference for All Available Models ---
                for name, model in models.items():
                    with torch.no_grad():
                        outputs = model(X_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs.data, 1)
                        
                    preds = predicted.cpu().numpy()
                    probs_np = probs.cpu().numpy()
                    
                    malicious_count = np.sum(preds == 1)
                    total_count = len(preds)
                    malicious_rate = malicious_count / total_count if total_count > 0 else 0
                    
                    results[name] = {
                        'malicious_count': malicious_count,
                        'total': total_count,
                        'rate': malicious_rate,
                        'preds': preds,
                        'probs': probs_np[:, 1] # Probability of being malicious
                    }
                
                # --- Visualizing Results ---
                st.markdown("### 📊 檢測結果比較 (Detection Results Comparison)")
                
                # Metric Columns
                cols = st.columns(len(results))
                for idx, (name, res) in enumerate(results.items()):
                    with cols[idx]:
                        st.subheader(f"{name} Model")
                        st.metric("惡意特徵檢出", f"{res['malicious_count']} / {res['total']}")
                        st.metric("惡意風險指數", f"{res['rate']*100:.1f}%")
                        
                        if res['rate'] > 0.5:
                            st.error(f"⚠️ 判定: 惡意流量")
                        else:
                            st.success(f"✅ 判定: 正常流量")

                st.markdown("---")
                
                # --- Comparison Chart (Agreement/Disagreement) ---
                if 'Transformer' in results and 'LSTM' in results:
                    st.markdown("### 🔍 模型一致性分析 (Model Consensus Analysis)")
                    
                    preds_t = results['Transformer']['preds']
                    preds_l = results['LSTM']['preds']
                    
                    agreement = np.sum(preds_t == preds_l)
                    disagreement = np.sum(preds_t != preds_l)
                    total = len(preds_t)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write(f"**總樣本數**: {total}")
                        st.write(f"**一致預測**: {agreement} ({agreement/total*100:.1f}%)")
                        st.write(f"**不一致預測**: {disagreement} ({disagreement/total*100:.1f}%)")
                    
                    with col2:
                        # Bar chart for probabilities
                        st.write("#### 惡意機率分佈比較 (Malicious Probability Distribution)")
                        chart_data = pd.DataFrame({
                            'Transformer': results['Transformer']['probs'],
                            'LSTM': results['LSTM']['probs']
                        })
                        st.line_chart(chart_data)

                st.markdown("---")
                st.markdown("### 📈 流量訊號視覺化")
                
                fig, ax = plt.subplots(figsize=(15, 4))
                start_t = raw_times[0]
                plot_times = [t - start_t for t in raw_times]
                
                # Use primary model for color decision (Transformer if available, else first one)
                primary_res = results.get('Transformer', list(results.values())[0])
                color = 'red' if primary_res['rate'] > 0.5 else 'green'
                
                ax.plot(plot_times, raw_sizes, color=color, alpha=0.7, linewidth=1)
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Packet Size (bytes)")
                ax.set_title(f"Packet Size Sequence")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
        if input_method == "上傳檔案 (.pcap)" and temp_file_obj:
            try:
                os.unlink(target_path)
            except:
                pass
