import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from scapy.all import PcapReader, IP
import matplotlib.pyplot as plt
import os
import tempfile
import platform

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
# --- 2. 定義模型架構 (Transformer) ---
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

# --- 2. 載入模型函式 (Debug 版 - 移除快取以免鎖死錯誤) ---
# @st.cache_resource  <-- 先註解掉，避免快取住 "找不到檔案" 的狀態
def load_model():
    INPUT_SIZE = 2
    D_MODEL = 64
    NUM_CLASSES = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MalwareDetectorTransformer(INPUT_SIZE, D_MODEL, NUM_CLASSES).to(device)
    
    # --- 強化的路徑搜尋邏輯 ---
    current_dir = os.path.dirname(os.path.abspath(__file__)) # src/
    project_root = os.path.dirname(current_dir)              # root/
    
    # 定義所有可能的路徑 (依優先順序)
    possible_paths = [
        os.path.join(project_root, "model", "iot_malware_model.pth"),   # 標準結構: root/model/
        os.path.join(project_root, "models", "iot_malware_model.pth"),  # 易錯結構: root/models/
        os.path.join(current_dir, "iot_malware_model.pth"),             # 放在 src/ 裡
        "iot_malware_model.pth"                                         # 當前執行目錄
    ]
    
    target_model_path = None
    
    # 遍歷尋找
    for path in possible_paths:
        if os.path.exists(path):
            target_model_path = path
            break
    
    if target_model_path is None:
        # 如果都找不到，顯示詳細 Debug 資訊
        st.error("❌ **嚴重錯誤：找不到模型檔案**")
        st.warning(f"系統已嘗試在以下路徑尋找，但都失敗：")
        for p in possible_paths:
            st.code(p)
        st.info("💡 請確認 `iot_malware_model.pth` 確實存在於上述任一路徑中。")
        return None, None

    try:
        model.load_state_dict(torch.load(target_model_path, map_location=device))
        model.eval()
        return model, device
        
    except RuntimeError as e:
        if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
            st.error("❌ **模型架構不匹配 (Model Mismatch)**")
            st.warning("偵測到舊版的模型檔案！程式碼已更新為 Transformer 架構，但 `model/iot_malware_model.pth` 仍是舊的模型。")
            st.info("💡 **解決方法**：請執行 `python src/2_train.py` 重新訓練模型，以覆蓋舊的檔案。")
            return None, None
        else:
            st.error(f"❌ 模型載入發生未預期錯誤: {e}")
            return None, None
    except Exception as e:
        st.error(f"❌ 模型載入發生錯誤: {e}")
        return None, None

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
st.set_page_config(page_title="IoT 加密流量偵測系統", page_icon="🛡️", layout="wide")

st.title("🛡️ IoT Encrypted Traffic Detection System")
st.markdown("### 基於深度學習 (Transformer) 之惡意流量行為分析")
st.markdown("---")

# 側邊欄
with st.sidebar:
    st.header("System Status")
    
    # 加入一個重新整理按鈕
    if st.button("🔄 重新載入模型"):
        st.cache_resource.clear()
        
    model, device = load_model()
    
    if model:
        st.success(f"✅ AI 模型運作中")
        st.caption(f"運算裝置: {device}")
    else:
        st.error("❌ 模型未就緒")
    
    st.markdown("---")
    st.header("Settings")
    max_analyze_packets = st.slider("最大分析封包數", 1000, 50000, 5000, 1000)
    st.info("💡 提示：若檔案過大 (>200MB)，請使用本機路徑模式。")

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
if target_path and model:
    if st.button("🚀 開始分析", type="primary"):
        with st.spinner(f"正在分析前 {max_analyze_packets} 個封包特徵..."):
            processed_data = preprocess_pcap(target_path, max_packets=max_analyze_packets)
            
            if processed_data:
                X_input, raw_sizes, raw_times = processed_data
                
                X_tensor = torch.from_numpy(X_input).to(device)
                with torch.no_grad():
                    outputs = model(X_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                
                preds = predicted.cpu().numpy()
                malicious_count = np.sum(preds == 1)
                total_count = len(preds)
                malicious_rate = malicious_count / total_count if total_count > 0 else 0
                
                st.markdown("### 📊 檢測結果分析")
                col1, col2, col3 = st.columns(3)
                col1.metric("分析序列數", f"{total_count} 組")
                col2.metric("惡意特徵檢出", f"{malicious_count} 組", delta_color="inverse")
                col3.metric("惡意風險指數", f"{malicious_rate*100:.1f}%")

                if malicious_rate > 0.5:
                    st.error(f"⚠️ 警告：偵測到惡意攻擊流量！ (DDoS/Malware)")
                else:
                    st.success(f"✅ 安全：此為正常 IoT 流量")

                st.markdown("---")
                st.markdown("### 📈 流量訊號視覺化")
                
                fig, ax = plt.subplots(figsize=(15, 4))
                start_t = raw_times[0]
                plot_times = [t - start_t for t in raw_times]
                color = 'red' if malicious_rate > 0.5 else 'green'
                ax.plot(plot_times, raw_sizes, color=color, alpha=0.7, linewidth=1)
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Packet Size (bytes)")
                ax.set_title(f"Packet Size Sequence ({'Attack Pattern' if malicious_rate > 0.5 else 'Normal Pattern'})")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
        if input_method == "上傳檔案 (.pcap)" and temp_file_obj:
            try:
                os.unlink(target_path)
            except:
                pass