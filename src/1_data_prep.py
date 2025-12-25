import numpy as np
from scapy.all import PcapReader, IP
import os

# --- 設定參數 ---
SEQUENCE_LENGTH = 20  # 每一個樣本看 20 個封包 (這就是我們的"視窗"大小)
MAX_SAMPLES_PER_FILE = 5000  # 每個檔案最多抓幾筆樣本 (避免單一檔案太大跑太久)

def process_pcap(file_path, label, seq_len=20):
    """
    讀取一個 pcap 檔，將其切分為多個序列樣本
    Output: 
      X: shape (n_samples, seq_len, 2) -> 特徵: [Packet Size, Inter-Arrival Time]
      y: shape (n_samples,) -> 標籤
    """
    print(f"正在處理: {file_path} ...")
    
    if not os.path.exists(file_path):
        print(f"錯誤: 找不到檔案 {file_path}")
        return [], []

    packet_sizes = []
    arrival_times = []
    
    # 1. 讀取封包特徵
    count = 0
    # 為了節省時間，我們讀取足夠的量就停 (例如: MAX_SAMPLES * seq_len * 1.5)
    limit = MAX_SAMPLES_PER_FILE * seq_len * 2 
    
    try:
        with PcapReader(file_path) as packets:
            for pkt in packets:
                if count >= limit:
                    break
                
                if IP in pkt:
                    packet_sizes.append(len(pkt))
                    arrival_times.append(float(pkt.time))
                    count += 1
    except Exception as e:
        print(f"讀取錯誤: {e}")
        return [], []

    print(f"  - 提取了 {len(packet_sizes)} 個封包，開始切分序列...")

    # 2. 轉換為時間差 (Inter-Arrival Time, IAT)
    # IAT[i] = Time[i] - Time[i-1]
    # 第一個封包的 IAT 設為 0
    iat = [0.0]
    for i in range(1, len(arrival_times)):
        diff = arrival_times[i] - arrival_times[i-1]
        iat.append(diff)

    # 3. 切分為序列 (Sliding Window 或 Non-overlapping)
    # 這裡使用簡單的 Non-overlapping (切完這段接下一段)
    X_data = []
    y_data = []
    
    num_sequences = len(packet_sizes) // seq_len
    
    for i in range(num_sequences):
        start_idx = i * seq_len
        end_idx = start_idx + seq_len
        
        # 提取這一段的特徵
        seq_sizes = packet_sizes[start_idx:end_idx]
        seq_iat = iat[start_idx:end_idx]
        
        # 組合特徵: [[size1, iat1], [size2, iat2], ...]
        # Normalize: 簡單做個正規化，讓數值不要太大
        # 封包大小通常 0-1500，除以 1500
        # 時間差通常很小，暫時不除，或根據觀察調整
        features = []
        for s, t in zip(seq_sizes, seq_iat):
            features.append([s / 1500.0, t]) 
            
        X_data.append(features)
        y_data.append(label)
        
        if len(X_data) >= MAX_SAMPLES_PER_FILE:
            break
            
    return X_data, y_data

if __name__ == "__main__":
    # 取得目前這支程式 (src/1_data_prep.py) 所在的資料夾
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 取得專案根目錄 (src 的上一層)
    project_root = os.path.dirname(current_dir)
    # 定義 data 資料夾路徑
    data_dir = os.path.join(project_root, "data")
    
    # 如果 data 資料夾不存在，自動建立它
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"建立資料夾: {data_dir}")

    # --- 1. 設定來源檔案路徑 ---
    path_benign = os.path.join(data_dir, "BenignTraffic.pcap")
    path_attack = os.path.join(data_dir, "DDoS-ICMP_Flood.pcap")

    # --- 2. 處理資料 ---
    all_X = []
    all_y = []

    # 處理良性 (Label = 0)
    X_benign, y_benign = process_pcap(path_benign, label=0, seq_len=SEQUENCE_LENGTH)
    all_X.extend(X_benign)
    all_y.extend(y_benign)

    # 處理惡意 (Label = 1)
    X_attack, y_attack = process_pcap(path_attack, label=1, seq_len=SEQUENCE_LENGTH)
    all_X.extend(X_attack)
    all_y.extend(y_attack)

    # --- 3. 轉換為 Numpy Array 並儲存到 data 資料夾 ---
    if len(all_X) > 0:
        X_array = np.array(all_X, dtype=np.float32)
        y_array = np.array(all_y, dtype=np.int32)

        print("-" * 30)
        print(f"資料處理完成！")
        print(f"特徵矩陣 X shape: {X_array.shape}")
        print(f"標籤矩陣 y shape: {y_array.shape}")
        print(f"  - 樣本總數: {X_array.shape[0]}")
        print(f"  - 序列長度: {X_array.shape[1]}")
        print(f"  - 特徵數量: {X_array.shape[2]} (Size, IAT)")
        
        # 修正：儲存到 data 資料夾
        save_x_path = os.path.join(data_dir, "X_data.npy")
        save_y_path = os.path.join(data_dir, "y_data.npy")
        
        np.save(save_x_path, X_array)
        np.save(save_y_path, y_array)
        
        print(f"\n已儲存為:\n -> {save_x_path}\n -> {save_y_path}")
        print("下一步：使用這些檔案來訓練 AI 模型。")
    else:
        print("沒有產生任何數據，請檢查檔案路徑是否正確。")