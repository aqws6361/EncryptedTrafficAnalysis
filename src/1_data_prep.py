import numpy as np
from scapy.all import PcapReader, IP
import os

# --- 設定參數 ---
SEQUENCE_LENGTH = 20  # 視窗大小
MAX_SAMPLES_PER_FILE = 5000  # 每個檔案最多抓幾筆樣本

def process_pcap(file_path, label, seq_len=20):
    """
    讀取一個 pcap 檔，將其切分為多個序列樣本
    Output: 
      X: shape (n_samples, seq_len, 2) -> 特徵: [Packet Size, Inter-Arrival Time]
      y: shape (n_samples,) -> 標籤
    """
    print(f"正在處理: {os.path.basename(file_path)} ...")
    
    if not os.path.exists(file_path):
        print(f"❌ 錯誤: 找不到檔案 {file_path}")
        return [], []

    packet_sizes = []
    arrival_times = []
    
    # 1. 讀取封包特徵
    count = 0
    # 讀取上限設定為樣本數的 2 倍左右，確保有足夠封包可切分
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
    iat = [0.0]
    for i in range(1, len(arrival_times)):
        diff = arrival_times[i] - arrival_times[i-1]
        iat.append(diff)

    # 3. 切分為序列
    X_data = []
    y_data = []
    
    num_sequences = len(packet_sizes) // seq_len
    
    for i in range(num_sequences):
        start_idx = i * seq_len
        end_idx = start_idx + seq_len
        
        # 提取這一段的特徵
        seq_sizes = packet_sizes[start_idx:end_idx]
        seq_iat = iat[start_idx:end_idx]
        
        # 組合特徵 & 正規化
        features = []
        for s, t in zip(seq_sizes, seq_iat):
            features.append([s / 1500.0, t]) 
            
        X_data.append(features)
        y_data.append(label)
        
        if len(X_data) >= MAX_SAMPLES_PER_FILE:
            break
            
    return X_data, y_data

if __name__ == "__main__":
    # --- 自動定位路徑 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"建立資料夾: {data_dir}")

    # --- 定義檔案來源 ---
    path_benign = os.path.join(data_dir, "BenignTraffic.pcap")
    
    # 在這裡加入所有您想訓練的攻擊類型
    # 記得把對應的 .pcap 檔案放入 data 資料夾
    attack_files = [
        os.path.join(data_dir, "DDoS-ICMP_Flood.pcap"), 
        os.path.join(data_dir, "DDoS-SYN_Flood1.pcap"),       
        os.path.join(data_dir, "Mirai-udpplain1.pcap"), # 您可以隨時取消註解並加入新檔案
    ]

    all_X = []
    all_y = []

    # 1. 處理良性 (Label = 0)
    print(f"--- 處理良性流量 ---")
    X_benign, y_benign = process_pcap(path_benign, label=0, seq_len=SEQUENCE_LENGTH)
    all_X.extend(X_benign)
    all_y.extend(y_benign)

    # 2. 處理多種惡意流量 (Label = 1)
    print(f"--- 處理惡意流量 ---")
    for attack_path in attack_files:
        if os.path.exists(attack_path):
            X_att, y_att = process_pcap(attack_path, label=1, seq_len=SEQUENCE_LENGTH)
            all_X.extend(X_att)
            all_y.extend(y_att)
        else:
            print(f"⚠️ 跳過 (找不到檔案): {os.path.basename(attack_path)}")

    # --- 3. 儲存 ---
    if len(all_X) > 0:
        X_array = np.array(all_X, dtype=np.float32)
        y_array = np.array(all_y, dtype=np.int32)

        print("-" * 30)
        print(f"資料處理完成！")
        print(f"特徵矩陣 X shape: {X_array.shape}") # (樣本數, 20, 2)
        print(f"標籤矩陣 y shape: {y_array.shape}")
        
        # 檢查資料平衡狀況
        n_benign = np.sum(y_array == 0)
        n_attack = np.sum(y_array == 1)
        print(f"  - 良性樣本數: {n_benign}")
        print(f"  - 惡意樣本數: {n_attack}")
        
        save_x_path = os.path.join(data_dir, "X_data.npy")
        save_y_path = os.path.join(data_dir, "y_data.npy")
        
        np.save(save_x_path, X_array)
        np.save(save_y_path, y_array)
        
        print(f"\n✅ 已儲存至 data 資料夾，下一步請執行 2_train.py")
    else:
        print("❌ 沒有產生任何數據，請檢查 pcap 檔案路徑。")