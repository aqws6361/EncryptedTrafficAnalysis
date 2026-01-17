import numpy as np
from scapy.all import PcapReader, IP
import os

# --- 設定參數 ---
SEQUENCE_LENGTH = 50  # 視窗大小 (配合您要做的 Transformer)
MAX_SAMPLES_PER_FILE = 5000  # 每個檔案最多抓幾筆樣本

def process_pcap(file_path, label, seq_len=20):
    """
    讀取一個 pcap 檔，將其切分為多個序列樣本
    """
    print(f"正在處理: {os.path.basename(file_path)} ...")
    
    if not os.path.exists(file_path):
        print(f"❌ 錯誤: 找不到檔案 {file_path}")
        return [], []

    packet_sizes = []
    arrival_times = []
    
    count = 0
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

    iat = [0.0]
    for i in range(1, len(arrival_times)):
        diff = arrival_times[i] - arrival_times[i-1]
        iat.append(diff)

    X_data = []
    y_data = []
    
    num_sequences = len(packet_sizes) // seq_len
    
    for i in range(num_sequences):
        start_idx = i * seq_len
        end_idx = start_idx + seq_len
        
        seq_sizes = packet_sizes[start_idx:end_idx]
        seq_iat = iat[start_idx:end_idx]
        
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
    
    # 1. 良性流量列表
    benign_files = [
        os.path.join(data_dir, "BenignTraffic.pcap"),
        os.path.join(data_dir, "BenignTraffic1.pcap"), 
        os.path.join(data_dir, "BenignTraffic2.pcap")
    ]
    
    # 2. 惡意流量列表
    attack_files = [
        os.path.join(data_dir, "DDoS-ICMP_Flood.pcap"), 
        os.path.join(data_dir, "DDoS-SYN_Flood1.pcap"),
        os.path.join(data_dir, "DDoS-PSHACK_Flood.pcap"), 
        os.path.join(data_dir, "DoS-TCP_Flood.pcap"), 
        os.path.join(data_dir, "Mirai-udpplain1.pcap"),
    ]

    all_X = []
    all_y = []

    # --- 處理資料邏輯 ---

    # 1. 處理多個良性流量 (Label = 0)
    print(f"--- 處理良性流量 ---")
    for b_path in benign_files:
        if os.path.exists(b_path):
            X_benign, y_benign = process_pcap(b_path, label=0, seq_len=SEQUENCE_LENGTH)
            all_X.extend(X_benign)
            all_y.extend(y_benign)
        else:
            print(f"⚠️ 跳過 (找不到檔案): {os.path.basename(b_path)}")

    # 2. 處理多個惡意流量 (Label = 1)
    print(f"\n--- 處理惡意流量 ---")
    for attack_path in attack_files:
        if os.path.exists(attack_path):
            X_att, y_att = process_pcap(attack_path, label=1, seq_len=SEQUENCE_LENGTH)
            all_X.extend(X_att)
            all_y.extend(y_att)
        else:
            print(f"⚠️ 跳過 (找不到檔案): {os.path.basename(attack_path)}")

    # --- 3. 儲存與平衡 ---
    if len(all_X) > 0:
        X_array = np.array(all_X, dtype=np.float32)
        y_array = np.array(all_y, dtype=np.int32)

        print("-" * 30)
        print(f"資料處理完成！")
        print(f"特徵矩陣 X shape: {X_array.shape}")
        print(f"標籤矩陣 y shape: {y_array.shape}")
        
        # 檢查資料平衡狀況
        n_benign = np.sum(y_array == 0)
        n_attack = np.sum(y_array == 1)
        print(f"  - 良性樣本數: {n_benign}")
        print(f"  - 惡意樣本數: {n_attack}")

        # --- 自動平衡機制 ---
        if n_benign > 0 and n_attack > 0:
            min_samples = min(n_benign, n_attack)
            if n_attack > n_benign * 1.5: # 只有當嚴重不平衡時才執行
                print(f"\n⚖️ 偵測到資料不平衡，正在執行欠採樣 (Undersampling)...")
                print(f"   目標數量: 各 {min_samples} 筆")
                
                idx_benign = np.where(y_array == 0)[0]
                idx_attack = np.where(y_array == 1)[0]
                
                np.random.seed(42)
                chosen_benign = np.random.choice(idx_benign, min_samples, replace=False)
                chosen_attack = np.random.choice(idx_attack, min_samples, replace=False)
                
                balanced_indices = np.concatenate([chosen_benign, chosen_attack])
                np.random.shuffle(balanced_indices)
                
                X_array = X_array[balanced_indices]
                y_array = y_array[balanced_indices]
                print(f"   平衡後 X shape: {X_array.shape}")
        
        save_x_path = os.path.join(data_dir, "X_data.npy")
        save_y_path = os.path.join(data_dir, "y_data.npy")
        
        np.save(save_x_path, X_array)
        np.save(save_y_path, y_array)
        
        print(f"\n✅ 已儲存至 data 資料夾，下一步請執行 2_train.py")
    else:
        print("❌ 沒有產生任何數據，請檢查 pcap 檔案路徑。")