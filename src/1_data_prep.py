import numpy as np
import os
import struct

# --- 設定參數 ---
SEQUENCE_LENGTH = 50  # 視窗大小 (Transformer)
MAX_SAMPLES_PER_FILE = 5000  # 每個檔案最多抓幾筆樣本

def process_pcap(file_path, label, seq_len=20):
    """
    讀取一個 pcap 檔，將其切分為多個序列樣本
    使用高效二進位讀取方式，跳過 Scapy 以應對數 GB 的大型 PCAP
    """
    print(f"正在處理: {os.path.basename(file_path)} ...")
    
    if not os.path.exists(file_path):
        print(f"[ERROR] 錯誤: 找不到檔案 {file_path}")
        return [], []

    packet_sizes = []
    arrival_times = []
    
    limit = MAX_SAMPLES_PER_FILE * seq_len * 2 
    
    try:
        with open(file_path, "rb") as f:
            global_header = f.read(24)
            if len(global_header) < 24:
                print("[ERROR] 錯誤: 檔案標頭無效")
                return [], []
                
            magic = global_header[:4]
            if magic == b'\xa1\xb2\xc3\xd4' or magic == b'\xa1\xb2\x3c\x4d':
                endian = '>'
            elif magic == b'\xd4\xc3\xb2\xa1' or magic == b'\x4d\x3c\xb2\xa1':
                endian = '<'
            else:
                print(f"[ERROR] 錯誤: 未知的 PCAP 魔術數字: {magic}")
                return [], []
                
            is_nano = (magic == b'\xa1\xb2\x3c\x4d' or magic == b'\x4d\x3c\xb2\xa1')
            
            count = 0
            while count < limit:
                hdr = f.read(16)
                if len(hdr) < 16:
                    break
                    
                ts_sec, ts_usec, caplen, origlen = struct.unpack(endian + "IIII", hdr)
                pkt_data = f.read(caplen)
                if len(pkt_data) < caplen:
                    break
                    
                if caplen >= 14:
                    eth_type = struct.unpack(">H", pkt_data[12:14])[0]
                    # 支援 VLAN tag
                    if eth_type == 0x8100 and caplen >= 18:
                        eth_type = struct.unpack(">H", pkt_data[16:18])[0]
                    if eth_type == 0x0800: # IPv4
                        packet_sizes.append(origlen)
                        t_val = ts_sec + (ts_usec / 1e9 if is_nano else ts_usec / 1e6)
                        arrival_times.append(t_val)
                        count += 1
                        
    except Exception as e:
        print(f"[ERROR] 讀取錯誤: {e}")
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

def extract_and_balance(benign_list, attack_list, label_suffix="", balance=True):
    all_X = []
    all_y = []
    
    print(f"\n--- 處理良性流量 ({label_suffix}) ---")
    for b_path in benign_list:
        if os.path.exists(b_path):
            X_benign, y_benign = process_pcap(b_path, label=0, seq_len=SEQUENCE_LENGTH)
            all_X.extend(X_benign)
            all_y.extend(y_benign)
        else:
            print(f"[SKIP] 跳過 (找不到檔案): {os.path.basename(b_path)}")
            
    print(f"\n--- 處理惡意流量 ({label_suffix}) ---")
    for a_path in attack_list:
        if os.path.exists(a_path):
            X_att, y_att = process_pcap(a_path, label=1, seq_len=SEQUENCE_LENGTH)
            all_X.extend(X_att)
            all_y.extend(y_att)
        else:
            print(f"[SKIP] 跳過 (找不到檔案): {os.path.basename(a_path)}")
            
    if len(all_X) == 0:
        return np.empty((0, SEQUENCE_LENGTH, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)
        
    X_array = np.array(all_X, dtype=np.float32)
    y_array = np.array(all_y, dtype=np.int32)
    
    n_benign = np.sum(y_array == 0)
    n_attack = np.sum(y_array == 1)
    print(f"[{label_suffix}] 原始樣本數 - 良性: {n_benign}, 惡意: {n_attack}")
    
    if balance and n_benign > 0 and n_attack > 0:
        min_samples = min(n_benign, n_attack)
        # 只要有一方是另一方的 1.5 倍以上就執行平衡
        if n_attack > n_benign * 1.5 or n_benign > n_attack * 1.5:
            print(f"[BALANCE] 偵測到 [{label_suffix}] 資料不平衡，執行欠採樣，目標數量: 各 {min_samples} 筆")
            idx_benign = np.where(y_array == 0)[0]
            idx_attack = np.where(y_array == 1)[0]
            
            np.random.seed(42)
            chosen_benign = np.random.choice(idx_benign, min_samples, replace=False)
            chosen_attack = np.random.choice(idx_attack, min_samples, replace=False)
            
            balanced_indices = np.concatenate([chosen_benign, chosen_attack])
            np.random.shuffle(balanced_indices)
            
            X_array = X_array[balanced_indices]
            y_array = y_array[balanced_indices]
            print(f"[{label_suffix}] 平衡後 X shape: {X_array.shape}")
            
    return X_array, y_array

if __name__ == "__main__":
    # --- 自動定位路徑 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")
    test_data_dir = os.path.join(project_root, "testData")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"建立資料夾: {data_dir}")

    # --- 1. 定義 Train/Val 資料來源 (data/) ---
    train_benign = [
        os.path.join(data_dir, "BenignTraffic.pcap"),
        os.path.join(data_dir, "BenignTraffic1.pcap"), 
        os.path.join(data_dir, "BenignTraffic2.pcap")
    ]
    train_attack = [
        os.path.join(data_dir, "DDoS-ICMP_Flood.pcap"), 
        os.path.join(data_dir, "DDoS-SYN_Flood1.pcap"),
        os.path.join(data_dir, "DDoS-PSHACK_Flood.pcap"), 
        os.path.join(data_dir, "DoS-TCP_Flood.pcap"), 
        os.path.join(data_dir, "Mirai-udpplain1.pcap")
    ]
    
    # --- 2. 定義 Test 資料來源 (testData/) ---
    test_benign = [
        os.path.join(test_data_dir, "BenignTraffic3.pcap")
    ]
    test_attack = [
        os.path.join(test_data_dir, "DDoS-ICMP_Flood16.pcap"),
        os.path.join(test_data_dir, "DDoS-ICMP_Flood16_encrypted.pcap"),
        os.path.join(test_data_dir, "DDoS-PSHACK_Flood10.pcap"),
        os.path.join(test_data_dir, "DoS-TCP_Flood8.pcap")
    ]
    
    # --- 3. 處理 Train/Val 數據 ---
    print("=== 正在生成 訓練與驗證 數據 ===")
    X_train_val, y_train_val = extract_and_balance(train_benign, train_attack, label_suffix="Train/Val", balance=True)
    
    if len(X_train_val) == 0:
        print("[ERROR] 無法從 data/ 資料夾生成訓練資料，請檢查 PCAP 檔案。")
        exit()
        
    # 切分為 Train (85%) 和 Val (15%)
    print(f"\n[Split] 正在將 Train/Val 數據執行分層切分 (85% / 15%)...")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
    )
    
    # --- 4. 處理 Test 數據 ---
    print("\n=== 正在生成 獨立測試 數據 (testData/) ===")
    X_test, y_test = extract_and_balance(test_benign, test_attack, label_suffix="Test", balance=True)
    
    if len(X_test) == 0:
        print("[WARN] 無法從 testData/ 生成獨立測試集，將使用 Train/Val 的一部分暫代...")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
        )
        
    # 列印最終統計
    print("\n[Stats] 各資料集最終劃分統計:")
    print(f"  - 訓練集 (Train) shape: {X_train.shape} | 良性: {np.sum(y_train==0)}, 惡意: {np.sum(y_train==1)}")
    print(f"  - 驗證集 (Val)   shape: {X_val.shape} | 良性: {np.sum(y_val==0)}, 惡意: {np.sum(y_val==1)}")
    print(f"  - 測試集 (Test)  shape: {X_test.shape} | 良性: {np.sum(y_test==0)}, 惡意: {np.sum(y_test==1)}")
    
    # 儲存舊有檔名維持相容性
    np.save(os.path.join(data_dir, "X_data.npy"), X_train_val)
    np.save(os.path.join(data_dir, "y_data.npy"), y_train_val)
    
    # 儲存切分後的資料
    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "X_val.npy"), X_val)
    np.save(os.path.join(data_dir, "y_val.npy"), y_val)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)
    
    print(f"\n[SUCCESS] 資料已成功儲存至 data/ 資料夾！")