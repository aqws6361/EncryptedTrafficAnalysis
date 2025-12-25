from scapy.all import rdpcap, PcapReader, IP
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def extract_signal_from_pcap(pcap_path, max_packets=1000):
    """
    從 Pcap 檔中提取「封包大小序列」與「到達時間序列」
    """
    # 檢查檔案是否存在，避免 scapy 報出看不懂的錯誤
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"找不到檔案: {pcap_path}")

    print(f"正在讀取檔案: {pcap_path} ... (僅讀取前 {max_packets} 個封包)")
    
    packet_sizes = []
    arrival_times = []
    
    # 使用 PcapReader 逐行讀取，避免記憶體爆炸
    count = 0
    try:
        with PcapReader(pcap_path) as packets:
            for pkt in packets:
                if count >= max_packets:
                    break
                    
                # 我們只關心 IP 層以上的封包 (濾掉單純的 Layer 2 雜訊)
                if IP in pkt:
                    # 特徵 1: 封包長度 (這就是波形的振幅)
                    packet_sizes.append(len(pkt))
                    
                    # 特徵 2: 到達時間 (這是時間軸)
                    arrival_times.append(float(pkt.time))
                    
                    count += 1
    except Scapy_Exception as e:
        print(f"Scapy 讀取錯誤: {e}")
        return [], []
    
    # 將時間正規化 (從 0 開始)
    start_time = arrival_times[0] if arrival_times else 0
    relative_times = [t - start_time for t in arrival_times]
    
    return relative_times, packet_sizes

def plot_traffic_signal(benign_path, attack_path):
    """
    畫出 良性 vs 惡意 的波形對比圖
    """
    # 1. 提取訊號
    try:
        # 這裡會去呼叫上面的函式，傳入你設定的路徑
        b_times, b_sizes = extract_signal_from_pcap(benign_path, max_packets=200)
        a_times, a_sizes = extract_signal_from_pcap(attack_path, max_packets=200)
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        print("請確認路徑正確，且檔案名稱沒有打錯 (包含副檔名 .pcap)。")
        return
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
        return

    # 2. 畫圖
    plt.figure(figsize=(15, 6))
    
    # 子圖 1: 良性流量
    plt.subplot(1, 2, 1)
    if b_times:
        plt.plot(b_times, b_sizes, color='green', marker='o', linestyle='-', linewidth=1, markersize=3)
        plt.title("Normal IoT Traffic (Signal)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Packet Size (bytes)")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No Data / File Error", ha='center', va='center')
    
    # 子圖 2: 攻擊流量
    plt.subplot(1, 2, 2)
    if a_times:
        plt.plot(a_times, a_sizes, color='red', marker='x', linestyle='-', linewidth=1, markersize=3)
        plt.title("Malicious Attack Traffic (Signal)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Packet Size (bytes)")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No Data / File Error", ha='center', va='center')
    
    plt.tight_layout()
    plt.show()
    
    print("圖表繪製完成！")

if __name__ == "__main__":
    # --- 設定檔案路徑 ---
    
    path_to_benign = r"C:\Users\Admin\Desktop\碩士論文\data\BenignTraffic.pcap"
    path_to_attack = r"C:\Users\Admin\Desktop\碩士論文\data\DDoS-ICMP_Flood.pcap"
    
    print(f"設定良性流量路徑: {path_to_benign}")
    print(f"設定攻擊流量路徑: {path_to_attack}")
    
    plot_traffic_signal(path_to_benign, path_to_attack)