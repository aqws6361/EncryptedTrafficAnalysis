from scapy.all import rdpcap, PcapReader, IP
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def extract_signal_from_pcap(pcap_path, max_packets=1000):
    """
    從 Pcap 檔中提取「封包大小序列」與「到達時間序列」
    """
    print(f"正在讀取檔案: {pcap_path} ... (僅讀取前 {max_packets} 個封包)")

    packet_sizes = []
    arrival_times = []

    # 使用 PcapReader 逐行讀取，避免記憶體爆炸
    count = 0
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

    # 將時間正規化 (從 0 開始)
    start_time = arrival_times[0]
    relative_times = [t - start_time for t in arrival_times]

    return relative_times, packet_sizes


def plot_traffic_signal(benign_path, attack_path):
    """
    畫出 良性 vs 惡意 的波形對比圖
    """
    # 1. 提取訊號
    # 註：這裡假設你已經下載了檔案，請修改成你電腦上的真實路徑
    # 如果還沒下載，這個程式會報錯 FileNotFoundError
    try:
        b_times, b_sizes = extract_signal_from_pcap(benign_path, max_packets=200)
        a_times, a_sizes = extract_signal_from_pcap(attack_path, max_packets=200)
    except FileNotFoundError as e:
        print(f"錯誤: 找不到檔案 - {e.filename}")
        print("請確認你已經從 CIC 官網下載了 .pcap 檔並修改程式碼中的路徑。")
        return

    # 2. 畫圖
    plt.figure(figsize=(15, 6))

    # 子圖 1: 良性流量
    plt.subplot(1, 2, 1)
    plt.plot(b_times, b_sizes, color='green', marker='o', linestyle='-', linewidth=1, markersize=3)
    plt.title("Normal IoT Traffic (Signal)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Packet Size (bytes)")
    plt.grid(True, alpha=0.3)

    # 子圖 2: 攻擊流量
    plt.subplot(1, 2, 2)
    plt.plot(a_times, a_sizes, color='red', marker='x', linestyle='-', linewidth=1, markersize=3)
    plt.title("Malicious Attack Traffic (Signal)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Packet Size (bytes)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("圖表繪製完成！這就是你要給教授看的「訊號」。")


if __name__ == "__main__":
    # --- 設定你的檔案路徑 ---
    # 請將這裡改成你實際下載的路徑
    # 範例：'./CIC_IoT_2023/Benign/Benign-01.pcap'
    path_to_benign = 'Benign_sample.pcap'
    path_to_attack = 'Attack_sample.pcap'

    # 如果你還沒有檔案，可以先傳入空字串或隨便一個檔案測試程式邏輯(會報錯但那是正常的)
    plot_traffic_signal(path_to_benign, path_to_attack)