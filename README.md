# 基於深度學習之 IoT 加密流量惡意行為偵測系統  
**IoT Encrypted Traffic Malicious Behavior Detection Using Deep Learning**

## 📌 Project Overview（專案簡介）

本研究旨在解決物聯網（IoT）環境中 **加密流量（Encrypted Traffic）** 的資安檢測問題。  
透過深度學習技術 **Long Short-Term Memory (LSTM)**，在 **不解密封包內容（Payload）** 的前提下，  
僅利用流量的 **時序訊號特徵**（如封包大小序列、封包到達時間間隔），  
即可準確識別多種惡意攻擊行為（如 **DDoS、Mirai Botnet**）。

本專案涵蓋完整流程，包括：

- 封包資料前處理（PCAP → 特徵序列）
- 深度學習模型訓練（LSTM）
- 效能評估（Confusion Matrix、Classification Report）
- Web 系統展示（Streamlit）

---

## ⚙️ System Requirements（系統需求）

- **Python**: 3.9 或以上  
- **建議環境**: 支援 GPU（CUDA）

### 核心套件
- `torch`（建議安裝 GPU 版本）
- `scapy`（封包處理）
- `numpy`, `pandas`（資料處理）
- `matplotlib`, `seaborn`（繪圖）
- `scikit-learn`（評估指標）
- `streamlit`（Web 介面展示）

---

## 📦 Installation（安裝方式）

### 建立與啟動虛擬環境 (Virtual Environment)
建議使用虛擬環境以避免套件衝突。

```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境 (Windows)
.\venv\Scripts\activate

# 啟動虛擬環境 (Mac/Linux)
source venv/bin/activate
```

### 安裝 PyTorch（請依照您的 CUDA 版本調整）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 安裝其他相依套件
```bash
pip install scapy matplotlib pandas seaborn scikit-learn streamlit
```

---

### 📁 Project Structure（專案結構）
建議維持以下目錄結構（程式碼已內建自動路徑定位）：
```bash
My_Thesis_Project/
├── data/                  # [資料區] 原始 PCAP 與處理後資料
│   ├── BenignTraffic.pcap
│   ├── DDoS-ICMP_Flood.pcap
│   ├── X_data.npy         # 自動產生
│   └── y_data.npy         # 自動產生
├── model/                 # [模型區] 訓練完成的模型權重
│   └── iot_malware_model.pth
├── src/                   # [程式碼區]
│   ├── 0_plot_figures.py  # 論文用高品質訊號圖繪製
│   ├── 1_data_prep.py     # PCAP → NPY 特徵工程
│   ├── 2_train.py         # LSTM 模型訓練
│   ├── 3_evaluate.py      # 模型效能評估
│   └── 4_demo.py          # Streamlit Web 展示系統
└── README.md
```

---

### 🔄 Workflow（執行流程）
請依照以下順序執行程式：

#### Step 0：繪製論文素材（Optional）
讀取原始 PCAP 檔案，繪製良性與惡意流量的 時序訊號對比圖，
適用於論文第三章（高解析度輸出）。
```bash
python src/0_plot_figures.py
```

📤 產出：
- `figure_traffic_signal.png`

---

#### Step 1：資料前處理（Data Preparation）
從 data/ 讀取 PCAP 檔案，提取時序特徵並轉換為 Numpy 格式，
供深度學習模型訓練使用。
```bash
python src/1_data_prep.py
```

📤 產出：
- `data/X_data.npy`
- `data/y_data.npy`

---

#### Step 2：模型訓練（Training）
使用 LSTM 模型進行訓練，程式會自動偵測並啟用 GPU 加速。
```bash
python src/2_train.py
```

📤 產出：
- `model/iot_malware_model.pth`
- 模型準確率約 98%+

---

#### Step 3：效能評估（Evaluation）
載入訓練完成的模型，對測試集進行推論並輸出評估結果。
```bash
python src/3_evaluate.py
```
📤 產出：
- `confusion_matrix.png`
- Classification Report（Console 輸出）

---

#### Step 4：系統展示（Demo）
啟動 Streamlit Web 介面，進行即時惡意流量偵測展示。
```bash
streamlit run src/4_demo.py
```
🔍 功能特色：

- 支援 PCAP 檔案上傳
- 支援 本機路徑讀取
- 即時顯示 惡意風險指數
- 視覺化流量訊號圖

---

### ⚠️ 注意事項（Notes）
若出現 FileNotFoundError，請先確認：

data/ 內的檔案名稱是否與程式碼設定一致

src/ 內所有程式皆使用 相對路徑定位

會自動尋找上一層的 data/ 與 model/

請勿隨意更動資料夾層級

---

### 📖 License

本專案僅供 學術研究與論文使用，請勿用於未授權之商業行為。

---

### 👤 Author
SHI-LIANG Deng
Master Thesis Project
Department of Information Engineering
