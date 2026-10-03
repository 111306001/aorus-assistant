# AORUS MASTER 16 AM6H AI Assistant

一個專為**資源有限的消費級筆電**設計的輕量化離線 RAG（Retrieval-Augmented Generation，檢索增強生成）問答系統。

本專案以 GIGABYTE AORUS MASTER 16 AM6H 系列產品規格為知識來源，透過自行實作的向量檢索取得相關規格，再交由本地端小型語言模型（SLM）生成精確回答。

系統的設計目標是在 **4GB 顯示記憶體 (VRAM)** 的嚴格硬體限制下，維持：
- 零高階框架依賴（不使用 LangChain、LlamaIndex）
- 繁體中文與英文混合問答
- 結構化規格的精準檢索與推論
- 串流輸出與可量化的效能測試

---

## 🛠️ 技術規格

| 項目 | 實作 |
|---|---|
| 程式語言 | Python 3.11+ |
| 套件管理 | `uv` |
| RAG 框架 | 純 Python 實作 (NumPy) |
| Embedding 模型 | `paraphrase-multilingual-MiniLM-L12-v2` |
| 語言模型 (LLM) | Qwen2.5-3B-Instruct |
| LLM 格式 | GGUF |
| 量化 (Quantization)| Q4_K_M (4-bit 權重量化) |
| 推論引擎 | `llama-cpp-python` |
| 向量檢索 | Semantic (Cosine) + Keyword + Domain 混合搜尋 |
| 內容長度 (Context)| 2048 tokens |
| GPU 卸載 | `-1`（全部卸載至 GPU） |

---

## 🧠 模型選擇與 4GB VRAM 限制評估

為了確保系統能在 4GB VRAM 的環境下運行，我們採用了 **SLM + Quantization + CPU/GPU 分工** 的策略：

1. **LLM 選擇 (Qwen2.5-3B-Instruct Q4_K_M)**：
   - 3B 模型原本約需 6GB 記憶體，透過 `Q4_K_M` (4-bit) 量化後，模型權重檔案大小縮減至約 **2.02 GB**。
   - 加上 2048 tokens 的 Context Window 運算保留區（KV Cache），LLM 推論時期的 VRAM 佔用可穩定控制在 **2.5 GB ~ 3.0 GB** 之間。
   - `llama-cpp-python` 設定 `n_gpu_layers=-1`，將推論完全卸載至 GPU，最大化生成速度而不爆顯存。

2. **Embedding 模型分離 (paraphrase-multilingual-MiniLM-L12-v2)**：
   - 系統需支援中英混合檢索。我們選擇小型多語系 Embedding 模型。
   - **關鍵策略**：Embedding 模型強制在 CPU 上執行（佔用極少 RAM），保留寶貴的 GPU VRAM 專供 LLM 生成使用。

---

## ✨ 系統特色

### 1. 純手刻輕量化 RAG Pipeline
不依賴 LangChain 或 LlamaIndex，直接實作核心邏輯，大幅降低框架負載：
- **Data Parser**: 將 HTML/文字規格表解析為 95 個獨立的 Key-Value Chunk。
- **Vector Store**: 基於 NumPy 實作 Cosine Similarity，並結合 Keyword 與 Domain 匹配進行 Hybrid Ranking。
- **Generator**: 串接 `llama-cpp-python` 進行串流 (Streaming) 輸出。

### 2. Hybrid Vector Search
針對結構化的產品規格，單純語意檢索容易混淆型號，因此結合三種分數決定檢索排名：
- **Semantic Similarity** (語意)
- **Keyword Matching** (關鍵字)
- **Domain Matching** (規格領域)

---

## 🚀 啟動步驟

### 1. 系統需求
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (快速 Python 套件管理器)
- Windows 環境 (或 Linux/macOS)

### 2. 環境建置
在專案根目錄下執行：
```bash
uv venv
uv sync
```

*若有支援 CUDA，建議依照 `llama-cpp-python` 官方文件，加上編譯參數以啟用硬體加速：*
```bash
set CMAKE_ARGS="-DGGML_CUDA=on"
uv pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 3. 模型下載
將量化模型 `qwen2.5-3b-instruct-q4_k_m.gguf` 放入 `models/` 目錄中：
```text
aorus-assistant/
└── models/
    └── qwen2.5-3b-instruct-q4_k_m.gguf
```

### 4. 執行評測
```bash
uv run python src/evaluate.py
```

---

## 📊 評測結果分析 (Evaluation Results)

系統內建 10 組涵蓋 CPU、GPU、RAM、網路、擴充槽與電源等硬體規格的 RAG 測試案例，進行定量與定性評估。

### 📈 定量指標 (Quantitative Metrics)

最新實測基準結果 (Baseline)：

| 指標 | 實測平均值 | 說明 |
|---|---|---|
| **Accuracy** | **10 / 10 (100%)** | 10 組問題皆正確檢索規格並正確配對生成。 |
| **Average TTFT** | **1483.09 ms** | 首字延遲 (Time To First Token)。包含模型載入的 Cold Start (3443.62 ms)，後續題目多在 790ms ~ 1800ms 之間，反應迅速。 |
| **Average TPS** | **25.47 tokens/sec** | 生成速度。在 4GB 限制的硬體下，這代表流暢且即時的使用者體驗。 |
| **Stream Chunks/s**| **25.08 chunks/sec** | 串流回傳區塊速度，與 TPS 高度一致。 |

*註：測試環境為 Windows 10，LLM 完全卸載至 GPU，Embedding 運行於 CPU。*

---

### 🔬 定性分析 (Qualitative Analysis)

本專案將 RAG 評估拆分為 **Retrieval (檢索)** 與 **Generation (生成)** 兩階段分析。

#### 1. 結構化資料的防混淆能力 (Test 2 & Test 3)
* **問題**：「請列出 BZH、BYH、BXH 三個版本分別搭載什麼 GPU？」
* **分析**：在多型號查詢中，系統透過 Hybrid Search 精準找回三個版本的規格 Chunk，LLM 成功遵循 Prompt 約束（禁止重新配對），正確回答 `BZH → RTX 5090`、`BYH → RTX 5080`、`BXH → RTX 5070 Ti`，沒有發生「幻覺」或「張冠李戴」的實體屬性映射錯誤 (Entity-Attribute Mapping Error)。

#### 2. 混合式檢索的優勢 (Test 6 & Test 7)
* **問題**：「Thunderbolt 4 與 Thunderbolt 5 分別位於哪一側？」
* **分析**：針對包含多重條件（介面版本 + 位置）的問題，單純的 Cosine Similarity 分數可能不夠，但結合 Keyword 匹配後，系統成功將包含「左側」與「右側」的 Chunk 排入 Top-K。LLM 也能準確整合兩段 Context 給出完整答案。

#### 3. 未來改進方向 (Limitations & Improvements)
雖然目前 10 題皆達 100% 正確率，但系統仍有優化空間：
- **Context 干擾**：部分查詢的 Top-K 檢索結果中會包含無關資訊（如詢問螢幕時混入 GPU 資訊），目前依賴 LLM 的注意力機制過濾。未來可實作純 Python 版本的 **Reranker** 來淨化 Context。
- **資料擴充**：目前的 95 個 Chunks 針對 AM6H/BZH 等型號。未來擴大 Dataset 時，需加入 `Hit Rate` 與 `MRR` 等檢索專用評測指標。
- **格式化輸出**：針對純規格問答，未來可實作 Structured Generation，強制 LLM 輸出 JSON，交由應用層渲染表格，進一步提升穩定性。

---

## 📂 專案結構

```text
aorus-assistant/
│
├── pyproject.toml         # uv 環境與依賴設定
├── README.md              # 專案說明與評測報告
│
├── models/                # 存放 GGUF 模型
│   └── qwen2.5-3b-instruct-q4_k_m.gguf
│
└── src/
    ├── data_parser.py     # 負責處理網頁結構化資料轉 Chunks
    ├── vector_store.py    # 純 Numpy 實作的 Hybrid Vector 檢索系統
    └── evaluate.py        # 負責 RAG 串接、Streaming 輸出與效能測量
```

---

## 📄 授權條款 (License)

本專案原始碼以 MIT 授權釋出。
所使用的 Qwen、Sentence Transformers、llama.cpp 及其他第三方元件的商標與版權均屬其各自所有者。GIGABYTE、AORUS 為技嘉科技之註冊商標。