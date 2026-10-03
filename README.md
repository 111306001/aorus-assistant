# AORUS MASTER 16 AM6H AI Assistant

一個專為**資源有限的消費級筆電**設計的輕量化離線 RAG（Retrieval-Augmented Generation，檢索增強生成）問答系統。

本專案以 GIGABYTE AORUS MASTER 16 AM6H 系列產品規格為知識來源，使用向量檢索取得相關規格，再交由本地端 GGUF LLM 生成回答。

系統的設計目標是在有限 GPU VRAM 的環境下，維持：

- 輕量化
- 本地端推論
- 中英文混合問答
- 結構化規格檢索
- 低額外框架依賴
- 可量化的 Retrieval / Generation 測試

本專案**不使用 LangChain、LlamaIndex 等高階 RAG 框架**，核心流程以 Python、NumPy、Sentence Transformers 與 `llama-cpp-python` 自行實作。

---

## 🛠️ 技術規格

| 項目 | 實作 |
|---|---|
| 程式語言 | Python 3.11+ |
| 套件管理 | `uv` |
| RAG Framework | 無 |
| Embedding Model | `paraphrase-multilingual-MiniLM-L12-v2` |
| LLM | Qwen2.5-3B-Instruct |
| LLM 格式 | GGUF |
| Quantization | Q4_K_M |
| 推論引擎 | `llama-cpp-python` |
| Vector Search | NumPy Cosine Similarity + Keyword / Domain Hybrid Search |
| Context Size | 2048 tokens |
| GPU Layers | `-1`（全部卸載至 GPU） |
| Top-K | 5 |
| 語言 | 繁體中文 / English |

---

## ✨ 系統特色

### 1. Lightweight RAG

不依賴 LangChain 或 LlamaIndex，直接實作：

```text
User Query
    ↓
Query Analysis
    ↓
Hybrid Vector Search
    ↓
Top-K Specification Chunks
    ↓
RAG Prompt
    ↓
Local GGUF LLM
    ↓
Streaming Answer
```

這樣可以降低框架層級的額外依賴，同時讓 Retrieval、Prompt 與 Generation 流程保持透明且容易修改。

---

### 2. Hybrid Vector Search

Vector Store 不只使用語意相似度，也會結合：

- Semantic Similarity
- Keyword Matching
- Domain Matching

最終根據多項分數決定 Retrieval 排名。

例如：

```text
Query:
What GPUs are available for the AORUS MASTER 16 BZH, BYH, and BXH?

Retrieved:

BYH → RTX 5080 Laptop GPU
BZH → RTX 5090 Laptop GPU
BXH → RTX 5070 Ti Laptop GPU
```

這對產品型號、介面名稱與規格名稱等高度結構化資訊尤其重要。

---

## 🧠 Model Selection

### LLM：Qwen2.5-3B-Instruct

本專案使用：

```text
qwen2.5-3b-instruct-q4_k_m.gguf
```

選擇 3B 級模型主要考量：

1. 模型規模相對較小
2. 適合本地端推論
3. GGUF Q4_K_M 可降低模型記憶體需求
4. 能處理繁體中文與英文混合問題
5. 足以完成本專案所需的規格型問答

目前測試設定：

```text
GPU layers : -1
Context    : 2048
```

`n_gpu_layers=-1` 代表將模型可卸載的 layers 全部交由 GPU 執行，以降低 CPU 推論負擔。

---

### Embedding：paraphrase-multilingual-MiniLM-L12-v2

本專案實際使用：

```text
paraphrase-multilingual-MiniLM-L12-v2
```

而不是大型 embedding model。

選擇 multilingual embedding 的主要原因是系統需要同時處理：

```text
English Query
        +
繁體中文規格資料
```

例如：

```text
What CPU does the AORUS MASTER 16 AM6H use?
```

以及：

```text
這台筆電最高支援多少 RAM？
```

都可以透過同一套 embedding pipeline 進行檢索。

Embedding model 在 CPU 上執行，避免佔用 LLM 所需的 GPU VRAM。

---

# 📂 Project Structure

```text
aorus-assistant/
│
├── pyproject.toml
├── .gitignore
├── README.md
│
├── models/
│   └── qwen2.5-3b-instruct-q4_k_m.gguf
│
└── src/
    ├── data_parser.py
    ├── vector_store.py
    └── evaluate.py
```

### `data_parser.py`

負責：

- 解析產品規格
- 將結構化規格轉換成適合 Retrieval 的文字 Chunk
- 建立 specification chunks

目前測試資料共：

```text
95 specification chunks
```

---

### `vector_store.py`

負責：

- Embedding
- Vector storage
- Semantic similarity
- Keyword matching
- Domain matching
- Hybrid ranking
- Top-K Retrieval

---

### `evaluate.py`

負責：

- 初始化 Vector Store
- 載入 Embedding Model
- 載入 GGUF LLM
- 執行 RAG Query
- Streaming Generation
- TTFT 測量
- Stream Chunks/sec 測量
- Retrieval / Generation 測試

---

# 🚀 Quick Start

## 1. 安裝 `uv`

請先安裝 Python 3.11+ 與 `uv`。

確認：

```bash
python --version
uv --version
```

---

## 2. 建立環境

在專案根目錄：

```bash
uv venv
uv sync
```

---

## 3. 準備 GGUF Model

由於 GGUF 模型通常超過 GitHub 的單檔大小限制，因此模型檔不納入 Git repository。

請將：

```text
qwen2.5-3b-instruct-q4_k_m.gguf
```

放入：

```text
models/
```

最終路徑：

```text
models/qwen2.5-3b-instruct-q4_k_m.gguf
```

---

## 4. 執行 Evaluation

```bash
uv run python src/evaluate.py
```

系統會依序：

```text
Initialize Data Parser
        ↓
Create 95 specification chunks
        ↓
Load multilingual embedding model
        ↓
Build vector store
        ↓
Load Qwen2.5-3B GGUF
        ↓
Run 10 RAG tests
        ↓
Measure TTFT / Stream speed
```

---

# 📊 Evaluation Results

目前版本使用 10 組人工設計的 RAG 測試案例進行測試。

測試範圍涵蓋：

1. CPU
2. GPU 型號
3. GPU 記憶體
4. Display Resolution / Refresh Rate
5. RAM
6. M.2 Storage
7. Thunderbolt 4 / Thunderbolt 5
8. Wired / Wireless Networking
9. Windows Hello / TPM
10. Battery / AC Adapter

---

## 📈 Accuracy

最新測試結果：

```text
Correct Answers : 10 / 10
Accuracy         : 100%
```

所有 10 組測試的 Retrieval 結果均包含回答問題所需的關鍵規格資訊，Generation 亦正確保留主要的：

- 型號 → 規格
- 分類 → 規格
- 位置 → 介面
- 數量 → 規格
- 數值 → 單位

對應關係。

---

## ⚡ Performance

最新實測：

```text
Average TTFT:
1519.52 ms

Average Stream:
24.35 chunks/sec
```

### Important Note

`Stream Chunks/sec` **不等同於 Tokens/sec（TPS）**。

`llama.cpp` 的 streaming callback 所回傳的 chunk 不一定恰好對應一個 token，因此本專案將該指標稱為：

```text
Stream Chunks/sec
```

而不是：

```text
Tokens/sec
```

這可以避免將 chunk 數量直接解讀為模型 token generation speed。

---

# 🧪 Test Results

| Test | Topic | Result |
|---:|---|:---:|
| 01 | CPU | ✅ |
| 02 | GPU Model Mapping | ✅ |
| 03 | GPU Memory | ✅ |
| 04 | Display | ✅ |
| 05 | RAM | ✅ |
| 06 | M.2 / PCIe | ✅ |
| 07 | Thunderbolt / Port Position | ✅ |
| 08 | Wireless / Wired Network | ✅ |
| 09 | Windows Hello / TPM | ✅ |
| 10 | Battery / AC Adapter | ✅ |
| **Overall** | **10 / 10** | **100%** |

---

# 🔍 Retrieval vs Generation Analysis

本專案特別將 RAG 評估拆成兩個階段：

```text
Retrieval
    ↓
Generation
```

這可以區分：

> 「系統沒有找到資料」

以及：

> 「系統找到了資料，但 LLM 回答時配對錯誤」

---

## Test 2：Multi-Model Mapping

問題：

```text
What GPUs are available for the AORUS MASTER 16 BZH, BYH, and BXH?
```

Retrieval 找到：

```text
BYH → RTX 5080
BZH → RTX 5090
BXH → RTX 5070 Ti
```

最終回答：

```text
BZH → RTX 5090
BYH → RTX 5080
BXH → RTX 5070 Ti
```

測試證明 Prompt 中的「型號 → 規格」對應約束可以降低小型模型重新配對規格的情況。

---

## Test 8：Category Mapping

問題：

```text
這台筆電支援哪些無線與有線網路規格？
```

Retrieval 找到：

```text
無線網路 → Wi-Fi 7
Wi-Fi 規格 → 802.11be 2x2
有線網路 → 1G LAN
```

最終回答：

```text
無線網路：
Wi-Fi 7 802.11be 2x2、Bluetooth 5.4

有線網路：
1G LAN
```

這個測試特別驗證了：

```text
Category → Specification
```

的資訊對應是否能在 Generation 階段被保留。

---

# 🧠 Generation Prompt Design

由於本專案使用的是較小型的 3B LLM，因此 Prompt 特別強調**規格對應關係的保留**。

主要規則包括：

```text
只能使用參考規格中的資訊。

不可以猜測、補充或推導。

必須保留：
「規格名稱 → 規格值」
的原始對應關係。

多個型號：
「型號 → 規格」

多個分類：
「分類 → 規格」

不可交換、顛倒或重新配對規格。
```

這對以下類型問題尤其重要：

```text
BZH → RTX 5090
BYH → RTX 5080
BXH → RTX 5070 Ti
```

以及：

```text
Wireless → Wi-Fi 7 / Bluetooth 5.4
Wired → 1G LAN
```

---

# ⚠️ Current Limitations

雖然目前 10 組測試達到 100% accuracy，但這並不代表系統在所有問題上的準確率都是 100%。

目前測試規模為：

```text
10 test cases
```

因此結果應理解為：

> **在目前 10 組人工設計測試案例中，回答正確率為 100%。**

而不是宣稱系統對所有可能問題都能達到 100% accuracy。

---

## Retrieval 仍存在少量無關結果

部分 Query 的 Top-K Retrieval 中仍會出現與問題無直接關係的 Chunk。

例如網路規格問題的 Retrieval 結果中，可能同時出現：

```text
BZH → RTX 5090 Laptop GPU
```

雖然該資訊與問題無關，但目前 Generation 能夠忽略干擾資料並產生正確答案。

因此目前的主要改善方向不是單純追求「Top-K 完全沒有無關資料」，而是進一步評估：

- Retrieval Precision
- Recall
- Reranking
- Top-K 最佳值
- Hard Negative Retrieval
- Generation 對干擾資訊的抗性

---

# 🔬 Future Improvements

後續可以從以下方向進行：

### 1. 擴充 Evaluation Dataset

目前：

```text
10 tests
```

後續可以增加至：

```text
30
50
100+
```

並加入：

- 多型號比較
- 中英文混合
- 不存在的規格
- 相似規格
- 數量問題
- 位置問題
- 多分類問題
- 干擾資訊
- Hard Negative Query

---

### 2. Retrieval Evaluation

除了最終答案 Accuracy，也可以加入：

```text
Recall@K
Precision@K
MRR
Hit Rate
```

以獨立量化 Retrieval 品質。

---

### 3. Reranking

目前使用 Hybrid Search。

未來可以加入 reranker：

```text
Query
  ↓
Hybrid Retrieval
  ↓
Top-N Candidates
  ↓
Reranker
  ↓
Top-K Context
  ↓
LLM
```

降低無關 Chunk 進入 Context 的機率。

---

### 4. Structured Generation

對高度結構化的產品規格，可以進一步要求模型使用固定格式：

```text
型號：規格
分類：規格
位置：規格
```

甚至可以讓 Generation 輸出 JSON，再由程式進行驗證。

這可以進一步降低：

```text
Model A → Specification B
```

這類 Entity-Attribute Mapping Error。

---

### 5. Automated Evaluation

目前 Accuracy 主要依賴人工檢查。

未來可以建立：

```text
Question
Expected Answer
Retrieved Context
Generated Answer
Evaluation Result
```

並自動計算：

```text
Accuracy
Retrieval Hit Rate
TTFT
Stream Chunks/sec
```

形成完整的 regression test system。

---

# 📌 Current Baseline

截至目前測試版本：

```text
AORUS MASTER 16 AM6H AI Assistant
────────────────────────────────────
Specification Chunks : 95

Embedding:
paraphrase-multilingual-MiniLM-L12-v2

LLM:
Qwen2.5-3B-Instruct
Q4_K_M GGUF

Context:
2048 tokens

GPU Layers:
-1

Top-K:
5

Evaluation Cases:
10

Accuracy:
10 / 10 (100%)

Average TTFT:
1519.52 ms

Average Stream:
24.35 chunks/sec
```

這個版本可作為後續 Retrieval、Prompt、Reranking 與模型最佳化的 **Baseline**。

---

# 📄 License

本專案的程式碼、資料與模型使用方式應依各自套件、資料來源與模型的授權條款使用。

Qwen、Sentence Transformers、llama.cpp 及其他第三方元件的商標與版權均屬其各自所有者。