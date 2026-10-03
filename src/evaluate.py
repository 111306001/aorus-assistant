# src/evaluate.py
import time
import os
from llama_cpp import Llama
from data_parser import AorusDataParser
from vector_store import SimpleVectorStore

def main():
    # 1. 準備資料與向量檢索
    print("Initializing Data Parser & Vector Store...")
    parser = AorusDataParser()
    docs = parser.get_chunks()

    vector_store = SimpleVectorStore()
    vector_store.add_documents(docs)

    # 2. 載入 GGUF 模型 (限 4GB VRAM 優化設定)
    model_path = os.path.abspath("models/qwen2.5-3b-instruct-q4_k_m.gguf")
    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型檔案於 {model_path}，請確認檔名與路徑！")
        return

    print(f"Loading LLM from {model_path} (n_gpu_layers=-1 for 4GB VRAM)...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,       # 限制 Context 視窗以節省記憶體
        n_gpu_layers=-1,  # 全部卸載到 GPU 運行
        verbose=False
    )

    # 3. 測試問題（支援繁體中文與英文混合）
    test_query = "What is the screen display and refresh rate of AORUS MASTER 16 AM6H? 這台筆電的處理器是什麼？"
    print(f"\n[使用者提問]: {test_query}\n")

    # 4. 檢索相關 Context
    retrieved_docs = vector_store.search(test_query, top_k=2)
    context = "\n".join(retrieved_docs)
    print(f"[檢索到的規格片段]:\n{context}\n")

    # 5. 建構 Prompt 與串流生成
    prompt = f"""請根據以下 GIGABYTE AORUS MASTER 16 AM6H 產品規格回答問題。支援繁體中文與英文混合回答。

【參考規格】：
{context}

【使用者問題】：{test_query}
【回答】："""

    print("-" * 40)
    print("[AI 回答串流輸出]: ", end="", flush=True)

    start_time = time.time()
    first_token_time = None
    token_count = 0

    stream = llm(
        prompt,
        max_tokens=256,
        temperature=0.1,
        stream=True
    )

    for output in stream:
        if first_token_time is None:
            first_token_time = time.time()
        text = output['choices'][0]['text']
        print(text, end="", flush=True)
        token_count += 1

    end_time = time.time()

    # 6. 計算定量效能指標 (System Evaluation)
    ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
    total_time = end_time - first_token_time if first_token_time else (end_time - start_time)
    tps = token_count / total_time if total_time > 0 else 0

    print("\n" + "-" * 40)
    print(f"\n📊 【系統效能評測指標】")
    print(f" - TTFT (首字延遲): {ttft:.2f} ms")
    print(f" - TPS (生成速度): {tps:.2f} tokens/sec")

if __name__ == "__main__":
    main()