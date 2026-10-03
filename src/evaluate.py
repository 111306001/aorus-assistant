import time
import os

from llama_cpp import Llama
from data_parser import AorusDataParser
from vector_store import SimpleVectorStore


def evaluate_query(llm, vector_store, query, top_k=5):
    """
    執行單一問題的 RAG 測試。

    評估項目：
    - Retrieval：檢索到的規格片段
    - Generation：LLM 最終回答
    - TTFT：Time To First Token
    - Stream Chunks/sec：Streaming chunk 產生速度

    注意：
    llama.cpp streaming chunk 不一定等於一個 token，
    因此不能直接將 chunk/sec 稱為 Token Per Second。
    """

    print("\n" + "=" * 60)
    print(f"[使用者問題]\n{query}")

    # =========================================================
    # 1. Retrieval
    # =========================================================

    retrieved_docs = vector_store.search(
        query,
        top_k=top_k
    )

    # 為每個 chunk 加上編號，幫助 LLM 維持資訊關係
    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):
        context_parts.append(
            f"[規格片段 {i}]\n{doc}"
        )

    context = "\n\n".join(context_parts)

    print("\n[檢索到的規格片段]")
    print("-" * 40)
    print(context)

    # =========================================================
    # 2. RAG Prompt
    # =========================================================

    messages = [
        {
            "role": "system",
            "content": (
                "你是一個嚴格的產品規格問答助手。\n"
                "只能根據【參考規格】回答，不可以使用參考規格以外的知識。\n"
                "\n"

                "【核心規則】\n"
                "1. 只能使用參考規格中明確提供的資訊。\n"
                "2. 不可以猜測、補充、推導或自行計算。\n"
                "3. 如果參考規格沒有答案，只回答：「參考規格中沒有相關資訊。」\n"
                "4. 回答時必須保留參考規格原本的「規格名稱 → 規格值」對應關係。\n"
                "5. 絕對不可交換、顛倒、重新配對或自行重組不同規格的資訊。\n"
                "\n"

                "【型號對應】\n"
                "6. 如果問題涉及多個不同型號，必須逐一建立「型號 → 規格」對應。\n"
                "7. 每個型號只能使用參考規格中明確屬於該型號的資訊。\n"
                "8. 絕對不可將某一型號的規格套用到其他型號。\n"
                "\n"

                "【分類對應】\n"
                "9. 如果問題涉及不同分類，例如無線網路與有線網路，"
                "必須按照參考規格中明確標示的分類回答。\n"
                "10. 無線網路只能使用明確屬於無線網路的規格，例如 Wi-Fi、802.11、Bluetooth。\n"
                "11. 有線網路只能使用明確屬於有線網路的規格，例如 LAN、Ethernet。\n"
                "12. 不可以將一個分類的規格值套用到另一個分類。\n"
                "\n"

                "【位置與數量】\n"
                "13. 如果問題涉及左側、右側、位置或介面，必須完全按照參考規格中的位置回答，不可顛倒。\n"
                "14. 數量必須直接依照參考規格回答，不可以自行相加、平均、推算或重複計算。\n"
                "15. 如果參考規格同時提供總數與分類數量，必須分開理解，不可混用。\n"
                "\n"

                "【數值與單位】\n"
                "16. 數值與單位必須忠實保留，例如 GB、TB、W、Wh、MHz、Hz、GHz。\n"
                "17. 不可以修改、轉換或推導參考規格中的數值。\n"
                "\n"

                "【回答方式】\n"
                "18. 如果參考規格已經明確提供答案，直接整理原始規格即可。\n"
                "19. 多個型號時，使用「型號：規格」逐項回答。\n"
                "20. 多個分類時，使用「分類：規格」逐項回答。\n"
                "21. 回答使用繁體中文。\n"
                "22. 只回答問題需要的資訊，不要額外解釋。\n"
                "23. 回答完立即停止。"
            )
        },
        {
            "role": "user",
            "content": (
                f"【參考規格】\n"
                f"{context}\n\n"
                f"【使用者問題】\n"
                f"{query}"
            )
        }
    ]

    # =========================================================
    # 3. LLM Streaming Generation
    # =========================================================

    print("\n[AI 回答]")
    print("-" * 40)

    start_time = time.time()
    first_token_time = None

    generated_text = ""
    output_chunks = 0

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.1,
        stream=True
    )

    for output in stream:

        # 第一個 streaming chunk 到達時間
        if first_token_time is None:
            first_token_time = time.time()

        delta = output["choices"][0]["delta"]

        if "content" in delta:

            text = delta["content"]

            print(
                text,
                end="",
                flush=True
            )

            generated_text += text
            output_chunks += 1

    end_time = time.time()

    # =========================================================
    # 4. Performance Metrics
    # =========================================================

    if first_token_time is not None:

        ttft = (
            first_token_time - start_time
        ) * 1000

        generation_time = (
            end_time - first_token_time
        )

        stream_chunks_per_sec = (
            output_chunks / generation_time
            if generation_time > 0
            else 0
        )

    else:

        ttft = 0
        stream_chunks_per_sec = 0

    # =========================================================
    # 5. Evaluation Output
    # =========================================================

    print("\n" + "-" * 40)

    print("📊 【系統效能評測】")

    print(
        f" - TTFT：{ttft:.2f} ms"
    )

    print(
        f" - Stream Chunks/sec："
        f"{stream_chunks_per_sec:.2f}"
    )

    print(
        f" - Retrieved Chunks："
        f"{len(retrieved_docs)}"
    )

    print("\n📋 【Retrieval / Generation 分離評估】")

    print(
        " - Retrieval："
        "請確認檢索結果是否包含回答問題所需的正確規格。"
    )

    print(
        " - Generation："
        "請確認 AI 回答是否忠實保留規格中的型號、"
        "位置、數量、數值與規格名稱對應關係。"
    )

    return {
        "query": query,
        "retrieved_docs": retrieved_docs,
        "answer": generated_text,
        "ttft_ms": ttft,
        "stream_chunks_per_sec": stream_chunks_per_sec,
    }


def main():

    # =========================================================
    # 1. 初始化 Data Parser
    # =========================================================

    print("=" * 60)
    print("Initializing Data Parser & Vector Store...")
    print("=" * 60)

    parser = AorusDataParser()

    docs = parser.get_chunks()

    print(
        f"Total specification chunks: "
        f"{len(docs)}"
    )

    # =========================================================
    # 2. 初始化 Vector Store
    # =========================================================

    vector_store = SimpleVectorStore()

    vector_store.add_documents(docs)

    print(
        "Vector Store initialized successfully."
    )

    # =========================================================
    # 3. 載入 GGUF LLM
    # =========================================================

    model_path = os.path.abspath(
        "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    )

    if not os.path.exists(model_path):

        print(
            f"\n錯誤：找不到模型檔案於：\n"
            f"{model_path}\n"
            f"請確認模型檔案名稱與路徑。"
        )

        return

    print("\nLoading LLM...")
    print(f"Model: {model_path}")
    print("GPU layers: -1")
    print("Context size: 2048")

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )

    print("LLM loaded successfully.")

    # =========================================================
    # 4. RAG 測試問題
    # =========================================================

    test_queries = [

        # CPU
        "What CPU does the AORUS MASTER 16 AM6H use?",

        # GPU
        "What GPUs are available for the AORUS MASTER 16 BZH, BYH, and BXH?",

        # GPU VRAM
        "BZH, BYH 和 BXH 的 GPU 記憶體容量分別是多少？",

        # Display
        "What is the screen resolution and refresh rate of the AORUS MASTER 16 AM6H?",

        # Memory
        "這台筆電最高支援多少 RAM？記憶體規格與插槽是什麼？",

        # Storage
        "How many M.2 slots does the AORUS MASTER 16 AM6H have, and what PCIe generations do they support?",

        # Thunderbolt
        "AORUS MASTER 16 AM6H 的 Thunderbolt 4 和 Thunderbolt 5 分別位於哪一側？支援哪些功能？",

        # Network
        "這台筆電支援哪些無線與有線網路規格？",

        # Security
        "Does the AORUS MASTER 16 AM6H support Windows Hello and TPM?",

        # Battery
        "What is the battery capacity and AC adapter wattage?",
    ]

    # =========================================================
    # 5. 執行測試
    # =========================================================

    results = []

    print("\n")
    print("=" * 60)
    print(
        f"開始執行 {len(test_queries)} 組 RAG 測試"
    )
    print("=" * 60)

    for i, query in enumerate(
        test_queries,
        start=1
    ):

        print("\n")
        print(
            f"################ "
            f"TEST {i}/{len(test_queries)} "
            f"################"
        )

        result = evaluate_query(
            llm=llm,
            vector_store=vector_store,
            query=query,
            top_k=5,
        )

        results.append(result)

    # =========================================================
    # 6. 測試結果摘要
    # =========================================================

    print("\n")
    print("=" * 60)
    print("📊 RAG 測試結果摘要")
    print("=" * 60)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"Test {i:02d} | "
            f"TTFT: {result['ttft_ms']:.2f} ms | "
            f"Stream: "
            f"{result['stream_chunks_per_sec']:.2f} "
            f"chunks/sec | "
            f"Retrieved: "
            f"{len(result['retrieved_docs'])}"
        )

    # =========================================================
    # 7. 平均效能
    # =========================================================

    if results:

        avg_ttft = (
            sum(
                r["ttft_ms"]
                for r in results
            )
            / len(results)
        )

        avg_stream = (
            sum(
                r["stream_chunks_per_sec"]
                for r in results
            )
            / len(results)
        )

        print("\n" + "-" * 60)
        print("📈 平均效能")
        print("-" * 60)

        print(
            f"平均 TTFT："
            f"{avg_ttft:.2f} ms"
        )

        print(
            f"平均 Stream Chunks/sec："
            f"{avg_stream:.2f}"
        )


if __name__ == "__main__":
    main()