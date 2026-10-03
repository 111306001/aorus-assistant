import time
import os

from llama_cpp import Llama

from data_parser import AorusDataParser
from vector_store import SimpleVectorStore


def evaluate_query(llm, vector_store, query, top_k=5):
    """
    執行單一 RAG Query，並評估：
    1. Retrieval
    2. TTFT (Time To First Generated Content)
    3. Generated Tokens
    4. Generation Time
    5. TPS (Tokens Per Second)
    6. Stream Chunks/sec
    """

    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    # ============================================================
    # 1. Retrieval
    # ============================================================

    retrieved_docs = vector_store.search(query, top_k=top_k)

    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):
        context_parts.append(
            f"[規格片段 {i}]\n{doc}"
    )

    context = "\n\n".join(context_parts)

    print("\n[Retrieved Context]")

    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\n--- Chunk {i} ---")
        print(doc)

    # ============================================================
    # 2. System Prompt
    # ============================================================

        system_prompt = """
你是 AORUS 筆電規格查詢助手。

你只能根據提供的「規格片段」回答，不得使用外部知識、猜測、推導或補充資料。

請嚴格遵守：

1. 完整回答使用者問題中的所有要求。
   如果問題要求「數量 + 規格」，兩者都必須回答。

2. 嚴格維持「實體 → 規格」的對應關係。
   不可以交換不同型號、版本、介面、位置或規格的資訊。

   例如：
   BZH → RTX 5090
   BYH → RTX 5080
   BXH → RTX 5070 Ti

3. 如果問題涉及多個項目，必須逐項回答。
   不可以只選其中一個。

4. 如果同一類規格有多個項目，必須全部列出。
   例如：
   1 x PCIe Gen4x4 M.2
   1 x PCIe Gen5 M.2
   必須回答為共 2 個 M.2 插槽，而不是 1 個。

5. 位置、介面與功能必須保持正確對應。
   例如：
   左側 → Thunderbolt 5 → DisplayPort 2.1
   右側 → Thunderbolt 4 → DisplayPort 1.4
   不可以交換。

6. 必須保留規格中的精確數值與單位，例如：
   64GB、DDR5 5600MHz、2 x SO-DIMM、
   PCIe Gen5、PCIe Gen4x4、240Hz、2560×1600、
   99Wh、330W。

7. 如果規格片段不足以回答問題，回答：
   「提供的規格資料不足以回答此問題。」

8. 使用繁體中文，回答簡潔直接，可使用條列式。

回答前請確認：
- 是否回答所有問題要求？
- 是否遺漏數量、單位或規格？
- 是否交換不同項目的對應關係？

只輸出最終答案，不要輸出分析過程。
"""

        user_prompt = f"""
以下是 AORUS 筆電規格資料：

{context}

使用者問題：
{query}

請嚴格根據以上規格片段回答。

回答時：
1. 完整回答問題中的所有要求。
2. 保留每個型號、介面、位置、數量與規格值之間的正確對應關係。
3. 如果有多個項目，請逐項列出。
4. 不要交換不同項目的規格。
5. 不要省略問題要求的任何規格。
6. 只輸出最終答案，不要輸出分析過程。
"""

    # ============================================================
    # 3. LLM Generation
    # ============================================================

    print("\n[Answer]")

    start_time = time.perf_counter()

    first_content_time = None
    generated_text = ""
    output_chunks = 0

    stream = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_tokens=256,
        temperature=0.1,
        stream=True,
    )

    for chunk in stream:
        choice = chunk.get("choices", [{}])[0]

        content = choice.get("delta", {}).get("content", "")

        if content:
            # 第一次收到實際文字內容
            if first_content_time is None:
                first_content_time = time.perf_counter()

            generated_text += content
            output_chunks += 1

            print(content, end="", flush=True)

    end_time = time.perf_counter()

    print()

    # ============================================================
    # 4. Performance Metrics
    # ============================================================

    # TTFT
    if first_content_time is not None:
        ttft_ms = (first_content_time - start_time) * 1000
    else:
        ttft_ms = None

    # Generation time
    if first_content_time is not None:
        generation_time_sec = end_time - first_content_time
    else:
        generation_time_sec = 0.0

    # ------------------------------------------------------------
    # Token count
    # ------------------------------------------------------------
    #
    # llama.cpp streaming chunks != tokens
    #
    # 因此不能用 output_chunks 當 TPS。
    # 這裡把最後生成的文字重新交給 llama.cpp tokenizer，
    # 得到實際 token 數。
    #
    generated_tokens = len(
        llm.tokenize(
            generated_text.encode("utf-8"),
            add_bos=False,
        )
    )

    # TPS
    if generation_time_sec > 0:
        tps = generated_tokens / generation_time_sec
    else:
        tps = 0.0

    # Stream chunks/sec
    if generation_time_sec > 0:
        stream_chunks_per_sec = output_chunks / generation_time_sec
    else:
        stream_chunks_per_sec = 0.0

    # ============================================================
    # 5. Print Metrics
    # ============================================================

    print("\n" + "-" * 70)
    print("[Performance]")
    print("-" * 70)

    if ttft_ms is not None:
        print(f"TTFT:                {ttft_ms:.2f} ms")
    else:
        print("TTFT:                N/A")

    print(f"Generated Tokens:    {generated_tokens}")
    print(f"Generation Time:     {generation_time_sec:.3f} sec")
    print(f"TPS:                 {tps:.2f} tokens/sec")
    print(f"Stream Chunks/sec:   {stream_chunks_per_sec:.2f}")
    print(f"Retrieved Chunks:    {len(retrieved_docs)}")

    # ============================================================
    # 6. Return Evaluation Result
    # ============================================================

    return {
        "query": query,
        "retrieved_docs": retrieved_docs,
        "answer": generated_text,
        "ttft_ms": ttft_ms,
        "generated_tokens": generated_tokens,
        "generation_time_sec": generation_time_sec,
        "tps": tps,
        "stream_chunks_per_sec": stream_chunks_per_sec,
    }


def main():

    # ============================================================
    # 1. Initialize Data Parser
    # ============================================================

    print("=" * 70)
    print("Initializing Data Parser & Vector Store...")
    print("=" * 70)

    parser = AorusDataParser()

    documents = parser.get_chunks()

    print(f"Total specification chunks: {len(documents)}")

    # ============================================================
    # 2. Initialize Vector Store
    # ============================================================

    vector_store = SimpleVectorStore()

    print("Loading embedding model...")

    vector_store.add_documents(documents)

    print("Vector store initialized.")

    # ============================================================
    # 3. Load LLM
    # ============================================================

    model_path = "models/qwen2.5-3b-instruct-q4_k_m.gguf"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Please make sure the GGUF model exists in the models directory."
        )

    print("\n" + "=" * 70)
    print("Loading Qwen2.5-3B-Instruct...")
    print("=" * 70)

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )

    print("LLM loaded successfully.")

    # ============================================================
    # 4. Evaluation Queries
    # ============================================================

    test_queries = [
        # Test 1
        "AORUS MASTER 16 AM6H 的 CPU 是什麼？",

        # Test 2
        "請列出 BZH、BYH、BXH 三個版本分別搭載什麼 GPU？",

        # Test 3
        "BZH、BYH、BXH 的 GPU 記憶體容量分別是多少？",

        # Test 4
        "AORUS MASTER 16 的螢幕解析度與更新率是多少？",

        # Test 5
        "這台筆電最高支援多少 RAM？規格與插槽數量是多少？",

        # Test 6
        "AORUS MASTER 16 有幾個 M.2 SSD 插槽？分別支援什麼 PCIe 規格？",

        # Test 7
        "Thunderbolt 4 與 Thunderbolt 5 分別位於哪一側？各支援哪些功能？",

        # Test 8
        "這台筆電的無線網路、Bluetooth 和有線網路規格是什麼？",

        # Test 9
        "這台筆電是否支援 Windows Hello？有沒有 TPM？",

        # Test 10
        "這台筆電的電池容量與 AC Adapter 功率是多少？",
    ]

    # ============================================================
    # 5. Run Evaluation
    # ============================================================

    results = []

    print("\n")
    print("=" * 70)
    print("Starting RAG Evaluation")
    print("=" * 70)

    for i, query in enumerate(test_queries, start=1):

        print("\n")
        print("#" * 70)
        print(f"# Test {i}/{len(test_queries)}")
        print("#" * 70)

        result = evaluate_query(
            llm=llm,
            vector_store=vector_store,
            query=query,
            top_k=5,
        )

        results.append(result)

    # ============================================================
    # 6. Summary
    # ============================================================

    print("\n\n")
    print("=" * 70)
    print("Evaluation Summary")
    print("=" * 70)

    print(
        f"{'Test':<6}"
        f"{'TTFT(ms)':<14}"
        f"{'Tokens':<10}"
        f"{'TPS':<12}"
        f"{'Chunks/s':<14}"
        f"{'Retrieved':<10}"
    )

    print("-" * 70)

    for i, result in enumerate(results, start=1):

        ttft = result["ttft_ms"]
        tokens = result["generated_tokens"]
        tps = result["tps"]
        chunks_sec = result["stream_chunks_per_sec"]
        retrieved = len(result["retrieved_docs"])

        ttft_str = f"{ttft:.2f}" if ttft is not None else "N/A"

        print(
            f"{i:<6}"
            f"{ttft_str:<14}"
            f"{tokens:<10}"
            f"{tps:<12.2f}"
            f"{chunks_sec:<14.2f}"
            f"{retrieved:<10}"
        )

    print("-" * 70)

    # ============================================================
    # 7. Average Metrics
    # ============================================================

    valid_ttft = [
        r["ttft_ms"]
        for r in results
        if r["ttft_ms"] is not None
    ]

    valid_tps = [
        r["tps"]
        for r in results
        if r["tps"] > 0
    ]

    valid_chunks_sec = [
        r["stream_chunks_per_sec"]
        for r in results
        if r["stream_chunks_per_sec"] > 0
    ]

    if valid_ttft:
        avg_ttft = sum(valid_ttft) / len(valid_ttft)
    else:
        avg_ttft = 0.0

    if valid_tps:
        avg_tps = sum(valid_tps) / len(valid_tps)
    else:
        avg_tps = 0.0

    if valid_chunks_sec:
        avg_chunks_sec = sum(valid_chunks_sec) / len(valid_chunks_sec)
    else:
        avg_chunks_sec = 0.0

    print("\nAverage Performance")
    print("-" * 70)

    print(f"Average TTFT:              {avg_ttft:.2f} ms")
    print(f"Average TPS:               {avg_tps:.2f} tokens/sec")
    print(f"Average Stream Chunks/sec: {avg_chunks_sec:.2f}")

    # ============================================================
    # 8. Final RAG Evaluation
    # ============================================================

    print("\n")
    print("=" * 70)
    print("RAG Evaluation Result")
    print("=" * 70)

    print(f"Total Tests: {len(results)}")

    print("\nAll tests completed.")

    print("=" * 70)


if __name__ == "__main__":
    main()