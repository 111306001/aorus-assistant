import re
import numpy as np
from sentence_transformers import SentenceTransformer


class SimpleVectorStore:

    # ---------------------------------------------------------
    # Domain definitions
    # ---------------------------------------------------------

    DOMAIN_TERMS = {
        "cpu": [
            "cpu",
            "processor",
            "core ultra",
            "intel",
            "amd",
            "core",
            "thread",
        ],

        "gpu": [
            "gpu",
            "graphics",
            "rtx",
            "geforce",
            "gddr",
            "vram",
            "graphics memory",
        ],

        "ram": [
            "ram",
            "memory",
            "ddr4",
            "ddr5",
            "so-dimm",
            "sodimm",
            "system memory",
        ],

        "storage": [
            "storage",
            "ssd",
            "m.2",
            "pcie",
            "nvme",
        ],

        "display": [
            "display",
            "screen",
            "panel",
            "oled",
            "resolution",
            "refresh rate",
            "hz",
            "wqxga",
            "nits",
            "dci-p3",
        ],

        "port": [
            "port",
            "usb",
            "usb-a",
            "usb-c",
            "thunderbolt",
            "hdmi",
            "displayport",
            "micro sd",
            "microsd",
            "audio",
            "power delivery",
        ],

        "network": [
            "network",
            "wifi",
            "wi-fi",
            "bluetooth",
            "lan",
            "ethernet",
            "802.11",
        ],

        "security": [
            "windows hello",
            "tpm",
            "intel ptt",
            "security",
            "authentication",
            "webcam",
            "ir webcam",
        ],

        "battery": [
            "battery",
            "wh",
            "li-ion",
            "adapter",
            "ac adapter",
            "power adapter",
            "watt",
        ],
    }

    # Terms which should be treated as especially important.
    TECHNICAL_PATTERNS = [
        r"\brtx\s*\d+\b",
        r"\b5090\b",
        r"\b5080\b",
        r"\b5070\s*ti\b",

        r"\b\d+\s*gb\b",
        r"\b\d+\s*tb\b",
        r"\b\d+\s*wh\b",
        r"\b\d+\s*w\b",
        r"\b\d+\s*hz\b",
        r"\b\d+\s*mhz\b",
        r"\b\d+\s*mb\b",
        r"\b\d+\.\d+\s*ghz\b",

        r"\bwifi\s*\d+\b",
        r"\bwi-fi\s*\d+\b",

        r"\bthunderbolt\s*[345]\b",
        r"\busb\s*[0-9.]+\b",

        r"\bddr[45]\b",
        r"\bpcie\s*[a-z0-9x.-]+\b",

        r"\bwqxga\b",
        r"\boled\b",
        r"\bhdmi\s*[0-9.]+\b",
        r"\bdisplayport\s*[0-9.]+\b",

        r"\bbluetooth\s*[0-9.]+\b",
        r"\b802\.11[a-z0-9]+\b",

        r"\bbzh\b",
        r"\bbyh\b",
        r"\bbxh\b",

        r"\bso[- ]?dimm\b",
        r"\bm\.2\b",
        r"\bnvme\b",
        r"\bgddr7\b",
        r"\bvram\b",
    ]

    # Generic English words which should not contribute
    # strongly to technical keyword matching.
    STOPWORDS = {
        "what",
        "does",
        "do",
        "the",
        "is",
        "are",
        "was",
        "were",
        "for",
        "of",
        "and",
        "or",
        "to",
        "in",
        "on",
        "at",
        "with",
        "from",
        "how",
        "many",
        "which",
        "where",
        "when",
        "who",
        "can",
        "could",
        "would",
        "should",
        "use",
        "used",
        "using",
        "support",
        "supports",
        "supported",
        "available",
        "highest",
        "lowest",
        "spec",
        "specs",
        "specification",
        "specifications",
        "information",
        "tell",
        "me",
        "about",
    }

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        candidate_k: int = 15,
        similarity_threshold: float = 0.30,
    ):
        """
        Args:
            model_name:
                SentenceTransformer embedding model.

            candidate_k:
                First-stage semantic retrieval size.

            similarity_threshold:
                Minimum final score required for a document
                to remain in the final result.
        """

        print(f"Loading embedding model: {model_name}...")

        self.model = SentenceTransformer(model_name)

        self.documents = []
        self.embeddings = None

        self.candidate_k = candidate_k
        self.similarity_threshold = similarity_threshold

    # =========================================================
    # Document indexing
    # =========================================================

    def add_documents(self, docs: list[str]):
        """
        將文件切片轉換成 normalized embedding vectors。
        """

        self.documents = docs

        print(
            f"Encoding {len(docs)} documents into vectors..."
        )

        self.embeddings = self.model.encode(
            docs,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    # =========================================================
    # Keyword extraction
    # =========================================================

    def _extract_keywords(self, query: str) -> list[str]:
        """
        擷取 query 中具有辨識度的技術關鍵詞。

        優先保留：
        - GPU 型號
        - RAM / VRAM 規格
        - CPU / GPU 型號
        - Thunderbolt / USB / HDMI
        - Wi-Fi / Bluetooth
        - PCIe / M.2
        - 型號 BZH / BYH / BXH
        - 數字規格
        """

        query_lower = query.lower()

        keywords = []

        # -----------------------------------------------------
        # 1. Technical patterns
        # -----------------------------------------------------

        for pattern in self.TECHNICAL_PATTERNS:
            matches = re.findall(
                pattern,
                query_lower,
            )

            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]

                match = match.strip()

                if match:
                    keywords.append(match)

        # -----------------------------------------------------
        # 2. English technical words
        # -----------------------------------------------------

        english_words = re.findall(
            r"[a-zA-Z][a-zA-Z0-9\-\.]+",
            query_lower,
        )

        for word in english_words:
            if word in self.STOPWORDS:
                continue

            # Avoid adding very short generic words.
            if len(word) <= 2:
                continue

            keywords.append(word)

        # -----------------------------------------------------
        # 3. Remove duplicates
        # -----------------------------------------------------

        keywords = list(dict.fromkeys(keywords))

        return keywords

    # =========================================================
    # Domain detection
    # =========================================================

    def _detect_domains(self, query: str) -> set[str]:
        query_lower = query.lower()

        domains = set()

        for domain, terms in self.DOMAIN_TERMS.items():

            for term in terms:

                term = term.lower()

                # Technical terms
                if any(ch in term for ch in ["-", ".", " ", "/"]):
                    matched = term in query_lower

                # Normal words
                else:
                    matched = bool(
                        re.search(
                            rf"\b{re.escape(term)}\b",
                            query_lower
                        )
                    )

                if matched:
                    domains.add(domain)
                    break

        # Explicit GPU memory
        if any(
            term in query_lower
            for term in [
                "gpu",
                "vram",
                "gddr",
                "graphics memory",
            ]
        ):
            domains.add("gpu")

        # Explicit RAM
        if any(
            term in query_lower
            for term in [
                "ram",
                "ddr5",
                "ddr4",
                "so-dimm",
                "sodimm",
            ]
        ):
            domains.add("ram")

        return domains

    # =========================================================
    # Keyword score
    # =========================================================

    def _keyword_score(
        self,
        query: str,
        document: str,
    ) -> float:
        """
        Technical keyword matching score.

        Exact technical terms receive more importance than
        generic English words.
        """

        keywords = self._extract_keywords(query)

        if not keywords:
            return 0.0

        document_lower = document.lower()

        score = 0.0
        total_weight = 0.0

        for keyword in keywords:

            # Technical terms receive higher weight.
            is_technical = any(
                re.fullmatch(pattern, keyword)
                for pattern in self.TECHNICAL_PATTERNS
            )

            weight = 2.0 if is_technical else 1.0

            total_weight += weight

            if keyword in document_lower:
                score += weight

        if total_weight == 0:
            return 0.0

        return score / total_weight

    # =========================================================
    # Domain score
    # =========================================================

    def _domain_score(
        self,
        query: str,
        document: str,
    ) -> float:
        """
        Domain-aware scoring.

        Positive:
            query domain matches document domain.

        Negative:
            query asks RAM but document mainly contains GPU VRAM,
            etc.
        """

        query_domains = self._detect_domains(query)

        if not query_domains:
            return 0.0

        document_lower = document.lower()

        document_domains = set()

        for domain, terms in self.DOMAIN_TERMS.items():

            for term in terms:

                if term in document_lower:
                    document_domains.add(domain)
                    break

        if not document_domains:
            return 0.0

        # -----------------------------------------------------
        # Positive domain match
        # -----------------------------------------------------

        matched_domains = (
            query_domains & document_domains
        )

        positive_score = len(matched_domains) / len(
            query_domains
        )

        # -----------------------------------------------------
        # Domain mismatch penalties
        # -----------------------------------------------------

        penalty = 0.0

        # RAM vs GPU memory is the most important distinction
        # for the current dataset.
        if "ram" in query_domains and "gpu" in document_domains:
            if "ram" not in document_domains:
                penalty += 0.50

        if "gpu" in query_domains and "ram" in document_domains:
            if "gpu" not in document_domains:
                penalty += 0.50

        # Display queries should not strongly prefer
        # unrelated hardware chunks.
        if "display" in query_domains:
            if (
                "cpu" in document_domains
                or "battery" in document_domains
            ):
                if "display" not in document_domains:
                    penalty += 0.20

        # Battery queries should not prefer GPU specs.
        if "battery" in query_domains:
            if "gpu" in document_domains:
                if "battery" not in document_domains:
                    penalty += 0.30

        return positive_score - penalty

    # =========================================================
    # Semantic similarity
    # =========================================================

    def _semantic_search(
        self,
        query_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate cosine similarity.

        Since both document and query embeddings are normalized,
        dot product == cosine similarity.
        """

        return np.dot(
            self.embeddings,
            query_embedding,
        )

    # =========================================================
    # Search
    # =========================================================

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[str]:
        """
        Two-stage hybrid retrieval.

        Stage 1:
            Semantic retrieval -> candidate_k

        Stage 2:
            Re-ranking using:
                - semantic similarity
                - technical keyword score
                - domain score

        Final:
            threshold filtering
            + top_k
        """

        if (
            self.embeddings is None
            or len(self.documents) == 0
        ):
            return []

        # =====================================================
        # 1. Query embedding
        # =====================================================

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        # =====================================================
        # 2. Semantic similarity
        # =====================================================

        semantic_scores = self._semantic_search(
            query_embedding
        )

        # =====================================================
        # 3. First-stage candidate retrieval
        # =====================================================

        candidate_k = min(
            self.candidate_k,
            len(self.documents),
        )

        candidate_indices = np.argsort(
            semantic_scores
        )[::-1][:candidate_k]

        # =====================================================
        # 4. Calculate hybrid scores
        # =====================================================

        results = []

        for idx in candidate_indices:

            document = self.documents[idx]

            semantic_score = float(
                semantic_scores[idx]
            )

            keyword_score = self._keyword_score(
                query,
                document,
            )

            domain_score = self._domain_score(
                query,
                document,
            )

            # -------------------------------------------------
            # Final score
            #
            # Semantic:
            #     0.65
            #
            # Keyword:
            #     0.20
            #
            # Domain:
            #     0.15
            # -------------------------------------------------

            final_score = (
                0.55 * semantic_score
                + 0.30 * keyword_score
                + 0.15 * domain_score
            )

            results.append(
                {
                    "index": idx,
                    "document": document,
                    "semantic": semantic_score,
                    "keyword": keyword_score,
                    "domain": domain_score,
                    "final": final_score,
                }
            )

        # =====================================================
        # 5. Sort by final score
        # =====================================================

        results.sort(
            key=lambda x: x["final"],
            reverse=True,
        )

        # =====================================================
        # 6. Similarity threshold
        # =====================================================

        filtered_results = [
            result
            for result in results
            if result["final"]
            >= self.similarity_threshold
        ]

        # =====================================================
        # 7. Final Top-K
        # =====================================================

        final_results = filtered_results[:top_k]

        # =====================================================
        # 8. Debug output
        # =====================================================

        print("\n" + "=" * 70)
        print("[Hybrid Vector Search]")
        print("=" * 70)

        print(f"Query: {query}")

        print(
            f"Detected domains: "
            f"{sorted(self._detect_domains(query))}"
        )

        print(
            f"Keywords: "
            f"{self._extract_keywords(query)}"
        )

        print(
            f"Candidates: {len(candidate_indices)}"
        )

        print(
            f"Final results: {len(final_results)}"
        )

        print("-" * 70)

        for rank, result in enumerate(
            final_results,
            start=1,
        ):

            print(
                f"{rank}. "
                f"final={result['final']:.4f} "
                f"| semantic={result['semantic']:.4f} "
                f"| keyword={result['keyword']:.4f} "
                f"| domain={result['domain']:.4f}"
            )

            print(
                f"   {result['document']}"
            )

        print("=" * 70)

        # =====================================================
        # 9. Return documents
        # =====================================================

        return [
            result["document"]
            for result in final_results
        ]
