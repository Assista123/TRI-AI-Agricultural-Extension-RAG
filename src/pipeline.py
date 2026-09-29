"""
Hybrid RAG pipeline:
BM25 + dense (BGE) → RRF → dual cross-encoder reranking.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from .bm25 import BM25Okapi
from .query_expansion import expand_agri_query, generate_hyde_doc


class HybridRAGPipeline:
    def __init__(
        self,
        documents: list[str],
        device: str | None = None,
        top_lexical: int = 60,
        top_dense: int = 60,
        top_fused: int = 30,
        top_final: int = 10,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_lexical = top_lexical
        self.top_dense = top_dense
        self.top_fused = top_fused
        self.top_final = top_final
        self.documents = documents

        # Lexical index
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Dense retriever + rerankers
        self.retriever = SentenceTransformer(
            "BAAI/bge-large-en-v1.5", device=self.device
        )
        self.reranker_bge = CrossEncoder(
            "BAAI/bge-reranker-v2-m3", device=self.device
        )
        self.reranker_mxbai = CrossEncoder(
            "mixedbread-ai/mxbai-rerank-large-v1", device=self.device
        )

        # Pre-compute document embeddings
        with torch.autocast(
            device_type=self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ):
            self.doc_embeddings = self.retriever.encode(
                documents,
                batch_size=32 if self.device == "cuda" else 16,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

        self.bge_prefix = "Represent this sentence for searching relevant passages: "

    def retrieve(self, query: str) -> list[int]:
        """Return indices of the top-k documents for a single query."""
        expanded_q = expand_agri_query(query)
        hyde_passage = generate_hyde_doc(query, expanded_q)

        # 1. Lexical
        bm25_scores = self.bm25.get_scores(expanded_q.lower().split())
        top_bm25 = np.argsort(bm25_scores)[::-1][: self.top_lexical]

        # 2. Dense (HyDE)
        formatted = self.bge_prefix + hyde_passage
        q_emb = self.retriever.encode(
            formatted, convert_to_tensor=True, normalize_embeddings=True
        )
        dense_scores = torch.matmul(self.doc_embeddings, q_emb).cpu().numpy()
        top_dense = np.argsort(dense_scores)[::-1][: self.top_dense]

        # 3. RRF
        rrf_scores = {}
        for rank, idx in enumerate(top_bm25):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank + 1)
        for rank, idx in enumerate(top_dense):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank + 1)

        candidates = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )[: self.top_fused]

        # 4. Dual cross-encoder rerank
        pairs = [[query, self.documents[i]] for i in candidates]
        with torch.autocast(
            device_type=self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ):
            scores_bge = self.reranker_bge.predict(
                pairs, batch_size=32, show_progress_bar=False
            )
            scores_mxbai = self.reranker_mxbai.predict(
                pairs, batch_size=32, show_progress_bar=False
            )

        norm_bge = (scores_bge - np.min(scores_bge)) / (np.ptp(scores_bge) + 1e-8)
        norm_mxbai = (scores_mxbai - np.min(scores_mxbai)) / (
            np.ptp(scores_mxbai) + 1e-8
        )
        ensemble = 0.5 * norm_bge + 0.5 * norm_mxbai

        top_offsets = np.argsort(ensemble)[::-1][: self.top_final]
        return [candidates[i] for i in top_offsets]
