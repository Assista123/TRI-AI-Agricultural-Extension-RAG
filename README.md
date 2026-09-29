# TRI-AI-Agricultural-Extension-RAG
Hybrid RAG system for agricultural extension queries | Team Tsavo (TRI AI Cohort 10 with Google DeepMind) | 9th of 25 

# Agricultural Extension RAG – Smart Retrieval for Farmers

**Team Tsavo** · TRI AI Saturday Cohort 10 (AI Research Foundations with Google DeepMind)  
**Result:** 9th out of 25 teams

## Problem
Agricultural extension officers and farmers need fast, reliable answers drawn from technical guidance documents. The challenge required building a retrieval system that returns the most relevant documents for natural-language queries about crop health, pests, nutrients, and agronomic practices.

## Approach
We built a hybrid retrieval pipeline that combines lexical and dense search, then re-ranks candidates with cross-encoders.

### Pipeline
1. **Query expansion** – Domain-specific agricultural dictionary (symptoms, crops, nutrients, pests) to enrich short farmer-style queries.
2. **HyDE** – Generate a hypothetical technical passage from the expanded query to improve dense retrieval.
3. **Lexical retrieval** – BM25 (custom Okapi implementation) over the full document corpus.
4. **Dense retrieval** – BAAI/bge-large-en-v1.5 embeddings with cosine similarity.
5. **Fusion** – Reciprocal Rank Fusion (RRF) of the top lexical and dense results.
6. **Reranking** – Dual cross-encoder ensemble (BGE-reranker-v2-m3 + mxbai-rerank-large-v1) to produce the final top-10 ranking.

## Corpus
- 695 agricultural extension documents
- 200 test queries
- Evaluation via competition submission format (QueryId → ranked DocumentIds)

## Tech Stack
- Python, PyTorch
- sentence-transformers (BGE large + cross-encoders)
- Custom BM25
- pandas, NumPy

## Role
Team Leader – Team Tsavo. Coordinated pipeline design, retrieval experiments, and final submission.

## Key Design Choices
- Agricultural query expansion to bridge everyday farmer language and technical documents
- HyDE to improve dense retrieval on short queries
- RRF instead of simple score combination for more stable fusion
- Dual cross-encoder ensemble for more robust final ranking

## Results
Placed **9th out of 25** teams in the TRI AI Saturday Cohort 10 competition.

## Repository Contents
- Notebook / scripts for the full retrieval pipeline
- Query expansion dictionary
- Submission generation code

## Acknowledgements
TRI AI Saturday Cohort 10 · AI Research Foundations with Google DeepMind
