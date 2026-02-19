# Epstein Documents Analysis System

> [!NOTE]
> **For investigators, journalists, NGOs, and independent researchers**
>
> We’ve also developed a **convenient web frontend** for people who prefer not to run the system locally.
> To balance demand, access is **provided on request**.

## Overview
This repository provides a high-precision interface for querying the Epstein Documents dataset. Our approach is engineered to solve the specific challenges of information retrieval within large-scale, noisy OCR datasets where accuracy and auditability are paramount.

## Technical Architecture & Problem Solving
The primary challenge of this dataset is the high density of specific entities - such as phone numbers, flight tail numbers, and timestamps - buried within unstructured, extracted text from scanned documents. Standard vector-only search often fails to capture the granularity required for these "needle-in-a-haystack" investigative queries. 

We solved this by implementing an entity-aware ingestion pipeline and a sophisticated 3-stage retrieval architecture:

### 0. Intelligent Metadata Extraction
Before indexing, the system conducts automated metadata extraction to structure the unstructured OCR text:
- **Entity Identification**: Utilizing a combination of specialized Regex patterns and a BERT-based Named Entity Recognition (NER) model (`dslim/bert-base-NER`) to identify phone numbers, email addresses, flight tail numbers, dates, and capitalized names/locations.
- **Enriched Indexing**: This extracted metadata is injected into the document chunks, ensuring that specific, high-value investigative tokens are prioritized during both semantic and lexical search stages.

### 1. Hybrid Retrieval Strategy (Vector + Keyword)
To maximize recall, the system employs a dual-stream search mechanism:
- **Semantic Search**: Utilizing `BAAI/bge-m3` embeddings to capture the conceptual context and intent of user queries.
- **Lexical Matching**: Utilizing a `BM25` retriever to handle specific numeric and entity-based queries (e.g., flight logs or phone records) where exact string matching is superior.
- **Reciprocal Rank Fusion (RRF)**: Results from both streams are fused using RRF to prioritize documents that demonstrate strength across both semantic and lexical domains.

### 2. Neural Reranking (Cross-Encoder)
Initial retrieval candidates often contain noise due to OCR artifacts. We utilize the `BAAI/bge-reranker-base` cross-encoder to perform a secondary, deep-learning-based relevance check on the top 50 candidates. This process effectively identifies the most relevant context and pushes it to the top of the context window, significantly improving the system's precision and the subsequent LLM generation.

### 3. Page-Level Auditing & Citations
Transparency is critical in legal and investigative document analysis. Our implementation uses a specialized `CitationQueryEngine` that provides verifiable page-level evidence for every assistant response, allowing users to cross-reference findings directly with the source materials.

## Performance Benchmarks
The architecture was rigorously tested using a synthetic evaluation suite:
- **Hit Rate (Top-5)**: **81.99%**
- **Mean Reciprocal Rank (MRR)**: **0.7110**

These metrics indicate that correct responses appear in the first or second result position in the vast majority of queries, demonstrating a robust foundation for further analysis and investigation.

## Quickstart with Docker

This repo includes a **working Docker Compose setup** (FastAPI app + Ollama) intended as the simplest way to run the system locally.

**Prerequisites**
- Docker Desktop / Docker Engine

**Start the stack**
```bash
# from repo root
cd epstein-chat

# start Ollama + the API
docker compose up -d --build

# watch startup logs (first run downloads HF embedding + reranker models)
docker compose logs -f app
```

**Health check + sample query**
```bash
curl -s http://localhost:8000/health | jq .

curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"Who Censored Roger Rabbit?"}' \
  | jq .
```

## Manual run

### 1. Download the Dataset
We use the [**Epstein Files OCR Dataset**](https://huggingface.co/datasets/ishumilin/epstein-files-ocr-datasets-1-8-early-release).

OCR was performed using a proprietary model provided by [Wild Ma-Gässli](https://wildma.ch).

```bash
huggingface-cli download ishumilin/epstein-files-ocr-datasets-1-8-early-release --repo-type dataset --local-dir data --local-dir-use-symlinks False
```

### 2. Evaluation Pipeline (Optional)
To verify the retrieval accuracy before running the full chat system, you can ingest and evaluate a subset of documents.

**Step 2a: Ingest Evaluation Data**
This indexes only the documents required for the evaluation dataset.
```bash
# Ensure you are in the epstein-chat directory
cd epstein-chat

# Ingest eval subset (pointing to your downloaded data)
python eval/ingest_eval_dataset.py --data-dir ../data --recursive
```

**Step 2b: Run Evaluation**
Run the evaluation script to measure Hit Rate and MRR.
```bash
python eval/eval_run.py --data-dir ../data --recursive
```
*Results will be saved to `eval_results.csv`.*

### 3. Full System Ingestion
To use the interactive chat with the entire dataset, you must ingest all documents.
*Note: This process can take a significant amount of time depending on your hardware.*

```bash
# Ingest all data (pointing to your downloaded data)
python ingest.py --data-dir ../data --recursive
```

### 4. Interactive Chat
Once ingestion is complete, you can start the chat interface.

**CLI Mode:**
```bash
python chat.py
```
*Or for a single query:*
```bash
python chat.py "Who Censored Roger Rabbit?"
```

## Configuration
The system uses environment variables for configuration. You can set these in your shell or a `.env` file.

> [!NOTE]
> **Dataset licensing:** The OCR dataset is downloaded separately from Hugging Face and remains subject to the dataset’s own license/terms.
> This repository’s MIT license applies to the source code only.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `OCR_DATA_DIR` | Path to the folder containing markdown files | `data` |
| `CHROMA_PERSIST_DIR` | Path to store the Vector Database | `./chroma_db` |
| `VLLM_API_BASE` | URL of the LLM API (OpenAI compatible) | `http://ollama:11434/v1` |
| `VLLM_MODEL_NAME` | Name of the LLM model to use | `llama3` |