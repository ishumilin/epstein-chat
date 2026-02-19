import json
import argparse
import pandas as pd
import os
import glob
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import LLMMetadata
from llama_index.core.evaluation import RetrieverEvaluator, HitRate, MRR
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import QueryBundle
from llama_index.core.schema import NodeWithScore
import chromadb

# Configuration
# Defaults are for local development; override via environment variables in Docker/remote.
EVAL_FILE = os.getenv("EVAL_FILE", "eval_dataset.json")
DATA_DIR = os.getenv("OCR_DATA_DIR", "../data")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "../chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "epstein_docs")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://10.10.10.1:12345/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "deepseek-ai/DeepSeek-R1")


def _index_md_files_by_basename(data_dir: str, *, pattern: str = "*.md") -> dict[str, str]:
    """Recursively index markdown files by basename.

    Needed for HF layout: pages/<range>/page_<N>.md while eval references just page_<N>.md.
    """
    idx: dict[str, str] = {}
    for p in Path(data_dir).rglob(pattern):
        idx.setdefault(p.name, str(p))
    return idx


def _node_key(nws: NodeWithScore) -> str:
    """Stable-ish key for merging nodes coming from multiple retrievers."""
    node = nws.node
    node_id = getattr(node, "node_id", None)
    if node_id:
        return str(node_id)
    md = getattr(node, "metadata", None) or {}
    file_name = md.get("file_name") or ""
    page_label = md.get("page_label") or ""
    # Avoid hashing entire content; a short prefix is usually enough.
    content_prefix = (node.get_content() or "")[:200]
    return f"{file_name}|{page_label}|{hash(content_prefix)}"


def _rrf_fuse(
    ranked_lists: list[list[NodeWithScore]],
    *,
    k: int = 60,
    top_k: int = 50,
) -> list[NodeWithScore]:
    """Reciprocal Rank Fusion without any LLM query expansion.

    This lets evaluation run even when no OpenAI-compatible endpoint is available.
    """
    merged: dict[str, tuple[NodeWithScore, float]] = {}

    for nodes in ranked_lists:
        for rank, nws in enumerate(nodes):
            key = _node_key(nws)
            score = 1.0 / (k + rank + 1)
            if key in merged:
                existing_nws, existing_score = merged[key]
                merged[key] = (existing_nws, existing_score + score)
            else:
                merged[key] = (nws, score)

    fused = [NodeWithScore(node=nws.node, score=score) for (nws, score) in merged.values()]
    fused.sort(key=lambda x: (x.score or 0.0), reverse=True)
    return fused[:top_k]


def _normalize_source_files(item: dict) -> list[str]:
    """Support both legacy single-label datasets and current multi-label datasets."""
    if "source_files" in item and item["source_files"]:
        return item["source_files"]
    if "source_file" in item and item["source_file"]:
        return [item["source_file"]]
    return []
def _build_bm25_corpus_files(
    *,
    data_dir: str,
    eval_items: list[dict],
    include_eval_sources_only: bool,
    min_size: int,
    recursive: bool,
) -> list[str]:
    """Return a list of markdown file paths used for the BM25 corpus."""
    if include_eval_sources_only:
        required = sorted({
            src
            for item in eval_items
            for src in _normalize_source_files(item)
        })
        filepaths: list[str] = []

        # Fast path (flat layout)
        missing: list[str] = []
        for fname in required:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath) and os.path.getsize(fpath) >= min_size:
                filepaths.append(fpath)
            else:
                missing.append(fname)

        # HF path layout fallback
        if missing and recursive:
            basename_to_path = _index_md_files_by_basename(data_dir, pattern="page_*.md")
            for fname in missing:
                fpath = basename_to_path.get(fname)
                if fpath and os.path.exists(fpath) and os.path.getsize(fpath) >= min_size:
                    filepaths.append(fpath)

        return filepaths

    # Fallback: build BM25 on the entire OCR directory.
    if recursive:
        all_files = [str(p) for p in Path(data_dir).rglob("page_*.md")]
    else:
        all_files = glob.glob(os.path.join(data_dir, "page_*.md"))
    return [f for f in all_files if os.path.getsize(f) >= min_size]

def get_page_metadata(filename):
    try:
        base = os.path.basename(filename)
        page_num = base.split('_')[1].split('.')[0]
        return {"page_label": page_num, "file_name": base}
    except Exception:
        return {"file_name": filename}

class LocalOpenAI(OpenAI):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=32768,
            num_output=self.max_tokens or 2048,
            is_chat_model=True,
            model_name=self.model,
        )

def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation.")
    parser.add_argument("--eval-file", default=EVAL_FILE, help="JSON file with generated questions")
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Directory containing OCR markdown files (HF layout supported with --recursive)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover page_*.md (needed for HF pages/<range>/ layout).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of eval questions (smoke testing / quick runs).",
    )

    parser.add_argument(
        "--retrieval-mode",
        default=os.getenv("EVAL_RETRIEVAL_MODE", "rrf"),
        choices=["rrf", "query_fusion", "vector", "bm25"],
        help=(
            "Retrieval strategy. 'rrf' is LLM-free reciprocal-rank fusion (recommended for evaluation). "
            "'query_fusion' uses LlamaIndex QueryFusionRetriever and may require an LLM endpoint for query expansion."
        ),
    )

    parser.add_argument(
        "--vector-top-k",
        type=int,
        default=int(os.getenv("EVAL_VECTOR_TOP_K", "50")),
        help="How many candidates to retrieve from the vector index.",
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=int(os.getenv("EVAL_BM25_TOP_K", "50")),
        help="How many candidates to retrieve from BM25.",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=int(os.getenv("EVAL_CANDIDATE_TOP_K", "50")),
        help="How many fused candidates to keep before reranking.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=int(os.getenv("EVAL_RRF_K", "60")),
        help="RRF constant k. Larger k makes rank positions matter less.",
    )
    parser.add_argument(
        "--fusion-num-queries",
        type=int,
        default=int(os.getenv("FUSION_NUM_QUERIES", "4")),
        help="Only used for --retrieval-mode=query_fusion. num_queries>1 enables LLM-based query expansion.",
    )

    parser.add_argument(
        "--bm25-eval-only",
        action="store_true",
        help="Build BM25 corpus only from the pages referenced by the eval dataset (recommended).",
    )
    parser.add_argument(
        "--bm25-min-size",
        type=int,
        default=int(os.getenv("BM25_MIN_SIZE", "0")),
        help="Minimum file size (bytes) for a page to be included in the BM25 corpus.",
    )
    args = parser.parse_args()

    # Load resources (same as app.py)
    print("Loading Index...", flush=True)
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    # Only initialize the LLM when explicitly requested.
    # (Evaluation should be able to run without any LLM dependency.)
    if args.retrieval_mode == "query_fusion" and args.fusion_num_queries > 1:
        llm = LocalOpenAI(
            api_base=VLLM_API_BASE,
            api_key="EMPTY",
            model=VLLM_MODEL_NAME,
        )
        Settings.llm = llm

    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    
    vector_retriever = vector_index.as_retriever(similarity_top_k=args.vector_top_k)

    # Load Questions
    with open(args.eval_file, "r") as f:
        eval_data = json.load(f)

    if args.limit is not None:
        eval_data = eval_data[: args.limit]
        print(f"Limiting evaluation to {len(eval_data)} questions.", flush=True)

    # Build BM25 (make corpus consistent with the eval dataset)
    print("Loading documents for BM25...")
    bm25_files = _build_bm25_corpus_files(
        data_dir=args.data_dir,
        eval_items=eval_data,
        include_eval_sources_only=args.bm25_eval_only,
        min_size=args.bm25_min_size,
        recursive=args.recursive,
    )
    if not bm25_files:
        raise RuntimeError(
            f"BM25 corpus is empty. DATA_DIR={args.data_dir!r}, eval_file={args.eval_file!r}. "
            "If running in Docker, ensure OCR_DATA_DIR is set to the mount (e.g. /ocr_output_vllm). "
            "If using HF nested pages/<range>/ layout, pass --recursive."
        )

    documents = SimpleDirectoryReader(input_files=bm25_files, file_metadata=get_page_metadata).load_data()
    
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=documents,
        similarity_top_k=args.bm25_top_k,
        stemmer=None,
        language="english",
    )

    retriever = None
    if args.retrieval_mode == "query_fusion":
        print("Setting up QueryFusionRetriever...")
        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            num_queries=args.fusion_num_queries,
            similarity_top_k=args.candidate_top_k,
            mode=FUSION_MODES.RECIPROCAL_RANK,
            use_async=False,
        )
    elif args.retrieval_mode == "vector":
        print("Using vector retriever only...")
        retriever = vector_retriever
    elif args.retrieval_mode == "bm25":
        print("Using BM25 retriever only...")
        retriever = bm25_retriever
    else:
        print("Using LLM-free RRF fusion (vector + BM25)...")

    print("Setting up Reranker...", flush=True)
    reranker = SentenceTransformerRerank(
        model="BAAI/bge-reranker-base", 
        top_n=5
    )

    print(f"Evaluating on {len(eval_data)} questions...", flush=True)
    
    # Simple Hit Rate / MRR Evaluation manually for now to avoid complex Ragas setup dependencies
    results = []
    
    for i, item in enumerate(eval_data):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(eval_data)}", flush=True)
        question = item["question"]
        expected_sources = _normalize_source_files(item)
        if not expected_sources:
            # If the dataset doesn't specify expected sources, we cannot score it.
            continue
        
        # 1. Retrieve
        if args.retrieval_mode == "rrf":
            vec_nodes = vector_retriever.retrieve(question)
            bm25_nodes = bm25_retriever.retrieve(question)
            nodes = _rrf_fuse(
                [vec_nodes, bm25_nodes],
                k=args.rrf_k,
                top_k=args.candidate_top_k,
            )
        else:
            nodes = retriever.retrieve(question)
        
        # 2. Rerank
        nodes = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(question))
        
        retrieved_files = [n.metadata.get("file_name") for n in nodes]
        
        # Check Hit (any correct source found)
        hit = any(source in retrieved_files for source in expected_sources)
        
        # Calculate Reciprocal Rank (of the first correct answer found)
        rank = float('inf')
        for source in expected_sources:
            if source in retrieved_files:
                r = retrieved_files.index(source) + 1
                if r < rank:
                    rank = r
        
        if rank == float('inf'):
            mrr = 0.0
        else:
            mrr = 1.0 / rank
            
        results.append({
            "question": question,
            "expected": expected_sources,
            "hit": hit,
            "mrr": mrr,
            "retrieved": retrieved_files
        })

    df = pd.DataFrame(results)
    print("\n--- Evaluation Results ---")
    print(f"Hit Rate (Top-5): {df['hit'].mean():.2%}")
    print(f"MRR (Top-5): {df['mrr'].mean():.4f}")
    
    df.to_csv("eval_results.csv", index=False)
    print("Detailed results saved to eval_results.csv")

if __name__ == "__main__":
    main()
