from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
import threading
from typing import List, Optional

# Re-use the engine loading logic from chat.py
# We import specific functions or classes if possible, or just copy the setup logic to avoid CLI argument conflicts
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import LLMMetadata
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
import chromadb
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Epstein Chat API")

# Configuration (mirrors chat.py)
DATA_DIR = os.getenv("OCR_DATA_DIR", "data")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "epstein_docs")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://ollama:11434/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "llama3") # Default to a smaller model for local docker
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "600"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "0"))
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", str(LLM_CONTEXT_WINDOW)))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "query_fusion")
FUSION_NUM_QUERIES = int(os.getenv("FUSION_NUM_QUERIES", "4"))
BM25_START_PAGE = int(os.getenv("BM25_START_PAGE", "0")) # Simplified defaults
BM25_END_PAGE = int(os.getenv("BM25_END_PAGE", "999999"))
BM25_MIN_SIZE = int(os.getenv("BM25_MIN_SIZE", "500"))
BM25_CHUNK_SIZE = int(os.getenv("BM25_CHUNK_SIZE", "512"))
BM25_CHUNK_OVERLAP = int(os.getenv("BM25_CHUNK_OVERLAP", "64"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "8"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "12"))
FUSION_TOP_K = int(os.getenv("FUSION_TOP_K", "12"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
CITATION_CHUNK_SIZE = int(os.getenv("CITATION_CHUNK_SIZE", "256"))
SOURCE_TOP_K = int(os.getenv("SOURCE_TOP_K", "5"))

# Global query engine
query_engine = None
engine_load_error: Optional[str] = None
engine_loading: bool = False

class ChatRequest(BaseModel):
    query: str

class SourceNode(BaseModel):
    file_name: str
    page_label: str
    text_preview: str
    score: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[SourceNode]

def get_page_metadata(filename):
    try:
        base = os.path.basename(filename)
        # Attempt to parse page number if format permits, otherwise just filename
        if "page_" in base:
            page_num = base.split('page_')[1].split('.')[0].split('_')[0] 
            return {"page_label": page_num, "file_name": base}
        return {"file_name": base}
    except Exception:
        return {"file_name": filename}

class LocalOpenAI(OpenAI):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=LLM_CONTEXT_WINDOW,
            num_output=self.max_tokens or 2048,
            is_chat_model=True,
            model_name=self.model,
        )

def load_engine():
    logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    logger.info(f"Connecting to LLM at: {VLLM_API_BASE} with model: {VLLM_MODEL_NAME}...")
    llm = LocalOpenAI(
        api_base=VLLM_API_BASE,
        api_key="EMPTY",
        model=VLLM_MODEL_NAME,
        temperature=0.1,
        max_tokens=LLM_MAX_TOKENS,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
        additional_kwargs={
            # Ollama OpenAI-compat uses `options` in the request JSON body.
            # With openai-python v2.x, this must be passed via `extra_body`.
            "extra_body": {"options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": LLM_MAX_TOKENS}},
        },
    )
    Settings.llm = llm

    logger.info(f"Loading Vector Store from: {PERSIST_DIR}...")
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Check if index exists
    try:
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise RuntimeError("Vector store not ready. Did ingestion run?")

    # Build BM25 Index
    logger.info("Building BM25 Index...")
    all_files = glob.glob(os.path.join(DATA_DIR, "**/*.md"), recursive=True)
    # Filter valid files
    filtered_files = [f for f in all_files if os.path.isfile(f) and os.path.getsize(f) >= BM25_MIN_SIZE]
    
    if not filtered_files:
        logger.warning("No files found for BM25 index! Retrieval might be degraded.")
        bm25_retriever = None
    else:
        documents = SimpleDirectoryReader(input_files=filtered_files, file_metadata=get_page_metadata).load_data()
        # Important: Sentence-level chunking is critical. The OCR markdown pages can be very large,
        # and passing entire pages into BM25 (and later into the LLM prompt) can lead to extremely
        # slow inference on CPU and frequent request timeouts.
        splitter = SentenceSplitter(chunk_size=BM25_CHUNK_SIZE, chunk_overlap=BM25_CHUNK_OVERLAP)
        bm25_nodes = splitter.get_nodes_from_documents(documents)
        bm25_retriever = BM25Retriever.from_defaults(nodes=bm25_nodes, similarity_top_k=BM25_TOP_K)

    vector_retriever = vector_index.as_retriever(similarity_top_k=VECTOR_TOP_K)
    
    # Hybrid Setup
    if bm25_retriever:
        if RETRIEVAL_MODE == "rrf" or FUSION_NUM_QUERIES <= 1:
             hybrid_retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                num_queries=1,
                similarity_top_k=FUSION_TOP_K,
                mode="reciprocal_rerank",
                use_async=False,
            )
        else:
            hybrid_retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                num_queries=FUSION_NUM_QUERIES,
                similarity_top_k=FUSION_TOP_K,
                mode="reciprocal_rerank",
                use_async=False,
            )
    else:
        logger.warning("Falling back to Vector-only retriever.")
        hybrid_retriever = vector_retriever

    logger.info("Setup Reranker...")
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=RERANK_TOP_N)

    engine = CitationQueryEngine.from_args(
        vector_index,
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],
        similarity_top_k=FUSION_TOP_K,
        citation_chunk_size=CITATION_CHUNK_SIZE,
    )
    return engine

@app.on_event("startup")
async def startup_event():
    """Start engine loading in a background thread.

    Loading embeddings + connecting to the LLM can take minutes on first start.
    We keep the HTTP server responsive and return 503 from /chat until ready.
    """

    global engine_loading

    def _load() -> None:
        global query_engine, engine_load_error, engine_loading
        engine_loading = True
        engine_load_error = None
        try:
            query_engine = load_engine()
            logger.info("Query Engine loaded successfully.")
        except Exception as e:
            engine_load_error = str(e)
            logger.exception("Query Engine failed to load")
        finally:
            engine_loading = False

    # Avoid starting multiple background threads if reload happens.
    if query_engine is None and not engine_loading:
        threading.Thread(target=_load, daemon=True).start()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not query_engine:
        detail = {
            "message": "Query engine not initialized yet",
            "loading": engine_loading,
            "error": engine_load_error,
        }
        raise HTTPException(status_code=503, detail=detail)
    
    try:
        response = query_engine.query(request.query)
        
        sources = []
        for node in response.source_nodes[:SOURCE_TOP_K]:
            sources.append(SourceNode(
                file_name=node.node.metadata.get("file_name", "Unknown"),
                page_label=node.node.metadata.get("page_label", "Unknown"),
                text_preview=node.node.get_content()[:200],
                score=node.score
            ))
            
        return ChatResponse(
            response=str(response),
            sources=sources
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine_loaded": query_engine is not None,
        "engine_loading": engine_loading,
        "engine_error": engine_load_error,
    }
