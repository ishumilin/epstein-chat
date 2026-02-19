import os
import glob
import argparse
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import LLMMetadata
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank

# Configuration
DATA_DIR = os.getenv("OCR_DATA_DIR", "data")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "epstein_docs")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://10.10.10.1:12345/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "deepseek-ai/DeepSeek-R1")
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "query_fusion")
FUSION_NUM_QUERIES = int(os.getenv("FUSION_NUM_QUERIES", "4"))
BM25_START_PAGE = int(os.getenv("BM25_START_PAGE", "9900"))
BM25_END_PAGE = int(os.getenv("BM25_END_PAGE", "10100"))
BM25_MIN_SIZE = int(os.getenv("BM25_MIN_SIZE", "500"))

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

def load_query_engine():
    # 1. Setup Embeddings
    print(f"Loading embedding model: {EMBED_MODEL_NAME}...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    # 2. Setup LLM
    print(f"Connecting to LLM at: {VLLM_API_BASE}...")
    llm = LocalOpenAI(
        api_base=VLLM_API_BASE,
        api_key="EMPTY",
        model=VLLM_MODEL_NAME,
        temperature=0.1,
        max_tokens=4096,
    )
    Settings.llm = llm

    # 3. Load Vector Store
    print(f"Loading Vector Store from: {PERSIST_DIR}...")
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )

    # 4. Build BM25 Index
    print("Building BM25 Index...")
    all_files = glob.glob(os.path.join(DATA_DIR, "page_*.md"))
    def _page_id(f):
        try:
            return int(os.path.basename(f).split('_')[1].split('.')[0])
        except Exception:
            return -1
    filtered_files = [
        f for f in all_files
        if BM25_START_PAGE <= _page_id(f) <= BM25_END_PAGE and os.path.getsize(f) >= BM25_MIN_SIZE
    ]
    documents = SimpleDirectoryReader(input_files=filtered_files, file_metadata=get_page_metadata).load_data()
    bm25_retriever = BM25Retriever.from_defaults(nodes=documents, similarity_top_k=5)

    # 5. Setup Hybrid Retriever
    vector_retriever = vector_index.as_retriever(similarity_top_k=50)
    if RETRIEVAL_MODE == "rrf" or FUSION_NUM_QUERIES <= 1:
        hybrid_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            num_queries=1,
            similarity_top_k=50,
            mode="reciprocal_rerank",
            use_async=False,
        )
    else:
        hybrid_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            num_queries=FUSION_NUM_QUERIES,
            similarity_top_k=50,
            mode="reciprocal_rerank",
            use_async=False,
        )

    # 6. Setup Reranker
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=5)

    # 7. Create Query Engine
    query_engine = CitationQueryEngine.from_args(
        vector_index,
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],
        similarity_top_k=50,
        citation_chunk_size=256,
    )
    return query_engine

def print_response(response):
    print("\n" + "="*50)
    print("ASSISTANT RESPONSE:")
    print("-" * 50)
    print(str(response))
    print("-" * 50)
    print("SOURCES:")
    seen_pages = set()
    for node in response.source_nodes:
        page = node.node.metadata.get("page_label", "Unknown")
        file_name = node.node.metadata.get("file_name", "Unknown")
        text_preview = node.node.get_content()[:200].replace("\n", " ")
        if page not in seen_pages:
            print(f"- Page {page} ({file_name}): {text_preview}...")
            seen_pages.add(page)
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Epstein Documents CLI Chat")
    parser.add_argument("query", nargs="?", help="A single question to ask the system")
    args = parser.parse_args()

    try:
        query_engine = load_query_engine()
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    if args.query:
        # Single query mode
        print(f"\nQUERY: {args.query}")
        response = query_engine.query(args.query)
        print_response(response)
    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_query = input("\nQ: ").strip()
                if not user_query:
                    continue
                if user_query.lower() in ["exit", "quit"]:
                    break
                
                print("Thinking...")
                response = query_engine.query(user_query)
                print_response(response)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()