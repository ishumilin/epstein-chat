import os
import json
import chromadb
import argparse
from pathlib import Path
import time
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from transformers import pipeline

# NOTE: We keep this file self-contained but reuse the same metadata extraction
# approach as `ingest.py` so evaluation uses comparable document enrichment.
import re
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configuration
EVAL_FILE = os.getenv("EVAL_FILE", "eval/eval_dataset.json")
DATA_DIR = os.getenv("OCR_DATA_DIR", "../data")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "epstein_docs")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")


def extract_metadata_from_text(text: str, ner_pipe=None) -> dict:
    """Extract structured metadata from a page.

    This mirrors `ingest.py` (regex + optional BERT NER). Keep values small enough
    for LlamaIndex metadata limits.
    """

    metadata: dict = {}

    def _truncate_list(items, max_chars=400):
        s = ", ".join(sorted(list(set(items))))
        if len(s) > max_chars:
            return s[:max_chars] + "..."
        return s

    dates = re.findall(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text)
    word_dates = re.findall(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        text,
        re.IGNORECASE,
    )
    all_dates = dates + word_dates
    if all_dates:
        metadata["dates"] = _truncate_list(all_dates)

    phones = re.findall(r"\b(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b", text)
    if phones:
        metadata["phone_numbers"] = _truncate_list(phones)

    emails_from = re.findall(r"From:\s*(.*)", text, re.IGNORECASE)
    emails_to = re.findall(r"To:\s*(.*)", text, re.IGNORECASE)
    if emails_from:
        metadata["email_from"] = _truncate_list([e.strip() for e in emails_from if e.strip()])
    if emails_to:
        metadata["email_to"] = _truncate_list([e.strip() for e in emails_to if e.strip()])

    tail_numbers = re.findall(r"\bN\d{2,5}[A-Z]{1,2}\b", text)
    if tail_numbers:
        metadata["flight_tail_numbers"] = _truncate_list(tail_numbers)

    ssn_patterns = re.findall(r"\b\d{3}-\d{2}-\d{4}\b", text)
    if ssn_patterns:
        metadata["potential_ids"] = _truncate_list(ssn_patterns)

    # Locations/Names from tables
    table_cells = re.findall(r"\|(.*?)\|", text)
    potential_locs = set()
    for cell in table_cells:
        content = cell.strip()
        if not content:
            continue
        if content.upper() in [
            "DATE",
            "TIME",
            "NUMBER",
            "DURATION",
            "CITY",
            "STATE",
            "BILLED PHONE",
            "DEST NUMBER",
        ]:
            continue
        if re.match(r"^[A-Z\s,]+$", content) and len(content) > 3:
            potential_locs.add(content)

    if ner_pipe:
        try:
            entities = ner_pipe(text[:2000])
            names = set()
            orgs = set()
            locs = set()
            for ent in entities:
                if ent.get("entity_group") == "PER":
                    names.add(ent.get("word"))
                elif ent.get("entity_group") == "ORG":
                    orgs.add(ent.get("word"))
                elif ent.get("entity_group") == "LOC":
                    locs.add(ent.get("word"))

            if names:
                metadata["extracted_names"] = _truncate_list(names)
            if orgs:
                metadata["extracted_orgs"] = _truncate_list(orgs)
            if locs:
                potential_locs.update(locs)
        except Exception as e:
            print(f"NER Error: {e}")

    if potential_locs:
        metadata["locations"] = _truncate_list(potential_locs)

    return metadata

def _index_md_files_by_basename(data_dir: str) -> dict[str, str]:
    """Recursively index markdown files by basename.

    This is needed for the HF dataset layout where files are nested under
    `pages/<range>/page_<N>.md` but eval references use just `page_<N>.md`.
    """
    idx: dict[str, str] = {}
    root = Path(data_dir)
    for i, p in enumerate(root.rglob("*.md"), start=1):
        # Prefer the first occurrence; collisions are unlikely but possible.
        idx.setdefault(p.name, str(p))
        if i % 5000 == 0:
            print(f"  indexed {i} files...")
    return idx

def get_page_metadata(filename):
    """
    Extracts page number from filename (e.g., 'page_123.md' -> {'page_label': '123'})
    """
    try:
        base = os.path.basename(filename)
        page_num = base.split('_')[1].split('.')[0]
        return {"page_label": page_num, "file_name": base}
    except Exception:
        return {"file_name": filename}

def _normalize_source_files(item: dict) -> list[str]:
    """Support both legacy single-label datasets and current multi-label datasets."""
    if "source_files" in item and item["source_files"]:
        return item["source_files"]
    if "source_file" in item and item["source_file"]:
        return [item["source_file"]]
    return []

def main():
    parser = argparse.ArgumentParser(description="Ingest evaluation subset into ChromaDB.")
    parser.add_argument("--eval-file", default=EVAL_FILE, help="Path to eval dataset json")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory containing OCR markdown files")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover page_*.md under --data-dir (required for HF pages/<range>/ layout).",
    )
    parser.add_argument("--persist-dir", default=PERSIST_DIR, help="Directory to store ChromaDB")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Chroma collection name")
    parser.add_argument("--embed-model", default=EMBED_MODEL_NAME, help="Embedding model name")
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Limit number of resolved eval files to ingest (smoke test / debugging).",
    )
    args = parser.parse_args()

    t0 = time.time()
    print(f"[1/5] Loading evaluation dataset from {args.eval_file}...")
    try:
        with open(args.eval_file, "r") as f:
            eval_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Eval file not found at {args.eval_file}")
        return
    print(f"  loaded {len(eval_data)} eval items in {time.time() - t0:.2f}s")

    # Extract all unique source files needed for evaluation
    required_files = set()
    for item in eval_data:
        required_files.update(_normalize_source_files(item))
    
    sorted_files = sorted(list(required_files))
    print(f"[2/5] Found {len(sorted_files)} unique files referenced in evaluation dataset.")

    # Build file index if needed
    basename_to_path: dict[str, str] = {}
    if args.recursive:
        t1 = time.time()
        print(f"[3/5] Indexing markdown files under {args.data_dir} (recursive)...")
        basename_to_path = _index_md_files_by_basename(args.data_dir)
        print(f"  indexed {len(basename_to_path)} markdown files in {time.time() - t1:.2f}s")

    # Locate these files in DATA_DIR
    files_to_ingest = []
    missing_files = []
    
    print("[4/5] Resolving eval source files to local paths...")
    for i, fname in enumerate(sorted_files, start=1):
        if args.recursive:
            fpath = basename_to_path.get(fname)
            if fpath and os.path.exists(fpath):
                files_to_ingest.append(fpath)
            else:
                missing_files.append(fname)
        else:
            fpath = os.path.join(args.data_dir, fname)
            if os.path.exists(fpath):
                files_to_ingest.append(fpath)
            else:
                missing_files.append(fname)

        if i % 2000 == 0:
            print(f"  resolved {i}/{len(sorted_files)}...")
            
    if missing_files:
        print(f"Warning: {len(missing_files)} referenced files not found in {args.data_dir}:")
        print(missing_files[:5], "..." if len(missing_files) > 5 else "")
    
    if args.limit_files is not None:
        before = len(files_to_ingest)
        files_to_ingest = files_to_ingest[: args.limit_files]
        print(f"Limiting ingestion to {len(files_to_ingest)}/{before} resolved files (--limit-files).")

    if not files_to_ingest:
        print("No files found to ingest. Exiting.")
        return

    print(f"Resolved {len(files_to_ingest)} files to ingest.")

    # 1. Setup Embeddings
    t_embed = time.time()
    print(f"[5/5] Initializing embeddings model: {args.embed_model} ...")
    embed_model = HuggingFaceEmbedding(model_name=args.embed_model)
    Settings.embed_model = embed_model
    Settings.llm = None # We don't need LLM for ingestion

    # Match ingest.py chunking so eval runs on comparable chunks.
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 100
    print(f"  embeddings ready in {time.time() - t_embed:.2f}s")

    # 2. Load Documents
    t_load = time.time()
    print("Loading documents (SimpleDirectoryReader)...")
    documents = SimpleDirectoryReader(
        input_files=files_to_ingest, 
        file_metadata=get_page_metadata
    ).load_data()
    print(f"  loaded {len(documents)} document chunks in {time.time() - t_load:.2f}s")

    print("Initializing BERT NER pipeline...")
    try:
        ner_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    except Exception as e:
        print(f"Failed to load NER pipeline: {e}")
        ner_pipe = None

    print("Extracting metadata...")
    for i, doc in enumerate(documents):
        if i % 10 == 0:
            print(f"Processing doc {i}/{len(documents)}")
        extracted = extract_metadata_from_text(doc.text, ner_pipe)
        doc.metadata.update(extracted)

    # 3. Setup Vector Store
    t_db = time.time()
    print(f"Persisting to {args.persist_dir}...")
    db = chromadb.PersistentClient(path=args.persist_dir)
    chroma_collection = db.get_or_create_collection(args.collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print(f"  Chroma ready in {time.time() - t_db:.2f}s")

    # 4. Create Index (Ingest)
    print("Indexing (embedding + writing vectors) ... this can take a while.")
    t_index = time.time()
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print(f"  indexing finished in {time.time() - t_index:.2f}s")
    
    print(f"Ingestion complete! Total elapsed: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
