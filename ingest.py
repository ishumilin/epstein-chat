import os
import argparse
import glob
import re
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from transformers import pipeline

# Configuration
DATA_DIR = os.getenv("OCR_DATA_DIR", "data")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "epstein_docs"
EMBED_MODEL_NAME = "BAAI/bge-m3"  # High quality, multi-lingual, long context

def get_page_metadata(filename):
    """
    Extracts page number and document name from filename 
    (e.g., 'DocName_page_123.md' -> {'page_label': '123', 'document_name': 'DocName'})
    """
    try:
        base = os.path.basename(filename)
        # New format: {pdf_name}_page_{page_id}.md
        parts = base.split('_page_')
        if len(parts) == 2:
            doc_name = parts[0]
            page_num = parts[1].split('.')[0]
            return {"page_label": page_num, "document_name": doc_name, "file_name": base}
        
        # Fallback for old format: page_123.md
        if base.startswith("page_"):
            page_num = base.split('_')[1].split('.')[0]
            return {"page_label": page_num, "file_name": base}
            
        return {"file_name": base}
    except Exception:
        return {"file_name": filename}

def extract_metadata_from_text(text, ner_pipe=None):
    """
    Extracts structured metadata from text content using regex and BERT NER.
    """
    metadata = {}
    
    def _truncate_list(items, max_chars=400):
        """Helper to keep metadata strings within LlamaIndex limits."""
        s = ", ".join(sorted(list(set(items))))
        if len(s) > max_chars:
            return s[:max_chars] + "..."
        return s

    # Dates (MM/DD/YYYY, MM/DD/YY, or Month DD, YYYY)
    dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', text)
    word_dates = re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b', text, re.IGNORECASE)
    all_dates = dates + word_dates
    if all_dates:
        metadata["dates"] = _truncate_list(all_dates)

    # Phone Numbers (7+ digits, various formats)
    phones = re.findall(r'\b(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b', text)
    if phones:
        metadata["phone_numbers"] = _truncate_list(phones)

    # Emails
    emails_from = re.findall(r'From:\s*(.*)', text, re.IGNORECASE)
    emails_to = re.findall(r'To:\s*(.*)', text, re.IGNORECASE)
    if emails_from:
        metadata["email_from"] = _truncate_list([e.strip() for e in emails_from if e.strip()])
    if emails_to:
        metadata["email_to"] = _truncate_list([e.strip() for e in emails_to if e.strip()])

    # Flight Tail Numbers (N + digits + letters)
    tail_numbers = re.findall(r'\bN\d{2,5}[A-Z]{1,2}\b', text)
    if tail_numbers:
        metadata["flight_tail_numbers"] = _truncate_list(tail_numbers)

    # SSNs or similar patterns
    ssn_patterns = re.findall(r'\b\d{3}-\d{2}-\d{4}\b', text)
    if ssn_patterns:
        metadata["potential_ids"] = _truncate_list(ssn_patterns)

    # Locations/Names from tables (Capitalized words in table cells)
    # Heuristic: Match content between pipes | ... |
    table_cells = re.findall(r'\|(.*?)\|', text)
    potential_locs = set()
    for cell in table_cells:
        content = cell.strip()
        if not content: continue
        # Filter headers
        if content.upper() in ["DATE", "TIME", "NUMBER", "DURATION", "CITY", "STATE", "BILLED PHONE", "DEST NUMBER"]:
            continue
        # Check if ALL CAPS (often locations) or Title Case (names)
        if re.match(r'^[A-Z\s,]+$', content) and len(content) > 3:
             potential_locs.add(content)
    
    # Use BERT NER if available
    if ner_pipe:
        try:
            # Truncate text to avoid token limit issues (BERT max 512 tokens usually)
            # We scan the first 2000 chars as a heuristic for headers/intro
            entities = ner_pipe(text[:2000])
            names = set()
            orgs = set()
            locs = set()
            
            for ent in entities:
                if ent['entity_group'] == 'PER':
                    names.add(ent['word'])
                elif ent['entity_group'] == 'ORG':
                    orgs.add(ent['word'])
                elif ent['entity_group'] == 'LOC':
                    locs.add(ent['word'])
            
            if names:
                metadata["extracted_names"] = _truncate_list(names)
            if orgs:
                metadata["extracted_orgs"] = _truncate_list(orgs)
            if locs:
                # Merge with heuristic locations
                potential_locs.update(locs)
                
        except Exception as e:
            print(f"NER Error: {e}")

    if potential_locs:
        metadata["locations"] = _truncate_list(potential_locs)

    return metadata

def main():
    parser = argparse.ArgumentParser(description="Ingest OCR markdown files into ChromaDB.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory containing .md files")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover *.md files under --data-dir (recommended for HF 'pages/<range>/' layout).",
    )
    parser.add_argument("--persist-dir", default=PERSIST_DIR, help="Directory to store ChromaDB")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to ingest (for testing)")
    parser.add_argument("--start-page", type=int, default=None, help="Start page number (inclusive)")
    parser.add_argument("--end-page", type=int, default=None, help="End page number (inclusive)")
    parser.add_argument("--min-size", type=int, default=0, help="Minimum file size in bytes to include")
    parser.add_argument("--clear", action="store_true", help="Clear existing ChromaDB before ingesting")
    args = parser.parse_args()

    print(f"Initializing embedding model: {EMBED_MODEL_NAME} ...")
    # Use BGE-M3 for state-of-the-art retrieval quality
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model
    # Set chunk size to accommodate metadata extraction
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 100

    print(f"Reading files from {args.data_dir} ...")
    # Read files
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return

    # Discover files
    # Support both flat layouts and nested (HF pages/<range>/...) layouts.
    if args.recursive:
        all_files = [str(p) for p in Path(args.data_dir).rglob("*.md")]
    else:
        # Use glob to get files, optionally limit
        # Support both new format (*_page_*.md) and old format (page_*.md)
        all_files = glob.glob(os.path.join(args.data_dir, "*.md"))
    
    # Sort helper
    def sort_key(f):
        base = os.path.basename(f)
        try:
            if "_page_" in base:
                parts = base.split('_page_')
                return (parts[0], int(parts[1].split('.')[0]))
            if base.startswith("page_"):
                return ("", int(base.split('_')[1].split('.')[0]))
        except:
            pass
        return (base, 0)

    all_files.sort(key=sort_key)
        
    # Filter by page range if specified
    if args.start_page is not None or args.end_page is not None:
        def _page_id(f):
            meta = get_page_metadata(f)
            try:
                return int(meta.get("page_label", -1))
            except:
                return -1
        sp = args.start_page or 0
        ep = args.end_page or 999999999
        all_files = [f for f in all_files if sp <= _page_id(f) <= ep]
        print(f"Filtered to pages {sp}-{ep}: {len(all_files)} files.")

    # Filter by minimum file size
    if args.min_size > 0:
        all_files = [f for f in all_files if os.path.getsize(f) >= args.min_size]
        print(f"Filtered by min size {args.min_size}: {len(all_files)} files.")

    if args.limit:
        all_files = all_files[:args.limit]
        print(f"Limiting ingestion to first {args.limit} files.")

    reader = SimpleDirectoryReader(input_files=all_files, file_metadata=get_page_metadata)
    documents = reader.load_data()
    
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
        # LlamaIndex embeds metadata into text by default. 
        # We also want it stored as metadata in vector store for filtering.
    
    print(f"Loaded {len(documents)} documents with metadata.")

    print(f"Setting up ChromaDB in {args.persist_dir} ...")
    # Initialize ChromaDB
    db = chromadb.PersistentClient(path=args.persist_dir)
    if args.clear:
        try:
            db.delete_collection(COLLECTION_NAME)
            print("Cleared existing collection.")
        except Exception:
            pass
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Indexing documents (this may take a while)...")
    # Create Index (chunking + embedding + storage)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    print("Ingestion complete. Index saved to disk.")

if __name__ == "__main__":
    main()
