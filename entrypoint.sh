#!/bin/bash
set -e

DATA_DIR=${OCR_DATA_DIR:-data}
CHROMA_DIR=${CHROMA_PERSIST_DIR:-./chroma_db}

echo "Starting Epstein Chat Container..."

# 1. Check Data Download
# If directory is empty or missing, download
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR")" ]; then
    echo "Data directory ($DATA_DIR) is empty. Attempting download from Hugging Face..."
    # Use python module for stability across huggingface_hub CLI versions.
    # Users can override by mounting their own data into /data.
    mkdir -p "$DATA_DIR"

    # Optionally limit download size for faster local setup.
    # - DOWNLOAD_MAX_PAGE: downloads up to that page number (rounded up to 1000-page buckets)
    #   Example: DOWNLOAD_MAX_PAGE=5000 downloads pages/00000-00999 ... pages/04000-04999
    DOWNLOAD_MAX_PAGE_EFFECTIVE="${DOWNLOAD_MAX_PAGE:-}"
    if [ -z "$DOWNLOAD_MAX_PAGE_EFFECTIVE" ] && [ -n "$LIMIT_INGEST" ]; then
        DOWNLOAD_MAX_PAGE_EFFECTIVE="$LIMIT_INGEST"
    fi

    python - <<'PY'
import os, math
from huggingface_hub import snapshot_download

repo_id = "ishumilin/epstein-files-ocr-datasets-1-8-early-release"
local_dir = os.environ.get("OCR_DATA_DIR", "data")
download_max_page = os.environ.get("DOWNLOAD_MAX_PAGE_EFFECTIVE")

allow_patterns = None
if download_max_page:
    try:
        max_page = int(download_max_page)
        if max_page > 0:
            buckets = int(math.ceil(max_page / 1000))
            allow_patterns = [
                ".gitattributes",
                "README.md",
                "LICENSE",
            ]
            for i in range(buckets):
                start = i * 1000
                end = start + 999
                allow_patterns.append(f"pages/{start:05d}-{end:05d}/*.md")
    except Exception:
        pass

print(f"Downloading dataset repo={repo_id} -> {local_dir}")
if allow_patterns:
    print(f"Using allow_patterns ({len(allow_patterns)}): {allow_patterns[:10]}{'...' if len(allow_patterns) > 10 else ''}")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=allow_patterns,
)
PY
else
    echo "Data directory found at $DATA_DIR. Skipping download."
fi

# 2. Check Ingestion
# If ChromaDB is empty/missing, run ingest
if [ ! -d "$CHROMA_DIR" ] || [ -z "$(ls -A "$CHROMA_DIR")" ]; then
    echo "ChromaDB not found at $CHROMA_DIR. Starting ingestion..."
    # We limit to 5000 as requested for this specific Mac test scenario, 
    # but normally this would be full. 
    # I will use an env var LIMIT_INGEST if present, else default to full.
    
    INGEST_CMD="python ingest.py --data-dir $DATA_DIR --recursive"
    if [ ! -z "$LIMIT_INGEST" ]; then
        echo "Limiting ingestion to $LIMIT_INGEST files..."
        INGEST_CMD="$INGEST_CMD --limit $LIMIT_INGEST"
    fi
    
    $INGEST_CMD
else
    echo "ChromaDB found. Skipping ingestion."
fi

# 3. Check Ollama (if using local service)
if [[ "$VLLM_API_BASE" == *"ollama"* ]]; then
    echo "Checking Ollama connection at $VLLM_API_BASE..."
    # Basic wait-for-it loop.
    OLLAMA_BASE_URL="http://ollama:11434"
    for i in {1..60}; do
        if curl -sf "$OLLAMA_BASE_URL/api/tags" > /dev/null; then
            echo "Ollama is ready."
            break
        fi
        echo "Waiting for Ollama... ($i/60)"
        sleep 2
    done
    
    # Extract hostname from VLLM_API_BASE
    OLLAMA_HOST=$(echo $VLLM_API_BASE | sed -e 's|^[^/]*//||' -e 's|/.*$||' -e 's|:.*$||')
    
    if [ "$OLLAMA_HOST" = "ollama" ]; then
         # Ollama versions differ in supported endpoints; /api/tags is stable.
         # It returns models with names like "llama3:latest".
         if curl -sf "$OLLAMA_BASE_URL/api/tags" | grep -q "\"name\":\"${VLLM_MODEL_NAME}"; then
             echo "Ollama model '$VLLM_MODEL_NAME' already present. Skipping pull."
         else
             echo "Triggering model pull for $VLLM_MODEL_NAME on internal ollama service..."
             # This call streams progress logs until done.
             curl -sS -X POST "$OLLAMA_BASE_URL/api/pull" -d "{\"name\": \"$VLLM_MODEL_NAME\"}" || echo "Failed to trigger pull, assuming model exists or manual pull needed."
         fi
    fi
fi

# 4. Start Server
echo "Starting FastAPI server..."
uvicorn api:app --host 0.0.0.0 --port 8000
