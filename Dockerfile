FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
# Install hf cli and requirements
RUN pip install --no-cache-dir huggingface_hub[cli] -r requirements.txt

# Copy application code
COPY . .

# Setup entrypoint
RUN chmod +x entrypoint.sh

# Expose API port
EXPOSE 8000

# Default command
ENTRYPOINT ["./entrypoint.sh"]
