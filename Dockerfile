FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf_cache \
    PORT=7860

# 1. Install CPU-only PyTorch first to avoid ~2.5GB NVIDIA CUDA downloads
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install Python dependencies
COPY requirements.txt pyproject.toml setup.py __init__.py ./
RUN pip install --no-cache-dir -r requirements.txt

# 3. Pre-cache the CrossEncoder reranker model in image layer
RUN mkdir -p /tmp/hf_cache && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" && \
    chmod -R 777 /tmp/hf_cache

# 4. Copy application source and install package mapping (`solvemycase -> .`)
COPY . .
RUN pip install --no-cache-dir --no-deps -e . && \
    mkdir -p /app/data/qdrant_storage /app/data/telemetry && \
    chmod -R 777 /app/data

EXPOSE 7860

CMD ["sh", "-c", "streamlit run ui/app.py --server.address 0.0.0.0 --server.port ${PORT:-7860} --server.headless true --browser.gatherUsageStats false"]
