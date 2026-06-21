FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.api.txt .
# Install CPU-only PyTorch first to avoid pulling the 900MB CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.api.txt

COPY src/ ./src/
COPY static/ ./static/

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
