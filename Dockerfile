# Use official lightweight Python image
FROM python:3.10-slim

# Set environment vars
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://host.docker.internal:11434


# Set working directory
WORKDIR /usr/src/app

# Install system dependencies (build tools + git for HuggingFace models)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY call_input_guardrails.py .
COPY app ./app
COPY templates ./templates
COPY Files ./Files
COPY Embedding_Model ./Embedding_Model

# Expose FastAPI default port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "call_input_guardrails:app", "--host", "0.0.0.0", "--port", "8000"]