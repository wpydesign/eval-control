FROM python:3.11-slim

WORKDIR /app

# Install system deps (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir .[api]

# Copy source
COPY *.py .
COPY *.jsonl .
COPY LICENSE .

# Data volume for logs
VOLUME ["/app/data"]

# Environment defaults
ENV EVAL_CONTROL_LOG_DIR=/app/data
ENV EVAL_CONTROL_API_KEYS=""
ENV EVAL_CONTROL_HOST=0.0.0.0
ENV EVAL_CONTROL_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
