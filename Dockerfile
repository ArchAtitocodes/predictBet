# Multi-stage lightweight Dockerfile for PredictBet Application
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Final runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy codebase
COPY . /app/

# Environment defaults
ENV PORT=8080
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/api/health')"

# Run non-root for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8080"]
