# MOOForge — multi-objective EMO toolkit
# Reproducible image: pinned Python, CPU-only, deterministic benchmark.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install runtime deps first (layer cache).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package + entry points.
COPY mooforge ./mooforge
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Default: run the full deterministic benchmark and emit benchmark.json.
ENTRYPOINT ["python", "-m", "mooforge.cli", "run"]
CMD ["--out", "/app/benchmark.json"]
