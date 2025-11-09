FROM python:3.11.14-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (locked for reproducibility)
RUN uv sync --locked --no-dev

# Copy application code
COPY src/ ./src/
COPY train.py ./

# Create directories for artifacts
RUN mkdir -p models data

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["uv", "run", "python", "train.py", "--help"]
