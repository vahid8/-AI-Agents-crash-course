FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml .

# Install dependencies using uv (no project install, just deps)
RUN uv pip install --system -r pyproject.toml

# Create Day1 directory for imports
RUN mkdir -p /app/Day1_Ingest_and_index_your_data

# Copy Day1 code (we reuse read_repo_data from Day1)
# Named day1_ingest.py to avoid circular import with main.py
COPY day1_main.py /app/Day1_Ingest_and_index_your_data/day1_ingest.py

# Copy application code
COPY main.py .

# Run the application
CMD ["python", "main.py"]
