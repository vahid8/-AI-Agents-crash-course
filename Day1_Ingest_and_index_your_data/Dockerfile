FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml .

# Install dependencies using uv (no project install, just deps)
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY main.py .

# Run the application
CMD ["python", "main.py"]
