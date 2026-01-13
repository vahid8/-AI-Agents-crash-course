FROM python:3.11-slim

WORKDIR /app

# Install uv for fast package management
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY *.py .

# Install dependencies
RUN uv pip install --system -r pyproject.toml

# Default command
CMD ["python", "main.py", "--demo-text"]
