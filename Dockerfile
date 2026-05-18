# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY mlruns/ mlruns/

# Copy MLflow database (optional wildcard to prevent build failure if missing)
COPY mlflow.d[b] .

# Expose ports (8000 for API, 8501 for Dashboard, 5000 for MLflow)
EXPOSE 8000 8501 5000

# Default command (API with Gunicorn for production performance)
CMD ["gunicorn", "src.serve_validated:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--timeout", "120"]
