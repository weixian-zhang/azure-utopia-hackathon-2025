# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --disable-pip-version-check --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy .env file if exists (optional, better to use environment variables)
# COPY .env .env

# Expose port 8080
EXPOSE 8080

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8080/health || exit 1

# Run FastAPI app with uvicorn on port 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]