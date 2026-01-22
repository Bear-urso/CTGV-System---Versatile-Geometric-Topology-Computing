# CTGV System Docker Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install CTGV system
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash ctgv
RUN chown -R ctgv:ctgv /app
USER ctgv

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["python", "-c", "import ctgv; print('CTGV System ready!')"]