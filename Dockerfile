FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY handler.py .
COPY runpod.toml .

# Create necessary directories
RUN mkdir -p workflows

# Copy workflow files
COPY workflows/ workflows/

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "handler.py"]
