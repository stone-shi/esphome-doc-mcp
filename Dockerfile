# Use official Python slim image for runtime efficiency
FROM python:3.12-slim

# Install system dependencies (git is required for documentation synchronization)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory inside the container
WORKDIR /app

# Copy dependency definition and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config files
COPY . .

# Expose server port (8000 is default for FastAPI)
EXPOSE 8000

# Create volume mount point for stateful database and doc repository
VOLUME ["/data"]

# Define container runtime environment defaults
ENV DATA_DIR=/data
ENV SYNC_INTERVAL_HOURS=24
ENV PORT=8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
