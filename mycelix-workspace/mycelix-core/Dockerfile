FROM python:3.11-slim

# Metadata
LABEL maintainer="tristan.stoltz@luminousdynamics.com"
LABEL description="Byzantine-Resilient Federated Learning with 100% Detection Rate"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    netcat-openbsd \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY run_distributed_fl_network_simple.py .
COPY conductor_wrapper.py .
COPY live_dashboard.py .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV FL_NODES=10
ENV FL_ROUNDS=100
ENV FL_BASE_PORT=9000
ENV FL_BYZANTINE_RATIO=0.1

# Expose ports for FL network (9000-9010)
EXPOSE 9000-9010

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 -c "import socket; s=socket.socket(); s.connect(('localhost', 9000)); s.close()"

# Default command
CMD ["python3", "run_distributed_fl_network_simple.py", \
     "--nodes", "${FL_NODES}", \
     "--rounds", "${FL_ROUNDS}", \
     "--base-port", "${FL_BASE_PORT}"]
