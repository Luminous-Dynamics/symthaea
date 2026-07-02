# Zero-TrustML Holochain - Production Dockerfile
# Fully decentralized P2P federated learning node

FROM python:3.11-slim

LABEL maintainer="Luminous Dynamics <dev@luminousdynamics.org>"
LABEL description="Zero-TrustML Holochain - Decentralized Federated Learning Node"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package files
COPY setup.py README_PACKAGE.md ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch numpy && \
    pip install --no-cache-dir -e .

# Create data directory
RUN mkdir -p /data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ZEROTRUSTML_="node-default"
ENV ZEROTRUSTML_="/data"
ENV ZEROTRUSTML_="ws://localhost:8888"
ENV ZEROTRUSTML_="krum"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from zerotrustml import Node; print('healthy')" || exit 1

# Expose ports (if needed for monitoring)
EXPOSE 8000

# Default command
CMD ["python", "-c", "from zerotrustml import Node, NodeConfig; import asyncio; import os; config = NodeConfig(node_id=os.getenv('ZEROTRUSTML_NODE_ID'), data_path=os.getenv('ZEROTRUSTML_DATA_PATH'), holochain_url=os.getenv('ZEROTRUSTML_HOLOCHAIN_URL'), aggregation=os.getenv('ZEROTRUSTML_AGGREGATION')); node = Node(config); asyncio.run(node.start())"]
