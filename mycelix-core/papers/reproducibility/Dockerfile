# Mycelix Paper Reproducibility Container
# Ensures exact environment for reproducing paper results

FROM python:3.11.7-slim-bookworm

LABEL maintainer="Luminous Dynamics"
LABEL description="Reproducibility container for Mycelix MLSys 2026 paper"
LABEL version="1.0.0"

# Set environment variables for reproducibility
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=42
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Disable GPU to ensure CPU-only reproducibility
ENV CUDA_VISIBLE_DEVICES=""

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements_exact.txt .

# Install Python dependencies with exact versions
RUN pip install --no-cache-dir -r requirements_exact.txt

# Copy configuration
COPY config.yaml .

# Copy experiment scripts
COPY experiments/ experiments/

# Copy expected results for validation
COPY expected_results/ expected_results/

# Copy main scripts
COPY run_all.py .
COPY validate.py .

# Create output directories
RUN mkdir -p output/results output/figures output/tables

# Set up volume mount point for output
VOLUME ["/app/output"]

# Default command runs all experiments
CMD ["python", "run_all.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import numpy; import scipy; print('OK')" || exit 1

# Usage:
# Build: docker build -t mycelix-paper .
# Run:   docker run -v $(pwd)/output:/app/output mycelix-paper
#
# For quick mode:
#   docker run -v $(pwd)/output:/app/output mycelix-paper python run_all.py --quick
#
# For individual experiments:
#   docker run -v $(pwd)/output:/app/output mycelix-paper python experiments/table1_byzantine_detection.py
