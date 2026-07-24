#!/usr/bin/env bash
# Fetch MNIST IDX files into data/benchmarks/mnist/ so the documented HDC
# results (best 88.49%, examples/benchmark_mnist_hdc.rs) are reproducible
# again — the data was never in-tree and yann.lecun.com now 403s.
# Mirror: the S3 bucket torchvision uses (stable, versioned by content).
#
# Usage: bash symthaea/scripts/fetch-mnist.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/benchmarks/mnist"
MIRROR="https://ossci-datasets.s3.amazonaws.com/mnist"
FILES=(
  train-images-idx3-ubyte.gz
  train-labels-idx1-ubyte.gz
  t10k-images-idx3-ubyte.gz
  t10k-labels-idx1-ubyte.gz
)

mkdir -p "$DIR"
for f in "${FILES[@]}"; do
  out="$DIR/${f%.gz}"
  if [ -f "$out" ]; then
    echo "have    $out"
    continue
  fi
  echo "fetch   $MIRROR/$f"
  curl -fsSL "$MIRROR/$f" -o "$DIR/$f"
  gunzip -f "$DIR/$f"
done

echo
echo "MNIST ready in $DIR:"
ls -la "$DIR"
echo
echo "Run: cargo run --release --example benchmark_mnist_hdc"
echo "NOTE: data/ is gitignored (symthaea/.gitignore **/data/** trap) — the"
echo "files stay local; this script is the committed reproducibility artifact."
