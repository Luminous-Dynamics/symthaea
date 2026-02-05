#!/bin/bash
# Download LibriSpeech data for testing
#
# LibriSpeech is a corpus of read English speech derived from audiobooks.
# https://www.openslr.org/12/
#
# Subsets:
#   dev-clean    (~350 MB, ~5.4 hours)  - Development, clean speech
#   test-clean   (~350 MB, ~5.4 hours)  - Test, clean speech
#   dev-other    (~320 MB, ~5.3 hours)  - Development, other conditions
#   test-other   (~340 MB, ~5.1 hours)  - Test, other conditions
#   train-clean-100 (~6.3 GB, ~100 hours) - Training subset
#   train-clean-360 (~23 GB, ~363 hours)  - Large training subset
#   train-other-500 (~30 GB, ~496 hours)  - Other training subset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data/librispeech"

# Default: download dev-clean and test-clean
SUBSETS="${1:-dev-clean test-clean}"

BASE_URL="https://www.openslr.org/resources/12"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "LibriSpeech Download Script"
echo "==========================="
echo "Data directory: $DATA_DIR"
echo "Subsets to download: $SUBSETS"
echo ""

for subset in $SUBSETS; do
    TARFILE="${subset}.tar.gz"
    URL="${BASE_URL}/${TARFILE}"

    if [ -d "LibriSpeech/${subset}" ]; then
        echo "[$subset] Already exists, skipping..."
        continue
    fi

    echo "[$subset] Downloading from $URL..."

    if command -v wget &> /dev/null; then
        wget -c "$URL" -O "$TARFILE"
    elif command -v curl &> /dev/null; then
        curl -L -C - -o "$TARFILE" "$URL"
    else
        echo "Error: wget or curl required"
        exit 1
    fi

    echo "[$subset] Extracting..."
    tar -xzf "$TARFILE"

    # Optionally remove tarball to save space
    # rm "$TARFILE"

    echo "[$subset] Done!"
done

echo ""
echo "Download complete!"
echo ""
echo "Contents:"
ls -la LibriSpeech/ 2>/dev/null || echo "(LibriSpeech directory not found)"
echo ""
echo "Usage:"
echo "  cargo run --release --example librispeech_test"
echo "  cargo run --release --example librispeech_test -- --subset test-clean --limit 100"
