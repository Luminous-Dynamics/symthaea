#!/bin/bash
# Operation Leviathan: Download Whale Acoustic Data
# Source: Watkins Marine Mammal Sound Database (WHOI)

set -e

WHALE_DIR="data/whales/sperm"
mkdir -p "$WHALE_DIR"

echo "=================================================="
echo "  OPERATION LEVIATHAN: Whale Data Acquisition"
echo "=================================================="
echo ""

# Sperm Whale Codas (Social Communication)
# These rhythmic click patterns are the "language" we want to decode
echo "Downloading Sperm Whale Social Codas..."

# Multiple coda recordings for variety
wget -nc -q --show-progress -O "$WHALE_DIR/coda_01.wav" \
    "https://cis.whoi.edu/science/B/whalesounds/WhaleSounds/61000/61025.wav" 2>/dev/null || \
    curl -s -o "$WHALE_DIR/coda_01.wav" \
    "https://cis.whoi.edu/science/B/whalesounds/WhaleSounds/61000/61025.wav"

wget -nc -q --show-progress -O "$WHALE_DIR/coda_02.wav" \
    "https://cis.whoi.edu/science/B/whalesounds/WhaleSounds/61000/61026.wav" 2>/dev/null || \
    curl -s -o "$WHALE_DIR/coda_02.wav" \
    "https://cis.whoi.edu/science/B/whalesounds/WhaleSounds/61000/61026.wav"

wget -nc -q --show-progress -O "$WHALE_DIR/coda_03.wav" \
    "https://cis.whoi.edu/science/B/whalesounds/WhaleSounds/61000/61027.wav" 2>/dev/null || \
    curl -s -o "$WHALE_DIR/coda_03.wav" \
    "https://cis.whoi.edu/science/B/whalesounds/WhaleSounds/61000/61027.wav"

# Echolocation clicks (Hunting/Navigation)
# These should cluster differently - functional vs social
echo "Downloading Sperm Whale Echolocation Clicks..."

wget -nc -q --show-progress -O "$WHALE_DIR/echolocation_01.wav" \
    "https://cis.whoi.edu/science/B/whalesounds/WhaleSounds/61000/61028.wav" 2>/dev/null || \
    curl -s -o "$WHALE_DIR/echolocation_01.wav" \
    "https://cis.whoi.edu/science/B/whalesounds/WhaleSounds/61000/61028.wav"

wget -nc -q --show-progress -O "$WHALE_DIR/echolocation_02.wav" \
    "https://cis.whoi.edu/science/B/whalesounds/WhaleSounds/61000/61029.wav" 2>/dev/null || \
    curl -s -o "$WHALE_DIR/echolocation_02.wav" \
    "https://cis.whoi.edu/science/B/whalesounds/WhaleSounds/61000/61029.wav"

echo ""
echo "Whale Data Summary:"
echo "-------------------"
ls -la "$WHALE_DIR"/*.wav 2>/dev/null || echo "  (checking for files...)"

echo ""
echo "Operation Leviathan: Data acquisition complete."
