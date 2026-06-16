#!/bin/bash
# Download Sleep-EDF Database (Sleep Cassette subset)
# Source: PhysioNet - https://physionet.org/content/sleep-edfx/1.0.0/
#
# This is a well-established clinical sleep dataset with:
# - EEG (Fpz-Cz, Pz-Oz channels)
# - EOG (horizontal)
# - EMG (submental)
# - Oro-nasal respiration
# - Rectal body temperature
# - Event markers
#
# Total: ~100 whole-night PSG recordings from 78 healthy subjects

set -e

DATASET_DIR="datasets/sleep-edf/sleep-cassette"
PHYSIONET_URL="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"

echo "=============================================="
echo "  Sleep-EDF Database Downloader"
echo "  Clinical PSG data for Consciousness Sensing"
echo "=============================================="
echo ""

mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

# Download a subset for testing (SC4001 = subject 01, night 1)
# PSG = polysomnography recording, Hypnogram = sleep stage annotations

echo "[1/4] Downloading SC4001E0-PSG.edf (50MB - full night recording)..."
if [ ! -f "SC4001E0-PSG.edf" ]; then
    wget -q --show-progress "$PHYSIONET_URL/SC4001E0-PSG.edf" -O SC4001E0-PSG.edf
else
    echo "      Already exists, skipping."
fi

echo "[2/4] Downloading SC4001EC-Hypnogram.edf (sleep stage annotations)..."
if [ ! -f "SC4001EC-Hypnogram.edf" ]; then
    wget -q --show-progress "$PHYSIONET_URL/SC4001EC-Hypnogram.edf" -O SC4001EC-Hypnogram.edf
else
    echo "      Already exists, skipping."
fi

echo "[3/4] Downloading SC4002E0-PSG.edf (second subject for validation)..."
if [ ! -f "SC4002E0-PSG.edf" ]; then
    wget -q --show-progress "$PHYSIONET_URL/SC4002E0-PSG.edf" -O SC4002E0-PSG.edf
else
    echo "      Already exists, skipping."
fi

echo "[4/4] Downloading SC4002EC-Hypnogram.edf..."
if [ ! -f "SC4002EC-Hypnogram.edf" ]; then
    wget -q --show-progress "$PHYSIONET_URL/SC4002EC-Hypnogram.edf" -O SC4002EC-Hypnogram.edf
else
    echo "      Already exists, skipping."
fi

echo ""
echo "Download complete! Files in $DATASET_DIR:"
ls -lh *.edf

echo ""
echo "=============================================="
echo "  Dataset ready for clinical integration tests"
echo "=============================================="
echo ""
echo "Signal channels available:"
echo "  - EEG Fpz-Cz (100Hz) - Primary consciousness signal"
echo "  - EEG Pz-Oz (100Hz)  - Secondary"
echo "  - EOG horizontal     - Eye movement (REM detection)"
echo "  - EMG submental      - Muscle tone (REM atonia)"
echo "  - Oro-nasal resp     - Breathing"
echo "  - Rectal temp        - Circadian rhythm"
echo ""
echo "Sleep stages in hypnogram:"
echo "  - W  = Wake"
echo "  - 1  = N1 (drowsy)"
echo "  - 2  = N2 (light sleep)"
echo "  - 3  = N3 (deep sleep / SWS)"
echo "  - 4  = N3 (legacy - now merged with 3)"
echo "  - R  = REM"
echo "  - M  = Movement artifact"
echo ""
