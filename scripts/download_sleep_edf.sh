#!/usr/bin/env bash
#
# Project Hypnos - Sleep-EDF Dataset Downloader
# Downloads the Sleep-EDF Expanded database from PhysioNet for consciousness detection validation
#
# Sleep-EDF Database: https://physionet.org/content/sleep-edfx/1.0.0/
# Citation: Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL.
#           Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG.
#           IEEE-BME 47(9):1185-1194 (2000).
#
# Usage:
#   ./scripts/download_sleep_edf.sh           # Download minimal subset (2 subjects)
#   ./scripts/download_sleep_edf.sh --full    # Download full dataset (153 subjects)
#   ./scripts/download_sleep_edf.sh --check   # Verify existing downloads
#

set -euo pipefail

# Configuration
PHYSIONET_BASE="https://physionet.org/files/sleep-edfx/1.0.0"
DATA_DIR="${PWD}/datasets/sleep-edf"
CASSETTE_DIR="${DATA_DIR}/sleep-cassette"
TELEMETRY_DIR="${DATA_DIR}/sleep-telemetry"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info() { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║        🧠 Project Hypnos - Sleep-EDF Downloader 🌙             ║"
    echo "║                                                                ║"
    echo "║  Validating Symthaea's LTC dynamics on clinical sleep data    ║"
    echo "║  Computational Biomarker for Consciousness Detection          ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check dependencies
check_deps() {
    local missing=()

    for cmd in curl sha256sum; do
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$cmd")
        fi
    done

    if [ ${#missing[@]} -ne 0 ]; then
        error "Missing required commands: ${missing[*]}"
        echo "Install with: nix-shell -p curl coreutils"
        exit 1
    fi
}

# Download a single file with progress and verification
download_file() {
    local url="$1"
    local output="$2"
    local desc="${3:-$(basename "$output")}"

    if [ -f "$output" ]; then
        success "Already exists: $desc"
        return 0
    fi

    info "Downloading: $desc"
    mkdir -p "$(dirname "$output")"

    if curl -# -L -o "$output" "$url"; then
        success "Downloaded: $desc"
        return 0
    else
        error "Failed to download: $desc"
        rm -f "$output"
        return 1
    fi
}

# Download minimal subset for initial development (2 subjects)
download_minimal() {
    info "Downloading minimal subset (2 subjects, ~80MB)"
    echo ""

    # Subject SC4001: Night 1 - Full night recording
    # - SC4001E0-PSG.edf: PSG recording (EEG Fpz-Cz, Pz-Oz, EOG, EMG, event marker)
    # - SC4001EC-Hypnogram.edf: Sleep stage annotations (expert-scored)

    local subjects=("SC4001" "SC4002")

    for subj in "${subjects[@]}"; do
        echo -e "${YELLOW}--- Subject: $subj ---${NC}"

        # PSG recording (contains EEG channels)
        download_file \
            "${PHYSIONET_BASE}/sleep-cassette/${subj}E0-PSG.edf" \
            "${CASSETTE_DIR}/${subj}E0-PSG.edf" \
            "${subj} PSG recording"

        # Hypnogram annotations (sleep stages)
        download_file \
            "${PHYSIONET_BASE}/sleep-cassette/${subj}EC-Hypnogram.edf" \
            "${CASSETTE_DIR}/${subj}EC-Hypnogram.edf" \
            "${subj} Hypnogram annotations"

        echo ""
    done

    success "Minimal dataset downloaded to: $CASSETTE_DIR"
}

# Download extended subset (20 subjects, good for training)
download_extended() {
    info "Downloading extended subset (20 subjects, ~800MB)"
    echo ""

    local subjects=(
        "SC4001" "SC4002" "SC4011" "SC4012" "SC4021" "SC4022"
        "SC4031" "SC4032" "SC4041" "SC4042" "SC4051" "SC4052"
        "SC4061" "SC4062" "SC4071" "SC4072" "SC4081" "SC4082"
        "SC4091" "SC4092"
    )

    for subj in "${subjects[@]}"; do
        echo -e "${YELLOW}--- Subject: $subj ---${NC}"

        download_file \
            "${PHYSIONET_BASE}/sleep-cassette/${subj}E0-PSG.edf" \
            "${CASSETTE_DIR}/${subj}E0-PSG.edf" \
            "${subj} PSG"

        download_file \
            "${PHYSIONET_BASE}/sleep-cassette/${subj}EC-Hypnogram.edf" \
            "${CASSETTE_DIR}/${subj}EC-Hypnogram.edf" \
            "${subj} Hypnogram"
    done

    success "Extended dataset downloaded to: $CASSETTE_DIR"
}

# Download full dataset (all 153 cassette recordings)
download_full() {
    info "Downloading full Sleep-EDF Cassette dataset (153 recordings, ~8GB)"
    warn "This will take a while..."
    echo ""

    # Get the file listing
    local record_list="${DATA_DIR}/RECORDS"
    download_file "${PHYSIONET_BASE}/sleep-cassette/RECORDS" "$record_list" "Record list"

    if [ ! -f "$record_list" ]; then
        # Create minimal record list if download failed
        info "Creating minimal record list..."
        mkdir -p "$DATA_DIR"
        for i in $(seq -w 1 78); do
            echo "SC40${i}E0-PSG" >> "$record_list"
        done
    fi

    # Download each recording
    while IFS= read -r record; do
        [ -z "$record" ] && continue

        local subj="${record:0:6}"

        download_file \
            "${PHYSIONET_BASE}/sleep-cassette/${record}.edf" \
            "${CASSETTE_DIR}/${record}.edf" \
            "${record}"

        # Also get the hypnogram if it's a PSG file
        if [[ "$record" == *"-PSG"* ]]; then
            local hypno="${subj}EC-Hypnogram"
            download_file \
                "${PHYSIONET_BASE}/sleep-cassette/${hypno}.edf" \
                "${CASSETTE_DIR}/${hypno}.edf" \
                "${hypno}"
        fi
    done < "$record_list"

    success "Full dataset downloaded to: $DATA_DIR"
}

# Verify downloaded files
verify_downloads() {
    info "Verifying downloaded files..."
    echo ""

    local total=0
    local valid=0
    local invalid=0

    shopt -s globstar nullglob 2>/dev/null || true
    for edf in "$DATA_DIR"/**/*.edf; do
        [ ! -f "$edf" ] && continue
        ((total++))

        # Check EDF header magic number (first 8 bytes should be "0       ")
        if head -c 8 "$edf" 2>/dev/null | grep -q "^0"; then
            ((valid++))
        else
            ((invalid++))
            error "Invalid EDF: $(basename "$edf")"
        fi
    done

    echo ""
    info "Verification Results:"
    echo "  Total files: $total"
    echo "  Valid EDF:   $valid"
    echo "  Invalid:     $invalid"

    if [ $invalid -eq 0 ] && [ $total -gt 0 ]; then
        success "All files verified!"
    elif [ $total -eq 0 ]; then
        warn "No EDF files found. Run without --check to download."
    fi
}

# Print usage information
print_usage() {
    cat << EOF

${CYAN}Sleep-EDF Database Information:${NC}

  The Sleep-EDF Expanded database contains 197 whole-night polysomnographic
  sleep recordings with expert-annotated sleep stages.

  ${YELLOW}Signals per recording:${NC}
    - EEG Fpz-Cz (frontal, 100 Hz)       ← Symthaea Frontal LTC
    - EEG Pz-Oz (occipital, 100 Hz)      ← Symthaea Occipital LTC
    - EOG horizontal (100 Hz)
    - Submental EMG (1 Hz)
    - Oronasal respiration (1 Hz)
    - Rectal body temperature (1 Hz)
    - Event marker (1 Hz)

  ${YELLOW}Sleep Stages (30-second epochs):${NC}
    - Wake (W)     → ConsciousnessState::Awake
    - Stage 1 (N1) → ConsciousnessState::LightSleep
    - Stage 2 (N2) → ConsciousnessState::LightSleep
    - Stage 3 (N3) → ConsciousnessState::DeepSleep (High Φ)
    - Stage 4 (N4) → ConsciousnessState::DeepSleep (merged with N3 in AASM)
    - REM (R)      → ConsciousnessState::REM (Paradoxical: high activity, low Φ)
    - Movement (M) → ConsciousnessState::Unknown

${CYAN}Usage:${NC}
  $0              Download minimal subset (2 subjects, ~80MB)
  $0 --extended   Download extended subset (20 subjects, ~800MB)
  $0 --full       Download full dataset (153 recordings, ~8GB)
  $0 --check      Verify existing downloads
  $0 --help       Show this help message

${CYAN}Quick Start:${NC}
  1. Download data:    ./scripts/download_sleep_edf.sh
  2. Run benchmark:    cargo run --example sleep_stage_benchmark --release
  3. Compare results:  See results/sleep_edf_benchmark.json

${CYAN}Files Downloaded:${NC}
  datasets/sleep-edf/sleep-cassette/
    ├── SC4001E0-PSG.edf        # PSG recording (EEG, EOG, EMG)
    ├── SC4001EC-Hypnogram.edf  # Sleep stage annotations
    ├── SC4002E0-PSG.edf
    ├── SC4002EC-Hypnogram.edf
    └── ...

${CYAN}PhysioNet Citation:${NC}
  Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
  Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit,
  and PhysioNet: Components of a New Research Resource for Complex Physiologic
  Signals. Circulation 101(23):e215-e220; 2000.

EOF
}

# Main entry point
main() {
    print_banner
    check_deps

    case "${1:-}" in
        --help|-h)
            print_usage
            ;;
        --check)
            verify_downloads
            ;;
        --full)
            download_full
            ;;
        --extended)
            download_extended
            ;;
        *)
            download_minimal
            ;;
    esac

    echo ""
    info "Data directory: $DATA_DIR"
    info "To run the benchmark:"
    echo "    cargo run --example sleep_stage_benchmark --release"
    echo ""
}

main "$@"
