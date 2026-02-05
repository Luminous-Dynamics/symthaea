# NixOS shell for AI benchmark dataset downloading
# Usage: nix-shell --run "python scripts/download_benchmarks.py --all"

{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "symthaea-ai-benchmarks";

  buildInputs = with pkgs; [
    # Python with required packages
    (python311.withPackages (ps: with ps; [
      datasets      # Hugging Face datasets library
      requests      # HTTP requests
      tqdm          # Progress bars
      pandas        # Data manipulation
      numpy         # Numerical operations
    ]))

    # Additional tools
    wget          # Alternative download method
    curl          # Alternative download method
    unzip         # Extract archives
    gzip          # Compression
  ];

  shellHook = ''
    echo "🧠 Symthaea AI Benchmarks Environment"
    echo "====================================="
    echo ""
    echo "Available commands:"
    echo "  python scripts/download_benchmarks.py --all    # Download all datasets"
    echo "  python scripts/download_benchmarks.py --mmlu   # Download MMLU only"
    echo "  python scripts/download_benchmarks.py --gsm8k  # Download GSM8K only"
    echo "  python scripts/download_benchmarks.py --check  # Check download status"
    echo ""
    echo "Dataset storage: ./data/"
    echo ""

    # Create data directory if it doesn't exist
    mkdir -p data

    # Set cache directory for Hugging Face
    export HF_HOME="$PWD/.hf_cache"
    export HF_DATASETS_CACHE="$PWD/.hf_cache/datasets"
    mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

    echo "HuggingFace cache: $HF_HOME"
    echo ""
  '';
}
