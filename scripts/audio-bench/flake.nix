{
  description = "Symthaea Audio Benchmark - MIR analysis, CLAP scoring, FAD";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
    python = pkgs.python3.withPackages (ps: with ps; [
      numpy
      scipy
      librosa
      soundfile
      torch
      transformers
      pip
      setuptools
      wheel
    ]);
  in {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [ python pkgs.ffmpeg ];
      shellHook = ''
        echo "Symthaea Audio Benchmark Environment"
        echo "  Python: $(python3 --version)"
        echo "  Torch:  $(python3 -c 'import torch; print(torch.__version__)')"
        echo ""
        echo "Install CLAP (first time only):"
        echo "  pip install --break-system-packages laion-clap frechet-audio-distance"
        echo ""
        echo "Run analysis:"
        echo "  python3 ../analyze_audio.py ../../audio_output/demo_*.wav"
        echo "  python3 clap_score.py ../../audio_output/demo_*.wav"
      '';
    };
  };
}
