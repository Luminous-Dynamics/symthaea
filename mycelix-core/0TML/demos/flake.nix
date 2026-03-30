# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
{
  description = "Professional Demo System for ZeroTrustML Grants";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowBroken = true;  # Allow matplotlib's tkinter backend
            allowUnfree = true;   # For any unfree packages we might need
          };
        };

        # Python environment with all visualization tools
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          # Core dependencies
          numpy
          pandas

          # PyTorch for FL
          pytorch
          torchvision

          # Visualization
          plotly
          dash
          networkx
          pyvis
          matplotlib
          seaborn

          # Interactive notebooks
          jupyter
          ipywidgets
          ipython

          # Data processing
          scipy
          scikit-learn

          # Monitoring clients
          prometheus-client

          # Holochain/blockchain
          websockets
          msgpack
          web3

          # Utilities
          pyyaml
          requests
          aiohttp
          # asyncio is in standard library, not a separate package
          tqdm
          colorama
          rich

          # Testing
          pytest
          pytest-asyncio
          pytest-benchmark
        ]);

      in
      {
        # Main demo environment
        devShells.default = pkgs.mkShell {
          name = "zerotrustml-demo-full";

          buildInputs = with pkgs; [
            # Python with all packages
            pythonEnv

            # Monitoring & metrics
            prometheus
            grafana

            # Video recording
            obs-studio
            ffmpeg-full

            # Terminal recording
            asciinema

            # Terminal beautification
            bat
            eza  # modern ls replacement (formerly exa)
            ripgrep
            fzf

            # System utilities
            htop
            iftop
            netcat

            # Holochain (if needed)
            # holochain

            # Docker (for multi-node demos)
            docker-compose

            # Documentation
            pandoc
            texlive.combined.scheme-full

            # Git for version control
            git

            # JSON/YAML tools
            jq
            yq
          ];

          shellHook = ''
            echo ""
            echo "╔════════════════════════════════════════════════════════════╗"
            echo "║                                                            ║"
            echo "║       🎬 ZeroTrustML Grant Demo Environment                   ║"
            echo "║                                                            ║"
            echo "║  Professional visualization & benchmarking tools           ║"
            echo "║                                                            ║"
            echo "╚════════════════════════════════════════════════════════════╝"
            echo ""
            echo "📊 Available Tools:"
            echo ""
            echo "  Visualization:"
            echo "    - Plotly Dash     (web dashboards)"
            echo "    - NetworkX/PyVis  (network graphs)"
            echo "    - Matplotlib      (static charts)"
            echo "    - Jupyter         (interactive notebooks)"
            echo ""
            echo "  Monitoring:"
            echo "    - Prometheus      (metrics collection)"
            echo "    - Grafana         (dashboards)"
            echo ""
            echo "  Recording:"
            echo "    - OBS Studio      (screen recording)"
            echo "    - FFmpeg          (video processing)"
            echo "    - Asciinema       (terminal recording)"
            echo ""
            echo "🚀 Quick Start:"
            echo ""
            echo "  1. Network topology:   python visualizations/network_topology.py"
            echo "  2. FL dashboard:       python visualizations/fl_dashboard.py"
            echo "  3. Run benchmarks:     python benchmarks/run_benchmarks.py"
            echo "  4. Full demo:          python scripts/orchestrate_full_demo.py"
            echo "  5. Jupyter notebook:   jupyter notebook"
            echo ""
            echo "📁 Project Structure:"
            echo ""
            echo "  demos/"
            echo "    ├── scripts/           # Demo orchestration"
            echo "    ├── visualizations/    # Interactive visualizations"
            echo "    ├── benchmarks/        # Performance testing"
            echo "    ├── recording/         # Video production"
            echo "    └── outputs/           # Generated assets"
            echo ""
            echo "💡 Documentation: See GRANT_DEMO_ARCHITECTURE.md"
            echo ""

            # Set up Python path
            export PYTHONPATH="$PWD:$PYTHONPATH"

            # Create output directories
            mkdir -p outputs/{videos,images,data,interactive}
            mkdir -p benchmarks/results

            # Grafana setup (if using)
            export GRAFANA_PORT=3000

            # Prometheus setup (if using)
            export PROMETHEUS_PORT=9090

            echo "✅ Environment ready! Start with: python scripts/01_start_network.sh"
            echo ""
          '';
        };

        # Minimal environment (just visualization, no monitoring)
        devShells.viz = pkgs.mkShell {
          name = "zerotrustml-demo-viz";

          buildInputs = with pkgs; [
            pythonEnv
            bat
            jq
          ];

          shellHook = ''
            echo "🎨 Visualization-only environment loaded"
            echo "Run: python visualizations/network_topology.py"
          '';
        };

        # Benchmark-only environment
        devShells.benchmark = pkgs.mkShell {
          name = "zerotrustml-demo-benchmark";

          buildInputs = with pkgs; [
            pythonEnv
            prometheus
            htop
            iftop
          ];

          shellHook = ''
            echo "📊 Benchmark environment loaded"
            echo "Run: python benchmarks/run_benchmarks.py"
          '';
        };

        # Recording-only environment
        devShells.recording = pkgs.mkShell {
          name = "zerotrustml-demo-recording";

          buildInputs = with pkgs; [
            obs-studio
            ffmpeg-full
            asciinema
          ];

          shellHook = ''
            echo "🎥 Recording environment loaded"
            echo "Start OBS: obs"
            echo "Record terminal: asciinema rec demo.cast"
          '';
        };

        # CI environment (no GUI tools)
        devShells.ci = pkgs.mkShell {
          name = "zerotrustml-demo-ci";

          buildInputs = with pkgs; [
            pythonEnv
            ffmpeg-full
            jq
          ];

          shellHook = ''
            echo "🤖 CI environment (no GUI tools)"
            export HEADLESS=1
          '';
        };

        # Package for installing globally
        packages.default = pkgs.stdenv.mkDerivation {
          name = "zerotrustml-demo-tools";
          src = ./.;

          buildInputs = [ pythonEnv ];

          installPhase = ''
            mkdir -p $out/bin
            mkdir -p $out/share/zerotrustml-demo

            # Copy scripts
            cp -r scripts $out/share/zerotrustml-demo/
            cp -r visualizations $out/share/zerotrustml-demo/
            cp -r benchmarks $out/share/zerotrustml-demo/

            # Create wrapper scripts
            cat > $out/bin/zerotrustml-demo <<EOF
            #!/bin/sh
            cd $out/share/zerotrustml-demo
            python scripts/orchestrate_full_demo.py "\$@"
            EOF

            chmod +x $out/bin/zerotrustml-demo
          '';
        };
      }
    );
}
