# Scripts

Python and Shell scripts for development, analysis, and benchmarking.

## Python Scripts

### Benchmarking
- `benchmark_causal_consciousness.py` - Causal consciousness benchmarks
- `benchmark_enhanced_causal.py` - Enhanced causal analysis

### Analysis
- `analyze_nixos_config.py` - NixOS configuration analysis
- `analyze_pyphi_results.py` - PyPhi result analysis
- `analyze_unrecoverable.py` - Error analysis
- `error_analysis.py` - General error analysis
- `error_analysis_pure.py` - Pure error analysis

### Meta-Learning
- `meta_learner.py` - Meta-learning experiments
- `meta_learner_v2.py` - Enhanced meta-learning

### Causal Discovery
- `hdc_causal_discovery.py` - HDC-based causal discovery
- `improved_causal.py` - Improved causal methods
- `targeted_neural_causal.py` - Neural causal discovery

### Utilities
- `aggregate_metrics.py` - Metrics aggregation
- `check_regressions.py` - Regression checking
- `ci_check_lanes.sh` - Run the focused `core`, `gpu`, `python-research`, and `coding-validation` validation lanes
- `run_coding_validation.sh` - Run code-generation honesty tests and the verified HumanEval-style smoke benchmark
- `format_summary.py` - Summary formatting
- `generate_benchmark_report.py` - Report generation
- `gpu_smoke.sh` - NVIDIA/CUDA environment preflight and optional Broca CUDA smoke test
- `post_to_dashboard.py` - Dashboard posting
- `precompute_ethics_embeddings.py` - Ethics embeddings
- `pyphi_comparison.py` - PyPhi comparison

### Prototyping
- `accn_prototype.py` - ACCN prototype

## Shell Scripts

- `benchmark-consciousness.sh` - Run consciousness benchmarks
- `check-integration-status.sh` - Check integration status
- `download-models.sh` - Download model weights
- `integrate-phase1.sh` - Phase 1 integration
- `run_tier_3_full_validation_overnight.sh` - Overnight validation

## Usage

```bash
# Enter a focused shell first
nix develop .#python-research

# Run package-backed Python scripts
python scripts/analyze_nixos_config.py
python scripts/benchmark_causal_consciousness.py

# Run Python smoke checks
uv run --no-sync pytest tests/python -q
uv run --no-sync ruff check python/symthaea_research scripts/analyze_nixos_config.py tests/python

# Run lane checks
./scripts/ci_check_lanes.sh core
./scripts/ci_check_lanes.sh python-research
./scripts/ci_check_lanes.sh coding-validation

# Run shell scripts
./scripts/benchmark-consciousness.sh
./scripts/gpu_smoke.sh
```
