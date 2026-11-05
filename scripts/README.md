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
- `muse_style_coverage.py` - Source-derived Muse route audit in both directions; `--report` emits per-constellation gaps and unreferenced executable styles, `--check` fails on frozen-baseline drift
- `muse_choro_input_manifest.py` - Fail-closed Choro source gate; checks exact artifact SHA-256/byte counts, source-record identity, selected work/movement IDs and selection basis, review-scope digest, four rights layers, use/jurisdiction scope, and an independent approval receipt. A blocked empty template is not authorization.
- `muse_style_capability_inventory.py` - Extracts per-style grammar family/dispatch mode, composition spec, meter, tempo range, census-informed intent-axis declarations, and instrument-pool choices from Rust source; pins the census-source digest without executing the census or claiming render/quality evidence
- `examples/style_control_response_curves.rs` - Records 696 control curves / 3,364 points across 29 styles and four fixed baseline seeds, with exact fingerprints, 11 structural-layer distances, tonic, tempo/meter, duration, velocity/spread, section intensity, emphasis, pitch/range, optional form metadata, and no-change triage flags; symbolic evidence only
  - `.github/workflows/muse-style-response-curves.yml` executes the probe for relevant source changes and uploads a source-bound JSON report plus SHA-256 receipt, including a per-style/control seed-robustness table and exact baseline seeds for no-change triage.
  - The dedicated workflow publishes a per-style capability table and source blob digests in its job summary; the JSON report joins catalog entries to their runtime profiles and marks catalog-only rows explicitly.
- `gpu_smoke.sh` - NVIDIA/CUDA environment preflight and optional Broca CUDA smoke test
- `post_to_dashboard.py` - Dashboard posting
- `precompute_ethics_embeddings.py` - Ethics embeddings
- `pyphi_comparison.py` - PyPhi comparison

### Prototyping
- `accn_prototype.py` - ACCN prototype

## Shell Scripts

- `benchmark-consciousness.sh` - Run consciousness benchmarks
- `check-integration-status.sh` - Check integration status
- `check_cargo_lock_closure.sh` - Run full `cargo metadata --locked` and emit a machine-readable lock-closure disposition. If the committed lock is stale, preserves Cargo-generated proposal bytes/patch plus metadata and toolchain hashes, restores `Cargo.lock`, and exits `FAIL_LOCK_DRIFT` before downstream qualification; never hand-edits lock contents.
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

# Inspect Muse style catalog and executable-routing gaps (metadata, not musical-quality evidence)
# Inspect source-declared runtime controls and grammar engine/fallback dispatch
# Measure fixed-seed, one-control-at-a-time curves on the real grammar dispatch path
cargo test -p symthaea-music-theory --example style_control_response_curves
MUSE_STYLE_RESPONSE_JSON_OUT=/tmp/muse-style-response.json cargo run --release -p symthaea-music-theory --example style_control_response_curves
python3 scripts/muse_style_coverage.py --report > /tmp/muse-style-coverage-report.json
python3 scripts/muse_style_coverage.py --check
python3 tests/python/test_muse_style_coverage.py -v
python3 tests/python/test_muse_choro_source_intake.py -v
python3 scripts/muse_choro_input_manifest.py --check
python3 tests/python/test_muse_choro_input_manifest.py -v
# For an empirical run, use --require-authorized --artifact-root <private-reviewed-files-dir>; current template is intentionally blocked.
python3 scripts/muse_style_capability_inventory.py > /tmp/muse-style-capability-inventory.json
python3 tests/python/test_muse_style_capability_inventory.py -v

# Run Python smoke checks
uv run --no-sync pytest tests/python -q
uv run --no-sync ruff check python/symthaea_research scripts/analyze_nixos_config.py scripts/muse_style_coverage.py scripts/muse_style_capability_inventory.py tests/python

# Run lane checks
./scripts/ci_check_lanes.sh core
./scripts/ci_check_lanes.sh python-research
./scripts/ci_check_lanes.sh coding-validation

# Exercise the fail-closed Cargo lock-closure classifications with a stub Cargo executable
bash tests/shell/test-cargo-lock-closure.sh
# Run the current workspace lock gate (LOCK_CLEAN is prerequisite evidence, not a Rust test result)
bash scripts/check_cargo_lock_closure.sh target/cargo-lock-closure

# Run shell scripts
./scripts/benchmark-consciousness.sh
./scripts/gpu_smoke.sh
```
