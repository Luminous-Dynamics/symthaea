# Symthaea-HLB Project Structure


> ⚠️ **METRIC CLARIFICATION**: Throughout this document, "Φ" refers to λ₂ (algebraic
> connectivity), NOT IIT integrated information. Validation revealed r = -0.62 correlation
> (nearly opposite). See `docs/METRIC_CLARIFICATION.md` for details.
**Total**: ~320K lines of executable code | ~914 source files | 28 directories

---

## Directory Overview

```
symthaea-hlb/
├── src/                    # Core Rust implementation (263K LOC)
├── examples/               # Usage examples (101 files, 22K LOC)
├── tests/                  # Integration tests (40 files, 9.5K LOC)
├── benches/                # Criterion benchmarks (29 files, 5K LOC)
├── crates/                 # Sub-crates (1.2K LOC)
│
├── scripts/                # Python/Shell tooling (26 files)
├── papers/                 # Academic publications (60+ files)
├── validation/             # PyPhi cross-validation
├── benchmarks/             # External benchmark datasets
│
├── docs/                   # Documentation (~300 files)
├── data/                   # Training/test data
├── datasets/               # Larger datasets
├── models/                 # Pre-trained models
├── figures/                # Generated visualizations
│
├── api/                    # API server
├── dashboard/              # Web monitoring UI
├── tools/                  # Debug/inspection tools
├── demos/                  # Demo applications
│
├── nix/                    # NixOS packaging
├── systemd/                # Service files
├── completions/            # Shell completions
├── tla/                    # TLA+ formal specs
│
├── logs/                   # Runtime logs
├── studies/                # Research studies
├── zenodo-dataset/         # Publication dataset
│
├── .archive-*/             # Archived code (4 directories)
├── .venv*/                 # Python virtual environments
├── target/                 # Rust build output
└── [root files]            # Config, results, scripts
```

---

## Core Code Directories

### src/ (263,271 LOC, 503 files)
The main Rust implementation organized by subsystem:

| Module | Files | LOC | Purpose |
|--------|-------|-----|---------|
| **hdc/** | 139 | 85,195 | Hyperdimensional Computing core |
| **consciousness/** | 71 | 58,771 | Consciousness theories & graph |
| **language/** | 40 | 33,311 | Natural language understanding |
| **benchmarks/** | 12 | 8,998 | Internal benchmarks |
| **observability/** | 23 | 8,176 | Monitoring & telemetry |
| **brain/** | 12 | 8,144 | 12 neural subsystems (Actor Model) |
| **shell/** | 15 | 6,665 | REPL interface |
| **bin/** | 5 | 5,300 | Binary entry points |
| **physiology/** | 8 | 4,701 | Embodiment systems |
| **perception/** | 12 | 4,511 | Vision, multimodal |
| **intelligence/** | 9 | 4,109 | Reasoning engine |
| **infrastructure/** | 10 | 3,442 | Config, logging |
| **memory/** | 6 | 3,045 | Memory systems |
| **synthesis/** | 7 | 2,985 | Program synthesis |
| **databases/** | 8 | 2,726 | Storage backends |
| **gui_bridge/** | 5 | 2,521 | GUI integration |
| **voice/** | 7 | 1,823 | TTS/STT |
| **web_research/** | 7 | 1,709 | Web queries |
| **action/** | 2 | 1,330 | Action execution |
| **nix_verification/** | 4 | 1,227 | NixOS validation |
| **safety/** | 4 | 1,059 | Safety constraints |
| **embeddings/** | 4 | 1,007 | Text embeddings |
| **api/** | 4 | 799 | External API |
| **core/** | 4 | 793 | Stable public facade |
| **phi_engine/** | 4 | 700 | Phi computation |
| **integration/** | 2 | 585 | External integrations |
| **substrate/** | 1 | 557 | Hardware abstraction |
| **symthaea_swarm/** | 3 | 458 | P2P networking |
| **sophia_swarm/** | 3 | 458 | Swarm intelligence |
| **soul/** | 2 | 422 | Value system |
| **hierarchical_cantor_ltc/** | 1 | 394 | Fractal LTC |

### examples/ (21,710 LOC, 101 files)
Usage examples demonstrating various capabilities:
- `phi_engine_quick_demo.rs` - Φ calculation demo
- `tier_3_exotic_topologies.rs` - Topology validation
- `brain_actor_model_demo.rs` - Brain subsystem demo
- `c_elegans_validation.rs` - Neuroscience validation
- `eeg_pattern_generation.rs` - EEG simulation
- ... and 96 more

### tests/ (9,492 LOC, 40 files)
Integration tests organized by module.

### benches/ (5,080 LOC, 29 files)
Criterion benchmarks for performance testing.

### crates/ (1,250 LOC, 12 files)
Sub-crates for specific functionality:
- **sophia-gym/** - Reinforcement learning environment
- **symthaea-gym/** - Consciousness training environment

---

## Tooling & Scripts

### scripts/ (~2,500 LOC, 26 files)
Python and Shell scripts for development:

| Script | Language | Purpose |
|--------|----------|---------|
| `benchmark_*.py` | Python | Performance benchmarking |
| `analyze_*.py` | Python | Result analysis |
| `pyphi_comparison.py` | Python | PyPhi cross-validation |
| `meta_learner*.py` | Python | Meta-learning experiments |
| `hdc_causal_discovery.py` | Python | Causal inference |
| `*.sh` | Shell | Build/run scripts |

### validation/
PyPhi cross-validation for verifying Φ calculations.

### tools/
- **symthaea-inspect/** - Debugging tool for consciousness traces

---

## Research & Academic

### papers/ (60+ files)
Academic publication pipeline:

| Paper | Topic | Status |
|-------|-------|--------|
| Paper 01 | Master Equation (Φ topology) | **Submission Ready** |
| Paper 02 | AI Consciousness | Draft |
| Paper 03 | Clinical Validation | Draft |
| Paper 04 | Binding Problem | Draft |
| Paper 05-15 | Various consciousness topics | Outlines |

Also contains:
- `generate_paper*.py` - Figure generation scripts
- `analysis/` - Statistical analysis
- `figures/` - Paper figures

### studies/
- `PHI_CAPABILITY_ETHICS_STUDY.md` - Ethics research

### zenodo-dataset/
Dataset prepared for publication with DOI.

---

## Data & Models

### data/
Training and test data:
- `eeg/` - EEG recordings
- `connectomes/` - Neural connectivity data

### datasets/
Larger datasets (not in git).

### models/
Pre-trained model weights.

### figures/
Generated visualizations and plots.

### benchmarks/
External benchmark datasets:
- `external/tuebingen/` - Causal discovery benchmark

---

## Infrastructure

### api/
HTTP API server for external integration.

### dashboard/
Web-based monitoring dashboard for real-time consciousness visualization.

### nix/
NixOS packaging configuration.

### systemd/
Systemd service files for deployment.

### completions/
Shell completion scripts (bash, zsh, fish).

### tla/
TLA+ formal specifications for consciousness properties.

---

## Documentation

### docs/ (~300 files)
Organized documentation:

| Folder | Purpose |
|--------|---------|
| `docs/VISION.md` | Project thesis |
| `docs/ARCHITECTURE.md` | Technical overview |
| `docs/HONEST_STATUS.md` | What works vs aspirational |
| `docs/ROADMAP.md` | Development roadmap |
| `docs/MODULE_MAP.md` | Complete code breakdown |
| `docs/sympoietic/` | Sympoietic partnership |
| `docs/planning/` | Historical roadmaps |
| `docs/paper/` | Academic submission |
| `docs/research/` | Research notes |
| `docs/architecture/` | Architecture docs |
| `docs/sessions/` | Work session logs |
| `docs/weekly/` | Weekly reports |

---

## Root Files

### Essential
| File | Purpose |
|------|---------|
| `Cargo.toml` | Rust project configuration |
| `Cargo.lock` | Dependency lock file |
| `flake.nix` | Nix flake configuration |
| `flake.lock` | Nix dependency lock |
| `README.md` | Project overview |
| `CLAUDE.md` | Claude AI context |
| `START_HERE.md` | Quick start guide |
| `CONTRIBUTING.md` | Contribution guide |
| `CHANGELOG.md` | Version history |
| `LICENSE` | MIT license |

### Scripts
| File | Purpose |
|------|---------|
| `quickstart.sh` | Quick setup script |
| `run_benchmarks.sh` | Run all benchmarks |
| `run_causal_benchmarks.sh` | Causal benchmarks |
| `run-consciousness-benchmarks.sh` | Consciousness benchmarks |
| `validate_phi_fix.sh` | Phi validation |
| `verify_phi_fix.sh` | Phi verification |
| `verify_week_1.sh` | Week 1 milestone verification |
| `test_word_learning.sh` | Word learning test |

### Python (Figure Generation)
| File | Purpose |
|------|---------|
| `generate_figures.py` | Main figure generation |
| `generate_supplementary_figures.py` | Supplementary figures |
| `prepare_zenodo_dataset.py` | Dataset preparation |

### Results (Should Move to results/)
| File | Purpose |
|------|---------|
| `*_results.json` | Experiment results (8 files) |
| `*_results.txt` | Validation results (4 files) |
| `*_results.csv` | CSV results (1 file) |
| `profiling_*.txt` | Profiling output (2 files) |
| `pyphi.log` | PyPhi log |
| `cargo-test.log` | Test log |

### Binary Data (Should Move to data/)
| File | Purpose |
|------|---------|
| `concept_store.bin` | Serialized concepts |
| `consciousness.bin` | Serialized consciousness state |

### Other
| File | Purpose |
|------|---------|
| `reproducibility.toml` | Reproducibility config |
| `test_phi_fix.rs` | Standalone test file |

---

## Archive Directories

| Directory | Date | Contents |
|-----------|------|----------|
| `.archive-2025-12-20/` | Dec 20, 2025 | Old code |
| `.archive-2025-12-29-pattern-matching/` | Dec 29, 2025 | Pattern matching experiments |
| `.archive-2026-01-04/` | Jan 4, 2026 | Old code |
| `.archive-2026-01-11/` | Jan 11, 2026 | Old code |

---

## Build Artifacts (Not in Git)

| Directory | Purpose |
|-----------|---------|
| `target/` | Rust build output |
| `.venv/` | Python virtual environment |
| `.venv-pyphi/` | PyPhi virtual environment |
| `__pyphi_cache__/` | PyPhi computation cache |
| `proptest-regressions/` | Property test regressions |

---

## Recommended Cleanup

### Move to results/
```
adaptive_reasoning_results.json
causal_explanation_results.json
dimension_synergies_results.json
meta_primitives_results.json
metacognitive_results.json
multi_objective_evolution_results.json
primitive_evolution_tier2_results.json
primitive_reasoning_results.json
primitive_validation_tier1_results.json
phi_validation_results_NEW.txt
PHI_VALIDATION_RESULTS_REDESIGNED.txt
TIER_3_VALIDATION_RESULTS_20251228_182858.txt
pyphi_validation_results.csv
profiling_post_session7c.txt
profiling_results_session_7a.txt
```

### Move to logs/
```
pyphi.log
cargo-test.log
```

### Move to data/
```
concept_store.bin
consciousness.bin
```

### Consolidate Archives
```
.archive-2025-12-20/
.archive-2025-12-29-pattern-matching/
.archive-2026-01-04/
.archive-2026-01-11/
→ .archive/2025-12-20/
→ .archive/2025-12-29-pattern-matching/
→ .archive/2026-01-04/
→ .archive/2026-01-11/
```

---

*Last updated: January 12, 2026*
