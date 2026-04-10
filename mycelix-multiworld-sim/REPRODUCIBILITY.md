# Reproducibility Guide

Every result from this simulator can be exactly reproduced given the same seed, configuration, and toolchain.

## Toolchain

```
rustc 1.94.0 (4a4ef493e 2026-03-02)
cargo 1.94.0 (85eff7c80 2026-01-15)
```

Install: `rustup install 1.94.0 && rustup default 1.94.0`

## Determinism Guarantee

The simulator uses a custom xoshiro256++ PRNG (`src/stochastic.rs`) initialized from a 64-bit seed. Identical seed + identical config → bit-exact identical output. This is verified by `tests/analytical_benchmarks.rs::same_seed_same_result`.

## Quick Start

```bash
cd mycelix-multiworld-sim

# Run all tests (573+ tests, ~15 min)
cargo test --lib --test proptest_invariants --test analytical_benchmarks

# Generate reference outputs for 10 canonical seeds
cargo run --release --bin generate_reference > data/reference_outputs/canonical_150yr.csv

# Run ablation study (quantify each mechanism's contribution)
cargo run --release --bin ablation_study -- --seeds 20

# Run sensitivity analysis (parameter importance ranking)
cargo run --release --bin sensitivity_tornado -- --seeds 10

# Run a single simulation with CSV output
# (modify the binary or use the library API — see below)
```

## Library API for Custom Experiments

```rust
use mycelix_multiworld_sim::config::{SimulationConfig, PolicyConfig};
use mycelix_multiworld_sim::constants::SimulationParams;
use mycelix_multiworld_sim::MultiWorldSimulator;

// Default 150-year config
let mut config = SimulationConfig::default_150_year();
config.seed = 42;

// Customize policy (toggle mechanisms)
config.policy.disasters_enabled = false;
config.policy.trust_weighted_governance = true;

// Customize physical parameters (spoilage, radiation, etc.)
let params = SimulationParams {
    spoilage_food: 0.05,  // Harsher food spoilage
    ..SimulationParams::default()
};

// Run with CSV output
let mut sim = MultiWorldSimulator::new(config).with_params(params);
sim.enable_csv_output("output/my_experiment.csv".into());
let report = sim.run();

println!("Final CVS: {:.4}", report.final_cvs);
println!("Population: {}", report.final_population);
```

## Canonical Seeds

All multi-seed experiments use these 10 seeds for comparability:

```
42, 123, 789, 1337, 2718, 3141, 4242, 5555, 7777, 9999
```

These are fixed, arbitrary values chosen for reproducibility (not cherry-picked).

## Statistical Analysis

The `statistics` module (`src/statistics.rs`) provides:

| Function | Purpose | Citation |
|----------|---------|----------|
| `bootstrap_ci()` | 95% confidence intervals | Efron (1979) |
| `paired_t_test()` | Hypothesis testing | Student (1908) |
| `cohens_d()` | Effect size measurement | Cohen (1988) |
| `detect_convergence()` | Plateau detection | — |
| `describe()` | Descriptive statistics | — |

## Experiment Binaries

| Binary | Purpose | Typical Runtime |
|--------|---------|-----------------|
| `generate_reference` | Canonical seed → output table | ~30 min (10 seeds × 150yr) |
| `ablation_study` | Mechanism contribution ranking | ~2 hr (9 mechanisms × 20 seeds) |
| `sensitivity_tornado` | Parameter importance ranking | ~1 hr (7 params × 3 levels × 10 seeds) |
| `multi_seed` | Multi-seed statistical summary | ~30 min |
| `governance_comparison` | 5 governance model comparison | ~15 min |
| `scenario_comparison` | TOML scenario A/B testing | ~10 min |

## Output Formats

- **CSV**: `enable_csv_output()` — per-tick per-world time series (17 columns)
- **JSONL**: `enable_jsonl_output()` — per-tick aggregate metrics
- **JSON**: `CivilizationReport` via serde — final summary
- **Markdown**: `StandardizedReport` — human-readable summary

## File Structure

```
src/
├── lib.rs              # Simulator core (MultiWorldSimulator)
├── config.rs           # PolicyConfig (25+ toggles), SimulationConfig
├── constants.rs        # SimulationParams (30+ tunable physical constants)
├── statistics.rs       # Bootstrap CI, t-test, Cohen's d, convergence
├── csv_output.rs       # CSV time-series writer
├── stochastic.rs       # Deterministic xoshiro256++ PRNG
├── world.rs            # World state, resources, ColonyParams builder
├── consciousness.rs    # Individual + collective Phi, coherence engine
├── governance.rs       # Consciousness-gated authority transitions
├── disasters.rs        # 40+ disaster types, cascade engine
├── economy.rs          # 8-sector Cobb-Douglas with demurrage
├── population.rs       # Demographics, genetics, pair bonding
├── needs.rs            # Allostatic load, Spinozist affect state
├── education.rs        # Peer-to-peer learning, TEND rewards
├── viability.rs        # EROI, scaling laws, thermodynamic axioms
└── bin/                # 30+ experiment binaries
```
