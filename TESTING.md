# Testing Guide

Comprehensive test infrastructure for the Luminous Dynamics monorepo. 320+ tests across 32 files, covering unit tests, property-based invariants, adversarial fuzzing, extreme mission scenarios, consciousness ablation, multi-seed robustness, sensory consciousness, and performance benchmarks.

## Key Finding (Experiment 9)

Symthaea has TWO consciousness measurement systems:
- **Behavioral proxy** (`compute_consciousness_level`): 4-input weighted average, used for motor safety gating. Mean ~0.36 in text mode.
- **SpectralMIP Phi** (ConsciousnessEngine): O(n³) IIT approximation, validated r=0.9998 vs exact Phi. Produces Φ=5-9, active on 54% of cycles.

All ablation tests measure the proxy. Experiment 9 confirmed the engine is active and producing real integrated information values.

## Quick Start

```bash
# Bridge-common (fastest — pure Rust, no features needed)
cargo test -p mycelix-bridge-common

# Symthaea core math
cargo test -p symthaea-core --lib

# Manipulator (standalone crate)
cargo test -p symthaea-manipulator

# AUV (standalone crate, includes extreme scenarios)
cargo test -p symthaea-auv

# HDC-core genetics library
cargo test -p hdc-core

# Health validation (pure Rust, no HDK)
cargo test -p health-validation

# Robotics integration (requires humanoid feature, ~3min)
CARGO_TARGET_DIR=target cargo test -p symthaea --features humanoid --test robotics_integration

# Full robotics suite (requires 4 features, ~5min)
CARGO_TARGET_DIR=target cargo test -p symthaea --features humanoid,helicopter,flight,vehicle \
  --test robotics_integration --test robotics_scenarios

# Substrate-embodiment cross-domain
CARGO_TARGET_DIR=target cargo test -p symthaea --features humanoid \
  --test substrate_embodiment_integration

# Determinism tests
CARGO_TARGET_DIR=target cargo test -p symthaea --test determinism_tests

# Defense cascade (immune system)
CARGO_TARGET_DIR=target cargo test -p symthaea --test defense_cascade_tests
```

## Test Suites

### Bridge-Common (Mycelix Governance Security)

**Location**: `crates/mycelix-bridge-common/tests/`

| File | Tests | What It Proves |
|------|-------|---------------|
| `consciousness_profile_tests.rs` | 48 | Tier mapping, hysteresis, governance evaluation, reputation decay, bootstrap credentials, vote weights, NaN sanitization |
| `routing_registry_tests.rs` | 21 | Cross-cluster dispatch correctness, partition integrity, security invariants (self-to-self blocked, unknown zomes rejected) |
| `saga_offline_tests.rs` | 26 | Saga state machine (happy path, compensation, timeout), offline credential tier degradation (24h/72h/168h), BLAKE3 attestation signing/verification |
| `migration_tests.rs` | 16 | Migratable trait with mock V1->V2->V3 schema evolution, malformed JSON, version boundaries, data preservation |
| `proptest_extended_properties.rs` | 8 | Vote weight monotonicity/bounds, reputation decay monotonicity, offline degradation monotonicity, saga terminal idempotency, NaN/Infinity sanitization, hysteresis stability |
| `proptest_adversarial_governance.rs` | 7 | Attacker simulations: expired credential replay, clock manipulation, blacklist bypass, NaN injection, offline extension, grace period exploitation, vote weight manipulation |
| `proptest_gating_invariants.rs` | 25 | (Pre-existing) Tier monotonicity, score-tier consistency, gate soundness, bootstrap cap/expiry |
| `security_regression.rs` | 15 | (Pre-existing) Replay attacks, clock skew, credential forgery, tier boundaries |

**Total**: 166 tests (126 new + 40 pre-existing)

**Run**: `cargo test -p mycelix-bridge-common` (~6 seconds)

### Robotics Integration (Symthaea Cognitive-Embodiment Loop)

**Location**: `symthaea/tests/`

| File | Tests | Features | What It Proves |
|------|-------|----------|---------------|
| `robotics_integration.rs` | 12 | humanoid (+helicopter,flight,vehicle) | Full cognitive loop closure: thought HV -> motor -> physics -> proprioceptive HV -> next cycle. Multi-body fusion, consciousness degradation halts motors, long-horizon stability (200 cycles), step interval decoupling |
| `robotics_scenarios.rs` | 3 | humanoid,helicopter,flight,vehicle | Mission-level: helicopter SAR extended mission (100 cycles, 3 phases), vehicle extended driving (100 cycles, 5 scenarios), multi-body consciousness divergence |
| `substrate_embodiment_integration.rs` | 5 | humanoid | Substrate-embodiment cross-domain: substrate switching mid-mission, all 8 substrates stable with embodiment, substrate chain transfer, feasibility in metadata |
| `determinism_tests.rs` | 5 | — | Same seed -> same consciousness, different seeds diverge, near-reproducibility over 30 cycles, input sensitivity, metadata consistency |
| `defense_cascade_tests.rs` | 12 | — | Graduated defense: Green=no actions, Yellow=monitoring, Orange=intervention, Red=emergency halt. Escalation ordering, moral filter approval, severity bounds, expiry enforcement |

**Total**: 37 tests

**Run**: `CARGO_TARGET_DIR=target cargo test -p symthaea --features humanoid,helicopter,flight,vehicle --test robotics_integration --test robotics_scenarios --test substrate_embodiment_integration --test determinism_tests --test defense_cascade_tests -- --test-threads=1`

**Note**: The `--test-threads=1` is recommended for robotics tests since each test runs 50-200 cognitive cycles and is CPU-intensive.

### Manipulator (7-DOF Industrial Arm)

**Location**: `symthaea/crates/symthaea-manipulator/tests/`

| File | Tests | What It Proves |
|------|-------|---------------|
| `trajectory_safety_tests.rs` | 20 | IK failure modes (unreachable targets), multi-waypoint trajectory chaining, workspace clearance gradients, safety level ordering, simulator long-horizon stability (1000 steps), FK-IK roundtrip consistency |
| `proptest_kinematics.rs` | 7 | FK always finite, within workspace, deterministic. Jacobian correct shape/finite. IK respects joint limits. FK-IK roundtrip accuracy. within_limits consistency |

**Total**: 27 tests (+ 34 pre-existing inline = 61 total in crate)

**Run**: `cargo test -p symthaea-manipulator`

### AUV (Autonomous Underwater Vehicle)

**Location**: `symthaea/crates/symthaea-auv/tests/`

| File | Tests | What It Proves |
|------|-------|---------------|
| `chemical_hydrodynamics_tests.rs` | 12 | WHO threshold detection for all 8 contaminants, plume Gaussian falloff, clean water validation, simulator stability (moderate thrust, alternating thrust, full thrust — the previously-crashing test) |
| `extreme_scenarios.rs` | 7 | Abyssal dive/ascent (1000 steps), thermal downdraft (200N force), contamination plume transit (2000 steps), asymmetric thruster failure, emergency full-reverse, multi-axis extreme, 5000-step long-duration mission |

**Total**: 19 tests (+ 52 pre-existing = 71 total in crate)

**Run**: `cargo test -p symthaea-auv`

### Symthaea-Core (Math Foundation)

**Location**: `symthaea/symthaea-core/tests/`

| File | Tests | What It Proves |
|------|-------|---------------|
| `proptest_linear_algebra.rs` | 12 | Matrix addition commutativity/identity, transpose involution, trace linearity, identity multiplication. Determinant: identity=1, transpose invariant, scalar multiple. Vector: dot commutativity, norm non-negative, unit normalization, Cauchy-Schwarz |

**Run**: `cargo test -p symthaea-core --test proptest_linear_algebra`

### HDC-Core (Genetics)

**Location**: `mycelix-health/crates/hdc-core/tests/`

| File | Tests | What It Proves |
|------|-------|---------------|
| `extended_tests.rs` | 25 | Similarity search (index, top-k, threshold), confidence scoring (boundaries, probability ordering), DNA encoding (error handling, case insensitivity, determinism, mutation sensitivity), batch operations, cross-module pipeline (encode -> index -> search -> confidence), seed determinism |

**Run**: `cargo test -p hdc-core`

### Health Validation (Pure Rust)

**Location**: `mycelix-health/crates/health-validation/`

| Tests | What It Proves |
|-------|---------------|
| 23 (18 unit + 5 proptest) | MRN format validation, DID format validation (6 methods), confidence score bounds, NaN rejection, score range checking. Proptest: valid formats always pass, out-of-range always fails |

**Run**: `cargo test -p health-validation`

## Property-Based Testing

42 proptest properties generating ~38,000+ test cases. These catch edge cases that hand-written tests miss.

**Bridge-common** (`proptest = "1"` in dev-deps):
- 8 invariant properties + 7 adversarial properties + 25 pre-existing = 40 total
- Covers: vote weight monotonicity, reputation decay, offline degradation, NaN injection, blacklist bypass, grace period exploitation

**Manipulator** (`proptest = "1"` in dev-deps):
- 7 IK/FK properties (finite, bounded, deterministic, joint-limit-respecting)

**Symthaea-core** (`proptest = "1.4"` in dev-deps):
- 12 linear algebra properties (commutativity, identity, Cauchy-Schwarz)

**Health-validation** (`proptest = "1"` in dev-deps):
- 5 validation properties (MRN, DID, confidence, score range)

## Benchmarks

### Criterion (Performance Regression)

```bash
# Cognitive cycle latency
cargo bench --bench cognitive_cycle

# Embodied cycle overhead (humanoid vs disembodied)
cargo bench --bench embodied_cycle --features humanoid
```

### External Validation (Real Datasets)

```bash
# Full scorecard (ETHICS, MMLU, GSM8K, etc.)
cargo run --example benchmark_scorecard --release

# Individual benchmarks
cargo run --example benchmark_arc_reasoning --release
cargo run --example benchmark_mnist_hdc --release
cargo run --example benchmark_gsm8k --release
```

### Benchmark Runner

```bash
# Quick mode (scorecard only)
./scripts/run_all_benchmarks.sh --quick

# Full mode (all 61 examples)
./scripts/run_all_benchmarks.sh --full

# List available benchmarks
./scripts/run_all_benchmarks.sh --list
```

## Key Bug Fixes

### AUV Full-Thrust NaN (Fixed: `38f48c469`)

**Symptom**: `SimpleAuvSimulator` produced NaN at step 35 under full thrust (all 8 thrusters at 1.0).

**Root cause**: Explicit Euler integration with stiff quadratic angular drag. Angular inertia (0.5 kg·m²) was too small relative to drag forces (623 Nm at 1 rad/s), causing oscillation -> divergence -> NaN.

**Fix**: Physical velocity clamping (5 m/s linear, 3 rad/s angular) + NaN guard. Equivalent to implicit drag floor.

**Verification**: `cargo test -p symthaea-auv --test extreme_scenarios`

## Coverage Philosophy

- **Unit tests**: Correctness of individual functions (bridge-common, manipulator, AUV, health)
- **Property tests**: Mathematical invariants that must hold for ALL inputs (38K+ generated cases)
- **Adversarial tests**: Simulated attacker strategies against the security boundary
- **Integration tests**: Full cognitive loop closure (perception -> motor -> physics -> proprioception)
- **Mission scenarios**: Meaningful embodied behavior (SAR, driving, diving, multi-body)
- **Extreme scenarios**: Physics at breaking point (full thrust, thermal downdraft, thruster failure)
- **Determinism**: Same seed produces reproducible consciousness dynamics
- **Performance**: Criterion benchmarks for regression detection

## Dataset Management

See `symthaea/data/DATASET_REGISTRY.md` for the full inventory of 42 datasets (~71GB).

```bash
# Download core datasets
./symthaea/scripts/fetch_datasets.sh

# Download specific dataset
./symthaea/scripts/fetch_datasets.sh arc
```
