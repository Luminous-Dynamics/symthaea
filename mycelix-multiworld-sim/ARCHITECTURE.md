# Architecture: Multiworld Simulation

## Core

`mycelix-multiworld-sim` (48K LOC Rust) is the central simulation engine. It runs monthly ticks through 12+ phases: demographics, genetics, psychology, education, economy, inter-world trade, governance, consciousness, disasters, and epoch evaluation.

## Dependency Graph

```
                    ┌─────────────────────────┐
                    │  mycelix-multiworld-sim  │
                    │    (48K LOC, monthly)    │
                    └──────────┬──────────────┘
                               │
           ┌───────────────────┼────────────────────┐
           │                   │                    │
           ▼                   ▼                    ▼
  symtropy-sim-bridge   symtropy-world       sol-atlas-core
  (Bevy Resources)      (threaded bridge)    (static timeline)
  Unidirectional        Bidirectional        Hardcoded calibration
           │                   │
           └─────────┬─────────┘
                     ▼
              Symtropy Game (Bevy 0.18)
```

## Consumers

### symtropy-sim-bridge
**Path**: `symtropy/crates/symtropy-sim-bridge/`

Wraps multiworld-sim types as Bevy ECS Resources for the Symtropy game. Imports: `WorldGovernance`, `WorldEconomy`, `FactionEngine`, `World`, `PolicyConfig`, `StochasticEngine`. One sim tick every 5 game-seconds. Clean Bevy plugin pattern.

### symtropy-world
**Path**: `symtropy/crates/symtropy-world/`

Threaded bridge: sim runs on background thread with monthly ticks, game polls at 60fps via bounded crossbeam channels (4 snapshots buffered). `SimSnapshot` structs serialize governance/economy/events for game consumption. Interpolation hides monthly granularity. Player actions flow back via `PlayerAction` channel.

### sol-atlas-core
**Path**: `sol-atlas-core/src/simulation.rs`

Static timeline visualization. Hardcoded tech milestones and growth rates calibrated from multiworld-sim's `empirical.rs` constants. No runtime integration — "spiritual" connection only. Used for Sol Atlas Bevy renderer.

## Internal Bridges

### mycelix_bridge.rs
Converts sim agents → production Holochain `ConsciousnessProfile` types. Same types deploy to real Mycelix governance without translation. Functions: `agent_to_profile()`, `location_to_body()`, `governance_latency_penalty()`.

### symtropy_bridge.rs
Converts sim state → game-facing types: `PlayerDecision`, `GameSnapshot`, `NarrativeSnippet`. Extracts decision points from sim state (empty project queue, independence readiness, crisis).

## Engine Versions

### V1 (Active)
Per-tick Bernoulli rolls for 40+ disaster types. 480,000 rolls per 1000-year sim. Simple, well-tested.

### V2 (Complete, Not Integrated)
**File**: `src/engine_v2.rs`

Gillespie algorithm with Poisson scheduling — skips empty ticks. Mathematically proven equivalent to V1 (within 5%). `EventQueue` priority heap, `PoissonProcess` scheduling. Ready to activate as a drop-in replacement.

## Scientific Infrastructure

| Module | Purpose |
|--------|---------|
| `statistics.rs` | Bootstrap CI, paired t-test, Cohen's d, convergence detection |
| `csv_output.rs` | Per-tick per-world CSV time series for R/Python analysis |
| `constants.rs` | `SimulationParams` — 30+ TOML-configurable physical parameters |
| `stochastic.rs` | Deterministic xoshiro256++ PRNG (seed → bit-exact replay) |
| `validation.rs` | Historical comparison (1970-2024 UN/NASA data) |
| `bin/ablation_study.rs` | 9-mechanism contribution ranking with CI/d/p |
| `bin/sensitivity_tornado.rs` | 7-parameter importance ranking |
| `bin/generate_reference.rs` | 10-seed canonical output table |

## Key Design Decisions

1. **Monthly ticks** — Coarser than daily, finer than yearly. Matches demographic and resource cycles.
2. **Agent-based** — Individual agents with 6D consciousness, 8-sector skills, psychological needs. Expensive but captures emergent dynamics.
3. **Same types as production** — `mycelix_bridge.rs` imports real Holochain types. What you test in sim deploys to Mycelix.
4. **Deterministic** — Custom PRNG guarantees reproducibility. Same seed → same output.
5. **Consciousness-gated governance** — Authority transitions require collective Phi + stability thresholds, not epoch timers.
