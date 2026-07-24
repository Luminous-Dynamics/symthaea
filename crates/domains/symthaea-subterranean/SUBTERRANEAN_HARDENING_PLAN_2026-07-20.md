# Symthaea Subterranean Hardening Plan

Date: 2026-07-20

## Goal

Turn `symthaea-subterranean` from a promising platform skeleton into a causally
truthful, testable subterranean control stack in which physical hazards,
learned control, active inference, and plant dynamics all affect the deployed
actuator path.

The governing control contract is:

```text
physical state + mission intent
    -> shared HDC control context
    -> HDC-LTC learned proposal
    -> consciousness / moral / physical safety envelope
    -> hazard-specific command arbitration
    -> plant
    -> command-conditioned prediction residual
```

## Patch sets prepared in this campaign

### Set 1 — Physical safety authority

Commit: `a5f41ad`

- Added deterministic hazard assessment across thermal, flood, gas, roof,
  escape, localization, communications, battery, and spoil conditions.
- Physical hazards now escalate `MotorSafetyLevel` independently of phi,
  manual overrides, and the moral gate.
- Added hazard-specific command arbitration.
- Added command-level regression tests proving dangerous learned outputs are
  replaced rather than merely labeling the state unsafe.

### Set 2 — Shared control context and deployable checkpoints

Commit: `10ee210`

- Added a common role-bound `perception + intent` HDC representation for both
  training and deployment.
- Training uses a deterministic neutral intent when no mission intent is
  supplied.
- Added versioned, serializable controller checkpoints with dimension,
  actuator-count, length, and finite-value validation.
- Added trainer export and embodiment checkpoint-loading paths.

### Set 3 — Timestep-correct plant dynamics

Commit: `f8b6cc0`

- Replaced fixed-per-step decay and recovery coefficients with rates or
  exponential continuous-time relaxation.
- Removed operating-mode-driven plant repair and localization recovery.
- Added realistic ambient thermal relaxation.
- Added tests preventing idle seal collapse and spontaneous extreme gas risk.
- Added a 50 Hz versus 400 Hz equivalent-rollout invariance gate.

### Set 4 — Construction, typing, normalization, and encoder hardening

Commit: `5f8f664`

- Added checked configuration constructors and explicit validation errors.
- Added typed actuator addressing and named constants for all simulator-used
  state channels.
- Normalized FEP observations to the encoder's physical channel ranges.
- Precomputed cumulative level hypervectors, reducing encoding from repeated
  per-channel cumulative construction to one lookup and bind per channel.
- Added checked encoder construction and normalization tests.

### Set 5 — Deployed FEP perception and genuine prediction residual

Commit: `d316ee9`

- Added the FEP perception/tau path to deployed embodiment operation.
- Replaced temporal frame difference mislabeled as prediction error with a
  command-conditioned one-step model residual.
- Added runtime FEP and deterministic model-agreement tests.

### Set 6 — Operational fallback truth

- Replaced the single misleading `VentAndRetreat` stage with explicit stages:
  nominal, thermal arrest, controlled withdrawal, position hold, energy
  conservation, and policy stop.
- Made fallback stage telemetry agree with the command arbiter's actual action.
- Updated safety tests to exercise the complete deployed step path.

## Required verification before merge

Run from the real Symthaea workspace, not from this isolated crate snapshot:

```bash
cargo fmt --check -p symthaea-subterranean
cargo clippy -p symthaea-subterranean --all-targets -- -D warnings
cargo test -p symthaea-subterranean
```

Also run the workspace's canonical Nix verification lane if this crate is part
of one. The isolated archive did not include `../../core/symthaea-core`,
`../../core/symthaea-fep`, or a Rust toolchain, so this campaign could perform
source inspection, delimiter/diff checks, and independent numerical replay of
the simulator equations, but not the authoritative Rust build.

## Next patch sets

### Set 7 — Scenario curriculum and held-out evaluation

Add deterministic scenario descriptions for:

- aquifer breach,
- gas pocket,
- roof instability,
- thermal runaway,
- spoil jam,
- relay loss,
- localization drift,
- battery reserve failure,
- sensor bias and dropout,
- combined hazards.

Training and evaluation seeds must be disjoint. Report intervention latency,
minimum safety margin, retreat success, mission progress, energy use, and false
abort rate. A lower imitation loss alone is not a safety result.

### Set 8 — Explicit recovery actuators and plant-policy separation

The current six-actuator plant cannot truthfully dewater, place a relay, repair
a seal, stabilize a roof, or execute a validated surfacing trajectory. Add
explicit actions or model those capabilities as unavailable. No operating mode
may directly improve plant state.

Suggested additions:

- seal closure or grout deployment,
- dewatering pump distinct from thermal cooling,
- relay deployment,
- roof-support deployment,
- localization scan,
- return-path follower or surfacing planner.

### Set 9 — Active-inference policy arbitration

Move beyond tau modulation by defining discrete high-level policies such as
continue, probe, stabilize, withdraw, deploy relay, isolate flood, and surface.
Use expected free energy only at this policy layer; retain the hard physical
safety arbiter as non-negotiable authority.

Required ablation:

```text
learned controller only
vs learned controller + FEP policy selection
vs reflex baseline
```

The FEP path should not be promoted unless it improves held-out safety or
mission metrics.

### Set 10 — Real-time and allocation gates

Add Criterion or workspace-native benchmarks for:

- state normalization,
- HDC encoding,
- control-context binding,
- LTC evolution,
- hazard assessment and arbitration,
- complete embodiment step.

Record p50, p95, and p99 latency at the production HDC dimension. Add an
allocation-free steady-state target for the control loop; the current
controller still copies final-layer features every forward pass for training.
Separate training and inference controller modes so deployment need not retain
training caches.

### Set 11 — External sensor/plant boundary

Split the deterministic simulator from the embodiment bridge through explicit
sensor and actuator interfaces. Prediction error only becomes scientifically
meaningful when the predicted state is compared with an independently observed
state rather than a second execution of the same deterministic model.

Add timestamp, freshness, calibration, missing-channel, and out-of-range
validation. Physical safety must fail closed when critical sensor confidence is
insufficient.

## Merge order

Sets 1 through 6 are designed to apply in order. Sets 7 and 8 should precede a
strong active-inference claim. Set 10 should precede real-time deployment. Set
11 should precede hardware-safety claims.

## Promotion boundary

After Sets 1 through 6, the crate can accurately claim:

- deterministic HDC-LTC control with a shared training/deployment input
  contract,
- deployable learned output checkpoints,
- hazard-derived command safety,
- normalized FEP-informed temporal modulation in training and deployment,
- timestep-consistent reference simulation,
- command-conditioned model-residual telemetry.

It should not yet claim autonomous underground deployment safety, validated
active-inference action selection, or a calibrated geological/thermal plant
model.
