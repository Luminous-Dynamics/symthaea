# Symthaea AI System Development Lifecycle (SDLC)

Classification: Internal | Version: 1.0 | Date: 2026-03-07
Owner: Tristan Stoltz, Luminous Dynamics
ISO 42001 Control: A.3.3 (AI system lifecycle processes)

---

## Purpose

This document formalizes Symthaea's software development lifecycle for compliance with ISO/IEC 42001:2023 Annex A.3.3. It describes the processes governing how AI system changes move from conception to production.

---

## 1. Development Phases

### 1.1 Design

- **Architecture Decision Records (ADRs)**: Significant design changes are documented in `docs/compliance/adr/` using the ADR template. Each ADR captures context, decision, consequences, and alternatives considered.
- **Risk assessment**: Changes to safety-critical modules (Class A) require risk review against `AI_RISK_REGISTER.md` before implementation.
- **Value alignment**: Changes are evaluated against the Eight Harmonies framework (IEEE 7000-2021 traceability).

### 1.2 Implementation

- **Feature flags**: New capabilities are gated behind Cargo feature flags (88 flags currently). This enables incremental integration and independent testing.
- **Threshold registry**: All numeric parameters are defined as named constants in `src/cognitive_loop/thresholds.rs` with scientific citations. No magic numbers in safety-critical paths.
- **Module boundaries**: New modules follow the established pattern — sub-crates in `crates/` for reusable components, `src/` modules for cognitive loop integration.

### 1.3 Verification

- **Automated testing**: CI pipeline (`symthaea-ci.yml`) runs on every commit:
  - `cargo fmt --check` (formatting)
  - `cargo clippy` (linting)
  - `cargo test --lib` (unit tests, ~3,735+)
  - Feature matrix testing (39 combinations)
  - Sub-crate testing (45 crates)
  - Documentation build (`cargo doc`)
- **Property-based testing**: Proptest suites verify stability across parameter perturbations (`proptest_feedback_stability.rs`, `proptest_threshold_sensitivity.rs`).
- **Soak testing**: Long-duration tests (500-1000 cycles) verify safety properties under sustained conditions.
- **Adversarial testing**: 26 adversarial moral algebra tests verify robustness against edge cases.

### 1.4 Validation

- **Psych-bench regression**: Weekly CI job runs psychological benchmark battery against normative baselines (Stroop, Flanker, N-back, StopSignal, CPT, PVT, DualTask, UG, RME).
- **Phi validation**: Spectral MIP finder validated against exhaustive search (r=0.99, rho=0.93, 62 data points).
- **Moral accuracy**: Ethics pipeline validated at 91.1% moral classification accuracy.

### 1.5 Deployment

- **Pre-deployment checklist**:
  1. All CI checks green
  2. No Class A safety test regressions
  3. Safety soak tests pass (15 tests)
  4. Adversarial moral tests pass (26 tests)
  5. Substrate validation tests pass (37 tests)
- **Post-deployment monitoring**: SafetyAgent (NRC-style Green/Yellow/Orange/Red) provides continuous runtime monitoring with per-cycle telemetry.

---

## 2. Change Classification

### Class A: Safety-Critical

Files where changes require additional review and testing:

| File | Reason |
|------|--------|
| `src/safety/agent.rs` | Safety level assessment logic |
| `src/safety/audit.rs` | Compliance audit report generation |
| `src/safety/gate.rs` | Safety gating decisions |
| `src/cognitive_loop/ethics_engine.rs` | Moral evaluation pipeline |
| `src/hdc/moral_algebra.rs` | Consent violation detection |
| `src/cognitive_loop/thresholds.rs` | All safety-critical numeric parameters |
| `src/cognitive_loop/consciousness_engine.rs` | Consciousness measurement (Phi) |

**Class A change requirements**:
- Commit message prefix: `safety:` or `safety(scope):`
- Pre-commit hook validates prefix (`scripts/check-class-a-changes.sh`)
- Soak test suite must pass before merge
- ADR required for behavioral changes

### Class B: Core Pipeline

Files in the cognitive loop that affect system behavior but are not directly safety-critical. Standard CI verification sufficient.

### Class C: Supporting Infrastructure

Documentation, tooling, CI configuration, sub-crate utilities. Standard review process.

---

## 3. Version Control

- **Branching**: Feature branches from `main`, merged via pull request.
- **Commit conventions**: Scoped prefixes (`feat:`, `fix:`, `refactor:`, `safety:`, `docs:`, `test:`).
- **History**: Full git history maintained. No force-pushes to `main`.

---

## 4. Incident Response

Safety incidents follow the procedures in `INCIDENT_RUNBOOK.md`:
- SEV-1 (Red safety level): Immediate halt, root cause analysis
- SEV-2 (Orange sustained): Investigation within 24 hours
- SEV-3 (Yellow sustained): Review within 1 week
- SEV-4 (Informational): Logged for quarterly review

---

## 5. Continuous Improvement

- **Quarterly review**: Compliance matrix updated, risk register reviewed.
- **Annual validation**: Full psych-bench regression, Phi re-validation, moral accuracy re-assessment.
- **CalibrationHistory**: Automated drift detection identifies systematic parameter drift (>75% same-direction threshold).
- **SelfAssessmentMonitor**: EMA tracking of prediction error, coherence, confidence, and attention for auto-calibration.

---

## References

- ISO/IEC 42001:2023, Annex A.3.3 — AI system lifecycle processes
- ISO/IEC 12207:2017 — Software lifecycle processes (informative)
- `docs/compliance/AI_RISK_REGISTER.md` — Risk identification
- `docs/compliance/INCIDENT_RUNBOOK.md` — Incident procedures
- `docs/compliance/adr/` — Architecture Decision Records

---

*This document is reviewed quarterly or when significant process changes occur.*
