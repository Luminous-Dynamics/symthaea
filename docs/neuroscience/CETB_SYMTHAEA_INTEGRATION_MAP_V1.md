# CETB ↔ Symthaea Integration Map V1

**Status:** Architecture proposal; no new workspace crate or dependency is introduced by this document.

## Principle

Reuse working components, do not duplicate them. The first milestone is a reproducible external-data analysis and evidence receipt, not a second benchmark framework inside an already large workspace.

| Existing component | Current responsibility | CETB integration |
|---|---|---|
| docs/research/DECISION_RELEVANT_PREDICTABILITY_CHAOS_BENCHMARK_V0_2026-11-17.md | General forecast/decision contract, calibration, resource accounting and split integrity | Reuse common prediction, uncertainty, event-time and resource fields; keep neuroscience-specific endpoints separate |
| symthaea-psych-bench | Large set of mostly synthetic architecture and cognition-related probes | Keep as synthetic-mechanism lane; add explicit labels where a benchmark is not externally grounded |
| symthaea-futures-core | Typed forecast distributions and outcomes | Use for theory prediction cards after outcome-space semantics are reviewed |
| symthaea-futures-calibration | Brier, log score, CRPS and finite-score handling | Reuse proper-score implementations; document the multiclass-vs-binary Brier convention in every report |
| src/consciousness/recursive_improvement/calibration.rs | Rolling prediction-calibration tracking for the cognitive loop | Useful for internal calibration diagnostics, not a substitute for frozen, independent theory scoring |
| Butlin probe qualification design/runtime/report | Construct-validity and evidence-tier qualification | Reuse the distinction between structural wiring, runtime proxy, functional support and external validation |
| symthaea-evidence-plane | Candidate shared provenance/evidence contract | Inspect and map its schema before implementing a neuroscience receipt adapter |
| Nix / workspace tooling | Pinned reproducible development environment and checks | Use a standalone research tool initially; do not add a new root workspace member until dependency and CI impact are reviewed |

## Proposed boundaries

### 1. External dataset intake

A small, isolated adapter should record:

- provider and DOI/version;
- data-use / licence statements and their provenance;
- file manifest and digests;
- BIDS version, modality, participant and run counts;
- inclusion / exclusion flags;
- preprocessing versions and output paths.

Raw neuroimaging, EEG and participant-level health data must not be committed to the public repository.

### 2. Frozen analysis

The first ds006623 reproduction should invoke or faithfully reproduce the archived author analysis. The benchmark receipt records command, environment, analysis code digest, input manifest digest, participant set, configuration and result digest.

Only after output parity is established should a Rust implementation be introduced as a separate method version.

### 3. Theory predictions and scoring

Use typed forecast distributions and existing proper-score code. Store each forecast separately from its resolution. Preserve the scoring convention, outcome coding, horizon, calibration sample and baseline. Do not allow Symthaea to grade its own hypothesis without an independent evaluator path.

### 4. Qualification

Qualification should be a separate step from analysis. It checks the declared protocol, data lineage, split integrity, missingness, statistical plan, negative controls and replay receipt. A qualified computation is not proof that the operational construct is identical to phenomenal experience.

## Repository strategy

- No changes to the root Cargo workspace, dependency list or the broad CI matrix in this increment.
- No new dependency on MATLAB or proprietary software is introduced into production Rust crates.
- Authors' MATLAB/BCT code remains the reproduction reference; a Rust port follows only after reference parity and licence review.
- The first code addition, if needed, should live in an isolated research tool or an existing benchmark adapter rather than creating another broad crate.
- Public docs and protocols may be reviewed independently of CI; code claims remain pending until the pinned toolchain runs successfully.
