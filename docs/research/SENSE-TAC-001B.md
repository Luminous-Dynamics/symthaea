# SENSE-TAC-001B — Deterministic Tactile Transducer / Front-End Fault Corpus

Status: frozen synthetic reference-corpus draft  
Parent: #5831  
Prerequisites: #5828 and the comparison/reference semantics in #5827 / #5863 / #5866

## Scope

The corpus at [SENSE_TAC_001B_FAULT_CORPUS_V0_1.json](fixtures/SENSE_TAC_001B_FAULT_CORPUS_V0_1.json) contains 30 deterministic synthetic fixtures. Its closed schema is [SENSE_TAC_001B_FAULT_CORPUS_V0_1.schema.json](fixtures/SENSE_TAC_001B_FAULT_CORPUS_V0_1.schema.json).

It is deliberately a synthetic fault model, not a model of any specific capacitive, piezoresistive, optical, magnetic, or commercial tactile sensor. Numeric signals use the dimensionless synthetic unit arb. These values must never be promoted to pascals, newtons, or measured accuracy.

## Model

For the two synthetic channels:

    normal_raw = gain_normal * normal_truth
               + shear_to_normal * shear_truth
               + bias_normal
               + drift_normal * elapsed_ms
               + temp_coeff_normal * (temperature_c - reference_temperature_c)
               + shared_offset
               + deterministic_noise_normal

    shear_raw = gain_shear * shear_truth
              + normal_to_shear * normal_truth
              + bias_shear
              + drift_shear * elapsed_ms
              + temp_coeff_shear * (temperature_c - reference_temperature_c)
              + shared_offset
              + deterministic_noise_shear

The declared processing order is linear response, history/environment effect, dead-zone, clipping, then deterministic quantization. Quantization uses nearest-step rounding with ties away from zero. Faults that affect identity, timing, transport, or missingness override the numeric response and are not imputed.

The candidate-facing input and evaluator-only truth are structurally separated. Candidate code may receive only candidate_input. The hidden physical_truth, injected fault, and expected disposition are evaluator-only.

## Frozen fault coverage

The 30 fixtures cover nominal response; bias; gain; both cross-axis directions; saturation; dead zone; quantization; drift; temperature dependence; hysteresis; dropout; stuck-at; intermittent open; channel swap; unit-scale mismatch; clock offset; readout latency; timestamp jitter; exact duplicate; conflicting replay; shared ADC offset; shared reference gain; wrong-article calibration; firmware-generation change; remount offset; curvature outside applicability; self-test missing scale drift; thresholded representation loss; and HDC representation aliasing.

They also preserve the central independence law: raw channels, transformed features, and HDC vectors derived from one physical acquisition do not become independent physical witnesses.

## Expected interpretation

- missing, stale, misidentified, conflicting, saturated, or out-of-profile evidence is never silently converted into a valid zero;
- a common ADC/reference error remains a shared failure cause, not two independent votes;
- self-test success is not proof of scale accuracy or calibration currentness;
- a changed installation, article, or firmware generation requires applicable calibration evidence;
- raw measurements remain addressable when a derived HDC representation is lossy or aliases two evaluator-distinct traces;
- a task feature may be useful for a declared task while remaining unfit as a physical measurand.

## Qualification plan

1. Independently validate the JSON against the closed schema, unique fixture IDs, required source/evaluator separation, and frozen enum vocabulary.
2. Implement a minimal Rust reference evaluator in a standalone tool, with no imports from production perception/sensor code.
3. Require 100% known-answer disposition agreement on all 30 fixtures and explicit negative controls for missingness, saturation, common-mode dependence, wrong article calibration, clock ambiguity, and representation aliasing.
4. Only after the reference oracle is stable, use it to qualify adapters. Production code must never import evaluator-only physical truth or expected outputs.

## Executable reference oracle

The standalone Rust reference lives at `tools/sense-tac-001b-oracle`. It:

- rejects duplicate JSON object members before semantic checks;
- validates the tracked corpus against its closed v0.1 JSON Schema;
- pins all 30 fixture identities, families and fault kinds;
- recomputes the declared two-channel numeric transforms and time/replay relations;
- derives the expected evidence/task/independence/representation dispositions from fault kind;
- runs mutation controls for missingness-as-zero, saturation, shared-reference independence inflation, replay conflicts, HDC aliasing, schema drift and self-test blind spots.

The model constants are explicit: reference temperature 20 °C, drift epoch 0 ms, default channel gain 1 and default bias 0. The fixture parameter semantics are recorded in the corpus model. The signals remain dimensionless synthetic arb.

Run from the repository root:

```sh
cargo fetch --locked --manifest-path tools/sense-tac-001b-oracle/Cargo.toml
cargo fmt --manifest-path tools/sense-tac-001b-oracle/Cargo.toml --all -- --check
cargo test --locked --offline --manifest-path tools/sense-tac-001b-oracle/Cargo.toml
cargo clippy --locked --offline --manifest-path tools/sense-tac-001b-oracle/Cargo.toml --all-targets -- -D warnings
cargo run --locked --offline --manifest-path tools/sense-tac-001b-oracle/Cargo.toml
```

The exact-head workflow pins Git blob identities for both corpus files and runs the same locked/offline sequence. The local authoring environment has no Rust toolchain, so no local Rust build/test/format/Clippy PASS is claimed. Workflow pending/skipped is not PASS.

## Claim ceiling

The corpus can establish only deterministic synthetic fixture/reference semantics. It does not establish physical sensor models, fault coverage probabilities, calibration, real drift rates, tactile performance, safety, or any commercial comparison.
