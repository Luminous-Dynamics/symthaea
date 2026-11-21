# CETB RCAF kernel

A small, dependency-free Rust reference kernel for Repeatability-Calibrated Architecture Fidelity (RCAF).

This is a research metric library, not a consciousness detector. It does not read EEG/fMRI files, estimate correlation matrices, perform preprocessing, or make clinical classifications. It consumes already-computed correlation matrices / edge weights and participant-level within- and between-condition similarities.

## Contents

- Weighted Jaccard for non-negative edge-weight vectors.
- Pair-aware correlation-matrix modes: absolute, positive edges in both matrices, negative edges in both matrices, and signed-shift.
- Matrix validation for symmetry, correlation range and shape.
- Participant-paired repeatability summaries and deterministic percentile bootstrap intervals for the median paired gap.
- Nine unit-test cases for hand-computed values, sign-mask semantics, invalid inputs and paired summaries.

The positive-only and negative-only modes apply a shared sign mask: an edge is included only when it has the requested sign in both matrices. This matches the pairwise masking convention used in the published ds006623 revision code. It is not equivalent to replacing non-matching edges with zero and including them in the denominator.

## Run

The crate is intentionally outside the root Cargo workspace and has no third-party dependencies:

    cargo test --manifest-path research/rcaf/Cargo.toml

No such command has been run in the current environment because rustc/cargo are unavailable. The source is staged for review; it is not represented as compiled or tested.

## Statistical contract

For each participant, first collapse repeated segment placements using the prespecified within-condition and between-condition summaries. Then pass equal-length participant vectors to summarize_rcaf. The participant is the inferential unit; repeated windows must never be passed as independent participants.

Positive paired gap means between-state similarity is lower than the within-state reference. The normalized excess-distance fraction is descriptive only; it is not a variance decomposition or causal attribution.


## v1.2 alignment-safe diagnostics

- `exact_sign_test` computes a two-sided exact participant-level sign test on paired gaps. Exact-zero gaps are ties and excluded from the binomial denominator. This is a distribution-free supplement to the effect distribution and participant bootstrap; it is not a randomization or causal test.
- `summarize_edge_sign_transitions` counts polarity reversals and transitions to/from a neutral correlation band. Evaluate a neutral-threshold grid frozen before inspecting state contrasts.
- Pair-aware positive/negative WJ still excludes polarity-reversed edges from that selected-edge denominator. Therefore report direct sign transitions beside sign-filtered WJ; do not interpret either metric alone.

The report contract is `rcaf-report.schema.json`. The accompanying `fixtures/synthetic-rcaf-report.json` contains synthetic values only and must never be presented as a biological observation.

The kernel now has fourteen authored unit-test cases. They are not yet compiled or executed in this environment.


## Executable report validation

The isolated research-data validator is at `tools/validate_report.py`. It checks the Draft 2020-12 report schema (when supplied) and semantic invariants that schemas cannot enforce: participant-key uniqueness, per-row gap arithmetic, denominator consistency, recomputed medians and sign-test counts/p-values, interval ordering, and claim ceilings.

It is a research validation helper, not a production Symthaea dependency. It uses the Python standard library for semantic checks; schema checking uses the optional development dependency in `requirements-validation.txt`.

Run:

```sh
python3 -m pip install -r research/rcaf/requirements-validation.txt
python3 research/rcaf/tools/validate_report.py \
  research/rcaf/fixtures/synthetic-rcaf-report.json \
  --schema research/rcaf/rcaf-report.schema.json
python3 -m unittest discover -s research/rcaf/tools -p 'test_*.py'
```

The validator never promotes a synthetic fixture to biological evidence. It also does not prove source-data truth, correct preprocessing, absence of hidden confounds, or clinical validity; these require separate evidence gates.


## Focused validation workflow

`.github/workflows/rcaf-research-validation.yml` runs only for changes to the standalone RCAF research package. It checks:
- the Python report-validator unit tests;
- the synthetic report against the committed Draft 2020-12 schema and semantic validator;
- Rust 1.96 formatting, all kernel tests and strict Clippy.

This workflow does not add the standalone crate to the root Cargo workspace or alter production CI. A workflow run is evidence only for the exact commit it checks; skipped, queued or absent runs are not passes.
