# SYMPsi synthetic qualification harness

**Status:** offline software/simulation qualification only. This tool does not ingest human data, does not run an AI participant, and does not test or establish psi / anomalous information acquisition.

## Purpose

The harness qualifies the statistical and evidence-handling plumbing before any real AIA study can be considered. It has two explicit profiles:

| Profile | Choices | Chance baseline | Planned trials per campaign | Provisional SESOI |
|---|---:|---:|---:|---:|
| four-choice | 4 | 25% | 7,300 | +2 percentage points |
| five-choice | 5 | 20% | 6,300 | +2 percentage points |

The four-choice profile preserves the original SCI-PSI-002 contract. The five-choice profile matches the five-alternative AI replication protocol in SYMPsi AIA-01. Their chance baselines, critical values and power calculations are kept separate.

Both profiles use a one-sided exact binomial test with alpha = 0.005. The frozen manifest sets 100,000 null campaigns and 20,000 SESOI power campaigns. A discrete exact-binomial test may have a realized type-I rate below the nominal alpha; the receipt reports both.

## What it checks

- Exact one-sided binomial critical values and discrete null-tail probabilities.
- Empirical null false-positive calibration against the exact binomial model.
- Empirical power calibration at the manifest-declared smallest effect of interest.
- A strong planted synthetic effect that must cross the frozen decision threshold.
- Strict four-choice / five-choice profiles, with no implicit chance-rate conversion.
- Unique trial IDs, fixed trial count, response-before-target sequence, target-manifest binding, response-payload digest, valid status and synthetic-only status.
- Directly planted target-bearing side-channel metadata.
- Target visibility to the analyst or runner.
- Outcome-dependent stopping and post-hoc exclusions.
- Repeated experimental units presented to an iid analysis path.
- Label recoding invariance and deterministic target-order permutation/re-scoring.
- Deterministic receipts bound to the exact Git HEAD and exact manifest bytes.

The direct leakage control only detects the known planted direct-target channel in its fixture. It does **not** claim to detect every possible unknown side channel. The target-order permutation control checks marginal preservation and score recomputation; it is not evidence of psi.

## Reproduce

Run from the repository root. Use the exact commit being qualified:

    GIT_HEAD="$(git rev-parse HEAD)"

    cargo fmt --manifest-path tools/sympsi-synthetic-harness/Cargo.toml --all -- --check
    cargo test --manifest-path tools/sympsi-synthetic-harness/Cargo.toml
    cargo clippy --manifest-path tools/sympsi-synthetic-harness/Cargo.toml --all-targets -- -D warnings

    cargo run --manifest-path tools/sympsi-synthetic-harness/Cargo.toml --       --profile four-choice       --git-head "$GIT_HEAD"       --manifest tools/sympsi-synthetic-harness/fixtures/manifest-v0.1.json       --out /tmp/sympsi-four-choice-receipt.json

    cargo run --manifest-path tools/sympsi-synthetic-harness/Cargo.toml --       --profile five-choice       --git-head "$GIT_HEAD"       --manifest tools/sympsi-synthetic-harness/fixtures/manifest-v0.1.json       --out /tmp/sympsi-five-choice-receipt.json

The output file must not already exist. The runner writes a deterministic JSON receipt with the exact Git SHA, manifest SHA-256, synthetic seed identifiers, null/power calibration, control dispositions, and a SHA-256 over the serialized receipt body.

## Randomness boundary

The deterministic SplitMix64 generator and seed derivation in this package are for synthetic fixtures only. They are not cryptographic, are not an entropy source, and must never be used to select targets in a real experiment.

A real future-target experiment needs a separate independently operated target service, pinned future-beacon group configuration, independent beacon-proof verification, response-before-target witnessed chronology, and an unbiased target mapping. Those components are outside this package.

## Limitations

- The iid binomial model is used only for the synthetic independent-trial profile. Repeated participant/session units are rejected by that path; a real clustered design needs its own preregistered model and simulation-based power analysis.
- The current receipt is a software qualification artifact, not a scientific trial receipt.
- The response digest in these fixtures is an integrity check over synthetic payload bytes, not a production cryptographic commitment protocol.
- A passing harness validates only the declared synthetic cases and statistical implementation. It says nothing about whether psi exists or whether Symthaea has anomalous abilities.
- No human-subject collection is authorized by this harness. Human research needs separate ethics/privacy review, informed consent, prospective registration and independent oversight.
