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

## Researcher-degrees-of-freedom stress test

Each profile also simulates 100,000 **null campaigns** with ten interim looks. Each look applies the same unadjusted one-sided alpha = 0.005 test to the cumulative data; the campaign stops at the first nominally significant look. The receipt reports the empirical campaign-level familywise false-positive rate, its 99% Wilson interval and the inflation multiple relative to the fixed-sample alpha.

This is deliberately an invalid analysis policy. The harness is expected to demonstrate that repeated unadjusted looks inflate false positives by at least 1.5x in this deterministic null simulation, while the trial-bundle validator rejects optional-stopping protocols. This is a metascience diagnostic—not a candidate procedure for the real experiment and not a psi test.

An independent 100,000-campaign Python cross-check produced approximate null FWERs of 2.239% (four-choice) and 2.278% (five-choice) for the ten-look design. The exact Rust deterministic estimates are captured in the run receipts.

The analytic reference values from the frozen manifest are:

| Profile | Critical hit count | Exact discrete type-I rate | Exact power at SESOI |
|---|---:|---:|---:|
| four-choice | 1,922 / 7,300 | 0.00472063 | 0.90429489 at 27% |
| five-choice | 1,343 / 6,300 | 0.00492610 | 0.90744448 at 22% |

These are exact-binomial model calculations for independent synthetic trials, not measured performance. The empirical campaigns are a deterministic Monte Carlo check against those analytical values.

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
- A pure four-/five-choice target-mapping primitive over **already verified** 32-byte beacon randomness: length-prefixed fields, domain-separated SHA-256, explicit scheme/chain/round/trial/response-commitment binding, and rejection sampling rather than modulo bias. Frozen known-answer vectors are cross-checked against an independent Python implementation.
- Deterministic receipts bound to the exact Git HEAD and exact manifest bytes.

The direct leakage control only detects the known planted direct-target channel in its fixture. It does **not** claim to detect every possible unknown side channel. The target-order permutation control checks marginal preservation and score recomputation; it is not evidence of psi.

The target-mapping function consumes a caller-supplied `verified_randomness` field and a response-commitment digest, but it does **not** verify a drand signature, trust root, timestamp, or response-before-target chronology. Those remain the separate responsibilities of SCI-PSI-003. Its frozen known-answer vector maps the specified test beacon/context/response commitment to target index 2 under both four- and five-choice mappings; independent target selection for actual experiments must use the complete versioned manifest and verifier receipts.

The known-answer inputs use Quicknet chain hash `52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971`, scheme `bls-unchained-g1-rfc9380`, round 1000, the published example's signature-derived randomness `fe290beca10872ef2fb164d2aa4442de4566183ec51c56ff3cd603d930e54fdd`, and the SHA-256 digest of the synthetic response-commitment string `sympsi-response-commitment-000042`. The frozen SHA-256 target transcript is `cd4ed4b3a8ab1da6185ea3a535bd6b87b71862f606a9d03cc5a388e8b854bde6`; its unbiased 4-way and 5-way outputs are both zero-based index 2. The drand input values are from the published [drandite-core Quicknet verification example](https://docs.rs/drandite-core/latest/drandite_core/), not a live experiment.

## Independent cross-language mapping check

A dependency-free Go reference implementation lives at `tools/sympsi-mapping-verifier-go/`. It re-encodes the transcript independently, pins the same Quicknet known-answer digest, checks 4-way and 5-way results, and tests round/response-commitment domain separation plus the 5-way rejection boundary. It is a test oracle only; Rust remains the production-side implementation candidate.

Reproduce:

    cd tools/sympsi-mapping-verifier-go && go test -count=1 -v ./...

CI pins Go 1.27.4 for this cross-language check. Agreement between two implementations verifies transcript interoperability for the frozen vectors; it does not verify a real beacon signature, trust root, freshness or response chronology.

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
