# ENG-REA-009: extraction fidelity benchmark

This is a provider-neutral scorecard contract for generated, inert firmware-container fixtures. It moves the evaluation target from “a provider emitted a plausible report” to “the provider recovered the known answers and accounted honestly for every source byte.”

## Why this matters

Unblob advertises recursive extraction, byte-region reporting and unknown-chunk carving; its format catalogue also documents format-specific partial-support limitations: <https://unblob.org/formats/> and <https://github.com/onekey-sec/unblob>. Binwalk documents signature/offset mapping, recursive extraction, and entropy analysis: <https://binwalk.app/> and <https://github.com/ReFirmLabs/binwalk>. Those are capabilities, not evidence that either tool is correct on every image.

The same independent oracle should score every provider. A parser's own report cannot define its own truth.

## Files

- `cases.json`: 13 deterministic fixture recipes covering nested transforms, leading/trailing bytes, adjacent containers, path collisions, false-positive magic, corrupt/truncated inputs, depth limits, padding, single-byte offsets, and empty files.
- `oracle-policy.json`: meaning of pass/partial/fail/abstain, metrics, and proposed acceptance targets.
- `example-oracle.json` / `example-observation.json`: an intentionally synthetic 10-byte contract fixture used to exercise the scorer.
- `verify-contract.sh`: validates the catalog and runs positive/negative end-to-end scorecard controls.
- `schemas/rea-differential-*.schema.json`: versioned contracts for the catalog, per-case oracle, provider observation, and scorecard.
- `crates/domains/symthaea-rea-evidence-verifier/src/differential.rs`: provider-neutral scoring library.
- CLI: `symthaea-rea-evidence-verifier score-case --oracle ORACLE.json --observation OBSERVATION.json`.

## Status contract

- **pass** — source identity exact; every expected output matches path, logical ID, digest and byte length; no extra outputs; regions exact; byte ledger is gap-free and non-overlapping; no unknown, unsupported, or ambiguous spans; completeness claim accepted.
- **partial** — source/output/region findings did not falsify the expected tree, but some bytes remain unknown, unsupported, ambiguous or unexplained, or the provider conservatively does not claim completeness.
- **fail** — wrong source, wrong/missing/extra file, wrong content hash, path collision, false/missing region, malformed path/range, overlapping byte ledger, or false completeness claim.
- **abstain** — provider explicitly says the case is unsupported/unknown and does not emit a purported extraction or claim completeness. Abstention earns **no recall credit**.

An abstention is not a success and is not counted as a correct extraction. Publish it as its own metric.

## Proposed benchmark targets (not measurements)

| Metric | Goal on provider-supported known-answer cases |
|---|---:|
| Exact known-file recall | 100% |
| False-positive emitted files | 0 |
| Unexplained source bytes | 0 |
| Double-accounted source bytes | 0 |
| Exact expected region precision/recall | 100% |
| False complete-extraction claims | 0 |
| Repeated-run semantic projection agreement | 100/100 |
| Rejection of enumerated injected contract defects | 100% |

Score at least five separate dimensions: **detector accuracy**, **recovery fidelity**, **byte-accounting integrity**, **coordinate-space/transform fidelity**, and **provenance/lineage fidelity**. Runtime and memory should be reported separately, never traded against silent byte loss.

Do not report one aggregate “firmware extraction accuracy” percentage without the denominators, eligible provider-format matrix, abstention rate, fixture-family split, and confidence intervals. Providers that do not support a format should not be punished as if they claimed to parse it; equally, an unsupported result must never be converted to a pass.

## Run

From the repository root:

    cargo fmt --package symthaea-rea-evidence-verifier
    cargo test -p symthaea-rea-evidence-verifier
    sh benchmarks/rea-differential/v1/verify-contract.sh

Score a provider adapter output:

    cargo run -p symthaea-rea-evidence-verifier -- score-case \
      --oracle case-oracle.json \
      --observation provider-observation.json

Only a full pass returns exit status 0. Partial, failed and abstained scorecards are emitted as JSON, then return a non-zero status.

## Honest scope boundary

This tranche contains the scoring contract, fixture taxonomy, schemas, and synthetic positive/negative controls. **It does not yet contain end-to-end Unblob/Binwalk adapters for all 13 recipes**, so it makes no cross-tool SOTA claim. The next gate is a deterministic fixture builder plus one adapter per pinned provider, with raw output retained and independently scored. Then add authorized, held-out real firmware (permission/licensing reviewed) as a separate evaluation set. Never execute extracted firmware, flash hardware, or run recovered device code as part of this extraction benchmark.
