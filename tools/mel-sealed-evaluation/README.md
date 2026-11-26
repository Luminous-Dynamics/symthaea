# MEL Sealed Evaluation v1

A standalone Rust reference scorer for a leakage-resistant music-judge study. The protocol separates three byte-addressed inputs:

1. A frozen candidate manifest. It contains candidate/audio/prompt digests, generator provenance, rights-receipt references, the judge identity, comparison protocol, split declaration and precommitted order seed. It contains no human preference labels or item strengths.
2. A frozen prediction bundle. The judge returns two separately executed presentations per pair in reversed order. The bundle binds the raw manifest bytes by SHA-256 and contains the revealed order seed, the judge responses and a freeze-receipt digest.
3. A human-reference release. This is created and access-controlled separately. It binds both the raw manifest and prediction bytes by SHA-256 and contains human-derived item strengths and pair-specific uncertainty. It must be released after the prediction freeze.

The scorer checks the byte-level links, schema identities, candidate and pair coverage, seed commitment, actual swapped-order presentation, response digests, reference coverage and calendar-valid declared chronology. It reports stable directional decisions, ties, abstentions, order-inconsistent judgments, human-reference coverage, conditional/effective accuracy, rank-gap-stratified results and a reference-probability MSE when the judge supplies probabilities. That MSE compares the judge probability with the logistic transform of a fitted human Bradley–Terry strength gap; it is not a conventional Brier score computed against raw binary human outcomes. It does not silently convert ties or abstentions into votes.

## Run

From the repository root:

    cargo run --manifest-path tools/mel-sealed-evaluation/Cargo.toml -- \
      tools/mel-sealed-evaluation/fixtures/manifest-v1.json \
      tools/mel-sealed-evaluation/fixtures/predictions-v1.json \
      tools/mel-sealed-evaluation/fixtures/human-reference-v1.json

    cargo test --manifest-path tools/mel-sealed-evaluation/Cargo.toml
    cargo clippy --manifest-path tools/mel-sealed-evaluation/Cargo.toml --all-targets -- -D warnings

The CLI writes a JSON report to stdout and exits non-zero when a structural, digest, identity, order or chronology check fails.

## Operational separation

The intended run sequence is:

1. Freeze and independently publish/anchor the candidate manifest commitment and order-seed commitment.
2. Give the judge only the candidate manifest's permitted fields and candidate assets. Do not mount, expose or send the human-reference package to the judge process, tool context, retrieval layer, cache or reward-model training path.
3. Collect both order presentations. Freeze the complete prediction file and anchor its digest independently.
4. Only then release the human-reference package to the scoring process.
5. Run this evaluator in a scorer environment that has access to all three files but is not used for judge inference.
6. Preserve the three source files, report, exact source revision and independent receipts as one evidence bundle.

A digest link proves that supplied bytes match the referenced bytes. It does not prove that the judge process lacked another route to the human labels. Likewise, the timestamp strings and receipt digests in v1 are checked as declared metadata; the crate does not validate a trusted timestamp authority, digital signature, key ownership, access-control log or isolated execution environment. Production qualification must add independently verifiable freeze/release receipts and access-boundary evidence.

## Important claim boundary

The included fixture is synthetic. A successful run establishes deterministic reference-tool conformance for the supplied bytes only. It is not a human listening result, generator ranking, reward-model result, proof of human preference, proof of rights ownership, proof of sealed execution or SOTA claim.

## Relationship to adjacent MEL work

- MEL-PREF-ASSAY-001 (#13212) checks human presentation assignment, rater exposure, identical-output controls and order-conditioned preferences.
- MEL-JUDGE-GAP-001 (#13218) checks machine-judge swap stability and agreement with a human-derived Bradley–Terry reference, including rank-gap strata.
- MEL-SEALED-EVAL-001 binds the held-out candidate, prediction and reference-release stages so the judge's predictions can be frozen before the gold reference is released. It is an orchestration and leakage-boundary contract, not a replacement for either evaluator.
