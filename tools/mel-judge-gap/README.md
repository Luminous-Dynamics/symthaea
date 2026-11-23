# MEL Rank-Gap-Aware Judge Qualification v1

A standalone Rust evaluator for qualifying a **machine pairwise-preference judge** against an independent, human-derived Bradley–Terry reference.

It answers a different question from a music generator leaderboard: does the evaluator preserve candidate identity when display order changes, and does it recover the human reference ranking—especially on comparisons where the human reference says the difference is clear?

## Run

\`\`\`sh
cargo run --manifest-path tools/mel-judge-gap/Cargo.toml -- \
  tools/mel-judge-gap/fixtures/valid-judge-gap-v1.json
cargo test --manifest-path tools/mel-judge-gap/Cargo.toml
cargo clippy --manifest-path tools/mel-judge-gap/Cargo.toml --all-targets -- -D warnings
\`\`\`

## Evaluator protocol

Every machine-judge comparison has two separate presentations. The first/second sequence is committed by a SHA-256 seed and deterministically assigned per pair. The second presentation reverses which item is on the left. Both outputs are mapped back to stable item identity.

- **Swap-stable directional:** both presentations select the same item.
- **Swap-stable tie:** both presentations return tie.
- **Swap-inconsistent:** both responses are complete but select different items, or one ties and the other selects an item.
- **Incomplete:** either response abstains or is invalid.

The judge identity/version, adapter and protocol digests, prompt digests, response digests, human-reference cohort, and reference-estimator digest travel with the record. Digests bind supplied evidence; they do not prove that an API actually executed the stated model or that the UI displayed a prompt.

## Human reference and pair difficulty

The reference supplies a human-derived Bradley–Terry logit strength per item and a pair-specific standard error for the *difference* between the two item strengths. The pair-specific uncertainty is important: estimates from the same Bradley–Terry fit are correlated, so treating two item standard errors as independent would be an unjustified shortcut.

A pair is called directionally grounded only when:

\`\`\`text
abs(reference_strength_A - reference_strength_B)
    - confidence_z * reference_gap_standard_error
    > minimum_directional_gap
\`\`\`

Pairs that fail this lower-bound test remain in the gap-bin counts but are not scored as unambiguous human wins/losses. The validator reports, per gap bin:

- coverage of directionally grounded pairs;
- accuracy conditional on a stable directional judge decision;
- effective accuracy against all directionally grounded pairs (so abstentions and swap failures are not hidden);
- swap stability, swap inconsistency, and incomplete counts.

A ridge-regularized Bradley–Terry fit is computed from stable directional judge outcomes on grounded pairs. The resulting item strengths are compared with the independent human-reference strengths using Spearman rank correlation. If the comparison graph is disconnected, the tool does not emit a ranking.

## Scope limits

The included fixture has five synthetic items and ten synthetic pairs. It is a known-answer conformance test, not a live LLM judge result, human listening study, music-quality measurement, or SOTA claim.

The tool does not authenticate a model provider, enforce request-context isolation, prove that a prompt hash matches a network request, validate audio licensing, or certify that a human reference panel is representative. Production studies need independently timestamped precommit receipts, rights-cleared audio, a rater-separated human reference, and a held-out item set.
