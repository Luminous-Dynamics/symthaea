# IG-MSG-UTILITY-001 — message-level evidence utility v0.1

**Owner:** [Symthaea #12805](https://github.com/Luminous-Dynamics/symthaea/issues/12805)  
**Composes with:** #11337/#11360, #12762, #12775, #9431, #639 and #3095  
**Status:** deterministic synthetic qualifier; no LLM message-pruning campaign has run.

## Research thesis

Measure whether each message contributes decision-relevant evidence for its specific recipient, then preserve, compact, omit or request it under explicit cost and information-integrity constraints.

The direct external comparator is **MessageSieve** (EMNLP 2026): https://aclanthology.org/2026.emnlp-main.2179/. Its authors report 19–49% evaluation-token reduction and +2.59 to +4.10 macro-average points over dense Full-MAS in their tested model/task settings. Those are external published results, not Symthaea results.

## Included

- 20 deterministic fixtures covering source-root echoes, decisive minority evidence, negative evidence, polarity reversal, stripped provenance/uncertainty/scope/freshness, unauthorized disclosure, stale facts, omission audit loss, compact message cost regressions and required evidence omission.
- Versioned JSON Schema.
- Standalone Rust structural qualifier.
- Exact-head GitHub Actions workflow using Rust 1.96.0.

## Core invariants

- A message is identified by an evidence root, not by its sender, wording or message ID alone.
- Multiple messages derived from one root add at most one independent root.
- An omitted message is absent from the receiver prompt but its source/audit record remains present.
- A compact message must preserve source, root, evidence ID, polarity, scope, freshness, uncertainty and authorization.
- Compact messages must retain a digest/reference to the original message and cost fewer tokens than the original.
- An unauthorized message must not be transmitted.
- A required root cannot be silently omitted in a deterministic fixture.
- A stale/unknown fact must not be presented as current.
- Negative evidence and minority evidence receive the same provenance-preservation requirements as positive/majority evidence.

## Run

```bash
cargo fmt --manifest-path research/ig-msg-utility-001/Cargo.toml --check
cargo test --manifest-path research/ig-msg-utility-001/Cargo.toml
cargo clippy --manifest-path research/ig-msg-utility-001/Cargo.toml --all-targets -- -D warnings
cargo run --manifest-path research/ig-msg-utility-001/Cargo.toml -- \
  research/ig-msg-utility-001/MESSAGE_UTILITY_FIXTURES_V0_1.json
```

## Validation status

- The 20 frozen synthetic fixture expectations were independently cross-checked against the stated structural scoring semantics: **20/20 match**.
- The standalone Cargo manifest includes its own `[workspace]` boundary so Cargo does not accidentally load the parent repository workspace.
- Native Rust compilation, formatting, tests and Clippy have not yet run on this branch; the exact-head workflow is configured to perform them. No CI pass is claimed.
- No LLM message-pruning, summarization or provider calls have been run. The fixture results are not MessageSieve comparisons.

## Claim ceiling

A passing fixture establishes only the synthetic structural contract. It does not demonstrate real-model token savings, non-inferior task accuracy, privacy guarantees, semantic equivalence of generated summaries, or superiority to MessageSieve. Those require matched-backbone, matched-budget model experiments, held-out task families and independent scoring.
