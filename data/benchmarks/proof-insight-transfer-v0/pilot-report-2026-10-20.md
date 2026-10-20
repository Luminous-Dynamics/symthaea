# Proof Insight Transfer v0: pilot source audit

**Date:** 2026-10-20  
**Protocol:** [evaluation-protocol.md](evaluation-protocol.md)  
**Source snapshot:** OpenAI `math` at `1581d4cbdd69a0faf34b2089614fe2a814316a6a`  
**Status:** two unscored candidate annotations; two direct transfer hypotheses rejected; no measured benchmark result.

## Why this pilot exists

A benchmark of proof insights must be able to reject a strategy that merely sounds general. The goal is not to annotate every proof with a persuasive summary; it is to capture preconditions and failure boundaries, then demand evidence that the move helps on an independently relevant target.

## Candidate A — family 156: finite obstruction through faithful representation

**Family:** `156`, “Borsuk's conjecture fails in dimension nine” (**train** in the frozen manifest).  
**Record:** [156-borsuk-nine-finite-obstruction.json](candidates/156-borsuk-nine-finite-obstruction.json).  
**Primary source:** `lean/OAI/Geometry/Borsuk/Counterexample.lean`.

The proof translates a hypothetical ten-set small-diameter cover of a projector set into an `Admissible` representation, derives a finite support-face `TenSystem`, and contradicts a finite obstruction. The candidate record explicitly requires preservation of coverage and diameter constraints; its current formal-verification, semantic-review, and mutation statuses are all unattempted/unreviewed.

## Direct-transfer test — family 156 → family 157: rejected

**Target:** family `157`, “Graph coloring, clique minors, and Colin de Verdière invariants” (**train**). Its relevant main result is `OAI.HadwigerCounterexample.not_hadwiger_conjecture`.

The target proof chain was inspected only after checking the frozen split. Its construction uses analytic parameters and sampled/finite graph transfer; then it composes a distinct graph-invariant argument. In `HadwigerMatching/MinorBound.lean`, the bridge bounds the Hadwiger number by graph order and connected-matching number. In `HadwigerMatching/Main.lean`, a separate inequality uses independence number and rational arithmetic to put the Hadwiger number strictly below the chromatic number.

The source insight's particular preconditions—Borsuk cover representation, support faces, a `TenSystem`, and a geometric finite-obstruction theorem—do not match this target. No corresponding target obligation was identified; no source lemma or tactic was transferred. **Disposition: reject this direct strategy candidate.** A generic resemblance (“both proofs have stages”) is not enough to count as transfer.

This is a legitimate negative result about the *proposed mapping*, not proof that all abstract proof-decomposition ideas are useless. The source record marks the transfer to family 157 as `rejected` and records why.

## Candidate B — family 157: counterexample via a bounded-invariant bridge

**Record:** [157-hadwiger-invariant-chain.json](candidates/157-hadwiger-invariant-chain.json).  
**Primary source:** `lean/OAI/Combinatorics/HadwigerCounterexample/Main.lean`.  
**Bridge:** `lean/OAI/Combinatorics/HadwigerMatching/Main.lean` and `MinorBound.lean`.

The formal chain is:

1. Construct arbitrarily large finite graphs with independence number at most two and connected-matching number (c<m/100).
2. Apply the matching–minor bound (3h(G)\le m+4c(G)+2), where (h(G)) is the Hadwiger number.
3. Deduce (h(G)<26m/75+2/3<m/2\le\chi(G)) for the stated size range.
4. Use the strict inequality (h(G)<\chi(G)) to refute the universal Hadwiger inequality.

The reusable *hypothesis* here is not “use a finite obstruction” but “separate expensive witness construction from a reusable invariant-comparison bridge and a small strict-inequality discharge.” A transfer requires the target to supply compatible invariants and the bridge's premises. This record is candidate-only; it does not claim the session rebuilt Lean, reran Comparator, or discovered a new theorem.

## Second direct-transfer control — family 157 → family 184: rejected

**Target:** family `184`, “Correspondence coloring with a fixed forbidden subgraph” (**train**). Its `CliqueFreeLog.logarithmic_independence_bound` theorem requires a lower bound on the independence number in terms of graph order and average degree for clique-free graphs.

The source lemma `card_le_twice_chromaticNumber_of_indepNum` assumes `indepNum G ≤ 2` and derives `|V(G)| ≤ 2χ(G)`: it uses an **upper bound** on independent-set size to lower-bound chromatic number. It cannot be reversed to prove a lower bound on the same independence number under different hypotheses. The target instead uses a weighted optimizer, degree cutoff, and maximum-degree independence argument. Disposition: reject direct transfer with `inequality_direction`, `predicate_mismatch`, `precondition_mismatch`, `domain_mismatch`, and `quantifier_scope` codes. No tactic or source lemma was applied to the target. The typed record is in [`candidates/157-hadwiger-invariant-chain.json`](candidates/157-hadwiger-invariant-chain.json); these are curator-level negative controls, not empirical prover failures.

## Integrity and exposure controls

- Both family 156 and family 157 were confirmed as `train` in the frozen split before their proof sources were inspected for these candidate annotations.
- Family `089` remains known-exposed for this workflow because it was inspected before its frozen `test` assignment was checked. See [exposure-audit.md](exposure-audit.md). Its frozen assignment remains unchanged, but exclude it from confirmatory scoring and report original versus adjusted denominators.
- Candidate records live under `candidates/`; they are not eligible benchmark rows and are not included in scores.
- Source repository and commit, manuscript and Lean source hashes, statement text digest, toolchain, and Mathlib revision are recorded. The statement digest is over stored UTF-8 declaration text—not a normalized elaborated Lean expression digest.
- Manuscript license status remains `needs_review`.
- No target transfer was attempted with a prover, and no proof-verification, successor-test, independent reviewer, mutation-run, or SOTA result is claimed.

## Engineering validation status (2026-10-20)

The dedicated GitHub Actions workflow was observed at run `39436173031` on earlier branch head `500a0456e6ab2c343164af57ea8f04a745377a4a`. The pinned Rust toolchain setup, isolated-crate staging, and `cargo fmt --check` completed successfully, but the next step (`cargo test --locked`) failed **before compiling or running the tests**: Cargo said the copied isolated `Cargo.lock` needed updating. Later steps, including the independent JSON Schema validator and source-index reproducibility check, were skipped. This is a lockfile/isolated-manifest graph mismatch, not a unit-test assertion failure.

A follow-up root-lock patch `0c17139e142f640e8ac6299f7e62c6e55d750a69` adds the `const-oid 0.9.6` dependency edge to the locked `digest 0.10.7` package to reflect the workspace's `sha2/oid` feature used by the isolated crate. The workflow was also updated to validate both checked-in candidate records rather than only family 156 (`1dd188825f653faac10b4ce9b4f544f793a1e229`). **No post-fix run has been observed yet**, so the lockfile repair and candidate-validator execution remain unverified. Do not infer success from the source change.

## What the pilot taught us

1. **Transfer needs a type system, not a similarity score.** Domain objects, bridge lemmas, invariants, and proof obligations must line up. A superficial shared proof shape should be rejected.
2. **Negative outcomes belong in the dataset.** Keep rejected candidates and their reasons; otherwise retrieval systems are trained only on appealing stories and suffer survivor bias.
3. **A proof insight needs at least three independent tests:** provenance/statement fidelity, correct matching of target preconditions, and measured target progress under matched compute.
4. **The next experiment must be domain-matched.** For the invariant-bridge candidate, inspect a separate training-family graph theorem only after checking the immutable split; then write target obligations before trying any transfer. Do not inspect untouched test-family proofs to search for convenient examples.

## Next acceptance gates

1. Run the Rust indexer and its unit tests with a pinned Lean/math source checkout; compare all 372 family assignments and the source/partition digests against `splits.json`.
2. Validate the record schema with an independent JSON Schema Draft 2020-12 validator; add a schema-validation command to the future dataset gate.
3. Independently review the two candidate records and verify their source statement/artifact digests. For formal claims, run the pinned project and proof audit with the recorded axiom policy.
4. Build a domain-matched training/validation pilot with explicit positive and near-miss transfer pairs. Freeze the proposed target proof obligations before runs.
5. Only then run the pre-registered baseline arms. A “win” requires the protocol's semantic-gate outcome and confidence interval; compilation-only gains do not count.
