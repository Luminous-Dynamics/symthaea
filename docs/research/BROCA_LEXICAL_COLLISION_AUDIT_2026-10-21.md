# Broca Exact Lexical-Prime Collision Audit

Date: 2026-10-21
Status: research prototype; runtime evidence pending

## Problem

A lexicon can assign distinct words the same semantic-prime sequence. The HDC projection then cannot distinguish those words from that sequence alone. The observed `mother` / `father` pair maps to `[Someone, Live]` for both entries. This is a representational collision; it is not evidence that the words are synonyms.

Fixing one pair by hand would hide the systemic issue. The first useful capability is to make the collision set observable and reproducible before changing the representation.

## Implementation in this change

- `LexicalGrounding::identical_prime_groups()` enumerates exact **ordered** prime-sequence collisions.
- Result groups and member words are sorted deterministically, so reports do not depend on randomized `HashMap` iteration order.
- A unit test checks determinism and verifies that the known `mother` / `father` collision is surfaced with its exact shared prime sequence.
- `cargo run -p symthaea-core --example lexical_alias_audit` emits a versioned JSON report with lexicon size, group count, number of words involved, groups, and an optional `GIT_COMMIT_SHA` source revision.

The report intentionally does not claim synonymy or contextual equivalence. It audits exact collisions in the symbolic lexical table only; it does not yet measure collisions induced by blending with genesis vectors, tokenization, contextual composition, or downstream generation.

## Research direction

The paradigm shift is from treating a semantic embedding as a complete meaning to treating it as a **typed, auditable semantic projection**:

1. Preserve shared semantic-core features.
2. Preserve an explicit lexical residual for meaning that the core does not encode.
3. Keep relation labels—equivalent, antonym, related-but-not-equivalent, role-reversed, ambiguous—separate from representation similarity.
4. Require an independent verifier to score whether the output retains the intended relation and role.
5. Report collisions and uncertainty rather than converting missing distinctions into confident claims.

Do not add arbitrary pseudo-universal primes simply to force words apart. The feature model should be explicitly typed and linguistically reviewed, with the shared core still available for generalization.

## Evaluation plan

### Gate A — audit completeness

- Every exact collision group is reported exactly once.
- Group order and word order are deterministic.
- Reported group count equals the number of groups emitted; participating-word count equals the sum of group sizes and never exceeds total lexicon size.
- The report records schema version and exact source revision where supplied.
- No report labels an exact collision as semantic equivalence.

### Gate B — independently reviewed relation set

Build a versioned dataset with balanced strata for:
- true or near equivalence;
- antonymy and polarity reversals;
- related-but-non-equivalent terms (including family roles);
- subject/object or agent/patient reversals;
- temporal and causal relations;
- unsupported and ambiguous terms.

Split confirmation data by lexical family and construction template, not by randomly splitting pairs. Use multiple reviewers or a documented adjudication process for ambiguous cases.

### Gate C — proposed quantitative targets

These are targets to preregister before opening confirmatory data, not current results:

- 100% completeness of exact symbolic collision reporting;
- at least 95% macro precision and recall for the relation classifier on the independently labeled set;
- zero silent certainty upgrades when the relation is unknown or ambiguous;
- report a false-collapse rate separately for each semantic relation class, with paired confidence intervals;
- target less than 1% false collapse on the curated high-confidence *related-but-not-equivalent* stratum, then expand the corpus before making broad claims;
- evaluate unseen lexical families and composed phrases; report known-word controls separately;
- keep semantic initialization opt-in until exact-head code validation and paired held-out evidence support enabling it.

The symbolic alias audit is deterministic and should have no statistical confidence interval; model and relation-classification scores do need uncertainty estimates. Never merge these evidence types into a single headline score.

## Industry comparison and claim boundary

Use established linguistic evaluation as complementary context rather than pretending this narrow audit replaces it:

- Holmes evaluates more than 200 datasets across morphology, syntax, semantics, reasoning, and discourse, and reports that no tested model consistently leads across all the covered areas: https://aclanthology.org/2024.tacl-1.88/
- CheckList motivates minimum-functionality, invariance, and directional-expectation tests because ordinary held-out accuracy can miss important behavioral failures: https://aclanthology.org/2020.acl-main.442/

Compare a frozen Broca baseline, semantic initialization, and relevant external models on the same inputs and independent scorer where technically possible. Record model revision, prompt, decoding settings, compute budget, and dataset digest. Report per-stratum performance, false collapses, latency and memory; do not infer general SOTA from this targeted slice.

## Execution status

- GitHub source read-back confirms the implementation and test are present on this branch.
- The JSON audit example is committed, but it has **not** been executed in this environment.
- No Cargo test, format, Clippy, CI pass, or measured model advantage is claimed.
- Follow-up issue: https://github.com/Luminous-Dynamics/symthaea/issues/8259
