# Benchmark-development exposure audit

**Audit date:** 2026-10-20  
**Frozen split artifact:** `splits.json`, OpenAI `math` commit `1581d4cbdd69a0faf34b2089614fe2a814316a6a`  
**Purpose:** preserve a truthful record of researcher-side exposure after the family split was frozen.

## Event E-001: family 089 was inspected before its split assignment was checked

During pilot selection, the bounded-treewidth (L_1)-embedding family was inspected to understand whether it offered a usable source-backed example. The proof file, comparator configuration, scope note, and related model definitions were read before checking the frozen split assignment. The later split lookup showed family `089` is assigned to `test`.

- **Affected family:** `089` — “Bounded-distortion (L_1) embeddings of planar and bounded-treewidth graphs.”
- **Affected sources inspected:** `CONTENTS.md`, `lean/formalization.yaml`, `lean/docs/089.md`, `lean/OAI/Combinatorics/TreewidthL1/Main.lean`, `lean/OAI/Combinatorics/TreewidthL1/Model.lean`, `lean/ComparatorChallenges/BoundedTreewidthL1.json`, and `lean/ComparatorChallenges/README.md`.
- **What did not happen:** no insight record from family `089` was committed; no model was trained on this material; no held-out model output or score was inspected; no test result was used to tune a model.
- **Disposition:** family `089` is benchmark-development/exploratory-only. Exclude it from the confirmatory test denominator for this benchmark version, keep its original frozen assignment unchanged, and do not use it for confirmatory claims.

This is a researcher-side exposure, not evidence that any model has pretraining contamination. Public-source pretraining exposure remains unknown across the dataset. The exclusion is intentionally recorded separately from `splits.json`; do not rewrite or reshuffle the frozen assignment to hide the event.

## Candidate E-002: family 156 selected after split lookup

Family `156` — “Borsuk's conjecture fails in dimension nine” — was checked in the frozen manifest before its proof was opened for candidate annotation. The manifest assigns it to `train`. Its candidate record is stored under `candidates/`, not in `records.jsonl`, and is currently unreviewed and unscored.

This sequence is recorded to preserve the distinction between using a training-family source to develop strategy representations and protecting untouched families for future confirmatory evaluation.

## Required protocol effect

1. Include this audit in every benchmark run manifest.
2. Preserve family `089` in the immutable split manifest but mark it as known-exposed and excluded from the confirmatory test denominator.
3. Report original split counts and post-screen/exposure-adjusted counts separately.
4. Never claim the remaining test families are contamination-free; audit cross-family dependencies, near-duplicates, imported declarations, and unknown pretraining exposure separately.
5. If further test-family source content is opened before a confirmatory run, append an audit entry before any model result is inspected; do not silently reassign the family.

## Event E-003: family 012 inspected for a development-only near-neighbour negative control

After checking the immutable `splits.json` assignment, family `012` was inspected to test a likely false-positive transfer suggested by shared words around primes, density, and asymptotics. The family is assigned to `validation`; this inspection makes it development-exposed for the current benchmark version.

- **Affected family:** `012` — “Independent largest prime factors of consecutive integers.”
- **Affected sources inspected:** pinned `CONTENTS.md`, `lean/docs/012.md`, `lean/ComparatorChallenges/JointDickman.lean`, and `lean/ComparatorChallenges/PrimeGaps.lean`, all at OpenAI `math@1581d4cbdd69a0faf34b2089614fe2a814316a6a`.
- **Research question:** can the family-026 “eventual inclusion preserves positive linear count” insight transfer from a lower-density statement about prime-index ratio increases to the family-012 statement about largest prime factors of consecutive integers?
- **Finding:** the stated predicates differ. Family 026 concerns indices n where p_n/n < p_(n+1)/(n+1), with p_n the nth prime; family 012 concerns integers n where P⁺(n) < P⁺(n+1), with P⁺ the largest prime factor, and a stronger joint Dickman distribution statement. The inspected family-012 statement gives no established eventual-inclusion bridge from the family-026 set and does not supply the transfer precondition by itself.
- **Disposition:** provisional curator rejection of this direct transfer as a development-only negative control (`object_type`, `predicate_mismatch`, `premise_unavailable`). This is not proof that no indirect lemma could connect the results, and it is not an observed model false-accept.
- **What did not happen:** no transfer model was run; no model output, proof outcome, or score was inspected; no source record was promoted to verified.
- **Protocol effect:** leave the immutable split assignment unchanged, but exclude family `012` from validation-based baseline selection and threshold tuning for this benchmark version. It may be retained as an explicitly development-only negative-control illustration. It must not be counted as a confirmatory validation result.

This is researcher-side exposure of a validation family, not evidence of pretraining contamination. Report original split counts and exposure-adjusted counts separately.