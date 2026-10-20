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
