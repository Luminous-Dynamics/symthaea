# VART-WORLD-CREATIVE-001 — Post-Unblinding Analysis Audit v1

Status: post-seal scientific audit contract. This document does **not** alter Freeze v3, the frozen analysis contract, hypotheses, endpoints, thresholds, multiplicity, or missingness rules.

## Purpose

The first unblinded analysis is immutable evidence about what the frozen analysis program actually produced. If that program or its human-readable summary is inconsistent with the frozen design, the original analysis receipt remains preserved and a **correction lineage** may be produced only by applying the pre-existing frozen contract mechanically to the same sealed evidence root.

No correction may introduce a new endpoint, estimator, threshold, exclusion, weighting rule, scalar aggregate, or preferred subset after unblinding.

## Frozen design accounting

Freeze v3 contains 64 revision-trials:

- H1 / 001A core: 8 independent clusters × 3 policies at revision 0 = 24 trials.
  - FULL: 8
  - RANDOM_VALID: 8
  - HEURISTIC: 8
- H2 / 001A longitudinal continuation: 8 FULL lineages × revisions 1..3 = 24 additional FULL trials.
- H3 / 001B MemoryTrap: 8 independent clusters × 2 policies at revision 0 = 16 trials.
  - FULL: 8
  - NO_LEDGER: 8

Therefore:

- the H1 confirmatory paired comparison uses exactly 8 FULL r0 trials, 8 RANDOM_VALID r0 trials, and 8 HEURISTIC r0 trials;
- the total campaign contains 40 FULL trials, not 24;
- H2 uses the 8 FULL lineages with r0..r3, but those repeated revisions are not additional independent H1 replicates;
- H3 uses 8 paired MemoryTrap clusters.

## Exact-test resolution

For an exact one-sided sign-flip/randomization test with `n` independent nonzero paired units, the smallest attainable p-value is `1 / 2^n`.

Consequences:

- with 8 independent H1 paired clusters, an exact one-sided sign-flip p-value cannot be smaller than `1/256 = 0.00390625` unless the frozen analysis contract prospectively specifies a different exact randomization space;
- with 8 negative H2 lineage slopes and no ties, `p = 1/256` is attainable for the one-sided all-negative extreme;
- for a binary H3 confusion endpoint with 7 positive paired differences and 1 zero difference, only 7 nonzero signs are informative, so the one-sided sign test floor is `1/128 = 0.0078125`; retaining the zero as two duplicate sign assignments does not change that probability;
- if H3's second preregistered endpoint (task performance) has 8 nonzero paired differences, it may separately attain `1/256`, subject to the frozen multiplicity rule.

## Co-primary H1 claim admission

A claim of H1 vector advantage requires the frozen co-primary family to be evaluated on the eight paired r0 clusters. At minimum, the claim packet must report separately:

- independently reconstructed prediction/calibration error if H1 prospectively includes it;
- declared-goal consequence with its frozen superiority criterion;
- physical validity with its frozen noninferiority or pass criterion;
- protected side effects with its frozen noninferiority margin;
- FULL vs RANDOM_VALID and FULL vs HEURISTIC contrasts as required by Freeze v3;
- the frozen multiplicity/gatekeeping result.

Means alone are not sufficient for co-primary claim admission. No scalar `world_quality`, `creative_score`, `intelligence_score`, or equivalent aggregate may be introduced.

## H2 wording boundary

A negative fitted slope is evidence of a negative longitudinal trend. It is **not** equivalent to a strictly monotonic revision-by-revision sequence.

If all eight preregistered world-level slopes are negative and the frozen exact test passes, an admissible conclusion is:

> Prediction error decreased over repeated interventions across the eight preregistered FULL lineages under the frozen trend estimator.

The result does not by itself establish a general causal world model, general intelligence, or universal learning ability.

## H3 wording boundary

The NO_LEDGER ablation can support a causal contribution of ledger-mediated provenance separation **within the preregistered MemoryTrap benchmark and architecture**, provided both frozen endpoints and multiplicity rule are satisfied.

It does not establish universal necessity of a particular implementation for all cognition, memory, or persistent-world systems.

## Correction lineage

If the first analysis fails this audit:

1. Preserve the original pre-unblinding anchor, analysis program bytes, stdout/stderr, output files, and analysis receipt unchanged.
2. Record a post-unblinding audit receipt that identifies each deterministic contract mismatch.
3. Correct only the analysis implementation or summary mapping necessary to conform to the already-frozen contract.
4. Hash and record the corrected program before execution.
5. Re-run against the exact same sealed evidence root, with no exclusions or data changes.
6. Emit a correction receipt linking:
   - Freeze v3 SHA-256;
   - campaign receipt SHA-256;
   - original analysis receipt SHA-256;
   - post-unblinding audit receipt SHA-256;
   - corrected program SHA-256;
   - corrected output closure SHA-256.
7. Report both the original and corrected analysis histories. Never delete or overwrite the first analysis.

## Claim ceiling

Until the corrected claim packet passes the frozen contract and this post-unblinding audit, the appropriate status is:

`UNBLINDED_RESULTS_PROMISING_CLAIM_ADMISSION_PENDING`

not `EvidenceBoundExperienceConditionedWorldImprovementQualified`.
