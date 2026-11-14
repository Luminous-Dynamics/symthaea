# Decision-time evidence coverage — research note

**Date:** 2026-11-14  
**Status:** proposed metric and development smoke test; no comparative result.

## Why retrieval coverage is not enough

TEMPO (EMNLP 2026) introduced temporal precision/coverage measures for reasoning-intensive retrieval over 13 domains. Its corpus contains 1,730 complex queries and 3,976 decomposed retrieval steps; the best reported system, DiVeR, reached 32.0 NDCG@10 and 71.4% Temporal Coverage@10. This shows that retrieving some relevant documents is not the same as recovering evidence across every required time period.

Reference: https://aclanthology.org/2026.emnlp-main.291/

Our task is narrower and stricter. We do not only ask whether a retrieval system found a temporally relevant document. We ask whether a decision system used the correct source versions, authority states, claim dimensions, effective intervals and knowledge cutoffs for the decision it actually made.

A retriever can have high temporal coverage and still return an issuer outside its authority, use a source outside its claim competence, or confuse a correction's effective time with the date the institution learned about it.

## Proposed metric: Decision-Time Evidence Coverage (DTEC)

For each adjudicated decision query (q), define a required set (R_q) of critical source/version/authority references. Let (M_q) be the subset correctly recovered, source-bound and temporally applicable in the system's packet.

`DTEC = |{q : R_q ⊆ M_q}| / |Q|`

This is an all-required-evidence metric: a query slice is complete only when every critical reference needed for that slice is recovered and bound to the correct effective-time and knowledge-time semantics.

Report alongside it:

- **Critical evidence recall:** fraction of required critical references recovered.
- **Evidence precision:** fraction of returned material references relevant to the query and within source competence.
- **Version binding accuracy:** correct source version and digest for each relied-on claim.
- **As-of reconstruction accuracy:** correct view for each `(effective_at, known_at)` pair.
- **Authority-time accuracy:** whether the delegated authority was valid and observed at the time of the decision/action.
- **Decision replay accuracy:** whether the decision disposition and reason codes can be reproduced from the frozen packet.
- **Uncertainty and abstention quality:** whether missing, stale, contested or unavailable evidence remains explicit.
- **Over-collection/disclosure burden:** evidence requested or replicated beyond the minimum required for the decision.

Do not collapse these metrics into one score. Safety failures cannot be compensated by retrieval speed or document count.

## Current IIC-010 smoke test

IIC-010 contains two temporal slices:

1. Historical effective time 2026-02-15, known-at 2026-11-14.
2. Current effective time 2026-11-14, known-at 2026-11-14.

It requires the system to retain the intermediate 60-day policy and the current 30-day policy, without applying the latter retroactively or losing the earlier decision-time view. The Rust evaluator counts a slice complete only when every oracle-mapped candidate reference for that slice is recovered with the exact source span and source version.

Two slices are a smoke test of metric semantics, not a statistically meaningful benchmark. Expansion should create at least 200 adjudicated temporal slices across policy amendments, late corrections, authority changes, source outages and supersession.

## Prospective research targets (not results)

- At least 98% DTEC on a held-out, adjudicated set, with confidence intervals and per-domain breakdowns.
- Zero false critical clears on temporal hard negatives.
- At least 95% authority-time correctness on held-out delegation/revocation cases.
- At least 90% dependency-invalidation recall while keeping unaffected-control invalidations below 5%.
- At least 30% reduction in unnecessary evidence requests or expert reconstruction effort, without increasing privacy exposure or critical errors.

## Metamorphic properties

- A later correction may change the current effective view but must not rewrite the earlier known-at view.
- Adding an unrelated source must not change the decision or required evidence set for an independent claim.
- Reordering independent source records must not change the reconstructed decision.
- A missing source remains missing; the system must not infer either positive or negative facts from absence.
- A source with a valid signature but insufficient competence must not satisfy a claim outside its scope.
- A source-version change must invalidate only dependencies that rely on the changed version.

## Important limits

TEMPO's NDCG and Temporal Coverage are retrieval metrics; DTEC is a proposed decision-evidence metric and is not directly comparable to their leaderboard. IIC-Eval v0.2 is a synthetic development corpus. Its oracle is not independently adjudicated and no Luminous, incumbent-vendor or general-purpose-model performance has been measured.
