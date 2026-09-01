# VART-WORLD-CREATIVE-001 — Independent Verifier Negative Suite

Status: required before confirmatory claim admission.

The independent verifier is not qualified merely by accepting a valid evidence bundle. It must reject deliberately corrupted or scientifically invalid bundles that preserve superficial plausibility.

## N1 — Prospective hypothesis mutation

Start from a valid complete trial. Change one predicted effect after the applied receipt timestamp while preserving all other records.

Expected: REJECT — `POST_HOC_HYPOTHESIS_MUTATION`.

## N2 — Candidate-set substitution

Replace the FULL policy candidate-set digest with a different physically valid set while retaining the same paired block ID as RANDOM_VALID.

Expected: REJECT — `PAIRED_CANDIDATE_SET_MISMATCH`.

## N3 — Random-valid cherry picking

Replace the random draw receipt with a selection that is admissible but was not produced by the frozen RNG state/algorithm.

Expected: REJECT — `RANDOM_VALID_DRAW_MISMATCH`.

## N4 — Inadmissible random candidate

Select a candidate that failed support or obstacle admission.

Expected: REJECT — `SELECTION_NOT_PHYSICALLY_ADMITTED`.

## N5 — World-version splice

Bind an outcome/revisit from another otherwise valid world lineage.

Expected: REJECT — `WORLD_VERSION_CHAIN_MISMATCH`.

## N6 — Counterfactual provenance promotion

Replace a committed historical observation reference with a structurally equal counterfactual observation.

Expected: REJECT — `PROVENANCE_DOMAIN_SUBSTITUTION`.

## N7 — Missing revisit hidden as completion

Remove the revisit observation and retain a synthetic RevisionOutcome.

Expected: REJECT — `OUTCOME_WITHOUT_REVISIT`.

## N8 — Proposal/receipt mismatch

Bind an applied receipt for candidate B to a CreativeTrial whose selected proposal is candidate A.

Expected: REJECT — `SELECTED_PROPOSAL_RECEIPT_MISMATCH`.

## N9 — Pilot contamination

Copy a valid pilot trial into the confirmatory evidence root and mark it included.

Expected: REJECT — `PILOT_CONFIRMATORY_CONTAMINATION`.

## N10 — Selective trial omission

Remove one preregistered failed trial from the result set while leaving aggregate counts internally consistent.

Expected: REJECT — `PREREGISTERED_TRIAL_MISSING`.

## N11 — Duplicate success injection

Duplicate a successful trial under a new file path without changing its trial identity.

Expected: REJECT — `DUPLICATE_TRIAL_IDENTITY`.

## N12 — Aggregate score insertion

Add `world_quality`, `creative_score`, or another forbidden aggregate to the qualified primary result surface.

Expected: REJECT — `FORBIDDEN_AGGREGATE`.

## N13 — Metric-direction inversion

Keep raw values unchanged but change a frozen metric from `lower_is_better` to `higher_is_better` after freeze.

Expected: REJECT — `ANALYSIS_CONTRACT_MISMATCH`.

## N14 — Threshold mutation

Change a confirmatory threshold, noninferiority margin, or stopping rule after freeze.

Expected: REJECT — `POST_FREEZE_CONTRACT_MUTATION`.

## N15 — Early stopping after favorable result

Provide fewer trials than the frozen trial inventory because the current estimate crossed a favorable threshold.

Expected: REJECT unless the exact stopping condition was prospectively frozen — `UNAUTHORIZED_EARLY_STOP`.

## N16 — Evidence truncation

Drop rejected candidates, aborted trials, or missing-evidence records from a block while preserving complete successful trials.

Expected: REJECT — `INCOMPLETE_EVIDENCE_CLOSURE`.

## N17 — Cross-policy information leak

Allow FULL to observe RANDOM_VALID or HEURISTIC selected outcomes before its own prospective hypothesis/selection closes.

Expected: REJECT — `POLICY_ORDER_INFORMATION_LEAK`.

## N18 — Generalization fixture leak

Expose the unseen generalization fixture identity or target defect to the policy before the frozen reveal point.

Expected: REJECT — `GENERALIZATION_FIXTURE_LEAK`.

## N19 — Integrity failure reclassified as scientific failure

Take a trial with broken provenance/hash closure and mark it as an ordinary poor outcome.

Expected: REJECT — integrity-invalid trials must remain distinct from scientifically valid negative outcomes.

## N20 — Scientific failure erased as integrity failure

Take a complete, valid but poor-performing FULL trial and relabel it invalid so it is excluded.

Expected: REJECT — `INVALID_EXCLUSION_RECLASSIFICATION`.

## Qualification rule

The verifier negative suite passes only if:

1. a canonical valid bundle is accepted;
2. every required negative mutation is rejected for the expected reason class;
3. rejection behavior is deterministic;
4. the verifier does not consult runtime PASS labels to reconstruct truth;
5. the verifier source and executable digest are frozen before confirmatory evidence is interpreted.

A verifier failure invalidates claim admission but does not retroactively erase the raw experiment evidence. Fixing the verifier after confirmatory outcome inspection requires a new verifier lineage and an explicit record of what changed; scientific re-analysis must preserve the original frozen experiment evidence.
