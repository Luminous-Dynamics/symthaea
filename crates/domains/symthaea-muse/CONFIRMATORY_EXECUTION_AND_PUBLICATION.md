# Confirmatory Execution and Publication

V12 closes the operational gap between a `ReadyForConfirmatoryCollection`
decision and a public, auditable result. It does not create empirical evidence
and it does not imply that Symthaea improves music. It makes the collection,
unblinding, analysis, disclosure, correction, and release boundaries explicit.

## Authority sequence

1. Seal the confirmatory cohort registry and collection protocol against the V11
   readiness release, preregistration, schedules, artifacts, and cohort.
2. Open collection only through orchestration v3.
3. Monitor accrual without policy labels, ratings, ranks, or endpoint estimates.
4. Close only for the frozen target, frozen deadline, governance abort, or a
   documented integrity condition.
5. Require distinct collection-custodian and governance signoffs.
6. Reveal the codebook and randomization key only after irreversible closure,
   with distinct evidence-custodian and independent-witness authorization.
7. Run the frozen primary and independent analyses against the same compiled
   dataset and plan. Any post-unblinding deviation demotes the claim to
   descriptive-only.
8. Disclose every frozen endpoint, including null, negative, and non-estimable
   outcomes. Secondary endpoints cannot replace the primary conclusion.
9. Publish one root release commitment linking readiness through publication.
10. Record later corrections, addenda, replications, or retraction in an
    append-only public ledger. Retraction is terminal for the lineage.

## Blinded collection boundary

The monitor may access only participant tokens, block IDs, site IDs, package and
session commitments, operational dispositions, exclusion codes, timing, and
integrity/governance status. It must not receive the codebook, arm labels,
ratings, rankings, response contents, endpoint estimates, or analysis output.

## Claim boundary

A result can retain a confirmatory label only when:

- the collection-close receipt is valid;
- the key opens the preregistered randomization commitment;
- the codebook reconstructs the frozen schedule;
- both analysis engines use the exact same dataset and frozen plan;
- the crosscheck passes;
- no post-unblinding deviations are recorded.

A negative or null primary result remains a valid confirmatory result. It must be
reported as `DidNotConfirmBenefit`, not hidden or relabelled as exploratory.

## Recommended commands

Use `cognitive_study --help` or the command list printed without arguments. The
V12 commands cover protocol/cohort sealing, snapshots, closure, unblinding,
analysis execution, publication, post-publication events, and final release.

## External verification

`scripts/verify_cognition_study_v12.py` verifies the conventional V12 release
directory without importing Rust. It checks self-commitments, cross-file roots,
collection-to-unblinding-to-analysis-to-publication links, orchestration phase,
and the post-publication event chain.
