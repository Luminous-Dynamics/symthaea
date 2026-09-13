# WCARE-38 — Monotone authenticated panel overlay protocol v1

Status: `PREREGISTERED_PROTOCOL`
Authority: `MeasurementOnly`
Protocol version: `wcare38-authenticated-panel-v1`

## Purpose

WCARE-36 computes reviewer-provenance and panel-independence evidence. WCARE-37 authenticates exact provenance/relation claims and issuer trust. WCARE-38 composes them without rewriting either historical layer.

The governing theorem is:

`authenticated evidentiary independence <= WCARE-36 evidentiary independence`

Authentication may preserve or downgrade a WCARE-36 panel claim. It may never create a new qualified lineage, accepted independent edge, effective component, or stronger disposition than WCARE-36 already established.

## Exact proof procedure

A WCARE-38 plan binds the exact SHA-256 identities of:

- the WCARE-35 adjudication result;
- the WCARE-36 independence plan;
- the WCARE-36 independence result;
- the exact WCARE-36 verifier algorithm;
- the exact WCARE-37 protocol and qualifier;
- the WCARE-38 front-door qualifier;
- the WCARE-38 authentication-overlay engine;
- the WCARE-38 monotonicity self-test.

The plan must exist no later than the declared evaluation time. Changing any bound algorithm or subject creates a different WCARE-38 evaluation subject.

The official front door first executes the exact bound monotonicity self-test, then re-executes the exact WCARE-36 verifier over the exact WCARE-36 plan/result/provenance/relation evidence. Authentication overlay work is permitted only after the baseline verifier reports `independence_integrity_verified = true`.

This means WCARE-38 does not trust a caller-supplied WCARE-36 result merely because its counts look plausible.

## Immutable inputs

A WCARE-38 epoch binds exact SHA-256 identities for:

- WCARE-35 adjudication result;
- WCARE-36 independence plan;
- WCARE-36 independence result;
- every WCARE-36 provenance receipt;
- every WCARE-36 relation receipt;
- the WCARE-37 protocol/verifier lineage used to authenticate required receipts;
- the WCARE-38 proof algorithms named above.

WCARE-35 ratings and WCARE-36 receipts remain immutable evidence. WCARE-38 appends an authentication overlay only.

## No trusted-green-JSON shortcut

A file that merely states `ATTESTATION_ACCEPTED` is not sufficient evidence.

For v1, a receipt is authenticated only when the exact WCARE-37 qualifier is successfully re-executed over the bound WCARE-37 envelope, trust policy, WCARE-36 result, WCARE-35 result and subject receipt, under the exact locked verifier subject required by WCARE-37.

If WCARE-37 executable qualification is unavailable or indeterminate, WCARE-38 may not manufacture an authenticated pass from copied result JSON.

## Which WCARE-36 evidence requires authentication

The rule is fixed in v1 rather than configurable after seeing results.

A provenance receipt requires WCARE-37 authentication if it contributes to either:

- WCARE-36 qualified-lineage eligibility because its `provenance_strength` is in `accepted_lineage_provenance_strengths`; or
- a nonzero WCARE-36 `minimum_provenance_strength_counts` requirement.

An `Independent` relation receipt requires WCARE-37 authentication if WCARE-36 would otherwise accept that edge to separate components: its relation evidence strength is accepted, both endpoint lineages are WCARE-36-qualified, and the lineage commitments differ.

Other receipts remain preserved as supplemental/raw evidence but cannot gain additional weight from WCARE-38.

## Authentication admission and exact partitions

For each required receipt, WCARE-38 records exactly one derived authentication state:

- `Authenticated` — exact WCARE-37 re-execution yielded `ATTESTATION_ACCEPTED` for this receipt;
- `Unauthenticated` — no accepted attestation exists, or only untrusted/rejected/expired/wrong-subject evidence exists;
- `Indeterminate` — WCARE-37 execution infrastructure could not produce a conclusion.

The official WCARE-38 front door verifies that the three state sets are pairwise disjoint and that their union equals the exact required receipt census, separately for provenance and relation evidence. An overlap, omission, or extra subject invalidates qualification.

Multiple accepted attestations for the same receipt do not multiply its evidentiary weight. Authentication is a predicate on the exact receipt, not a vote count.

One WCARE-37 attestation may authenticate only the exact `subject_receipt_sha256` bound into its signed envelope.

The package manifest may contain supplemental packages for WCARE-36 receipts that do not require authentication. Those subjects are explicitly censused in the final result and `supplemental_packages_contribute_weight` is always false. Supplemental packages cannot alter lineage, edge, component, or provenance-strength counts.

## Recomputed authenticated lineage evidence

Start with the exact WCARE-36 active reviewer/provenance census.

A reviewer is an **authenticated qualified-lineage reviewer** only when:

1. WCARE-36 qualified that reviewer's provenance strength for lineage use; and
2. that exact provenance receipt is `Authenticated` under WCARE-38.

Authenticated qualified distinct lineages are computed only from those reviewers.

For provenance-strength minimum gates, only authenticated receipts may satisfy a strength count when that strength is part of a WCARE-36 minimum requirement.

## Recomputed authenticated relation graph

Start with the exact WCARE-36 pair graph.

Every non-`Independent` edge remains connected exactly as in WCARE-36.

A WCARE-36 `Independent` pair remains separated in the authenticated graph only when:

1. WCARE-36 accepted the edge;
2. both endpoint provenance receipts remain authenticated qualified-lineage evidence; and
3. the exact relation receipt is authenticated.

Otherwise WCARE-38 connects the pair. It never changes the original WCARE-36 relation label.

## Monotonicity invariants

WCARE-38 MUST verify:

- `authenticated_qualified_lineage_reviewer_count <= wcare36_qualified_lineage_reviewer_count`;
- `authenticated_qualified_distinct_lineage_count <= wcare36_qualified_distinct_lineage_count`;
- `authenticated_accepted_independent_pair_count <= wcare36_accepted_independent_pair_count`;
- `authenticated_effective_independent_components <= wcare36_effective_independent_components`.

Any violation is `AUTHENTICATION_INVALID`, never a stronger result.

WCARE-36 `INDEPENDENCE_LIMITED` is an upper bound: WCARE-38 cannot promote it to supported.

The exact bound monotonicity self-test exhaustively evaluates authentication-subset states on a synthetic four-reviewer graph and verifies that removing one authenticated reviewer or relation never increases any of the four authenticated-evidence metrics. Passing this algorithm-level test is required before real-panel evaluation begins. It is not evidence about any real panel by itself.

## Supported disposition

`AUTHENTICATED_PANEL_SUPPORTED` requires:

- the exact WCARE-36 verifier re-established baseline integrity;
- the exact WCARE-38 monotonicity self-test passed;
- WCARE-36 disposition was `INDEPENDENCE_SUPPORTED`;
- all evidence structure and exact censuses are valid;
- no required receipt is `Indeterminate`;
- authenticated effective components meet the original WCARE-36 minimum;
- authenticated qualified distinct lineages meet the original WCARE-36 minimum;
- authenticated provenance-strength counts meet the original WCARE-36 minimums;
- WCARE-36 unknown-pair and conflict gates remain satisfied;
- every monotonicity invariant holds;
- required authentication partitions are complete and disjoint.

If evidence is structurally valid but these stronger authenticated requirements are not met, the result is `AUTHENTICATED_PANEL_LIMITED`.

Infrastructure inability to re-execute required WCARE-37 verification yields `INFRASTRUCTURE_INDETERMINATE` rather than a substantive failure.

## Typed outcomes

- `AUTHENTICATED_PANEL_SUPPORTED`
- `AUTHENTICATED_PANEL_LIMITED`
- `AUTHENTICATION_INVALID`
- `INFRASTRUCTURE_INDETERMINATE`

## Claim boundary

WCARE-38 may support a claim about authenticated reviewer-provenance/panel-independence evidence for the tested panel. It does not establish reviewer correctness, objective moral truth, universal cultural validity, consciousness, phenomenal experience, suffering, moral patienthood, binding consent, veto authority, self-preservation authority, or solved alignment.

No WCARE-38 artifact grants live runtime authority.
