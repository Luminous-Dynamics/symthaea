# Actuation Enforcement Qualification Campaign V1

Status: DESIGN / EVIDENCE CONTRACT ONLY — NOT QUALIFIED

Parent code subject before this documentation-only commit:

`b1dd08e54bc111f66f197434b72bd92e8add4366`

This document freezes the next hardening step for the closed-world actuation-enforcement evidence introduced by #1549. It does not alter runtime authority, qualify an adapter, or claim physical resource enforcement.

## Why this is required

The V1 structural evidence set already requires exactly one satisfied record for each of nine fixed enforcement obligations and binds every record to the same enforcement profile, authentication profile and backend identity.

That is necessary but not sufficient for a rigorous qualification campaign. Records with stable implementation identities could otherwise be assembled from different laboratory epochs or different physical/emulated environments. Structural completeness must therefore eventually distinguish:

`same implementation identities != same qualification campaign != same execution environment`.

## Additive V2 shape

Do not rewrite or reinterpret existing V1 record IDs after qualification has begun. Introduce an additive campaign-bound wrapper or V2 record schema only after #1549 and its parents compile and qualify.

The campaign must have a canonical, domain-separated identity binding at minimum:

- exact `StructurallyCompleteActuationEnforcementEvidenceId`;
- exact `ActuationEnforcementBoundaryProfileId`;
- exact `ActuationInterlockAuthenticationProfileId`;
- exact backend ID and backend implementation lineage;
- non-zero qualification campaign nonce/digest;
- qualification harness implementation digest;
- scenario-suite manifest digest;
- environment manifest digest;
- topology / dependency manifest digest;
- hardware / firmware capability manifest digest where physical hardware participates;
- toolchain / container / Nix realization digest used to execute the evidence campaign;
- campaign start and end timestamps;
- evidence-manifest digest covering the exact nine evidence record IDs in canonical obligation order.

All digests are descriptive until verifier-owned admission establishes their truth.

## Campaign coherence rules

A campaign-bound evidence object must fail closed unless:

1. all nine V1 obligation records are present in the underlying complete set;
2. every record belongs to the exact same enforcement/authentication/backend context;
3. every record observation time lies within the declared campaign interval;
4. campaign start and end are non-zero and ordered;
5. all environment/harness/topology/hardware/toolchain digests are non-zero;
6. the evidence manifest covers the exact canonical nine record IDs, neither fewer nor additional substituted records;
7. the wrapper binds the exact complete-set ID rather than reconstructing a new notion of completeness;
8. no campaign metadata is treated as evidence truth merely because it hashes correctly.

## Adversarial qualification cases

The eventual tests must reject at least:

- eight records from campaign A plus one record from campaign B;
- same backend ID but different harness digest;
- same backend ID but different firmware/hardware capability manifest;
- one observation timestamp outside the campaign interval;
- campaign interval rollback or zero timestamps;
- evidence-manifest digest omitting one obligation;
- evidence-manifest digest substituting a different record for the same obligation;
- topology/environment drift between stale-holder and crash-restart scenarios;
- a caller-provided environment digest that is never independently verified;
- a campaign-bound wrapper from an older verifier/profile being reused after verifier adoption changes.

## Relationship to #1550

Verifier-owned admission must qualify the exact campaign-bound evidence object, not merely the underlying V1 complete-set ID, once campaign binding is implemented.

The verifier must independently validate the claimed campaign/environment manifests and require fresh current-verifier evidence before the campaign is accepted for adapter admission.

## Non-claims

Even a verifier-qualified campaign does not prove the live target resource currently enforces fencing. It establishes evidence about a specific qualified backend/boundary implementation in a specific qualification world.

The physical #1528 theorem still requires backend/resource qualification showing that the same live boundary which mutates the resource rejects stale generations, replay, deny/emergency-stop cases, duplicate consumption and rollback of fence state.

## Implementation gate

Do not implement this additive campaign schema on top of source-only parents while GitHub CI remains queued. Preserve #1549's V1 IDs until formatter/compiler/clippy/tests/security workflows execute successfully or expose defects that require repair first.
