# HUM-HRI-001C — Consent-Bound Human-Contact Intent

Status: source-design candidate
Issue: #4682
Authority: semantic contact intent only; **no motor authority**

## Purpose

Bind one intentional human-contact proposal to the exact semantic consent scope currently admitted by the HUM-HRI-001B lifecycle.

This tranche closes an important substitution gap: consent-epoch equality alone is insufficient. Two different scopes can share the same epoch, so downstream contact intent must prove that the exact admitted scope—not merely an equal epoch—authorizes the proposed region/site/contact class.

## Core theorem

For any proposed human-contact intent `I` and consent scope `S`:

`bind(I, S)` is admissible only when `S` is exactly the scope currently active in the runtime session and `S` explicitly permits the proposed human region, robot site, contact class, and current time.

A broader same-epoch scope is rejected.

## Runtime strengthening

`HumanContactAuthoritySessionV1` now retains a private snapshot of the exact admitted semantic scope while Active.

- admission stores the exact scope snapshot;
- revocation destroys the active snapshot before entering `WithdrawRequired`;
- `is_exact_scope_active()` requires full semantic equality plus current scope liveness;
- same-epoch but broader/substituted scope values fail closed.

The runtime still authenticates nothing and grants no motor authority.

## Bound intent

`ConsentBoundHumanContactIntentV1` records:

- participant ID;
- session ID;
- consent epoch;
- explicit contact class;
- one explicit human body region;
- one explicit robot contact site;
- one spatial-goal ID;
- a finite root-frame target.

The type intentionally does not contain production force, pressure, temperature, energy, speed, actuator, or controller values.

## Integration with existing WholeBodyIR work

Draft PR #1068 defines a separate `HumanContact` whole-body objective with force/speed demands and a `HumanContactConsentMaintained` invariant.

This tranche does not import or duplicate #1068. When that draft lineage is rebased onto the current human-contact authority stack, its `AssistHuman -> HumanContact` compilation path should consume a `ConsentBoundHumanContactIntentV1` (or a successor binding) rather than accepting only a spatially bound skill permit.

The intended join is therefore:

`semantic proposal -> spatial binding -> exact scoped-consent binding -> existing live authority -> WholeBodyIR -> qualified contact safety -> HAL`

and never:

`proposal -> WholeBodyIR -> motor`.

## Tests

Focused source regressions cover:

- exact active scope binds successfully;
- broader same-epoch scope substitution fails;
- region substitution fails;
- robot-site substitution fails;
- revoked session cannot bind;
- expired scope cannot remain eligible;
- malformed spatial target fails.

## Nonclaims

This tranche does not establish:

- authentic participant consent evidence;
- a cryptographic consent identity;
- adult eligibility/age assurance;
- contact-site hardware qualification;
- force/pressure/temperature safety;
- tactile sensor validity;
- actuator or motor authority;
- physical human-contact safety;
- human-trial authorization;
- legal/product-safety certification.

Exact-head format/compile/test/Clippy evidence is required before qualification is claimed.
