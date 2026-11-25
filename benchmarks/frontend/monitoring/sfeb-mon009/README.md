# SFEB MON-009 — sink-boundary effect qualification

**Status:** research protocol and reference transaction design; no PostgreSQL integration has been run from this branch.  
**Tracking:** [Issue #13638](https://github.com/Luminous-Dynamics/symthaea/issues/13638).  
**Parent:** [MON-008 PR #13620](https://github.com/Luminous-Dynamics/symthaea/pull/13620).

MON-009 begins the next engineering increment after the MON-007 semantic model and MON-008 authenticated checkpoint preflight. Its target is a real, reversible PostgreSQL sink that atomically enforces tenant/resource scope, stable operation identity, request digest, authority revision and fencing.

It then adds a destination-issued receipt, a separately authorized exact-revision readback, a declared observation boundary with independently anchored watermarks, and a portable auditor.

The key research distinction is:

- **record integrity:** who signed which bytes and whether they were transparently registered;
- **effect truth:** what the destination committed and what a separate principal observed;
- **bounded completeness:** which mutation paths and intervals are actually covered, and which bypasses remain outside the claim.

No universal completeness, global exactly-once execution, production security or SOTA superiority is claimed.

See [SINK-BOUNDARY-QUALIFICATION-v1.md](SINK-BOUNDARY-QUALIFICATION-v1.md) for the semantics, PostgreSQL reference design, observer independence matrix, coverage model, adversarial corpus and W0-W6 benchmark extension.


Physical-action composition is treated separately in [PHYSICAL-EFFECT-INTEROP-v1.md](PHYSICAL-EFFECT-INTEROP-v1.md). Digital dispatch, physical engagement and independently measured physical outcome are distinct claims; a valid physical engagement receipt is not a safety case.


The latest action/decision interoperability comparison is in [AGENT-DECISION-EFFECT-INTEROP-v1.md](AGENT-DECISION-EFFECT-INTEROP-v1.md). Agent Action Capsule -05 already defines a confirmed-effect response binding, but its own security model treats that as producer-observed bytes—not external-world state. MON-009 composes that profile with destination commit receipts and separate-principal exact-revision readback. Decision Provenance -03 supplies a separate approval/decision-record comparison. None of these active Internet-Drafts is a published RFC.
