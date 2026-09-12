# ADR-008: RSK Governance Command and Recovery Separation

**Date**: 2026-09-12  
**Status**: Proposed  
**Change Class**: A (Safety-Critical)

## Context

The RSK reference ledger intentionally exposes direct in-process methods for lineage registration, quarantine, revocation, and related mutations. Those APIs model state-machine semantics, but they are not a production authorization boundary.

Production governance creates a different class of risk from ordinary replication authorization:

- a safety monitor should be able to add bounded negative facts without minting positive grants;
- a revoker should be able to revoke a bounded target without recovering or widening authority;
- a policy authority should be able to establish policy only inside its governed scope;
- an ordinary operator should not become a recovery root;
- clearing quarantine, resolving a fork, or creating a fresh epoch must require a distinct recovery ceremony;
- no old active grant should cross a recovery epoch implicitly.

This ADR contains no physical replication mechanism, molecular design, biological implementation, fabrication recipe, or autonomous manufacturing path.

## Decision

Adopt a signed, exact-target governance-command boundary and a structurally distinct recovery-authority boundary.

The production model separates at least:

```text
VerifiedPolicyAuthority
VerifiedSafetyMonitor
VerifiedQuarantineAuthority
VerifiedRevocationAuthority
VerifiedRecoveryQuorum
```

Exact type names are non-normative. Role separation is normative.

## Fundamental separation

1. **Replication authority != governance authority.**
2. **Governance authority != recovery authority.**
3. **Negative authority does not imply positive authority.**
4. **Recovery is not the inverse of quarantine/revocation.**
5. **Fork resolution is not a storage heuristic.**
6. **Fresh-epoch recovery does not inherit active grants.**

## Exact command binding

Every authenticated governance command must bind the exact action being authorized, including at minimum:

- command schema/version;
- command kind;
- command ID / mutation ID / nonce;
- target kind and exact target ID(s);
- target epoch;
- expected ledger sequence/head digest;
- issuer identity/key/profile;
- issuer role;
- policy/trust snapshot digest or epoch;
- issued-at/not-before/expiry/freshness as applicable;
- reason code / bounded incident reference where applicable;
- command-specific parameters;
- canonical command digest/signature.

A command valid for one kind/target/epoch/head cannot be replayed or reinterpreted as another.

## Role matrix

### Safety monitor

May, according to policy:

- veto/freeze new replication authority;
- propose or commit bounded quarantine/negative facts;
- report evidence/incidents.

Must not:

- mint positive grants;
- widen ceilings;
- clear quarantine/revocation;
- select a winning fork;
- establish a new recovery epoch.

### Quarantine authority

May add quarantine within explicit scope.

Must not implicitly:

- revoke unrelated scopes;
- mint grants;
- unquarantine;
- recover/fork-resolve;
- widen ceilings.

### Revocation authority

May revoke explicitly authorized grant/subject/lineage/policy/key scope.

Must not implicitly:

- clear revocation;
- mint replacement authority;
- select fork winner;
- establish new epoch.

### Policy authority

May establish/supersede policy inside exact scope according to verified policy-governance rules.

Must not:

- bypass constitutional floors;
- rewrite committed negative state;
- recover a fork merely by publishing a new ordinary policy;
- inherit recovery powers unless separately authorized.

### Recovery quorum

May perform only the explicitly governed recovery ceremony, with exact old-terminal-state binding and fresh-epoch semantics.

Recovery authority is a high-trust root and must be protected by a stronger policy than ordinary operations.

## Negative command asymmetry

Negative controls are intentionally easier to authorize than positive recovery in many profiles.

This does **not** mean arbitrary callers may deny service. Negative commands remain authenticated, scoped, freshness-bound, auditable, and policy-controlled.

The asymmetry is:

```text
add bounded negative fact -> ordinary negative-governance path may be sufficient
remove/override negative fact -> recovery path required
```

This prevents a compromised subject/operator from using an ordinary API to erase its own containment state.

## Quarantine versus revocation

Quarantine and revocation remain distinct:

- **quarantine**: affected scope is non-authorizing pending investigation/recovery;
- **revocation**: explicit monotonic invalidation according to scope/policy within the epoch.

Neither is automatically destructive.

Unavailability or uncertain evidence normally causes freeze, not automatic revocation.

## Recovery ceremony

A recovery ceremony binds at least:

- old epoch ID;
- old terminal ledger head/checkpoint digest;
- all known fork branch heads when recovering from equivocation;
- incident/recovery reason;
- recovery policy/version/digest;
- recovery quorum signer identities/failure domains;
- exact preserved subjects/lineages if any;
- newly declared immutable/hard ceilings;
- new trust/policy roots as applicable;
- new epoch ID/genesis commitment;
- explicit statement that no active grant crosses implicitly;
- recovery evidence digest and signatures.

A successful recovery produces a new epoch/genesis. It does not mutate the old history to make the incident disappear.

## Fork handling

For conflicting valid successors to the same predecessor:

1. mark the affected authority scope forked/non-operational;
2. retain every conflicting branch as incident evidence;
3. deny new positive replication authority;
4. prohibit last-write-wins, highest timestamp, autonomy choice, or receipt-majority winner unless a separately specified recovery policy explicitly and safely defines a governed procedure;
5. recover through the distinct recovery ceremony into a fresh epoch.

Transparency receipts can reveal equivocation but do not select authority.

## Grant carry-over prohibition

No active grant, authorization token, quorum capability, or verified monitor witness becomes valid in the new epoch merely because the subject survives recovery.

Positive authority in the new epoch must be explicitly re-established and verified against the new epoch's trust/policy/lineage state.

## Current-reference compatibility

The existing in-memory direct mutation APIs remain reference semantics. This ADR does not require changing them until the production adapter/verifier layer exists.

Production code must not expose those direct methods to an untrusted caller as a governance API without the verified command boundary.

## Threat coverage

Primary threats:

- T16 negative-state suppression;
- T20 fork/equivocation;
- T24 governance command substitution;
- T25 recovery-role confusion;
- T26 grant carry-over across recovery;
- T30 catastrophic recovery-root compromise.

## Evidence discipline

This ADR freezes the target semantics only.

Implementation promotion requires:

- canonical signed governance command vectors;
- wrong role/kind/target/epoch/head rejection;
- stale/replayed command rejection;
- bounded negative-scope tests;
- monitor cannot mint grants/widen ceilings;
- ordinary operator/revoker cannot recover;
- fork recovery preserves all incident branches;
- fresh epoch has no active grant carry-over;
- recovery threshold/failure-domain tests;
- exact-head Class A CI and retained evidence;
- current negative-state dominance after all integration.

## Consequences

### Positive

- removes implicit "god mode" from operator/revoker/monitor roles;
- makes governance actions exact-action bound and replay-resistant;
- recovery becomes explicit, auditable, and structurally harder than ordinary operations;
- forks become safety incidents rather than database conflict resolution;
- old positive authority cannot leak through recovery.

### Cost

- more operational roles/key custody;
- recovery is deliberately slower/heavier;
- fail-closed forks can reduce availability;
- production adapter must authenticate and verify every governance command.

## Related work

- #1335 production-admission umbrella
- #1668 verified positive evidence
- #1670 governance/recovery separation
- #1676 threat model
- #1762 threat-model candidate
- #1766 verified-evidence candidate

**Production admission remains DENIED.**