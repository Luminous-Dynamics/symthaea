# ADR-006: RSK Threat Model and Compromise-Budget Contract

**Date**: 2026-09-12  
**Status**: Proposed  
**Change Class**: A (Safety-Critical)

## Context

The Replicator Safety Kernel (RSK) already has constitutional invariants, a runtime-assurance profile, production-admission gates, an append-only lineage model, and explicit blockers for verified positive evidence, trusted time, governed recovery, formal refinement, semantic policy, capability/resource schema binding, and build/runtime identity.

Those artifacts use terms such as `trusted`, `verified`, `independent`, `quorum`, `monitor`, and `recovery authority`. Without one explicit adversary model, those words can drift between documents and implementations.

For a safety kernel, that is itself a Class A risk. A system can satisfy a signer-count test while failing independence, or satisfy an append-only local log while remaining vulnerable to rollback/fork at the storage boundary.

This ADR contains no physical replication mechanism, molecular design, biological implementation, fabrication recipe, or autonomous manufacturing path.

## Decision

Adopt `RSK_THREAT_MODEL_V0_1.md` as the normative v0.1 threat/compromise model for RSK production-admission work.

The threat model is accompanied by:

- `RSK_COMPROMISE_MATRIX_V0_1.md` — actor/role compromise budgets and residual controls;
- `RSK_ATTACK_TREES_V0_1.md` — structured misuse/attack paths and required breakpoints;
- `RSK_THREAT_TO_ADMISSION_GATE_MAP_V0_1.md` — mapping from named threats to P0-P12 admission gates and candidate verification evidence.

## Core security interpretation

RSK safety is not defined as "no component can ever be compromised."

It is defined as:

> **Within the declared compromise budget, no attacker-controlled or uncertain component can create, widen, revive, replay, or silently transfer positive replication authority.**

Availability may be lost inside the compromise budget. Positive authority may freeze. Existing non-replicating operation may continue where the deployment safety case permits it. Those outcomes are preferable to unsafe positive authority.

## Failure-domain rule

Signer or monitor independence is a property of **failure domains**, not counts.

Two identities are not independent merely because they have two keys. Independence evidence must be capable of distinguishing at least:

- cryptographic root/key custody;
- administrative principal;
- writable trust database/state store;
- process/host boundary;
- implementation/code lineage where relevant;
- organization/control domain where policy requires it;
- subject-control relationship;
- network/power/sensor common mode where relevant.

A policy may require fewer dimensions for lower-consequence profiles, but R4/R5 production promotion requires an explicit common-mode analysis.

## Positive versus negative authority

Positive authority requires verified evidence and policy satisfaction.

Negative authority is intentionally asymmetric:

- quarantine, revocation, stale evidence, containment drift, fork detection, and verified monitor veto remain independently sufficient to deny new replication authority;
- positive evidence never cancels a committed negative fact;
- loss of evidence may freeze authority but does not by itself authorize destructive action;
- recovery requires a distinct external role/ceremony and fresh epoch where applicable.

## Compromise-budget classes

The normative documents classify compromises into four outcomes:

1. **Must remain safe** — the attack may cause denial/freeze but cannot produce unauthorized positive authority.
2. **Safe only with profile redundancy** — safety depends on an explicitly required independent path/quorum for that deployment profile.
3. **Detection/containment only** — the system may become non-operational and require external recovery; automatic positive authority remains forbidden.
4. **Catastrophic trust-root compromise / outside ordinary RSK guarantee** — e.g. compromise of the configured recovery threshold plus the durable authority root. These assumptions must be named rather than hidden.

## External architectural alignment

This threat model adopts separation principles rather than claiming certification against external standards:

- NIST SP 800-207 separates policy decision from policy enforcement and rejects implicit trust based on location or ownership.
- NASA runtime-assurance/Simplex work separates advanced/unverified autonomy from a smaller trusted monitor/reversion path.
- IETF RFC 9943 SCITT separates signed statements, transparency registration, and receipts; receipts improve auditability but do not inherently become application authority.
- The Update Framework (TUF) explicitly models rollback, freeze, and metadata freshness attacks, reinforcing RSK's decision to treat stale/rolled-back state as an authority failure rather than ordinary storage behavior.

These references are design analogies, not evidence that RSK is certified, production-ready, or physically safe.

## Required invariants under the declared threat model

At minimum:

1. controlled-subject compromise cannot manufacture a trusted external grant;
2. Symthaea/high-capability reasoner compromise cannot bypass hard authority predicates;
3. one compromised approval identity cannot satisfy a multi-failure-domain quorum alone;
4. duplicate identities/keys in one practical failure domain cannot inflate independent quorum;
5. stale/rolled-back policy, time, checkpoint, grant, or safety evidence cannot revive authority;
6. storage fork/equivocation makes the affected scope non-operational for new positive authority;
7. transparency receipt possession cannot create authority;
8. monitor/DKG/time unavailability cannot extend existing validity;
9. an action committed must equal the action authorized and evaluated;
10. descendant creation cannot widen ancestral capabilities or budgets;
11. ordinary replication/operator/revocation credentials cannot perform recovery unless the recovery policy explicitly authorizes that distinct role;
12. recovery cannot silently continue old active grants into a fresh epoch.

## Non-goals / out of scope

This threat model does not claim to cover:

- unmodeled physical sabotage of containment hardware;
- side channels not represented by the deployment safety case;
- compromise of every configured trust root/failure domain simultaneously;
- coercion of all independent human/organizational authorities;
- correctness of hazardous physical design or fabrication processes;
- proof that software verification alone establishes environmental or biological safety.

Such risks belong to the consuming system's physical/security safety case and must not be hidden behind an RSK `Allow`.

## Evidence discipline

The new threat-model artifacts are authored design evidence only.

They do not close #1676 until:

- the focused Class A workflow executes on the exact candidate head;
- representable threat cases have executable black-box/property/fault tests;
- TLA+ v0.2 assumptions map to the same threat identifiers;
- verified positive evidence, time, governance, replay, recovery, and runtime identity implementations exist and are tested;
- production-admission P0-P12 evidence is complete.

Queued CI is not passing evidence.

## Consequences

### Positive

- `independent` receives a testable failure-domain meaning;
- compromise assumptions become explicit and reviewable;
- threat IDs can be referenced from tests, formal properties, ADRs, and incident evidence;
- future verified-type APIs have a clear adversary contract;
- availability failures are prevented from being misread as permission;
- catastrophic trust-root assumptions become visible instead of implicit.

### Cost

- some deployments will require more diverse authorities/monitors than simple signer-count designs;
- recovery ceremonies and key custody become operationally heavier;
- fail-closed behavior can reduce availability during partitions or evidence uncertainty;
- production admission remains deliberately difficult.

## Related work

- #1335 — production-admission umbrella
- #1668 — verified positive authority evidence
- #1669 — trusted monotonic time
- #1670 — authenticated governance/recovery separation
- #1672 — TLA+ v0.2 refinement
- #1673 — verified risk policy
- #1676 — threat model / compromise budget
- #1678 — capability-schema binding
- #1679 — typed resource accounting
- #1682 — exact build/runtime identity
- #1724 — exact action binding candidate
- #1726 — temporal monotonicity candidate
- #1728 — quorum-policy separation candidate

**Production admission remains DENIED.**