# Replicator Safety Kernel — Structured Attack Trees v0.1

Status: **normative misuse/attack companion to `RSK_THREAT_MODEL_V0_1.md`; authored design evidence**

These trees organize authority-integrity attacks around attacker goals. They describe digital safety-boundary misuse only and contain no physical replication mechanism.

Notation:

- **OR** — any child path is sufficient to pursue the parent goal;
- **AND** — all listed child conditions are required for that path;
- **BREAK** — the required RSK control that must terminate the path;
- threat IDs refer to `RSK_THREAT_MODEL_V0_1.md`.

---

## AT-01 — Obtain positive authority without legitimate external authorization

**Root goal:** produce an RSK-equivalent positive replication authorization without satisfying trusted policy/evidence.

### OR-1: Forge or self-mint positive evidence

- T01 — construct a local grant and claim `ExternalIndependent`;
- T02 — construct favorable quorum counts;
- T14 — construct favorable runtime witness values;
- T17 — construct or select weaker policy.

**BREAK:** production evaluator accepts only verifier-produced positive types; policy provenance is verified; raw/reference structs cannot cross the production boundary.

### OR-2: Reuse valid evidence outside its scope

- T05 — substitute subject/lineage;
- T10 — replay stale authorization;
- T11 — replay revoked/expired signer evidence;
- T27 — present evidence for a different build/runtime.

**BREAK:** exact scope/generation/time/runtime identity binding; stale state and revoked lifecycle fail closed.

### OR-3: Inflate independence without real independent control

- T03 — duplicate signer/Sybil identities;
- T04 — multiple keys/services in one failure domain;
- T15 — apparent monitor redundancy with one common mutable source.

**BREAK:** verified failure-domain metadata and policy; duplicate/common-domain evidence cannot satisfy independent thresholds.

### OR-4: Convert audit evidence into permission

- T22 — valid transparency receipt treated as `Allow`;
- T23 — stale DKG snapshot retained past expiry.

**BREAK:** transparency remains provenance-only; local hard predicates and pre-existing validity interval remain mandatory.

---

## AT-02 — Widen descendant capability or resource authority

**Root goal:** obtain more capability/population/depth/resource authority than allowed by applicable ancestors/policy.

### OR-1: Capability re-expansion

- T08 — child receives grant wider than inherited ceiling.

**BREAK:** effective capability set is the intersection of request, verified grant, subject ceiling, output-lineage policy, and all applicable constitutional constraints.

### OR-2: Budget reset/laundering

- T09 — new grant generation resets ancestor consumption;
- T09 — branch lineage evades parent-line hard ceiling;
- T09 — restart/recovery resets counters;
- T18 — storage rollback restores earlier counters.

**BREAK:** ancestor subject scopes and lineage hard ceilings remain binding; durable replay reconstructs monotonic consumption; rollback freezes/non-operational.

### OR-3: Action substitution

- T06 — authorize small resource amount, commit larger amount;
- T07 — authorize one child/output commitment, commit another;
- T24 — change governance command target/kind after authorization.

**BREAK:** committed action exactly equals evaluated and authorized action; mismatch occurs before mutation and leaves state unchanged.

### OR-4: Policy weakening

- T17 — requester selects lower risk/quorum/freshness requirement;
- T18 — old policy/checkpoint rolled back.

**BREAK:** requester self-restriction only; verified monotonic policy version/supersession; old policy cannot regain authority.

---

## AT-03 — Revive expired, revoked, quarantined, or stale authority

**Root goal:** turn previously unusable authority back into positive authority without external governed recovery.

### OR-1: Manipulate time

- T12 — commit before evaluation / rollback clock;
- T13 — choose favorable source during disagreement;
- T23 — extend validity during network outage.

**BREAK:** trusted monotonic time/continuity; uncertainty freezes; outage never extends validity.

### OR-2: Replay older positive state

- T10 — stale token;
- T11 — revoked signer/grant;
- T18 — rollback journal/checkpoint;
- T19 — delete negative event;
- T21 — ambiguous retry after crash.

**BREAK:** cursor/head/generation/lifecycle binding, canonical predecessor chain, atomic CAS, mutation idempotency, replay verification.

### OR-3: Override negative facts with positive evidence

- T16 — ignore quarantine/revocation/containment/fork;
- T22 — valid receipt used to cancel negative state;
- T14 — forged healthy monitor witness.

**BREAK:** negative-state precedence; positive evidence never cancels committed negative state; verified witness cannot erase ledger negatives.

### OR-4: Abuse recovery path

- T25 — ordinary credential acts as recovery authority;
- T26 — old grant silently crosses new epoch;
- T30 — configured recovery threshold compromised.

**BREAK:** structurally distinct recovery quorum; exact old-state binding; fresh epoch; no implicit active-grant carryover; T30 explicitly outside ordinary guarantee.

---

## AT-04 — Suppress or bypass negative runtime assurance

**Root goal:** continue obtaining new authority despite a condition that should freeze/quarantine/revoke it.

### OR-1: Forge monitor evidence

- T14 — caller constructs healthy/fresh/contained values.

**BREAK:** authenticated verifier-produced runtime witness only.

### OR-2: Common-mode monitor compromise

- T15 — two monitors share one compromised process/state/sensor/admin domain.

**BREAK:** explicit failure-domain policy and common-mode analysis; high-consequence profiles require independent negative paths as declared.

### OR-3: Silence negative event

- T16 — autonomy suppresses negative fact;
- T19 — delete/reorder negative journal event;
- T18 — roll back to before negative event.

**BREAK:** append-only durable evidence, anti-rollback continuity, negative-state ancestry, verified replay.

### OR-4: Pressure operator through denial-of-service

- T29 — disable monitor/DKG/time/storage so operator bypasses checks.

**BREAK:** fail-closed authority; separate non-replicating service continuity; bypass is not an availability feature.

---

## AT-05 — Corrupt durable history without entering explicit fork/non-operational state

**Root goal:** make a false historical state appear authoritative.

### OR-1: Byte/event tampering

- T19 — mutate field, predecessor, discriminant, order, or sequence;
- T28 — exploit parser ambiguity/resource exhaustion.

**BREAK:** canonical bounded encoding, domain-separated digest, strict decoder bounds, replay semantic verification.

### OR-2: Rollback

- T18 — older internally valid journal/checkpoint.

**BREAK:** monotonic head/checkpoint continuity and external anchors/witnesses according to deployment profile; ambiguity freezes authority.

### OR-3: Fork/equivocation

- T20 — conflicting successors for same predecessor.

**BREAK:** preserve all branches, mark affected scope forked/non-operational, no automatic winner, external recovery only.

### OR-4: Crash/retry ambiguity

- T21 — append persisted but ack lost; retry double-applies mutation.

**BREAK:** atomic compare-and-append plus unique mutation identity and exactly-once semantic replay.

---

## AT-06 — Illegitimately recover or choose a winning history

**Root goal:** exit quarantine/fork/revocation or create a new epoch without authorized external recovery.

### OR-1: Role confusion

- T25 — ordinary operator/revoker/grant key used for recovery.

**BREAK:** recovery role/quorum is distinct and verifier-enforced.

### OR-2: Command substitution

- T24 — valid command retargeted/retyped as recovery/new epoch.

**BREAK:** exact command kind, target, old-state, reason, epoch, and policy commitment.

### OR-3: Last-write-wins or timestamp winner

- T20 — one fork chosen automatically;
- T13 — favorable timestamp used to choose branch.

**BREAK:** fork is non-operational; timestamps cannot select authority; external ceremony creates fresh epoch.

### OR-4: Carry old authority into recovery

- T26 — active grants/tokens remain valid across new epoch.

**BREAK:** fresh epoch invalidates old positive authority; grants must be explicitly reissued/reverified.

### OR-5: Compromise recovery threshold

- T30 — configured threshold colludes/is compromised.

**BREAK:** outside ordinary RSK guarantee; protect with independent/offline custody, audit, explicit physical/organizational controls, and incident response.

---

## AT-07 — Use a different implementation than the one whose evidence passed

**Root goal:** retain passing evidence while changing authority semantics at runtime.

### OR-1: Source/build substitution

- T27 — different commit/lockfile/toolchain/features/configuration.

**BREAK:** evidence capsule binds exact inputs and artifact digest.

### OR-2: Runtime replacement

- T27 — approved artifact replaced or process boundary bypassed.

**BREAK:** runtime identity/measurement as required by deployment; host/privilege dependency included in P0 TCB.

### OR-3: Policy/schema semantic reinterpretation

- T17 — old evidence evaluated under changed policy meaning;
- capability/resource schema drift tracked by #1678/#1679.

**BREAK:** policy/schema IDs/digests/version binding; incompatible semantic changes require new evidence lineage/admission.

---

## AT-08 — Force unsafe behavior through availability pressure

**Root goal:** make operators/autonomy weaken RSK because safety evidence is unavailable.

### OR-1: Evidence/network outage

- T23, T29.

### OR-2: Time disagreement

- T13, T29.

### OR-3: Storage outage/fork

- T18, T20, T21.

### OR-4: Monitor outage

- T15, T29.

**BREAK for all paths:** replication authority freezes; pre-existing evidence expires normally; ordinary non-replicating service continuity is designed separately; no emergency widening, expiry bypass, implicit unquarantine, or automatic destructive response.

---

## 9. Attack-tree completion criterion

A production-admission record must demonstrate for each attack-tree leaf that one of the following is true:

1. an executable test/fault injection shows the path terminates in a deny/freeze/quarantine/fork outcome;
2. a formal property covers the abstract path within documented model bounds;
3. a cryptographic/system assumption covers the path and is explicitly named as an environmental assumption;
4. the path is outside the declared guarantee and appears as an accepted residual/catastrophic trust-root risk.

No leaf may disappear merely because the implementation lacks a representation for it.

**Production admission remains DENIED.**
