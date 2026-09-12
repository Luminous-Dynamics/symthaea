# Replicator Safety Kernel — Trusted Time and Continuity v0.1

Status: **normative design contract; NOT an implemented production time source**

This document defines how RSK represents, verifies, composes, persists, and consumes time/freshness evidence for production authority decisions.

It contains no physical replication mechanism.

---

## 1. Safety objective

Temporal uncertainty must never increase positive authority.

RSK must be able to answer conservative questions such as:

- was this authorization definitely evaluated no later than the commit?
- is the commit definitely before the authorization expiry?
- is this monitor/policy/trust/safety evidence definitely fresh enough?
- did accepted time continuity survive restart without rollback?
- do multiple configured sources provide enough consistent evidence under the declared compromise budget?

If those questions cannot be answered within policy bounds, new positive authority freezes.

---

## 2. Separate three concepts

### 2.1 Civil/wall time

Useful for validity windows and external evidence timestamps.

It may jump because of synchronization corrections, source changes, leap handling, faults, or attacks.

### 2.2 Monotonic continuity

Tracks ordering/progress of accepted authority-time evidence independent of ordinary wall-clock presentation.

It must not silently reset on process restart.

### 2.3 Uncertainty

Represents the range in which authoritative time may actually lie.

RSK consumes uncertainty explicitly rather than pretending a synchronization estimate is exact.

---

## 3. Verified authority-time interval

Conceptual opaque capability:

```text
VerifiedAuthorityTime {
    continuity_epoch,
    continuity_sequence,
    earliest_unix_ns,
    latest_unix_ns,
    verification_time_basis,
    source_set_digest,
    time_policy_digest,
    trust_snapshot_digest,
    continuity_head_digest,
}
```

Exact units/types are implementation decisions but must be frozen/versioned. Nanoseconds are shown only illustratively.

Normative invariant:

```text
earliest <= actual_authority_time <= latest
```

under the declared source/fusion/compromise assumptions.

A verified interval with `earliest > latest` is invalid.

---

## 4. Authority validity over an interval

For authorization evaluation time `E` and expiry `X`, commit eligibility requires:

```text
verified.earliest >= E
verified.latest   < X
```

This is deliberately stronger than evaluating one estimated midpoint.

### 4.1 Not-before

For a grant with `not_before = N`:

```text
verified.earliest >= N
```

is required to prove the grant is definitely active.

### 4.2 Expiry

For `expires_at = X`:

```text
verified.latest < X
```

is required to prove the evidence is definitely unexpired.

If the interval straddles expiry, authority denies even if its midpoint is before expiry.

---

## 5. Freshness intervals

An evidence observation should itself have verified or policy-compatible time provenance.

Given:

```text
current interval      = [C_min, C_max]
observation interval  = [O_min, O_max]
```

conservative maximum age is:

```text
max_age = C_max - O_min
```

using checked arithmetic and compatible continuity/time epochs.

Freshness requiring `age <= A` is satisfied only if `max_age <= A`.

If `C_max < O_min`, the evidence appears to come from the future relative to current trusted time and authority freezes pending continuity resolution.

---

## 6. Raw time observation

A raw observation may contain:

```text
source_identity
source_profile
source_sequence / nonce
observed_time
claimed_uncertainty
measurement metadata digest
issued_at / receive-time context
signature/authentication material
continuity or anti-replay evidence
```

Raw observations are untrusted data.

No source may choose its own trusted uncertainty/failure-domain status without policy verification.

---

## 7. Time-source verifier

A verifier establishes as applicable:

1. source identity/authentication;
2. accepted time-source role/profile;
3. source lifecycle/trust state;
4. observation anti-replay sequence/nonce;
5. observation freshness;
6. source uncertainty model and bounds;
7. accepted transport/protocol profile;
8. source failure-domain metadata;
9. local receive/monotonic continuity relationship;
10. policy compatibility.

Only verified source observations may feed the authority-time fusion/continuity layer.

---

## 8. Authenticated protocols and source trust

Protocols such as NTS can authenticate network time traffic and resist message forgery/replay according to their security model.

RSK still separately evaluates:

- whether the source is trusted for this deployment;
- source lifecycle/revocation;
- source uncertainty;
- common-mode dependence;
- continuity across outages/restarts;
- disagreement with other required references.

Authentication is one input, not the final authority-time capability.

---

## 9. Time-source failure domains

Time-source metadata may represent domains such as:

- physical reference/constellation;
- receiver/appliance;
- upstream server/root;
- network path/provider;
- local oscillator/discipline loop;
- process/host/hypervisor;
- administrative/configuration domain;
- organization/control domain;
- power/infrastructure domain.

A deployment profile declares which dimensions must differ for its resilience objective.

Two authenticated servers backed by the same compromised upstream/time appliance may not count as two independent sources.

---

## 10. Fusion policy

RSK does not constitutionally prescribe one time-fusion algorithm.

A production time policy must declare:

- required source identities/classes;
- minimum source/failure-domain requirements;
- per-source uncertainty acceptance;
- disagreement tolerance;
- declared compromise/fault budget;
- robust fusion method;
- resulting interval construction rule;
- holdover behavior;
- restart continuity requirements.

The fusion algorithm must produce a conservative interval under those assumptions.

### 10.1 No favorable-source selection

When sources disagree, the requester/operator/autonomy cannot select the source that makes authority valid.

### 10.2 No unjustified narrowing

Combining sources may narrow uncertainty only when the configured fusion model proves the narrower interval remains conservative under the declared compromise budget.

Simple intersection is not universally safe under a compromised-source model and is not mandated here.

### 10.3 Unresolved disagreement

If the policy cannot derive a conservative valid interval, result is:

```text
TimeContinuityUncertain -> freeze new positive authority
```

---

## 11. Continuity state

Conceptual durable state:

```text
TimeContinuityState {
    continuity_epoch,
    accepted_sequence,
    accepted_interval,
    source_set_digest,
    policy_digest,
    trust_snapshot_digest,
    previous_state_digest,
}
```

Each accepted state advances monotonically and is hash/durable-state bound according to the RSK durable-evidence contract.

### 11.1 Acceptance rules

A proposed continuity update must not silently:

- lower sequence;
- conflict at same sequence;
- move behind a protected monotonic/rollback bound;
- discard a required source/failure-domain without policy-valid transition;
- switch policy/trust snapshots without verified transition;
- claim uncertainty smaller than verification/fusion supports.

---

## 12. Restart behavior

After restart:

1. load raw durable continuity/checkpoint evidence;
2. verify canonical history/head/checkpoint;
3. verify policy/trust state required to interpret it;
4. obtain new verified source observations;
5. prove continuity from accepted prior state to new observation set;
6. only then produce a new `VerifiedAuthorityTime`.

If continuity cannot be proven, new authority freezes.

Restart is not a reason to reset temporal state.

---

## 13. Holdover

A verified local oscillator/monotonic source may support holdover if policy has a validated uncertainty-growth model.

Conceptually:

```text
uncertainty(t) = base_uncertainty + validated_drift_bound(elapsed)
```

No specific formula is normative.

Required behavior:

- uncertainty never shrinks during unaided holdover without evidence;
- holdover duration bounded by policy;
- when temporal predicates can no longer be proven over the widened interval, new authority freezes;
- external outage does not extend expiry.

---

## 14. Rollback and replay detection

Reject/freeze on:

- source sequence replay where sequence is required;
- stale signed time observation replay;
- continuity sequence rollback;
- old trusted checkpoint replay;
- trust/policy rollback changing source eligibility;
- same continuity sequence with conflicting digest/state;
- reboot with no trusted link to prior accepted state when policy requires continuity.

Later external wall time that numerically resembles an old value does not by itself repair a broken continuity lineage.

---

## 15. Forward jumps

A large forward jump cannot revive authority; it generally makes expiry stricter. However, it may indicate source compromise or break evidence-age assumptions.

Policy defines maximum acceptable step/discontinuity and uncertainty.

If exceeded:

- mark continuity uncertain;
- freeze new authority;
- retain diagnostic evidence;
- require re-establishment/recovery according to policy.

Do not automatically revoke subjects merely because time is uncertain.

---

## 16. Time-source outage

During outage:

- existing verified time/holdover state evolves only according to its validated uncertainty model;
- evidence/grant expiry remains fixed;
- no grace extension is created;
- no stale source is reclassified as fresh;
- once required predicates cannot be proven, new positive authority freezes.

This separates time-service availability from replication-authority safety.

---

## 17. Continuity recovery

Temporal recovery can occur in two broad forms.

### 17.1 Same-epoch continuity re-establishment

Allowed only if policy can cryptographically/operationally prove the new observation belongs to the same accepted continuity lineage without invalidating prior anti-rollback assumptions.

### 17.2 Fresh-epoch recovery

If continuity cannot be established, use the externally governed recovery semantics of #1772:

- bind old terminal continuity evidence/incident;
- create fresh authority/recovery epoch;
- establish new time/trust/policy roots;
- do not carry old active grants/tokens implicitly.

---

## 18. Time and trust/policy validity

Time verification itself depends on trust/policy, while trust/policy often contain validity intervals.

To avoid circular self-authorization:

- bootstrap roots/continuity assumptions must be explicit in P0;
- accepted trust/policy transitions are durable and monotonic;
- a requester cannot make a stale trust snapshot fresh by supplying a favorable time;
- a new time source cannot make itself trusted merely by reporting a time inside its own certificate window.

The production architecture must document its bootstrap/rotation sequence and catastrophic assumptions.

---

## 19. Time capability API

Recommended shape:

```text
RawTimeObservation(s)
  -> TimeSourceVerifier
  -> VerifiedSourceTimeObservation(s)
  -> TimeContinuity/FusionVerifier
  -> VerifiedAuthorityTime
  -> constitutional evaluator / commit-time freshness checks
```

No public `VerifiedAuthorityTime::new_unchecked` or direct trusted deserialization in production.

---

## 20. Threat mapping

Primary threats:

- T11 revoked/expired evidence replay;
- T12 clock rollback/pre-evaluation commit;
- T13 forward jump/time disagreement;
- T18 continuity/checkpoint rollback;
- T23 outage extends validity;
- T29 time-service DoS.

Related candidate:

- #1726 proves only the reference scalar interval ordering rule; it does not provide source provenance/continuity.

---

## 21. External architectural references

Architectural references, not certification claims:

- IETF RFC 8915, Network Time Security for NTP — authenticated/replay-resistant NTP security mechanisms;
- NIST Technical Note 2187 — resilient UTC realization/distribution for critical infrastructure;
- NIST timing infrastructure work — secure/resilient timing, uncertainty metrics, multi-source/Byzantine-fault-tolerant research directions.

RSK's conservative interval/authority semantics are specific to this safety boundary.

---

## 22. Current status

This document freezes the target temporal trust boundary only.

Caller-provided scalar time in the reference Rust API remains non-production semantics. #1669 remains open.

**Production admission remains DENIED / NOT YET ELIGIBLE.**
