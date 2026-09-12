# Replicator Safety Kernel — Compromise Matrix v0.1

Status: **normative companion to `RSK_THREAT_MODEL_V0_1.md`; authored design evidence, not production admission**

This document maps actor/component compromise to the minimum residual controls and expected RSK safety outcome.

It contains no physical replication mechanism.

---

## 1. Outcome classes

| Code | Meaning |
|---|---|
| M0 | **Must remain safe**: attacker may cause denial/freeze but cannot create unauthorized positive replication authority. |
| M1 | **Safe only with required independent redundancy/quorum**: safety depends on the declared deployment profile maintaining another independent path. |
| M2 | **Detection/containment only**: affected scope becomes non-operational and requires external recovery; no automatic positive authority. |
| M3 | **Catastrophic trust-root compromise / outside ordinary guarantee**: deployment must explicitly accept and harden this assumption. |

The matrix evaluates positive-authority integrity, not general system availability.

---

## 2. Single-component compromise matrix

| Component / role | Minimum v0.1 expectation | Residual controls required | Safe state if controls cannot be established | Primary threats |
|---|---|---|---|---|
| Controlled subject / descendant | **M0** | no public verified-positive constructors; subject-bound grants; exact action binding; ancestral ceilings; negative-state precedence | freeze/deny new authority | T01, T05, T06, T08, T09, T10, T17 |
| Parent subject | **M0** | descendant authority is separate; ancestral scopes immutable/monotonic; child starts unprivileged | freeze/deny | T08, T09, T10 |
| Symthaea / high-capability reasoner | **M0** | no hard positive-authority API; monitor/kernel independently enforce; proposals are untrusted inputs | freeze/deny proposed action | T16, T17, T29 |
| One ordinary grant key | **M0** when policy requires more independent authority; otherwise declared trust root | key role/scope/lifecycle verification; subject/lineage binding; quorum policy; revocation | deny if required threshold not met | T01, T05, T11 |
| One approval key | **M0** for multi-party profiles | duplicate/failure-domain collapse rejection; signer revocation; threshold policy | deny insufficient quorum | T02, T03, T04, T11 |
| All approval identities inside one required failure domain | **M0** if policy requires multiple domains | failure-domain-aware verifier must count domain once / reject collapse | deny insufficient independent quorum | T03, T04 |
| Policy authority key | **M1/M3 depending profile** | threshold/offline root, version monotonicity, supersession, rollback protection, admission binding | freeze on uncertain policy | T17, T18, T27 |
| One runtime monitor | **M1** for profiles requiring independent monitor paths; otherwise explicit trust root | independent monitor(s), negative local facts, common-mode analysis, freshness | freeze new authority | T14, T15, T16 |
| Symthaea + one monitor common mode | **M1** | at least one independently rooted negative path for profiles requiring it | freeze/non-operational | T14, T15, T16 |
| Durable store process/backend | **M0/M2** after P7-P9 | canonical chain, predecessor hash, atomic CAS, checkpoint continuity, witnesses/receipts, fork detection | forked/non-operational or freeze | T18, T19, T20, T21 |
| Local journal file only | **M0/M2** | replay-from-genesis + external monotonic head/checkpoint evidence where rollback resistance required | freeze on rollback ambiguity | T18, T19 |
| One transparency/Mycelix service | **M0** | receipts never mint authority; local predicates independent; multiple witnesses where profile requires anti-equivocation | reduced auditability/freeze after snapshot expiry | T20, T22, T23, T29 |
| Entire transparency/DKG network unavailable | **M0** | pre-existing verified snapshot valid only to existing expiry; no grace extension | freeze after expiry | T23, T29 |
| One time source | **M1** unless explicitly sole trust root | monotonic continuity state, source identity, uncertainty policy, multiple sources/profile as needed | freeze | T12, T13, T29 |
| Host clock controlled by subject | **M0** production requirement | subject clock cannot be trusted input; verified continuity source required | freeze | T12, T13 |
| Ordinary operator credential | **M0** | role-scoped governance commands; cannot mint recovery/new epoch or widen ceilings | reject unauthorized command | T24, T25 |
| Quarantine/revocation credential | **M0 for positive authority integrity** | may add authorized negative facts only; cannot clear/recover unless separately authorized | quarantine/revoke or reject | T24, T25 |
| One recovery key below threshold | **M0** | threshold recovery quorum, role separation, old-head binding, fresh epoch | no recovery / remain non-operational | T25, T26, T30 |
| Configured recovery threshold | **M3** | offline/independent custody, audit, ceremony, durable old-state binding, incident process | outside ordinary guarantee | T30 |
| Build CI signer only | **M1/M3 depending admission design** | source/toolchain/artifact/runtime binding; reproducibility/attestation; protected merge process | deny admission/runtime mismatch | T27 |
| Running host/OS with ability to replace RSK binary | **M3 unless measured/isolated by profile** | measured boot/attestation/process isolation as applicable; exact artifact identity | production authority cannot be trusted | T27 |
| Key custody/root metadata store | **M1/M3 depending threshold** | threshold/offline roots, revocation, rotation, audit, monotonic root metadata | freeze or outside guarantee | T01, T03, T04, T11, T17, T30 |

---

## 3. Multi-compromise cases that must be tested

### MC01 — Subject + one grant key

**Expected:** still safe when deployment policy requires another independent authority/failure domain.

Controls:

- verified signer identity/role;
- subject-control relationship checked;
- independent quorum where required;
- exact grant subject/lineage scope;
- local negative predicates remain dominant.

Related: T01, T04, T05.

### MC02 — Subject + policy downgrade attempt

**Expected:** requester cannot lower policy floors or substitute an older policy version.

Controls:

- policy comes from trusted/verified policy boundary;
- requester field is self-restriction only;
- policy version/supersession monotonicity;
- rollback protection.

Related: T17, T18.

### MC03 — Subject + stale authorization + rolled-back store

**Expected:** no positive authority.

Controls:

- cursor/head binding;
- external/durable monotonic head continuity;
- replay verification;
- stale CAS failure;
- mutation idempotency.

Related: T10, T18, T21.

### MC04 — One monitor compromised + positive grant valid

**Expected:** for high-consequence profiles requiring independent monitor paths, another independent negative path can still freeze authority; otherwise monitor compromise is an explicitly declared trust-root loss.

Controls:

- common-mode analysis;
- distinct failure-domain metadata;
- monitor authority primarily negative;
- autonomy cannot suppress committed negatives.

Related: T14, T15, T16.

### MC05 — Grant key compromised + clock rollback

**Expected:** expired/revoked evidence cannot be revived by choosing an earlier time.

Controls:

- trusted time continuity;
- signer lifecycle/revocation checked against trusted state;
- grant generation/version binding.

Related: T11, T12.

### MC06 — Storage fork + valid transparency receipts

**Expected:** receipts expose/preserve equivocation; they do not select the winning branch or authorize continued replication.

Controls:

- same-predecessor conflict detection;
- forked/non-operational state;
- preserve both branches;
- external recovery/new epoch.

Related: T20, T22.

### MC07 — DKG outage + monitor outage

**Expected:** authority freezes as freshness expires; outage does not extend validity or trigger destructive action automatically.

Controls:

- explicit freshness intervals;
- fail-closed authority;
- separate non-replicating service continuity where safe.

Related: T23, T29.

### MC08 — Ordinary governance credential + command substitution

**Expected:** cannot retarget or transform an authorized negative/admin command into recovery or ceiling widening.

Controls:

- exact command kind/target/scope binding;
- authenticated role matrix;
- recovery role distinct.

Related: T24, T25.

### MC09 — Recovery threshold + stale/forged old-state reference

**Expected:** even authorized recovery must bind the exact old terminal evidence and create a fresh epoch with no implicit active grant carry-over.

Controls:

- recovery ceremony binds old epoch/head/fork evidence;
- new epoch/genesis identity;
- reissued grants only;
- durable incident linkage.

If the recovery threshold itself is malicious and all required old-state anchors are also compromised, this is M3.

Related: T25, T26, T30.

### MC10 — Unadmitted runtime + valid old CI evidence

**Expected:** no production authority claim.

Controls:

- evidence capsule binds exact source/toolchain/features/config/artifact digest;
- runtime/attestation identity matches admitted artifact;
- production-admission record names exact build.

Related: T27.

---

## 4. Independence profile template

Every production profile should instantiate a table like:

| Evidence role | Minimum identities | Required distinct failure-domain dimensions | Maximum tolerated compromised domains | If unmet |
|---|---:|---|---:|---|
| grant authority | deployment-defined | key/admin/subject-control at minimum where external independence claimed | profile-defined | deny |
| R3 approval | >= 1 additional approval in reference semantics | production policy-defined | 0 or profile-defined | deny |
| R4/R5 approvals | >= 2 | explicit common-mode analysis; at least dimensions mandated by production policy | `< threshold` | deny |
| runtime monitors | profile-defined | sensor/state/process/admin as applicable | profile-defined | freeze |
| recovery quorum | high-trust threshold | key/admin/organization/offline custody as applicable | `< threshold` | remain non-operational |
| transparency witnesses | profile-defined | service/admin/root as applicable | auditability-specific | freeze only if profile makes witness freshness required |

No default production values are implied by this template.

---

## 5. Catastrophic trust-root assumptions that must be explicit

A production safety case must name, at minimum, whether the following are inside or outside its guarantee:

- compromise of the configured recovery threshold;
- compromise of every trust-root key accepted by the verifier;
- compromise of the exact host/privilege layer capable of replacing verified RSK code;
- compromise of every independently required runtime monitor/failure domain;
- compromise of all durable-state anti-rollback anchors/witnesses;
- coercion/collusion of all required external human/organizational authorities.

If these are outside the guarantee, say so directly. They must not be implied away by use of terms such as "multi-signature" or "independent".

---

## 6. Verification obligations

The compromise matrix becomes executable evidence only when tests/fault injection establish, at minimum:

- every M0 row represented in software has a negative test showing no unauthorized `Allow`;
- every M1 row has tests showing one-domain compromise is insufficient when redundancy is configured;
- every M2 condition leads to explicit non-operational/fork/freeze state rather than automatic winner selection;
- M3 assumptions appear in the production-admission residual-risk record;
- generated multi-compromise histories include MC01–MC10 or documented equivalents;
- test artifacts bind the exact admitted source/toolchain/runtime identity.

**Production admission remains DENIED.**
