# Replicator Safety Kernel — Trusted Time and Continuity Test Plan v0.1

Status: **normative verification plan; authored design evidence only**

This document defines the minimum evidence required before a production RSK authority path can rely on verified time/freshness/continuity.

It contains no physical replication mechanism.

---

## 1. Core properties

A production implementation must establish:

1. caller-selected wall time cannot create positive authority;
2. accepted time uncertainty is explicit and conservative;
3. rollback/replay/restart discontinuity cannot revive authority;
4. source disagreement cannot be resolved by choosing the favorable source;
5. holdover uncertainty grows according to validated policy and eventually freezes authority if predicates become unprovable;
6. outages never extend grant/evidence validity;
7. valid trusted time cannot override hard negative authority facts.

---

## 2. Interval-boundary tests — TM-INT

Use authorization evaluation `E` and expiry `X`.

Required cases:

- `[E, E]` may satisfy lower bound if all other facts pass;
- `[E, X-1]` may satisfy full interval at chosen resolution;
- `[E-1, E]` fails because earliest < E;
- `[X-1, X]` fails because latest >= X;
- `[X, X]` fails;
- `[E+1, X-1]` passes temporal interval predicate;
- invalid interval earliest > latest rejects;
- arithmetic/resolution conversion overflow rejects;
- reference #1726 scalar semantics agree with zero-width interval behavior.

No test may use midpoint validity as the authority criterion.

---

## 3. Not-before/expiry tests — TM-WINDOW

For a grant/evidence window `[N, X)`:

- interval entirely before `N` denies;
- interval straddling `N` denies until earliest >= N;
- interval entirely inside `[N, X)` may pass;
- interval straddling `X` denies;
- interval entirely at/after `X` denies;
- outage cannot move `X`;
- source switch cannot reset `N` or `X`.

---

## 4. Freshness tests — TM-FRESH

Given verified observation interval and current time interval:

- conservative maximum age uses latest current / earliest observation;
- exact max-age boundary passes only according to declared inclusive/exclusive policy;
- one unit beyond max age denies;
- observation apparently in future relative to current interval freezes;
- stale monitor/policy/trust/safety snapshot denies even if another source reports a favorable midpoint;
- checked arithmetic handles zero/max values without wraparound.

---

## 5. Source authentication/lifecycle tests — TM-SOURCE

For every admitted source profile:

- valid authenticated observation accepted structurally;
- invalid authentication/signature/MAC rejects;
- unknown source rejects;
- wrong source role/profile rejects;
- retired/revoked source rejects;
- stale trust snapshot rejects;
- replayed source sequence/nonce rejects where applicable;
- unsupported protocol/profile rejects;
- claimed uncertainty smaller than trusted source/profile minimum cannot be self-selected;
- source cannot declare itself an independent failure domain.

Authenticated NTS or another protocol proves only its defined message/source security properties; policy/failure-domain/uncertainty tests still apply.

---

## 6. Source-disagreement/fusion tests — TM-FUSION

Under each configured deployment fusion profile:

- consistent source set produces conservative interval;
- one source disagrees within tolerated modeled bounds -> output remains conservative under declared fault budget;
- disagreement exceeds policy -> freeze/no verified authority time;
- requester cannot select one favorable source from disputed set;
- ordering of source input does not alter deterministic accepted result;
- duplicate source identity does not increase source quorum;
- two source identities sharing a required-distinct failure domain do not inflate independence;
- missing required failure-domain metadata fails closed;
- source removal/change requires policy-valid transition;
- fusion never returns narrower uncertainty than justified by its declared compromise model.

If a profile claims tolerance of `f` compromised sources, tests/fault analysis must actually model that assumption. No universal Byzantine threshold is implied by this specification.

---

## 7. Rollback tests — TM-ROLLBACK

Inject:

- wall-clock regression;
- source observation sequence replay;
- continuity sequence rollback;
- same continuity sequence / different state digest collision;
- older continuity checkpoint after restart;
- older trust snapshot making a revoked source appear active;
- older policy allowing weaker time source/freshness;
- old signed observation replay;
- rollback of holdover starting point.

Expected result: no new positive authority until verified continuity/recovery resolves the condition.

Threats: T12, T18.

---

## 8. Forward-jump tests — TM-JUMP

Inject forward jumps:

- within allowed adjustment/discontinuity policy;
- at exact maximum allowed boundary;
- beyond maximum;
- from one source only;
- common-mode across dependent sources;
- accompanied by evidence freshness changes.

Required behavior:

- beyond-profile jump causes uncertainty/freeze;
- a jump may expire authority but cannot revive it;
- no automatic destructive/revocation response solely because the jump is unexplained;
- diagnostic/incident evidence retained.

Threat: T13.

---

## 9. Holdover tests — TM-HOLD

For every supported holdover profile:

- initial uncertainty matches verified synchronized state;
- uncertainty never shrinks without new evidence;
- validated drift bound grows monotonically with elapsed holdover;
- exact holdover-duration limit enforced;
- when widened interval first straddles expiry/freshness boundary, new authority freezes;
- source reacquisition requires continuity verification before uncertainty narrows;
- reboot during holdover cannot silently reset uncertainty/drift age;
- outage does not extend grant/evidence expiry.

---

## 10. Restart/reboot continuity tests — TM-RESTART

Scenarios:

- clean restart with valid durable continuity state and consistent new observations;
- restart with missing continuity state;
- restart with older checkpoint;
- restart after trust/policy rotation;
- restart while external sources unavailable;
- restart with conflicting time sources;
- restart after fresh-epoch recovery.

Properties:

- process restart alone cannot create fresh trusted continuity;
- missing/ambiguous continuity freezes authority;
- stale checkpoint rejected/non-operational;
- accepted previous state cannot be forgotten silently;
- fresh-epoch recovery starts with explicitly new continuity/trust semantics and no old positive-authority carryover.

---

## 11. Time-service outage tests — TM-OUTAGE

Inject loss of:

- one source;
- all network sources;
- local high-quality reference;
- authentication provider;
- trust/revocation source;
- time-policy source.

Expected behavior:

- validated holdover only within existing policy;
- uncertainty expands;
- pre-existing expiry unchanged;
- no grace extension;
- once predicates cannot be proven, new authority freezes;
- ordinary non-replicating service may continue only according to its separate safety case;
- outage does not authorize destructive action.

Threats: T23, T29.

---

## 12. Bootstrap/circularity tests — TM-BOOT

Time often verifies certificate/policy validity while certificates/policies determine accepted time sources. Test the declared bootstrap architecture explicitly.

Cases:

- stale trust snapshot + favorable untrusted source cannot mutually validate each other;
- new source cannot establish its own key validity window using only its own time;
- policy rollback cannot change time acceptance and then use that time to validate the rolled-back policy;
- root/initial continuity assumptions match P0 production-admission record;
- rotation transition can be replayed deterministically from retained evidence;
- catastrophic root/bootstrap assumptions are explicitly identified.

---

## 13. Negative-state composition tests — TM-NEG

For a fully valid `VerifiedAuthorityTime`, independently add:

- subject quarantine;
- lineage revocation;
- grant revocation;
- fork/non-operational state;
- stale ledger cursor/head;
- containment drift;
- invalid quorum/policy;
- unadmitted runtime/build.

Result: no positive authority. Time is never an override channel.

---

## 14. Property/state-machine suite — TM-PROP

Generate histories including:

- source observations;
- uncertainty changes;
- source additions/removals;
- trust/policy rotations;
- backward/forward jumps;
- outages;
- holdover;
- restarts;
- authorization evaluation/commit/expiry;
- negative-state mutations;
- fresh-epoch recovery.

Global properties:

1. successful commit interval entirely satisfies temporal window;
2. accepted continuity sequence never regresses in-epoch;
3. same sequence never accepts two different states;
4. uncertainty never narrows without supporting verified evidence/fusion transition;
5. outage never extends existing expiry;
6. restart never resets continuity implicitly;
7. unresolved source disagreement never produces favorable positive authority;
8. requester cannot choose source/profile to lower verified policy;
9. temporal validity never overrides hard negative state;
10. recovery with broken continuity requires explicit governed fresh-epoch semantics where same-epoch proof is unavailable.

---

## 15. Formal-model properties — TM-FORMAL

TLA+ v0.2 or companion temporal model should cover abstractly where practical:

- `SuccessfulCommitNeverPrecedesEvaluation`;
- `SuccessfulCommitAlwaysBeforeExpiry`;
- `UncertainIntervalCannotAuthorize`;
- `ContinuitySequenceNeverRegresses`;
- `ContinuityCollisionCannotAuthorize`;
- `OutageCannotExtendAuthorityLifetime`;
- `TimeRollbackCannotRestoreAuthority`;
- `RestartWithoutContinuityCannotAuthorize`;
- `NegativeStateDominatesValidTime`;
- `FreshEpochDoesNotCarryOldTemporalAuthority`.

Cryptographic source authentication and real clock-accuracy assumptions remain environmental unless explicitly modeled.

---

## 16. Fuzz/bounds tests — TM-FUZZ

Fuzz raw time observation/continuity decoders with:

- arbitrary/truncated bytes;
- extreme timestamps;
- interval endpoints near integer limits;
- invalid earliest/latest ordering;
- oversized source sets/metadata;
- unknown profile/domain tags;
- duplicate source records;
- invalid uncertainty encodings;
- repeated/cyclic continuity references if format permits.

No malformed input may panic/overflow or produce verified authority time.

---

## 17. Exact evidence subject

Executed evidence binds:

- exact code commit/tree;
- Cargo.lock/toolchain/Nix environment;
- time source/profile/fusion policy versions;
- trust/policy snapshot fixture digests;
- fault schedule/random seeds;
- hardware/security/OS clock profile where deployment-specific;
- artifact/runtime digest where applicable.

---

## 18. Promotion rule

Production trusted-time admission requires green applicable evidence for:

- TM-INT;
- TM-WINDOW;
- TM-FRESH;
- TM-SOURCE;
- TM-FUSION;
- TM-ROLLBACK;
- TM-JUMP;
- TM-HOLD;
- TM-RESTART;
- TM-OUTAGE;
- TM-BOOT;
- TM-NEG;
- TM-PROP;
- TM-FORMAL as modeled;
- TM-FUZZ.

Authored plans and queued CI are not executed evidence.

**Production admission remains DENIED / NOT YET ELIGIBLE.**
