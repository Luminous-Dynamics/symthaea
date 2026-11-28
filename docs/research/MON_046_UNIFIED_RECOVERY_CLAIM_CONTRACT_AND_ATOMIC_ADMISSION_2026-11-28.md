# MON-046 — Unified Recovery Claim Contract and Atomic Admission Boundary

**Profile:** proposed mon046-unified-recovery-claim-v1  
**Status:** research contract and structural schema; no runtime implementation or production authorization.  
**Date:** 2026-11-28

## Executive thesis

Recovery assurance should be a compiler over independently sourced evidence, not a new backup engine or restore-test scheduler. Its output is a typed, time-bounded claim whose subject, scope, evidence custody, evaluator, policy, dependencies and freshness can be independently inspected.

MON-041, MON-042, MON-044 and MON-045 address important pieces. Their current boundaries remain intentionally offline: MON-041 consumes supplied trust-key and nonce snapshots and does not fetch/re-hash artifact bytes; MON-044 hashes a supplied required-scope inventory but does not authenticate its origin; MON-045 consumes caller-supplied authentication flags and graph digests and does not authenticate the declared dependency set. MON-046 defines the interface between these modules without pretending those limitations are solved.

The central design rule:

> An observation is not an authenticated observation; authenticated evidence is not a complete scope inventory; a complete inventory is not proof of successful recovery; and a qualified recovery claim is not itself authorization to mutate production.

These are four separate propositions.

## 1. Four-plane assurance model

| Plane | Question | Responsibility | Required output |
|---|---|---|---|
| Observation | What bytes, events or state were observed, where and when? | Adapter / capture agent | Immutable artifact descriptor and capture receipt |
| Custody and appraisal | Who produced it, was it altered, was the signer trusted at capture, and is trust fresh now? | MON-041 plus independent trust bootstrap | Authenticated appraisal result, trust checkpoint and replay-ledger checkpoint |
| Scope and semantics | What state had to be captured, what was covered, and did the restored system satisfy its invariants? | MON-044 plus recovery evaluators | Authenticated scope manifest, capability assessment and semantic results |
| Claim and reliance | Which conclusion follows, what invalidates it, and may a relying party act on it? | MON-045 plus MON-046 relying-party gate | Signed claim bundle; optional, separately bounded admission token |

A verifier must never convert a missing upstream plane into a positive result. A structurally valid envelope can still have an indeterminate campaign disposition.

## 2. Unified claim envelope

The companion structural schema defines the envelope, not a complete proof calculus. Semantic validators must enforce the following invariants.

### 2.1 Subject binding

Every claim binds stable system and environment identifiers, a recovery-boundary digest identifying the scenario/target/recovery point, the source-state interval or snapshot used for adjudication, the exact predicate and a versioned policy digest.

Database restore, Kafka capture completeness, application readiness and end-user service availability are different predicates. They must not collapse into one generic success bit.

### 2.2 Evidence references

Each evidence descriptor binds evidence class, source identity, artifact SHA-256 and byte length, media type and immutable content address, capture interval and clock uncertainty, capture principal, connector/build identity and configuration digest, MON-041 receipt digest and verifier appraisal digest.

The consumer MUST retrieve or receive the artifact bytes and recompute length and digest before parsing or relying on them. A signed digest descriptor alone does not show that the referenced bytes were observed by the verifier.

Artifact digest, receipt digest, appraisal-result digest and final claim digest are different objects. Never reuse one digest field for all four.

### 2.3 Authentication is an appraisal, not a boolean

MON-045's current authenticated boolean is a caller assertion and must not become a trust boundary. The integration adapter must require a content-addressed appraisal result containing:
- verifier identity and verifier binary/build digest;
- trust-anchor-set digest and independently pinned bootstrap identity;
- trust-snapshot digest and monotonic trust epoch;
- signer identity, key validity and authenticated revocation checkpoint;
- durable nonce-ledger checkpoint and atomic-consumption result;
- artifact byte re-hash result;
- appraisal policy digest, result digest and reason codes.

A missing trust anchor, stale revocation checkpoint or unavailable nonce ledger is indeterminate. A cryptographically invalid receipt, scope mismatch, digest mismatch or confirmed replay is blocked. A caller-supplied authenticated=true value is never sufficient.

Trust bootstrap is a separate root-of-trust problem. The trust snapshot must be authenticated by a root configured outside the evidence being evaluated, or by a verifiable threshold/delegation chain rooted there. A verifier must reject rollback to an older trust epoch.

### 2.4 Scope and negative space

The required-scope manifest must itself carry authenticated provenance. It must identify the authority that enumerated required state surfaces; source system/schema/application boundary and enumeration time; enumeration method; required, conditional, excluded, unsupported and unknown surfaces; exceptions and the policy/authority approving them.

MON-044 can evaluate consistency of a supplied inventory. It cannot establish that the inventory is complete unless its origin and enumeration procedure are independently evidenced.

A mandatory surface with missing evidence is indeterminate. A mandatory surface declared unsupported or excluded without an accepted exception is blocked. Positive CDC reconciliation cannot compensate for an unenumerated DDL, schema-history, sequence, large-object, privilege, job, routing or application-state surface.

### 2.5 Dependency closure

MON-045's required edges must be derived from or checked against an authenticated dependency declaration. The claim pins the exact scope-manifest digest, all required evidence/appraisal digests, capability/conformance digests, prerequisite claim digests, policy digest and evaluator digest.

The validator must distinguish an absent required node from a deliberately absent optional context node. Optional context can enrich an explanation; it cannot satisfy a required edge. If authenticated inventory completeness is unavailable, the claim remains indeterminate even when the declared graph is internally consistent.

## 3. Time, revocation and freshness

A single wall-clock timestamp is too strong an assumption for distributed recovery evidence. Capture time should be represented as an interval [earliest, latest], with explicit uncertainty and clock-source quality.

For a key with effective revocation time t_r:
- if the entire capture interval is before t_r, the historical signature may remain valid under policy;
- if the entire interval is at or after t_r, the receipt is blocked;
- if the interval overlaps t_r, the receipt is indeterminate unless an independently trusted time source resolves the ordering.

This preserves MON-041's as-of-capture principle while refusing to manufacture certainty at a clock boundary. A later compromise finding may carry an authenticated compromise-effective interval; policy, not an untrusted caller timestamp, decides retrospective scope.

A claim's effective expiry is the earliest applicable limit:

claim_not_after = min(evidence expiries, conformance expiries, trust checkpoint expiry, policy max age, source freshness deadline, revocation freshness deadline)

Subtract clock uncertainty conservatively. Missing or stale revocation watermarks must shorten the usable horizon or force indeterminate; they must never imply that no revocation occurred.

Revocation and invalidation are monotonic within a trust epoch: later evidence may create a new claim version, but must not erase the historical fact that an earlier claim was invalidated.

## 4. Event-sourced claim lifecycle

Treat evidence, appraisals, trust updates, scope changes, claim evaluations and revocations as immutable events. A current claim is a deterministic projection over the content-addressed event set, policy digest, evaluator build digest, trust epoch/revocation checkpoint and scope-manifest digest.

A correction is a new event referencing the prior event, not an in-place rewrite. Every derived claim records its dependency-trace digest and invalidation generation. This permits deterministic replay, retrospective review and exact explanation of why a prior green claim is no longer usable.

For efficiency, recompute only the transitive closure of changed mandatory dependencies. Scoped invalidation is safe only after the required-edge set is authenticated. Otherwise, "unaffected" means only "not connected in the graph we were given", not "independently shown unaffected".

## 5. Atomic admission boundary (future, separately gated)

A qualified recovery claim must not directly set production_action_permitted=true. A relying party may optionally issue a short-lived, one-use admission token bound to:
- exact claim digest and policy digest;
- target environment/resource and exact operation;
- maximum expiry/TTL;
- current trust/revocation epoch;
- current claim-invalidation generation or fencing token;
- unique nonce and intended relying party.

At the actual control-plane mutation boundary, the enforcement point must atomically:
1. verify token signature and exact target/operation binding;
2. verify TTL, trust epoch, revocation checkpoint and current claim digest;
3. compare invalidation/fencing generation with the current generation;
4. atomically consume the nonce;
5. perform or commit the authorized mutation under the same serialization boundary.

A dashboard check followed by a later unguarded API call is vulnerable to TOCTOU. If the resource API cannot enforce the final fencing/consumption step, MON must report admission as unavailable/indeterminate and must not claim safe production authorization.

MON-046 v1 defines this contract only. It does not implement an authorizer, issue production tokens or change any existing production_action_permitted=false invariant.

## 6. Interoperability profile

Do not create a proprietary signing and attestation stack unnecessarily.

- Use a versioned in-toto Statement-style subject/predicate separation for claim meaning and exact artifact identity.
- Use DSSE or an equivalent reviewed envelope for domain-separated signing of typed payloads.
- Use RFC 8785 JSON Canonicalization Scheme only where JSON is selected as the signed representation; publish cross-language test vectors and reject duplicate property names and unsupported numeric ranges.
- Consider an append-only transparency log for claim, policy and trust-root history. A log proves inclusion/ordering under its trust assumptions; it does not prove the evidence is true or complete.
- Keep transport, evidence serialization, predicate semantics and relying-party policy independently versioned.

References:
- IETF RATS architecture: https://www.rfc-editor.org/rfc/rfc9334.html
- JSON Canonicalization Scheme: https://www.rfc-editor.org/rfc/rfc8785.html
- DSSE protocol: https://github.com/secure-systems-lab/dsse/blob/master/protocol.md
- in-toto Statement: https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md
- Sigstore Rekor transparency log: https://docs.sigstore.dev/logging/overview/

These are interoperability building blocks, not claims that MON conforms to them yet.

## 7. Benchmark design: prove the evaluator before comparing vendors

MON-043 should add the MON-046 contract as an integration arm and separately score each layer. Baseline arms must receive the same source state, workload, recovery boundary, fault schedule, target and evidence budget.

### 7.1 Hard safety metrics

| Metric | Proposed gate |
|---|---:|
| Critical false-qualified states in fixed negative controls | 0 |
| Forged or replayed evidence accepted | 0 |
| Required scope omission accepted as complete | 0 |
| Stale trust/revocation checkpoint treated as fresh | 0 |
| Dependent claim left qualified after mandatory revocation | 0 |
| Relying-party token usable for a different target or operation | 0 |
| Admission after invalidation-generation change | 0 |

These are deterministic mutation-suite gates, not statistical claims of zero real-world risk.

### 7.2 Quality and efficiency metrics

Report with denominators and confidence intervals:
- false-qualified, false-blocked and indeterminate dispositions separately;
- required-scope omission recall and scope enumeration coverage;
- evidence custody/replay attack acceptance;
- impacted-claim precision and recall;
- unrelated-claim invalidation rate;
- stale-claim exposure time and revocation-to-enforcement latency;
- evaluator mutation score and oracle escape count;
- p50/p95/p99 incremental evaluation latency and overhead;
- storage, compute, egress and operator time per qualified recovery claim;
- actual service RTO, RPO and application-invariant restoration.

Never combine safety failures and cost into one weighted score. Report a Pareto frontier and a selective-risk/coverage curve so an evaluator that returns indeterminate for every case cannot appear successful.

### 7.3 Independence and statistical design

The experimental unit is the independently sampled dependency cluster, not replay count. The failure-domain inventory must cover compute/hypervisor, storage, network, identity, control plane, clock, capture path, trust service, source inventory, evaluator/oracle and external dependencies. Shared or transitively connected domains collapse into one cluster. Unenumerated common causes remain a stated limitation.

Use deterministic mutation fixtures for evaluator correctness and separate operational campaigns for reliability inference. Seal holdouts. Pre-register sampling, exclusions, stopping rules, oracle adjudication and analysis. Never infer four-nines reliability from a few dozen replays.

## 8. Required adversarial scenarios

The companion scenario manifest defines the initial contract-level attack set. The first release should include at least:
1. caller flips an authentication boolean without producing a verifier appraisal;
2. attacker replaces artifact bytes after signing the descriptor;
3. trust snapshot is unsigned, stale or rolled back;
4. two concurrent admissions consume the same nonce;
5. capture-time uncertainty interval overlaps effective key revocation time;
6. revocation checkpoint becomes stale between evaluation and action;
7. a required scope item or dependency edge is omitted from an otherwise self-consistent graph;
8. same Kafka event is omitted from both capture records and reconciliation arrays;
9. required DDL/schema-history/non-row state is missing;
10. optional context dependency fails and incorrectly downgrades or qualifies a mandatory claim;
11. shared control plane is declared as two independent failure domains;
12. root claim is absent while an orphan invalid node exists;
13. evaluator mutation changes blocked/indeterminate precedence;
14. all-indeterminate output attempts to pass the benchmark;
15. token is replayed against another environment, operation or older fencing generation.

Each case needs an exact expected disposition and reason family. Tests must assert both result and evidence trail, not merely a non-zero exit code.

## 9. Industry comparison and matched baseline

MON is an assurance overlay. The baseline comparison must credit each product for its existing capabilities:

| Baseline | Native strength | MON augmentation to measure |
|---|---|---|
| AWS Backup Restore Testing + validation | Scheduled restores, recovery-point selection, restore duration and event-driven validation | Authenticated independent state inventory, evidence provenance and claim dependencies |
| Veeam SureBackup | Isolated virtual lab, live application tests, backup verification and content scan | Portable appraisal outputs, source-scope closure and cross-vendor claim lifecycle |
| Azure Site Recovery | Recovery plans, ordered failover tasks and test failover | Independent proof that every required dependency and service invariant was met |
| PostgreSQL pg_verifybackup + actual test restore | Backup-manifest verification plus real server startup/data validation | Typed application-level state claims and cross-system composition |
| Jepsen / fault-injection harness | Distributed consistency under controlled faults | Recovery-boundary, custody, coverage and operational-readiness claims |

Official references:
- AWS restore testing: https://docs.aws.amazon.com/aws-backup/latest/devguide/restore-testing.html
- AWS validation: https://docs.aws.amazon.com/aws-backup/latest/devguide/restore-testing-validation.html
- Veeam SureBackup: https://helpcenter.veeam.com/docs/vbr/userguide/surebackup_job.html
- Azure recovery plans: https://learn.microsoft.com/en-us/azure/site-recovery/recovery-plan-overview
- PostgreSQL pg_verifybackup: https://www.postgresql.org/docs/current/app-pgverifybackup.html
- NIST SP 1339 (June 2026): https://csrc.nist.gov/pubs/sp/1339/final

Do not publish superiority claims until matched experiments exist and are independently adjudicated.

## 10. Delivery sequence

| Order | Increment | Exit condition |
|---|---|---|
| P0 | Freeze envelope, threat model and structural schema | Schema and fixture parse; review of semantic invariants |
| P1 | Cross-language canonicalization and signed claim vectors | Go/Rust produce byte-identical vectors; mutation tests catch malformed/ambiguous inputs |
| P2 | MON-041 verifier appraisal output | Trust bootstrap, artifact re-hash, authenticated revocation checkpoint and durable atomic nonce ledger are evidenced |
| P3 | MON-044 authenticated scope provenance | Inventory origin and enumeration completeness have independent evidence |
| P4 | MON-045 integration | No caller-supplied authentication boolean; graph closure is rooted in authenticated manifests |
| P5 | MON-043 oracle and matched runner | Sealed holdout, evaluator mutation suite and independent adjudication |
| P6 | Admission reference monitor | Atomic nonce consumption and fencing enforced at actual mutation boundary |
| P7 | Matched live campaigns | Reproducible cross-engine results with independent failure-domain audit |

P2-P6 are not prerequisites for reviewing the research contract. No runner status is treated as a pass.

## 11. Explicit non-claims

This change does not restore backups or replace AWS, Veeam, Azure, PostgreSQL, Kafka or Jepsen; authenticate a trust root, source inventory, capture adapter or live clock; prove that a declared graph contains every real dependency; certify any vendor connector; demonstrate operational recovery reliability or SOTA performance; or authorize production traffic promotion/failover/mutations.

The research contribution is a composable and falsifiable contract for recovery assurance whose claims can be independently checked, invalidated and safely consumed by relying parties.
