# SAT External Effect Receipt Verifier — Implementation and Benchmark Plan v0.1

**Date:** 2026-11-26  
**Status:** engineering plan; schemas and examples authored; implementation and execution pending.  
**Parent:** `SAT_EXTERNAL_EFFECT_RECEIPT_PROFILE_V0_1_2026-11-26.md`  
**Target crate:** `crates/core/symthaea-sat-trace-verifier`

## 1. Design decision

The verifier is a read-only evidence evaluator. It MUST NOT execute, authorize, retry, compensate, or silently reconcile external operations. It computes a vector of bounded dispositions from frozen inputs and evidence bundles. Unknown or contradictory outcomes remain held for an external owner to reconcile.

The verifier MUST NOT turn a valid signature, transparency receipt, source label, or matching hash into a general assertion of physical or business truth. It MUST report separate facts for receipt structure, attempt binding, signature/key verification, transparency inclusion/consistency, source independence, source-state monotonicity, effect correspondence, and action disposition.

## 2. Normative verification pipeline

The implementation should execute the following phases in fixed order. Stop at a hard identity or cryptographic failure, but retain all independent diagnostics that can safely be computed without trusting failed inputs.

1. **Parse and schema:** strict JSON; reject duplicate keys, unknown critical fields, invalid formats and unsupported versions. Validate the attempt and receipt against their pinned schemas.
2. **Freeze context:** recompute the manifest, attempt, authority, disclosure, effect-binding and receipt-statement commitments under the declared canonicalization and domain-separation profile.
3. **Bind identities:** exact run ID, manifest digest, attempt ID, idempotency key, intent digest, authority/disclosure digests, effect binding and frozen trust-policy digest.
4. **Verify signatures:** resolve signer identity and key from the frozen trust bundle; validate key usage, algorithm allowlist, certificate/credential chain, validity and revocation semantics at signing time and verification time; verify the source statement bytes, not an ambiguous reconstructed subset.
5. **Verify transparency:** validate the selected SCITT COSE receipt or Sigstore Rekor bundle, signed checkpoint, inclusion proof and consistency proof. Compare against a prior checkpoint stored outside the subject's control. Missing prior state is `unanchored` or `inconclusive`, never silently accepted as continuous.
6. **Evaluate source trust:** verify source role and trust-policy binding; evaluate the independence dependency graph and source-specific sequence monotonicity. Distinct key IDs are not sufficient independence.
7. **Reconcile outcome:** compare every observed effect field to the exact authorized effect projection. Preserve applied, not-applied, partial, unknown, rejected, mismatch and conflict as distinct outcomes.
8. **Update local evidence state:** append accepted checkpoint and source-sequence observations transactionally. A failed or uncertain persistence operation must leave the previously accepted state intact and return `hold_reconcile`.
9. **Emit report:** produce a deterministic signed/unsigned report vector, artifact digest list, verifier version, policy digest, trust-bundle digest and limitations. A SAT qualification report is a separate evaluator decision.

## 3. Rust module boundary

Add a new `effect_receipt` module to `symthaea-sat-trace-verifier`. Keep the current event-chain verifier and checkpoint structural inspector intact. No cryptographic pass flag is allowed until an adapter actually verifies the required primitive and trust inputs.

Proposed public concepts:

- `EffectAttempt`: frozen run, attempt, intent and exact-effect identity.
- `EffectReceipt`: receipt identity, reported outcome, optional observed projection and source attestations.
- `EffectVerificationContext`: expected run/manifest/policy digests and declared effect profile.
- `EffectVerificationReport`: orthogonal dispositions for structure, binding, authenticity, transparency, independence, source freshness, effect match and mutation control.
- `AttemptLedger`: caller-supplied immutable prior state; owns idempotency-to-intent binding, current attempt IDs, terminal states and source sequence pins.
- `EvidenceResolver`: resolves content-addressed evidence references from an offline evidence bundle. It has no network or mutation capability.
- `SignatureVerifier`: provider-specific COSE/JWS/DSSE verification adapter; returns verified signer identity and verified signed bytes.
- `TransparencyVerifier`: provider-specific SCITT/Rekor proof adapter; returns exact statement inclusion and prior-head consistency results.
- `IndependenceEvaluator`: applies a separately pinned dependency graph and quorum policy; returns `satisfied`, `not_satisfied` or `unverified` with common-cause cut sets.

All adapters should be passed explicitly. The core reconciliation logic must be deterministic and testable without network access. Network collection is a separate, untrusted evidence-acquisition process.

## 4. Attempt state machine

The `idempotency_key` identifies stable intent. The `attempt_id` identifies one execution attempt. A retry preserves the idempotency key, intent digest and exact-effect binding but receives a new attempt ID.

Allowed semantic transitions:

- `Prepared -> Submitted -> Applied`
- `Prepared -> Submitted -> NotApplied -> RetryPrepared`
- `Prepared -> Submitted -> Unknown -> Hold`
- `Hold -> ReconciledApplied` or `Hold -> ReconciledNotApplied`
- `Submitted -> Partial -> Hold/CompensationRequired`
- Any contradictory terminal evidence -> `Conflicting -> Hold/Investigation`

Forbidden transitions:

- `Unknown -> NewMutation` without independent reconciliation.
- `Applied -> Retry` for the same idempotency key.
- `AttemptA receipt -> settle AttemptB` even when the idempotency key is the same.
- Same idempotency key with a different intent/effect binding.
- Second terminal settlement overwriting the first accepted terminal record.

`NotApplied` only permits another attempt after its required evidence sources are authenticated, sufficiently independent, current and bound to the exact attempt. A timeout, HTTP error or connector failure is not proof of non-application.

## 5. Cryptographic and transparency interoperability

Use existing standards rather than defining a SAT-specific signature or ledger:

- **SCITT:** use RFC 9943 COSE Signed Statements and COSE Receipts (RFC 9942). RFC 9943 is a transparency architecture for identifiable signed statements; registration proves that a statement was registered, not that its contents are accurate. Each source should sign its own assertion. A SAT aggregator must not rewrite several parties' assertions into a single apparent issuer statement.
- **SCRAPI:** implement against the published `draft-ietf-scitt-scrapi-11` API profile while it remains an Internet-Draft. Pin the exact draft/profile version in adapter metadata; do not present it as a final RFC.
- **Sigstore Rekor:** consume provider-native bundles and verify the signed entry/checkpoint and the corresponding inclusion and consistency proofs. Persist prior accepted checkpoints locally. A new checkpoint without a proof from the prior pinned state is not continuity evidence.
- **Canonicalization:** the existing trace verifier implements a restricted JCS-like SAT v0.1 profile with safe-integer-only numbers. It is not a claim of full RFC 8785 conformance. Keep that profile explicitly named and versioned, represent precise quantities as strings, and add cross-language vectors. A future full RFC 8785 profile requires independent conformance vectors before migration.
- **Key lifecycle:** bind key usage, algorithm policy, issuer identity, signing time, trust-bundle version and revocation snapshots. Distinguish ordinary retirement from compromise and apply explicit effective-time rules. Never fetch live key material during offline qualification and silently substitute it for the run's pinned trust bundle.

SCITT transparency is per signed statement/issuer. A source statement's truth remains a source-reliability question; the receipt's truth remains an effect-reconciliation question; the transparency receipt only proves the applicable statement was included under the log's rules.

## 6. Independence is a dependency graph

Model at least these nodes and relations:

- legal/operator control and beneficial ownership;
- signing-key custodian and identity provider;
- cloud/hosting tenant and administrator;
- source application and database;
- network path, region and time source;
- sensor calibration and physical measurement chain;
- transparency-service operator and witness governance;
- auditor, evaluator and qualification subject conflicts.

The verifier should calculate the minimum common-cause cut set that can defeat the required quorum. A policy may require `k` attestations, but `k` distinct key IDs do not imply `k` independent sources. If a dependency edge is unknown or undisclosed, independence remains `unverified`; do not infer independence from missing data.

For physical actuation, a controller receipt and a second software log are not independent physical observation. Prefer separately sampled position/force/current or other calibrated sensors with declared uncertainty and tamper boundary. SAT does not claim sensor output equals physical truth without measurement assurance.

## 7. Benchmark programme

Preserve native benchmark tasks, attack protocols and scoring. SAT adds evidence/qualification envelopes and labels adapted baselines honestly.

| Benchmark | Native role | SAT extension | Comparison discipline |
|---|---|---|---|
| AgentDojo | Prompt-injection robustness on tool-use tasks | Attempt-bound effect receipts, source read-back and disclosure evidence | Preserve native tasks, graders, attacks and benign controls |
| AgentHarm | Harmful agent capability | Verify whether harmful actions reach the emulated or real effect boundary | Keep native harmfulness score separate from SAT effect outcomes |
| DelegateGuard | Purpose-bound delegated capability enforcement | Add independent source truth, evidence custody, exact effect and transparency | Reproduce published policies; do not claim capability attenuation as SAT's invention |
| AuditBench | Hidden-behavior discovery | Require evidence-backed findings, falsifiable tests, independent regrade and benign calibration | Hide positive/negative target labels from investigators and SAT subjects |
| ControlArena / SHADE-Arena | Control protocol, monitor and attacker evaluation | Wrap trajectories in portable, independently re-verifiable event and effect evidence | Freeze protocol version, monitor visibility, attacker budget and environment |
| LADDER | Long-horizon manipulation | Actuator authority, revocation, calibrated read-back and physical effect correspondence | Preserve task horizon, embodiment, scene, success and cost measurements |

### Proposed initial targets (research targets, not results)

| Measure | Initial target | Evidence requirement |
|---|---:|---|
| Exact-effect mismatch accepted as matched | 0 | All finite negative controls; independent regrade |
| Stale/cross-run receipt accepted for another attempt | 0 | Replay, retry and identity-binding test matrix |
| Invalid signature/key/proof accepted | 0 | Known-good/known-bad cryptographic test vectors, including expiry and revocation-time cases |
| Invalid/missing transparency continuity accepted as anchored | 0 | Inclusion, consistency, split-view and stale-pin negatives |
| Unknown outcome followed by unauthorized new mutation | 0 | Stateful timeout/reconnect campaign with external sink oracle |
| Required independent source omitted but reported sufficient | 0 | Independence graph and common-cause cut-set cases |
| Benign exact-effect task falsely blocked | <=1% | Paired benign controls and confidence intervals by task family |
| Verification availability | 100% of required evidence states explicit | Missing artifacts must return unavailable/inconclusive, not pass |
| Evidence verification p50/p95 latency and cost | Measure before setting threshold | Include fetching, verification, witness, monitor and evaluator overhead |
| Human unaided capability after withdrawal | Non-inferior within preregistered 5% margin | Proper power, participant clustering, attrition and delayed retention controls |

Zero observed failures do not mean zero risk. Publish numerators, denominators, upper confidence bounds and clustered analyses. Do not use a rule-of-three bound as though repeated steps from one environment or attacker were independent trials.

### Baseline arms

- `B0`: unrestricted execution / native system.
- `B1`: static tool allowlist.
- `B2`: purpose-bound capability and delegation control.
- `B3`: label-flow / information-flow control.
- `B4`: output monitor / content guard.
- `B5`: benchmark-native control policy.
- `B6`: SAT composition with external observation and verifier.
- `B7+`: SAT ablations removing one observation, authority, custody or continuity boundary at a time.

Run matched task/seed pairs across arms. Freeze model revision, system and developer prompts, memory, tools, connectors, permissions, runtime, network, environment, attacker budget, evaluator, monitor configuration and resource accounting. If any dimension differs, label the result adapted or unmatched.

## 8. Execution stages and evidence levels

- **Stage A — contract conformance:** schema meta-validation, example validation, fixture uniqueness, cross-language canonicalization vectors. This is a source/contract check, not a Rust test pass.
- **Stage B — reference semantics:** Rust compilation, unit tests, property tests and mutation tests executed on exact commit SHA.
- **Stage C — cryptographic adapters:** valid/invalid COSE, DSSE, key-status, inclusion and consistency test bundles, run offline against pinned trust roots.
- **Stage D — integration:** owner-native authority/connector/sink adapters, emulator-backed external state read-back and unknown-outcome recovery.
- **Stage E — independent campaign:** sealed tests, adaptive attackers, benign controls, independent evaluator and preregistered statistics.
- **Stage F — field assurance:** production sink integrations, incident/rollback handling, drift monitoring, time-bounded qualification and change-triggered requalification.

Every published claim records exact repository SHA, manifest, policy, fixture set, test runner and evidence artifacts. A skipped workflow is `skipped`; an authored test not run is `not executed`.

## 9. Integration ownership

The SAT verifier observes and qualifies; it must not become another authority or action owner.

- `CogSec`: security posture, isolation and security telemetry evidence.
- `INTX`: external connector and disclosure boundary evidence.
- `Authority`: permit, delegation, purpose, expiry and revocation decisions.
- `Symtropy`: execution/replay, physical or simulated sink adapters and recovery evidence.
- `SAT`: frozen evaluation contract, independent receipt verification, evidence disposition and qualification report.

No duplicate authorization state machine, payment state machine or robot safety controller should be introduced. For unknown outcome, the owner responsible for the external sink performs reconciliation under its own policy; SAT independently records and grades the evidence.

## 10. Immediate implementation order

1. Add receipt/attempt parsing and cross-binding to the current Rust verifier, with authenticity and source independence explicitly `unknown`/`unverified`.
2. Implement typed reconciliation and prior-attempt/source-sequence state, then execute the 40 fixture catalog.
3. Add offline evidence bundle resolution and COSE/DSSE signature verification as separate adapters.
4. Implement SCITT RFC 9942/9943 and Rekor proof adapters; persist checkpoint pins outside subject-controlled storage.
5. Implement common-cause independence policy and source-specific read-back adapters.
6. Add benchmark adapters and start with deterministic emulators before real-world sinks.

No step may promote a lower evidence level by documentation-only changes.

## References (status is important)

- IETF [RFC 9943 — SCITT Architecture](https://www.rfc-editor.org/rfc/rfc9943.html), Proposed Standard, June 2026.
- IETF [RFC 9942 — COSE Receipts](https://www.rfc-editor.org/rfc/rfc9942.html), June 2026.
- [SCITT Reference APIs draft-ietf-scitt-scrapi-11](https://datatracker.ietf.org/doc/html/draft-ietf-scitt-scrapi-11), Internet-Draft, June 2026; work in progress.
- [Sigstore Rekor specification](https://github.com/sigstore/architecture-docs/blob/main/rekor-spec.md) and [security model](https://docs.sigstore.dev/about/security/).
- [RFC 8785 — JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785.html), Informational, not Standards Track.
- [AgentDojo](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html), NeurIPS 2024.
- [AgentHarm](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c493d23af93118975cdbc32cbe7323f5-Abstract-Conference.html), ICLR 2025.
- [DelegateGuard](https://arxiv.org/abs/2610.16005), October 2026 preprint.
- [AuditBench](https://alignment.anthropic.com/2026/auditbench/), March 2026.
- [LADDER](https://ladderrobotics.ai/research), long-horizon manipulation.