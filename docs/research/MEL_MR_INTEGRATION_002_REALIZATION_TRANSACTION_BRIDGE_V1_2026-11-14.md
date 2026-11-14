# MEL-MR-INTEGRATION-002 — Capability-qualified realization transactions

**Version:** 1.0 design contract  
**Date:** 2026-11-14  
**Status:** Research/specification; implementation and exact-head qualification pending  
**Parent issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12139

## 1. Purpose

This document defines the integration boundary between:

- **MEL-REAL-001** — capability-qualified musical realization (#10196, draft #10202); and
- **MEL-MR-EVAL-040** — replayable, budgeted edit-transaction decisions (#12107).

The integration makes backend choice replaceable without weakening musical intent, authorization, preservation requirements or evidence. A generator or renderer may propose a candidate, but cannot certify that candidate or authorize its commitment.

This work intentionally does not define a second provider abstraction. The realization contract owns backend capabilities and execution receipts. EVAL-040 owns candidate attempts, budget accounting, independent evidence gates and terminal decisions. This document defines how those two contracts compose.

## 2. System principle

**Stable intent; replaceable realization; independent verification; explicit human authority.**

A single musical request may be realized by a local symbolic composer, a MIDI instrument, an audio editor, a full-song generator or a live stream backend—but only where that exact backend/version exposes the required controls and observations.

The system must never:

- silently translate a hard note/score obligation into best-effort prompt steering;
- infer a capability from provider marketing, model name or a successful HTTP response;
- accept a provider-supplied digest without hashing the actual returned bytes;
- reuse one candidate's renderer, acoustic, preservation, metamorphic or human evidence for another candidate;
- interpret a timeout as proof that the provider did not execute;
- retry an uncertain mutation without reconciliation;
- let preference, similarity or creative novelty compensate for a hard-gate failure;
- infer publishing, training, sharing or overwrite permission from an edit transaction's technical acceptance.

## 3. Composed transaction flow

Musician intent + source + authorization  
→ Backend capability negotiation  
→ Exact MusicalRealizationRequestV1  
→ Selected realizer and immutable request identity  
→ Exact output bytes + MusicalRealizationReceiptV1  
→ Independent byte rehash / materialization  
→ EVAL-040 immutable candidate attempt  
→ Operation, renderer, acoustic, preservation, metamorphic, rights and human evidence (as required by policy)  
→ Accept / Repair / Abstain / Reject  
→ Separate musician-authorized commit/export/share

A missing evidence producer is an explicit unavailable or unresolved state. It is never converted to success by a default value, a high similarity score, a model confidence value or a second backend's output.

## 4. Identity and evidence bindings

Every realization attempt is bound to one immutable upstream intent and one source lineage. Every returned candidate receives its own attempt identity and evidence chain.

The EVAL-040 attempt must bind at least:

| Binding | Required content |
|---|---|
| Request | realization request ID, schema version and exact request digest |
| Capability | capability manifest digest, backend ID/version, adapter version and declared environment identity |
| Execution | provider/session/idempotency references where available, submission state, execution diagnostics and timestamps |
| Source | source artifact digest, source lineage, intent digest and authorization/consent references |
| Candidate | actual output digest, modality, byte length and independently observed output hash |
| Controls | requested, accepted, ignored, best-effort and unavailable controls kept distinct |
| Observations | exact available event/score/stem/timing observations; missing observations remain explicitly unavailable |
| Verification | candidate-specific symbolic, renderer, acoustic, preservation, metamorphic and human evidence receipts, each with verifier/policy version |
| Budget | submissions, render calls, repair iterations, elapsed duration and materialized bytes actually consumed |

### 4.1 Actual-byte rule

The transaction boundary must hash the exact output bytes it receives or materializes. A digest supplied by the provider or copied from a receipt is only a claim until independently checked against those bytes.

Output container digest and decoded PCM digest remain distinct when both are available. Symbolic score/MIDI digest, rendered WAV/container digest and decoded PCM digest must not be conflated.

Replay of a receipt proves integrity of the receipt structure and its declared bindings. It does not re-read a filesystem or re-execute a remote provider. Those operations require fresh observation/materialization.

### 4.2 Candidate isolation

Candidate A and candidate B may share intent, source and policy. They may not share candidate-specific verification merely because they came from the same request, provider, prompt, model or renderer.

A downstream report is reusable only where its contract explicitly names the shared artifact and proves that all relevant inputs and policies are identical. Otherwise it is candidate-scoped.

## 5. Provider execution state machine

Provider execution state is separate from the EVAL-040 terminal decision.

| Execution state | Meaning | Budget / next action |
|---|---|---|
| NotSubmitted | Dispatch boundary has not been crossed | No provider submission consumed; dispatch may proceed if policy permits |
| Submitted | Submission is evidenced; completion is not yet known | Count the submission; await or reconcile |
| Completed | Output and receipt are available and bound | Rehash bytes, then begin independent verification |
| Failed | Terminal failure is reconciled to this request | Count the submission; retain diagnostics |
| OutcomeUnknown | Timeout, disconnect, process crash or incomplete observation leaves execution/charge uncertain | Count conservatively; block blind retry or new mutation |

### 5.1 Unknown outcome is not permission to retry

When a call times out after crossing the dispatch boundary, the system cannot infer that no work occurred. The attempt enters OutcomeUnknown.

A retry is permitted only after reconciliation by a backend-supported idempotency/status mechanism or an independently checkable execution record that binds the original request. If the backend cannot resolve the state, the transaction remains unresolved and EVAL-040 must abstain.

A reconciled retry must be distinguishable from a new candidate and must not double-count a verified output. If a provider cannot offer idempotency or status lookup, that limitation is recorded in its capability manifest and the supported retry policy is stricter.

Every submitted request consumes the declared attempt/render budget, including failed, partial, cancelled, malformed and unusable responses. No budget is refunded merely because the output failed verification.

### 5.2 Explicit degradation

A provider fallback, model/version change, capability-manifest change, output modality change, or material change in effective controls creates a new realization attempt and receipt. It cannot silently inherit the old attempt's identity or verification.

## 6. Candidate portfolio and selection

A portfolio may contain bounded alternatives from multiple compatible realizers. Every candidate shares the same frozen request, source lineage, upstream musical obligations and authorization, while retaining unique realization and output identities.

The evaluator returns a vector of separate results, not a universal quality score:

- hard musical obligations: satisfied / violated / indeterminate;
- requested target change and protected-content preservation;
- collateral-change and locality observations;
- acoustic quality and audible artifacts;
- human judgments and uncertainty, when available;
- backend capability coverage and effective-control fidelity;
- provenance completeness and replay availability;
- latency, cost, render count, repair count and failure burden;
- abstention and unsupported-domain reasons.

Preference can help order candidates that have passed the applicable hard validity gates. It cannot override a missing rights decision, a protected-content violation, a hard musical obligation failure or an unresolved integrity state.

When several results are valid alternatives, preserve them as alternatives. Do not manufacture a single "correct" waveform when the musical contract permits multiple realizations or when reviewers disagree.

## 7. Authority and permission boundaries

The following claims remain distinct:

1. Capability declaration is not capability conformance.
2. Capability conformance is not successful execution on a particular request.
3. Output byte identity is not musical correctness.
4. Symbolic correctness is not acoustic correctness.
5. Audio similarity is not instruction execution.
6. Human preference is not proof of preserved material.
7. A valid reviewer signature is not proof of reviewer expertise or musical truth.
8. Rights admission is not benchmark qualification.
9. EVAL-040 acceptance is not permission to publish, train, federate, distribute or overwrite.

The final commit/export/share action must be separately authorized by the musician and applicable contribution/rights scope. Technical acceptance qualifies an exact candidate against declared evidence gates; it does not transfer authorship or expand permission.

## 8. Offline deterministic acceptance suite

The first integration must use local mock realizers. It must not require vendor credentials, remote services, paid accounts or network access.

Minimum known-answer and adversarial cases:

| ID | Scenario | Expected behavior |
|---|---|---|
| RT-01 | Same intent routed to two compatible backends | Exact intent and authorization remain identical; candidate receipts remain distinct |
| RT-02 | Audio-only backend receives exact-note request | Refuse before dispatch or explicitly downgrade only with new human-approved intent |
| RT-03 | Best-effort/advisory control offered for a hard requirement | Dispatch blocked |
| RT-04 | Backend ignores a requested control | Ignored/effective-control state recorded; downstream gates decide outcome |
| RT-05 | Capability manifest changed or stale | Attempt invalidated; no inherited conformance |
| RT-06 | Output bytes mutated after provider receipt | Independent rehash detects substitution |
| RT-07 | Candidate A receipt presented for candidate B | Reject lineage/candidate mismatch |
| RT-08 | Partial, malformed, failed or cancelled output | Typed failure; submission still consumes budget |
| RT-09 | Timeout after submission | OutcomeUnknown; no blind retry or mutation |
| RT-10 | Idempotency/status resolves an unknown call | Reconciliation binds original request; no duplicate acceptance or double count |
| RT-11 | Fallback to another provider or version | New attempt and receipt required |
| RT-12 | Render, repair, time or byte budget exhausted | Deterministic Repair/Abstain according to reason; never unbounded retry |
| RT-13 | Preferred candidate violates a hard obligation | Cannot outrank/replace a valid candidate or be accepted |
| RT-14 | EVAL-040 Accept followed by publish/share request without authorization | Separate action blocked |

Replay must reject reordered attempts, changed policy, stale manifests, candidate/source substitution, changed output digests, altered budget counters and accept-with-pending evidence.

## 9. Benchmark and competitive evaluation

Compare each backend on its supported task intersection. Do not rank a full-song generator on note-level control it does not expose, or a live improviser on long-form production if that is outside its task contract.

Report separate measurements:

- capability-claim precision and unsupported-request refusal;
- hard-control fidelity and ignored-control detection;
- success / failure / partial / unknown execution rates;
- verified acceptance at fixed candidate, render and wall-clock budgets;
- false acceptance on hard-negative controls;
- candidates, submissions, renders and repairs per verified acceptance;
- p50/p95/p99 time and cost per verified acceptance;
- abstention coverage and selective risk;
- protected-content preservation and target-obligation success;
- cross-backend and held-out renderer/domain degradation.

Required baselines include full regeneration, one-backend greedy repair, fixed-probe editing, local symbolic/MIDI editing and provider-native workflows, where those baselines support the same operation. Match source, instruction, scope, output duration, model version, controls, budget and evaluator. Publish incompatible task tracks separately.

A prospective product objective is to reach the same or better hard-gate and human-judged outcome with fewer candidate renders and lower p95 time/cost, without increasing abstention concealment or collateral-change risk. This is a hypothesis until measured.

## 10. Relationship to adjacent programmes

| Programme | Owns | Does not replace |
|---|---|---|
| MEL-REAL-001 #10196 / PR #10202 | Backend capability/request/receipt contract | Execution truth, musical verification or permission |
| EVAL-040 #12107 | Attempt ledger, budget and terminal decision | Backend capability negotiation |
| EVAL-035 #12007 | Actual-byte materialization and byte receipts | Rights authorization or perceptual correctness |
| EVAL-037 #12044 / PR #12050 | Musical obligations, protected invariants and reference adequacy | Evaluation of actual audio |
| EVAL-038 #12083 / PR #12084 | Metamorphic relations and replay | Correctness by itself |
| EVAL-039 #12098 / PR #12106 | Authenticated pseudonymous review judgments | Reviewer competence/independence governance |
| EVAL-015 #9044 / EVAL-016 | Renderer execution, event lineage and acoustic evidence | Human preference and legal rights |
| EVAL-021 #9184 | Blinded listener calibration and unseen-renderer generalization | Exact symbolic oracle |
| RIGHTS / integration #12035 | Purpose-specific rights-to-byte admission | Musical benchmark qualification |
| MEL-013A #10051 | Studio artifact transaction and commit authority | Automatic permission from EVAL-040 |
| RMI-Bench #10088 | Causal reciprocal musical interaction | Editing quality or generation preference |

## 11. Implementation sequence

1. **Schema bridge:** add a transaction-side reference to the exact MEL-REAL-001 request, manifest, receipt, execution state and byte-verified candidate output. Keep the realization contract and EVAL-040 modules independently versioned.
2. **Offline mock adapters:** implement one deterministic symbolic/MIDI realizer and one audio-only/streaming mock with deliberately different capabilities.
3. **Attempt reconciler:** exercise Submitted/Completed/Failed/OutcomeUnknown, idempotency, partial results and budget consumption.
4. **Candidate isolation:** prove no evidence transfer across candidate identities and no silent provider fallback.
5. **Independent verifier composition:** bind EVAL-035 byte hashes and task-specific EVAL-036/037/015/016 reports to each attempt; require EVAL-038/039/021/rights evidence where the frozen policy requires them.
6. **Studio integration:** display candidate portfolio and evidence status; preserve explicit musician commit/export/share authority.
7. **Real backend adapters:** add one backend at a time only after version-specific capability probes, terms/rights review, actual-byte receipt capture and failure-reconciliation behavior are demonstrated.

Exact-head formatting, compilation, unit tests, fixture replay and Clippy must be reported separately. Queued/skipped checks are not passes. No external benchmark or production-quality claim follows from schema fixtures.

## 12. Claim ceiling

This contract, when implemented and qualified, can establish that a particular realization attempt was capability-qualified, its exact output bytes were observed, its costs/outcomes were accounted for, and its required independent evidence gates were replayably satisfied.

It cannot by itself establish musical beauty, universal cultural validity, human preference, legal authorship, consent, consciousness, provider honesty, or state-of-the-art superiority. Those require separate evidence programmes and explicit claim scopes.
