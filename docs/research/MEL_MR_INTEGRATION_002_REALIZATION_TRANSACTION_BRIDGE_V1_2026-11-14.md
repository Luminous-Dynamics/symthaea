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


## 13. Current ecosystem implications (14 November 2026 snapshot)

This section is a deployment-design snapshot, not a claim of integration or qualification with any named vendor. Product capabilities, availability, pricing and terms can change; an adapter must pin the exact endpoint/product/model/version and re-check the applicable terms before use.

### 13.1 Generators and studios are already converging

- **Suno Studio 2.0** documents a chat bar for generating audio/MIDI clips and arranging, MIDI piano-roll editing, a built-in wavetable synthesizer, track effects, automation and custom chat-designed plugins. Its current documentation says Studio 2.0 is a Premier-tier product and does not support VST or Audio Units plugins. This means conversational generation, MIDI, effects and automation are not differentiators by themselves.
  - Official source: https://help.suno.com/en/articles/13670529
  - Product launch notes: https://suno.com/blog/studio-2

- **Moises Studio** documents simultaneous shared browser sessions, multitrack recording, AI-generated complementary parts, recorded/generated-to-MIDI transcription, agentic editing with visible and reversible steps, stem separation across 30+ instrument/vocal categories, MIDI/MusicXML export and DAWproject session export. Moises also states that user uploads are not used to train its models. These are real product capabilities and trust commitments; Melothaea should compete through interoperable verification and independently replayable evidence rather than pretending collaboration, reversibility or export are absent elsewhere.
  - Official source: https://moises.ai/newsroom/product-announcements/moises-launches-studio-collaborative-workspace/

- **Google Lyria RealTime** documents a persistent bidirectional WebSocket music stream with weighted prompts and controls including guidance, BPM, density, brightness, scale, bass/drum muting, generation mode and seed. The documented output is raw stereo PCM; the model is labelled experimental. Its documentation notes that BPM/scale changes require a context reset and abrupt prompt changes can need gradual weighting/cross-fading. These are useful creative controls, but they are not a claim of exact note-event editing or deterministic symbolic preservation.
  - Official source: https://ai.google.dev/gemini-api/docs/realtime-music-generation

### 13.2 Adapter classification

| System class | Useful role | Evidence and control boundary |
|---|---|---|
| Local symbolic/MIDI editors and renderers | Exact note/event edits, lineage, score-preserving transforms | Candidate control may be precise; still require independent rendering and acoustic verification |
| Lyria-like streaming generators | Improvisation, rapid texture/arrangement proposals, live performance candidates | Audio-stream output and high-level controls; exact note identity and stem isolation are unavailable unless independently produced |
| Suno-like generative studios | Rapid song/clip ideas, generated stems/MIDI and creative exploration | Treat exported artifacts and available project metadata as untrusted inputs; do not assume a public machine API from UI feature documentation |
| Moises-like collaborative studios | Stem workflows, co-editing, arrangement, external DAW handoff | Integrate at explicit import/export/project boundaries unless a supported API and terms authorize deeper integration |
| Independent DAWs/renderers | Final production, high-fidelity playback and user-owned session continuity | Maintain portable project state; bind each render to exact project, plugin, renderer and asset versions |

A capability manifest must say which controls are exact, bounded, advisory, best-effort, unavailable or merely observed after generation. Do not convert a high-level prompt or a UI action into an exact symbolic guarantee.

### 13.3 Product position: a portable evidence layer, not a closed DAW

The product should be able to work with a musician's current studio rather than demand that the musician migrate their entire workflow. Prioritize:

1. A portable musical-session/intent sidecar bound to source files, project revision, contribution permissions, target obligations and protected invariants.
2. Non-destructive candidate branches with aligned A/B preview, symbolic/audio change summaries and explicit musician commit.
3. A versioned evidence receipt that follows exported/imported audio, MIDI and project handoffs where the receiving format permits it; retain an external signed sidecar when it does not.
4. Capability-aware routing across local symbolic tools, streaming generators and external studios, with no silent downgrade from exact control to prompt-only steering.
5. Per-candidate provenance, renderer/acoustic checks and purpose-specific rights decisions that survive movement between tools.

This positions Melothaea as the **trust, intent and verification layer across the music-production stack**, while leaving generation and final rendering replaceable.

## 14. Public benchmark calendar and matched-scope strategy

The following dates and scopes were checked against the organisers' public pages on 14 November 2026.

- **ICASSP 2027 Audio Editing Challenge / MMAE:** the official page says the public development test set has 2,000 examples and 17,741 atomic rubrics. The hidden challenge set and submission SDK were released 10 November 2026, with separate 500-example Single Model and Agent tracks; registration closed 8 October; submissions close 25 November and final rankings are scheduled for 8 December. Agent components must have publicly released model versions/weights before 1 November. If already enrolled, use the official SDK and report the hidden track independently. Otherwise, reproduce the public development baseline and publish a clearly labelled non-challenge result; do not imply hidden-test access or eligibility.
  - Official source: https://audio-editing-challenge.github.io/
  - Official leaderboard/protocol: https://audio-editing-challenge.github.io/leaderboard/

- **SMEBench:** the 27 October 2026 preprint describes a balanced 3,600-example synthetic test set and a separate 416-example real-audio complement. Its authors report that metric choice can materially change system rankings. Keep synthetic and real-audio results separate; report source preservation, target agreement and textual compatibility as distinct axes, and first qualify the reference/operation rather than treating every paired waveform as infallible.
  - Paper: https://arxiv.org/abs/2610.36715

- **XACLE 2027:** the challenge explicitly targets audio-text evaluation generalization to unseen audio-generation systems. Its public schedule lists test release on 19 November 2026 and submission on 3 December. This is an evaluator-generalization challenge, not a music-editing leaderboard; use it to test whether Melothaea's semantic/audio evaluator survives a backend it did not train on.
  - Official source: https://xacle.org/2027/

- **MIREX 2026 CMI-RewardBench:** the published core result reports 73.9% pairwise human-preference agreement for the leading listed system on 249 core pairs. This is a useful adjacent evaluator-quality reference, not an edit-success score and not statistically interchangeable with MMAE or SMEBench.
  - Official task/results: https://www.music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench

### 14.1 Qualification targets (proposals, not results)

| Track | Reproduction / reference | Melothaea qualification target |
|---|---|---|
| MMAE Agent, same 2,000-example public scope | Reproduce the published baseline: IFR 44.06%, CR 74.63%, EMR 7.45% | First gate: IFR ≥44.06%, CR ≥76.63%, EMR ≥12%; stretch: IFR ≥50%, CR ≥80%, EMR ≥15% |
| MMAE hidden Agent track | Official 500-example track with private rubrics and frozen model release eligibility | Only report if enrolled and eligible; target top-quartile or better with full reproducibility evidence, not a proxy score on development data |
| SMEBench synthetic | 3,600 paired synthetic cases | ≥95% target-property success on supported held-out operations, with preservation and text compatibility reported separately |
| SMEBench real-audio complement | 416 real-audio cases | Separate operation-stratified report; no claim of real-audio generalization from synthetic results alone |
| Hard integrity/protected-content controls | Frozen byte, lineage, rights, stale-receipt and protected-invariant negatives | Zero observed false Accepts; additionally publish confidence bounds. For example, 0/3,000 independent negatives gives an approximate 95% upper bound of 0.1% (rule-of-three approximation) |
| Counterfactual probing | Exhaustive-render oracle on a bounded development subset | Pair recall ≥0.95, precision ≥0.80 at ≤30% of exhaustive render cost; prospective until measured |
| Human validity calibration | Blinded, signed multi-reference adjudication | ≥90% acceptance of independently validated legitimate alternatives and ≤1% acceptance of invalid references, with panel disagreement and confidence intervals reported |
| Cross-backend/renderer generalization | Held-out renderer, soundfont, backend version and audio modality | No silent capability downgrade; report absolute scores and per-domain degradation, with a proposed stretch ceiling of 5 percentage points on matched supported tasks |
| Reciprocal interaction | RMI-Bench: contingent interaction versus yoked replay, fixed-response and imitation controls | Demonstrate significant partner-intervention contingency and low false contingency under yokes; zero unapproved commits; evaluate human agency/ownership separately from artifact preference |
| Efficiency | Full regeneration, greedy repair, fixed-probe and single-backend baselines at matched hard-gate success | ≥30% lower render/compute cost per verified acceptance without increasing false acceptance, collateral changes or hidden abstentions |

The numerical thresholds above are research targets. Their use in a public claim requires frozen datasets, matched task scope, versioned model/backend configurations, pre-registered evaluators, confidence intervals and independent reproduction.

### 14.2 Reporting rule

Publish a capability × operation × modality × renderer matrix. Report failures, partial outputs, unknown outcomes, abstentions and budget exhaustion in the denominator. A single aggregate score must not conceal a backend that performs well on easy generation but cannot execute a hard preservation obligation.
