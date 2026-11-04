# MEL-RECOVERY-001 — Veto during pending musical response: deterministic race protocol v1

**Date:** 4 November 2026  
**Status:** research/engineering contract; no Rust execution, CI, device, DAC, loopback or musician result is claimed.  
**Parent programmes:** [#10390 — Timing, Recovery and Grounded Outcomes](https://github.com/Luminous-Dynamics/symthaea/issues/10390), [#10382 — Decision-to-effect adapter](https://github.com/Luminous-Dynamics/symthaea/issues/10382).

## 1. Purpose

Qualify the smallest high-consequence recovery transition in live musical participation:

> A collaborator grants a bounded invitation, Melothaea begins preparing a response, and the collaborator vetoes that response before all of its output has passed through the application queue.

This experiment tests whether an authorization is invalidated across producer rendering, application enqueue, callback admission and later output observation, while preserving an exact account of work that already crossed an irreversible boundary. It deliberately does not attempt to qualify every kind of musical recovery at once.

## 2. Existing owners: compose, do not duplicate

| Responsibility | Existing owner | Required composition |
| --- | --- | --- |
| Streaming WAIT / RESPOND / CONTINUE / YIELD trace | #10309 | Exact session, turn, input prefix, decision time and receipt |
| Note-source immediate invalidation | #10311 | Separate unconsumed notes from already emitted notes |
| Application-queued PCM generation fence | #10327 | Monotonic output epoch, bounded stale scan and callback admission boundary |
| CPAL playback-time to output-frame mapping | #10345 | Stream/configuration-bound fit; point-estimate-only until uncertainty is qualified |
| Runner monotonic to CPAL clock mapping | #10391 / #10381 | Same runner origin as decision/effect receipts |
| Selective act/withhold policy | #10408 / #10402 | Preserve WAIT, ASK and ABSTAIN as distinct dispositions |
| Grounded action authorization | #10418 / #10409 | Exact artifact, prefix-bounded evidence, action predicate, authorization ID and expiry |
| Decision-to-effect event ledger | #10382 | Append-only event sequence and separate downstream statuses |
| System-level recovery panel | #10390 | Timing, recovery, grounded outcome and conditional multiparty reporting |

The callback remains real-time safe: no allocation, lock acquisition, blocking channel operation, filesystem access, logging or unbounded queue scan. Control and ledger operations stay off the callback.

## 3. Vocabulary and the irreversibility boundary

For every output frame or frame range, preserve these distinct facts:

1. **Rendered:** synthesis created samples under a specific decision, authorization, plan digest and captured output epoch.
2. **Application-enqueued:** samples entered the application-owned output queue.
3. **Stale-discarded:** the callback removed samples because their epoch was no longer current.
4. **Callback-admitted:** a frame passed the callback's current-epoch admission check.
5. **CPAL-predicted:** the backend supplied a predicted playback time for the callback buffer.
6. **Loopback-observed:** an independent, calibrated electrical/acoustic capture observed output.

A request, renderer completion, application enqueue, callback admission, backend prediction and physical observation are not interchangeable forms of success.

### 3.1 Admission linearization point

For the application-queue guarantee, define the callback's atomic epoch read/check as the **admission linearization point** for each frame. A frame that reads the old epoch immediately before a concurrent invalidation may be admitted and written afterward. The epoch fence cannot recall it.

The strongest software claim is therefore:

> No frame tagged with a revoked epoch is admitted by a callback whose admission check observes the new epoch.

Do not claim that no sound can occur after the collaborator presses veto, after the control thread receives the message, or after a local UI event. A frame may already have passed callback admission, and samples may already be in downstream OS/device buffers. Count callback-admitted/in-flight frames separately from stale frames rejected after the fence. Never relabel a pre-fence admission as cancelled.

## 4. State carried by the experiment

The replay fixture and effect ledger bind, at minimum:

- session, turn and participant/authority identity;
- monotonically ordered control-event sequence;
- exact input artifact digest, sample rate, channels, total frames and permitted-prefix end;
- evidence-bundle digest and grounding/annotation policy versions;
- decision ID, decision sequence, selected action and action-policy version;
- authorization ID, scope, not-before time, expiry and runner clock-domain/origin identity;
- response plan/renderer digest and rendered artifact/frame ranges;
- output stream instance/configuration identity and current output epoch;
- producer snapshot epoch and exact authorization/plan snapshot used to render;
- queue enqueue order and application-queue disposition;
- callback admission order, epoch observed and frame range;
- CPAL callback/playback anchors and fit-bundle digest, when available;
- independent loopback artifact digest, calibration identity and uncertainty, when available;
- explicit unresolved status for absent or mismatched evidence.

A device fingerprint is not an anonymization guarantee. Device-specific traces need an explicit privacy and publication policy.

## 5. Normative veto protocol

### 5.1 Authorize a bounded response

1. Accept the exact audio artifact and permitted input prefix.
2. Validate the temporal-grounding receipt and action-specific RESPOND predicate. Confidence or a fluent explanation alone is insufficient.
3. Verify that no valid veto, stop, expiry or superseding commitment exists for the target turn.
4. Create one authorization ID bound to the decision receipt, response plan digest, effect window, session/turn and current control sequence.
5. Capture the current output epoch together with authorization and plan identity on the producer/control side.
6. Render and enqueue only frames tagged with that captured epoch and bound to that authorization. A producer may not read the current epoch later and use it to relabel an already-rendered response.

### 5.2 Accept a veto

1. Validate the veto source and its authority for this session/turn. If authority is ambiguous, fail closed and ask for clarification rather than resolving the conflict by loudness, recency or model preference.
2. Serialize the veto with authorization and output-control commands.
3. Mark the authorization revoked and advance the output epoch monotonically. Epoch exhaustion or a failed control operation must fail closed; it must never wrap or reuse an earlier epoch.
4. Bind the veto event to the old authorization ID, prior epoch, new epoch, participant authority and control sequence.
5. Make all later producer output use a new authorization/epoch. The revoked response may not be silently retagged or replayed.
6. Persist/replay the control event and its effect observations. If durable ledger persistence fails after the output fence has advanced, keep the output fenced and report the ledger gap as unresolved; do not restore the revoked authorization.

### 5.3 Callback behavior

For each output frame:
- admit only if its captured epoch equals the current epoch at the atomic admission check;
- otherwise discard it under the existing bounded stale-drain policy;
- count stale discard separately from underrun/zero-fill;
- count any pre-fence callback admission as admitted/in-flight;
- never block waiting for a producer or control-plane lock.

If the stale-scan budget is exhausted, continue bounded draining on a later callback. Do not convert a stale queue into a false claim of immediate physical silence.

## 6. Deterministic adversarial schedule

The primary fixture pauses the producer between epoch capture and queue publication.

| Step | Actor | Event | Expected evidence |
| --- | --- | --- | --- |
| 0 | Controller | Start session/turn with stream S and epoch E | Session, clock origin and stream descriptor agree |
| 1 | Evidence adapter | Supply grounded invitation for input prefix P | Exact artifact/prefix and evidence bundle validate |
| 2 | Decision adapter | Commit RESPOND decision N and bounded authorization A | Decision and authorization receipts bind same turn and plan |
| 3 | Producer | Capture (A, plan digest, E), render, then pause before enqueue | Producer snapshot receipt records E and A |
| 4 | Control plane | Validate collaborator veto V at sequence N+1 | Veto authority and target turn validate |
| 5 | Control plane | Revoke A and advance E to E+1 | One ordered control transition; new epoch receipt |
| 6 | Producer | Resume and enqueue old rendered frames tagged E | Frames retain E and A; no retagging |
| 7 | Callback | Drain old queued frames after observing E+1 | Stale discard; zero callback admission for those frames |
| 8 | Decision adapter | Attempt to reuse N/A | Rejected as revoked/stale |
| 9 | Collaborator | Issue a new explicit invitation | A new decision/authorization/epoch is required |
| 10 | Producer/callback | Produce a response to the new invitation | Only new authorization and current-epoch frames are admissible |

### 6.1 Required race interleavings

- **Veto-before-admission:** callback checks the frame after E+1 is visible. The old frame must be discarded.
- **Admission-before-veto:** callback checks E and pauses; veto advances to E+1 before the callback writes the already-admitted frame. Count this as pre-fence admitted/in-flight; do not claim recall.
- **Veto-between-render-and-enqueue:** producer resumes after E+1 and enqueues E-tagged frames. A later callback must reject them.
- **Concurrent callback and invalidation:** the observed atomic order decides whether the frame is pre-fence admitted or post-fence stale. Every frame belongs to exactly one category.
- **New invitation after veto:** a new authorization may proceed only under the new control sequence and a non-reused epoch. Old producer work remains stale.
- **Veto while authorization is being committed:** serialized control order must yield either a completed authorization subsequently revoked or a rejected authorization. No untracked intermediate authority is allowed.

## 7. Invariants and falsifiers

| ID | Invariant | Falsifier |
| --- | --- | --- |
| VETO-01 | A valid veto permanently invalidates the targeted authorization for the turn | Revoked authorization admits a new producer effect |
| VETO-02 | A frame retains the epoch/authorization snapshot captured before rendering | Producer relabels old frames with the new epoch |
| VETO-03 | A callback that observes E+1 does not admit a frame tagged E | E-tagged frame is admitted after the E+1 check |
| VETO-04 | A frame admitted before E+1 remains explicitly accounted as in-flight | Pre-fence frame is reported as recalled/cancelled |
| VETO-05 | Epochs are monotonic, non-reused and fail closed at exhaustion | Wraparound, ABA reuse or fallback epoch |
| VETO-06 | Veto ordering is independent of model producer message order | Delayed model command overrides a later valid veto |
| VETO-07 | Identical duplicate event IDs are idempotent; conflicting payloads reject | Conflicting replay mutates the accepted event |
| VETO-08 | Queue discard, callback underrun and callback admission are separate | One counter is substituted for another |
| VETO-09 | Every attempted frame has a terminal or unresolved accounting disposition | A missing frame range disappears from the denominator |
| VETO-10 | Callback work remains bounded and non-blocking | Lock, allocation, log, wait or unbounded scan in callback |
| VETO-11 | Missing/expired clock fit prevents qualified timing conversion | Point estimate is presented as calibrated physical timing |
| VETO-12 | A new invitation cannot authorize replay of the revoked fragment | Old rendered material reappears without a new plan |

Fail closed on missing authority, mismatched artifact/turn, expired authorization, invalid clock identity, sequence regression, conflicting duplicate, unavailable evidence or incomplete event accounting.

## 8. Controls and later variants

The minimal deterministic harness includes:
1. Always-respond producer control (must fail veto cases).
2. Always-withhold control (must fail valid invitation cases).
3. No-epoch/delayed-invalidation control (must expose stale output admission).
4. Reset-and-replay control (must expose duplicate/replayed fragments).
5. Correct epoch fence and serialized-veto treatment.
6. Yoked/stale authorization from another session/turn (must fail identity binding).
7. Conflicting duplicate veto and out-of-order control event.
8. Stale-prefix, unavailable-stream and expired-window cases.

After this vertical slice passes offline, add separate recovery strata: tempo/rubato/meter change during a pending forecast; phrase-internal breath that must not become a handoff; dropout/restart and clock-segment invalidation; role reassignment and multiparty authority conflict; renderer failure, underrun and full anchor-ring evidence loss.

## 9. Metrics

Report exact numerators, denominators and uncertainty intervals:

- veto validation → control-plane linearization time;
- control-plane linearization → epoch-advance observation;
- producer render completion and enqueue delay;
- old-epoch frames rendered, enqueued and stale-discarded;
- callback-admitted/in-flight frame count and frame interval;
- callback stale-scan work and budget exhaustion count;
- underrun/zero-fill frames;
- rejected effects after veto by authorization and sequence;
- duplicate/replayed fragment rate;
- time to a newly authorized valid continuation, when a new invitation exists;
- CPAL-predicted presentation interval;
- independently loopback-observed veto-to-output tail, only with calibrated capture.

For timing, report p50/p95/p99 stratified by device, stream/session, callback size, action and intervention type. Do not compute device-frame latency from incompatible or expired clock fits. Do not substitute CPAL prediction for independent loopback.

## 10. Claim ladder and acceptance gates

| Level | Evidence required | Permitted claim |
| --- | --- | --- |
| L0 | Frozen schedule and schemas validate | Fixture contract is well-formed |
| L1 | Deterministic reducer and adversarial schedule tests execute | Offline software state-machine behavior is qualified |
| L2 | Rust module compiles/tests on exact PR head; independent replay | Exact-head implementation behavior is qualified |
| L3 | Physical CPAL trace with exact stream descriptor and clock-fit receipts | Application/backend timing observations are available; not physical acoustic cancellation |
| L4 | Independently calibrated electrical/acoustic loopback | Measured output tail and physical effect timing for the tested setup |
| L5 | Blinded musician pilot with preregistered protocol | Human collaboration, agency and recovery outcomes for the sampled cohort |

A claim must not skip levels. A synthetic fixture, source inspection, branch mergeability or CI job marked skipped is not a Rust execution result. A device prediction is not loopback. A loopback result is not evidence of human musical usefulness.

Initial software acceptance targets:
- 100% exact event accounting in the synthetic fixture;
- zero post-fence admission of old-epoch frames;
- zero authorization reuse after veto;
- 100% capture of pre-fence admitted/in-flight frames as such;
- zero epoch reuse/wrap;
- deterministic replay equality of the event ledger;
- no regression of the callback bounded-work/non-blocking contract.

These are targets to be tested, not achieved results.

## 11. Next integration sequence

1. Reconcile existing PR heads with current main without weakening #10327, #10345 or #10391 guarantees.
2. Implement the deterministic race harness against the epoch primitive before wiring the live runner.
3. Add the append-only event model to the #10382 adapter, consuming exact receipts from #10309/#10408/#10418.
4. Connect live producer and callback receipts while keeping callback work bounded.
5. Run exact-head Rust formatting, tests, Clippy and feature-matrix checks; preserve queued != pass.
6. Only then collect physical CPAL traces, independent loopback and musician-pilot evidence.

## 12. Explicit non-claims

This protocol does not establish artistic quality, consciousness, subjective experience, human-equivalent musical understanding, universal timing precision, physical silence, legal authorship or causal influence merely because an agent's output changes. Reciprocal causal influence (#10088), expressive adaptation (#10210), forecast transparency (#10190), multiparty agency (#10249) and longitudinal non-assimilation (#10133) remain separately measured programmes.

The target is narrower and more useful: a musical participant whose contributions can be revoked at a known software boundary, whose pre-boundary effects remain visible, and whose recovery can be independently replayed and audited.
