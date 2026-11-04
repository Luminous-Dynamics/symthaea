# MEL-EVIDENCE-001 — Grounded evidence to selective participation v1

**Status:** independent evidence-bundle receipt contract, action authorization validator, four strict schemas, four conformance fixtures, CLI example and 20 authored Rust tests are committed. Rust compilation, test execution, integration with the #10265 temporal-grounding producer, runner prefix isolation and real-audio evaluation remain outstanding.

## Objective

Bind a musical participation decision to independently produced temporal evidence from the exact audio artifact and permitted input prefix.

This module is a **consumer-side authorization bridge**. It does not duplicate or replace the temporal-grounding evaluator in #10265. The temporal-grounding producer emits claims and its exact receipt digests; the bridge checks whether a decision cites claims that are valid, current, artifact-matched and sufficient for the requested action.

The path is:

`exact audio artifact + permitted prefix
  → independently grounded event/context claims
  → committed evidence bundle
  → action-specific evidence predicate
  → decision receipt + authorized effect window`

## Evidence bundle

The bundle records:
- session, turn and exact audio SHA-256 / sample rate / channel count / frame count;
- temporal-grounding evaluator and annotation-policy versions;
- the permitted input-prefix end frame;
- runner monotonic clock-domain ID and exact origin ID for decision/effect timestamps;
- claims with stable IDs, event/context kind, grounding status, source frame interval, observed-through frame, commit time, source grounding-receipt digest and reason when not grounded;
- SHA-256 of the exact serialized claim vector;
- SHA-256 of the entire evidence-bundle payload (excluding its own digest).

The full-bundle commitment binds metadata as well as claims. Changing the annotation policy, evaluator identity, artifact, prefix or claim vector invalidates the commitment. This prevents a decision receipt from silently being replayed under modified evidence metadata.

V1 uses the Rust `serde_json` serialization of the typed structs for the commitment. It does not claim a cross-language canonical JSON standard. The digest commits the bytes/structure supplied to this consumer; it does not independently authenticate the upstream grounding producer.

## Action-specific evidence predicates

- **RESPOND:** cite a grounded invitation claim.
- **CONTINUE:** cite a grounded active-phrase claim.
- **YIELD:** cite a grounded handoff/stop claim.
- **WAIT:** when policy requires grounding, cite a grounded retained-floor, intentional-space or collaborator-veto claim.
- **ASK:** cite ambiguous grounding; ASK is not treated as a failed attempt to find a grounded event.
- **ABSTAIN:** cite explicit non-grounded evidence (ambiguous, unavailable, contradicted or not estimable) and withhold the contribution.

All cited claims for active interventions and WAIT must be grounded and committed before the decision. ASK/ABSTAIN may cite non-grounded claims, but non-grounded claims require a reason. A grounded stream-unavailable claim requires ASK or ABSTAIN; it cannot be silently re-labelled as intentional WAIT. A valid collaborator veto blocks RESPOND, CONTINUE and YIELD even if an invitation is otherwise grounded.

Active interventions require an effect window in the explicitly identified runner-monotonic clock domain and origin. Its not-before timestamp must be at least the policy's minimum lead after the decision time, its expiry must follow its start, and its duration must not exceed policy. This is deliberately separate from the source-audio sample-frame clock used by evidence intervals. WAIT/ASK/ABSTAIN must not claim an output effect window.

## Temporal and artifact checks

The validator rejects:
- wrong/mismatched session or turn;
- wrong artifact digest or sample format clock fields;
- altered claim-set or full-bundle digest;
- duplicate/missing claim IDs;
- malformed or beyond-artifact claim intervals;
- claims beyond the permitted input prefix;
- claim receipts committed after the bundle or decision;
- evidence older than policy permits;
- missing action-specific grounded claim;
- ambiguous evidence used to justify an active contribution;
- veto or stream-unavailable contradiction;
- runner-clock-origin mismatch, missing or premature effect windows, invalid/overlong effect windows.

A decision must carry the exact full evidence-bundle and claim-set commitments. The bridge checks these relations and action predicates; it does not independently re-read the WAV bytes or prove the live runner exposed only the declared prefix. Runtime input isolation is a separate assurance obligation.

## Conformance fixtures

The checked-in fixture set contains four cases:
1. **Grounded invitation → RESPOND:** accepted with a bounded future effect window in the shared runner clock domain.
2. **Ambiguous invitation evidence → ASK:** accepted without an audio effect.
3. **Unavailable input evidence → ABSTAIN:** accepted as an explicit epistemic withhold.
4. **Grounded invitation + collaborator veto → RESPOND:** rejected due to veto authority.

The test suite also challenges mismatched artifact digests, ambiguous evidence used for RESPOND, prefix leakage, expired claims, premature or overlong effects, runner-clock-origin mismatch, tampered full-bundle metadata and unavailable-stream masquerading as WAIT.

These are controlled conformance cases, not a real audio-localization result or model score.

## Reproduction

Run:

`cargo run -p symthaea-muse --example mel_grounded_participation`

The CLI replays the same checked-in cases used by the Rust unit test and fails if an expected disposition changes.

## Relationship to other programmes

- #10265 owns temporal grounding, MusTBench/TAG-Bench/MusicGroundingBench adapters and multi-occurrence event localization.
- #10402 / PR #10408 owns selective participation, WAIT/ABSTAIN/ASK and paired act/withhold policy evaluation.
- #10393 owns waveform-grounded musical counterfactual fixtures.
- #10307/#10309 own streaming input-prefix and action-trace integrity.
- #10382 owns action/effect authority, veto and downstream execution status.
- #10390 owns timing/recovery/grounded-outcome reporting.

This bridge checks whether the decision cites sufficient grounded evidence. It does not re-score temporal localization or silently combine the benchmarks into one score.

## Qualification boundary

This v1 establishes a replayable, typed authorization contract and synthetic conformance cases. It does not establish that a model can detect invitations from raw audio, that the upstream temporal evidence is accurate, that the runtime enforced the permitted prefix, that audio reached the device, or that the resulting contribution is musically appropriate. Those require independent temporal-grounding results, runner isolation, physical output observation and musician evaluation.
