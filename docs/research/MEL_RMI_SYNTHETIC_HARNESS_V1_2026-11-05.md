# RMI-Bench v1 — synthetic Track A implementation

**Date:** 5 November 2026  
**Protocol:** `crates/domains/symthaea-muse/RMI_BENCH_V1_PREREGISTRATION_2026-11-02.md`  
**Implementation:** `tools/mel-rmi-synthetic/`  
**Pull request:** [#10524](https://github.com/Luminous-Dynamics/symthaea/pull/10524)

## Scope and claim ceiling

This is a deterministic Rust implementation of the preregistered offline Track A benchmark-integrity milestone. It tests whether the evaluator distinguishes specific synthetic policies and whether its event, intervention, content-digest, negotiation, and scoped-authority checks reject declared invalid cases.

It does **not** evaluate the Melothaea model, human musical value, perceived ownership, live performance, hardware timing, commercial model quality, consciousness, legal authorship, or rights. Passing the synthetic controls is not a SOTA result.

## Causal design

Each task uses four paired symbolic observations sharing one task/session/seed/frozen-history identity:

1. **Control:** untreated partner observation.
2. **Targeted:** only the declared target feature changes.
3. **Sham:** the target feature is unchanged.
4. **Unrelated:** a different feature changes while the target feature remains fixed.

The intervention validator rejects event-ID duplication, task/session/history/seed leakage, wrong dimensions, a no-op treatment, a changed sham, and mismatched assignment order. The deterministic Fisher–Yates scheduler binds order to the recorded seed for replay. It is a synthetic fixture scheduler, not an approved randomization procedure for a human study.

Policy arms execute in that recorded order, but output actions are stored in canonical condition slots for paired contrasts. Each arm calls `fork_from_frozen_state` with the same task seed; the API contract requires a fresh policy restored from the same pre-intervention checkpoint. A stateful-counter test checks that each arm starts at ordinal one and receives the same seed. The reverse-direction controlled partner is now independently forked per condition as well, using the same task seed and registered condition order; a stateful-partner test checks fresh per-fork call state, shared seed and invocation order. These traits cannot prove that an arbitrary production adapter obeys the contract; real adapters still need their own isolation tests.

## Baselines and dimensions

The fixed five-feature suite is expanded to **32 deterministic variants per dimension (160 total)**:

- motif transformation;
- rhythmic response;
- tempo drift;
- role exchange;
- texture change.

The policy controls are:

- rule-based positive oracle;
- fixed response;
- imitation-only;
- one-way prompt response;
- cross-session yoked replay;
- unrelated-feature shortcut (input-sensitive for the wrong reason).

The rule-based relation is task-authored synthetic ground truth. It is intentionally not described as general musical complementarity or perceptual equivalence.

## Bidirectional contingency

Partner-to-agent and agent-to-controlled-partner effects remain separate. Both directions report targeted, sham, and unrelated-feature contrasts. The agent-to-partner oracle also checks whether the partner follows the task-declared reverse response relation; a partner that changes in response to the agent for an arbitrary reason does not pass that oracle.

A joint synthetic reciprocal flag requires the partner-to-agent oracle and all reverse-direction gates. This is a controlled simulator result only. A human-partner causal claim requires a separately approved, randomized or otherwise defensible human study; observational performance traces alone cannot support the same causal label.

## Multi-valid response negotiation

`NegotiatedResponseContractV1` represents a set of admissible symbolic alternatives without ranking them. Partner authority is separate: it may contain an explicit selection and a veto set. The agent can commit, branch, ask for clarification, or abstain.

The evaluator separately reports contract validity, veto compliance, explicit-choice compliance, ambiguity preservation, action shape, and qualification. It rejects a premature single choice under unresolved multi-option preference, a vetoed branch, a branch that omits viable alternatives, an invalid response after all alternatives are removed, and branching beyond the declared fan-out budget. An explicit partner selection must be followed; if all choices are vetoed, abstention is the valid disposition.

This is a symbolic interaction/authority contract, not a perceptual-equivalence evaluator. Compose it with [#9307 / PR #9325](https://github.com/Luminous-Dynamics/symthaea/pull/9325) for Score/event-level realization equivalence rather than duplicating that logic here.

The canonical runner has **11 negotiation cases**: six expected qualified outcomes and five expected rejections. The Rust unit suite also enumerates **3,464** small-state dispositions across non-empty accepted-offset sets for cyclic domains of size 2–5, source values, veto subsets, and unresolved/explicit choices.

## Scoped authority-event checks

`PartnerAuthorityEventV1` binds a selection/veto payload structurally to the exact session, task, source observation event, control sequence, monotonic clock origin, effective time, exclusive expiry, and an opaque receipt reference. The current suite has **seven authority-scope cases**: one structurally valid event and six rejected cross-event, stale-sequence, wrong-clock, premature, expired, and missing-reference cases.

This crate does **not** authenticate the actor, verify the receipt signature, or evaluate purpose-specific consent. The receipt reference is opaque. A production adapter must verify actor and authority through the MEL-013A / trusted identity boundary before using this structural evaluator. A digest or opaque string does not prove consent.

## Artifact content evidence

`ArtifactPayloadSetV1` accepts exact byte arrays for a synthetic Score JSON, MIDI, WAV or little-endian f32 PCM representation and the exact serialized transaction-receipt bytes. The pinned `sha2 = 0.10.9` implementation binds SHA-256 and byte length per representation and hashes the receipt bytes. A second call recomputes those digests against the supplied payload sets and rejects changed bytes, missing sets, duplicate formats, or event rebinding.

**Important limitation:** the helper does not access Studio storage, persist artifacts, reopen them, verify receipt signatures, authenticate the actor, validate consent scope, or attest to the verifier binary. The canonical example passes its generated payload arrays back to the verifier; this proves deterministic byte equality for those in-memory synthetic sets only. The helper provides a seam for a real adapter to pass independently re-fetched bytes, but does not itself prove such a read-back occurred.

The `verifier_id` and `verifier_version` fields are caller-supplied labels, not a signed attestation. Hashing a synthetic receipt is not evidence that it originated from Studio or that its signer is trusted. A typed `CommittedArtifactReadbackV1` trait now defines the integration seam: the verifier requests each of the eight expected event IDs (four inputs, four outputs) from a committed-storage adapter, then checks the returned bytes/receipt hashes against the manifest-bound evidence. Tests verify exact event lookup and reject a post-commit byte substitution. The fake read-back provider tests this contract; it does not show that Studio persistence was actually reopened.

The production integration must reuse MEL-013A: resolve Studio-owned artifacts, independently re-read exact bytes after commit, authenticate the transaction receipt and purpose-specific actor authority, and bind the verified transaction into the RMI trial. Do not create a parallel artifact registry.

### Synthetic MIDI semantics (added 6 November 2026)

The canonical example no longer uses marker-only MIDI files. The new renderer writes Standard MIDI File format 0 with one track, a 4/4 time signature, actual note-on/note-off events, and dimension-specific control events. Motif values transpose a fixed melodic contour; rhythm values rotate a note-onset pattern; tempo values produce a tempo meta-event; role values select General MIDI program and register; texture values change chord polyphony. Target and protected features are encoded on separate MIDI channels. An output artifact is generated from the agent action plus the same protected-feature value as its input.

The semantic MIDI tests verify that all five target dimensions change their expected event representation while the protected dimension's note events remain identical; protected tempo is compared as a tempo map, not inferred from note events. These files are actual symbolic MIDI event streams, but they remain procedural fixtures—not expressive performances, audio renders, or human judgments of musical equivalence. The extension raises the authored Rust unit-test count from 64 to 66; the expanded version still needs an exact-head workflow run before any pass claim.

## Executable suite and output

The standalone example writes:

- a trial summary CSV for six policies across 160 fixtures (960 policy trials);
- a four-condition reciprocal summary for 160 rule-based-agent / responsive-partner simulator trials;
- a **4,480-row condition-level event CSV**, with four rows for every policy and reciprocal trial. Each row binds task/session/seed, registered condition order, input/output event IDs, initial/target/unrelated feature values, action value/mechanism, both directional contrasts, and reciprocal outcome where applicable;
- per-dimension descriptive agent and partner summaries;
- an artifact-digest CSV with **17,920 rows** for the synthetic records;
- eleven negotiation rows and seven authority-scope outcomes.

The dedicated workflow pins Rust 1.96.0, stages the crate outside the root workspace, checks formatting, runs the unit tests, executes the example, and enforces these synthetic gates:

| Gate | Expected |
|---|---:|
| Rule-based positive oracle | 160 / 160 |
| Fixed response qualified | 0 / 160 |
| Imitation-only qualified | 0 / 160 |
| One-way prompt qualified | 0 / 160 |
| Yoked replay qualified | 0 / 160 |
| Unrelated-feature shortcut qualified | 0 / 160 |
| Rule-consistent reciprocal simulator | 160 / 160 |
| Negotiation cases | 6 qualified / 5 rejected |
| Scoped authority cases | 1 structurally valid / 6 rejected |
| Synthetic artifact byte-set reverifications | 1,120 |
| Artifact digest rows | 17,920 |
| Agent summaries | 30 (six policies × five dimensions) |
| Partner summaries | 5 dimensions |

These counts are finite fixture-control results, not benchmark accuracy estimates or evidence of generalization. The per-dimension means/standard deviations are descriptive summaries across constructed deterministic variants; no confidence intervals, statistical population inference, or cross-dimension composite score are reported.

## Verification history

- The **42-test** artifact-binding revision passed its exact-head dedicated workflow on commit `618fa4d719b68c6fe0b401c072e7177e2a2cd3ed`, run `42025215684`: 42 passed, 0 failed; the five-dimension positive/negative-control gates passed; 1,120 synthetic byte sets were re-verified; the workflow checked 17,920 digest rows, 30 agent summaries, and five partner summaries. Repository-wide CI, governance, benchmark, and showroom workflows were skipped, not passed.
- The **63-test** revision added multi-valid response negotiation, the 3,464-state finite model check, and scoped authority events; it passed on exact head `95bb88bdf7b41223d564f6af19686e41db93bf3c`.
- The **64-test** revision added symmetric controlled-partner counterfactual forks and passed on exact head `19f4386580a7548b09788094dd6abbb97f394e71`: fresh per-condition partner state, identical common seed, and registered invocation order were verified.
- The event-ledger revision passed the dedicated exact-head workflow on `55ce85fbab8e5477a28540c697a5dd222b8293fd`: 64/64 tests, the existing 160-case causal and interaction/authority gates, 17,920 artifact-digest rows, and the new 4,480-row / 40-column condition-level event ledger with zero malformed rows. The evidence bundle uploaded as artifact `13443490488`.
- These are dedicated synthetic-harness workflow results. Repository-wide CI, governance, benchmark, and showroom workflows remain skipped, not passed.
- No musician pilot, human-partner study, hardware/device loopback, real Studio read-back, receipt-signature verification, or external commercial benchmark run has been performed.

## Next integration steps

1. Implement the MEL-013A adapter against the existing Studio artifact lifecycle—not a parallel store.
2. Fetch the committed source/result artifact bytes from durable Studio storage on an independent read path and bind the actual transaction receipt.
3. Verify the actor/signature and action/purpose scope through a trusted authority provider; unavailable verification must fail closed.
4. Run a sealed symbolic task split with unseen motifs, rhythms, tempo patterns, role changes, and negotiated alternatives.
5. Add actual renderer/backend adapters and log intended action, output artifact, callback admission, predicted playback, and loopback-observed playback separately.
6. Pilot with musicians under approved consent and ethics review; estimate variance/ICC before preregistering power and primary outcomes.
7. Compare production quality against blind instrumental/vocal preferences, editing against native IFR/CR/EMR and preservation measures, and agency against the RMI human study. Never combine these into one score.
