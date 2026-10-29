# MEL-MR-EVAL-013 — Render Execution Receipts and Performed-Time Maps

**Status:** research/design RFC; no renderer attestation or benchmark gain claimed  
**Tracking:** [Issue #9026](https://github.com/Luminous-Dynamics/symthaea/issues/9026)  
**Builds on:** [MEL-MR-EVAL-010](https://github.com/Luminous-Dynamics/symthaea/pull/8989), [MEL-MR-EVAL-011](https://github.com/Luminous-Dynamics/symthaea/pull/9008), and [MEL-MR-EVAL-012](https://github.com/Luminous-Dynamics/symthaea/pull/9025)

## Executive finding

The current contract-bound full-Score API is a meaningful improvement: it checks instruction, source/target Score, preservation-profile and renderer-manifest digests; binds pitch edits to the target MIDI note and transposition to the requested delta; enforces profile-backed tolerances; and replays the evidence envelope.

It still does not prove that a renderer actually consumed the target artifact. Also, the manifest's constant `render_latency_frames` is insufficient to represent a score that underwent expressive timing, swing, strumming, humanization, or a different performed-MIDI export path.

**The next paradigm shift is to bind evidence to the performed event timeline and the actual render invocation, not just to the abstract score and a renderer description.**

This RFC deliberately separates local execution evidence from cryptographic attestation. A trusted local adapter can record which bytes it actually passed to which process and which bytes it read back; that is stronger than a caller-supplied claim. It is not a guarantee against a malicious host or a malicious renderer that deliberately ignores its input. Those stronger claims need a separate attestation design.

## Source audit: why the timing boundary matters

Muse currently has at least two relevant render paths:

- The theory realization path in [`theory_realize.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/research/mel-mr-eval-012-score-contract-binding-20261029/crates/domains/symthaea-muse/src/theory_realize.rs) creates a `Timeline` from `Rubato::from_score_with_dialect` and swing. The score-to-seconds map is therefore not simply `beat * 60 / tempo`. The realization pipeline generates performance voices, including timing/articulation transformations, renders an arrangement, then applies mastering.
- The optional [`fluid_render.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/research/mel-mr-eval-012-score-contract-binding-20261029/crates/domains/symthaea-muse/src/fluid_render.rs) path consumes a performed MIDI file through a locally discovered FluidSynth executable and SoundFont, then reads and masters the resulting WAV.

Those paths have different inputs, assets, timing transformations and post-processing. They are not interchangeable under one generic renderer ID. The current MEL-MR-EVAL-008 full-Score mask derives nominal note intervals from beat positions, sample rate, tempo and one constant frame offset. That is appropriate only where the actual execution path obeys that timing contract. Do not claim the current mask is calibrated for the expressive realization or external MIDI path until an explicit performed-time map is available.

## 1. Chain of custody

Represent the pipeline as a sequence of artifacts with domain-separated identities:

1. **Source Score:** exact serialized Score hash.
2. **Target Score:** exact serialized target Score hash.
3. **Frozen edit contract:** request/instruction digest, supported operation, target event, required invariants, approved profile digest and contract hash.
4. **Performed-event representation:** an explicit ancestry map from each relevant Score event to the performed event(s) it generated after all timing and articulation transformations.
5. **Renderer input:** exact MIDI or other byte sequence delivered to the backend, hashed after serialization and immediately before invocation.
6. **Execution configuration:** renderer executable identity/version, arguments, sample rate, channels, instrument bank/SoundFont digest, effects and mastering configuration, timing transform version and any declared environment inputs.
7. **Output artifact:** exact output file/WAV byte hash plus a distinct canonical PCM digest and decoded format metadata.
8. **Independent audit:** recompute operation-conditioned checks using the performed-event map, source/target audio, exact output and the frozen profile.

A receipt should bind adjacent artifacts explicitly. A chain of independent hashes without the edges between them is insufficient: a target Score hash and an unrelated output WAV hash do not establish that the WAV was rendered from that Score.

## 2. Proposed data contracts

### PerformedEventMap

The map should contain one record per logical source event and its realized descendants:

- deterministic Score-event identity, defined relative to the full Score digest and a stable event identity scheme;
- zero, one, or many performed event IDs, with a declared reason for a zero/many mapping;
- target MIDI pitch, part/role, instrument voice and note attributes after realization;
- performed onset and end time in seconds and frame coordinates;
- pre-humanization and post-humanization times when both exist and are materially different;
- timing-transform chain and version: dialect, rubato/swing, seed, strum/humanization policy, sample rate and rounding rules;
- mapping certainty / unresolved status—ambiguous mappings must abstain.

Do not map events solely by nearest onset or pitch. Repeated notes and repeated motifs are valid counterexamples. Event ancestry must be preserved during symbolic-to-performed transformation, or independently established by a deterministic alignment with explicit ambiguity rejection.

### RendererExecutionReceipt

The adapter-generated record should include:

- version and domain-separated receipt digest;
- the edit-contract digest and exact source/target Score digests;
- performed-event-map digest;
- exact renderer input artifact type, byte hash, format and size;
- renderer executable digest and reported version where obtainable;
- instrument bank/SoundFont, effect-chain and mastering configuration digests;
- explicit effective arguments and allowlisted environment inputs—not arbitrary secrets or the full process environment;
- process exit status, timeout/error status, output format and frame count;
- exact output-file hash and canonical decoded-PCM hash;
- deterministic run seed / relevant parameters;
- limitations and trust level: in-process call, local subprocess, remote signed attestation, or unknown.

The constructor should live inside the trusted adapter path. A free-standing caller should not be able to assert successful execution simply by supplying a manifest. Nevertheless, any serialized receipt must be revalidated; an unkeyed digest is not a signature and does not authenticate the adapter.

## 3. Correctly deriving masks

The edit gate must use the performed-time map for the actual rendering pathway.

For a pitch-only edit, the changed pitch is allowed to alter spectrum in the relevant performed interval. The note's attack and energy envelope remain required. Protected neighboring events remain audited in their actual output positions. The acoustic mask and onset-protection mask must remain separate.

If timing transformations are deterministic, a score event's expected frame interval should be derived by applying the recorded transform chain and the same frame-rounding rules used for rendering. If the system cannot map a target event or transformation to output time with justified certainty, it must abstain instead of widening the mask.

Effects and mastering can produce energy beyond a note's nominal end. Model those tails conservatively using a profile-backed boundary guard or an explicitly measured impulse/tail model. Do not treat a caller-selected wide mask as permission to ignore arbitrary spillover.

## 4. Adapter architecture

Prefer two narrowly typed adapters instead of one flag-heavy generic renderer:

### Native score realization

- Receive the exact target Score and a frozen realization specification.
- Use a realization API that exposes performed-event ancestry alongside the audio result; do not reconstruct ancestry after the fact from a flattened note list.
- Compute the input Score/spec/seed/timing-transform identity within the adapter.
- Capture the actual output from the native renderer and mastering chain.
- Return audio plus a locally generated execution record and performed-event map.

### Performed MIDI / external renderer

- Accept exact MIDI bytes or create them from the tracked performed-event representation.
- Write the bytes to a unique, adapter-controlled temporary location; hash them after writing and before process invocation.
- Resolve the exact executable and SoundFont, hash both, record effective arguments and explicitly selected settings.
- Invoke the renderer, check exit status, read the exact output bytes, parse format metadata and hash output and decoded PCM.
- Capture post-processing separately. Never silently switch to a fallback renderer under the original manifest; a fallback creates a different execution record.
- Clean up safely, and test temporary-file collision and stale-output attacks.

A local adapter receipt establishes observed local invocation and returned bytes under the trusted-adapter assumption. If the threat model includes a hostile renderer or host, add a separately reviewed signed/remote attestation mechanism; do not overstate what local hashes prove.

## 5. Adversarial and conformance tests

Minimum independent mutation families:

- swap the target Score while retaining the old performed MIDI;
- substitute MIDI bytes after the input digest was computed;
- substitute renderer binary, version, SoundFont, effect or mastering configuration;
- report the right manifest but return a prior invocation's WAV;
- change the output file or decoded PCM after receipt creation;
- change swing/rubato/dialect/seed while retaining an old event map;
- use a constant latency offset for a transformed timeline;
- delete, delay, duplicate or attenuate the edited target note;
- alter a neighboring note outside the true performed interval;
- expand the authorization mask to cover protected material;
- truncate or change sample rate/channels while preserving a misleading score hash;
- create ambiguous repeated-note mappings or duplicate event IDs;
- cause process failure, timeout, missing executable/SoundFont or renderer fallback.

Positive controls should include exact deterministic render replay, a pitch edit with preserved attack/envelope, supported transposition, and a candidate that deliberately fails the envelope check.

Acceptance requires the exact intended artifact chain and expected verdict, not simply that receipts can be generated. A receipt failure, missing event map or unsupported renderer path must reject or abstain.

## 6. Benchmark plan and targets

Keep public MMAE IFR/CR/EMR separate from musical preservation. Run the same task IDs, source assets, editor, seeds, evaluator and tool budgets across:

A. current score/audio binding without execution receipt;
B. adapter-generated render receipt only;
C. performed-event map only;
D. receipt + event map + operation-conditioned audio gate;
E. D plus Symthaea diagnostic selection/repair.

Report:

- task-level paired EMR and confidence interval on compatible public MMAE tasks;
- critical false accepts by renderer substitution, timeline drift and target-deletion family;
- event-map coverage, ambiguity abstention, unsupported renderer coverage and output replay success;
- note-onset and interval mapping error against known-truth synthetic fixtures;
- protected-region leakage and target-change/envelope success;
- p50/p95 adapter overhead, wall time, render cost and repeatability;
- performance split by native renderer versus external FluidSynth, instrument bank, dialect, and transformation type.

Initial goals, not results:

1. zero false accepts in all frozen deterministic critical-negative controls;
2. at least 384 independent critical negatives with zero false accepts before claiming an approximate two-sided 95% Wilson upper bound below 1%;
3. exact replay of every artifact edge and the performed event map for supported deterministic fixtures;
4. 100% fail-closed behavior for missing/ambiguous maps and unsupported render paths;
5. quantify adapter overhead before setting a final latency ceiling;
6. the renderer receipt must not worsen EMR or critical false acceptance compared with the same deterministic audio gate without the receipt.

## 7. Integration order

1. Add a test-only performed-event lineage representation to a small deterministic realization fixture. Prove the mapping across rubato, swing, strum and humanization without introducing a production renderer dependency.
2. Expose the event map at the point where score events become performed notes. Preserve ancestry through one-to-many voice realization; avoid guessing it later.
3. Build the native in-process adapter receipt first, because its execution boundary is easiest to control.
4. Add the external MIDI/FluidSynth adapter as a distinct backend with explicit binary/assets/args identity and no silent fallback.
5. Derive operation masks from performed intervals and add mutation fixtures for timing-map drift and artifact substitution.
6. Add Symthaea planning/diagnostic selection only after the event map and receipt have passed deterministic conformance. The cognitive planner can choose which checks to run first; it cannot rewrite timing maps, widen masks or certify its own output.

## Decision rule

Promote a renderer path only when its adapter records exact inputs and outputs, the performed-event map is replayable, all critical negative controls fail closed, and independent audit reproduces the expected verdict. Do not advertise broad real-renderer attestation until the hostile-host boundary has been addressed.

**No renderer-attestation implementation, Rust test execution, benchmark run, or SOTA claim is made by this RFC.** It identifies the next trust boundary and defines the data and experiments needed to implement it honestly.
