# MEL-MR-EVAL-005 — Symbolic-to-audio edit fidelity and locality

**State:** protocol draft / preregistration scaffold. No audio benchmark results are claimed by this document.
**Tracks:** follows MEL-MR-EVAL-004 (symbolic edit locality and content-bound receipts), but is a separate cross-modal evaluation. Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8799

## Why this is the next leap

A symbolic verifier can prove that a MIDI-like event sequence contains the requested change and preserves specified event properties. It cannot prove that a render or audio-editing model sounds correct, preserves instruments/timbre, avoids clicks at edit boundaries, or leaves other sources intact.

Recent work makes the missing interface clear:
- The 2026 audio-editing survey separates acoustic, semantic and instance edits and explicitly calls for annotations of targets, operations and non-target preservation regions: https://david-pigeon.github.io/audioeditsurvey_project/
- TimingBench represents events and target times as machine-readable schemas and calibrates its detector's localization error against human annotation. Its authors warn against treating scores finer than the measurement floor as meaningful: https://taan.ai/research/rushing-or-dragging/
- The public audio-editing benchmark catalog describes MMAE (2,000 cases / 17,741 rubrics), ZoME-Bench (1,100 music edit cases), and MelodiaEdit (2,015 edit pairs). Their reported metric trade-offs demonstrate why target alignment, preservation and sound quality must be reported separately, not collapsed into one number: https://github.com/MM-Speech/AudioEditSurvey
- The 2026 music-generation survey concludes that no single metric covers all properties and distinguishes symbolic properties from audio/perceptual properties: https://link.springer.com/article/10.1007/s10462-026-11582-x

These benchmarks are comparison points, not interchangeable datasets. Their task scope, data source, evaluator, and score definitions must be retained verbatim when results are compared.

## Unit of evaluation

One case describes an exact source audio asset, its symbolic representation, a declared edit, an expected symbolic target, target audio regions, protected regions, a fixed reference renderer configuration, and versioned property oracles. The evaluated system receives the source and instruction; its output is scored without changing thresholds after seeing test results.

The schema `mel-crossmodal-edit-case-schema-v1.json` defines asset digests, licensing/provenance, renderer parameters, region annotations, operation family, critical failure modes, and annotation state. Assets are referenced by relative paths and SHA-256 digests; the schema itself does not certify that the files exist, their licences are valid, or their audio matches the symbolic score. Those are ingest-gate responsibilities.

### v1 is deliberately constrained

Start with cases where a symbolic target is explicit and independently checkable: transposition/register, local pitch, onset/duration, and rhythmic edits. Treat source/instrument/texture/arrangement edits as separate strata that enter scored evaluation only when the expected target and the right preservation property can be independently annotated. Do not quietly treat a missing oracle as success; unsupported cases are reported as unsupported/abstained.

## Evaluation protocol

### A. Before scoring

1. Verify every file digest and codec header; reject absolute paths, POSIX traversal, backslash paths, and missing assets.
2. Enforce cross-field invariants that JSON Schema cannot express alone: `edit.target_symbolic_sha256 == assets.expected_target_symbolic.sha256`; every region must satisfy `0 <= start_seconds < end_seconds <= source_audio.duration_seconds`; target/protected masks must follow the case's declared overlap policy; renderer configuration digest must identify the exact config bytes.
3. Verify license, attribution, source lineage, renderer name/version/config digest, sample rate, channels, gain/loudness policy, and exact preprocessing.
3. Deduplicate source recordings, performances, MIDI variants and near-duplicates before splitting.
4. Group splits by source family, composition, performer/recording lineage, and arrangement where applicable. No closely related arrangement may cross dev/calibration/test.
5. Freeze detector versions, thresholds and target-region rules on development/calibration data. Publish a preregistration digest before test evaluation.

### B. Independent score channels

**1. Symbolic target correctness.** Apply the MEL-MR-EVAL-004 verifier to source/target symbolic events when that representation is valid. Report requested-property pass, protected-property pass, and unsupported cases separately.

**2. Audio target-change success.** Score the requested property within its declared target region: for pitch/register use a fixed, versioned pitch-tracking pipeline with voiced/unvoiced confidence and a human-calibrated error floor; for rhythm/onsets use event-localization with per-family calibration and missed-event controls; for instrument/source changes use a validated, versioned source-aware detector plus human adjudication where the detector is not authoritative. Report detector failures as unscorable, not automatically as success.

**3. Audio preservation/locality.** Evaluate protected non-target regions and tracks separately from the target. Depending on task, report chroma/interval consistency, onset/rhythm preservation, source/stem similarity, loudness and stereo changes, plus a fixed boundary-artifact metric around the edit seam. No single embedding distance proves preservation. Run unchanged/no-op and wrong-region negative controls to reveal detectors that mistake general activity for the instructed event.

**4. Human evaluation.** Randomize paired outputs and hide system identity. Ask separate questions for (a) requested edit achieved, (b) non-target content preserved, (c) audible artifacts, and (d) overall musical quality/usefulness. Use multiple independent raters, adjudicate only under a preregistered procedure, and report agreement and source-family bootstrap confidence intervals. Preference is not a substitute for exact symbolic correctness; symbolic correctness is not a substitute for preference.

**5. Operational cost.** Report p50/p95 wall-clock latency, real-time factor, model/renderer versions, compute/device, retries, failures, abstentions and edit iteration count. Compare systems at matched input, output duration and compute budget; report best-of-N separately from single-shot results.

### C. Joint success without score laundering

Publish every metric and per-family confusion matrix. Define *joint edit success* only if:
- the requested target property passes;
- all critical protected properties pass;
- no predeclared critical failure is detected; and
- the case was in scope and scored by an evaluator with sufficient calibrated measurement precision.

Quality and listener preference remain separate axes. Unsupported or unscorable cases never count as passes. Do not average a catastrophic preservation failure away with high instruction adherence.

## Controls and adversarial cases

Required controls include:
- unchanged-source and explicit no-op cases;
- instruction omitted (target edit absent);
- edit performed at the wrong timestamp or wrong instrument;
- correct target edit plus leakage into one or more protected regions;
- pitch/rhythm changed together when only one axis was authorized;
- event-density distractors around the requested timestamp;
- boundary clicks, discontinuities, phase/coherence shifts and gain jumps;
- codec, sample-rate conversion, loudness normalization and trim-offset controls;
- detector failure, low confidence, unsupported operation, malformed/missing asset, hash mismatch and license-pending controls.

For onset/timing scores, measure detector localization error against human annotation per operation family. The allowed timing tolerance must exceed the evaluator's justified uncertainty floor; otherwise mark the case unscorable rather than awarding precision the apparatus does not support.

## Baselines

1. **Oracle transform + fixed renderer:** apply the expected edit directly to symbolic score and render through the declared reference renderer. This estimates the quality ceiling and the renderer's own artifacts; it is not a learned-system competitor.
2. **Conventional edit pipeline:** deterministic MIDI/DAW operation, region masks, crossfade rules, and fixed render chain.
3. **Symthaea / Melothaea candidate:** identical case input and budget, recording every edit and verification attempt.
4. **Open and commercial editors:** include only systems that support the operation and provide auditable input/output records. Publish unavailable features and scope limits; do not compare incompatible tasks as if equivalent.

Use the same source clip and instruction across systems. Normalize playback level for listener preference comparisons, but retain unnormalized renders and document the normalization. Paired bootstrap over source families is preferable to treating repeated edits of one source as independent examples.

## Proposed gates (targets, not results)

- 100% agreement on deterministic symbolic contract controls once Rust tests are executed.
- Zero false accepts on predeclared critical no-op, wrong-region, leakage and content-drop controls.
- At least 95% target-property success and at most 5% critical protected-property violations on held-out, human-annotated, in-scope edits before making reliability claims.
- Detector measurement error must be materially smaller than the scored tolerance; report error distributions, not just a mean.
- For perceptual results, preregister a powered paired-listener design and publish confidence intervals. Establish a pilot first rather than claiming a preference advantage from a small or selectively sampled set.
- Every reported number must be reproducible from asset hashes, exact configuration, frozen evaluator version and output manifests.

These are ambitious engineering gates, not expected scores inferred from current systems. The pilot should estimate variance and confidence interval width before finalizing human-preference acceptance thresholds.

## Limitations and release gate

This schema is not a dataset, and a protocol is not an empirical result. Do not label MEL-MR-EVAL-005 a completed benchmark until the repo (or a stable dataset host) contains a rights-cleared pilot with exact asset digests, renderer manifest, expected symbolic outputs, human-reviewed target/protected regions, versioned evaluator settings, negative controls, and a script that reproduces the report. No proprietary system should be ranked unless it was tested under a comparable protocol.
