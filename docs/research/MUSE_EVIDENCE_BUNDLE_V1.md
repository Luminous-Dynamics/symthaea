# MUSE Evidence Bundle v1 — cross-layer verification manifest

**Status:** manifest schema, synthetic fixtures and integration contract drafted. No executable bundle verifier or qualification result is claimed yet.

**Tracking:** [MUSE-EVIDENCE-BUNDLE-001 / #12981](https://github.com/Luminous-Dynamics/symthaea/issues/12981)

**Inputs under integration:**
- [PR #12927 — Render Execution Receipt v1](https://github.com/Luminous-Dynamics/symthaea/pull/12927)
- [PR #12951 — Paired Edit Evidence Benchmark v1](https://github.com/Luminous-Dynamics/symthaea/pull/12951)
- [PR #12977 — WAV Gain-Envelope Observer v1](https://github.com/Luminous-Dynamics/symthaea/pull/12977)
- [Issue #11389 — Independent Score-to-Audio Conformance](https://github.com/Luminous-Dynamics/symthaea/issues/11389)
- [Issue #12971 — Calibrated Acoustic and Human Observers](https://github.com/Luminous-Dynamics/symthaea/issues/12971)

## Paradigm shift

Move from a single render receipt, a single reward score or a single WAV hash to a **typed evidence graph**. Each producer makes a bounded claim and seals its own result. The bundle verifier independently validates the component receipt, confirms exact artifact/case bindings, then derives an aggregate outcome without letting one layer compensate for another.

A bundle is not a new reward model. It is a verifier of the evidence graph.

## V1 schema

The manifest is \`muse-evidence-bundle-v1\`:

- \`docs/research/schemas/muse-evidence-bundle-v1.schema.json\`
- Synthetic complete, incomplete, failure-precedence and cross-binding mismatch fixtures live in \`docs/research/fixtures/\`.

The manifest records:
- exact case, source/candidate Score, source/candidate audio and candidate MIDI SHA-256 digests;
- component status, receipt digest and producer/verifier identity/version;
- explicit binding digests for each component;
- derived pass/fail/abstain status and gate counts;
- a domain-separated, recursively key-normalized bundle receipt seal.

The digest recipe is a versioned project profile; it is not claimed to be RFC 8785/JCS.

## Component binding rules

### Symbolic edit

Verify the symbolic result using the public \`symthaea_muse::edit_evidence::verify_result_seal\` API. Its case, source Score and candidate Score digests must equal the bundle's corresponding artifact digests.

The symbolic sub-gate is the conjunction of target adherence, protected-content preservation and correspondence. This is **not** StrictEMR: instruction compatibility, render integrity, acoustic realization, score-to-audio conformance and human acceptance remain independent gates.

### Render execution

Verify the render receipt through the public \`muse_render_receipt::verify_receipt_bytes\` and \`validate_receipt\` APIs. For a completed render, the full artifact directory must also pass \`verify_bundle\`.

Cross-bind:
- render source/candidate Score digests to the bundle Score digests;
- \`midi.midi_sha256\` to candidate MIDI;
- \`execution.input_midi_sha256\` to the same MIDI;
- \`execution.output_audio_sha256\` and \`audio.audio_sha256\` to candidate audio;
- \`midi.score_sha256\` to candidate Score;
- the receipt's independent \`verification\` channels to their component receipt hashes.

The render receipt exposes explicit \`midi_score_conformance\`, \`rendered_score_conformance\`, \`protected_audio_preservation\` and \`human_evaluation\` evidence channels. Their verdicts and receipt hashes must agree with the bundle's corresponding independent component. \`outcome_unknown\` is never a successful render and never permission to retry.

### Acoustic observation

Verify the observer result through \`symthaea_muse::audio_edit_observer::verify_gain_envelope_observation_seal\`. Contract, source WAV and candidate WAV digests must match the bundle's acoustic and audio bindings. A \`full_mix\` observation cannot satisfy an instrument-local claim; an \`isolated_stem\` identity still requires upstream stem/material lineage proof.

The acoustic observer's RMS gain result is not a timbral-quality, score-to-audio or human-perceptibility verdict.

### Score-to-audio conformance and human acceptance

These are required top-level gates, but their independently verifiable receipt formats/validators are not yet implemented in this workstream. Until registered public verification APIs exist, these gates must remain \`not_run\` or \`abstain\`. A manifest-provided string saying \`pass\` is not sufficient.

The human receipt must bind the exact case and candidate Score/WAV digests, the frozen listening protocol version and the preregistered rater quorum. It cannot authorize a different candidate or replace machine-verifiable preservation.

## Outcome lattice (non-compensating)

- **fail:** any independently verified required gate fails; any receipt seal fails; or any case/source/candidate/MIDI/WAV/contract/receipt binding differs.
- **abstain:** no known failure exists, but any required gate is missing, \`not_run\`, unsupported, ambiguous, low-confidence, out of scope or lacks an independently verifiable receipt.
- **pass:** all five required gates independently pass, all receipt seals are valid, and every exact artifact/case binding agrees.

Failure dominates abstention. Abstention dominates pass. No average or high reward score can reverse this order.

The schema constrains the basic result shape and conditional status counts. Cross-object digest equality, receipt cryptography, semantic pass criteria and precedence are runtime verifier responsibilities; JSON Schema alone cannot establish them.

## Synthetic fixtures

- \`muse-evidence-bundle-v1.complete.synthetic.json\`: shape-level example where all five component claims are marked pass. Its repeated hex strings are sentinels, not actual cryptographically verified receipts. It is **not** evidence that a real bundle can currently pass.
- \`muse-evidence-bundle-v1.incomplete.synthetic.json\`: symbolic/render/acoustic pass, score-to-audio and human are \`not_run\`; outcome abstains.
- \`muse-evidence-bundle-v1.failure-dominates-abstain.synthetic.json\`: one acoustic failure plus two missing gates; outcome fails, not abstains.
- \`muse-evidence-bundle-v1.case-lineage-mismatch.synthetic.json\`: symbolic receipt bound to a different case digest; runtime must fail it.
- \`muse-evidence-bundle-v1.candidate-audio-mismatch.synthetic.json\`: acoustic receipt bound to a different candidate WAV; runtime must fail it.

These fixtures are currently checked for schema/shape and intended lattice semantics. Their receipt digests are deliberately placeholders. They must never be presented as passing cryptographic tests.

## Adversarial qualification matrix

Before calling the verifier qualified, test:
1. wrong case, source Score, candidate Score, MIDI, source WAV or candidate WAV;
2. stale or tampered component receipt, invalid bundle seal, unknown component field or unsupported producer version;
3. render receipt \`outcome_unknown\`, incomplete execution, nonzero exit or MIDI/audio hash mismatch;
4. protected-content failure accompanied by otherwise passing target edit;
5. missing acoustic, score-to-audio or human receipt;
6. full-mix observer used for an instrument-local claim;
7. stale stem identity or source-lineage receipt;
8. human receipt from a different candidate or listening protocol;
9. nested JSON key reordering, manifest component reordering and duplicate component entries;
10. failure plus abstention, confirming failure precedence.

Targets (prospective, not measured): 100% rejection of deterministic wrong-binding and tampered-receipt controls; zero false pass on missing required evidence; identical canonical bundle digest for semantically identical reordered JSON; and O(number of components) lookup/verification cost.

## Implementation order

1. Verify each component using its existing public API; do not recreate its digest logic in the bundle verifier.
2. Add closed manifest wire validation and the recursive key-order-independent bundle seal.
3. Add exact cross-bindings and derive the non-compensating outcome lattice.
4. Use render verifier's full \`verify_bundle\` on the artifact directory; receipt-only validation is insufficient for a full integrity claim.
5. Register the score-to-audio and human receipt validators before allowing their component statuses to pass.
6. Add a CLI after the library API is stable, plus pinned JSON Schema Draft 2020-12 and exact-head Rust tests.

## Claim boundary

This bundle is evidence of a specific case execution and its defined checks. It does not prove universal musical quality, authorship, publication rights, instrument identity without lineage evidence, listener preference outside the registered cohort, or that hashes alone prove an untrusted renderer consumed a file. Artistic acceptance remains human-governed.
