# Muse Analyst v1

Muse Analyst reduces routine listening work without claiming that software can
decide whether music is beautiful, meaningful, memorable, or culturally
authentic.

## Authority order

1. Native score, recipe, grammar plan, motif lineage, and provenance.
2. Rendered-audio verification.
3. Learned perceptual predictions with uncertainty.
4. Human observations.

Later layers may detect a problem or add evidence. They do not rewrite facts
from an earlier layer. Every external metric uses AnalystEvidenceLayer;
predictions and human observations cannot share a field.

## Implemented in v1

- AnalystPieceBundle, versioned through the shared Muse protocol.
- Requested-versus-realized intent and score measurements.
- Grammar-plan obligation checks.
- Transformation-aware motif occurrence summaries.
- Phrase, cadence, climax, recurrence, and ending evidence.
- Source-note traceability through the renderer.
- Explicit uncertainty and reviewer-specific escalation.
- Mandatory expert escalation for culturally qualified output.
- The /api/piece/{id}/analyst-bundle endpoint in Muse Studio.
- A Research-mode Analyst panel.
- muse_analyst_pack, which builds hash-addressed reports for paired packs.
- Leave-one-premise/motif-block-out nuisance evaluation with training-only
  normalization, exact permutation evidence, and block-bootstrap intervals.
- Separate composer-assertion and symbolically-verified trace records.
- Adversarial fixtures for omitted motif events, false literal transforms,
  false cadences, unsupported obligation fulfilment, and invalid spans.
- A deterministic random audit lane for otherwise accepted output.
- Direct grammar-owner traces from PeriodSentence, GrooveCycle,
  ProcessAdditive, and ModalArcInformed, using stable voice/index score-event
  references that are translated into public event IDs.
- Independent verification of structural spans, motif sequence and distance,
  cadential markers, obligation lifecycle continuity, responsible passes, and
  evidence references. Composer assertions never self-promote.
- A native rendered-audio integrity layer for decoding, silence, true peak,
  clipping, DC offset, first/second differences, broadband impulse outliers,
  and a clearly labelled high-frequency proxy.
- An adversarial condition matrix covering assertion-source corruption,
  reversed/missing/overlapping structure, motif order and region errors,
  cadence/obligation corruption, renderer note loss, silence, clipping, and
  injected impulses.

The paired-pack command is:

    cargo run -p symthaea-muse --bin muse_analyst_pack -- PACK_DIRECTORY

It writes analyst_bundle.json and analyst.html beside the playable clips.

Existing hash-addressed packs can be augmented without replacing their
original structural truth:

    cargo run -p symthaea-muse --features studio --bin muse152_listening_pack -- augment-evidence PACK_DIRECTORY

This appends `structural_truth_with_composer_trace_v2.json` and
`audio_integrity_by_sha256.json`; the Analyst pack builder prefers the newer
trace artifact while preserving the original file.

## Next implementation order

1. Add four-way comparison bundles keyed by premise_id and motif_id.
2. Measure within-family form topology, harmonic route, development path,
   climax strategy, and ending diversity—not only activity curves.
3. Add the pinned external audio-verification sidecar for calibrated onset,
   loudness, spectrum, tempo, pitch, and panning evidence. Native integrity
   checks remain the always-on first gate.
4. Add learned embeddings only as versioned shadow evidence.
5. Calibrate motif-recognition, lure-rejection, and grammar-confusion
   predictions against blinded human judgments.
6. Route uncertainty, model disagreement, novelty, cultural review, and
   flagship candidates into a bounded human queue.

## Sidecar boundary

The audio sidecar should return JSON only. Each value records artifact hash,
analyzer and model version, measurement layer, units, uncertainty, and
limitations. Historical evidence is immutable; upgrading a tool creates a new
measurement rather than overwriting an old one.

Suitable reference tools include music21 for external computational-musicology
work, Essentia/librosa for audio verification, Basic Pitch for an independent
audio-to-note check, and MERT as one learned perceptual view. None becomes an
executive composer or a universal quality oracle.

## Human listening policy

Fully automate duplicate detection, plan compliance, motif-use screening,
renderer fidelity, regression gates, and batch anomalies.

Escalate uncertain motif recognition, family distinction, ending predictions,
style fit, and replay predictions.

Require people for flagship promotion, artistic meaning, emotional impact,
long-term memorability, and culturally specific authenticity.
