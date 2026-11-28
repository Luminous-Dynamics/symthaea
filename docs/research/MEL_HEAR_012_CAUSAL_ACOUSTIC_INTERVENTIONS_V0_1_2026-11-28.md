# MEL-HEAR-012 — Causal acoustic interventions v0.1

**Status:** manifest/receipt contract and byte-binding validator. No audio transformation is executed in this crate.

## Research thesis

Passive audio similarity cannot by itself identify why a recording changed. A difference may arise from musical content, performance realization, instrument/source, room/channel, production processing or corruption. The evaluator needs a controlled intervention manifest, an exact record of what actually ran, and independent measurement of whether the declared musical property behaved as predicted.

Move from passive matching to counterfactual acoustic science:

1. Preserve the immutable baseline recording and its lineage.
2. Declare one intervention factor, implementation, parameters, deterministic seed and source-time support.
3. State the expected effect on one explicitly named musical property.
4. Materialize the intervention and bind the exact output bytes in a receipt.
5. Independently evaluate the output; do not accept the intervention's expected label as evidence.
6. Retain ambiguity and non-identifiability rather than forcing an answer.

## What this slice enforces

The manifest and receipt schemas bind:
- Work and performer/session cluster identities through digests.
- Baseline audio, rights/consent evidence, annotation evidence and parameter manifests.
- Intervention factor, family, target property and expected invariant/equivariant/non-invariant/not-estimable relation.
- Deterministic seed and explicit sample support.
- Exact manifest bytes, exact input audio bytes and exact output audio bytes.
- Implementation and execution-environment identities.
- Hidden reference annotations; a model-facing reference label is not allowed in the manifest contract.

The Rust validator rejects source/output substitution, replay against a changed manifest, mismatched parameter/implementation/seed/scope bindings, unblinded reference, non-deterministic benchmark cases, invalid sample geometry and sham controls that claim the target property changed.

## Important limits

A valid receipt proves that the supplied bytes and execution metadata are mutually bound. It does not prove that the intervention actually implements its name, that the declared expected effect is true, or that a human listener perceives the property as preserved. Those require independent transform qualification, property-specific audio evaluation and human/perceptual evidence.

The next slice should implement known-answer room/channel interventions and sham controls, then test property-relative counterfactual outcomes against #13908 and #14147. Active analysis selection should compare expected risk reduction per unit compute, latency and human-review cost against fixed-resolution, random and uncertainty-only baselines.
