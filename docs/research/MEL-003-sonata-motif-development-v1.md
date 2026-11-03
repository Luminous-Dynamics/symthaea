# MEL-003-MD1 — Sonata motif-development realizer V1

**Status:** implementation and adversarial tests authored; not yet compiled or executed  
**Date:** 2026-11-03  
**Parent:** [#3898](https://github.com/Luminous-Dynamics/symthaea/issues/3898)  
**Capability contract:** [#10231](https://github.com/Luminous-Dynamics/symthaea/issues/10231)  
**Action-specific tracking:** [#10233](https://github.com/Luminous-Dynamics/symthaea/issues/10233)  
**Adapter:** `crates/domains/symthaea-muse/src/sonata_motif_development.rs`

## Research objective

Turn the cognitive action `DevelopMotif` into a bounded, score-changing operation whose exact source, transformation, target region and protected fields can be inspected and independently checked.

This is distinct from `ReturnOpeningMaterial`. A return restores earlier material; development transforms a named source idea in a designated later section.

## Contract

The adapter accepts only:
- a plan-bearing `SonataRealization`;
- a trace whose action is exactly `DevelopMotif`;
- a driving obligation that exists in the realization's plan, is still pending, and exactly matches the trace's obligation kind, motif ID and transformation;
- a plan-defined source region (`sonata.primary` or `sonata.secondary`) and the Development target.

It refuses unsupported motif IDs, missing regions, empty or ambiguous note mappings, chromatic pitches that cannot be mapped exactly into the section's diatonic scale, incompatible source/target event counts or durations, stale obligations, mismatched trace metadata and prohibited transformations.

## V1 scope

Only target-melody pitches are mutable. Note count, event order, onset, duration, velocity, voice role, part identity, emphasis, section intensity, tempo, meter, key, total duration and all non-target notes are protected.

The candidate set currently contains an identity baseline and one requested transformation: diatonic transposition or inversion around the first source degree. The selected candidate remains unset. This is deliberate: candidate generation is not a license to make an unqualified creative choice.

The motif family is derived deterministically from the exact plan-defined source score region. Its ID, version, structural-signature digest, source projection digest and full family digest are retained. This is a reproducible session-scoped family extraction—not a claim that the motif has been admitted to a persistent, reviewed Motif Foundry library.

## Independent evidence

For each candidate, retain:
- exact source/baseline score digest;
- source and target section and key;
- exact ordered source/target note indices and source-index digest;
- source-region event-projection digest;
- derived motif-family ID/version/signature digest and family digest;
- exact candidate score digest and full event-projection digest;
- transformation and capability receipt digest;
- motif-family identity verdict;
- exact target-only pitch diff result;
- full score theory-validation report;
- score-side Sonata obligation verification.

A receipt is not itself proof of musical success. A candidate with an unrelated identity verdict, failed locality check, failed theory validation or unfulfilled obligation remains a failed candidate; the adapter does not auto-select it.

## Adversarial tests authored

1. The ordinary composer output is the no-op baseline and does not falsely fulfil the explicit development transformation.
2. The requested transposition changes target pitches and is independently accepted by the plan's score-side obligation verifier.
3. A trace that reuses the correct obligation ID but changes the requested transformation is rejected.
4. An obligation already fulfilled or waived in the plan cannot be replayed as pending work.
5. Chromatic source pitches abstain instead of being rounded to a scale degree.
6. A fixed source score/region deterministically derives the same family identity.
7. Pitch-only application preserves every protected field and all non-target notes.
8. A prohibited family transformation is rejected before publication.
9. The action capability matrix names this adapter separately from the thematic-return adapter; unsupported actions remain unsupported and shadow-only remains non-executable.

## Qualification targets

- 100% exact target-only pitch changes for candidates marked locality-valid.
- 100% preservation of declared protected fields.
- Zero false acceptance of missing, stale, mismatched or already-fulfilled obligations.
- Zero silent fallback or action substitution.
- 100% of candidate outcomes retained, including identity baseline and any failed validation.
- Byte-identical fresh-process receipts for fixed inputs and environment.
- Human motif-recognition, creative-value and preference thresholds remain pilot-derived; they are not inferred from symbolic identity metrics.

## Claim boundary and next step

This adapter establishes a bounded symbolic transformation path only if its Rust tests and exact-head CI pass. It does not yet demonstrate that the production Studio cognitive session selects or dispatches the adapter, that a musician recognizes the transformation, that it improves music, or that it generalizes beyond the Sonata form.

The next experiment is a frozen-input shadow-versus-enabled comparison in which the cognitive proposal is bound to the exact pending obligation and the selected `Score` digests are compared. The live Studio route remains unchanged until that causal score-level bridge is qualified.
