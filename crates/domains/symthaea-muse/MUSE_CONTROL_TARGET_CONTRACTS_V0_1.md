# Muse Control-Target Contracts v0.1

**Status:** Research/evaluation contract for the M0c response-curve probe. This document specifies what the probe is allowed to claim; it is not a report of completed measurements. The exact-head workflow remains the source of runtime evidence.

## Purpose

A change in output is not proof that a control achieved its target. Likewise, failure to change an exact fingerprint is not proof that a control is dead: the fingerprint can intentionally quotient out a dimension such as transposition, while dynamics and form may change independently.

Every control evaluation therefore keeps four questions separate:

1. **Target fidelity:** did the realized property match a declared target?
2. **Intervention response:** which measured channels changed as the control varied?
3. **Non-target invariance:** which independently predeclared properties stayed stable?
4. **Other evidence:** did rendering succeed, and is there separate perceptual or cultural evidence?

No aggregate “control quality” score is defined. Missing output, a missing seed, an absent plan field, or an undeclared target remains unknown/not-predeclared rather than a pass.

The current probe also computes a small set of **candidate non-target invariants** from the one-factor-at-a-time grids: absolute key tonic (except during a tonic sweep), key tonality (except when valence is an active tonal target), meter, and tempo (except during an arousal sweep). For valence sweeps where a preset pins the mode or the RagaModalArc route owns its modal default, the probe instead checks that realized tonality remains locked across the sweep. Each curve reports its baseline value, number of checked points, number of deviations, and point-level deviations. These are diagnostic rather than automatic failures; the purpose is to expose leakage and unexpected coupling without collapsing it into a single score. This follows the separation of invariance and transformation response used in the [ICASSP 2026 controllable-music evaluation study](https://doi.org/10.1109/ICASSP55912.2026.11461451), while keeping the Muse checks grounded in observable symbolic fields.

For numeric control grids, the probe also reports **Spearman's rank correlation** between the control value and selected numeric output descriptors. Tied output values receive average ranks; a constant output or non-numeric axis returns an explicit status with a null coefficient rather than an invented zero. The coefficient characterizes rank ordering across the tested points only; it does not establish control-target correctness, perceptual meaning, or causation beyond the one-factor intervention. This complements the response-shape label with a signed measure, consistent with the survey's discussion of rank-based control evaluation.

## Versioned control contracts

| Control | Current target contract | Applicability and explicit limits |
|---|---|---|
| **Tonic** | Compare the realized score-key tonic pitch class with the requested pitch class (modulo 12). Report the nine candidate transposition-invariant structure layers separately. | Tonal-center adherence is mechanically checkable. The nine-layer invariance set is diagnostic, not an automatic quality gate; styles may impose justified constraints that need investigation. This is not a claim that transposition preserves register-sensitive orchestration or absolute-key harmony. |
| **Valence** | Where mode is not pinned by the style or owned by a grammar default, negative values target minor and positive values target major. Exactly zero is neutral and is excluded from the directional match denominator. | This is a **tonal proxy**, not a measurement of perceived emotional valence. Styles with a pinned mode or an engine-owned modal default are explicitly not scored on this mapping. The report must disclose tested, scored, neutral-unscored, matching, and mismatching point counts separately. |
| **Arousal** | Compare realized tempo with the source-declared tempo curve: spec.tempo(arousal) multiplied by the applicable attitude adjustment. The dedicated Blues call-response and Jazz chorus engines use the unadjusted spec tempo. Current numerical tolerance: 0.001 BPM. | This checks fidelity to the implementation’s tempo mapping, **not perceived arousal**. Motif selection and phrase rhetoric may also respond to arousal, so non-target invariants remain undeclared until justified per grammar route. |
| **Energy** | For the PeriodSentence family only, compare the emitted phrase model with the composer decision rule: sentence when arousal > 0.6 or energy > 0.75, otherwise period. | This checks the threshold policy and the recorded plan against the same composer rule; it is not a perceptual energy metric. Other grammar families remain unscored until their intended energy semantics are specified independently. |
| **Requested bars** | For BluesCallResponse and JazzChorus, check that the plan preserves the input request, applies its declared whole-chorus rounding rule, and agrees with an independently observed score span. Expected realized bars are ceil(max(requested_bars, 1) / bars_per_chorus) * bars_per_chorus. Observed span is max(note.onset + note.duration) / meter; current tolerance: 0.001 bars. | The score span is measured from note onsets/durations rather than inferred from section count or the plan alone. Other grammar routes remain not_predeclared: exact duration cannot be assumed when a family may quantize requests into larger musical units or express duration through a non-bar-based process. |
| **Seed** | No musical target is assigned. Seeds form the robustness dimension: report output/channel changes per expected seed and preserve identity coverage. | Seed variation cannot substitute for target fidelity. Duplicate, unexpected, malformed, or missing seed identities invalidate complete-coverage claims. |

### Why zero valence is unscored

The former binary predicate treated valence == 0 as a major-mode target because it used valence >= 0. That made the neutral point count as directional adherence even though the metric had no neutral semantic target. The probe now uses strict sign comparisons: negative = minor proxy, positive = major proxy, zero = neutral/unscored. The neutral point remains visible in the report, but is not counted as either a match or mismatch. This change only corrects evaluation semantics; it does not change the composer.

### Why bar count is scoped to two grammar families

A universal equality test (realized_bars == requested_bars) would be wrong for a route whose documented contract rounds to a complete chorus. Conversely, accepting plan metadata alone would be circular: a plan can claim a duration that the emitted note events do not occupy. The current check is intentionally restricted to the Blues and Jazz chorus plans, which expose request, realized length, and chorus unit. It checks the plan's rounding policy and the note-event span independently. A mismatch is useful diagnostic evidence; it should not be “fixed” by loosening tolerance or converting a missing plan into success.

For a different grammar family, the next step is to add an explicit typed duration contract (for example, an exact-bar, quantized-unit, or process-duration contract), then add a regression test that compares the contract with realized note timing. Do not infer it from a style label, form_section_count, or fingerprint displacement.

## Evidence and pass discipline

- Run the probe through compose_with_grammar_plan, the routed path used by compose_styled; a nearby lower-level composer entry point is not equivalent evidence for a dedicated engine.
- Change one control at a time, preserve the unchanged baseline in each grid, and repeat across the frozen seed set 90210, 130363, 222731, and 918273.
- Keep target-fidelity counters separate from channel-change indicators, structural-layer distances, monotonicity labels, and seed robustness.
- A measured contract says that the target was evaluated. It does not mean every point matched; matching and mismatching counts must remain visible.
- Source/toolchain/runner/report digests and exact sweep cardinality must bind the generated artifact to the checked-out head. Queued or skipped workflows and absent artifacts are not passes.
- Symbolic score evidence does not establish audio-render quality, subjective emotional impact, style distinctness, or cultural authenticity. Those need separately defined evidence and appropriately scoped claims.

## Research basis

1. Wei Duan, Yi Yu, and Jean-Pierre Briot, “Recent advances in music generation: methods, evaluation, and challenges,” Artificial Intelligence Review 59 (2026), published 2 June 2026. The survey discusses separate symbolic metrics for tonality, rhythm, structure, alignment/control and rank-based response analysis; it cautions that no single objective metric covers music quality. https://doi.org/10.1007/s10462-026-11582-x
2. Laura Ibáñez-Martínez et al., “Evaluating Disentangled Representations for Controllable Music Generation,” ICASSP 2026. Their probing framework examines informativeness, equivariance, invariance and disentanglement, and reports mismatches between intended and observed representation semantics. https://arxiv.org/abs/2602.10058
3. Hüseyin Yücebağ, “Musical Representation of Genre Information in Generative Artificial Intelligence: The Case of Suno,” The Turkish Online Journal of Design Art and Communication 16(4), published 1 October 2026. In a qualitative 75-output case study across five genres and five style-influence settings, response patterns varied by genre and were not universally monotonic. This is contextual evidence, not a substitute for Muse-specific measurements. https://doi.org/10.7456/tojdac.1993191

## Next evidence gate

After the exact-head workflow produces an artifact, inspect the neutral-valence denominator and the Blues/Jazz bar-span comparisons first. Only then decide whether any composer behavior needs a change. Extend valence, energy, or duration contracts to other grammar routes only when their target semantics are supported by implementation evidence and their output metric has a testable, route-specific contract.
