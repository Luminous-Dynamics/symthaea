# MEL-PRE-001 — Musical Forecast V1 and Qualification Plan

**Status:** research contract / implementation baseline  
**Date:** 2026-11-03  
**Parent:** [#10190](https://github.com/Luminous-Dynamics/symthaea/issues/10190)  
**Code:** `crates/domains/symthaea-muse/src/musical_forecast.rs`

## Research question

Can a musical agent make a bounded next-action commitment inspectable before realization, so a collaborator can anticipate, steer, veto or branch from it without being expected to obey it?

The object being exposed is actionable musical intent (symbolic event slots, role, beat position, duration and expression), plus field-level uncertainty. It is not hidden chain-of-thought, and it is not a post-hoc explanation.

## Paradigm shift

Move from a black-box response to an **inspectable future commitment**:

1. The agent commits an immutable, versioned forecast before its target realization.
2. A forecast is bound to a session, turn, monotonic clock, input-history digest and immutable policy identity digest.
3. A later score/MIDI/audio artifact and a separately versioned event projection are bound into the evaluation receipt.
4. Revisions append a new forecast that explicitly supersedes an earlier one; the original preview is never rewritten.
5. Delivery/display is logged separately from forecast generation. A display record does not prove the musician read, understood or acted on it.
6. The collaborator can accept, steer, veto, branch, ignore or abstain. Departing from a preview is not disobedience or an agent failure.

The V1 Rust module provides the symbolic event ledger and evaluator. Each ledger event is SHA-256 hash-linked to its predecessor; the evaluation context records an exact event-sequence high-water mark at realization and the receipt binds that prefix hash. This disambiguates same-tick events and excludes later display/revision records. Publishing or signing that head outside the ledger is still required to anchor the history against a full rewrite. It does not yet wire the forecast object into the live Studio candidate/score path or a UI.

## V1 invariants

- Schema version and forecast/turn/session IDs are mandatory.
- Sequence IDs increase strictly; event ticks are monotonic inside one session/clock-domain ledger. The realization receipt pins an exact sequence high-water mark in addition to the tick; same-tick events are resolved by sequence.
- A commit cannot be backdated relative to the append-only ledger and must occur strictly before its target execution window begins.
- The execution window is non-empty; the forecast is committed before its start and remains valid through its expiry.
- Every predicted event has a unique stable slot ID and positive duration.
- Every predicted field has an explicit probability or an explicit unavailable reason.
- Probability values are integer parts-per-million in `[0, 1,000,000]`.
- A new forecast can supersede only an active previous forecast in the same session, turn and clock domain.
- A forecast is not eligible if created after realization, outside its target window, expired, superseded or invalidated. A display receipt cannot resurface an expired, superseded or invalidated preview as current.
- Input-history and immutable-policy-identity mismatches fail closed; a yoked forecast cannot pass as contingent.
- The evaluation receipt binds the exact realization artifact digest and the canonical digest of the symbolic event projection used by the scorer.
- A single forecast's Brier contribution is not called calibration. Calibration requires a held-out population, stratified by field, forecast horizon, task and collaborator.

## Evaluation arms

All arms must share the same frozen input, action budget, generation budget, model/policy identity and task schedule wherever possible.

| Arm | Purpose | Required result |
|---|---|---|
| No preview | Reactive baseline | Establishes ordinary coordination |
| Post-hoc explanation | Timing negative control | Must not pass anticipatory-timing gate |
| Yoked preview | Contingency negative control | Different input-history digest must be rejected |
| One precommitted forecast | Core treatment | Must be visible before bound realization |
| Uncertainty-aware alternatives | Avoid false precision | Proper scoring and useful branching |
| Steerable preview | Agency treatment | Accept, constrain, veto, branch and ignore are available without penalty |

Human-human and musician-alone baselines should be added on a matched subset when feasible. Preview-only and steerable-preview ablations must not receive additional generation calls or compute.

## Outcomes — do not collapse into one score

### Integrity and provenance

- Forecast committed before realization: **100% of scored forecasts**.
- Append-only revision history: **100%**; zero rewritten prior forecasts.
- Stale, expired, superseded and post-hoc forecasts accepted as current: **0**.
- Yoked or wrong-model-identity previews accepted as contingent: **0**.
- Every prediction field has confidence coverage or an explicit unavailable reason.
- Every scored realization has an artifact digest and a reproducible event-projection digest.

### Forecast performance

- Field-wise precision/recall for beat position, duration, pitch, velocity, role and articulation.
- Proper score (Brier; log loss where the event distribution and outcome semantics justify it).
- Reliability diagrams, calibration error and confidence intervals over held-out forecast populations.
- Accuracy by forecast horizon, style, density, task and collaborator.
- Revision frequency, cancellation, stale rate and forecast churn.
- Do not credit a precise prediction if the realization itself is not bound to the exact source artifact/projection.

### Coordination and agency

- Anticipation accuracy before hearing the output.
- Join/entry timing error and phase error, in milliseconds and beat fractions.
- Missed entries, collisions, interruption burden, repair count and time-to-recovery.
- Steer/veto/branch/ignore rates, successful constraint satisfaction and time-to-intent.
- Perceived control, authorship/ownership and desire to continue.
- Response diversity, including valid departures from the forecast.
- Anchoring, conformity and premature closure as potential harms, not success.

### Runtime

- Forecast creation, delivery, display and update latency: p50/p95/p99.
- Lead time distribution, stale-preview rate, cancellation propagation and clock drift.
- CPU, memory, network and audio/MIDI timing impact.
- Exact-process replay, deterministic event order and receipt verification.

No global 'transparency score' should trade away creative agency for a higher forecast-accuracy average.

## Qualification targets

These are preregistered targets, not measured results:

- **Integrity gate:** 100% precommit and append-only history, zero false acceptance of post-hoc/yoked/stale/superseded inputs.
- **Forecast gate:** calibration and accuracy must beat both post-hoc and yoked controls on preregistered metrics; establish thresholds from a pilot before confirmatory evaluation.
- **Coordination gate:** improve at least one musician-valued coordination outcome versus no preview, with confidence intervals and a pilot-derived non-inferiority bound on agency/ownership/diversity.
- **Agency gate:** veto, branch and ignore must be usable without system penalty or forced return to the forecast.
- **Robustness gate:** report every participant/task/horizon subgroup and missingness; no pooling failures away.

Do not pick universal p95 latency or participant-count thresholds before a device/performer pilot establishes the operating envelope and variance.

## Industry comparison and position

- [Artificial Analysis AA-Music-Instrumental v1.1](https://artificialanalysis.ai/music/leaderboard/instrumental?tab=leaderboard) is a blind pairwise preference arena. On 2026-11-03 it reports Suno v6 at 1148 (95% interval 1126–1170), Suno v6-mini at 1114 (1093–1135), and a Mureka/Lyria cluster around 1078–1084. This is evidence about preference for standalone instrumental outputs, not causal interaction, forecast utility or edit fidelity.
- Its [methodology](https://artificialanalysis.ai/methodology/music) uses one output per prompt, roughly three minutes, blind paired comparisons, Bradley–Terry estimation and loudness/delivery normalization. Our 90s/180s internal banks are not a reproduction of this arena; a comparable claim requires matched prompts, modality, settings and independent listeners.
- [Suno Studio 2.0](https://help.suno.com/en/articles/13670529) is now a generative DAW with timeline workflow, MIDI, automation, effects and synths. Melothaea must compete as an interaction/cognition layer, not by claiming that chat plus audio clips is a novel studio.
- [Lyria RealTime](https://ai.google.dev/gemini-api/docs/realtime-music-generation) is a low-latency streaming system with bidirectional controls. The relevant differentiator is whether a symbolic forecast can be inspected, revised and causally tied to the next performance event—not whether a prompt slider exists.
- [jam_bot (NIME 2026)](https://nime.org/proc/nime2026_73/index.html) already addresses expressive velocity, varying-tempo call-and-response, throughput and MIDI latency compensation. We must benchmark those dimensions too; a forecast UI is not a substitute for good timing.
- [Agents in Concert (IUI 2026)](https://www.media.mit.edu/publications/agents-in-concert/) reports musician co-design and visual forecasts that helped musicians anticipate AI responses, while raising agency/spontaneity tensions. This is a research signal and design precedent, not proof that every preview improves musicianship.
- The [ICASSP 2027 Audio Editing Challenge](https://audio-editing-challenge.github.io/) has a public 2,000-item/17,741-rubric MMAE development set, with a 500-item private test set per track opening November 10. Its published context says current systems are below 5% Exact Match Rate. Registration closed October 8 and the Agent Track model-release cutoff was November 1; eligibility must be verified rather than assumed.

**Strategic decision:** do not try to outspend Suno on raw waveform generation first. Keep the renderer replaceable. Build the strongest score-first planning, verified edit, reciprocal control, predictive transparency and collaborator-specific memory layer, and measure raw preference as a separate benchmark family. The 2026 Qwen-Music report is also an architectural signal—semantic planning followed by a distinct high-fidelity renderer—but its comparative results are author-reported, not independent confirmation.

## Remaining work after this V1

1. Add forecast alternatives and per-field intervals/distributions rather than pretending every future is a point estimate.
2. Bind the forecast object to the real Studio score-selection bridge only after the score-level counterfactual gate in #3898 passes.
3. Add a canonical artifact resolver and byte verification; a SHA-256 field alone does not retrieve or authenticate artifact bytes.
4. Implement the display/interaction client and separately record generated, delivered, displayed, acknowledged, steered, vetoed and ignored states.
5. Run deterministic symbolic fixtures and adversarial regressions; then pilot with musicians under appropriate consent/ethics review.
6. Compare symbolic/MIDI paths only to systems that expose those event-level observables; do not impute note-level accuracy from audio-only outputs.

## Claim boundary

The current code is a Rust schema, append-only event ledger and deterministic symbolic evaluator. It does not yet establish that the live Melothaea path emits forecasts, that a musician sees or understands them, that the model anticipates subjectively, that the user is more creative, or that a system is conscious. No Rust build/test result is claimed until exact-head local or CI evidence exists.
