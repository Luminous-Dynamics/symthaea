# MEL-003 — Sonata Action-to-Realizer Capability V1

**Status:** two narrow action adapters authored; no live-path behavior change  
**Date:** 2026-11-03  
**Parent:** [#3898](https://github.com/Luminous-Dynamics/symthaea/issues/3898)  
**Tracking issue:** [#10231](https://github.com/Luminous-Dynamics/symthaea/issues/10231)  
**Code:** `crates/domains/symthaea-muse/src/sonata_intervention.rs`

## Why this boundary is necessary

The active Sonata Studio route builds an HDC/CfC/FEP temporal session, but passes a literal `Some(CognitiveGoal::Recapitulate)` when constructing the symbolic observation. The bridge gives that explicit goal priority over the terminal FEP action and therefore selects `ReturnOpeningMaterial`.

Removing that hard-coded goal alone is not safe: the current Sonata candidate generator only knows how to realize `ReturnOpeningMaterial`. The function rejects other actions with `UnsupportedAction`. A direct one-line change would either make the live path fail or invite an unsafe action substitution.

## V1 contract

`qualify_sonata_action_v1(action, mode)` returns a versioned, serializable receipt containing:

- the exact symbolic action;
- requested mode (`execute` or `shadow_only`);
- status (`supported`, `unsupported`, or `shadow_only`);
- for supported actions, a named realizer, input/output modality and explicit evidence outputs;
- for unsupported or shadow-only actions, a reason that remains in the receipt.

The executable registry is deliberately narrow:

| Symbolic action | Execute mode |
|---|---|
| Maintain | Unsupported |
| DevelopMotif | **Supported by `sonata-motif-development-v1`** |
| IntroduceContrast | Unsupported |
| IncreaseHarmonicInstability | Unsupported |
| ModulateToRelatedKey | Unsupported |
| IncreaseDensity | Unsupported |
| StrengthenCadence | Unsupported |
| AddCounterline | Unsupported |
| ReturnOpeningMaterial | **Supported by `sonata-thematic-return-v1`** |
| ThinTexture | Unsupported |

Shadow-only mode is available for every action and does not authorize candidate generation.

The Sonata-return adapter declares a plan-bearing Sonata realization as input and a symbolic score as output. It emits theory-validation, obligation-verification, outside-target skeleton comparison, symbolic outcome measurement and intervention-prediction evidence.

The separate Sonata motif-development adapter accepts only a `DevelopMotif` trace bound to an exact pending plan obligation. It derives a session-scoped motif family from the named exposition region, proposes an identity baseline and one permitted pitch-only transformation, and records the full source score, note-index projection, motif-family version/signature and output score digests. It preserves all non-target note fields and score metadata. It emits theory validation, score-side obligation verification, motif-family identity and exact target-only diff evidence. This derived family is reproducible from the exact score region; it is not claimed to be a persistent, human-reviewed Motif Foundry family.

These are evidence channels; they are not promises that every candidate passes every check.

A successful `SonataInterventionBatch` contains the exact capability qualification used to authorize it. Unsupported execution returns an error carrying the qualification. Mismatched `PredictionContext.action` and `CognitiveDecisionTrace.proposal.action` are rejected independently. Legacy serialized batches missing the new receipt decode as shadow-only, never as current execution authority.

## What this does not do

- It does not change the live Studio hard-coded recapitulation goal.
- It does not claim the HDC/CfC/FEP session causally changes the current live selected score.
- It does not map an unsupported action to `ReturnOpeningMaterial`.
- It does not implement score realizers for the other eight actions.
- It does not claim musical improvement, listener preference, collaboration value or consciousness.

## Second action-family tranche: DevelopMotif

The first additional action family is implemented as a separate `sonata-motif-development-v1` adapter (tracking: [#10233](https://github.com/Luminous-Dynamics/symthaea/issues/10233)). It uses the exact exposition/development windows in the plan; it does not perform approximate phrase search. The V1 transformation set is deliberately limited to diatonic transposition and inversion because both can change pitch while preserving event count, order, onset, duration, velocity, role, part identity, emphasis and section intensity.

The batch carries exact source-score, source-note-index, source-region-projection, derived-family/version/signature, candidate-score and full event-projection digests. The adapter checks that the trace's driving obligation exists, is still pending in the realization's plan, and exactly matches the trace's named motif/transformation. This prevents stale or forged observation metadata from asking for a transformation different from the one the independent score-side verifier evaluates.

No candidate is automatically selected. A musician or separately qualified policy must select among the retained candidates; the baseline and transformed candidate both carry independent theory, obligation, identity and locality evidence. No live Studio dispatch is added by this tranche.

The next action family should be selected only after this adapter is reviewed and exact-head Rust verification is available. The production Studio route should remain shadow/control plus an explicit opt-in path until a same-frozen-input cognitive-versus-shadow experiment proves score-level influence.

## Benchmark targets

Keep these measurements separate:

1. **Action fidelity:** requested-to-realized action match; unsupported-action false execution rate must be zero.
2. **Theory validity:** candidate score validation and obligation verification; report all failures, not only selected candidates.
3. **Edit locality:** exact event/score diffs and preservation outside the declared target region.
4. **Causal score influence:** same frozen Sonata plan, seed, input history and candidate budget; compare cognition-shadow and cognition-enabled selected score hashes. A different trace with the same selected score is zero score-level influence.
5. **Musical quality:** blind preference, motif recognition, formal coherence and listener utility, measured separately from action compliance.
6. **Runtime:** candidate generation latency and failure/abstention rates stratified by action and score complexity.

## Qualification status

Rust unit tests are authored for the capability matrix, separate realizer identity/evidence, plan/trace obligation binding, stale-obligation rejection, exact target-only field preservation, chromatic-source abstention, prohibited transformations, deterministic family derivation and score-side verification. They have not been compiled or run in this work session. Neither this note nor the PR claims exact-head CI qualification.
