# MEL-003 — Sonata Action-to-Realizer Capability V1

**Status:** initial qualification contract; no live-path behavior change  
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
| DevelopMotif | Unsupported |
| IntroduceContrast | Unsupported |
| IncreaseHarmonicInstability | Unsupported |
| ModulateToRelatedKey | Unsupported |
| IncreaseDensity | Unsupported |
| StrengthenCadence | Unsupported |
| AddCounterline | Unsupported |
| ReturnOpeningMaterial | **Supported by `sonata-thematic-return-v1`** |
| ThinTexture | Unsupported |

Shadow-only mode is available for every action and does not authorize candidate generation.

The Sonata-return adapter declares a plan-bearing Sonata realization as input and a symbolic score as output. It emits theory-validation, obligation-verification, outside-target skeleton comparison, symbolic outcome measurement and intervention-prediction evidence. These are evidence channels; they are not promises that every candidate passes every check.

A successful `SonataInterventionBatch` contains the exact capability qualification used to authorize it. Unsupported execution returns an error carrying the qualification. Mismatched `PredictionContext.action` and `CognitiveDecisionTrace.proposal.action` are rejected independently. Legacy serialized batches missing the new receipt decode as shadow-only, never as current execution authority.

## What this does not do

- It does not change the live Studio hard-coded recapitulation goal.
- It does not claim the HDC/CfC/FEP session causally changes the current live selected score.
- It does not map an unsupported action to `ReturnOpeningMaterial`.
- It does not implement score realizers for the other nine actions.
- It does not claim musical improvement, listener preference, collaboration value or consciousness.

## Next action-family tranche

Implement one additional symbolic action family in its own adapter. Require same-frozen-input shadow-versus-enabled selection, exact baseline and candidate score byte digests, theory validation, explicit Preserve/Change contract, obligation verification and non-target preservation. Add negative controls in which the realizer is unavailable or the requested action is wrong; these must abstain rather than fall back.

Only after at least one non-hardcoded FEP action has a qualified realizer should the production Studio route be offered an opt-in cognition-controlled arm. Keep the existing behavior as the control/shadow path.

## Benchmark targets

Keep these measurements separate:

1. **Action fidelity:** requested-to-realized action match; unsupported-action false execution rate must be zero.
2. **Theory validity:** candidate score validation and obligation verification; report all failures, not only selected candidates.
3. **Edit locality:** exact event/score diffs and preservation outside the declared target region.
4. **Causal score influence:** same frozen Sonata plan, seed, input history and candidate budget; compare cognition-shadow and cognition-enabled selected score hashes. A different trace with the same selected score is zero score-level influence.
5. **Musical quality:** blind preference, motif recognition, formal coherence and listener utility, measured separately from action compliance.
6. **Runtime:** candidate generation latency and failure/abstention rates stratified by action and score complexity.

## Qualification status

Six Rust tests are authored in this initial slice: full execute-mode action matrix; shadow-only mode for every action; unsupported action with no coercion; action/context mismatch; successful batch receipt binding; legacy batch decoding to shadow-only. They have not been compiled or run in the current environment. This note and PR do not claim exact-head CI qualification.
