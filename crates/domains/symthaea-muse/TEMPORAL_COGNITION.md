# Temporal Symthaea Cognition in Muse

## Status

V7 introduces one narrow temporal cognitive path for Sonata return previews.
It replaces Studio's constructed terminal inference with a sequence of real
calls through Symthaea's HDC, CfC/LTC, and active-inference components.

The path is intentionally limited to the existing cognitive Sonata endpoint. It
does not become the default composer, bypass theory validation, or grant a
cognitive scalar authority over musical quality.

## Exact data flow

For a frozen Sonata score and seed, Muse performs the following steps:

1. Read the declared Sonata plan from the theory realization.
2. Divide every completed section from the opening through the end of the
   development into a frozen number of observation windows. The target
   recapitulation is deliberately not observed.
3. Measure each window with the versioned symbolic score profile.
4. Convert that profile into six bounded sensory proxies:
   brightness, flux, rhythmic complexity, harmonic tension, energy, and noise.
5. Encode the channels, section identity, and sequence position into a
   genesis-seeded continuous hypervector.
6. Evolve the encoded observation through the seeded HDC/CfC network using the
   score window's real duration as `dt`.
7. Mix a bounded fraction of the temporal state into the six FEP channels.
8. Perceive the resulting observation with `MusicalInferenceEngine`.
9. Select an action by expected free energy using an explicit musical-state
   goal vector.
10. Commit that action to the FEP agent so the following observation can
    produce a temporal-difference transition update.
11. Apply the selected action and sensory feedback to the evolving
    `MusicalState`.
12. At the recapitulation boundary, retain the complete pre-target frame
    trajectory and use its terminal inference to create the constrained
    symbolic proposal.

The proposal then enters the unchanged V6 authorities:

- canonical theory validation decides admissibility;
- the parameterized world model predicts candidate effects;
- the musical policy evaluates desired outcomes;
- Studio preview and commit preserve artist control.

## Reproducibility contract

The temporal session records:

- cognitive-session contract version;
- backend identity;
- piece seed;
- derived FEP RNG seed;
- HDC dimension and genesis namespace;
- CfC layer sizes;
- observation-window count and pre-target section boundaries;
- temporal blend and feedback strength;
- installed FEP goal preferences and precision;
- every symbolic profile and sensory vector;
- HDC input and temporal-state fingerprints;
- `MusicalState` before and after every frame;
- every selected and committed FEP action;
- free-energy and precision evidence;
- committed-action and temporal-difference learning statistics;
- terminal state and inference;
- a whole-session integrity fingerprint.

Identical inputs are required to replay exactly in the supported build and
runtime environment. The recipe still records source, renderer, soundfont,
binary, and environment identities separately. The session FNV fingerprints
are integrity and regression aids, not cryptographic artifact signatures and
not a promise of cross-architecture floating-point identity.

## Musical goals

The active-inference engine previously stored a preference vector without
installing it in expected-free-energy evaluation. V7 closes that boundary.

A declared `MusicalState` now produces an inspectable six-channel goal:

- valence influences preferred brightness;
- arousal influences rhythmic complexity, stability, and energy;
- prediction error influences consonance and noise tolerance;
- consciousness level determines goal precision.

These are bounded engineering mappings. They are not claims that emotional
experience has been measured or that the six channels exhaust musical intent.

## Temporal learning

The proposal-only `MusicalInferenceEngine::infer` remains unchanged for legacy
callers. The V7 session uses `infer_and_commit`.

Committing the chosen action records it as the hypothesized cause of the next
observation. On the following frame, the FEP agent can update its transition and
observation models through the configured temporal-difference learner. Session
validation rejects traces where:

- an action was not committed;
- frame or cycle counts disagree;
- any target-recapitulation frame leaks into the decision history;
- transition history is missing;
- the FEP seed or goal differs from its declared derivation;
- numeric evidence is non-finite;
- terminal inference or session fingerprint is inconsistent.

## Frozen temporal ablation

`cognitive_session_experiment` defines the first mechanistic V7 experiment.
Each paired trial holds fixed:

- score and frozen input digest;
- seed and FEP RNG stream;
- musical goals;
- observation windows;
- HDC/CfC architecture;
- state feedback;
- all configuration except temporal blend.

The control still executes HDC/CfC but sets `temporal_blend = 0`, preventing its
latent state from altering the FEP observation. The treatment uses the frozen
positive blend.

The descriptive gate requires:

- sixteen unique paired seeds and frozen inputs;
- valid, internally consistent traces;
- identical raw symbolic observations in each pair;
- identical FEP seeds and goals;
- no configuration drift other than temporal blend;
- mean per-channel temporal sensory change of at least `0.005`;
- at least one divergent paired FEP action.

A pass establishes only that HDC/CfC temporal state had a measurable causal
influence on the FEP trajectory. It does not establish that the influence was
helpful.

## Evidence still required

Three questions remain separate:

1. **Mechanism:** Does temporal HDC/CfC state change the FEP trajectory?
   The paired V7 ablation addresses this.
2. **Prediction:** Does the learned world model predict intervention outcomes
   better on frozen prequential holdout data? The V5 adaptive experiment
   addresses this.
3. **Usefulness:** Does the resulting policy improve theory-valid musical
   outcomes, artist workflow, or blinded listener judgment over fixed,
   random-valid, and hand-authored controls? The four-arm experiment addresses
   this.

A positive result on one question must not be reported as a positive result on
another.

## Explicit non-claims

V7 does not show that:

- symbolic sensory proxies are equivalent to hearing rendered audio;
- HDC or CfC improves music;
- active inference outperforms the hand-authored heuristic;
- temporal-difference updates generalize beyond the observed piece;
- IIT, Phi, consciousness level, or free energy measures musical quality;
- a listener will recognize or prefer the selected return;
- the system should receive authority outside the constrained Sonata preview.

## Next gate

Before expanding to other forms, run the following order:

1. compile, format, lint, and test V1 through V7 in the complete workspace;
2. generate frozen same-input replay fixtures on the supported CI platforms;
3. run the sixteen-pair temporal ablation without changing thresholds;
4. inspect negative controls and publish null results;
5. run the prequential world-model holdout;
6. run the blinded four-arm musical and workflow evaluation.

Only then should temporal cognition be generalized to rondo, erosion, lineage,
opera, or ordinary Studio generation.

## V8 confirmatory mechanism gate

V8 adds `temporal_confirmatory`, which assigns related temporal-ablation inputs
to pilot or confirmatory families before analysis. At least twenty-four frozen
confirmatory pairs are required. Paired bootstrap intervals must place sensory
influence above zero while the observed mean clears the existing `0.005`
practical threshold. Mean action divergence must be at least ten percent with a
lower interval bound above zero.

This stronger gate still supports only the statement that temporal state
influenced the FEP trajectory under the frozen Sonata setup.
