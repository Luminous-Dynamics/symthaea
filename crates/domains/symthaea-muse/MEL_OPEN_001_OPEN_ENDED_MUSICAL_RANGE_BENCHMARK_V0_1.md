# MEL-OPEN-001: Open-Ended Musical Range and Combinatorial Generalization Benchmark v0.1

**Status:** Research contract / proposed benchmark. Not an implementation, benchmark result, or claim of musical quality.  
**Scope:** Public Symthaea / Melothaea (Muse composition and rendering path).  
**Relationship:** Complements [Muse style system v2, issue #7417](https://github.com/Luminous-Dynamics/symthaea/issues/7417), [instrument support, issue #7416](https://github.com/Luminous-Dynamics/symthaea/issues/7416), [style expansion PR #7764](https://github.com/Luminous-Dynamics/symthaea/pull/7764), [instrument capability inventory PR #7464](https://github.com/Luminous-Dynamics/symthaea/pull/7464), and [musician-level performance profiles PR #7714](https://github.com/Luminous-Dynamics/symthaea/pull/7714). It does not replace, stack onto, or advance those branches.

## 1. Research question

Can Melothaea grow beyond a fixed catalog by composing validated musical capabilities in combinations that were not explicitly enumerated or hand-wired—while retaining structural validity, useful controllability, playable realization, traceable provenance, and listener value?

The hypothesis is **capability-first, open-world composition**:

> The system should grow mainly by learning or adding composable musical capabilities—not by continually adding one more hard-coded genre or instrument enum variant.

A style label is a useful user interface, not a sufficient implementation. An instrument name is identity metadata, not proof that a sample or renderer exists. A generated composition is an output, not proof that a requested idea was realized.

This is an evaluation and architecture proposal for combining existing work, not a duplicate style-profile implementation. Issue #7417 already proposes versioned multidimensional recipes and lineage; issue #7416 covers source-instrument identity and rendering coverage. This contract adds the cross-cutting question: **how do we measure growth in the system's ability to form new, useful combinations?**

## 2. Can the range be infinite?

**Not literally as a claim about a deployed finite computer, renderer, or validated instrument set.** The actual synthesis, compute, numeric representations, available assets, and runtime are bounded.

But the catalog need not be a closed list. Melothaea can approach open-ended range if its architecture can accept new profiles and instrument adapters and compose them through reusable, validated rules. An open schema can support arbitrarily many user-authored profile records over time; a compositional grammar can express many combinations never named in advance. Every particular generated artifact still has finite resources and explicitly bounded behavior.

Treat “infinite range” as three measurable engineering objectives:

1. **Unbounded extension surface:** no source-code edit or new enum variant is required for a new profile when its semantics fit an existing declared schema and the necessary renderer capability is available.
2. **Combinatorial generalization:** hold out combinations of known components and measure whether the composer can synthesize them without bespoke per-combination code.
3. **Evidence-preserving novelty:** accept new combinations only with receipts showing what was requested, which constraints were satisfied, what rendered, and where substitution or uncertainty occurred.

Novelty alone is not success. Melothaea must explore a growing space while remaining useful, controllable, and honest.

## 3. Paradigm shifts

### 3.1 From style catalog to executable capability graph

Represent capabilities as reusable, versioned nodes and typed relations instead of making every combination a new style variant.

Candidate node families:

- **Structure:** form, section roles, phrase grammar, recurrence, transformation, cadence, thematic memory.
- **Pitch:** tonal/modal systems, tuning, register, pitch hierarchy, ornament constraints.
- **Time:** meter, cycles, subdivision, accent maps, swing, rubato, phrase-level timing.
- **Texture/orchestration:** roles, polyphony, density, role-to-part assignment, doubling, register allocation.
- **Instrument capability:** source identity, playable range, polyphony, sustained/struck/plucked/noise behavior, articulation set, control dimensions, available renderer/sample/SoundFont paths, and provenance.
- **Performance:** articulation, timing and velocity models, expressive controls, performer/ensemble constraints.
- **Acoustic production:** spatial model, dynamics, room, effects, mix and delivery target.
- **Intent:** mood, energy, motion, tension, intimacy, density and functional use.

A profile can compose capabilities across these dimensions without claiming a new historical genre. A historical/tradition label remains an attributed metadata claim with the review requirements already described by issue #7417.

Each capability should state its preconditions, produced guarantees, resource bounds, conflicts, and evidence type. Hard constraints are validated before rendering; advisory preferences guide search without masquerading as guarantees.

### 3.2 From one-shot generation to structure-first, constraint-aware search

Generate a symbolic plan first. Validate form, motif behavior, phrase continuity, harmony/pitch logic, and requested transitions before assigning orchestration and detailed performance. Then realize each role using currently available instruments and renderers.

This is a debugging advantage: failed structure, infeasible instrumentation, and poor sound are separately attributable. Search and ranking can then use a multi-objective frontier rather than collapse creativity, correctness, and taste into one opaque score.

### 3.3 From a fixed enum to manifest-driven extension

Where appropriate, let new profiles and source banks be registered through versioned data rather than code changes. An instrument descriptor is not automatically executable: it must bind to a tested procedural renderer, a verified sample mapping, a declared external preset, or an explicit unavailable state.

A new profile may extend descriptive identity without extending sound rendering. These are separate accomplishments.

A capability receipt should distinguish at least:

- `described`
- `structurally_executable`
- `renderer_available`
- `asset_present`
- `asset_decoded`
- `rendered`
- `signal_checked`
- `perceptually_reviewed`
- `tradition_reviewed` (only when applicable)

These are separate evidence states, not a single ladder where one successful stage proves all later ones.

### 3.4 From a single creativity score to a Pareto frontier

Report correctness, user-control fidelity, novelty, diversity, usefulness, rendering fidelity, provenance, resource use, and human preference separately. Search should expose the tradeoff between novelty and utility and let users steer it.

Use automatic metrics to detect exact defects and support controlled comparisons, not to replace musicians' judgements about artistry. Preserve disagreement in aesthetic ratings rather than smoothing it away as measurement error. For tradition-specific work, distinguish technical execution from attribution/authenticity review and rights.

## 4. MEL-OPEN-001 benchmark protocol

### Split A — Compositional generalization

Build a deterministic test grid from **components, not named genre labels**. Each held-out case should combine components that have individually appeared in development, while the exact combination is withheld.

Vary one or more of:

- structural grammar × groove;
- structure × pitch language/tuning;
- ensemble roles × instrument profiles;
- performance dialect × structural style;
- acoustic/production profile × rendering path;
- three-way combinations across structure, time, and instrumentation.

Freeze data splits, source/profile digests, engine commit, seed set, candidate budget, and evaluation rules. Avoid leakage by verifying that a supposedly held-out combination does not appear verbatim in profile examples or fixtures. Report each family separately; do not let easy families hide failures on rare or under-supported ones.

### Split B — Extension without bespoke code

Feed a valid, previously unseen manifest/profile into the registry. Measure whether it can be parsed, validated, composed, rendered (where a compatible renderer exists), and audited without adding a hard-coded branch for that specific profile.

Separate tests that add a **new composition profile** from tests that add a **new rendering capability**: the former should not imply the latter.

### Split C — Controlled edits

For each case, change one requested dimension while freezing unrelated variables and seeds. Examples: change only instrument assignment; change only groove; change only phrase-development behavior; change only the room profile.

Report both:
- **target sensitivity:** did the intended, measured feature move in the requested direction?
- **non-target invariance:** did unrelated features remain stable within an explicitly declared tolerance?

Do not claim invariance where a changed instrument's range or articulation legitimately forces a disclosed rewrite.

### Split D — Instrument and renderer truthfulness

For each selected part, require a receipt binding the requested identity to the chosen render path, supported range/articulations, substitutions, unavailable assets, and any lossy conversion. Exercise missing assets, corrupt samples, absent SoundFont programs, unknown profile IDs, unsupported articulations, and unsupported source semantics as negative controls.

Import/export round trips should preserve every represented instrument assignment and timed change exactly. Unsupported semantics must be preserved, explicitly rejected, or handled via a visible user-approved downgrade—not silently dropped.

### Split E — Blind listening and practitioner review

Use randomized, matched prompts and equal candidate budgets. Evaluate style-family recognition, prompt/intent adherence, perceived musical coherence, novelty/usefulness, and preference against a declared baseline. Keep the one-shot result separate from best-of-2/4/8; do not compare cherry-picked best-of-N outputs to a one-shot baseline.

For culturally specific traditions, use appropriate practitioner review and provenance disclosure. Technical benchmark scores must not be promoted as a claim of authenticity.

## 5. Proposed benchmark targets

These are **proposed targets, not measured results**. Run a baseline first, publish exact failures, and revise thresholds only through a documented version change—not to make a failed run pass.

| Measure | Initial target | Guardrail |
|---|---|---|
| Hard structural constraints | 100% of outputs accepted as valid | Invalid candidates fail closed with attributable reasons |
| Determinism | 100% same-input/same-version/same-seed symbolic and receipt identity | Record external renderer and asset versions separately |
| Held-out recombination | ≥95% hard-constraint satisfaction on each declared test family | Publish counts and every exception; no aggregate-only pass |
| One-factor control response | ≥90% of preregistered directional probes meet their target | Report response magnitude and non-target drift per feature |
| Orchestration-only edit | 100% preserve canonical symbolic skeleton when no instrument constraint requires rewriting | Any allowed rewrite is explicit and reason-coded |
| Capability / fallback accounting | 100% of assigned parts have a truthful render-path status | Zero silent substitution or silent loss of source semantics |
| Supported import/export round trips | 100% exact preservation of represented assignments and event times | Unsupported fields remain explicit failures or explicit loss receipts |
| Manifest extension | No code change for schema-expressible profile registration | A descriptor alone does not count as audio capability |
| Style-family listening | ≥0.80 broad-family macro-F1 on the reviewed benchmark subset, as a stretch target | Freeze evaluation protocol; report confusion and chance context |
| Human pairwise preference | >60% over the strongest matched baseline, with 95% CI lower bound >50% | Blind, held-out prompts/listeners, fixed candidate budget |
| Novelty/usefulness | Publish a novelty–utility Pareto curve, not a single score | Include exact/near-duplicate detection and listener utility |
| Runtime and resource use | p50/p95 latency, memory, score size and audio duration for 8/16/32-bar cases on named hardware | Publish cost per accepted distinct result and failures |

The style-family target intentionally starts below perfect classification: styles can share real ancestry or blend. Confusion matrices and structural provenance are more informative than demanding every label be perfectly separated. This complements rather than replaces the style-contract measures already proposed in PR #7764.

## 6. How to measure open-ended growth

Do not use the number of catalog rows as the headline metric. Publish an **Open-Ended Range Report** with:

- number of registered component profiles by capability family and evidence state;
- number of valid combinations exercised, split into seen and withheld combinations;
- distinctness after canonical structural fingerprinting, plus duplicate/near-duplicate rates;
- constraints met, failures, substitutions, and unsupported capability requests;
- novelty versus utility and preference on the Pareto frontier;
- new-manifest onboarding cost: code edits, manual steps, time, and evidence produced;
- p50/p95 compute/resource cost per accepted, structurally distinct output;
- human-reviewed areas and areas with no listener/practitioner evidence.

A simple multiplicative “number of styles × instruments” estimate will be misleading because many combinations conflict, map to the same behavior, or cannot render. Report valid, distinct, supported combinations separately from theoretical combinations.

## 7. Minimum deterministic suite before human trials

1. **Pair recombination:** withhold pairings across at least four axes while ensuring each individual component is represented.
2. **Three-way transfer:** withhold structure × rhythm × instrumentation combinations.
3. **Unknown profile:** confirm that a new schema-valid profile can register without an enum edit, and that unsupported execution fails visibly.
4. **Renderer substitution:** remove one sample/preset and confirm a visible unavailable/substitute receipt, never a false success.
5. **Control sensitivity and invariance:** change one dimension at a time with frozen seeds and compare the declared feature vector.
6. **Negative controls:** invalid grammar composition, contradictory constraints, missing capabilities, malformed manifests, sample decoding failure, unsupported source event, and deterministic receipt mismatch.
7. **Novelty controls:** exact duplicate, transposition-only copy, instrumentation-only copy, structure-preserving paraphrase, and genuinely changed formal development should be reported separately.

Every report binds source commit, benchmark version, profile/asset digests, seeds, command, exit status, and complete failure details. Queued workflows, authored tests, or static inspection are not runtime passes.

## 8. Staged implementation

**M0 — Baseline (read-only inventory).** Join existing style-routing, source-instrument, sample-capability, and performance-profile inventories. Emit a deterministic cross-axis map and identify capability families that only exist as labels, source records, or theoretical proposals.

**M1 — Benchmark harness.** Add fixtures for controlled one-factor changes, held-out combinations, negative controls, and machine-readable receipts. Do not expand the public catalog yet.

**M2 — Capability compiler prototype.** Choose a small verified set of existing components and resolve a MusicRecipeV1-like request into a typed execution plan, using the profile architecture in issue #7417 rather than creating a parallel taxonomy.

**M3 — Open extension.** Test manifest-based profiles and an instrument capability adapter through the existing versioned instrumentation-sidecar work in issue #7416. Keep source identity, renderer choice, and acoustic evidence distinct.

**M4 — Multimodal evaluation.** Capture deterministic symbolic outputs and renders, then run blind preference and style-family evaluation with a frozen candidate-selection protocol.

**M5 — Expand based on frontier gaps.** Add capability primitives where tests reveal missing expressive possibilities, not just because a genre/instrument list looks short.

## 9. External research and standards

- [MIREX 2026: Symbolic Music Generation](https://www.music-ir.org/mirex/wiki/2026%3ASymbolic_Music_Generation) uses prompt-conditioned continuation and blind subjective ratings of coherence, creativity, structuredness, and musicality. This is useful for one constrained task, but it does not alone test open-world compositional generalization.
- [MIREX 2026: CMI-RewardBench results](https://music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench_Results) evaluates automatic preference metrics against human preferences. Its reported performance varies across instrumental and vocal subsets, supporting separate subdomain reporting rather than reliance on one evaluator score.
- Schapiro et al., [*Combinatorial Creativity: A New Frontier in Generalization Abilities* (2025)](https://arxiv.org/abs/2509.21043), formalizes novelty/utility trade-offs for open-ended generative tasks. Transfer to music is a research hypothesis to test, not established proof for Melothaea.
- Grötschla et al., [*Benchmarking Music Generation Models and Metrics via Human Preference Studies* (ICASSP 2025)](https://doi.org/10.1109/ICASSP49660.2025.10887745), reports human preference studies across generation systems and highlights the challenge of mapping automatic metrics to listener preferences.
- The [MIDI Association MIDI 2.0 overview](https://midi.org/midi-2-0) describes capability inquiry, profiles, property exchange, increased resolution and per-note expressive control, which are relevant interoperability references. MIDI 2.0 is a useful device interoperability layer, not a replacement for Melothaea's internal compositional and acoustic capability model.
- Existing project contracts: [MEL-005A / PR #7230](https://github.com/Luminous-Dynamics/symthaea/pull/7230), [style system v2 / issue #7417](https://github.com/Luminous-Dynamics/symthaea/issues/7417), [instrument support / issue #7416](https://github.com/Luminous-Dynamics/symthaea/issues/7416), [style expansion and contract benchmark / PR #7764](https://github.com/Luminous-Dynamics/symthaea/pull/7764), [instrument capability inventory / PR #7464](https://github.com/Luminous-Dynamics/symthaea/pull/7464), [musician benchmarks / PR #7714](https://github.com/Luminous-Dynamics/symthaea/pull/7714).

## 10. Decision

Treat “infinite range” as **open-ended, combinatorial, capability-backed extensibility**. The first productive step is to benchmark the ability to recombine existing capabilities and to onboard new ones honestly. More style labels should follow evidence that new rules or capabilities create valid and meaningfully different music—not precede it.

No benchmark results are claimed by this document. No runner state, authored test, or static audit can substitute for exact-head execution or blind human evidence.
