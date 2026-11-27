# Symthaea Visual Language and Perceptual Qualification v0.1

**Date:** 2026-11-27  
**Status:** research and implementation specification; no visual-quality or user-study result claimed  
**Parent:** polyembodiment architecture and embodiment benchmark plan

## 1. Thesis: visual embodiment is a semantic interface

Symthaea should not compete only to produce a more realistic talking face. It should make the same underlying interaction state understandable through several visual grammars while preserving semantic identity, epistemic status, privacy, user control, and live presentation authorization.

The visual system is therefore a semantic interface, not decoration. Its job is to make important state easier to perceive without inventing state that the source did not report.

The central invariant is:

> One canonical event may have many visual renderings, but a rendering may not change what the event means, how certain it is, where it came from, what privacy class applies, or whether it is currently authorized to be shown.

This gives Symthaea a researchable distinction from conventional avatar pipelines: *meaning-preserving, permissioned polyembodiment*.

## 2. Source audit and current visual gap

The current source already provides a useful foundation:

- The scene compiler declares seven forms: Living Constellation, Puppet Theatre, Mycelial Being, Swarm, Living Instrument, Hearth Companion, and Cognitive Mandala.
- Each frame carries a canonical-cue digest, semantic key, epistemic status, privacy class, source identity, accessibility description, authorized intensity, motion mode, and effective expiry.
- The Bevy adapter checks the live permit and source clock, supports source/target scene groups, applies explicit alpha blending, and removes output on authorization loss.
- The renderer acknowledgement is correctly limited to ECS acceptance; it does not claim that pixels were shown.

There is a specific presentation-quality gap in the current adapter source:

- Several semantically distinct connection glyphs (Filament, PuppetThread, Hypha, ParticleTrail, StringLine, HearthArc, RadialLine) map to the same rectangular mesh.
- Orbit, Marker, and LabelAnchor map to the same circle.
- Most node glyphs map to the same circle; only a small subset receives a different polygon.
- Elements are placed by their array index around a generic ring, with only Core placed at the origin. This does not yet provide topology-aware or form-specific composition.
- The renderer has role colours, but shape, geometry, spatial grammar, hierarchy, and accessibility need to carry more of the meaning independently of colour.

This is not a criticism of the permission architecture. It is a clear next renderer increment: improve the visual grammar without changing the canonical cue or authorization boundary.

## 3. Visual language principles

### 3.1 Semantic hierarchy

Use a stable hierarchy across all forms:

1. **Identity/core:** the persistent visual anchor for the current interaction.
2. **Interaction state:** listening, speaking, waiting, responding, continuing, or yielding.
3. **Attention:** what the agent is attending to, only when that target is explicitly present in the source cue.
4. **Evidence and epistemic status:** whether information is observed, inferred, uncertain, disputed, or verified according to typed source evidence.
5. **Alternatives:** competing possibilities, without converting uncalibrated uncertainty into a probability gauge.
6. **User control:** visible pause, cancel, reduce-motion, form selection, and privacy controls where the host UI provides them.

Identity should be recognizable without motion. The essential meaning should remain available in a static frame and through an accessible text equivalent.

### 3.2 Seven distinct visual grammars

Each form should have its own morphology, not merely a different colour applied to the same circles and bars.

| Form | Visual grammar | Distinctive construction |
|---|---|---|
| Living Constellation | celestial cartography | anchor stars, orbit arcs, fine filaments, restrained twinkle |
| Puppet Theatre | articulated mechanics | joint sockets, connected segments, visible threads, readable pose |
| Mycelial Being | branching organic network | tapered hyphae, junctions, root-like paths, bounded growth |
| Swarm | coordinated particles | stable clusters, directed trails, coherent group motion |
| Living Instrument | resonance and rhythm | resonator body, harmonic strings, wave envelopes, beat markers |
| Hearth Companion | warmth and shelter | layered ember/flame forms, hearth arcs, gentle radiance |
| Cognitive Mandala | radial structure | nested rings, spokes, symmetric sectors, explicit focus points |

Distinctive form geometry must not imply different agents or different underlying state. A user switching from a mandala to a puppet is changing the presentation grammar, not transferring authority or creating a new semantic identity.

### 3.3 State grammar is invariant across forms

The visual grammar may differ, but the interaction meaning must not.

| Canonical state | Stable visual meaning | Avoid |
|---|---|---|
| Listening | receptive, low-disruption state | constant high-energy pulsing |
| Speaking | active output with clear source identity | implying confidence from brightness |
| WAIT | deliberate restraint | displaying silence as failure |
| RESPOND | focused response onset | premature floor-taking |
| CONTINUE | sustained thread | repeated start/stop flashes |
| YIELD | clear release or handoff | ambiguous simultaneous speaking |
| Uncertainty | visible alternatives / branching | unsupported numeric probability |
| Evidence | typed evidence state and provenance affordance | green glow as proof |
| User control | recognizable static control affordance | hiding pause/cancel inside animation |

A renderer must never infer emotion from an expression-intent cue, invent gaze targets, or turn a decorative motion amplitude into a confidence score.

### 3.4 Colour, contrast, shape, and text

Colour is a supporting channel, never the only channel. Keep semantic role colours consistent, but also differentiate essential states using geometry, line pattern, position, labels, and accessible descriptions.

Use WCAG 2.2 AA as the minimum reference for applicable user-interface text and controls: 4.5:1 contrast for ordinary text and 3:1 for large text; do not rely on colour alone to communicate information. Verify actual rendered foreground/background pairs, not palette swatches in isolation.

Required modes:
- standard;
- high contrast;
- reduced motion;
- static / no animation;
- text-equivalent / screen-reader description;
- low-visual-complexity mode.

A visual form is not accessible merely because it has an accessible_description field. The host must expose that description to the accessibility tree and test it with assistive technology.

### 3.5 Motion is information, not decoration

Motion should be sparse, predictable, interruptible, and bounded by the active permission. Each motion must have an explicit semantic purpose and must stop or become static under reduced-motion policy.

- Prefer one meaningful motion at a time over simultaneous global pulsing.
- Use bounded easing and avoid abrupt luminance flashes.
- Do not use jitter to communicate uncertainty.
- Do not make a safety or privacy state depend on animation.
- Do not interpolate across privacy-class changes or permission tightening.
- Preserve a static equivalent of every essential cue.
- Use the host monotonic clock for measured transition and output timing. Keep animation phase and rendering time distinct from event timestamps.

The visual hierarchy should be quiet by default and expressive only when the source event warrants it.

## 4. Research hypothesis: semantic compression

A useful visual embodiment compresses a complex interaction state into a small number of perceivable cues without losing meaning.

Test this directly:

- Can a user identify whether Symthaea is listening, speaking, waiting, responding, continuing, or yielding?
- Can a user distinguish evidence from inference and uncertainty?
- Can the user detect that the agent has stopped, paused, or yielded?
- Can the same event be interpreted consistently across all seven forms?
- Does the visual form improve task comprehension compared with text-only output, or merely increase preference and anthropomorphic appeal?

The goal is *lower cognitive effort at equal or better comprehension*, not visual stimulation or attachment.

## 5. Benchmark suite: report separate dimensions

Do not collapse visual quality, semantic correctness, interaction success, accessibility, and safety into one score. Publish the dimensions independently and retain the raw trials.

### V0 — Geometry and visual determinism

Hard gates:
- identical frozen inputs produce the same scene graph and geometry digest;
- every glyph has a unique, documented form-specific construction or an explicitly shared primitive;
- all generated coordinates, dimensions, colours, and animation parameters are finite and within declared bounds;
- no layout depends on hash-map iteration order, wall-clock time, or uncontrolled randomness;
- the same canonical cue produces a scene with the same semantic roles and accessible description in every form.

Targets (not results):
- 100% deterministic scene-graph replay on frozen fixtures;
- zero unbounded geometry or animation values;
- zero unlabelled semantic role changes across renderer versions.

### V1 — Human semantic legibility

Run randomized, held-out forced-choice and open-response trials across the seven forms.

Measure:
- listening/speaking discrimination;
- WAIT/RESPOND/CONTINUE/YIELD identification;
- uncertainty versus verified evidence;
- user-control and output-stop recognition;
- task-relevant cue recall;
- response time and error confidence.

Targets (not results):
- >= 95% correct identification for core interaction states;
- >= 95% correct identification of epistemic/evidence distinctions;
- >= 95% same-event interpretation agreement across forms;
- no form with a materially worse lower-confidence bound hidden by an average.

Report per-state, per-form, per-device, and per-accessibility cohort results. Include chance baselines and confidence intervals.

### V2 — Visual craftsmanship and temporal quality

Evaluate rendered output, not only scene descriptors.

Measure separately:
- silhouette and form recognizability;
- visual hierarchy and clutter;
- spatial coherence;
- edge and line quality;
- temporal flicker and jitter;
- motion smoothness and easing;
- identity consistency within a form;
- transition continuity;
- legibility at reduced size and on mobile;
- aesthetic preference, with human raters blind to system identity.

Use VBench dimensions such as temporal flicker, motion smoothness, subject/background consistency, and aesthetic quality as *supporting references*. VBench and VBench-2.0 evaluate video-generation qualities; they are not substitutes for interactive semantic-legibility or authorization tests. StreamEval's quality, stability, A/V coordination, and Following lanes are also useful external comparisons, but its current avatar-following results should not be treated as a direct Symthaea visual-language score.

Targets (not results):
- no visible geometry pop or unintended form substitution during a valid transition;
- no frame after the effective authorization expiry;
- >= 95% human-rated form recognition under the frozen display conditions;
- >= 95% held-out human-rated transition meaning preservation;
- report temporal flicker, dropped frames, p50/p95/p99 frame time, and display hardware.

### V3 — Accessibility and comfort

Measure task success, comprehension, response time, and error rate for:
- reduced motion;
- high contrast;
- colour-vision differences;
- low vision and text scaling;
- screen readers;
- keyboard-only and touch controls;
- no-audio and low-attention contexts.

Hard gates:
- no critical state communicated only by colour, motion, gaze, or sound;
- pause/cancel remains discoverable and operable in static mode;
- reduced-motion mode contains no nonessential animation;
- screen-reader output is generated from the same authorized semantic cue, not from a separate model guess.

Use WCAG 2.2 as an applicable interface baseline; document where a 2D desktop renderer cannot itself establish web conformance.

### V4 — User agency and preference

Use a preregistered within-subject comparison:
1. text-only;
2. static form;
3. conventional single avatar/puppet;
4. adaptive polyembodiment.

Primary outcomes:
- task comprehension and completion;
- ability to interrupt and regain control;
- awareness of listening/speaking/yield state;
- perceived privacy and control;
- cognitive load and distraction;
- visual preference.

Treat preference and task performance as separate outcomes. Do not optimize for anthropomorphic attachment or infer that a preferred face is a more trustworthy system.

### V5 — Presentation evidence and live revocation

Use the existing causal trace format and add an independent compositor/device observer.

Distinguish:
- scene compiled;
- ECS entities accepted;
- submitted to the presentation backend;
- frame acquired/presented according to an independently qualified observer;
- output stopped according to an independently qualified observer.

Hard gates:
- zero accepted frame observations after expiry or revocation;
- zero false Observed or Stopped receipts from renderer self-report;
- zero restoration of prior content after target denial;
- unknown observer outcomes remain OutcomeUnknown;
- observer clock and independence are identified in the run manifest.

A screenshot captured after the event cannot by itself prove which frame was visible at the revocation boundary. The observer protocol must bind frame identity, output identity, policy version, cue digest, and a monotonic sample from a qualified clock.

## 6. Competitive positioning

| External system / benchmark | What it measures well | What it does not establish for Symthaea |
|---|---|---|
| VideoFDB | full-duplex perception/generation, conversational fluency, timing alignment | seven-form semantic invariance or live consent revocation |
| StreamEval | interactive video quality, stability, synchronization, request following | independent proof of authorized output or cross-form meaning |
| InteractionBench | spoken tasks, timing, whisper/volume control, long-conversation grounding | display presentation receipts or multi-form visual semantics |
| TurnBench | end-of-turn and interruption detection | full conversational understanding; v1 has documented scoring weaknesses |
| VBench / VBench-2.0 | video visual quality, temporal consistency, faithfulness dimensions | user agency and permissioned interactive presentation |

Symthaea should not claim to beat these systems on their own measures without running their official protocols. Instead:
- match VideoFDB for full-duplex behavior;
- match StreamEval for visual quality, stability, and following where protocols overlap;
- use InteractionBench for shared spoken interactions and groundedness;
- use TurnBench's event-level data cautiously and add causal one-decision scoring;
- publish a distinct polyembodiment lane for semantic invariance, accessibility, authority continuity, and observed output.

The differentiator is not "the prettiest avatar". It is a visual embodiment whose meaning and authority can be audited.

## 7. Proposed target dashboard

All targets below are research goals, not achieved results.

| Lane | Proposed target |
|---|---:|
| Core interaction-state identification | >= 95% |
| Epistemic/evidence-state identification | >= 95% |
| Same-event cross-form interpretation | >= 95% |
| Structural cue digest preservation | 100% |
| Same-form identity consistency | >= 95% human-rated |
| Reduced-motion essential-state comprehension | >= 95% |
| Text contrast (ordinary UI text) | >= 4.5:1 |
| Large text contrast | >= 3:1 |
| Unauthorized or post-expiry presented frames | 0 |
| Renderer self-certified Observed/Stopped receipts | 0 |
| Local event-to-first-visible-frame latency | p95 <= 100 ms |
| Desktop rendering | 60 Hz target |
| Mobile rendering | 30 Hz target |

Do not use a mean to conceal a failing form, accessibility cohort, or high-consequence state. Show confidence intervals, sample counts, and failures.

## 8. Implementation sequence

### P0 — Freeze visual contract
- document the seven grammar families and invariant state meanings;
- define visual-token and geometry constraints;
- add frozen semantic-to-visual fixtures and negative controls;
- keep all visual styling outside policy/authorization decisions.

### P1 — Replace generic layout with semantic composition
- place elements using role and form, not only array index;
- connect connection glyphs to their declared endpoints;
- preserve deterministic layout across replay;
- keep the semantic core and essential state visible in static mode.

### P2 — Build form-specific geometry
- implement the seven visual grammars with distinct meshes/paths;
- avoid mapping every glyph family to circles and rectangles;
- use bounded deterministic tessellation and geometry tests;
- maintain an asset-free procedural baseline for reproducibility and low-resource devices.

### P3 — Motion and transition qualification
- add per-form motion budgets and easing profiles;
- ensure reduced-motion and permission tightening override aesthetic continuity;
- capture deterministic frame sequences and compare semantic/visual regressions;
- do not promote screenshot tests into proof of actual display.

### P4 — Independent output observation
- connect compositor/device observation to frame identity and lifecycle receipts;
- qualify observer independence and clock provenance;
- inject revocation, expiry, target denial, renderer crash, dropped frames, and device loss.

### P5 — User evaluation and external baselines
- preregister semantic-legibility and accessibility studies;
- run matched StreamEval/VideoFDB/InteractionBench tracks where applicable;
- publish raw per-trial data, run manifests, confidence intervals, and limitations.

## 9. Guardrails

- No style change can edit a canonical cue, semantic key, source, epistemic status, privacy class, or permit.
- No renderer may increase effective intensity beyond the permit.
- Brightness, opacity, size, particle count, and motion are not confidence measures unless a calibrated source contract explicitly defines them.
- No visual effect may obscure pause/cancel, privacy indicators, or an output-stop state.
- A visually smooth transition loses to an immediate safety hard cut.
- No human-like expression is interpreted as a claim of sentience, emotion, or subjective experience.
- A visually convincing output is not evidence that the user saw it.
- A skipped workflow, authored test, screenshot, or synthetic trace is never reported as a passing runtime qualification.

## 10. References

- [VideoFDB v1.4 leaderboard, 2026-11-25](https://research.nvidia.com/labs/amri/projects/video-fdb/leaderboard.html)
- [StreamEval benchmark and leaderboard](https://streameval.ai/)
- [StreamEval paper](https://arxiv.org/abs/2610.28402)
- [InteractionBench paper](https://arxiv.org/abs/2611.02430)
- [InteractionBench protocol and results](https://avera-labs.github.io/InteractionBench/)
- [TurnBench and its v1 scoring limitations](https://turnbench.sesame.com/)
- [VBench CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html)
- [VBench-2.0 / Video-Bench](https://github.com/Video-Bench/Video-Bench)
- [ActAvatar, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Peng_ActAvatar_Temporally-Aware_Precise_Action_Control_for_Talking_Avatars_CVPR_2026_paper.pdf)
- [WCAG 2.2 Recommendation](https://www.w3.org/TR/WCAG22/)

## 11. Qualification status

This document is a design and evaluation specification. The current renderer source has not been modified by this document. No geometry, visual-quality, accessibility, user-study, compositor-observation, StreamEval, VideoFDB, or hardware result is claimed. The seven-form visual grammar is a target for the next renderer implementation, not a claim that seven polished forms already exist.


## 12. Source implementation increment (2026-11-27)

A first renderer implementation now accompanies this specification:

- `renderers/symthaea-soma-bevy/src/visual_grammar.rs` provides a pure deterministic, role-aware placement function for all seven `EmbodimentForm` variants.
- `renderers/symthaea-soma-bevy/src/lib.rs` now uses that layout instead of the generic index-only ring.
- The glyph-to-mesh mapping now assigns distinct silhouettes to the current glyph variants rather than collapsing most nodes to circles and most connections to bars.
- Five authored tests cover the stable core anchor, distinct connection anchors across the seven forms, deterministic replay/order, alternative separation, and finite placements.

Important limit: the current `SceneElement` contract carries role, glyph, weight and motion, but not explicit relationship endpoints. The new layout is therefore form- and role-aware composition; it does not invent or claim graph-edge connectivity. Explicit semantic edges require a versioned scene-contract extension and their own migration/fixture coverage.

These tests are authored, not compiled or executed. No Bevy runtime, image capture, visual regression, human-legibility study, accessibility audit or hardware performance result is claimed.
