# Piano composition: plan-to-score obligation audit

**Status:** research/design addendum, 2026-11-01. This document proposes an independently verifiable composition-evidence layer. It does not claim that the current generator has achieved the proposed results, and it does not introduce a new composer or a universal aesthetic score.

## Why this is the next useful layer

The piano excellence programme (#9729) already separates composition, performance, acoustic realization and human listening. The wider Muse programme also has:

- first-class grammar profiles and plan evidence in `symthaea-music-theory::grammar`;
- structural traces and composition fingerprints, including motif-return features;
- intent-conditioned quality-diversity work (#7933);
- causal musical-cognition work (#3898);
- performance-representation fidelity work (#9476), plus meter/time-signature migration (#9473).

We should connect these pieces rather than create a competing planner, fingerprint, or listening harness.

The next seam is **stronger plan-to-score accountability**, but it is important not to duplicate what already exists. A source audit found that Muse already has:

- `GrammarStructuralTrace` and composer-authored motif, cadence, structural-span and obligation-transition assertions;
- protocol-level `ComposerStructuralTrace`, `TraceVerificationReport`, and `MotifEvidenceReconciliation`;
- `Analyst::verify_composer_trace`, which checks exact event references, motif-family/region bindings, event order, approximate contour/rhythm distance, cadence-arrival markers, and obligation lifecycle/evidence references;
- `Analyst::reconcile_motif_evidence`, which reconciles composer-asserted motif occurrences with the independent score-window scanner by shared event identities;
- `ObligationKind` and `ObligationLedger` in the music-theory crate, plus `MotifReturnEvidence` for transformation-aware symbolic comparison.

That is a meaningful existing foundation. The missing work is not another COE record format or another generic verifier. It is to strengthen the semantics of the existing Analyst path so that a valid-looking assertion cannot be mistaken for proof that the promised musical event actually occurred.

This is not an attempt to formalize beauty. It is a way to make selected compositional claims more inspectable and falsifiable while leaving artistic judgment to listeners.

## Fresh benchmark landscape: MIREX 2026

The MIREX 2026 piano-continuation task asks systems to continue a four-measure piano prompt with twelve measures, producing a sixteen-measure piece. Its published results now include four systems:

| System | Prompt coherence | Structure | Creativity | Musicality |
|---|---:|---:|---:|---:|
| DAMN (Nested Transformer) | 3.68 ± 0.16 | 3.39 ± 0.18 | 3.18 ± 0.16 | 3.47 ± 0.14 |
| Anticipatory Music Transformer | 3.47 ± 0.20 | 3.34 ± 0.16 | 3.34 ± 0.16 | 3.61 ± 0.17 |
| PhraseSplitGen | 2.95 ± 0.18 | 2.74 ± 0.15 | 2.89 ± 0.15 | 2.61 ± 0.14 |
| MuseCoco (1.2B) | 2.79 ± 0.17 | 2.76 ± 0.15 | 3.42 ± 0.12 | 2.68 ± 0.16 |

The reported values are means ± standard error. Under the report's corrected Wilcoxon comparison, DAMN and Anticipatory Music Transformer share the higher significance group for prompt coherence, structure and musicality; all four systems share the same significance group for creativity. The evaluation used eight prompts across pop, classical and jazz, 12 participants, 38 ratings per model, and a process in which one candidate was cherry-picked from eight per prompt. These results give us a concrete task-matched comparison set, but the small prompt/participant panel and best-of-eight selection mean they should be treated as a task-specific reference, not a ranking of general compositional ability.

**Implication:** Symthaea should reproduce this task with disclosed candidate budgets and report both first-candidate and best-of-N performance. It must also run a separate no-cherry-picking, long-form study. A 16-bar continuation score cannot establish 32/64-bar thematic development.

Sources:
- MIREX 2026 task: https://www.music-ir.org/mirex/wiki/2026:Symbolic_Music_Generation
- MIREX 2026 results snapshot (2026-10-31): https://futuremirex.com/mirex/wiki/2026:Symbolic_Music_Generation_Results

## Research insight: representation, composition, and performance are different problems

Three current comparisons should not be collapsed into one leaderboard:

1. **Composition:** MIREX 2026 measures short-prompt continuation and rates coherence, structure, creativity and musicality.
2. **Performance:** Pianist Transformer (ICML 2026) is a large-scale expressive piano-performance renderer, reporting pretraining on 10B MIDI tokens. It is a performance baseline, not a symbolic-composition baseline: https://proceedings.mlr.press/v306/you26d.html
3. **Representation:** Agogic (2026) reports a controlled study where performance-resolution symbolic tokens (10 ms timing, per-note velocity and multi-track texture) substantially improve distributional FMD versus beat-grid representations. The reported FMD difference is not a human preference or beauty result: https://arxiv.org/abs/2608.03999

For expressive performance, a DAFx 2026 study also reports that adding velocity and sustain-pedal information improves FAD against a note-only representation on its setup: https://cnmat.berkeley.edu/publications/evaluating-tokenization-strategies-expressive-classical-piano-performance-generation

MAESTRO remains a useful aligned piano performance corpus (approximately 200 hours; around 3 ms MIDI/audio alignment), but its CC BY-NC-SA 4.0 terms must be respected. Dataset lineage, source-work partitions and license-compatible use must be frozen in the benchmark manifest: https://magenta.withgoogle.com/datasets/maestro

**Architectural consequence:** retain separate identities for notated beat time, performed event time, and rendered sample time. Do not replace the current exact symbolic score with a 10 ms lattice merely because one paper found a distributional advantage for a performance-token model. Benchmark the conversion and round-trip error first (#9476).

## Proposed extension: strengthen the existing Analyst, do not add a second evidence system

The current verifier already performs valuable independent checks, but the implementation leaves several semantic gaps that should become explicit research targets.

### Verified foundations

The present `verify_composer_trace` checks:
- that asserted event IDs exist in the exact `ListenCompositionBundle`;
- that structural span endpoints are ordered and parent spans exist;
- that motif families and formal regions exist, that motif events exist and are ordered in one voice, and that claimed distances are in range;
- a recomputed contour/duration-shape distance against the declared motif occurrence;
- that a cadence arrival event is marked cadential and that its evidence event IDs exist;
- that obligation lifecycle transitions are continuous, responsible passes are named, and a fulfilled transition cites at least one existing score event.

The existing motif reconciler then compares asserted motif occurrences to independently scanned motif occurrences. Keep this mechanism and its explicit states (matched, not recovered, inferred but not asserted, ambiguous, rejected).

### Semantic gaps to investigate and close

1. **Obligation type is not bound into the protocol transition.** The current `ObligationTransitionTrace` carries an ID, lifecycle state, event IDs, responsible pass and optional transformation string, but not the typed `ObligationKind` or the corresponding immutable ledger definition. Consequently the verifier can establish that a fulfilled transition cites real notes, but cannot establish that those notes satisfy a `ReturnMotif`, `ReachKey`, `ReachClimax`, `ResolveAlteredDegree`, or other promised condition.

2. **Cadence type is not independently re-derived.** The current verifier checks that the arrival event has a cadential marker and that evidence references resolve. Its own report explicitly discloses that harmonic cadence type is not re-derived. A composer could therefore label a cadence “authentic” while the actual harmonic preparation/arrival does not support that label.

3. **Motif claims are not yet fully transformation-verified.** The verifier checks an approximate structural-distance measurement and literal-distance consistency, but does not fully recompute every declared item in `transformation_chain` against the source motif and exact target notes. The separate `MotifReturnEvidence` logic is available and should be integrated or compared without treating its symbolic similarity as proof of human recognition.

4. **Trace validity and score realization must remain separate.** An event reference being valid is not evidence that the event is musically relevant to the obligation. Likewise, shared note IDs between an assertion and a scanner result should not by themselves establish semantic equivalence if the transformation or expected relationship differs.

5. **Coverage must be explicit.** The report should distinguish verified, contradicted, ambiguous, and insufficiently evidenced obligations. Unsupported grammar semantics should yield an explicit non-pass/unknown state, not be silently accepted or forced into tonal rules.

### Minimal compatible design

Extend existing protocol/Analyst contracts additively where needed; do not introduce a parallel `CompositionObligationEvidenceV1` container.

- Bind each transition to its typed obligation definition (or to an immutable, score-bound ledger entry whose canonical digest and full definition are available to the verifier).
- Keep the exact score digest, plan/trace provenance, stable event IDs and existing `TraceVerificationReport` as the evidence envelope.
- Re-derive obligation-specific conditions from exact score facts and the grammar-specific verifier. For a motif return, compare the promised source material with the cited target span under the requested transformation. For a destination key, inspect the declared tonal evidence in the destination region. For a cadence, inspect the harmonic and melodic evidence rather than trusting the arrival label.
- Reuse `MotifReturnEvidence`, `ObligationKind`, `ObligationLedger`, and existing protocol reconciliation types wherever their semantics fit.
- Version any wire additions and preserve backward compatibility: legacy traces that do not carry the typed obligation definition must be reported as insufficient evidence for semantic fulfilment, not grandfathered into verified status.
- Preserve grammar-specific domains. A drone/process work should not fail because it lacks an authentic cadence; an intentionally open form should not be required to return its opening motif.

The verifier's output remains mechanical evidence about declared score properties, not an aesthetic score and not an assertion that a listener will recognize or enjoy the motif.

## Counterfactual composition experiment

The key experiment is a paired intervention:

**same intent + brief + seed + candidate budget + grammar + realization settings; change one declared compositional mechanism.**

Start with a small set of interpretable interventions:

1. motif memory enabled vs. removed;
2. planned transformation vs. identity-preserving repetition;
3. planned return vs. no return;
4. phrase/cadence trajectory enabled vs. neutralized;
5. long-range form plan enabled vs. locally valid phrases assembled without that plan.

For each pair, verify:

- the intended obligation changes in the score;
- protected obligations remain satisfied;
- no unrelated score field changes without an explicit budget;
- the score and plan digests bind to the exact comparison;
- candidate and compute budgets are matched.

This tests whether the architecture's compositional machinery has a causal, observable effect. It does **not** prove that the intervention makes the music more beautiful. That second question requires independent listening.

### Required negative controls

- A plan claims a motif return but the score omits it.
- A motif is copied with no declared transformation but labeled “inverted” or “fragmented.”
- The section labels are changed while notes remain byte-identical.
- A score is altered after the plan digest is sealed.
- An evidence span points to the wrong part or outside the score.
- A different seed, score, or motif is substituted while the receipt retains the original identity.
- A valid non-tonal/process work is incorrectly rejected for lacking a tonal cadence.
- A no-return grammar is falsely required to provide a return.

The target is zero false certifications on the frozen adversarial fixture set, not a claim of zero risk on all possible music. Publish each fixture and rejection reason.

## Evaluation design

### Task families

- Official MIREX-parity 4+12 measure continuation.
- Original 32-bar pieces.
- Original 64-bar works with at least three sections and a declared long-range identity plan.
- Grammar-diverse controls: period/sentence, contrapuntal, ground variation, call-response, groove cycle, modal arc, additive process and ambient/textural forms.

### Arms

At matched generation and selection budgets:

- current canonical Muse generator;
- explicit grammar-plan path;
- plan path with one obligation mechanism ablated;
- plan path with the strengthened existing Analyst verifier enabled;
- available MIREX systems/organizer outputs where reproducible and lawfully accessible.

Report first-candidate results separately from best-of-N. If using eight candidates to mirror MIREX, disclose all eight and do not quietly compare Symthaea's best-of-eight with a baseline's first output.

### Mechanical measurements

- obligation verification rate by grammar and obligation type;
- false-certification and false-rejection rates on positive/negative fixtures;
- score/plan identity binding and deterministic replay;
- motif occurrence and transformation evidence;
- section-level contrast/return evidence;
- harmonic/voice-leading invariants only where the selected grammar requires them;
- originality/near-duplicate screening against the frozen, rights-cleared corpus;
- rejection reasons, completion rate, candidate count, wall time and resource use.

Keep validity, originality, intent adherence, diversity and artistic preference as separate axes.

### Human measurements

Use blinded, randomized, level-matched pairs with ties and “neither” options. Ask separately about phrase direction, thematic identity, coherence, surprise, memorability, emotional consequence and desire to listen again. Stratify experienced pianists/composers from general listeners; include hidden repeats and a delayed return-listening task. Freeze the protocol and perform power/precision analysis before recruiting the confirmatory panel.

Automated FMD/FAD/CLAP or the symbolic verifier may explain specific properties. None is a substitute for listeners.

## Initial research gates (targets, not results)

- 100% of required obligations in the frozen positive fixtures are found.
- 100% of preregistered malformed/contradictory evidence fixtures are rejected.
- Zero false certifications on the frozen adversarial panel.
- Every declared obligation resolves to exact score evidence or an explicit non-pass status.
- Identical inputs, code, corpus and candidate budget reproduce identical mechanical reports.
- Composer/work/source-disjoint evaluation; all exclusions and near-duplicate alerts disclosed.
- Human pairwise preference target remains >60% against the strongest matched baseline, with the 95% confidence-interval lower bound above 50%, replicated on a sealed split. This is an ambitious research target, not an achieved score or universal beauty test.

## Implementation sequence

1. Extend the existing `ObligationTransitionTrace`/ledger linkage so the verifier can see the typed obligation definition, with a versioned additive migration.
2. Add obligation-specific score re-derivation in `Analyst::verify_composer_trace`; use exact score events and existing motif-return/cadence primitives.
3. Add deterministic positive and adversarial fixtures, including fulfilled-but-irrelevant events, false cadence labels, and false transformation labels.
4. Reuse the existing `TraceVerificationReport` and `MotifEvidenceReconciliation`; do not create a second evidence container or listening harness.
5. Add paired structural interventions to the existing listening pack and the 32/64-bar piano brief suite with exact manifest/seed/candidate accounting.
6. Connect to #9729's blind listening protocol only after mechanical obligation semantics and artifact lineage are stable.

No training run, renderer capability, compile, test, or listener result is claimed by this design note. The next source PR should be narrow and should not be opened as merge-ready until exact-head Rust checks and independent fixture evidence exist.


## Implementation handoff

The implementation work is tracked separately in [#9797 — Verify typed compositional obligations against realized scores](https://github.com/Luminous-Dynamics/symthaea/issues/9797). The issue turns the audit findings into explicit work packages and adversarial fixtures. This document remains a design addendum; #9797 is the source implementation target.

## Research and implementation continuation — 2026-11-01

### Evaluator reliability is a separate problem

MIREX 2026 now includes a Music Evaluation via CMI-RewardBench task that tests whether automatic music evaluators agree with human pairwise choices. Its published result page reports a 249-pair core set and a 618-sample extended set with vocal/instrumental subsets. The top listed CloudEar-overall evaluator reports 73.9% core accuracy and 72.6% on the instrumental extension. These are broad music evaluator results, not piano-only calibration evidence. The ICML 2026 CMI-RewardBench paper adds CMI-Pref human annotations and reward models; its evaluation must be treated as an evaluator baseline, not listener ground truth.

Use this as a new research layer: independently test all automated judges against a piano-only, composition-disjoint, blinded human panel. Keep musicality, intention adherence, performance plausibility and audio integrity separate. Evaluator outputs remain diagnostic and may not become a live generation reward.

Sources:
- https://music-ir.org/mirex/wiki/2026:Music_Evaluation_via_CMI-RewardBench_Results
- https://proceedings.mlr.press/v306/ma26am.html
- https://github.com/Haiwen-Xia/CMI-RewardBench

Implementation: [#9832 — score-grounded typed obligation verification](https://github.com/Luminous-Dynamics/symthaea/pull/9832). Evaluator follow-up: [#9833 — piano evaluator reliability against human preferences](https://github.com/Luminous-Dynamics/symthaea/issues/9833).

### Industry scope boundary

Qwen-Music (2026) is a substantial end-to-end audio/song-generation comparator. Its authors describe a semantic tokenizer, music LLM with melody planning, a generative renderer, training on more than five million hours of multilingual music, and evaluations against current proprietary text-to-music services. It is not a like-for-like symbolic solo-piano composer or expressive piano-performance model. Compare it in a separate end-to-end audio/instruction lane; do not conflate whole-song generation with piano score quality.

Source: https://arxiv.org/abs/2607.11699

MAESTRO v3.0.0 is the recommended aligned piano-performance corpus baseline where its CC BY-NC-SA 4.0 terms are compatible. Pin the exact release and split; v3 reports 1,276 performances, 198.7 hours and 7.04 million notes. MAESTRO v2 added sostenuto (CC 66) and una-corda (CC 67) alongside sustain (CC 64), which makes it especially important not to collapse expressive piano performance to note-only MIDI.

Source: https://magenta.withgoogle.com/datasets/maestro
### Physical-model renderer conformance

Pianoteq 9's official manual documents progressive partial sustain-pedal travel, distinct sostenuto and una-corda pedals, a renderer-specific harmonic pedal, sympathetic string resonance, repedaling, continuous velocity response, hammer-hardness/voicing and physical parameters. The harmonic pedal is a modeled effect, not a universal acoustic-piano control. Treat each control as a renderer-specific capability cell; never infer it from the generic piano identity.

Sources:
- https://www.modartt.com/user_manual?product=pianoteq
- https://www.modartt.com/pianoteq_features