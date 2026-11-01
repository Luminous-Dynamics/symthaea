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

The missing scientific seam is **plan-to-score accountability**. A composer can emit a plan that says “introduce a theme, transform it in the middle, and bring it home” while its finished score fails to realize that development. A plan label or a plausible fingerprint is not proof. We need a separate verifier that checks each declared compositional obligation against the exact symbolic score.

This is not an attempt to formalize beauty. It is a way to make compositional claims inspectable and falsifiable.

## Fresh benchmark landscape: MIREX 2026

The MIREX 2026 piano-continuation task asks systems to continue a four-measure piano prompt with twelve measures, producing a sixteen-measure piece. Its published results now include four systems:

| System | Prompt coherence | Structure | Creativity | Musicality |
|---|---:|---:|---:|---:|
| DAMN (Nested Transformer) | 3.68 ± 0.16 | 3.39 ± 0.18 | 3.18 ± 0.16 | 3.47 ± 0.14 |
| Anticipatory Music Transformer | 3.47 ± 0.20 | 3.34 ± 0.16 | 3.34 ± 0.16 | 3.61 ± 0.17 |
| PhraseSplitGen | 2.95 ± 0.18 | 2.74 ± 0.15 | 2.89 ± 0.15 | 2.61 ± 0.14 |
| MuseCoco (1.2B) | 2.79 ± 0.17 | 2.76 ± 0.15 | 3.42 ± 0.12 | 2.68 ± 0.16 |

The reported values are means ± standard error. The report describes eight prompts across pop, classical and jazz; 12 participants; 38 ratings per model; and a process in which eight candidates per prompt were returned for cherry-picking. These results give us a concrete current comparison set, but the small prompt/participant panel and best-of-eight selection mean they should be treated as a task-specific reference, not a ranking of general compositional ability.

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

## New proposal: Composition Obligation Evidence (COE)

A grammar/planning component may declare obligations, but it may not certify its own success. A downstream verifier receives the exact score plus the plan/trace and returns one status per obligation:

- `verified`: the required score evidence was independently found;
- `not_met`: evidence contradicts or fails the declared obligation;
- `unknown`: the available score representation or evidence cannot decide;
- `not_applicable`: this grammar does not make that kind of claim;
- `invalid_evidence`: malformed identity, digest, span, or contradictory evidence.

Examples of obligations:

| Obligation | Evidence the verifier must locate |
|---|---|
| Motif introduction | Exact opening score span and canonical motif identity |
| Motif transformation | Later span plus a recognized transformation relation to the source motif |
| Motif return | A later, score-bound occurrence satisfying the declared return relation |
| Section contrast | Distinct, measurable structural evidence across declared sections—not only different labels |
| Call and response | Ordered phrase spans and a declared response relationship |
| Ground variation | Persistent ground identity plus measurable variation in the dependent material |
| Cadential/ending intent | Exact final phrase and the applicable grammar-specific closure evidence |

The verifier must not force every musical language into common-practice tonal norms. A drone, modal unfolding, additive process, groove cycle, ambient texture or intentionally open ending needs its own declared grammar contract. An unsupported property is `unknown` or `not_applicable`, never a silent pass or a false failure.

### Proposed versioned evidence record

The first implementation can be a small additive type in the music-theory crate. Its semantic fields should include:

- schema and verifier version;
- exact canonical score digest;
- grammar family and plan digest;
- stable obligation ID and obligation kind;
- source motif/ground identity digest, where applicable;
- exact part and score-time span references;
- observed transformation/evidence references;
- status and machine-readable failure reasons.

Do not infer a stable note identity from pitch, index, channel, or timing. Reuse the producer-assigned identity contract where available. A content digest establishes integrity, not authorship or signature authority. The composer supplies claims; an independent verifier recomputes evidence.

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
- plan path with all proposed COE verification enabled;
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

Automated FMD/FAD/CLAP or the COE verifier may explain specific properties. None is a substitute for listeners.

## Initial research gates (targets, not results)

- 100% of required obligations in the frozen positive fixtures are found.
- 100% of preregistered malformed/contradictory evidence fixtures are rejected.
- Zero false certifications on the frozen adversarial panel.
- Every declared obligation resolves to exact score evidence or an explicit non-pass status.
- Identical inputs, code, corpus and candidate budget reproduce identical mechanical reports.
- Composer/work/source-disjoint evaluation; all exclusions and near-duplicate alerts disclosed.
- Human pairwise preference target remains >60% against the strongest matched baseline, with the 95% confidence-interval lower bound above 50%, replicated on a sealed split. This is an ambitious research target, not an achieved score or universal beauty test.

## Implementation sequence

1. Audit the current grammar plans, structural traces, obligation verifier and score fingerprint; reuse their semantics.
2. Define the smallest versioned COE schema and a verifier that reads exact score/plan artifacts without trusting composer-supplied “verified” flags.
3. Add deterministic positive and adversarial fixtures for the supported grammar families.
4. Add paired structural interventions to the existing listening/benchmark pack rather than creating another pack framework.
5. Add the 32/64-bar piano brief suite and exact manifest/seed/candidate accounting.
6. Connect to #9729's blind listening protocol only after mechanical obligations and artifact lineage are stable.

No training run, renderer capability, compile, test, or listener result is claimed by this design note. The next source PR should be narrow and should not be opened as merge-ready until exact-head Rust checks and independent fixture evidence exist.
