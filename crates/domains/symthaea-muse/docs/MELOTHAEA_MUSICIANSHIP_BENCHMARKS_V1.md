# Melothaea Musicianship Benchmarks v1

**Status:** research protocol and proposed targets; not an established standard and not a result claim.  
**Scope:** generated performance, instrument-specific expressive control, ensemble interaction, composition, and listener preference.

## 1. The goal: outperform on defined capabilities, not win a vague “best artist” score

Can Melothaea exceed exceptional drummers, guitarists, bassists, and artists? It is plausible for a system to outperform humans on some bounded dimensions—repeatability, rapid arrangement variants, constraint satisfaction, response speed, search breadth, and exact reproduction—while remaining weaker on groove, touch, phrase intention, timbral nuance, taste, or social interaction. There is no single accepted scalar measurement of “best artist.”

Do not claim superiority from more styles, more notes, lower MIDI error, musical-theory validity, a higher model score, or a generated clip that sounds impressive to its developer. The target is a public, reproducible matrix of tasks with strong human baselines and listener evaluations.

## 2. Paradigm shift: musician = adaptive performance policy, not “notes plus random jitter”

The low-level legacy performance path maps a system-state value (Phi) to Gaussian timing jitter. That is a controllable heuristic, not evidence that higher Phi means better time, lower Phi means expressive looseness, or either parameter represents instrumental skill.

The new PerformanceProfileV1 introduces explicit timing, downbeat tightness, velocity variation, and duration controls without using Phi as a skill proxy. Its default values are engineering priors, **not** calibrated human statistics. The new API is a parameter boundary, not a full musician model and not yet a claim about the output used by the whole product.

The research direction is an event-sequence policy that jointly conditions on:
- instrument role and physical affordances;
- style/tradition, meter, tempo and subdivision;
- phrase position, harmonic obligation and section intent;
- interaction with other parts (kick–bass lock, drum fill response, guitar space around vocals);
- articulation, dynamics and timbre capabilities;
- observed musician/session context and an editable intent contract.

Notes must be realized in relation to the phrase and ensemble, not independently jittered. The policy should be able to choose tight, laid-back, swung, pushing, pulling, sparse, or deliberately unstable feels when requested—and explain which control changed.

Keep symbolic composition, performance realization, synthesis fidelity and listener judgment as separate evidence layers.

## 3. Benchmark layers and gates

### Gate A — execution fidelity

For a given written score or MIDI performance target, measure:
- event precision/recall and pitch accuracy;
- onset and offset error in milliseconds **and** beat-relative units;
- velocity and dynamic-curve error;
- missing, duplicated, stuck or invalid notes;
- phrase timing, articulation and note-release behavior;
- export/render reproducibility, latency and resource use.

Report distributions, tails and failures by instrument role. Averages must not hide an occasional catastrophic dropped bar or a badly wrong fill.

### Gate B — musician-specific behavior

Evaluate each role with appropriate tasks. Drum, guitar and bass are different physical and musical problems; do not score them as one undifferentiated note sequence.

**Drummer**
- Source data: the [Groove MIDI Dataset (GMD)](https://magenta.withgoogle.com/datasets/groove) includes 1,150 MIDI files, more than 22,000 measures and 13.6 hours of aligned human-performed expressive drumming, with 10 drummers and multiple styles.
- Tasks: beat/groove continuation; style-preserving variations; fill placement into transitions; response to tempo, density and dynamics changes; stable performance across several minutes.
- Measures: onset F1 at predeclared 10/25/50 ms tolerances; kick/snare/hat class accuracy; beat-relative microtiming distributions; velocity accent contour; groove descriptor distance; fill-onset and transition appropriateness; variation across repeated choruses.
- Avoid optimizing for low onset error alone: professional drummers intentionally place notes ahead of or behind the beat. Score style-conditional fit and listener preference too.

**Guitarist**
- Sources: [GuitarSet](https://github.com/marl/GuitarSet) and the [Fraunhofer IDMT-SMT-Guitar dataset](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html). IDMT-SMT-Guitar documents normal, muted, picked/finger-style playing, bends, slides, vibrato, harmonics and dead notes, as well as realistic licks.
- Tasks: preserve a riff while varying picking/articulation; respond with complementary phrases; realize a bend, slide, vibrato, palm-mute or harmonic intention; maintain voicing and rhythm under tempo/style changes.
- Measures: note/onset/offset accuracy where annotations support it; articulation recognition and correct realization; pitch-bend trajectory; vibrato rate/depth; pick-attack/timbre recognition; fretboard feasibility; chord/voice-leading accuracy; riff identity after transformation.
- MIDI note timing alone cannot assess guitar touch or tone. Use annotated audio and controlled synthesis/recording paths for the acoustic dimensions.

**Bassist**
- Sources: [IDMT-SMT-Bass](https://www.idmt.fraunhofer.de/en/publications/datasets/bass.html) and [IDMT-SMT-Bass-Single-Track](https://www.idmt.fraunhofer.de/en/publications/datasets/bass_lines.html). The latter contains 17 annotated bass lines with pattern occurrences, pitch/onset/offset, string/fret positions and plucking/expression-style labels; the former covers multiple plucking and expression techniques over a low-register instrument range.
- Tasks: lock with kick while preserving groove; support harmonic changes; vary a repeated bass line without losing identity; create fills only in available phrase gaps; sustain or mute notes intentionally.
- Measures: note/pitch/onset accuracy; root/guide-tone support; kick–bass onset relationship; low-register/octave accuracy; silence/rest placement; repetition-with-variation; articulation/timbre class; playability by string/fret constraints when ground truth exists.
- A bassist is not good merely because every onset matches the grid. The pocket, interaction, note choice and restraint must be evaluated.

Use these datasets within their stated licenses and terms; do not commit source recordings or derived redistribution of restricted data to this repository. GMD is CC BY 4.0. Fraunhofer IDMT guitar/bass datasets are commonly offered under CC BY-NC-ND 4.0; verify the exact dataset-specific license before use. Separate training, calibration, and test data by performer/session or source work to prevent leakage.

### Gate C — ensemble musicianship

A band is more than three individually plausible tracks. Define matched multi-part challenges where all candidate systems receive the same chart/brief and constraints.

- Kick–bass alignment and intentional displacement, measured with note-role-aware timing rather than global onset statistics.
- Harmonic-change support and voice leading without occupying the same register unnecessarily.
- Guitar chord/lead frequency and rhythmic-space management.
- Fills at phrase transitions without covering a vocal or primary motif.
- Dynamics and density that rise into a climax and recede at the intended structural boundary.
- Reaction to a human tempo/density/section cue within a bounded response time.
- Multi-minute stability: no stuck notes, lost state, unexplained tempo drift, or repeated fill loops.

Proposed initial engineering gates:
- zero malformed MIDI events and zero stuck/dropped notes in the deterministic synthetic panel;
- 100% of declared semantic cue changes applied to the intended part and musical boundary;
- p99 cue-response latency below one eighth-note duration at the declared tempo, with any stricter musical deadline defined per task;
- 30-minute stress performance with no state reset, stuck note or unplanned silence;
- every transformation annotated with its preservation contract and actual symbolic diff.

These are candidate goals, not current results; validate measurement tooling and define the exact fixtures first.

### Gate D — composition and artistry

Do not reduce “artist” to musicianship mechanics or one audio embedding. Each challenge should specify the brief, style family, desired emotional/structural arc, instrumentation, duration, restrictions, reference assets and permitted degrees of freedom.

Measure distinct dimensions:
- long-range form and section contrast;
- motif development, return, transformation and payoff;
- harmonic/rhythmic coherence under controlled novelty;
- instrument-role differentiation and arrangement clarity;
- emotional arc consistency with the brief, assessed by listeners;
- memorability and identity after a delay;
- variation quality across repeated generation seeds;
- edit locality: when asked to change one thing, did the rest remain intact?
- controllability: can musicians reliably elicit the requested changes in one or two revisions?
- originality/diversity within the allowed style, without rewarding noise or mere deviation.

A valid originality test needs a frozen reference corpus, duplication checks, explicit licensing, and a human novelty judgment. A style label or a similarity embedding does not prove originality or cultural authenticity.

## 4. Human preference challenge protocol

The decisive artistic comparison is a blinded, matched-brief study—not a developer selecting a favorite demo.

For each task, compare:
1. Melothaea with a fixed, recorded version identifier;
2. a strong contemporary baseline/system under the same brief;
3. an expert human performance produced for the same brief;
4. optionally, an expert-produced reference arrangement or interpretation.

Match duration, loudness policy, input chart/brief, instrument availability and production chain. Randomize order. Keep identities blind until judgments are frozen. Collect separate ratings for groove, execution, composition, emotion, instrument authenticity, coherence, novelty, and overall preference. Allow “tie/no meaningful preference” and record expertise/background without treating it as ground truth.

Pre-register exclusions, analysis, stop rule and confidence interval. Use a pilot to estimate variance and statistical power before fixing the sample size. The published [music-generation human-preference benchmark](https://arxiv.org/abs/2506.19085) compared 12 systems across 6,000 songs with more than 15,000 pairwise judgments from over 2,500 participants; that scale is a useful precedent, not a minimum requirement for every experiment.

Proposed headline target for a named task/domain:
- Melothaea wins at least 60% of matched comparisons against a declared strong baseline, with a 95% confidence interval lower bound above 50%;
- no critical instrument/genre subgroup has an unexplained collapse;
- listeners' preference agrees with the intended brief, not simply loudness, novelty or polished mixing;
- results replicate on a held-out prompt set and in a second listening cohort.

Passing this target supports only that named task/domain and cohort. It does **not** prove Melothaea is better than every top artist or better for every listener.

## 5. Quality, generalization and statistical controls

- Split by performer, session and source piece, not by randomly splitting adjacent bars.
- Keep a held-out genre/tradition and a held-out prompt set for generalization.
- Report per-role, per-style and per-tempo results; aggregate only with published weights.
- Publish confidence intervals and failure cases, not just means or Elo ranks.
- Separate calibration metrics from confirmatory metrics. Do not tune the profile on the test musicians and then report success on the same recordings.
- Compare against quantized, random-humanized, role-aware baseline and strong existing system ablations.
- Evaluate deterministic replay and seed diversity separately: one tests reproducibility, the other meaningful range.
- Preserve every negative control. Queue/skipped/missing CI does not equal pass.

## 6. Milestones

**M0 — Baseline truth:** version and hash the existing performance path, freeze synthetic test scores, and document current known limits. No “better than human” claim.

**M1 — Dataset adapters:** create legally compliant, read-only adapters for GMD; GuitarSet/IDMT guitar; IDMT bass; and matched performance prompts. Record splits, source hashes, licenses and annotations.

**M2 — Role-aware scorecard:** compute metrics by performer, part and style; add known-good and mutation controls to prove the metrics react to the intended defects.

**M3 — Learnable expression profiles:** fit PerformanceProfileV1 from training performers only, compare against the unchanged legacy Phi-based path, and freeze parameters per role/style. Avoid overfitting to a global average performer.

**M4 — Interactive ensemble policy:** model cross-part timing/phrasing and context-specific fills, then test on held-out human sessions and adversarial tempo/section changes.

**M5 — Matched human studies:** blinded expert and listener comparison on fixed briefs with preregistered analysis.

**M6 — Limited superiority statement:** publish only the specific tested capabilities where Melothaea beats the chosen human/system baseline with uncertainty bounds and successful replication.

## 7. Immediate code boundary in this PR

PerformanceProfileV1 is an explicit, bounded API for timing variance, downbeat timing scale, velocity variation and duration scaling. It rejects non-finite/out-of-range parameters before mutation and uses a fixed seed for deterministic application. Regression tests cover zero-variation identity, deterministic/bounded output and invalid-profile rejection.

It is **not yet wired into the entire production composition/export path**, its defaults are not fitted to any performer dataset, and this PR has not been compiled or tested in the current environment. Do not treat the parameter API as proof of better playing.

## Sources

- Gillick et al., *Learning to Groove with Inverse Sequence Transformations*, and [Groove MIDI Dataset](https://magenta.withgoogle.com/datasets/groove) (human-performed aligned MIDI/audio, CC BY 4.0).
- [GuitarSet](https://github.com/marl/GuitarSet), guitar transcription dataset and annotations.
- Fraunhofer IDMT, [IDMT-SMT-Guitar](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html), [IDMT-SMT-Bass](https://www.idmt.fraunhofer.de/en/publications/datasets/bass.html), and [IDMT-SMT-Bass-Single-Track](https://www.idmt.fraunhofer.de/en/publications/datasets/bass_lines.html).
- Grötschla et al., [Benchmarking Music Generation Models and Metrics via Human Preference Studies](https://arxiv.org/abs/2506.19085).
- Duan, Yu & Briot, [Recent advances in music generation: methods, evaluation, and challenges](https://link.springer.com/article/10.1007/s10462-026-11582-x), published 2 June 2026. The survey highlights the lack of a unified, accepted evaluation protocol and the limits of objective metrics for expressiveness and interaction.
