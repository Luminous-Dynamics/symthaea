# MEL-GROOVE-001 — Ensemble Interlock and African Groove Benchmark v1

**Status:** research/engineering protocol plus a source-independent analysis primitive. This document does not claim that any African genre has an executable route, that a synthetic fixture is an authentic performance, or that Melothaea exceeds a commercial music-generation system.

**Implementation:** `symthaea-music-theory::ensemble_groove` on the branch named below.

## 1. Why a new primitive is needed

The current main branch has 29 composer-backed styles. Afrobeat, Highlife, Juju, Soukous, Amapiano and Gqom are visible research identities in the 152-entry catalog but have no callable composer routes.

The engine already contains several valuable, narrower mechanisms:

- `groove_cycle` makes phrase boundaries and long-form sections obey a named cycle (Son clave 3–2, Flamenco compás, Bossa and Habanera).
- `rhythmic_identity` measures post-composition melodic placement against a bar grid.
- `Accompaniment` contains several reusable cells, including Montuno, Compás-Gait and BossaComp.
- Muse's performance and renderer layers are separate from symbolic composition.

The remaining gap is not a lack of one more pattern. It is a general, measurable account of relationships between independently repeating parts. A one-bar drum pattern cannot, by itself, express or verify the full relation between a time-line, bass ostinato, guitar part, lead drummer, log-drum phrase, or improvised lead.

## 2. Paradigm shift: model the ensemble, not a flattened beat

The unit of analysis should be an interacting set of role layers:

`EnsembleGroove = {pulse, layers, cycles, phase, accent, constraints}`

Each layer has its own cycle length and attack positions. The full joint horizon is their least common multiple (LCM). Pairwise collision, phase and separation properties are then measured across that complete horizon, not a convenient one-bar excerpt.

This approach is motivated by ethnomusicological accounts of:
- a steady ostinato framework with concentric rhythmic layers and variable lead-drumming manipulation (Anku);
- time-line patterns, polyrhythm and lead-drum narratives as distinct but interacting practices (Agawu);
- cyclic cross-rhythm in which interwoven figures create tension while retaining a shared underlying pulse (Arom).

The code in this tranche implements the exact timing substrate and three explicit pairwise relationship types. It deliberately does not infer role names, cultural meaning, groove quality, or genre authenticity.

### 2.1 Current executable contract

The new `ensemble_groove` module provides:

- versioned `EnsembleGrooveV1`, `GrooveLayerV1`, `GrooveEventV1` and explicit pairwise constraints;
- integer tick timing, avoiding floating-point coincidence decisions;
- full LCM supercycle calculation for unequal layer cycles;
- per-layer attack counts, attacks per beat, accent histograms and distinct phase positions;
- pairwise coincident-attack counts, coincidence rate and minimum circular separation;
- fail-closed validation, including unknown layer references, invalid event positions, invalid accent levels, excessive supercycles, attack count limits and aggregate pairwise-work limits;
- deterministic synthetic tests for unequal cycles, exact collisions, accent distributions, invalid records, bounded expansion and intentional rests.

The supported relationship constraints are deliberately small:
1. `NoCoincidentAttacks` — an exact non-overlap requirement where the arrangement contract calls for it.
2. `MinimumCircularSeparation` — a lower bound on the circular distance between attacks in two layers.
3. `CallResponseWithinWindow` — a directed requirement that each caller-layer attack receive a responder-layer attack within a declared circular delay window, with a minimum coverage threshold in integer basis points.

Intentional unisons remain valid unless the author explicitly forbids them. The analyzer must never treat “no collision” as a universal property of African music or ensemble music in general.

### 2.2 What this does not do

It does not:
- generate notes or audio;
- model cue-conditioned section-state transitions beyond the directed timing/coverage contract;
- map a named genre to a pattern;
- classify a recording;
- infer authentic percussion or guitar roles from a genre name;
- model swing, microtiming, ornamentation, timbre or cultural performance practice;
- validate a human's interpretation of a tradition.

Those require separate, evidence-bound layers.

## 3. Six distinct research tracks

The six styles below are not a single “African groove” template. Each needs a period- and scene-specific model with its own sources, reference repertoire, nearest-neighbour controls, and knowledgeable review.

| Style | Research-grounded starting point | Candidate engineering focus | Important boundary |
|---|---|---|---|
| Afrobeat | Repetitive, interlocked guitar parts can establish the foundation while horns and voice provide variable/improvised material. | Independent guitar/bass ostinati, percussion/horn role graph, long-form vamp and lead/ensemble transitions. | Afrobeat is not Afrobeats. Do not reduce Afrobeat to generic 4/4 percussion or three guitar tracks. |
| Ghanaian Highlife | Distinct guitar-band and dance-band histories; studied guitar patterns such as mainline, Yaa Amponsah and Dagomba; keyboard parts can be auxiliary to guitar patterns. | Guitar-role vocabulary, bass/guitar interlock, band-vs-guitar-band arrangement and phrase/harmonic pattern interaction. | “Highlife” is internally diverse and changes over time. The literature warns that timeline and drum pattern alone can misclassify contemporary recordings. |
| Juju | Nigerian popular style with a changing history; performance research describes instrumental, verbal and non-verbal directing, including lead guitar and talking-drum cues. | Leader/follower signals, cue-conditioned transitions, layered percussion, call/response and ensemble state changes. | Do not hardcode talking drum as the only leader or treat all Juju periods/ensembles as identical. |
| Soukous / Congolese rumba lineage | Multi-guitar and bass roles support a rhythmic/harmonic fabric; lead guitar contributes interludes and improvisation; sebene is a structurally meaningful instrumental section. | Role-separated rhythm/lead guitars, bass relationship, sectional transition into sebene, lead improvisation over a persistent foundation. | Soukous is a historical family with overlapping variants. Do not collapse Congolese rumba, soukous, cavacha and later dance variants into one template. |
| Amapiano | South African electronic dance music with a historically situated production scene; scholarship describes the log drum's emergence as a prominent sonic and musical voice. | Treat log drum as a melodic-rhythmic bass voice with its own phrases, call/response and relation to kick, shaker, keys and vocal material. | A log-drum sample is not sufficient to make Amapiano. Model time, scene and production variants. |
| Gqom | Durban-origin electronic music is distinguished in scholarship by rhythmic emphasis, reduced harmonic/melodic foregrounding, raw/dark production and slow-burn structure. | Sparse rhythmic state, carefully managed silence, long-range tension, kick/bass relationships and section energy transitions. | Do not equate Gqom with generic techno or Kwaito. The sound has evolved and travelled into new scenes, including Afrikaans Gqom. |

### 3.1 Research implications for each track

**Afrobeat:** the key test is whether the fixed/repeating ensemble fabric and variable lead layers remain independently legible. The symbolic model should allow horn lines and vocal phrases to vary without destroying the foundational ostinato relationships.

**Highlife:** investigate guitar-band and dance-band forms separately. The studied guitar vocabulary is more informative than simply adding guitar to a generic pop arrangement. A source-grounded pattern library should capture named idioms, their chord compatibility and how bass, percussion and keyboard support them.

**Juju:** move beyond note-pattern composition. Some arrangements are directed in performance through musical cues and verbal/non-verbal coordination. A future model should have event-conditioned transitions: a lead cue can request a break, pickup, fill, chorus return or ending. The response of the ensemble should be testable as a coordinated state transition.

**Soukous:** model form and instrumental roles together. A sebene transition is not just a new guitar patch; it changes the role of the lead guitar and the energy/function of the section while the supporting layers continue to establish the dance foundation.

**Amapiano:** the log drum should not be a decorative kick replacement. It is often melodic and rhythmic at once. A later recipe should support pitch-contoured bass phrases, syncopated placement, conversational response with other parts, and variation across sections.

**Gqom:** silence and sparse pressure are compositional resources. A generic density-maximizing drum generator will systematically miss this. A proper contract needs negative-space targets, repeated-beat identity, slow-burn section development and counterexamples where adding notes is an error.

## 4. Research and data intake

The immediate data task is to construct a rights-cleared, period-labelled, source-linked reference set. A public URL is not a licence. A scholarly analysis, transcription, sound recording, composition, arrangement and metadata may have different rights and consent requirements.

For every reference item, record:
- work/recording/arrangement identity and version;
- country, region, language, artist, ensemble and relevant period (only where the source supports it);
- source URI, artifact hash and rights/consent review per modality;
- annotation provenance and annotator role;
- known variants, uncertainty and disagreement;
- what the item can and cannot establish.

Do not ingest recordings or transcriptions until all required rights layers and community-consent requirements are cleared. Where no rights-cleared reference is available, keep the profile at hypothesis level and use synthetic metric fixtures only for software correctness.

### 4.1 Annotation model

A qualified annotation should distinguish:
- underlying pulse and tactus;
- time-line/reference patterns where present;
- cycle lengths and phase origin;
- role-specific attack positions and accent hierarchy;
- cross-rhythm/polyrhythm and intentional coincidence;
- bass and melodic-rhythmic interaction;
- lead/response and cue/transition behavior;
- section function and long-range energy;
- microtiming, ornamentation, articulation and production where relevant.

Each annotation should also identify what is unknown. Do not impute a missing part from another track, from a genre tag, or from assumed instrumentation.

## 5. Benchmark design: MEL-GROOVE-001

The benchmark must measure capability in stages. The first stage is not a listening contest; it is an exact oracle proving that the representation and analyzer behave correctly.

### G0 — synthetic known-answer mechanics

- Recompute LCM supercycles exactly.
- Detect every planted collision and no others.
- Verify pairwise circular separation.
- Verify accent and density summaries.
- Reject malformed patterns, unknown layer references, invalid accents and resource-exhaustion inputs.
- Test metamorphic transformations: translate all layer phases by the same offset, reorder JSON layer/event arrays, and duplicate the full joint supercycle. Equivalent patterns must produce equivalent structural measurements.

**Target:** 100% of frozen known-answer and negative-control cases pass; zero false acceptance for a violated hard constraint.

### G1 — source-grounded symbolic analysis

Use a separate set of rights-cleared, musician-annotated reference patterns.

Report:
- layer-role and event alignment precision/recall;
- cycle-period and phase error;
- accent-tier agreement;
- pairwise interlock/collision agreement;
- cross-rhythm and reference-pulse agreement;
- uncertainty and source coverage per style.

Do not merge annotation disagreements into one “ground truth” without preserving the disagreement. Report per-annotator agreement and adjudication method.

**Proposed target:** at least 0.90 precision and recall on adjudicated, explicitly notated mechanical events; at least 0.80 inter-annotator agreement on categorical role/relationship labels before using them as confirmatory labels. These are pilot gates, not SOTA claims.

### G2 — production composer control

Call the real composition path with controlled interventions:
- baseline;
- one layer changed;
- cycle length changed;
- phase offset changed;
- accent hierarchy changed;
- two simultaneous changes;
- same changes in reversed application order;
- an invalid or conflicting constraint as a negative control.

Measure target sensitivity and protected non-target invariance separately. A change to the bass line should not silently destroy the reference timeline, section boundaries or melodic identity unless the contract explicitly permits it.

**Proposed target:** at least 95% of preregistered target interventions produce the intended measurable effect; at least 95% of protected invariants remain within declared tolerance; all invalid hard-constraint cases are rejected or explicitly reported unsupported.

### G3 — style identity and nearest-neighbour discrimination

Use matched prompts, durations, instrumentation availability and fixed output counts. Randomize and blind the outputs. Include the six styles and close controls (for example Afrobeat vs Afrobeats; Highlife vs palmwine/guitar-band variants; Juju vs generic percussion-led pop; Soukous vs Congolese rumba; Amapiano vs South African house; Gqom vs techno/Kwaito).

Report:
- balanced accuracy and per-class precision/recall;
- full confusion matrix;
- pairwise identity judgments;
- within-style diversity;
- nearest-neighbour confusion;
- abstention where the style contract is underspecified.

**Proposed pilot target:** 70% balanced accuracy across the six scoped identities, with a 95% confidence interval and no style hidden by macro-averaging. Confirmatory thresholds and sample size must be preregistered after a power analysis and practitioner review.

### G4 — rendered audio and human review

Run the same symbolic candidates through:
- neutral/reference renderer settings;
- declared style/performance profile;
- instrument/production substitutions;
- a fixed renderer and instrument-bank environment with exact hashes.

Blind practitioners and independent listeners. Ask separate questions:
1. Is the musical identity recognizable?
2. Is the arrangement idiomatic within the declared scope?
3. Is the performance/execution credible?
4. Is the result musically compelling?
5. What evidence supports or contradicts the judgment?

**Proposed target:** at least 60% pairwise preference on a preregistered task against a named baseline, with the 95% interval lower bound above 50%, followed by replication. Identity and authenticity require separate gates and cannot be inferred from preference.

### G5 — open-ended combinations

Only after individual systems pass their gates:
- combine a chosen rhythmic layer with a different structural owner;
- retain or explicitly adapt the source's role/phrase behavior;
- test held-out combinations, not merely sampled combinations from the same preset library;
- evaluate whether the resulting hybrid sounds coherent while being honestly labelled as a hybrid, not as a historically authentic genre.

**Target:** measurable compositional generalization to held-out layer combinations without violating protected source constraints; no claim of authenticity without relevant practitioner review.

## 6. Comparison with SOTA and the music industry

Do not confuse three different leaderboards or tasks.

### Commercial full-audio generation

The current Artificial Analysis Music Arena is based on blind listener preference. Its 9 November 2026 snapshot reports:
- Suno v6: **1148 ± 22** on instrumental generation (2,208 samples);
- Suno v6-mini: **1114 ± 21** (2,332 samples);
- Mureka V9.5: **1084 ± 21** (2,297 samples);
- Lyria 3 Pro: **1080 ± 21** (2,329 samples);
- Stable Audio 3 Large: **1014 ± 21** (2,291 samples).

For vocals, Suno v6 is **1144 ± 18** (3,063 samples), Mureka V9.5 is **1101 ± 18**, and Lyria 3.5 is **1050 ± 18**. These are blind-preference Elo results, not measures of cultural fidelity, controllability or exact musical structure. <https://artificialanalysis.ai/music/leaderboard/instrumental> · <https://artificialanalysis.ai/embed/music-generation-leaderboard/leaderboard/vocals>

### Symbolic composition

MIREX 2025 evaluated a deliberately narrow task: continue a four-measure, 4/4 piano prompt into a 16-measure piece. It reported separate coherency, structure, creativity and musicality scores. Its subjective evaluation used 20 participants, eight prompts, and allowed cherry-picking from eight candidates before evaluation. This is useful but not a complete measure of controllable, ensemble-level music generation. <https://www.music-ir.org/mirex/wiki/2025%3ASymbolic_Music_Generation_Results>

### Evaluation frontier

A 2026 survey concludes that no single metric is sufficient and that symbolic generation, audio generation, structure, diversity, faithfulness and controllability need distinct evaluation methods. It also identifies long-term structure, cross-voice coordination, groove/polyphonic density, timbral realism, and richer intermediate representations as open challenges. <https://doi.org/10.1007/s10462-026-11582-x>

A 2025 study found that only 5.7% of the music hours in the datasets it surveyed were labelled as non-Western genres. That is evidence of under-representation in those studied datasets, not a universal estimate of every commercial model's training data. <https://aclanthology.org/2025.findings-naacl.258/>

### What “better than SOTA” should mean for Melothaea

We should compete on two independent tracks:

**Track A — production quality:** matched prompts, fixed durations, identical output count, blind pairwise comparison, no cherry-picked candidates, and current commercial systems as named comparators. The goal is to eventually outperform leading systems on the same task—not to compare our symbolic unit tests against their finished audio.

**Track B — musical agency and controllability:** exact style/role constraints, structure-preserving edits, score/MIDI transparency, independent part control, reproducible improvisation and source-bound provenance. This is where Melothaea can develop a different and potentially stronger capability than black-box audio generation.

The decisive result could be: commercial systems win on one-shot audio preference, while Melothaea wins on verified control, editability, structure and tradition-specific role fidelity. That would be a meaningful, testable distinction rather than a fabricated overall SOTA claim.

## 7. Research basis

1. Willie Anku, “Circles and Time: A Theory of Structural Organization of Rhythm in African Music,” *Music Theory Online* 6(1), 2000. [https://doi.org/10.30535/mto.6.1.2](https://doi.org/10.30535/mto.6.1.2)
2. Kofi Agawu, “The Rhythmic Imagination,” in *The African Imagination in Music*, 2016. [https://doi.org/10.1093/acprof:oso/9780190263201.003.0005](https://doi.org/10.1093/acprof:oso/9780190263201.003.0005)
3. Simha Arom, “African rhythmics,” in *African Polyphony and Polyrhythm*, Cambridge University Press. [https://www.cambridge.org/core/books/abs/african-polyphony-and-polyrhythm/african-rhythmics/F675DEAFE6569381B013C9982B3C5762](https://www.cambridge.org/core/books/abs/african-polyphony-and-polyrhythm/african-rhythmics/F675DEAFE6569381B013C9982B3C5762)
4. “Gigging on the World Stage: Bossa Nova and Afrobeat after De-reification.” [SciELO article](https://www.scielo.br/j/cint/a/3TTy5RCKtKQDJLghWzb833b/?format=html&lang=en)
5. M. M. Coffie, “Redefining Ghanaian Highlife Music in Modern Times,” 2020. [University of Education, Winneba publication record](https://www.uew.edu.gh/index.php/dme/staff/mmcoffie/publications/29265/detail)
6. Eric Sunu Doe, “Palmwine music styles in Ghana,” *African Music* 14(1), 2025. [https://doi.org/10.21504/amj.v14i1.3039](https://doi.org/10.21504/amj.v14i1.3039)
7. A. Alaja-Browne, “A diachronic study of change in juju music,” *Popular Music*, 2008. [Cambridge University Press](https://www.cambridge.org/core/journals/popular-music/article/abs/diachronic-study-of-change-in-juju-music/6C359B104E11472083E09727E7958672)
8. “Origin and development of Directing in juju music performance.” [Academic Journals](https://academicjournals.org/journal/JMD/article-full-text/777EB6547545)
9. “The Genesis of Urban Music in Zaïre.” [*African Music* journal article](https://journal.ru.ac.za/index.php/africanmusic/article/download/1945/1020/1027)
10. D. M. Eaby-Lomas, “Historicising gqom as a post-kwaito phenomenon,” *Journal of the Musical Arts in Africa* 18(1), 2021/2022. [https://doi.org/10.2989/18121004.2021.2020964](https://doi.org/10.2989/18121004.2021.2020964)
11. “Ase trap tse ke pina tsa ko kasi”: Sonics After the Struggle: Mapping Amapiano and Black Aesthetics in Post-Apartheid South Africa, 2026. [https://doi.org/10.1080/13696815.2026.2634896](https://doi.org/10.1080/13696815.2026.2634896)

## 8. Implementation sequence

1. Qualify the source-independent LCM/constraint analyzer and its synthetic negative controls.
2. Add source-grounded profile manifests for six research identities, each with explicit unknowns and period boundaries.
3. Define a rights-cleared, musician-annotated pattern set and adjudication protocol.
4. Extend constraints to directed call/response, cue-conditioned state transitions, allowed coincidence and reference-timeline ownership only when an annotated tradition requires them.
5. Integrate the model with the real composer path; keep each style's structure, rhythm, bass, melody and orchestration contracts separately inspectable.
6. Add performance/rendering evidence only after the symbolic model passes.
7. Conduct blind musician and listener evaluations; publish failures, confidence intervals, per-style results and source/asset digests.

**Current claim boundary:** the code is a new symbolic analysis primitive with authored synthetic tests. The six tradition profiles remain research hypotheses. No authentic recordings, licensed corpora, genre-specific composer routes, renderer qualifications, listener studies or SOTA wins are claimed.

## 2.3 Directed cue and response state machine

The new `ensemble_cues` module models a cue protocol as a finite-state transition system:

`current_state + (cue_type, source_layer) -> target_state + (response_layer, response_action)`

An authored sequence records cue events, realized state changes and response actions. The verifier checks:

- the cue is valid for the current state and declared source layer;
- the expected state change occurs within a declared latency bound;
- the designated response layer performs the required action inside its delay window;
- cues are processed in deterministic chronological order;
- a failed state change does not silently advance the state machine.

This is a reusable cue protocol, not a Juju template. A profile may later define instrument-led, verbal, gesture-derived or hybrid cue vocabularies, but those vocabularies and their meanings require source and practitioner review. The current implementation verifies declared cue/action traces; it does not infer body gestures or cues from audio/video.

### Juju-specific implication

Olusakin's 2014 fieldwork-based study distinguishes instrumental, verbal and non-verbal forms of directing in Juju performance. Its examples include lead-guitar and talking-drum signals that request changes in dynamics, interludes, accompaniment, pauses and endings. It also notes that leadership and directing practice varies between bands. This supports an event-conditioned transition model, not one universal master-drummer controller. [Olusakin, 2014, DOI: 10.5897/JMD2014.0027](https://doi.org/10.5897/JMD2014.0027).

## 2.4 Inter-onset space and rhythmic pressure

The new `groove_space` module measures circular inter-onset gaps per layer over the same complete supercycle. It reports:

- sorted inter-onset gap lengths;
- shortest and longest gaps;
- exact mean-gap ratio (horizon / unique attacks);
- count of gaps above a caller-selected threshold;
- fraction of the supercycle covered by those long gaps, in basis points.

The name is intentionally **inter-onset space**, not acoustic silence. The current event representation has attacks but not note durations, release envelopes, reverberation or sample tails. A long inter-onset gap can contain sustained bass or a pad; it is therefore not proof of audible silence. A future rendered-audio track must combine event timing with duration/envelope and signal measurements before making an acoustic-silence claim.

### Gqom-specific implication

Eaby-Lomas's historical analysis describes Gqom's rhythmic beat as foregrounded over harmony and melody, a raw/dark production character, a slow-burn structure, and later transformations of those features. This supports testing sparsity, inter-onset gaps and long-range pressure as candidate measurements, but not declaring a specific gap distribution canonical for all Gqom. [Eaby-Lomas, “Historicising gqom as a post-kwaito phenomenon”](https://doi.org/10.2989/18121004.2021.2020964).

## 2.6 Offline executable known-answer harness

The crate now includes `examples/ensemble_groove_contract_benchmark.rs`, a deterministic offline harness covering four synthetic conditions:

1. unequal-cycle interlock with a valid no-collision/separation contract;
2. a planted collision negative control;
3. sparse versus dense circular inter-onset-gap profiles;
4. a complete cue/state/response chain and a missing-response negative control.

Run it with:

```sh
cargo run -p symthaea-music-theory --example ensemble_groove_contract_benchmark
```

The harness exits with an error if a known-answer assertion fails and emits a JSON report labelled `synthetic_known_answer_mechanics_only`. It has no source recording inputs and must not be interpreted as genre recognition, audio-quality, listener-preference or authenticity evidence.

## 2.5 Added synthetic gates

The new cue and space modules add deterministic synthetic tests for:

- ordered cue → state → response chains;
- missing or delayed responses;
- wrong state changes and state-machine non-advancement;
- deterministic cue ordering under array permutation;
- duplicate cue-transition keys;
- sparse-vs-dense inter-onset profiles;
- circular gap conservation across the entire supercycle;
- single-event cycles and explicit empty/rest layers.

These tests qualify the mechanics only. They do not qualify any named musical tradition.

## 5.1 Adversarial control and preservation matrix

Style controls should be tested under interventions, not one at a time only. The proposed confirmatory matrix is:

| Intervention | Target behavior | Protected properties / negative controls |
|---|---|---|
| Add or alter a leader cue | Expected state transition and response | Non-target layers remain unchanged outside declared response window |
| Remove the required response | Response coverage fails | A state change alone must not pass the whole cue contract |
| Add rhythmic attacks to a sparse layer | Density and gap profile change | Cycle ownership and unrelated layer phase remain stable |
| Shift all layer phases by one common offset | Relative layer relationships remain identical | Pairwise coincidence, circular separation and gap statistics are invariant |
| Change one layer's cycle length | Joint LCM and phase pattern update | Unchanged layers keep their own cycle/event definition |
| Break a long gap deliberately | Long-gap coverage decreases | Unrelated layer metrics remain unchanged |
| Reorder events and JSON records | Same canonical result after sorting | No dependence on serialization order |
| Remove a required source/consent/annotation record | Evidence admission fails | No fallback to genre-name assumptions or public URL availability |

For the composer path, every intervention should be run through the actual production composer with fixed seeds and exact source/artifact digests. Synthetic inputs are appropriate for oracle validation; style-specific fixtures require separately sourced evidence.

## 6.1 Updated industry comparison snapshot (9 November 2026)

Artificial Analysis's current AA-Music-Instrumental v1.1 page reports a blind-pairwise-preference Elo ranking. Its visible snapshot lists Suno v6 at 1148 ± 22 (2,208 samples), Suno v6-mini at 1114 ± 21, Mureka V9.5 at 1084 ± 21, and Lyria 3 Pro at 1080 ± 21. This is a preference leaderboard, not a score for structural control, cultural specificity, editability or source provenance. [AA Music Instrumental leaderboard](https://artificialanalysis.ai/music/leaderboard/instrumental?tab=leaderboard).

The June 2026 *Artificial Intelligence Review* survey covers symbolic, audio and multimodal generation, including interactive/co-creative settings, and discusses distinct models, datasets and objective metrics. It reinforces that commercial audio preference and controlled symbolic capability are separate evaluation targets. [Duan, Yu & Briot (2026)](https://doi.org/10.1007/s10462-026-11582-x).

## 7.1 New research references

12. Oluniyi Michael Olusakin. “Origin and development of 'Directing' in juju music performance.” *Journal of Music and Dance*, 2014. [https://doi.org/10.5897/JMD2014.0027](https://doi.org/10.5897/JMD2014.0027).
13. Dion Malcolm Eaby-Lomas. “Historicising gqom as a post-kwaito phenomenon.” *Journal of the Musical Arts in Africa*, 2021. [https://doi.org/10.2989/18121004.2021.2020964](https://doi.org/10.2989/18121004.2021.2020964).
14. Wei Duan, Yi Yu & Jean-Pierre Briot. “Recent advances in music generation: methods, evaluation, and challenges.” *Artificial Intelligence Review*, 2026. [https://doi.org/10.1007/s10462-026-11582-x](https://doi.org/10.1007/s10462-026-11582-x).
