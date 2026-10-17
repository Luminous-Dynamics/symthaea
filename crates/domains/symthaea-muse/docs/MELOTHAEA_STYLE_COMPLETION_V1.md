# Melothaea Style Completion v1

**Status:** first implementation tranche; not a claim that all 152 styles are finished.  
**Scope:** the public Muse 152 registry, composer routes, compositional identity, and human-review gates.

## Baseline and honest completion measure

The catalog has 152 stable public IDs across 19 constellations. On this branch, 35 entries have a callable composer route, up from 29. The remaining 117 entries have no callable route.

A callable route is not the same as a finished or authentic style. Existing catalog statuses remain Developing or ExpertReviewRequired; no style is Foundation merely because it generates a score. Some existing entries route to a close related preset (for example, Chicago Blues to the general Blues engine or Motet to SacredChoral), so style-specific identity and blind listening still need to be established.

| Constellation | Callable routes | No route yet |
|---|---:|---:|
| Classical Lyric & Character | 7/8 | 1 |
| Baroque and Early Dance Forms | 3/8 | 5 |
| Developmental and Large Forms | 3/8 | 5 |
| Contrapuntal and Polyphonic | 3/8 | 5 |
| Ground, Ostinato and Transformation | 3/8 | 5 |
| Song and Narrative | 2/8 | 6 |
| Blues, Gospel and Soul | 1/8 | 7 |
| Jazz and Improvisatory Forms | 1/8 | 7 |
| Latin and Caribbean Cycles | 3/8 | 5 |
| European and North Atlantic Folk | 3/8 | 5 |
| Mediterranean, Middle Eastern and North African Modal | 1/8 | 7 |
| South Asian Raga and Tala | 1/8 | 7 |
| East and Southeast Asian Traditions | 0/8 | 8 |
| African and Diasporic Groove | 0/8 | 8 |
| Minimal, Process and Experimental | 1/8 | 7 |
| Ambient and Electronic Texture | 1/8 | 7 |
| Club and Beat Music | 0/8 | 8 |
| Pop and Rock | 0/8 | 8 |
| Dramatic, Screen and Stage | 2/8 | 6 |
| **Total** | **35/152** | **117/152** |

## Newly callable styles

This tranche adds four structural presets using existing named engines, plus two Baroque presets with distinct rhythmic identities/form control instead of mapping those catalog names to a generic genre label:

- **Rondo** — fixes the form pool to A–B–A–C–A, using the current Rondo realizer.
- **Theme and Variations** — fixes the form pool to the duration-preserving variation realizer.
- **Erosion** — fixes the ground-form pool to the existing progressive ground-loss realizer.
- **Lineage** — fixes the ground-form pool to the existing parent-to-descendant ground transformation realizer.
- **Sarabande** — uses an explicit 3/4 accompaniment cell with the primary accent on beat two, separated from the existing Waltz/OomPah pattern. Its formal model and stylistic details remain Developing pending review.
- **Minuet and Trio** — uses a 3/4 minuet-specific first-beat accent cell and fixed ternary A–B–A return. The A/ReturnA harmony now keeps its quarter-note Minuet gait, while the B/Trio changes to eighth-note arpeggiated harmony after the opening thinning; instrument/timbre contrast remains unfinished and the style stays Developing pending expert review.

All six styles have a named catalogue mapping and a defined tempo band. Rondo, Theme and Variations, Erosion, and Lineage have fixed-form regression checks. The Minuet and Trio declares its B-section Arpeggio override in the spec, has both the A–B–A return test and a score-event test that requires offbeat harmony events in B while A and ReturnA retain on-beat quarter-note harmony, and surfaces that texture change in the generated identity description. Sarabande and Minuet each have rhythmic-cell tests for beat positions and accent hierarchy. Erosion and Lineage derive their starting instrumentation/motif/harmonic constraints from the Passacaglia preset and still need their own authored melodic DNA; they are structurally callable, not fully differentiated timbral or melodic identities.

The Style::ALL registry and the catalog's executable-route tests now cover the expanded 35. The known inert progression-field audit and intent-axis audit also include the ground-form additions. A style whose dedicated form owns its own harmonic/length logic must not be made to look complete by changing a progression field that the engine does not consume.

## Cultural and tradition-specific safety

The catalog no longer treats all unmapped styles in a culturally rooted constellation as review-free by default. Every entry in these constellations is marked as requiring knowledgeable musical review before any authenticity/foundation claim:

- Blues, Gospel and Soul;
- Jazz and Improvisatory Forms;
- Latin and Caribbean Cycles;
- European and North Atlantic Folk;
- Mediterranean, Middle Eastern and North African Modal;
- South Asian Raga and Tala;
- East and Southeast Asian Traditions;
- African and Diasporic Groove.

This is a review requirement, not a claim that every style is equally well understood or that a generated score is authentic. A culturally informed review may require specialist performers, instrument-specific constraints, tuning/raga/maqam knowledge, rhythmic-cycle correctness, phrasing, ensemble role, oral/aural transmission, and explicit terminology boundaries. The UI should prefer “inspired by” or a transparent hybrid label unless evidence justifies a stronger term. Review must be performed with the right practitioners; unit tests alone cannot establish it.

## Definition of done for each style

A style should only be promoted through these layers:

1. **Explicit identity:** structure/grammar, phrase behavior, harmonic system, rhythm, melodic language, ensemble role, performance dialect, production assumptions, and provenance for every borrowed tradition-specific claim.
2. **Executable route:** fixed and tested structural mechanisms; validated score; no empty, malformed, or out-of-range output; a clear map from the public catalog ID to the actual preset.
3. **Own signature:** tests show at least the declared core properties (for example, the rondo return, variation theme skeleton, repeated ground or cycle phase). A different enum value or tempo band is not enough.
4. **Determinism and controlled diversity:** same source/seed/version is reproducible; new seeds yield meaningful within-style variation; cross-style comparisons catch accidental duplicates and label bleed.
5. **Capability sensitivity:** tests identify which intent axes affect output and which are intentionally inert. A dead knob must be disclosed or wired into the actual engine.
6. **Fair baseline comparison:** compare matched lengths, loudness/renderer and task constraints to the existing baseline; retain negative controls; report confidence intervals and failed cases.
7. **Blind listening:** compare samples without labels, with experts and target listeners. Record task, sample identities, seed/config, reviewer expertise, outcomes and uncertainty. Do not promote to Foundation from implementation tests alone.
8. **Qualified review:** traditions requiring expert review cannot make authenticity/foundation claims without that evidence. User-created and open/public-domain corpora still need source and license provenance.

## Paradigm shift: from genre labels to executable, falsifiable contracts

A “style” should stop being a preset name attached to a shared generator. Treat it as a versioned, testable contract:

1. **Identity is a bundle of constraints, not one embedding or label.** A Sarabande contract should state its metre, characteristic beat-two weighting, phrase/harmonic behavior, intended pace, and ensemble assumptions. A Minuet-and-Trio contract must additionally demonstrate a recognizable A–B–A return and an audible/structural distinction between A and the Trio. Requirements differ by style; the contract declares which are invariant and which allow variation.
2. **Separate composition, performance, and audio.** A correct symbolic score does not prove believable rubato, articulation, instrumentation, mix, or sound design. Score-level tests, performance-realization metrics, and audio/listener tests must remain separate linked scorecards.
3. **One grammar engine, many honest presets — but no false equivalence.** Reuse a real engine when its invariants match. Route every named style through the public production dispatcher and expose the selected grammar/form plan. If two named styles share an engine or motif bank, record that fact and measure whether their signatures actually differ. Never mark a style complete merely because an enum, tempo band, or catalog mapping exists.
4. **Measure causes, not correlations alone.** Change valence, arousal, energy, duration, or tonal center one factor at a time, hold the rest fixed, and report which features actually move. An advertised control that is inert is a disclosed gap; a feature that moves for unrelated changes is not evidence of causal control.
5. **Diversity must be conditional.** Maximize within-style variation without erasing the style's invariants; penalize cross-style collisions without rewarding random noise. Report exact duplicates, structural neighbors, style-conditional diversity and human-labeled confusion separately.
6. **Test relationships, not just outputs.** Metamorphic tests apply controlled transformations where some properties should remain invariant. A tonic shift from C to G can reveal whether rhythmic placement and within-role intervals survive transposition, while absolute pitch/register is tracked separately. The initial transposition probe is diagnostic, not a universal gate: range-sensitive behavior must be identified and reviewed before turning this expectation into a hard contract.
7. **Human judgment is an instrument, not a final decoration.** Blind listeners should evaluate style recognition, coherence, creativity, structure, performance quality and preference separately. Tradition-specific evaluations need knowledgeable practitioners and must not infer authenticity from a classifier score.

### New executable benchmark

A Rust example, **examples/style_contract_benchmark.rs**, audits the real public composition path, **compose_with_grammar_plan(style.grammar_profile(), intent, style.spec())**. It reports selected grammar-plan families, declared form/accompaniment pools, score-validation findings, deterministic score/structure replay, seed-level exact-fingerprint diversity, matched-intent cross-style collisions, nearest structural neighbors, and declared-vs-observed sensitivity for valence/arousal/energy/bars. Paired sensitivity probes distinguish exact symbolic-score changes from structural changes. It also runs a **C-versus-G tonic-transposition metamorphic probe** for every style across the fixed seed/intent set, reporting the rate at which rhythmic placement and within-role pitch-interval signatures are preserved, the rate of exact event-stream fingerprint changes, and structural distance. A small unit test verifies that the signature is invariant to a uniform pitch shift.

The new **style-identifiability audit** compares each style against itself across different seeds while holding intent fixed, then compares different styles under matched seed/intent cases. The eight-seed sweep yields 3,920 within-style seed-pair observations across 35 styles and four intents, and 19,040 matched between-style comparisons across 595 style pairs. It reports within-style seed variation, matched between-style distance, their ratio, and every style-pair margin. For this analysis only, a **layer-balanced RMS distance** gives each of the fingerprint's 11 named layers equal aggregate weight, avoiding the automatic dominance of a longer layer simply because it contains more dimensions. Its symmetry, identity and equal-layer-weight properties have dedicated unit tests. The audit now reports **per-layer separation** for all 11 feature layers, including normalized within-style variation, matched between-style distance, and between-minus-within margins. Each style pair also exposes per-layer distances against each member's own seed-variation baseline, to show whether form, harmony, rhythm, contour, orchestration or another layer accounts for its apparent separation. The new **distributional diagnostics** add interpolated p10/p25/median/p75/p90, min/max, mean, finite/non-finite counts for within-style, between-style, nearest-neighbor, per-style-pair and per-layer distance samples. Every style pair also receives an **observed tail-separation margin**: p10 of matched between-style distance minus the larger of the two styles' p90 within-style seed-variation distances. A positive margin means the observed lower tail separates beyond both observed upper tails, a stricter diagnostic than a difference in means; the same margin is reported per fingerprint layer. This makes seed-sensitive tails and outliers visible instead of allowing a mean to hide a weak subset; empty samples are encoded as null statistics rather than fabricated zeroes. Non-finite distance values are excluded from quantile calculations but explicitly counted. The fixed sample counts are small and the quantiles are not confidence intervals; these remain structural diagnostics, not a learned perceptual metric. Do not turn them into a release gate until calibrated against blinded human style-confusion labels. The JSON report schema is now version 6. The fixed style-identity sweep uses eight seeds [0, 1, 7, 13, 29, 42, 73, 101] and four intents: 32 matched configurations per style pair, and 28 cross-seed comparisons per intent for each style.

The benchmark now also **tests its own validation oracle** with seeded malformed-score mutations: non-finite tempo, zero note duration, pitch below the configured MIDI floor, and overlapping melody events. The audit must detect the expected universal invariant rule for each mutation, and it gates the overall mechanical result on those detections. This separates “the generated sample passed” from “the validator demonstrably catches known faults.”

Run it with source provenance captured:

    GIT_COMMIT="$(git rev-parse HEAD)" cargo run --release -p symthaea-music-theory --example style_contract_benchmark

Or write a machine-readable report:

    GIT_COMMIT="$(git rev-parse HEAD)" cargo run --release -p symthaea-music-theory --example style_contract_benchmark -- /tmp/style-contract.json

The JSON marks the source commit as null when GIT_COMMIT is not supplied; preserve the JSON beside the exact source revision and CI evidence. Structural replay is checked independently of the score hash so a stable note stream cannot hide a nondeterministic generated form. The benchmark also runs the canonical **score_validation::validate_score** auditor, gates on its declared style-independent invariants (metadata, note bounds and voice monophony), and reports but does not universally gate style-dependent voice-leading findings or warnings. This prevents a classical tonal rule from being mistaken for an obligation shared by every idiom. The top-level **control_contract_audit** counts both declared-live controls with no observed effect at the sampled points and declared-inert controls that do show an effect; either discrepancy prevents **all_mechanical_checks_passed** from being true. This is a bounded counterfactual screen, not proof that a control is globally active or inactive.

The diversity census now uses **Style::ALL** for its current causal-diversity pass so new presets enter by default. Its historical 29-style pilot remains fixed for reproducibility. The new contract benchmark runs through the actual grammar dispatcher, including dedicated grammar families; that distinction matters because the older diversity census uses the shared composition-and-form path and is a different measurement layer. A named-style contract audit now inspects the generated harmony events for Minuet A, Trio B and da-capo ReturnA: it gates the mechanical result unless Trio B contains actual offbeat arpeggio attacks while A and ReturnA remain on the beat. This is an event-level composition test, not a claim of timbral authenticity.

**Neither tool evaluates human preference, style authenticity, cultural legitimacy or artistic superiority.** Both must emit diagnostic evidence, not a claim that the style passed a listener test.

### Structural skeleton before texture

A useful architectural principle is to treat a composition as a sequence of separately inspectable layers: first create and validate a form/motif/harmonic **skeleton**, then realize orchestration, accompaniment and performance details against that skeleton. Section texture is now declarative in `TextureSpec.section_accompaniment_overrides`: a style can select an accompaniment pattern by `SectionRole` rather than relying on a style-name conditional embedded in the renderer. The field defaults empty when loading old saved specs, duplicate role overrides are rejected, and identity descriptions disclose the section-specific texture. This gives future forms a reusable way to specify contrast that can be tested at the resulting event stream. This makes failures localizable: a structurally sound Minuet–Trio plan can be tested independently of whether the Trio's texture truly contrasts. Recent work on symbolic generation's long-range structure explores explicit skeleton guidance before texture realization ([CAST, *Scientific Reports*, 14 April 2026](https://doi.org/10.1038/s41598-026-46750-0)); Melothaea can test this principle in its symbolic grammar without adopting a neural model. Keep structure, texture, control adherence and human judgments as separate metrics.

## Proposed benchmark targets

These are **prospective engineering goals**, not current results. First collect a reproducible baseline and publish every failure.

| Dimension | Proposed target | Conditions and caveats |
|---|---|---|
| Mechanical correctness | 100% of presets validate; zero empty scores, non-finite/non-positive timelines, meter mismatches or deterministic-replay failures | Initial smoke sweep, then at least 10,000 fixed-seed/config cases. Fail closed on any unexplained violation. |
| Validator-oracle sensitivity | Detect 100% of preregistered universal-invariant mutations in the self-test suite | Expand the mutation families over time; seeded tests calibrate rule sensitivity but do not establish general test coverage or musical quality. |
| Control sensitivity | At least 95% of *declared-live* style/control pairs show an effect on a pre-registered intended feature across a seed sweep; every remaining no-effect case is documented | Keep intentionally inert controls distinct from bugs; report exact and structural changes separately. |
| Style identity, broad families | Macro-F1 ≥ 0.80 across expert-reviewed family labels on held-out seeds/templates | Use a frozen evaluator only as a diagnostic, with a blinded human panel as the deciding evidence. Balance classes and report per-family results. |
| Style identity, exact labels | Top-1 ≥ 0.65 and top-3 ≥ 0.85 on the reviewed style subset | Initial target for a 35-style tranche, not a universal 152-class claim. Report a confusion matrix and confidence intervals; do not collapse culturally distinct labels to inflate the score. |
| Pairwise preference | ≥ 60% preference over the strongest *matched* baseline, with the 95% confidence-interval lower bound above 50%, replicated on held-out prompts and listeners | Score composition, style adherence, and performance separately. Match duration, loudness, instruments/rendering and selection budget. |
| Objective metric validity | Rank correlation ρ ≥ 0.50 between the pre-registered aggregate metric and expert judgments on held-out styles, with confidence intervals | Validate metric agreement rather than optimizing for the metric; publish the dimensions and groups where agreement fails. |
| Cross-style bleed | Reduce human-confirmed cross-style confusion by ≥ 30% relative to the frozen baseline without reducing within-style coherence or preference | Calibrate the structural-neighbor threshold using blinded human labels; raw feature distance alone is not ground truth. |
| Style identifiability under seed variation | Eight fixed seeds, 32 matched configs per style pair, per-pair/per-layer distributions and the p10-between minus max(p90-within-A, p90-within-B) observed tail margin | Publish all weak pairs; the margin is a diagnostic, not a confidence bound. After acquiring blinded labels, target positive tail margin for ≥90% of expert-reviewed style pairs on held-out seeds while reporting all exceptions. Do not qualify authenticity from geometry alone. |
| Reproducibility | Every report records engine version, source/head SHA, configuration, seed set and corpus provenance; reruns reproduce exact symbolic hashes | Keep benchmark fixtures immutable, version threshold changes, and never count queued/skipped/stale CI as a pass. |
| Metamorphic behavior | Establish a measured transposition-invariance baseline; after practitioner review, target ≥ 95% rhythmic/interval-signature preservation for styles that declare transposability | Report exceptions by style and register; do not hide range-specific instrumentation or pitch limits behind an aggregate. The current C→G probe is diagnostic, not a pass gate. |
| Runtime and resource use | Report p50/p95 generation latency, memory and score size for 8/16/32-bar cases on named reference hardware | Establish the baseline before adopting a hard latency SLO; symbolic composition and audio rendering need separate budgets. |

### Benchmark design that resists self-deception

- **Matched comparisons:** same intent, tonic, length, rendering loudness and instrument budget; compare against the current engine, a simple style-agnostic baseline and human-authored controls where licensing permits.
- **Held-out evaluation:** hold out seeds, motif templates, prompts, and—where the tradition calls for it—composers/performers. Report tuning items separately from final items.
- **Negative controls:** intentionally remove the defining accent, change the required metre, shuffle the style label, or replace a dedicated preset with the nearest generic fallback. The benchmark should detect those regressions. Retain controls even when the system improves.
- **Blinded panels:** randomize labels and output order; collect independent ratings before discussion. Use clustered/bootstrap confidence intervals over both listeners and source prompts; report expertise, disagreements and missing responses.
- **Cultural governance:** recruit qualified reviewers for the specific tradition, pay/credit them appropriately, and let them flag a misleading label. Do not average away serious authenticity concerns into one universal “music score.”
- **No hidden cherry-picking:** record candidate count, rejection/selection rules and any manual edits. Report one-shot plus best-of-2/4/8 results as separate budget curves; never compare a best-of-N result with a one-shot baseline. Calibrate selectors against blinded human choices and retain the selector-free outputs.
- **Release gates:** mechanics first, frozen objective report second, blind listening third, qualified cultural review where needed. A style is Foundation only after the relevant evidence exists, not because an enum and tests landed.

### External evaluation references

- [MIREX 2026 Symbolic Music Generation](https://www.music-ir.org/mirex/wiki/2026%3ASymbolic_Music_Generation) uses blind subjective ratings for prompt coherence, creativity, structuredness and overall musicality. It generates eight candidates per test sample and permits cherry-picking from that fixed budget; Melothaea should therefore report **one-shot** and **best-of-8** results separately, with selection rules and candidate budget frozen in advance. Otherwise a larger search budget can masquerade as a better composer.
- [MIREX 2026 CMI-RewardBench results](https://music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench_Results) compare automatic reward models against human pairwise preference. They demonstrate a practical evaluator benchmark, but its scores are not directly comparable to Melothaea's proposed style-recognition or composition targets.
- [Recent advances in music generation: methods, evaluation, and challenges](https://doi.org/10.1007/s10462-026-11582-x) reviews persistent long-range structure, controllability and evaluation gaps. It supports keeping structural diagnostics, causal control tests, and human evaluation as separate result dimensions.
- [A Survey on Evaluation Metrics for Music Generation](https://arxiv.org/abs/2509.00051) highlights limitations in objective-metric/human-perception alignment, cross-cultural bias and standardization.
- [Benchmarking Music Generation Models and Metrics via Human Preference Studies](https://arxiv.org/abs/2506.19085) reports a large human-preference study (12 systems, 6,000 generated songs, 15,000 pairwise comparisons, 2,500 participants). This gives a scale reference—not a minimum viable requirement for our first controlled panel.

## Recommended remaining sequence

**Next: deepen Classical and Baroque form identity.** The initial Minuet-versus-Trio rhythmic contrast is now implemented, but the Trio's distinct instrumentation/timbre and fuller performance dialect remain open. Build Gavotte, Gigue, Allemande, Toccata, French Overture, Rondo-adjacent developmental forms and Elegy; expand the Sarabande's current rhythm-focused preset into a reviewed form/performance model from actual formal/rhythmic differences, not aliases. Reuse existing form machinery when its invariants really match; add a new engine only when the structure cannot be represented faithfully.

**Then: song, blues and jazz.** Build verse/chorus, through-composed, sea shanty, art song/Lied, Delta/Country/Chicago blues variants, Gospel call-response, Soul/R&B/Neo-Soul, swing, bebop, cool/hard/modal/free jazz and jazz waltz. Require distinct groove/harmonic/phrase identity tests before making these callable under specific labels.

**Then: cycles and regional folk.** Build explicit cycle phase, instrument roles, and allowed subdivisions for son, salsa, mambo, cha-cha, samba, tango variants, Irish/Scottish/English/Nordic/Balkan/Klezmer traditions; acquire knowledgeable reviewers for culturally specific styles.

**Then: modal traditions and Asian/African styles.** Do not map maqam, makam, dastgah, raga/tala, gagaku, honkyoku, min'yo, guqin, sizhu, gugak, gamelan, piphat, Afrobeat, highlife, juju, soukous, amapiano, gqom, reggae or dub to Western functional-harmony substitutes and call them complete. Each needs domain-specific tuning/modes or cycle/ensemble models and informed evaluation.

**Then: process/electronic/club/pop/screen.** Complete phase music, additive process, spectral/aleatoric, dark ambient, drone, Berlin School, IDM/glitch/vaporwave/electroacoustic, house/deep house/techno/trance/DnB/UK garage/breakbeat/hip-hop, pop/indie/dream/synthpop/rock/prog/post-rock/folk-rock, musical theatre, noir/epic/horror/sci-fi and adaptive game-score states. Include arrangement, production and long-form tests as relevant; notes and tempo alone cannot finish these styles.

These batches are a prioritization, not permission to skip any of the 117 remaining unmapped catalog IDs.

## Verification status

This branch is the first style-completion tranche, not the final 152-style delivery. Six style routes, catalog coverage tests, control-contract probes, a diagnostic tonic-transposition audit, a layer-balanced style-identifiability audit, and validator fault-injection self-tests are authored. Rust formatting, compilation, unit tests, benchmark execution and listening checks must pass or be explicitly reported on the exact final commit before this tranche is qualified. A queued, skipped, stale or missing CI result is not a pass.

No audio samples or third-party recordings are added to this branch. Listening and benchmark corpora must use original works, verified public-domain recordings, or explicitly licensed/authorized sources.
