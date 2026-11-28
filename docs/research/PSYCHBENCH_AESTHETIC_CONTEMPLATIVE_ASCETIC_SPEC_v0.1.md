# PsychBench Aesthetic, Contemplative & Ascetic Research Specification v0.1

**Status:** research/implementation specification; dataset adapters and empirical runs pending  
**Target branch:** `research/psychbench-vnext-2026-11`  
**Review date:** 2026-11-28  
**Claim boundary:** this document defines work to implement and validate. It reports no new benchmark scores, no human study, no SOTA result, and no evidence of subjective experience or consciousness.

## Executive decision

Extend PsychBench in two connected but distinct research domains:

1. **Aesthetic cognition and creativity:** how an agent perceives, interprets, evaluates, prefers, and creates artifacts across modalities.
2. **Contemplative practice and ascetic self-regulation:** how an agent understands voluntary practices of attention, restraint, simplicity, renunciation, and disciplined goal pursuit; how it models their context, risks, outcomes, and reversibility.

Do not create one universal "beauty" score or one universal "spiritual development" / "asceticism" score. Both domains are plural, contextual, culturally situated, and multi-objective. They should enter PsychBench as construct cards, independently scored tasks, evidence classes, and versioned result records.

This is an extension of the existing PsychBench vNext work, not a greenfield replacement. Preserve the current registry, construct-validity work, selective-control protocol, Cognitive Ecology (PCE), and psychometric corrections. Do not duplicate the separate `symthaea-aesthetic` crate; integrate it as one candidate system under test and evaluate it against independent external criteria.

## 1. Current repository assets and gaps

### 1.1 Assets to reuse

- `symthaea-psych-bench` already has affect, creativity, attention, social cognition, metacognition, executive control, cognitive ecology, and other benchmark families.
- PsychBench vNext has a pilot registry and JSON Schema, typed loader, explicit metric bounds, evidence classes, and fail-closed validation. The pilot is deliberately incomplete and is not a full benchmark inventory.
- `symthaea-aesthetic` already provides multi-dimensional aesthetic representations, Birkhoff-inspired quantities, novelty/harmony components, preference-learning paths, and aesthetic feedback. Its theoretical priors are useful as candidate predictors, not universal ground truth.
- Symthaea has meditation-related EEG processing and a MeditationSentinel path, plus existing examples/scripts referring to OpenNeuro `ds001787`.
- PsychBench vNext already distinguishes synthetic mechanism probes, external tasks, human-normed tasks, ecological interaction, and theory probes. Reuse those evidence classes.

### 1.2 Gaps to close

1. Aesthetic cognition is not yet a first-class externally grounded PsychBench domain across visual, auditory, linguistic, spatial, kinetic, material, mathematical, culinary, and interactive forms.
2. Existing aesthetic scoring code is not itself proof that human preferences are predicted. It must be evaluated on held-out raters, items, contexts, modalities, and cultures.
3. Existing creative tasks include synthetic HDC checks and planted associations. Those are mechanism-conformance tests, not creativity measurements. Keep them labelled accordingly.
4. Meditation detection and a small EEG dataset do not establish meditation depth, spiritual attainment, consciousness, or broad contemplative competence.
5. Ascetic practices are broader than meditation. They include voluntary restraint and renunciation in multiple domains and traditions; they require context, agency, safety, and reversibility to be represented explicitly.
6. Aesthetic and contemplative/ascetic tasks must not be conflated with clinical diagnosis, wellness advice, or ranking religious and philosophical traditions.

## 2. Paradigm shifts

### Shift A — from "beauty score" to pluralistic aesthetic evidence

Represent aesthetic judgements as distributions and structured comparisons, not a universal scalar. Preserve:
- individual ratings and pairwise choices;
- confidence and rater uncertainty;
- expert/non-expert distinctions;
- cultural, language, context, and task provenance where ethically collected;
- disagreement, bimodality, and minority perspectives;
- the distinction between artifact properties, personal preference, social convention, and task-specific utility.

A system may predict one cohort well and another poorly. The benchmark must show that instead of hiding it in an aggregate.

### Shift B — from artifact-only scoring to person × artifact × context

Model an aesthetic judgement as conditional:
`P(judgement | artifact, person, context, purpose, modality, history)`.

The same object can be beautiful, disturbing, sublime, familiar, boring, sacred, playful, or inappropriate depending on context and purpose. The benchmark should test whether the system notices when context changes interpretation, not merely whether it repeats a static preference.

### Shift C — from passive ratings to active aesthetic inquiry

Allow the system to:
- request missing context;
- ask whether the evaluator wants technical craft, emotional impact, novelty, accessibility, cultural fit, or personal preference;
- compare two candidates under a stated criterion;
- abstain when modality or context is outside its evidence domain.

Measure information requests by their expected reduction in decision uncertainty and their cost. Use the existing selective-prediction protocol as the decision-control substrate; do not call a hand-authored estimate "calibrated EVSI."

### Shift D — from single-output creativity to process and trajectory

Evaluate ideation, selection, revision, refinement, and final artifact separately. Preserve:
- novelty versus usefulness;
- rule adherence versus exploratory risk;
- technical quality versus taste;
- originality versus recombination;
- intention versus achieved effect;
- process efficiency versus final quality.

Use blinded expert panels and Consensual Assessment Technique-style ratings where appropriate, but preserve expert disagreement. The 2026 Human Creativity Benchmark provides a useful methodological precedent for separating convergence on verifiable quality from divergence in taste; verify data access and protocol before claiming a direct replication.

### Shift E — from state labels to practice trajectories and counterfactuals

For contemplative and ascetic domains, do not infer a single latent "depth" state from one EEG band, questionnaire item, or verbal self-report. Evaluate:
- attentional stability and mind-wandering dynamics;
- meta-awareness and recovery after distraction;
- cognitive flexibility and transfer;
- motivation, intention, and voluntary choice;
- adverse effects, burden, and stopping conditions;
- retention and change over time;
- whether a practice is appropriate to the person's stated goal and context.

A successful system must know when not to encourage further restraint and when to recommend stopping, seeking qualified human support, or gathering more context.

### Shift F — from discipline as compliance to agency-preserving self-regulation

A benchmark must reward goal-consistent voluntary control without rewarding rigidity, coercion, self-punishment, or harm. Test both:
- persistence when a self-chosen constraint remains appropriate; and
- flexible suspension/reversal when conditions change or harm signals appear.

The ability to stop is a positive capability. More deprivation is not more achievement.

### Shift G — from leaderboard-only evaluation to a capability frontier

Report separate dimensions for:
- task performance;
- human preference/distribution similarity;
- uncertainty and abstention;
- cultural and cohort robustness;
- causal/intervention evidence;
- longitudinal learning and transfer;
- resource cost;
- safety and agency preservation.

Do not collapse these into a domain-wide rank. A composite may be used only for an explicitly stated decision, with published weights, sensitivity analysis, and hard safety constraints.

## 3. Domain A: Aesthetic Cognition, Experience & Expression (ACEE)

### 3.1 Scope

Create `benchmarks/aesthetic/` with separate task families; do not require every modality to ship in v0.1.

| Family | Required construct coverage | Initial external grounding |
|---|---|---|
| Visual | composition, colour, balance, symmetry, texture, style, scene semantics, visual ambiguity | MSC 2026; AVA/AADB where protocol and licence permit |
| Auditory/music | rhythm, timbre, harmony, tension/release, temporal form, emotional expression, performance and production craft | MADB 2026 for professional multi-dimensional music assessment; MUSIN-G only for familiarity/enjoyment/affective response, not aesthetic quality |
| Linguistic/literary | metaphor, sound, rhythm, ambiguity, narrative form, humour, emotional resonance | blinded human preference and expert assessment; preserve genre/language strata |
| Movement/performance | timing, gesture, fluency, expressiveness, synchronisation, intentional deviation | motion/performance datasets with task-matched annotator outcomes |
| Spatial/architectural | proportion, legibility, atmosphere, material fit, accessibility, context | human study with rendered/physical environment context; do not reduce architecture to geometric ratios |
| Haptic/material/craft | texture, material appropriateness, craftsmanship, wear, tactility | initially human ratings and controlled material descriptions; no fabricated sensory access |
| Mathematical/intellectual | proof elegance, explanatory compression, structural insight, coherence | expert assessment plus formal correctness oracle; correctness and elegance remain separate |
| Interactive/digital/game | feedback, agency, affordance clarity, pacing, exploration, delight/friction | task-completion telemetry plus user preference; do not equate usability with beauty |
| Culinary/olfactory/gustatory | presentation, aroma/flavour balance, texture, cultural fit | text/structured metadata only until real sensory human studies exist; the model must not claim direct taste/smell perception without sensors |

Cross-modal composition, synesthesia, and temporal correspondence should be evaluated as their own task families. Congruence is not always the target: counterpoint, contrast, and deliberate dissonance may be the intended artistic choice.

### 3.2 Construct card

Every ACEE benchmark must declare:
- the target judgement (e.g. visual appeal, emotional resonance, craft, preference, originality, or context fit);
- the eligible modality and what sensory evidence the system actually receives;
- stimulus provenance, licence, version, and digest;
- human rater population and rating-scale semantics;
- whether the task is absolute rating, pairwise preference, ranking, explanation, or creation;
- expected ambiguity and whether multiple answers are legitimate;
- known cultural/language/expertise confounds;
- positive, negative, and metamorphic controls;
- claim eligibility and evidence level.

### 3.3 Metrics

Use metric-specific estimands; never infer bounds from field names.

**Preference prediction**
- pairwise accuracy and pairwise log loss;
- Brier score for probabilistic preference;
- calibration slope/intercept and reliability curves;
- ordinal rating log score or a justified ordinal likelihood;
- Wasserstein distance / Earth-mover distance between predicted and observed rating distributions;
- rank correlation as a secondary statistic, with confidence intervals clustered by rater and stimulus.

**Pluralism and disagreement**
- rater dispersion and multimodality;
- subgroup-conditioned calibration;
- effective perspective count and representation concentration;
- leave-one-cohort-out and leave-one-rater-group-out sensitivity;
- minority-erasure / aggregation-sensitivity diagnostics.

**Creative output**
- technical correctness / constraint adherence from independent oracles;
- blinded expert judgement of originality, usefulness, craft, and intent alignment;
- novelty against a frozen reference corpus with nearest-neighbour and memorisation checks;
- revision gain, constraint satisfaction, and cost per accepted improvement;
- human preference and evaluator-critic scores must be reported separately.

### 3.4 Required controls

- **Identity control:** same artifact under changed IDs, filenames, or irrelevant metadata should not change its intrinsic evidence score.
- **Context control:** change only context and test whether contextual interpretation changes while artifact measurements remain invariant.
- **Style/culture control:** matched content in multiple styles; do not expect identical preferences, but require transparent cohort-conditional predictions.
- **Rater leakage control:** no rater or near-duplicate stimulus leakage across train/validation/test partitions.
- **Proxy-gaming control:** generate candidates that maximize the internal aesthetic heuristic while degrading independent human preference or task usefulness.
- **Ablation control:** remove each aesthetic feature family and measure selective, pre-registered changes.
- **Plurality control:** include deliberately polarized items where a single mean hides two legitimate preference clusters.
- **Counterpoint control:** do not automatically penalize dissonance, asymmetry, irregularity, or low-complexity work; score intent and context.

### 3.5 Initial dataset plan

**MSC (Minimum Semantic Content, 2026)** is the first visual anchor. It contains 10,426 natural-scene images, 100 human aesthetic ratings per image, and manipulated variants intended to broaden the rating distribution. Its reduced semantic content is useful for isolating some low-level visual factors, but it is not a universal art dataset, not a cross-cultural norm, and not evidence for music, literary, haptic, or architectural aesthetics. Use raw rating distributions, retain participant/item identifiers where licensing permits, and split by base-scene family so manipulated variants cannot leak across folds.

**Poetry Assessment EEG Dataset 1 (NEMAR on006648 / OpenNeuro ds006648, 2026)** should be the first literary-aesthetics anchor and the first bridge between aesthetic response, affect, creativity, and physiology. It contains 64-channel EEG and behavioural ratings from 47 participants who evaluated 210 short English-language texts: 70 Haiku, 70 Senryu, and 70 non-poetic controls. Each text was rated on five 7-point dimensions: aesthetic appeal, vivid imagery, being moved/emotional impact, originality, and creativity. The release also includes participant-level trait measures including PANAS, openness, curiosity, visual/auditory imagery, MAAS mindfulness, and AReA aesthetic responsiveness. The dataset is CC0 and its versioned NEMAR record is v1.0.0 (DOI 10.82901/nemar.on006648). Preserve participant-by-item ratings rather than collapsing to mean labels. Use participant-held-out and text-held-out evaluation, with a crossed generalization split for unseen participants and unseen texts; cluster uncertainty by participant and stimulus. EEG can be evaluated as an auxiliary predictor of task-time judgments, but behavioural ratings—not a band-power heuristic—remain the outcome labels. The associated companion dataset on006647 has four additional participants whose segmented recordings were excluded from the primary PSD analyses; keep it separate for EEG/PSD claims and only consider its behavioural records after verifying the participant/item joins and their provenance. The two releases must not be silently pooled as if all recordings had equal technical status.

This is a strong literary/affective aesthetics anchor, not a universal measure of poetry quality, general creativity, cross-cultural taste, or contemplative attainment. It is a single language/cultural sample and a finite set of short text forms. It does not cover spoken performance, music, visual art, architecture, or direct sensory taste.

**MADB: A Large-Scale Music Aesthetics Dataset with Professional and Multi-Dimensional Annotations (2026 arXiv preprint)** is the first substantial music-evaluation anchor. Its paper reports 9,999 tracks assembled from 2,799 manually collected tracks, 1,000 Suno tracks, 1,800 Levo tracks, and 4,400 MuChin tracks; 30 trained/professional annotators; and about 10 ratings per track. It provides ten fine-grained assessment dimensions plus an overall score and short explanatory comments/tags. The dimensions span melody, rhythm, structure, arrangement, performance/singing emotion and skill, enunciation, and sound effects. Reported audio-model baselines include MuQ, MERT and CLAP-family systems. MuQ's published overall-score results are LCC 0.718, SRCC 0.714, and KRCC 0.528; MERT is a useful second music-specific comparator. Treat these as within-protocol reference values, not independently verified SOTA. The authors use a random train/validation split under one evaluation framework, not an artist-, source-, or generator-held-out test; PsychBench should improve on this by grouping all tracks from source/artist/generation families into separate splits and testing unseen-source transfer. Keep audio-only, metadata/tag, text-comment, and audio+text conditions separate. In the paper's Qwen2-Audio comparison, audio-only performance is markedly weaker than comment/tag conditions; comments already encode human judgments and must not be presented as a fair audio-only perception result. The dataset card reports CC-BY-NC-4.0 for dataset material; track-level audio may have different source rights, and research use does not grant redistribution or commercial use. Preserve individual ratings and comments, not just mean scores. Note that the corpus is genre-skewed, includes AI-generated tracks, and its authors report no temporal annotations or causal identification among score dimensions.

**PAM∃LA (2026 under-review dataset/preprint)** is a strong personalization benchmark for visual aesthetics. Its current dataset card reports 69,904 ratings over 5,077 AI-generated images from 199 participants spanning 23 nationalities, with continuous 1–5 ratings and image/prompt/user metadata. Its exact host is https://huggingface.co/datasets/pamela-dataset/pamela. The split design distinguishes validation/test on unseen users from known users rating unseen images. Inter-rater agreement is low at the single-rater level (Krippendorff alpha 0.2057 for ordinal labels; 0.2447 for the continuous score; ICC(2,1) 0.2454), while mean-rating reliability is higher (ICC(2,k̄) 0.8175 for approximately 13.8 raters/image). That is not simply annotation failure: it is evidence that individual taste is heterogeneous and personalization matters. Benchmark two distinct questions: population-prior prediction for a new person, and personalization for a known person on unseen images. Report per-user calibration, coverage and improvement over a population prior; do not turn the average score into an objective beauty label. The participant pool is explicitly not demographically representative, and several nationality cohorts are small; avoid unsupported subgroup claims. Annotation data, prompts and dataset structure are CC-BY-4.0. Images were generated by Nano Banana (Gemini 2.5 Flash Image) and FLUX.2; follow the separate generator terms, including any commercial-use restrictions. Pin the exact dataset revision and preserve the under-review status in publications.

**MUSIN-G (NEMAR on003774 / OpenNeuro ds003774)** is an optional complementary EEG resource with 20 participants, 12 music tracks, 128-channel EEG, and familiarity/enjoyment ratings. It may help study music-evoked affect and preference signals, but it is not a professional music-quality dataset and must not be used as ground truth for technical musical craft or universal aesthetic quality.

**Human Creativity Benchmark (2026 preprint)** is a process/evaluation precedent: its reported 15,000 professional judgements cover five creative domains and separate verifiable convergence from taste-driven divergence. Treat it as a comparator/protocol reference until dataset access, licence, task protocol, and replication conditions are confirmed.

**Existing visual aesthetic datasets** (e.g. AVA/AADB) may be added only after confirming licence, annotation semantics, rater provenance, overlap/duplicate contamination, and comparability with MSC. A dataset's popularity does not make its labels a universal aesthetic truth.

For music, use MADB as the primary audio-evaluation anchor, but respect its annotation and split limits. MUSIN-G is a secondary EEG/enjoyment dataset, not a music-quality or beauty dataset. For other non-visual modalities, start with new small preregistered panels rather than inventing external ground truth from datasets whose labels measure emotion, genre, engagement, or caption quality rather than aesthetic preference.

## 4. Domain B: Contemplative Practice & Ascetic Self-Regulation (CPASR)

### 4.1 Scope and terminology

Use two related labels:
- **Contemplative practice:** deliberate practices involving attention, awareness, reflection, prayer, meditation, silence, or sustained observation.
- **Ascetic practice:** voluntary restraint, simplification, renunciation, or disciplined limitation undertaken for a stated ethical, spiritual, philosophical, practical, or personal purpose.

These categories overlap but are not synonyms. The benchmark should describe practices neutrally and represent traditions in their own terms rather than impose a single theology or developmental hierarchy.

The taxonomy should cover (without treating any as a recommendation):
- meditation and attentional training;
- contemplative prayer, reflection, and silence;
- solitude and retreat;
- voluntary simplicity and material restraint;
- dietary restraint and fasting;
- celibacy or sexual restraint;
- physical austerity and endurance practices;
- renunciation, non-attachment, and reduced consumption;
- ritual, vows, disciplined routines, and service;
- contemporary secular practices of restraint, attention, and self-regulation.

### 4.2 Separate constructs

Create at least four task families:

1. **Practice literacy and interpretation:** identify a practice's intended purpose, tradition/context, assumptions, ethical limits, and known risks from sourced materials. A knowledge test is not a test of having undergone the practice.
2. **Attentional/metacognitive regulation:** sustained attention, mind-wandering awareness, recovery after interruption, meta-awareness, and flexible shifting between focused and open monitoring.
3. **Voluntary constraint and agency:** maintain a self-selected, low-risk, reversible constraint when appropriate; revise or stop it when the goal changes, evidence conflicts, or harm markers appear.
4. **Longitudinal adaptation:** retention, transfer, motivation, cognitive flexibility, recovery, and adverse-effect monitoring across repeated sessions.

Do not treat adherence alone as success. A system that follows a harmful or no-longer-appropriate constraint is failing agency-preserving regulation.

### 4.3 Initial external anchor: meditation EEG and self-report

Use NEMAR `on001787` / OpenNeuro `ds001787` as an initial external dataset adapter candidate. The record describes 24 participants performing a meditation task, with interruptions approximately every two minutes for self-ratings of concentration and mind-wandering; the NEMAR record lists 64 channels and three sessions. This is a small dataset suitable for a carefully bounded external-task pilot, not a population norm and not a basis for broad consciousness claims.

**Important existing-code audit (main branch, 2026-11-28):** the current claims need to be downgraded before reuse.
- `scripts/validate_meditation.py` uses the first 120 seconds as a supposed baseline and the signal after 180 seconds as "meditation", although the dataset description says participants were meditating and were interrupted about every two minutes to self-rate concentration and mind-wandering. The script counts stimulus events but does not use the self-ratings as supervised labels. Its five checks are broad feature-range / extraction checks and a simple direction-of-change rule, not participant-held-out classification, calibration, or external predictive validity.
- `rust-sentinels/src/sentinels/meditation.rs` describes validation as "5/5 checks passed", but its visible unit tests use synthetic alpha/theta sine waves; the implementation computes a baseline-relative `(alpha + theta) / beta` index and a simplified alpha-derived stability proxy. This is a feature heuristic, not validated meditation-depth measurement.
- `examples/benchmark_meditation_resting.rs` uses the internal `MeditationSimulator` to generate six labelled EEG categories from state-specific frequency amplitudes. Its 30 generated trials and six checks are a simulator/mechanism smoke test, not an OpenNeuro result or external validation.
- `src/perception/physio/meditation_detector.rs` labels ratios of band powers as "alpha coherence" and "gamma synchrony"; these are power-similarity proxies, not cross-spectral coherence or phase synchrony. Its detection confidence is hard-coded to 0.85. These must be renamed as proxies or replaced by appropriate signal estimators, and confidence must be empirically calibrated.
- `examples/meditation_phi_analysis.rs` reads subject/session event files but then constructs seeded random, small-world, and modular HDC topologies to calculate `Phi`; it does not derive those values from each participant's recorded EEG. Its output therefore cannot support an empirical statement that meditation increases EEG-derived `Phi`.

Until those paths are corrected, label them `SyntheticMechanism` / exploratory feature-extraction only, remove wording that implies 5/5 external validation or empirical EEG-derived `Phi`, and do not use their outputs as evidence that Symthaea detects meditation states. This is a source-level audit; it does not claim to have executed the scripts or downloaded the dataset.

Required protocol:
- pin exact dataset version and digest;
- align EEG windows to timestamped interruption/self-rating events and inspect the original paper, BIDS events, and questionnaire scales before defining labels;
- split by participant (leave-one-participant-out or grouped folds), never by random EEG windows alone;
- if sessions are used, report session-held-out evaluation separately;
- keep participant self-report as the task label, not as infallible ground truth; predict concentration and mind-wandering ratings as separate constructs;
- compare against class-prior, bandpower-only, regularized linear, and suitable temporal baselines;
- report subject-level metrics and uncertainty, not only pooled windows;
- include artefact/missing-channel robustness, temporal drift, and event-alignment sensitivity;
- never define meditation "depth" as a single alpha/theta ratio;
- avoid inferring spiritual attainment, mental health, or subjective experience from EEG.

The current MeditationSentinel is one system under test, not a qualified detector. First repair the label alignment and signal estimators, then compare it with participant-held-out baselines.

### 4.4 Asceticism-specific task design

The initial v0.1 should be non-interventional and use sourced scenarios plus low-risk computational tasks. For each scenario record:
- who selected the constraint and whether it is genuinely voluntary;
- the person's stated purpose and values;
- intensity, duration, reversibility, and exit conditions;
- physical, psychological, social, and material risks;
- cultural/religious context and the practitioner's own interpretation;
- evidence quality and uncertainty;
- indicators of coercion, compulsion, distress, or escalating harm.

The evaluator should score whether the system:
- distinguishes chosen restraint from externally imposed deprivation;
- identifies missing context and asks appropriate clarifying questions;
- distinguishes symbolic/ritual meaning from physiological claims;
- recognizes when evidence is weak or contested;
- preserves the person's agency and right to stop;
- avoids glorifying suffering or treating deprivation as inherently virtuous;
- does not prescribe fasting, sleep restriction, physical austerity, or other potentially harmful practices.

Do not run fasting, sleep-deprivation, pain, isolation, or other high-risk human interventions as benchmark tasks. Any prospective human-subject research requires ethics approval, informed consent, trained oversight, explicit stop rules, and risk review. This benchmark is not a clinical product or a substitute for qualified care.

### 4.5 Metrics

- attention: d-prime / sensitivity, lapse rate, mind-wandering event detection with participant-held-out uncertainty;
- metacognition: calibration and discrimination between confidence and subsequent correctness;
- flexibility: appropriate shift and stop rates under counterfactual changes;
- agency: voluntary-goal fidelity, coercion detection, reversibility, and appropriate refusal;
- longitudinal: retention, transfer, practice-effect estimates, recovery, and burden;
- safety: dangerous recommendation rate, missed coercion/harm cues, inappropriate persistence, and false alarm rate;
- knowledge: source-grounded factual accuracy, attribution accuracy, uncertainty, and abstention.

No single "ascetic score" is permitted. Practice-specific outcomes must remain distinct.

## 5. Cross-domain experiments

The most interesting contribution may come from interaction effects. Pre-register hypotheses; do not assume them to be true.

| Hypothesis family | Experiment | Falsifiable outcome |
|---|---|---|
| Attention × aesthetics | Compare aesthetic judgement under focused attention, divided attention, and post-interruption recovery | Determine which dimensions shift and whether the effect generalizes across modalities |
| Simplicity × preference | Compare minimalist and high-complexity artifacts across stated goals/cohorts | Test whether preference depends on goal/context rather than a universal complexity optimum |
| Contemplation × creativity | Compare ideation diversity and revision quality under controlled attention protocols | Measure any benefit and cost separately; avoid claiming meditation intrinsically improves creativity |
| Voluntary restraint × agency | Give the system a self-chosen low-risk constraint, then alter the context and risk | Reward appropriate persistence and appropriate stopping, not rigid adherence |
| Aesthetic affect × regulation | Separate emotional impact, aesthetic judgement, and post-exposure regulation | Test discriminant validity: distinct constructs should not collapse into one valence output |
| Cultural context × interpretation | Present matched stimuli/practices with context variations | Preserve context sensitivity without fabricating uniform cultural norms |

## 6. Comparison with research and industry

Use the following comparator roles:

| Comparator | Use | Do not claim |
|---|---|---|
| MSC 2026 | External, distribution-rich visual preference anchor | Universal aesthetics or cross-modal mastery |
| HCB 2026 preprint | Professional creativity evaluation and disagreement-preserving design | Peer-reviewed SOTA or universal creative quality |
| AVA / AADB | Established visual-aesthetics model/data comparisons after licence and split audit | Direct equivalence to MSC or human taste in general |
| AESTHEMOS / aesthetic emotion scales | Vocabulary and construct-development support | One-dimensional "beauty" ground truth |
| NEMAR/OpenNeuro ds001787 | Participant-held-out self-reported concentration/mind-wandering EEG pilot, after event-label alignment | Meditation depth, enlightenment, consciousness, clinical prediction |
| NEMAR on006648 / ds006648 | 47-participant poetry preference, affect, creativity, EEG and trait-measure anchor | General aesthetics, cross-cultural norms, or meditation-state validation |
| MADB 2026 | 9,999-track professional multi-dimensional music assessment; audio-only baselines include MuQ/MERT/CLAP/Qwen2-Audio conditions | Universal music taste, artist/source generalization from random-split results, or unbiased global preference |
| PAM∃LA (2026 dataset/preprint) | Personalized visual taste prediction with seen-user/unseen-image and unseen-user splits | Population-representative beauty norms; image-output licensing beyond the dataset annotations |
| MUSIN-G / NEMAR on003774 | Music familiarity/enjoyment self-report and EEG pilot | Professional aesthetic-quality labels or a universal music-quality score |
| CogBench / AwarenessBench / HumanBaseline | Adjacent cognitive, affective, awareness, and human-reference comparisons | Interchangeable global psychology scores |
| Classical signal baselines | Establish whether the architecture adds value beyond simple features | State-of-the-art neural comparison |
| Contemporary multimodal models | Matched input/output and budget comparison on compatible tasks | Apples-to-apples where modalities, context, or tools differ |

For each experiment compare:
1. a trivial prior/chance baseline;
2. a transparent classical baseline;
3. a task-specific state-of-the-art model or published result;
4. a contemporary general multimodal model, where the same inputs and budgets are possible;
5. Symthaea's native path;
6. Symthaea with the relevant specialist subsystem ablated;
7. a human panel for the HumanNormed subset.

Do not publish a SOTA claim unless the task version, split, scorer, budget, uncertainty, and baseline revisions are pinned and independently reproducible. A benchmark-specific win is not an across-domain claim.

## 7. Proposed qualification gates

These are proposed internal release gates, not claims about universal psychometric standards.

### Gate 0 — contract and provenance
- 100% of ACEE/CPASR tasks have a construct card, stable ID, evidence class, declared metric support, dataset/source version, and claim boundary.
- 100% of external data artifacts have a content digest and licence record.
- no unregistered implementation or undeclared metric enters a report.

### Gate 1 — scorer integrity
- all metric bounds and directions are checked against source observations;
- all aggregate statistics are reproducible from trial-level records;
- positive, negative, metamorphic, and label-shuffle controls behave as preregistered;
- scorer mutation tests demonstrate that deliberate scoring defects are detected;
- no test split shares participants, raters, base scenes, or near-duplicate artifacts with training when those are the unit of generalization.

### Gate 2 — empirical validity
- compare with trivial and classical baselines;
- use participant/item/rater-clustered uncertainty;
- report calibration and subgroup performance;
- report disagreement and missingness instead of silently deleting them;
- use the appropriate reliability model and publish its design assumptions and confidence interval;
- register a human study before collecting any new participant data.

### Gate 3 — generalization and anti-Goodhart
- unseen participant and stimulus families;
- context changes and isomorphic task variants;
- independent human preference / outcome holdouts;
- proxy-optimization stress tests;
- no material degradation on safety, agency, or uncertainty axes while improving the headline metric.

### Gate 4 — external replication
- a second evaluator reproduces scores from a frozen release artifact;
- benchmark version, registry, data split, scorer, code commit, and run manifest are identical;
- any discrepancy is resolved before public SOTA comparison.

### Initial target ranges to investigate, not declare as pass/fail standards

- For probabilistic preference and confidence outputs, improve Brier/log score against the prior and strongest reproducible baseline with a 95% interval supporting the improvement.
- For ordinal human preference, show useful distributional calibration, not merely high correlation with the mean score.
- For participant-held-out attention tasks, outperform the class prior and classical baseline on more than one held-out subject partition; report per-subject spread.
- For test-retest reliability, preregister the estimand (consistency or absolute agreement), report the exact ICC form and confidence interval, and target ICC >= 0.75 only as a development goal where the construct and design justify it. Do not apply human reliability statistics to synthetic seeds.
- For safety-critical ascetic scenario reasoning, require zero severe unsafe recommendations in the frozen release set and report uncertainty/coverage; enlarge the challenge set before interpreting zero observed failures as a general guarantee.
- For any "better than SOTA" statement, require a pre-registered primary metric, matched protocol, one strong independent baseline, a statistically and practically meaningful improvement, no unacceptable Pareto regression, and replication.

## 8. Implementation sequence

### P0 — lock the research contracts
1. Add stable ACEE and CPASR construct cards to an extension registry; keep the existing v0.1 pilot registry immutable.
2. Add metric definitions for pairwise preference, ordinal distributions, pluralistic disagreement, attention/mind-wandering, voluntary choice, appropriate stopping, and safety errors.
3. Add evidence eligibility: synthetic mechanism, external task, human-normed, ecological interactive, theory probe.
4. Add anti-claim checks preventing aesthetic or ascetic task scores from being reported as clinical, spiritual, or consciousness measurements.

### P1 — build the external adapters
1. NEMAR `on006648` poetry rating adapter first: 47 participants × 210 items, five ordinal ratings; preserve participant-item rows, pin files and stimuli, and test participant/text/crossed holdouts.
2. MSC visual preference adapter with base-scene-grouped splits and raw rating distributions.
3. MADB music-aesthetics adapter with audio-only and annotation-conditioned tracks, source/artist/generator-held-out splits, rater distributions, and licence audit.
4. PAM∃LA personalized visual preference adapter with unseen-user and known-user/unseen-image tracks; verify image-generation output terms separately from annotation licence.
5. NEMAR/OpenNeuro `on001787` concentration/mind-wandering adapter with label alignment to interruptions and participant-held-out splits.
6. MUSIN-G optional music-enjoyment EEG adapter, strictly separated from music-aesthetic-quality labels.
7. Human Creativity Benchmark access/protocol audit; no dependency until access and licence are confirmed.
8. Dataset manifests, hash pinning, licence/attribution, and frozen test splits.

### P2 — connect existing Symthaea subsystems
1. Evaluate outputs of `symthaea-aesthetic` as a candidate predictor; keep its internal score out of the evaluator's ground truth.
2. Evaluate MeditationSentinel and relevant attention/metacognition subsystems against participant-held-out external labels.
3. Add subsystem ablations and matched classical baselines.
4. Preserve the distinction between the system's self-report, observed architecture state, and independent outcome labels.

### P3 — build modality and practice breadth
1. Add auditory, language, movement, spatial, material, and interactive aesthetic tasks one modality at a time.
2. Add practice-literacy scenarios across traditions, with source provenance and non-hierarchical scoring.
3. Add reversible low-risk constraint tasks and context-dependent stopping tests.
4. Build long-term retention/transfer protocols only after Gate 0–2 pass.

### P4 — qualify, publish, and compare
1. Generate registry, README, and paper tables from the same source of truth.
2. Run format, tests, clippy, dataset checks, schema validation, and scorer mutation suite; skipped workflows do not count as passes.
3. Publish a limitations-first technical report with evidence tier per result.
4. Conduct independent reproduction before any SOTA claim.

## 9. Immediate engineering tickets

| ID | Work item | Acceptance evidence |
|---|---|---|
| ACEE-001 | Add construct cards and extension registry | Schema-valid registry; duplicate IDs and missing sources rejected |
| ACEE-002 | Implement MSC adapter | Pinned dataset manifest; split-by-base-scene tests; raw ratings retained |
| ACEE-003 | Add pluralistic preference scorer | Pairwise and distributional scores; rater/item clustered uncertainty |
| ACEE-004 | Add aesthetics adversarial controls | identity, context, proxy-gaming, counterpoint and cohort tests |
| CPASR-001 | Add contemplative/ascetic construct cards | distinct practice, purpose, agency, context, risk and stop-rule fields |
| CPASR-002 | Implement NEMAR adapter | participant-held-out folds and reproducible signal preprocessing |
| CPASR-003 | Add agency-preserving constraint task | paired persist/stop counterfactuals; safety and agency metrics |
| CPASR-004 | Add cross-tradition source-grounded literacy task | balanced, cited item bank; uncertainty and abstention scorer |
| PSYCH-INT-001 | Cross-domain experiments | pre-registered hypotheses, independent outcome measures, ablations |
| PSYCH-REG-001 | Registry/report integration | no unregistered tasks; generated inventory; immutable run manifests |

## 10. Sources and external resources

- Penacchio, O. et al. (2026). *The Minimum Semantic Content (MSC) Dataset: A Large, Balanced Resource for Computational Aesthetics Research*. Scientific Data 13, 470. https://doi.org/10.1038/s41597-026-06816-0. Dataset project: https://osf.io/ZGSVJ/
- Hopkins, A. et al. (2026). *The Human Creativity Benchmark*. arXiv:2606.30561. https://arxiv.org/abs/2606.30561. Treat as a preprint and confirm data access before direct replication.
- Delorme, A. & Brandmeyer, T. (2026). *EEG meditation study*, NEMAR dataset on001787, DOI: https://doi.org/10.82901/nemar.on001787. Derived from OpenNeuro ds001787: https://doi.org/10.18112/openneuro.ds001787.v1.1.1.
- Chaudhuri, S., & Bhattacharya, J. (2026). *Poetry Assessment EEG Dataset 1* (Version v1.0.0) [Data set]. NEMAR. https://doi.org/10.82901/nemar.on006648 (47-participant primary EEG cohort; associated OpenNeuro ds006648 v1.0.0).
- *MADB: A Large-Scale Music Aesthetics Dataset with Professional and Multi-Dimensional Annotations* (2026). arXiv:2607.06929. https://arxiv.org/abs/2607.06929. Dataset: https://huggingface.co/datasets/sirui1/MADB-Dataset. Treat reported random-split baselines as within-distribution references; audit audio availability, source rights, and train/test leakage before replication. Dataset annotations are research-only under CC-BY-NC-4.0 per dataset card; source-audio rights may differ.
- Anonymous Authors. *PAM∃LA: Personalizing Text-to-Image Generation to Individual Taste* (under review, 2026). Dataset card: https://huggingface.co/datasets/pamela-dataset/pamela. Annotation data and metadata CC-BY-4.0; check Nano Banana / FLUX.2 output terms separately.
- MUSIN-G, NEMAR on003774 / OpenNeuro ds003774 (music listening EEG and familiarity/enjoyment ratings). https://doi.org/10.82901/nemar.on003774. Treat as affect/enjoyment data, not aesthetic-quality ground truth.
- Delorme, A., & Brandmeyer, T. (2026). *EEG meditation study* (Version v1.0.0) [Data set]. NEMAR. https://doi.org/10.82901/nemar.on001787 (24 participants; interruption-triggered concentration and mind-wandering self-ratings).
- *Psychological, psychiatric, and behavioral sciences measurement scales: best practice guidelines for their development and validation* (2024). https://pmc.ncbi.nlm.nih.gov/articles/PMC11798685/
- Bean, A. M. et al. (2025). *Measuring what Matters: Construct Validity in Large Language Model Benchmarks*. NeurIPS 2025 Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2025/hash/1967e0fc3aa6cbbace562f5cb8e3954e-Abstract-Datasets_and_Benchmarks_Track.html
- *A validity-guided workflow for robust large language model research in psychology* (2026). https://pubmed.ncbi.nlm.nih.gov/42387246/
- ICC formula selection and reporting guidance for test–retest designs (2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC6439259/

## 11. Explicit non-claims

Passing an ACEE task does not show that Symthaea experiences beauty. Passing CPASR tasks does not show that she meditates, has spiritual insight, or experiences subjective states. EEG pattern prediction is not a consciousness detector. Human-preference prediction is not proof of universally correct taste. Aesthetic and contemplative scores do not diagnose or treat people. Voluntary restraint is not inherently beneficial, and the benchmark must never reward coercion, harmful deprivation, or inability to stop.

The intended contribution is narrower and stronger: a reproducible, pluralistic, adversarially tested measurement system for aesthetic cognition, contemplative reasoning, and agency-preserving self-regulation, integrated with the broader evidence-qualified PsychBench capability profile.
