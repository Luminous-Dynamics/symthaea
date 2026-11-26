# ErgoStream Active Decision and Multimodal Reliability Addendum

**Date:** 2026-11-26  
**Status:** proposed benchmark protocol addendum; no new benchmark result or superiority claim  
**Programme:** Symthaea #13401 / #13402; Symtropy #7442 / #7497 / #7516 / #7518 / #7529 / #7555 / #7558 / #7556 / #7577  
**Purpose:** extend the frozen task-agnostic acquisition preregistration with evaluator-qualified decision-sufficient acquisition and a multi-axis reliability protocol for asynchronous multimodal alignment.

## 1. New evidence that changes the competitive framing

Two developments make a single pooled synchronization score scientifically inadequate.

### 1.1 AV synchronization metrics are measurement instruments

Sharma and Tapadiya's August 2026 preprint, *What Do Audio-Visual Synchronization Metrics Actually Measure?* (arXiv:2608.25157), audits AV-Align, ImageBind AV-relevance, JavisScore and Synchformer/DeSync using controlled-distortion monotonicity, preprocessing sensitivity, rank uncertainty, cross-metric agreement and a PEAVS human-aligned proxy.

The paper reports an axis split:
- Synchformer/DeSync is strongest on temporal-offset tracking (reported Kendall tau 0.84).
- ImageBind/JavisScore better match the PEAVS human-aligned proxy (reported tau 0.20) and content-disruption families.
- Cross-metric agreement is low (reported Krippendorff alpha 0.066).
- Linear and simple k-nearest-neighbour score fusion do not improve agreement with the human-aligned proxy over the best individual metric.

The work is described as an ECCV 2026 workshop poster and non-archival preprint. Treat the values as the paper's reported results, not as a definitive field-wide consensus. Its main methodological lesson is still important: a synchronization evaluator must itself be qualified as a measurement instrument. A new metric should not be considered better merely because it agrees with one other metric or produces a stronger leaderboard rank.

Reference: https://arxiv.org/abs/2608.25157

### 1.2 Generated-audio/video quality is a distinct task from feature alignment

Two 2026 benchmarks address generated audiovisual content rather than the feature-extraction task targeted by AV-SyncBench.

**SyncReward** (arXiv:2610.16338, 12 October 2026) reports SyncReward-Data-80K with 80,000 professional synchronization ratings on clips from eight audiovisual generators, plus a 1,000-clip held-out benchmark with multiple professional ratings per clip. Its taxonomy covers Speech, Onset, Instrument and Ambient sound families. The preprint reports that current specialized metrics and omni-modal models can diverge from human ratings. The project page says release of data, benchmark, model and code is planned; verify availability, license and exact artifact versions before treating it as an executable external baseline.

Reference: https://arxiv.org/abs/2610.16338  
Project: https://github.com/WeChatCV/SyncReward

**Beyond Time Shifts / SyncBench** (ECCV 2026) proposes a reference-free evaluator for generated audiovisual synchronization using Qwen2.5-Omni-3B, SynthSync and human preference data. The public project reports approximately 306,000 pairwise human annotations and, for its trained evaluator, NDCG 0.9435, Kendall tau 0.4899, MRR 0.7674 and pairwise accuracy 72.38%. These are native SyncBench results, not AV-SyncBench results and not clock-offset classification accuracy.

Reference: https://chenhaoqcdyq.github.io/BeyondTimeShifts/  
Repository: https://github.com/chenhaoqcdyq/BeyondTimeShifts

These datasets measure a different target from AV-SyncBench:
- AV-SyncBench: whether a feature model ranks the original audio/video pair above controlled temporal or semantic perturbations.
- SyncReward / SyncBench: whether an evaluator of generated clips aligns with human judgements of generated audiovisual quality.

They must remain distinct benchmark lanes.

## 2. Multimodal reliability card — required scorecard

Every ErgoStream audiovisual or multimodal result should report a reliability card, not a single synchronization number.

| Dimension | What it measures | Example metrics / controls |
|---|---|---|
| Clock registration | Recovery of device timebases | offset/rate error, held-out anchor residual, rank, singular spectrum, condition diagnostic |
| Event correspondence | Which observations refer to the same event | pairwise precision/recall, false association, set-valued correctness, false-unique rate |
| Temporal sensitivity | Response to controlled timing errors | monotonicity across offset/jitter/speed tiers, local-jitter response |
| Semantic integrity | Correct source/event association when timing is held fixed | source/timbre substitution detection, event-count and event-type consistency |
| Human alignment | Agreement with independent human assessment | rank correlation, pairwise preference accuracy, calibration and confidence intervals |
| Metric stability | Sensitivity to harmless pipeline changes | decoder/window/preprocessing perturbations, rank-flip rate, bootstrap ranking uncertainty |
| Missingness and conflict | Behavior under incomplete/disagreeing modalities | coverage, failure denominators, correct abstention, selective risk |
| Cost and latency | Practical acquisition/evaluation burden | media seconds, compute, memory, wall time, energy where measured, cost per qualified pair |

The card must display per-stratum denominators, confidence intervals, ties, missing outputs and failed samples. A scalar may be included for a precisely defined native task, but it cannot replace the card.

### Required metric-audit controls

Before a metric is used as a training reward, model-selection signal or leaderboard authority, test:

1. **Known-distortion monotonicity:** score must move in the expected direction under controlled offset, jitter, speed, event deletion/addition and semantic substitution.
2. **Benign-transform stability:** decoder, crop, frame-rate, sample-rate and windowing changes that preserve the relevant event should not arbitrarily reverse model rankings.
3. **Axis specificity:** a temporal-only change should primarily affect temporal metrics; a source-only change should primarily affect semantic/source metrics.
4. **Human agreement:** use independent multi-rater labels on a held-out sample; report agreement and uncertainty rather than assuming one reference metric is ground truth.
5. **Rank uncertainty:** bootstrap at video/scene/source level rather than treating correlated crops or perturbation pairs as independent.
6. **Shortcut/identity controls:** duplicate media, source memorization, background leakage and easy low-level cues must not reveal the positive/negative assignment.
7. **Selective validity:** ambiguous, missing-source or conflicting-modality cases should permit abstention; confidence must be calibrated on independent strata.

Metric fusion is a hypothesis, not an automatic improvement. It must demonstrate incremental human agreement and stable ranking under paired bootstrap evaluation.

## 3. External comparator hierarchy and fair comparisons

Keep each external benchmark on its native task and protocol.

| Lane | Primary comparison | Secondary comparator | ErgoStream extension |
|---|---|---|---|
| Real AV feature alignment | AV-SyncBench official protocol | Synchformer, SparseSync and compatible frozen models | active observation acquisition and reusable event-grammar transfer after native reproduction |
| Generated AV evaluation | SyncReward-Bench and SyncBench where artifacts are released and licensed | PEAVS and the metric-audit protocol | source/event explanation, uncertainty, event completeness and decision relevance |
| Clock registration | MATH-SYNC v1.1 | paired-anchor OLS / Theil-Sen / Huber | active anchor selection and corrupted-reference abstention |
| Structural transfer | exact finite graph matching and graph-transfer suites | FOR/ETC, graphon diagnostics, ReCoG-style relative contexts | explicit witness ambiguity and counterexamples on unseen families |
| Decision-focused learning | robust worst-case regret minimization | finite Bayes-risk / EVSI and exact finite-horizon policy oracles | minimize observation cost subject to evidence constraints and a predeclared regret ceiling |

### Important 2026 adjacent work

- ReCoG (arXiv:2611.01753, 1 November 2026) reports that removing a large proportion of source graph edges improved transfer for four graph pretraining methods, and reports gains of 7.61 percentage points in mean one-shot cross-domain node classification and 4.47 points in cross-type transfer to heterophilous graphs across 18 target datasets. Include full-graph, edge-thinned and relative-context arms wherever graph assumptions fit. This is a result to reproduce, not evidence that edge-thinning always helps every relation task. https://arxiv.org/abs/2611.01753
- Yamao et al., *Robust Decision-Focused Learning via Worst-Case Regret Minimization*, UAI 2026 / PMLR 337, models both measurement-error uncertainty around observed coefficients and distribution-shift uncertainty through a Wasserstein ambiguity set, then minimizes worst-case decision regret. It is a relevant competitor to prediction-only objectives: https://proceedings.mlr.press/v337/yamao26a.html
- MATH-SYNC v1.1, archived 2 October 2026, adds G12-A/B/C/D estimator, anchor geometry, reference sensitivity and convergence/scaling experiments. The authoritative Zenodo record states that Harvard and WESAD remain controlled schedule-replay experiments, not absolute physical clock measurements: https://zenodo.org/records/23106783

## 4. MATH-SYNC v1.1 reproduction checklist

The exact v1.1 archive is 6.9 MB and publishes SHA-256:

\`7f029bc9b7f892e65dcfa29fd4241eb362a8eb223356ab0037b21a7d852ea3a1\`

Download and verify the archive bytes before trusting the checksum or running any experiment. Preserve the frozen archive, release commit, environment, dependency versions, data manifests, participant exclusions, random seeds and score aggregation.

The current article reports the following supplementary results:

- Alternative estimators: OLS, Huber, LAD, Tukey and RANSAC. Under nominal/stress contamination, robust estimators substantially outperform OLS; Huber and LAD converged in all 150 trials, while Tukey had 33 failures and RANSAC required approximately 1.1–1.2 seconds per realization in that implementation.
- Anchor geometry: 2,700 trials; 63 rank-deficient cases, 60 with four anchors and three with six anchors. Increasing temporal span from 25% to 100% reduced median RMSE across the reported scenario/anchor combinations.
- Reference sensitivity: six possible references; a controlled smooth non-affine 20 ms reference perturbation worsened all 300 paired comparisons. This is a synthetic stressor, not an estimate of real hardware reference bias.
- Tolerance: at relative tolerance 1e-4, 50/50 trials converged after median one iteration but median RMSE rose to 1.431 ms; at 1e-6, 50/50 converged with median five iterations and RMSE 1.088 ms; at 1e-8, 49/50 converged with median 11 iterations and RMSE 1.087 ms.
- Scaling: 1,000 trials across 20 stream/anchor configurations; 936 converged and 64 reached the iteration limit. The largest 40-stream/128-anchor configuration had 4,622 observations and 206 unknowns, with approximately 462 ms median solve-only time on the study workstation.

These numbers define a reproduction target. They do not imply that ErgoStream has reproduced the experiments. The Rust v0.1 reference currently has a lower 8-stream/2,048-observation bound; extending it to the external scale envelope requires a separate qualification increment.

Paper: https://www.mdpi.com/2227-7390/14/20/3720  
Archive: https://zenodo.org/records/23106783  
Release: https://github.com/jrhechavarriah/MATH_SYNC_v1.0/releases/tag/v1.1

## 5. Decision-sufficient active acquisition — evaluator contract

The active-acquisition benchmark must keep three objects separate:

1. **Candidate hypothesis set:** candidate-local explanations and their predicted decision-equivalence classes.
2. **Evaluator-admissible witness set:** the complete set of structural witnesses compatible with the frozen public evidence prefix, established by independent owner-side validation.
3. **Ground-truth probe outcome support:** the possible outcome set for each declared probe under each evaluator-admissible witness, held outside the candidate process.

A candidate is not allowed to establish its own truth set. The evaluator must use an independent structural validator to bind candidate-local hypothesis IDs to canonical evaluator witness identities.

### Non-compensable evaluator checks

- **False exclusion:** admissible witness omitted by the candidate. A candidate cannot obtain DecisionSufficient by shrinking its own hypothesis set.
- **False inclusion:** candidate hypothesis does not map to an admissible witness.
- **False decision equivalence:** candidate groups witnesses that imply materially different downstream decisions.
- **False decision split:** candidate separates witnesses that are decision-equivalent under the frozen domain utility/tolerance contract.
- **False probe utility:** candidate predicts a separating outcome partition but evaluator truth shows a shared outcome remains compatible with decisions that disagree.
- **False outcome support:** candidate omits a possible outcome or predicts a spurious outcome.
- **Unnecessary abstention:** candidate returns Unresolved despite an affordable, legal, truth-discriminating probe.
- **Decision-sufficient validity:** complete admissible witness coverage and evaluator-confirmed decision equivalence are both required.

The evaluator-only reference implementation is in Symtropy draft PR #7577, stacked on candidate policy PR #7558 and tracked by #7555. It reports each failure count separately; it does not authorize probe execution, certify physical sensors or replace blind episode custody.

### Multi-step policy qualification

The first #7558 policy is a one-step minimax decision-partition baseline. It must not be presented as an optimal sequential researcher.

The next policy comparison should include:
- no acquisition/passive schedule;
- cheapest-first and uniform coverage;
- random legal probe;
- maximum-hypothesis-split;
- one-step minimax decision-disagreement reduction;
- calibrated entropy and EVSI only where the likelihood model is justified;
- two-step and finite-horizon lookahead;
- evaluator-only exact finite-horizon policy on tiny worlds.

For every method, compare cost to a predeclared terminal decision-regret ceiling, not merely the number of hypotheses removed. Report regret, decision error, false-sufficiency, false-exclusion, unnecessary abstention, accepted and failed probe attempts, observation latency, energy and compute. Preserve all failures in the denominator.

A robust extension should minimize worst-case downstream regret over evaluator-admissible hypotheses and a declared outcome uncertainty set. Probabilistic/expected EVSI is a separate arm and requires a frozen, calibrated outcome model; entropy is not a substitute for decision utility.

## 6. Preregistered target metrics and claim thresholds

These are prospective targets, not achieved results:

| Objective | Target / qualification rule |
|---|---|
| Exhaustive finite witness correctness | Zero false-unique, false-exclusion and false-sufficiency on the complete tiny-world oracle suite |
| Probe correctness | Zero false probe-utility claims on exhaustive finite fixtures |
| Active acquisition efficiency | At least 20% lower acquisition cost than the strongest matched passive baseline at equal or lower held-out decision regret, with no false-sufficiency regression |
| False-sufficiency rate | Target below 1%; zero observed false-sufficiency among 300 independent confirmatory negative cases only supports an approximate 95% upper bound near 1% under an IID/binomial interpretation; cluster dependence requires a cluster-aware bound and usually more cases |
| MATH-SYNC reproducibility | Reproduce G12 outputs, failures and participant-level results under the exact frozen v1.1 code/data/environment protocol before any comparative claim |
| AV-SyncBench reproduction | Reproduce the official score path, reference models, preprocessing variants, ties and every stratum before adding candidate comparisons |
| Human-aligned generated AV | Run only after dataset/model artifacts and licenses are available; evaluate on sealed clips and preserve independent human raters |
| Transfer contribution | Explicit witnesses must show incremental predictive or decision value beyond the strongest applicable M1/M2 diagnostic on sealed generator families |

No aggregate score can compensate for false causal transfer, false-sufficiency, false-unique or unsupported physical claims.

## 7. Integration and status

- Symtropy #7556: independent Jacobi-SVD rank audit and MATH-SYNC v1.1 reproduction delta. Exact-head Rust 1.96 workflow passed for the currently recorded PR head; this is bounded rank-diagnostic qualification, not full MATH-SYNC reproduction.
- Symtropy #7558: one-step decision-sufficient minimax probe selector. Exact-head Rust 1.96 formatting, tests and strict Clippy passed on its recorded head; this is a candidate policy reference, not empirical acquisition superiority.
- Symtropy #7577: evaluator-only decision-sufficiency and probe-claim scorer. Exact-head workflow #45856103031 passed Rust 1.96 formatting, all 7 evaluator tests, and strict Clippy at head `91fe21319657bf833fc071003ca8de249cd10ab7`. This is a finite reference evaluator, not process isolation or blind episode integration.
- Symtropy #7555: active disambiguation research contract.
- Symtropy #7516: AV-SyncBench adapter/reproduction owner.
- Symtropy #7518 / #7529: MATH-SYNC reproduction and active calibration ownership.
- Symthaea #11171: production research-policy integration owner.
- Existing blind episode, source custody, and evidence receipt owners remain authoritative. Do not create another episode runner, sensor/timebase service, evidence ledger, truth store or production scheduler.

**Current limitation:** the evaluator reference is a separate package in the same public repository and has not yet been integrated into the blind owner-side execution service. Package separation is not process or OS isolation. No hidden answer keys should be placed in a candidate-accessible process or artifact.

No claim of SOTA, physical synchronization, human-level audiovisual judgment, causal discovery, general intelligence or production safety follows from this addendum.


## 2. Streaming perception and sequential access comparators

The existing Symtropy #6675 issue already owns the policy-level benchmark for choosing what to acquire, when to wait, when to repeat, when to stop, and when to respond under deadlines and costs. ErgoStream should plug its epistemic decision-sufficiency layer into that tournament rather than build a second active-media scheduler.

The following external benchmarks should be retained as distinct comparison lanes:

| Comparator | Native task and reported evidence | What ErgoStream should learn from it | What it does not establish |
|---|---|---|---|
| Video Active Perception (VAP), arXiv:2605.01662 | Training-free frame/keyframe acquisition for long-form video QA; reports up to 5.6× frame efficiency over standard sampling on EgoSchema, NExT-QA, ActivityNet-QA, IntentQA and CLEVRER | Matched frame-budget and cost-to-answer baselines; acquire moments that differ from expectations rather than uniformly sampling | It is task-question-conditioned frame selection, not task-agnostic structural acquisition or decision-sufficiency qualification |
| SEW-Bench / Access-Structured Perception, arXiv:2608.22975 | On a 4,096-token episodic retrieval budget, reports 75–94% retrieval accuracy for query-conditioned access vs 3–19% for equal-budget query-independent sampling | Strong baseline for access policy and resource allocation under a fixed budget | The authors report that ASP did not beat a verbatim-only baseline on any backbone, and two of four preregistered falsification criteria fired; this is a necessary warning against assuming compressed state is automatically useful. Natural-video benchmarks were not run due to dataset agreements |
| TRACE, arXiv:2609.30670 | 1,240 records across 517 videos; evaluates eight models/systems in eight configurations; measures temporal validity, response triggers, completion, workload and reliability | Require causal evidence-availability timestamps, response delay, false alarms, missed windows, completion and workload alongside task accuracy | Streaming video QA/response evaluation is not itself task-agnostic acquisition or event-identity transfer |
| SPOT-Bench, NeurIPS 2026 E&D | Fully proactive streaming-video benchmark; every prediction across the whole stream is evaluated using Timeliness-F1. Its live detection table lists Ctrl-SPOT baseline T-F1@5 of 20.9 at 1Hz | Include no-response dead time, prediction spam, missed events, timely response and silence quality; evaluate the entire stream, not only sampled question points | The current repository states that interaction/intervention held-out test splits are planned for a later challenge release; some leaderboard tasks are detection-only at present |
| DIVE, arXiv:2610.28405 | 525 audio/video samples with human annotations; reports sub-second average response latency for leading systems, while average response-window rates remain at or below 53.77% | Separate latency from response-window coverage and participation quality; include multi-party, ongoing and proactive interaction strata | Full-duplex interaction quality is not a clock-registration score or a measure of structural correspondence |

### Consequence for ErgoStream

A useful benchmark should make the full observation/decision timeline explicit:

1. Evidence exists in the world.
2. Evidence becomes available to the candidate after acquisition/transport delay.
3. The candidate may acquire, wait, repeat, stop, answer or abstain.
4. The decision deadline may expire before a high-fidelity observation arrives.
5. The evaluator scores the decision against all admissible worlds and the actual evidence available at the time.

This makes it possible to distinguish:
- **frame efficiency** from **decision efficiency**;
- **low latency** from **timely, correct participation**;
- **information gain** from **decision value**;
- **structural uniqueness** from **decision sufficiency**;
- **memory compression** from **useful, independently validated reuse**.

The policy tournament in Symtropy #6675 is the owner for sequential action scheduling, deadline and resource accounting, EIG/EVSI comparison and exact finite-horizon policy-tree oracle. ErgoStream #7555/#7558 supply the hypothesis-set and one-step decision-partition layer; #7577 supplies evaluator-side qualification of candidate claims. No duplicate scheduler or memory engine should be introduced.

### Matched evaluation requirements

For every method, report identical:
- source stream and evidence availability trace;
- legal action set and source-dependency/common-cause groups;
- acquisition, processing and waiting costs;
- response deadline and stale-evidence policy;
- candidate inference and memory budgets;
- held-out outcome and hidden evaluator probe sets.

Report separately:
- answer/task correctness;
- time-to-valid-decision and p50/p95/p99 response delay;
- response-window coverage and false-response rate;
- missed-target rate and no-response dead-time;
- acquired frames/audio duration/bytes and modality mix;
- candidate inference cost, memory and energy where measured;
- terminal decision regret and false-sufficiency / false-unique rates.

Do not compare VAP frame-efficiency, SEW retrieval accuracy, TRACE QA accuracy, SPOT Timeliness-F1, DIVE response-window rate, AV-SyncBench pairwise accuracy or MATH-SYNC clock RMSE as though they were measurements of the same variable. They are different tasks and different estimands.


## 8. Fresh streaming frontier audit: online interaction and spatial persistence

A further November 2026 review adds two important external comparators that sharpen the next engineering priorities. Their task contracts differ; the numbers below must remain on their native scorecards.

### 8.1 OmniInteract: online response policy is a first-class capability

**OmniInteract** (EMNLP 2026, https://aclanthology.org/2026.emnlp-main.490/) evaluates native online inference over unbroken audio-visual streams, without future content. It contains 250 videos and 1,430 temporally grounded response slots: 1,062 1Q1A slots spanning real-time, proactive and nested interaction, plus 368 1QnA slots for continuous monitoring and step guidance. Each slot specifies a trigger, response window and target answer.

The paper evaluates correctness, response timing, invalid outputs, interruption handling and context continuity. Its reported best overall Interaction-Aware Quality-Timeliness F1 (IA-QTF1) is 0.368; best 1QnA IA-QTF1 is only 0.052. This is a strong indication that continuous multi-step engagement is a harder capability than single responses. These are paper-reported benchmark results, not an ErgoStream result.

**Required ErgoStream treatment**
- Run both 1Q1A and 1QnA tracks; do not let good single-turn performance hide weak continuous task completion.
- Preserve invalid outputs, interruptions, premature answers, late answers, missed triggers and context discontinuities in denominators.
- Score response quality, timing, interruption behavior and chain completion separately before reporting the native IA-QTF1.
- Keep the candidate online and causal: no future frames, answer slots, hidden trigger labels or target responses.
- Compose intervention decisions through Symthaea #5169 and the existing AV-STREAM #12837 path; do not implement a parallel turn controller.

### 8.2 OVO-S-Bench: persistent spatial understanding is not solved by retrieval

**OVO-S-Bench** (EMNLP 2026 Oral, official implementation: https://github.com/InternLM/OVO-S-Bench) provides a hierarchical streaming spatial-intelligence benchmark. Its released version contains 1,680 questions over 348 source videos from nine datasets, organized into 30 task types and four cumulative levels. The mean observed prefix at query time is 8.8 minutes; median evidence spans are 2.0 seconds (L1), 36.8 seconds (L2), 2.0 seconds (L3), and 278.7 seconds (L4).

The four levels separate:
- **L1 — instantaneous egocentric perception:** current-view distances, scale, clearance and local spatial relations.
- **L2 — spatiotemporal context:** recall of spatial evidence that appeared earlier but is no longer visible.
- **L3 — spatial simulation and reasoning:** reorientation, removal consequences, feasibility and route planning.
- **L4 — allocentric mapping:** global direction, topology and trajectory-map alignment across viewpoints.

The published comparison reports 38 evaluated systems. Gemini-3.1-Pro reaches 59.2 overall, Qwen3-VL-235B-A22B 53.6, human streaming 86.6 and human offline 92.2. L4 is the weakest level for 28 of 34 systems in the reported analysis. The strongest open model trails streaming human performance by roughly 33 points. These scores use OVO-S-Bench's own streaming protocol and are not directly comparable to OmniInteract, SPOT-Bench or DIVE.

**The critical negative results matter as much as the leaderboard:**
- Across HERMES, StreamingTOM and FluxMem, evidence recall has near-zero per-query correlation with correctness (reported Pearson r from -0.07 to 0.00).
- An oracle evidence sampler and doubling the frame budget each improve accuracy by no more than 0.3 points over the reported uniform 128-frame baseline.
- 13 of 15 evaluated specialized streaming/spatial variants trail their comparable general backbone on overall accuracy; L4 is especially vulnerable to regressions.

This directly falsifies a simplistic "more memory + more frames = better understanding" strategy. Memory retrieval and frame selection remain useful components, but they are not substitutes for spatial state estimation, map construction, geometric reasoning, temporal consistency and grounded evidence use.

The official protocol truncates each video at the annotated query timestamp. Most systems receive 128 uniformly sampled frames from that prefix; systems with native sequential ingestion are evaluated at their published streaming rate. Any reproduction must pin which protocol path each arm uses, document the frame/compute budget, and avoid treating a sampled-prefix model as a native continuous-stream model.

### 8.3 Revised architecture hypothesis: evidence-indexed spatial state

The next architecture experiment should connect existing owners rather than introduce another memory or controller subsystem:

```
causally arriving modality-native observations
  -> source/time-qualified event and entity hypotheses
  -> uncertain local egocentric state
  -> persistent allocentric/topological spatial state
  -> evidence-indexed event memory (references, not replacement truth)
  -> decision-sufficient probe / acquisition proposal
  -> existing sequential acquisition and turn-control owners
  -> answer, intervene, wait, revise, or abstain
```

Test the following hypotheses independently:
1. Persistent allocentric/topological state improves L4 mapping and cross-viewpoint consistency beyond flat frame retrieval.
2. Evidence-indexed memory improves long-horizon query support only when each recalled claim resolves to the original source interval and survives contradictory/revision evidence.
3. Active acquisition improves decision quality only when it reduces evaluator-confirmed decision regret or response misses at a matched resource budget; evidence-recall or entropy gains alone are insufficient.
4. HDC retrieval, CfC/LTC continuous state, explicit metric/topological maps and dense visual features each earn their place through matched ablations. No substrate receives an assumed advantage.
5. An interaction controller that models both response windows and no-response dead time improves timeliness without increasing unsolicited or incorrect interventions.

Implementation ownership stays with:
- Symthaea #3934 / #8152 for visual/world-state and representation comparisons;
- Symthaea #10644 / #9075 / #9177 for temporal retrieval, valid-time/known-at semantics and evidence context;
- Symthaea #5169 for floor, stop, yield and interruption semantics;
- Symthaea #12837 for continuous audiovisual benchmark integration;
- Symtropy #6675 for sequential acquire/wait/repeat/stop/answer/abstain policy tournaments and exact finite-horizon oracles;
- Symtropy #7555 / #7558 / #7577 for decision-sufficient hypotheses, candidate selection and evaluator qualification.

### 8.4 New benchmark lanes and prospective targets

The following targets are proposed research gates, not achieved results or claims about current Symthaea capability. They are meaningful only after the exact native benchmark protocol, model revisions, input budgets, scoring code and data permissions are pinned.

| Benchmark lane | Current external reference | Prospective target | Qualification condition |
|---|---:|---:|---|
| OVO-S overall spatial streaming | Gemini-3.1-Pro 59.2; Qwen3-VL-235B-A22B 53.6 | ≥62.0 overall | Same prefix-only information and documented frame/streaming budget; no hidden future frames |
| OVO-S allocentric L4 | Gemini-3.1-Pro 54.9 | ≥55.0 L4 | No material L1/L2 regression; include map/trajectory counterfactuals and evidence-span correctness |
| OmniInteract overall IA-QTF1 | 0.368 | ≥0.45 | Official 1Q1A/1QnA denominators, invalid-output and interruption accounting |
| OmniInteract continuous 1QnA IA-QTF1 | 0.052 | ≥0.12 | Full chain completion, context continuity and response-window scoring |
| SPOT-Bench T-F1@5 at 1 Hz | Ctrl-SPOT 20.9 | ≥25.0 | Full-stream predictions, dead-time, false response and missed event accounting |
| DIVE response-window rate | Best reported average ≤53.77% | ≥70% | Keep response quality, participation and timing distinct; report false intervention and no-trigger controls |
| Resource frontier | Benchmark-specific; not pooled | ≥20% lower measured acquisition/processing cost at a preregistered non-inferiority margin | Matched model quality, hardware, context, frame/audio budget and end-to-end accounting |

A target crossing one leaderboard score is not enough. Require paired source/video-level uncertainty, repeated seeds where applicable, no leakage, a frozen evaluation plan, and no increase in unsupported claims, false event persistence, false interventions, false-sufficiency or unsafe abstention behavior. Where the benchmark cannot be run under the same model/input protocol, report it as a non-comparable external reference rather than a ranking.

### 8.5 Immediate execution order

1. Keep the small synthetic evaluator and exact-head policy tests as the fast local gate; no remote CI dependency is required to continue design work.
2. Build an OVO-S adapter/receipt contract first (model revision, prefix cutoff, 128-frame or native-streaming path, query ID, answer, score, failure, latency, resource record). Run a small smoke subset only after checking dataset access and source-video terms.
3. Add an allocentric L4 diagnostic breakdown and explicit evidence-span provenance checks; do not modify the upstream scoring function.
4. Add OmniInteract response-window, interruption and 1QnA chain receipts to the AV-STREAM evaluation contract; use its official metrics unchanged.
5. Extend the existing #6675 tournament with candidate policy traces from #7558 and evaluator qualification from #7577 only at the adapter boundary. Do not copy its scheduler, oracle or turn controller into another package.
6. Before choosing a Symthaea model architecture, run the matched baseline ladder: best-practical modular pipeline, pinned end-to-end multimodal model, native continuous path, hybrid path, and active hybrid. Include memory-disabled, HDC-disabled, no-map, no-active-sensing and no-response-controller ablations.

### Sources

- OmniInteract (EMNLP 2026): https://aclanthology.org/2026.emnlp-main.490/
- OVO-S-Bench official implementation, benchmark protocol and leaderboard: https://github.com/InternLM/OVO-S-Bench
- SPOT-Bench official project and leaderboard: https://dibschat.github.io/SPOT-Bench/
- DIVE: https://arxiv.org/abs/2610.28405
