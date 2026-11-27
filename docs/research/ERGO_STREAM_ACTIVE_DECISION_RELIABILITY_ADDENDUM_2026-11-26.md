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
| OVO-S allocentric L4 | Gemini-3.1-Pro 54.9 | ≥58.0 L4 | No material L1/L2 regression; include map/trajectory counterfactuals and evidence-span correctness |
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


## 9. Cross-programme synthesis: decision-preserving memory for streaming spatial state

A repository ownership audit found that the next memory-related increment already has a natural owner: **MACE #13284** formalizes decision-sufficient memory under resource/evidence constraints, while **MACE #13261** owns reproduction of strong retrieval and state-first baselines. ErgoStream should supply the streaming/spatial task distribution to those evaluators rather than create a second memory engine.

### 9.1 The paradigm shift

The optimization target is not maximum recall, maximum retained frames, or minimum embedding distortion. It is the **smallest admissible state that preserves the distinctions needed for a declared downstream decision**.

For a query/DecisionNeed (q), evidence prefix (E_{\leq t}), retained memory state (S), frozen policy \(\pi\), evaluator truth \(y\), and task utility \(U_y\):

- Absolute regret: \(R_y(S)=\max_a U_y(a)-U_y(\pi(q,S))\).
- Compression regret gap: \(G_y(S)=R_y(S)-R_y(E_{\leq t})\).

The full-prefix reference is an information baseline, not a claim that the prefix is complete or physically true. Preserve absolute regret against evaluator optimum as well as compression regret gap. A compressed state qualifies only when its preregistered regret margin passes **and** every hard evidence constraint passes.

Hard constraints are not soft penalties:
- each claim must resolve to original observation IDs, source intervals and content/revision digests;
- measurement time and known-at time remain distinct;
- scope, permission, source rights, supersession and revocation are checked at use;
- common-cause evidence is not counted as independent corroboration;
- unresolved identity, geometry, clock alignment or source disagreement remains explicit;
- derived summaries are retrieval/decision aids, never replacement observations.

This makes memory quality a cost–regret frontier under an evidence contract, not a similarity-retrieval score.

### 9.2 Streaming-spatial adaptation

Run the frontier over OVO-S-Bench's prefix-only queries, preserving its native protocol and level partitions. The official project leaderboard distinguishes human streaming (86.6 overall) from human offline (92.2); these are different access conditions and must not be conflated. Gemini-3.1-Pro's reported 59.2 is the strongest listed model overall, with L4 allocentric mapping at 54.9. The key comparison is not simply whether the memory retrieves the annotated evidence interval, but whether the retained state supports the correct spatial decision and map consistency at the query prefix.

Add paired counterfactual strata:
1. Same scene statistics, different object identity or route topology.
2. Identical visible endpoint, different hidden path/history.
3. Occlusion and reappearance with a plausible identity swap.
4. Camera ego-motion versus independently moving objects.
5. Late evidence that corrects an earlier position, event order or entity association.
6. Clock drift, reordered chunks and missing observations.
7. Misleading but semantically similar retrieved episodes.
8. Ambiguous or unobservable cases where the correct output is a set, a query, or abstention.

For every query, freeze the candidate-visible prefix and memory snapshot before opening evaluator truth. Hidden scenario state, future frames, answer text, target map and response-window labels remain evaluator-only.

### 9.3 Matched memory arms

Use the baseline ownership and source pins from #13261. Keep the following arms distinct and matched on backbone, input prefix, total context and all-in resource budget:

- Full-prefix/full-context reference and no-memory baseline.
- Verbatim-only event/frame retention (a strong control, not a strawman).
- Uniform/fixed-rate sampling and query-independent top-k.
- OASIS-style hierarchical event memory with intent-conditioned retrieval.
- BM25, dense, hybrid RRF, entity/date/literal and neighboring-context retrieval.
- State-first/supersession memory and causal-graph retrieval.
- Typed-content HDC and event-ID HDC as separate arms.
- Explicit spatial/topological graph plus dense visual features.
- HDC-disabled, map-disabled, hierarchy-disabled and compression-disabled ablations.

The SEW-Bench/ASP paper's negative result is a required falsification control: its access-structured method improves the fixed-budget comparison over query-independent sampling, but does not beat verbatim-only on any evaluated backbone, and its compressed state was not independently useful in that setting. Conversely, OASIS reports gains from on-demand hierarchical event retrieval, and HERMES/StreamScout provide additional memory and active-access comparators. Reproduce each on its own task protocol before adapting it; do not transplant leaderboard numbers between tasks.

The strongest current comparators make an important distinction: **adaptive access is already a competitive technique; it is not, by itself, the new contribution.** Our hypothesis must be stronger: preserve decision-relevant state, verify the evidence that warrants it, and adapt acquisition to residual decision regret.

Reported native-protocol reference points:
- **OASIS (CVPR 2026):** on Qwen3-VL-8B / StreamingBench, Real-Time All rises from 72.83 to 78.22; ACU 35.63 to 42.74; MCU 35.74 to 49.60; SQA 43.90 to 48.40. The study uses the StreamingBench All Context setting (five questions per video), Qwen3-VL-8B at 0.5 fps and one A800 80GB. Reported StreamingBench tokens/query fall from 29,517.6 to 9,544.2 (~67.7%); however, on OVO-Bench, end-to-end query latency rises from 4.14 s to 6.52 s even though request-processing delay is 0.19 s and retrieval itself is 0.11 s. This makes OASIS a strong accuracy/token-efficiency baseline, not automatic evidence of lower response latency. It is not a spatial-map correctness result. https://openaccess.thecvf.com/content/CVPR2026/papers/Liang_OASIS_On-Demand_Hierarchical_Event_Memory_for_Streaming_Video_Reasoning_CVPR_2026_paper.pdf
- **HERMES (ACL 2026):** hierarchical KV-cache reuse reports up to 10× faster time-to-first-token, up to 68% fewer video tokens than uniform sampling, and up to 11.4% gains on streaming datasets. Its main comparator is cache/memory efficiency and real-time response, not evaluator-held spatial witness completeness. https://aclanthology.org/2026.acl-long.381/
- **StreamScout (arXiv:2609.00291):** uses a staged glance → look-back → query-salient retrieval cascade with a learned stop/escalate boundary. Its abstract reports a 14.65-point improvement for Qwen3-VL-8B on OVO-Bench with 59% fewer tokens than uniform sampling and 1.04 s average answer time. OVO-Bench is not OVO-S-Bench; do not transfer the score. https://arxiv.org/abs/2609.00291
- **SEW-Bench / ASP:** query-conditioned access is a strong control, but the paper reports that ASP does not beat its verbatim-only baseline on any evaluated backbone and triggers two preregistered falsification criteria. This prevents us from assuming that an extra compressed state is always worth maintaining. https://arxiv.org/abs/2608.22975

The ErgoStream differentiation should therefore be tested against all three families: hierarchical cache compression (HERMES), on-demand event retrieval (OASIS), and learned stop/escalate access (StreamScout). Add a fourth arm: **decision-sufficient access**, in which the next memory expansion or sensor query is selected only when it can lower evaluator-confirmed downstream regret enough to justify its full cost, while the returned claim carries exact source evidence and remains revisable.

### 9.4 Scorecard and proposed gates

Report a frontier, not a single composite score:

| Dimension | Required result |
|---|---|
| Task outcome | Native OVO-S L1–L4 accuracy and query-level decision/action correctness |
| Regret | Absolute regret, compression regret gap, tail/CVaR and worst-family result |
| Evidence fidelity | Exact source-span coverage, provenance completeness, stale/retracted/unauthorized-use rate |
| Spatial consistency | Entity identity continuity, map/topology error, route/trajectory consistency and uncertainty coverage |
| Epistemic behavior | False-unique rate, unsupported-claim rate, set-valued correctness and justified abstention |
| Revision | Correction propagation, supersession latency and contradiction recovery |
| Resource | Stored bytes plus sidecars, tokens/context, indexing/update/replay work, calls, peak memory, wall-time p50/p95/p99 and measured energy where available |

Prospective research gates (not achieved results):
- compression-induced task-success loss no greater than 1 absolute percentage point at the selected operating point;
- worst-family regression no greater than 2 points;
- at least 25% lower **all-in** memory/resource cost at matched task quality, including provenance sidecars, updates, replay and verification;
- zero use of known retracted, revoked, wrong-scope or unauthorized evidence in deterministic adversarial fixtures;
- no regression in false-unique, unsupported-spatial-claim or false-persistence rates.

These targets align with MACE #13284's proposed qualification gates. They must be calibrated on development data and frozen before confirmatory evaluation. Use paired source/video-level clusters, not independent crops, for uncertainty estimates. Preserve every failed/invalid attempt and per-stratum denominator.


### Stretch tier after first native reproductions

These are deliberately harder second-stage targets. They are only meaningful once the relevant external baselines have been reproduced on the same dataset version, split, prefix, backbone or model-class budget, scorer and runtime contract:

| Lane | Stretch target | Comparison condition |
|---|---:|---|
| OVO-S overall | ≥65.0 | Exceed Gemini-3.1-Pro 59.2 and the strongest reproduced open baseline; paired cluster interval must support the improvement |
| OVO-S L4 | ≥62.0 | Exceed the reported 54.9 Gemini L4 result without a material L1–L3 regression |
| OmniInteract overall | ≥0.50 IA-QTF1 | Official slot mix, full invalid-output and interruption accounting |
| OmniInteract 1QnA | ≥0.15 IA-QTF1 | Continuous chain completion and context continuity |
| SPOT-Bench | ≥28 T-F1@5 at 1 Hz | Full-stream scoring, including no-response periods |
| DIVE | ≥75% response-window rate | False interventions and no-trigger controls remain within preregistered bounds |
| Efficiency | ≥30% lower all-in measured cost | No more than 1 absolute task-score point loss against the strongest matched adaptive-access baseline |

Do not target a number without its protocol. A score that exceeds a weak baseline, uses a larger model or frame budget, or silently excludes invalid attempts is not a SOTA result. A credible win is a paired improvement over the strongest compatible baseline with uncertainty bounds, complete cost accounting and unchanged integrity/safety gates.

### 9.5 Owner composition and first implementation slice

- **MACE #13284:** cost–decision-regret frontier, epsilon-sufficiency and active evidence acquisition.
- **MACE #13261:** strong raw-trajectory and industry memory baseline reproduction.
- **AV-STREAM #12837:** prefix-only OVO-S/OmniInteract adapters and multimodal receipts.
- **Symtropy #6675:** sequential acquire/wait/repeat/stop/answer/abstain policy tournament and finite-horizon oracle.
- **Symtropy #7577:** evaluator-side qualification of decision-sufficiency and probe claims; not the memory engine or episode custodian.
- **Vision Mind #3934, temporal retrieval #10644, bitemporal lifecycle #9075, proof-aware context #9177:** existing world-state, source-time and evidence-packet owners.

First implementation slice:
1. Freeze an OVO-S query receipt schema: dataset/query ID, prefix cutoff, model/backbone/config digest, memory snapshot digest, answer/action, source evidence IDs/spans, unsupported/abstained state, native score, latency and resource receipt.
2. Add a small deterministic paired-world pilot that tests map/identity preservation and correction-at-use without public raw media.
3. Run the strongest available verbatim-only, OASIS-like/hierarchical, dense/hybrid and state-first baselines under matched caps.
4. Publish the cost–regret curve and failure strata before connecting the candidate to #6675's sequential acquisition policy.

Do not create a parallel production memory store, evidence ledger, spatial map owner, sensor scheduler or conversational turn controller. The intended contribution is a shared, independently evaluated **decision-preserving state contract** across streaming perception and long-horizon memory.


## 10. Spatial audio fields: DCASE 2026 Task 3 as a complementary anchor

OVO-S-Bench evaluates spatial reasoning from streaming visual prefixes; DCASE 2026 Task 3 evaluates reconstruction of a spatial acoustic scene. They should remain separate native benchmark lanes, but together motivate a stronger cross-modal research direction: **maintain a coherent allocentric scene state from visual geometry and spatial acoustic fields without treating either modality's model output as ground truth**.

### 10.1 Native DCASE task contract

DCASE 2026 Task 3, *Semantic Acoustic Imaging for Sound Event Localization and Detection from Spatial Audio and Audiovisual Scenes*, is completed and publishes its baseline, development data, evaluation data and results.

- Development provides 32-channel Eigenmike recordings and high-definition acoustic maps.
- Evaluation provides only four-channel spatial audio (tetrahedral and/or first-order ambisonics), with video available for the audiovisual track.
- Output is a dynamic semantic acoustic image: per-frame class masks/energy points on a 360×180 equirectangular field at 10 FPS.
- Evaluation includes mask mAP over COCO IoU thresholds, Pearson correlation and relative distance error; source matching uses a 20° great-circle threshold and Hungarian assignment.
- The challenge reports 29 submissions from eight teams for Task 3: 14 audio-only and 15 audiovisual.

Official references:
- Task description, data, protocol and results: https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes
- DCASE 2026 challenge index/results: https://dcase.community/challenge2026/

### 10.2 Published results and interpretation

The published best-per-team table reports the following leading entries:

| Track | Submission | Macro Mask mAP | Macro Pearson r |
|---|---|---:|---:|
| Audio-only | Kong_CUHK_task3a_2 | 0.1080 | 0.3962 |
| Audio-only | Jeong_Medisensing_task3a_4 | 0.0662 | 0.2765 |
| Audiovisual | Yang_WHUiasp_task3b_2 | 0.0131 | 0.4002 |
| Audiovisual | Kim_SAMSUNG_task3b_4 | 0.0218 | 0.2420 |

**Important scoring nuance:** the audiovisual rank-1 entry does not have the highest Mask mAP in the displayed best-per-team table. The challenge ranks by cumulative rank across its metrics, with Mask mAP used as a tie-breaker. Therefore, do not call the rank-1 audiovisual entry the best mAP system or collapse the metrics into one interpreted capability. Report every native metric and ranking rule separately.

The unexpectedly low audiovisual Mask mAP relative to the audio-only leaders is not evidence that video hurts in general: the tracks, submissions and methods differ. It is a reason to reproduce the exact official pipeline and matched audio-only versus audiovisual ablations before inferring modality benefit.

### 10.3 Symthaea research hypothesis

DCASE Task 3 adds a field-valued spatial audio representation to the existing OVO-S global visual map question. The experiment should test:

1. **Acoustic-field continuity:** source extent and energy maps remain temporally consistent across motion, overlap and partial occlusion.
2. **Cross-modal source association:** sound-field regions associate with visual entities only when timing, geometry and source evidence support the join.
3. **Allocentric consistency:** acoustic bearings and visual landmarks map into a common world frame only when camera pose, array pose and cross-clock transform are qualified.
4. **Uncertainty preservation:** diffuse, overlapping, low-SNR or geometrically ambiguous sources produce broad/set-valued support or abstention rather than fabricated point estimates.
5. **Decision value:** an acoustic map is useful only when it improves a downstream task—e.g. finding an off-screen source, choosing a safe direction, distinguishing two visually similar events, or deciding whether another view is worth acquiring.

Keep the official DCASE output and scoring unchanged. Add separate ErgoStream diagnostics for source-event lineage, cross-modal association, clock/pose uncertainty, map persistence and downstream decision regret. A dense acoustic image is a prediction, not an authenticated physical field.

### 10.4 DCASE extensions and execution sequence

Use the completed 2026 results as a reproducibility target rather than inventing a new pooled benchmark:

- First reproduce Task 3 audio-only and audiovisual tracks separately, with the exact 4-channel evaluation input contract and official scoring.
- Compare a qualified audio-only SELD/acoustic-imaging pipeline, audiovisual fusion, and the same audiovisual system with video removed. Keep audio/video training data access explicit.
- Use the 32-channel development recordings only under the official data terms and train/evaluation split; do not allow unavailable 32-channel data or labels into evaluation-time inference.
- Add Task 4 spatial semantic segmentation, Task 5 audio-dependent QA and Task 6 long-audio moment retrieval as separate follow-on lanes, not one composite score.
- Join only timestamp/pose-qualified acoustic and visual claims into the #12837 event/world-state adapter; preserve independent modality receipts and source roots.

DCASE's 2026 challenge ran 1 April–15 June 2026 and the 2026 workshop took place 27–29 October 2026. The results are therefore a completed comparison, not an upcoming submission opportunity.

### 10.5 OVO-S human-anchor protocol correction

The OVO-S project repository's displayed leaderboard explicitly distinguishes **Human (streaming) = 86.6** from **Human (offline) = 92.2**, while Gemini-3.1-Pro is 59.2 overall and 54.9 on L4. The EMNLP abstract text describes the 92.2 human figure as being evaluated under the same prefix-access protocol, which is inconsistent with the project leaderboard's streaming/offline labels.

For current comparisons, use the official project leaderboard's labelled table as the operational protocol reference: compare streaming systems to 86.6 human streaming, and report 92.2 as human offline. Flag the paper-abstract discrepancy in any report rather than silently selecting the larger human number. Do not use either human figure as a deployable model target; use them as separate reference anchors.

Sources:
- OVO-S official leaderboard/protocol: https://github.com/InternLM/OVO-S-Bench
- EMNLP 2026 paper abstract: https://aclanthology.org/2026.emnlp-main.2374/

### 10.6 Revised multi-modal spatial benchmark ladder

| Lane | Official baseline anchor | First target | Stretch target | Required comparison |
|---|---|---|---|---|
| OVO-S streaming spatial | Gemini-3.1-Pro 59.2; Qwen3-VL-235B-A22B 53.6; Human streaming 86.6 | ≥62 overall | ≥65 overall | Prefix-only; report L1–L4; preserve streaming/offline human distinction |
| OVO-S L4 allocentric map | Gemini-3.1-Pro 54.9; Qwen3-VL-235B-A22B 45.7 | ≥58 | ≥62 | No material L1–L3 regression; evidence-span and map consistency |
| DCASE Task 3 audio-only | Best listed Mask mAP 0.1080 | Reproduce official top system | Paired improvement over reproduced leader | Four-channel evaluation input; exact map rendering and official scorer |
| DCASE Task 3 audiovisual | Rank-1 cumulative entry 0.0131 Mask mAP / 0.4002 Pearson r | Reproduce official rank and all metrics | Improve both native quality and matched modality ablation | Do not infer superiority from cumulative rank alone |
| Cross-modal spatial decision | No shared official score | Zero false source joins on deterministic negative fixtures | Lower decision regret at lower acquisition cost | Hidden paired worlds; independent audio/video evidence and qualified transforms |

These are research gates, not achieved Symthaea results. DCASE-native reproduction, OVO-S-native reproduction and internal cross-modal decision evaluation must be reported separately.


## 11. OVO-S benchmark refresh: retrieval is not the primary bottleneck

The current official OVO-S-Bench project page now reports 38 evaluated systems and includes Qwen3.5 models. This supersedes the earlier secondary-tracker note that described post-April model coverage as absent; the official table itself now includes Qwen3.5-397B-A17B, Qwen3.5-27B, Qwen3.5-9B and Qwen3.5-4B. Do not repeat the older claim that these models have no OVO-S scores.

### 11.1 Current official leaderboard snapshot

All values below are the official streaming-protocol multiple-choice accuracy table:

| Model | L1 | L2 | L3 | L4 | Overall |
|---|---:|---:|---:|---:|---:|
| Gemini-3.1-Pro | 61.9 | 64.0 | 55.9 | 54.9 | 59.2 |
| Qwen3-VL-235B-A22B | 52.5 | 55.2 | 61.2 | 45.7 | 53.6 |
| Qwen3.5-397B-A17B | 49.6 | 55.4 | 58.1 | 45.4 | 52.1 |
| Qwen3.5-27B | 51.5 | 55.2 | 52.4 | 47.7 | 51.7 |
| GPT-5.4 | 54.6 | 57.6 | 50.8 | 40.5 | 50.9 |
| InternVL-3.5-241B-A28B | 55.6 | 55.7 | 51.6 | 40.5 | 50.9 |
| Human (streaming) | 93.2 | 81.0 | 86.4 | 79.2 | 86.6 |
| Human (offline) | 97.0 | 86.2 | 94.2 | 89.2 | 92.2 |

The best reported model remains Gemini-3.1-Pro at 59.2 overall / 54.9 L4; the strongest listed open baseline remains Qwen3-VL-235B-A22B at 53.6 / 45.7. The official streaming-human anchor is 86.6 / 79.2 L4. The table is a dated list of evaluated systems, not a guarantee that all currently available models have been tested. Never impute scores for untested systems.

### 11.2 The important negative result: more recall is not enough

The updated OVO-S project analysis reports:
- L4 is the lowest-scoring level for 28 of 34 systems with L4 results, with an average 9.3-point gap from L1–L3.
- Evidence Recall has near-zero per-query Pearson correlation with correctness for HERMES, StreamingTOM and FluxMem: (r\in[-0.07,0.00]).
- An oracle-evidence sampler and doubling the frame budget each improve by no more than +0.3 points over uniform 128-frame sampling.
- 13 of 15 streaming/spatially fine-tuned variants underperform their comparable general backbone overall; 13 of 15 regress on L4 (mean L4 change −6.1).
- In the evaluated CoT error analysis, 60–80% of wrong traces for three model families are attributed to mis-grounded visual evidence.

This materially changes the research priority. **Do not treat OVO-S as primarily a retrieval or frame-budget problem.** Better access to evidence does not guarantee that the system constructs a correct global map from it. We should prioritize grounded entity association, coordinate-frame consistency, trajectory integration, topology construction, loop-closure validation, map revision and downstream spatial decision quality.

This does not mean that long-horizon memory is solved in general. OVO-S tests a particular spatial task family; 4DMemoryBench and EgoMemReason show important persistent-memory weaknesses under much longer temporal and multi-history conditions. Keep these benchmark lanes distinct.

### 11.3 Revised architecture experiment

The primary controlled experiment should hold evidence constant to isolate state construction:
- Native causal stream (candidate's own admitted sampling policy).
- Common sampled evidence (same 128-frame/prefix budget for every candidate).
- Oracle evidence spans (same annotated evidence intervals for every candidate; diagnostic upper bound, never a deployable score).
- Common map-state handoff (same downstream evaluator consumes every candidate's state projection).

If models remain weak under common/oracle evidence, the limiting factor is not retrieval. The candidate must produce an explicit, revisable state with source-linked geometry and topology rather than a prose summary or an ungrounded chain of thought.

Add direct score planes:
- entity/landmark identity, false merge and false split;
- camera/world transform and pose uncertainty;
- trajectory and map alignment;
- topological edge precision/recall, false adjacency and missed connectivity;
- false loop closure and route-history reconstruction;
- claim-to-source-span grounding and epistemic-origin integrity;
- map revision under occlusion, late contradictory evidence and calibration changes;
- downstream DecisionNeed-specific regret, risk/coverage and abstention.

Spatial specialization is only admitted after paired matched-backbone factorial ablations. The current OVO-S results are a warning against assuming a spatial fine-tune is an improvement.

### 11.4 Official source and coverage note

Official OVO-S results and analysis: https://internlm.github.io/OVO-S-Bench/ and https://github.com/InternLM/OVO-S-Bench.

The current official project page, rather than an older secondary tracker snapshot, is the source of the model list, per-level scores and diagnostic findings in this section.

## 12. Long-horizon spatial memory is a separate challenge

Newer benchmarks confirm that the field has not solved persistent multimodal memory; they do not invalidate the OVO-S finding that evidence recall alone is insufficient for allocentric mapping.

### 12.1 4DMemoryBench

4DMemory introduces LongVideo with approximately 887–900 hours of source video, 5,214 manually reviewed questions, multi-history queries and a maximum temporal extent of 97.98 hours. Its harness combines question-independent persistent memory, question-conditioned retrieval and specialist tools. The paper evaluates 13 API-served models with a separate Gemini-3.1-Pro media observer, which gives every evaluated model access to prepared media evidence. This observer-mediated setup must be reproduced and reported as such; it is not a same-input standalone model leaderboard.

Under the paper's majority-judge Overall column, the leading entries include Muse Spark-1.1 at 77.79, Grok-4.6 at 75.55 and Gemini-3.7-Flash at 73.34. The paper also reports that the best observed Overall is lower than tool-invocation rates (at least 98.70% for 12/13 systems), showing that calling tools is not equivalent to successfully executing ordered evidence operations and answering correctly.

Source: https://arxiv.org/abs/2610.32332

### 12.2 EgoMemReason and LongSpace

- **EgoMemReason:** 500 week-long egocentric questions across entity, event and behavior memory; the best reported system achieves 39.6% overall accuracy. Use it to measure multi-day state changes, event chronology and behavioral abstraction—not only spatial-map accuracy. https://arxiv.org/abs/2605.09874
- **LongSpace-Bench / LongSpace (EMNLP 2026):** room-tour spatial memory with scene perception, spatial relations and long-horizon recall; LongSpace uses sequential chunks, early 3D structural cues and layer-aware memory. Reproduce its method as a spatial-memory comparator, but do not equate its offline/chunked protocol with OVO-S causal streaming. https://aclanthology.org/2026.emnlp-main.227/

### 12.3 Revised benchmark portfolio

| Research question | Primary benchmark | Diagnostic companions |
|---|---|---|
| Can the system construct an allocentric map from a causal prefix? | OVO-S-Bench | Spatial state construction fixtures / map-state evaluator |
| Does a better memory help when the evidence spans hours or days and multiple histories? | 4DMemoryBench | EgoMemReason, LongSpace |
| Can the system associate sounds with spatial sources and preserve acoustic fields? | DCASE 2026 Task 3 | OVO-S spatial map diagnostics |
| Does the system act, wait or abstain at the right moment? | OmniInteract / SPOT / DIVE | #6675 sequential policy tournament |
| Does compressed memory preserve future decisions? | MACE #13284 | Multi-DecisionNeed spatial sequences |

Do not pool these into a single score. Keep their evidence access, model observer, temporal horizon, task semantics, judge and resource accounting explicit.



## 13. Constraint-carrying allocentric state

The new spatial-state programme [VIS-012](https://github.com/Luminous-Dynamics/symthaea/issues/13855) proposes a stronger world-model substrate than a single best-estimate map: a **versioned set of source-grounded spatial constraints and admissible map alternatives**.

An observation should create claims/constraints with explicit provenance, frame, time, uncertainty and epistemic origin. The system should retain alternative identity associations, pose relations and topological edges when the observations do not uniquely determine them. A current DecisionNeed may use a decision-equivalent quotient without collapsing the underlying alternatives.

### Candidate invariants

- **Frame correctness:** image coordinates do not become world coordinates without admitted calibration/pose and a transform uncertainty envelope.
- **Cycle consistency:** transform composition around a closed pose-graph cycle must agree with identity within a preregistered residual bound. A failed closure remains a rejected/alternative constraint rather than being silently optimized away.
- **Topological validity:** direct adjacency requires an observation or typed inference rule; sequential visits alone do not establish a doorway/connection.
- **Identity consistency:** repeated appearance is not sufficient for loop closure when trajectory or geometric constraints conflict.
- **Temporal consistency:** a late correction updates current map belief while preserving earlier known-at state and audit history.
- **Decision sufficiency:** resolve only the distinctions that change the exact current DecisionNeed's action; retain source-linked residuals for later queries.

### Evaluation consequences

Add pose-graph cycle residuals, transform/uncertainty coverage, false loop-closure rate, topological edge support class, false adjacency, map hypothesis count, contradiction localization and minimum affected-subgraph revision cost. Report map quality separately from answer accuracy and retrieval recall.

LongSpace-Bench is a relevant method comparator because LongSpace injects frozen 3D geometric cues into early decoder layers and uses sensory/working/long hierarchical KV memory for query-guided recall. Its EMNLP protocol is room-tour long-video reasoning, not OVO-S causal streaming. Compare the representation/hierarchy as a baseline, but keep protocol access and temporal horizon explicit. Official paper: https://aclanthology.org/2026.emnlp-main.227/.

The contribution hypothesis is not “HDC replaces SLAM” or “a constraint graph beats all VLMs”. It is that multimodal models, HDC/CfC state, explicit geometric/topological constraints and independent decision evaluation can be composed so that every map claim is grounded, revisable and no more specific than its evidence permits.


## 14. Classical geometric validation: separate semantic spatial reasoning from SLAM

VIS-012 now composes the native OVO-S semantic/allocentric question benchmark with classical geometric datasets. This is necessary because correct long-horizon spatial answers do not prove metrically consistent pose/map reconstruction, and low trajectory error does not prove semantic scene understanding.

### 14.1 Dataset lanes

| Dataset | Qualified research lane | Native metric/contract | Critical limitation |
|---|---|---|---|
| TUM RGB-D | RGB-D visual odometry and SLAM | official trajectory evaluation; ATE/RPE under documented alignment | 39 sequences; Kinect RGB/depth 640×480 at 30 Hz; motion capture trajectory at 100 Hz; calibration and association matter |
| EuRoC MAV | stereo-inertial VIO/SLAM | pose/trajectory error with camera/IMU calibration and timestamps | 2×20 FPS stereo, 200 Hz IMU; Vicon room and Leica machine-hall ground truth differ; the dataset documents time synchronization limitations |
| KITTI Odometry | monocular/stereo visual odometry; separate LiDAR/multimodal track | official subsequence translation error (%) and rotation error (deg/m) | sequences 00–10 have training GT; 11–21 are hidden-test; never mix monocular, stereo, LiDAR or visual-LiDAR scores as a common method class |
| OpenLORIS-Scene | lifelong/multi-session SLAM | repeated trajectory consistency, relocalization and map-change handling | multiple visits, dynamic humans and day/night; GT is OptiTrack in office and offline LiDAR-SLAM for other scenes, so GT provenance differs by scene |

Sources:
- TUM RGB-D paper / dataset: https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf ; https://vision.in.tum.de/data/datasets/rgbd-dataset
- EuRoC MAV: https://projects.asl.ethz.ch/datasets/euroc-mav/
- KITTI Odometry: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
- OpenLORIS-Scene: https://lifelong-robotic-vision.github.io/dataset/scene

### 14.2 Metric and protocol discipline

- ATE is reported only after the official alignment; explicitly state rigid SE(3) versus Sim(3) alignment. Do not erase monocular scale ambiguity by silently scaling an unobservable result.
- RPE uses frozen time/distance intervals.
- KITTI's translation (%) and rotation (deg/m) are computed over official subsequence lengths; these are not ATE.
- Pose-graph cycle residual, loop-closure precision/recall, reprojection or point-cloud alignment are separate diagnostics with declared calibration and input geometry.
- OpenLORIS map-change/false-deletion metrics must be reported per scene and per GT source class.
- Every run pins sensor modality, camera/IMU rate, calibration, dataset revision, trajectory alignment, hardware, runtime, memory and compute budget.
- Do not pool TUM, EuRoC, KITTI and OpenLORIS into one scalar or compare LiDAR methods to camera-only systems without explicit track separation.

### 14.3 Required factorial and negative controls

- RGB-only, RGB-D, stereo, stereo+IMU, and LiDAR only where actually supported.
- Camera-intrinsics and camera–IMU-extrinsic perturbations.
- Timestamp offset, drift, reset and sensor-specific clock confusion.
- Wrong transform direction/frame identity and cycle-inconsistent pose graph.
- Dynamic object motion versus camera ego-motion.
- False loop closure from visually similar locations.
- Revisit after illumination, furniture, scene and people changes.
- Stale/missing depth and IMU gaps.
- Shuffled or reversed trajectories.
- Metric-scale ambiguity and unsupported world-coordinate claims.

### 14.4 Composition with ErgoStream

Keep the lanes independent:
- **OVO-S:** semantic spatial question answering from causal prefixes.
- **TUM / EuRoC / KITTI:** metric pose and geometric trajectory qualification.
- **OpenLORIS:** long-term/lifelong map maintenance and relocalization.
- **DCASE Task 3:** spatial acoustic-field estimation.
- **4DMemoryBench / EgoMemReason / LongSpace:** long-horizon, multi-history memory and spatial recall.
- **Symtropy #6184:** deterministic paired counterfactuals for frame, topology, false-loop, contradiction and DecisionNeed errors.

VIS-012 #13855 owns the candidate projection and controlled adapter. Vision Mind #3934 owns persistent belief/geometry integration. HDC/dense/hybrid representation comparisons remain in #8152. Symtropy retains hidden truth and independent scoring. No second SLAM backend, map store, sensor service or scheduler is introduced by the benchmark protocol.

The target is a joint Pareto frontier over semantic spatial reasoning, geometric accuracy, topological correctness, long-term map revision, uncertainty calibration and measured resource cost—not a pooled leaderboard.


## 15. KITTI Odometry current modality-specific reference

The official KITTI odometry table currently lists:
- **SOFT2 (stereo):** 0.53% translation error and 0.0009 deg/m rotation error.
- **V-LOAM (LiDAR):** 0.54% and 0.0013 deg/m.
- **LOAM (LiDAR):** 0.55% and 0.0013 deg/m.

This is not one pooled "SLAM SOTA" comparison: SOFT2 uses stereo imagery, V-LOAM/LOAM use LiDAR point clouds. Use SOFT2 as a stereo reproduction anchor and keep LiDAR references separate.

KITTI's official protocol has 22 stereo sequences (00–10 with public training ground truth; 11–21 hidden test), requires fully automatic evaluation with one parameter set across test sequences, and computes translation (%) / rotation (deg/m) over 100–800m subsequences. The benchmark policy requires significant novel work leading to a peer-reviewed publication for official submissions; it also reports a server restore on 7 September 2026 to the 7 August 2026 backup, so intervening submission history may need resubmission.

Source: https://www.cvlibs.net/datasets/kitti/eval_odometry.php

A prospective stereo goal is to reproduce SOFT2 first, then beat 0.53% / 0.0009 deg/m on the official hidden-test protocol using the same modality and fixed-parameter rules. A train-split result is a development result, not leaderboard superiority. This target is research-only and not achieved by Symthaea.


## 16. Method comparators for spatial-state construction

The external comparison must distinguish complete SLAM systems, geometry priors, frozen features and long-video memory:

- **ORB-SLAM3:** a complete feature-based visual, visual-inertial and multi-map SLAM system for monocular, stereo and RGB-D modes. It is the classical tracking/relocalization/multi-session baseline. The paper reports 3.5 cm average stereo-inertial accuracy on EuRoC under its own evaluation protocol; reproduce the official dataset metric and do not treat that figure as a universal cross-dataset score. https://doi.org/10.1109/TRO.2021.3075644
- **MASt3R-SLAM (CVPR 2025):** dense monocular SLAM built around a two-view reconstruction prior with pointmap matching, camera tracking/fusion, graph construction, loop closure and global optimization. The authors report 15 FPS and state-of-the-art performance across the benchmarks they evaluated with known calibration. They also identify lens-distortion/pinhole-training and full-resolution decoder-throughput limitations. Use it in a matched monocular lane, not against stereo/RGB-D numbers. https://openaccess.thecvf.com/content/CVPR2025/html/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.html
- **DINOv3:** a frozen dense-feature arm, not a metric SLAM system. Meta reports 7B-scale backbones trained on 1.7B images and high-resolution features for downstream visual tasks. It belongs in #8152's Patch-HDC vs DINOv3 vs hybrid representation test with one fixed downstream geometry/association backend. It does not provide metric scale, trajectory correctness or map topology by itself. https://ai.meta.com/research/dinov3/
- **LongSpace:** room-tour long-video spatial-memory comparator using sequential chunks, early 3D structural cues and layer-aware memory. Reproduce under its own protocol, separate from OVO-S causal streaming and classical SLAM. https://aclanthology.org/2026.emnlp-main.227/

### Matched method ladder

1. ORB-SLAM3 where sensor modalities match.
2. MASt3R-SLAM on the same monocular stream and calibration assumptions.
3. DINOv3 frozen features through the common geometry/association backend.
4. Patch-HDC features through the exact same backend.
5. DINOv3/HDC hybrid through the common backend.
6. Constraint-carrying map state with DecisionNeed-aware acquisition.
7. No-map/no-geometry/no-persistence/no-loop-closure/no-active-acquisition lesions.

Pin code commit, model and weights digests, preprocessing, camera model, calibration, resolution, precision, hardware and resource budget. Keep off-the-shelf and trained/adapted results separate. The paper-reported 15 FPS is not a comparable runtime number until input and hardware are matched.

The hypothesis is not that HDC replaces SLAM or that a large VLM should be the map authority. It is that replaceable perceptual specialists (including learned dense features and HDC) can feed source-grounded, joint pose/map constraints, temporal revision, topological validation and decision-sufficient acquisition, with an independent evaluator catching unsupported claims. This composition must beat the strongest appropriate specialist at matched cost, not merely outperform weak/no-map baselines.
