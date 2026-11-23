# ErgoStream Competitive Benchmark & Evaluation Plan

**Date:** 2026-11-23 · **Status:** research planning note; no baseline has been executed by this note
**Programme:** Symthaea #13401 · Symtropy #7149

## Executive position

ErgoStream should not claim to be a new general-intelligence leaderboard. It is a focused evaluation of **task-agnostic structural acquisition**: can a learner build a useful, revisable structural library from unlabeled, causally arriving streams before the learner is told what downstream questions matter?

Existing suites measure complementary capabilities. They should be used as adapters and independent probe sets, not pooled into a single score. The new benchmark must test the acquisition phase that many external tasks do not isolate.

## 1. Competitive benchmark map

| Track | External benchmark / system | What it tests | Fit to ErgoStream | Limitation / correct use |
|---|---|---|---|---|
| Ansatz-free scientific discovery | [Unsatz](https://arxiv.org/abs/2611.03713) | Learner proposes variables, hidden state, interactions and measurement/model form, buys experiments and predicts held-out interventions/conditions/horizons | **Closest direct comparator** for structural/model-form discovery | Agent starts with a scientific task and a model-program language; it is not a generic pre-task acquisition benchmark. Reproduce its exact task protocol before comparing |
| Interactive exploration | [ARC-AGI-3](https://arcprize.org/arc-agi/3) | Explore unfamiliar interactive environments, infer goals/rules, build adaptive world models and learn from actions without natural-language instructions | Strong external probe for acquired structures supporting exploration and transfer | It is task/game-centric and does not independently measure a pre-task structural library. Final ARC Prize 2026 results are scheduled for 2026-12-04; do not treat them as published as of this note |
| Visual causal reasoning | [CLEVRER](https://clevrer.csail.mit.edu/) | Descriptive, explanatory, predictive and counterfactual questions over object-interaction videos | Good post-acquisition probe for temporal/event relations and causal predictions | Synthetic, short, object-collision videos; task labels/questions are exposed at evaluation, so this is not task-agnostic acquisition by itself |
| Interactive physical reasoning | [PHYRE](https://phyre.ai/) | Sample-efficient solving and generalization across 2D physical puzzles | Useful for testing whether discovered dynamics/relations support action selection | Puzzle goals are provided; not a test of open-ended or goal-free acquisition |
| Real audiovisual grounding | [DCASE 2026 Task 3](https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes) | Reconstruct semantic spatial acoustic energy maps / dynamic masks from low-channel audio, optionally audiovisual | Strong sensory and cross-modal downstream qualification adapter | Measures a specific sound-imaging task, not broad structural discovery. 2026 challenge reports 29 Task 3 submissions from 8 teams, split audio-only and audiovisual |
| Dynamic manipulation | [RoboDyna](https://robodyna.github.io/robodyna/) | Physically grounded object dynamics and closed-loop manipulation when the world continues evolving during inference | High-value Symtropy probe for whether acquired event/dynamics structure matters to decisions under latency | Robotics task performance is not a direct metric of generic acquisition; retain sync/async, expert-demonstration and human comparisons |
| Moving-target manipulation | [LIBERO-Dynamic](https://arxiv.org/abs/2610.24579) | Dynamic extension of LIBERO with moving objects, model-specific inference latency and dynamic demonstrations | Complementary test of temporal/embodied transfer and planning under stale observations | Fine-tuning and demonstrations affect results; match data and training budget before attributing benefit to the architecture |

## 2. What the current state of the art tells us

### Unsatz is the closest scientific competitor

Unsatz (Ji, Wang and Sun, 2026-10-31) evaluates scientific discovery without a supplied ansatz over 80 simulated systems and 20 domains. The authors report that nine LLMs solve between 21% and 60% of systems. It exposes a key failure mode: agents may identify which variables exist but not how they act, and may prefer a lower training error over better held-out prediction. These are reported paper results, not a reproduced Symthaea comparison.

**Response:** use Unsatz as the direct ansatz-free scientific-discovery comparator within the #11042 lane. ErgoStream adds a separate pre-task acquisition phase and a library snapshot that is tested on later undisclosed probes. Do not reimplement the Unsatz task DSL or claim superiority from a different synthetic task.

### Interactive world models need closed-loop outcome evaluation

The 2026 *Survey of World Model Benchmarks* catalogues 130 representative benchmarks and separates capability, protocol, metric family and data source. Its survey snapshot identifies a marked imbalance between open-loop prediction protocols and closed-loop interaction, and between prediction-level and downstream-outcome metrics.

**Response:** evaluate both whether a system predicts/reconstructs structure and whether that structure improves decisions in environments where actions and inference latency affect outcomes. Prediction accuracy alone is not enough.

### Dynamic robotics is a harder test than frozen-video prediction

RoboDyna reports 20 conceptual dynamic manipulation tasks across five categories plus 10 household tasks, 3,500 expert demonstrations and synchronous/asynchronous evaluation. Its authors report sub-20% success across evaluated policies, and that inference latency can flip individual episode outcomes. LIBERO-Dynamic similarly makes target motion and policy inference latency explicit across dynamic variants of the LIBERO tasks.

**Response:** don't claim robotics transfer from offline video QA. Reuse Symthaea's existing RoboDyna/LIBERO-Dynamic adapter (#13389) after the learner has passed the smaller causal acquisition probes. Report success, partial completion, progress, recovery, latency and human gap separately.

### Audiovisual grounding is a modality-specific qualification layer

DCASE 2026 Task 3 makes the sensory problem concrete: map four-channel spatial-audio input to a high-resolution semantic acoustic scene, with optional visual input. The challenge received 29 submissions from 8 teams. That is useful for validating source localization, event masks and cross-modal alignment, but it should not be represented as a general intelligence test.

**Response:** adapt DCASE Task 3 through the AV-STREAM path (#12837) as an external sensory specialist comparison, retaining separate audio-only and audiovisual tracks.

## 3. Required baseline ladder

All arms should receive the same stream, split, modality availability, observation/intervention budget, clock information, hardware, wall-time and memory budget. Where a model requires training, equalize and report training data and compute separately from inference cost.

1. **Simple controls:** no-change/persistence, random acquisition, passive next-frame, shuffled temporal order, shuffled modality alignment, and constant/IID-noise controls.
2. **Acquisition policies:** surprise/prediction-error only, random-network-distillation novelty, learning-progress, information gain, cost-normalized information gain, predictive compression / MDL, and a hybrid acquisition policy.
3. **Representation controls:** strong modality-specific encoders with event association; frozen dense multimodal embeddings; HDC only; CfC/LTC only; explicit typed graph/hypergraph; bounded symbolic/e-graph; HDC–CfC plus graph/e-graph hybrid.
4. **World-model controls:** object/event-centric recurrent and transformer models; the strongest compatible action-conditioned world model; a no-transfer/fresh-learning control.
5. **Scientific-discovery controls:** Unsatz-compatible agent/program interface, PySR or another pinned symbolic-regression baseline when the candidate grammar and task class match, SINDy only when its sparse-regression assumptions are satisfied, and exact tiny-world enumeration as evaluator diagnostic only.
6. **Human reference:** where the task supports meaningful human interaction, include a human sample-efficiency / action-efficiency reference. It is not a substitute for a matched machine baseline.

No name-only reproduction. Record exact code commit, weights, model/API revision, prompt, tool access, preprocessing, operator set, task subset, compute, retries, failed runs and licensing/data exclusions. For proprietary APIs, report black-box evaluation conditions and do not imply implementation-level parity.

## 4. Scorecard and target frontier

Do not collapse the results into a single ErgoScore. Report three non-substitutable axes:

### Capability
- Unit/event segmentation and equivalence-class recovery.
- Typed relation / motif recovery and compositional generalization.
- Held-out continuation and intervention prediction.
- Cross-modal association, timing and missing-modality robustness.
- Transfer to new generator families, compositions and downstream tasks.

### Epistemic reliability
- False relation, false causal claim, false shared-grammar / false-invariance rates.
- Correct set-valued response on non-identifiable cases.
- Calibration, risk–coverage, abstention and outside-support detection.
- Library reuse precision, negative-transfer cost, valid-scope retention and retraction latency.

### Efficiency / operations
- Total observations, active queries and intervention cost.
- Candidate/equation/model evaluations, model calls/tokens, training and inference compute.
- CPU/GPU time, p50/p95/p99 latency, peak memory and energy when measured.
- Cost to first independently validated structure; incomplete attempts remain in denominators.

Publish Pareto frontiers: quality vs all-in cost; transfer vs false-structure rate; latency vs task outcome. Include worst-family and worst-environment results, not only pooled means.

## 5. Proposed internal go/no-go targets

These are prospective engineering targets, not external standards or achieved results. Establish pilot variance and freeze the precision/power plan before confirmatory testing.

- Exact agreement with independent oracles on every in-scope finite micro-world.
- Zero hidden-truth/future-frame leakage in serialized child views and negative controls.
- No false causal promotion on controls where only correlation/co-occurrence is provided.
- No false unique identification on evaluator-defined non-identifiable cases.
- At least six structurally distinct generator families before a confirmatory transfer claim.
- Stretch target: at least +10 percentage points in primary structural-transfer success over the strongest matched open baseline, **or** at least 20% lower all-in observation/compute cost at a preregistered non-inferiority margin no wider than 2 percentage points. These thresholds must be revisited if pilot variance/power analysis shows they are under- or over-demanding.
- No SOTA statement unless the comparator was directly run under the same contract, source artifacts are pinned, complete costs and all failures are reported, and paired uncertainty intervals support the claim.

## 6. Recommended staged programme

**Stage 0 — protocol integrity:** schemas, tiny finite generator, independent exact evaluator, hidden-field checks, action accounting and deterministic replay.

**Stage 1 — passive structure acquisition:** discrete motif streams; freeze the acquired library; score held-out motif relations and compositions. Include random, passive, MDL and learning-progress controls.

**Stage 2 — active disambiguation:** matched-marginal worlds and unidentifiable prefixes; allow bounded observations/interventions; score cost to validated distinction and safe unresolved behavior.

**Stage 3 — multimodal acquisition:** cross-modal delayed/missing alignment; adapt AV-STREAM and DCASE tracks; retain modality-native timing/provenance.

**Stage 4 — physical transfer:** CLEVRER, PHYRE, then RoboDyna and LIBERO-Dynamic. Separate synthetic, benchmark and real physical evidence tiers.

**Stage 5 — scientific discovery:** matched Unsatz/ACDB/Symtropy model-proposal comparisons, held-out ranges, interventions and longer horizons.

**Stage 6 — confirmatory study:** six or more generator families, family-held-out test, preregistered uncertainty plan, matched strongest baselines, full resource accounting and external review.

## 7. References

- Unsatz: https://arxiv.org/abs/2611.03713
- ARC-AGI-3: https://arcprize.org/arc-agi/3
- CLEVRER: https://clevrer.csail.mit.edu/
- PHYRE: https://phyre.ai/
- DCASE 2026 Task 3: https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes
- RoboDyna: https://robodyna.github.io/
- LIBERO-Dynamic: https://arxiv.org/abs/2610.24579
- Survey of World Model Benchmarks: https://world-model-benchmarks.github.io/
- Gromov, *Ergostructures, Ergologic and the Universal Learning Problem*: https://www.ihes.fr/~gromov/wp-content/uploads/2018/08/ergologic31.pdf