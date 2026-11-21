# Temporal associative-retrieval capacity (v0.1)

**Status:** research-only protocol and executable example  
**Issue:** [#13099](https://github.com/Luminous-Dynamics/symthaea/issues/13099)  
**Example:** `crates/core/symthaea-hdc-ltc/examples/temporal_associative_capacity.rs`  
**Protocol version:** `symthaea.temporal-associative-capacity/0.1`

## Research question

How much key–value information remains recoverable from an HDC bundle after the state is allowed to evolve over elapsed time and receive unrelated events?

This is a narrow measurement bridge between static VSA capacity and continuous-time temporal dynamics. It is deliberately not a learned episodic-memory claim: v0.1 performs **passive retention** of a static superposition. It does not train the neuron to store, consolidate, or recall episodes.

## Existing algebra and task

For each seed, dimension (D), and fan-in K, generate independent bipolar key/value codebooks:

[
 k_i,v_iin{-1,+1}^D.
]

Store the superposition using the crate's existing operators:

[
 M=operatorname{bundle}_{i=1}^{K}(k_iodot v_i).
]

For query key (k_i), unbind and rank the frozen value codebook:

[
 q_i=k_iodot h,qquad
 hat i=argmax_joperatorname{cos}(q_i,v_j).
]

No new HDC binding or bundling primitive is introduced. The static HDC bundle is the incumbent reference.

## Experimental conditions

| Condition | Initial state / query | Interpretation |
|---|---|---|
| `static_hdc_bundle` | Original bundle, original keys | Static retrieval reference |
| `static_mismatched_pairing_control` | Bundle keys with the next value in a cyclic permutation; original expected labels | Deliberately wrong association; should not recover the original pairing |
| `static_random_query_control` | Original bundle, independent random query keys | Null control; top-1 accuracy should be near (1/K) over enough independent seeds |
| `hdc_ltc_zero_input` | Original bundle evolved under zero input | Passive state evolution without new sensory content |
| `hdc_ltc_distractor` | Original bundle evolved under deterministic unrelated bipolar inputs | Retention under interference |

For dynamic conditions, the same input-event count is run with regular and deterministic irregular time gaps summing approximately to the requested horizon. Each gap is supplied causally; no future input is visible. The current neuron floors each integration interval at (10^{-9}) seconds, so the runner rejects smaller positive gaps to prevent the evolved state and reported logical time from diverging. A positive horizon must contain at least one evolution event. Dimensions below 8 are rejected because the neuron constructor orthogonalizes five internal hypervectors and very small dimensions make that initialization degenerate. Static controls are emitted once per corpus, not copied into each temporal cohort.

The neuron is initialized with the exact same bundle via the existing public state setter and the same deterministic model seed across the two dynamic conditions. Its configuration, weights and state are therefore paired for each seed/dimension/load.

## Metrics

Every query produces a JSONL row containing the chosen value index, correctness, tie/validity flags, target similarity, best non-target similarity and target-vs-best-distractor margin.

Each cohort also emits:
- top-1 accuracy, with invalid/tied outcomes not silently counted as correct;
- chance accuracy (1/K);
- mean target/distractor similarity and retrieval margin;
- a **finite-sample plug-in Fano diagnostic**;
- initial/final state L2 norms, so the neuron's post-step global norm cap (5.0 in the current implementation) is visible rather than hidden in a cosine-only metric;
- neuron API update count, neuron-reported logical time, and absolute discrepancy from the requested horizon;
- separate wall-time observations for codebook construction, model setup, event/gap preparation, state evolution and query decoding;
- the exact positive interval schedule for each dynamic cohort, its coefficient of variation, and each distractor input seed. The single-event case is not duplicated as an "irregular" cohort because one interval cannot exhibit timing irregularity.

For K balanced labels and top-1 error rate e, Fano's inequality gives the diagnostic:

```text
I(Y; Ŷ) ≥ max(0, log₂K − h₂(e) − e·log₂(K−1))
```

The JSON field is named `plug_in_fano_information_lower_bound_bits`, but the value is calculated from the finite cohort's observed error rate. It is a descriptive plug-in diagnostic—not a confidence interval, a distribution-free finite-sample guarantee, or a substitute for estimating mutual information under a specified sampling distribution. The diagnostic is not the same as retrieval accuracy: a deterministic but wrong permutation can still preserve information about the label while failing the original semantic lookup. The per-query rows are retained so that uncertainty estimation and alternative estimators can be applied later.

Wall-clock fields are observations from a single process run, not energy measurements. Codebook creation, neuron model setup, event/gap preparation, the neuron-evolution loop and query decoding are timed separately; the evolution loop includes the neuron method's own post-step statistics work. The codebook preparation duration is repeated in each summary for that paired corpus, but it is incurred only once per seed/dimension/fan-in combination. The API update counter counts public evolution calls, not floating-point operations or internal component operations. The seed derivation is domain-separated: key codes use domain 1, value codes domain 2, the gap generator domain 3, neuron initialization domain 4, distractor vectors domain 5, and random null-query keys domain 6. The exact intervals and distractor seeds are emitted in the summary so replay does not depend on reconstructing an implicit schedule. The per-event u64 seeds are emitted as fixed-width hexadecimal strings to avoid loss of integer precision in JSON consumers using IEEE-754 numbers. Cohort IDs encode the exact f64 horizon bit pattern, not a rounded decimal rendering. The initial and final state norms are diagnostics because the current HDC-LTC implementation rescales a state above its global norm limit after evolution; comparisons across dimension/fan-in must account for this nonlinear amplitude effect.

## Running

From the repository root:

```bash
cargo run --release --manifest-path crates/core/symthaea-hdc-ltc/Cargo.toml \
  --example temporal_associative_capacity -- \
  --dims 1024,4096 --loads 4,8,16 --horizons 1,10 \
  --events 1,8 --seeds 5 --source-revision "$(git rev-parse HEAD)" \
  --out temporal-capacity.jsonl
```

Arguments:
- `--dims`: vector dimensions (8–262144);
- `--loads`: memory fan-in (2–512);
- `--horizons`: finite non-negative horizons in seconds; zero is reserved for identity/static controls;
- `--events`: number of evolution calls for each dynamic cohort; the codebook and event-input arrays are capped at 8,388,608 dimension-elements per configured cohort to avoid unbounded allocations;
- `--seeds`: number of deterministic corpus/model seeds;
- `--source-revision`: optional 40- or 64-character hexadecimal Git SHA; use the current source commit for publishable measurements. The summary receipt stores null when omitted;
- `--out`: JSONL destination; omit it or pass `-` for stdout.

The JSONL stream contains one `query` record per item followed by one `summary` record for each cohort. Each summary carries the source revision used for the run (or null when the caller did not pin one). A summary with K queries uses all K queries as its denominator; invalid and tied queries remain visible and cannot improve the correctness count.

## Capacity targets for the next qualification phase

**Seed-count warning:** the default three seeds in the raw runner are smoke-scale only. They are not suitable for a confirmatory claim. The independent unit is the corpus/model seed; the K queries within one bundled memory are correlated observations and must never be counted as K independent replicates.

Do not define capacity as the largest fan-in observed once. The target is a temporal capacity surface with a two-level reliability requirement:

```text
K95(D,T,E,CV) =
  largest tested fan-in K for which:
  (1) a seed passes when its semantic top-1 accuracy >= 95%, and
  (2) the one-sided exact 95% Clopper-Pearson lower bound on
      the probability that an independent seed passes is >= 95%.

T90(D,K,E,CV) =
  largest tested horizon T for which:
  (1) a seed passes when its semantic top-1 accuracy >= 90%, and
  (2) the one-sided exact 95% Clopper-Pearson lower bound on
      the probability that an independent seed passes is >= 95%.
```

Here D is representation dimension, T is elapsed horizon, E is event count, and CV is the coefficient of variation of the positive inter-event gaps. The confidence unit is the independent corpus/model seed—not the per-query result.

With zero failures, at least **59 independent seeds** are required for a single one-sided exact 95% lower bound on seed-success probability to reach 95% (59/59 gives approximately 0.95049). Selecting a maximum across multiple tested loads or horizons is a multiple-comparison problem, so the frontier analyzer applies a **Bonferroni family-wise correction** within each fixed D/condition/gap/horizon/event family for K95, and within each fixed D/condition/gap/fan-in/event family for T90. For L tested levels it uses per-level alpha = 0.05/L and requires at least ceil(log(0.05/L) / log(0.95)) seeds even when every seed passes. Thus 2 tested levels require 72 all-passing seeds; 3 levels require 80; 5 levels require 90. Any observed seed failures require more. The analyzer also requires identical seed sets across fan-in levels for K95 and across horizon levels for T90. A missing or unpaired seed set prevents qualification. This correction is within each fixed family only; it does not correct for selecting among multiple dimensions, conditions, gap profiles, event counts, representations or model families. Confirmatory claims across those outer families require a preregistered primary family or an additional outer multiplicity correction (for example, Holm or Bonferroni).

The new Rust post-processor is `examples/temporal_capacity_frontier.rs`. Its output contract is defined in [`data/schemas/symthaea_temporal_capacity_frontier_v0_1.schema.json`](../../data/schemas/symthaea_temporal_capacity_frontier_v0_1.schema.json). It reads the raw JSONL summaries and emits per-cohort seed-reliability records plus discrete K95/T90 records. It requires a valid source Git SHA in every summary, rejects duplicate seeds and malformed counts, groups results by exact source revision, and records the tested-level count, per-level alpha and minimum all-pass seed count. Cohort reliability rows are pointwise diagnostics; K95/T90 use the family-wise Bonferroni correction described above. It never interpolates between untested loads or horizons. A null K95/T90 means no tested point met the full gate; it does not mean the system has zero capacity or zero retention.

Run the raw experiment and frontier analysis from a clean, committed worktree:

```bash
test -z "$(git status --porcelain)" || { echo "worktree must be clean"; exit 1; }
cargo run --release --manifest-path crates/core/symthaea-hdc-ltc/Cargo.toml \
  --example temporal_associative_capacity -- \
  --dims 1024,4096 --loads 4,8,16 --horizons 1,10 \
  --events 1,8 --seeds 80 --source-revision "$(git rev-parse HEAD)" \
  --out temporal-capacity.jsonl

cargo run --release --manifest-path crates/core/symthaea-hdc-ltc/Cargo.toml \
  --example temporal_capacity_frontier -- \
  --input temporal-capacity.jsonl --min-seeds 59 \
  --out temporal-frontier.jsonl
```

Report the whole capacity surface with:
- semantic top-1 accuracy and target-vs-best-distractor margin;
- random-query false-match rate and the number of comparisons performed;
- the plug-in Fano diagnostic, clearly separated from semantic accuracy;
- information retained per stored byte and per evolution update; report joules only when physical energy is measured;
- horizon, event count, irregularity CV and time-to-recovery after a regime change.

Cohort-reliability rows are unadjusted pointwise diagnostics; the K95/T90 summaries are Bonferroni-adjusted within the fixed family. The outer-family limitation above still applies.

The 95% / 90% operating points are **proposed qualification targets**, not existing Symthaea results or universal thresholds. A candidate wins only if its confidence-qualified capacity surface and resource frontier outperform matched baselines under the same workload and measurement boundary.

## What v0.1 can and cannot establish

It can characterize, on synthetic bipolar key/value codebooks:
- static associative retrieval as fan-in and dimension vary;
- passive state retention under the selected HDC-LTC configuration;
- sensitivity to elapsed horizon and event-gap irregularity;
- the additional effect of unrelated event inputs;
- whether a simple random-query null control behaves around chance;
- basic update/setup/preparation/decode cost components.

It cannot establish:
- learned episodic memory, consolidation, online adaptation or continual learning;
- a formal information-retention law or certified mutual-information estimate;
- superiority to state-space, transformer, neural CDE, SNN or commercial retrieval systems;
- real-world data performance, hardware throughput or energy efficiency;
- a universal HDC capacity threshold.

## Matched baseline matrix for later SOTA comparison

The v0.1 runner is a synthetic *associative recall* probe. It must not be presented as a forecasting, continual-learning or neuromorphic-system benchmark. Those comparisons need their own task adapters and measurement contracts.

| Evaluation question | Required comparison family | Fairness rule |
|---|---|---|
| Static VSA capacity | MAP-I, MAP-B, sparse binary and the algebraically applicable HRR/FHRR/GHRR / Symthaea profiles | Reproduce the same symbolic task and error target. IBM's 2026 JAIR analysis concerns set membership and set-intersection estimation; this key/value recall workload is related but not the same benchmark. |
| Key/value associative recall | Existing BinaryHV and ContinuousHV incumbents plus matched MAP / HRR / FHRR / sparse VSA baselines | Match dimension, precision, stored information, query count, bundling load, cleanup oracle, and exact candidate set. Include flat exact scan as the correctness oracle. |
| Continuous-time state modelling | Original CfC variants (including CfC-mmRNN), HDC-LTC, NIALIM/DFA-CfN/PRDFA-CfN, Neural CDE / ODE-RNN, and strong GRU-D / GRU / LSTM controls | Use the same continuous-time task, timestamps, interpolation policy, initial conditions, training data and tuning budget. A passive-retention test is not a substitute for a forecasting or sequence-learning test. |
| Irregular-time forecasting | BITS and TIME-compatible data/tasks, plus strong task-appropriate time-series models | Use their split/no-leakage protocols and report results by missingness, sampling irregularity and pattern, not only a single average. Do not relabel synthetic v0.1 results as BITS or TIME results. |
| Continual learning / neuromorphic relevance | NeuroBench algorithm tasks where applicable; deployed system track only after a real target implementation exists | Algorithm quality/complexity is separate from end-to-end deployed timing and efficiency. |
| Agent memory / experience transfer | MemoryArena, AMA-Bench and same-backbone memory systems; connect to Symthaea #9516 MACE and #9677 replay/stability–plasticity | Evaluate downstream dependent-task decisions, not recall alone. Keep HDC primitive performance separate from full agent-stack performance. |
| Retrieval at scale | Exact scan first; then conventional vector index / ANN and cache baselines where corpus size justifies them | Candidate-generation recall and final exact reranking are separate. Match hardware, concurrency, filters, data residency, cold-start and update/rebuild costs. |

### Agent-memory and industry-system comparison

A mechanism-level HDC associative-recall result is not an agent-memory result. The downstream qualification must preserve a distinct agent/system track:

- **MemoryArena (ICML 2026):** multi-session Memory–Agent–Environment tasks with dependent subtasks, where experience learned in earlier interactions must improve later actions. The authors report that near-saturated performance on long-context recall benchmarks can still fail to transfer to this setting. Its public task families include web navigation, preference-constrained planning, progressive information search and sequential formal reasoning.
- **AMA-Bench (ICML 2026):** long-horizon agent trajectories containing states, actions, observations and tool outputs. Its paper reports that similarity-heavy memory can lose causal and objective information; AMA-Agent combines causality graphs and tool-augmented retrieval and reports 57.22% accuracy, 11.16 percentage points above its strongest baseline on that benchmark. This is a paper-reported result, not a Symthaea comparison.
- **WorldMemArena (May 2026):** 400 multi-session multimodal tasks spanning evolving lifelong state and agentic execution, with stage-level diagnosis of memory writing, maintenance/update, retrieval and use. Its paper reports that better writing/storage alone does not guarantee downstream performance and that multimodal memory remains unstable on realistic trajectories. It belongs in a separate multimodal MACE adapter; do not mix its results with text-only MemoryArena or AMA-Bench. https://arxiv.org/abs/2605.29341
- **Commercial/open memory implementations:** Mem0, Letta/MemGPT, LangMem and Zep/Graphiti are reasonable systems to include in a same-backbone comparison. They must use the identical agent backbone, tool permissions, task set, turn budget, token budget, storage limits and outcome evaluator. A third-party benchmark's single-run recall or vendor-reported LongMemEval/LoCoMo number is context, not an apples-to-apples SOTA ranking.

This downstream work is already owned by [Symthaea #9516 — causal experience transfer across interdependent agent sessions](https://github.com/Luminous-Dynamics/symthaea/issues/9516), which defines MACE and explicitly plans to reuse MemoryArena and AMA-Bench where licensing permits. The mechanism-level bridge is:

- #13099: passive HDC-LTC associative retention and capacity as a low-level representation/control arm;
- #9677: online learning, replay, stability–plasticity, forgetting and retention byte-time;
- #13019: when to refresh slow context;
- #9666: joint sensing, memory, compute and verification under temporal error debt;
- #9516: whether qualified experience is transferred into later dependent agent decisions.

Do not duplicate these owners or collapse their metrics. A future MACE adapter should carry a frozen HDC memory configuration as one candidate arm, preserve the common agent/action authority, and report final task completion, regret, false causal claims, stale-policy reuse, memory bytes and end-to-end cost.

### Current state-of-the-art context

- The [2026 JAIR VSA paper](https://research.ibm.com/publications/capacity-analysis-of-vector-symbolic-architectures) derives capacity bounds for MAP-I, MAP-B and sparse-binary VSAs on set-membership and set-intersection tasks. Symthaea should reproduce those exact workloads under #11145 before extending the conclusion to other algebras.
- The 2026 [BITS benchmark](https://arxiv.org/abs/2609.33303) covers 11 datasets across nine domains and reports that rankings vary with irregularity characteristics and metric choice; it explicitly finds no single modelling strategy consistently dominates. [TIME (2026)](https://arxiv.org/abs/2602.12147) adds 50 datasets and 98 forecasting tasks with strict zero-shot/no-leakage evaluation. Those results support regime-specific Pareto comparisons rather than one universal leaderboard score.
- The 2026 [NIALIM/DFA-CfN/PRDFA-CfN paper](https://doi.org/10.1109/TNNLS.2026.3669345) reports a linear-input-interpolation approach with an error-bound analysis and results across six tasks, including physical dynamics, human activity, PhysioNet, sentiment, remote sensing and energy forecasting. It is a key comparator for the #13031/#13042 temporal-integration programme. Reproduce the pinned public implementation and verify that caller-supplied timestamps are honored before making a matched comparison.
- The original [CfC work](https://www.nature.com/articles/s42256-022-00556-7) established a solver-free approximate formulation for LTC dynamics. It remains a necessary baseline, but newer input-interpolation methods mean that matching only against zero-order-held CfC would be inadequate.
- [NeuroBench](https://neurobench.ai/) separates a hardware-independent algorithm track from a deployment-aware system track. Symthaea should follow that separation: algorithmic retrieval/forecasting quality and complexity first, then measured wall latency, throughput, memory and physical energy on a declared hardware boundary.

### What "better than SOTA" should mean here

Do not optimize for one headline score. A credible win requires a preregistered operating regime and a Pareto improvement across:
1. **Quality:** correct retrieval / forecast / task utility, calibration and failure rate.
2. **Information efficiency:** useful bits retained or recovered per stored byte and per update, with the task distribution stated.
3. **Temporal fidelity:** trajectory error, irregular-gap stability, recovery after bursts, and decision freshness.
4. **Resource cost:** memory footprint, bytes moved, allocation, CPU/GPU time, p50/p95/p99 latency, and physical joules only when directly measured.
5. **Robustness:** missing, delayed, duplicated, reordered, adversarial and out-of-distribution events.
6. **Operational integrity:** deterministic replay, versioned representations, provenance, resumability and fail-closed behavior.

A candidate may win in one regime and lose in another. Publish the frontier and the losing regimes; do not silently promote an architecture globally.

## Next qualification stages

1. Verify exact retrieval and random-query null behavior over a larger independent seed set.
2. Add paired confidence intervals and explicit bootstrap/randomization tests at the seed/corpus level; queries within one bundle must not be treated as independent replicates.
3. Compose work accounting with the existing temporal ledger (#13060–#13062), without creating a second ledger.
4. Compare representation families using #11145's algebraic-conformance contract (MAP, HRR/FHRR, GHRR, sparse, BinaryHV and ContinuousHV).
5. Add matched dynamical controls: frozen state, liquid-only, HDC-only, bounded iterative evolution and an independent high-accuracy reference where an equivalent trajectory is defined.
6. Only then test online updates, retention under new learning, consolidation/replay, capacity-versus-forgetting, and task-level utility.

The benchmark should be judged by a quality/cost/robustness frontier with matched information, labels, splits, seeds and system boundaries—not by vector width, a single top-1 score, or one unpaired timing.

## Research context

- Static VSA capacity is already being studied in #7772 and #11145. IBM's 2026 JAIR work studies capacity bounds for MAP-I, MAP-B and sparse-binary VSAs on set-membership and set-intersection tasks; those results motivate task-specific empirical curves but are not directly reproduced by this key/value recall protocol.
- Irregular-time input integration is already owned by #13031 and the NIALIM/DFA-CfN source audit by #13042. This example deliberately does not claim to solve time-varying-input integration.
- Temporal work instrumentation and decision freshness are owned by #13060–#13062 and #13091. This example reports only the simple neuron-local counters available from its crate and should later consume the canonical ledger.
- The wider irregular-time comparison should map to BITS/TIME-style time-series evaluation and NeuroBench's separation of algorithm and deployed-system tracks where the tasks genuinely match.
