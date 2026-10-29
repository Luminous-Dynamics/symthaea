# Symthaea Architecture Engineering Program v0.1

**Status:** research protocol frozen for review; architecture search NOT IMPLEMENTED, experiments NOT EXECUTED, candidates NOT QUALIFIED, NOT PASS. The matrix validator, paired-cohort evaluator, and known-optimum synthetic benchmark are authored; Rust compilation/tests remain unexecuted in this environment.  
**Tracking issue:** [#8888](https://github.com/Luminous-Dynamics/symthaea/issues/8888)  
**Prerequisite audit:** [#8308 — Liquid Broca learning signal and benchmark validity](https://github.com/Luminous-Dynamics/symthaea/issues/8308)  
**Related experiment:** [#8362 — context-local adaptation vs extra thought tokens](https://github.com/Luminous-Dynamics/symthaea/issues/8362)

## 1. Research question

Can Symthaea discover architecture choices that improve the quality–cost frontier for a named workload and hardware target, using fewer expensive experiments than standard search baselines?

This is deliberately narrower than “invent a better neural network.” A valid result must identify:
- **which workload** improved;
- **which baseline** was beaten;
- **under which training and inference budgets**;
- **on which exact hardware/software stack**;
- **with what uncertainty and failure modes**; and
- **whether the finding replicated on held-out data and independent runs**.

The current post-Transformer product direction in `docs/ROADMAP.md` is unchanged. This is an isolated research track: Transformers, state-space/recurrent models, and hybrids are competing candidates, not ideologies.

## 2. Paradigm shifts to test

### 2.1 Optimize a Pareto frontier, not a single “intelligence score”

Track task quality, training compute, prefill latency, decode latency, throughput, peak memory, and energy independently. A candidate dominates another only when it is no worse on every declared objective and strictly better on at least one. A weighted composite may be shown as a secondary view, but it must never replace the metric vector.

This avoids “faster because much smaller,” “higher score because much more compute,” and “higher Φ therefore better language model” confounds.

### 2.2 Make search hardware- and workload-conditioned

Architecture configurations are evaluated against a named workload profile: context-length distribution, prompt/output ratio, batch/concurrency, latency objective, memory limit, and target accelerator. A universal winner is not assumed. At minimum, keep separate profiles for:
- small/interactive local inference;
- long-context retrieval and state tracking; and
- throughput-oriented batched inference.

The search space must exclude candidates whose operators/kernels are unavailable on the target, unless kernel implementation cost is explicitly included as a research objective.

### 2.3 Treat search efficiency as a first-class benchmark

The scientific claim is not just “our search found a good model.” It is also “the Symthaea-guided policy found better candidates per trial, compute-hour, or unit cost than simpler search.”

Compare, at the same evaluation budget:
1. random search;
2. a conventional adaptive method (for example TPE/Bayesian optimization, if available);
3. the existing Symthaea architecture-search policy, after adapting it to measured task outcomes.

Report best-so-far quality, Pareto-front hypervolume, valid configurations, wall-clock/GPU-hours, and time-to-target after each evaluation. Run a synthetic response-surface control with known optima before interpreting search-policy differences on expensive model training. Include a label-shuffling negative control; if the search still “wins” after outcome labels are shuffled, the analysis is invalid.

The `symthaea-phi-search` crate may supply mutation/evolution infrastructure, but Φ is not the selection objective for language quality. Use Φ only as a secondary feature if a pre-registered experiment tests its incremental predictive value.

### 2.4 Search compositions with causal ablations

Treat each component as an intervention, not a marketing label. Use staged and factorial ablations for:
- hidden size, depth/width, MLP-to-attention parameter allocation, and GQA;
- full versus local/global attention patterns;
- recurrent/state-space layer substitutions;
- HDC-based associative retrieval as a sidecar; and
- context-local adaptation / fast state, only after the static path is valid.

The control is a competent dense Transformer. Keep some full-attention layers available in the grammar: recent expressiveness theory establishes that recurrent/linear-attention hybrids cannot replace full attention for every sequential composition task. A hybrid is a hypothesis to test, not the default winner.

For HDC, isolate a retrieval-augmentation path first. Do not replace token embeddings, attention, or the language loss with HDC similarity unless a separate experiment justifies it.

### 2.5 Make every claim evidence-carrying

Every run must record the exact source commit/tree, config, tokenizer, dataset/split digests, initialization/checkpoint identity, seed, optimizer/schedule, precision, training tokens and FLOPs (or a declared proxy), device, runtime/kernel versions, decoding settings, cost traces, per-example outputs, and failures.

Candidate generation and evaluation are separate authorities: Symthaea proposes; the independent evaluator computes task metrics and rejects invalid runs. Search workers must not read the locked final test set. Preserve failed runs and negative results rather than deleting them from aggregate summaries.

## 3. Literature and external references

1. Bian et al., **Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs**, ICLR 2026. The study evaluates more than 200 models from 80M to 3B parameters and reports architecture-aware choices that improve accuracy and inference throughput versus its specified baselines at matched training budget. This supports testing architecture-aware parameter allocation; it does not imply Symthaea can reproduce the gains.  
   https://proceedings.iclr.cc/paper_files/paper/2026/hash/dd2eb5250696753ea37141bbd89bb569-Abstract-Conference.html
2. Ye et al., **A Provable Expressiveness Hierarchy in Hybrid Linear-Full Attention**, ICML 2026. This argues for preserving full-attention controls on tasks requiring sequential function composition.  
   https://proceedings.mlr.press/v306/ye26h.html
3. Lahoti et al., **Mamba-3: Improved Sequence Modeling using State Space Principles**, ICLR 2026. The paper reports better retrieval/state tracking at 1.5B scale while explicitly noting practical hardware-utilization issues for linear models.  
   https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html
4. Yen et al., **HELMET: How to Evaluate Long-context Models Effectively and Thoroughly**, ICLR 2025. Its seven application-oriented categories show that context tasks can have distinct behavior and that needle-in-a-haystack alone is not sufficient.  
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/f5332c8273d02729730a9c24dec2135e-Abstract-Conference.html
5. Bai et al., **LongBench v2**, ACL 2025. A deeper long-context benchmark covering single-/multi-document QA, long in-context learning, dialogue history, repository understanding, and structured data.  
   https://aclanthology.org/2025.acl-long.183/
6. MLCommons, **MLPerf Client / Endpoints**. Useful measurement conventions for TTFT, output speed, end-to-end response time, and serving trade-offs. These system results must not be represented as architecture-only evidence unless the serving setup is controlled.  
   https://mlcommons.org/benchmarks/client/  
   https://docs.mlcommons.org/inference/

## 4. Candidate grammar (proposed, not implemented)

The machine-readable matrix is `docs/research/fixtures/ARCHITECTURE_ENGINEERING_CANDIDATES_V0_1.json`. It is intentionally all-null for measurements.

| ID | Candidate family | Primary question | Required control |
|---|---|---|---|
| AE-BASE-01 | Dense decoder, full attention | Can we reproduce the baseline loss and cost? | Fixed config, fixed data/tokenizer |
| AE-ALLOC-01 | Dense attention + MLP/GQA allocation sweep | Does architecture allocation improve quality/throughput at matched budget? | Random/default allocations, same training FLOPs |
| AE-LG-01 | Local attention with explicit global-attention layers | Does sparse locality improve long-context cost without harming global composition? | Dense full-attention model, same task/compute |
| AE-SSM-01 | Strong available recurrent/state-space baseline (prefer Mamba-3 only if a compatible checkpoint/kernel is actually available) | Where do recurrence/state tracking and hardware limits help or hurt? | Dense baseline, same workload/hardware; separately report training-path differences |
| AE-HYB-01 | Full-attention + recurrent/state-space layer mix | Is there a reproducible quality–cost frontier gain from selective full attention? | Pure dense and pure recurrent controls |
| AE-HDC-01 | HDC retrieval sidecar + unchanged language backbone | Does associative retrieval improve correct recall per added latency/memory? | Backbone alone; randomized/irrelevant retrieval negative control |
| AE-TTT-01 | Context-local adaptation / temporary fast state | Is adaptation more useful than extra generated thought tokens at equal total inference cost? | Static model and equal-budget extra-thought arm; follows #8362 |

Do not silently approximate an unavailable architecture with a differently named implementation. Mark it unavailable and preserve the reason.

## 5. Benchmark contract

### 5.1 Language-modeling quality

- Held-out negative log-likelihood (NLL) and perplexity on a frozen, documented corpus.
- Bits-per-byte where tokenizer differences would otherwise make perplexity comparison misleading.
- Primary task metric declared before the run; accuracy, exact match, or another task metric only when the task protocol is valid.
- Train/dev/test separation by document/source, not merely adjacent chunks, to reduce leakage.
- Report tokenizer, vocabulary, data filtering, deduplication, and any contamination limitations.

### 5.2 Long-context behavior

Use a small, executable early suite at the context lengths that fit the reference hardware, then increase lengths only after cost and correctness are stable. Prefer application-diverse HELMET / LongBench v2 subsets and controlled state-tracking or sequential-composition tasks. RULER-like synthetic probes are diagnostic, not a substitute for realistic evaluation.

Report score by length bucket, position of evidence, distractor count, evidence conflicts, retrieval versus reasoning, and failure type. A nominal context window or a successful single-needle example is not evidence of effective context use.

### 5.3 Reliability and robustness

- Repeated-run variance and confidence intervals.
- Distractor insertion, irrelevant/reordered evidence, conflicting facts with explicit source precedence, and distribution-shift splits.
- Calibration (for probability outputs), such as Brier score / ECE, where appropriate.
- NaN/divergence, invalid configuration, OOM, timeout, incomplete output, and evaluator failures remain visible and count toward operational yield.
- No claim of robustness from a synthetic suite alone.

### 5.4 Inference and resource cost

Measure separately:
- prefill latency and time-to-first-token (TTFT);
- per-output-token latency (TPOT) and p50/p95/p99;
- output tokens/s per stream and aggregate system throughput;
- peak host RAM and accelerator memory;
- training tokens, training FLOPs or a clearly stated estimate, wall-clock, retries, and failed-run compute;
- joules/token only when the measurement source is identified. Telemetry-derived board power is not wall-energy; label estimate versus calibrated direct measurement.

Fix the workload mix, prompt/output lengths, batch/concurrency, warm/cold state, precision, runtime, and kernel versions. Report a throughput–interactivity frontier, not just the best tokens/s datapoint.

## 6. Evaluation and promotion gates

### Gate 0 — source and metric validity

Do not spend architecture-search compute until the #8308 learning-signal and task-validity defects are resolved or fenced off from the experiment. Verify that the measured loss changes with predictions and that external benchmark implementations condition on the actual task input. Include a deterministic tiny overfit test and a deliberately broken-evaluator negative control.

### Gate 1 — baseline qualification

Freeze the dataset/tokenizer/config/source identity. Demonstrate a decreasing training loss on a tiny fixed subset, sane held-out NLL, reproducible scoring, and exact per-example accounting. Measure a baseline throughput/memory profile before proposing a candidate sweep. A smoke test is not a successful baseline reproduction.

### Gate 2 — bounded search screen

Begin with a budget that fits the available hardware and produces valid baseline training. Screen candidates with at least 3 seeds where training stochasticity matters. Compare search policies at the same trial and compute budget. Use cheaper fidelities only after checking that rankings predict higher-fidelity results; do not assume small-model winners transfer to larger scales.

### Gate 3 — confirmatory finalist

For promotion claims, use at least 5 independent training seeds where stochasticity applies, a locked evaluation set, paired bootstrap confidence intervals (or a suitable pre-registered equivalent), and correction/controls for selecting from many candidates. Publish raw per-example outcomes and the full candidate ledger.

### Proposed practical promotion thresholds

Pre-register the primary outcome before running experiments. A finalist advances only if it demonstrates at least one:
- **Quality win:** ≥1.0 absolute percentage point on a designated primary accuracy metric, or a statistically supported held-out NLL/BPB improvement at matched training compute; and no >0.5 percentage-point regression on designated guardrail tasks.
- **Efficiency win:** ≥20% higher measured throughput or ≥15% lower measured energy/token at statistically non-inferior quality (predeclared margin: ≤0.5 accuracy points or ≤1% relative NLL, where applicable).

These are proposed research bars, not results, not universal SOTA criteria, and not permission to ship. A long-context win, a low-power edge win, and a quality-at-equal-compute win are separate claims.

### Search-policy promotion threshold

The claim that Symthaea is a *better architecture engineer* needs its own result separate from a better architecture. Under equal wall-clock/compute budgets and the same candidate grammar, require either:
- at least **20% fewer full-fidelity candidate evaluations** to reach a preregistered quality/cost target than the better of random search and the conventional adaptive baseline; or
- at least **10% higher normalized Pareto hypervolume** at the same budget, using normalization bounds and reference point fixed before the search.

Search-policy promotion also requires at least 5 independent search seeds per policy across 2 preregistered workload profiles, uncertainty intervals over search runs, and a failed/negative label-shuffle control. If a threshold cannot be tested with the available compute, label the result preliminary instead of weakening it after seeing results.

## 7. First implementation sequence

1. Close the evaluation-integrity gaps under #8308 before relying on Liquid Broca language metrics.
2. Audit the actual Rust training/model path, supported attention/state-space kernels, GPU memory behavior, and license/checkpoint availability.
3. Authored the frozen-matrix Rust validator and unit tests in `crates/domains/symthaea-architecture-engineering`; Rust compilation/tests are still pending. Next: specify and validate provenance-bearing run receipts before implementing a search scheduler.
4. Establish a valid dense baseline plus one strong feasible recurrent/state-space comparator.
5. Reproduce the architecture-allocation result direction at small scale; publish a failed reproduction as a useful outcome.
6. Run search-policy ablations against random search using equal evaluation budgets and a known synthetic response surface.
7. Only then expand to local/global attention, HDC retrieval, or context-local adaptation.
8. Promote candidates based on held-out evidence and publish both winners and negative results.

The implementation should be Rust-first and use existing training frameworks/kernels where they work. Do not build a full training framework just to avoid using a supported one. Do not add production Python or a new JavaScript stack for orchestration if a small Rust CLI is sufficient.

## 8. Frozen candidate-matrix validator

The `symthaea-architecture-engineering` Rust crate now contains a fail-closed validator for the frozen v0.1 JSON fixture.

Run from the workspace root:

```bash
cargo run -p symthaea-architecture-engineering -- validate-matrix docs/research/fixtures/ARCHITECTURE_ENGINEERING_CANDIDATES_V0_1.json
cargo test -p symthaea-architecture-engineering
```

The validator checks the schema/version and canonical tracking issue, fixed candidate/search/negative-control counts, globally unique IDs, valid architecture-control references, `NOT_RUN` status, null measurement fields, an explicit locked-final-test policy, and HTTPS research references. Unit tests cover the committed fixture, a populated measurement, duplicate IDs, final-test leakage policy, and malformed JSON.

This is a **design-fixture integrity check only**. It does not validate training runs or scientific claims. The implementation and tests are authored, but no compiler/test result is claimed in this turn.

## 8. Paired-cohort evaluation module

The Rust crate now includes `paired_evaluation`, a deterministic evaluation primitive tracked by [#8911](https://github.com/Luminous-Dynamics/symthaea/issues/8911). Its aim is to prevent candidate search from rewarding easy query allocation, lucky single runs, or prompt-template quirks.

The module requires:
- a frozen candidate list with a configuration SHA-256 for every candidate;
- source commit, runtime/environment digest, run-config digest, dataset digest, and evaluator identity/digest;
- exact input digests, task family, prompt-template identity, metric bounds, and metric direction for every case;
- the same exact case × seed grid for every candidate, with at least three declared seeds;
- an explicit median or predeclared trimmed-mean policy;
- an output digest per trial and a canonical SHA-256 over the plan, trial outcomes, and derived summaries.

It emits per-case absolute scores, centered relative merit for search feedback, per-task-family/metric summaries, per-template/metric summaries, and an exact receipt digest. Relative merit is not the final quality score; raw/normalized absolute outcomes and all strata remain available. The helper `pareto_front` refuses incomplete metric points rather than pretending an unknown cost is zero, and rejects any comparison whose workload/hardware profile digests differ. Peak memory is an integer byte count.

Run from the workspace root:

```bash
cargo test -p symthaea-architecture-engineering
cargo run -p symthaea-architecture-engineering -- validate-matrix docs/research/fixtures/ARCHITECTURE_ENGINEERING_CANDIDATES_V0_1.json
cargo run -p symthaea-architecture-engineering -- evaluate-paired <evaluation-input.json>
```

The `evaluate-paired` input is a JSON envelope with a `plan` matching `EvaluationPlan` and an `observations` array matching `TrialObservation`. Use exact rendered-input and raw-output digests from the evaluator; do not use placeholder hashes for real experiments.

**Validation boundary:** eighteen unit tests are authored for paired scoring, difficulty-offset cancellation, robust aggregation, incomplete/duplicate observations, receipt determinism and Pareto trade-offs. They have not been compiled or run in this authoring environment. This module is not an LLM trainer, and its existence does not qualify any scientific result.

### External search-method findings

- [RAAS (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_RAAS_LLM_Agentic_System_Architecture_Search_with_GRPO_CVPR_2026_paper.html) uses cohorts of candidate workflows on identical queries and multiple execution trials to reduce query-difficulty and execution-noise confounding. We use these as baseline design patterns, not as a novelty claim.
- [SAGE-NAS (ICML 2026)](https://proceedings.mlr.press/v306/lin26ac.html) combines semantic search with a graph-based evaluator and zero-cost proxies; we should test whether cheap predictors actually rank higher-fidelity candidates on Symthaea's workloads before using them to prune.
- [The Evaluation Scaling Law (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.62/) reports that prompt-template diversity improves assessment reliability. Our case identity therefore includes prompt-template ID and summaries preserve per-template performance rather than hiding prompt sensitivity in one mean.
- [Data-quality scaling-law study (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6544c0b53cc1ea51d0c7086b536830a8-Abstract-Conference.html) finds data-quality interventions can change scaling behavior. Architecture-only claims therefore require frozen identical data/tokenizer/training conditions; data-quality interventions should be a separately designed factor, not an uncontrolled confound.

## 9. Known-optimum synthetic search benchmark

The `synthetic_search` module adds a deterministic, exhaustively enumerable 256-configuration space with two landscapes:
- `SmoothTradeoff`: a single global quality optimum with increasing resource costs;
- `Deceptive`: the same global optimum plus a planted local trap surrounded by lower-quality neighbors.

The benchmark compares random sampling, local search, and a feature-only proxy-ranked policy. Proxy modes include a noisy feature prior, shuffled proxy labels, and an explicitly anti-correlated prior. The proxy does not read the hidden task-quality objective. Every policy is charged the exact number of full candidate evaluations it receives.

Reported outcomes include:
- simple regret to the known global optimum;
- exact-optimum hit rate over independent seeds;
- recall of the exhaustive top-10 candidates and the exact multi-objective Pareto front;
- normalized two-objective quality/latency hypervolume;
- evaluations needed to reach the preregistered quality target; and
- run-to-run sample standard deviation.

The full known optimum and exact Pareto front come from exhaustive enumeration, independently of which candidates a search policy evaluates. The synthetic objective has intentionally artificial units and cannot predict language-model accuracy, training time, or GPU performance.

Run the full small synthetic comparison from the workspace root:

```bash
cargo test -p symthaea-architecture-engineering
cargo run -p symthaea-architecture-engineering -- synthetic-sweep smooth_tradeoff 20
cargo run -p symthaea-architecture-engineering -- synthetic-sweep deceptive 20
```

Each sweep emits JSON containing the oracle and aggregates at budgets 16, 32, 64, and 128 for random/local search and correlated/shuffled/anti-correlated proxy ranking. The output remains `SYNTHETIC_ONLY_NOT_MODEL_EVIDENCE`. Eleven synthetic unit tests are authored, including budget accounting, permutation invariance, the deceptive trap, proxy-mode controls, and aggregate integrity. They have not been compiled or executed here.

The next acceptance step is to run these commands on the exact branch head and inspect the raw JSON before using this benchmark to choose a real search policy. A win on this synthetic surface qualifies only the benchmark/search implementation for that surface, not a language architecture.

### Updated search-method findings

- [ZAPS (September 2026)](https://arxiv.org/abs/2609.14184) reports that zero-cost proxies can be strongly correlated and that naive proxy aggregation may compound shared errors. Its reported 52.3%/65.8% top-100 recovery at 200 evaluations belongs to its NAS-Bench-201 experiments, not Symthaea. This motivates explicit shuffled/anti-correlated proxy controls and measuring top-K recall under limited evaluation budgets.
- [LUMINA (March 2026)](https://www.microsoft.com/en-us/research/publication/lumina-llm-guided-gpu-architecture-exploration-via-bottleneck-analysis/) investigates LLM-guided GPU design-space exploration and reports Pareto-front gains on its own hardware-design problem. Symthaea should compare search efficiency and frontier quality in a small known-optimum environment before investing in an expensive real architecture search.
- [MLPerf Endpoints v1.0 (opened for submissions October 13, 2026)](https://mlcommons.org/2026/10/open-submissions-endpoints-v1-0/) is now the current industry reference for endpoint comparisons. It uses rolling submissions, reports the concurrency/interactivity/aggregate-throughput frontier, and adds mandatory provisioned-power reporting for the Standardized division. Its first v1.0 results were expected in November 2026, so as of this protocol date we should use the published v0.7 results as historical context and target the v1.0 protocol for future comparable submissions. For real model validation, use named profiles and separate first-token latency, per-user token rate, aggregate throughput, memory, and measured versus estimated energy.

## 10. Non-goals and claim ceiling

This design does not:
- implement a Transformer, state-space model, architecture scheduler, trainer, or benchmark runner;
- claim any proposed candidate is already available or faster;
- reproduce the ICLR 2026 results;
- establish a SOTA win or independent replication;
- equate Phi, HDC similarity, or internal coherence with language intelligence;
- overturn the post-Transformer roadmap; or
- authorize production promotion without exact-head verification and the gates above.

**Current disposition: DESIGN ONLY / NOT EXECUTED / NOT QUALIFIED / NOT PASS.**
