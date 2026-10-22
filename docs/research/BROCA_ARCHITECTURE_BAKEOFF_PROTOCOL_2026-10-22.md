# Liquid Broca Architecture Bake-off Protocol

**Status:** preregistration proposal; no experiments reported  
**Date:** 2026-10-22  
**Scope:** Symthaea's Liquid Broca / Mamba + HDC + temporal projection + CfC/LTC pipeline

This document freezes the minimum comparison contract before architecture results are observed. Thresholds are practical engineering targets, not claims about current capability or universal definitions of SOTA.

## 1. Research question

Does adding associative thought projection, temporal projection, liquid-state control, or logit intervention improve a defined language-generation task at equal or lower compute cost than a simple recurrent language baseline?

A mechanism is not justified by its biological, cognitive, topological, or consciousness-related name. It must improve a declared metric, have a reproducible effect, and justify its runtime and operational cost.

## 2. Falsifiable architecture hypotheses

| Arm | Configuration | Question |
|---|---|---|
| A | Pretrained Mamba-130M, no Broca projection/control | Reference language substrate |
| B | A + fixed HDC-to-SSM projection | Does thought conditioning help at all? |
| C | B + trained projection | Does learned projection beat fixed projection? |
| D | C + temporal projection | Does chunked temporal conditioning help versus one vector? |
| E | D + CfC/LTC bottleneck | Does slower adaptive state improve outcomes? |
| F | E + per-token Δ/B modulation | Does dynamic state modulation add value? |
| G | F with semantic attraction, gating, and veto ablated individually | Which intervention, if any, helps? |
| H | Sparse explicit HDC-keyed memory with bounded retrieval, using A as the language substrate | Does explicit associative retrieval outperform compressing all thought through the projection? |
| I | Budget-matched Transformer and, where practical, Mamba-2/3, RWKV-7, and xLSTM baselines | Is the effect specific to Broca or a general sequence-model property? |
| J | Context-specific training of a small temporary adapter/state versus spending the same compute on extra generated thought tokens | Is a targeted update a better use of inference compute than more open-ended rehearsal for a given long-context task? |

Do not use only A-vs-all-features-on. Report the individual effects and selected interaction terms. When full factorial evaluation is too expensive, preregister the smaller fractional-factorial design before runs.

For every comparison, disclose checkpoint source, model parameters, trainable parameter count, training tokens/steps, data, tokenizer, precision, batch/sequence length, and any difference in initialization or tuning budget. If those differ, label the comparison as a system comparison rather than an architecture-only comparison.

## 3. Evaluation gates

### Gate 0 — validity of the training signal

- Teacher-forced loss must score the actual target under logits conditioned on the correct prior tokens.
- Projection gradients must use the actual projected context and the full teacher-forced prefix, with the final target scored only after every preceding target token has updated recurrent state.
- Backpropagate from the SSM output gradient through the actual forward projection graph: HDC bottleneck, LayerNorm, activations, optional deep bottleneck, temporal chunk projection, adapter, attention and positional encoding as applicable. Do not pass target-token hidden states where projection activations are expected.
- Apply projection updates after scoring the complete teacher-forced target sequence so weights remain fixed while its per-token losses are accumulated.
- A missing gradient, invalid target, malformed context, or non-finite loss must fail closed; it must not be silently replaced by a zero vector or a success-shaped metric.
- Add finite-difference or independent-reference tests for both the Mamba context-anchor gradient and each projection Jacobian (including temporal attention where enabled). Contract tests alone do not establish gradient correctness.
- Empty or invalid evaluation splits must return an error / unavailable result, never a numeric score that looks like success.

### Gate 1 — held-out language quality

Primary metric: mean teacher-forced negative log-likelihood (NLL) per target token on a fixed held-out canonical thought-to-text set, supplemented by bits per byte where tokenizer differences make token perplexity incomparable.

Secondary metrics: token top-1/top-k accuracy, calibration (including Brier score or ECE with binning details), length-normalized target fidelity, unsupported-claim rate, and blinded human preference.

Standard tasks such as LAMBADA and HellaSwag are secondary cross-checks. Reuse a documented implementation such as EleutherAI's lm-evaluation-harness and pin the task revision and tokenization/scoring configuration. Do not compare generation from an intent index against a benchmark answer and call it task accuracy.

### Gate 2 — memory, state tracking, and long context

Use a mixture of controlled diagnostic tasks and application-oriented long-context tests.

1. **Controlled recurrent state tests:** retrieve a fact after lags of 128, 512, 2,048, 8,192 and longer tokens where supported; distinguish update-after-conflict from stale recall; vary distractor count, fact collisions, write frequency, and query position. Score accuracy by lag and report state/RAM growth.
2. **Multi-hop / realistic long-context tests:** select compatible tasks from HELMET and/or HELM Long Context (for example retrieval QA, multi-hop QA, summarization, narrative multiple-choice and multi-round co-reference). Respect the tested model's context limit and exact task format.
3. **Semantic continuity:** generate from structured intents and measure preservation of required entities, relations, constraints, and uncertainty. Score a blinded, pre-defined rubric; separately count unsupported additions and omissions.
4. **Robustness:** repeat with reordered evidence, paraphrased queries, distractors, contradictions, and irrelevant high-salience text.

A synthetic needle-in-a-haystack score is a diagnostic only. The HELMET study found that simple needle-in-a-haystack performance is a poor proxy for downstream long-context quality; complex tasks and application diversity matter.

### Gate 3 — practical cost and reliability

Record at least:

- prompt/prefill latency and per-token decode p50/p95;
- tokens/second under batch size 1 and the separately stated batch throughput regime;
- peak RSS and VRAM, parameter memory, recurrent state size, and memory growth as context grows;
- energy per generated token if trustworthy device-level measurement is available;
- failures, timeouts, non-finite values, EOS behavior, and sensitivity to seed / prompt variation.

Specify hardware, power mode, GPU/CPU driver/runtime, compiler flags, Cargo features, quantization, thread count, and thermal conditions. Compare on the same physical machine where possible. Do not combine figures measured on different hardware into a single speedup claim.

## 4. Context-compute allocation experiment

Do not assume that increasing free-form thought tokens is the best way to spend inference compute. Include these alternatives on the same held-out long-context tasks:

1. **Static:** no test-time parameter/state update; answer using the original recurrent state and available context.
2. **More rehearsal:** spend an explicitly measured additional token/FLOP budget on internal analysis or candidate generation.
3. **Targeted update:** update only a small temporary adapter or context-local state using unlabeled context or a permitted support prefix; never use the answer key or held-out target in the update.
4. **Associative retrieval:** use bounded HDC-keyed memory to select relevant records, then answer with the same language backbone.
5. **Hybrid:** targeted retrieval plus a small context-local update, measured separately from the cost of producing the answer.

Match or report the actual total budget (preprocessing, update, retrieval, answer generation, memory, and latency), rather than comparing only decode tokens. Test on long-context QA, multi-hop retrieval, summarization, conflict resolution and complex instruction following. Include a no-update control and a random/unrelated-context update as negative controls. Any temporary update must be reset between examples unless cross-example learning is the explicit experiment; otherwise leakage invalidates the result.

This experiment is motivated by *Let's (not) just put things in Context: Test-Time Training for Long-context LLMs* (ICLR 2026), which reports substantial gains from targeted context-specific updates in its tested Qwen3-4B settings on subsets of LongBench-v2 and ZeroScrolls. Those results are evidence that the allocation of test-time compute is worth testing; they are not evidence that the same method will work for Liquid Broca. See https://proceedings.iclr.cc/paper_files/paper/2026/hash/b619cd6dcc986856b8a8da2b08d89396-Abstract-Conference.html.

## 5. Provisional acceptance thresholds

Finalize these thresholds before evaluation begins.

- **Quality win:** at least 1% relative NLL improvement with a paired 95% confidence interval excluding zero, or at least a 2-percentage-point increase on the preregistered Broca task with a paired confidence interval excluding zero.
- **Efficiency win:** at least 20% lower p95 latency, peak memory, or energy per token while quality remains within a preregistered non-inferiority margin.
- **Component inclusion:** the primary metric improves sufficiently to pass the chosen threshold, or a declared secondary capability improves without material regression in primary quality, calibration, or safety.
- **Replication:** at least five independent seeds for stochastic training/decoding, plus paired bootstrap confidence intervals over evaluation examples. Report all seeds, not only the best one.
- **No false leaders:** no task-invalid result, skipped run, incomplete dataset, or missing checkpoint can appear as a numeric leaderboard entry.

Do not change thresholds after seeing outcomes. Any exploratory post-hoc analysis must be labelled separately.

## 6. Evaluation receipt

Every run must write a machine-readable receipt including:

- git repository, branch and exact commit SHA; clean/dirty status and patch digest;
- full serialized model and Broca configuration, enabled/disabled feature list, Cargo features, compiler/runtime versions;
- model/checkpoint file hashes, tokenizer revision/hash, data source/revision/license and dataset digest;
- named split, per-example identifiers and scores, target-token counts, aggregate formulas, exclusions and failure reasons;
- seed, decoding options, prompt/template format, context length, truncation and normalization rules;
- hardware identity, CPU/GPU, driver/runtime, precision, thread count, memory and latency methodology;
- command line, timestamps, exit status, stderr/log digest, and whether any result was skipped or recovered from a corrupt/mismatched checkpoint.

The receipt should permit an independent person to answer: exactly what code ran, on what data and checkpoint, using what settings, and why is this number the reported result?

## 7. Pareto selection rule

Do not choose a single winner from an opaque weighted score. Plot quality versus latency and memory, with uncertainty intervals; identify Pareto-dominated configurations. Select a default only from configurations that clear the quality/reliability gates, then choose according to the actual deployment budget.

A mechanism that increases quality slightly but doubles latency may be useful for a high-value offline task, while being inappropriate for edge generation. Preserve those distinctions.

## 8. Industry and research baselines

Use the following papers and evaluation frameworks as comparison references, not as directly transferable score claims:

- Mamba-3, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html
- RWKV-7 "Goose": https://arxiv.org/abs/2503.14456
- xLSTM 7B, ICML 2025: https://proceedings.mlr.press/v267/beck25b.html
- EleutherAI lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- HELMET long-context benchmark: https://arxiv.org/abs/2410.02694 (seven application-centric categories; the ICLR 2025 paper reports weak correlation between simple needle-in-a-haystack performance and downstream long-context quality).
- HELM Long Context leaderboard and transparent run details: https://crfm.stanford.edu/helm/long-context/latest/
- Context-specific test-time training (ICLR 2026): https://proceedings.iclr.cc/paper_files/paper/2026/hash/b619cd6dcc986856b8a8da2b08d89396-Abstract-Conference.html

Published claims use different scales, training data, hardware, and inference stacks. The fairest achievable comparison here is against local baselines run under matching budgets; any broader industry comparison must explicitly state incompatibilities.

## 9. What would count as a paradigm-level win?

The most interesting prospective result is not "all features enabled wins." It is a compact design where:

1. the pretrained recurrent model remains a strong language substrate;
2. explicit HDC-addressed memory improves targeted retrieval and semantic persistence beyond latent projection alone;
3. the controller intervenes only when an observed condition predicts value;
4. removing a mechanism predictably removes the capability it provides;
5. the resulting system lands on a better quality/latency/memory Pareto frontier than simpler alternatives.

This remains a falsifiable hypothesis. No architecture win or state-of-the-art claim is implied by this protocol.
