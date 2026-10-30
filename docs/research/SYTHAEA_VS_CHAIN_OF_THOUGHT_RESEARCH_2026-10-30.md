# Symthaea vs explicit and latent chain-of-thought reasoning

**Status:** research plan and source-audit correction, 2026-10-30. No comparative benchmark run or SOTA claim is made in this document.

## Executive assessment

Symthaea is not currently a general-purpose replacement for a frontier reasoning LLM. It is a heterogeneous cognitive architecture with at least three distinct reasoning/generation paths:

1. Broca's native HDC/CfC autoregressive decoder;
2. the separate typed `ConsciousReasoningEngine` for epistemic assessment, action planning, tool gating and counterfactual analysis;
3. optional external LLM backends in the wider repository.

These should not be collapsed into a single model comparison. Broca's native decoder is not initialized from a pretrained frontier language model. It starts from Genesis-seeded HDC token embeddings and network parameters, with a minimal or 4K tokenizer and task-specific training pairs. Neither the current CPU path nor the optional Candle GPU path performs temporal backpropagation through time. The broader reasoning engine does not receive an arbitrary natural-language problem and produce a general proof transcript; it consumes a typed `ReasoningContext` containing theory metrics, available actions, tool descriptors, code context and simulation inputs.

The defensible claim today is **architectural differentiation**, not superior general reasoning.

## What chain-of-thought actually provides

Explicit chain-of-thought (CoT) is an inference-time method: a language model emits intermediate text before its answer. It is valuable because it supplies serial computation and a visible surface trace. The original CoT work showed major gains on arithmetic, commonsense and symbolic reasoning tasks for sufficiently large language models.

But a generated explanation is not automatically a faithful record of the computation that caused the answer. Turpin et al. showed that models can rationalize answers driven by biased input features without disclosing those features. Anthropic's intervention work found task-dependent reliance on the displayed CoT and reported that faithfulness does not increase monotonically with model capability. The 2025 study *Reasoning Models Don't Always Say What They Think* likewise found that reasoning-model CoTs disclose some influential hints, but often fail to disclose them.

Recent latent-reasoning research creates a stronger comparison than “text CoT versus HDC”:

- **LOTUS (June 2026)** supervises recurrent latent blocks with gold CoT-step tokens. It reports matching explicit-CoT performance at 3B scale in its tested settings and 2.5–6.9× lower thought-phase latency.
- **Latent Recurrent Thoughts (Findings of EMNLP 2026)** uses a small recurrent latent reasoner in front of a frozen LLM decoder. Its experiments report gains over non-thinking-mode CoT prompting on the same decoder, prompt, data and training budget.
- **GRACE (EMNLP 2026)** provides human-annotated, step-level faithfulness labels for context-grounded reasoning, with separate inference and grounding error tracks.
- **LongCoT (ICML 2026)** contains 2,500 long-horizon problems whose solution paths require tens to hundreds of thousands of reasoning tokens. At publication, GPT-5.2 reached 9.8% and Gemini 3 Pro 6.1%, showing that long-horizon dependency management remains difficult even for frontier systems.

The 2026 position paper *LLM Reasoning Is Latent, Not the Chain of Thought* argues for treating latent-state trajectories as a default object of study. This is a useful research hypothesis, not settled consensus or a result about Symthaea.

## Source audit: what Symthaea implements

### A. Broca's native HDC/CfC language path

Relevant sources:
- `crates/domains/symthaea-broca/src/controller.rs`
- `crates/domains/symthaea-broca/src/generator.rs`
- `crates/domains/symthaea-broca/src/encoder.rs`
- `crates/domains/symthaea-broca/src/thought_chunk.rs`
- `crates/domains/symthaea-broca/src/training.rs`

The controller encodes the 43-channel thought representation into a 16,384D ContinuousHV. Its token-step input is composed as:

`thought_hv ⊗ token_embedding(previous_token) ⊗ permute(position_base, position)`

It then evolves the HDC-LTC/CfC network in closed form, normalizes the output state, and computes token logits through weight-tied similarity against the token embeddings. This is a recurrent autoregressive generator; it is not a Transformer decoder.

Important differences from frontier LLMs:

- Token embeddings and network state are Genesis-seeded, not a large pretrained language representation.
- The documented tokenizer options include a 208-token minimal vocabulary and a 4K vocabulary. These are not comparable with modern LLM vocabularies and learned representations.
- The training corpus is formed from `(ThoughtChannels, target_text)` pairs, including outputs collected from an existing LLM organ. Teacher text is a target, not evidence that a teacher's intermediate reasoning is correct.
- `ThoughtChunk` is a semantic-output container: an HDC vector, output kind, optional target string/token span, and scalar `psi`, `confidence`, and spectral entropy.
- The current chunk-to-program conversion creates a linear predecessor chain. It is not a dependency-aware proof DAG.
- `psi` is a system control/consciousness-related metric. The `confidence` field is a caller-supplied scalar. Neither is currently a calibrated probability that a proposition is true.

#### Temporal credit assignment: a real constraint

An audit of `LanguageController::backward_step` found that it reads the current network state, reconstructs the current input, and forms local targets. It does not retain the activation history or propagate gradients from token position t back to t−1. The approximate error signal does propagate across layers; what is absent is temporal credit assignment.

Therefore Broca's current CPU and optional GPU trainers use teacher-forced token losses plus a local per-step recurrent update. The GPU path batches operations on the selected device and propagates an approximate signal through layers, but `GpuCfcNetwork::backward_and_update` still constructs local targets from the current state; it does not retain prior token states or propagate gradients back across time. Its cell backward path also uses a simplified local evolution derivative rather than the full gated forward transition. The `bptt_window` field is a legacy name for the maximum teacher-forced positions processed per pair, not a BPTT unroll window. The module, configuration, GPU implementation and training-test labels have now been corrected.

The repository already contains the recommended next experiment:
- `docs/TEMPORAL_CREDIT_ASSIGNMENT_STEP4_DESIGN_2026-07-31.md`
- `docs/TEMPORAL_BENCHMARK_V2_PREREGISTRATION_2026-07-31.md`
- `crates/domains/symthaea-broca/examples/broca_state_reset_ablation.rs`
- `crates/core/symthaea-evidence-plane/src/task_validator.rs`
- `crates/domains/symthaea-psych-bench/src/benchmarks/temporal/`

The frozen pre-registration requires a validated task, a mechanically guaranteed Static negative control, a tuned EMA baseline and explicit compute accounting. It sets a +0.10 absolute ambiguous-point-accuracy gate over the best non-Static baseline, paired bootstrap 95% CI excluding zero, at least eight disjoint seeds, and a 3× EMA-compute ceiling per correct ambiguous prediction. It also records that the previous HDC-LTC coupling showed no demonstrated predictive superiority to EMA at a 4–5× compute premium. This prior must be tested, not explained away.

The first action should be to run the existing `NORMAL` versus `RESET` Broca state ablation on the canonical held-out suite. If accumulated recurrent state does not help, adding temporal-credit machinery before fixing the dynamics/representation is not justified. If it helps, the already designed eligibility-trace/e-prop intervention is the lower-blast-radius first step; a genuine activation-tape BPTT implementation remains a possible later alternative.

### B. ConsciousReasoningEngine

Relevant sources:
- `src/consciousness/reasoning_engine/mod.rs`
- `src/consciousness/reasoning_engine/types.rs`
- `src/consciousness/temporal_planning/`
- `src/consciousness/counterfactual/`
- `src/consciousness/epistemic_conflict/`

This subsystem is a typed decision/planning controller. It detects theory conflicts, estimates reliability, computes an evidence-modulated effective-phi signal, decides whether simulation is worth its cost, optionally plans with MCTS, gates tool actions and runs counterfactual analysis when identified. It emits structured `ReasoningEvent` telemetry.

It is not a general natural-language CoT model:
- `ReasoningContext` is typed, not a raw language query.
- MCTS acts over supplied actions and a supplied/default simulation state.
- Counterfactual analysis depends on a structured causal DAG and reference-harness outcomes.
- An event log is not a faithful free-form thought transcript.

A source/documentation mismatch was found and corrected in this branch: comments claimed Tier 1 ran micro-MCTS, but the executable path explicitly excludes Tier 1 with `budget.tier != BudgetTier::Tier1`. An existing test also asserts that Tier 1 does not plan. The module documentation now says Tier 1 currently performs reduced-budget epistemic assessment without MCTS; `MctsConfig::tier1()` exists but is not exercised by this path. Do not claim a Tier 1 MCTS capability until the path and its tests are deliberately changed.

### C. The unified CognitiveLoop

`crates/domains/symthaea-broca/src/cognitive_loop.rs` orchestrates vision telemetry, semantic monologue, program nodes and optional Geodesic synthesis. Its Broca monologue path is feature-gated on `mamba-cpu`; without that feature it emits an empty default monologue. This orchestration example is not itself evidence of end-to-end natural-language reasoning quality.

## The paradigm shift: latent cognition plus explicit proof

Do not choose between a private latent trajectory and a public textual CoT. Use two coordinated representations with different jobs.

### 1. Latent recurrent workspace

Use HDC/CfC state for compact, iterative computation. The latent workspace should encode:
- current task state and unresolved goals;
- active hypotheses and alternative branches;
- temporal context and causal state;
- uncertainty and resource budget;
- references to evidence/proof nodes, not unchecked semantic payloads.

The latent trajectory should be learned and evaluated, not assigned a high `psi` score and assumed correct.

### 2. Explicit typed proof workspace

Use a graph of typed facts, rules, dependencies, actions and verification receipts. Broca's emerging derivation-witness work is a natural foundation. Every user-visible factual conclusion should be traceable to source premises, a rule version, and an independently checkable result where possible.

A proof witness is not required to reproduce every internal activation. It should capture the verifiable obligations needed to justify a claim.

### 3. Query-driven refinement

An answer plan becomes a set of proof obligations. At each recurrent step:
- choose the most decision-relevant unresolved obligation;
- retrieve or compute the missing evidence;
- update the latent state;
- update the proof graph;
- measure uncertainty, contradiction and expected value of further compute;
- stop when proof obligations are discharged, the budget is exhausted, or the system should abstain.

This is better than endlessly generating token-level rationales. It also lets Symthaea allocate extra compute to the parts of the problem that actually remain uncertain.

### 4. Train intermediate state transitions, not just prose

Use LLMs as teachers for candidate decompositions and language realization, but never assume their CoT is ground truth. Distill:
- state before/after each valid operation;
- task-specific latent thinking states;
- typed rule applications and proof edges;
- explicit failure, contradiction and abstention signals.

The LRT/LOTUS research suggests that supervised latent thinking states can preserve task-relevant structure without decoding every step into user-facing text. Symthaea can test a CfC/HDC equivalent, with a proof workspace as an external audit channel.

### 5. Preserve a readable explanation as a separate product

Generate a short explanation *from the verified proof graph*, not as a raw transcription of latent state. If no proof is available, say so. If a readable explanation is requested, render selected verified steps and citations. Do not present the generated explanation as a transcript of the hidden computation unless an intervention study has established that claim.

## Evaluation plan: separate capability from representation

A fair comparison has two tracks.

### Track A — Same decoder, different reasoning substrate

Use the same frozen LLM decoder, tokenizer, prompt, output format and inference settings for:
1. direct-answer baseline;
2. explicit textual CoT;
3. explicit CoT plus independent verifier;
4. Symthaea structured planner/proof workspace, with no textual CoT exposed to the answerer;
5. Symthaea latent recurrent refinement plus proof workspace;
6. hybrid Symthaea state/proof planner plus the same decoder.

This isolates the quality of the reasoning substrate more cleanly than comparing a small native Broca generator to a frontier LLM end-to-end.

### Track B — Native system comparison

Evaluate Broca-native generation and the broader ReasoningEngine as separate systems. Report their own model size, training corpus, tokenizer, context, hardware, features, compute, latency, accuracy and calibration. Do not combine them into a single Symthaea score or compare unlike model classes as though they had equivalent pretraining.

### Benchmark portfolio

| Benchmark | What it measures | Use |
|---|---|---|
| **LongCoT** (ICML 2026) | Long-horizon dependency graphs and verifiable final answers across chemistry, math, CS, chess and logic | Primary stress test for long-range proof/state management |
| **GRACE** (EMNLP 2026) | Human-annotated step-level context faithfulness; inference and grounding errors | Check whether explanations and proof steps actually follow evidence |
| **MemDerive** | Derived state from evolving events/rules; intermediate DAGs and supporting events | Query-driven premise gathering and incremental truth maintenance |
| **MiniF2F / Lean proofs** | Formal theorem proof and machine-checkable validity | Hard correctness gate on formal tasks |
| **Temporal Benchmark V2** | Ambiguous next-state decisions that demonstrably require history | Broca/CfC temporal-credit experiment |
| **MemCalib / COLLABMEM / MemEvoBench** | Memory use calibration, multi-agent collaboration, and memory corruption | Long-term context governance, calibrated influence and agent boundaries |

### Required metrics

Report all metrics by task family and difficulty, with paired confidence intervals:

- exact final-answer accuracy;
- proof/witness precision, recall and F1;
- percentage of proof steps that are valid and source-supported;
- GRACE-style step grounding and inference error rates;
- counterfactual sensitivity: remove/change one premise and test whether only dependent conclusions change;
- abstention accuracy and coverage;
- Brier score, expected calibration error and risk-coverage curve;
- token count and context use;
- wall-clock p50/p95 latency, accelerator time, peak memory and energy where measurable;
- write, retrieval and persistence cost;
- performance under distractors, delayed facts, reordered history, rule updates and long dependency chains.

A single aggregate score must not conceal a failure in proof validity, grounding or access control.

## Proposed research targets (hypotheses, not achieved results)

1. **Hybrid parity:** with the same LLM decoder and matched answer budget, match explicit CoT accuracy within 2 percentage points while reducing total inference tokens or latency by at least 30%.
2. **Faithfulness:** at least 90% source-supported proof steps on a held-out GRACE-derived evaluation, with every incorrect derivation counted as an error even when the final answer happens to be right.
3. **Formal validity:** 100% of emitted proofs accepted by the independent formal checker on formalizable tasks; abstain rather than fabricate a proof.
4. **Long-horizon:** establish a reproducible LongCoT baseline first, then target at least 5 absolute points above the reproduced matched baseline without increasing total compute. Report the official score and the cost curve.
5. **Temporal learning:** obey the already frozen Temporal Benchmark V2 gates. Do not relax its thresholds or replace its Static/EMA/PermutationVsa arms after looking at results.
6. **Safety:** zero unauthorized tool actions and zero unauthorized evidence materializations in adversarial tests.

The first two targets apply to the hybrid same-decoder track. They are not predictions for the current native Broca generator.

## Experiment sequence

1. **Audit and baseline freeze.** Record exact source SHA, feature flags, checkpoint digest, dataset hash, tokenizer, hardware and model settings. Fix the current stale BPTT/Tier-1 documentation before measurement (done in this branch).
2. **Run existing Broca NORMAL-vs-RESET ablation.** Use the canonical held-out suite. Publish both means, per-case deltas, non-finite skips, checkpoint SHA and hardware. Do not implement temporal traces before this test.
3. **Run Temporal Benchmark V2 task validation first.** It must pass each family at its intended history depth; its Static negative control must be mechanically chance-level on ambiguous points.
4. **Reproduce hard baselines.** Tuned EMA bank and permutation VSA must be included. A 4–5× compute premium without an accuracy win is a loss, not an architectural success.
5. **Only then implement eligibility traces.** Use the existing Step 4 design; preserve cross-layer credit and instrument trace updates. Compare HdcLtc+Traces against HdcLtc, EMA and PermutationVsa under the frozen gates.
6. **Build the proof-aware hybrid.** Feed Symthaea's typed state/proof graph to the same frozen decoder used in CoT baselines. Verify each source/premise independently.
7. **Run LongCoT and GRACE.** Maintain a held-out split; publish all metrics, errors and inference budgets. Add counterfactual step interventions to assess faithfulness.
8. **Only make narrow SOTA claims.** State exact benchmark version, split, model, prompt, compute and confidence interval. If no matched comparison exists, call it a research result, not SOTA.

## External references

- Wei et al., [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903), 2022.
- Turpin et al., [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388), NeurIPS 2023.
- Anthropic, [Measuring Faithfulness in Chain-of-Thought Reasoning](https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning), 2023.
- Chen et al., [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410), 2025.
- Fan et al., [Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers](https://www.microsoft.com/en-us/research/publication/bridging-the-gap-between-latent-and-explicit-reasoning-with-looped-transformers/), June 2026.
- Chen & Fu, [Latent Recurrent Thoughts](https://aclanthology.org/2026.findings-emnlp.578/), Findings of EMNLP 2026.
- Pham et al., [GRACE: Step-Level Benchmark for Faithful Reasoning over Context](https://aclanthology.org/2026.emnlp-main.1582/), EMNLP 2026.
- Motwani et al., [LongCoT: Benchmarking Long-Horizon Chain-of-Thought Reasoning](https://proceedings.mlr.press/v306/motwani26a.html), ICML 2026.
- Wang, [LLM Reasoning Is Latent, Not the Chain of Thought](https://arxiv.org/abs/2604.15726), 2026 position paper.

## Non-claims

This document does not claim Symthaea beats frontier LLMs, that HDC/CfC is inherently superior to Transformers, that `psi` is answer correctness, or that an explicit proof witness reveals the full latent computation. It establishes a source-audited comparison and a falsifiable experiment sequence.
