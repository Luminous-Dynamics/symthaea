# Broca Semantic Token Initialization Experiment — 2026-10-21

## Purpose

Test whether lexically grounded, compositional HDC token initialization improves Broca's sample efficiency or held-out semantic generalization relative to genesis-seeded token vectors.

This is an **ablation hypothesis, not a result**. The default stays unchanged (enable_semantic_token_embeddings = false) until paired, exact-head evidence supports changing it.

## Paradigm shift

The conventional baseline treats each token ID as its own seed-addressed vector and asks the sequence learner to discover useful relationships. The experimental initializer injects a small, inspectable prior:

- direct lexical decomposition from the existing LexicalGrounding;
- conservative prefix/suffix reduction, admitted only when the reduced stem resolves to a grounded lexical entry;
- a genesis-seeded fallback for unknown tokens;
- configurable semantic/genesis blending for admitted entries.

This does **not** claim that Natural Semantic Metalanguage provides a complete or universally accepted semantic ontology. The HDC vector is a compact prior; it is not a parse tree, a factual source, or proof of comprehension.

## Existing research and external baselines

1. Waldis et al., *Holmes: A Benchmark to Assess the Linguistic Competence of Language Models*, TACL 2024. Holmes spans morphology, syntax, semantics, reasoning, and discourse and specifically distinguishes linguistic probes from instruction-following performance: https://aclanthology.org/2024.tacl-1.88/
2. Ismayilzada et al., *Evaluating Morphological Compositional Generalization in Large Language Models*, NAACL 2025. In Turkish and Finnish experiments, models' generalization weakened on novel roots and as morphological complexity increased: https://aclanthology.org/2025.naacl-long.59/
3. EleutherAI LM Evaluation Harness, a reusable framework with standard academic tasks and multiple model backends: https://github.com/EleutherAI/lm-evaluation-harness
4. HELM's maintainers report that the original framework entered maintenance mode on 2026-06-01. Its published taxonomy and transparent artifact model remain useful references, but it should not be the sole living baseline: https://github.com/stanford-crfm/helm

These sources motivate test design. They provide no evidence that this experiment improves Symthaea.

## Implementation boundary

- SemanticQuarkBasis is now exposed as a Broca module and has an opt-in controller initialization path.
- The normal controller constructor and serialized older BrocaConfig records retain the genesis-seeded default.
- The semantic initializer uses the same tokenizer supplied to Broca; its embedding row count is checked against vocab_size.
- The GPU/candle cache is constructed from the chosen initial embedding table rather than first allocating a second random table.
- Morphological heuristics alone no longer authorize a semantic decomposition in production embedding paths. A candidate must reach a lexically grounded stem within a fixed search bound.
- Coverage counters must account for every tokenizer row: direct + grounded morphology + fallback = total.

Checkpoint loading should restore checkpoint embeddings verbatim; the experiment is intended for newly initialized models and controlled comparisons, not for silently mutating a trained checkpoint.

## Initial within-repository ablation

broca_fusion_bench now has three relevant arms:

1. **Genesis token baseline:** existing initialization, semantic token initialization disabled.
2. **Semantic initialization only:** change initialization while compositional-logit refinement and algebraic gating remain off.
3. **Semantic initialization + comp/alg:** assess a candidate interaction separately from the isolated effect.

The benchmark trains each arm independently with the same training split, epoch count, optimizer configuration, and seed domain when --epochs N is supplied. It evaluates on a separate small held-out phrase split. Without --epochs, results are initialization-only smoke observations and must not be described as trained-model performance.

The phrase split is intentionally small. It validates the experimental wiring and can reveal obvious regressions; it is **not** sufficient to establish SOTA, broad semantic fidelity, or robust statistical generalization.

## Proposed evaluation ladder

### Gate A — implementation invariants

Required for every exact head:

- all embedding rows match the tokenizer vocabulary and HDC dimension;
- coverage counters sum exactly to the number of token rows;
- identical genesis, vocabulary, and configuration produce repeatable initialized vectors;
- absent/unknown lexemes use the documented fallback;
- false-positive lexical controls such as unicorn, disaster, king, thing, and wonder do not obtain semantics from accidental affix spelling;
- positive morphology controls preserve the grounded root's primes when a supported affix is applied;
- equivalent lexical prime sequences are near-identical, while a multi-pair contrastive probe separates them from opposite/contrastive concepts;
- non-finite genesis-blend weights fall back to the documented default instead of propagating non-finite embedding values;
- the training and held-out smoke fixtures have intent labels and epistemic ordinals aligned with their intended targets (channel 4 is uncertainty, not answer/certainty);
- legacy configuration deserialization keeps semantic initialization disabled unless explicitly enabled;
- checkpoint load/save and GPU cache paths remain aligned with the exact token IDs.

### Gate B — a real compositional corpus

Before making any capability claim, build a larger Rust-native or language-neutral fixture over typed semantic programs and surface forms. Use synthetic known-answer generation only for development; keep confirmation cases sealed before tuning.

Suggested strata:

- negation and polarity;
- quantifiers and cardinality;
- temporal order and aspect;
- subject/object role reversal;
- source attribution and quotation;
- causal-strength changes;
- morphology over seen versus held-out roots;
- one versus multiple interacting affixes;
- novel compositions of familiar lexemes;
- ambiguity / out-of-scope cases that should remain indeterminate.

Every semantic change must have a paired meaning-preserving control. Split by root family, templates, and semantic program—not random sentences alone—to reduce leakage.

### Gate C — paired measured advantage

Proposed qualification targets (not observed results):

- use at least 1,000 cases for the first meaningful corpus and a separately sealed confirmatory partition;
- report precision, recall, false-positive rate, false-negative rate, abstention/indeterminate rate, and performance per phenomenon;
- run at least five seeds during development, then freeze a fresh confirmatory execution;
- require a paired improvement of at least 5 percentage points on preregistered held-out compositional cases, with a 95% paired confidence interval whose lower bound is above zero;
- no more than 2 percentage points regression on the prespecified in-vocabulary control group;
- no aggregate pass if a required critical stratum fails;
- disclose confidence intervals, sample counts, failed cases, model/config digests, and resource costs.

These numerical thresholds must be reviewed and frozen before confirmatory data is opened. Change them only under a new benchmark revision.

### Gate D — external comparison

Compare Symthaea and reference systems on identical frozen inputs and identical scoring rules. Include:

- a simple symbolic/template baseline;
- the genesis-seeded Broca baseline;
- the semantic-initialized Broca variant;
- strong local open-weight models that can be pinned to exact weights/revisions;
- current hosted model APIs, with observed model IDs, request/response digests, and dated execution receipts where available;
- human/native-language references for tests that require native judgments.

Use Holmes or compatible publicly documented linguistic probes for breadth, the NAACL 2025 morphology paradigm as inspiration for novel-root challenge design, and the LM Evaluation Harness for standard-task interoperability. Do not replace external tasks with a custom score and call it industry SOTA.

For public model snapshots, record model/provider/version, prompt, decoding settings, latency, token/resource cost, date, and whether exact replay is possible. Do not compare a small untrained Broca instance to frontier models and infer that initialization alone explains any raw-score difference.

## What would count as a paradigm-level result?

A convincing result would be a reproducible, independently scored reduction in the number of training examples needed to preserve semantic relationships on **unseen roots and unseen compositions**, while remaining competitive on ordinary language controls. This would be more informative than improved fluency or a higher HDC similarity score alone.

A stronger follow-on could make language generation operate over a typed semantic program first and surface realization second, with an independent verifier rejecting polarity, quantifier, temporal, attribution, or epistemic-status drift before release. Existing Broca semantic-fidelity work (V16–V25) remains the authority for how that future capability must be qualified; this experiment does not bypass those prerequisites.

## Claim ceiling

A passing smoke test establishes only that the option is wired and local structural invariants hold. A passing held-out microbenchmark is evidence only for that exact synthetic profile. Neither establishes general language understanding, factual truth, multilingual competence, consciousness, production readiness, or superiority over state-of-the-art systems.
