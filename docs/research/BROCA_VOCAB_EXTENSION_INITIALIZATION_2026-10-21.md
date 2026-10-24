# Broca Vocabulary-Extension Initialization Experiment

Date: 2026-10-21
Status: opt-in API and initialization-only smoke benchmark; runtime validation pending

## Research question

Does the way Broca initializes a newly appended token materially affect (1) preservation of existing-token behavior, (2) the geometry and scale of the new vector, (3) the probability mass assigned to the new token before training, and (4) the amount of follow-up training needed to learn a useful representation?

Do not conflate these with the separate from-scratch V1/V2 genesis-seed experiment in PR #8270. A from-scratch symbol table and a new token appended to a model with an existing embedding table are different tasks.

## Literature and implications

- Mundra et al., CoNLL 2024, compare vocabulary-expansion initialization methods for RoBERTa and LLaMA 2 over four languages and five tasks. They motivate initialization within the convex hull of existing embeddings and introduce Constrained Word2Vec; their simple baselines are competitive with more advanced methods in the studied setting: https://aclanthology.org/2024.conll-1.8/
- Tejaswi et al., Findings of EMNLP 2024, report that early model performance does not always predict final adapted performance and that optimal vocabulary/adaptation choices are language-dependent: https://aclanthology.org/2024.findings-emnlp.614/
- Joshi et al., arXiv, 4 August 2026, compare more than 20 token-initialization methods for Hindi vocabulary extension in a pretrained Nemotron model. Their reported gains include subword composition and norm calibration; they also find initialization loss alone is unreliable as a proxy for downstream convergence: https://arxiv.org/abs/2608.03494
- Chen et al., arXiv, 2 April 2026, study grounded initialization for newly added semantic-ID tokens in generative recommendation. Their analysis highlights that naive mean initialization can collapse distinct new tokens into a degenerate subspace: https://arxiv.org/abs/2604.02324

These studies differ from Broca's HDC/CfC architecture, base embeddings, data, and tasks. They establish valuable hypotheses and relevant baselines, not an expectation that a published improvement will transfer.

## Implementation in this change

The existing `LanguageController::append_token` remains unchanged for compatibility. A separate `append_token_with_initialization` API adds explicit opt-in strategies:

1. `SeededRandomV1`: the exact legacy genesis-seeded token vector.
2. `ContextBundleNormMatched`: bundle supplied constituent/context vectors and scale the candidate to the median norm of existing embedding rows.
3. `ContextBundleNormMatchedWithResidual`: blend the context bundle with a deterministic, name-specific genesis residual, then match the median existing row norm. The residual weight must be finite and in [0, 1].

Context-based initialization fails before mutating the vocabulary when context is missing, dimension-mismatched, non-finite, or degenerate; invalid residual weights are rejected. The append operation returns the inserted index, preserves old rows, and rebuilds the optional GPU cache consistently with the legacy path.

The benchmark compares four arms: the seeded-random control, the legacy context-bundle behavior, the norm-matched context bundle, and context bundle plus a 0.15 residual. It operates on synthetic vector contexts from a fixed genesis phrase and deliberately duplicates one context set as a collapse negative control. Those contexts have **no linguistic labels**, so the benchmark is intentionally about geometry and output-distribution effects—not semantic correctness.

Run from the repository root:

```sh
GIT_COMMIT_SHA=$(git rev-parse HEAD) cargo run -p symthaea-broca --bin broca-vocab-extension-bench -- --json-out extension.json
```

## Smoke receipt fields

- bit-identical preservation of all old embedding rows;
- maximum absolute change in old-token logits on the same fixed input;
- whether the old-vocabulary top-1 token remains unchanged;
- median new-row norm relative to the median old-row norm;
- mean similarity to supplied context rows;
- mean and maximum absolute pairwise cosine among added rows;
- mean and maximum probability mass assigned to new tokens across 16 deterministic fixed queries, plus non-finite logit count;
- KL divergence between the original old-vocabulary distribution and the extended distribution renormalized over old tokens;
- finite/non-finite coordinate counts;
- schema version, fixed fixture digest, source revision, and explicit claim boundary.

The benchmark uses a bank of 16 deterministic queries. It reports the fraction of queries where the original top-1 token remains top-1 across the full extended vocabulary, and the minimum old-best-vs-new-best logit margin. The KL metric isolates changes within the old-token conditional distribution. If extension leaves old logits unchanged, that conditional KL should be near zero; total probability mass assigned to newly added tokens is reported separately because softmax's denominator changes.

A literature result from CoNLL 2024 motivates constraining new embeddings to the convex hull of existing rows under a dot-product scoring argument. Broca's current logits use cosine similarity, so we do **not** assume that guarantee transfers: vector normalization changes the geometry of the score. This branch measures greedy preservation empirically instead. The next quality-stage version should collect representative hidden/query states from a fixed calibration corpus, tune or reject candidates using old-vs-new top-1 margins on calibration data, and verify on a sealed holdout; finite calibration cannot prove preservation for every possible hidden state.

## Proposed evaluation gates

Targets below must be preregistered and are not current results:

### Gate A — invariants

- 100% of old embedding rows bit-identical after extension.
- Old logits for a fixed input change by at most numerical tolerance (target `<= 1e-6` in deterministic CPU tests).
- Zero non-finite coordinates in new vectors.
- Invalid context or residual-weight input fails closed without changing vocabulary size.
- Norm-matched variants achieve a median norm ratio in [0.99, 1.01] under the declared metric.
- Equal seed/name/context/strategy gives byte-identical initialization.

### Gate B — initialization geometry

Use at least 5 seeds and multiple context sets. Report context alignment, pairwise cosine/collapse, vector norm, and probability mass on new tokens separately. Include controls with duplicate contexts and opposing contexts to test degeneracy handling.

### Gate C — task value

Use independently labelled unseen lexical families and templates. Train each arm under identical conditions. Measure task quality by stratum, number of tokens/steps and compute required to reach a fixed held-out quality target, old-vocabulary non-inferiority, and calibration. Require a paired confidence interval excluding zero before asserting a quality advantage.

A reasonable research target is at least 2x lower median steps-to-target than seeded-random controls, with no more than a 2 percentage-point regression on established old-vocabulary controls. This is a hypothesis to test, not a prediction or result.

## Paradigm shift

Treat vocabulary extension as a constrained, measurable state transition rather than blindly appending another row. The system should know what evidence supports a new token (known constituent vectors, caller-supplied context, or none), preserve already learned behavior, make uncertainty explicit, and expose distribution changes for review.

The next likely iteration is a **preservation-constrained initializer**: select a candidate in the existing embedding space, optionally use subword or grounded directions, add a small name-specific residual to prevent collapse, and choose scale through an explicit norm/calibration rule. Avoid a mean-only approach without testing pairwise collapse; avoid calling context vectors "semantic" unless the contexts and labels support that claim.

## Geometry finding: stored norm is not a logit control

Broca's current `compute_logits` path is explicitly `logit_scale * cosine_similarity(output_hv, token_embedding)`; the accelerated path also describes embeddings as pre-normalized. A positive rescaling of a token row therefore leaves its cosine logit unchanged (subject only to floating-point tolerance). Norm matching is still a stored-vector invariant and may matter to other consumers, but it must not be presented as a mechanism for controlling this path's new-token activation or preserving old predictions.

A metamorphic unit test now checks this scale invariance through `compute_logits`, including any enabled backend selected by the build. The next quality-oriented work is angular candidate selection with labelled protected-old and intended-new calibration contexts, plus a held-out false-activation measurement. See [issue #8285](https://github.com/Luminous-Dynamics/symthaea/issues/8285). This is a design finding, not benchmark evidence; Rust tests remain unrun in this environment.

## Validation boundary

- GitHub read-back confirms the experimental strategy API, benchmark, Cargo registration, and regression tests are present on this branch.
- Cargo tests, rustfmt, Clippy, and benchmark execution have not been run in this environment.
- The receipt is a smoke diagnostic only; no downstream quality or SOTA claim is made.

Related work:
- [Issue #8271 — benchmark initialization for vocabulary extension](https://github.com/Luminous-Dynamics/symthaea/issues/8271)
- [PR #8270 — versioned genesis-token baseline experiment](https://github.com/Luminous-Dynamics/symthaea/pull/8270)
- [Issue #8259 — preserve lexical distinctions beyond shared primes](https://github.com/Luminous-Dynamics/symthaea/issues/8259)
