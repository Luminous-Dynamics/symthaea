# Preregistered comparison protocol v0

**Status:** protocol and synthetic schema control only. No model inference, timed benchmark run, or SOTA result has been performed by this plan.

## Objective

Measure the value of source-witness/context-aware reasoning over strong, simple retrieval baselines under comparable evidence and tool budgets. The primary question is not "does the answer sound plausible?" It is whether a system makes the right source-conditioned claim, changes when relevant evidence changes, stays stable under irrelevant changes, preserves valid alternatives, and respects access restrictions.

The JSON Schema contracts are `comparison-plan.schema.json`, `pair-result.schema.json`, and `score-report.schema.json`. `synthetic-plans.jsonl`, `synthetic-pair-results.jsonl`, and `synthetic-source-pack.jsonl` are intentionally fabricated qualification fixtures. Their digests, system labels, and observation labels are fake and must never be treated as real runs or model results. The `score-paired-results` Rust binary scores adjudicated paired results under a preregistered plan, validates the generated report against `score-report.schema.json`, and writes a digest-bound report; it does not execute a model or verify human labels against source material. Before scoring, it recomputes and compares the exact bytes of the paired case manifest, source catalog, and source pack with the predeclared SHA-256 digests; a mismatch fails closed. It additionally requires the source-pack baseline/mutated snapshots to match the pair states field-for-field and evidence-item-for-evidence-item, and checks non-synthetic source rights/access against the catalog.

## Current benchmark landscape and resulting controls

This design should be compared to, not substituted for, adjacent benchmarks:

- [BRINK (EACL 2026)](https://aclanthology.org/2026.eacl-long.114/) argues that knowledge-graph RAG evaluations can accidentally reward reading a directly present triple instead of reasoning with incomplete knowledge. For MythologyBench, explicitly separate one-edge lookup from multi-hop composition, remove necessary links as a paired intervention, and score abstention when the remaining graph cannot justify the answer.
- [KGHaluBench (Findings EACL 2026)](https://aclanthology.org/2026.findings-eacl.206/) uses dynamically generated knowledge-graph questions and checks hallucinations at conceptual and correctness levels. The corresponding opportunity here is a deterministic, digest-pinned case generator that creates fresh source-family-safe contrastive questions and negative controls after the schema, generator and grader are frozen.
- [Measuring Meta-Cultural Competency (ICML 2026)](https://proceedings.mlr.press/v306/saha26a.html) argues that cultural structure/pluralism are not captured by exact-answer accuracy alone. We should report structure-preservation and unfamiliar-tradition transfer as separate endpoints, while explicitly prohibiting graph-structure similarity from serving as evidence of historical origin or shared meaning.
- [CulturalBench](https://aclanthology.org/2025.acl-long.1247/) highlights multi-answer and region-coverage failures; use human-adjudicated answer sets and report worst-group performance, not only a pooled score.

These papers motivate additional benchmark tracks; they do **not** show that Symthaea has beaten those systems. The current ten paired interventions remain synthetic contract tests, and the source catalog still has no eligible held-out set.

## Baseline ladder

Run as much of the following ladder as the research question needs, with pinned identities:

1. **BM25 / lexical retrieval** — low-complexity, interpretable retrieval control.
2. **Dense retrieval** — embedding retriever, fixed index/source pack.
3. **Standard RAG** — retrieval + generator with citations, no witness/variant-specific guard.
4. **Source-aware RAG** — source-family, witness/edition and time/context filters.
5. **Typed narrative graph** — explicit typed nodes/edges, relation and provenance gates.
6. **Typed graph + Symthaea HDC** — the proposed system, adding compositional HDC/graph reasoning on top of the same admissible evidence.
7. **Ablations** — disable source identity, context, variant, typed relation, counterevidence, or rights gates one at a time.
8. **Expert reference** — blinded subject-expert labels / adjudicated claim and alternative sets, not a model-like exact string.

Before any output is inspected, freeze `candidate_system_id` alongside the full baseline list. The candidate cannot be changed after seeing scores. Synthetic systems in this repository are scorer controls only; their names do not imply that BM25 or Symthaea was actually executed.

An LLM baseline should have a pinned model identifier/revision or dated API identity, not just a product label. Pin retriever, embedding model, index build, reranker, prompt, decoding, tool implementation and all evidence/source digests. Report proprietary models as API baselines if weights are unavailable; don't claim full reproducibility beyond what their API exposes.

## Equal-budget rule

Each plan freezes one shared budget envelope: maximum evidence tokens, context tokens, output tokens, tool calls, wall-clock milliseconds and retries. A system may use less. Log actual tokens, tool calls, latency, memory and cost. No contender may receive more source text or more calls and then be declared better without a separate budget-sensitivity experiment.

Use the same evaluation questions, allowed source pack and answerability labels. If a retriever returns different passages, cap *evidence tokens delivered to the model*, not merely the number of passages. Report retrieval quality separately from answer-generation quality. Include a no-retrieval or closed-book condition only as an explicitly separate baseline, never as an implicitly under-resourced competitor.

## Primary endpoint and non-compensatory gates

Choose one primary endpoint *before* inspecting held-out output. Recommended first endpoint: paired-intervention fidelity (semantic output change/invariance must match the predeclared relationship). Recommended secondary metrics include citation precision, unsupported-claim rate, typed-relation macro-F1, variant isolation, plurality retention, evidence-withdrawal accuracy, calibrated risk/coverage, rights violations, worst-group performance, latency and memory.

Rights/access violations, fabricated citations, source-family leakage and unsupported promotion of `resembles` to `historically_influences` are hard gates. A model with perfect accuracy cannot offset these failures. Report accuracy, evidence quality, safety gates, latency, cost and user-facing usefulness as distinct dimensions.

## Split and source policy

- Split by `source_family_id`—not question, URL, paraphrase or individual passage.
- Keep all variants / URLs under one work-family together when leakage is plausible.
- Freeze source IDs, exact editions/witnesses, locators, source-pack manifest and digests.
- Never use source families seen in development for a purported unseen-family test without explicitly reporting that limitation.
- Unknown rights imply metadata-only use. Review copyright/licensing and community access expectations before including content in an evaluation pack.
- The held-out manifest stays sealed until prompts, graders, model/config IDs, metrics and thresholds are frozen.

Current source catalog entries are all development-only, metadata-only, rights-unknown and unreviewed. **There is no eligible held-out corpus yet**, so a `preregistered_held_out` manifest should fail closed until a rights-reviewed, source-family-disjoint split exists.

## Statistics and human evaluation

- For paired tasks, evaluate each system on the same input/intervention pairs. For deterministic systems, one execution may be enough for exact reproducible outputs; for stochastic or hosted APIs, predeclare seeds/repeats and report variability.
- Bootstrap confidence intervals by **connected source-family clusters**, not individual questions or edges. If a paired intervention references multiple families, those families join one connected component; pairs sharing any family are resampled together. The scorer uses the preregistered seed/replicate count, the same cluster resamples for candidate and all baselines, and percentile intervals on paired differences. For held-out comparisons against multiple baselines, the scorer requires a preregistered familywise correction (`bonferroni`); the report records the correction method and cluster count. A held-out inference run also requires at least 30 independent connected clusters; this is a conservative eligibility floor, not a guarantee of statistical power. Smaller or highly connected corpora remain exploratory and cannot produce a qualifying held-out comparison. The pilot must still preregister a power/sample-size rationale and report cluster-size imbalance; clustered-inference guidance warns that small cluster counts can distort uncertainty, and recommended thresholds depend on the design.
- Report paired differences against each baseline and confidence intervals on the primary endpoint; correct for multiple comparisons if making multiple primary claims.
- Report mean, lower-tail and worst-group scores by tradition, period/genre, source type, language/translation, answerability and access status where sample sizes permit. Make sparse-group uncertainty explicit.
- Require blinded reviewers for open-ended interpretations. Preserve their independent labels and disagreement. Report agreement/reliability before adjudication; split compound claims into atomic claims rather than forcing one label.
- Keep the held-out set fixed across model versions; publish per-case outputs and digests when rights allow, otherwise publish safe aggregates and audit receipts.

## Provisional engineering targets (not standards or achievements)

These are hypotheses for a pilot, not reported performance:
- 100% on audited release-blocker controls; zero rights violations, false citations or unsupported historical-lineage promotions in qualification cases.
- At least 0.98 citation precision on answerable closed-world cases once a credible annotated set exists.
- At most 1% unsupported factual claims on answerable closed-world cases, reported with confidence intervals.
- At least 0.90 paired-intervention fidelity *per major intervention family*, not just pooled.
- An engineering ambition of +5 percentage points absolute on the preregistered primary endpoint versus the strongest comparable baseline, with no evaluated subgroup regressing by more than 2 points. Only claim improvement if the confidence interval supports it and human annotation is reliable.

Set or revise thresholds using a pilot and power analysis **before opening the final test**. The initial 10 synthetic pairs cannot support statistical superiority.

## Cluster uncertainty policy

The scorer resamples connected source-family clusters, not individual questions, with one shared bootstrap draw per candidate/baseline comparison. If a pair bridges two families, it connects them; observations inside the resulting component are not treated as independent. This approach is intentionally conservative about leakage, though a heavily connected source graph can leave few effective clusters.

The current **30 independent-cluster minimum** is a provisional eligibility gate, not a universal theorem or a guarantee of power. General clustered-inference guidance warns that few clusters can distort uncertainty; suggested sample thresholds differ across analysis designs and study characteristics ([small-cluster analysis comparison](https://academic.oup.com/ije/article/47/1/321/4091562); [cluster-robust inference guide](https://doi.org/10.1016/j.jeconom.2022.04.001)). Those studies do not validate this particular benchmark. Therefore any real held-out evaluation must preregister a pilot power/sample-size rationale, report cluster-size distribution and effective cluster count, and test sensitivity to cluster definitions. Do not interpret a positive percentile-bootstrap interval as enough by itself for an SOTA claim.

## What would count as beating the state of the art?

A defensible claim needs all of the following: a source-family-held-out set with transparent inclusion/exclusion rules; rights review; independent subject-expert annotation; a published baseline ladder; equal budgets; frozen model/retriever/index/prompt/decoding revisions; preregistered primary endpoint; uncertainty intervals; ablations showing which component caused the gain; worst-group and critical-gate reporting; and third-party reproduction where feasible. If improvements disappear when source-family split or evidence budgets are enforced, the claimed advance does not survive.

The paired scorer computes relation accuracy, preservation rate, required-change satisfaction, citation identity rate, forbidden-behavior rows, per-dimension paired fidelity, typed-relation macro-F1, worst-dimension score, critical-gate violations, and source-family-clustered candidate-versus-baseline confidence intervals. Bootstrap samples connected source-family clusters, uses the preregistered seed/replicate count, and reuses the same resamples for candidate and all baselines. Held-out comparisons against multiple baselines require preregistered Bonferroni familywise correction; the report records the correction method and cluster count. A lower confidence bound above zero is a statistical signal for the specified test distribution, not an SOTA claim. The held-out scorer requires at least 30 independent connected source-family clusters plus a preregistered power rationale; 30 is an eligibility floor, not a guarantee of power. In `preregistered_held_out` phase, incomplete system/pair coverage fails closed and candidate critical-gate failures return a nonzero exit after preserving the report. This paired-result scorer currently supports `paired_intervention_fidelity`, `citation_identity_rate`, `typed_relation_macro_f1`, and `worst_dimension_score` as primary endpoints; other endpoint families need their own matching annotation/scoring contract.

The synthetic control files in this directory are *not* experiment results. Only after these conditions are met should a paper or product claim say that Symthaea outperforms a named baseline on a defined metric and test distribution.
