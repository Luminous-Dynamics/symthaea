# Structural Transfer Scorecard v0

**Status:** preregistration proposal; not yet an expert-adjudicated benchmark or a model score. The ten entries in `metamorphic-pairs.jsonl` are synthetic development controls only.

## The paradigm shift

A cultural or mythological knowledge system should not be scored as if a living tradition, historical period, or manuscript corpus were one canonical graph. The unit of evidence is a **contexted claim tied to a witness**; the unit of robustness is a **controlled intervention pair**. A good system must preserve evidence lineage and method while allowing the answer to change when the evidence, witness, variant, context, or access policy changes.

This complements three directions in recent evaluation work:

- [CulturalBench (ACL 2025)](https://aclanthology.org/2025.acl-long.1247/) uses 1,696 human-written and human-verified questions across 45 regions and reports multi-answer failures and uneven regional performance.
- [Measuring Meta-Cultural Competency (ICML 2026)](https://proceedings.mlr.press/v306/saha26a.html) evaluates cultural knowledge structure as well as answer matching and explores spectral comparisons across eight models, nine cultural domains, and 170 countries. It motivates structural evaluation, but does not validate a mythology-specific spectral score.
- [FrameNet-Cultures (Findings ACL 2026)](https://aclanthology.org/2026.findings-acl.491/) uses open-ended frame semantics across 18 frames and 20 countries, while explicitly treating the reference profiles as comparison artifacts rather than definitive descriptions of every society.

Our proposed distinction is narrower and more auditable: **evidence-bounded structural transfer** across source witnesses, versions and contexts, with explicit checks for false lineage and erased plurality.

## Machine-readable paired interventions

`metamorphic-pair.schema.json` defines a pair as:

1. a baseline state with an explicit evidence pack;
2. exactly one declared intervention, such as a context swap, source swap, evidence removal, relation mutation, version switch, counterevidence addition, rights revocation, equivalent alias, or prompt-injection insertion;
3. a mutated state;
4. the expected relationship between outputs; and
5. criteria defining what must remain invariant, what must change, and what must never occur.

The initial ten pairs are explicitly synthetic. They qualify the test machinery, not Egyptology, another tradition, or the model. After expert-reviewed pairs exist, each scenario should have a stable ID, evidence digest, mutation digest, annotation provenance and a separately kept test split.

Score structured supported claims, relation labels and citation targets—not raw string equality. Paraphrases can be equivalent; fluent wording must not hide citation drift.

## Scorecard: report dimensions separately

| Dimension | Primary measure | Failure captured |
|---|---|---|
| Expected-change fidelity | Fraction of intervention pairs where required semantic changes occur without violating preserved invariants | Model is insensitive to new context/evidence or changes unrelated facts |
| Expected-invariance fidelity | Fraction of pairs where semantic claims remain stable under declared irrelevant/equivalent changes | Alias noise or source-embedded instructions alter reasoning |
| Citation identity | Claim-level precision and exact witness/edition/locator correctness | Citation looks plausible but points to the wrong version |
| Typed relation integrity | Macro-F1 and confusion matrix for `precedes`, `causes`, `resembles`, `historically_influences`, and other adjudicated relations | Sequence becomes causation; resemblance becomes lineage |
| Variant isolation | Unsupported cross-variant claim/edge rate | Different tellings are silently merged into one canon |
| Plurality retention | Recall of independently supported alternatives; false-consensus rate | An arbitrary single answer erases valid disagreement |
| Evidence withdrawal | Revision/abstention accuracy after removing the sole support | A previous answer survives after its evidence disappears |
| Access and rights | Unauthorized retrieval/derived-content count after policy mutation | Public discoverability is mistaken for reuse permission |
| Calibration | Risk–coverage curve and selective risk at fixed coverage | Confident answers persist where evidence is missing |
| Structural transfer | Context-conditioned typed-edge/role agreement on held-out source families; worst-group score and bootstrap interval | A system fits one source family while failing on unfamiliar witnesses |
| Efficiency | Retrieval/generation latency, tokens, memory, tool calls and total cost per accepted answer | Quality is purchased with an unfair budget |

Never collapse the first eight dimensions into an unconstrained scalar. **Rights violations, fabricated lineage, and critical evidence/provenance breaks are release gates, not errors that good prose or high average accuracy can offset.** Report a balanced panel by tradition, period, language/translation, genre, source type, answerability and access status. Publish both mean and worst-group / lower-tail performance.

## Structural and spectral comparison

Where multiple qualified annotators independently annotate a source-conditioned graph, compare distributions of typed edges, node roles, motifs and (exploratorily) graph spectra. Keep disagreements and alternate graphs rather than averaging them into an unqualified universal graph.

Before using spectral distance, normalize or stratify for graph size, sparsity and degree distribution; compare against degree-preserving/permutation nulls; bootstrap by source family, not individual edges. Report sensitivity to graph construction and annotation aggregation. A similar spectrum or embedding is **not** evidence of identical meaning, cultural consensus, contact, borrowing or historical influence. Treat these graph summaries as diagnostic features whose predictive value must be tested against held-out transfer outcomes.

## Baselines and protocol

Use the same frozen cases, source pack, allowed evidence, prompt budget and output schema for:

1. keyword/BM25 retrieval;
2. dense retrieval;
3. conventional RAG with citations;
4. source-aware RAG with witness/period constraints;
5. typed temporal/context graph retrieval;
6. typed graph retrieval plus Symthaea HDC composition;
7. ablations removing context, witness identity, typed relations, counterevidence, or rights/provenance gates.

Record model/version and date, retrieval/embedding/reranker versions, prompt and decoding configuration, seed/repeats, index/source digests, evidence-window budget, latency, token counts, hardware/tool calls and cost. Hold out complete source families/work-recensions; do not use random question splits as the only generalization test. Reserve the held-out test until graders and thresholds are frozen.

For expert-reviewed answer pairs, use multiple blinded reviewers, keep per-reviewer labels and disagreement, and measure inter-rater reliability before making a metric a gate. Automatic graders may triage but must not be the sole judge. Check whether each annotation is answerable from the source pack; abstention is correct only when required evidence is absent or access is denied.

## Provisional acceptance targets (not achieved results or universal standards)

After a pilot establishes reliable annotations, preregister the thresholds and primary endpoint:

- **Critical controls:** 100% correct policy decision on audited release-blocker controls; zero unauthorized retrieval or unsupported historical-influence promotion in the qualification set.
- **Citation precision:** target at least 0.98 on answerable, closed-world claims; exact source/witness identity reported separately.
- **Unsupported factual claims:** target at most 1% of factual claims on answerable, closed-world cases, with claim-level annotation and confidence interval.
- **Clear relation / temporal cases:** target at least 0.90 macro-F1 or accuracy where labels are reliably adjudicable.
- **Paired intervention fidelity:** target at least 0.90 on each major intervention family, not merely as a pooled average.
- **Comparative claim:** require a preregistered primary metric, equal budgets, repeated runs where outputs are stochastic, source-family bootstrap confidence intervals, and a positive interval for improvement over the strongest baseline. An engineering ambition is at least +5 percentage points absolute on the primary endpoint while no audited subgroup regresses by more than 2 points; these are proposed gates, not established community norms.
- **Expert interpretation:** report rubric-wise blind pairwise wins, disagreements and uncertainty. Do not force one global target where subject specialists cannot reliably agree.

A 10-pair synthetic control pack is far too small for a superiority claim. Before a public leaderboard, aim for at least 200 adjudicated held-out pairs across several source families and multiple task families, with the final count driven by pilot power analysis and annotator agreement. The current scorer imposes a provisional minimum of 30 *independent connected source-family clusters* for held-out inference; a pair that references multiple families joins them into one dependence cluster, so ten pairs do not magically yield ten independent sources. This is an eligibility floor, not a guarantee of power or a universal standard. Clustered-inference studies warn that few independent clusters can inflate error rates and advise small-sample corrections; one review notes recommended thresholds vary by analysis and design ([small-cluster analysis review](https://academic.oup.com/ije/article/47/1/321/4091562), [cluster-robust inference guide](https://doi.org/10.1016/j.jeconom.2022.04.001)). These studies concern clustered statistical designs rather than this exact benchmark, so we borrow the conservatism, not their thresholds as direct validation. A pilot power/sample-size rationale and cluster-size distribution must be reported. A cross-tradition claim additionally requires meaningful, rights-cleared, source-family-disjoint data for the named traditions; Egypt-only results do not establish universal transfer.

## Decision rule

A release qualifies only if its critical gates pass and its performance panel meets preregistered thresholds. Otherwise report exactly which gate failed, the exact dataset/config/model digests, and the next falsifying experiment. A passing data contract is not a passing model benchmark; a queued runner is not a pass; structural similarity is not historical evidence.

