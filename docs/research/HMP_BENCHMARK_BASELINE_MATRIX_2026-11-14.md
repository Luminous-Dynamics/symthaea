# HMP-Bench: Industry Baseline Matrix and Tournament Protocol

**Benchmark programme:** HMP-001  
**Snapshot date:** 2026-11-14  
**Status:** Baseline roster and protocol proposal. No candidate has been scored by Luminous on HMP-Bench.  
**Purpose:** Freeze comparable model baselines and prevent unsupported “better than SOTA” claims.

## 1. Why a frozen, task-specific tournament

General model leaderboards measure broad capabilities and are useful for shortlisting. They do not determine which system best preserves a community's own account, distinguishes scholarship from testimony, handles a permission revocation, or predicts a held-out social outcome.

HMP-Bench therefore measures:
- evidence-grounded attribution and interpretation;
- diversity of viewpoints and faithful preservation of disagreement;
- calibrated uncertainty and abstention;
- policy enforcement for operation-specific rights;
- philosophical argument quality against expert assessments;
- cross-cultural construct validity in psychological measures;
- empirically calibrated social-system predictions.

No single aggregate score is a valid substitute for these tracks. Rights violations are hard failures. Human ratings are references with uncertainty, not metaphysical ground truth.

## 2. Frontier API roster to freeze for the first tournament

The provider model identifiers below were verified in current official documentation as of the snapshot date. Re-check availability, exact snapshot/version, region and pricing immediately before each run; aliases may move.

| Provider | Model | API identifier | Official reference |
|---|---|---|---|
| OpenAI | GPT-6 Astra | \`gpt-6-astra\` | https://developers.openai.com/api/docs/models/gpt-6-astra |
| Anthropic | Claude Fable 5.2 | \`claude-fable-5-2\` | https://www.anthropic.com/claude/fable |
| Google | Gemini 4 Carbon | \`gemini-4-carbon\` | https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/get-started-with-gemini-4 |
| DeepSeek | DeepSeek-V4.1-Flash (updated 2026-10-17) | \`deepseek-flash\` | https://api-docs.deepseek.com/updates/?release=deepseek-coder |

This is the **flagship frontier comparison set**, not the whole test matrix. Add current cost/latency-oriented sibling models for each provider before the full tournament. Include a strong open-weight model that is actually runnable under a declared inference budget, and a transparent classical/statistical baseline for tasks where one is appropriate.

Do not describe a model as “open source” merely because an API is available. Record weight availability, license, checkpoint digest, quantization and deployment route separately. DeepSeek's announcement links a weight checkpoint, but its very large MoE configuration still requires an explicitly documented inference setup; API testing is not local self-hosting.

## 3. External benchmark anchors

These are complementary benchmarks, not interchangeable measures. Check dataset access, consent, terms and licensing before direct reuse.

| Domain | Existing work | What HMP should borrow | What HMP must add |
|---|---|---|---|
| Cultural knowledge | CulturalBench (ACL 2025), 1,696 verified questions across 45 regions | Human-authored items, geographic breadth, annotator verification | Per-claim source provenance, intra-community disagreement, rights status and translation ambiguity |
| Cross-cultural understanding | CCU-Bench (2026), 3,029 QA pairs; reported 57.8% mean across 19 systems | Multi-format cross-cultural tasks | Separate factual, symbolic, pragmatic and value-description outcomes |
| Cultural framing | FrameNet-Cultures (ACL 2026), 18 frames across 20 countries | Explicit frames and structured country/context references | Treat profiles as situated references, never as canonical summaries of societies |
| Values | DiverValue-Bench (IJCAI 2026), 23,763 instances across 74 countries/regions | Population-diverse evaluation and subgroup reporting | Measure faithful explanation of difference, not conformity to a preferred value profile |
| Conceptual reasoning | LMCA, ACCoRD and DTBench in the 2026 Conceptual Reasoning Index | Argument-level expert rubrics and consistency checks | Preserve attributed philosophical positions and disagreement without pretending all conclusions have binary ground truth |
| Religious literacy | ICJS interfaith initiative; FMG-Bench | Domain-expert review and carefully bounded use cases | Broader traditions/languages, practitioner/community review, attribution and permission gates |
| Social simulation | 2026 operational-validation forum study; 2026 social-network simulation survey | Multi-run empirical replication and comparison of classical / generative approaches | Held-out mechanism, network, temporal, intervention and transfer validity |

## 4. Tournament design

### A. Two distinct evaluation lanes

**Lane 1 — Base-model capability**
- No external retrieval, no tools, no custom domain corpus.
- Fixed system prompt, user prompt, output schema and token budget.
- Run each model under documented standard and high-reasoning settings where available.
- Measures underlying model capability without hiding weaknesses behind tooling.

**Lane 2 — Complete research-system capability**
- Same frozen source corpus, retrieval budget, tools, schema and policy interface for every candidate.
- Compare raw model, model + retrieval, model + provenance/rights enforcement, and full HMP pipeline.
- Include ablations: remove provenance, remove disagreement representation, remove rights gate, remove cross-lingual source support.
- Measure end-to-end success, quality, rights correctness, latency and cost per correctly qualified answer.

Do not compare a tool-using HMP system with a bare model and attribute all differences to the model. Report both lanes.

### B. Evaluation controls

- Freeze and hash the item set, prompts, rubric, tools, corpus, model IDs and scoring code before the test.
- Store provider, exact model ID/snapshot, request date/time, reasoning level, sampling configuration, tool calls, input/output token counts and all errors/refusals.
- Use paired items across systems, blinded human scoring, independent adjudication for serious disagreements and inter-rater agreement.
- Use repeated runs for nondeterministic outputs; specify seed support and its limits.
- Use a calibration/dev set and a locked test set. No prompt tuning on locked items.
- Stratify by task, language, region, source type and claim type. Publish sample sizes and uncertainty intervals.
- Record API price snapshot and observed cost per successful qualified result. Include caching, retrieval, tools and retries.
- Keep rights and safety tests separate from answer-quality scoring; one unauthorized use fails the relevant release gate.

## 5. Per-track metrics

| Track | Primary measures | Hard failure examples |
|---|---|---|
| ATTR | Attribution precision, scope error, false universalization rate | Treating an individual testimony as universal doctrine |
| SOURCE | Citation precision/recall, passage entailment, unsupported claim rate | Fabricated or irrelevant citation presented as support |
| PLURAL | Perspective coverage, disagreement recall, false-consensus rate | Silently erasing a material disagreement |
| UNCERTAINTY | Brier score/ECE where probability elicitation is valid, selective risk, abstention quality | Presenting unknown/non-empirical claims as established empirical facts |
| RIGHTS | Unauthorized operation count, revocation latency, policy-negative-control pass rate | Any denied/revoked/restricted operation performed |
| PHILO | Expert rubric agreement, premise/objection fidelity, argument attribution, consistency | Misattributing a position or fabricating an argument |
| PSYCH | Reliability, invariance, subgroup calibration and out-of-sample predictive validity | Treating a translated instrument as valid without checking measurement properties |
| SOC | Held-out distributional, network, temporal and intervention errors | Causal recommendation from observational fit alone |
| CORRECT | Correction propagation, stale-view rate, retraction compliance | A withdrawn claim remains retrievable as current |

For expert-scored philosophical items, report human-human agreement and uncertainty. The 2026 Conceptual Reasoning Index authors estimate a ceiling near 91 on their aggregate because human ratings themselves are noisy; model scores should be interpreted relative to that ceiling and the benchmark's exact weighting, not as absolute philosophical correctness. See https://alignment.anthropic.com/2026/conceptual-reasoning-index/.

## 6. What “better than SOTA” must mean

A result is competitive only if the comparison is matched, preregistered and independently reproducible. HMP should aim to outperform the strongest baseline on a declared task and metric, not every model on every task.

Proposed proof obligations:

1. **Source fidelity:** materially reduce unsupported claims and scope errors without reducing supported-answer coverage.
2. **Pluralism:** reduce false universalization and false consensus while preserving the actual differences in source material.
3. **Uncertainty:** improve selective risk / abstention quality against calibrated baselines.
4. **Rights:** zero unauthorized operations across adversarial negative controls; this is a gate, not an optimization metric.
5. **Cross-cultural validity:** improve performance on the weakest supported language/context slices without erasing genuine value differences.
6. **Social simulation:** demonstrate positive held-out predictive skill beyond classical/statistical baselines on both aggregate and network/temporal outcomes; validate intervention forecasts only against credible intervention evidence.
7. **Efficiency:** lower cost/latency/energy per correctly attributed, qualified result, not just per generated token.

### Ablation evidence

The full pipeline should be compared with:
- base model only;
- base model + retrieval;
- retrieval + citation/provenance;
- retrieval + provenance + disagreement representation;
- full pipeline with Mycelix policy enforcement;
- human expert baseline.

This identifies which component provides the measurable benefit. If provenance does not improve citation faithfulness, or the permission gate does not block unauthorized access under fault injection, the system has not earned the claimed advantage.

## 7. Release gates and claims policy

- No published leaderboard result before the test set, prompts, rubric and evaluation code are frozen.
- No “SOTA” label without a matched comparison with current model versions and uncertainty intervals.
- No single weighted score across rights, epistemics, philosophy, psychology and sociology.
- No aggregate-only reporting; publish per-group/per-language slices where sample sizes and consent allow.
- Any hard rights failure blocks release of that configuration.
- Report failures, abstentions, missing data and excluded cases; do not discard them silently.
- External replication is required before a headline performance claim.

## 8. Immediate execution

1. Preserve this 2026-11-14 roster as a dated reference; refresh aliases before each run.
2. Build the HMP-Bench item format and synthetic negative-control suite first.
3. Obtain dataset permissions and expert/community reviewers before assembling real cultural/religious examples.
4. Run a small 200-item pilot with human annotation and inter-rater checks.
5. Run Lane 1 and Lane 2 against the frozen roster.
6. Publish the complete receipt: item-set digest, model IDs, prompts, config, scoring code revision, run IDs, cost, latency, per-track results, intervals and limitations.

**Current status:** roster and protocol specified only. No model was called, no benchmark item was scored, and no comparative performance result is claimed in this document.
