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

## 2. Provider baseline roster — snapshot 2026-11-14

The roster includes both frontier/high-reasoning and cost/throughput siblings. It is a candidate matrix, not a claim that every identifier is available to every account, region or contract. Check current access, policy, pricing, data-retention and region restrictions before running any evaluation.

| Provider | Model | API identifier | Role in tournament | Official reference |
|---|---|---|---|---|
| OpenAI | GPT-6 Astra | gpt-6-astra | Flagship, high-reasoning reference | https://developers.openai.com/api/docs/models |
| OpenAI | GPT-6.1 Sol | gpt-6.1-sol | Strong cost/intelligence trade-off | https://developers.openai.com/api/docs/models |
| OpenAI | GPT-6.1 Terra | gpt-6.1-terra | Efficient and visual/interaction-capable reference | https://developers.openai.com/api/docs/models |
| OpenAI | GPT-6 Luna | gpt-6-luna | High-volume efficiency reference | https://developers.openai.com/api/docs/models |
| Anthropic | Claude Opus 5.5 | claude-opus-5-5 | Current general flagship for long-running knowledge work | https://platform.claude.com/docs/en/models/opus-5-5/overview |
| Anthropic | Claude Fable 5.2 | claude-fable-5-2 | Demanding reasoning / long-horizon agentic reference | https://platform.claude.com/docs/en/models/fable-5-2/overview |
| Anthropic | Claude Sonnet 5.5 | claude-sonnet-5-5 | Speed/intelligence balance | https://platform.claude.com/docs/en/models/overview |
| Anthropic | Claude Haiku 5.5 | claude-haiku-5-5 | High-volume and latency-sensitive reference | https://platform.claude.com/docs/en/models/overview |
| Google | Gemini 4 Carbon | gemini-4-carbon | Frontier model for complex knowledge and long-horizon tasks | https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/4-carbon |
| Google | Gemini 4 Argon | gemini-4-argon | Frontier reasoning comparator | https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/google-models |
| Google | Gemini 4 Hydrogen | gemini-4-hydrogen | Efficiency/throughput comparator | https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/google-models |
| DeepSeek | DeepSeek V4 Pro | deepseek-v4-pro | High-capability DeepSeek reference | https://api-docs.deepseek.com/updates/ |
| DeepSeek | DeepSeek V4.1 Flash-1017 | deepseek-flash | Cost/throughput and multimodal reference; alias resolves to the latest Flash release | https://api-docs.deepseek.com/updates/ |

The official DeepSeek changelog records the V4.1 Flash release on 2026-09-10 and the Flash-1017 update on 2026-10-17; the API alias remains deepseek-flash. The DeepSeek V4 Pro API name is deepseek-v4-pro. Record the exact observed model/version from provider responses where available.

### Model-version and access controls

- A provider alias is not an immutable checkpoint. Store the exact API identifier, provider-returned version/snapshot if available, request timestamp, region, endpoint, account tier and terms snapshot.
- Run the same locked item set at comparable reasoning/effort settings. Where providers expose non-equivalent effort controls, report settings rather than pretending they are identical.
- Do not compare API-only execution with a local model and call it a pure model comparison. Record the inference harness, quantization, context, tools, sampling and memory policies separately.
- Record provider-specific data-retention and safety restrictions. Do not send human-subject or community-controlled material to any provider unless the relevant permission, contract and data-handling assessment explicitly allows it.
- The frontier roster is not the full matrix. Add at least one current, genuinely runnable open-weight checkpoint with a documented license and hardware budget, plus classical/statistical baselines where appropriate.
- Never infer that an API model's existence means its weights are open or locally deployable.

## 3. External benchmark anchors and their limits

These benchmarks are useful reference points, but their constructs and sampling frames differ. They are not interchangeable, and country coverage does not prove population representativeness.

| Domain | Existing work | What HMP should borrow | Limits HMP must preserve / what HMP adds |
|---|---|---|---|
| Cultural knowledge | CulturalBench (ACL 2025): 1,696 human-written, five-annotator-verified questions across 45 global regions and 17 topics | Human-authored challenge items, red-teaming, underrepresented-region coverage | A benchmark item does not represent a whole culture. HMP adds situated source attribution, intra-community disagreement, permission state and temporal scope. https://aclanthology.org/2025.acl-long.1247/ |
| Cross-cultural symbolic understanding | CCU-Bench (2026): 3,029 QA pairs in five formats, tested on 19 models; reported mean 57.8% | Multi-format reasoning about images, connotation and emotion; explicit error analysis | It is grounded in Hofstede's dimensions and focuses on China, Japan and Mexico. It is not a global survey of cultures, religions or philosophies. HMP adds source/context-sensitive tasks and community review. https://doi.org/10.1631/ENG.ITEE.2026.0083 |
| Cultural frames | FrameNet-Cultures (ACL 2026): 18 frames across 20 countries, compared with EveryCulture reference profiles | Open-ended generation, frame semantics, comparison of neutral vs culture-conditioned prompts | The authors describe profiles as structured references, not definitive portraits. Neutral prompts showed European-leaning alignment and weaker alignment for African and Oceanian profiles. HMP adds explicit source limitations, local variants and reviewable claims. https://aclanthology.org/2026.findings-acl.491/ |
| Value preferences | DiverValue-Bench (IJCAI 2026): 23,763 instances from 1,396 users across 74 countries/regions; sample-level human audit | Subgroup-aware evaluation; reporting disparities hidden by aggregate scores | Authors explicitly warn that samples are not population-representative for country-level values. It measures profile-conditioned value preference alignment, not comprehensive cultural, religious or moral truth. HMP adds pluralism without optimizing people toward one shared preference. https://ijcai.org/proceedings/2026/850 |
| Conceptual reasoning | Conceptual Reasoning Index: LMCA, ACCoRD and DTBench | Argument-level expert rubrics, consistency checks and explicit human-rating uncertainty | CRI is a defined suite, not philosophy as a whole. LMCA rates arguments against positions; ACCoRD checks logical consistency; DTBench tests decision-theoretic questions. HMP adds attribution of positions and preservation of irreducible disagreement. https://alignment.anthropic.com/2026/conceptual-reasoning-index/ |
| Religious/interfaith literacy | ICJS Interfaith Benchmark Initiative announced 2026-07-28 | Domain expertise; accuracy, tone, bias and constructive dialogue evaluation | The initiative was described as in development at announcement; do not report it as a released scored benchmark until publication. HMP adds rights, scope, language and internal-diversity controls. https://icjs.org/resource/icjs-ai-interfaith-benchmark/ |
| Religious/pastoral boundary tests | FMG-Bench | Narrow, high-stakes domain-specific review and abstention requirements | Do not generalize pastoral guidance scores to all religions or spirituality; benchmark performance is not authorization for unsupervised pastoral deployment. https://github.com/FideAI/fmg-bench |
| Social simulation | 2026 Voat-like forum replication: 30 independent 30-day simulations vs 30 empirical windows, using stateless Dolphin Mistral 24B agents | Multiple runs, empirical comparison, confidence intervals, activity/network/toxicity/topic measures | Single platform, domain and model design; activity matches coexisted with divergences in comments, thread length, network core and toxicity allocation. HMP adds mechanism-specific, held-out and transfer validation. https://link.springer.com/article/10.1140/epjds/s13688-026-00674-x |
| Indigenous knowledge governance | CARE Principles for Indigenous Data Governance | Collective Benefit, Authority to Control, Responsibility and Ethics | CARE is governance guidance, not a dataset license or substitute for community-specific authority. HMP must apply it in context. https://www.gida-global.org/careprinciples |

### Consequence for HMP-Bench design

Do not create a “culture score” that ranks societies, religions or individuals. Evaluate whether the system:
- attributes a statement to the source and scope that actually support it;
- distinguishes community self-description, participant testimony, scholarship, archive and model synthesis;
- preserves internal disagreement and historical change;
- recognizes untranslatable or non-comparable concepts;
- distinguishes empirical claims, doctrinal/metaphysical claims, normative positions and lived experiences;
- respects operation-specific permissions, correction and withdrawal;
- reports sampling gaps and uncertainty instead of filling them with stereotypes.

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
