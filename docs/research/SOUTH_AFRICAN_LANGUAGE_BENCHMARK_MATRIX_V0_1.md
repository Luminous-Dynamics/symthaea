# South African Language Benchmark Matrix v0.1

**Status:** research and dataset-admission matrix; not a benchmark result.  
**Date:** 2026-11-05  
**Parent:** [Symthaea KNOW-INT-001 / #10494](https://github.com/Luminous-Dynamics/symthaea/issues/10494)  
**Factuality track:** [SA-FACT-001 / #10531](https://github.com/Luminous-Dynamics/symthaea/issues/10531)

> Do not combine the resources below into one pooled “South African language score”. They measure different constructs, were created under different conditions, and have different maturity and licensing status.

## 1. Comparison matrix

| Resource | Scope / language claims | Tasks / useful measures | Current admission status | Strategic role |
|---|---|---|---|---|
| **Multi-Hall-SA** (Findings EACL 2026) | English + isiZulu, isiXhosa, Sepedi, Sesotho | Factual vs. hallucinated statements; temporal, entity, numerical, location errors; missed-hallucination, false-hallucination and cross-language discrepancy | Strongest direct factuality anchor. Hugging Face card declares CC BY 4.0. Pin exact revision and resolve paper/release count differences. | Primary first factuality evaluation |
| **MzansiBench** (UCT 2026 project showcase) | Project describes all 11 official written languages | 13 tasks / 16 datasets; prompt-based and fine-tuning pipelines | Emerging project. Audit exact source code, dataset matrix, licences, model revisions, result artifacts and reproducibility; do not presume every task covers every language. | Broad language/task coverage map and potential shared leaderboard |
| **Uhura** (LREC 2026) | Six typologically diverse African languages; inspect released manifest for exact South African-language coverage | Scientific QA (ARC-Easy) and truthfulness/safety (TruthfulQA) | Open benchmark/platform; tasks human-translated from English. Translation lineage and technical-term equivalence are separate error sources. Verify dataset license and exact data/code revisions. | Scientific and safety factuality |
| **MzansiText / MzansiLM** (LREC 2026) | South African multilingual corpus and 125M-parameter decoder-only reference model; paper targets all 11 official written languages | NLU/NLG under task-specific and multilingual adaptation | Pin model/tokenizer/data revisions and licences. Task-specific BLEU and classification F1 are not comparable to reasoning/factuality scores. | Local small-model baseline and tokenizer/data analysis |
| **AFROBENCH** (Findings ACL 2025) | 64 African languages in aggregate | 15 tasks / 22 datasets across several NLP areas | Aggregate counts do not mean every language is in every task. Build a language × dataset × task × split matrix; verify individual licences. | Broad African-language reference and external comparability |
| **Afri-MCQA** (ACL 2026) | Paper: 15 African languages / 12 countries; current released index differs | 7.5k native-speaker-created cultural QA pairs; text and speech | Current Hugging Face card declares CC BY-NC 4.0. Current index lists 16 subsets including Sesotho, Setswana and a Zulu-labelled subset; isiXhosa is not listed in that reviewed index. Non-commercial conditions require separate handling. | Cultural QA and text/speech evaluation, not a general factuality substitute |

## 2. Primary sources and version-control requirements

### Multi-Hall-SA
- Paper: https://aclanthology.org/2026.findings-eacl.330/
- UCT archive: https://pubs.cs.uct.ac.za/id/eprint/1804/
- Dataset: https://huggingface.co/datasets/sello-ralethe/Multi-Hall-SA
- Dataset card declares CC BY 4.0.
- Paper reports up to 23.6% fewer hallucinations detected in South African target languages than English, and an average 59.4% reduction in cross-language gap under its knowledge-augmented condition.
- Its knowledge augmentation retrieves up to five relevant triples and is described as an idealized upper-bound setting. Do not present the figure as evidence that any deployed RAG system achieves the same gain.
- The paper distribution table and current dataset-card row count differ (3,500 statements per language condition in the paper table versus 11,373 rows in the currently visible card). Record the exact release revision and inspect actual files before reproducing or interpreting counts.

### MzansiBench
- UCT showcase: https://showcase.sit.uct.ac.za/event/showcase-2026/mzansibench-36
- Project technical overview: https://projects.cs.uct.ac.za/honsproj/cgi-bin/view/2026/karriem_vondoorze.zip/index.html
- Treat its 13-task / 16-dataset / 11-language description as the project's current stated scope, not proof of full language/task coverage or an independently reproduced leaderboard. Inspect repository state, result files and splits before use.

### Uhura
- Paper: https://aclanthology.org/2026.lrec-1.115/
- Two task families: ARC-Easy science QA and TruthfulQA truthfulness/safety.
- The paper says six typologically diverse African languages and describes human translation from English. Do not infer exact languages from the abstract; inspect released manifests and translation protocol.

### MzansiText / MzansiLM
- Paper: https://aclanthology.org/2026.lrec-1.109/
- Model: 125M parameters.
- Reported task-specific values include 20.65 BLEU on an isiXhosa data-to-text task and 78.5% macro-F1 on an isiXhosa news topic-classification task. These numbers are reference results for those tasks/configurations only, not a general language capability score.

### AFROBENCH
- Paper: https://aclanthology.org/2025.findings-acl.976/
- Code/data: https://github.com/McGill-NLP/AfroBench
- Build a per-task/per-language admission manifest; aggregate language count is not per-task coverage.

### Afri-MCQA
- Paper: https://aclanthology.org/2026.acl-long.1869/
- Dataset: https://huggingface.co/datasets/Atnafu/Afri-MCQA
- Current card declares CC BY-NC 4.0; do not use the corpus for commercial training or commercial product fixtures without separate permission. Pin the exact release and honour attribution/non-commercial restrictions.

## 3. Required dataset-admission record

No resource enters a qualification run until its manifest records:

- paper identifier, DOI, publication version and exact citation;
- repository / dataset revision or immutable commit;
- file-level SHA-256 digests and exact row counts;
- language label, script, orthographic variety, dialect/variety where known;
- task, subset, split, class distribution and source-family grouping;
- provenance of original source material and any translation/transcription stages;
- license identifier, attribution, permitted purpose and restrictions;
- raw asset modalities (text/audio/image) and access limitations;
- preprocessing, normalization, filtering, translation and deduplication lineage;
- exact model/revision, prompt, evaluator, scoring implementation and compute budget;
- train/development/test overlap, benchmark contamination and source-family leakage assessment;
- native-speaker/domain-review status and known annotation limitations.

If any field is unknown, record Unknown. Do not infer license, language identity or source rights from a paper citation or display label.

## 4. Matched baseline requirements

For a given task, compare as applicable:

1. Published paper baseline with exact settings, where reproducible.
2. Strong current open-weight multilingual baseline pinned by exact revision.
3. A local/South-African-focused baseline where task fit exists (e.g. MzansiLM for its supported tasks).
4. Zero-shot and few-shot settings, reported separately.
5. BM25, multilingual dense and hybrid retrieval where the task has evidence retrieval.
6. Native-language RAG, translate-to-English RAG and source-aware retrieval under matched evidence/context budgets.
7. Symthaea and Mycelix evidence-bundle + Symthaea variants.
8. Human/native-speaker reference or adjudicated gold labels, with disagreement reported.

Do not use one system's better tokenizer, translation, retrieval corpus, context allowance or number of calls without recording it. Use paired comparisons, source/document-clustered confidence intervals, preregistered primary endpoints and explicit multiple-comparison correction.

## 5. Metrics by lane

### Factuality / hallucination
- Per-language and per-error-type precision, recall, macro-F1, AUROC/AUPRC where appropriate.
- Missed-hallucination rate, false-hallucination rate and cross-language discrepancy.
- Evidence-span precision/recall, source attribution and unsupported-claim rate.
- Brier score, reliability, selective risk/coverage and useful abstention.

### Scientific QA / truthfulness
- Exact/rubric-scored correctness, evidence support and calibrated abstention.
- Stratify health, law, finance, politics and scientific domains; high-stakes output has a separate release gate.
- Translation/technical-term equivalence is independently evaluated.

### Broad language/task competence
- Macro-F1 / exact task metric per dataset, language, task and split.
- Worst-language and worst-task results, not only pooled averages.
- Language identification, code-switching, orthographic and entity transliteration robustness.
- Training-data/task exposure and few-shot vs fine-tuned status.

### Narrative/cultural reasoning
- Event-anchor localization, temporal/causal relation quality, perspective, source-witness attribution and multiple-interpretation recall.
- Preserve cultural context and legitimate disagreement; avoid a universal culture score.

## 6. Competitive strategy

There are two distinct goals:

**Coverage and parity:** reduce severe quality gaps across languages/tasks without lowering absolute quality in better-served languages.

**Evidence-grounded factuality:** improve correct evidence attribution, provenance independence, temporal applicability, calibrated uncertainty and abstention.

A system that reduces language disparity by becoming equally poor in every language is not an improvement. A system that scores well on translated MCQ but fails on source-native factuality or local named entities is not a robust South African knowledge system.

The first public comparison should be a per-language/per-task dashboard with raw per-case outputs, model/data revisions, confidence intervals, compute/cost and worst-group outcomes. No single aggregate “Mzansi intelligence score” should govern claims.

## 7. Implementation sequencing

1. Pin and admit Multi-Hall-SA as the first factuality dataset.
2. Build the dataset/task/language/licence manifest and reproducibility report.
3. Reproduce its published zero-shot baseline where exact model/prompt details are recoverable.
4. Add matched BM25, multilingual dense, hybrid, native-language RAG, translate-to-English and source-aware retrieval baselines.
5. Audit MzansiBench, Uhura, MzansiLM, AFROBENCH and Afri-MCQA for exact data/code revisions and licensing.
6. Compose the existing Rust-first knowledge benchmark harness; do not create a second scoring or evidence framework.
7. Expand language/task coverage only with native-speaker review and rights-cleared source material.

**Current status:** research matrix only. No dataset has been downloaded, ingested, scored, fine-tuned or independently reviewed by this change. No new leaderboard results or SOTA claims are made.


### Released row schema and scoring unit

The current viewer exposes `pair_id`, `file_id`, `fact_index`, `language`, `hallucination_type`, plus:
- `english_factual_statement`;
- `english_hallucinated_statement`;
- `target_factual_statement`;
- `target_hallucinated_statement`.

Each row contains four aligned statements. A full evaluation records four model predictions per pair row, distinguishing statement language (English/target) from statement kind (factual/hallucinated). `pair_id` is the matched fact identifier; `file_id` is a provisional source-document cluster key that must be confirmed against the exact underlying release metadata before clustered inference.

The current viewer shows four splits and 11.4k total rows, with 2.87k displayed in the selected isiZulu split; its file statistics list 127 `file_id` values. These are mutable viewer values, not a frozen versioned manifest. Pin the dataset revision and hash the exact files before using any row count as a benchmark denominator.


## 8. New EMNLP 2026 African-language resources and industry-style comparator

### 8.1 AfrIFact — end-to-end evidence-grounded fact checking

[ AfrIFact (EMNLP 2026) ](https://aclanthology.org/2026.emnlp-main.133/) covers ten African languages and English, with an explicit pipeline for information retrieval, evidence extraction and fact checking. The authors report weak cross-lingual retrieval even for leading embedding models, and greater difficulty retrieving healthcare documents than cultural/news documents. Their reported few-shot prompting gain reaches 43% for AfriqueQwen-14B; task-specific fine-tuning improves fact-checking accuracy by up to 26%.

This is the most important new companion to Multi-Hall-SA because it evaluates the **whole evidence chain**, not only detection of controlled statement perturbations. Exact language lists, South African language overlap, release revision, splits and license still require direct dataset admission.

Execution: [AFR-FACT-001 / #10544](https://github.com/Luminous-Dynamics/symthaea/issues/10544).

### 8.2 TukaBench — culturally grounded multilingual safety

[TukaBench (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.692/) evaluates jailbreak robustness across seven African languages and distinguishes:
- human-translated prompts;
- prompts adapted to African contexts then translated;
- locally authored culturally grounded prompts;
- code-switched prompts.

It labels responses Jailbroken, Refused and Deflected; the Deflected class helps separate unsafe compliance from language-comprehension failure. The paper also reports lower LLM-judge/human agreement in lower-resource languages and less commonly supported scripts. Prompt data is accessed separately from the code repository; exact language, dataset revision, access and license need admission review.

Execution: [AFR-SAFE-001 / #10545](https://github.com/Luminous-Dynamics/symthaea/issues/10545).

Safety is a distinct evaluation dimension. Do not combine jailbreak/refusal results with factuality, evidence support or confidence scores. A system that refuses everything is not a safe/helpful winner.

### 8.3 MzansiBench: clarify its coverage and maturity

The UCT project page describes 13 tasks across 16 datasets spanning all 11 South African official written languages; it includes zero-/few-shot prompting via LM Evaluation Harness and a separate encoder fine-tuning pipeline. The project reports eight fine-tuning datasets tested end-to-end and prompt-based AfroBench result reproduction under a matched setup.

- Main project: https://projects.cs.uct.ac.za/honsproj/cgi-bin/view/2026/karriem_vondoorze.zip/index.html
- UCT showcase: https://showcase.sit.uct.ac.za/event/showcase-2026/mzansibench-36

Coverage is dataset/pipeline-dependent, not a complete 13-task × 11-language Cartesian product. Treat it as a valuable South African evaluation and integration project, with an external implementation review before depending on its leaderboard outputs.

### 8.4 DataLens Africa: useful market context, not a gold SOTA scoreboard

The [DataLens Africa LLM Leaderboard](https://www.datalens.africa/llm-leaderboard) shows four benchmark columns—AfriMCQA, AfriMMLU, MasakhaNEWS and AfriMedQA—and publishes a Version 1.5 snapshot last updated 16 October 2026. The displayed ranking lists Gemini 3.8 Flash at 85.32% overall and GPT-6.1 Sol at 81.56%, among other provider-labelled systems.

Treat this as **secondary industry-style context only**. The page says it ranks models selected by the publisher and that the four displayed benchmarks form one of seven evaluation dimensions; the open page does not expose enough of the exact prompts, revisions, scoring code, inference settings and independent replication protocol to establish a definitive SOTA result. We should:
- reproduce its benchmark slices where source data and licensing permit;
- compare exact model identities and dated API versions;
- publish matched cost/token/latency and per-language metrics;
- never treat the leaderboard's pooled score as an independent scientific benchmark result.

### 8.5 Current evaluation portfolio

| Lane | Core evaluation question | Reference |
|---|---|---|
| Controlled hallucination detection | Can the system detect temporal, entity, numerical and location errors across aligned languages? | Multi-Hall-SA |
| Evidence retrieval and verification | Can it find, localize and correctly interpret supporting/contradicting evidence? | AfrIFact |
| Scientific knowledge and truthfulness | Can it answer scientific questions safely in low-resource languages? | Uhura |
| Broad South African task coverage | Which languages and tasks work, and where are data/quality gaps? | MzansiBench, AFROBENCH |
| Safety and comprehension parity | Does it avoid harmful compliance without mistaking incomprehension for refusal? | TukaBench |
| Local small-model comparison | What is achievable with locally focused small models and corpora? | MzansiLM |
| Cultural/narrative interpretation | Can it preserve source context, event structure and multiple valid interpretations? | NarraBench, LitVISTA, LitBench |

Do not create one pooled “African intelligence score”. Keep task constructs, source lineage, language, annotation origin and license visible. The portfolio is intentionally multi-dimensional: factuality, evidence quality, task competence, safety, cultural interpretation and efficiency are separate capabilities.


## 9. Industry-style African-language model snapshot (DataLens Africa, Version 1.5)

The [DataLens Africa LLM leaderboard](https://www.datalens.africa/llm-leaderboard) currently displays a Version 1.5 snapshot last updated **16 October 2026**. Its four visible benchmark columns are:
- AfriMCQA — multimodal culturally grounded MCQ, accuracy;
- AfriMMLU — human-translated MMLU across African languages, accuracy;
- MasakhaNEWS — news topic classification, macro-F1;
- AfriMedQA v2 — clinical QA, macro-accuracy.

The current published ranking shown on that page is:

| Rank | Model | Overall score |
|---:|---|---:|
| 1 | Gemini 3.8 Flash | 85.32% |
| 2 | Gemini 3.6 Flash | 83.66% |
| 3 | GPT-6.1 Sol | 81.56% |
| 4 | GPT-5.6 Sol | 80.33% |
| 5 | Grok 4.6 | 77.27% |
| 6 | Claude Opus 4.6 | 77.19% |
| 7 | DeepSeek-V4-Pro | 76.26% |

This is a **market context snapshot**, not a definitive scientific SOTA ranking. The page explicitly says it ranks models selected by the publisher, and that these four benchmark columns are one of seven dimensions. The other listed dimensions are token efficiency/cost-to-serve, cultural appropriateness, factuality/hallucination, safety, instruction-following/reasoning, and dialect/code-switch robustness. It reports 15k+ questions, 16+ languages and 32 specialties, but the visible page does not supply enough information to independently reproduce every model result (exact model revision, prompt, full split manifest, decoding, provider access and raw per-case outputs).

The four reported metrics are not one common statistical construct. Averages across subtask scores are useful for an overview but can obscure poor clinical, cultural or language-specific performance. We should:
- reproduce individual benchmark slices where the data and licenses permit;
- record exact language/task coverage and per-language outcomes;
- ask for / inspect model identifiers, inference date, prompt/config, raw outputs and cost/latency receipts;
- report score by benchmark and worst-language slice, not just overall average;
- use this leaderboard as one external comparator, not as the held-out research benchmark;
- compare our source-grounded factuality and safety work on Multi-Hall-SA / AfrIFact / TukaBench separately, because the DataLens four-column overview is not a substitute for those task-specific evaluations.

## 10. What the MzansiBench project actually contributes

The [UCT showcase description](https://showcase.sit.uct.ac.za/event/showcase-2026/mzansibench-36) clarifies MzansiBench's implementation and reported status:
- a modular evaluation library, central dataset repository and public leaderboard;
- 13 tasks across 16 datasets spanning the 11 official written languages, with special attention to nine low-resource Niger-Congo B languages;
- a prompt pipeline validated against AfroBench and reported to closely reproduce published results in a matched environment;
- a separate encoder-only fine-tuning pipeline for sequence classification, paired-text classification and token classification, validated with functional testing, published baseline comparisons and repeated seeds;
- reported substantial gaps between English/Afrikaans and low-resource Niger-Congo B languages, including serious dataset coverage gaps for Tshivenda and isiNdebele.

This materially improves its value as an external South African evaluation-integration reference. It is not proof that every one of the 13 tasks appears in every one of the 11 languages; the exact task × dataset × language × split matrix still has to be admitted and pinned. Nor does its prompt-based reproduction establish that it covers source-grounded retrieval, narrative interpretation, rights, provenance or abstention.

Project references:
- UCT showcase: https://showcase.sit.uct.ac.za/event/showcase-2026/mzansibench-36
- Technical project website: https://projects.cs.uct.ac.za/honsproj/cgi-bin/view/2026/karriem_vondoorze.zip/index.html

## 11. Competitive strategy against industry

We should not try to beat frontier models by making one pooled accuracy number higher. The programme's competitive thesis should be:

1. **Quality parity:** reduce the worst-language and worst-domain gaps without reducing strong-language absolute accuracy.
2. **Evidence-grounded factuality:** improve exact retrieval, evidence localization, provenance independence, temporal applicability, supported verdicts and useful abstention.
3. **Safety parity:** reduce harmful compliance in local languages without falsely treating misunderstanding or blanket refusal as safety success.
4. **Cultural and interpretive competence:** use source-native, locally authored and expert-adjudicated data; preserve multiple legitimate interpretations.
5. **Efficiency:** publish accuracy/calibration/safety against tokens, calls, latency, energy or hardware budget and cost-to-serve.
6. **Governance:** rights, authority, consent, update/retraction and auditable evidence lineage are hard gates, not tradeable score weights.

The initial industry-facing comparison should pin one current frontier API baseline, one open-weight multilingual baseline and one locally relevant South African baseline per task where possible. We must match data split, evidence budget, prompt exposure, context, decoding and tool access, and publish per-case outputs plus confidence intervals. A claim of superiority is allowed only for the exact task, split, language and budget tested.


| **Poly-FEVER** (Findings EMNLP 2026) | 11 languages: English, Chinese, Hindi, Arabic, Bengali, Japanese, Korean, Tamil, Thai, Georgian and Amharic; no isiZulu, isiXhosa, Sepedi or Sesotho in the released language list | 77,973 aligned Supported/Refuted claims; 22-topic metadata; prompt families, rephrasing robustness and DPR/Wikipedia retrieval | Strong global multilingual factuality comparator. Its translations are Google Cloud Translation with targeted human audit, GEMBA and COMETKiwi. Pin release revision and verify exact data license before admission. | Large-scale cross-lingual method control, not South African coverage |

## 2.1 Additional cross-lingual comparator: Poly-FEVER

[Poly-FEVER (Findings EMNLP 2026)](https://aclanthology.org/2026.findings-emnlp.540/) extends FEVER, Climate-FEVER and SciFact to 77,973 aligned claims across 11 languages. The current dataset card reports 46,808 Supported and 31,165 Refuted claims, and translation-quality metadata for each target language.

Use it as a global comparator for:
- language-aligned claim verification;
- topic-stratified analysis;
- translation-quality covariates;
- prompt/rephrasing robustness;
- retrieval-augmented factuality comparisons.

It does **not** include the four South African languages in Multi-Hall-SA. Its alignment is translation-based, whereas Multi-Hall-SA extracts facts from parallel government document pairs and generates controlled factual/hallucinated statement pairs in parallel. These datasets test different sources of cross-language failure; never pool their scores or treat either as a replacement for the other.

- Paper: https://aclanthology.org/2026.findings-emnlp.540/
- Dataset card: https://huggingface.co/datasets/HanzhiZhang/Poly-FEVER

The currently visible card does not provide a license field in the retrieved summary. Verify the dataset repository's exact license, source licensing, and data-use conditions before downloading, copying, fine-tuning, or including it in commercial fixtures. No license is inferred from public availability.

## 2.2 Multi-Hall-SA design limits to preserve in evaluation reports

The current [Multi-Hall-SA card](https://huggingface.co/datasets/sello-ralethe/Multi-Hall-SA) states:
- Each row has the English factual statement, English hallucinated counterpart, target factual statement and target hallucinated counterpart.
- Facts are extracted from aligned document pairs in parallel rather than translating the English hallucination, specifically to avoid translation artefacts that a detector could exploit.
- Hallucinations are synthetically perturbed across four error types and balanced by construction.
- The benchmark's stated purpose is to measure the **cross-lingual detection gap**; its English condition is intentionally not difficult.
- A `prompts/` configuration includes recovered fact-extraction and hallucination-generation templates, model/decoding details and wording artefacts. The dataset card says detection prompts were not recorded with the generation pipeline.

Consequences:
1. Report absolute accuracy and per-class metrics, but do not interpret this resource as a comprehensive real-world hallucination-difficulty benchmark.
2. Add a separate source-grounded factuality evaluation using independently authored/adjudicated claims and realistic evidence packs; do not treat synthetic perturbation detection as full fact verification.
3. Reproduce the published detection setup only where the paper specifies enough detail. A generic detection prompt found in a construction repository is not proof of exact reproduction.
4. Keep source text external/read-only; verify source and benchmark rights, revision, original government-source metadata, and human validation documentation before use.
5. Audit perturbation-generator artefacts and near-duplicate leakage; use source-family/document splits and separately report synthetic vs independently authored cases.

Dataset card: https://huggingface.co/datasets/sello-ralethe/Multi-Hall-SA  
Paper: https://aclanthology.org/2026.findings-eacl.330/

## 8. New international comparators and detector method track

### Poly-FEVER — translated multilingual factuality

[Poly-FEVER (Findings EMNLP 2026)](https://aclanthology.org/2026.findings-emnlp.540/) aligns 77,973 Supported/Refuted claims across 11 languages, extending FEVER, Climate-FEVER and SciFact. It uses Google Cloud Translation with targeted human validation, prompt variation and DPR retrieval over Wikipedia.

Use this as a broad **translated-claim** comparator, distinct from Multi-Hall-SA's parallel government-document facts. Pin exact data revision/digests, splits and upstream source rights. Report translation quality/lineage as a separate factor; do not combine Poly-FEVER and Multi-Hall-SA into one score.

Dataset: https://huggingface.co/datasets/HanzhiZhang/Poly-FEVER

### MultiHaluDet — hidden-state probing

[MultiHaluDet (MeLLM 2026)](https://aclanthology.org/2026.mellm-1.6/) uses full hidden-state trajectories from frozen models with a calibrated stacked detector. The paper reports up to 98.55% AUROC on English HaluEval/TriviaQA using Mistral-7B and LLaMA2-7B, with transfer experiments in French, Bangla and Amharic.

This is a white-box method requiring access to internal model states. Treat it as an optional open-weight detector baseline, not a black-box/frontier or source-verification equivalent. Reproduce it only with the same model revision and held-out source families; add a direct South African-language transfer experiment.

### LitBench — human preference for generated writing

[LitBench (EACL 2026)](https://aclanthology.org/2026.eacl-long.362/) reports 43,827 training story pairs and 2,480 test pairs; its strongest tested zero-shot judge achieved 73% human preference agreement, versus 78% for specialized reward models.

This is a creative-writing preference benchmark, not literary knowledge or historical/cultural correctness. Keep it in a separate generation/evaluator lane.

### Method-level comparison requirement

The new portfolio should compare independent channels:
- external evidence relevance, provenance lineage and time-scope verification;
- internal model hidden-state / HDC temporal uncertainty signals;
- paired cross-view and counterfactual invariance tests;
- calibration, selective prediction and explicit abstention.

A detector that flags an answer as suspicious has not established the correct answer. A source verifier that finds a matching passage has not established its relevance or authenticity. The research question is whether the channels complement each other under matched compute/context budgets and preserve failure explanations.


## 9. African evidence retrieval and safety tracks

### AfrIFact — end-to-end African-language fact checking

Paper: https://aclanthology.org/2026.emnlp-main.133/

AfrIFact covers ten African languages plus English and includes the full automated fact-checking chain: information retrieval, evidence extraction and fact checking. The paper reports that current embedding systems struggle with cross-lingual retrieval and that cultural/news documents are easier to retrieve than healthcare material. It reports up to 43% gain from few-shot prompting in AfriqueQwen-14B and up to 26% from task-specific fine-tuning under the paper's settings.

**Use:** the end-to-end evidence-retrieval track for Mycelix evidence-plane + Symthaea. Measure retrieval Recall@k / MRR / nDCG; source-family recall; evidence-span precision/recall; claim-evidence relation macro-F1; verdict macro-F1; calibration and selective coverage. Report healthcare separately from culture/news.

**Admission:** exact language list, overlap with South African language targets, dataset revision, file digests, license, source rights, and split lineage remain to be checked before use. Do not infer South African coverage from the aggregate “ten African languages” description. Do not use the paper's reported gains as Symthaea/Mycelix results. Issue: [AFR-FACT-001 / #10544](https://github.com/Luminous-Dynamics/symthaea/issues/10544).

### TukaBench — multilingual safety and comprehension

Paper: https://aclanthology.org/2026.emnlp-main.692/  
Code: https://github.com/McGill-NLP/tukabench  
Dataset: https://huggingface.co/datasets/Victor-Akinode/tukabench

TukaBench evaluates seven African languages plus English. Paper prompt settings include human translation, culturally adapted translation, locally curated prompts and code-switching. Labels separate **Jailbroken**, **Refused** and **Deflected** (off-target/incoherent comprehension failure).

The currently visible Hugging Face release declares **CC BY-NC 4.0** (non-commercial research use only) and exposes three configs—`afri-jbb-harm`, `afri-jbb-benign`, `afri-jbb-culture`—with 100 prompts per config/language split. Visible splits include English, Amharic, Hausa, Igbo, Nyanja, Swahili, isiXhosa and Yoruba. The code repo says prompt data is not included in the repo and refers users to separate data access. Verify exact access and whether all paper settings are released before reproducing.

**Use:** safety/comprehension qualification, not truth evaluation. Measure harmful compliance, appropriate refusal, false refusal on benign prompts, deflection/comprehension failure, safe helpfulness and human-vs-judge agreement per language/prompt origin. Do not maximize refusal at the expense of allowed-task usefulness. Native-speaker annotation is critical because the paper finds judge-human agreement degrades for lower-resource languages/scripts. Keep harmful prompts in restricted evaluation assets, not general source fixtures. Issue: [AFR-SAFE-001 / #10545](https://github.com/Luminous-Dynamics/symthaea/issues/10545).

### How these tracks fit together

| Track | Primary capability | Must not be confused with |
|---|---|---|
| Multi-Hall-SA | Detect controlled factual perturbations in parallel source-native statements | Full evidence retrieval / real-world fact-checking |
| AfrIFact | Retrieve evidence, extract spans and verify a claim | A pure detector or refusal benchmark |
| TukaBench | Resist jailbreaks while preserving benign helpfulness and comprehension | Factual truth or evidence support |
| MzansiBench / Uhura / AFROBENCH | Broad language/task and scientific QA coverage | One universal South African score |

Use shared run receipts and identity/provenance fields, but preserve separate labels, metrics, rights and safety thresholds.
