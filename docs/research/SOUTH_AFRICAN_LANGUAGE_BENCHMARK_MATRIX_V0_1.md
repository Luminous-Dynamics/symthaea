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
