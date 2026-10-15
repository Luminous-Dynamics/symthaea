# CRISPR Cost-Aware Evidence Bench — Research Contract v1

**Status:** research contract / proposed first proving vertical  
**As of:** 2026-10-15  
**Ownership:** Symthaea = discovery, modeling and evaluation design; Mycelix = consent, provenance, evidence exchange and replication lineage  
**Evidence ceiling:** proposal until data rights, implementation, benchmark results and any prospective external validation are separately qualified.

## Executive thesis

The highest-value first step is not to invent a new CRISPR editor or claim Symthaea can already optimize gene therapies. It is to establish whether Symthaea's composed scientific-discovery system improves decisions over strong specialist tools while preserving uncertainty, safety constraints, data rights and a reproducible evidence trail.

The target question is:

> Can a Symthaea-coordinated, ancestry-aware model council and information-gain loop identify better-supported genome-editing options, generalize to held-out biological contexts, and reduce experimental resources needed to establish a result—without worsening the safety profile?

The near-term deliverable is a Rust-first, reproducible **CRISPR Cost-Aware Evidence Bench** over lawfully usable published datasets. It is not a treatment recommender, clinical decision system, wet-lab automation controller or claim of therapeutic efficacy.

Compose the existing [Symthaea biology discovery program (#6271)](https://github.com/Luminous-Dynamics/symthaea/issues/6271), rather than building a second scientific-agent framework. Mycelix's [prospective-evidence federation contract (#3331)](https://github.com/Luminous-Dynamics/mycelix/issues/3331) is the intended evidence-custody boundary.

## Why now: research frontier

These publications motivate the research questions. None proves Symthaea is competitive, and none is local qualification evidence.

1. **AI-designed genome editors are experimentally plausible.** Madani et al., Nature (2025), reported human-genome editing using a programmable editor designed with AI models of biological sequences: https://www.nature.com/articles/s41586-025-09298-z
2. **Precision is separate from raw efficiency.** Chauhan et al., Nature (2025), reported a prime editor with comparable efficiency and up to 60-fold fewer insertion/deletion errors than prior systems in the tested settings: https://www.nature.com/articles/s41586-025-09537-3
3. **Editor engineering is multi-objective.** Tao et al., Nature Biotechnology (2026), used AI-guided redesign of reverse-transcriptase domains and reported improved stability/expression and increased editing efficiency in multiple experimental contexts: https://doi.org/10.1038/s41587-026-03149-6
4. **Tissue-selective delivery is a major translational bottleneck.** Ngo et al., Nature Biotechnology (2026), review established and emerging delivery mechanisms and remaining limits on tissue-selective editing: https://www.nature.com/articles/s41587-025-02945-w
5. **Delivery barriers include more than organ accumulation.** A 2025 review of extrahepatic lipid-nanoparticle delivery discusses formulation, biodistribution, endosomal escape, immunogenicity and scale-up: https://www.sciopen.com/article/10.26599/NR.2025.94907422
6. **Manufacturing affects accessibility.** A 2025 review of automated cell-therapy manufacturing identifies labor, contamination risk, scalability and tech transfer as operational constraints: https://doi.org/10.1016/j.jconrel.2025.02.057
7. **Price is not production cost.** CMS documents a $2.2 million US list price reported for Casgevy and an access model using outcomes-based agreements. This illustrates the distinction between scientific performance, delivery burden, list price, payer economics and patient access: https://www.cms.gov/priorities/innovation/cgt-access-model-frequently-asked-questions

The implication is not “AI solves CRISPR.” Editor design, precision, delivery, manufacturability, evidence quality and economics must be evaluated together, with independent evidence for each claim.

## Paradigm shifts to test

### 1. From best editor to best qualified treatment system

A candidate cannot be judged by on-target efficiency alone. The relevant system includes edit class, target and cell context, specificity, bystander outcomes, delivery, transient exposure, immune response, manufacturing requirements, patient eligibility and uncertainty.

There is no universal scalar score that may trade away a safety constraint. Report a vector of outcomes and a Pareto frontier. Higher editing efficiency coupled with unacceptable specificity is not an improvement.

### 2. From editor-first to delivery-aware design

Delivery is not a downstream checkbox. Represent separate evidence for reaching the intended tissue or cell population, productive intracellular delivery, intended editing in relevant cells, unwanted exposure elsewhere, immune/tolerability outcomes where measured, and formulation manufacturability.

Predicted biodistribution remains distinct from measured biodistribution. An in-vitro result cannot establish in-vivo tissue selectivity.

### 3. From random data splits to biological generalization

Random-row splitting can overstate performance when near-duplicate sequences, target loci, experimental batches, cell lines, papers or related model/data ancestry appear on both sides.

Use provenance-aware, group-held-out evaluations. Depending on the task, separate by target/locus, near-duplicate cluster, assay batch, paper, laboratory, cell context, editor family or publication time. Report which leakage risks could not be ruled out.

Retrospective success must not be called prospective discovery. Later source/model exposure must not silently rewrite historical results.

### 4. From model agreement to ancestry-aware disagreement

Compare specialist models and simple baselines rather than rebuilding all frontier protein and RNA models. Record exact versions where knowable, inputs, configurations, exposure limits, output artifacts and common ancestry.

Several models using overlapping data are not independent witnesses. Disagreement can reveal an applicability boundary or help select a discriminating measurement.

### 5. From plausible recommendations to preregistered, falsifiable predictions

Before holdout outcomes are revealed, freeze task, data generation, model/configuration, split identity, candidate ranking, uncertainty, metrics, exclusions and scoring rule. Retain the complete evaluation census, including failures, null results, non-convergence, invalid inputs and abstentions.

Symthaea must not choose the winning metric after seeing results, revise candidates and receive retroactive prospective credit, or act as its own independent verifier.

### 6. From “lower price” to cost per qualified result and cost of care

Keep separate: compute/data-preparation cost; experiments per qualified finding; cost of failed or non-reproducible experimental work; manufacturing and quality-control cost; delivery and administration burden; downstream care/monitoring; and list price, reimbursement and patient expense.

The first benchmark can measure research resources and validated-discovery yield. It cannot infer the final price of a therapy from a model score.

### 7. From closed silos to consent-bound evidence commons

Mycelix should preserve source lineage, permissions, protocol identity, artifact digests, attribution, assessments, replications, contradictions and null outcomes. Sensitive patient/genomic data must not become public merely because hashes or decentralized storage are used. V1 uses public, lawfully usable datasets; any sensitive-data collaboration requires an appropriate lawful basis, access controls, consent/governance and institutional review.

A digest can detect an artifact change under a declared procedure. It does not prove biological truth, representativeness, clinical relevance, legal permission or independence.

## V1 scope: benchmark published evidence first

The first pass is a **read-only computational benchmark**, not new biological design generation.

### Benchmark families

1. **On-target outcome prediction** on published CRISPR/Cas9 datasets, preserving label and assay context.
2. **Off-target assessment** using genome-wide or deep-sequencing datasets where available. Dataset construction must be audited: an unreported off-target is not automatically a verified biological negative.
3. **Prime-editing outcome prediction**, including efficiency and unintended editing outcomes where published measurements support them.
4. **Cross-context generalization** across held-out loci, assays, cell contexts, laboratories or editor families.
5. **Uncertainty and abstention**: identify when inputs are outside the evaluation envelope or confidence is poorly calibrated.
6. **Experiment prioritization**: compare fixed candidate selection against a declared information-gain/discrimination policy, using only information available at selection time.

Candidate sources for audit:
- GUIDE-seq, CIRCLE-seq and related off-target benchmark literature: https://academic.oup.com/bib/article/24/3/bbad131/7130974
- Multi-method off-target dataset compilation: https://www.nature.com/articles/s42003-025-08275-6
- Prime-editing data and context-generalization work: https://www.nature.com/articles/s41587-024-02268-2
- DeepPrime7 repository and evaluation assets: https://github.com/yumin-c/DeepPrime7

These are candidate sources, not pre-approved inputs. Before ingestion, record publisher/source, version and retrieval date, license and use/redistribution permissions, exact artifact identity, transformations and limitations. Unresolved rights block use. Do not redistribute source data or model weights without confirmed rights; prefer source-hosted artifacts or reproducible download instructions.

### Baselines and fair comparison

At minimum compare:
- a transparent task-appropriate baseline;
- a strong published specialist model;
- a specialist ensemble where comparable and reproducible;
- the same specialist method plus Symthaea's coordination/selection layer;
- ablations disabling one proposed Symthaea contribution at a time.

Possible prime-editing comparators include PRIDICT2.0, DeepPrime/DeepPrime7 and OPED, subject to reproducible access, exact version, license, input compatibility and fair evaluation protocol. Relevant off-target predictors should be selected by the same criteria. An inaccessible or incomparable tool is reported, not silently omitted.

Do not claim a Symthaea win merely because it wraps a stronger external model. Separate the specialist model's contribution from improvements due to context handling, disagreement analysis, uncertainty calibration or experiment selection.

## First proving vertical: decision-centric off-target triage

Start here before attempting novel editor design. This is the most defensible initial test because a current published benchmark exposes a real decision bottleneck and distributes reproducibility artifacts.

### Published evidence anchor

Kaufmann et al. (2026), *Systematic Benchmarking of CRISPR-Cas9 Off-Target Prediction Tools Reveals Limitations and Implications for Preclinical Assessment* (Human Gene Therapy; first published online 26 July 2026), compared 14 prediction tools: https://doi.org/10.1177/10430342261466407

The paper reports:
- 3,827 deep-sequenced sites across 26 guide-RNA/Cas9 combinations, using published CRISPRoffT data and several human-cell contexts;
- 383 observations above the paper's operational indel-frequency threshold of 0.1%;
- after collapsing repeated genomic sites, 1,806 unique sites and 118 operational positives for ranked classification;
- a further no-bulge subset for some ML comparisons, with 1,768 unique sites and 114 positives;
- a shared candidate universe of 324,375 sites assembled from several genome-wide nomination tools;
- approximately 77% recall among the top 500 ranked sites and up to 83% among the top 1,250 in the reported comparison, with weak-to-moderate score/indel-frequency correlation and only modest gains from tool combinations.

These quantities are specific to that cohort, operational label threshold, site universe, and analysis. They are not universal clinical false-negative rates and do not prove a guide or therapy is safe.

The authors' public [reproducibility repository](https://github.com/maren-ha/OffTargetPredictionBenchmark) provides code and a small canonical truth-table input, with larger standardized prediction artifacts distributed through [Zenodo record 20627722](https://zenodo.org/records/20627722). Its documented workflow reproduces the analysis from standardized outputs; it does **not** rerun every predictor in its native environment. We should therefore describe a first replay as a reproduction/secondary analysis, not an independent re-execution of all prediction tools.

### Paradigm shift: rank for evidence acquisition, not merely for a leaderboard

A usual score leaderboard asks, "Which predictor has the best aggregate metric?" The more operationally meaningful question is:

> Under a fixed review/validation budget, which policy surfaces the largest fraction of independently observed off-target sites while preserving uncertainty and generalization limits?

This becomes a bounded **decision-under-measurement-budget** benchmark. Symthaea should initially operate only on published prediction outputs and recorded measurements. It must not autonomously generate wet-lab instructions, authorize experiments, make clinical recommendations, or label a guide as safe.

### Experiment contract

1. **Freeze the exact source generation.** Record the manuscript/version, Git commit, Zenodo record and artifact IDs, license/terms, retrieval date, artifact digests, schema, and any transformations. Rights or artifact identity unresolved means no qualification.
2. **Reconstruct source-defined cohorts.** Independently reproduce the full-cohort and no-bulge cohort counts and label/collapse rules from the paper. Keep the two cohorts separate where the source does. Do not mix predictor outputs calculated on different candidate universes without an explicit mapping.
3. **Use a source-faithful first replay.** Begin with the paper's standardized outputs. Any later native rerun of tools is a separate lineage requiring native environment, exact tool/version, reference genome, parameters and source input identity.
4. **Prevent guide leakage.** Random-row splits are not sufficient. Primary generalization analysis holds out complete guide-RNA/Cas9 groups; additional cell-context/lab/source-group evaluations are allowed only where IDs and cohort structure support them. Because there are only 26 guide/Cas9 combinations in this benchmark, uncertainty intervals must respect group clustering rather than pretending every site is independent.
5. **Compare fair policies at equal budget.** Include each reproducibly represented published tool, transparent score/rank baselines, simple aggregation where justified, and a Symthaea policy that uses only permitted pre-outcome information. Report when a tool or candidate universe cannot be fairly compared rather than silently dropping it.
6. **Preregister the decision rule.** Commit candidate universe, group split, rank/fusion policy, validation budget(s), primary metric, tie-break behavior, exclusions, and evaluation cutoff before exposing held-out labels. The selected policy and output list must be immutable after commitment.
7. **Rebuild metrics independently.** A separate verifier recomputes the top-K recall/precision and other preregistered metrics from exact prediction and outcome artifacts, checks all eligible groups, and reports a complete terminal disposition census.

### Primary outcome vector

Report the full vector; do not optimize one metric at the expense of hidden failure modes:

- recall at preregistered review budgets (for example, top-K site counts, interpreted only within the benchmark's candidate universe);
- precision and number of false negatives at each budget;
- precision-recall curve / average precision when labels and candidate-universe rules support it;
- performance by held-out guide/Cas9 group and supported biological context;
- uncertainty/calibration only where scores have a defensible, separately evaluated probability interpretation;
- abstention and out-of-domain behavior;
- performance and ranking stability under duplicate collapse, label-threshold sensitivity, and permitted cohort changes;
- resource cost and number of candidates reviewed per detected benchmark-positive site.

Do not use overall accuracy as the decisive measure on an imbalanced cohort. Do not treat untested or unreported sites as confirmed true negatives. A higher recall at top-K on this retrospective dataset is an algorithmic benchmark result, not proof of clinical safety.

### What would show a real Symthaea advantage?

A useful result would show that the Symthaea-assisted **selection policy**, compared with strong existing score/rank baselines under identical budget and held-out groups, improves a preregistered decision metric with uncertainty intervals and no material collapse in subgroup performance. The result must persist when the strongest external specialist model is held constant, so any measured gain can be attributed to Symthaea's coordination/selection layer rather than to merely choosing a stronger predictor.

A null result is informative. If no policy beats the best fair baseline, report that result and retain the benchmark as a qualified reference. Do not convert a small pooled-data gain into a general claim about novel editor design, safety, or treatment cost.

### Mycelix evidence record

Use the existing Mycelix-DeSci event/provenance contracts rather than a second ledger. Preserve:
- exact source, dataset, standardized prediction file, and artifact-digest identities;
- each tool's known version/configuration and common data/model ancestry;
- frozen cohort/split and selection-policy commitments;
- complete output census, exclusions and transformation lineage;
- observed-label source and the operational definition used by the paper;
- independent metric-verification receipt;
- failed, null, contradictory and superseded evaluation attempts.

These records establish lineage and reproducibility only. They do not prove data authenticity, independent biological truth, or clinical utility.

### First exit gate

The first gate passes only when the team can, from permitted source artifacts:
- reconstruct the published cohort definitions and key counts;
- independently recompute the key baseline metrics to a stated tolerance;
- pass deliberate split-leakage, label-mutation, artifact-mismatch and metric-manipulation negative controls;
- measure at least one preregistered equal-budget comparison over held-out guide/Cas9 groups;
- clearly separate source reproduction, independent reanalysis, native tool rerun, and biological validation.

No new production model type is authorized by this plan. Compose the existing biology benchmark (#6324), source/contract oracle and evidence-plane infrastructure. Open a new adapter only after the ownership audit proves an actual gap.

---

## Symthaea contribution hypotheses

These are hypotheses to test, not established capabilities.

- **Mechanistic/context graph:** link target context, edit class, editor system, assay conditions, measured outcomes, source paper and known limitations.
- **Ancestry-aware model council:** preserve disagreement and shared model/data ancestry; do not collapse outputs into unqualified consensus.
- **Calibration and applicability:** report task/context calibration and detect out-of-domain inputs. Abstention can be correct.
- **Falsifier-first evaluation:** keep competing explanations and test for leakage, assay artifacts and selection bias.
- **Information-aware prioritization:** measure whether a declared strategy increases prospectively qualified findings per experiment or resource unit.
- **Persistent negative-result memory:** failed, null, contradictory and procedurally invalid outcomes remain first-class records.
- **HDC role is initially narrow:** test hyperdimensional representations for relational/context indexing or analogy after specialist baselines are established. Do not presume HDC will outperform sequence models at raw sequence prediction.
- **CfC/active-inference role is also a hypothesis:** test whether temporal/context or action-selection components add measurable value. Do not add them merely because Symthaea contains them.

## Metrics: no single “CRISPR score”

Report, where supported by the exact labels and assay: task-appropriate ranking/discrimination; calibration and proper scoring rules; false-negative and false-positive behavior; group-held-out performance; uncertainty/abstention coverage; robustness to duplicates, label construction, missingness and assay shifts; experiments per independently qualified outcome; resource cost per qualified discovery; and realized information value per resource unit.

Keep efficiency, specificity, bystander outcomes, indels, delivery, immune outcomes, manufacturability and economics as separate axes unless an externally justified utility function is preregistered. Missing values mean “not measured,” not zero risk.

## Rust-first implementation shape

Reuse the scientific-discovery, exposure, prospective-scorecard, causal, evidence-plane and model-invocation owners already identified by BIO-MILLENNIUM-000 (#6271). Do not create a second generic model council, information-gain engine, evidence ledger or biology ontology.

Initial tooling is deterministic and report-only. A run manifest binds task/data source generation, exact split and grouping policy, model/provider/version/config where knowable, source/transformation digests, metrics, uncertainty procedure, full prediction/result census and stable diagnostics for excluded/invalid examples. Prediction, observation, assessment and replication remain distinct lanes.

Keep orchestration, schema validation, split construction, metric calculation and receipt verification Rust-first. A specialist runtime may be called through a pinned/documented adapter, but it must not become an undeclared dependency or erase output origin/configuration. This contract requires no production Python dependency.

## Qualification gates

**Gate 0 — source/rights audit.** Every dataset/model input has exact identity, terms/license, retrieval date, transformation lineage, permission and limitations. Unresolved rights block use. No patient-identifying or sensitive data enter V1.

**Gate 1 — split/label audit.** Independent review checks target leakage, duplicates, paper/batch ancestry, label semantics, invalid negatives, missingness and known training-data contamination. Deliberate leakage fixtures must make the verifier fail.

**Gate 2 — reproducible baseline.** At least one task is evaluated against a simple baseline and a qualified specialist comparator. Record the exact toolchain/configuration and every terminal disposition. Queued, skipped, or stale-head CI is not a pass.

**Gate 3 — ablation and prospective commitment.** Preregister model/configuration, split, ranking, metrics, exclusions, calibration and resource budget. Evaluate sealed predictions only after holdout/prospective results are available. Distinguish Symthaea's contribution from the external model's.

**Gate 4 — independent review.** An evaluator that did not generate predictions replays the data/metric contract and checks output census, lineage and negative controls.

**Gate 5 — external laboratory validation (future only).** Any physical study requires qualified collaborators, institutional/laboratory authorization, appropriate biosafety and ethics review, justified validation design and independently captured outcomes. A computational result, registry record, publication or digest grants no experimental or clinical authority.

No later gate is implied by passing an earlier one.

## Mycelix evidence boundary

Compose [DESCI-BIO-002 (#3331)](https://github.com/Luminous-Dynamics/mycelix/issues/3331); do not create a parallel ledger.

Expected flow:

**frozen source/task contract → prospective prediction/ranking commitment → authorized external test or holdout reveal → immutable raw-result reference → independent assessment → independent replication where available → evidence-relative projection**

Proposal, prediction, computational result, external observation, assessment, replication and any clinical/official criterion evidence remain distinct. Model agreement, contributor reputation, publication, schema validation and matching digests cannot promote evidence to a stronger class.

Mycelix governs local provenance, permissions, attribution, collaboration and evidence exchange. It does not certify a therapy or redefine external acceptance criteria.

## Cost-reduction theory of change

The immediate causal claim to test is:

**better-calibrated ranking + fewer invalid experiments + greater reproducibility → lower research cost per qualified result.**

Every arrow requires measurement. This does not imply lower medicine prices, lower clinical-administration costs or a clinically useful therapy. These require separate evidence about manufacturing yield, batch failures, quality-control burden, delivery, care pathways, reimbursement and access.

Later manufacturing optimization is a separate vertical. Do not conflate automated ex-vivo cell therapy manufacturing with in-vivo editing or assume savings transfer across platforms.

## Nonclaims

This contract does not claim:
- Symthaea outperforms specialist gene-editing models;
- any novel editor, guide, delivery formulation or therapy has been discovered;
- a predicted off-target result establishes clinical safety;
- model scores are suitable for individual medical decisions;
- lower research cost necessarily reduces final treatment price;
- a public registry makes sensitive genomic data safe to publish;
- cryptographic receipts authenticate biological truth;
- any laboratory experiment has been run, authorized or independently replicated.

Success means producing a fair, reproducible result—including a negative result—about whether Symthaea improves genome-editing research efficiency without weakening evidence or safety requirements.
