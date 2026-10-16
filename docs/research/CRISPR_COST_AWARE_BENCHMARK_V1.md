# CRISPR Cost-Aware Evidence Bench — Research Contract v1

**Status:** research contract / proposed first proving vertical  
**As of:** 2026-10-16  
**Ownership:** Symthaea = discovery, modeling and evaluation design; Mycelix = consent, provenance, evidence exchange and replication lineage  
**Evidence ceiling:** proposal until data rights, implementation, benchmark results and any prospective external validation are separately qualified.

## Executive thesis

The highest-value first step is not to invent a new CRISPR editor or claim Symthaea can already optimize gene therapies. It is to establish whether Symthaea's composed scientific-discovery system improves decisions over strong specialist tools while preserving uncertainty, safety constraints, data rights and a reproducible evidence trail.

The target question is:

> Can a Symthaea-coordinated, ancestry-aware model council and information-gain loop identify better-supported genome-editing options, generalize to held-out biological contexts, and reduce experimental resources needed to establish a result—without worsening the safety profile?

The near-term deliverable is a Rust-first, reproducible **CRISPR Cost-Aware Evidence Bench** over lawfully usable published datasets. It is not a treatment recommender, clinical decision system, wet-lab automation controller or claim of therapeutic efficacy.

Compose the existing [Symthaea biology discovery program (#6271)](https://github.com/Luminous-Dynamics/symthaea/issues/6271), rather than building a second scientific-agent framework. Mycelix's [prospective-evidence federation contract (#3331)](https://github.com/Luminous-Dynamics/mycelix/issues/3331) is the intended evidence-custody boundary.

Machine-readable planning contract and benchmark gates: [`docs/research/data/CRISPR_OFFTARGET_BENCHMARK_CONTRACT_V1.json`](data/CRISPR_OFFTARGET_BENCHMARK_CONTRACT_V1.json). Its validation flags deliberately remain false until the exact data artifacts are retrieved, rights-reviewed, and independently replayed.

The companion [CRISPR triage scorecard and value-of-information protocol](CRISPR_TRIAGE_SCORECARD_V1.md) turns the paradigm shift into preregistered tracks for native nomination, conditional ranking, review-budget efficiency, calibration/abstention, shift robustness, Symthaea incrementality, and adversarial verification. Its machine-readable counterpart is [`docs/research/data/CRISPR_TRIAGE_SCORECARD_V1.json`](data/CRISPR_TRIAGE_SCORECARD_V1.json); both remain planning contracts, not execution evidence.

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

**Artifact-rights audit (2026-10-16):** the retrieved Zenodo record lists `offtarget_prediction_benchmark_zenodo_artifacts_v1.zip` (190.7 MB) with displayed MD5 `611cb28ab0516e471c5dc59ea44a330a`, but its visible Rights/License section yielded no license value. Treat this as **unresolved**, not as evidence of either permission or prohibition. Do not retrieve or ingest the archive for model/benchmark development until explicit record metadata or written rights-holder confirmation establishes permission for the intended use. A declared MIT license for the reproduction code does not establish the dataset/artifact license. The compact GitHub-hosted truth table also needs its own data-rights check. Archive bytes were not downloaded and SHA-256 has not been verified.

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

## Quantitative benchmark targets and critical interpretation rules

This section fixes measurable research targets before implementation. Thresholds labeled **engineering targets** are proposed decision rules for this project, not field-wide standards, clinical safety thresholds, or known achievable Symthaea performance. Final acceptance must bind an exact data/model/source generation and remain open to a well-supported null result.

### First critical split: nomination is not ranking

The Kaufmann et al. benchmark exposes a critical methodological distinction:

- **Nomination/coverage:** did a genome-wide search method include a measured positive site in its candidate set?
- **Ranking/scoring:** conditional on a site being in the scored candidate universe, how high did a predictor rank it?
- **Assay confirmation:** did an experimental measurement method detect and quantify the event under its specified conditions?

These are different questions and require separate metrics. The paper's starting union of native search-tool outputs contained 324,375 unique sites, but 46 validated positive sites (representing 53 raw positive records) were absent before the authors explicitly added validated positives through 66 coordinate-resolved rows. This addition was intentional so rankings could be compared independently of nomination coverage. It means a ranking replay of the positive-seeded universe cannot be reported as proof that the original search methods found all positive sites. The exact final row/site count after this step must be reconstructed from source artifacts, not inferred from the pre-seeding count. Source: https://journals.sagepub.com/doi/10.1177/10430342261466407

Accordingly, reports must publish separate values for:
1. pre-seeding nomination coverage on the measured-positive reference subset;
2. scoring/ranking quality conditional on the explicitly declared universe;
3. the count and identities of reference positives injected into the union for rank-only analysis;
4. assay sensitivity/precision where appropriate independent experimental evidence exists.

The measured-positive set is itself incomplete. A benchmark miss is a miss relative to that reference cohort, not proof of universal biological falsehood; an unassayed candidate is unknown, not a true negative.

### Source-replay acceptance targets

**Target A — exact source and data identity**
- Record source repository URL and exact commit SHA, manuscript DOI/version, artifact identifier, retrieval timestamp, artifact byte size and SHA-256 digest.
- Audit code license and dataset/artifact rights separately. A code repository's MIT license does not automatically license bundled, linked or third-party biological data.
- Pin reference genome/build, cohort transformations, duplicate key, threshold, and each candidate universe as independent immutable inputs.
- Unresolved use rights, digest mismatch or ambiguous cohort provenance blocks qualification.

**Target B — source cohort reconstruction**

Reconstruct these reported anchors independently:
- 3,827 sequence-resolved deep-sequenced observations across 26 guide/Cas9 combinations;
- 383 observations at/above the paper's operational 0.1% indel threshold;
- after repeated-site collapse, 1,806 unique labeled sites including 118 positives;
- no-bulge ML subset: 1,768 unique labeled sites including 114 positives;
- published sequence reconstruction agreement: 3,824 / 3,827 (reported as 0.9992);
- pre-seeding search-union omission: 46 validated sites / 53 raw positive observations; 66 coordinate-resolved rows were added for positive seeding.

The first five quantities must match the source's exact declared transformation/collapse semantics. For values reported only as rounded percentages, reproduce within **1.0 percentage point**; when exact result artifacts exist, compare exact integer counts and compute metrics locally instead of treating paper-rounded values as ground truth. Record both pre-seeding and post-seeding counts separately.

**Target C — independent metric reconstruction**
- Independently recalculate recall@K, precision@K, false-negative counts, coverage, and PR-AUC only where labels/universe make the metric well-defined.
- Reproduce the source's primary per-guide/Cas9 ranking summaries at matched K and exact cohort; do not replace them with pooled-row averages without explicitly reporting both.
- A report is not complete until the full expected prediction/result census and all exclusions reconcile.
- Negative controls for altered cutoff, candidate-universe mismatch, label mutation and missing rows must fail closed.

### Engineering targets for a Symthaea-added decision policy

These are **aspirational gates**, not predictions about current capability:

- **Fair comparison:** compare at least one source-supported transparent baseline and the strongest reproducibly available specialist baseline under exactly the same universe, split, K/budget, outcome labels and preprocessing. Freeze inclusion reasons for all candidate tools before evaluating outcomes.
- **Primary impact target:** on pre-registered guide/Cas9-group holdouts, either (a) improve macro-averaged recall at a fixed, predeclared candidate budget by **at least 5 absolute percentage points** over the best fair baseline, or (b) use **at least 20% fewer candidate review slots** to reach the baseline's declared recall target. These alternatives are not interchangeable; report the complete precision/false-negative trade-off.
- **Uncertainty:** report group-aware bootstrap or another justified interval that respects guide/Cas9 clusters. A headline point estimate without an uncertainty interval does not qualify a win. Require the pre-registered primary interval to exclude no improvement before using “demonstrated improvement”; otherwise report inconclusive/null.
- **Partial-label robustness:** when unknown outcomes exist, report a paired worst/best-case Recall@K-delta bound on the same candidate universe and shared label mapping. Distinguish an observed-label improvement from a label-robust improvement; the latter requires the paired worst-case lower bound to exceed zero.
- **Generalization:** any gain must be evaluated without group leakage and repeated on at least one independent, use-permitted cohort before describing it as robust across datasets. A 26-group cohort alone is a first test, not broad generalization.
- **No hidden regression:** publish group-level outcomes and predefined subgroup checks. Do not let gains for common groups hide a collapse for a specific held-out group.
- **Ablation:** keep the strongest specialist model fixed and compare it with and without Symthaea's selection/coordination layer. Additional gains from a stronger wrapped model do not establish a Symthaea contribution.
- **Operational burden:** measure data/compute/runtime and candidates reviewed. Candidate-slot reduction is a laboratory-review-burden proxy only; it is not a measured dollar saving or a treatment-cost reduction.

The top-500 and top-1,250 figures from the paper (approximately 77% and 83% recall for top methods, in the particular evaluation) are useful reference points, not universal standards or a guarantee about a held-out split. The proposed +5 pp / 20% targets must be declared before examining new held-out results. If the sample or cluster count cannot support a stable interval, that is a benchmark limitation; do not weaken the statistical rule post hoc.

### Additional frontier: orthogonal measurement is part of the system

The field is moving beyond purely in-silico ranking toward explicit comparison of computational nomination and experimental nomination methods. Kinney et al., *Nature Communications* (2026), reported UNCOVERseq results of 97.6% analytical sensitivity and 78% precision on an inter-method confirmation benchmark; the sensitivity/precision calculations cover two guide RNAs, so these numbers must not be treated as universal across editor, cell or assay contexts. The paper discusses the need for orthogonal methods and limits imposed by library preparation, library complexity, read depth and analytical procedures: https://www.nature.com/articles/s41467-026-74623-7

**Important data-rights gate:** the UNCOVERseq paper explicitly restricts the deposited study data to non-commercial research and prohibits commercial use—including training/developing predictive models or integrating into commercial software—without prior written permission. Until written permission for the intended project/use is secured, this project must not ingest those datasets or use them for model training/development. The published paper may inform a literature review, but its restricted measurements are not a permitted CRISPR training/evaluation corpus by default. The article had an author correction dated 12 October 2026; use the current version and preserve the correction lineage: https://doi.org/10.1038/s41467-026-78544-3

The paradigm shift here is a **measurement-aware evidence system**: computational nomination, ranking, orthogonal observation and independent assessment are connected but never collapsed into one “off-target score.” Actual assay selection/execution remains under qualified external human/laboratory authority and outside this computational benchmark.

### Cost claims: targets we can and cannot measure in v1

V1 can measure **reproducibility, ranking at a fixed candidate budget, candidate-slot burden, software runtime and computational resource use**. It can estimate neither actual laboratory savings nor drug-price reduction unless those downstream quantities are observed in a suitable study.

Later programs should separately measure:
- cost per independently qualified finding;
- assays/candidates needed to meet a predeclared evidence threshold;
- repeat-run and batch failure rates;
- manufacturing yield and cost of goods;
- quality-control and delivery burden;
- total treatment-pathway costs, payer/reimbursement effects and patient access.

Do not extrapolate percentage reductions from simulated candidate slots directly into monetary savings or a therapy's sale price.

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

**Partial-label identification is mandatory when outcomes are incomplete.** Report observed-label metrics plus worst/best-case label-assignment bounds within the exact candidate universe. If P is the eligible measured-positive count, pK the measured positives in the top K, U the unknown/unassayed candidate count in that universe, and uK the unknowns in the top K, report conditional recall bounds [pK / (P + U - uK), (pK + uK) / (P + uK)] and precision bounds [pK/K, (pK+uK)/K] where K slots exist. These are not confidence intervals, posterior probabilities, or safety claims; they cannot bound positives omitted from the candidate universe. Never convert unknown/unassayed outcomes into negatives merely to obtain a point estimate. The executable metric kernel is proposed at [scripts/crispr_triage_verifier.rs](../../scripts/crispr_triage_verifier.rs); the detailed contract lives in [CRISPR_TRIAGE_SCORECARD_V1.md](CRISPR_TRIAGE_SCORECARD_V1.md).

For policy comparisons, use paired label-sensitivity bounds on the same candidate IDs and outcome mapping, assigning each unknown site's latent outcome jointly across both rankings. Separate policy bounds are not a substitute. The project may claim **observed-label improvement** when the preregistered group-aware statistical interval supports it; it may claim **label-robust improvement within the declared universe** only when the paired worst-case Recall@K-delta lower bound is above zero. Neither statement accounts for true sites absent from that universe.

**Exact-budget cardinality:** every policy compared at budget K must supply at least K candidates, and K must be positive. If it cannot, label the metric `not_estimable_at_budget_K`; do not evaluate a shorter list and report it as the same budget.

**Multi-objective decision frontier:** report a Pareto frontier over preregistered, context-appropriate metrics rather than manufacturing a universal CRISPR score. Possible axes include macro recall at a fixed candidate budget, review slots at fixed recall, worst-group performance, label-sensitivity bounds, valid calibration loss, and measured compute. Every compared policy must use the same objective definitions, units, directions, episodes and availability rules. Do not replace missing/not-estimable values with zero; list excluded policies and reasons. Pareto membership is not, by itself, proof of improvement.

The dependency-free helper [scripts/crispr_decision_frontier.rs](../../scripts/crispr_decision_frontier.rs) implements objective-direction-aware dominance. Its ten Rust unit tests are authored but not compiled or run in this authoring environment because rustc is unavailable.

Context matters: recent reviews stress that off-target activity and its clinical interpretation differ by endpoint, editing system, cell/tissue context, and intended use. See [Stewart et al., Nature Genetics (2026)](https://doi.org/10.1038/s41588-025-02428-3) and [On-target and off-target activities of CRISPR therapeutics across scales, Trends in Biotechnology (2026)](https://doi.org/10.1016/j.tibtech.2026.07.015). V1 is a computational triage benchmark, not a universal biological or clinical-risk model.

Keep efficiency, specificity, bystander outcomes, indels, delivery, immune outcomes, manufacturability and economics as separate axes unless an externally justified utility function is preregistered. Missing values mean “not measured,” not zero risk.

## Rust-first implementation shape

Reuse the scientific-discovery, exposure, prospective-scorecard, causal, evidence-plane and model-invocation owners already identified by BIO-MILLENNIUM-000 (#6271). Do not create a second generic model council, information-gain engine, evidence ledger or biology ontology.

Initial tooling is deterministic and report-only. A run manifest binds task/data source generation, exact split and grouping policy, model/provider/version/config where knowable, source/transformation digests, metrics, uncertainty procedure, full prediction/result census and stable diagnostics for excluded/invalid examples. Prediction, observation, assessment and replication remain distinct lanes.

Keep orchestration, schema validation, split construction, metric calculation and receipt verification Rust-first. A specialist runtime may be called through a pinned/documented adapter, but it must not become an undeclared dependency or erase output origin/configuration. This contract requires no production Python dependency.

A dependency-free, synthetic-testable metric kernel has been added at [scripts/crispr_triage_verifier.rs](../../scripts/crispr_triage_verifier.rs). It currently validates typed inputs and metrics rather than parsing the JSON contract or hashing artifact bytes; the eventual runner must supply a strict input parser, trusted byte hashing, manifest binding and independent replay. The current authoring environment lacks `rustc`, so the committed Rust tests have not been compiled or run here.

## Qualification gates

**Gate 0 — source/rights audit.** Every dataset/model input has exact identity, terms/license, retrieval date, transformation lineage, permission and limitations. Unresolved rights block use. No patient-identifying or sensitive data enter V1.

**Gate 1 — split/label audit.** Independent review checks target leakage, duplicates, paper/batch ancestry, label semantics, invalid negatives, missingness and known training-data contamination. Deliberate leakage fixtures must make the verifier fail.

**Gate 2 — reproducible baseline.** At least one task is evaluated against a simple baseline and a qualified specialist comparator. Record the exact toolchain/configuration and every terminal disposition. Queued, skipped, or stale-head CI is not a pass.

**Gate 3 — ablation and prospective commitment.** Preregister model/configuration, split, ranking, metrics, exclusions, calibration and resource budget. Evaluate sealed predictions only after holdout/prospective results are available. Distinguish Symthaea's contribution from the external model's.

**Gate 4 — independent review.** An evaluator that did not generate predictions replays the data/metric contract and checks output census, lineage and negative controls.

**Gate 5 — external laboratory validation (future only).** Any physical study requires qualified collaborators, institutional/laboratory authorization, appropriate biosafety and ethics review, justified validation design and independently captured outcomes. A computational result, registry record, publication or digest grants no experimental or clinical authority.

**Gate 6 — incomplete-label sensitivity.** Whenever unknown/unassayed candidates exist in a fixed candidate universe, report partial-identification bounds or an explicit not-estimable state; do not imply completeness beyond the observed/reference scope.

**Gate 7 — exact fixed-budget cardinality.** For any fixed K comparison, every compared policy must provide at least K candidates and K must be positive; otherwise the policy metric is not estimable at that budget.

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


## Additional gate: probability-sampled verification audit (T15 / G15)

Outcome labels can be selectively observed because candidate nomination or rank affects which sites receive experimental verification. Complete-case metrics on that observed subset can therefore be optimistic or otherwise unrepresentative. Add a retrospective audit lane that freezes the full ranking and candidate frame, then samples candidates with known, nonzero inclusion probabilities under a declared outcome-blind design. This is motivated by partial-verification bias in diagnostic evaluation ([methodological overview](https://pubmed.ncbi.nlm.nih.gov/35124189/)) and the CRISPR-specific separation between candidate nomination and conditional ranking ([Kaufmann et al., 2026](https://doi.org/10.1177/10430342261466407)).

For audited candidate i with label y_i and inclusion probability pi_i, estimate totals using sum(y_i / pi_i); separately calculate the corresponding top-K positive total using only ranks in the frozen top K. Form ratio estimates for recall and precision, and publish naïve audit-subset metrics beside them. The count estimators require a valid probability sample; the ratio estimates are not finite-sample unbiased. The point-estimate kernel does not compute confidence intervals.

**Hard requirements:** independently verify the complete rank census; freeze frame, ranking and K before outcome reveal; provide known, nonzero inclusion probabilities for every target-frame candidate; make selection outcome-blind conditional on the declared design; retain unknown labels as unknown; reject convenience samples with missing propensities and zero-probability strata. These estimates apply only within the declared frame and do not bound positives omitted from all nomination tools. Real-cohort claims require design-appropriate uncertainty, group-aware evaluation, source permissions and independent replay.

Implementation: [scripts/crispr_verification_bias.rs](../../scripts/crispr_verification_bias.rs). Scorecard and synthetic fixture: [CRISPR_TRIAGE_SCORECARD_V1.md](CRISPR_TRIAGE_SCORECARD_V1.md), [machine-readable scorecard](data/CRISPR_TRIAGE_SCORECARD_V1.json), and [CRISPR-OT-008 #7633](https://github.com/Luminous-Dynamics/symthaea/issues/7633). The Rust kernel remains authored but uncompiled/unexecuted; the F14 arithmetic check is synthetic only.


Independent estimator check: the conditional Bernoulli-design oracle enumerated 2,048 deterministic finite-population fixtures and 128,512 non-empty sample subsets, and confirmed design-unbiased Horvitz–Thompson candidate/positive counts (including top-K counts) to floating-point tolerance under the enumerated designs. This is mathematical estimator evidence only; it is not Rust execution, a real audit-design verification, confidence-interval coverage, biological evidence, or Symthaea improvement.


Resolved-label guard: the Rust kernel accepts only positive, assay-defined-negative, or explicitly unknown outcome states; it rejects an unknown row and requires inclusion probabilities for the final resolved-outcome sample, not just test invitations. The sampling-design verifier must handle nonresponse/assay failure explicitly; this kernel does not estimate those propensities.
