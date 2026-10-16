# Enzyme Kinetics Paradigm-Shift Roadmap V0

**Status:** research proposal; not a validated platform or a claim of novel scientific discovery.  
**Scope:** a Rust-first, provenance-preserving route from published enzyme measurements to reproducible kinetic claims and honest prediction.  
**Related work:** [Industrial Enzyme Benchmark V0](INDUSTRIAL_ENZYME_BENCHMARK_V0.md), [kcat admission profile](KCAT_ADMISSION_PROFILE_V0.json), [SLAC calibration reference manifest](SLAC_CALIBRATION_REFERENCE_MANIFEST.json).  
**Machine-readable companion:** [ENZYME_EVIDENCE_GRAPH_CONTRACT_V0.json](ENZYME_EVIDENCE_GRAPH_CONTRACT_V0.json) defines the graph contract. The initial [Rust verifier](ENZYME_EVIDENCE_GRAPH_VERIFIER_V0.md) now checks structural/provenance invariants and locally acquired bytes; it does not yet replay kinetics or establish scientific validity.

## Executive thesis

The most promising change is not to build a larger predictor around a table of kcat values. It is to change what the field's computational substrate treats as the primary object.

**A kinetic parameter is not an orphan scalar. It is a claim derived from a particular catalyst, reaction, assay context, measurement series, calibration, preprocessing path, model, and uncertainty analysis.** A row containing kcat = x s^-1 is the end of a chain, not the complete evidence.

This direction is consistent with the existing standards landscape, not a replacement for it. STRENDA specifies information needed to interpret and reproduce enzymology measurements and explicitly does not define a general score of scientific data quality. EnzymeML V2 already provides a structured container for reaction conditions, molecular species, measurements/time courses, equations, kinetic models, parameters, and references. The paradigm shift proposed here is to connect these capabilities to exact-byte provenance, independent replay, evaluation by transport distance, explicit uncertainty, and decision-focused experiment selection.

The intended advantage is **reliable decision infrastructure**, not a presumption that Symthaea will outperform conventional methods. The system must be able to conclude that a conventional baseline is better, that a datum is not comparable, that a fit is non-identifiable, or that the evidence is insufficient.

## What current results do and do not establish

The pinned SLAC calibration smoke check exercises five calibration JSON artifacts at pH 3 across 25, 30, 35.5, 40 and 45 °C. It can check source-byte identity and reproduce a declared linear calibration within its stated range. That is useful pipeline-reproduction evidence, but it is one enzyme/substrate context and does not demonstrate broad enzyme-function generalization.

The related thesis' time-course analysis also illustrates why stage boundaries matter: calibration, blank/control correction, deriving concentrations, estimating rates, and fitting a kinetic model are distinct transformations. A matching final parameter does not prove that the raw-signal pathway was reproduced. A model with omitted time-dependent inactivation can produce a materially different fitted kcat; this reported example should become a future, source-matched regression test only after the exact raw data, model, and analysis lineage are pinned.

A 2026 bioRxiv preprint on enzyme-engineering benchmarks raises a related concern: aggregation across heterogeneous assay conditions can inflate apparent generalization. It is a preprint and should be treated as a motivating hypothesis, not definitive consensus. This roadmap addresses the concern directly by separating task types and requiring increasingly difficult holdouts.

## Existing work and the genuine opportunity

We must not claim that a new kinetic predictor, uncertainty estimation, or out-of-distribution testing is itself novel:

- **EnzymeML already addresses interoperable, FAIR representations** of reaction conditions, time courses, kinetic models and estimated parameters, including data flow between experimental and modeling software ([EnzymeML, Nature Methods, 2023](https://www.nature.com/articles/s41592-022-01763-1)). This project should compose with EnzymeML/SBML rather than invent a competing biochemical document format.
- **CatPred already predicts in-vitro kcat, Km and Ki and explicitly addresses dataset standardization, dissimilar-sequence evaluation and uncertainty quantification** ([Nature Communications, published 28 February 2025](https://www.nature.com/articles/s41467-025-57215-9)). These are established neighboring capabilities, not an originality claim for this work.
- **EnzyArena (2026 bioRxiv preprint) argues that pooling unmatched assay conditions can inflate apparent generalization** and benchmarks campaign-matched data, with a prospective 150-mutant comparison reported by its authors ([preprint](https://www.biorxiv.org/content/10.64898/2026.03.29.715152v2)). Because this is a preprint, its conclusions need independent review; nevertheless, it strongly motivates matched-condition evaluation and prospective utility checks.
- **Over-parameterized kinetic models can be non-identifiable**, so the system should report identifiability and parameter uncertainty rather than reward fit complexity ([AppliedChem, 9 April 2026](https://www.mdpi.com/2673-9623/6/2/25)).

The credible differentiation to investigate is an *evidence compiler*: connect exact raw-data bytes through each declared calibration, control correction, unit conversion, rate extraction, model fit and uncertainty analysis; retain every failure/exclusion; then separately evaluate prediction under stricter study/family/protocol holdouts and whether a decision beats simple baselines. That is a testable research objective, not a claim that an end-to-end implementation or advantage exists today.

## Seven paradigm shifts

### 1. From orphan parameters to a provenance graph

Model each result as a linked evidence graph, not an isolated row:

source record → raw signal → calibration/blank correction → derived concentration or rate → model fit → parameter estimate → comparison or prediction

Each edge records the operation, tool/version, configuration, units, input hashes, output hash, and relevant exclusions. A result that cannot be linked to its source artifacts is not silently promoted to reproduced evidence.

**Implementation principle:** reuse EnzymeML/SBML identifiers and structures where they fit. Add a narrow provenance/transform manifest only for gaps that cannot be represented losslessly. Do not create a second competing biochemical ontology.

### 2. From one context-free number to a context-conditioned kinetic object

For the primary endpoint, preserve the reported kcat, but always bind it to the enzyme form (including whether turnover is referenced to a subunit or oligomer), reaction, substrate, assay conditions, model equation, and the source's definition of the parameter.

The ambitious next step is a **context-conditioned kinetic response surface** where enough data exist—for example, response across temperature, pH, substrate concentration, ionic environment, or cofactor level. Do not force this surface to exist for sparse data. Report the observed context and the interpolation/extrapolation domain, and abstain outside it.

Maintain distinct types for:
- directly measured signals;
- derived concentrations or rates;
- model-estimated parameters;
- fitted or extrapolated response surfaces;
- model predictions for unseen sequences or conditions.

No type conversion should imply that one evidence class is a measurement from another class.

### 3. From point estimates to identifiable models and uncertainty

A numeric fit without uncertainty and identifiability diagnostics is incomplete for decision support. Where the source makes them available, capture confidence intervals or standard errors, replicate hierarchy, parameter covariance, fit residuals, boundary hits, and sensitivity to prespecified alternatives. Where only a point estimate is available, record uncertainty as unavailable—not zero.

The independent reproduction track should report at least:
- the source model and parameterization;
- objective/residual definition and fitting constraints;
- software/version and optimization settings;
- reported and reproduced parameters with declared tolerances;
- failed or non-identifiable fits and the reason;
- sensitivity/ablation results prescribed before looking at held-out labels.

Never choose the model solely because it yields the most convenient kcat. Where multiple models remain plausible, preserve model uncertainty.

### 4. From random-split leaderboards to a transportability ladder

Different splits answer different questions. Report them as separate benchmark tracks; do not compress them into a single score.

| Track | Held-out boundary | Question answered |
|---|---|---|
| A. Artifact replay | Frozen source artifacts and declared pipeline | Can this exact transformation be reproduced? |
| B. Experimental replay | Independent replicate/series within a source, with a frozen model | Does the analysis reproduce the source's parameter estimate? |
| C. Study transport | Whole paper/study held out | Does it transfer beyond a source's reporting/experimental habits? |
| D. Sequence-family transport | Sequence families or homology clusters held out | Does it generalize beyond close relatives? |
| E. Protocol/laboratory transport | Assay protocol, instrument, or laboratory held out when supported by metadata | Does it survive measurement-system shift? |
| F. Context transfer | Prespecified new condition ranges, separating interpolation from extrapolation | Is the condition-response prediction reliable inside or outside observed support? |
| G. Prospective decision | Predictions frozen before new measurements are observed | Does using the system improve a decision under a real budget? |

Leakage control must group re-hosted copies, duplicate records, replicates, closely related sequences, and shared publications before splitting. Report intersections and denominators. A source-held-out split does not prove sequence-family generalization; a sequence-family split does not prove laboratory transport. If the corpus cannot support a dimension, mark that track **not estimable** instead of claiming success.

The leaderboard should be a *matrix of transport tests*, not one leaderboard rank.

### 5. From accuracy-only to calibrated decisions and principled abstention

Compare Symthaea against transparent baselines on each frozen track: global median, context-group median, descriptor/neighbour methods, and specialist baselines only where the task and licensing permit. Preregister the target transform, metric, exclusion rules, candidate pool, and uncertainty procedure before examining held-out labels.

Report denominators, subgroup performance, interval coverage/calibration when uncertainty is claimed, and errors by context completeness and out-of-domain status. Keep failed predictions, abstentions, failed fits, and excluded records in explicit accounting. A model that abstains appropriately can be more useful than one that returns a confident extrapolation, but that value must be measured by a predeclared decision-loss rule.

Negative controls should include shuffled labels, source/study identity, context-only prediction, duplicate leakage, and deterministic replay. Any convincing performance from a nuisance control blocks predictive claims until investigated.

### 6. From passive prediction to value-of-information experiment selection

Once offline evaluation is sound, move from “predict the most active enzyme” to “what next observation would most reduce uncertainty or change the decision?” A future experiment planner can rank candidate measurements by expected reduction in decision loss or model uncertainty, adjusted for time, material cost, feasibility, and uncertainty.

Start fully offline with simulated candidate measurements and synthetic fixtures. For real prospective work, freeze the proposed predictions and protocol before the measurement, require qualified human review and appropriate biosafety/risk oversight, and bind each resulting observation back to the prediction and the exact experimental provenance. The system must not authorize or execute wet-lab protocols autonomously.

### 7. From one-off analyses to an interoperable evidence commons

Make the reproducibility receipt portable: a manifest references the rights-cleared source version, exact bytes, transformations, schemas, model configuration, splits, metrics, and the independent verifier's output. Source rights and attribution remain explicit, and metadata completeness is not confused with scientific validity.

EnzymeML should remain the primary exchange-format candidate for experimental data and models; STRENDA should remain the reporting-guidance reference; SBML remains relevant for suitable model representations. Mycelix or another provenance/governance service may record source-rights review and contribution receipts later, but trust scores or community consensus must never be converted into biological validation.

### 8. From fitting one preferred model to actively distinguishing plausible models

A fit can look numerically adequate while competing mechanisms remain observationally indistinguishable. The next scientific question is not only “which parameter values best fit this dataset?” but also “which additional observation would distinguish the remaining explanations?”

Keep two experiment-design objectives separate:

- **Parameter identifiability:** choose a measurement condition that narrows prespecified profile-likelihood intervals or otherwise improves practical identifiability for parameters in a declared model.
- **Model discrimination:** choose a condition where prespecified competing models make meaningfully different, experimentally resolvable predictions.

These objectives can conflict. An experiment that estimates parameters precisely under one model may do little to distinguish that model from a rival. Never collapse the objectives into one opaque score: report both, their constraints, and why the selected design trades one against the other.

**Proposed implementation path:** begin with synthetic ODE/kinetic fixtures whose parameters and generating model are known. Register candidate models, noise assumptions, feasible measurement conditions, selection objective, budget and stopping rule before scoring designs. Compare the chosen design against random and simple local-sensitivity baselines. Score parameter recovery, profile-likelihood identifiability, model-selection accuracy, robustness to misspecification, and total measurement cost. Preserve cases where no available design can discriminate the candidates. Only later adapt this to a rights-cleared enzyme assay, with expert review before prospective measurements.

This direction is supported by a September 2026 *Mathematical Biosciences* paper on optimal experiment design for practical identifiability and model discrimination. That paper demonstrates the distinction using simple ODE examples, not enzyme-specific prospective validation; this roadmap treats it as a transferable method to test, not evidence that the proposed enzyme workflow already works.

## Target architecture

1. **Source and rights layer:** source/version ID, rights and attribution review, access path, artifact checksums, publication/record locators.
2. **Evidence/artifact layer:** exact raw signals and controls where available, immutable derived artifacts, measurement identities, units, and explicit missingness.
3. **Transformation layer:** versioned calibration, baseline/blank correction, signal-to-concentration conversion, rate extraction, model fitting, and documented exclusions; each transform names input and output hashes.
4. **Kinetic-claim layer:** equation, variable definitions, parameterization, estimates, uncertainty, fit diagnostics, identifiability, and contextual applicability.
5. **Evaluation layer:** frozen corpus and split manifests, independent scoring, transportability track, denominators, uncertainty, negative controls, and abstention decisions.
6. **Decision layer:** candidate ranking and value-of-information analysis only after evaluation gates pass; predictions and measured outcomes remain separate data types.

A thin independent verifier should read the frozen artifacts independently of the candidate/model generator. A hash proves byte identity, not truth or scientific quality. A schema validator proves conformance to declared structure, not that a reaction equation is balanced or an experiment is unbiased. Keep these distinctions visible in every receipt.

## A concrete, fail-closed implementation sequence

### Phase 0 — Freeze the contract

- Keep the current kcat admission profile explicitly in policy-review status.
- Resolve exact kcat definitions, catalyst concentration basis, parameter uncertainty semantics, unit rules, and terminal statuses (success, null_result, failed, excluded, out_of_domain, blocked).
- Review profile rules against STRENDA Level 1A/1B and the exact EnzymeML V2 specification, without duplicating their data models.
- Exit only when each field has a declared schema, provenance source, missingness policy, and manual-versus-automated validation boundary.

### Phase 1 — Reproduce the pilot's calibration pipeline

- Run the synthetic Rust tests and the exact-head calibration suite.
- Verify the source URLs, sizes, SHA-256, Git blob identity, and canonical file metadata against the pinned source version.
- Reproduce the calibration fits from the source-declared range; retain out-of-range controls and exclusions explicitly.
- Do not claim kinetic-model reproduction from calibration-line agreement.

### Phase 2 — Build the evidence-graph replay for one assay

**Implemented starting point:** the Rust `enzyme-evidence-graph-check` verifies graph references, endpoint typing, DAG structure, exact local artifact hashes/lengths, source-specific rights-status gates, parameter value/unit pairing, strict rejection of unknown fields, artifact-reference closure, parameter-claim provenance edges (`reported_by` / `estimates`), and code/configuration/environment artifact identities for successful transformations. Regression fixtures are synthetic; this is structural integrity, not a kinetic replay, and the exact-head workflow must pass before execution is claimed.

- Pin the matching raw readings, controls, calibration(s), EnzymeML/OMEX model, schema/software revisions, and source analysis.
- Connect the acquired artifacts to transformation records and verify the exact source code, configuration, environment lock, and all intermediate outputs. Independent replay must execute the declared transforms, not merely compare hashes or re-use the producer's summary.
- Reproduce blank/control handling, signal conversion, initial-rate or time-course processing, and final model fit as separately hashed stages.
- Report stage-wise discrepancies, parameter uncertainty, model identifiability, failed fits and preregistered tolerances. Use the source's inactivation example as a preregistered regression/ablation only after its exact data and model lineage are confirmed.

### Phase 2.5 — Falsify model and design logic on synthetic truth

**Implemented starting point:** `xtask/src/enzyme_model_falsification.rs` adds a deterministic, synthetic-only Rust harness, exposed as `cargo run -p xtask -- enzyme-model-falsification`. It checks recovery of known Michaelis–Menten parameters with an explicit 2% fixture gate; fits both Michaelis–Menten and substrate-inhibition families on two deliberately separable synthetic truths and gates selection using in-sample BIC; chooses a candidate condition that maximizes separation between two fixed toy hypotheses; and separately computes a local finite-difference sensitivity proxy for Km. It emits a machine-readable JSON report and has five unit tests.

This is an executable sanity fixture, not yet a statistical benchmark. It uses fixed residual vectors, coarse parameter grids, two deliberately separable model-selection examples, fixed hypothesis parameters for experiment-design scoring, and an assumed noise scale. It does not test model-selection calibration across broad scenarios, estimate calibrated intervals, implement profile likelihoods, account for physical assay constraints, or demonstrate real enzyme behavior. See [ENZYME_MODEL_FALSIFICATION_BENCHMARK_V0.md](ENZYME_MODEL_FALSIFICATION_BENCHMARK_V0.md) for the scorecard and proposed targets.

- Expand to preregistered scenario matrices over parameter ranges, experimental designs, noise levels, model misspecification, and deterministic seeds.
- Add ambiguous/non-identifiable cases where the correct result is an explicit abstention, not a forced model selection.
- Measure nominal-interval coverage, parameter-recovery error, model discrimination balanced accuracy, false certainty, and measurement-budget efficiency independently.
- Compare active design with uniform-random and simple sensitivity-based design under matched budgets.
- Keep synthetic gates distinct from the later rights-cleared real-assay replay and held-out transport benchmark.

### Phase 3 — Make the independent benchmark boring and trustworthy

- Freeze source rights, corpus, study/family/campaign groupings, split, target, baseline model versions, scoring code, uncertainty methods, decision metric, and negative controls before held-out labels are examined.
- Have an independent verifier reproduce every baseline metric, exclusion, denominator, and confidence interval from the same staged artifacts.
- Evaluate under the same assay-context/campaign groupings wherever possible; never let related replicates, re-hosted records, or measurements from the same mutagenesis campaign leak across train and test.
- Retain the four existing transparent baselines (constant median, context-group median, descriptor k-nearest-neighbor, and Symthaea-assisted) as the first, low-cost rung. Once permissible data/model licenses and compute are checked, add a predeclared set of strong external baselines rather than selectively choosing them after seeing results: a published kinetic predictor (for example CatPred/UniKP where their intended inputs and model artifacts can be fairly reproduced), a zero-shot sequence-fitness baseline or consensus where scientifically appropriate, and random ranking for candidate-selection tasks.
- Score both prediction and decision utility: appropriate endpoint error (including log-scale error for wide-range positive kinetics), fraction within one order of magnitude, prediction-interval coverage/calibration when intervals are produced, ranking enrichment/top-k hit rate, and cost-normalized utility. Pre-register primary metrics and guard against picking whichever metric makes a method look best.
- Run each transportability track only where the corpus supports it. If family, protocol, or campaign metadata are inadequate, mark that track not estimable rather than laundering it into a score.
- Do not claim improvement unless matched-split baselines, uncertainty analysis, denominators, and negative controls support it. An in-database retrospective result is not evidence of prospective utility.

**Why this is stricter than a leaderboard:** CatPred already reports query-specific uncertainty and sequence-distribution transfer tests. The 2026 EnzyArena bioRxiv preprint reports that supervised kinetic-parameter predictors lose advantage on independent mutagenesis datasets and that one tested supervised predictor underperformed random selection in its 150-mutant prospective campaign, while a zero-shot sequence model ranked better. Treat that as a consequential but provisional result until peer-reviewed or independently reproduced; use it to design fair tests, not to assume a universal winner. A Symthaea result is valuable only if it earns its place in this shared, preregistered comparison.

### Phase 4 — Move to prospective value

- Preregister a bounded, low-risk, human-reviewed experiment chosen for a clearly stated decision or information-value reason.
- Freeze all model versions, scores, intervals, abstentions, random seed, candidate eligibility rules, and the random-selection baseline before observations are captured.
- Compare the resulting decision with predeclared baselines under the same assay, budget, sample size, and safety constraints; report the complete denominator and unsuccessful outcomes.
- Prioritize decision metrics that survive the cost of testing—such as useful hits per assay budget or enrichment at a fixed experimental capacity—not only correlation or retrospective parameter error. Preserve null findings when the system does not improve outcomes.

## Exit criteria that prevent self-deception

- **Integrity exit:** every admitted artifact and transform is hash-pinned, rights reviewed, and path-contained; negative controls prove that tampering and leakage are detected.
- **Replay exit:** independent runs reproduce each required stage within predeclared tolerances. Matching only the final parameter is insufficient.
- **Scientific-fit exit:** the model equation and parameter interpretation are explicit; uncertainty/identifiability and failures are accounted for.
- **Generalization exit:** each claimed transfer dimension has a dedicated held-out track, relevant baselines, uncertainty, denominator, and working negative controls.
- **Decision-value exit:** prospective use measurably improves a preregistered decision enough to justify cost/risks; otherwise retain the null result.

No exit can be inferred from queued CI. A workflow result only verifies the exact commit and job it actually executed.

## What would be genuinely paradigm-changing?

The strongest contribution would be a community-usable **reproducibility-to-decision pipeline** in which an enzyme kinetic claim can be traced and independently replayed from source artifacts, uncertainty is preserved rather than discarded, predictions are evaluated by how far they must transfer, and the next measurement is selected based on its potential to change a decision. That would connect reporting standards, data exchange, kinetic fitting, benchmarking, and experiment design into one auditable workflow.

The innovation should be judged by measurable outcomes: fewer unreproducible parameters, fewer silent unit/context errors, better calibration of uncertainty, honest estimates of out-of-domain behavior, and better prospective decisions. It should not be judged by architecture size, number of fields, number of predictions, or a single headline leaderboard score.

## Research references

- [STRENDA Guidelines — official current guidance](https://www.beilstein-institut.de/en/projects/strenda/guidelines/): reporting of catalyst identity, assay context, parameter definitions, software, uncertainty and raw-data expectations.
- [STRENDA database and guideline overview](https://www.beilstein-strenda-db.org/strenda/public/guidelines.xhtml): explicitly distinguishes useful/reliable reporting from a universal data-quality metric.
- [EnzymeML V2 formal specification](https://github.com/EnzymeML/enzymeml-specifications/blob/main/specifications/v2.md): structured reaction conditions, molecular species, measurements/time courses, equations, parameters and references.
- [EnzymeML documentation](https://enzymeml.org/): exchange format and FAIR data rationale.
- [CatPred: a comprehensive framework for deep learning in vitro enzyme kinetic parameters](https://www.nature.com/articles/s41467-025-57215-9) (Nature Communications, 2025): established kcat/Km/Ki prediction and query-level uncertainty; baseline for scientific comparison, not novelty to claim.
- [On Over-Parameterisation and Parameter Estimation of Enzyme Kinetics](https://www.mdpi.com/2673-9623/6/2/25) (AppliedChem, 9 April 2026): warns that added kinetic model parameters can impair practical identifiability and fit noise.
- [A Practical Identifiability Criterion Leveraging Weak-Form Parameter Estimation](https://link.springer.com/article/10.1007/s11538-026-01639-x) (Bulletin of Mathematical Biology, 31 March 2026): makes identifiability a measurable gate rather than an informal warning.
- [Optimal experiment design for practical parameter identifiability and model discrimination](https://pubmed.ncbi.nlm.nih.gov/42176891/) (Mathematical Biosciences, September 2026): treats parameter identifiability and choosing between competing models as related but distinct experiment-design objectives; demonstrations use simple ODEs, so enzyme-specific validation remains open.

- [On the reproducibility of enzyme reactions and kinetic modelling](https://pubmed.ncbi.nlm.nih.gov/35357794/) (2022): practical guidance on experiment design, kinetic modelling and reproducible enzyme data.
- [Benchmarking and Experimental Validation of Machine Learning Strategies for Enzyme Engineering](https://www.biorxiv.org/content/10.64898/2026.03.29.715152v2) (bioRxiv preprint, 2026): a recent, not-yet-definitive argument that heterogeneous assay conditions can confound apparent generalization.
- [Industrial Enzyme Benchmark V0 contract](INDUSTRIAL_ENZYME_BENCHMARK_V0.md): task scope and separation of calibration reproduction, kinetic-model replay, and prediction.
