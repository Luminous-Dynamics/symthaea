# Mechanism identifiability under interventions, nuisance models and finite samples V1

**Tracking:** #10737 — SCI-CONNECTOMICS-094
**Composes:** #10603, #10656, #10661, #10670, #9587, #9522, #9575, #9578, #9255
**Status:** synthetic exact contract; no biological mechanism or performance claim is made.

## 1. The missing assurance object

Query-directed intervention selection (#10670) can identify a scientific decision across reconstruction worlds. It does not necessarily identify the mechanism producing the response. Direct synaptic transmission, polysynaptic propagation, extrasynaptic signalling, shared input, cell-intrinsic state and observation-kernel artifacts can remain observationally compatible.

V1 introduces a bounded **mechanism-identifiability certificate**. It answers whether a declared intervention and measurement envelope separates the supplied mechanism hypotheses at a preregistered finite replicate budget.

## 2. Keep query and mechanism identity separate

Every mechanism hypothesis carries its own scientific-query decision and mechanism class. Query identifiability is checked only across pairs with opposing query decisions. Mechanism identifiability is checked across all distinct mechanism classes.

The benchmark preserves four possible outcomes:

| Query identified? | Mechanism identified? | Status |
|---|---|---|
| Yes | Yes | QUERY_AND_MECHANISM_IDENTIFIED |
| Yes | No | QUERY_IDENTIFIED_MECHANISM_UNIDENTIFIED |
| No | Either | QUERY_UNIDENTIFIED |
| Cannot evaluate | Cannot evaluate | NOT_ESTIMABLE |

`QUERY_IDENTIFIED_MECHANISM_UNIDENTIFIED` is not a failed result. It is often the scientifically correct conclusion.

## 3. Generative mechanism object

A mechanism hypothesis is not just a graph label. The biological model must eventually bind:
- a reconstruction world and source evidence;
- direct edges and candidate intermediate paths;
- latent mediators and possible common inputs;
- parameter/state families, including neuromodulation;
- intervention semantics and off-target envelope;
- observation/indicator/registration kernel;
- predicted *observable* response distributions.

Keep latent states and the measured signal separate. A direct edge is not a conductance measurement; a measured response does not prove a direct synapse; extrasynaptic signalling must not be rewritten as an EM edge.

## 4. Robust finite-family identifiability

For action `a`, mechanism `m`, nuisance profile `eta`, and observed outcome `y`, the supplied model gives `P(y | m, a, eta)`.

For each pair of mechanisms and each *shared physical nuisance profile*, compute:

`TV(P_i, P_j) = 0.5 * sum_y |P_i(y) - P_j(y)|`.

Robust single-action separation is the minimum TV across the declared nuisance profiles. A zero value means at least one declared nuisance profile makes the pair observationally equivalent under that action.

For `n` independent biological replicates, V1 computes the exact equal-prior optimal pairwise testing error:

`R_n = 0.5 * sum_(y_1,...,y_n) min(P_i^n(y_1,...,y_n), P_j^n(y_1,...,y_n))`.

The reported robust risk is the maximum `R_n` over the declared common nuisance profiles. This is a finite categorical software oracle, not a general biological power calculation.

Cells, time points, and multiple neurons nested within one animal are not independent biological replicates. In real datasets, the independent unit must be animal/session/target according to the sampling design, with cluster-aware uncertainty. The synthetic fixture explicitly declares independent synthetic replicates.

A positive TV establishes distributional separation only inside the supplied finite model/nuisance envelope. It does not prove biological truth or sufficient power. Finite-sample risk is a separate quantity.

## 5. Exact decision contract

For a given intervention and replicate budget:
- `query_identified` is true only if every pair with opposing query decisions has worst-case pairwise testing risk at or below the preregistered alpha;
- `mechanism_identified` is true only if every pair with distinct mechanism classes has worst-case pairwise testing risk at or below alpha;
- unresolved pairs are returned explicitly;
- a feasible policy is selected by minimum total cost among actions that meet both gates; ties are broken by stable action ID;
- disallowed, inaccessible, unassessed and infeasible interventions can never be selected.

The pairwise equal-prior testing risk is a controlled identifiability diagnostic, not the final multi-class decision rule for a biological study. Confirmatory analyses must use preregistered class priors/losses, multiplicity control and clustered uncertainty.

## 6. Synthetic known-answer fixture

The fixture declares three mechanisms:

| Mechanism | Class | Query decision |
|---|---|---:|
| direct | DIRECT_SYNAPTIC | true |
| extra | EXTRASYNAPTIC_LATENT | true |
| shared | SHARED_INPUT | false |

Two nuisance profiles are included: nominal and indicator saturation.

Five actions are evaluated:

| Action | Core behavior | Cost | Feasibility |
|---|---|---:|---|
| passive | All three mechanisms predict the same measured distribution | 1 | Feasible |
| query-probe | Separates query-true from query-false mechanisms but makes direct and extrasynaptic predictions identical | 2 | Feasible |
| selective-pathway | Appears to separate mechanisms under nominal conditions; saturation collapses all predictions | 1 | Feasible |
| orthogonal-joint | Three observable categories robustly separate all three mechanisms under both nuisance profiles | 3 | Feasible |
| infeasible-perfect | Perfectly separates all hypotheses but is prohibited | 0.5 | Disallowed |

At seven independent synthetic replicates and alpha 0.05:
- passive identifies neither query nor mechanism;
- query-probe identifies the query but not the mechanism;
- selective-pathway cannot certify either because one declared nuisance profile collapses the predictions;
- orthogonal-joint identifies both query and mechanism;
- infeasible-perfect is excluded despite its perfect predicted separation.

The cheapest feasible action satisfying both identification gates is `orthogonal-joint`. This is a constructed mathematical fixture, not a claim about an actual laboratory perturbation.

## 7. Adversarial controls

V1 requires:
1. Passive observational equivalence.
2. Query identified while direct versus extrasynaptic attribution remains unresolved.
3. Nuisance-induced collapse of an apparently discriminating action.
4. A finite sample budget that is too small to meet alpha despite nonzero TV.
5. A disallowed perfect discriminator.
6. Missing prediction cells in the action × mechanism × nuisance Cartesian product.
7. Duplicate and dangling identities.
8. Outcome probability sums other than one.
9. Inconsistent outcome vocabularies.
10. A changed query decision that would change which pairs are tested.

## 8. Industry and scientific comparators

- ConnectomeBench2 V2: predictive split/merge proofreading. It does not directly measure mechanism identifiability.
- NEURD and RoboEM: reconstruction and candidate generation. Their corrections are inputs to candidate mechanism worlds, not mechanism truth.
- Randi et al. (Nature 2023): direct optogenetic stimulation and whole-brain calcium imaging across 23,433 C. elegans neuron pairs and 113 animals. The atlas measures *effective* signal propagation over all paths and reports extrasynaptic contributions; its statistical framework distinguishes connected, equivalently non-connected and insufficient-evidence pairs. It is an ideal abstention/overclaim control, but not a direct monosynaptic ground-truth table.
- #9587: broad blind intervention benchmark and mechanism scorecard; this V1 is its finite-family identifiability kernel.
- #10670: query-directed intervention selection; this V1 supplies a separate mechanism-identification acceptance condition.


### Matched modern baselines and benchmark denominator discipline

| Comparator | What it establishes | Required matched comparison |
|---|---|---|
| ConnectomeBench2 V2 (2026) | Predictive split/merge proofreading across four species; paper page reports 716,485 expert decisions and >4.5M images. | Reproduce classification/calibration/transfer separately. The current dataset card reports 401,170 rows, with 37,926 test rows and EM on 63% overall, but only 38% of merge-edit and 23% of split-edit rows. Pin revision and task filters; these are not interchangeable denominators. |
| NEURD (Nature 2025) | Morphology-informed proofreading; selected high-confidence merge subsets report 99%/95% agreement with expert splits, nearly 150k automatic edits and >3× speed-up for multi-soma review. | Use as a candidate mechanism/reconstruction generator; preserve selected-subset and workflow denominators rather than treating agreement as global reconstruction accuracy. |
| RoboEM (Nature Methods 2024) | Self-steering 3D neurite tracing; ~400× lower computational annotation cost in studied cortical analyses. | Compare reconstruction/candidate generation and all-in annotation cost; do not describe the result as 400× less expert labor. |
| Connectomics VLMs (arXiv:2609.36492, 2026 preprint) | LoRA adaptation on a few thousand labels brings some open VLMs near specialist proofreading performance on evaluated tasks, with selected unseen-species merge-transfer gains. | Include as a candidate generator and cross-species shift baseline; reproduce exact split, data filters and preprint task definitions. |
| NPI (Nature Methods 2025) | Learns a surrogate of large-scale fMRI dynamics and virtually perturbs it to estimate effective connectivity; validates on generative networks and compares with cortico-cortical evoked potentials. | Use as a virtual-perturbation baseline, but do not equate model-internal perturbation with a measured optogenetic intervention or single-cell connectome mechanism. |
| GO-CBED (TMLR 2026) | Goal-oriented, non-myopic sequential Bayesian design targets information about a specified causal quantity; uses a learned policy and variational approximation. | Directly compare query targeting and multi-step policy value against entropy/EIG and the exact finite minimax oracle here. Its approximate Bayesian objective is not the same as nuisance-worst-case finite-sample mechanism risk. |
| C. elegans signal-propagation atlas (Nature 2023) | Direct optogenetic perturbation and whole-brain calcium imaging for 23,433 neuron pairs across 113 animals; includes extrasynaptic effects and explicit insufficient-evidence outcomes. | Use as measured perturbation data and abstention/overclaim controls; its effective connections are not a direct monosynaptic ground-truth table. |
| Global graph-invariant stopping (2026 preprint) | Fits global invariant convergence to estimate proofreading effort/stopping. | Matched global convergence baseline only; its authors explicitly caution that global proxies do not measure progress toward a specific scientific query. |

Sources: ConnectomeBench2 [paper/project](https://synthneuro.org/publications/connectomebench2-a-unified-benchmark-for-automated-connectomic-proofreading/) and [current dataset card](https://huggingface.co/datasets/jeffbbrown2/ConnectomeBench2); [NEURD](https://www.nature.com/articles/s41586-025-08660-5); [RoboEM](https://www.nature.com/articles/s41592-024-02226-5); [VLM proofreading preprint](https://arxiv.org/abs/2609.36492); [NPI](https://www.nature.com/articles/s41592-025-02654-x); [GO-CBED](https://tmlr.infinite-conf.org/paper_pages/6dVCZT7pO2.html); [C. elegans signal-propagation atlas](https://www.nature.com/articles/s41586-023-06683-4); [graph-invariant stopping](https://www.biorxiv.org/content/10.64898/2026.06.30.735414v1).

These systems answer different questions. The mechanism-identifiability benchmark is not a new classifier leaderboard. It tests whether competing causal explanations can be discriminated under declared intervention, nuisance, observation and finite-sample limits. Report task accuracy, reconstruction fidelity, effective-connectivity prediction, mechanism attribution and expert cost as separate scorecards.

## 9. Promotion targets

These are aspirational, not achieved results:

- 100% exact classifications on bounded synthetic fixtures;
- zero mechanism-identifiability promotions when a declared nuisance profile yields identical observable distributions;
- 100% abstention on exact observational-equivalence controls;
- zero direct-synapse promotions from effective response alone;
- zero functional-zero promotions from non-significance, low power or inaccessible data;
- publish risk versus replicate budget, not a universal sample count;
- at least 20% lower total experimental cost per correctly resolved mechanism class than the strongest preregistered baseline, with paired uncertainty supporting improvement;
- no degradation in query-level accuracy/calibration when mechanism abstention is enabled.

## 10. Staged execution and claim ceiling

Stage 1: exact synthetic finite-distribution qualification. Stage 2: retrospective evaluation on the C. elegans perturbation atlas using animal-level lockboxes and exact observation kernels. Stage 3: Drosophila CPG competing mechanism models under exact artifact custody (#9934). Stage 4: prospective blinded interventions only with appropriate collaborators, approvals and data rights.

A passing synthetic oracle validates finite-family mathematics and software semantics. It cannot prove direct synaptic causation, biological mechanism, complete nuisance coverage, or generalization to real animals. Those claims require independent measurements, source-bound evidence, blinded adjudication, adequate biological replication and prospective validation.