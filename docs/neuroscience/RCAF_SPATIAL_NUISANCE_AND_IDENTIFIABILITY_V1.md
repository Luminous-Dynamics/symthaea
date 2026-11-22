# RCAF Spatial Nuisance and Identifiability Challenge v1

**Status:** proposed research contract; no spatial-nuisance implementation or empirical qualification is claimed by this document.  
**Parent:** executable common-mode calibration in `research/rcaf-adversarial-calibration-v1`.  
**Purpose:** specify the next independent nuisance family without changing the existing common-mode model or its acceptance tests.

## 1. Why this is the next increment

The current synthetic kernel has participant-level global and module factors, region residuals, and one shared AR(1) observation-only nuisance factor. That is a useful first counterexample, but it has no explicit ROI geometry, graph structure, localized fields, or transient spatial artifacts.

This gap matters. Head motion has been shown to induce complex signal changes that can be shared across much of the brain, persist after motion stops, and alter functional-connectivity correlations in a distance-dependent way. A spatially constant nuisance factor cannot exercise that entire failure surface. The 2024 systematic connectomics study evaluated 768 processing pipelines over 13,824 pipeline/criterion/data evaluations; only 9 met its complete multi-criterion requirements, and some unsuitable pipelines produced reproducible effects in the wrong direction. The 2025 comparison of 239 pairwise interaction statistics likewise found that method rankings depend on the property being evaluated. These results support a qualification system, not a single-score leaderboard.

The next step is therefore a **spatially structured, low-rank, observation-only challenge** with explicit geometry, nuisance/target alignment, and temporal coherence. It must be a new family alongside—not a replacement for—the common-mode challenge.

## 2. Core model and contracts

Let (p) index participants, (r) ROIs, (t) time points, (k) spatial nuisance modes, and (c) one of the four counterfactual cells.

The clean observation model can be represented as

[
X^{(c)}_{p,r,t}
= g_{p,r}G_{p,t}
+ m^{(c)}_{p,r}M_{p,t}
+ epsilon_{p,r,t}.
]

The spatial nuisance intervention adds

[
A_{p,r,t}(lambda)
= lambdasum_{k=1}^{K}B_{r,k}Z_{p,k,t},
qquad
Y^{(c)}_{p,r,t}=X^{(c)}_{p,r,t}+A_{p,r,t}(lambda).
]

Here (Binmathbb{R}^{R	imes K}) is a predeclared spatial loading matrix derived from an explicit ROI graph or coordinates; (Z) is a set of temporally structured nuisance processes; and (lambda) is a nuisance dose. The signal intervention changes (m^{(c)}), not the nuisance process. The nuisance intervention changes (A), not the latent target.

For a variance-normalized factor model, the generator must check each ROI's total factor variance and reject configurations that leave non-positive residual variance. Report both the requested loading dose and the realized nuisance variance fraction; raw loading alone is not comparable across different ranks or basis normalizations.

### Four matched cells

| Cell | Latent target | Spatial nuisance | Required interpretation |
|---|---|---|---|
| Clean null | unchanged | absent | ordinary same-distribution null |
| Signal only | changed by known intervention | absent | clean directional recovery |
| Nuisance only | unchanged | present | operational artifact-only false alarm |
| Signal + nuisance | changed by same intervention | same nuisance draw as nuisance-only | nuisance-conditioned recovery and effect drift |

Use common random numbers within each replicate and participant. The nuisance-only and combined cells must share the same nuisance innovations, nuisance spatial basis, and nuisance parameters. Their only intended difference is the planted latent signal intervention. Use independently keyed deterministic substreams for latent, residual, and nuisance innovations so adding a new family cannot silently shift the old generator's random draw order.

At exact zero spatial dose, all spatial-nuisance and combined-cell outputs must exactly reproduce their corresponding clean cells. Existing common-mode results must remain unchanged when spatial nuisance is disabled.

## 3. Geometry must be explicit

A spatial challenge must never infer anatomy from ROI array order. A synthetic chain or grid may be used as a clearly labelled toy graph, but must not be described as an anatomical atlas.

The geometry contract should accept:

- a unique, ordered ROI-label vector;
- a symmetric, non-negative weighted adjacency matrix or an explicitly sourced coordinate set and distance rule;
- a declared graph-construction version and digest;
- a declared treatment of disconnected components, isolated nodes, and missing ROIs.

Use the weighted graph Laplacian (L=D-W), where (W) is the adjacency matrix and (D) is the diagonal weighted degree matrix. Candidate low-rank bases include low-frequency Laplacian subspaces and heat-kernel-smoothed seed fields (H_\tau=exp(-\tau L)). The spatial basis must be deterministic under a fixed geometry contract. Where eigenvalue multiplicity makes individual eigenvectors non-unique, define the subspace/projector or a deterministic orthogonalization convention rather than treating an arbitrary eigenvector orientation as meaningful.

The constant graph mode is excluded from the first spatial family because it duplicates the existing common-mode challenge. It can be reintroduced only as an explicit cross-family interaction test.

## 4. Distinct spatial nuisance families

The first executable increment should implement the first two families. Keep the third as a separately versioned extension; do not mix it into a first result without a distinct report field.

### S1 — graph-smooth low-rank field

Generate (K) temporally varying nuisance factors whose ROI loadings occupy a predeclared low-frequency graph subspace. This tests broad but non-constant spatial fields and low-rank structured contamination. Sweep rank (K), spatial smoothness, temporal AR(1) coefficient, and dose.

### S2 — localized heat-kernel field

Select seed ROI(s) from a predeclared seed policy and spread their spatial loading through the graph heat kernel. Sweep heat-kernel scale, number of seeds, and spatial overlap with the signal module. Include both single-focus and multi-focus fields.

### S3 — motion-like transient spatial field (follow-on)

Use a sparse event train and a post-event decay envelope,

[
h_t=\sum_j a_j\exp[-(t-t_j)/\tau],\mathbf{1}(t\ge t_j),
qquad
A_{r,t}=q_r h_t,
]

where (q_r) is a declared spatial field. Include a broad component plus a distance-dependent component, and vary event timing relative to the analysis window. This is a stylized challenge, not a validated physical model of head motion. It is motivated by published evidence that motion-related BOLD changes can persist after movement and affect correlation structure as a function of inter-regional distance.

A static positive per-ROI multiplicative gain should not be treated as a generic correlation-matrix nuisance: Pearson correlation is invariant to constant positive rescaling of each ROI time series. A gain challenge must therefore specify the time-varying, additive, nonlinear, or preprocessing interaction that makes it capable of changing the endpoint.

## 5. Alignment and temporal-coherence axes

A spatial artifact is most dangerous when it resembles the target. Do not remove this case by always orthogonalizing nuisance modes against the target.

Define a target spatial subspace (T) and projector (P_T). Record nuisance-to-target spatial alignment, for example,

[
a(B,T)=\frac{\|P_TB\|_F^2}{\|B\|_F^2},
]

with the exact normalization fixed in the implementation contract. Include at least orthogonal/low-overlap, partial-overlap, and target-aligned settings. Report the realized alignment, not only its requested value.

Also vary temporal coherence between nuisance and target processes. For a controlled scalar coupling parameter (\gamma\), one possible construction is

[
Z_t=\gamma M_t+\sqrt{1-\gamma^2}\,U_t,
qquad 0\le\gamma\le1,
]

where (U) is an independent process with the declared temporal law. This creates a path from independent artifacts to a deliberately indistinguishable boundary case.

The grid must cross:

- spatial dose and realized nuisance variance fraction;
- nuisance rank and spatial smoothness/localization;
- target-overlap level (a);
- temporal coherence (\gamma);
- temporal autocorrelation and segment length;
- target effect size and participant response fraction.

Keep development and confirmatory seeds disjoint. Avoid selecting spatial modes, ranks, dose points, or acceptance thresholds after observing confirmatory outcomes.

## 6. An explicit identifiability boundary

This challenge should introduce a first-class outcome: **not identifiable from the available observations**.

Consider a target intervention that adds (\Delta u z_t) to the observed ROI vector. An artifact-only intervention can add (\lambda v a_t). If (v=u), (a_t=z_t), and (\lambda=\Delta), then the two observed time series are exactly equal:

[
X_t+\Delta u z_t
\equiv
X_t+\lambda v a_t.
]

No estimator operating on those observations alone can tell whether the contribution came from a latent target change or an observation artifact. This is a constructive non-identifiability case, not an algorithm failure. A benchmark that demands a positive/negative causal verdict here rewards unsupported certainty.

Therefore:

- identifiable cells should be scored for calibrated recovery, artifact false alarms, bias, and coverage;
- weakly identifiable cells should report uncertainty and abstention performance;
- observationally equivalent cells should require an explicit `not_identifiable`/abstain disposition;
- a system must not claim that a denoiser solved the boundary unless it uses genuinely independent information (for example, a motion trace, physiological recording, acquisition metadata, or an independently justified nuisance sensor) and that information is included in the evidence record.

Measure correct abstention on non-identifiable cells separately from false abstention on clean, identifiable signal cells. Do not merge them into a single accuracy score.

## 7. Required outcomes

Each family × rank × alignment × coherence × dose cell should report:

1. classical same-distribution null calibration;
2. nuisance-only operational false-alarm rate and its interval;
3. clean directional recovery and wrong-direction rejection;
4. signal-plus-nuisance directional recovery and wrong-direction rejection;
5. estimated effect bias, RMSE, interval coverage, and interval width;
6. nuisance-induced effect drift in absolute units and as a fraction of the preregistered minimally meaningful effect;
7. artifact amplification (combined estimate minus clean estimate), with uncertainty;
8. correct abstention on non-identifiable cells and false abstention on identifiable cells;
9. participant attrition / data sufficiency, runtime, memory, and deterministic replay status.

Keep the existing distinction intact: nuisance-only rejection under changed observations is not classical Type-I error under an identical observed-data distribution.

The primary result should be a **qualification surface / identifiability envelope**, not a scalar rank. Show where a pipeline is supported, nuisance-sensitive, underpowered, or not identifiable. Preserve the Pareto trade-off among recovery, calibration, artifact sensitivity, coverage, transportability, abstention, compute cost, and replayability.

## 8. Metamorphic and negative-control contract

Required tests for the spatial generator:

- exact zero-dose equivalence with the clean path;
- exact nuisance-innovation matching between nuisance-only and combined cells;
- latent target/loading digest unchanged in both nuisance-only cells;
- ROI relabeling/permutation equivariance when data, labels, coordinates, and graph are permuted together;
- rejection when ROI labels are mismatched or graph/data dimensions differ;
- invariance to a joint orthogonal rotation of an equal-variance nuisance-factor basis (the generated spatial covariance must be unchanged);
- invariance to a coupled basis sign flip;
- deterministic replay for identical configuration and seeds;
- rejection of asymmetric/negative/invalid graph weights, non-finite coordinates, invalid heat-kernel scales, invalid coherence, and variance-budget overflow;
- negative control: permuting the time series without permuting the graph must not be accepted as an equivalent geometry;
- negative control: a spatial mode deliberately aligned with the target must trigger the expected elevated identifiability risk, not be silently projected away.

A metamorphic pass qualifies only the declared mathematical property. It does not validate an atlas, preprocessing workflow, biological interpretation, or empirical nuisance distribution.

## 9. Proposed qualification targets (not achieved results or universal standards)

For predeclared identifiable operating regions:

- primary null families: at least 10,000 Monte Carlo replicates, with simultaneous uncertainty bounds across the confirmatory grid;
- screening: at least 1,000 replicates per cell before confirmatory lock;
- moderate nuisance-only false alarms: target no greater than 5%, with the upper simultaneous confidence bound reported;
- directional recovery: retain the existing endpoint-specific minimally meaningful effect and power target; report participant-level effective (N) independently of Monte Carlo replicate count;
- effect drift: target at most 10% of the minimally meaningful effect in the declared operating envelope, unless the endpoint contract justifies another threshold;
- identifiability boundary: correct abstention is required; recovery is not an admissible target where the observations are observationally equivalent;
- replay: exact deterministic replay for synthetic fixtures and content-addressed receipts for every execution.

These are engineering gates for preregistration, not externally established standards. Do not change acceptance thresholds after looking at confirmatory outcomes.

## 10. Industry comparison: same inputs, explicit boundaries

The goal is not to replace mature preprocessing and postprocessing tools. It is to add a claim-specific qualification layer over fair, reproducible comparisons.

A fair comparator campaign should pin the data release, atlas, preprocessing release/container, nuisance strategy, censoring, smoothing, run-combination, connectivity definition, exclusions, and compute environment. Use the same participant-level split manifest and never choose a comparator configuration on the external test cohort.

Useful operational comparators include:

- **fMRIPrep** for BIDS-oriented preprocessing and confound/QA outputs. Its documentation explicitly states that it does not generally perform downstream denoising; its confound outputs include motion, global signals, CompCor components and other regressors.
- **XCP-D** for downstream denoising, censoring, connectivity, and postprocessing. Its 26.1.1 interface exposes nuisance-regressor strategies, despiking, smoothing, motion filtering, run handling, and output choices.
- **CONN 25b** as an established alternative denoising/connectivity workflow, including its denoising and quality-control tools.
- **ABCD Coordinating Center v4.1.0** as a useful pinned operational reference: its 18 November 2026 release notes specify fMRIPrep 25.1.4, XCP-D 26.1.1, and ReproTM outputs. However, the v4.1.0 minimal fMRIPrep derivative release explicitly excludes preprocessed BOLD time series, confounds, and surface-projected functional data. It therefore cannot be assumed to provide every input needed for an independent RCAF re-analysis. Either use the documented full-derivative generation path, qualify available XCP-D derivatives for the precise endpoint, or select a different complete source release. Record this availability boundary and participant attrition in the comparison. Also report a current fMRIPrep 25.2.7 comparison where compatible; do not silently mix versions.

Compare both native recommended configurations and a preregistered matched configuration. Report participant attrition, coverage, QC outcomes, endpoint-specific recovery, artifact-only false alarms, effect inflation, site transport, runtime, memory, and exact replay. Do not call one pipeline superior because it wins one synthetic metric.

## 11. Execution order

1. Add a separate `SpatialNuisanceConfig` and deterministic substream derivation; leave `InjectionRecoveryConfig` defaults and common-mode outputs unchanged.
2. Implement S1 graph-smooth low-rank fields and the four matched cells.
3. Implement S2 heat-kernel localized fields and the alignment/coherence grid.
4. Add the explicit non-identifiable construction and an abstention result contract.
5. Add schema validation, semantic tests, negative controls, metamorphic tests, and artifact-receipt support.
6. Run a small development pilot, inspect failure modes, freeze the generator and confirmatory grid, then run at least 10,000-replicate null campaigns.
7. Only after synthetic qualification, compare fMRIPrep → XCP-D and CONN on the same frozen empirical inputs and independent participant/site splits.
8. Keep empirical cohort analysis blocked until dataset provenance, preprocessing, atlas, endpoint, exclusions, and external split are frozen.

## 12. References and implementation anchors

- Power, J. D. et al. (2014). *Methods to detect, characterize, and remove motion artifact in resting state fMRI*. NeuroImage. https://doi.org/10.1016/j.neuroimage.2013.08.048
- *Systematic evaluation of fMRI data-processing pipelines for consistent functional connectomics* (2024). Nature Communications. https://doi.org/10.1038/s41467-024-48781-5
- Liu, Z.-Q. et al. (2025). *Benchmarking methods for mapping functional connectivity in the brain*. Nature Methods. https://doi.org/10.1038/s41592-025-02704-4
- fMRIPrep outputs and confounds: https://fmriprep.org/en/latest/outputs.html
- XCP-D 26.1.1 usage and nuisance/postprocessing options: https://xcp-d.readthedocs.io/en/26.1.1/usage.html
- CONN toolbox release information: https://web.conn-toolbox.org/
- ABCD Coordinating Center release notes: https://docs.abcdstudy.org/latest/documentation/imaging/abcc_start_page.html

## Claim limits

This document is a design specification. No new Rust implementation, synthetic spatial-nuisance run, external-data analysis, pipeline comparison, or biological qualification is claimed here. The existing common-mode result remains a separate synthetic result with its original scope. The next code PR must make the spatial family executable and test the contracts above before any numerical robustness claim is made.
