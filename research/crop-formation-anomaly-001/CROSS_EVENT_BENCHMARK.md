# CROP-INTEL-001 — Cross-event generalization and evaluation benchmark

**Version:** 0.1  
**Status:** design plus event-grouped split primitive; no multi-event model evaluation has been run.

## 1. The key paradigm shift

Do not ask whether a model can explain the same formation from which its rules were inferred. Ask whether a frozen measurement/representation/grammar pipeline works on a completely unseen physical event.

Images, crops, crops-of-crops, traced SVGs, JSON exports, alternate digitizations and derived feature tables from one formation are not independent events. They must remain in one split group.

There are three distinct evaluation regimes:

1. **Within-event reconstruction:** hold out primitives/regions from one event. Useful for reconstruction diagnostics only; it does not demonstrate transfer.
2. **Event-held-out generalization:** train/tune on some physical events and evaluate on unseen events.
3. **Site/source-held-out generalization:** hold out entire field locations or source families to test geographic/source leakage. This is a stricter, separate evaluation—not a substitute for event-held-out evaluation.

## 2. Current data inventory and maturity

### CCAT Zeals Knoll 2026

- One event, 60 digitized circle records.
- CC BY 4.0 derived geometry CSV imported with exact source Git blob SHA.
- Image-space pixels, not field metres.
- No per-circle uncertainty, independent annotator agreement or original photograph in CROP-INTEL.
- Good for fixed-input software replay; **not** an event-generalization dataset.

See [CCAT replication and audit](CCAT_REPLICATION.md).

### FEEDIT Field Edit candidate corpus

The public FEEDIT archive states 11 formations, 1,195 rendered elements, 1990–2008 and CC BY-SA 4.0 parametric geometry. The candidate metadata index is in [records.json](datasets/feedit-candidate-index/records.json).

Important data-quality distinctions:
- Some pages describe individually traced circles/geometry, e.g. Stonehenge Julia Set (125 rendered / 125 declared).
- Some expand a compact generator: Windmill Hill Triple Julia (94 / 32), Woodborough Hill sunflower (310 / 16), and Milk Hill Galaxy (409 / 69).
- The Milk Hill Galaxy page explicitly says satellite geometry is representative, while the source survey covers major circles.
- Page metadata identifies Claude-Fable-5 as digitizer for most entries, with dates in August 2026; independent human re-digitization and measurement covariance are not supplied.
- One catalogue entry (Barbury Castle Tree of Life) had a detail page returning 404 in this audit.
- The source web pages give approximate location coordinate uncertainty from ±300 m to ±1,000 m for accessible records.
- CROP-INTEL has **not imported the underlying FEEDIT geometry JSON**. Only public metadata is indexed here; the site says the JSON records are CC BY-SA 4.0 and the GitHub repository link is forthcoming.

Therefore FEEDIT currently supplies useful candidate **generator specifications and geometry metadata**, not 1,195 independent ground-truth measurements. Before numerical import, obtain the source JSON records and preserve CC BY-SA 4.0 attribution/share-alike conditions. Do not redistribute source photographs without separate permission.

### Adjacent crop-lodging measurement benchmarks

These are not crop-circle datasets, but they give a realistic baseline for the biological/3D measurement channel:

- Ravichandran et al. (2025), [Precision phenotyping in wheat: LiDAR-based plant height estimation and lodging classification using uncrewed ground vehicles](https://doi.org/10.1080/07038992.2025.2516742), reports a study over six Canadian research stations, 90 plots and 30 historical cultivars, with lodging classification up to 0.97 quadratic-weighted kappa and 98.51% Macro-F1 for the evaluated 2-/3-class settings. Treat these as the paper's reported results, not a CROP-INTEL result or a directly comparable crop-circle score.
- Reena et al. (2025), [Wheat3D PartNet: Annotated dataset for 3D wheat part segmentation](https://doi.org/10.1016/j.compag.2025.110697), describes 1,303 3D wheat point clouds over three cultivars with manual ear/non-ear labels. The article's data-availability statement says the point-cloud dataset will be made available on request; access must be confirmed before treating it as a public download. It is relevant to stem/leaf/ear segmentation methods, not to circle geometry.
- Çelik (2026), [EO4CerealStress lodging maps](https://doi.org/10.17026/PT/TFYJBN), provides georeferenced EnMAP/Sentinel-2 prediction maps (GeoTIFFs) from several regression approaches under CC BY-ND 4.0. These are model outputs, not raw, independently labelled training data. Do not modify or redistribute derivatives under a different licence.

The 2026 review [Multi-scale lodging in rice and wheat](https://doi.org/10.3389/fpls.2026.1956108) stresses the distinction between remote-sensing canopy posture proxies and intrinsic mechanical measurements (flexural rigidity, breaking force/moment, section modulus). The field protocol must keep these as separate data channels.

## 3. Event identity and leakage policy

Each sample row must have:
- stable `sample_id`;
- canonical `event_id`;
- `leakage_group_id` for the split unit in the current benchmark;
- source assertion IDs and source media IDs;
- derivative/parent IDs and content hashes;
- site/field ID, source family and acquisition session where known;
- rights state and label provenance.

For event-held-out evaluation, `leakage_group_id` is the canonical physical event identity. For site-held-out evaluation, it is the canonical field/site identity. For source-held-out evaluation, it is the source/acquisition family. Run these as separate benchmark tracks and report the grouping rule.

The same physical event cannot cross folds under any of these tasks. If identity is unresolved or conflicting, use a conservative shared leakage group until adjudicated.

### Rust split primitive

`lab/src/event_split.rs` contains:
- deterministic approximately sample-balanced grouped K-fold allocation;
- grouping by caller-supplied `leakage_group_id`;
- unique sample ID and required metadata validation;
- complete validation coverage;
- exact train/validation complement checks;
- cross-fold leakage detection;
- reproducibility through a recorded seed and stable tie-break.

This implementation is a split primitive, not a complete training/evaluation runner. The generated plan must be written into a benchmark receipt, including the exact sample order and source hashes, before model fitting.

The reference crate also exposes `CircleGrammarFamily`: a frozen list of class-count/contact-threshold configurations. Evaluating a family recomputes the centre, radius/radial clustering, contact graph and centred motifs for each configuration. Combine each predeclared score across that complete family with `familywise_max_statistic_p_value`, rerunning the same family on each null replicate. This current family covers only the declared class-count/tolerance grid; other choices (centre estimator, boundary selection, primitive extraction, motif families and image preprocessing) must be added whenever those are explored.

## 4. Data provenance classes

Assign each geometric primitive one of the following observation classes. Do not collapse them into a single "ground truth" label.

| Class | Meaning | Permitted use |
|---|---|---|
| M0 | Direct field survey with instrument/calibration and independent check points | Measurement ground truth, subject to documented uncertainty |
| M1 | Independently human-traced geometry from rights-cleared source imagery with repeat annotations | Geometric reference; propagate annotation disagreement |
| M2 | Single human/AI-assisted trace from an image or diagram | Exploratory reference only; requires independent audit before accuracy claims |
| M3 | Parametric generator primitive declared by the dataset author | Test of reconstruction/generator hypotheses; not an independent observation |
| M4 | Symmetry-expanded/repeated/generated instances derived from fewer declared primitives | Augmentation/known generator output only; never count as independent measured samples |
| M5 | Uncertain or conflicting geometry | Preserve as unresolved; do not force into scored test labels |

FEEDIT's page-level "fidelity level" is source metadata, not an externally calibrated accuracy statistic. Also retain whether an element is directly traced, estimated, copied, mirrored, rotated, procedurally generated or representative.

## 5. Benchmark tracks

### Track A — fixed-input replay

Purpose: verify implementations agree on exactly the same supplied geometry.
- CCAT: reproduce 60 records, five radius classes, 66 contacts at 45 px, 12 A–E contacts and the 10/10/8 motif counts.
- FEEDIT: when source JSON is obtained, validate schema, units, primitive types, transform hierarchy and declared-versus-rendered counts.
- Report exact count agreement, numeric tolerance, runtime and all mismatches.

This track tests software fidelity, not field measurement accuracy or causal interpretation.

### Track B — geometric reconstruction accuracy

Inputs: rights-cleared photographs with independent annotations and a declared image/field transform; or surveyed field geometry.
Compare:
- human annotator A vs B;
- classical Hough/contour/ellipse/circle fitting;
- feature-based registration;
- suitable image segmentation/detection models;
- parametric geometric reconstruction.

Report per-event:
- centre RMSE and p95 error in calibrated units;
- radius MAE/RMSE and relative error;
- boundary Chamfer and Hausdorff distances;
- primitive precision/recall/F1;
- transform/registration checkpoint errors;
- calibration and uncertainty coverage;
- abstention rate and human review minutes.

Do not use the same circle annotations to tune and test a digitizer.

### Track C — event-held-out structural representation

Goal: learn a representation of geometric relations that transfers to unseen events without assuming all events share one construction grammar.

Candidate features:
- normalized primitive geometry and scale-free ratios;
- symmetry group and residuals;
- spatial/radial/angular distributions;
- contact/topology graph;
- graph motifs and motif uncertainty;
- geometric description length / MDL;
- reconstruction residual and parameter count.

Baselines:
- hand-engineered radial/symmetry descriptors;
- circle/arc/polygon primitive statistics;
- contact graph and graph-kernel baselines;
- minimum-description-length models;
- CCAT-style radius/radial/contact/motif grammar;
- unconstrained flexible model with matched complexity;
- simple null/nearest-neighbour models.

Report event-level macro-F1 only for a label task with adjudicated labels. For continuous reconstruction/prediction, use proper predictive scores and geometry error. Include event-level bootstrap intervals; the independent unit is the event, not each circle.

### Conditional radius-to-location permutation null

The standalone lab implements a specific geometric conditional null:
- preserve every observed circle centre exactly;
- preserve the complete multiset of measured radii;
- permute which radius is assigned to which centre;
- rerun the full frozen radius-class/contact-threshold family for every seed.

This tests whether the observed association between circle size and spatial position is unusual **conditional on the observed centre configuration and radius distribution**, under the hypothesis that radii are exchangeable across centres. It does not preserve the observed contact graph, does not generate a natural formation model, and does not test all possible human construction processes. It must be reported as one null family, never as "random crop circles".

The Rust APIs are `permute_circle_radii` and `run_radius_permutation_ensemble`. Pair their score matrix with `familywise_max_statistic_p_value` to obtain a max-statistic-adjusted empirical p-value across the entire declared class-count/tolerance family. The integrated unit test demonstrates the intended composition on the imported CCAT data; it remains unexecuted pending Rust tooling.

### Track D — generative grammar and reconstruction transfer

For each event, fit a model on a preregistered subset of primitives/regions and predict the held-out geometry. Evaluate:
- normalized Chamfer/Hausdorff error;
- primitive-count prediction error;
- reconstruction coverage and interval calibration;
- model description length (data residual + model complexity);
- parameter count and sensitivity;
- out-of-domain performance on entirely held-out events.

Compare grammar models at matched description/parameter budgets. A model that uses more hidden constants, mirrored copies or post-hoc exceptions must pay that complexity cost.

FEEDIT M3/M4 elements may be used as generator-specific synthetic controls, but they must be segregated from the M0/M1 measurement benchmark.

### Track E — geometric information/decoder tournament

- Freeze extraction, segmentation, orientation, transform, grid dimensions, codebook, bit order and decoding rules before test analysis.
- Run the entire family on positive, negative and matched geometric controls.
- Use a max-statistic family-wise test; every null replicate reruns the complete declared family.
- Include non-payload designs with comparable symmetry, density, primitive count and visual complexity.
- Report all attempts and "no supported decoding" as a valid result.
- Exact held-out payload recovery and independent blind replication are required for S3.

The Rust helper `familywise_max_statistic_p_value` rejects unequal family widths and computes the empirical p-value using the maximum of the entire family per null replicate. It does not make an invalid null exchangeable. The current 2×2 switch null is not qualified for inferential use.

### Track F — crop-flow, 3D pose and measured mechanics

Use the [field protocol](FIELD_PROTOCOL.md) and separate upright, mechanical-lodging and naturally lodged controls. Compare ground-level stem orientation, 3D/LiDAR, RGB texture direction and crop-flow descriptors. Calibrate posture proxies against raw force-displacement tests rather than inferring flexural rigidity or breaking moment from images. The schema now records specimen-level mechanical tests and raw curves; the lab includes idealized first-order beam reference equations only.

### Track G — controlled mechanism discrimination

Use [MECHANISM_BENCHMARK.md](MECHANISM_BENCHMARK.md) to define mechanism-specific predictions before collection. Split by independent plot/event and acquisition session; do not split stems from the same manipulated plot across train and test. First test blinded, controlled reference plots, then move to site/event holdout. Report mechanism confusion matrices, proper predictive scores, calibration, selective risk, abstention and event-clustered intervals. A model must predict held-out intervention outcomes or independent observations, not only reproduce the final footprint.

## 6. Split strategy and reporting

For a corpus of (N) distinct events:
- use grouped K-fold (e.g. 5 folds only if at least 5 independent groups exist);
- preserve site/source families for separate blocked evaluation;
- perform all feature choice and hyperparameter tuning inside training folds;
- keep final test event IDs sealed until the protocol and model are frozen;
- run a final leave-one-event-out analysis where feasible;
- report per-event results and event-clustered confidence intervals;
- never select a model using the final test event and then report that event as held out.

With only 11 FEEDIT candidate records, a 5-fold event-held-out score will have high uncertainty and low power. Use the corpus first for software replay and feasibility, not leaderboard claims. Do not count all generated sub-elements as independent samples.

## 7. Proposed performance targets

These are programme targets, not current results or universal SOTA claims.

| Area | Target |
|---|---|
| Split integrity | 100% of samples assigned once to validation; zero event/site/source leakage under declared grouping |
| Geometry parser | 100% exact fixture round-trip; strict rejection of malformed/ambiguous units |
| Primitive reconstruction | F1 ≥ 0.95 on synthetic controls; then establish empirically achievable held-out measured-data targets |
| Geometric accuracy | Centre/radius and boundary errors reported in calibrated units with 95% interval coverage assessed |
| Structural transfer | Outperform radial-only and simple graph/MDL baselines on event-held-out proper scores, with confidence intervals excluding no improvement before claiming superiority |
| Grammar efficiency | Better held-out reconstruction at equal or lower description length/parameter budget |
| Decoder | Family-wise α ≤ 0.01 only after null exchangeability, complete family replay and positive/negative control qualification |
| Plant posture | Reproduce known-angle synthetic fields; compare real sensor outputs to blinded manual/3D references |
| Calibration | Publish Brier/log score, coverage, selective risk, false-confident rate and abstention |
| Reproducibility | Exact dataset hashes, source/rights manifests, code SHA, configuration, random seed, folds and exclusions |

Do not claim a leaderboard win on one case, one split, one metric or an unrelated task.

## 8. Immediate execution plan

1. Add grouped event split tests and the family-wise max-statistic evaluator to the standalone Rust lab.
2. Validate those algorithms with synthetic multi-event fixtures before importing more data.
3. Obtain FEEDIT's machine-readable JSON geometry under CC BY-SA 4.0; keep M3/M4 generator records distinct from traced measurements.
4. Add at least two independent human digitizations for one rights-cleared case before claiming reconstruction accuracy.
5. Acquire multi-event, multi-site measurement references and negative geometric controls.
6. Run event-held-out and site-held-out benchmarks separately.
7. Only after those controls, assess whether any grammar generalizes across events.

**Current claim ceiling:** event-split, frozen geometry-family and family-wise statistical reference functions are authored, not executed. One CCAT event is available for fixed-input replay; FEEDIT contributes a metadata candidate index but its geometry JSON is not imported. No event-held-out model result exists yet.
