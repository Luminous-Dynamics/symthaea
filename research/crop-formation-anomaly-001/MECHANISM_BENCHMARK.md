# CROP-INTEL-001 — Mechanism discrimination and plant mechanics benchmark

**Version:** 0.1 research design + analytical reference implementation  
**Status:** first-order Rust equations authored; not compiled or executed; no field dataset or mechanical campaign exists  
**Claim ceiling:** this work does not identify the cause of any reported formation. It adds measurements and falsifiable comparators, not evidence for an extraordinary mechanism.

## Executive change: stop ranking anomalies; start discriminating mechanisms

A visually unusual pattern is not a physical explanation. The scientific target is not to maximize an anomaly score or train a classifier on a poorly defined "genuine" label. The target is to establish which observations distinguish plausible mechanisms, what those observations can identify, and when the answer is not identifiable.

The new programme loop is:

1. **Specify competing mechanisms** before seeing the target measurements.
2. **Predict observables** at plant, canopy, field and event scales.
3. **Measure in calibrated coordinates** with instrument uncertainty and matched controls.
4. **Test mechanics** against measured force-displacement and plant state.
5. **Predict held-out observations**, not just the already observed outline.
6. **Choose the next measurement** that most reduces disagreement among surviving models, subject to permission, power, safety and cost.
7. **Abstain** when the mechanisms remain observationally equivalent.

The crucial paradigm shift is from *pattern recognition* to *causal identifiability*. A model is valuable when it can be wrong in a specific, predeclared way.

## 1. Do not collapse mechanisms into one label

Keep authorship, physical process, crop response, geometric structure, information content and evidence quality orthogonal. For the physical channel, use a hypothesis set such as:

| Candidate | Mechanism family | Example discriminating observations to test |
|---|---|---|
| H1 | Distributed natural loading (wind, gust, rain, soil saturation) | Time-aligned weather and terrain; coherent field-scale orientation; root-plate versus stem failure; crop-stage and edge effects |
| H2 | Localized human mechanical flattening (foot, board, roller, rope, hand tools) | Directional bends, repeated access/turning paths, localized compression, broken nodes, contact/track evidence, experimentally matched controls |
| H3 | Mechanical layout/construction using a central anchor, rope or radial guide | Geometry and plant deformation may share a centre or angular structure; test anchor/access traces and reproduce with documented construction controls |
| H4 | Machinery or vehicle interaction | Row/track alignment, soil compaction, wheel/implement widths, crop damage and access geometry |
| H5 | Mixed, transformed or unknown process | Multiple signatures or observations that are inconsistent with all registered models |

These are hypothesis families, not unique fingerprints. Wind can be directional; a human method can be radial; later weather can alter a human-made formation. No single crop-flow vector, circle, boundary or damaged stem is exclusive to one cause. Record combinations of evidence and their dependencies.

A "not identifiable" result is scientifically correct when two mechanisms predict nearly the same observable distribution at the available resolution.

## 2. The multiscale state to measure

### Individual plant / stem

Where access and crop condition permit, capture:

- directed base-to-tip 3D stem-axis vector in a gravity-referenced field frame;
- stem inclination, azimuth, bend point and curvature profile;
- root plate / root anchorage condition, stem break, kink, crushing and node damage as separate labels;
- major and minor stem diameters, wall thickness, internode length and cross-section orientation;
- sheath, leaf and ear posture separately from the main stem;
- moisture, cultivar/species, phenological stage, soil condition and time since deformation;
- repeated measurement uncertainty, occlusion, instrument, calibration and assessor.

Aerial RGB posture is a proxy. 3D point-cloud or photogrammetry outputs are still measurements with reconstruction error, occlusion and segmentation uncertainty. Preserve source point clouds/images and reference a manually measured audit subset.

### Plant cluster / canopy

Retain the complete local orientation distribution, not only a global mean. Estimate multimodality, circular concentration, neighbour alignment, local bend/flatten density and radial/tangential components only when the formation centre is independently surveyed in the same coordinate frame.

### Field / event

Record georeferenced boundary transitions, standing/bent/flattened/broken regions, crop-row direction, slope, soil, access/track evidence, acquisition time, weather observations and uncertainty. A centre or boundary estimated from an image must not silently become exact ground truth.

The independent statistical unit is the formation, field or controlled plot according to the target claim—not the number of stems. Plants are nested in transects, locations, plots and events.

## 3. Mechanical reference equations added to the Rust lab

New module: `lab/src/mechanics.rs`.

It supplies a small set of transparent SI-unit calculations:

- Concentric hollow elliptical cross-section second moments about both principal axes:
  - (I_x = \frac{\pi}{4}(a_o b_o^3-a_i b_i^3))
  - (I_y = \frac{\pi}{4}(a_o^3 b_o-a_i^3 b_i))
- Principal-axis section moduli: (Z_x=I_x/b_o), (Z_y=I_y/a_o).
- Three-point bending stiffness estimate: (EI=kL^3/48), where (k) is the linear force/displacement slope and (L) is support span.
- Three-point bending peak moment: (M_{max}=FL/4).
- Cantilever tip-load estimate: (EI=FL^3/(3\delta)); base moment (M=FL).
- Linear local curvature: (κ=M/EI).
- Single-axis outer-fibre stress: (|\sigma_{max}|=|M|/Z).

The implementation validates finite positive dimensions and loads, supports solid or concentric hollow elliptical sections, and tests principal-axis anisotropy. The authored unit tests are analytical checks; they do not validate a living-plant constitutive model.

### Assumptions and explicit limitations

These are first-order Euler-Bernoulli beam relationships for idealized, small-deflection, linear-elastic cases. Crop stems can be tapered, hollow, elliptical, anisotropic, layered, viscoelastic, hydrated, locally damaged and coupled to leaves, nodes, roots and soil. The modulus and effective section may change with direction, moisture, loading rate and test preparation. A single global (EI) is not an adequate representation of a nonuniform stem unless that approximation is justified.

Do not use the equations to infer material modulus from posture alone, infer a historic force from a final shape without boundary conditions, or infer the source of a formation. Preserve raw force-displacement curves and fit/calibrate model parameters from controlled measurements. For spatially varying stems, use segment-wise or finite-element models only after cross-section, material and boundary assumptions are independently justified.

### Protocol implications

- Standardize specimen orientation, support span, fixture shape, load point, preload/preconditioning, moisture, temperature, displacement rate and time since collection.
- Record the force-displacement curve, not just a derived stiffness or failure score.
- Separate nondestructive stiffness measurements from destructive breaking tests.
- Record whether the sheath remains attached; it can materially change wheat-stem response.
- Distinguish root lodging, stem bending and stem breaking. They are different failure modes.
- Do not mix test rates across comparison groups without a registered rate model or stratification.

Tabaracci, Vos & Robertson (2024) report rate effects in maize and wheat stem testing: the tested rate had a small effect on some three-point bending estimates in their mature, dry samples, while rind-puncture measurements were strongly rate-dependent. They emphasize consistent span lengths and displacement rates, and note that green stems may show larger effects. This is exactly why a measured mechanical endpoint needs a standardized protocol and why puncture force must not be treated as a universal surrogate for whole-stem bending behaviour. [Plant Methods 20, 125 (2024)](https://doi.org/10.1186/s13007-024-01253-9).

## 4. Falsifiable experiment matrix

Do not build a mechanism classifier from photographs with subjective origin labels. First create controlled, blinded reference data in permitted experimental plots and instrumented samples. A staged design:

### Stage A — analytical / instrument controls

1. Check the Rust formula implementation against hand-calculated fixtures and a second independent numerical implementation.
2. Test circular, elliptical-solid and elliptical-hollow reference specimens in at least two bending orientations.
3. Calibrate 3D axis reconstruction against rods or reference objects at known inclinations and azimuths.
4. Quantify inter-annotator agreement and repeat-measurement uncertainty before field claims.

### Stage B — controlled mechanical signatures

Within an approved experimental area, document controlled examples of distinct mechanical actions and naturally generated reference conditions where feasible. Do not modify a reported formation. Capture pre/post geometry, exact method, applied load/time, soil state, crop type/stage, weather and plant-level damage. Randomize sample labels and blind lab assessors. Use a separate control plot and preserve all exclusions.

The design should include both:
- a mechanism-development set, on which models and features are chosen; and
- sealed, independently created test plots/events, acquired later or by another team.

Replicates are independent plots / sessions, not multiple plants from one manipulated patch.

### Stage C — external transfer

Evaluate on different fields, cultivars, crop stages, acquisition systems and weather regimes. Hold out complete event/site/source groups. A model that only works on plots from the same camera, field, cultivar or experimental session has not demonstrated field transfer.

### Stage D — reported-event case study

Only after Stages A–C have fixed the measurement and analysis pipeline, consider a permitted reported event with available matched controls. A single reported event may support a detailed case study; it cannot establish general mechanism accuracy or non-human origin.

## 5. Benchmark tracks and scorecard

| Track | Primary question | Required outputs |
|---|---|---|
| M0 — equation/instrument | Are the reference equations and sensors correct on known inputs? | Exact fixtures, calibration residuals, uncertainty, repeatability |
| M1 — stem geometry | Does 3D sensing measure stem axes/cross-sections accurately? | Angular error, dimensional error, p95 error, interval coverage, abstention |
| M2 — mechanical response | Does the model predict controlled force-displacement behaviour? | Curve NRMSE, EI error, peak-moment error, failure-mode agreement |
| M3 — posture structure | Does the model describe local and canopy-scale crop-flow reliably? | Orientation distribution distance, boundary error, field-level calibration |
| M4 — controlled mechanism discrimination | Can blinded, independent plots be distinguished across registered mechanism families? | Event-level macro-F1, confusion matrix, proper scores, calibration, abstention |
| M5 — external causal transfer | Does predictive discrimination survive new sites/events and different acquisition conditions? | Leave-site/event-out performance, event-clustered intervals, selective risk |

### Proposed engineering gates

These are programme targets to be tested, not achieved results or claimed universal SOTA:

- **Equation reference:** analytical fixtures agree with independent calculations to relative error ≤ (10^{-9}) for non-degenerate test values.
- **3D stem axis:** median absolute angular error ≤ 5° and p95 ≤ 15° against blinded manual/reference measurements, with uncertainty coverage reported.
- **Cross-section:** relative error ≤ 10% on calibration artefacts, then establish biological repeatability empirically.
- **Mechanical response:** force-displacement curve NRMSE ≤ 10% on the declared linear-elastic fixture/test regime; report failure prediction separately.
- **Repeatability:** target ICC ≥ 0.90 for primary non-destructive mechanical and posture measures on reference repeats; report confidence intervals and between-day/operator components.
- **Uncertainty:** nominal 95% predictive intervals should cover 90–98% of held-out reference observations; include calibration curves and negative-control false-confidence rates.
- **Mechanism discrimination:** only set a numeric classifier target after independent plot-level labels exist. A future candidate gate is macro-F1 ≥ 0.90 on blinded held-out controlled plots, while simultaneously beating a simple physical-feature baseline and meeting calibration/abstention gates. This is not a target for classifying historic formations.
- **External transfer:** no superiority claim without event/site-held-out evaluation, event-clustered confidence intervals and performance against the same frozen baselines.

Never optimize a single score. Include confusion matrices, per-mechanism recall, Brier/log scores, calibration, selective risk versus coverage, abstention, p95 error, cost, runtime and human review time. If an unknown mechanism is not represented in training, the system must be allowed to abstain rather than force a known label.

## 6. Industry and research SOTA: what is comparable and what is not

The adjacent agricultural field is advancing quickly, but most published results answer *where is lodging?*, *what is the canopy posture?* or *which plant part is this?* They do not identify the cause of a crop formation.

- **Wheat ground-LiDAR field phenotyping (2025):** Ravichandran et al. report 90 plots, 30 historical cultivars, six Canadian research stations and two field seasons; their article reports up to 0.97 quadratic-weighted kappa and 98.51% macro-F1 in its evaluated lodging classification settings. This is a strong adjacent benchmark for field posture/lodging sensing, but its classes, sampling, ground truth and task are not comparable to crop-formation mechanism attribution. [DOI: 10.1080/07038992.2025.2516742](https://doi.org/10.1080/07038992.2025.2516742).
- **Soybean UAV phenotyping (2026):** Zhao et al. report lodging classification accuracy of 0.81–0.92 across diverse soybean genotypes, combined with transcriptomic analysis and candidate-gene identification. This is a useful multimodal agronomy comparator, not a crop-circle benchmark. [DOI: 10.1016/j.indcrop.2026.124188](https://doi.org/10.1016/j.indcrop.2026.124188).
- **3D point-cloud phenotyping (2026):** a recent single-plant segmentation framework reports improved F1 across crop datasets and combines zero-shot segmentation with point-cloud completion and trait extraction. It provides a comparator for plant-instance segmentation and morphology, not causal mechanism identification. [Holistic 3D framework, Computers and Electronics in Agriculture (2026)](https://www.sciencedirect.com/science/article/pii/S0168169926001407).
- **Cross-scale lodging review (2026):** the recent review argues for connecting tissue/organ mechanics, root-shoot coupling, whole-plant failure and remote-sensing structure, and highlights cross-environment validation and standardized biomechanics as priorities. [DOI: 10.3389/fpls.2026.1956108](https://doi.org/10.3389/fpls.2026.1956108).

### How CROP-INTEL should outperform the current workflow

Do not claim to beat a 0.97 lodging classifier with an unrelated crop-circle score. First reproduce relevant baselines on their own task and match their ground-truth quality. Then demonstrate advantages on currently under-served dimensions:

1. **Causal/mechanical grounding:** posture predictions calibrated against force-displacement, failure mode and root-plate observations rather than only visual labels.
2. **Cross-event generalization:** complete site/event holdouts, not random plant-level or within-image splits.
3. **Uncertainty and abstention:** calibrated unknown handling, p95 errors, selective risk and false-confident rate.
4. **Measurement traceability:** raw sensor data, calibration, uncertainty, rights, sample custody and exact reproducible receipts.
5. **Intervention prediction:** predict how measured geometry changes under a known loading protocol, then evaluate on a sealed independent intervention.
6. **Efficiency and accessibility:** publish low-cost ground-photogrammetry / phone-depth baselines as well as LiDAR; compare cost, operator time and measurement accuracy, not just model accuracy.

This is the route to a real paradigm shift: make plant posture, geometry and mechanical response one calibrated evidence chain, while refusing to infer a cause from a visual resemblance.

## 7. Statistical and governance safeguards

- Freeze primary endpoints, mechanism definitions, test protocols, exclusion rules, split IDs and analysis code before unblinding.
- Treat event/site/plot as the independent unit. Use hierarchical models for plants nested within transects and plots.
- Keep measurement uncertainty separate from biological variation and model uncertainty.
- Preserve null, inconclusive and unknown outcomes.
- Include matched human-authored geometric controls and naturally lodged crop controls; no one null model covers all human or weather processes.
- Correct for all model, feature, threshold and hypothesis searches that informed the result.
- Have a second analyst replay the result from frozen inputs.
- Permission, safety, landowner agreement and privacy take precedence over data collection.
- Provenance demonstrates how a record was handled; it does not prove the record's causal interpretation.

## 8. Implementation status and next tests

Implemented in this increment:
- `lab/src/mechanics.rs` added with validated elliptical-section properties, three-point bending, cantilever, curvature and stress reference calculations.
- Eleven Rust unit tests authored for analytical values, axis anisotropy, hollow-section behaviour and invalid inputs.
- `lib.rs` exports the mechanics API.

Not done:
- Rust formatting, compilation, tests and Clippy have not run in this environment.
- No force-displacement dataset, instrument calibration, field measurements or mechanical validation exists.
- No formation mechanism has been assigned.
- The reference model has not been coupled to Symtropy or to a production point-cloud pipeline.

Exact-head commands remain mandatory:

```sh
cd research/crop-formation-anomaly-001/lab
cargo fmt -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

## References

1. Tabaracci, K., Vos, J. & Robertson, D. J. (2024). [The effect of testing rate on biomechanical measurements related to stalk lodging](https://doi.org/10.1186/s13007-024-01253-9). *Plant Methods*, 20, 125.
2. Ravichandran et al. (2025). [Precision phenotyping in wheat: LiDAR-based plant height estimation and lodging classification using uncrewed ground vehicles](https://doi.org/10.1080/07038992.2025.2516742).
3. Zhao et al. (2026). [A breeding-oriented UAV phenotyping framework for scalable lodging assessment and candidate gene identification in soybean](https://doi.org/10.1016/j.indcrop.2026.124188).
4. Jing et al. (2026). [Multi-scale lodging in rice and wheat: bridging biomechanical traits, phenotypic detection, and intelligent harvesting systems](https://doi.org/10.3389/fpls.2026.1956108).
5. Liu et al. (2026). [A holistic 3D framework for single-plant point cloud segmentation and phenotypic trait extraction](https://www.sciencedirect.com/science/article/pii/S0168169926001407).
