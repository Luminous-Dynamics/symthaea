# FDM Quality Benchmark Protocol v0.1

**Status:** research protocol proposal; no physical results claimed  
**Scope:** fused-filament fabrication (FFF/FDM) only  
**Owner:** Symthaea fabrication kernel, composing existing manufacturing, uncertainty, metrology, and authorization owners  
**Related work:** [MFG-VALUE-001 #7608](https://github.com/Luminous-Dynamics/symthaea/issues/7608), [MFG-ACTIVE-002 #7638](https://github.com/Luminous-Dynamics/symthaea/issues/7638), [MFG-ACTIVE-003 #7670](https://github.com/Luminous-Dynamics/symthaea/issues/7670), [INTX-FAB-ADAPTER-001 #7706](https://github.com/Luminous-Dynamics/symthaea/issues/7706), [MFG-PROC-000 #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686).

This is an FDM-specific physical-validation protocol that instantiates the existing production-value and experiment-selection work. It does not create a second manufacturing ontology, benchmark ledger, machine-control authority, or universal quality score.

## 1. Research question

Under equal time, material, machine, operator, and safety constraints, can a proposed calibration/optimization workflow reach a declared part-quality threshold with less operator effort, less calibration waste, or better conforming output than a strong conventional workflow?

Compare against a documented baseline such as a pinned OrcaSlicer calibration workflow and the printer's established firmware calibration tools. The baseline is not “factory defaults” unless that is the actual comparison being tested.

A useful system improves **conforming parts per total constrained resource**, not nominal print speed, an opaque AI-quality score, or simulation performance alone.

## 2. Paradigm shift

Move from a calibration wizard that produces preferred settings to a **machine-specific, evidence-bearing process envelope**:

1. Declare the part's critical features, dimensional tolerances, material/process constraints, inspection plan, and unacceptable outcomes.
2. Bind the exact machine configuration, nozzle/tooling, firmware, slicer, material lot, conditioning history, and measurement instruments.
3. Run a conventional baseline and/or select one discriminating experiment using the existing active-learning work.
4. Preserve raw observations, uncertainty, failures, rejected measurements, and the reason each candidate was chosen.
5. Freeze the winning candidate before a separate confirmatory run on held-out parts/runs/lots.
6. Accept a profile only for the exact tested applicability envelope. Drift, maintenance, hardware swaps, material changes, or calibration expiry may narrow or invalidate applicability; they must not rewrite historical results.

A generated toolpath is not a conforming part. A completed job is not proof of dimensional accuracy. A simulated or synthetic result is not physical evidence.

## 3. Scope and non-goals

### In scope

- FDM/FFF dimensional and geometric capability.
- Conventional calibration versus offline or shadow-mode adaptive experiment selection.
- Repeatable test artifacts and metrology plans.
- Machine/material/process profile identity and applicability.
- Deterministic fixture and recorded-data replay.
- A staged bridge toward a supervised physical pilot.
- Leptos/Bevy views for inspecting toolpaths, profile changes, uncertainty and benchmark evidence after the underlying data contract is stable.

### Explicitly out of scope for v0.1

- Claiming ISO/ASTM 52902 conformity or reproducing copyrighted standard test-artifact designs.
- Qualifying material strength, fatigue life, chemical resistance, biocompatibility, fire performance or safety-critical parts from dimensional coupons.
- Generalizing FDM thresholds to resin, powder-bed fusion, CNC, or other processes.
- Autonomous live printer control, arbitrary G-code submission, or replacing firmware interlocks.
- Treating a webcam image or model confidence as dimensional metrology.
- Claiming that Symtropy is already a high-fidelity thermal/structural/flow solver.
- Claiming the current synthetic DefectPredictor training example establishes real defect prediction: its training labels come from a code-defined quality function.

## 4. Existing-tool baseline

Freeze exact software versions and profile exports for every benchmark generation.

- Slicer baseline: OrcaSlicer or PrusaSlicer, pinned to an exact release and configuration export.
- Firmware baseline: the exact installed firmware and configuration, beginning with Klipper where available.
- Calibration sequence: record nozzle/bed temperature, maximum volumetric speed, pressure advance, flow, retraction, tolerance and resonance/input-shaping methods when applicable. Keep the exact sequence and stopping rule fixed before scoring.
- Control integration: treat the current Symthaea MockPrinter as a test double. The codebase documents OctoPrint/Moonraker live transports as unavailable until real transports exist and are tested.
- Metrology baseline: use an explicit measurement plan, instrument identification, calibration status, resolution, environmental conditions, and uncertainty. Use the current QIF/FIELD/metrology owners as the integration destination rather than inventing a new quality standard.

Relevant references:
- ISO/ASTM 52902:2023, geometric capability assessment using additive-manufacturing test artifacts: https://www.iso.org/standard/79683.html
- OrcaSlicer calibration guide: https://www.orcaslicer.com/wiki/calibration_guide/
- Klipper resonance-compensation guidance, including the ringing-versus-smoothing trade-off: https://www.klipper3d.org/Resonance_Compensation.html
- NIST Digital Twins for Advanced Manufacturing: https://www.nist.gov/programs-projects/digital-twins-advanced-manufacturing
- ISO 23247-1:2021, digital twin framework for manufacturing: https://www.iso.org/standard/75066.html

These references guide the protocol; linking them is not evidence of compliance or endorsement.

## Research update (reviewed 2026-10-18)

The current evidence supports a measurement-first, uncertainty-aware benchmark rather than a single learned quality score.

- **Standards boundary:** ISO/ASTM 52902:2023 defines AM test-artifact geometry and the quantities/qualities used to characterize geometric capability, including measurement-uncertainty considerations. It does not prescribe a universal slicer recipe or a particular manufacturing procedure. Our artifact and measurement plan should be explicitly versioned independently from candidate calibration policies. [ISO/ASTM 52902:2023](https://www.iso.org/standard/79683.html).
- **Integrated calibration artifacts:** a 2025 PLA FFF study presents a test artifact combining defect identification, measurement, evaluation and compensation, together with a fit kit for tolerance evaluation. This is a useful baseline direction: our benchmark should test whether a workflow's measurements and compensation improve on an integrated conventional artifact, not merely on uncalibrated defaults. [He et al., *Polymer Testing* (2025), DOI 10.1016/j.polymertesting.2025.108834](https://doi.org/10.1016/j.polymertesting.2025.108834).
- **Layer-wise measurement lineage:** a September 2025 study describes calibrated imaging, a G-code-derived digital reference, contour-error quantification in physical units, and synchronized process/thermal measurements. This motivates retaining measurement calibration, reference geometry, process state and source identity together. Vision can be a useful low-cost observation channel, but its calibration and measurement uncertainty must be validated against physical artifacts before its output is treated as dimensional evidence. [Haseltalab et al., *International Journal of Advanced Manufacturing Technology* (2025), DOI 10.1007/s00170-025-16429-3](https://doi.org/10.1007/s00170-025-16429-3).
- **Calibrated prediction, not just high accuracy:** a 2025 FFF modeling paper reports calibrated prediction intervals and multi-objective optimization across PLA, ABS, PETG and TPU using a hybrid synthetic/literature-anchored dataset. Its reported results motivate uncertainty calibration and material-specific profiles, but do not establish cross-printer transfer or the performance of this repository. Require held-out profiles/lots and physical confirmatory runs before transfer claims. [Aquisman Asare et al., *Measurement* (2025), DOI 10.1016/j.measurement.2025.119350](https://www.sciencedirect.com/science/article/abs/pii/S0263224125027095).
- **Digital-twin credibility:** NIST's 2026 manufacturing digital-twin workshop summary identifies interoperability, verification/validation/uncertainty quantification (VVUQ), cybersecurity and workforce readiness as persistent priorities. A visually convincing twin is not evidence of model validity. [NIST, *Digital Twins Workshops Summary Report*, published July 21, 2026](https://www.nist.gov/publications/digital-twins-workshops-summary-report).

### Spatial capability is a hypothesis to test

A single global printer profile may conceal variation by build location, orientation, axis, motion regime, or thermal history. ISO/ASTM 52902:2023 treats test artifact position/orientation and measurement uncertainty as part of meaningful capability characterization. ASTM's published scope describes the need to consider build-platform coverage and repeatability for a complete machine-performance study. These ideas motivate the separate [FDM-SPATIAL-001 workstream](https://github.com/Luminous-Dynamics/symthaea/issues/7998).

A research example is Edoimioya, Chou and Okwudire, “Vibration Compensation of Delta 3D Printer with Position-varying Dynamics using Filtered B-Splines” ([arXiv:2209.06791](https://arxiv.org/abs/2209.06791)), which studies position-varying dynamics and reports simulation and physical experiments for a delta-printer context. This result does not imply the same compensation applies to CoreXY, Cartesian or other machine designs. It makes spatial dependence a testable hypothesis, not a universal assumption.

For each physical specimen, bind its position and orientation to a declared machine coordinate frame. For a multi-position campaign, stratify results by location and orientation; do not let a strong overall mean hide localized critical failures. Predictions between measured locations remain estimates and must include a validated applicability domain and uncertainty. A spatial correction is a proposal until independently tested; the benchmark must not silently modify live G-code.

### Consequences for this protocol

1. Keep geometry/artifact identity, process recipe, canonical machine profile, observed machine behavior and measurement system as separate versioned identities; reuse the machine-profile owner for build bounds rather than maintaining a parallel envelope model.
2. Score the measurement system itself: repeatability, reproducibility where feasible, bias against a reference, resolution, uncertainty and missingness.
3. Preserve both signed error (to reveal systematic bias) and absolute error (to measure magnitude); do not let positive and negative deviations cancel in one aggregate.
4. Separate model training, adaptive discovery and held-out confirmation by run/profile/material-lot identity. Randomly splitting individual measurements from the same print across train and test is not sufficient evidence of generalization.
5. Treat camera/vision observations as a distinct measurement method with its own calibration and error model. They may provide fine-grained layer-wise trends, but visual plausibility alone cannot certify dimensions, interlayer strength or internal defects.
6. Report calibration effort and waste alongside quality. A predictor that reduces a statistical loss but does not reduce the resource cost of reaching conforming parts has not demonstrated the desired operational benefit.
7. Do not assert that these research papers validate our system; their methods motivate the experiment design only.

## 5. Benchmark artifact family

Create or select openly licensed, revision-pinned artifacts that cover distinct failure modes. Do not optimize all decisions against one decorative test print.

| Artifact | Primary observations | Limits |
|---|---|---|
| Datum-bearing dimensional coupon | X/Y/Z dimensions, squareness, repeatability, position dependence | Does not establish every geometry's accuracy |
| Hole/pin and clearance array | Hole/pin error, fit classes, clearance feasibility | Fit conclusions apply only to the tested orientation, feature scale and process |
| Thin-wall and small-feature coupon | Feature survival, wall width, minimum reproducible gap | Not proof of exact wall thickness without suitable metrology |
| Bridge/overhang coupon | Unsupported-span behavior, sag/edge deviation, support dependence | Visual grades alone are not dimensional inspection |
| Resonance/ringing geometry | Ringing amplitude, corner error, smoothing/detail loss | Must follow firmware guidance and record settings that affect the result |
| Optional tensile coupon | Strength/strain data using an appropriate test method | Requires separate specimen, test-machine, conditioning, and material-method qualification |

Each artifact revision must identify its intended measurands, nominal dimensions, critical features, build orientation, placement rules, acceptance rules, and known limitations. Where a formal standard is used, cite the exact edition and obtain the needed rights; do not imply compliance merely because an artifact resembles a standard.

## 6. Minimum experimental identity

Each run manifest must bind, or explicitly mark unavailable:

- Benchmark protocol version and exact source revision.
- Artifact source and mesh digest, units, mesh validation result, orientation, placement, and slicer project digest.
- Machine identity at the strongest verified level; kinematics; motion system; frame/toolhead changes; nozzle diameter and wear state.
- Bind the exact canonical `MachineProfile` used by the fabrication-kernel validator. Build-volume bounds should be taken from that profile, not duplicated into a competing envelope schema. Keep a digest of the embedded profile and treat it as identity, not a signature or proof of trusted origin.
- Spatial samples should bind their declared frame and nominal position/orientation to that same profile. An origin-in-bounds check is only a precondition; it does not prove the whole specimen fits or clearances are collision-free.
- Firmware release and configuration digest; controller and relevant motion settings.
- Slicer release, profile digest, relevant process settings, and generated G-code digest.
- Material family, manufacturer, product, lot/spool identity where available, declared diameter, storage/conditioning history, and known uncertainty.
- Ambient/environment observations where relevant.
- Instrument identity, calibration status, method, resolution, operator or automation profile, raw observations, units, uncertainty, and missing features.
- Experiment selector/policy version, candidate set, hard-gate dispositions, random seed if applicable, and reason for choosing the experiment.
- Start/end and duration semantics, operator interventions, failures, aborts, and reprints.
- Receipt identities that point to authoritative machine/process/metrology/evidence owners; do not duplicate their authority.

The manifest must distinguish specified settings, measured configuration, reported telemetry, inferred values, synthetic values, and unknowns. Unknown is not zero and does not satisfy an acceptance condition.

## 7. Experimental design

### Arm A — conventional baseline

Use the pinned, documented conventional workflow. Preserve its total operator time, print time, calibration material, consumed filament, configuration changes, and final settings. A successful calibration is not assumed; failures remain in the denominator.

### Arm B — candidate workflow

Initially, use Symthaea only in offline proposal or shadow mode. It may choose among declared feasible experiments using the existing experiment-selection interface. It cannot bypass hard process constraints, the machine profile, metrology requirements, or authorization gates.

A predicted high-quality score is not acceptance. The candidate profile must pass the same independent confirmatory checks as Arm A.

### Fair-comparison rules

- Use the same eligible machine class, artifact revisions, material-lot blocks, measurement system, environmental controls where feasible, and resource budget.
- Randomize or interleave order within declared blocks when safe; record order effects and maintenance interventions.
- Separate adaptive discovery from confirmation. Confirmation items must not have been used to choose the winning setting.
- Freeze the primary endpoint, guardrails, exclusions, data split and analysis before examining confirmatory results.
- Use paired or blocked comparisons and an a priori sample-size/power or precision rationale. Do not choose a sample count because it makes an early result look favorable.
- Report all attempted jobs, invalid or missing measurements, failed prints, reprints and censored “not found within budget” campaigns.
- If sample size or measurement quality cannot distinguish improvement from no effect, report inconclusive.

## 8. Metrics — never collapse into one score

### Primary endpoint (choose one per campaign)

Choose before the confirmatory phase, for example:
- Operator minutes to first independently conforming part; or
- Total calibration resource cost to first independently conforming part, including artifact material, energy and machine occupancy; or
- Absolute error for a named set of critical dimensions, subject to predeclared bounds on time, material and yield.

### Required secondary metrics

- Per-feature signed error, absolute error, repeatability, measurement uncertainty and spatial dependence across bed position/Z height.
- First-pass yield and reprint/scrap rate, with denominator and missing results disclosed.
- Time and filament spent on calibration artifacts; total machine occupancy.
- Energy per independently conforming part when metering quality is known.
- Surface/ringing metrics and a distinct smoothing/fine-feature penalty.
- Calibration stability after declared time/use, configuration change, nozzle/toolhead change, or material-lot change.
- Prediction calibration and error on held-out data only when the target is measurable and the dataset supports a meaningful estimate.
- Uncertainty/unknown rates; rejected data; failed/censored campaigns; traceability completeness.
- User effort and task completion for the application, measured separately from physical manufacturing outcomes.

### Hard guardrails

- A candidate that violates any declared critical dimension or process limit fails even if faster or cheaper.
- Do not trade a critical quality failure for throughput/energy gain through a composite score.
- Unmeasured critical features remain Unknown; no pass-by-default.
- Old calibration evidence cannot certify a modified machine/profile without an explicit applicability assessment.
- Failure to detect a defect is counted as a miss when a confirmed defect is in the evaluation set.
- Job-completed, status-reported, simulation-passed, or digest-matched states cannot mint physical conformity.

## 9. Proposed improvement targets

These are **research targets**, not measured performance claims or industry norms. Set the actual thresholds after baseline variance and instrument capability are known.

- **Protocol determinism:** 100% byte-stable normalized synthetic/replay outputs for identical manifest, code, policy and seed.
- **Negative controls:** 100% rejection of the frozen invalid-profile, stale-calibration, missing-critical-measurement, altered-digest, leaked-holdout and critical-limit-violation fixtures.
- **Practical pilot hypothesis:** at least 20% less operator calibration time or calibration-artifact waste to reach the same predeclared quality threshold.
- **Quality hypothesis:** aim for at least 10% reduction in error on the preselected critical-feature set, or a demonstrably better time/quality Pareto point. This is secondary unless selected as the campaign's primary endpoint.
- **No-regression guardrails:** no worsening beyond the predeclared non-inferiority margin for first-pass yield, any critical feature, or process limits.
- **Evidence completeness:** every claimed confirmatory result links to its exact run, profile, artifact and measurement records; unknown/missing fields remain visible.
- **Reproducibility:** an independent evaluator recomputes derived metrics from raw admitted observations, not from candidate-reported summary scores.

The physical target is achieved only if uncertainty intervals and the pre-registered decision rule support it on the confirmatory set. Otherwise report Worse, Inconclusive, or NotEvaluable.

## 10. Staged gates

### G0 — protocol and synthetic fixtures

Freeze schemas, artifact identity, test cases, invalid/negative controls and deterministic receipts. Passing G0 proves protocol/implementation behavior only.

### G1 — offline toolpath/profile validation

Use existing slicer adapters or pinned file exports to compare configuration and statically inspect generated programs. Add independent G-code parsing/simulation before making stronger collision or machine-state claims. No live command submission.

### G2 — recorded-data replay

Admit only lawful, versioned records with source, unit, quality, timestamps and missingness semantics preserved. Keep synthetic, simulated, replayed and physical evidence classes distinct.

### G3 — shadow-mode recommendation

Compare candidate recommendations with the baseline without changing live machine settings. Evaluate feasibility escapes, prediction error, applicability detection and the rate of recommendations that would violate guardrails.

### G4 — supervised physical pilot

Only after the live adapter, authentication, authorization, machine profile, failure reconciliation and independent stop path are explicitly qualified. Require operator approval, a bounded non-safety-critical part, documented maintenance state, verified measurement setup and an agreed rollback/abort plan. Start with a single machine and a small preregistered campaign.

### G5 — cross-machine generalization

A profile trained on one machine does not transfer automatically. Test transfer to a separate machine/configuration and report any required recalibration. Preserve the original result's profile boundary.

Queued or absent CI, a successful mock, and an unexecuted physical protocol are not passing evidence.

## 11. Cross-project integration

- **Symthaea:** existing fabrication kernel, process reasoning and experiment-selection policy; the benchmark consumes existing identities and uncertainty/measurement owners.
- **Symtropy:** visualization, deterministic replay, and simulation where the specific model is validated for the question. Simulation never substitutes for physical confirmation.
- **Mycelix:** later source/lot/process and custody provenance, controlled recipe sharing, and work-order linkage. Do not require distributed identity or governance to test the local offline benchmark.
- **Sol-Atlas:** later workshop/fleet/resource planning (capacity, energy and material availability), not part of the single-printer quality gate.
- **Leptos:** configuration, experiment plan, job history, evidence report, accessibility and multi-printer/fleet workflows.
- **Bevy:** interactive machine/toolpath twin, layer-by-layer replay, collision visualization, motion telemetry and inspection overlays. The UI must label observed, predicted, simulated and unknown data distinctly.

Shared domain types and durable evidence contracts should be in ordinary Rust crates; the rendering engine must not own manufacturing truth or live-control authority.

## 12. First implementation tranche

1. Add this protocol as a stable, versioned specification and review the ownership links.
2. Audit existing artifact/manifest, machine-profile, defect-prediction and metrology types; record actual coverage and semantic gaps before adding types.
3. Implement the smallest deterministic FDM benchmark manifest/fixture corpus in Rust, with independent metric recomputation and frozen expected outcomes.
4. Add hostile tests for stale profile, altered toolpath digest, missing critical measurement, duplicated/reordered observation, leaked confirmatory sample, a faster-but-out-of-tolerance candidate, and synthetic labels presented as physical truth.
5. Bind a first read-only observation path only after the existing adapter issue's source audit establishes the exact interface. Keep hardware simulation default.
6. Publish baseline output, code head, toolchain and corpus digests before selecting a physical pilot.

## 13. Exit criteria

The first increment is complete only when the exact-head report contains:
- Protocol and corpus digests.
- Explicit source/evidence class for every observation.
- Reproducible baseline/candidate comparison with complete outcomes.
- Independent evaluator agreement and all required negative controls.
- No live hardware claims unless the corresponding transport and physical evidence actually exist.
- A clear list of unresolved physical/metrology gates.

Passing this protocol does not establish industry-leading quality, safe parts, universal machine compatibility, production savings, standards conformity or manufacturing readiness. It establishes a falsifiable route to testing those claims.
