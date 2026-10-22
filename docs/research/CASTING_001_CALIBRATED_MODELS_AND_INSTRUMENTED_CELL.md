# CASTING-001 — Calibrated Models and an Instrumented Foundry Cell

**Status:** Research design / first software slice  
**Owner:** Symthaea physics and manufacturing  
**Scope:** Low-volume, non-ferrous casting research; initial process family is sand-cast test coupons  
**Claim ceiling:** Analytical prototype only. No process, machine, material, or part is qualified by this document or the accompanying code.

## Executive decision

Build a measured, model-calibrated casting workflow rather than an opaque defect classifier or a one-shot “AI optimized” recipe.

The first software slice adds a Chvorinov-style solidification-time model fit to measured observations. Its parameters are calibrated from data, its units are explicit, its fit diagnostics are exposed, and it refuses predictions outside its observed thermal-modulus range. It is a narrow first-order model—not a fluid-flow, feeding, microstructure, shrinkage, strength, or acceptance solver.

The first machine-design target is an instrumented, guarded, low-volume non-ferrous casting cell with independent machine safety controls and a read-only Symthaea monitoring/advisory path. Start with test coupons and documented manual sign-off. Do not begin with autonomous pouring, pressure casting, vacuum systems, or production-critical parts.

## Architecture and ownership

The broader manufacturing contract is already addressed by [MFG-PROC-000 (#5686)](https://github.com/Luminous-Dynamics/symthaea/issues/5686). This work is a process-model specialization under that architecture, not a competing process registry.

Keep the following claims separate:

    process family represented
    != process model calibrated
    != prediction accurate on held-out castings
    != equipment capability demonstrated
    != recipe qualified
    != physical execution authorized
    != casting accepted for service

Symthaea owns process-model and manufacturability reasoning. The fabrication kernel/HAL owns machine-program validation, execution authority, operator commands, and hardware release. Mycelix owns distributed operations, work orders, supplier/resource offers, and production lineage. Physical measurements and inspection require their appropriate metrology/evidence path.

The initial Rust model is analysis-only. It adds no device driver, controller, recipe loader, machine command, automatic process release, or safety bypass.

## First model: thermal modulus and solidification time

The model uses the familiar empirical relation

    t = B × (V / A)^n

where t is solidification time in seconds, V is casting volume in cubic metres, A is the declared effective cooling surface area in square metres, and B and n are calibrated parameters. With V/A in metres, B has units of seconds per metre raised to n.

The form is based on Chvorinov-style solidification scaling; it is deliberately not treated as a universal constant law. The selected cooling-area definition and observations must be consistent. A process population must not mix alloy families, mold materials, tooling state, mold temperature regimes, or measurement definitions without an explicit hierarchical model that accounts for those differences.

The Rust implementation:

- validates finite, positive observations and physical geometry inputs;
- requires at least three observations for the numerical fit (not as a claim of statistical sufficiency);
- fits the exponent and constant by ordinary least squares in log space;
- reports the sample count and in-sample log-space R-squared;
- records the observed thermal-modulus range and rejects predictions outside it;
- does not infer defect absence, alloy properties, strength, or casting acceptance;
- includes synthetic arithmetic regression tests, explicitly not physical validation.

R-squared is a diagnostic, not uncertainty quantification. The next calibration phase needs repeated physical observations, measurement uncertainty, held-out validation, and prediction intervals before the model can support higher-stakes decisions.

## Process architecture: close the loop with evidence

    product requirements + CAD geometry + alloy identity
        -> manufacturability / thermal-modulus screening
        -> baseline filling-and-solidification simulation
        -> candidate mold, gate, riser, vent, and process designs
        -> safety-reviewed, operator-approved trial plan
        -> controlled test-coupon casting campaign
        -> sensor records + metrology + defect classification
        -> model residuals, uncertainty, and discrepancy ledger
        -> next experiment chosen for information gain and practical value
        -> updated calibration and held-out evaluation

Run a multi-fidelity workflow. Use fast analytical checks to reject obviously poor candidates, an independently validated flow/thermal solver for geometry-dependent decisions, and physical trials for calibration and acceptance evidence. Do not use an inexpensive model to claim it has reproduced full CFD, feeding behavior, microstructure, or defect physics.

The important paradigm shift is from finding one nominal “best” setting to finding a robust, measurable process window: settings should remain acceptable across known variation in material lot, mold condition, geometry, ambient conditions, and machine behavior. Optimize yield, quality, energy per accepted unit, cycle time, material use, and maintainability together. Safety and required quality gates are constraints, not costs that can be traded away.

## Reference machine: instrumented non-ferrous casting cell

This is a subsystem architecture for qualified foundry engineers to detail—not a build-ready hot-metal machine design.

### Material and mold path

- Traceable charge preparation with alloy/lot identity, incoming condition, measured mass, and documented contamination controls.
- Furnace and crucible selected and qualified for the chosen alloy, batch size, atmosphere, and operating envelope by competent personnel.
- Repeatable mold/pattern fixtures with recorded identity, mold recipe/revision, mold condition, and documented effective cooling-area convention.
- Defined locations for calibrated melt and mold temperature measurements; use sensors and sheaths suitable for the actual environment.
- Instrumented charge/pour mass and pouring-motion observations where the chosen cell can measure them with justified uncertainty.
- A post-cast path for dimensional metrology, appropriate non-destructive examination, and destructive coupon testing when required.

### Machine, instrumentation, and safety boundary

- A rigid, guarded fixture with documented alignment, service access, and changeover references.
- A measured pouring-motion interface (for example, angle/position and mass change over time) before attempting automatic pour optimization.
- Independent safety-rated or otherwise appropriately engineered interlocks, emergency stop, guarding, exclusion zone, dry feed/mold controls, spill containment, ventilation/fume extraction, and lockout/tagout. Actual requirements depend on the process and site and must be reviewed by qualified professionals.
- Calibrated instrumentation with calibration identity, measurement range, timestamps, sample rates, missing-data markers, and uncertainty metadata.
- A safety/control PLC and machine interlocks that remain authoritative. Symthaea starts read-only; it may flag anomalies and propose experiments, but it cannot directly command pouring, furnace output, or interlocks.
- Local data capture with append-only run manifests; loss of telemetry must not bypass machine safety or silently turn a run into qualified evidence.

Do not freeze actuator sizing, crucible dimensions, furnace power, pour temperature, or cycle settings into a generic repository document. Those require a specific alloy, batch, geometry, equipment datasheet, thermal analysis, guarding design, and site-specific safety review.

## Casting run record

Each trial should link the exact subjects needed to interpret the result:

- design and CAD revision; geometry digest; nominal and as-built dimensions;
- alloy family, declared composition/specification, charge lot, mass, and incoming inspection;
- mold/pattern/tooling identity and revision; material and conditioning history;
- machine/furnace/crucible identity and relevant maintenance/calibration state;
- process-plan/recipe revision and operator-approved run identifier;
- raw sensor streams, units, calibration references, timestamps, missing samples, and uncertainty;
- model source/version, calibration dataset digest, fitted parameters, domain, and prediction;
- post-cast observations, inspection method, defect taxonomy, measured dimensions, and uncertainty;
- final disposition: accepted for the stated test, rejected, rework, or unresolved.

A commanded process or a completed machine cycle is not an observation of the resulting material state. A clean visual check does not prove absence of internal porosity. A simulation result cannot populate the as-built measurement fields.

## Experiment design

1. Select one non-critical test-coupon geometry and one declared alloy/mold/process population. Freeze inclusion/exclusion rules before collecting data.
2. Establish repeatable measurement procedures for geometry, solidification-time proxy, relevant temperatures, and post-cast results. Have the metrology and foundry procedure reviewed by competent staff.
3. Run a small feasibility campaign to verify instrumentation, time synchronization, repeatability, and data completeness. Do not train and test on the same casting while claiming generalization.
4. Use a predeclared calibration/validation split. Where the dataset permits, hold out complete geometries or tooling states, not only randomly selected rows from a nearly identical geometry.
5. Compare the calibrated model against a documented fixed-exponent baseline, a simple mean baseline, and—when available—an independently validated numerical solver.
6. Record model residuals and disagreements. Large disagreement or out-of-domain conditions should trigger “insufficient evidence / require engineering review,” not a confident automated recipe change.
7. Use sequential design of experiments only after baseline repeatability is established. Choose the next trial by expected information gain and operational value, subject to an independently approved safe experimental envelope.
8. Inspect and classify defects using a documented method. Use disciplined root-cause analysis rather than assuming a predicted thermal field identifies the cause of every defect.

## Benchmark gates

All numeric values below are proposed research targets, not measured results or universal foundry-industry thresholds.

### Gate A — software and scientific correctness

- Exact synthetic-data recovery of known positive model parameters within a frozen numerical tolerance.
- Tests for units, invalid/non-finite values, degenerate data, negative fitted exponent, and out-of-domain prediction rejection.
- Independent calculation or test implementation reproduces the published reference fixtures.
- No missing or invalid input can be upgraded to a calibrated estimate.
- Synthetic tests are explicitly labelled as arithmetic-only; they cannot satisfy physical validation.

### Gate B — calibration and generalization

Report MAE and median absolute relative error for solidification-time estimates, separately for in-domain and held-out geometry/tooling groups. Publish sample counts, calibration range, measurement uncertainty, excluded observations, and all split rules.

Proposed first target: median absolute relative error no greater than 15% on a frozen, in-domain held-out coupon set, provided the chosen time measurement has materially smaller uncertainty. If that cannot be met, diagnose model mismatch instead of widening tolerances after observing the data.

Next, add prediction intervals and measure empirical coverage on a held-out set. A nominal 90% interval should be judged by observed coverage and width; a point estimate plus R-squared is not a substitute.

### Gate C — physical process improvement

Establish a measured baseline using the same part requirements, alloy/lot policy, inspection rules, and accounting boundary. Then test a predeclared candidate process window with sufficient independent repetitions.

Proposed directional targets:

- reduce scrap/rework mass per accepted mass by at least 20% relative to the frozen baseline;
- reduce metered energy per accepted unit by at least 10%, without worsening required material properties or defect acceptance;
- reduce avoidable cycle-time variance and explain any change in the median cycle time;
- preserve 100% traceability for all critical trial inputs, model versions, observations, and dispositions;
- demonstrate that a deliberately stale calibration, missing critical sensor, incompatible material identity, or out-of-domain geometry blocks a qualified recommendation.

Targets only become claims after measured results, uncertainty, sample counts, and replication are published. A synthetic optimizer win is not a foundry productivity result.

### Gate D — bounded machine integration

Only after Gates A–C and a separate process/equipment qualification review should an adapter be considered. Initial deployment remains advisory and shadow-mode. Any later closed-loop behavior must have a separately qualified boundary, independent machine safety controls, explicit operator approval, bounded authority, safe fallback, fault injection, and a documented way to stop/recover. Model confidence must never override an interlock.

## Comparison with the state of practice

Foundry simulation and process-control tools are already mature; we should not claim Symthaea is “better than SOTA” merely because it combines physics and AI. The measurable differentiation to pursue is an auditable loop that connects model assumptions, calibration scope, geometry, machine observations, inspection results, uncertainty, and process economics—and rejects unsupported extrapolation.

Start with baselines that are easy to reproduce: fixed-parameter Chvorinov estimate, a simple statistical predictor, and available validated casting simulation. In later phases, compare calibrated hybrid models on held-out geometries and whole production lots, not just random train/test rows. Report error, calibration, runtime, memory, experimental cost, defect rates, and energy per accepted unit.

## Delivery sequence

- **CASTING-001A — first slice (this change):** calibrated Chvorinov-style solidification model, explicit SI semantics, diagnostics, synthetic regression tests, and reference cell/process design.
- **CASTING-001B — run-record contract:** bind process-model and sensor observations to the existing manufacturing/process/evidence contracts; do not duplicate MFG-PROC identity.
- **CASTING-001C — metrology and inspection:** define repeatability, uncertainty, defect taxonomy, test coupon fixtures, and independent acceptance evidence.
- **CASTING-001D — filling/thermal model adapter:** evaluate existing validated solvers before creating a new solver; compare their predictions with measured trials.
- **CASTING-001E — experiment selection:** bounded multi-objective design of experiments with frozen baselines and explicit out-of-domain states.
- **CASTING-001F — equipment adapter:** only after independent physical/process qualification, behind existing fabrication/HAL/operator authority.

## Research references

- [Essam et al. (2026), computational modeling, AI, and digital twins in metal casting](https://link.springer.com/article/10.1007/s43939-026-00685-5) — discusses physics/data/model integration and the limitations of industrial-scale validation and generalization.
- [American Foundry Society: Casting Defect Analysis](https://www.afsinc.org/courses/casting-defect-analysis) — emphasizes systematic root-cause evaluation and collecting process data instead of assuming a classifier alone determines corrective action.
- [American Foundry Society: Aluminum Casting Performance Initiative](https://www.afsinc.org/news/2025/05/15/afs-begins-work-aluminum-casting-performance-initiative-acpi-projects) — examples of current industry work on defect recognition and process-related quality.
- [Niiyama, A Modification of Chvorinov's Rule on Solidification Time of Sand Castings](https://www.jstage.jst.go.jp/article/imono/44/12/44_1047/_article) — source for the classic volume/surface-area solidification relation and its limitations.
- [NIST: Real-Time Monitoring and Control of Additive Manufacturing Processes](https://www.nist.gov/programs-projects/real-time-monitoring-and-control-additive-manufacturing-processes) — useful methodological precedent for calibrated sensing, validated physics-based models, and carefully qualified feedback-control research. It is not direct evidence that the same controller works for casting.

## Nonclaims

This change does not establish physical casting accuracy, defect-detection performance, manufacturing capability, production economics, part integrity, machine safety, legal compliance, or autonomous foundry readiness. The synthetic fixtures are software checks only. Physical trials and equipment integration require qualified engineers, controlled conditions, validated measurement methods, and exact-head evidence.
