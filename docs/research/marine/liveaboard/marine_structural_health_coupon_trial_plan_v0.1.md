# Marine structural-health coupon validation trial — V0.1

**Status:** proposed laboratory experiment; not performed.  
**Parent:** [MARINE-STRUCTURAL-HEALTH-001 #11089](https://github.com/Luminous-Dynamics/symthaea/issues/11089) / draft [PR #11094](https://github.com/Luminous-Dynamics/symthaea/pull/11094)

## 1. Question

On a declared metallic welded-joint coupon and controlled laboratory load case, can synchronized strain and acoustic-emission (AE) observations identify and localize fatigue-related events more reliably than a simple single-sensor baseline, while keeping false alarms, missed events, uncertainty and sensor spacing visible?

This tests a sensing hypothesis in a repeatable specimen. It does not validate a vessel structure, predict fleet fatigue life, establish a crack-growth law, set an inspection interval or qualify an NDT procedure.

## 2. Article, method and safety boundary

Use a material/joint coupon whose alloy/grade, product form, thickness, weld process, joint geometry, heat treatment, surface condition and fabrication record are fully documented. The responsible materials / welding / fatigue laboratory selects the specimen geometry, loading spectrum, fixtures, stress range, frequency, stop criteria and safety enclosure under its approved procedure.

The programme must use:
- a competent laboratory with a rated fatigue test frame, guarding/containment and emergency stop;
- a qualified test plan and independent materials / structural review;
- calibrated load, displacement/strain, AE acquisition and environmental instrumentation;
- pre-test dimensional, visual and appropriate NDT characterization;
- a clearly documented reference-state specimen group and a separately identified flaw/damage progression group, if approved by the lab;
- independent post-test NDT and, where suitable, destructive sectioning / metallography to establish ground truth.

Do not infer or invent stress limits, fatigue cycles, crack sizes, loading frequencies, weld acceptance criteria or sample counts in this plan. They must be supplied by the responsible laboratory and applicable material/joint qualification basis. Do not use a vessel, live pressure system, fuel tank, battery compartment, seawater intake or operational hull as the first test article.

## 3. Experimental arms

Compare on the same specimen family and controlled exposure:
1. **Reference / ground-truth workflow:** pre/post independent inspection, calibrated load history and documented specimen identity.
2. **Strain-only:** qualified strain measurement at declared positions.
3. **AE-only:** synchronized AE sensors with declared coupling, spacing, bandwidth, threshold and noise characterization.
4. **Strain + AE:** fused timing and context, with each channel's independent evidence retained.
5. **Context-aware fusion:** add temperature, fixture/environmental noise and load context; no model may treat context as direct crack evidence.

Vary sensor distance/spacing and include a declared region outside effective coverage. The exact sensor locations must be frozen from the specimen geometry and method plan before data collection. Do not tune placement on the blind evaluation specimens and then report the same specimens as held-out validation.

## 4. Ground truth and blind evaluation

- Give every coupon a unique ID, material/weld batch, process revision and geometry/datum record.
- A separate inspection team, blinded to algorithm outputs, performs the pre/post reference inspection using qualified methods for the specimen/material/defect type.
- Keep raw load, strain, AE and environment data time-synchronized; record clock uncertainty, sampling, calibration, coupling checks, trigger thresholds, missing data and saturation.
- Label each event only after independent review; distinguish fatigue-related source, fixture friction, impacts, electrical noise and unknown source.
- Preserve event-level and specimen-level splits. Do not randomly split adjacent AE windows from the same coupon across train/test.
- Use a held-out coupon batch or geometry for confirmation. Keep the reference ground truth sealed until model/policy outputs are frozen.
- Where justified by the lab, post-test sectioning/metallography can provide a stronger independent crack-location/size reference than sensor estimates alone.

## 5. Negative controls

1. Healthy reference coupon under the same declared load path: sensor noise must not become a damage event.
2. Fixture friction / grip slip / cable motion: AE event must not automatically become a crack.
3. Temperature change or coupling degradation: apparent sensor drift must be detected or cause abstention.
4. Sensor detachment / stale calibration / channel saturation: no positive health disposition.
5. Deliberately changed sensor spacing or a blind region: localization confidence/coverage must degrade visibly.
6. Wrong specimen ID, weld revision or coordinate datum: evidence rejected.
7. A damage event outside the calibrated load/environment/material envelope: unsupported or indeterminate.
8. Independent NDT disagreement: no algorithmic override of the qualified inspector's finding.
9. Missing high-load segment / time-sync gap: no “no indication” disposition over the gap.
10. Post-test result differs from model: preserve the miss and recalibrate only on a future, separate dataset.

## 6. Metrics

Report by specimen, load block, damage state, sensor configuration and event source:
- detection probability and false-alarm rate at preregistered thresholds;
- false negative rate by reference damage severity;
- event classification precision/recall, but keep unknown/noise/fixture events as a class;
- source localization error with sensor geometry and uncertainty;
- time from independent event onset/reference to detection;
- probability/interval calibration and OOD/abstention rate;
- sensitivity to sensor distance, coupling and environment;
- usable-data fraction, sensor uptime, calibration/maintenance burden;
- NDT confirmation and disagreement rate;
- repeatability across coupon batch / joint geometry;
- test cost and elapsed laboratory time.

Do not report a single F1 score without class balance, event counts, specimen-level split, false-negative severity and calibration. Do not transfer coupon metrics directly to hull-scale structures.

## 7. Predeclared interpretation

A sensing approach advances only if it improves a preregistered decision-relevant metric on held-out specimens without increasing critical false negatives or hiding out-of-domain cases. Higher raw AE hit count, higher model confidence or better training-set F1 is not a win.

No fatigue crack-growth / RUL claim is made unless a separate, material/joint-specific physics model is independently verified and validated against additional specimens. Any sensor-driven inspection recommendation remains advisory and cannot override required inspections or competent authority decisions.

## 8. Required output package

- Specimen/material/weld/process manifest and exact test revision.
- Laboratory-approved test plan and risk review.
- Sensor installation / calibration / coupling records.
- Frozen load/environment and synchronized raw-data manifests.
- Pre/post independent NDT reports and ground-truth adjudication.
- Blind held-out predictions with uncertainty and per-specimen event ledger.
- Reproducible analysis code/toolchain manifest and exact output digest.
- Deviations, excluded runs and failures, with reasons.
- Permission, rights and retention decision for raw waveforms, imagery and specimen records.

## 9. Claim ceiling

A successful coupon trial demonstrates only the tested sensor/method on the exact coupon family, load, environment, geometry, spacing and measurement chain. It does not establish vessel-scale detection performance, structural adequacy, crack-growth prediction, inspection interval, seaworthiness, class/flag approval or permission to defer inspection.
