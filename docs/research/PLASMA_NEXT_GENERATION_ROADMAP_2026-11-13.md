# Luminous Dynamics Plasma Technology — Next-Generation Research Roadmap

**Date:** 13 November 2026  
**Portfolio owners:** MICROPLASMA-001 #11780; PLASMA-VV-001 #7588; PLASMA-SURFACE-001 #11795; MICROPROP-001 Symtropy #4912  
**Purpose:** cross-programme strategy and benchmark alignment  
**Status:** evidence-bounded research roadmap, not a product specification or physical-device authorization

## Executive thesis

The field does not need another generic claim that a plasma source can be made smaller, sensorized, modular or connected to Industry 4.0. Industrial suppliers already offer inline systems, rotating and multi-head architectures, optical plasma monitoring, power/voltage measurement, gas-flow/pressure control, fieldbus connections, recipe logging and maintenance monitoring.

Luminous should pursue a harder and more valuable objective:

> Build plasma process systems whose measured internal state is linked to independently measured useful product outcomes, whose performance transfers across material lots and production contexts, and whose complete lifecycle economics outperform the best matched incumbent.

This is a research hypothesis, not a demonstrated competitive advantage. It requires exact application identity, calibrated observations, downstream outcome tests, held-out validation and complete resource accounting.

## 1. What is already state of the art

### Industrial process equipment

- **Plasmatreat Openair-Plasma PCU:** the manufacturer describes continuous optical monitoring of the emitted plasma stream, power/voltage measurement, process-gas flow, back-pressure, jet rotation, recipe storage and EtherCAT/CANopen integration. The PCU can control and monitor multiple jets separately. Plasmatreat also reports availability above 99%; this remains a vendor claim, not an independent matched reliability benchmark. Sources: https://www.plasmatreat.com/en/products-and-services/openair-plasmar-process-monitoring and https://www.plasmatreat.com/en/products-and-services/process-control.
- **Surfx SPM:** a product listing describes real-time forward/reflected power, gas-flow and head-temperature monitoring, automatic RF matching and closed-loop coolant control. Source: https://www.dkshsce.com/Products-en/Surfx-SPM.
- **AcXys:** reports application-dependent production speeds above 100 m/min and examples of 750 automotive dashboard parts/day or 2,000 PP tubes/vials/hour. These are manufacturer examples, not matched independent energy/quality trials. Source: https://www.acxys.com/faq/.
- **Tantec SpinTEC:** advertises rotating dual nozzles, 40–150 mm treatment area, robotic/line integration and speeds up to 10 m/min. Source: https://tantec.com/atmospheric-plasma-treater-spintec/.
- **Yamato PSA:** its product brochure reports 900 W per head, 30 standard L/min compressed air per head at 60 psi, 38–152 mm treatment width and multi-head configurations. Source: https://www.yamato-usa.com/web/content/52905.
- **Relyon plasmacell P300:** uses a base workstation with generator, nozzle and XYZ portal, with compressor and extraction as optional modules; extraction remains a required system consideration. Source: https://www.relyon-plasma.com/wp-content/uploads/2021/03/F0354301_BA_PLASMACELL_EN_copyright-2021.pdf.

### Research state

- A 2026 CAP wettability model reports test R²=0.9079 for relative contact-angle reduction on a literature-derived 534-observation/53-paper corpus. The split/augmentation pipeline needs exact reproduction and grouped holdout testing before claiming deployment transfer. DOI: https://doi.org/10.1002/ppap.70246.
- A 2026 CFRP/zinc study reports up to 4.21 ± 0.12 MPa pull-off adhesion and an 84% increase over untreated CFRP in its most intense air-plasma condition. This does not establish aerospace lightning-protection qualification. DOI: https://doi.org/10.1016/j.compositesa.2026.109797.
- A 2026 review of ML-enabled in-situ diagnostics for plasma semiconductor manufacturing describes virtual metrology, anomaly detection, predictive maintenance and optimization, while identifying labeled-data scarcity, interpretability, domain transfer and lab-to-fab transfer as open challenges. This is adjacent evidence, not direct proof for atmospheric surface activation. DOI: https://doi.org/10.1021/acsami.6c04768.
- The 2026 large-scale plasma roadmap identifies scale-up controllability, grid-to-outcome energy, reliability/maintainability and outcome-relevant diagnostics as deployment challenges. DOI: https://doi.org/10.1049/hve2.70241.

**Competitive conclusion:** modular hardware, multi-head arrays, process telemetry and routine process monitoring are not unique differentiators. Luminous must compete at the downstream quality, evidence-transfer and full lifecycle level.

## 2. Five paradigm shifts

### A. From plasma-source performance to accepted useful service

Do not use peak discharge power, optical intensity, plume length, contact angle or single-coupon adhesion as a universal success metric.

Define an application service unit:
- accepted bonded area meeting an adhesion/durability floor;
- accepted conductive coating area meeting mechanical/electrical requirements;
- clean-air volume after ozone/NOx abatement;
- water volume meeting contaminant-specific and by-product requirements;
- conforming nanoparticle mass meeting size distribution/yield constraints;
- mission delta-v or total impulse for spacecraft propulsion.

A candidate can win only if it provides the same service under the same quality, safety and system boundary.

### B. From "make a smaller machine" to physics-preserving numbering-up

Compare one enlarged discharge with arrays of independently characterized cells and microfluidic reactors. Measure the scale-out law, not just cell count:
`G_N = Q_N / (N Q_1)`

Include manifold imbalance, cross-talk, shared supplies, fault isolation, degraded operation, service time and utility scaling. The proposed target G_N ≥ 0.90 is a stretch objective, not a published standard.

### C. From plasma-state monitoring to downstream-outcome qualification

The industrial market already monitors power, voltage, optical emission, flow, pressure, temperature, movement and jet rotation. Luminous should test whether these signals predict the **actual product outcome**:
- bond strength and failure mode;
- coating adhesion and electrical continuity;
- aging and material-property retention;
- clean-air delivery after abatement;
- contaminant mineralization/toxicity;
- conforming synthesis yield.

The differentiator is not more sensors. It is a calibrated and transferable link between telemetry, material identity and useful result, with uncertainty and an explicit HOLD/UNKNOWN disposition outside the validated domain.

### D. From one "best recipe" to a qualified process window

A process optimum is a distribution across material lots, ambient conditions, geometry, storage delay, consumable state and production day. Learn the acceptable region and its failure boundaries, not a single maximum result.

Use held-out lots and dates, replicate distributions, failure-mode analysis and post-process aging. Store every attempted run and every rejected specimen.

### E. From trial-and-error to evidence-aware experimental design

Reuse the existing robust experiment-selection infrastructure (OLI-EXPERIMENT-001 #11446, UAP-OED-001 #11050, AR-5D #9332) to select experiments that distinguish competing physical hypotheses and reduce uncertainty per laboratory cost.

Do not create a second generic EIG/BO optimizer. Plasma work owns the material/process features, observable contracts and physical interpretation. The shared experiment-selection owner remains authoritative.

## 3. Application portfolio and priorities

Priority levels describe research sequencing, not commercial certainty.

### P0 — near-term product hypothesis: durable polymer surface activation

Issue: [PLASMA-SURFACE-001 #11795](https://github.com/Luminous-Dynamics/symthaea/issues/11795); draft implementation [PR #11815](https://github.com/Luminous-Dynamics/symthaea/pull/11815).

Why first:
- non-clinical materials-processing outcome;
- standardized adhesive/coating performance tests can be selected for a specific geometry;
- direct industrial incumbent comparisons exist;
- high-mix / complex-part / selective activation can be evaluated without claiming generic line-speed superiority.

Differentiation hypothesis: durable bondability window, material-aware outcome prediction, lower lifecycle cost per accepted bonded area, lower scrap/rework on a matched high-mix process.

### P0 — intelligence enabler: grouped plasma outcome generalization

Issue: [PLASMA-ML-GENERALIZATION-001 #11839](https://github.com/Luminous-Dynamics/symthaea/issues/11839); draft [PR #11842](https://github.com/Luminous-Dynamics/symthaea/pull/11842).

Why:
- current literature baseline is high on a random split;
- true transfer to unseen papers, devices, labs and materials remains the more relevant deployment question;
- split/preprocessing leakage and source provenance can be checked independently before original data access;
- downstream outcomes must remain separate from contact angle.

### P1 — high-value, longer-cycle: CFRP conductive metallization

Issue: [PLASMA-CFRP-METAL-001 #11838](https://github.com/Luminous-Dynamics/symthaea/issues/11838); draft [PR #11840](https://github.com/Luminous-Dynamics/symthaea/pull/11840).

Why:
- high-value conductive coating application on lightweight composites;
- recent 2026 source provides an experimental anchor for zinc/CFRP plasma pretreatment;
- electrical continuity, coating integrity and laminate preservation matter alongside adhesion.

Risk: aerospace/LSP qualification requires a separate authority-approved component-level programme. A 4.21 MPa coupon result is not an aircraft lightning protection pass.

### P1/P2 — microfluidic plasma synthesis

Parent: MICROPLASMA-001 #11780.

Focus on high-value material synthesis with conforming yield, particle-size distribution, channel-to-channel uniformity, fouling, membrane lifetime and recovered product per full-system energy. Preserve the validated microscale interface and number-up; do not assume throughput scales with channel count.

### P2 — ambient-air DBD clean-air delivery

Issue: [PLASMA-AIR-DBD-001 #11804](https://github.com/Luminous-Dynamics/symthaea/issues/11804); source work in [PR #11808](https://github.com/Luminous-Dynamics/symthaea/pull/11808).

A 2026 GAPS study reports up to 4.5-log E. coli bioaerosol reduction but reports approximately 9 ppm preliminary residual ozone and calculates EEO using discharge power rather than full wall-plug input. Any occupied-air claim requires post-abatement ozone/NOx measurement and realistic safety assessment. Compare complete-system CADR and energy to HEPA/UV alternatives.

### P2 — water treatment and plasma-liquid catalysis

Issue: [PLASMA-WATER-QUAL-001 #11805](https://github.com/Luminous-Dynamics/symthaea/issues/11805); source work in PR #11808.

Keep microbial inactivation, parent-contaminant conversion, mineralization, transformation-product toxicity, catalyst synergy and pre-treatment as separate lanes. Measure TOC/COD and carbon balance, by-products/toxicity, residual oxidants, catalyst leaching/deactivation, real-matrix performance and kWh per conforming cubic metre. Stop when matched non-plasma alternatives are safer or cheaper.

### Separate programme — miniature plasma propulsion

Symtropy MICROPROP-001 #4912 remains separate from process plasma. Score mission delta-v, total impulse, specific impulse, wet/dry mass, duty cycle, power-processing losses, propellant-feed reliability, plume, endurance and flight heritage. Do not reuse surface-treatment service units.

## 4. Competitive benchmark strategy

### Common denominator: full system to accepted outcome

For every use case report:
- grid-side energy including power supply, gas/compressed air, cooling, extraction, pumps, controls, standby, startup and after-treatment;
- all consumables and replacement intervals;
- accepted useful output, yield, scrap, rework and downtime;
- equipment + utilities + automation + enclosure footprint/mass;
- uncertainty, sample count, calibration, failed runs and material provenance;
- a matched best incumbent under the same service and quality boundary.

### Prospective portfolio targets (not standards/achievements)

| Measure | Proposed research target |
|---|---:|
| Full-system lifecycle cost or energy per accepted service | ≥25% improvement over best matched incumbent |
| Spatial output variation | CV ≤5%, only if measurement uncertainty supports it |
| Number-up efficiency | G_N ≥0.90 at each tested scale |
| Validation transfer | independent material lot, device and production day |
| Model-guided experiments | ≥25% fewer experiments to reach same frozen decision quality |
| Data/evidence completeness | every attempt has a terminal disposition; unknown never becomes zero |

No target is a universal number. The acceptable limit must be chosen from the application service and uncertainty budget.

## 5. What would constitute "better than SOTA"

A claim should pass all four gates:

1. **Matched incumbent:** same substrate/coating/adhesive or process matrix, geometry, service endpoint, duty cycle and complete utility boundary.
2. **Non-inferior product function:** quality, adhesion/conductivity/clean-air/water requirement, durability and substrate preservation are all preserved.
3. **Independent transfer:** held-out lot/device/day and, where relevant, held-out source paper/lab or production line.
4. **Economically meaningful gain:** measured lifecycle energy/cost, scrap/rework, maintenance or uptime improvement beyond measurement uncertainty.

An improvement in a plasma diagnostic metric alone is not a market advantage.

## 6. Evidence ladder and claim ceiling

- S0: synthetic contract and negative-control replay; evaluator behavior only.
- S1: source reconstruction; source-specific values and unknowns.
- S2: independent matched laboratory campaign; raw data, calibration, preregistration and full failed-run accounting.
- S3: held-out process qualification on new lots/days/devices.
- S4: production-scale duty cycle, yield, maintenance, lifecycle economics and EHS.
- S5: sector-specific qualification/certification where required (e.g. aerospace component-level LSP programme).

Synthetic replay cannot become physical evidence. Solver convergence cannot become experiment validation. Pull-off adhesion cannot become LSP compliance. Discharge power cannot become wall-plug process efficiency.

## 7. Immediate execution priorities

1. Finish and review the polymer bondability evaluator PR #11815; exact-head Rust validation remains pending because this authoring environment has no cargo/rustc/rustfmt.
2. Continue ML grouped-holdout integrity work in PR #11842; obtain the original data and supplement only through lawful author access; do not allege leakage without verifying the pipeline.
3. Advance CFRP source-method reconstruction and materials-preservation benchmark in PR #11840; do not presume the source's pull-off standard.
4. Review the array evaluator PR #4935 findings: missing cross-talk coverage, expected-metric checks and blocked energy-score suppression.
5. Continue air/water qualification in PR #11808 with the ozone/by-product and full-system energy boundaries intact.
6. Keep propulsion and fusion in separate physical and economic model families.

## 8. References / industry anchors

- Plasmatreat process monitoring: https://www.plasmatreat.com/en/products-and-services/process-control
- Surfx SPM listing: https://www.dkshsce.com/Products-en/Surfx-SPM
- AcXys FAQ: https://www.acxys.com/faq/
- Tantec SpinTEC: https://tantec.com/atmospheric-plasma-treater/
- Yamato PSA brochure: https://www.yamato-usa.com/web/content/52905
- Relyon plasmacell P300 manual: https://www.relyon-plasma.com/wp-content/uploads/2021/03/F0354301_BA_PLASMACELL_EN_copyright-2021.pdf
- 2026 CAP wettability ML: https://doi.org/10.1002/ppap.70246
- 2026 CFRP/Zn metallization: https://doi.org/10.1016/j.compositesa.2026.109797
- 2026 in-situ plasma diagnostic ML review: https://doi.org/10.1021/acsami.6c04768
- 2026 plasma scale-up roadmap: https://doi.org/10.1049/hve2.70241
