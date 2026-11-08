<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Micro-LED AR microdisplay yield-aware co-design corpus v0.1

**Owner:** [MICRODISP-YIELD-001 #7868](https://github.com/Luminous-Dynamics/symthaea/issues/7868)  
**Corpus ID:** `MICRODISP-YIELD-CORPUS-001`  
**Status:** `UNQUALIFIED` — synthetic workload definitions, not wafer/panel measurements.

## Research basis

The July 2026 *Laser & Photonics Reviews* review of micro-LED/CMOS integration identifies coupled electrical, thermal, optical, placement, interconnect, process-integration and yield limits at sub-10 μm and smaller pixel pitch. It describes the bottleneck shifting as architectures scale, so optimizing only pixel pitch can move the dominant failure elsewhere. [Park et al., 2026](https://doi.org/10.1002/lpor.71634).

A separate 2026 review treats extremely high transfer yield as a central commercialization bottleneck and examines multiple assembly approaches plus defect-control, redundancy and repair strategies. Its 99.99999% figure is a research frontier, not an achieved Luminous result or an automatic gate. [Hong et al., 2026](https://doi.org/10.1088/2631-7990/ae25a8).

## Files

- `microdisplay-yield-corpus-v0.1.json`: 20 deterministic proposed cases, all `UNQUALIFIED`.
- `microdisplay-yield-corpus.schema.json`: JSON Schema Draft 2020-12 structure.
- This protocol fixes the evidence and optimization boundaries.

## Benchmark planes

1. **Yield arithmetic:** denominator validation and expectations with explicit independence assumptions.
2. **Defect topology:** independent versus clustered/row/column failures.
3. **Inspection:** class-wise precision/recall, false-positive burden, coverage and local-vs-global claim scope.
4. **Placement/interconnect:** declared tolerance propagation and stage-specific failure accounting.
5. **Repair/redundancy:** first-pass yield, repair-adjusted yield, post-repair defects, area/power overhead and retest.
6. **Electro-thermal-optical:** uncertainty propagation rather than false precision from unknown material/device parameters.
7. **Visual utility:** usable FOV, MTF, luminance, color and eye-box are hard feasibility constraints; higher yield cannot compensate for violating them.
8. **Reliability/metrology:** distinguish initial, thermal-soak, aged and post-repair state; bind device/config/calibration identity.
9. **Economics/optimization:** cost per qualified-good display, calibration burden and Pareto-front membership without an opaque single score.
10. **Evidence/provenance:** raw artifacts, denominators, calibration and source rights are explicit gates.

## Example yield arithmetic

For (N=100{,}000{,}000) emitters with independent per-emitter success probability (p), expected failures are (N(1-p)). Thus:
- (p=0.9999) implies 10,000 expected failures.
- (p=0.9999999) implies 10 expected failures.

Those are expectations under the stated simple model, not guaranteed counts, assembled-panel yield or a model of correlated defects. Clustering and stage dependencies require separate evidence or scenario bounds.

## Acceptance targets (proposals, not results)

- 100% agreement with independent synthetic arithmetic references.
- Zero false accepts for denominator mismatch, incomplete inspection coverage, duplicate repair accounting, stale calibration or missing critical inputs.
- Keep all defect-stage denominators and correlation assumptions visible.
- A claimed optimization must improve a preregistered primary metric such as repair-adjusted yield or cost per qualified-good module **while meeting every optical and reliability floor**.
- Long-term stretch study: ≥20% lower cost per qualified-good display or ≥20% better repair-adjusted yield against a named baseline, without material regression in usable FOV, MTF, luminance, power or reliability. This is a study target; revise if baseline data demonstrate it is inappropriate.
- Physical yield requires actual process counts, calibrated inspection/metrology, uncertainty, raw artifacts and independent review.

## Execution sequence

1. Validate this schema and corpus independently, including mutation negatives.
2. Implement deterministic arithmetic/semantic tests in the existing Rust-first engineering infrastructure.
3. Reuse SEMI-EQP-MET-001A (#5914) for inspection coverage/calibration; PHOT-ENG-000 (#5668) for optical model ownership; ENG-DEVICE-000 (#5670) for device subsystem identity.
4. Add bounded electro-thermal-optical models only after input closure and source-supported ranges exist.
5. Plan physical coupon/sample experiments only with qualified fabrication/metrology partners and a separately reviewed safety/authority contract.

## Limits

No micro-LED device, fab process or transfer method is selected or qualified. No production yield, repair efficiency, optical quality, device lifetime or economics result is claimed. Model predictions, synthetic tests and physical evidence remain separate.
