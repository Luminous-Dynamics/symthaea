# Camera, sensor fusion and quality-observer benchmark

Status: research design, 3 November 2026. This is not a validated camera model or cleaning controller.

## Competitive boundary

Miele announced the G8 Diamond / AutoVision on 2 September 2026. The official release says its integrated camera uses image processing and AI-based recognition to detect load level and item types; it increases cleaning intensity for pots, optimizes internal bottle cleaning and selects drying for plastics. It is integrated with AutoDos and variable hot-air ExcellentDry, and is due in Germany, Austria and Switzerland in April 2027; other countries follow.

This makes camera-based load classification, item-type recognition and adaptive wash/dry selection a known announced incumbent. The release does not give comparable numeric cycle energy/water or independent cleaning results. Do not infer them.

Official source: https://www.miele.de/en/m/world-premiere-for-the-g8-diamond-8453.htm

Other relevant incumbents:
- GE CleanSensor / auto cycles: optical turbidity-based adaptation.
- ENERGY STAR consumer guidance: wash-water sensing used to adjust operation.
- V-ZUG AdoraDish V6000: load and soil sensing, partial-load programmes, heat-pump thermal architecture and OptiStart cold pre-wetting to soften soil before a lower-temperature wash.
- Fisher & Paykel DishDrawer: independent half-load compartments and Quick/Sanitise/Dry+ options.

## Candidate observers

A. Manual/programme-table baseline.

B. Turbidity-only / existing auto-sensing baseline.

C. Camera-only load observer: occupancy, item type, geometry, bottle opening/internal-cleaning need, pot/baked-on cookware flag, plastic/recessed drying class. Camera does not estimate residual surface soil reliably and cannot claim a surface is clean.

D. Multi-sensor process observer: turbidity trend, conductivity, pressure, flow, pump current, water temperature and real electrical power.

E. Fused observer: camera-derived geometry/item type + process signals + load/arrival context + calibrated uncertainty.

F. Sensor-ablated observers: omit one channel at a time; test whether each sensor adds measurable value beyond simpler alternatives.

## Computer-vision benchmarks

Create an exact-SKU / load image set with independent labels for:
- item type and material class;
- count/occupancy and occupied rack volume;
- dish orientation, nesting and spray-shadow risk;
- bottle neck/opening visibility and internal lance eligibility;
- plastic/recessed-item drying category;
- occlusion, glare/reflections, wet lens, steam/condensation and low-light conditions.

Split by entire physical loading session, camera unit and day; never split near-identical frames into train/test. Report class precision/recall/F1, occupancy absolute error, calibration error / Brier score, and abstention rate. Include out-of-distribution tests and uncertain/occluded loads. CV metrics validate recognition only, not cleaning.

## Process and outcome benchmarks

Ground truth:
- witness coupons and gravimetric residual soil;
- soil-specific assay or standardized image/color evaluation;
- rinse carry-over / detergent-residue assay;
- drying residual moisture per material;
- hydraulic delivery, temperature trajectory and electrical energy.

Compare observers under identical cleaning/rinse/dry endpoints. Report energy/water per successful item, time-to-release, rewash rate, false early-stop rate, uncertainty calibration, sensor abstention and fallback rate. Use held-out soil batches and days.

## Safety and privacy invariants

- No early-stop or clean-success claim from camera classification or turbidity alone.
- A sensor/camera fault, occlusion, fouling, calibration drift, contradictory measurements or out-of-scope item must trigger abstention and a conservative cycle.
- The deterministic safety supervisor stays independent of the learned observer.
- Provide a usable no-camera path.
- Prefer local inference; define whether images are processed transiently, retained locally, or transmitted. Default to no raw-image retention unless explicitly needed and consented for research. Minimize personal data and avoid remote image upload as a requirement for appliance operation.
- Record model version, sensor health, calibration, scope, uncertainty and fallback reason with each control decision.

## Aspirational acceptance gates

- At least 10% reduction in energy or water per verified successful clean versus the best matched incumbent, without worse cleaning/rinse/drying outcomes.
- No statistically significant false-success increase relative to the matched baseline; establish a pre-registered upper confidence bound from independent held-out cycles.
- Camera/item classification calibration and coverage reported per class, not a single aggregate accuracy.
- Fused observer must materially outperform the best individual input or prove equal performance at lower sensor/hardware cost.
- Any camera-only gain must be compared with the added hardware cost, privacy burden and service failure modes.

## Important limitations

The Miele G8 is announced product information; it has no public matched performance results in the cited release and does not reach DACH markets until April 2027. Exact SKU, product-label values and independent tests must be obtained before quantitative comparison. This document defines a benchmark, not a claim that InstantClean already exceeds the G8.
