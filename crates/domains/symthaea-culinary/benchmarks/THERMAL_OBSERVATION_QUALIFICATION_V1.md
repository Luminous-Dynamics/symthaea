# Thermal observation log qualification benchmark v1

**Status:** structural contract and synthetic fixture only. Not a validated thermal or food-safety model.

## Research basis

- USDA FSIS distinguishes doneness (a sensory state) from safety and explains why internal temperature, probe placement, and spatial temperature variation matter: https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/food-thermometers
- USDA FSIS explains minimum internal temperatures and rest times for specific products; those thresholds are outside this generic observation schema and must not be inferred by it: https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/safe-temperature-chart
- FAO recipe calculations require cooking adjustments and prepared-food mass to be represented separately: https://www.fao.org/4/y4705e/y4705e23.htm

## Acceptance targets

1. 100% of observations resolve to exactly one existing process step and one declared sensor.
2. Zero duplicate observation IDs or duplicate sequence indices within one sensor/step.
3. 100% of sensor logs are monotonic in time with respect to declared sequence order.
4. 100% of temperature readings carry explicit unit, location, type, value interval and source/version/time/evidence status, or explicit unknown reason.
5. Zero synthetic calibration records accepted as evidence for a sensor marked calibrated.
6. No safe/unsafe, pasteurized/not-pasteurized, or doneness pass/fail output from this validator.
7. Stream/order permutation must not change the summary when sequence indices remain identical.
8. The independent oracle corpus must include timestamp reversal, stale/missing sensor identity, incorrect placement metadata, duplicate sequences, wide uncertainties, unknown readings and synthetic calibration spoofing.
9. Rust formatting, compilation, tests and Clippy must run against the exact head before qualification. Skipped/queued/absent workflows are not pass evidence.

## Next empirical benchmark

Record repeated batches using sensors checked against a traceable reference. Keep a minimum of product geometry, material state, sensor type, location, accuracy/calibration record, sampling timestamps, and raw readings. Report:
- temperature error and bias against reference, with median/p95 and worst case;
- repeatability across batches;
- missing-sample rate and sensor dropouts;
- spatial disagreement between core/surface/medium/ambient observations;
- thermal-model residuals and interval coverage on held-out batches;
- results stratified by cooking method and food geometry.

Keep sensor calibration qualification, thermal-model validation, food safety, and sensory doneness as distinct gates. A thermometer trace can support a model; it does not validate the model or safety claim by itself.
