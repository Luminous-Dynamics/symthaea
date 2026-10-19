# Thermal observation log fixture v1

All values and evidence records in this fixture are synthetic; they are not cooking instructions, food-safety evidence, or instrument performance measurements.

The log binds three estimated process-medium temperature intervals to the `boil` step from the synthetic rice/pea process trace. The sensor is deliberately marked `unverified`, and the records are not qualified.

## Why this is a paradigm shift

Symthaea Culinary already contains a time/temperature trajectory model. The missing connective tissue is a traceable observation log that records which process step was measured, what type of temperature was being observed, where the probe was placed, which sensor produced the observation, what uncertainty was reported, and what calibration status is claimed.

The USDA FSIS guidance explicitly distinguishes doneness from safety and notes that placement, food geometry, and temperature variation within a food affect the accuracy and representativeness of a reading: https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/food-thermometers

## Contract

- Every log binds to the trace ID and process ID.
- Every observation references an existing process step and declared sensor.
- Observation IDs and per-sensor/per-step sequence indices are unique.
- Time must strictly increase with sequence index.
- Temperature is Celsius with explicit lower/point/upper values or a reasoned unknown.
- Sensor type, stated accuracy, calibration status, last calibration time, and calibration evidence are explicit.
- Synthetic calibration evidence cannot substantiate a sensor marked calibrated.
- Passing validation does not calculate lethality, determine safe cooking, certify doneness, or validate a thermal model.

## Qualification boundary

Before using logs to evaluate a physical model, compare instruments against traceable references, preregister probe placement and sampling cadence, capture measurement uncertainty, and preserve raw observations. Do not convert a structurally valid log into a food-safety verdict. Safety policies require separate validated limits and authoritative assessment.
