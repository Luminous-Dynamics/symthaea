# Synthetic paired culinary experiment fixture v1

Every value, process-trace reference, randomization seed and evidence locator in this fixture is synthetic. The fixture is a software-contract test, not a cooking experiment and not evidence of improved yield, reduced waste, energy savings or sensory acceptance.

## Study shape

- Two preregistered conditions: `baseline` and `variant`.
- Two paired blocks; each block has exactly one run per condition.
- Randomized order differs across blocks and is checked against each run's declared execution order.
- Three independent outcome dimensions: prepared edible yield (g), discarded material (g), and energy input (Wh).
- Every run has an explicit process-trace ID, time window and one record per declared outcome.
- Outcome records carry units, point/lower/upper values, method and source/version/time/evidence status.

All outcomes are marked `estimated` and all source records `synthetic` intentionally. The validator should accept the structural design while refusing to mark the experiment ready for empirical paired analysis.

## What this fixture tests

It checks paired-block identity, assignment-order consistency, protocol preregistration before data collection, outcome completeness, unit agreement, finite ordered intervals, unique run/block/outcome identifiers, and the boundary between structural completeness and evidence-qualified observations.

A valid dataset is not automatically causal evidence. Source locators and `qualified` labels are declared metadata; they do not authenticate a source or validate a study design independently.