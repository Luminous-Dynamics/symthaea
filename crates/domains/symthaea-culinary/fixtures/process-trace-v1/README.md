# Culinary process-trace fixture v1

All values in this fixture are synthetic contract examples, not measured cooking results, nutrition data, safety evidence, or operating recommendations.

## What this represents

One single-batch preparation and cooking sequence with an explicit input/output graph:

- Inputs: 100 g raw staple, 200 g added water, and 8 g added oil.
- Preparation: 98 g prepared intermediate and 2 g trimming.
- Cooking: 245 g edible product, 53 g evaporated water, and 8 g utensil residue.

The synthetic point values sum to 308 g on each whole-batch boundary. That arithmetic is illustrative only: the intervals are deliberately uncertain, and this module does not make any mass-balance verdict. The shared PIE-001 layer owns exact/possible/impossible mass closure.

## Contract

Each stream has a stable ID, a material class, a disposition, source and destination, mass quantity status, and source/version/time/evidence metadata. A stream must be one of:

- external to a step (boundary input),
- one step to a later step (internal flow), or
- a step to external (boundary output).

Step input/output references must agree with stream endpoints. IDs must be unique, streams may not be orphaned, and internal flows must follow the declared step order. A quantity marked unknown has null value/lower/upper and an explicit reason; it is never coerced to zero.

The root boundary-inventory status is an explicit claim (claimed_complete, known_incomplete, or unknown), with its own provenance. The validator can check the record's structure, not independently prove the claim that all physical streams were observed.

## PIE-001 integration boundary

The mapper should preserve input provenance and convert grams to the shared PIE mass unit explicitly (1 g = 0.001 kg). A stream internal to the trace appears as a step output for its producer and a step input for its consumer; it must not be counted as a whole-batch external input/output. PIE-001 remains the sole owner of mass-balance classifications. This Culinary module reports only structural export readiness and explicit blockers.

## Qualification limits

The fixture and evidence URIs are synthetic. Passing structural validation means only that a trace is connected and sufficiently quantified for a mapping attempt. It does not authenticate sources, verify physical completeness, demonstrate conservation, validate nutrient retention, establish food safety, or predict taste/texture.
