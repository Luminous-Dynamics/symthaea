# Affective Emergence v0.2 — Scenario Constructibility and Parameter Freeze

Status: **normative design-only / blocked on Native Interoception v0.1 qualification**

This contract governs how the abstract X00–X11 scenario families become exact deterministic numeric arms after the observatory implementation exists.

## Principle

A scenario may need numerical search to satisfy a prospectively declared mechanical contrast. That search must not become candidate optimization.

The allowed question is:

> Can the native v0.1 dynamics construct a valid arm/pair that satisfies the already-frozen X-family mechanical constraints?

The forbidden question is:

> Which valid parameters make a preferred E-candidate look strongest, most affect-like, or most predictive?

## Two-phase materialization

### Phase A — constructibility search

Allowed inputs:

- frozen v0.1 native dynamics;
- frozen X-family structural constraints;
- native state/configuration/drives/interventions;
- mechanical homeostasis/allostasis quantities needed to establish the X-family equality/inequality constraint;
- deterministic solver/search diagnostics.

Forbidden inputs:

- E00–E11 rankings or promotion scores;
- Y0–Y3 functional evaluation results;
- semantic affect interpretation;
- unblinded condition labels beyond what is mechanically required to construct the contrast;
- post-hoc plot appearance;
- any oracle information not explicitly required by a diagnostic-only X family.

Candidate implementations may be compiled during this phase only when needed to prove API/availability compatibility, but their comparative numerical outputs cannot be used as the parameter-search objective.

### Phase B — parameter freeze

Once every required X family has at least one valid constructible realization, freeze one `ExploratoryScenarioParameterManifest` before exploratory candidate comparison.

The manifest must bind:

- exact X-family and arm IDs;
- full native initial states;
- dynamics configs;
- drives/interventions and timing;
- exact cut points/windows;
- matched-group identities;
- all mechanical equality/inequality constraints and tolerances;
- constructibility search algorithm/version;
- search space/bounds;
- deterministic tie-breaking rule;
- rejected-construction accounting or digest;
- v0.1 source/model-semantics identity;
- canonical digest.

After freeze, changing any outcome-relevant numeric parameter creates a new exploratory scenario lineage.

## Selection among multiple valid constructions

If multiple parameterizations satisfy a frozen X-family constraint, choose without E/Y outcome inspection using a prospective deterministic rule, for example:

1. minimize absolute distance from default v0.1 configuration;
2. then minimize total intervention/drive magnitude;
3. then lexicographically minimize the canonical parameter vector.

The exact rule must be frozen before the search returns candidate-dependent results.

Do not choose the realization with the largest E-candidate separation.

## Constructibility failure

If an X-family cannot be realized inside the declared valid v0.1 domain:

- preserve the failed search evidence;
- classify the family as `NotConstructibleUnderCurrentSubstrate`;
- do not weaken the mechanical constraint after inspecting candidate outputs;
- revise/supersede the design prospectively if the discriminator is still scientifically necessary.

Inability to construct a crossed-sign or matched-state case is itself information about the substrate/design.

## Independence from promotion targets

Y0–Y3 are not available during scenario construction or parameter selection.

The parameter freeze must predate the frozen candidate-payload and later-outcome artifacts used by `V02_FUNCTIONAL_EVALUATION_AND_PROMOTION.md`.

## Adversarial tests

Future implementation should include known-bad parameter selectors that:

- maximize E05 separation;
- minimize E01 baseline performance;
- select the most dramatic post-hoc cut point;
- inspect Y1/Y2 outcomes while choosing arms;
- iterate until one preferred candidate wins.

Each must be rejected by provenance/objective-dependency checks.

A valid control uses only mechanical X-family constraints and the frozen deterministic tie-break rule.

## Claim boundary

This contract makes scenario materialization an engineering/constructibility step rather than a hidden model-selection step. It does not establish that the resulting worlds are biologically realistic or that any candidate measured in them is affect, emotion, feeling, mood, suffering, sentience, or consciousness.
