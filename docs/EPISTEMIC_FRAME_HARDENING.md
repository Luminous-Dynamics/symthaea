# Epistemic Frame Hardening

## Status
Research/design artifact for GIS hardening. This document deliberately separates confidence in a claim from confidence in the frame that produced the claim.

## Why this matters

GIS currently has a useful ignorance taxonomy and 3D uncertainty model, but IgnoranceType::None is documented as complete knowledge. That semantic is too strong for an embodied or open-world cognitive system.

A system can have high confidence that a proposition follows from its evidence, low measured uncertainty inside its current model, and still be wrong because the model, schema, ontology, or excluded variables were inappropriate.

Recent work on quantitative inference makes the same distinction: rigor earned inside a finite frame is not the same thing as confidence that the frame itself is appropriate.

> No detected ignorance means no ignorance detected under the current evidence, model, schema, and ontology. It must never mean proof of completeness.

## Four uncertainty layers

1. Proposition uncertainty — Is claim X supported?
2. Model uncertainty — Is the explanatory/predictive model appropriate?
3. Ontology uncertainty — Are the entities, categories, and variables being represented the right ones?
4. Frame uncertainty — Is the overall representation of the problem appropriate, including what has been excluded?

The fourth layer is not merely another scalar. It is partly a statement about the limits of what the current representation can express.

## Proposed EpistemicFrame

A future executable representation should make the reasoning frame inspectable:
- evidence boundary
- perceptual/measurement assumptions
- definitions
- ontology/schema identifier
- causal/model assumptions
- excluded variables or relations
- known blind spots
- alternative frames considered
- frame revision/correction history

A claim should therefore be understood as: Claim C is supported by evidence E within frame F, rather than Claim C equals 0.87 confidence.

The confidence value remains useful; it is simply no longer allowed to masquerade as frame validation.

## Pipeline

The cognitive provenance chain should remain explicit:

observation → percept → interpretation → hypothesis → ontology/schema → causal model → prediction → intervention → outcome

Ontology and frame are not just one downstream stage. They constrain what can count as an observation, entity, variable, cause, or outcome in the first place.

## Minimal implementation path

1. Change the semantics/documentation of IgnoranceType::None from complete knowledge to no ignorance detected under the active frame.
2. Add deterministic tests proving that no-detected-ignorance does not imply completeness.
3. Add an explicit frame descriptor with stable identity/versioning.
4. Attach frame metadata to IgnoranceDetection.
5. Add model/schema/frame qualification to the consciousness decision gate.
6. Preserve backward compatibility for existing callers while introducing stronger APIs.
7. Add correction events so ontology/frame revisions can invalidate or qualify prior conclusions without deleting provenance.

## Deterministic adversarial tests

### Same evidence, different schema
The same observations are evaluated under two schemas that expose different variables. Proposition confidence may remain high in both, while frame divergence is non-zero.

### Ontology revision exposes a missing variable
A previously accepted model receives a new ontology version that makes a previously unrepresented entity or relation explicit. Prior confidence remains historically valid for the old frame but is not silently promoted to the new frame.

### High proposition confidence, high structural uncertainty
A claim can be locally well-supported while its model structure remains uncertain. The decision gate must retain the structural qualification.

### Collective agreement with shared blind spot
Multiple agents can agree on the same claim while sharing the same ontology/schema. Agreement increases evidence about agreement, not proof that the frame is complete.

### Correction after frame revision
A frame revision creates a correction event linked to the original claim, preserving the original observation and decision history.

## Design invariant

> Consensus is not completeness. Confidence is not ontology validation. A clean inference is not evidence that the frame was complete.

## Relationship to existing GIS

This proposal complements, rather than replaces:
- Graceful Ignorance
- 3D uncertainty
- Dark Spot DHT
- Epistemic Decision Gate
- Epistemic Mirror
- Rashomon / alternate perspectives
- Socratic questioning
- curiosity / expected information gain

The new primitive is the missing connective tissue: explicit provenance for the frame in which ignorance and confidence are evaluated.

## External research note

2026 work on uncertainty estimation argues that existing predictive uncertainty methods capture only partial sources of epistemic uncertainty. Separate 2026 work argues that automated inference has a structural frame limitation: what lies outside the chosen specification may not appear as uncertainty at all. These findings support treating frame uncertainty as a first-class design concern rather than assuming that a sufficiently calibrated scalar closes the epistemic problem.

## Success condition

A future GIS implementation should be able to say, deterministically and honestly:

> Within frame F, given evidence E, claim C has confidence X. Frame F has these assumptions and exclusions. Alternative frame F2 changes these variables. Therefore the confidence value is conditional, not a claim of completeness.

That is a materially stronger epistemic contract than simply returning a high confidence score.