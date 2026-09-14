# Generativity Evidence Vector v1

## Purpose

The generativity evidence contract describes how an action, project, resource, or policy may change future productive capacity, optionality, diversity, and regeneration while making dependency, concentration, and irreversibility risks visible.

It is deliberately not a universal score.

## Relationship to existing exploration and EFE

Symthaea already has surprise-driven exploration and Expected Free Energy machinery. Those answer different questions:

- surprise-driven exploration asks whether prediction error warrants exploration and whether subsequent surprise falls;
- EFE represents pragmatic and epistemic contributions for action evaluation;
- the generativity vector records evidence about what useful capacity and future optionality an intervention creates, preserves, or constrains.

Keeping these concepts separate prevents a descriptive measurement model from silently acquiring execution authority.

## Dimensions

Positive-capacity dimensions:

- `immediate_utility`
- `epistemic_gain`
- `option_value`
- `diversity`
- `capability_gain`
- `diffusion`
- `commons_gain`
- `regeneration`

Risk dimensions:

- `dependency_risk`
- `concentration_risk`
- `irreversibility_risk`

Every dimension contains both a normalized estimate and an explicit confidence value. Assessments also retain evidence, assumptions, context, and unresolved uncertainty.

## Non-authority invariant

A `GenerativityAssessment` is evidence, not permission.

No assessment, regardless of its values or confidence, is sufficient by itself to:

1. execute an action;
2. change a MAGI execution mode or EFE weighting;
3. grant a capability or access right;
4. modify identity, reputation, MATL, or MYCEL;
5. change governance or voting weight;
6. allocate money, compute, land, water, or other commons resources.

Any future bridge to one of those systems must define a separate, explicit policy and authority boundary.

## Why there is no canonical scalar score

Scalarization would embed a value system into what should remain an evidence contract. It would also create a simple reward target that could be Goodharted or reward-hacked.

Consumers should preserve the vector. A later comparison layer may use, for example:

- Pareto dominance;
- quality-diversity archives;
- domain-specific policies;
- community-governed preference profiles;
- constrained optimization with hard floors or ceilings.

Those policies must remain separate from the evidence representation itself.

## V1 validation invariants

- all numeric values and confidence values are finite;
- normalized values are in `[0, 1]`;
- the schema identifier is explicit and versioned;
- subject and context are non-empty;
- evidence entries have stable identities and explicit kinds;
- unknown schema versions fail validation rather than being silently interpreted.

## Follow-on tranches

1. Add typed exploration outcomes that separate surprise reduction from knowledge gain, diversity retention, and option creation.
2. Add Pareto-dominance and archive semantics over generativity vectors without introducing a universal score.
3. Bind evidence items to Symthaea's existing evidence/provenance plane and resolution contracts.
4. Define a compatible Mycelix contribution-lineage attestation so enabling contributions can be traced without granting ownership over descendants.
5. Define domain-specific regeneration observations for commons resources.
6. Add adversarial tests for proxy gaming, fabricated evidence, self-referential scoring, concentration feedback loops, and confidence laundering.

## Design principle

A successful intervention should be able to leave behind more capability, knowledge, optionality, resilience, or regenerative capacity than it consumed, while preserving explicit evidence about trade-offs and uncertainty.
