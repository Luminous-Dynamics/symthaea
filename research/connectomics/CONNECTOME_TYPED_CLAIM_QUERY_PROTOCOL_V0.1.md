# Typed multiplex connectome claims and query grammar V0.1

**Research issue:** [#10349](https://github.com/Luminous-Dynamics/symthaea/issues/10349)  
**Schema:** `connectome-typed-claim-bundle-v0.1.schema.json`

## Purpose

A connectome analysis can combine anatomical synapses, reconstruction correspondence, statistical dependence, functional association and causal effects. These are different relation types with different evidence and error models. V0.1 prevents one from silently promoting a claim across these layers.

A typed relation is not a truth assertion. The schema uses `CLAIMED_*` status values to make the producer-authored boundary explicit.

## Relation registry

- **Anatomical synapse edge:** binds a synapse candidate, source/target neuron or compartment, presence, polarity, pre-/post-partner claims, image revision and coordinate frame. This does not establish functional efficacy or causality.
- **Reconstruction correspondence:** binds objects across named source releases using a typed mapping receipt. It does not create a within-release anatomical synapse.
- **Statistical dependence edge:** binds model, sample frame, conditioning set, hypothesis ID and family, multiplicity procedure, sampling design, effect estimate and uncertainty. A q-value applies only to its declared test family.
- **Functional association edge:** binds recordings, acquisition protocol, signal processing, state/context and a frozen temporal window. It is not direct anatomical evidence.
- **Causal effect claim:** binds intervention protocol, assignment mechanism, target population, outcome, estimand, identification strategy, assumptions, positivity and selection diagnostics. Topology and observational dependence cannot satisfy this contract.

The 2026 pathwise feature-selection paper is a comparator for statistical local-graph inference, not an EM proofreading or synapse-detection baseline: https://pubmed.ncbi.nlm.nih.gov/42120384/

## Typed query grammar

Every query freezes a formal definition, analysis digest, single-layer or declared-mixed mode, requested claim type, allowed relation-type path sequences, claim IDs used by each result path and producer-declared result digest/status.

The semantic verifier must recompute relation-type sequences from referenced claim IDs and compare them with the allowed path grammar. A producer cannot label an anatomical-plus-statistical path as causal. Mixed evidence remains mixed evidence until a separate relation-specific claim and evidence adapter justifies any stronger interpretation.

For multi-claim paths, adjacent **traversable** relations must compose exactly: the previous claim's target object ID, object kind and world/release must equal the next claim's source object ID, kind and world/release. A mismatch is `REJECTED_PATH_COMPOSITION`, even when the relation-type sequence is permitted. Reconstruction correspondence may intentionally cross releases, but the next relation must begin at its declared target-side object and world.

A `CAUSAL_EFFECT_CLAIM` is not a traversable anatomical/statistical edge. It is an attached estimand/identification assertion and is checked separately for intervention, target population, outcome, positivity and selection. Its endpoints therefore do not participate in ordinary graph-edge continuity. This keeps causal inference attached to a declared query without pretending an intervention record is another anatomical synapse.

## Non-laundering rules

1. Every claim has exactly one primary relation type.
2. Evidence profiles must match the relation type.
3. Claim and query IDs are unique.
4. Every path references existing claims and its relation-type order is recomputed.
5. Path assertions must be allowed by the query's frozen grammar.
6. Anatomical edge construction requires presence, polarity and pre-/post-partner support. Unresolved components yield ambiguous or not-estimable status.
7. Cross-release correspondence never becomes a within-release anatomical edge.
8. Statistical q-values are bound to a declared complete hypothesis family and multiplicity procedure.
9. Statistical dependence cannot promote anatomical presence or causal effect.
10. Functional association cannot promote anatomy or causal effect.
11. Causal promotion requires an independently authenticated intervention and identification profile, including adequate positivity and resolved selection.
12. Evidence ancestry/common training/data sources are recorded; correlated sources cannot be counted as independent.
13. Exact result bytes are digest-bound.
14. The verifier reports structural replay only and keeps `trusted_promotion=false` until evidence bytes and authorities are authenticated.

## Known-answer fixture

The synthetic bundle includes one anatomical synapse claim with separate presence/polarity/partner declarations; one cross-release neuron correspondence; one statistical dependence edge with an explicit hypothesis family and q-value; one functional association edge with a frozen time window; and one causal claim marked not estimable because positivity/selection requirements are unresolved.

It also contains an intentionally invalid producer attempt to infer a causal effect from anatomical and statistical evidence. The attempt is preserved as a claim so a future semantic verifier can reject it without erasing the audit trail.

## Required semantic adversarial tests

Reject statistical evidence labelled anatomical; anatomical evidence labelled causal; correspondence used as a within-release synapse; unresolved polarity used as a directed edge; source/target world mismatch; duplicate or unknown IDs; path references to missing claims; relation-type order inconsistent with the frozen grammar; mixed evidence promoted to causal without a causal claim; q-values not bound to their hypothesis family; multiplicity procedure changed after results; positivity violation on a causal claim marked supported; observational association promoted to causality; query code/grammar changed after result digest; correlated model ancestry counted as independent; and supported output without a result digest.

## Proposed qualification targets (not results)

- 100% rejection of invalid relation-type promotions in exhaustive synthetic fixtures.
- Zero anatomical edges promoted from statistical dependence alone.
- Zero causal claims promoted from topology or observational dependence alone.
- 100% rejection of malformed or unbound statistical hypothesis-family claims.
- Deterministic replay for identical typed claim bundles and query inputs.
- Statistical false-discovery control evaluated only under assumptions of the selected method; it is not a synapse precision claim.
- Query-level sensitivity and robustness evaluated across plausible reconstruction worlds, preserving NOT_IDENTIFIABLE and NOT_ESTIMABLE.

## Claim ceiling

V0.1 schema and fixtures validate only a typed representation contract. They do not validate biological edges, authenticate evidence, establish causality, or reproduce the statistical graph estimator. Relation-specific evidence adapters and an independent semantic verifier remain required.
