# GEOM-001B — Native Cognitive-State Projection Authority

## Status

Foundational measurement tranche under #3127, stacked for repository-lineage purposes on GEOM-003B2 exact head `4e7d9522ec4f8d47d01481722bbeaf62eeea2dfe`.

This tranche freezes how Symthaea's native per-cycle cognitive projection becomes the simplex coordinates consumed by GEOM-001. It exists to prevent projection choice from becoming a hidden researcher degree of freedom after lesion results are known.

## Native source

The source observable is `CycleResult.thought_vector`.

At the current frozen implementation, the cognitive loop constructs this vector by partitioning the cycle HDV into 32 contiguous chunks and taking the arithmetic mean of each chunk. The GEOM projection layer therefore treats 32 dimensions as part of the measurement contract.

A dimension other than 32 is **environment / implementation drift**, not an invitation to pad, truncate, interpolate, or silently adapt the projection.

## Semantic boundary

The output of this tranche is a non-negative unit-sum vector on a simplex.

It is **not** claimed to be:

- a calibrated belief distribution;
- a probability distribution over external-world states;
- a probability that Symthaea is conscious;
- IIT Phi;
- a quantum state;
- a gravitational state.

It is a fixed coordinate representation used to measure trajectories with probability-simplex geometry.

## Primary projection

For signed native coordinates `x_i`, the primary representation is

`p_i = |x_i| / sum_j |x_j|`.

Properties:

- no learned parameters;
- no temperature;
- invariant to global non-zero scale;
- invariant to global sign inversion;
- permutation equivariant;
- preserves relative coordinate magnitude linearly.

## Mandatory sensitivity projection

The secondary representation is

`q_i = x_i^2 / sum_j x_j^2`.

This emphasizes concentrated coordinates more strongly than the primary projection while remaining parameter-free and globally scale/sign invariant.

It is **mandatory** sensitivity analysis, not an optional fallback.

## Fail-closed input rules

Reject the observation if:

- vector length is not exactly 32;
- any coordinate is NaN or infinite;
- all coordinate magnitudes are zero.

Do not repair these cases in the measurement layer.

## Controls

The implementation freezes the following controls before target-system outcomes:

1. one-hot native states project to the same simplex vertex under both mappings;
2. equal coordinate magnitudes project to the uniform simplex point even with alternating signs;
3. multiplying the entire native vector by a non-zero scalar, including a negative scalar, does not change either projection;
4. permuting native dimensions permutes projected dimensions in exactly the same way;
5. the primary and sensitivity projections are demonstrably different on a non-uniform vector;
6. both outputs are non-negative and sum to one;
7. wrong dimensions, non-finite values, and zero vectors fail closed.

## Projection-sensitivity rule

Every scientific GEOM run using the thought-vector projection must retain both trajectories:

- primary L1-magnitude trajectory;
- squared-energy sensitivity trajectory.

No result may select whichever projection looks more favorable after observing outcomes.

If a qualitative conclusion depends materially on which of the two preregistered projections is used, report:

> projection-sensitive

rather than:

> robust geometric effect.

A later statistical tranche may define exact quantitative concordance thresholds, but it may not replace or hide either projection.

## Why no softmax

A softmax projection would require at least one temperature parameter and would make geometric concentration strongly temperature-dependent. That is useful for some models but undesirable as the first GEOM authority because it adds an avoidable researcher-controlled degree of freedom.

## Why no learned codebook

A semantic anchor/codebook projection may be scientifically useful later, but its anchor selection and training history introduce another model layer. GEOM-001B deliberately begins with the native 32D projection already emitted by Symthaea and adds no learned representation.

## Claim boundary

Allowed:

> Under the preregistered native-state projection, the Fisher-Rao trajectory changed by X under intervention Y, and the squared-energy sensitivity projection showed Z.

Not allowed:

> These simplex coordinates are probabilities of consciousness.

Not allowed:

> A robust geometric effect establishes consciousness.

Not allowed:

> Similar geometry establishes a physical link to gravity.

## Promotion gate

GEOM-001B can qualify projection integrity independently, but scientific use remains blocked on the qualified GEOM measurement and intervention lineages. No target-system lesion data should be used to revise these projection definitions.
