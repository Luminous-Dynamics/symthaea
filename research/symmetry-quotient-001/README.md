# Finite symmetry quotient reference v0.1

This crate is the first executable slice of [SYMM-QUOTIENT-001](https://github.com/Luminous-Dynamics/symthaea/issues/11616).

It starts with finite, exactly enumerable domains: C2 and C3 actions, plus a non-abelian D4 action on ordered pairs of square vertices. These make the unquotiented search small enough to enumerate independently and compare against exhaustive orbit oracles.

## Authority boundary

The compiler separates three questions:

1. Is the multiplication table a finite group?
2. Does the declared permutation family define an action of that group on the complete candidate universe?
3. Does every action element preserve the complete task semantics attached to every candidate?

Only when all three are verified does the compiler emit a quotient certificate. A candidate group or a mathematically valid action is not enough.

The exact discrete task contract records domain membership, admissibility, feasibility, integer objective cost, constraint signature, boundary-condition identity, and output-equivalence identity. Every field must be preserved exactly. Unknown or approximate semantics are not accepted.

## Certificate replay

The certificate includes problem/action/group identities, group order, candidate count, deterministic orbit representatives, every orbit member, forward and inverse group-element witnesses, and recomputed check counts.

The verifier does not trust the certificate's counts, orbit listing, or stabilizers. It revalidates the group and action, checks every forward and inverse witness, checks exact task-semantic equality, verifies that each orbit is complete under every group element, reconstructs each stabilizer from the action, checks subgroup closure, verifies |G| = |orbit| × |stabilizer|, and requires a disjoint cover of the original candidate universe.

The lift operation first replays the certificate, then maps a selected representative to every original candidate in its orbit. In this finite-indexed v0.1 reference, lifting returns candidate identities and transformation witnesses; it is not a continuous-state lifting mechanism.

This version does not use a cryptographic digest. The verifier receives the exact original task and action records. A later version should bind canonical serialized inputs with cryptographic digests and independently replayable proof artifacts.

## Reproduction

    cargo fmt --manifest-path research/symmetry-quotient-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/symmetry-quotient-001/Cargo.toml
    cargo clippy --locked --manifest-path research/symmetry-quotient-001/Cargo.toml --all-targets -- -D warnings

## Deliberate limits

This is not a general-purpose group-discovery engine. It does not cover continuous Lie groups, approximate actions, infinite domains, non-finite objectives, transformed output spaces, or physical step-map equivariance. It must abstain when its exact finite contract is incomplete or its resource budget is exceeded.

The D4 benchmark also compares an actual objective search against exhaustive enumeration: a 16-candidate ordered-pair problem has three verified orbits, the optimum found by evaluating one representative per orbit matches the exhaustive optimum, and lifting reconstructs all four original minimizers. This yields 16/3 = 5.33× fewer objective evaluations on this exact finite fixture; it is candidate-work reduction, not a wall-clock speedup claim. The tests independently enumerate candidate orbits, mutate the group table/action/task/certificate witnesses, and verify the optimum and full solution lift.


## Multi-environment common symmetry (v0.2 increment)

The first compiler qualified one task snapshot. The new API `qualify_common_task_preserving_subgroup` accepts a set of named task environments over the same candidate universe and computes the exact intersection of their task-preserving stabilizers.

A group element is retained only when it preserves **every** candidate's complete semantics in **every** declared environment: domain membership, admissibility, feasibility, objective, constraint signature, boundary condition and output-equivalence identity. For every excluded ambient element, the report records an environment and candidate pair witnessing the first semantic difference, including the differing task fields.

The retained set is checked for identity, inverse closure and multiplication closure, then materialized as a restricted finite group/action. `compile_multi_environment_quotient` compiles a separate quotient certificate for each environment using this same common subgroup. `verify_multi_environment_quotient` independently reconstructs the common subgroup and replays every per-environment certificate before reporting orbit counts.

This makes the safe symmetry regime explicit:
- D4 geometry and nominal objective may be invariant in one environment;
- a changed operating regime may preserve only C4, so reflections are excluded;
- a further boundary-condition or objective change may reduce the common subgroup to C1.

It is still exact, finite and discrete. No approximate/soft symmetry receives pruning authority. An empty common quotient beyond identity simply means no search reduction was qualified across the declared environment family.

Tests cover D4-to-C4 intersection (with witness records for the four excluded reflections), a third asymmetric environment that shrinks the common subgroup to identity, and rejection of a forged common-subgroup witness. The per-environment certificates are replayed separately.


## Contextual symmetry atlas (new v0.3 direction)

The common-subgroup compiler above is the correct choice when one quotient must remain valid across the entire declared environment family. It may be conservative when the operating context is known before search.

The new `compile_contextual_multi_environment_quotient` therefore constructs a separate, context-bound subgroup and quotient for each environment. Each certificate carries the exact `context_id`, its local task-preserving subgroup, and its own quotient certificate. The verifier recomputes and replays every local subgroup and quotient; relabelling one context's certificate for another is rejected.

The hand-auditable D4 ordered-pair fixture now distinguishes:
- isotropic context: D4 retained, 3 orbits over 16 candidates;
- rotating-regime context: C4 retained, 4 orbits;
- asymmetric context: C1 retained, 16 singleton orbits.

This is not permission to reuse a C4 certificate in the isotropic or asymmetric context. The caller must present the certificate whose exact context ID matches the active environment. A missing or unknown context should abstain rather than inherit another environment's pruning permission.

The common-intersection compiler remains available and returns C1 for the same three-environment family. The contextual atlas demonstrates the more useful alternative: **preserve the maximal exact symmetry locally, while binding pruning authority to the operating context**. Both approaches are exact finite-group constructions; neither accepts soft/spectral proposals as proof.
