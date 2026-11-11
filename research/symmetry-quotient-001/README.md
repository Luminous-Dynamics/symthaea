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

The verifier does not trust the certificate's counts or orbit listing. It revalidates the group and action, checks every forward and inverse witness, checks exact task-semantic equality, verifies that each orbit is complete under every group element, and requires a disjoint cover of the original candidate universe.

The lift operation first replays the certificate, then maps a selected representative to every original candidate in its orbit. In this finite-indexed v0.1 reference, lifting returns candidate identities and transformation witnesses; it is not a continuous-state lifting mechanism.

This version does not use a cryptographic digest. The verifier receives the exact original task and action records. A later version should bind canonical serialized inputs with cryptographic digests and independently replayable proof artifacts.

## Reproduction

    cargo fmt --manifest-path research/symmetry-quotient-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/symmetry-quotient-001/Cargo.toml
    cargo clippy --locked --manifest-path research/symmetry-quotient-001/Cargo.toml --all-targets -- -D warnings

## Deliberate limits

This is not a general-purpose group-discovery engine. It does not cover continuous Lie groups, approximate actions, infinite domains, non-finite objectives, transformed output spaces, or physical step-map equivariance. It must abstain when its exact finite contract is incomplete or its resource budget is exceeded.

The tests exhaustively enumerate candidate orbits independently of the certificate compiler. Adversarial tests mutate the group table, action permutations, task semantics, and certificate witnesses. No benchmark speedup is claimed from the tiny fixture; its purpose is to establish exactness before scaling.
