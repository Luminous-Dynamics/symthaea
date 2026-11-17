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

## Content-addressed certificate binding v1

The content_binding module adds canonical, domain-separated binary encodings for the full search problem, finite group/action, quotient certificate, and binding manifest. Integer values are fixed-width little-endian; collection lengths are encoded as u64; strings are UTF-8 with byte-length prefixes; each object begins with a distinct versioned domain tag. The encoding does not depend on JSON whitespace, map iteration order, or platform-sized byte representations.

A BoundQuotientPackageV1 records the SHA-256 digest of the canonical task/problem, the finite group/action, and the complete quotient certificate. Its manifest also binds the exact qualification budget, encoding byte budget, hash algorithm identifier and canonical encoding schema identifier.

bind_quotient_certificate first runs the independent quotient verifier and only then produces the binding. verify_bound_certificate recomputes all object and manifest digests and then replays the independent quotient verifier again. A hash match alone never authorizes pruning.

The implementation uses the maintained RustCrypto sha2 crate rather than a project-local hash implementation. The standard SHA-256 empty-string and abc vectors are checked in the test suite.

Reproduction:

    cargo fmt --manifest-path research/symmetry-quotient-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/symmetry-quotient-001/Cargo.toml
    cargo clippy --locked --manifest-path research/symmetry-quotient-001/Cargo.toml --all-targets -- -D warnings

The binary encoding is an identity format for the exact v1 Rust data model, not a general cross-language JSON protocol. A future interoperability version must publish a language-neutral decoder and frozen byte fixtures before claiming cross-implementation compatibility.

Security boundary: SHA-256 makes accidental or malicious mutation detectable when a trusted expected digest is retained. This package is not a digital signature, publisher identity, proof of Rust implementation correctness, proof of scientific model fidelity, formal theorem proof, or physical-equivalence proof. The mathematical/action/task verifier remains mandatory, and the search-pruning claim stays scoped to the exact finite/discrete contract.
