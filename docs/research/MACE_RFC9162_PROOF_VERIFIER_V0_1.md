# MACE RFC 9162 proof verifier v0.1

**Status:** Rust implementation and deterministic test suite authored. The algorithms were independently prototyped and exercised in Python in the authoring session, but the Rust crate could not be compiled or executed because this environment has no Rust toolchain. No CI, Rust test, deployed witness, or external security qualification is claimed.

## Purpose

The MACE decision-manifest verifier (#13625) currently depends on the AppendOnlyManifestWitnessLogVerifierV1 boundary. This increment adds a reusable Rust implementation of the underlying RFC 9162 Merkle Tree Hash proof algorithms without creating a second log, signer format, quorum protocol, or trust root.

Module:

- research/broca-mace-transfer-bench-v0-1/aml-adapter/src/rfc9162_proofs.rs

Export:

- pub mod rfc9162_proofs;

It implements:

- RFC 9162 leaf hashing: SHA-256(0x00 || leaf_data).
- RFC 9162 internal-node hashing: SHA-256(0x01 || left || right).
- Empty-tree hash.
- Inclusion proof verification with exact index, tree-size and proof-node bounds.
- Consistency proof verification from a pinned old checkpoint to a candidate new checkpoint.
- Strict lowercase SHA-256 digest wire parsing.

## Security boundary

A valid inclusion proof establishes only that a leaf is a member of the supplied Merkle root. A valid consistency proof establishes only that the candidate tree is an append-only extension of the supplied prior root.

Neither proof authenticates the checkpoint, proves that it is the newest checkpoint, prevents a log from presenting split views, authenticates the leaf's issuer, or establishes organizational independence.

The caller still needs:

1. An independently pinned checkpoint and log identity.
2. A separately verified checkpoint signature/quorum and trust-roster policy.
3. A prior checkpoint retained outside the candidate memory/evaluator rollback domain.
4. Consistency verification from that prior checkpoint, not a self-selected checkpoint returned with the proof.
5. An independent monitor/gossip or external transparency anchor when split-view resistance is claimed.
6. A MACE leaf binding the exact suite, episode-set root, sealed-manifest digest, assignment root, preregistration root, artifact root, candidate-access profile, reveal protocol and commit chronology.

The implementation deliberately does not add a new Ed25519 envelope or quorum schema. The next adapter should consume the existing MON-005 checkpoint/receipt protocol from #13518 and feed its already authenticated checkpoint/root and proof bundle into this verifier. The MON-005 v2/v3 reference reports 49/49 local tests, 496 inclusion checks and 2,016 consistency-proof pairs; its current one-host/one-process and fixture-key limitations remain. A Rust proof verifier does not upgrade those deployment claims.

## Test design

The Rust module includes an RFC-shaped tree/proof generator for tests only.

Authored tests cover:

- Every leaf in tree sizes 1 through 64: **2,080** inclusion paths.
- Every old/new size pair 1 <= m < n <= 64: **2,016** consistency proofs.
- Genesis extension with the exact empty-tree root.
- Same-size checkpoint identity and empty proof requirements.
- Wrong leaf, wrong root, out-of-range index, truncated and extended inclusion paths.
- Wrong old/new root, truncated and extended consistency paths, and tree-size rollback.
- Strict lowercase digest parsing and exact 32-byte length.

An independent Python implementation was exercised against all 2,080 inclusion paths and 2,016 consistency pairs for the same tree shape during authoring. That is a local algorithmic cross-check, not execution of the Rust source, an official RFC vector suite, or differential interoperability against a third-party CT implementation.

### Fixed cross-language reference vector

The Rust tests also pin a 13-leaf inclusion proof and an 11-to-13 consistency proof generated independently in Python. The deterministic leaves are the UTF-8 strings `mace-rfc9162-fixture-leaf-{i}` for indices 0 through 12.

- Tree size 13 root: `6adfcbd4eb5db2afdffcdaf31ee4231e0b899dc28228716130b5d3f202ceddf8`
- Leaf 7 hash: `806c793bf78be40e761f9baf94ff64166614a684e190ee363e8687711e345901`
- Tree size 11 root: `7f199e2a6d6e4a05e029092dd52e806710ab4068200d11b5108360e2f84e3517`

The inclusion audit path and consistency proof nodes are embedded verbatim in the Rust test. Their independent origin is a useful regression check, but a Rust test still has to execute before parity is established.

## Next integration steps

1. Run the Rust module tests, formatting and strict Clippy on the exact PR head.
2. Add published RFC 9162 vectors and differential tests against an independent mature CT library.
3. Define the exact MACE-to-MON-005 leaf projection; do not assume hashing the same JSON fields automatically yields the same MON-005 record leaf.
4. Implement a proof-provider adapter for the existing AppendOnlyManifestWitnessLogVerifierV1 trait that verifies the pinned checkpoint/quorum first, then validates the exact MACE leaf inclusion and prior-checkpoint consistency proofs.
5. Exercise both stale-data-only and common-rollback-domain T9/T10 restore cases against independently administered witness custody.
