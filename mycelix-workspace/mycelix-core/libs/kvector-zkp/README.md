# kvector-zkp

Zero-knowledge STARK proofs for K-Vector validation in Mycelix.

## License

Apache-2.0 -- see [LICENSE](LICENSE) for details.

Commercial licensing available from [Luminous Dynamics](https://luminousdynamics.org).

## Usage

```rust
use kvector_zkp::proof::KVectorProof;
use mycelix_core_types::k_vector::KVector;

// Create a K-Vector and generate a STARK proof of its validity
let kv = KVector::new(vec![0.8, 0.2, 0.5, 0.1]);
let proof = KVectorProof::prove(&kv, &bounds)?;

// Verify without revealing the underlying vector values
let valid = proof.verify(&public_inputs)?;
```

## Features

- STARK proofs for K-Vector range and consistency validation (built on Winterfell)
- Proof caching with LRU for repeated verifications
- Parallel proof generation (`parallel` feature, enabled by default)
- Experimental GPU acceleration (`gpu` feature)
- SHA-3 based commitments
- Integration with mycelix-core-types K-Vector

## Part of [Mycelix](https://mycelix.net)

Decentralized infrastructure for cooperative civilization.
