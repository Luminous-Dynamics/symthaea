# winterfell-pogq

Domain-specific STARK proofs for Proof-of-Gradient-Quality Byzantine detection in federated learning.

## License

Apache-2.0 -- see [LICENSE](LICENSE) for details.

Commercial licensing available from [Luminous Dynamics](https://luminousdynamics.org).

## Usage

```rust
use winterfell_pogq::{PogqProver, PogqVerifier, GradientCommitment};

// Commit to a gradient update
let commitment = GradientCommitment::from_gradients(&gradients);

// Generate a STARK proof that the gradient satisfies quality bounds
let proof = PogqProver::prove(&commitment, &quality_params)?;

// Verify the proof without seeing the raw gradients
let valid = PogqVerifier::verify(&proof, &commitment.public_inputs())?;
```

## Features

- STARK proofs for gradient quality validation (built on Winterfell)
- Proof-of-Gradient-Quality (PoGQ) for Byzantine-tolerant federated learning
- BLAKE3-based cryptographic commitments
- Benchmark binaries for proof generation performance
- Optional Python bindings via PyO3 (`python` feature)

## Part of [Mycelix](https://mycelix.net)

Decentralized infrastructure for cooperative civilization.
