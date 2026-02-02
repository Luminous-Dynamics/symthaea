# ZK Gradient Proof

Zero-knowledge proofs for gradient quality verification in federated learning using RISC-0.

## Overview

This module enables FL clients to prove their gradient quality without revealing the actual gradient values. This is crucial for:

- **Privacy**: Training data remains confidential
- **Byzantine resistance**: Proves gradients meet quality constraints
- **Audit trail**: Cryptographic proof of proper computation

## What Gets Proven

The zkVM guest verifies:
1. L2 norm is within `[min_norm, max_norm]`
2. All elements have magnitude ≤ `max_element_magnitude`
3. Dimension is ≥ `min_dimension`
4. All values are finite (no NaN/Inf)

## What Remains Private

- The actual gradient values (never leave the zkVM)
- Exact L2 norm
- Training data used

## Project Structure

```
zk-gradient-proof/
├── core/           # Shared types (witness, output, constraints)
├── methods/        # RISC-0 guest compilation
│   └── guest/      # zkVM guest program
└── host/           # Prover and verifier
```

## Prerequisites

1. Install Rust: https://rustup.rs
2. Install RISC-0 toolchain:
   ```bash
   cargo install cargo-risczero
   cargo risczero install
   ```

## Building

```bash
cd zk-gradient-proof
cargo build --release
```

First build takes ~5-10 minutes (compiles RISC-V target).

## Running the Demo

```bash
cargo run --release -p zk-gradient-host

# With logging
RUST_LOG=info cargo run --release -p zk-gradient-host
```

## Usage in Federated Learning

```rust
use zk_gradient_core::{GradientProofInput, GradientConstraints};
use zk_gradient_methods::{GRADIENT_QUALITY_ELF, GRADIENT_QUALITY_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};

// 1. Client prepares proof input
let input = GradientProofInput {
    gradient: my_gradient,
    global_model_hash: model_hash,
    epochs: 5,
    learning_rate: 0.01,
    client_id: "client-001".to_string(),
    round: current_round,
    constraints: GradientConstraints::for_federated_learning(),
};

// 2. Generate proof
let env = ExecutorEnv::builder()
    .write(&input)?
    .build()?;
let receipt = default_prover().prove(env, GRADIENT_QUALITY_ELF)?;

// 3. Extract public output
let output: GradientProofOutput = receipt.journal.decode()?;

// 4. Send (gradient_hash, receipt) to aggregator
// Aggregator verifies without seeing gradient values

// 5. Aggregator verifies proof
receipt.verify(GRADIENT_QUALITY_ID)?;
```

## Performance

| Operation | Time (release build) |
|-----------|---------------------|
| First proof | 30-120s (includes circuit compilation) |
| Subsequent proofs | 10-30s |
| Verification | <1s |
| Proof size | ~200KB |

## Security Properties

- **Soundness**: Cannot fake a valid proof for invalid gradient
- **Zero-knowledge**: Verifier learns only validity, not values
- **Non-interactive**: Proof is standalone, no back-and-forth
- **Deterministic**: Same input produces same commitment

## Integration with MATL

The proof output can feed into the Mycelix Adaptive Trust Layer:

```rust
// Update K-Vector based on proof validity
if proof_output.quality_valid {
    agent.k_vector.k_r += 0.01; // Reputation boost
    agent.k_vector.k_i += 0.01; // Integrity boost
}
```

## License

Apache 2.0
