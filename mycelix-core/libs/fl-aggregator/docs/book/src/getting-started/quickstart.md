# Quick Start

This guide will help you generate and verify your first zkSTARK proof in under 5 minutes.

## Prerequisites

- Rust 1.75 or later
- fl-aggregator with the `proofs` feature

## Step 1: Add Dependency

```toml
[dependencies]
fl-aggregator = { version = "0.1", features = ["proofs"] }
```

## Step 2: Generate a Range Proof

A range proof demonstrates that a value lies within a specific range without revealing the value.

```rust
use fl_aggregator::proofs::{
    RangeProof, ProofConfig, SecurityLevel, VerificationResult
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure proof parameters
    let config = ProofConfig {
        security_level: SecurityLevel::Standard128,
        parallel: true,
        max_proof_size: 0, // No limit
    };

    // Generate proof that 42 is in range [0, 100]
    let proof = RangeProof::generate(42, 0, 100, config)?;

    println!("Proof generated!");
    println!("  Size: {} bytes", proof.size());

    // Verify the proof
    let result = proof.verify()?;

    println!("Verification result:");
    println!("  Valid: {}", result.valid);
    println!("  Time: {:?}", result.verification_time);

    Ok(())
}
```

## Step 3: Verify a Gradient Proof

For federated learning, prove gradient integrity:

```rust
use fl_aggregator::proofs::{
    GradientIntegrityProof, ProofConfig, SecurityLevel
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ProofConfig {
        security_level: SecurityLevel::Standard128,
        parallel: true,
        max_proof_size: 0,
    };

    // Your gradient vector
    let gradients: Vec<f32> = vec![0.1, -0.2, 0.3, -0.1, 0.05];

    // Maximum allowed L2 norm
    let max_norm = 1.0;

    // Generate proof
    let proof = GradientIntegrityProof::generate(&gradients, max_norm, config)?;

    println!("Gradient proof generated!");
    println!("  Gradient elements: {}", gradients.len());
    println!("  Proof size: {} bytes", proof.size());

    // Verify
    let result = proof.verify()?;
    assert!(result.valid);

    Ok(())
}
```

## Step 4: Batch Verification

Verify multiple proofs efficiently:

```rust
use fl_aggregator::proofs::{
    RangeProof, ProofConfig, SecurityLevel,
    integration::{BatchVerifier, AnyProof}
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ProofConfig {
        security_level: SecurityLevel::Standard96,
        parallel: true,
        max_proof_size: 0,
    };

    // Generate multiple proofs
    let proofs: Vec<AnyProof> = (0..10)
        .map(|i| {
            let proof = RangeProof::generate(i * 10, 0, 100, config.clone()).unwrap();
            AnyProof::Range(proof)
        })
        .collect();

    // Batch verify
    let verifier = BatchVerifier::new();
    let result = verifier.verify_all(&proofs)?;

    println!("Batch verification:");
    println!("  Total: {}", result.total);
    println!("  Valid: {}", result.valid_count);
    println!("  Time: {:?}", result.total_time);

    Ok(())
}
```

## Next Steps

- [Configuration](./configuration.md) - Tune proof parameters
- [Proof Types](../proofs/overview.md) - Learn about all proof types
- [Integration](../integration/fl.md) - Integrate with federated learning
