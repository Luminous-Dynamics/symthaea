# VSV-STARK: Verifiable PoGQ Decision Logic

**Verifiable Symbiotic Voting via Succinct Transparent Arguments of Knowledge**

This directory contains the STARK proof system for verifying PoGQ-v4.1's per-round decision logic in Byzantine detection for federated learning.

## What Does It Prove?

VSV-STARK proves the correctness of one round of PoGQ decision logic:

1. **EMA Update**: `ema_t = β * ema_prev + (1-β) * x_t`
2. **Warm-up Logic**: No quarantine during first W rounds
3. **Hysteresis Counters**: Track consecutive violations/clears
4. **Conformal Threshold**: Compare hybrid score to threshold
5. **Quarantine Decision**: Enter/release/maintain based on state

**What it does NOT prove in v0**:
- PCA extraction (deferred to v1.1)
- Cosine similarity computation (deferred to v1.1)
- Hybrid score derivation from gradient (deferred to v1.1)
- SHA-256 preimage for gradient hash (deferred to v1.1)

**Threat model for v0**:
- Prover can choose arbitrary hybrid score `x_t` (witness)
- But prover CANNOT cheat on decision logic given `x_t`
- v1.1 will bind `x_t` to gradient hash via PCA/cosine proof

## Directory Structure

```
verif/
├── SPEC.md                    # Complete specification
├── README.md                  # This file
├── air/                       # Rust AIR implementation
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs             # Main library
│   │   ├── air.rs             # AIR constraints
│   │   └── public.rs          # Public inputs & witness
│   └── benches/               # Performance benchmarks
│       └── proof_bench.rs
├── python/                    # PyO3 wrapper
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs             # Python bindings
└── tests/
    └── integration_test.py    # End-to-end tests
```

## Installation

### Prerequisites

- Rust 1.70+ (`rustup install stable`)
- Python 3.8+
- maturin (`pip install maturin`)

### Build Rust Library

```bash
cd verif/air
cargo build --release
cargo test
```

### Build Python Extension

```bash
cd verif/python
maturin develop --release
```

This compiles the Rust code and installs the `vsv_stark` Python module.

### Verify Installation

```python
import vsv_stark

print(f"VSV-STARK version: {vsv_stark.__version__}")
print(f"Scale factor: {vsv_stark.SCALE}")

# Convert float to fixed-point
beta_fp = vsv_stark.to_fixed(0.85)
print(f"0.85 → {beta_fp} (fixed-point)")
```

## Usage

### From Python

```python
import vsv_stark
import json

# Public inputs (commitments + parameters + previous state)
public = {
    "h_calib": "a" * 64,          # SHA-256 of calibration parameters
    "h_model": "b" * 64,          # SHA-256 of model weights
    "h_grad": "c" * 64,           # SHA-256 of client gradient
    "beta_fp": 55705,             # EMA β = 0.85
    "w": 3,                       # Warm-up rounds
    "k": 2,                       # Violations to quarantine
    "m": 3,                       # Clears to release
    "egregious_cap_fp": 65535,    # Hybrid score cap
    "threshold_fp": 58982,        # Conformal threshold = 0.90
    "ema_prev_fp": 49152,         # Previous EMA = 0.75
    "consec_viol_prev": 1,        # Previous violation streak
    "consec_clear_prev": 0,       # Previous clear streak
    "quarantined_prev": 0,        # Previous quarantine status
    "current_round": 5,           # Current round number
    "quarantine_out": 0           # Expected output decision
}

# Private witness (only known to prover)
witness = {
    "x_t_fp": 52428,              # Hybrid score = 0.80
    "in_warmup": 0,               # Not in warm-up
    "violation_t": 1,             # Below threshold
    "release_t": 0                # Not releasing
}

# Generate proof
proof_bytes, proof_hex = vsv_stark.generate_round_proof(public, witness)
print(f"Proof size: {len(proof_bytes)} bytes")

# Verify proof
is_valid = vsv_stark.verify_round_proof(proof_bytes, json.dumps(public))
assert is_valid, "Proof verification failed"
print("✅ Proof verified successfully")
```

### From Command Line

```bash
# Generate proof for artifact directory
python scripts/prove_decisions.py results/artifacts_run_001/

# Output:
# - decision_proof.bin (STARK proof bytes)
# - decision_public.json (public inputs)

# Verify existing proof
python scripts/prove_decisions.py results/artifacts_run_001/ --verify-only
```

## Performance Targets (v0)

| Metric | Target | Actual |
|--------|--------|--------|
| Proof generation | < 100ms | TBD |
| Proof size | < 50 KB | TBD |
| Verification | < 10ms | TBD |
| Trace length | 1 row | ✅ 1 row |

## Test Vectors

See `SPEC.md` for complete test vectors covering:
1. Normal operation (no quarantine)
2. Enter quarantine (k consecutive violations)
3. Warm-up override (violations ignored during warm-up)
4. Release from quarantine (m consecutive clears)

## Development

### Running Tests

```bash
# Rust unit tests
cd verif/air
cargo test

# Python integration tests
cd verif
python -m pytest tests/

# Benchmarks
cd verif/air
cargo bench
```

### Debugging

Enable detailed logging:

```bash
export RUST_LOG=vsv_stark=debug
python scripts/prove_decisions.py results/artifacts_run_001/
```

### Adding New Constraints

1. Update `verif/air/src/air.rs` with new constraint logic
2. Update `build_trace()` to compute new columns
3. Add corresponding fields to `PublicInputs` or `PrivateWitness`
4. Update `SPEC.md` with constraint equations
5. Add test vectors in `tests/`

## Roadmap

### v0 (Current) - Decision Logic Only
- ✅ EMA update
- ✅ Warm-up logic
- ✅ Hysteresis counters
- ✅ Conformal threshold comparison
- ✅ Quarantine output
- ✅ PyO3 wrapper
- ✅ Python integration script

### v1.1 (Planned) - Complete Provenance
- PCA extraction proof
- Cosine similarity proof
- Hybrid score derivation
- SHA-256 preimage for gradient
- Bind x_t to H_grad commitment

### v1.2 (Planned) - Batch Proofs
- Multi-round batching (10 rounds per proof)
- Amortized cost < 10ms per round

### v2.0 (Vision) - On-Chain Verification
- Recursive STARK composition
- Groth16 wrapper for EVM verification
- Gas cost < 100K per proof

## FAQ

**Q: Why not prove PCA/cosine in v0?**

A: PCA requires floating-point matrix operations which are complex in fixed-point STARK constraints. Cosine requires division by norms (expensive gadget). v0 focuses on proving the decision logic is correct given the hybrid score, deferring the binding of hybrid score to gradient hash to v1.1.

**Q: What's the security level?**

A: Winterfell provides ~100 bits of security with default parameters (32 queries, 8 blowup factor). This is sufficient for research purposes. Production deployments would use 128+ bit security.

**Q: Can I use this for my federated learning system?**

A: Yes! The code is open source (MIT license). However, v0 only proves decision logic, not the full gradient→decision pipeline. Wait for v1.1 for production use, or adapt the code to your threat model.

**Q: How does this compare to other verifiable FL systems?**

A: Most verifiable FL systems use zero-knowledge proofs (SNARKs/STARKs) to prove gradient computation is correct. VSV-STARK is unique in proving the *detection logic* is correct, ensuring Byzantine detectors can't be gamed by adversaries who know the detection algorithm.

## References

1. **Winterfell**: https://github.com/facebook/winterfell
2. **STARK 101**: https://starkware.co/stark-101/
3. **PoGQ Paper**: Zero-TrustML (USENIX Security 2025, under review)
4. **Fixed-Point Arithmetic in ZK**: https://eprint.iacr.org/2020/1586

## License

MIT License - See LICENSE file for details.

## Contact

- **GitHub**: https://github.com/Luminous-Dynamics/Zero-TrustML
- **Email**: research@luminousdynamics.org
- **Issues**: https://github.com/Luminous-Dynamics/Zero-TrustML/issues

---

*"Making Byzantine detection verifiable, one proof at a time."*
