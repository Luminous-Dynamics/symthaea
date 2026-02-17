# Winterfell Migration Plan for VSV-STARK
**Date**: November 9, 2025
**Goal**: Achieve 3-10× proving speedup with custom AIR implementation

## Executive Summary

We're migrating from RISC Zero's general-purpose zkVM (46.6s proving) to a specialized Winterfell AIR (target: 5-15s) for PoGQ Byzantine detection. This provides the same cryptographic guarantees with dramatically better performance.

**Current RISC Zero Performance** (REAL benchmarks):
- Prove time: 46.6s ± 874ms
- Verify time: 92ms
- Proof size: 221KB
- Technology: General-purpose RISC-V zkVM

**Target Winterfell Performance**:
- Prove time: 5-15s (3-10× speedup)
- Verify time: <50ms
- Proof size: ~200KB (similar)
- Technology: Custom AIR for PoGQ state machine

## Why Winterfell?

### Problem with RISC Zero zkVM
- **General-purpose overhead**: Proves RISC-V instruction execution
- **Large trace**: Every CPU cycle becomes a constraint
- **Slow for simple logic**: PoGQ is ~150 lines, but zkVM proves thousands of instructions

### Winterfell AIR Advantages
- **Domain-specific**: Only proves PoGQ state transitions
- **Minimal trace**: 8 columns × trace_length (power of 2)
- **Direct constraints**: EMA update, hysteresis, conformal check = 5 polynomial constraints
- **Same security**: STARKs with equivalent cryptographic properties

## Implementation Status

### ✅ Completed (2 hours)

1. **Crate Structure**
   - Created `vsv-stark/winterfell-pogq/` workspace member
   - Dependencies: winterfell = "0.10", winter-math, winter-crypto
   - Binary: `winterfell-prover` CLI matching RISC Zero interface

2. **AIR Definition** (`src/air.rs`)
   - 8-column execution trace:
     ```
     [ema_t, viol_t, clear_t, quar_t, x_t, threshold, beta, round]
     ```
   - 5 transition constraints:
     - EMA update: `ema[t+1] = (β * ema[t] + (S - β) * x[t]) / S`
     - Violation counter
     - Clear counter
     - Quarantine state (hysteresis + warm-up)
     - Round increment
   - Boundary constraints: Initial state + final output assertion

3. **Trace Builder** (`src/trace.rs`)
   - Reuses `vsv_core::pogq::compute_decision` for correctness
   - Fills 8-column matrix row-by-row
   - Power-of-2 trace length requirement

4. **Prover Implementation** (`src/prover.rs`)
   - `PoGQProver::prove()`: Generate STARK proof
   - Returns timing breakdown + proof bytes
   - Compatible interface with RISC Zero

5. **Integration Tests** (`src/tests.rs`)
   - Mirrors 7 boundary tests from zkVM implementation
   - Tests: normal operation, enter quarantine, warm-up override, release, threshold boundaries, hysteresis edges

6. **Dual-Backend Benchmark** (`experiments/vsv_dual_backend_benchmark.py`)
   - Compares RISC Zero vs Winterfell
   - Same inputs to both backends
   - Generates Table VII-bis for paper

### 🚧 In Progress

7. **Build System Integration**
   - Added to workspace: `members = [..., "winterfell-pogq"]`
   - Issue: Permission conflicts in `target/` directory (likely due to other cargo processes)
   - Workaround: Build separately or wait for other processes

8. **Q16.16 Fixed-Point Conversion**
   - AIR uses BaseElement (128-bit field)
   - Need to verify Q16.16 → field → Q16.16 roundtrip
   - Current: Direct casting (may need refinement)

### ⏳ Pending

9. **Proof Verification** (de-prioritized)
   - Winterfell v0.10 doesn't expose `StarkProof::from_bytes`
   - Options:
     a. Upgrade to v0.13 (better serialization API)
     b. Custom deserialization
     c. Use v0.10's internal verification (happens during prove)
   - Decision: Focus on proving performance first

10. **Real-World Testing**
    - Generate proofs with actual PoGQ states from experiments
    - Compare proof sizes
    - Benchmark on same hardware as RISC Zero

11. **Paper Integration**
    - Add Table VII-bis: "zkVM vs AIR Comparison"
    - Section III.F: Discuss dual-backend strategy
    - Rationale: Winterfell for high-frequency, RISC Zero for complex logic

## Technical Design

### Execution Trace Example (trace_length=4)

```
Row | ema_t  | viol_t | clear_t | quar_t | x_t   | threshold | beta  | round
----|--------|--------|---------|--------|-------|-----------|-------|------
0   | 55705  | 0      | 0       | 0      | 60000 | 58982     | 55705 | 0
1   | 55800  | 0      | 1       | 0      | 61000 | 58982     | 55705 | 1
2   | 55900  | 0      | 2       | 0      | 62000 | 58982     | 55705 | 2
3   | 56000  | 0      | 3       | 0      | 63000 | 58982     | 55705 | 3
```

**Public Inputs**: β=55705, w=3, k=2, m=3, threshold=58982, ema[0]=55705, quar[3]=0 (expected)
**Private Witness**: x[0..3] = [60000, 61000, 62000, 63000]

### Constraint Evaluation (row t → t+1)

```rust
// Constraint 0: EMA update
ema[t+1] == (beta * ema[t] + (65536 - beta) * x[t]) / 65536

// Constraint 1: Violation counter
viol[t+1] == (x[t] < threshold) ? viol[t] + 1 : 0

// Constraint 2: Clear counter
clear[t+1] == (x[t] >= threshold) ? clear[t] + 1 : 0

// Constraint 3: Quarantine state
quar[t+1] == hysteresis_logic(quar[t], viol[t+1], clear[t+1], round[t+1], w, k, m)

// Constraint 4: Round increment
round[t+1] == round[t] + 1
```

### Security Properties

Both RISC Zero and Winterfell provide:
- **Computational soundness**: ~100 bits (configurable)
- **Zero-knowledge**: Optional (can disable for Byzantine detection)
- **Post-quantum**: Based on collision-resistant hash functions
- **Verifiable correctness**: Proof attests decision matches public params

**Scope of Guarantee** (CRITICAL for paper):
- ✅ Attests: "Given β, w, k, m, threshold, ema_prev, the decision `quar_out` is correct"
- ❌ Does NOT attest: Training data correctness, model quality, gradient truthfulness

## Build & Test Plan

### Phase 1: Local Testing (2-4 hours)
```bash
# Fix permission issues (kill other cargo processes or use separate target)
cd vsv-stark
cargo build --release -p winterfell-pogq

# Run unit tests
cargo test -p winterfell-pogq

# Run integration tests (7 boundary cases)
cargo test -p winterfell-pogq --test tests
```

### Phase 2: Benchmark Comparison (1-2 hours)
```bash
# Build both backends
cd vsv-stark && cargo build --release  # RISC Zero
cargo build --release -p winterfell-pogq

# Run dual-backend benchmark
cd ..
python experiments/vsv_dual_backend_benchmark.py --rounds 5 --output results/dual_backend_comparison.json

# Expected output:
# RISC Zero: 46.6s ± 874ms
# Winterfell: 5-15s (target)
# Speedup: 3-10x
```

### Phase 3: Paper Integration (1 hour)
- Generate Table VII-bis
- Update Section III.F with dual-backend discussion
- Add performance comparison figure

## Risks & Mitigations

### Risk 1: Winterfell may be slower than expected
**Likelihood**: Low (AIR overhead << zkVM overhead)
**Impact**: Medium (still need fast proving for 5-node demo)
**Mitigation**:
- If <3× speedup, investigate constraint degree
- Consider hardware acceleration (GPU prover in v0.13)
- Fall back to RISC Zero for paper, optimize later

### Risk 2: Q16.16 arithmetic may not match exactly
**Likelihood**: Medium (field arithmetic vs integer arithmetic)
**Impact**: High (proof would be invalid)
**Mitigation**:
- Extensive cross-validation with vsv_core reference
- Fuzzing with proptest (property-based testing)
- Formal verification of constraint equations

### Risk 3: Permission issues blocking build
**Likelihood**: High (currently happening)
**Impact**: Low (workaround exists)
**Mitigation**:
- Kill other cargo processes: `pkill -9 cargo` or `pkill -9 rust-analyzer`
- Build in separate workspace
- Use `cargo build --target-dir /tmp/winterfell-target`

## Performance Breakdown Estimate

### RISC Zero zkVM (current):
```
Trace building:     2-3s  (RISC-V execution)
Commitment phase:   10-15s (Merkle trees for large trace)
FRI proving:        20-25s (Polynomial commitments)
Serialization:      1-2s
-----------------------------------------
Total:              46.6s
```

### Winterfell AIR (target):
```
Trace building:     <100ms (8 cols × 256 rows = 2048 field elements)
Commitment phase:   2-4s   (Much smaller trace)
FRI proving:        2-8s   (Fewer constraints)
Serialization:      <100ms
-----------------------------------------
Total:              5-15s (optimistic: 3-5s, conservative: 10-15s)
```

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Implementation | 2h (DONE) | Winterfell crate skeleton |
| Build & Debug | 2-4h | Working binary |
| Benchmarking | 1-2h | Dual-backend comparison |
| Paper Update | 1h | Table VII-bis + Section III.F |
| **Total** | **6-9h** | Production-ready alternative backend |

## Success Criteria

### Minimum Viable (must have):
- [ ] Winterfell binary compiles
- [ ] Generates valid proofs (any performance)
- [ ] 7 boundary tests pass
- [ ] Proof size < 500KB

### Target (should have):
- [ ] Proving time: 5-15s (3-10× speedup)
- [ ] Verification: <50ms
- [ ] Proof size: ~200KB (similar to RISC Zero)
- [ ] Table VII-bis in paper

### Stretch (nice to have):
- [ ] Sub-5s proving (10× speedup)
- [ ] GPU acceleration (v0.13 upgrade)
- [ ] Formal verification of constraints
- [ ] Benchmark on 5-node demo

## Next Actions (Priority Order)

1. **Fix build issues** (30 min - 1 hour)
   - Kill conflicting processes: `pkill cargo` or `pkill rust-analyzer`
   - OR: Build with isolated target: `cargo build --target-dir /tmp/wf-target`
   - Verify binary exists: `file vsv-stark/target/release/winterfell-prover`

2. **Run unit tests** (15 min)
   - `cargo test -p winterfell-pogq`
   - Verify 7 boundary tests pass
   - Fix any Q16.16 conversion issues

3. **Benchmark comparison** (1-2 hours)
   - `python experiments/vsv_dual_backend_benchmark.py --rounds 5`
   - Generate dual-backend comparison JSON
   - Create Table VII-bis LaTeX

4. **Paper integration** (1 hour)
   - Add Section III.F subsection on dual-backend strategy
   - Include Table VII-bis
   - Discuss when to use zkVM vs AIR

## References

- Winterfell Docs: https://github.com/facebook/winterfell
- STARK Paper: https://eprint.iacr.org/2018/046
- RISC Zero Benchmarks: `results/vsv_stark_benchmarks_REAL.json`
- PoGQ Spec: `POGQ_V4_ENHANCED_SPEC.md`

---

**Status**: Crate implemented, blocked on build system permissions
**Next Step**: Resolve cargo build conflicts and run first benchmark
**ETA to completion**: 6-9 hours (blocked issues excluded)
