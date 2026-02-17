# VSV-STARK v0 Specification

**Purpose**: Prove PoGQ-v4.1's per-round decision logic for one client/round using STARK

**Scope (v0)**:
- ✅ EMA update
- ✅ Warm-up logic
- ✅ Hysteresis counters
- ✅ Conformal threshold comparison
- ✅ Quarantine output
- ❌ PCA extraction (defer to v1.1)
- ❌ Full training loop (defer to v1.1)
- ❌ Multi-round batch (v0 proves single round)

---

## Fixed-Point Arithmetic

**Scale Factor**: `S = 2^16 = 65536`

All fractional values are represented as `value_fp = floor(value * S)`.

**Example conversions**:
```
0.85 → 55705  (β for EMA)
0.90 → 58982  (conformal threshold)
0.75 → 49152  (previous EMA state)
1.00 → 65536  (egregious cap)
```

**Arithmetic operations**:
- Addition: `(a_fp + b_fp)` (stays in scale)
- Multiplication: `(a_fp * b_fp) / S` (rescale after multiply)
- Division: `(a_fp * S) / b_fp` (pre-scale dividend)
- Comparison: Direct comparison of fixed-point values

**Bounds**:
- All values assumed ∈ [0, 1] before scaling
- After scaling: ∈ [0, 65536]
- Field size: 128-bit (plenty of headroom)

---

## Public Inputs (JSON Commitments)

```json
{
  "H_calib": "<sha256_hex>",      // Hash of calibration_blob.json
  "H_model": "<sha256_hex>",      // Hash of model weights
  "H_grad": "<sha256_hex>",       // Hash of client gradient

  "params": {
    "beta_fp": 55705,             // EMA smoothing (0.85 * 65536)
    "W": 3,                       // Warm-up rounds
    "k": 2,                       // Hysteresis: violations to quarantine
    "m": 3,                       // Hysteresis: clears to release
    "egregious_cap_fp": 65535     // Cap for hybrid score (0.9999...)
  },

  "threshold_fp": 58982,          // Conformal threshold (0.90 * 65536)

  "prev_state": {
    "ema_prev_fp": 49152,         // Previous EMA (0.75 * 65536)
    "consec_viol_prev": 1,        // Previous violation streak
    "consec_clear_prev": 0,       // Previous clear streak
    "quarantined_prev": 0         // Previous quarantine status (0 or 1)
  },

  "current_round": 5,             // Round number t (for warm-up check)

  "decision": {
    "quarantine_out": 0           // Output decision (0=accept, 1=reject)
  }
}
```

**Hash commitments**:
- `H_calib`: Binds to PCA components, α, margin (not verified in v0)
- `H_model`: Binds to model state before aggregation
- `H_grad`: Binds to client's submitted gradient

---

## Private Witness (Not Revealed)

```json
{
  "x_t_fp": 52428,                // Hybrid score for this round (0.80 * 65536)

  "flags": {
    "in_warmup": 0,               // 1 if t ≤ W, else 0
    "violation_t": 1,             // 1 if x_t < threshold, else 0
    "release_t": 0                // 1 if releasing from quarantine, else 0
  }
}
```

**Derivation (not proven in v0)**:
- `x_t_fp` comes from `hybrid_score(pca_score, cosine_score, egregious_cap)`
- In v1.1, we'll add constraints proving x_t matches H_grad + H_calib

---

## AIR Constraints (Single-Step Trace)

### Trace Layout

Single row with columns:

| Column | Symbol | Description |
|--------|--------|-------------|
| 0 | `t` | Current round number |
| 1 | `W` | Warm-up threshold |
| 2 | `beta_fp` | EMA beta (fixed-point) |
| 3 | `threshold_fp` | Conformal threshold (fixed-point) |
| 4 | `x_t_fp` | Hybrid score this round (witness) |
| 5 | `ema_prev_fp` | Previous EMA state (public) |
| 6 | `ema_t_fp` | Computed EMA this round |
| 7 | `consec_viol_prev` | Previous violation streak |
| 8 | `consec_clear_prev` | Previous clear streak |
| 9 | `quarantined_prev` | Previous quarantine status (0/1) |
| 10 | `b_in_warmup` | Boolean: t ≤ W |
| 11 | `b_violation` | Boolean: x_t < threshold |
| 12 | `b_release` | Boolean: releasing from quarantine |
| 13 | `consec_viol_t` | Updated violation streak |
| 14 | `consec_clear_t` | Updated clear streak |
| 15 | `quarantine_out` | Final quarantine decision (0/1) |

### Constraint Equations

**1. Boolean Constraints**

All boolean columns must be 0 or 1:
```
b_in_warmup * (1 - b_in_warmup) = 0
b_violation * (1 - b_violation) = 0
b_release * (1 - b_release) = 0
quarantined_prev * (1 - quarantined_prev) = 0
quarantine_out * (1 - quarantine_out) = 0
```

**2. Warm-up Logic**

```
b_in_warmup = 1  if t ≤ W
            = 0  otherwise

Constraint: (b_in_warmup = 1) ⟹ (t ≤ W)
            (b_in_warmup = 0) ⟹ (t > W)

// Simplified: b_in_warmup * (W - t + 1) ≥ 0
// and: (1 - b_in_warmup) * (t - W) ≥ 0
```

**3. EMA Update**

```
ema_t_fp = (beta_fp * ema_prev_fp + (S - beta_fp) * x_t_fp) / S

Constraint: S * ema_t_fp = beta_fp * ema_prev_fp + (S - beta_fp) * x_t_fp
```

Where `S = 65536` is embedded as a constant.

**4. Violation Flag**

```
b_violation = 1  if x_t_fp < threshold_fp
            = 0  otherwise

// Comparison gadget (v0 simple version):
// We'll use a difference check:
// If b_violation = 1, then threshold_fp - x_t_fp > 0
// If b_violation = 0, then x_t_fp - threshold_fp ≥ 0

delta_pos = threshold_fp - x_t_fp  (if b_violation = 1)
delta_neg = x_t_fp - threshold_fp  (if b_violation = 0)

Constraint: b_violation * delta_pos + (1 - b_violation) * delta_neg ≥ 0
```

**5. Hysteresis Counter Updates**

```python
# Violation counter
if b_violation:
    consec_viol_t = consec_viol_prev + 1
    consec_clear_t = 0
else:
    consec_viol_t = 0
    consec_clear_t = consec_clear_prev + 1

# Constraints:
consec_viol_t = b_violation * (consec_viol_prev + 1) + (1 - b_violation) * 0
consec_clear_t = (1 - b_violation) * (consec_clear_prev + 1) + b_violation * 0
```

**6. Quarantine Logic**

```python
# Enter quarantine if k consecutive violations
b_enter = 1 if consec_viol_t ≥ k else 0

# Release if m consecutive clears
b_release = 1 if (quarantined_prev = 1 and consec_clear_t ≥ m) else 0

# Final state
if b_in_warmup:
    quarantine_out = 0  # Never quarantine during warm-up
elif b_release:
    quarantine_out = 0  # Releasing
elif b_enter:
    quarantine_out = 1  # Entering quarantine
else:
    quarantine_out = quarantined_prev  # Maintain previous state
```

**Constraints**:
```
// Enter condition
b_enter = 1 ⟺ consec_viol_t ≥ k

// Release condition
b_release = 1 ⟺ (quarantined_prev = 1) ∧ (consec_clear_t ≥ m)

// Warm-up override
quarantine_out = b_in_warmup * 0 + (1 - b_in_warmup) * (
    b_release * 0 + (1 - b_release) * (
        b_enter * 1 + (1 - b_enter) * quarantined_prev
    )
)
```

---

## Proof Generation Flow

```python
from verif.python import generate_round_proof, verify_round_proof
import json

# 1. Load public inputs from artifact
public_json = {
    "H_calib": artifacts["calibration_blob.json"].sha256,
    "H_model": artifacts["model_weights.npz"].sha256,
    "H_grad": artifacts["client_001_gradient.npz"].sha256,
    "params": {"beta_fp": 55705, "W": 3, "k": 2, "m": 3, "egregious_cap_fp": 65535},
    "threshold_fp": 58982,
    "prev_state": {"ema_prev_fp": 49152, "consec_viol_prev": 1,
                   "consec_clear_prev": 0, "quarantined_prev": 0},
    "current_round": 5,
    "decision": {"quarantine_out": 0}
}

# 2. Compute witness (from detector internals)
witness = {
    "x_t_fp": int(hybrid_score * 65536),
    "flags": {
        "in_warmup": 0,
        "violation_t": 1 if hybrid_score < 0.90 else 0,
        "release_t": 0
    }
}

# 3. Generate STARK proof
proof_bytes, proof_hex = generate_round_proof(public_json, witness)

# 4. Verify
is_valid = verify_round_proof(proof_bytes, json.dumps(public_json))
assert is_valid, "Proof verification failed"

# 5. Save proof
Path("results/artifacts_run_001/decision_proof.bin").write_bytes(proof_bytes)
```

---

## Performance Targets (v0)

- **Proof generation**: < 100ms (single round, simple constraints)
- **Proof size**: < 50 KB (no recursive composition)
- **Verification**: < 10ms (on-chain friendly)
- **Trace length**: 1 row (single-step proof)

---

## Out of Scope for v0

### Deferred to v1.1:
- **PCA extraction**: Proving `pca_score = (grad · components[0]) / norm(components[0])`
- **Cosine similarity**: Proving `cosine_score = grad · median / (norm(grad) * norm(median))`
- **Hybrid score derivation**: Proving `x_t = max(pca_score, cosine_score)` with egregious cap
- **Hash preimage**: Proving `SHA256(grad) = H_grad`
- **Multi-round batching**: Proving 10 rounds in single proof

### Why defer:
- PCA requires floating-point matrix ops (complex in fixed-point)
- Cosine requires division by norms (expensive gadget)
- Hash preimage requires SHA-256 circuit (large)
- v0 focuses on decision logic only, assuming honest x_t witness

**Threat model for v0**:
- Prover can cheat by choosing arbitrary `x_t_fp` (witness)
- But prover CANNOT cheat on EMA/hysteresis/quarantine logic given x_t
- v1.1 closes the loop by binding x_t to H_grad via PCA/cosine proof

---

## Test Vectors

### Test 1: Normal Operation (No Quarantine)

**Public**:
```json
{
  "params": {"beta_fp": 55705, "W": 3, "k": 2, "m": 3},
  "threshold_fp": 58982,
  "prev_state": {"ema_prev_fp": 60000, "consec_viol_prev": 0,
                 "consec_clear_prev": 2, "quarantined_prev": 0},
  "current_round": 5,
  "decision": {"quarantine_out": 0}
}
```

**Witness**:
```json
{
  "x_t_fp": 61000,
  "flags": {"in_warmup": 0, "violation_t": 0, "release_t": 0}
}
```

**Expected**:
- `ema_t_fp = (55705 * 60000 + 9831 * 61000) / 65536 ≈ 60153`
- `b_violation = 0` (61000 ≥ 58982)
- `consec_viol_t = 0`, `consec_clear_t = 3`
- `quarantine_out = 0` (maintain previous state)

### Test 2: Enter Quarantine

**Public**:
```json
{
  "prev_state": {"ema_prev_fp": 50000, "consec_viol_prev": 1,
                 "consec_clear_prev": 0, "quarantined_prev": 0},
  "current_round": 8,
  "decision": {"quarantine_out": 1}
}
```

**Witness**:
```json
{
  "x_t_fp": 50000,
  "flags": {"in_warmup": 0, "violation_t": 1, "release_t": 0}
}
```

**Expected**:
- `b_violation = 1` (50000 < 58982)
- `consec_viol_t = 2` (k=2, triggers quarantine)
- `consec_clear_t = 0`
- `quarantine_out = 1` (enter quarantine)

### Test 3: Warm-up Override

**Public**:
```json
{
  "prev_state": {"ema_prev_fp": 40000, "consec_viol_prev": 5,
                 "consec_clear_prev": 0, "quarantined_prev": 0},
  "current_round": 2,
  "decision": {"quarantine_out": 0}
}
```

**Witness**:
```json
{
  "x_t_fp": 30000,
  "flags": {"in_warmup": 1, "violation_t": 1, "release_t": 0}
}
```

**Expected**:
- `b_in_warmup = 1` (2 ≤ 3)
- `b_violation = 1` (30000 < 58982)
- `consec_viol_t = 6`
- `quarantine_out = 0` (warm-up override)

---

## Version History

- **v0** (Current): Prove decision logic only, assume trusted x_t witness
- **v1.1** (Planned): Add PCA/cosine proofs, bind x_t to H_grad
- **v1.2** (Planned): Multi-round batching (10 rounds per proof)
- **v2.0** (Vision): Recursive composition, on-chain verification
