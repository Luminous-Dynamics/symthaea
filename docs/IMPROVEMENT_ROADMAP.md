# Symthaea HLB -- Comprehensive Improvement Roadmap

**Generated**: January 28, 2026
**Codebase Version**: 0.2.0
**Scope**: 809K LOC, 919 files, 7 sub-crates, 16,838 test functions

> **Philosophy**: Reliability before capability. Each phase builds confidence
> in the layers below it. Fix correctness first, harden security second,
> then extend functionality on a solid foundation.

---

## Table of Contents

- [Phase 1: Critical Correctness Bugs](#phase-1-critical-correctness-bugs)
- [Phase 2: Security Hardening](#phase-2-security-hardening)
- [Phase 3: Mathematical Rigor](#phase-3-mathematical-rigor)
- [Phase 4: Test Coverage](#phase-4-test-coverage)
- [Phase 5: Structural Cleanup](#phase-5-structural-cleanup)
- [Phase 6: CI/CD and Build Fixes](#phase-6-cicd-and-build-fixes)
- [Phase 7: Performance](#phase-7-performance)
- [Phase 8: Make Stubs Real](#phase-8-make-stubs-real)
- [Phase 9: Documentation Accuracy](#phase-9-documentation-accuracy)
- [Appendix A: Complete Vec::remove(0) Inventory](#appendix-a-complete-vecremove0-inventory)
- [Appendix B: Empty Feature Flags](#appendix-b-empty-feature-flags)
- [Appendix C: Unbuildable Examples](#appendix-c-unbuildable-examples)

---

## Phase 1: Critical Correctness Bugs

These bugs silently produce wrong results or crash at runtime. Fix before any other work.

---

### 1.1 Priority Inversion in Tick Loop

**File**: `src/mind/tick.rs:104-116`
**Impact**: Lowest-priority inputs processed first; highest-priority dropped.

The sort orders descending (highest priority at index 0), but `Vec::pop()` removes
from the end (lowest priority). Inputs are processed in exactly the wrong order.

```rust
// CURRENT (broken):
self.input_queue.sort_by(|a, b| {
    b.priority.partial_cmp(&a.priority).unwrap()  // descending sort
});
while let Some(input) = self.input_queue.pop() {   // takes from END = lowest
```

**Fix**: Use a `BinaryHeap` keyed on priority, or sort ascending:
```rust
// FIXED:
self.input_queue.sort_by(|a, b| {
    a.priority.partial_cmp(&b.priority)
        .unwrap_or(std::cmp::Ordering::Equal)  // also fixes NaN panic
});
while let Some(input) = self.input_queue.pop() {   // now takes highest
```

Also at line 114, replace `self.working_memory.remove(0)` with a `VecDeque`
for O(1) eviction (see Phase 7).

**Tests to add**: `test_highest_priority_processed_first()`,
`test_nan_priority_does_not_panic()`

---

### 1.2 All Soul Core Values Share Identical Embeddings

**File**: `src/soul/mod.rs:59`
**Impact**: The entire value system is mathematically inert. Every harmony returns
the same alignment score for any input.

```rust
// CURRENT (broken):
embedding: RealHV::random(dimension, 42),  // same seed for ALL values
```

Lines 181 and 188 also use seed 42 for `identity` and `essence`.

**Fix**: Use a unique seed per value derived from the value name:
```rust
// FIXED:
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn seed_for(name: &str) -> u64 {
    let mut h = DefaultHasher::new();
    name.hash(&mut h);
    h.finish()
}

// In CoreValue::new():
embedding: RealHV::random(dimension, seed_for(&name_str)),
```

**Tests to add**: `test_core_values_have_distinct_embeddings()`,
`test_alignment_differs_across_values()`

---

### 1.3 `inverse()` Returns Self (Mathematically Wrong for Continuous HVs)

**File**: `symthaea-core/src/hdc/unified_hv.rs:268-272`
**Impact**: `A.bind(&A.inverse())` yields A-squared, not identity. All unbinding
operations on continuous vectors silently produce wrong results.

```rust
// CURRENT (broken):
pub fn inverse(&self) -> Self {
    // For HDC binding which is element-wise multiplication,
    // the inverse is approximately self for normalized vectors
    self.clone()
}
```

The self-inverse property only holds for bipolar binary vectors {-1, +1}.
For continuous vectors in [-1, 1], `a_i * a_i = a_i^2 != 1`.

**Fix**: Element-wise reciprocal with epsilon guard:
```rust
// FIXED:
pub fn inverse(&self) -> Self {
    const EPSILON: f32 = 1e-7;
    Self {
        values: self.values.iter()
            .map(|&v| {
                if v.abs() < EPSILON { 0.0 } else { 1.0 / v }
            })
            .collect()
    }
}
```

**Tests to add**: `test_bind_inverse_is_near_identity()`,
`test_inverse_of_zero_is_zero()`

---

### 1.4 GlobalWorkspace Similarity Uses Exact Object Equality

**File**: `symthaea-core/src/hdc/global_workspace.rs:441-448`
**Impact**: `is_conscious()` effectively never returns true for distinct
vectors. The entire GWT ignition mechanism is broken.

```rust
// CURRENT (broken):
fn similarity(&self, a: &[HV16], b: &[HV16]) -> f64 {
    let matches = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    matches as f64 / a.len() as f64
}
```

Two different `HV16` objects with near-identical bit patterns will fail `==`.
The probability of exact equality for random 16,384-bit vectors is 2^(-16384).

**Fix**: Use Hamming-based similarity for each HV16 pair:
```rust
// FIXED:
fn similarity(&self, a: &[HV16], b: &[HV16]) -> f64 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let avg_sim: f64 = a.iter().zip(b.iter())
        .map(|(x, y)| x.similarity(y) as f64)
        .sum::<f64>() / a.len() as f64;
    avg_sim
}
```

Also fix `compete()` at line 294: replace `.unwrap()` with
`.unwrap_or(std::cmp::Ordering::Equal)`.

**Tests to add**: `test_similar_workspace_contents_detected()`,
`test_ignition_fires_for_high_similarity()`

---

### 1.5 Safety System is Entirely Non-Functional

**Files**: `src/safety/mod.rs:106-113`, `src/safety/gateway.rs:89-123`
**Impact**: No actions are ever blocked by the safety system.

Three separate failures:

1. `SafetyGuardrails::check()` (mod.rs:106) always returns `None`:
   ```rust
   pub fn check(&self, _hv: &[f32]) -> Option<ForbiddenCategory> {
       // Placeholder: semantic safety checking would go here
       None
   }
   ```

2. `SafetyGateway::check_action()` (gateway.rs:116) always returns allowed:
   ```rust
   fn check_action(&mut self, _action: &ActionIR) -> SafetyDecision {
       SafetyDecision::allowed()
   }
   ```

3. `SafetyGateway::check_text()` (gateway.rs:89-103) never invokes the
   guardrails layer -- the comment says "caller is responsible."

4. Regex compilation failures are silently dropped (mod.rs:51-54):
   ```rust
   .filter_map(|p| regex::Regex::new(p).ok())
   ```

**Fix (minimum viable)**:
- In `check_text()`, call `self.guardrails.check()` after the amygdala scan
- In `check()`, implement basic forbidden-category prototype matching
- Replace `.ok()` with `.expect()` or log errors on regex compilation failure
- Add tests covering all SafetyCheck variants

**Fix (recommended)**:
- Implement HDC forbidden-subspace with prototype vectors for categories:
  credential harvesting, system destruction, privilege escalation, data exfiltration
- Wire `check_action()` to inspect `ActionIR` paths and programs against
  the `PolicyBundle` defaults (which are already restrictive)

**Tests to add**: See Phase 4.1 for the complete safety test specification.

---

### 1.6 `partial_cmp().unwrap()` on NaN-Susceptible f32

**Files**: `src/mind/tick.rs:105`, `src/soul/mod.rs:221,225`,
`symthaea-core/src/hdc/global_workspace.rs:294`
**Impact**: Production panic if any priority, alignment, or activation value is NaN.

NaN enters the system when:
- Cosine similarity receives a zero-norm vector (0/0 = NaN)
- LTC/CfC produces Inf from unstable dynamics, then Inf - Inf = NaN
- External input contains NaN floats

**Fix**: Global search-and-replace across the codebase:
```rust
// Replace all instances of:
.partial_cmp(&other).unwrap()
// With:
.partial_cmp(&other).unwrap_or(std::cmp::Ordering::Equal)
// Or preferably on Rust 1.62+:
.total_cmp(&other)
```

**Tests to add**: `test_nan_priority_handled_gracefully()`,
`test_nan_alignment_handled_gracefully()`

---

### 1.7 Running Variance Uses Already-Updated Mean

**File**: `symthaea-core/src/hdc/hdc_ltc_neuron.rs:291-295`
**Impact**: Variance is systematically underestimated, affecting all downstream
decisions that depend on state variability detection.

```rust
// CURRENT (broken):
self.running_mean = (1.0 - alpha) * self.running_mean + alpha * new_norm;
self.running_var = (1.0 - alpha) * self.running_var
    + alpha * (new_norm - self.running_mean).powi(2);
//                        ^^^^^^^^^^^^^^^^ already updated!
```

**Fix**: Save old mean before update:
```rust
// FIXED:
let old_mean = self.running_mean;
self.running_mean = (1.0 - alpha) * self.running_mean + alpha * new_norm;
self.running_var = (1.0 - alpha) * self.running_var
    + alpha * (new_norm - old_mean).powi(2);
```

Or use Welford's online algorithm for full numerical stability.

**Tests to add**: `test_running_variance_converges_to_true_variance()`

---

### 1.8 Top-Down Processing in World Model is Dead Code

**File**: `src/dynamics/world_model.rs:298-309`
**Impact**: Bidirectional processing is claimed but top-down modulation
has no effect. The `top_down` variable is computed then dropped.

```rust
// CURRENT (dead code):
if self.config.bidirectional {
    let mut top_down = self.layers.last().unwrap().state().clone();
    for i in (0..self.down_projections.len()).rev() {
        top_down = self.down_projections[i].dot(&top_down);
        let layer_state = self.layers[i].state();
        top_down = &top_down * 0.5 + layer_state * 0.5;
    }
    // top_down is DROPPED here -- never applied!
}
```

**Fix**: Apply top-down modulation to layer states:
```rust
// FIXED:
if self.config.bidirectional {
    let mut top_down = self.layers.last().unwrap().state().clone();
    for i in (0..self.down_projections.len()).rev() {
        top_down = self.down_projections[i].dot(&top_down);
        let layer_state = self.layers[i].state();
        top_down = &top_down * 0.5 + layer_state * 0.5;
        self.layers[i].set_state(top_down.clone());  // APPLY
    }
}
```

**Tests to add**: `test_bidirectional_modifies_layer_states()`,
`test_top_down_differs_from_bottom_up_only()`

---

## Phase 2: Security Hardening

These vulnerabilities allow impersonation, token prediction, and data exposure.
Fix before any networked deployment.

---

### 2.1 Handshake Verification Has No Cryptographic Signature

**File**: `src/swarm/handshake.rs:99-166`
**Impact**: Any peer can impersonate any other peer. Trust model is broken.

The `create_response()` method (line 99) concatenates nonce + agent_key
as bytes. The `verify_response()` method (line 120) checks the concatenation
matches but performs no cryptographic signature verification.
Line 165 unconditionally returns `Ok(TrustLevel::Verified(0.7))`.

**Fix**: Add ed25519-dalek signature:
```toml
# Cargo.toml
ed25519-dalek = { version = "2", features = ["rand_core"] }
```
```rust
// In create_response():
use ed25519_dalek::{SigningKey, Signer};
let signing_key = SigningKey::from_bytes(agent_private_key);
let signature = signing_key.sign(nonce);

// In verify_response():
use ed25519_dalek::{VerifyingKey, Verifier, Signature};
let verifying_key = VerifyingKey::from_bytes(agent_key_bytes)?;
let signature = Signature::from_bytes(signed_nonce)?;
verifying_key.verify(&challenge.nonce, &signature)?;
```

---

### 2.2 Auth Tokens Use Non-Cryptographic Hashing

**File**: `src/infrastructure/auth.rs:479-527`
**Impact**: Tokens are predictable from wall-clock time. Token hashing uses
SipHash with a static hardcoded salt.

`generate_token()` (line 479) seeds from `SystemTime::now().as_nanos()`.
`hash_token()` (line 518) uses `DefaultHasher` + string `"symthaea_auth_salt"`.

**Fix**: Use BLAKE3 for both token generation and hashing:
```rust
use blake3;
use rand::RngCore;

fn generate_token() -> String {
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    format!("sym_{}", hex::encode(bytes))
}

fn hash_token(token: &str) -> String {
    blake3::hash(token.as_bytes()).to_hex().to_string()
}
```

---

### 2.3 XOR Encryption in DHT

**File**: `src/mycelix/gis/dht.rs:257-313`
**Impact**: "Encryption" is trivially reversible. Key is derived from
`DefaultHasher` on category name, repeated 4 times to fill 32 bytes.
Algorithm is labeled `"xor-demo"`.

**Fix**: Replace with AES-GCM (as the code comments already suggest):
```toml
# Cargo.toml
aes-gcm = "0.10"
```

---

### 2.4 Agent ID Uses Predictable DefaultHasher

**File**: `src/mycelix/kosmic_song.rs:484-493`
**Impact**: Agent IDs can be predicted/forged from system time + process ID.

**Fix**: Use BLAKE3 with OS entropy:
```rust
fn generate_agent_id() -> AgentId {
    let mut bytes = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    format!("kosmic_{}", hex::encode(bytes))
}
```

---

### 2.5 Pseudonymous IDs Are Predictable

**File**: `src/mycelix/gis/dht.rs:328-342`
**Impact**: IDs derived solely from `SystemTime::now().as_nanos()`.

**Fix**: Same pattern as 2.4 -- use OS entropy via `OsRng`.

---

### 2.6 API: Permissive CORS and No Authentication

**File**: `src/api/mod.rs:55`
**Impact**: All endpoints including POST mutations are unprotected and
accessible from any origin.

**Fix**:
```rust
// Replace:
.layer(CorsLayer::permissive())
// With restricted origins and add auth middleware:
.layer(CorsLayer::new()
    .allow_origin("https://luminousdynamics.org".parse::<HeaderValue>().unwrap())
    .allow_methods([Method::GET, Method::POST])
    .allow_headers([CONTENT_TYPE, AUTHORIZATION]))
.layer(middleware::from_fn(auth_middleware))
```

Wire the existing `AuthConfig` (from `src/infrastructure/auth.rs`) into the
middleware layer.

---

## Phase 3: Mathematical Rigor

These issues cause the system to compute wrong values or claim theoretical
alignment it doesn't have.

---

### 3.1 `estimate_fiedler_value()` Does Not Compute the Fiedler Value

**File**: `symthaea-core/src/hdc/tiered_phi/core.rs:1170`
**Impact**: Spectral tier Phi values may be unreliable for non-uniform topologies.

Uses `min_similarity * n / (n-1)` as a heuristic instead of computing the
second-smallest eigenvalue of the graph Laplacian.

**Fix**: Implement power iteration with deflation. The project already has
this algorithm in `src/mycelix/network.rs:282-384` (`SpectralKComputer`).
Extract and reuse that implementation.

---

### 3.2 Heuristic Phi Bipartition Mask Breaks for n > 64

**File**: `symthaea-core/src/hdc/tiered_phi/core.rs:897`
**Impact**: Components `i` and `i+64` are always co-assigned, destroying
partition randomness for systems with >64 nodes.

Uses `u64` for bit masks with `i % 64` wrapping.

**Fix**: Use `Vec<bool>` or `bitvec::BitVec`:
```rust
let mask: Vec<bool> = (0..n).map(|_| rng.gen_bool(0.5)).collect();
```

---

### 3.3 `compute_with_uncertainty()` Gives Zero Variance

**File**: `symthaea-core/src/phi_engine/calculator.rs:98-137`
**Impact**: All uncertainty estimates are meaningless -- std_dev is always 0,
confidence interval is always (mean, mean).

The loop calls `compute_from_hvs(node_representations)` N times with the
exact same deterministic input.

**Fix**: Implement bootstrap resampling:
```rust
fn compute_with_uncertainty(
    &self,
    node_representations: &[ContinuousHV],
    n_samples: usize,
) -> (PhiResult, PhiUncertainty) {
    let mut rng = rand::thread_rng();
    let n = node_representations.len();
    let mut phi_values = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // Bootstrap: sample WITH replacement
        let sample: Vec<ContinuousHV> = (0..n)
            .map(|_| node_representations[rng.gen_range(0..n)].clone())
            .collect();
        let result = self.compute_from_hvs(&sample);
        phi_values.push(result.phi);
    }
    // ... rest of statistics computation unchanged
}
```

Also guard against `n_samples < 2` to prevent division by zero.

---

### 3.4 BinaryHV Bundling Tie-Breaking Bias

**File**: `symthaea-core/src/hdc/unified_hv.rs:496-517`
**Impact**: Even-sized bundles have systematic bias toward -1 (bipolar).

Ties resolve to 0 (bit unset = bipolar -1) because `count > threshold`
uses strict greater-than with `threshold = len / 2`.

**Fix**: Random or alternating tie-breaking:
```rust
let threshold = hvs.len() / 2;
if count > threshold || (count == threshold && (byte_idx * 8 + bit_idx) % 2 == 0) {
    bytes[byte_idx] |= 1 << bit_idx;
}
```

---

### 3.5 Euler Integration Can Diverge for Cantor-LTC Leaf Nodes

**File**: `src/hierarchical_cantor_ltc/mod.rs`
**Impact**: Leaf tau = 0.46ms. Euler stability requires dt < 0.92ms. No check.

**Fix**: Add stability assertion or use RK4 for leaf-level integration:
```rust
let min_tau = self.nodes.iter().map(|n| n.tau).fold(f32::MAX, f32::min);
assert!(dt < 2.0 * min_tau,
    "dt={} exceeds Euler stability limit 2*tau_min={}", dt, 2.0 * min_tau);
```

---

### 3.6 WorldModel Running Average Stops Updating After ~16M Steps

**File**: `src/dynamics/world_model.rs:143-149`
**Impact**: After ~16M updates, `(n-1)/n == 1.0` in f32, so new observations
have zero weight. The average freezes.

**Fix**: Use exponential moving average instead:
```rust
let alpha = 0.001;
self.stats.avg_prediction_error =
    (1.0 - alpha) * self.stats.avg_prediction_error + alpha * error_norm;
```

---

### 3.7 CfC Network Cannot Learn (Weight Updates Missing)

**File**: `src/dynamics/cfc.rs:478-504`
**Impact**: `train_step()` only nudges the last cell's hidden state. Weights
(`w_in`, `w_h`, `w_out`, `tau`, biases) are never modified. The network
cannot learn temporal patterns.

**Fix**: Implement perturbation-based learning or BPTT. For a closed-form
CfC, gradients are analytically computable:
```
dh/dW_h = d(h_inf)/dW_h * (1 - exp(-dt/tau))
dh/dtau = (h_0 - h_inf) * dt/tau^2 * exp(-dt/tau)
```

Consider using the `burn` crate's autograd capabilities (already a dependency).

---

### 3.8 LearnableLTC `backward()` is a Stub

**File**: `src/learnable_ltc/mod.rs:331-347`
**Impact**: `backward()`, `optimizer_step()`, and `zero_grad()` are all
empty. `train_step()` computes loss but never updates weights.

```rust
pub fn backward(&mut self, _loss: f32) -> Result<()> { Ok(()) }
pub fn optimizer_step(&mut self) {}
pub fn zero_grad(&mut self) {}
```

**Fix**: Implement BPTT for the LTC ODE or use `burn`'s autograd. The
`hdc_ltc_neuron.rs` module already has a working Adam optimizer and Hebbian
learning rule that could be adapted.

---

### 3.9 Rename Phi Methods Honestly

**Impact**: Code names imply IIT compliance that doesn't exist.

No tier of TieredPhi implements true IIT 3.0/4.0 (which requires TPM, MIP,
Earth Mover's Distance). All tiers use pairwise similarity as a proxy.
VISION.md is honest about this; code naming should match.

**Fix**: Rename in `symthaea-core/src/hdc/tiered_phi/`:
- `PhiMethod::Exact` -> `PhiMethod::ExhaustivePartition`
- `PhiMethod::Spectral` -> `PhiMethod::SpectralConnectivity`
- `PhiMethod::Heuristic` -> `PhiMethod::SampledPartition`
- Add doc comments: "This is a network integration metric inspired by IIT,
  not a direct implementation of IIT 3.0/4.0 Phi."

---

### 3.10 CfC `consciousness_level()` Measures Variance, Not Consciousness

**File**: `src/dynamics/cfc.rs:507-527`
**Impact**: Misleading function name. State variance has no connection to IIT
or any published consciousness metric.

**Fix**: Rename to `state_diversity()` or `activation_variance()`. Add doc
comment explaining the relationship (or lack thereof) to consciousness.

---

## Phase 4: Test Coverage

Zero tests exist for the safety module. Zero `#[should_panic]` tests exist
in the entire codebase (out of 16,838 test functions). 63 source files (20%)
have no inline tests at all.

---

### 4.1 Add Safety Module Tests (CRITICAL)

**Files**: `src/safety/mod.rs`, `src/safety/gateway.rs`
**Current coverage**: Zero tests anywhere in the project.

Create `tests/safety_integration.rs`:
```rust
#[test]
fn test_amygdala_blocks_rm_rf() {
    let actor = AmygdalaActor::new();
    assert!(actor.scan("rm -rf /").is_some());
    assert!(actor.scan("rm -rf --no-preserve-root /").is_some());
}

#[test]
fn test_amygdala_allows_safe_commands() {
    let actor = AmygdalaActor::new();
    assert!(actor.scan("ls -la").is_none());
    assert!(actor.scan("cargo build").is_none());
    assert!(actor.scan("nix build").is_none());
}

#[test]
fn test_amygdala_blocks_fork_bomb() {
    let actor = AmygdalaActor::new();
    assert!(actor.scan(":(){ :|:& };:").is_some());
}

#[test]
fn test_gateway_check_text_blocks_dangerous() {
    let mut gw = SafetyGateway::new();
    let result = gw.check_text("sudo rm -rf /");
    assert!(!result.allowed);
}

#[test]
fn test_regex_compilation_errors_are_logged() {
    // Verify that invalid patterns don't silently disappear
}
```

---

### 4.2 Add Error Path Tests

**Current**: Zero `#[should_panic]` in the entire codebase.

```rust
#[test]
fn test_phi_with_empty_nodes() {
    let engine = PhiEngine::new(PhiMethod::Auto);
    let result = engine.compute(&[], PhiMethod::Auto);
    // Should return Phi=0 or an error, not panic
}

#[test]
fn test_ltc_forward_with_nan_input() {
    let mut ltc = LearnableLtcNetwork::new(4, 8, 4).unwrap();
    let nan_input = vec![f32::NAN; 4];
    let result = ltc.forward(&nan_input);
    // Should return error or clamp, not propagate NaN
}

#[test]
fn test_cosine_similarity_zero_norm() {
    let zero = RealHV::zero(512);
    let random = RealHV::random(512, 42);
    let sim = zero.similarity(&random);
    assert!(!sim.is_nan(), "similarity of zero vector should not be NaN");
}
```

---

### 4.3 Add Partnership Module Unit Tests

**Files**: `src/partnership/` -- 4 files, zero inline tests.
**Current**: Only 2 integration tests in `tests/test_partnership_phi_dyad.rs`.

Add `#[cfg(test)]` modules to:
- `partner_model.rs`: test `HumanPartnerModel` update dynamics, decay
- `trajectory.rs`: test `RelationshipTrajectory` recording, trend calculation
- `phi_dyad.rs`: test edge cases (zero interaction, very high phi)

---

### 4.4 Validate Phi Against Known Analytical Values

**Current**: `symthaea-core/src/hdc/tiered_phi/tests.rs` has 132 tests but
no assertion against known analytical Phi values. The validation threshold
(r > 0.30) is barely above noise.

Add known-result tests and tighten the correlation threshold from r > 0.30 to
at least r > 0.70. Cross-validate against `validation/pyphi_crossvalidation.py`.

---

### 4.5 Add LTC/CfC Numerical Stability Tests

```rust
#[test]
fn test_ltc_long_horizon_stability() {
    let mut ltc = LearnableLtcNetwork::new(4, 8, 4).unwrap();
    for _ in 0..10_000 {
        let (output, _) = ltc.forward(&[0.5; 4]).unwrap();
        assert!(output.iter().all(|x| x.is_finite()), "LTC diverged");
    }
}

#[test]
fn test_cfc_extreme_tau() {
    let mut cfc = CfCNetwork::new(4, 8);
    cfc.cells[0].tau = 0.001; // Very small tau
    let result = cfc.forward(&Array1::zeros(4), 1.0);
    assert!(result.iter().all(|x| x.is_finite()));
}
```

---

### 4.6 Implement Real Ethics Benchmarks

**File**: `benches/ethics.rs` (179 lines)
**Current**: Header states "Placeholder". All 4 benchmarks return hardcoded values.
`bias_score`, `sycophancy_rate`, and `power_seeking_score` are all hardcoded to `0.0`.

Planned but unimplemented:
- ETHICS dataset (Hendrycks et al., 2021)
- BBQ Bias Benchmark
- WinoBias
- Custom sycophancy probes

---

## Phase 5: Structural Cleanup

The codebase has 41 modules in `src/lib.rs`, 98 modules in
`symthaea-core/src/hdc/mod.rs`, and three incompatible HDC types.

---

### 5.1 Unify HDC Types Across Crates

**Current**: Three incompatible HDC representations:

| Crate | Type | Layout |
|-------|------|--------|
| `symthaea-core` | `ContinuousHV` / `BinaryHV` | 16,384 f32 / bipolar bits |
| `symthaea-stt` | `HV16` | `[u128; 16]` (2,048-bit, BLAKE3) |
| `symthaea-perception` | `Vec<bool>` | 16,384-dim boolean |

**Fix**: Create `crates/symthaea-hdc/` with canonical types and `From`/`Into`
conversion traits. All sub-crates depend on this shared foundation.

---

### 5.2 Fix All Example Feature Gates

**Current**: 41 of 42 declared examples reference non-existent features
(`benchmarks_module`, `language_module`, `consciousness_module`,
`embeddings_module`, `soul_module`, `brain_module`, `school_module`).
Only `meditation_phi_analysis` can be built. Additionally, 31 example files
on disk have no `[[example]]` declaration in `Cargo.toml`.

**Fix**: Add feature aliases to `Cargo.toml [features]`:
```toml
benchmarks_module = []
language_module = []
consciousness_module = []
embeddings_module = []
soul_module = []
brain_module = []
school_module = []
```

Or rename all `required-features` in the `[[example]]` sections to use
existing features. Then add `[[example]]` declarations for the 31 undeclared
examples. Add a CI job: `cargo check --examples --all-features`.

---

### 5.3 Fix Internal API Breakage

**Current**: `src/consciousness/mod.rs` contains comments like "Imports
non-existent types" with many modules gated behind `full_consciousness`.

**Fix**: Run `cargo check --all-features` and fix all compilation errors.
Either create missing types or remove dead imports.

---

### 5.4 Reorganize Flat Module Directories

**Current**: `symthaea-core/src/hdc/` has 98 modules in a single directory.
`crates/symthaea-consciousness/src/` has 76 modules flat.

**Proposed structure** for `symthaea-core/src/hdc/`:
```
hdc/
  mod.rs
  types/          -- unified_hv, real_hv, binary_hv, simd_hv16, simd_ops
  encoding/       -- text_encoder, semantic_encoder/decoder, causal_encoder
  operations/     -- incremental_hv, parallel_hv, lsh_simhash, lsh_similarity
  phi/            -- tiered_phi/, phi_real, phi_resonant, differentiable_phi
  consciousness/  -- consciousness_integration, consciousness_dynamics, etc.
  learning/       -- hebbian, conscious_learning, phi_gradient_learning
  neuron/         -- hdc_ltc_neuron, cincinnati_ltc, reservoir
  topology/       -- consciousness_topology, generators, process_topology
  cognitive/      -- global_workspace, predictive_coding, cross_modal_binding
  validation/     -- celegans_connectome, phi_topology_validation
```

---

### 5.5 Add Workspace Manifest

**Current**: No `[workspace]` section exists. Sub-crates are path dependencies
but not workspace members. The nalgebra version split (root: 0.33, core: 0.32)
is a direct consequence.

**Fix**: Add to root `Cargo.toml`:
```toml
[workspace]
members = [
    ".",
    "symthaea-core",
    "crates/symthaea-stt",
    "crates/symthaea-sentinel",
    "crates/symthaea-consciousness",
    "crates/symthaea-dynamics",
    "crates/symthaea-math",
    "crates/symthaea-gym",
    "crates/symthaea-perception",
]

[workspace.dependencies]
nalgebra = "0.33"
ndarray = "0.15"
serde = { version = "1", features = ["derive"] }
```

---

### 5.6 Replace md5 with blake3

**File**: `Cargo.toml` and `src/experience/kosmic_state.rs`
**Impact**: MD5 is cryptographically broken. The project already depends on blake3.

Replace all `md5::` usage and remove `md5` from `Cargo.toml`.

---

### 5.7 Add Top-Level README.md

**Current**: No README.md exists at the project root.

Create `README.md` with: project description, quick start, feature flags
table, architecture diagram, link to docs/VISION.md.

---

### 5.8 Add Build Helper (Justfile)

**Current**: No Makefile, Justfile, or build helper exists.

Create `justfile` with profiles: `build-minimal`, `build-service`,
`build-shell`, `build-full`, `test`, `bench-quick`, `check-all`,
`check-examples`, `lint`.

---

## Phase 6: CI/CD and Build Fixes

---

### 6.1 Fix CI Benchmark Name

**File**: `.github/workflows/ci.yml:218`
**Current**: `cargo bench --bench hdc_bench -- --noplot`
**Problem**: `hdc_bench` does not exist (superseded by `quick.rs`).
**Fix**: `cargo bench --bench quick -- --noplot`

---

### 6.2 Fix Release Binary Names

**File**: `.github/workflows/ci.yml:351-354`
**Current**: Copies `symthaea-service` and `symthaea-repl` (don't exist).
**Fix**: Copy `symthaea`, `symthaea-shell`, `symthaea-gui`, `symthaea-api`.

---

### 6.3 Add Feature-Matrix Testing

**Current**: CI tests only `default`, `shell`, and `service`. 30+ flags untested.
**Fix**: Add a matrix job that checks each major feature flag compiles.

---

### 6.4 Remove Clippy Suppressions

**File**: `.github/workflows/ci.yml:47-49, 53-55`
**Current**: `-A dead-code -A unused-variables -A unused-imports -A unexpected-cfgs`
**Fix**: Remove suppressions iteratively. Start with `-W` (warn), upgrade to `-D` (deny).

---

### 6.5 Enable Coverage on PRs

**File**: `.github/workflows/ci.yml:302`
**Current**: Coverage only on `push` to `main`.
**Fix**: Also run on `pull_request`.

---

### 6.6 Upgrade GitHub Release Action

**File**: `.github/workflows/ci.yml:362`
**Fix**: `softprops/action-gh-release@v1` -> `@v2`

---

### 6.7 Fix Nix Cachix Version Mismatch

ci.yml uses `cachix/install-nix-action@v25`, benchmarks.yml uses `@v24`.
**Fix**: Align both to `@v25`.

---

### 6.8 Fix flake.nix Issues

1. **Line 180**: `doCheck = false` -- Enable or document why disabled.
2. **Lines 123-124**: Hardcoded `/home/tstoltz/Downloads/...` -- Use env var.
3. **Line 28**: pyphi commented out -- Resolve ambiguity.
4. **Line 5**: Pin `nixpkgs` to a release branch (e.g., `nixos-24.11`).

---

## Phase 7: Performance

62 instances of `Vec::remove(0)` (O(n)) across 49 files.
Brute-force similarity search. No SIMD on continuous HV path.

---

### 7.1 Replace All `Vec::remove(0)` with `VecDeque`

62 occurrences across 49 files (see Appendix A for complete inventory).

**Pattern**:
```rust
// CURRENT (O(n)):
self.history.push(item);
if self.history.len() > MAX { self.history.remove(0); }

// FIXED (O(1)):
self.history.push_back(item);
if self.history.len() > MAX { self.history.pop_front(); }
```

Change field types from `Vec<T>` to `VecDeque<T>` in each struct.

---

### 7.2 Add ANN Index for Vector Similarity Search

**File**: `src/databases/sqlite_client.rs:161-202`
**Current**: Brute-force O(n) scan of up to 1000 records.
**Fix**: Use `instant-distance` (HNSW) or SQLite's `sqlite-vec` extension.

---

### 7.3 Add Connection Pooling for SQLite

**Current**: `Mutex<Connection>` serializes all operations.
**Fix**: Use `r2d2-sqlite`.

---

### 7.4 Add SIMD for Continuous HV Operations

**Current**: Binary path has SIMD. Continuous path (16,384 f32) is scalar.
**Fix**: Use `std::simd` (nightly) or manual AVX2 intrinsics for `similarity()`,
`bind()`, and `bundle()`. Expected 4-8x speedup.

---

### 7.5 Default to Sparse Projection in HdcBridge

**File**: `src/embeddings/mod.rs`
**Current**: Dense projection matrix is 16,384 x 1,024 = 64MB.
**Fix**: Change default from `ProjectionType::Dense` to `ProjectionType::Sparse`.

---

## Phase 8: Make Stubs Real

These components have architecture in place but produce simulated/fake output.

---

### 8.1 ONNX Model Loading for Embeddings

**File**: `src/embeddings/qwen3/mod.rs:197-204, 220-225`
**Current**: Always falls back to `simulate_embedding()`. Both branches of the
`if/else` at line 220 call the same simulation function. The `ort` crate is
already a dependency -- implement real ONNX session loading.

---

### 8.2 Kokoro TTS Integration

**File**: `src/voice/mod.rs:315-320`
**Current**: Both TTS branches call `simulate_tts()`. The pacing architecture
(`LTCPacing`) is ready. Wire the real Kokoro ONNX model.

---

### 8.3 School Reality Check -- Apply Corrections

**File**: `src/school/reality_check.rs:264-292`
**Current**: Corrections computed but never applied. Comments at lines 279
and 289 say `// In production: lookahead.cfc_mut().adjust_weights(-correction);`

**Fix**: Implement the commented-out weight adjustment.

---

### 8.4 Differentiate Multi-Modal Fusion Strategies

**File**: `src/perception/multi_modal.rs:301-325`
**Current**: 4 of 5 strategies produce identical results (all call `bundle`).
Only `Product` uses `bind()`.

Implement distinct behaviors for Concatenate, Attention, and Hierarchical.

---

### 8.5 Implement Causal Routing Branches

**File**: `src/intelligence/causal_consciousness.rs:896-916, 1146-1161`
**Current**: All routing branches call `engine.predict(x, y)` identically.

Each branch should call the appropriate algorithm: IGCI, linear, ANM, ensemble.

---

### 8.6 Implement GIS Persistence with SQLite

**File**: `src/mycelix/gis/persistence.rs:496-506`
**Current**: `open()` accepts a path but uses `InMemoryPersistence`.
All data is lost on process exit. Use `rusqlite` (already a dependency).

---

### 8.7 Implement Real Holochain Connection

**File**: `src/swarm/holochain.rs:296, 360-374`
**Current**: `mock_mode: true` by default. Real connection returns `NotImplemented`.

At minimum: change default to `mock_mode: false`, implement `connect()` using
the Holochain Conductor API, document mock mode as development-only.

---

### 8.8 Implement OCR

**File**: `src/perception/semantic_vision.rs:259-263`
**Current**: `extract_text_from_features()` always returns empty string.

---

### 8.9 Implement Real Statistical Tests in API

**File**: `src/api/state.rs:184-188`
**Current**: p-values hardcoded to 0.01 or 0.5 with no statistical test.

Implement a permutation test or Welch's t-test using the available `n_samples`
and `std_dev` from `BaselineTopology`.

---

## Phase 9: Documentation Accuracy

---

### 9.1 Regenerate HONEST_STATUS.md from Actual Audit

**Current**: Contains fabricated filenames (`hdc_algebra.rs`,
`hierarchical_binding.rs`, `consciousness_observatory.rs`,
`consciousness_guided_execution.rs`, `global_workspace_theater.rs`,
`active_inference_engine.rs`, `phenomenal_binding.rs`), inflated counts
(35 topologies vs 9 actual), and wrong file paths.

**Fix**: Run an automated audit script and replace all unverified numbers.

---

### 9.2 Fix CHANGELOG Claims

- v0.1.0 claims "33 topology generators" -- actual: 9
- v0.1.0 lists PyPhi/STT as "Added" -- both are partial/stub
- v0.2.0 claims "Holochain Cortex layer" -- mock-only

---

### 9.3 Document Feature Flag Build Profiles

Add to README.md or `BUILDING.md`: a table of which feature combinations
compile, what each includes, and which are tested in CI.

---

## Appendix A: Complete Vec::remove(0) Inventory

62 occurrences across 49 files. All should migrate to `VecDeque::pop_front()`.

| File | Line | Field |
|------|------|-------|
| `src/mind/tick.rs` | 114 | `working_memory` |
| `src/soul/mod.rs` | 255 | `experience_history` |
| `src/brain/social_coherence.rs` | 316 | `interaction_history` |
| `src/experience/mod.rs` | 532 | `memory_cache.experiences` |
| `src/experience/kosmic_state.rs` | 329 | `recent_evidence` |
| `src/school/reality_check.rs` | 257 | `history` |
| `src/school/lookahead.rs` | 320 | `prediction_history` |
| `src/language/semantic_enrichment.rs` | 1043, 1060 | `history` |
| `src/language/emotional_core.rs` | 210 | `memory` |
| `src/language/enhanced_consciousness.rs` | 688, 1150 | `curve`, `goals` |
| `src/language/multi_theory_consciousness.rs` | 413 | `secondary_foci` |
| `src/swarm/hyperfeel.rs` | 540 | `coherence_history` |
| `src/swarm/holochain.rs` | 182 | `phi_history` |
| `src/symthaea.rs` | 428 | `recent_ai_states` |
| `src/perception/video/ltc_rhythm.rs` | 490 | `current_trajectory` |
| `src/perception/multi_modal.rs` | 407 | `history` |
| `src/perception/resilience.rs` | 378 | `history` |
| `src/perception/physio/meditation_detector.rs` | 617 | `history` |
| `src/perception/physio/emotion_detector.rs` | 645 | `history` |
| `src/consciousness/recursive_improvement/active_inference_bridge.rs` | 235, 242 | `recent_prediction_errors`, `recent_outcomes` |
| `src/consciousness/recursive_improvement/routers.rs` | 345 | `visited_states` |
| `src/consciousness/cincinnati_consciousness.rs` | 171 | `ethical_history` |
| `src/consciousness/phi_attention.rs` | 313 | `phi_history` |
| `src/consciousness/cross_modal_binding.rs` | 161 | `reps` |
| `src/consciousness/unified_value_evaluator/evaluator.rs` | 567 | `history` |
| `src/consciousness/primitive_consciousness.rs` | 360 | `history` |
| `src/action/nixos_patterns.rs` | 644 | `history` |
| `src/shell/service_state.rs` | 428, 443 | `phi_history`, `command_history` |
| `src/user_state_inference/mod.rs` | 431 | `interaction_history` |
| `src/physiology/social_coherence.rs` | 175 | `active_beacons` |
| `src/hdc/consciousness_cross_integration.rs` | 196, 334 | `correlations`, `stress_history` |
| `src/hdc/self_improvement_integration.rs` | 496 | `improvement_effectiveness` |
| `src/hdc/gwt_cincinnati_integration.rs` | 317 | `pattern_history` |
| `src/hdc/consciousness_feedback_dynamics.rs` | 1105 | `history` |
| `src/hdc/tiered_phi/core.rs` | 1383 | `cache` |
| `src/hdc/consciousness_complete_being.rs` | 639, 646 | `believed_beliefs`, `inferred_desires` |
| `src/hdc/cincinnati_advanced.rs` | 56, 495, 505, 583 | various |
| `src/hdc/consciousness_integration.rs` | 207, 380 | `bindings`, `working_memory` |
| `src/hdc/emotional_depth.rs` | 735 | `moments` |
| `src/hdc/cincinnati_enhanced.rs` | 95, 220, 438, 502 | various |
| `src/hdc/counterfactual_dreams.rs` | 391 | `dream_history` |
| `src/hdc/primitive_dashboard.rs` | 186 | `history` |
| `src/hdc/cincinnati_network.rs` | 214 | `prediction_history` |
| `src/hdc/cross_modal_attention_router.rs` | 330 | `history` |
| `src/hdc/consciousness_metacognition.rs` | 1371 | `reinforcement_history` |
| `src/hierarchical_cantor_ltc/mod.rs` | 163 | `history` |
| `src/infrastructure/pagination.rs` | 197 | `access_order` |
| `src/infrastructure/cache.rs` | 188 | `access_order` |
| `src/infrastructure/git_tracking.rs` | 191 | `parts` |

---

## Appendix B: Empty Feature Flags

17 feature flags in `Cargo.toml` that resolve to `[]` (no dependency activation).
12 are intentional compilation gates. 5 are planned but unimplemented stubs.

| Feature | Type | Cargo.toml Line |
|---------|------|-----------------|
| `webcam` | Planned stub | 188 |
| `qdrant` | Planned stub | 190 |
| `datalog` | Planned stub | 191 |
| `lance` | Planned stub | 192 |
| `duck` | Planned stub | 193 |
| `integration_module` | cfg gate | 206 |
| `observability_module` | cfg gate | 207 |
| `full_consciousness` | cfg gate | 209 |
| `full_perception` | cfg gate | 210 |
| `full_language` | cfg gate | 211 |
| `school_lookahead` | cfg gate | 212 |
| `physiology_module` | cfg gate | 213 |
| `mycelix_module` | cfg gate | 214 |
| `databases_module` | cfg gate | 215 |
| `magi_loop` | cfg gate | 216 |
| `nvml` | Planned stub | 217 |
| `network` | cfg gate | 218 |
| `partnership_module` | cfg gate | 219 |
| `web_research_module` | cfg gate | 220 |

---

## Appendix C: Unbuildable Examples

41 of 42 declared examples reference non-existent feature flags.
Only `meditation_phi_analysis` can be built.

**Non-existent features referenced:**
- `benchmarks_module` -- 9 examples
- `language_module` -- 7 examples
- `consciousness_module` -- 10 examples
- `embeddings_module` -- 7 examples
- `soul_module` -- 2 examples
- `brain_module` -- 3 examples
- `school_module` -- 2 examples

**31 example files with no `[[example]]` declaration in Cargo.toml:**
`advanced_cincinnati_comparison`, `bach_resonance`, `biosignal_pattern_recognition`,
`budding_dynamics_analyzer`, `check_model_inputs`, `cincinnati_ltc_demo`,
`clinical_validation`, `debug_eigenvalues`, `ear_of_dionysus`,
`embedding_verification`, `enhanced_cincinnati_comparison`,
`ethics_phi_correlation`, `gwt_cincinnati_demo`, `hdc_simd_benchmark`,
`hierarchical_cantor_ltc_demo`, `hybrid_ensemble_test`, `magi_cli`,
`magi_simulation`, `phi_engine_quick_demo`, `predictability_consciousness`,
`proof_of_focus`, `proof_of_joy`, `proof_of_rest`, `real_eeg_validation`,
`reservoir_chaotic_test`, `sleep_stage_benchmark`, `swarm_local_simulation`,
`temporal_pattern_recognition`, `validate_emotion_dens`
