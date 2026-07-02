# Rust/Python Architecture Plan for 0TML
**Date**: November 9, 2025
**Based on**: User architectural guidance

## Executive Summary

**Core Principle**: Rust for deterministic/verifiable logic, Python for experiments/analysis.

**Current Status**:
- ✅ Winterfell AIR implementation complete (~1800 LOC)
- ✅ vsv-core with PoGQ reference implementation
- ✅ RISC Zero zkVM backend working (46.6s proving)
- 🚧 Dual-backend needs Rust CLI integration
- ⏳ Python bindings (PyO3) needed for seamless interop

**Target**: Framework-agnostic FL with any PyTorch/TensorFlow/JAX model, Rust-verified PoGQ decisions.

---

## Language Responsibility Matrix

### ✅ Rust (Now - Production Critical)

| Component | Crate | Status | Priority |
|-----------|-------|--------|----------|
| **VSV-STARK Proving** | `winterfell-pogq` | 🚧 Build blocked | P0 |
| | `vsv-stark/host` | ✅ Working | P0 |
| **PoGQ State Machine** | `vsv-core` | ✅ Reference impl | P0 |
| **Fixed-Point Math** | `vsv-core::fixedpoint` | ✅ Q16.16 | P0 |
| **Dual-Backend Selector** | `vsv-prover` (new) | ⏳ Not started | P1 |
| **Provenance/Attestation** | `provenance` (new) | ⏳ Not started | P1 |
| **Holochain Zomes** | `holochain/zomes/*` | ✅ Scaffolds | P2 |

### ✅ Rust (Soon - Production Hardening)

| Component | Crate | Status | Priority |
|-----------|-------|--------|----------|
| **Sybil Weighting** | `sybil-defense` (new) | ⏳ Not started | P2 |
| **Network Services** | `vsv-daemon` (new) | ⏳ Not started | P3 |
| **WASM Verifier** | `vsv-verifier-wasm` (new) | ⏳ Not started | P2 |
| **Static CLI Verifier** | `vsv-cli` (new) | ⏳ Not started | P3 |

### ✅ Python (Keep - Research/Orchestration)

| Component | Location | Status | Priority |
|-----------|----------|--------|----------|
| **Experiment Orchestration** | `experiments/*.py` | ✅ Working | P0 |
| **Dataset Prep** | `experiments/prepare_*.py` | ✅ Working | P1 |
| **Analysis/Plotting** | `experiments/analyze_*.py` | ✅ Working | P1 |
| **LaTeX Generation** | `experiments/*_table_*.py` | ✅ Working | P1 |
| **Training Loops** | `src/training/*.py` | ✅ PyTorch | P0 |
| **Model Implementations** | `src/models/*.py` | ✅ Framework-native | P0 |

### 🔗 Interop (Critical - Boundary Layer)

| Component | Technology | Status | Priority |
|-----------|-----------|--------|----------|
| **Python → Rust PoGQ** | PyO3 / maturin | ⏳ Not started | P1 |
| **Python → Rust Verifier** | PyO3 / maturin | ⏳ Not started | P1 |
| **gRPC Service** | tonic + pyo3 | ⏳ Not started | P3 |
| **ONNX Model Exchange** | onnxruntime | ⏳ Not started | P2 |

---

## Phase 1: Stabilize Core Rust Crates (Week 3-4)

### 1.1 Complete Winterfell Build (P0 - 4 hours)

**Goal**: Get `winterfell-pogq` binary built and benchmarked.

```bash
# Fix build permissions
cd vsv-stark
cargo build --release -p winterfell-pogq --target-dir /tmp/wf-target

# Run integration tests
cargo test -p winterfell-pogq --target-dir /tmp/wf-target

# Benchmark vs RISC Zero
python experiments/vsv_dual_backend_benchmark.py --rounds 5
```

**Deliverable**: Table VII-bis with 3-10× speedup confirmation

### 1.2 Dual-Backend Rust CLI (P1 - 6 hours)

**Goal**: Single Rust binary that dispatches to either backend.

**New Crate**: `vsv-stark/vsv-prover/`

```rust
// vsv-prover/src/main.rs
use clap::{Parser, ValueEnum};

#[derive(ValueEnum, Clone)]
enum Backend {
    RiscZero,
    Winterfell,
}

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "winterfell")]
    backend: Backend,

    #[arg(long)]
    public: PathBuf,

    #[arg(long)]
    witness: PathBuf,

    #[arg(long)]
    output: PathBuf,
}

fn main() {
    match args.backend {
        Backend::Winterfell => winterfell_pogq::prove(...),
        Backend::RiscZero => risc_zero_host::prove(...),
    }
}
```

**Python Integration**:
```python
# experiments/vsv_stark_benchmark_integrated.py
result = subprocess.run([
    "vsv-prover",
    "--backend", "winterfell",  # or "risczero"
    "--public", public_json,
    "--witness", witness_json,
    "--output", proof_bin,
])
```

**Deliverable**: One binary, two backends, clean switch via `--backend` flag.

### 1.3 vsv-core Hardening (P0 - 4 hours)

**Goal**: Make `vsv-core` the single source of truth for all backends.

**Changes**:
1. **Add `no_std` support** (for WASM target)
   ```rust
   #![no_std]
   #[cfg(feature = "std")]
   extern crate std;
   ```

2. **Deterministic API guarantees**
   ```rust
   /// Compute PoGQ decision with deterministic Q16.16 arithmetic.
   ///
   /// # Determinism Guarantee
   /// Same inputs → same output across ALL platforms (x86, ARM, WASM).
   /// No floating-point, no platform-specific intrinsics.
   pub fn compute_decision(
       public: &PublicInputs,
       witness: &PrivateWitness,
   ) -> Result<DecisionOutput, Error>
   ```

3. **Property tests** (proptest)
   ```rust
   proptest! {
       #[test]
       fn ema_bounded(beta in 0..65536u32, ema in 0..65536u32, x in 0..65536u32) {
           let output = compute_ema(beta, ema, x);
           prop_assert!(output.to_raw() >= 0);
           prop_assert!(output.to_raw() <= 65536);
       }
   }
   ```

4. **Cross-backend validation**
   ```rust
   #[test]
   fn winterfell_matches_zkvm() {
       let inputs = test_vectors::load("tests/boundary_cases.json");
       for case in inputs {
           let zkvm_output = risc_zero_compute(case);
           let air_output = winterfell_compute(case);
           assert_eq!(zkvm_output, air_output);
       }
   }
   ```

**Deliverable**: Auditable, deterministic, no_std core library.

---

## Phase 2: Python Bindings (Week 4)

### 2.1 PyO3 Wrapper for PoGQ (P1 - 8 hours)

**Goal**: Call Rust PoGQ from Python with zero-copy efficiency.

**New Crate**: `vsv-core-py/`

```rust
// vsv-core-py/src/lib.rs
use pyo3::prelude::*;
use vsv_core::{pogq, fixedpoint::Fixed};

#[pyclass]
struct PublicInputs {
    #[pyo3(get, set)]
    beta: f64,
    #[pyo3(get, set)]
    w: u64,
    // ... etc
}

#[pyfunction]
fn compute_decision(public: &PublicInputs, witness: &PrivateWitness) -> PyResult<DecisionOutput> {
    let public_rust = public.to_rust();
    let witness_rust = witness.to_rust();

    let output = pogq::compute_decision(&public_rust, &witness_rust)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(DecisionOutput::from_rust(output))
}

#[pymodule]
fn vsv_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_decision, m)?)?;
    Ok(())
}
```

**Build with maturin**:
```toml
# vsv-core-py/pyproject.toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "vsv-core"
requires-python = ">=3.11"
```

**Python Usage**:
```python
from vsv_core import PublicInputs, PrivateWitness, compute_decision

public = PublicInputs(
    beta=0.85,
    w=10,
    k=3,
    m=5,
    threshold=0.10,
    # ...
)
witness = PrivateWitness(x_t=0.92)

output = compute_decision(public, witness)
print(f"Quarantine: {output.quarantine_out}")
```

**Deliverable**: `pip install vsv-core` → call Rust from Python notebooks

### 2.2 PyO3 Wrapper for Verifier (P1 - 6 hours)

**Goal**: Verify proofs from Python analysis scripts.

**New Crate**: `vsv-verifier-py/`

```rust
#[pyfunction]
fn verify_proof(proof_bytes: &[u8], public_json: &str, backend: &str) -> PyResult<bool> {
    let public = serde_json::from_str(public_json)?;

    let result = match backend {
        "winterfell" => winterfell_pogq::verify(proof_bytes, public),
        "risczero" => risc_zero_host::verify(proof_bytes, public),
        _ => return Err(PyValueError::new_err("Unknown backend")),
    }?;

    Ok(result)
}
```

**Python Usage**:
```python
from vsv_verifier import verify_proof

with open("proof.bin", "rb") as f:
    proof_bytes = f.read()

valid = verify_proof(
    proof_bytes,
    public_json='{"beta": 0.85, ...}',
    backend="winterfell"
)

assert valid, "Proof verification failed!"
```

**Deliverable**: `pip install vsv-verifier` → validate proofs in analysis

---

## Phase 3: Framework-Agnostic FL (Week 5)

### 3.1 Standardized Boundary Interface

**Goal**: Any framework can participate in federation.

**Contract** (Rust-defined, language-agnostic):
```rust
// vsv-core/src/fl_boundary.rs

/// FL round submission (framework-agnostic)
#[derive(Serialize, Deserialize)]
pub struct RoundSubmission {
    pub round: u64,
    pub agent_id: [u8; 32],

    // Model update (framework-specific format)
    pub model_delta: Vec<u8>,  // Serialized weights (e.g., safetensors, ONNX)

    // Metrics (standardized)
    pub train_loss: f32,
    pub train_accuracy: f32,
    pub val_loss: f32,
    pub val_accuracy: f32,
    pub sample_count: u64,

    // PoGQ inputs (computed by framework adapter)
    pub hybrid_score: f32,  // x_t (0.0 to 1.0)
    pub direction_cosine: f32,  // For direction prefilter

    // Attestation
    pub signature: [u8; 64],  // Ed25519 over canonical JSON
}
```

**Framework Adapters** (Python):
```python
# src/adapters/pytorch_adapter.py
class PyTorchAdapter:
    def prepare_submission(self, model, dataset, round_num) -> dict:
        # Train model
        train_loss, train_acc = self.train_epoch(model, dataset)

        # Validate
        val_loss, val_acc = self.validate(model, dataset.val)

        # Extract delta (vs global model)
        delta = self.compute_delta(model, self.global_model)

        # Compute hybrid score (TPR + FPR from validation)
        tpr, fpr = self.compute_metrics(model, dataset.val)
        hybrid_score = tpr - fpr  # Simplified

        # Serialize delta (safetensors format)
        delta_bytes = safetensors.torch.save(delta)

        return {
            "round": round_num,
            "agent_id": self.agent_id,
            "model_delta": delta_bytes,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "sample_count": len(dataset.train),
            "hybrid_score": hybrid_score,
            "direction_cosine": self.compute_direction(delta),
            "signature": self.sign(canonical_json),
        }

# src/adapters/tensorflow_adapter.py
class TensorFlowAdapter:
    # Same interface, TensorFlow implementation
    pass

# src/adapters/jax_adapter.py
class JAXAdapter:
    # Same interface, JAX implementation
    pass
```

**Rust Coordinator** (calls into vsv-core):
```rust
// vsv-coordinator/src/aggregator.rs
pub fn process_round(submissions: Vec<RoundSubmission>) -> AggregationResult {
    let mut decisions = Vec::new();

    for sub in submissions {
        // Validate signature
        verify_signature(&sub)?;

        // Run PoGQ decision
        let public = PublicInputs::from_submission(&sub);
        let witness = PrivateWitness { x_t_fp: Fixed::from_f32(sub.hybrid_score) };
        let decision = pogq::compute_decision(&public, &witness)?;

        decisions.push((sub.agent_id, decision));
    }

    // Aggregate (only non-quarantined agents)
    let honest_agents: Vec<_> = decisions.iter()
        .filter(|(_, d)| d.quarantine_out == 0)
        .collect();

    let global_delta = aggregate_deltas(&honest_agents);

    AggregationResult {
        global_delta,
        decisions,
        quarantined_count: decisions.len() - honest_agents.len(),
    }
}
```

**Deliverable**: PyTorch/TensorFlow/JAX all use same Rust verifier

---

## Phase 4: Production Hardening (Week 6+)

### 4.1 WASM Verifier (P2 - 8 hours)

**Goal**: Browser/lightweight nodes can verify proofs.

**New Crate**: `vsv-verifier-wasm/`

```rust
// vsv-verifier-wasm/src/lib.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn verify_proof_wasm(
    proof_bytes: &[u8],
    public_json: &str,
    backend: &str,
) -> Result<bool, JsValue> {
    // Same logic as PyO3 verifier
    let public = serde_json::from_str(public_json)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    match backend {
        "winterfell" => winterfell_pogq::verify(proof_bytes, public),
        _ => Err(JsValue::from_str("Unsupported backend in WASM")),
    }
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}
```

**JavaScript Usage**:
```javascript
import init, { verify_proof_wasm } from './vsv_verifier_wasm.js';

await init();

const proofBytes = await fetch('proof.bin').then(r => r.arrayBuffer());
const publicJson = JSON.stringify({ beta: 0.85, ... });

const valid = verify_proof_wasm(
    new Uint8Array(proofBytes),
    publicJson,
    "winterfell"
);

console.log("Proof valid:", valid);
```

**Deliverable**: In-browser proof verification for dashboards

### 4.2 Provenance & Attestation (P1 - 12 hours)

**Goal**: Ed25519 signing, SHA-256 manifests, receipt validation.

**New Crate**: `provenance/`

```rust
// provenance/src/manifest.rs
#[derive(Serialize, Deserialize)]
pub struct ProofManifest {
    pub version: u32,
    pub backend: Backend,
    pub round: u64,
    pub config_hash: [u8; 32],  // SHA-256 of PoGQ config
    pub nonce: [u8; 32],
    pub timestamp: u64,
    pub proof_hash: [u8; 32],
    pub signature: [u8; 64],  // Ed25519
}

impl ProofManifest {
    pub fn new(
        backend: Backend,
        round: u64,
        config: &PublicInputs,
        proof_bytes: &[u8],
        keypair: &Keypair,
    ) -> Self {
        let config_hash = sha256(serde_json::to_vec(config).unwrap());
        let proof_hash = sha256(proof_bytes);
        let nonce = rand::random();

        let canonical = Self::canonical_bytes(backend, round, &config_hash, &nonce, &proof_hash);
        let signature = keypair.sign(&canonical);

        Self {
            version: 1,
            backend,
            round,
            config_hash,
            nonce,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            proof_hash,
            signature: signature.to_bytes(),
        }
    }

    pub fn verify(&self, pubkey: &PublicKey, proof_bytes: &[u8]) -> Result<(), Error> {
        // Verify proof hash
        let computed_hash = sha256(proof_bytes);
        if computed_hash != self.proof_hash {
            return Err(Error::HashMismatch);
        }

        // Verify signature
        let canonical = Self::canonical_bytes(
            self.backend,
            self.round,
            &self.config_hash,
            &self.nonce,
            &self.proof_hash,
        );
        pubkey.verify(&canonical, &Signature::from_bytes(&self.signature))?;

        Ok(())
    }
}
```

**Deliverable**: Tamper-proof proof receipts with Ed25519 attestation

### 4.3 gRPC Network Service (P3 - 16 hours)

**Goal**: Production-grade daemon for multi-agent coordination.

**New Crate**: `vsv-daemon/`

```rust
// vsv-daemon/src/service.rs
use tonic::{Request, Response, Status};

#[derive(Debug, Default)]
pub struct VsvService {
    coordinator: Arc<Mutex<Coordinator>>,
}

#[tonic::async_trait]
impl vsv_proto::vsv_service_server::VsvService for VsvService {
    async fn submit_round(
        &self,
        request: Request<RoundSubmission>,
    ) -> Result<Response<SubmissionReceipt>, Status> {
        let submission = request.into_inner();

        // Validate signature
        verify_signature(&submission)?;

        // Run PoGQ
        let decision = self.coordinator.lock().await
            .process_submission(submission)?;

        // Generate proof (async, can take 5-15s with Winterfell)
        let proof = tokio::task::spawn_blocking(move || {
            vsv_prover::prove(&decision.public, &decision.witness, Backend::Winterfell)
        }).await??;

        // Create manifest
        let manifest = ProofManifest::new(...);

        Ok(Response::new(SubmissionReceipt {
            round: decision.round,
            quarantine: decision.quarantine_out,
            proof: proof.bytes,
            manifest: serde_json::to_vec(&manifest)?,
        }))
    }

    async fn verify_receipt(
        &self,
        request: Request<VerifyRequest>,
    ) -> Result<Response<VerifyResponse>, Status> {
        let req = request.into_inner();

        let manifest: ProofManifest = serde_json::from_slice(&req.manifest)?;
        manifest.verify(&req.pubkey, &req.proof)?;

        let valid = vsv_verifier::verify(&req.proof, &req.public, manifest.backend)?;

        Ok(Response::new(VerifyResponse { valid }))
    }
}
```

**Deliverable**: Production multi-agent coordinator with async proving

---

## Engineering Guardrails

### Determinism Checklist
- [ ] No floating-point in PoGQ core
- [ ] Q16.16 fixed-point everywhere
- [ ] No `rand` in critical paths (use seeded PRNG with nonce)
- [ ] Pin Rust toolchain in `rust-toolchain.toml`
- [ ] `#[cfg(test)]` determinism tests across architectures

### Testing Pyramid
```
           /\         Property Tests (proptest)
          /  \        Fuzzing (cargo-fuzz)
         /____\       Cross-backend Validation
        /      \      Integration Tests (7 boundary cases)
       /________\     Unit Tests (EMA, hysteresis, etc.)
      /__________\    Smoke Tests (CLI runs)
```

### Security
- [ ] Supply-chain pinning (`Cargo.lock` in git)
- [ ] `cargo deny` (licenses, advisories, bans)
- [ ] `cargo auditable` (embed dep info in binaries)
- [ ] Ed25519 signing on all proofs
- [ ] SHA-256 content addressing
- [ ] No secret material in logs

### Performance
- [ ] `criterion` benches for EMA/hysteresis
- [ ] Proving time benchmarks (zkVM vs AIR)
- [ ] Memory profiling (valgrind / heaptrack)
- [ ] Flamegraphs for hot loops

---

## Immediate Next Actions (Priority Order)

### 🔥 P0: Complete Winterfell (4 hours)
1. Fix cargo build permissions
   ```bash
   cd vsv-stark
   cargo build --release -p winterfell-pogq --target-dir /tmp/wf-target
   ```
2. Run 7 integration tests
   ```bash
   cargo test -p winterfell-pogq --target-dir /tmp/wf-target
   ```
3. Benchmark vs RISC Zero
   ```bash
   python experiments/vsv_dual_backend_benchmark.py --rounds 5
   ```

### 🔥 P0: Generate Table VII-bis (1 hour)
1. Create LaTeX table from benchmark JSON
2. Add to paper Section III.F
3. Document dual-backend strategy

### 🔥 P1: Dual-Backend Rust CLI (6 hours)
1. Create `vsv-prover` crate
2. Implement `--backend {winterfell,risczero}` flag
3. Update Python experiments to use unified CLI

### 🔥 P1: PyO3 PoGQ Bindings (8 hours)
1. Create `vsv-core-py` crate with maturin
2. Wrap `compute_decision` and `Fixed` types
3. Publish wheel: `pip install vsv-core`
4. Update experiments to use Rust PoGQ

### 📋 P2: WASM Verifier (8 hours)
1. Create `vsv-verifier-wasm` with wasm-bindgen
2. Build: `wasm-pack build --target web`
3. Demo in browser: `examples/verify-in-browser.html`

---

## Success Metrics

### Phase 1 (Week 3-4): Core Stabilization
- [ ] Winterfell proves 3-10× faster than zkVM
- [ ] vsv-core passes 100+ property tests
- [ ] Dual-backend CLI works seamlessly
- [ ] Table VII-bis in paper

### Phase 2 (Week 4): Python Interop
- [ ] `pip install vsv-core` works
- [ ] Experiments call Rust PoGQ (no Python reimplementation)
- [ ] Zero-copy performance (no JSON serialization overhead)
- [ ] All existing experiments pass with Rust backend

### Phase 3 (Week 5): Framework-Agnostic FL
- [ ] PyTorch adapter complete
- [ ] TensorFlow adapter complete
- [ ] JAX adapter (stretch goal)
- [ ] Same Rust verifier for all frameworks

### Phase 4 (Week 6+): Production Hardening
- [ ] WASM verifier runs in browser
- [ ] Ed25519 attestation on all proofs
- [ ] gRPC daemon handles 100+ agents
- [ ] Static CLI verifier for air-gapped audit

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1** | Week 3-4 (20h) | Core Rust crates stable |
| **Phase 2** | Week 4 (14h) | Python bindings + wheels |
| **Phase 3** | Week 5 (16h) | Framework-agnostic FL |
| **Phase 4** | Week 6+ (36h) | Production deployment |
| **Total** | 4-5 weeks | Production-ready 0TML |

---

## Decision Log

### ✅ Confirmed Decisions
1. **Winterfell for production proving** - 3-10× speedup justified
2. **vsv-core as single source of truth** - All backends call this
3. **PyO3 for Python interop** - Zero-copy, type-safe, fast
4. **Framework-agnostic boundary** - Any ML framework can participate
5. **Rust for all verification** - Deterministic, auditable, no_std

### ⏳ Pending Decisions
1. **ONNX vs safetensors** for model exchange (leaning safetensors)
2. **Winterfell v0.10 vs v0.13** (staying on v0.10 for now)
3. **gRPC vs REST** for network service (gRPC for type safety)
4. **Holochain conductor integration** timing (after Phase 2)

---

**Status**: Architecture defined, implementation roadmap clear
**Next Session**: Complete Winterfell build + benchmarks + Table VII-bis
**Long-term**: Full Rust core with Python/WASM/JS bindings for ecosystem
