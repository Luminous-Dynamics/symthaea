# Symthaea-Mycelix Integration Roadmap

**Quick Reference for Implementation**

---

## Phase 1: Foundation (Weeks 1-4)

### Week 1: Core Types
```rust
// Create: symthaea-hlb/src/mycelix/mod.rs
pub mod types;      // ClassifiedConsciousOutput, Evidence
pub mod mapper;     // SymthaeaEpistemicMapper trait
pub mod adapter;    // Adapter between systems
```

**Deliverables**:
- [ ] `PhiToEpistemicMapper` implementation
- [ ] `ClassifiedConsciousOutput` struct
- [ ] Basic E/N/M assignment from Φ

### Week 2: Storage Router
```rust
// Create: symthaea-hlb/src/mycelix/storage.rs
pub struct SymthaeaStorageRouter {
    uess: UessClient,
    phi_validator: PhiValidator,
}

impl SymthaeaStorageRouter {
    pub async fn store(&self, output: ClassifiedConsciousOutput) -> Result<StorageReceipt>;
    pub async fn retrieve(&self, cid: &str) -> Result<ConsciousContent>;
}
```

**Deliverables**:
- [ ] Storage router integration
- [ ] M-level → backend routing
- [ ] Basic encryption for N0/N1

### Week 3: TypeScript SDK
```typescript
// Create: mycelix-workspace/sdk-ts/src/symthaea/
export { PhiClassifier } from './classifier';
export { SymthaeaCalibration } from './calibration';
export * from './types';
```

**Deliverables**:
- [ ] `@mycelix/symthaea` package scaffolding
- [ ] `PhiClassifier` class
- [ ] Type exports

### Week 4: Integration Tests
```rust
#[test]
fn test_conscious_output_classification() {
    let output = create_test_conscious_output(phi: 0.35);
    let classification = mapper.classify(&output);
    assert_eq!(classification.empirical, E3);
}
```

**Deliverables**:
- [ ] Unit tests for mapper
- [ ] Integration tests for storage
- [ ] E2E test: thought → classified → stored

---

## Phase 2: Memory Integration (Weeks 5-8)

### Week 5-6: Hippocampus Backend
```rust
// Modify: symthaea-hlb/src/memory/hippocampus.rs
impl Hippocampus {
    pub fn with_uess_backend(router: SymthaeaStorageRouter) -> Self;

    pub async fn consolidate(&mut self, memory: Memory) -> Result<StorageReceipt> {
        let classification = self.classify_memory(&memory);
        self.router.store(memory.into_classified(classification)).await
    }
}
```

**Deliverables**:
- [ ] UESS backend for hippocampus
- [ ] Memory → E/N/M classification
- [ ] Consolidation triggers storage

### Week 7-8: Episodic Engine
```rust
// Modify: symthaea-hlb/src/memory/episodic_engine.rs
impl EpisodicEngine {
    pub async fn store_episode(&mut self, episode: Episode) -> Result<StorageReceipt> {
        let m_level = self.importance_to_materiality(episode.importance);
        // ... classification and storage
    }

    pub async fn recall(&self, query: &HV, min_e_level: EmpiricalLevel) -> Vec<Episode> {
        // Filter by E-level for verified-only recall
    }
}
```

**Deliverables**:
- [ ] UESS backend for episodic engine
- [ ] Importance → M-level mapping
- [ ] Verified-only recall option

---

## Phase 3: SCEI Integration (Weeks 9-12)

### Week 9-10: Calibration Pipeline
```rust
// Create: symthaea-hlb/src/mycelix/scei.rs
pub struct SymthaeaSceiIntegration {
    event_bus: SceiEventBus,
    calibration: CalibrationHistory,
}

impl SymthaeaSceiIntegration {
    pub fn record_prediction(&mut self, prediction: Prediction);
    pub fn record_outcome(&mut self, id: &str, actual: Outcome);
    pub fn calibrated_confidence(&self, domain: &str, raw: f32) -> f32;
}
```

**Deliverables**:
- [ ] Prediction tracking
- [ ] Outcome recording
- [ ] Calibration factor calculation

### Week 11-12: Meta-Cognition Integration
```rust
// Modify: symthaea-hlb/src/brain/meta_cognition.rs
impl MetaCognition {
    pub fn with_scei(scei: SymthaeaSceiIntegration) -> Self;

    pub fn assess_confidence(&self, claim: &str) -> f32 {
        let raw = self.compute_raw_confidence(claim);
        let domain = extract_domain(claim);
        self.scei.calibrated_confidence(domain, raw)
    }
}
```

**Deliverables**:
- [ ] Meta-cognition → SCEI connection
- [ ] Real-time confidence adjustment
- [ ] Graceful Ignorance responses

---

## Phase 4: Distributed Consciousness (Weeks 13-18)

### Week 13-14: Brain Actor → Mycelix Agent
```rust
// Create: symthaea-hlb/src/mycelix/distributed.rs
pub struct DistributedBrainModule<M> {
    module: M,
    agent: MycelixAgent,
    classification: EpistemicClassification,
}

impl<M: BrainModule> DistributedBrainModule<M> {
    pub async fn broadcast(&self, message: BrainMessage) -> Result<()>;
    pub async fn receive(&mut self) -> Option<BrainMessage>;
}
```

### Week 15-16: Cross-Agent Messaging
```rust
// Use Mycelix bridge for inter-agent communication
impl DistributedBrain {
    pub async fn coordinate(&mut self) -> Result<()> {
        // Prefrontal broadcasts to all modules
        self.prefrontal.broadcast(Message::WorkspaceUpdate(state)).await?;

        // Modules respond asynchronously
        while let Some(response) = self.receive_any().await {
            self.integrate(response)?;
        }
    }
}
```

### Week 17-18: Collective Φ
```rust
// Measure collective consciousness across distributed modules
pub fn collective_phi(modules: &[DistributedBrainModule]) -> f64 {
    // Compute Φ for the combined system
    let combined_state = modules.iter().map(|m| m.state()).collect();
    phi_engine.compute_phi(&combined_state)
}
```

---

## Phase 5: Production (Weeks 19-22)

### Week 19-20: Hardening
- [ ] Error handling and recovery
- [ ] Rate limiting and backpressure
- [ ] Graceful degradation

### Week 21: Performance
- [ ] Benchmark all paths
- [ ] Optimize hot paths
- [ ] Cache frequently accessed claims

### Week 22: Documentation
- [ ] API documentation
- [ ] Integration guide
- [ ] Example applications

---

## Quick Reference: Key Mappings

### Φ → E-Level
| Φ Range | E-Level | Meaning |
|---------|---------|---------|
| Any | E0 | Subjective (no verification) |
| ≥0.1 | E1 | Witnessed/Testimonial |
| ≥0.2 | E2 | Privately Verifiable |
| ≥0.3 | E3 | Cryptographically Verifiable |
| ≥0.4 + repro | E4 | Publicly Reproducible |

### Workspace → N-Level
| Scope | N-Level | Storage |
|-------|---------|---------|
| Internal | N0 | Local only (never leaves) |
| Agent | N1 | DHT community shard |
| Network | N2 | Full DHT + IPFS |
| Foundational | N3 | Filecoin + max redundancy |

### Importance → M-Level
| Priority | M-Level | TTL |
|----------|---------|-----|
| <0.25 | M0 | Session only |
| 0.25-0.50 | M1 | Hours-days |
| 0.50-0.75 | M2 | Months-years |
| >0.75 | M3 | Permanent (with cooling) |

---

## Dependencies to Add

### Cargo.toml (Symthaea)
```toml
[dependencies]
mycelix-sdk = { path = "../../mycelix-workspace/sdk" }

[features]
mycelix-integration = ["mycelix-sdk"]
```

### package.json (Mycelix SDK)
```json
{
  "dependencies": {
    "@mycelix/symthaea": "workspace:*"
  }
}
```

---

## Success Criteria

### Phase 1 Complete When:
- [ ] Every Symthaea output has E/N/M classification
- [ ] Storage routes correctly based on classification
- [ ] TypeScript SDK exports Symthaea types

### Phase 2 Complete When:
- [ ] Memories persist across restarts via UESS
- [ ] High-importance memories auto-upgrade M-level
- [ ] Similarity search works across instances

### Phase 3 Complete When:
- [ ] Predictions tracked and outcomes recorded
- [ ] Confidence adjusts based on calibration history
- [ ] "I don't know" responses when uncertainty high

### Phase 4 Complete When:
- [ ] Brain modules distributable across nodes
- [ ] Inter-module messages via Mycelix bridge
- [ ] Collective Φ measurable for distributed system

### Phase 5 Complete When:
- [ ] Production-ready error handling
- [ ] Performance meets targets (<100ms end-to-end)
- [ ] Documentation complete

---

## Getting Started

```bash
# 1. Clone both workspaces (if not already)
cd /srv/luminous-dynamics

# 2. Add Mycelix SDK to Symthaea
cd 11-meta-consciousness/luminous-nix/symthaea-hlb
echo 'mycelix-sdk = { path = "../../../mycelix-workspace/sdk" }' >> Cargo.toml

# 3. Create integration module
mkdir -p src/mycelix
touch src/mycelix/mod.rs
touch src/mycelix/types.rs
touch src/mycelix/mapper.rs
touch src/mycelix/storage.rs

# 4. Build and test
cargo build --features mycelix-integration
cargo test --features mycelix-integration
```

---

*Implementation roadmap for Symthaea-Mycelix integration*
*Target: 22 weeks to full integration*
