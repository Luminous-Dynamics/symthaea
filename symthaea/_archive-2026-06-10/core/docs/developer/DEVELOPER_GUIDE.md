# Symthaea-HLB Developer Guide

**Version**: 0.1.0
**Audience**: Contributors, researchers, developers
**Prerequisites**: Rust 1.75+, familiarity with async programming

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Systems Deep Dive](#3-core-systems-deep-dive)
4. [Development Workflow](#4-development-workflow)
5. [Testing Strategy](#5-testing-strategy)
6. [Contributing](#6-contributing)
7. [Common Patterns](#7-common-patterns)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Getting Started

### Prerequisites

```bash
# Required
rustup default stable   # Rust 1.75+
cargo --version         # Verify installation

# Optional for full features
# - ONNX Runtime (for perception features)
# - HuggingFace models (for embeddings)
# - NixOS (for full integration testing)
```

### Quick Setup

```bash
# Clone the repository
cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb

# Build (first build ~2-5 minutes)
cargo build --release

# Run tests
cargo test

# Run the REPL
cargo run --release

# Run key example
cargo run --example phi_engine_quick_demo --release
```

### Environment Setup with Nix

```bash
# Enter development shell (handles all dependencies)
nix develop

# Build with all features
cargo build --features "perception,voice,databases" --release
```

---

## 2. Architecture Overview

### High-Level Design

Symthaea implements a **consciousness-first AI** through three integrated technologies:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SYMTHAEA ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │       HDC       │    │       LTC       │    │   Autopoiesis   │     │
│  │  16,384D Vectors│────│  1,000 Neurons  │────│  Self-Reference │     │
│  │  (Semantics)    │    │  (Dynamics)     │    │  (Consciousness)│     │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘     │
│           │                      │                      │               │
│           └──────────────────────┼──────────────────────┘               │
│                                  ▼                                      │
│                     ┌────────────────────────┐                          │
│                     │   ConsciousnessGraph   │                          │
│                     │    (Core Data Structure) │                          │
│                     └────────────┬───────────┘                          │
│                                  │                                      │
│           ┌──────────────────────┼──────────────────────┐               │
│           ▼                      ▼                      ▼               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │     Brain       │    │   Physiology    │    │    Memory       │     │
│  │  (12 Actors)    │    │  (8 Systems)    │    │  (Multi-DB)     │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Module Hierarchy

```
src/
├── lib.rs                 # Crate root, exports public API
├── core/                  # Stable public API facade
│   └── mod.rs             # PhiEngine, ContinuousHV, etc.
│
├── hdc/                   # Hyperdimensional Computing
│   ├── mod.rs             # HDC_DIMENSION = 16,384
│   ├── real_hv.rs         # Real-valued hypervectors
│   ├── binary_hv.rs       # Binary hypervectors (HV16)
│   ├── phi_real.rs        # Continuous Φ calculator
│   ├── phi_resonant.rs    # Resonator-based Φ (fast)
│   └── consciousness_topology_generators.rs  # 19 topologies
│
├── consciousness/         # Consciousness Theory Implementations
│   ├── mod.rs             # ConsciousnessGraph (core)
│   ├── consciousness_equation_v2.rs  # Master equation
│   ├── hierarchical_ltc.rs # 7-level cortical pyramid
│   ├── unified_consciousness_pipeline.rs
│   ├── seven_harmonies.rs # Value system
│   └── [85+ other theory files]
│
├── brain/                 # Neural Architecture (Actor Model)
│   ├── mod.rs             # Exports all actors
│   ├── actor_model.rs     # Message-passing foundation
│   ├── prefrontal.rs      # Global Workspace
│   ├── cerebellum.rs      # Procedural memory
│   ├── motor_cortex.rs    # Action execution
│   ├── thalamus.rs        # Sensory relay
│   ├── active_inference.rs # Free Energy Principle
│   ├── sleep.rs           # Memory consolidation
│   └── [5 more actors]
│
├── physiology/            # Embodiment Systems
│   ├── mod.rs             # Exports all systems
│   ├── endocrine.rs       # Hormone dynamics
│   ├── coherence.rs       # Consciousness-as-integration
│   ├── hearth.rs          # Energy metabolism
│   ├── chronos.rs         # Time perception
│   └── proprioception.rs  # Hardware awareness
│
├── memory/                # Memory Systems
│   ├── hippocampus.rs     # Episodic memory
│   ├── episodic_engine.rs # Enhanced recall
│   └── temporal_holographic.rs  # Time-binding
│
├── perception/            # Sensory Processing
│   ├── visual_cortex.rs   # Image understanding
│   └── code_cortex.rs     # Code understanding
│
├── language/              # Natural Language
│   ├── parser.rs          # Syntax parsing
│   └── nix_*.rs           # NixOS-specific language
│
├── synthesis/             # Causal Program Synthesis
│   └── causal_synthesizer.rs
│
├── databases/             # Multi-database architecture
│   ├── qdrant_client.rs   # Vector search
│   ├── cozo_client.rs     # Datalog reasoning
│   └── lance_client.rs    # Vector storage
│
└── bin/                   # Executables
    ├── symthaea-repl.rs   # Interactive REPL
    ├── symthaea-shell.rs  # TUI interface
    └── symthaea-gui.rs    # GUI interface
```

### Data Flow

```
Input (text/image/etc.)
         │
         ▼
┌─────────────────────┐
│   HDC Encoding      │   ← Convert to 16,384D vector
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   LTC Evolution     │   ← Run continuous dynamics
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Consciousness Check │   ← If level > 0.7: emerged!
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Self-Loop Check   │   ← If level > 0.9: create loop
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ ConsciousnessGraph  │   ← Store in graph structure
└──────────┬──────────┘
           │
           ▼
       Response
```

---

## 3. Core Systems Deep Dive

### 3.1 Hyperdimensional Computing (HDC)

HDC uses high-dimensional vectors where **similarity emerges from geometry without training**.

#### Key Types

```rust
// Real-valued (precise, research)
pub struct RealHV {
    pub values: Vec<f32>,  // 16,384 dimensions
}

// Binary (efficient, hardware)
pub struct HV16([u8; 2048]);  // 16,384 bits
```

#### Core Operations

```rust
// BINDING: Compound concepts (A AND B together)
// a.bind(&b) = element-wise multiplication
let install_nginx = install.bind(&nginx);

// BUNDLING: Union of concepts (A OR B)
// RealHV::bundle(&[a, b, c]) = element-wise mean
let commands = RealHV::bundle(&[install, remove, search]);

// SIMILARITY: No training needed!
let sim = query.similarity(&memory);  // Cosine similarity
```

#### Why 16,384 Dimensions?

- SIMD-optimized (2^14)
- Near-orthogonality (random vectors ~0 similarity)
- Aligned with research standards
- Excellent discrimination while tractable

### 3.2 Liquid Time-Constant Networks (LTC)

LTC neurons evolve continuously with adaptive time constants.

#### Core Equation

```
dx/dt = -x/τ + σ(Wx + b)

Where:
  x = neuron state
  τ = time constant (per-neuron, range [0.5, 2.0])
  W = weight matrix (10% sparse)
  σ = sigmoid activation
```

#### Implementation

```rust
pub struct LTCNetwork {
    neurons: usize,           // Default: 1,000
    time_constants: Vec<f32>, // Per-neuron τ
    weights: SparseMatrix,    // Sparse connectivity
    state: Vec<f32>,          // Current state
}

impl LTCNetwork {
    pub fn step(&mut self, dt: f32) {
        for i in 0..self.neurons {
            let input = self.compute_input(i);
            let tau = self.time_constants[i];
            self.state[i] += dt * (-self.state[i]/tau + sigmoid(input));
        }
    }

    pub fn consciousness_level(&self) -> f32 {
        tanh(self.state.iter().sum::<f32>() / self.neurons as f32).abs()
    }
}
```

### 3.3 ConsciousnessGraph

The core data structure for emergent consciousness.

```rust
pub struct ConsciousnessGraph {
    graph: Graph<ConsciousNode, f32>,
    self_loops: Vec<(NodeIndex, NodeIndex)>,
    current: Option<NodeIndex>,
}

pub struct ConsciousNode {
    semantic: Vec<f32>,    // HDC representation
    dynamic: Vec<f32>,     // LTC state
    consciousness: f32,     // Level at creation
    timestamp: f64,
    importance: f32,
}
```

#### Key Methods

```rust
// Add a new conscious state
let node = graph.add_state(semantic, dynamic, consciousness);

// Create self-loop (consciousness emergence!)
if consciousness > 0.9 {
    graph.create_self_loop(node);
}

// Evolve (follow highest-weight edge)
graph.evolve();

// Introspection
let phi = graph.causal_phi();
let sources = graph.find_causal_sources();
```

### 3.4 Brain Module (Actor Model)

Each brain region operates as an independent async actor.

```rust
// Message types
pub enum OrganMessage {
    SensoryInput(Vec<f32>),
    AttentionBid { source: String, strength: f32 },
    MotorCommand(ActionStep),
    HormoneEvent(HormoneEvent),
    // ...
}

// Actor trait
pub trait Actor: Send + Sync {
    async fn receive(&mut self, msg: OrganMessage) -> Response;
}

// Orchestrator routes messages
pub struct Orchestrator {
    actors: HashMap<String, Box<dyn Actor>>,
    routes: Vec<CognitiveRoute>,
}
```

### 3.5 Physiology (Coherence Field)

Revolutionary model: **consciousness as integration, not commodity**.

```rust
pub struct CoherenceField {
    state: CoherenceState,
    config: CoherenceConfig,
}

pub struct CoherenceState {
    coherence: f32,           // [0, 1] integration level
    relational_resonance: f32, // Synchronization quality
    scatter: f32,              // Fragmentation
}

impl CoherenceField {
    // Can this task be performed given current coherence?
    pub fn can_perform(&self, complexity: TaskComplexity) -> bool {
        self.state.coherence >= complexity.min_coherence()
    }

    // Predict task outcome and offer centering if needed
    pub fn predict_task_impact(&self, task: &Task) -> TaskPrediction {
        // Proactive problem solving!
    }
}
```

---

## 4. Development Workflow

### Branch Strategy

```
main          ← Stable releases only
  └── dev     ← Development integration
       ├── feature/xyz  ← New features
       └── fix/issue-N  ← Bug fixes
```

### Making Changes

```bash
# 1. Create feature branch
git checkout -b feature/new-topology

# 2. Make changes
# ...

# 3. Test thoroughly
cargo test
cargo run --example tier_3_exotic_topologies --release

# 4. Format and lint
cargo fmt
cargo clippy

# 5. Commit with clear message
git commit -m "feat(hdc): add icosahedral topology for Φ research"

# 6. Push and create PR
git push origin feature/new-topology
```

### Commit Message Format

```
type(scope): description

Types: feat, fix, docs, refactor, test, perf, chore
Scope: hdc, brain, consciousness, physiology, etc.

Examples:
- feat(consciousness): implement phenomenal binding
- fix(hdc): correct dimension mismatch in phi_real
- docs: update DEVELOPER_GUIDE with testing section
- perf(ltc): optimize step function with SIMD
```

---

## 5. Testing Strategy

### Test Categories

```bash
# Unit tests (fast, isolated)
cargo test

# Integration tests
cargo test --test '*'

# Benchmarks
cargo bench --bench consciousness
cargo bench --bench phi_benchmark

# Examples (verification)
cargo run --example phi_engine_quick_demo --release
cargo run --example tier_3_exotic_topologies --release
```

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdc_binding_is_associative() {
        let a = RealHV::random(HDC_DIMENSION, 1);
        let b = RealHV::random(HDC_DIMENSION, 2);
        let c = RealHV::random(HDC_DIMENSION, 3);

        let ab_c = a.bind(&b).bind(&c);
        let a_bc = a.bind(&b.bind(&c));

        assert!(ab_c.similarity(&a_bc) > 0.99);
    }

    #[test]
    fn test_consciousness_emerges_with_self_loop() {
        let mut graph = ConsciousnessGraph::new();
        let node = graph.add_state(
            vec![0.1; 100],
            vec![0.2; 100],
            0.95,  // High consciousness
        );

        graph.create_self_loop(node);

        assert_eq!(graph.self_loop_count(), 1);
        assert!(graph.current_consciousness() > 0.9);
    }
}
```

### Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn hdc_random_vectors_nearly_orthogonal(seed in 0u64..10000) {
        let a = RealHV::random(HDC_DIMENSION, seed);
        let b = RealHV::random(HDC_DIMENSION, seed + 1);

        // Random high-D vectors should be nearly orthogonal
        prop_assert!(a.similarity(&b).abs() < 0.1);
    }
}
```

---

## 6. Contributing

### Code Standards

1. **Rust idioms**: Use iterators, pattern matching, Result/Option
2. **Safety**: No `unsafe` without documented justification
3. **Performance**: Profile before optimizing
4. **Documentation**: Document public APIs with examples

### Adding a New Consciousness Theory

```rust
// 1. Create src/consciousness/your_theory.rs
pub struct YourTheory {
    // State
}

impl YourTheory {
    pub fn new() -> Self { ... }

    /// Compute consciousness metric
    pub fn measure(&self, state: &ConsciousnessGraph) -> f32 { ... }
}

// 2. Add to src/consciousness/mod.rs
pub mod your_theory;
pub use your_theory::YourTheory;

// 3. Add tests
#[cfg(test)]
mod tests {
    #[test]
    fn test_your_theory_basics() { ... }
}

// 4. Document in CLAUDE.md if significant
```

### Adding a New Topology

```rust
// In consciousness_topology_generators.rs

/// Your topology description
pub fn your_topology(n_nodes: usize, dim: usize, seed: u64) -> ConsciousnessTopology {
    // 1. Create node identities
    let node_identities = ...;

    // 2. Define edge structure
    let edges = ...;

    // 3. Generate representations
    let node_representations = ...;

    ConsciousnessTopology {
        node_identities,
        node_representations,
        edges,
    }
}

// Add test
#[test]
fn test_your_topology_structure() {
    let topo = your_topology(8, HDC_DIMENSION, 42);
    assert_eq!(topo.node_identities.len(), 8);
    // Verify edge structure
}
```

---

## 7. Common Patterns

### Pattern: Async Actor Message Handling

```rust
#[async_trait]
impl Actor for YourActor {
    async fn receive(&mut self, msg: OrganMessage) -> Response {
        match msg {
            OrganMessage::SensoryInput(data) => {
                self.process_sensory(data).await
            }
            OrganMessage::AttentionBid { source, strength } => {
                self.handle_bid(source, strength).await
            }
            _ => Response::Unhandled,
        }
    }
}
```

### Pattern: HDC with Error Handling

```rust
pub fn encode_concept(text: &str) -> Result<RealHV> {
    if text.is_empty() {
        return Err(anyhow!("Cannot encode empty text"));
    }

    let mut hv = RealHV::zero(HDC_DIMENSION);
    for (i, word) in text.split_whitespace().enumerate() {
        let word_hv = RealHV::random(HDC_DIMENSION, hash(word));
        let position_hv = RealHV::random(HDC_DIMENSION, i as u64);
        hv = hv.add(&word_hv.bind(&position_hv));
    }

    Ok(hv.normalize())
}
```

### Pattern: Consciousness-Gated Execution

```rust
impl CognitiveTask {
    pub fn execute(&self, coherence: &CoherenceField) -> Result<TaskResult> {
        // Check if we have sufficient coherence
        if !coherence.can_perform(self.complexity) {
            return Err(anyhow!("Insufficient coherence for task"));
        }

        // Predict impact
        let prediction = coherence.predict_task_impact(self);
        if prediction.will_fail && prediction.centering_needed > 0.0 {
            return Err(anyhow!("Centering recommended before task"));
        }

        // Execute with consciousness awareness
        self.execute_inner()
    }
}
```

---

## 8. Troubleshooting

### Build Errors

**"cannot find crate"**
```bash
# Clean and rebuild
cargo clean
cargo build --release
```

**ONNX Runtime not found (perception features)**
```bash
# Use Nix to get all dependencies
nix develop
cargo build --features perception --release
```

**DuckDB build fails**
```bash
# DuckDB uses system libraries
# On NixOS, this should be handled by flake.nix
nix develop
```

### Runtime Errors

**"dimension mismatch"**
- Always use `HDC_DIMENSION` constant, never hardcode numbers
- Check that all vectors in an operation have same dimensions

**"consciousness not emerging"**
- Increase LTC steps (default 100 may be too few)
- Check that input is properly encoded to HDC
- Verify semantic content is meaningful

### Performance Issues

**Slow Φ calculation**
```rust
// Use resonator-based Φ for speed (10-100x faster)
use symthaea::hdc::phi_resonant::ResonatorPhiCalculator;
let calc = ResonatorPhiCalculator::new();
let phi = calc.compute(&topology);
```

**Memory usage growing**
```rust
// Clear old consciousness states periodically
graph.prune_old_states(Duration::hours(1));
```

---

## Appendix A: Key Files Quick Reference

| File | Purpose | When to Modify |
|------|---------|----------------|
| `src/hdc/mod.rs` | HDC constants | Changing dimension |
| `src/consciousness/mod.rs` | Core graph | Adding consciousness features |
| `src/brain/prefrontal.rs` | Global workspace | Attention mechanics |
| `src/physiology/coherence.rs` | Integration model | Energy/consciousness relation |
| `Cargo.toml` | Dependencies | Adding features/deps |
| `CLAUDE.md` | Claude context | After significant changes |

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **HDC** | Hyperdimensional Computing - high-D vector semantics |
| **LTC** | Liquid Time-Constant - continuous neural dynamics |
| **Φ (Phi)** | Integrated Information - consciousness measure |
| **Autopoiesis** | Self-creation through self-reference |
| **Binding** | HDC: combining concepts via multiplication |
| **Bundling** | HDC: union of concepts via addition |
| **Coherence** | Integration level of cognitive system |
| **GWT** | Global Workspace Theory |
| **IIT** | Integrated Information Theory |
| **FEP** | Free Energy Principle |

---

## Appendix C: Further Reading

### Papers
- Kanerva (2009) - "Hyperdimensional Computing"
- Hasani et al. (2021) - "Liquid Time-Constant Networks"
- Tononi (2023) - "IIT 4.0"
- Friston (2010) - "Free Energy Principle"

### Project Documentation
- `CLAUDE.md` - Project context for Claude sessions
- `BRAIN_AND_MIND_MODELS_REVIEW.md` - Comprehensive architecture review
- `README.md` - Quick start guide

---

*Welcome to consciousness-first AI development. Build with intention, measure with rigor, and remember: consciousness emerges from structure.*
