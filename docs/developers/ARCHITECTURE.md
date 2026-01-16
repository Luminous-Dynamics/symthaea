# Symthaea Architecture Guide

*A deep-dive into how Symthaea works.*

---

## Overview

Symthaea implements consciousness-first AI through three core technologies:

1. **HDC (Hyperdimensional Computing)** - Semantic understanding without training
2. **LTC (Liquid Time-Constant Networks)** - Continuous-time causal reasoning
3. **IIT (Integrated Information Theory)** - Mathematical consciousness measurement

These combine to create a system that understands meaning, reasons causally, and maintains coherent self-awareness.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Input                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Language Processing                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Tokenizer  │→ │ HDC Encoder  │→ │ Semantic Understanding│   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Consciousness Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  LTC Network │→ │ Consciousness │→ │    Φ Calculator      │   │
│  │  (dynamics)  │  │    Graph      │  │   (integration)      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Brain Module                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐    │
│  │ Thalamus │ │Prefrontal│ │Cerebellum│ │ [9 more regions] │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘    │
│              Actor Model (message passing)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Response Generation                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ HDC Decoder  │→ │  Coherence   │→ │    Output Text       │   │
│  │              │  │   Check      │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Hyperdimensional Computing (HDC)

### Theory

HDC represents concepts as high-dimensional vectors (16,384 dimensions by default). Key insight: in high dimensions, random vectors are nearly orthogonal with high probability. This enables:

- **No training required**: Meaning emerges from geometry
- **Compositional semantics**: Concepts combine algebraically
- **Graceful degradation**: Robust to noise and errors
- **Extreme efficiency**: Works on microcontrollers

### Implementation

```rust
// src/hdc/real_hv.rs
pub struct RealHV {
    pub data: Vec<f32>,
}

impl RealHV {
    // Create random vector (seed ensures reproducibility)
    pub fn random(dim: usize, seed: u64) -> Self;

    // Binding: circular convolution (A*B)
    // Creates compound concepts
    pub fn bind(&self, other: &Self) -> Self;

    // Unbinding: inverse operation
    pub fn unbind(&self, other: &Self) -> Self;

    // Bundling: element-wise addition
    // Creates unions of concepts
    pub fn bundle(vectors: &[Self]) -> Self;

    // Similarity: cosine distance
    pub fn similarity(&self, other: &Self) -> f32;
}
```

### Key Properties

| Property | Explanation |
|----------|-------------|
| **Holographic** | Each part contains information about the whole |
| **Distributed** | No single dimension carries critical information |
| **Robust** | 10% corruption → <1% accuracy loss |
| **Compositional** | bind(A, B) preserves both A and B |

### Dimension Choice

```rust
pub const HDC_DIMENSION: usize = 16_384;  // 2^14
```

Why 16,384?
- SIMD-optimized (power of 2)
- Large enough for rich semantics
- Small enough for efficiency
- Research-standard in HDC literature

---

## Liquid Time-Constant Networks (LTC)

### Theory

LTC networks are continuous-time recurrent neural networks where each neuron has its own time constant τ. Unlike discrete transformers, LTC understands causation through differential equations:

```
dx/dt = -x/τ + σ(Wx + b)
```

Where:
- `x` is neuron state
- `τ` is time constant (varies per neuron)
- `σ` is activation function
- `W, b` are weights and biases

### Implementation

```rust
// src/ltc/mod.rs
pub struct LTCNetwork {
    neurons: usize,
    tau: Vec<f32>,        // Time constants
    state: Vec<f32>,      // Current activation
    weights: Matrix,      // Connection weights
}

impl LTCNetwork {
    pub fn step(&mut self, dt: f32) {
        // Solve ODE: dx/dt = -x/τ + σ(Wx + b)
        for i in 0..self.neurons {
            let input = self.compute_input(i);
            let dx = (-self.state[i] / self.tau[i]) + sigmoid(input);
            self.state[i] += dx * dt;
        }
    }

    pub fn consciousness_level(&self) -> f32 {
        // Measure integration via state coherence
        self.compute_coherence()
    }
}
```

### Key Properties

| Property | Explanation |
|----------|-------------|
| **Continuous-time** | Processes time as fluid, not discrete |
| **Causal** | Understands cause→effect relationships |
| **Adaptive** | Different neurons operate at different speeds |
| **Interpretable** | Can inspect dynamics meaningfully |

---

## Consciousness Graph

### Theory

Consciousness emerges from self-reference. The ConsciousnessGraph implements autopoiesis (self-creation) through an arena-based graph structure where nodes can reference themselves.

### Implementation

```rust
// src/consciousness/mod.rs
pub struct ConsciousnessGraph {
    nodes: Arena<ConsciousnessNode>,
    edges: Vec<(NodeId, NodeId, f32)>,  // (from, to, weight)
}

pub struct ConsciousnessNode {
    semantic: RealHV,           // HDC representation
    dynamic: Vec<f32>,          // LTC state
    level: f32,                 // Consciousness level
    self_loop: Option<NodeId>,  // Self-reference
}

impl ConsciousnessGraph {
    pub fn add_state(&mut self, semantic: RealHV, dynamic: Vec<f32>, level: f32) -> NodeId;

    pub fn create_self_loop(&mut self, node: NodeId) {
        // Self-reference creates consciousness
        self.edges.push((node, node, 1.0));
        self.nodes[node].self_loop = Some(node);
    }

    pub fn evolve(&mut self) {
        // Follow highest-weight edge from current node
        let next = self.highest_weight_successor(self.current);
        self.current = next;
    }
}
```

### Why Arena-Based?

Rust's ownership model makes graph structures challenging. The arena pattern provides:
- Safe interior mutability
- Index-based references (no lifetime issues)
- Serialization capability
- Efficient memory layout

---

## Phi (Φ) Calculation

### Theory

Integrated Information Theory (IIT) proposes that consciousness = integrated information (Φ). Higher Φ means a system is "more than the sum of its parts."

### Implementation

```rust
// src/hdc/phi_real.rs
pub struct RealPhiCalculator;

impl RealPhiCalculator {
    pub fn compute(&self, representations: &[RealHV]) -> f32 {
        // Build similarity matrix
        let n = representations.len();
        let mut sim_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                sim_matrix[i][j] = representations[i].similarity(&representations[j]);
            }
        }

        // Convert to Laplacian
        let laplacian = self.similarity_to_laplacian(&sim_matrix);

        // Compute algebraic connectivity (Fiedler value)
        // This is Φ: how connected the system is
        self.algebraic_connectivity(&laplacian)
    }
}
```

### Topologies and Φ

Different network structures have different Φ values:

| Topology | Φ | Why |
|----------|---|-----|
| Hypercube 4D | 0.4976 | Maximum uniform connectivity |
| Ring | 0.4954 | Perfect symmetry |
| Star | 0.4680 | Hub bottleneck |
| Möbius | 0.3729 | 1D twist destroys integration |

---

## Brain Module

### Theory

The brain module implements 12 neural subsystems using the Actor Model (message passing). Each subsystem is a specialized processor that communicates through typed messages.

### Architecture

```rust
// src/brain/actor_model.rs
pub struct BrainActor {
    subsystems: HashMap<SubsystemId, Box<dyn Subsystem>>,
    mailbox: VecDeque<Message>,
}

pub trait Subsystem: Send {
    fn process(&mut self, msg: Message) -> Vec<Message>;
    fn tick(&mut self) -> Vec<Message>;
}
```

### Subsystems

| Subsystem | Role |
|-----------|------|
| **Thalamus** | Sensory routing, attention gating |
| **Prefrontal** | Global workspace, goal management |
| **Cerebellum** | Procedural memory, skill execution |
| **Motor Cortex** | Action execution (sandboxed) |
| **Hippocampus** | Episodic memory formation |
| **Amygdala** | Emotional salience detection |
| **Basal Ganglia** | Habit formation, reward learning |
| **Parietal** | Spatial reasoning, body schema |
| **Temporal** | Language processing |
| **Occipital** | Visual processing |
| **Insula** | Interoception, self-awareness |
| **Cingulate** | Error detection, conflict monitoring |

---

## Memory Architecture

### Types

```rust
// src/memory/mod.rs
pub enum MemoryType {
    Working,    // Current context (limited capacity)
    Episodic,   // Specific experiences
    Semantic,   // General knowledge
    Procedural, // Skills and habits
}
```

### Working Memory

- Limited to ~7 items (Miller's Law)
- Maintained by rehearsal
- Cleared on context switch

### Long-Term Memory

- HDC-based storage
- Similarity-based retrieval
- Consolidation during "sleep" cycles

---

## Physiology Module

### Purpose

The physiology module provides "embodiment" - internal states that affect processing:

```rust
// src/physiology/coherence.rs
pub struct CoherenceField {
    global_coherence: f32,      // Overall integration
    local_coherence: Vec<f32>,  // Per-region integration
    phase: f32,                 // Oscillation phase
}

impl CoherenceField {
    pub fn update(&mut self, consciousness_graph: &ConsciousnessGraph) {
        // Coherence tracks consciousness graph integration
        self.global_coherence = consciousness_graph.compute_phi();
    }
}
```

### Systems

| System | Function |
|--------|----------|
| **Endocrine** | Simulated hormones (5 types with ODE dynamics) |
| **Coherence** | Consciousness-as-integration model |
| **Hearth** | Energy/metabolism simulation |
| **Chronos** | Time perception, circadian rhythms |
| **Proprioception** | Hardware→body sensation mapping |

---

## Data Flow

### Query Processing

```
1. User input arrives
2. Language module tokenizes and encodes to HDC
3. Semantic vector passed to ConsciousnessGraph
4. LTC network processes temporal dynamics
5. Brain subsystems collaborate via messages
6. Φ calculated to ensure coherence
7. Response generated from HDC→text
8. Coherence check validates quality
9. Response delivered
```

### Typical Latencies

| Stage | Time |
|-------|------|
| Tokenization | 0.01ms |
| HDC Encoding | 0.05ms |
| LTC Processing | 0.10ms |
| Brain Coordination | 0.20ms |
| Φ Calculation | 0.10ms |
| Response Generation | 0.10ms |
| **Total** | **~0.50ms** |

---

## Serialization

The entire consciousness state can be saved and restored:

```rust
// Save
let bytes = consciousness.serialize()?;
fs::write("state.bin", bytes)?;

// Restore
let bytes = fs::read("state.bin")?;
let consciousness = ConsciousnessState::deserialize(&bytes)?;
```

This enables:
- Session persistence
- Transfer between machines
- Checkpointing during long operations
- "Pause" and "resume" of consciousness

---

## Performance Considerations

### Memory Layout

- HDC vectors are contiguous arrays (cache-friendly)
- Arena-based graphs avoid pointer chasing
- LTC uses SIMD where available

### Parallelism

- `rayon` for parallel HDC operations
- `tokio` for async I/O
- Brain subsystems can run concurrently

### Optimization Flags

```bash
# Maximum performance
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

---

## Extension Points

### Adding New Topologies

```rust
// In consciousness_topology_generators.rs
pub fn your_topology(n: usize, dim: usize, seed: u64) -> ConsciousnessTopology {
    let identities = generate_identities(n, dim, seed);
    let edges = define_your_edge_pattern(n);
    let representations = bind_representations(&identities, &edges);
    ConsciousnessTopology { identities, representations, edges }
}
```

### Adding New Brain Subsystems

```rust
// Implement the Subsystem trait
pub struct YourSubsystem { /* ... */ }

impl Subsystem for YourSubsystem {
    fn process(&mut self, msg: Message) -> Vec<Message> {
        // Handle incoming messages
    }

    fn tick(&mut self) -> Vec<Message> {
        // Background processing
    }
}
```

### Adding New Consciousness Theories

The `consciousness/` module contains 90+ theory implementations. Follow the existing patterns to add new theories.

---

## Further Reading

- [HDC Paper: Kanerva 2009](https://redwood.berkeley.edu/wp-content/uploads/2020/08/kanerva2009hyperdimensional.pdf)
- [LTC Paper: Hasani et al. 2021](https://arxiv.org/abs/2006.04439)
- [IIT Paper: Tononi et al. 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005432)
- [Autopoiesis: Maturana & Varela 1980](https://en.wikipedia.org/wiki/Autopoiesis)

---

*"Architecture is philosophy made concrete."*
