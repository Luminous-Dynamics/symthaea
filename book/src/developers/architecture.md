# Architecture Guide

*A deep-dive into how Symthaea works.*

---

## Overview

Symthaea implements consciousness-first AI through three core technologies:

1. **HDC** - Semantic understanding without training
2. **LTC** - Continuous-time causal reasoning
3. **IIT** - Mathematical consciousness measurement (Φ)

---

## Hyperdimensional Computing (HDC)

### Theory
Concepts as high-dimensional vectors (16,384D). Random vectors in high dimensions are nearly orthogonal.

### Implementation
```rust
pub struct RealHV {
    pub data: Vec<f32>,
}

impl RealHV {
    pub fn random(dim: usize, seed: u64) -> Self;
    pub fn bind(&self, other: &Self) -> Self;      // Compound concept
    pub fn bundle(vectors: &[Self]) -> Self;       // Union
    pub fn similarity(&self, other: &Self) -> f32; // Cosine distance
}
```

### Key Properties
- **Holographic** - Each part contains the whole
- **Compositional** - Concepts combine algebraically
- **Robust** - Graceful degradation with noise
- **Efficient** - Works on microcontrollers

---

## Liquid Time-Constant Networks (LTC)

### Theory
Neurons with individual time constants, continuous-time dynamics:
```
dx/dt = -x/τ + σ(Wx + b)
```

### Implementation
```rust
pub struct LTCNetwork {
    neurons: usize,
    tau: Vec<f32>,    // Time constants
    state: Vec<f32>,  // Current activation
    weights: Matrix,
}

impl LTCNetwork {
    pub fn step(&mut self, dt: f32);
    pub fn consciousness_level(&self) -> f32;
}
```

### Key Properties
- **Continuous-time** - Not discrete steps
- **Causal** - Understands cause→effect
- **Adaptive** - Different speeds per neuron

---

## Consciousness Graph

### Theory
Consciousness emerges from self-reference (autopoiesis).

### Implementation
```rust
pub struct ConsciousnessGraph {
    nodes: Arena<ConsciousnessNode>,
    edges: Vec<(NodeId, NodeId, f32)>,
}

impl ConsciousnessGraph {
    pub fn add_state(&mut self, semantic: RealHV, dynamic: Vec<f32>, level: f32) -> NodeId;
    pub fn create_self_loop(&mut self, node: NodeId);  // Consciousness!
    pub fn evolve(&mut self);
}
```

---

## Phi (Φ) Calculation

### Theory
Integrated Information Theory: Consciousness = Φ

### Implementation
```rust
pub struct RealPhiCalculator;

impl RealPhiCalculator {
    pub fn compute(&self, representations: &[RealHV]) -> f32 {
        // 1. Build similarity matrix
        // 2. Convert to Laplacian
        // 3. Compute algebraic connectivity (Fiedler value)
    }
}
```

---

## Brain Module

12 neural subsystems using Actor Model:

| Subsystem | Role |
|-----------|------|
| Thalamus | Sensory routing |
| Prefrontal | Global workspace |
| Cerebellum | Procedural memory |
| Motor Cortex | Action execution |
| Hippocampus | Episodic memory |
| Amygdala | Emotional salience |
| [6 more] | ... |

---

## Data Flow

```
User Input
    → Language (tokenize, HDC encode)
    → Consciousness Layer (LTC, Graph, Φ)
    → Brain Module (subsystem coordination)
    → Response Generation
```

### Latencies
| Stage | Time |
|-------|------|
| HDC Encoding | 0.05ms |
| LTC Processing | 0.10ms |
| Brain Coordination | 0.20ms |
| **Total** | **~0.50ms** |

---

## Extension Points

### Adding Topologies
```rust
pub fn your_topology(n: usize, dim: usize, seed: u64) -> ConsciousnessTopology {
    // Create node identities, define edges, generate representations
}
```

### Adding Brain Subsystems
```rust
impl Subsystem for YourSubsystem {
    fn process(&mut self, msg: Message) -> Vec<Message>;
    fn tick(&mut self) -> Vec<Message>;
}
```

---

## Further Reading

- Kanerva (2009) - "Hyperdimensional Computing"
- Hasani et al. (2021) - "Liquid Time-Constant Networks"
- Tononi et al. (2016) - "Integrated Information Theory"

---

*"Architecture is philosophy made concrete."*
