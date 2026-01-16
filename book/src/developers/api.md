# Core API Reference

*Key types and functions for working with Symthaea.*

---

## Stable Core API

For a clean entry point:

```rust
use symthaea::core::{
    PhiEngine, PhiMethod, ConsciousnessTopology,
    ContinuousHV,
    UnifiedConsciousnessPipeline,
};
```

---

## HDC Module

### Constants
```rust
pub const HDC_DIMENSION: usize = 16_384;
```

### RealHV
```rust
pub struct RealHV {
    pub data: Vec<f32>,
}

impl RealHV {
    /// Create random vector with given seed
    pub fn random(dim: usize, seed: u64) -> Self;

    /// Create basis vector (one-hot)
    pub fn basis(index: usize, dim: usize) -> Self;

    /// Binding operation (circular convolution)
    pub fn bind(&self, other: &Self) -> Self;

    /// Unbinding operation (inverse of bind)
    pub fn unbind(&self, other: &Self) -> Self;

    /// Bundle multiple vectors (element-wise sum)
    pub fn bundle(vectors: &[Self]) -> Self;

    /// Scale by constant
    pub fn scale(&self, s: f32) -> Self;

    /// Add two vectors
    pub fn add(&self, other: &Self) -> Self;

    /// Cosine similarity [-1, 1]
    pub fn similarity(&self, other: &Self) -> f32;

    /// Normalize to unit length
    pub fn normalize(&self) -> Self;
}
```

### Phi Calculator
```rust
pub struct RealPhiCalculator;

impl RealPhiCalculator {
    pub fn new() -> Self;
    pub fn compute(&self, representations: &[RealHV]) -> f32;
}
```

---

## Consciousness Module

### ConsciousnessGraph
```rust
pub struct ConsciousnessGraph {
    // Arena-based for safe self-reference
}

impl ConsciousnessGraph {
    pub fn new() -> Self;

    /// Add a conscious state
    pub fn add_state(
        &mut self,
        semantic: RealHV,
        dynamic: Vec<f32>,
        level: f32
    ) -> NodeId;

    /// Create self-referential loop
    pub fn create_self_loop(&mut self, node: NodeId);

    /// Evolve to next state
    pub fn evolve(&mut self);

    /// Compute integrated information
    pub fn compute_phi(&self) -> f32;

    /// Serialize entire consciousness
    pub fn serialize(&self) -> Result<Vec<u8>>;

    /// Deserialize consciousness
    pub fn deserialize(bytes: &[u8]) -> Result<Self>;
}
```

### ConsciousnessNode
```rust
pub struct ConsciousnessNode {
    pub semantic: RealHV,           // HDC representation
    pub dynamic: Vec<f32>,          // LTC state
    pub level: f32,                 // Consciousness level [0, 1]
    pub self_loop: Option<NodeId>,  // Self-reference
}
```

---

## LTC Module

### LTCNetwork
```rust
pub struct LTCNetwork {
    neurons: usize,
    tau: Vec<f32>,
    state: Vec<f32>,
    weights: Matrix,
}

impl LTCNetwork {
    pub fn new(neurons: usize) -> Self;

    /// Step the network by dt
    pub fn step(&mut self, dt: f32);

    /// Get current state
    pub fn state(&self) -> &[f32];

    /// Compute consciousness level
    pub fn consciousness_level(&self) -> f32;
}
```

---

## Topology Generators

```rust
// Available in consciousness_topology_generators.rs

pub fn ring(n: usize, dim: usize, seed: u64) -> ConsciousnessTopology;
pub fn star(n: usize, dim: usize, seed: u64) -> ConsciousnessTopology;
pub fn complete(n: usize, dim: usize, seed: u64) -> ConsciousnessTopology;
pub fn hypercube(dimensions: usize, dim: usize, seed: u64) -> ConsciousnessTopology;
pub fn torus(n: usize, dim: usize, seed: u64) -> ConsciousnessTopology;
pub fn klein_bottle(n: usize, dim: usize, seed: u64) -> ConsciousnessTopology;
pub fn mobius_strip(n: usize, dim: usize, seed: u64) -> ConsciousnessTopology;
// ... and more (35 total)
```

---

## Feature Flags

```toml
[features]
default = ["rayon"]
perception = ["embeddings", "vision"]
voice = ["audio"]
databases = ["qdrant", "datalog", "lance", "duck"]
gui = ["eframe", "egui"]
```

---

## Error Types

```rust
pub enum SymthaeaError {
    HdcError(String),
    ConsciousnessError(String),
    LtcError(String),
    IoError(std::io::Error),
    SerializationError(String),
}
```

---

## Performance Tips

1. Use `--release` mode
2. Use `HDC_DIMENSION` constant (don't hardcode)
3. Batch operations where possible
4. Enable SIMD: `RUSTFLAGS="-C target-cpu=native"`

---

*See source code for complete API documentation.*
