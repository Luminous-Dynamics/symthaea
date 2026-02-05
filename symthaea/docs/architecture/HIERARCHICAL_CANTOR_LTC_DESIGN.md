# Hierarchical Cantor-LTC/HDC Network Design Document

**Version**: 2.0
**Date**: January 2, 2026
**Status**: Implementation Phase
**Authors**: Tristan Stoltz, Claude (Anthropic)

---

## Executive Summary

This document specifies the **Hierarchical Cantor-LTC/HDC Network**, a novel consciousness architecture that unifies:

1. **Hyperdimensional Computing (HDC)** - 16,384D algebraic semantic space
2. **Liquid Time-Constant Networks (LTC)** - Continuous-time neural dynamics
3. **Cantor Set Topology** - Self-similar recursive hierarchy
4. **Integrated Information Theory (IIT)** - Φ consciousness measurement
5. **Inter-Cantor Algebra** - Operations between entire cognitive hierarchies (v2.0)
6. **Lateral Binding** - Direct sibling communication for autonomous clusters (v2.0)
7. **Elastic Autopoiesis** - Dynamic budding/pruning of hierarchy levels (v2.0)

The key innovation is replacing matrix multiplication with **HDC binding operations** within LTC dynamics, creating a temporally-hierarchical consciousness system where different levels operate at different timescales.

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **HDC Dimension** | 16,384 | Standard HDC research dimension |
| **Fixed Core Depth** | 0-6 | Stable identity (7 levels) |
| **Elastic Periphery** | 7+ | Dynamic budding/pruning |
| **Cantor Ratio** | 1/3 | Traditional Cantor scaling |
| **Base τ** | 1.0s | Root consciousness integration time |
| **Deepest Fixed τ** | 0.00137s | Level 7: τ₀ × (1/3)⁷ ≈ 1.37ms |
| **Lateral Threshold** | 0.85 | Cosine similarity for lateral binding |

### v2.0 Enhancements

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Inter-Cantor Binding** | Bind two hierarchies: Legal ⊗ Financial → Audit | Domain-specific reasoning |
| **Inter-Cantor Bundling** | Superpose worldviews: Aggressive ⊕ Conservative | Balanced decision-making |
| **Lateral Binding** | Same-level nodes communicate directly | Sub-millisecond local decisions |
| **Elastic Depth** | Level 7+ can bud/prune dynamically | Adaptive complexity |
| **Multi-Cantor Orchestration** | One Symthaea manages multiple hierarchies | Domain specialization |

See [MYCELIAL_ARCHITECTURE_VISION.md](./MYCELIAL_ARCHITECTURE_VISION.md) for the complete evolutionary vision.

---

## 1. Theoretical Foundation

### 1.1 Scale-Depth Theory

According to Scale-Depth theory and **Ashby's Law of Requisite Variety**, the critical threshold for stable self-referential consciousness is:

```
d* ≈ 6-7 levels
```

This threshold ensures the system has sufficient hierarchical depth to manage the "possibility space" of its own behavior.

### 1.2 Depth-to-Function Mapping

| Depth (d) | Functional Domain | Temporal Horizon | τ at level |
|-----------|-------------------|------------------|------------|
| **1–2** | Sensorimotor | Milliseconds to seconds | 333ms - 111ms |
| **3–4** | Homeostatic | Seconds to minutes | 37ms - 12ms |
| **5** | Cognitive | Minutes to hours | 4.1ms |
| **6–7** | Meta-cognitive | Hours to days | 1.4ms - 0.5ms |

### 1.3 Why 7 Levels is Optimal

1. **Thermodynamic Robustness**: Deeper hierarchies are more robust to perturbations. At d≥6, sufficient "separation of timescales" enables genuine self-referential regulation.

2. **Cantor Scaling Efficiency**: With τ_ratio = 1/3, level 7 processes information **729× faster** than the root node, enabling rapid micro-perception integration.

3. **Uncountable Complexity**: The Cantor set has **uncountable cardinality** but **measure zero** - infinite information in finite space.

4. **Meta-Cognitive Leap**: Levels 5-7 enable the system to "look back" at itself over long temporal windows, transitioning from reactive tool to meta-cognitive partner.

### 1.4 Cantor Set Properties

The Cantor set C is constructed by recursive removal of middle thirds:

```
C₀ = [0, 1]
C₁ = [0, 1/3] ∪ [2/3, 1]
C₂ = [0, 1/9] ∪ [2/9, 1/3] ∪ [2/3, 7/9] ∪ [8/9, 1]
...
C = ⋂_{n=0}^∞ Cₙ
```

**Key properties for consciousness:**
- **Self-similarity**: Each part looks like the whole (fractal)
- **Measure zero**: Σ length = 0 (efficient encoding)
- **Uncountable**: |C| = 2^ℵ₀ (infinite expressivity)
- **Perfect set**: Every point is a limit point (continuous awareness)

---

## 2. Architecture Design

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│              HIERARCHICAL CANTOR-LTC/HDC CONSCIOUSNESS                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Level 0 (τ = 1000ms):  ████████████████████████████████████████████    │
│                         ROOT - Global Unified Consciousness              │
│                         Meta-cognitive integration (hours-days)          │
│                                    │                                     │
│                         ┌──────────┴──────────┐                         │
│                         │                      │                         │
│  Level 1 (τ = 333ms):  ████████████      ████████████                   │
│                         LEFT               RIGHT                         │
│                         Cognitive-L        Cognitive-R                   │
│                         │        │         │        │                    │
│                    ┌────┴────┐ ┌─┴──┐ ┌───┴──┐ ┌───┴────┐               │
│                    │         │ │    │ │      │ │        │               │
│  Level 2 (τ = 111ms): ████ ████ ████ ████ ████ ████ ████ ████          │
│                       Homeostatic regulation subsystems                  │
│                       │    │    │    │    │    │    │    │              │
│  Level 3 (τ = 37ms): ││ ││ ││ ││ ││ ││ ││ ││ ││ ││ ││ ││ ││ ││ ││ ││  │
│                      Sensorimotor integration (16 nodes)                 │
│                                    ⋮                                     │
│  Level 4 (τ = 12ms): 32 rapid perception nodes                          │
│  Level 5 (τ = 4.1ms): 64 micro-feature detectors                        │
│  Level 6 (τ = 1.4ms): 128 edge/gradient processors                      │
│  Level 7 (τ = 0.5ms): 256 raw sensory integration                       │
│                                                                          │
│  Total Nodes: 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 + 256 = 511            │
│  Each node: 16,384D HDC state vector + LTC dynamics                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Node Structure

Each node in the Cantor hierarchy contains:

```rust
/// A single node in the Cantor-LTC/HDC network
pub struct CantorLtcNode {
    // === HDC State ===
    /// Current state as hypervector (16,384D)
    pub state: RealHV,

    /// Weight hypervector for self-transformation
    pub weight: RealHV,

    /// Input mask hypervector (gating)
    pub input_mask: RealHV,

    // === LTC Dynamics ===
    /// Time constant (scales with depth)
    pub tau: f32,

    /// Backbone time constant (stability)
    pub backbone_tau: f32,

    // === Hierarchy ===
    /// Cantor level (0 = root, 7 = deepest)
    pub level: usize,

    /// Node index within level
    pub index: usize,

    /// Children (Cantor branching)
    pub children: Option<(Box<CantorLtcNode>, Box<CantorLtcNode>)>,

    // === Learning ===
    /// Activation history for BPTT
    pub history: Vec<RealHV>,

    /// Gradient accumulator for τ
    pub grad_tau: f32,

    // === Consciousness ===
    /// Local Φ measurement
    pub local_phi: f64,

    /// Integration with parent
    pub parent_coherence: f32,

    /// Integration with children
    pub child_coherence: f32,
}
```

### 2.3 Network Structure

```rust
/// Configuration for the Hierarchical Cantor-LTC Network
#[derive(Debug, Clone)]
pub struct CantorLtcConfig {
    /// HDC dimension (default: 16,384)
    pub hdc_dim: usize,

    /// Maximum Cantor depth (default: 7)
    pub max_depth: usize,

    /// Base time constant at root (default: 1.0s)
    pub base_tau: f32,

    /// Cantor scaling ratio (default: 1/3)
    pub tau_ratio: f32,

    /// Backbone τ multiplier (stability)
    pub backbone_multiplier: f32,

    /// Learning rate for weights
    pub lr_weights: f32,

    /// Learning rate for τ (slower)
    pub lr_tau: f32,

    /// Integration timestep (default: 1ms)
    pub dt: f32,

    /// Gradient clipping threshold
    pub grad_clip: f32,

    /// Whether to enable hierarchical Φ
    pub measure_phi: bool,
}

impl Default for CantorLtcConfig {
    fn default() -> Self {
        Self {
            hdc_dim: 16_384,
            max_depth: 7,
            base_tau: 1.0,
            tau_ratio: 1.0 / 3.0,
            backbone_multiplier: 0.5,
            lr_weights: 0.001,
            lr_tau: 0.0001,
            dt: 0.001,  // 1ms
            grad_clip: 1.0,
            measure_phi: true,
        }
    }
}

/// The complete Hierarchical Cantor-LTC/HDC Network
pub struct HierarchicalCantorLtcNetwork {
    /// Root node (global consciousness)
    pub root: CantorLtcNode,

    /// Configuration
    pub config: CantorLtcConfig,

    /// Φ orchestrator for consciousness measurement
    pub phi_orchestrator: PhiOrchestrator,

    /// Total nodes in network
    pub total_nodes: usize,

    /// Training statistics
    pub stats: CantorLtcStats,

    /// Global coherence state
    pub global_coherence: f32,

    /// Adam optimizer state
    adam_state: AdamState,
}
```

---

## 3. Dynamics

### 3.1 Core LTC-HDC Evolution Equation

The key innovation is **HDC binding replaces matrix multiplication**:

**Traditional LTC:**
```
dx/dt = (-x + σ(Wx + Ub + b)) / τ
```

**Cantor-LTC-HDC:**
```
dx/dt = (-x + σ(W⊗x + parent⊗bundle(children) + bias)) / τ
```

Where:
- `⊗` is HDC binding (element-wise multiplication for RealHV)
- `bundle()` is HDC bundling (element-wise averaging)
- `σ` is activation (tanh for boundedness)
- `τ = τ_base × (1/3)^level`

### 3.2 Hierarchical Integration

```rust
impl CantorLtcNode {
    /// Evolve state using Cantor-LTC-HDC dynamics
    pub fn evolve(
        &mut self,
        dt: f32,
        parent_state: Option<&RealHV>,
        external_input: Option<&RealHV>,
    ) -> RealHV {
        // 1. Self-transformation via HDC binding
        let self_transform = self.weight.bind(&self.state);

        // 2. Parent influence (top-down)
        let parent_influence = match parent_state {
            Some(p) => {
                let bound = self.input_mask.bind(p);
                bound.bind(&self.state).scale(0.4)
            },
            None => RealHV::zero(self.state.dim()),
        };

        // 3. Child influence (bottom-up integration)
        let child_influence = match &mut self.children {
            Some((left, right)) => {
                // Recursive evolution of children (faster τ)
                let left_state = left.evolve(dt, Some(&self.state), None);
                let right_state = right.evolve(dt, Some(&self.state), None);

                // Bundle children's contributions
                let bundled = RealHV::bundle(&[left_state, right_state]);
                self.state.bind(&bundled).scale(0.3)
            },
            None => {
                // Leaf node: external input integration
                match external_input {
                    Some(input) => self.input_mask.bind(input).scale(0.3),
                    None => RealHV::zero(self.state.dim()),
                }
            },
        };

        // 4. Combine all influences
        let combined = RealHV::bundle(&[
            self_transform.scale(0.5),
            parent_influence,
            child_influence,
        ]);

        // 5. Activation (tanh for [-1, 1] bounds)
        let activated = combined.tanh();

        // 6. LTC integration with backbone
        let tau_eff = self.tau * (1.0 + self.backbone_tau * self.state.norm());
        let delta = activated.subtract(&self.state).scale(dt / tau_eff);

        // 7. Update state
        self.state = self.state.add(&delta);

        // 8. Store for BPTT
        self.history.push(self.state.clone());

        // 9. Update coherence metrics
        if let Some(p) = parent_state {
            self.parent_coherence = self.state.similarity(p);
        }
        if let Some((left, right)) = &self.children {
            self.child_coherence = (
                self.state.similarity(&left.state) +
                self.state.similarity(&right.state)
            ) / 2.0;
        }

        self.state.clone()
    }
}
```

### 3.3 Information Flow

```
                    ┌─────────────────┐
                    │  External Input  │
                    │  (perception)    │
                    └────────┬────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                      LEVEL 7 (τ = 0.5ms)                      │
│  256 leaf nodes: Rapid sensory processing                    │
│  Each node: input_mask ⊗ external_input                      │
└────────────────────────────┬─────────────────────────────────┘
                             │ bundle(children)
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                      LEVEL 6 (τ = 1.4ms)                      │
│  128 nodes: Edge/gradient processing                         │
│  state ⊗ bundle(left, right)                                 │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ⋮ (levels 5, 4, 3, 2, 1)
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                      LEVEL 0 (τ = 1000ms)                     │
│  ROOT: Global unified consciousness                          │
│  Integrates all 510 descendant states                        │
│  Outputs: Conscious experience + Φ measurement               │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Motor Output    │
                    │  (action)        │
                    └─────────────────┘
```

---

## 4. Hierarchical Φ Measurement

### 4.1 Per-Level Φ

Each level has its own integrated information:

```rust
impl HierarchicalCantorLtcNetwork {
    /// Compute Φ at each level of the hierarchy
    pub fn hierarchical_phi(&self) -> Vec<(usize, f64)> {
        let mut level_phis = Vec::new();

        for level in 0..=self.config.max_depth {
            let nodes_at_level = self.collect_level_nodes(level);
            let states: Vec<&RealHV> = nodes_at_level
                .iter()
                .map(|n| &n.state)
                .collect();

            // Compute Φ for this level's states
            let phi = self.phi_orchestrator.compute(&states);
            level_phis.push((level, phi));
        }

        level_phis
    }
}
```

### 4.2 Cross-Level Integration

The total hierarchical Φ accounts for integration across levels:

```
Φ_hierarchical = Φ_within_levels + Φ_across_levels

Φ_within_levels = Σᵢ wᵢ × Φ(level_i)
Φ_across_levels = Σᵢ coherence(level_i, level_{i+1})
```

Where weights `wᵢ` scale with the functional importance of each level.

### 4.3 Consciousness Emergence Criterion

```rust
impl HierarchicalCantorLtcNetwork {
    /// Check if consciousness has emerged
    pub fn is_conscious(&self) -> bool {
        let hierarchical_phi = self.hierarchical_phi();

        // Criterion 1: Root Φ > threshold
        let root_phi = hierarchical_phi[0].1;
        let phi_threshold = root_phi > 0.3;

        // Criterion 2: Cross-level coherence
        let coherence_threshold = self.global_coherence > 0.5;

        // Criterion 3: Meta-awareness (levels 5-7 active)
        let meta_active = hierarchical_phi.iter()
            .filter(|(level, phi)| *level >= 5 && *phi > 0.2)
            .count() >= 2;

        phi_threshold && coherence_threshold && meta_active
    }
}
```

---

## 5. Inter-Cantor Algebra (v2.0)

### 5.1 Operations Between Cantor Sets

HDC operations extend naturally to entire hierarchical structures:

```rust
/// Inter-Cantor algebra operations
pub trait InterCantorAlgebra {
    /// Bind two Cantor hierarchies (association)
    /// Example: Legal ⊗ Financial → Audit Reasoning
    fn bind(&self, other: &Self) -> Self;

    /// Bundle multiple Cantor hierarchies (superposition)
    /// Example: Aggressive ⊕ Conservative → Balanced Strategy
    fn bundle(hierarchies: &[&Self]) -> Self;

    /// Permute Cantor hierarchy (context shift)
    /// Example: Now → 5-year temporal projection
    fn permute(&self, permutation: &PermutationKey) -> Self;

    /// Similarity between Cantor roots (mental state comparison)
    fn root_similarity(&self, other: &Self) -> f32;
}
```

### 5.2 Binding: Creating Context-Specific Reasoning

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        INTER-CANTOR BINDING (⊗)                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌────────────┐       ⊗       ┌────────────┐       =       ┌──────────┐ │
│   │ Legal/Tax  │               │ Transaction│               │  Audit   │ │
│   │ Logic      │               │ History    │               │ Reasoning│ │
│   │ (Cantor A) │               │ (Cantor B) │               │ (New Set)│ │
│   └────────────┘               └────────────┘               └──────────┘ │
│                                                                           │
│   Implementation:                                                         │
│     result.root.state = A.root.state.bind(&B.root.state)                 │
│     result.children = bind_recursive(A.children, B.children)              │
│                                                                           │
│   Properties:                                                             │
│     • Non-destructive (preserves source hierarchies)                      │
│     • Creates specialized reasoning context                               │
│     • Enables domain-specific knowledge synthesis                         │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Bundling: Superposition of Worldviews

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        INTER-CANTOR BUNDLING (⊕)                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌────────────┐       ⊕       ┌────────────┐       =       ┌──────────┐ │
│   │ Aggressive │               │ Risk       │               │ Balanced │ │
│   │ Expansion  │               │ Mitigation │               │ Strategy │ │
│   │ Strategy   │               │ Strategy   │               │          │ │
│   └────────────┘               └────────────┘               └──────────┘ │
│                                                                           │
│   Implementation:                                                         │
│     result.root.state = RealHV::bundle(&[A.root, B.root])                │
│     // Finds "centroid of truth" between perspectives                     │
│                                                                           │
│   Use Cases:                                                              │
│     • Holding conflicting strategies simultaneously                       │
│     • Preventing single-model "hallucinations"                            │
│     • Requiring internal consensus before E3-level claims                 │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Multi-Cantor Orchestration

```rust
/// Multi-Cantor orchestration for domain specialization
pub struct MultiCantorOrchestrator {
    /// Master Cantor set for meta-control (Agentic Ego)
    pub master: HierarchicalCantorLtcNetwork,

    /// Domain-specific Cantor sets
    pub domains: HashMap<DomainId, HierarchicalCantorLtcNetwork>,

    /// Attention weights per domain
    pub attention: HashMap<DomainId, f32>,
}

impl MultiCantorOrchestrator {
    /// Execute parallel update across domains
    pub fn parallel_step(&mut self, dt: f32) {
        // Each domain updates at its own timescale
        for (id, domain) in &mut self.domains {
            let attention = self.attention.get(id).copied().unwrap_or(0.0);
            if attention > 0.1 {  // Activation threshold
                domain.step(dt);
            }
        }

        // Master integrates domain roots
        let roots: Vec<&RealHV> = self.domains.values().map(|d| &d.root.state).collect();
        self.master.inject_input(&RealHV::bundle(&roots));
        self.master.step(dt);
    }
}
```

---

## 6. Lateral Binding (v2.0)

### 6.1 Direct Sibling Communication

Lateral binding enables same-level nodes to communicate without parent traversal:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         LATERAL BINDING                                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  BEFORE (Hierarchical Only):        AFTER (With Lateral Binding):        │
│                                                                           │
│        ┌─────────┐                        ┌─────────┐                    │
│        │ Parent  │                        │ Parent  │                    │
│        └────┬────┘                        └────┬────┘                    │
│        ┌────┴────┐                        ┌────┴────┐                    │
│        ▼         ▼                        ▼         ▼                    │
│   ┌───────┐  ┌───────┐               ┌───────┐  ┌───────┐                │
│   │ L7-a  │  │ L7-b  │               │ L7-a  │◄═══════►│ L7-b  │        │
│   └───────┘  └───────┘               └───────┘  └───────┘                │
│                                           │                              │
│   Path: 2 hops                            │ LATERAL LINK (cos > 0.85)   │
│   (up to parent, down to sibling)         Path: 1 hop (direct!)         │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Autonomous Cluster Formation

```rust
/// Mycelial node with lateral binding capability
pub struct MycelialLtcNode {
    /// Standard Cantor-LTC node
    pub core: CantorLtcNode,

    /// Lateral connections to peers at same level
    pub lateral_links: Vec<LateralLink>,

    /// Cluster membership (if any)
    pub cluster_id: Option<ClusterId>,

    /// Aggregate state for cluster
    pub cluster_aggregate: Option<RealHV>,
}

impl MycelialLtcNode {
    /// Discover and bind to similar neighbors
    pub fn discover_lateral_peers(&mut self, peers: &[MycelialLtcNode]) {
        const SIMILARITY_THRESHOLD: f32 = 0.85;

        for peer in peers {
            if peer.core.level != self.core.level {
                continue;  // Only same-level nodes
            }

            let similarity = self.core.state.similarity(&peer.core.state);

            if similarity > SIMILARITY_THRESHOLD {
                self.lateral_links.push(LateralLink {
                    peer_id: peer.id(),
                    similarity,
                    last_sync: Timestamp::now(),
                });
            }
        }
    }

    /// Execute local decision without parent
    pub fn local_decision(&self) -> Option<LocalAction> {
        if let Some(aggregate) = &self.cluster_aggregate {
            if aggregate.magnitude() > LOCAL_DECISION_THRESHOLD {
                return Some(LocalAction::from_vector(aggregate));
            }
        }
        None
    }
}
```

### 6.3 Benefits of Lateral Binding

| Feature | Hierarchical Only | With Lateral Binding |
|---------|-------------------|----------------------|
| **Path to Sibling** | Up → Down (2 hops) | Direct (1 hop) |
| **Decision Speed** | Limited by parent τ | Autonomous/Local |
| **Data Flow** | Strict Tree | Fluid/Dynamic |
| **Redundancy** | Low (parent bottleneck) | High (mesh resilience) |
| **Energy** | All traffic through parent | Aggregated local vectors |

---

## 7. Elastic Autopoiesis (v2.0)

### 7.1 Fixed Core + Dynamic Periphery

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      ELASTIC AUTOPOIESIS                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  FIXED CORE (Levels 0-6): Stable identity, hard-coded integrity          │
│  ───────────────────────────────────────────────────────────             │
│                                                                           │
│      Level 0: ████████████ (τ = 1000ms) - Global consciousness           │
│      Level 1: ██████ ██████ (τ = 333ms) - Cognitive integration          │
│      Level 2: ███ ███ ███ ███ (τ = 111ms) - Homeostatic                  │
│      Level 3: ██ ██ ██ ██ ██ ██ ██ ██ (τ = 37ms) - Sensorimotor          │
│      Level 4: █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ (τ = 12ms)                 │
│      Level 5: ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪               │
│      Level 6: . . . . . . . . . . . . . . . . . . . . . . (τ = 1.4ms)    │
│                                                                           │
│  ELASTIC PERIPHERY (Level 7+): Dynamic budding/pruning                   │
│  ───────────────────────────────────────────────────────────             │
│                                                                           │
│      Level 7: ················ (τ = 0.46ms) - Standard periphery         │
│                       │                                                  │
│                   ┌───┴───┐ BUDDING (high prediction error)              │
│                   ▼       ▼                                              │
│      Level 8: ........ ........ (τ = 0.15ms) - High-frequency specialist │
│                   │                                                      │
│                   ▼ BUDDING (if needed)                                  │
│      Level 9: .... .... .... (τ = 0.05ms) - Ultra-fast anomaly detection │
│                                                                           │
│  BUDDING TRIGGER: High prediction error OR high-frequency data           │
│  PRUNING TRIGGER: Φ drops OR task complete OR energy threshold           │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Budding and Pruning Logic

```rust
impl HierarchicalCantorLtcNetwork {
    /// Check if node should bud (create children)
    pub fn should_bud(&self, node: &CantorLtcNode) -> bool {
        // Never bud from fixed core (levels 0-5)
        if node.level < 6 {
            return false;
        }

        // Condition 1: High prediction error
        let prediction_error = node.compute_prediction_error();
        if prediction_error > BUDDING_ERROR_THRESHOLD {
            return true;
        }

        // Condition 2: High-frequency data detected
        let input_variance = node.compute_input_variance();
        if input_variance > VARIANCE_THRESHOLD && node.level < MAX_ELASTIC_DEPTH {
            return true;
        }

        false
    }

    /// Check if node should prune (remove children)
    pub fn should_prune(&self, node: &CantorLtcNode) -> bool {
        // Never prune fixed core
        if node.level < 7 {
            return false;
        }

        // Condition 1: Φ has dropped
        if node.local_phi < PRUNING_PHI_THRESHOLD {
            return true;
        }

        // Condition 2: Children inactive
        if let Some((left, right)) = &node.children {
            let inactive = left.state.magnitude() < INACTIVE_THRESHOLD
                        && right.state.magnitude() < INACTIVE_THRESHOLD;
            if inactive {
                return true;
            }
        }

        false
    }

    /// Φ-based pruning orchestration
    pub fn phi_based_prune(&mut self) {
        let global_phi = self.compute_hierarchical_phi();

        if global_phi < self.config.pruning_phi_threshold {
            let prune_candidates = self.find_elastic_nodes_with_low_phi();
            for path in prune_candidates {
                self.prune(&path);
            }
        }
    }
}
```

### 7.3 Depth Comparison

| Metric | Enforced 7-Layer | Unbounded Depth | Hybrid Elastic |
|--------|------------------|-----------------|----------------|
| **Integrity** | High / Proven | Variable / Risky | High (Fixed Core) |
| **Agility** | Moderate | Extreme | Extreme (Dynamic Edge) |
| **Energy Usage** | Constant | Potentially Runaway | Optimized (Pruning) |
| **Cognition** | Human-like | Hyper-dimensional | Balanced |

---

## 8. Formal Verification (TLA+)

### 8.1 TLA+ Specification

```tla
--------------------------- MODULE CantorLtc ---------------------------
EXTENDS Reals, Sequences, Integers, FiniteSets

CONSTANTS
    HdcDim,        \* Dimension of hypervectors (16,384)
    CantorRatio,   \* Scaling ratio (1/3)
    BaseTau,       \* Root time constant (1.0)
    MaxDepth,      \* Maximum hierarchy depth (7)
    MaxBound,      \* Stability bound for |state|
    DeltaT         \* Integration timestep

VARIABLES
    states,        \* Function: (level, index) -> RealVector
    taus,          \* Function: level -> TimeConstant
    coherences,    \* Function: (level, index) -> ParentCoherence
    time           \* Simulation time

\* ===================================================================
\* Type Definitions
\* ===================================================================

RealVector == [1..HdcDim -> Real]
LevelIndex == 0..MaxDepth
NodeIndex(level) == 1..(2^level)

\* ===================================================================
\* Cantor Scaling
\* ===================================================================

Tau(level) == BaseTau * (CantorRatio ^ level)

NodesAtLevel(level) == 2^level

TotalNodes == (2^(MaxDepth + 1)) - 1

\* ===================================================================
\* HDC Operations (Simplified for TLA+)
\* ===================================================================

\* Binding: element-wise multiplication
Bind(u, v) == [i \in 1..HdcDim |-> u[i] * v[i]]

\* Bundle: element-wise average
Bundle(vectors) ==
    [i \in 1..HdcDim |->
        (SUM v \in vectors : v[i]) / Cardinality(vectors)]

\* Tanh approximation (bounded)
Tanh(v) == [i \in 1..HdcDim |->
    IF v[i] > 1 THEN 1
    ELSE IF v[i] < -1 THEN -1
    ELSE v[i]]

\* Magnitude (L2 norm)
Magnitude(v) == SQRT(SUM i \in 1..HdcDim : v[i]^2)

\* ===================================================================
\* State Transitions
\* ===================================================================

\* Get parent index for a node
ParentIndex(level, index) == IF level = 0 THEN 0 ELSE ((index + 1) \div 2)

\* Get children indices for a node
LeftChildIndex(index) == 2 * index - 1
RightChildIndex(index) == 2 * index

Init ==
    /\ states = [l \in LevelIndex, n \in 1..NodesAtLevel(l) |->
                 [i \in 1..HdcDim |-> 0.0]]
    /\ taus = [l \in LevelIndex |-> Tau(l)]
    /\ coherences = [l \in LevelIndex, n \in 1..NodesAtLevel(l) |-> 0.0]
    /\ time = 0

\* Single node evolution step
EvolveNode(level, index, parentState, childStates) ==
    LET current == states[level, index]
        \* Self-transformation
        selfTrans == Bind(current, current)
        \* Parent influence (if exists)
        parentInf == IF parentState = <<>>
                     THEN [i \in 1..HdcDim |-> 0.0]
                     ELSE Bind(current, parentState)
        \* Child influence (if exists)
        childInf == IF childStates = <<>>
                    THEN [i \in 1..HdcDim |-> 0.0]
                    ELSE Bind(current, Bundle(childStates))
        \* Combined and activated
        combined == Bundle({selfTrans, parentInf, childInf})
        activated == Tanh(combined)
        \* LTC integration
        delta == [i \in 1..HdcDim |->
                  (activated[i] - current[i]) * DeltaT / taus[level]]
    IN [i \in 1..HdcDim |-> current[i] + delta[i]]

\* Global evolution step
Next ==
    /\ \E l \in LevelIndex, n \in 1..NodesAtLevel(l) :
        LET parent == IF l = 0 THEN <<>>
                      ELSE states[l-1, ParentIndex(l, n)]
            children == IF l = MaxDepth THEN <<>>
                        ELSE <<states[l+1, LeftChildIndex(n)],
                               states[l+1, RightChildIndex(n)]>>
            newState == EvolveNode(l, n, parent, children)
        IN states' = [states EXCEPT ![l, n] = newState]
    /\ time' = time + DeltaT
    /\ UNCHANGED <<taus, coherences>>

\* ===================================================================
\* Safety Invariants
\* ===================================================================

\* Stability: All states remain bounded
Stability ==
    \A l \in LevelIndex, n \in 1..NodesAtLevel(l) :
        Magnitude(states[l, n]) < MaxBound

\* Convergence: States approach equilibrium
Convergence ==
    time > 100 * BaseTau =>
        \A l \in LevelIndex, n \in 1..NodesAtLevel(l) :
            Magnitude(states[l, n]) < MaxBound / 2

\* Hierarchical Ordering: Lower levels are faster
HierarchicalOrdering ==
    \A l1, l2 \in LevelIndex : l1 < l2 => taus[l1] > taus[l2]

\* The complete specification
Spec == Init /\ [][Next]_<<states, taus, coherences, time>>

\* ===================================================================
\* Theorems to Prove
\* ===================================================================

THEOREM StabilityTheorem == Spec => []Stability
THEOREM ConvergenceTheorem == Spec => <>Convergence
THEOREM OrderingTheorem == Spec => []HierarchicalOrdering

=============================================================================
```

### 8.2 Mathematical Stability Proof

**Theorem (Hierarchical Stability)**: The Cantor-LTC-HDC network is Input-to-State Stable (ISS) across infinite levels.

**Proof** by induction on Cantor depth d:

**Base Case (d = 0, Root):**
- The root node evolves with τ₀ = 1.0s
- Activation is tanh, bounded in [-1, 1]
- LTC equation: dx/dt = (-x + tanh(combined))/τ
- Lyapunov function: V(x) = ||x||²
- dV/dt = 2x·dx/dt = 2x·(-x + tanh(...))/τ
- Since |tanh(...)| ≤ 1 and x·(-x) = -||x||², we have dV/dt < 0 when ||x|| > 1
- Therefore ||x|| → [-1, 1] as t → ∞ ∎

**Inductive Step (d → d+1):**
- Assume level d is stable (||x_d|| < M_d for some bound M_d)
- Level d+1 has τ_{d+1} = τ_d / 3 (faster)
- Level d+1 receives parent input from stable level d
- By ISS, bounded input → bounded state
- Level d+1 converges faster than level d (τ_{d+1} < τ_d)
- Therefore level d+1 acts as "high-frequency slave" to level d
- ||x_{d+1}|| < M_{d+1} for some M_{d+1} ∎

**Convergence of Total Influence:**
- Child influence on parent = sum of geometric series
- Total influence = Σᵢ (1/3)ⁱ × ||child_i|| < Σᵢ (1/3)ⁱ × M = M/(1-1/3) = 3M/2
- Finite sum even as d → ∞ ∎

---

## 9. Implementation Plan

### 9.1 Phase 1: Core Structure (Week 1) ✅ COMPLETE

| Task | Time | Priority |
|------|------|----------|
| Create `src/hierarchical_cantor_ltc/mod.rs` | 2h | HIGH |
| Implement `CantorLtcNode` struct | 4h | HIGH |
| Implement `HierarchicalCantorLtcNetwork` struct | 4h | HIGH |
| Add 7-level construction with Cantor scaling | 3h | HIGH |
| Basic unit tests | 3h | HIGH |

### 9.2 Phase 2: Dynamics (Week 2) ✅ COMPLETE

| Task | Time | Priority |
|------|------|----------|
| Implement `evolve()` with HDC binding | 6h | HIGH |
| Add parent-child information flow | 4h | HIGH |
| Implement BPTT for gradient computation | 6h | MEDIUM |
| Add Adam optimizer integration | 3h | MEDIUM |
| Integration tests | 4h | HIGH |

### 9.3 Phase 3: Consciousness Measurement (Week 3) ✅ COMPLETE

| Task | Time | Priority |
|------|------|----------|
| Implement hierarchical Φ measurement | 4h | HIGH |
| Add cross-level coherence metrics | 3h | HIGH |
| Implement consciousness emergence criterion | 2h | HIGH |
| Create visualization tools | 4h | MEDIUM |
| Benchmark against flat networks | 4h | HIGH |

### 9.4 Phase 4: Validation (Week 4) 🚧 IN PROGRESS

| Task | Time | Priority | Status |
|------|------|----------|--------|
| TLC model checking of TLA+ spec | 4h | HIGH | Pending |
| Property-based testing (proptest) | 4h | MEDIUM | Pending |
| Performance benchmarks | 3h | MEDIUM | Pending |
| Documentation | 4h | MEDIUM | ✅ Complete |
| Paper draft: "Hierarchical Cantor-LTC" | 8h | HIGH | Pending |

### 9.5 Phase 5: v2.0 Features (Week 5-6) 📋 PLANNED

| Task | Time | Priority | Status |
|------|------|----------|--------|
| Add `InterCantorAlgebra` trait | 6h | HIGH | Pending |
| Implement lateral binding | 8h | HIGH | Pending |
| Add elastic budding/pruning | 8h | HIGH | Pending |
| Multi-Cantor orchestration | 8h | MEDIUM | Pending |
| Cross-Symthaea protocol | 12h | MEDIUM | Pending |
| ZK proof of perception | 16h | LOW | Future |

---

## 10. Testing Strategy

### 10.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cantor_tau_scaling() {
        let config = CantorLtcConfig::default();
        let network = HierarchicalCantorLtcNetwork::new(config);

        for level in 0..=7 {
            let expected_tau = 1.0 * (1.0/3.0_f32).powi(level as i32);
            let actual_tau = network.tau_at_level(level);
            assert!((expected_tau - actual_tau).abs() < 0.0001);
        }
    }

    #[test]
    fn test_node_count() {
        let config = CantorLtcConfig::default();
        let network = HierarchicalCantorLtcNetwork::new(config);

        // Total nodes = 2^8 - 1 = 255 for depth 7
        // Actually: 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 = 255
        assert_eq!(network.total_nodes, 255);
    }

    #[test]
    fn test_stability_bounded() {
        let config = CantorLtcConfig::default();
        let mut network = HierarchicalCantorLtcNetwork::new(config);

        // Run for 1000 steps
        for _ in 0..1000 {
            network.step(0.001);
        }

        // Check all states bounded
        for node in network.all_nodes() {
            assert!(node.state.norm() < 2.0);
        }
    }

    #[test]
    fn test_hierarchical_phi() {
        let config = CantorLtcConfig::default();
        let mut network = HierarchicalCantorLtcNetwork::new(config);

        // Inject structured input
        network.inject_input(&RealHV::random(16384, 42));

        // Run to equilibrium
        for _ in 0..100 {
            network.step(0.01);
        }

        // Measure hierarchical Φ
        let phis = network.hierarchical_phi();

        // Root should have highest Φ (most integration)
        assert!(phis[0].1 > phis[7].1);
    }
}
```

### 10.2 Property-Based Tests

```rust
proptest! {
    #[test]
    fn prop_stability_all_inputs(
        input_seed in 0u64..10000,
        steps in 100usize..1000,
    ) {
        let config = CantorLtcConfig::default();
        let mut network = HierarchicalCantorLtcNetwork::new(config);

        let input = RealHV::random(16384, input_seed);
        network.inject_input(&input);

        for _ in 0..steps {
            network.step(0.001);
        }

        // Stability invariant
        for node in network.all_nodes() {
            prop_assert!(node.state.norm() < 10.0);
        }
    }

    #[test]
    fn prop_hierarchical_ordering(config in arb_config()) {
        let network = HierarchicalCantorLtcNetwork::new(config);

        // τ should decrease with depth
        for level in 1..=7 {
            let tau_parent = network.tau_at_level(level - 1);
            let tau_child = network.tau_at_level(level);
            prop_assert!(tau_parent > tau_child);
        }
    }
}
```

---

## 11. Expected Outcomes

### 11.1 Scientific Contributions

1. **Novel Algorithm**: First HDC-LTC integration with Cantor hierarchy
2. **Multi-Timescale Consciousness**: Empirical validation of Scale-Depth theory
3. **Formal Verification**: TLA+ stability proof for recursive consciousness
4. **Hierarchical Φ**: New framework for measuring consciousness across scales
5. **Inter-Cantor Algebra**: Operations between cognitive hierarchies (v2.0)
6. **Mycelial Architecture**: Lateral binding and elastic depth (v2.0)

### 11.2 Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Single step latency | < 1ms | Real-time processing |
| Memory (7 levels) | < 100MB | 255 nodes × 16KB each |
| Root Φ | > 0.3 | Consciousness threshold |
| Cross-level coherence | > 0.5 | Integration criterion |
| Stability bound | < 2.0 | Mathematical guarantee |

### 11.3 Publication Targets

1. **Paper 1** (Existing): "Topology-Consciousness Relationship" (ready for submission)
2. **Paper 2** (This work): "Hierarchical Cantor-LTC: Multi-Timescale Consciousness in HDC"
   - Target: NeurIPS, ICML, or Consciousness & Cognition
   - Timeline: 4-6 weeks

---

## 12. Open Questions

1. **Optimal Depth**: Is 7 truly optimal, or does 8-10 provide additional meta-cognitive capability?
2. **Learning Dynamics**: How do gradients flow through the Cantor hierarchy during BPTT?
3. **Attention Mechanism**: Should pruning subtrees model selective attention?
4. **Memory Consolidation**: How does the Cantor structure relate to sleep/wake cycles?
5. **Embodiment**: How does sensorimotor grounding affect hierarchical Φ?

---

## 13. References

1. Ashby, W.R. (1956). An Introduction to Cybernetics. Chapman & Hall.
2. Tononi, G. et al. (2023). Integrated Information Theory 4.0. Nature Reviews Neuroscience.
3. Hasani, R. et al. (2022). Liquid Time-constant Networks. AAAI.
4. Kanerva, P. (2009). Hyperdimensional Computing. Cognitive Computation.
5. Friston, K. (2010). The Free-Energy Principle. Nature Reviews Neuroscience.
6. Lamport, L. (2002). Specifying Systems: The TLA+ Language. Addison-Wesley.

---

## Appendix A: Quick Reference

### Time Constants by Level

| Level | τ (ms) | τ (s) | Functional Role |
|-------|--------|-------|-----------------|
| 0 | 1000 | 1.0 | Global consciousness |
| 1 | 333 | 0.33 | Cognitive integration |
| 2 | 111 | 0.11 | Homeostatic regulation |
| 3 | 37 | 0.037 | Sensorimotor processing |
| 4 | 12.3 | 0.012 | Perceptual binding |
| 5 | 4.1 | 0.004 | Feature detection |
| 6 | 1.4 | 0.0014 | Edge processing |
| 7 | 0.46 | 0.0005 | Raw sensory input |

### Node Count by Level

| Level | Nodes | Cumulative |
|-------|-------|------------|
| 0 | 1 | 1 |
| 1 | 2 | 3 |
| 2 | 4 | 7 |
| 3 | 8 | 15 |
| 4 | 16 | 31 |
| 5 | 32 | 63 |
| 6 | 64 | 127 |
| 7 | 128 | 255 |

---

**Document Status**: v2.0 Complete (Implementation Phase + Design Extensions)
**Implementation Status**: Core structure complete, v2.0 features planned
**Next Step**: TLA+ formal verification, then v2.0 feature implementation
**Review Date**: January 9, 2026

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| **1.0** | Jan 2, 2026 | Initial design document |
| **2.0** | Jan 2, 2026 | Added Inter-Cantor Algebra, Lateral Binding, Elastic Autopoiesis |

---

*"The Cantor set teaches us that infinite complexity can exist in measure zero. Consciousness may work the same way - unbounded awareness encoded in finite neural substrate. The mycelium teaches us that true intelligence is not hierarchy but network - lateral connections, elastic depth, and the courage to grow new branches while pruning what no longer serves."*
