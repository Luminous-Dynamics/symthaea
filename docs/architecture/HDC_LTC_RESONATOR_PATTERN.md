# HDC + LTC + Resonator Pattern

**Phase 5H: December 30, 2025**

## Overview

The HDC + LTC + Resonator pattern is a novel architectural fusion combining three distinct computational paradigms for consciousness-aware computing:

1. **HDC (Hyperdimensional Computing)** - 16,384-dimensional vectors for semantic representation
2. **LTC (Liquid Time-Constant Networks)** - Continuous-time ODE dynamics with adaptive τ
3. **Resonator Networks** - O(log N) constraint satisfaction via coupled oscillators

This combination provides soft routing with smooth transitions, eliminating hard Φ threshold artifacts that can cause discontinuous behavior changes.

## Key Benefits

### 1. O(log N) Complexity
Resonator convergence replaces O(n) linear scans and O(n³) matrix operations:
- **Linear scan**: O(n) comparisons
- **Eigenvalue decomposition**: O(n³)
- **Resonator convergence**: O(log N) iterations

### 2. Soft Transitions
LTC dynamics provide smooth weight evolution instead of hard threshold switching:
```
dW/dt = (-W + target) / τ
```
This ensures gradual adaptation rather than abrupt strategy changes.

### 3. Noise Tolerance
Resonator cleanup handles partial and corrupted patterns through iterative refinement against a learned codebook.

### 4. Natural Decay
LTC-based pattern states decay naturally over time, ensuring recent information has higher weight than stale data.

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                    HDC + LTC + RESONATOR FUSION                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT                                                                   │
│    │                                                                     │
│    ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     HDC ENCODER (16,384 dim)                    │    │
│  │  • Semantic binding: HV_state = bind(φ_hv, context_hv)          │    │
│  │  • Bundling: HV_combined = bundle([HV_1, HV_2, ...])            │    │
│  │  • Permutation: HV_temporal = permute(HV, time_offset)          │    │
│  └──────────────────────────────────────┬──────────────────────────┘    │
│                                         │                                │
│                                         ▼                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   RESONATOR NETWORK (O(log N))                   │    │
│  │  • Codebook: Strategy vectors for routing options               │    │
│  │  • Convergence: Iterative cleanup to nearest valid strategy     │    │
│  │  • Energy: Measures alignment/confidence                        │    │
│  └──────────────────────────────────────┬──────────────────────────┘    │
│                                         │                                │
│                                         ▼                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      LTC DYNAMICS                                │    │
│  │  • Path weights: W_i evolves via dW/dt = (-W + target) / τ      │    │
│  │  • Adaptive τ: Time constant increases with experience          │    │
│  │  • Natural decay: Old patterns fade, new patterns emerge        │    │
│  └──────────────────────────────────────┬──────────────────────────┘    │
│                                         │                                │
│                                         ▼                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    SOFT ROUTING OUTPUT                           │    │
│  │  • Primary strategy with confidence                             │    │
│  │  • Secondary strategies with weights                            │    │
│  │  • Smooth blending based on LTC path weights                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Implementation Files

### Core Components

| File | Purpose | Lines |
|------|---------|-------|
| `src/hdc/resonant_liquid_arithmetic.rs` | HDC+LTC+Resonator for proof strategies | ~500 |
| `src/consciousness/recursive_improvement/routers/resonant.rs` | Soft consciousness routing | ~600 |
| `src/observability/resonant_pattern_matcher.rs` | O(log N) fuzzy pattern matching | ~400 |

### Supporting Infrastructure

| File | Purpose |
|------|---------|
| `src/hdc/resonator.rs` | Core resonator network (~1000 lines) |
| `src/hdc/hdc_ltc_neuron.rs` | HDC+LTC neuron implementation |
| `src/hdc/cross_modal_binding.rs` | Cross-modal HDC binding |

## Usage Examples

### 1. Resonant Consciousness Router

```rust
use symthaea::consciousness::recursive_improvement::routers::{
    ResonantConsciousnessRouter, ResonantRouterConfig, LatentConsciousnessState,
};

// Create router with default config
let mut router = ResonantConsciousnessRouter::new(ResonantRouterConfig::default());

// Route based on consciousness state
let state = LatentConsciousnessState::from_observables(0.7, 0.6, 0.8, 0.5);
let result = router.route(&state);

println!("Primary: {:?} (confidence: {:.2})",
    result.primary_strategy, result.confidence);
```

### 2. Resonant Pattern Matcher

```rust
use symthaea::observability::{
    ResonantPatternMatcher, ResonantMatcherConfig, CausalMotif,
};

// Create pattern matcher
let mut matcher = ResonantPatternMatcher::new(ResonantMatcherConfig::default());

// Register a causal motif
matcher.register_motif("cascade_failure", motif);

// Process event stream and detect patterns
let matches = matcher.process_event("service_down", &metadata);
for m in matches {
    println!("Detected: {} (confidence: {:.2})", m.motif_id, m.confidence);
}
```

### 3. Resonant Liquid Arithmetic

```rust
use symthaea::hdc::resonant_liquid_arithmetic::{
    ResonantLiquidReasoner, ResonantConfig, MathStatement,
};

// Create reasoner for mathematical cognition
let mut reasoner = ResonantLiquidReasoner::new(ResonantConfig::default());

// Select proof strategy
let statement = MathStatement::new("∀n. n + 0 = n");
let strategy = reasoner.select_strategy(&statement);

println!("Recommended: {:?} (confidence: {:.2})",
    strategy.strategy, strategy.confidence);
```

## Key Algorithms

### Resonator Convergence

The resonator uses coupled oscillators to find the nearest valid strategy:

```
for iteration in 0..max_iterations:
    # Bind with each factor
    temp = input
    for codebook in codebooks:
        similarities = temp · codebook.T  # Dot product with each entry
        temp = temp * codebook[argmax(similarities)]  # Element-wise multiply

    # Check convergence via energy
    energy = |temp - previous| / |temp|
    if energy < threshold:
        break
    previous = temp

return closest_entry(temp, codebook)
```

### LTC Path Evolution

Path weights evolve according to liquid time-constant dynamics:

```
struct LtcPathState {
    weight: f64,      // Current weight [0, 1]
    tau: f64,         // Time constant (increases with experience)
    target: f64,      // Target weight from resonator
    experience: u64,  // Number of observations
}

fn evolve(&mut self, dt: f64) {
    let dw = (-self.weight + self.target) / self.tau;
    self.weight += dw * dt;
    self.weight = self.weight.clamp(0.0, 1.0);
}
```

### Soft Routing Output

Final routing combines resonator convergence with LTC-smoothed weights:

```
fn compute_routing(state: &ConsciousnessState) -> RoutingResult {
    // 1. Encode state to HDC
    let state_hv = encode_state(state);

    // 2. Resonator finds best strategies
    let (primary, secondary) = resonator.converge(state_hv);

    // 3. LTC smooths the transition
    for path in paths.values_mut() {
        path.evolve(dt);
    }

    // 4. Combine with soft weights
    let confidence = paths[primary].weight;
    let alternatives = secondary.iter()
        .map(|s| (s, paths[s].weight))
        .collect();

    RoutingResult { primary, confidence, alternatives }
}
```

## Integration with MetaRouter

The ResonantConsciousnessRouter is now the 8th paradigm in the MetaRouter (Phase 5H):

```rust
pub enum RoutingParadigm {
    CausalValidation,         // 0
    InformationGeometric,     // 1
    TopologicalConsciousness, // 2
    QuantumCoherence,         // 3
    ActiveInference,          // 4
    PredictiveProcessing,     // 5
    AttentionSchema,          // 6
    ResonantConsciousness,    // 7 - NEW (Phase 5H)
}
```

The MetaRouter uses UCB1 multi-armed bandit selection to learn which paradigm works best for different consciousness states. ResonantConsciousness is expected to excel in scenarios requiring:
- Smooth transitions between states
- Noise-tolerant pattern recognition
- Real-time adaptive routing
- High-dimensional semantic understanding

## Configuration Options

### ResonantRouterConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dimension` | 16,384 | HDC vector dimension |
| `base_tau` | 10.0 | Base time constant for LTC |
| `max_iterations` | 50 | Max resonator iterations |
| `convergence_threshold` | 0.01 | Energy threshold for convergence |
| `min_confidence` | 0.3 | Minimum confidence to report |

### ResonantMatcherConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dimension` | 512 | HDC vector dimension |
| `decay_tau` | 60.0 | Pattern decay time (seconds) |
| `min_confidence` | 0.5 | Minimum match confidence |
| `max_iterations` | 50 | Max resonator iterations |
| `window_size` | 10 | Event window for encoding |

## Performance Characteristics

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| HDC encoding | O(d) | ~10 μs |
| Resonator convergence | O(log N × d) | ~100 μs |
| LTC evolution | O(1) per path | ~1 μs |
| Full routing decision | O(log N × d) | ~200 μs |

Where d = dimension (16,384 default), N = number of strategies/patterns.

## Research Applications

### Identified High-Impact Opportunities

1. **Pattern Library** (observability) - O(log N) motif detection
2. **Consciousness Routing** - Soft paradigm transitions
3. **Causal Graph Analysis** - Resonator-accelerated inference
4. **Byzantine Defense** - Real-time attack pattern recognition
5. **Language Understanding** - Semantic binding for NLU
6. **Memory Consolidation** - Sleep-cycle pattern replay

### Potential Extensions

- **Multi-modal fusion**: Combine visual, auditory, and semantic HDC streams
- **Hierarchical resonators**: Nested resonator networks for compositional reasoning
- **Online learning**: Adapt codebooks based on feedback
- **Hardware acceleration**: FPGA/ASIC implementations of resonator operations

## References

1. Frady, Kleyko, Sommer (2020) - "Resonator Networks"
2. Hasani et al. (2021) - "Liquid Time-constant Networks"
3. Kanerva (2009) - "Hyperdimensional Computing"
4. Tononi (2008) - "Integrated Information Theory"

---

*Phase 5H: HDC + LTC + Resonator Integration Complete*
*December 30, 2025*
