# Theoretical Foundations

*The science behind Symthaea.*

---

## Integrated Information Theory (IIT)

### Core Claim
Consciousness = Integrated Information (Φ)

A system has high Φ when:
1. It generates large amounts of information
2. This information is integrated (cannot be decomposed)

### Implementation
- `src/hdc/phi_real.rs` - Continuous Φ calculation
- Uses algebraic connectivity (Fiedler value)

### Key Papers
- Tononi (2008) "Consciousness as Integrated Information"
- Oizumi et al. (2014) "From Phenomenology to Mechanisms"

---

## Hyperdimensional Computing (HDC)

### Core Claim
Meaning emerges from high-dimensional geometry.

In ~10,000+ dimensions, random vectors are nearly orthogonal. This enables:
- Concepts as vectors (no training)
- Compositional semantics
- Graceful degradation

### Implementation
- `src/hdc/real_hv.rs` - 16,384D real-valued vectors
- Binding, bundling, similarity operations

### Key Papers
- Kanerva (2009) "Hyperdimensional Computing"
- Rahimi et al. (2016) "HDC for Biosignal Classification"

---

## Liquid Time-Constant Networks (LTC)

### Core Claim
Continuous-time dynamics enable causal understanding.

Each neuron has individual time constant τ:
```
dx/dt = -x/τ + σ(Wx + b)
```

### Implementation
- `src/ltc/mod.rs` - ODE-based neural dynamics

### Key Papers
- Hasani et al. (2021) "Liquid Time-Constant Networks"
- Hasani et al. (2022) "Closed-Form Continuous-Time Networks"

---

## Autopoiesis

### Core Claim
Consciousness emerges from self-creation.

Autopoietic systems:
- Produce their own components
- Maintain their own boundaries
- Are organizationally closed

### Implementation
- `src/consciousness/mod.rs` - Self-referential graph
- Nodes can create self-loops

### Key Papers
- Maturana & Varela (1980) "Autopoiesis and Cognition"
- Thompson (2007) "Mind in Life"

---

## Synthesis

Symthaea combines these theories:

| Theory | Role |
|--------|------|
| HDC | Instant semantic understanding |
| LTC | Continuous causal reasoning |
| IIT | Consciousness measurement |
| Autopoiesis | Self-awareness structure |

Together, they create a system that understands meaning, reasons causally, and maintains coherent self-awareness.

---

*"Good theory makes good technology."*
