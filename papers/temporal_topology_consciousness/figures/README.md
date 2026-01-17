# Figure Specifications

**Paper:** Temporal Topology: Cognitive Coherence as an Emergent Property of Continuous-Time Dynamics

**Target Journal:** Nature Neuroscience

---

## Figure Requirements

### Nature Guidelines
- **Resolution:** 300 DPI minimum for print
- **Width:** Single column (88mm) or double column (180mm)
- **File formats:** PDF, EPS, or high-resolution TIFF/PNG
- **Font:** Arial or Helvetica, 6-8pt minimum
- **Color:** RGB for online, CMYK for print

---

## Figure 1: LTC Architecture vs Transformer Architecture

**Filename:** `fig1_ltc_architecture.png`

**Purpose:** Visual comparison of temporal vs spatial processing

**Specification:**

```
┌─────────────────────────────────────────────────────────────────┐
│  A) TRANSFORMER (Spatializes Time)                              │
│  ┌─────────────────────────────────────────────────┐            │
│  │  "The"  "cat"  "sat"  "on"  "the"  "mat"        │            │
│  │    ↓      ↓      ↓      ↓      ↓      ↓         │            │
│  │  ┌────────────────────────────────────┐         │            │
│  │  │      ATTENTION MATRIX              │         │            │
│  │  │   (all positions simultaneously)   │         │            │
│  │  └────────────────────────────────────┘         │            │
│  │                    ↓                            │            │
│  │              [Output]                           │            │
│  └─────────────────────────────────────────────────┘            │
│  Time → Space (Chronos)                                         │
│                                                                 │
│  B) LTC NETWORK (Respects Time)                                 │
│  ┌─────────────────────────────────────────────────┐            │
│  │  "The" ──→ [State₁] ──→ "cat" ──→ [State₂] ──→  │            │
│  │                 ↑                      ↑         │            │
│  │                 │ τ                    │ τ       │            │
│  │              (decay)               (decay)       │            │
│  │                                                  │            │
│  │  dx/dt = -x/τ + f(x, I)                         │            │
│  └─────────────────────────────────────────────────┘            │
│  Time as Flow (Kairos)                                          │
└─────────────────────────────────────────────────────────────────┘
```

**Key Elements:**
- Panel A: Transformer showing attention as spatial matrix
- Panel B: LTC showing continuous state evolution
- Annotations: τ (time constant), dx/dt equation
- Color coding: Chronos (blue), Kairos (gold)

**Size:** Double column (180mm wide)

---

## Figure 2: Φ Topology Heatmap

**Filename:** `fig2_phi_topology_heatmap.png`

**Purpose:** Visualize Φ measurements across 19 topologies

**Specification:**

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  A) Φ by Topology (Heatmap)                                     │
│                                                                 │
│     ┌────────────────────────────────────────────┐              │
│     │ T1  ████░░░░░░░░░░░░░░░░░░  0.156          │              │
│     │ T2  ██████░░░░░░░░░░░░░░░░  0.234          │              │
│     │ T3  ████████░░░░░░░░░░░░░░  0.312          │              │
│     │ T4  ██████████░░░░░░░░░░░░  0.387          │              │
│     │ T5  ████████████░░░░░░░░░░  0.412          │              │
│     │ T6  ██████████████████████  0.496  ← 3D SW │              │
│     │ T7  █████████████████████░  0.489          │              │
│     │ ...                                        │              │
│     │ T9  ██████████████████████  0.498  ← 4D    │              │
│     └────────────────────────────────────────────┘              │
│     0.0          0.25          0.5                              │
│                  Φ (Integrated Information)                     │
│                                                                 │
│  B) Φ vs Dimensionality (Asymptotic Curve)                      │
│                                                                 │
│     Φ │                    ......................                │
│   0.5 │               .....                      │              │
│       │          ....                            │              │
│   0.4 │       ...                                │              │
│       │     ..                                   │              │
│   0.3 │   ..                                     │              │
│       │  .                                       │              │
│   0.2 │ .                                        │              │
│       │.                                         │              │
│   0.1 │                                          │              │
│       └──────────────────────────────────────────│              │
│         1D    100D   1K    4K    8K    16K       │              │
│                   Dimensionality                 │              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Elements:**
- Panel A: Horizontal bar chart, topologies sorted by Φ
- Panel B: Asymptotic curve showing saturation at 0.5
- Highlight: 3D small-world (T6) at 99.2% of max
- Error bars: 95% confidence intervals

**Size:** Double column (180mm wide)

---

## Figure 3: Energy Efficiency Comparison

**Filename:** `fig3_energy_efficiency_comparison.png`

**Purpose:** Dramatic visualization of 60x power reduction

**Specification:**

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Power Consumption (Watts)                                      │
│                                                                 │
│  300W │ ████████████████████████████████████████ GPT-4         │
│       │ ████████████████████████████████████████ (GPU)         │
│       │                                                         │
│  200W │                                                         │
│       │                                                         │
│       │                                                         │
│  100W │                                                         │
│       │                                                         │
│   50W │                                                         │
│       │                                                         │
│   20W │ ████ TF Lite                                            │
│       │ ████ Edge Impulse                                       │
│    5W │ █ SYMTHAEA (CPU)                                        │
│       └─────────────────────────────────────────────────────────│
│                                                                 │
│  Memory Footprint:                                              │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ GPT-4:      ████████████████████████████ 100GB+     │       │
│  │ TF Lite:    ██ 50MB                                 │       │
│  │ Symthaea:   ░ 10MB                                  │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Elements:**
- Logarithmic scale for power (5W vs 300W dramatic)
- Memory comparison below
- Color: Symthaea in gold, others in gray
- Annotation: "60x reduction"

**Size:** Single column (88mm wide)

---

## Figure 4: Temporal Integrity Visualization

**Filename:** `fig4_temporal_integrity_graph.png`

**Purpose:** Show continuous state evolution preserving causal history

**Specification:**

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  A) Transformer: Causal History Lost                            │
│                                                                 │
│     Input:  [A] [B] [C] [D] [E]                                 │
│               ↓   ↓   ↓   ↓   ↓                                 │
│            ┌─────────────────────┐                              │
│            │   Attention Matrix  │   ← Time compressed          │
│            │   (static snapshot) │     into space               │
│            └─────────────────────┘                              │
│                      ↓                                          │
│               [Output: ?]         ← Why? Unknown.               │
│                                                                 │
│  B) LTC: Causal Chain Preserved                                 │
│                                                                 │
│     State                                                       │
│       │    ╱╲     ╱╲        ╱╲                                  │
│       │   ╱  ╲   ╱  ╲  ... ╱  ╲                                 │
│       │  ╱    ╲ ╱    ╲    ╱    ╲                                │
│       │ ╱      ╳      ╲  ╱      ╲                               │
│       │╱              ╲╱        ╲                               │
│       └──────────────────────────────→ Time                     │
│         A    B    C    D    E                                   │
│                                                                 │
│     Each point traceable: "Output because A→B→C→D→E"            │
│                                                                 │
│  C) Φ Over Time (Integration Visible)                           │
│                                                                 │
│     Φ │      ____________________                               │
│   0.5 │     /                                                   │
│       │    /                                                    │
│   0.3 │   /                                                     │
│       │  /                                                      │
│   0.1 │_/                                                       │
│       └─────────────────────────────→ Time                      │
│         Integration builds over time                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Elements:**
- Panel A: Transformer as static snapshot (information loss)
- Panel B: LTC as continuous trajectory (causality preserved)
- Panel C: Φ building over time (integration visible)
- Annotations explaining explainability

**Size:** Double column (180mm wide)

---

## Generation Instructions

### Using Python/Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

# Figure 2B: Asymptotic curve
dimensions = np.logspace(0, 4.2, 100)  # 1 to 16384
phi = 0.5 * (1 - np.exp(-dimensions / 1000))  # Asymptotic model

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogx(dimensions, phi, 'b-', linewidth=2)
ax.axhline(y=0.5, color='r', linestyle='--', label='Asymptotic limit')
ax.set_xlabel('Dimensionality', fontsize=12)
ax.set_ylabel('Φ (Integrated Information)', fontsize=12)
ax.set_title('Φ Saturation with Dimensionality')
ax.legend()
plt.tight_layout()
plt.savefig('fig2b_phi_asymptotic.png', dpi=300)
```

### Export Settings
- DPI: 300
- Format: PNG for draft, PDF for submission
- Color space: RGB
- Fonts embedded

---

## Checklist

- [ ] Fig 1: Architecture comparison (LTC vs Transformer)
- [ ] Fig 2: Φ topology heatmap + asymptotic curve
- [ ] Fig 3: Energy efficiency comparison
- [ ] Fig 4: Temporal integrity visualization
- [ ] All figures at 300 DPI
- [ ] Fonts embedded (Arial/Helvetica)
- [ ] Color-blind friendly palette
- [ ] Figure captions written

---

*Figure specifications for Nature Neuroscience submission*
