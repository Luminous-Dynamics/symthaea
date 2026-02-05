# Publication Figures Specification

## Target Venue
NeurIPS / Nature Machine Intelligence / ICML

## Figure List

### Figure 1: The Phenomenal Corridor
**Type**: Heatmap + Line plot
**Data**: Layer-wise unity scores (L0-23) for phenomenal vs functional

```
Panel A: Heatmap
- X-axis: Layer (0-23)
- Y-axis: Concept class (Phenomenal, Functional)
- Color: Unity score (0.6-1.0)
- Highlight: L21-22 corridor

Panel B: Line plot with confidence bands
- X-axis: Layer (0-23)
- Y-axis: Unity difference (Phen - Func)
- Show: 95% CI from bootstrap
- Mark: Significant layers (**)
```

### Figure 2: Fine-Grained Corridor Analysis
**Type**: Bar chart with error bars
**Data**: L17-23 unity scores

```
- X-axis: Layer (17-23)
- Y-axis: Mean unity score
- Bars: Phenomenal (blue), Functional (orange)
- Error bars: 95% CI
- Annotations: p-values, Cohen's d
- Highlight: L22 peak
```

### Figure 3: The "Lobotomy" Experiment
**Type**: Before/after comparison + selectivity plot
**Data**: Causal ablation results

```
Panel A: Baseline vs Ablated unity scores
- Grouped bars: Baseline, L21 Shuffle, L22 Shuffle
- Two groups: Phenomenal, Functional
- Show: Effect reversal at L22

Panel B: Selectivity heatmap
- X-axis: Intervention type
- Y-axis: Layer (21, 22)
- Color: Selectivity score (positive = phen-selective)
```

### Figure 4: Binding Compression Asymmetry
**Type**: Scatter plot with regression
**Data**: Binding effect by layer

```
- X-axis: Layer
- Y-axis: Binding effect (persistence change)
- Points: Phenomenal (blue), Functional (orange)
- Lines: Regression fits
- Shading: Interaction region
```

### Figure 5: Theoretical Model
**Type**: Schematic diagram
**Content**:

```
     Input Layer
          │
          ▼
    ┌─────────────┐
    │ Early Layers│  Syntax, surface features
    │   (0-6)     │
    └─────────────┘
          │
          ▼
    ┌─────────────┐
    │Middle Layers│  Semantic content
    │   (7-17)    │  Functional > Phenomenal
    └─────────────┘
          │
          ▼
    ┌─────────────┐
    │ PHENOMENAL  │  ← Phenomenal structure emerges
    │  CORRIDOR   │  ← Peak at L22
    │  (18-22)    │  ← Structure-dependent encoding
    └─────────────┘
          │
          ▼
    ┌─────────────┐
    │ Output Layer│  Task-specific compression
    │    (23)     │  Effect partially preserved
    └─────────────┘
```

### Figure 6: XOR Binding Mechanism
**Type**: Vector diagram
**Content**:

```
Panel A: Within-class binding (Phenomenal)
  Phen_1 = Φ ⊕ Spec_1
  Phen_2 = Φ ⊕ Spec_2
  Bind(P1,P2) = Spec_1 ⊕ Spec_2  (Φ cancels)
  → Reduced complexity

Panel B: Cross-class binding (Oil & Water)
  Phen = Φ ⊕ Spec_P
  Func = F ⊕ Spec_F
  Bind(P,F) = Φ ⊕ F ⊕ Spec_P ⊕ Spec_F
  → Increased complexity (no cancellation)
```

---

## Supplementary Figures

### Supp. Fig. 1: Robustness Validation
- Bootstrap distributions
- Cross-validation results
- Random subset analysis

### Supp. Fig. 2: Betti Number Analysis
- β₀, β₁, β₂ by layer
- Persistence diagrams

### Supp. Fig. 3: Concept Examples
- Word clouds for each category
- Representative concepts

### Supp. Fig. 4: Activation Statistics
- L22 activation distributions
- Per-dimension discriminability

---

## Color Scheme

```
Phenomenal: #3498db (blue)
Functional: #e74c3c (orange/red)
Significant: #2ecc71 (green)
Non-significant: #95a5a6 (gray)
Corridor highlight: #f39c12 (gold)
```

## Data Sources

| Figure | Experiment File | Key Variables |
|--------|-----------------|---------------|
| Fig 1 | layer_topology_expanded.rs | layer_unity_scores |
| Fig 2 | phenomenal_corridor_finegrained.rs | layer_metrics |
| Fig 3 | causal_ablation_lobotomy.rs | ablation_results |
| Fig 4 | binding_layer_sweep.rs | binding_effects |
| Fig 5 | N/A (schematic) | - |
| Fig 6 | N/A (schematic) | - |

---

## Generation Notes

Figures can be generated using:
- Python + matplotlib/seaborn
- R + ggplot2
- Julia + Plots.jl

Recommend: Export experiment data to JSON, then use Python for publication-quality figures.
