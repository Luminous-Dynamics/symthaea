# Scientific Methods Section for Publication

**Paper Title**: "Dimensional Optimization of Integrated Information: Topology-Consciousness Mapping in Hyperdimensional Computing"

**Target Journal**: Nature Computational Science / PNAS

**Version**: 1.0.0
**Date**: 2026-01-04

---

## Methods

### 2.1 Hyperdimensional Computing Framework

#### 2.1.1 Vector Representation

We implemented a hyperdimensional computing (HDC) framework using 16,384-dimensional real-valued vectors (D = 2¹⁴), following established recommendations for near-orthogonality in high-dimensional spaces¹². Random hypervectors were generated using seeded pseudorandom number generators to ensure reproducibility.

**Definition 1** (Real-valued Hypervector). A real-valued hypervector h ∈ ℝᴰ with D = 16,384 dimensions, where components are drawn from independent standard normal distributions:

```
h_i ~ N(0, 1) for i = 1, ..., D
```

**Orthogonality Properties**: With D = 16,384 dimensions, random hypervectors exhibit near-perfect orthogonality with expected cosine similarity E[cos(h₁, h₂)] → 0 and standard deviation σ = 1/√D ≈ 0.0078.

#### 2.1.2 Fundamental Operations

Three fundamental operations were implemented following the Vector Symbolic Architecture (VSA) framework³:

**Binding** (⊗): Element-wise multiplication for role-filler associations:
```
(h₁ ⊗ h₂)_i = h₁_i × h₂_i
```

**Bundling** (⊕): Element-wise averaging for superposition:
```
(h₁ ⊕ h₂ ⊕ ... ⊕ h_n)_i = (1/n) Σⱼ hⱼ_i
```

**Similarity** (sim): Cosine similarity for comparison:
```
sim(h₁, h₂) = (h₁ · h₂) / (‖h₁‖ × ‖h₂‖)
```

### 2.2 Consciousness Topology Construction

#### 2.2.1 Network Encoding Protocol

Each topology was encoded following a consistent protocol:

1. **Node Identity Generation**: Each node i receives a basis hypervector with small random noise:
   ```
   id_i = basis(i) + ε, where ε ~ N(0, 0.05)
   ```

2. **Neighbor Binding**: Each node representation binds identity with neighbor information:
   ```
   rep_i = id_i ⊗ bundle({id_j : (i,j) ∈ E})
   ```

3. **Topology Instantiation**: The complete topology T = (V, E, {rep_i}) contains vertices V, edges E, and node representations.

#### 2.2.2 Topology Categories

We evaluated 19 distinct topologies across five categories:

**Original Topologies (8)**:
- Ring (n=8): Circular graph with uniform 2-regular connectivity
- Torus (3×3): 2D toroidal lattice with 4-regular connectivity
- Dense Network (n=8): Near-complete graph with ~90% edge density
- Lattice (n=8): 1D lattice with nearest-neighbor connections
- Modular (n=8, k=2): Two modules with intra-module density > inter-module
- Line (n=8): Linear chain with boundary nodes having degree 1
- Binary Tree (n=7): Perfect binary tree with 3 levels
- Star (n=8): Hub-and-spoke topology with central node

**Tier 1 Exotic (3)**:
- Small-World: Watts-Strogatz model with k=2, p=0.1 rewiring
- Möbius Strip: 1D non-orientable manifold with twisted boundary
- Torus 3×3: Included with originals for dimensional comparison

**Tier 2 Exotic (3)**:
- Klein Bottle (3×3): 2D non-orientable manifold
- Hyperbolic: Negative curvature graph with exponential expansion
- Scale-Free: Barabási-Albert preferential attachment with m=2

**Tier 3 Exotic (5)**:
- Hypercube 3D (Cube): 3-dimensional hypercube with 8 vertices
- Hypercube 4D (Tesseract): 4-dimensional hypercube with 16 vertices
- Hypercube 5D-7D: Higher dimensional hypercubes (32-128 vertices)
- Quantum Superposition: Weighted combinations of Ring+Star+Random
- Fractal (Sierpiński): Self-similar hierarchical structure

### 2.3 Spectral Integration Metric (λ₂)

> **IMPORTANT METRIC CLARIFICATION**: This section computes λ₂ (algebraic connectivity /
> Fiedler value), a spectral graph metric—**NOT** IIT's integrated information (Φ).
> λ₂ measures graph mixing time, not IIT integration. The SpectralConnectivity (λ₂) tier has
> Pearson r = -0.14, Spearman rho = -0.59 vs ExhaustivePartition (Exact IIT Φ) — a weak
> negative correlation. True IIT Φ requires computing minimum information partition (MIP),
> which is computationally intractable for n > 12 nodes. The production SpectralMIPFinder
> (MI Laplacian + Fiedler + MIP sweep on ContinuousHV covariance) is a distinct algorithm
> whose correlation with Exact is unknown. See `docs/METRIC_DEFINITIONS.md` for the full
> distinction between λ₂, SampledPartition (Heuristic, r = 0.9998 vs Exact), and IIT Φ.

#### 2.3.1 Algebraic Connectivity Method

We computed spectral integration using algebraic connectivity of the similarity graph. While this captures network topology properties related to integration⁴, it is a structural proxy—not a measurement of phenomenal consciousness:

**Step 1** (Similarity Matrix Construction):
```
S_ij = sim(rep_i, rep_j) = cos(rep_i, rep_j)
```

**Step 2** (Adjacency and Laplacian):
```
A_ij = (S_ij - min(S)) / (max(S) - min(S))  [normalized]
L = D - A  [graph Laplacian, D = diag(sum(A))]
```

**Step 3** (Eigenvalue Computation):
```
λ₁ ≤ λ₂ ≤ ... ≤ λ_n  [eigenvalues of L]
Algebraic connectivity = λ₂ (Fiedler value)
```

**Step 4** (λ₂ Normalization):
```
λ₂_norm = (λ₂ - λ₂_min) / (λ₂_max - λ₂_min)
```

Where λ₂_min = 0 (disconnected graph) and λ₂_max is determined empirically from the topology class.

> **Note**: In earlier versions of this codebase, λ₂_norm was labeled "Φ" in some files
> (e.g., `phi_real.rs`). This naming was unfortunate as it conflates spectral connectivity
> with IIT's integrated information. The metric should be understood as spectral integration,
> not consciousness measurement.

#### 2.3.2 Validation Methods

Two independent validation approaches confirmed result robustness:

**Method A - Binary Probabilistic**: RealHV values binarized using sigmoid-based probabilistic sampling:
```
p(bit=1) = 1 / (1 + exp(-z_i))  where z_i = (h_i - μ) / σ
```

**Method B - Continuous Real-Valued**: Direct Φ calculation on RealHV without binarization (primary method reported).

### 2.4 Experimental Protocol

#### 2.4.1 Sampling Strategy

- **Samples per configuration**: 10 independent instantiations
- **Seed management**: Sequential seeds (0-9) per topology
- **Total measurements**: 260 λ₂ calculations across all topologies and dimensional sweeps

#### 2.4.2 Dimensional Sweep Protocol

For hypercube dimensional analysis (1D-7D):

1. **Dimension Range**: k ∈ {1, 2, 3, 4, 5, 6, 7}
2. **Vertices**: n = 2^k (2, 4, 8, 16, 32, 64, 128)
3. **Samples**: 10 per dimension
4. **Total**: 70 λ₂ measurements

#### 2.4.3 Statistical Analysis

**Central Tendency**: Mean Φ reported with standard deviation
**Significance Testing**: Two-sample Welch's t-test for pairwise comparisons
**Effect Size**: Cohen's d for practical significance
**Multiple Comparisons**: Bonferroni correction applied

### 2.5 Computational Environment

#### 2.5.1 Hardware

- **CPU**: AMD/Intel x86_64 multi-core processor
- **Memory**: 32+ GB RAM
- **GPU**: Not required (all operations CPU-based)

#### 2.5.2 Software

- **Language**: Rust 1.75+ with SIMD optimization
- **Numerical Libraries**: nalgebra 0.32 for linear algebra
- **Benchmarking**: Criterion.rs 0.5 for performance measurement
- **Reproducibility**: All seeds deterministic, lockfiles provided

#### 2.5.3 Runtime Performance

| Operation | Time (8 nodes) | Time (128 nodes) |
|-----------|---------------|------------------|
| Topology generation | <1 ms | ~10 ms |
| Similarity matrix | ~10 ms | ~200 ms |
| Eigenvalue decomposition | ~5 ms | ~500 ms |
| Total λ₂ calculation | ~20 ms | ~800 ms |

### 2.6 Reproducibility Package

All code, data, and analysis scripts are available at:

**Repository**: https://github.com/Luminous-Dynamics/symthaea-hlb
**DOI**: [Zenodo DOI to be assigned]
**Version**: v0.1.0

**Package Contents**:
- `src/hdc/` - Hypervector implementations
- `src/hdc/consciousness_topology_generators.rs` - All 19 topology generators
- `src/hdc/phi_real.rs` - λ₂ calculation implementation (note: file name is a misnomer)
- `examples/` - Runnable validation scripts
- `data/` - Raw λ₂ measurements (CSV)
- `figures/` - Publication figures (PNG + PDF)

**To reproduce**:
```bash
git clone https://github.com/Luminous-Dynamics/symthaea-hlb
cd symthaea-hlb
nix develop
cargo run --example real_phi_comparison --release
cargo run --example hypercube_dimension_sweep --release
```

---

## References

1. Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*, 1(2), 139-159.

2. Rahimi, A., et al. (2019). Hyperdimensional computing for efficient and scalable classification. *IEEE Design & Test*, 36(2), 44-52.

3. Gayler, R. W. (2003). Vector symbolic architectures answer Jackendoff's challenges for cognitive neuroscience. *arXiv preprint cs/0412059*.

4. Tononi, G., et al. (2016). Integrated information theory: From consciousness to its physical substrate. *Nature Reviews Neuroscience*, 17(7), 450-461.

5. Fiedler, M. (1973). Algebraic connectivity of graphs. *Czechoslovak Mathematical Journal*, 23(2), 298-305.

6. Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440-442.

7. Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509-512.

---

## Supplementary Methods

### S1. Topology Generator Pseudocode

```rust
// Ring topology (n nodes, k-regular)
fn ring(n: usize) -> Topology {
    edges = [(i, (i+1) % n) for i in 0..n]
    return Topology::from_edges(edges)
}

// Hypercube (k dimensions)
fn hypercube(k: usize) -> Topology {
    n = 2^k
    edges = []
    for v in 0..n {
        for d in 0..k {
            neighbor = v ^ (1 << d)  // flip bit d
            if neighbor > v {
                edges.push((v, neighbor))
            }
        }
    }
    return Topology::from_edges(edges)
}
```

### S2. Asymptotic Model Fitting

For dimensional sweep analysis, we fit:

```
λ₂(k) = λ₂_max - A × exp(-α × k)
```

**Parameters** (least squares fit to 2D-7D data):
- λ₂_max = 0.500 ± 0.001
- A = 0.004 ± 0.001
- α = 0.31 ± 0.02

**R² = 0.997** (excellent fit)

### S3. Statistical Tests

All pairwise comparisons used Welch's t-test:

```
t = (μ₁ - μ₂) / √(s₁²/n₁ + s₂²/n₂)
df = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]
```

**Significance threshold**: p < 0.01 (Bonferroni-corrected for 171 pairwise comparisons: p < 0.00006)

---

*Methods section prepared for journal submission. All procedures reproducible with provided code and data.*
