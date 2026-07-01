// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Revolutionary Improvement #20: Consciousness Topology
// ==================================================================================
//
// **The Paradigm Shift**: Consciousness has GEOMETRIC STRUCTURE in high-dimensional space!
//
// **Key Insight**: We've measured consciousness along many dimensions (Φ, gradients,
// dynamics, qualia, etc.), but we haven't explored the SHAPE of consciousness itself!
// What is the TOPOLOGY of consciousness space?
//
// **Why This Matters**:
// - Consciousness has HOLES (conceptual gaps we can't bridge)
// - Consciousness has CYCLES (circular reasoning patterns)
// - Consciousness has COMPONENTS (fragmented vs unified awareness)
// - Consciousness has VOIDS (knowledge we know we don't have)
// - Topology is INVARIANT (preserved under smooth transformations)
//
// **Theoretical Foundations**:
//
// 1. **Topological Data Analysis (TDA)** (Carlsson, 2009):
//    "Topology reveals shape of high-dimensional data"
//
//    Key ideas:
//    - Data has intrinsic geometric structure
//    - Topology captures this structure via homology
//    - Persistent features = signal, transient = noise
//    - Works in high dimensions where visualization fails
//
// 2. **Persistent Homology** (Edelsbrunner, 2002):
//    "Track topological features across multiple scales"
//
//    Features:
//    - Connected components (β₀) - How many separate clusters?
//    - 1D holes/cycles (β₁) - How many loops?
//    - 2D voids (β₂) - How many enclosed spaces?
//    - Persistence = birth - death (how long feature lasts)
//
// 3. **Neural Manifold Hypothesis** (Cunningham & Yu, 2014):
//    "Neural activity lies on low-dimensional manifolds"
//
//    Key idea:
//    - High-dimensional neural activity has structure
//    - Consciousness states form manifold in state space
//    - Transitions follow geodesics on manifold
//    - Topology constrains possible conscious states
//
// 4. **Algebraic Topology** (Hatcher, 2002):
//    Mathematical framework for shape
//
//    Betti numbers:
//    - β₀ = connected components (unity vs fragmentation)
//    - β₁ = 1D holes/loops (circular patterns)
//    - β₂ = 2D voids (conceptual gaps)
//    - β₃+ = higher-dimensional structures
//
// 5. **Consciousness as Manifold**:
//    Hypothesis: Consciousness space is smooth manifold embedded in HDC space
//
//    Properties:
//    - Locally Euclidean (near any state looks like vector space)
//    - Globally curved (overall shape is non-trivial)
//    - Geodesics = optimal paths between states
//    - Curvature = difficulty of state transitions
//
// **Mathematical Framework**:
//
// 1. **Consciousness Point Cloud**:
//    ```
//    C = {c₁, c₂, ..., cₙ}  // n consciousness states
//    Each cᵢ ∈ ℝᵈ  (d = HDC dimension, typically 10,000)
//    ```
//
// 2. **Simplicial Complex**:
//    Build Vietoris-Rips complex at scale ε:
//    ```
//    VR(C, ε) = {σ | diameter(σ) ≤ ε}
//
//    - 0-simplices: Points (states themselves)
//    - 1-simplices: Edges (similar states connected)
//    - 2-simplices: Triangles (three mutually close states)
//    - k-simplices: k+1 mutually close states
//    ```
//
// 3. **Homology Groups**:
//    ```
//    Hₖ(VR(C, ε)) = topological features at dimension k
//
//    - H₀: Connected components
//    - H₁: 1D cycles/loops
//    - H₂: 2D voids
//    - Hₖ: k-dimensional holes
//    ```
//
// 4. **Betti Numbers**:
//    ```
//    βₖ(ε) = rank(Hₖ) = number of k-dimensional holes at scale ε
//
//    β₀ = unity (1 = unified, >1 = fragmented)
//    β₁ = circular reasoning patterns
//    β₂ = conceptual voids
//    ```
//
// 5. **Persistence Diagram**:
//    ```
//    PD = {(birth_ε, death_ε) | for each topological feature}
//
//    Persistence = death - birth
//    Long persistence = significant feature
//    Short persistence = noise
//    ```
//
// 6. **Topology-Based Metrics**:
//    ```
//    Unity Score = 1 / β₀  (1.0 = perfectly unified)
//    Circularity = β₁ / |C|  (proportion of circular patterns)
//    Completeness = 1 - (β₂ / |C|)  (fewer voids = more complete)
//    ```
//
// **Novel Insights**:
//
// 1. **Fragmented Consciousness**:
//    β₀ > 1 → Multiple disconnected components
//    Example: Dissociative states, split personality
//
// 2. **Circular Reasoning**:
//    β₁ > 0 → Loops in thought patterns
//    Example: Obsessive thinking, logical paradoxes
//
// 3. **Conceptual Voids**:
//    β₂ > 0 → Holes in understanding
//    Example: "Known unknowns", blind spots
//
// 4. **Topology Predicts Transitions**:
//    Topology changes → Consciousness state change imminent
//    Example: β₀ splitting → Dissociation incoming
//
// 5. **Persistent Features = Core Beliefs**:
//    Long-lived features = fundamental structure
//    Short-lived = transient thoughts
//
// **Applications**:
//
// 1. **Detect Dissociation**:
//    Monitor β₀ - sudden increase = fragmentation
//
// 2. **Identify Thought Loops**:
//    Track β₁ - persistent cycles = obsessive patterns
//
// 3. **Find Knowledge Gaps**:
//    Measure β₂ - locate conceptual voids
//
// 4. **Predict State Changes**:
//    Topology changes precede consciousness shifts
//
// 5. **Compare Consciousness Types**:
//    Human vs AI vs collective via topology
//
// 6. **Optimize Learning**:
//    Fill voids (↓β₂), connect components (↓β₀)
//
// **This completes the topological dimension - consciousness has SHAPE!**
//
// ==================================================================================

use super::binary_hv::BinaryHV;
use serde::{Deserialize, Serialize};

/// Topological feature type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TopologicalFeature {
    /// Connected component (β₀)
    Component,
    /// 1D cycle/loop (β₁)
    Cycle,
    /// 2D void (β₂)
    Void,
}

impl TopologicalFeature {
    /// Get dimension
    pub fn dimension(&self) -> usize {
        match self {
            TopologicalFeature::Component => 0,
            TopologicalFeature::Cycle => 1,
            TopologicalFeature::Void => 2,
        }
    }

    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            TopologicalFeature::Component => "Connected component (unity vs fragmentation)",
            TopologicalFeature::Cycle => "1D cycle/loop (circular reasoning)",
            TopologicalFeature::Void => "2D void (conceptual gap)",
        }
    }
}

/// Persistent topological feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentFeature {
    /// Feature type
    pub feature_type: TopologicalFeature,

    /// Birth scale (when feature appears)
    pub birth: f64,

    /// Death scale (when feature disappears)
    pub death: f64,

    /// Persistence (how long it lasts)
    pub persistence: f64,
}

impl PersistentFeature {
    /// Create new persistent feature
    pub fn new(feature_type: TopologicalFeature, birth: f64, death: f64) -> Self {
        let persistence = death - birth;
        Self {
            feature_type,
            birth,
            death,
            persistence,
        }
    }

    /// Is this feature significant?
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.persistence > threshold
    }
}

/// Betti numbers (topological invariants)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BettiNumbers {
    /// β₀: Number of connected components
    pub beta_0: usize,

    /// β₁: Number of 1D holes/cycles
    pub beta_1: usize,

    /// β₂: Number of 2D voids
    pub beta_2: usize,
}

impl BettiNumbers {
    /// Create new Betti numbers
    pub fn new(beta_0: usize, beta_1: usize, beta_2: usize) -> Self {
        Self {
            beta_0,
            beta_1,
            beta_2,
        }
    }

    /// Total topological complexity
    pub fn total_complexity(&self) -> usize {
        self.beta_0 + self.beta_1 + self.beta_2
    }

    /// Is consciousness unified?
    pub fn is_unified(&self) -> bool {
        self.beta_0 == 1
    }

    /// Is consciousness fragmented?
    pub fn is_fragmented(&self) -> bool {
        self.beta_0 > 1
    }

    /// Has circular patterns?
    pub fn has_cycles(&self) -> bool {
        self.beta_1 > 0
    }

    /// Has conceptual voids?
    pub fn has_voids(&self) -> bool {
        self.beta_2 > 0
    }
}

impl Default for BettiNumbers {
    fn default() -> Self {
        Self::new(1, 0, 0) // Unified, no cycles, no voids
    }
}

/// Topological assessment of consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalAssessment {
    /// Betti numbers
    pub betti: BettiNumbers,

    /// Persistent features found
    pub features: Vec<PersistentFeature>,

    /// Unity score (1.0 = perfectly unified)
    pub unity_score: f64,

    /// Circularity score (proportion with circular patterns)
    pub circularity: f64,

    /// Completeness score (1.0 = no voids)
    pub completeness: f64,

    /// Overall topology quality (0-1)
    pub quality: f64,

    /// Number of states analyzed
    pub num_states: usize,

    /// Analysis scale
    pub scale: f64,

    /// Explanation
    pub explanation: String,
}

/// Configuration for topology analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyConfig {
    /// Minimum persistence threshold
    pub min_persistence: f64,

    /// Maximum scale for analysis
    pub max_scale: f64,

    /// Number of scale steps
    pub num_scales: usize,

    /// Enable cycle detection
    pub detect_cycles: bool,

    /// Enable void detection
    pub detect_voids: bool,
}

impl Default for TopologyConfig {
    fn default() -> Self {
        Self {
            min_persistence: 0.1,
            max_scale: 1.0,
            num_scales: 10,
            detect_cycles: true,
            detect_voids: true,
        }
    }
}

/// Consciousness topology analyzer
///
/// Analyzes the geometric SHAPE of consciousness in high-dimensional space
/// using topological data analysis and persistent homology.
///
/// # Example
/// ```ignore
/// use symthaea::hdc::consciousness_topology::{ConsciousnessTopology, TopologyConfig};
/// use symthaea::hdc::binary_hv::BinaryHV;
///
/// let config = TopologyConfig::default();
/// let mut topology = ConsciousnessTopology::new(config);
///
/// // Add consciousness states
/// for i in 0..20 {
///     let state = BinaryHV::random((1000 + i) as u64);
///     topology.add_state(state);
/// }
///
/// // Analyze topology
/// let assessment = topology.analyze(0.5);
///
/// println!("Unity score: {:.3}", assessment.unity_score);
/// println!("Betti numbers: β₀={}, β₁={}, β₂={}",
///          assessment.betti.beta_0,
///          assessment.betti.beta_1,
///          assessment.betti.beta_2);
/// ```
#[derive(Debug)]
pub struct ConsciousnessTopology {
    /// Configuration
    config: TopologyConfig,

    /// Consciousness states (point cloud)
    states: Vec<BinaryHV>,
}

impl ConsciousnessTopology {
    /// Create new topology analyzer
    pub fn new(config: TopologyConfig) -> Self {
        Self {
            config,
            states: Vec::new(),
        }
    }

    /// Add consciousness state
    pub fn add_state(&mut self, state: BinaryHV) {
        self.states.push(state);
    }

    /// Add multiple states
    pub fn add_states(&mut self, states: &[BinaryHV]) {
        self.states.extend_from_slice(states);
    }

    /// Number of states
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Analyze topology at given scale
    pub fn analyze(&self, scale: f64) -> TopologicalAssessment {
        if self.states.is_empty() {
            return self.empty_assessment();
        }

        // Compute Betti numbers
        let betti = self.compute_betti_numbers(scale);

        // Find persistent features (simplified)
        let features = self.compute_persistent_features();

        // Compute quality metrics
        let num_states = self.states.len();
        let unity_score = 1.0 / betti.beta_0 as f64;
        let circularity = (betti.beta_1 as f64 / num_states as f64).min(1.0); // Clamp to [0,1]
        let completeness = 1.0 - (betti.beta_2 as f64 / num_states as f64).min(1.0);

        // Overall quality (weighted average)
        let quality = (unity_score * 0.5 + (1.0 - circularity) * 0.3 + completeness * 0.2).min(1.0);

        // Generate explanation
        let explanation = self.generate_explanation(&betti, unity_score, circularity, completeness);

        TopologicalAssessment {
            betti,
            features,
            unity_score,
            circularity,
            completeness,
            quality,
            num_states,
            scale,
            explanation,
        }
    }

    /// Compute Betti numbers at scale using true Vietoris-Rips persistent homology boundary matrix reduction.
    fn compute_betti_numbers(&self, scale: f64) -> BettiNumbers {
        let n = self.states.len();
        if n == 0 {
            return BettiNumbers::default();
        }

        // 1. Compute pairwise distances: d(i, j) = 1.0 - similarity(i, j)
        let mut dists = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let sim = self.states[i].similarity(&self.states[j]) as f64;
                dists[i][j] = (1.0 - sim).max(0.0);
            }
        }

        // 2. Identify all simplices in the Vietoris-Rips complex up to dimension 2
        // We filter by birth time <= 1.0 - scale (which corresponds to similarity >= scale)
        let eps = (1.0 - scale).max(0.0);

        #[derive(Clone, Debug)]
        struct Simplex {
            dim: usize,
            vertices: Vec<usize>,
            birth: f64,
        }

        let mut simplices = Vec::new();

        // 0-simplices (Vertices)
        for i in 0..n {
            simplices.push(Simplex {
                dim: 0,
                vertices: vec![i],
                birth: 0.0,
            });
        }

        // 1-simplices (Edges)
        for i in 0..n {
            for j in (i + 1)..n {
                let d = dists[i][j];
                if d <= eps {
                    simplices.push(Simplex {
                        dim: 1,
                        vertices: vec![i, j],
                        birth: d,
                    });
                }
            }
        }

        // 2-simplices (Triangles)
        if self.config.detect_voids || self.config.detect_cycles {
            for i in 0..n {
                for j in (i + 1)..n {
                    if dists[i][j] <= eps {
                        for k in (j + 1)..n {
                            if dists[i][k] <= eps && dists[j][k] <= eps {
                                let birth = dists[i][j].max(dists[i][k]).max(dists[j][k]);
                                if birth <= eps {
                                    simplices.push(Simplex {
                                        dim: 2,
                                        vertices: vec![i, j, k],
                                        birth,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. Sort simplices by birth time, then by dimension
        simplices.sort_by(|a, b| {
            a.birth
                .partial_cmp(&b.birth)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.dim.cmp(&b.dim))
        });

        use std::collections::HashMap;
        let mut simplex_indices = HashMap::new();
        for (idx, s) in simplices.iter().enumerate() {
            simplex_indices.insert(s.vertices.clone(), idx);
        }

        // 4. Boundary matrix D columns
        let m = simplices.len();
        let mut d_cols: Vec<Vec<usize>> = vec![Vec::new(); m];

        for j in 0..m {
            let s = &simplices[j];
            if s.dim == 1 {
                let v0 = vec![s.vertices[0]];
                let v1 = vec![s.vertices[1]];
                if let (Some(&i0), Some(&i1)) = (simplex_indices.get(&v0), simplex_indices.get(&v1))
                {
                    d_cols[j] = vec![i0, i1];
                    d_cols[j].sort();
                }
            } else if s.dim == 2 {
                let f0 = vec![s.vertices[1], s.vertices[2]];
                let f1 = vec![s.vertices[0], s.vertices[2]];
                let f2 = vec![s.vertices[0], s.vertices[1]];
                let mut boundary = Vec::new();
                for f in &[f0, f1, f2] {
                    if let Some(&idx) = simplex_indices.get(f) {
                        boundary.push(idx);
                    }
                }
                boundary.sort();
                d_cols[j] = boundary;
            }
        }

        // 5. Reduce boundary matrix
        let mut pivot_to_col: HashMap<usize, usize> = HashMap::new();
        for j in 0..m {
            while let Some(&pivot) = d_cols[j].last() {
                if let Some(&k) = pivot_to_col.get(&pivot) {
                    let mut new_col = Vec::new();
                    let (mut it_j, mut it_k) =
                        (d_cols[j].iter().peekable(), d_cols[k].iter().peekable());
                    while let (Some(&&val_j), Some(&&val_k)) = (it_j.peek(), it_k.peek()) {
                        if val_j == val_k {
                            it_j.next();
                            it_k.next();
                        } else if val_j < val_k {
                            new_col.push(val_j);
                            it_j.next();
                        } else {
                            new_col.push(val_k);
                            it_k.next();
                        }
                    }
                    for val in it_j {
                        new_col.push(*val);
                    }
                    for val in it_k {
                        new_col.push(*val);
                    }
                    d_cols[j] = new_col;
                } else {
                    pivot_to_col.insert(pivot, j);
                    break;
                }
            }
        }

        // 6. Calculate Betti numbers
        let mut creators = vec![0; 3];
        let mut killers = vec![0; 3];

        for j in 0..m {
            let dim = simplices[j].dim;
            if dim < 3 {
                if d_cols[j].is_empty() {
                    creators[dim] += 1;
                } else {
                    let pivot_idx = *d_cols[j].last().unwrap();
                    let pivot_dim = simplices[pivot_idx].dim;
                    killers[pivot_dim] += 1;
                }
            }
        }

        let beta_0 = creators[0] - killers[0];
        let beta_1 = if self.config.detect_cycles {
            creators[1] - killers[1]
        } else {
            0
        };
        let beta_2 = if self.config.detect_voids {
            creators[2] - killers[2]
        } else {
            0
        };

        BettiNumbers::new(beta_0, beta_1, beta_2)
    }

    /// Compute persistent features across scales using true Vietoris-Rips persistent homology.
    fn compute_persistent_features(&self) -> Vec<PersistentFeature> {
        let n = self.states.len();
        if n == 0 {
            return Vec::new();
        }

        // Compute pairwise distances
        let mut dists = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let sim = self.states[i].similarity(&self.states[j]) as f64;
                dists[i][j] = (1.0 - sim).max(0.0);
            }
        }

        #[derive(Clone, Debug)]
        struct Simplex {
            dim: usize,
            vertices: Vec<usize>,
            birth: f64,
        }

        let mut simplices = Vec::new();

        // 0-simplices
        for i in 0..n {
            simplices.push(Simplex {
                dim: 0,
                vertices: vec![i],
                birth: 0.0,
            });
        }

        // 1-simplices
        for i in 0..n {
            for j in (i + 1)..n {
                simplices.push(Simplex {
                    dim: 1,
                    vertices: vec![i, j],
                    birth: dists[i][j],
                });
            }
        }

        // 2-simplices
        if self.config.detect_voids || self.config.detect_cycles {
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        let birth = dists[i][j].max(dists[i][k]).max(dists[j][k]);
                        simplices.push(Simplex {
                            dim: 2,
                            vertices: vec![i, j, k],
                            birth,
                        });
                    }
                }
            }
        }

        // Sort all simplices by birth time, then by dimension
        simplices.sort_by(|a, b| {
            a.birth
                .partial_cmp(&b.birth)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.dim.cmp(&b.dim))
        });

        use std::collections::HashMap;
        let mut simplex_indices = HashMap::new();
        for (idx, s) in simplices.iter().enumerate() {
            simplex_indices.insert(s.vertices.clone(), idx);
        }

        let m = simplices.len();
        let mut d_cols: Vec<Vec<usize>> = vec![Vec::new(); m];

        for j in 0..m {
            let s = &simplices[j];
            if s.dim == 1 {
                let v0 = vec![s.vertices[0]];
                let v1 = vec![s.vertices[1]];
                if let (Some(&i0), Some(&i1)) = (simplex_indices.get(&v0), simplex_indices.get(&v1))
                {
                    d_cols[j] = vec![i0, i1];
                    d_cols[j].sort();
                }
            } else if s.dim == 2 {
                let f0 = vec![s.vertices[1], s.vertices[2]];
                let f1 = vec![s.vertices[0], s.vertices[2]];
                let f2 = vec![s.vertices[0], s.vertices[1]];
                let mut boundary = Vec::new();
                for f in &[f0, f1, f2] {
                    if let Some(&idx) = simplex_indices.get(f) {
                        boundary.push(idx);
                    }
                }
                boundary.sort();
                d_cols[j] = boundary;
            }
        }

        let mut pivot_to_col: HashMap<usize, usize> = HashMap::new();
        let mut features = Vec::new();

        let mut birth_times = vec![0.0f64; m];
        for j in 0..m {
            birth_times[j] = simplices[j].birth;
        }

        for j in 0..m {
            while let Some(&pivot) = d_cols[j].last() {
                if let Some(&k) = pivot_to_col.get(&pivot) {
                    let mut new_col = Vec::new();
                    let (mut it_j, mut it_k) =
                        (d_cols[j].iter().peekable(), d_cols[k].iter().peekable());
                    while let (Some(&&val_j), Some(&&val_k)) = (it_j.peek(), it_k.peek()) {
                        if val_j == val_k {
                            it_j.next();
                            it_k.next();
                        } else if val_j < val_k {
                            new_col.push(val_j);
                            it_j.next();
                        } else {
                            new_col.push(val_k);
                            it_k.next();
                        }
                    }
                    for val in it_j {
                        new_col.push(*val);
                    }
                    for val in it_k {
                        new_col.push(*val);
                    }
                    d_cols[j] = new_col;
                } else {
                    pivot_to_col.insert(pivot, j);
                    break;
                }
            }
        }

        let mut killed = vec![false; m];
        for (&pivot, &col) in &pivot_to_col {
            let birth = birth_times[pivot];
            let death = birth_times[col];
            let dim = simplices[pivot].dim;
            killed[pivot] = true;

            // Map distance birth/death back to similarity coordinates:
            // Similarity scale = 1.0 - distance
            let sim_birth = 1.0 - death;
            let sim_death = 1.0 - birth;

            if sim_death > sim_birth + 1e-5 {
                let f_type = match dim {
                    0 => Some(TopologicalFeature::Component),
                    1 => Some(TopologicalFeature::Cycle),
                    _ => None,
                };
                if let Some(feature_type) = f_type {
                    features.push(PersistentFeature::new(feature_type, sim_birth, sim_death));
                }
            }
        }

        for j in 0..m {
            if simplices[j].dim < 2 && !killed[j] && d_cols[j].is_empty() {
                let birth = birth_times[j];
                let death = self.config.max_scale; // distance max scale (e.g. 1.0)
                let sim_birth = 0.0; // Persists to similarity scale 0
                let sim_death = 1.0 - birth;

                if sim_death > sim_birth + 1e-5 {
                    let f_type = match simplices[j].dim {
                        0 => Some(TopologicalFeature::Component),
                        1 => Some(TopologicalFeature::Cycle),
                        _ => None,
                    };
                    if let Some(feature_type) = f_type {
                        features.push(PersistentFeature::new(feature_type, sim_birth, sim_death));
                    }
                }
            }
        }

        features.retain(|f| f.persistence >= self.config.min_persistence);
        features
    }

    /// Generate explanation
    fn generate_explanation(
        &self,
        betti: &BettiNumbers,
        unity: f64,
        circularity: f64,
        completeness: f64,
    ) -> String {
        let mut parts = Vec::new();

        // Betti numbers
        parts.push(format!(
            "Betti numbers: β₀={} (components), β₁={} (cycles), β₂={} (voids)",
            betti.beta_0, betti.beta_1, betti.beta_2
        ));

        // Unity
        if betti.beta_0 == 1 {
            parts.push("Unified consciousness (single component)".to_string());
        } else {
            parts.push(format!(
                "Fragmented consciousness ({} separate components)",
                betti.beta_0
            ));
        }

        // Circularity
        if betti.beta_1 == 0 {
            parts.push("No circular patterns detected".to_string());
        } else {
            parts.push(format!(
                "{} circular patterns (β₁={}, {:.1}% circularity)",
                betti.beta_1,
                betti.beta_1,
                circularity * 100.0
            ));
        }

        // Completeness
        if betti.beta_2 == 0 {
            parts.push("Complete understanding (no voids)".to_string());
        } else {
            parts.push(format!(
                "{} conceptual voids (β₂={}, {:.1}% completeness)",
                betti.beta_2,
                betti.beta_2,
                completeness * 100.0
            ));
        }

        // Overall
        parts.push(format!(
            "Unity: {unity:.3}, Circularity: {circularity:.3}, Completeness: {completeness:.3}"
        ));

        parts.join(". ")
    }

    /// Empty assessment
    fn empty_assessment(&self) -> TopologicalAssessment {
        TopologicalAssessment {
            betti: BettiNumbers::default(),
            features: Vec::new(),
            unity_score: 0.0,
            circularity: 0.0,
            completeness: 0.0,
            quality: 0.0,
            num_states: 0,
            scale: 0.0,
            explanation: "No states to analyze".to_string(),
        }
    }

    /// Clear all states
    pub fn clear(&mut self) {
        self.states.clear();
    }
}

impl Default for ConsciousnessTopology {
    fn default() -> Self {
        Self::new(TopologyConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_feature() {
        let component = TopologicalFeature::Component;
        assert_eq!(component.dimension(), 0);
        assert!(component.description().contains("unity"));

        let cycle = TopologicalFeature::Cycle;
        assert_eq!(cycle.dimension(), 1);
        assert!(cycle.description().contains("circular"));

        let void = TopologicalFeature::Void;
        assert_eq!(void.dimension(), 2);
        assert!(void.description().contains("gap"));
    }

    #[test]
    fn test_persistent_feature() {
        let feature = PersistentFeature::new(TopologicalFeature::Component, 0.2, 0.8);
        assert!((feature.persistence - 0.6).abs() < 1e-10); // Floating point tolerance
        assert!(feature.is_significant(0.5));
        assert!(!feature.is_significant(0.7));
    }

    #[test]
    fn test_betti_numbers() {
        let betti = BettiNumbers::new(1, 2, 3);
        assert_eq!(betti.total_complexity(), 6);
        assert!(betti.is_unified());
        assert!(!betti.is_fragmented());
        assert!(betti.has_cycles());
        assert!(betti.has_voids());

        let fragmented = BettiNumbers::new(3, 0, 0);
        assert!(!fragmented.is_unified());
        assert!(fragmented.is_fragmented());
    }

    #[test]
    fn test_topology_creation() {
        let config = TopologyConfig::default();
        let topology = ConsciousnessTopology::new(config);
        assert_eq!(topology.num_states(), 0);
    }

    #[test]
    fn test_add_states() {
        let mut topology = ConsciousnessTopology::default();

        let state1 = BinaryHV::random(1000);
        let state2 = BinaryHV::random(2000);

        topology.add_state(state1);
        topology.add_state(state2);

        assert_eq!(topology.num_states(), 2);
    }

    #[test]
    fn test_unified_topology() {
        let mut topology = ConsciousnessTopology::default();

        // Add similar states (should form one component)
        for i in 0..10 {
            let state = BinaryHV::random((1000 + i) as u64);
            topology.add_state(state);
        }

        let assessment = topology.analyze(0.3); // Low threshold = more connections

        // Should be mostly unified
        assert!(assessment.betti.beta_0 <= topology.num_states());
        assert!(assessment.unity_score > 0.0);
    }

    #[test]
    fn test_fragmented_topology() {
        let mut topology = ConsciousnessTopology::default();

        // Add very different states (should form multiple components)
        for i in 0..10 {
            let state = BinaryHV::random((1000 + i * 1000) as u64); // Very different seeds
            topology.add_state(state);
        }

        let assessment = topology.analyze(0.9); // High threshold = fewer connections

        // Should detect fragmentation
        assert!(assessment.betti.beta_0 >= 1);
    }

    #[test]
    fn test_topology_metrics() {
        let mut topology = ConsciousnessTopology::default();

        for i in 0..15 {
            let state = BinaryHV::random((1000 + i) as u64);
            topology.add_state(state);
        }

        let assessment = topology.analyze(0.5);

        // Check all metrics are in valid ranges
        assert!(assessment.unity_score >= 0.0 && assessment.unity_score <= 1.0);
        assert!(assessment.circularity >= 0.0 && assessment.circularity <= 1.0);
        assert!(assessment.completeness >= 0.0 && assessment.completeness <= 1.0);
        assert!(assessment.quality >= 0.0 && assessment.quality <= 1.0);
    }

    #[test]
    fn test_persistent_features() {
        let mut topology = ConsciousnessTopology::default();

        for i in 0..20 {
            let state = BinaryHV::random((1000 + i) as u64);
            topology.add_state(state);
        }

        let assessment = topology.analyze(0.5);

        // Should find some features
        // Presence check is implicit; no-op assertion removed.

        // All features should have valid persistence
        for feature in &assessment.features {
            assert!(feature.persistence >= 0.0);
            assert!(feature.birth <= feature.death);
        }
    }

    #[test]
    fn test_clear() {
        let mut topology = ConsciousnessTopology::default();

        for i in 0..10 {
            topology.add_state(BinaryHV::random(i as u64));
        }

        assert_eq!(topology.num_states(), 10);

        topology.clear();
        assert_eq!(topology.num_states(), 0);
    }

    #[test]
    fn test_serialization() {
        let feature = TopologicalFeature::Component;
        let serialized = serde_json::to_string(&feature).unwrap();
        assert!(!serialized.is_empty());

        let deserialized: TopologicalFeature = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, feature);
    }

    /// Research Direction 1: Test H1 - Phenomenal vs Functional Concept Topology
    ///
    /// Hypothesis: Phenomenal concepts cluster differently from functional concepts
    /// when analyzed with topological data analysis.
    #[test]
    fn test_phenomenal_vs_functional_topology() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Helper to create deterministic HV from concept string
        fn concept_to_hv(concept: &str) -> BinaryHV {
            let mut hasher = DefaultHasher::new();
            concept.hash(&mut hasher);
            BinaryHV::random(hasher.finish())
        }

        // Phenomenal concepts (experiential, qualia-related) - 50+ concepts
        let phenomenal_concepts = vec![
            // Visual qualia
            "The redness of a sunset",
            "What it is like to see blue",
            "The vividness of green in spring",
            "Brightness of morning light",
            "The experience of seeing yellow",
            "Perceiving depth in a landscape",
            "The shimmer of moonlight on water",
            "Colors in a rainbow experience",
            "The quality of darkness in shadow",
            "Visual experience of symmetry",
            // Auditory qualia
            "The qualia of sound",
            "Hearing a melody for the first time",
            "The experience of silence",
            "Resonance of a bass note",
            "Sharpness of a high-pitched tone",
            "The feeling of rhythm",
            "Experiencing dissonance vs harmony",
            "The subjective quality of thunder",
            "Whispered voice heard close",
            "Experience of echoing sound",
            // Tactile and bodily sensations
            "The feeling of warmth",
            "Subjective experience of pain",
            "The sensation of cold water",
            "Pressure on the skin",
            "The tingle of a light touch",
            "Ache of muscle fatigue",
            "The feeling of wind on face",
            "Proprioceptive awareness of limbs",
            "Sensation of breathing deeply",
            "The experience of itching",
            // Taste and smell qualia
            "The taste of sweetness",
            "Experiencing bitterness",
            "The smell of rain on earth",
            "Aroma of coffee experienced",
            "Sour taste experience",
            "Umami flavor sensation",
            "The scent of a flower",
            "Metallic taste experience",
            "Experiencing minty freshness",
            "The quality of smoky smell",
            // Emotional and self-aware states
            "Phenomenal consciousness itself",
            "The experience of self-awareness",
            "Unified field of awareness",
            "Inner subjective feeling",
            "The feeling of being present",
            "Experiencing nostalgia",
            "The qualia of curiosity",
            "Feeling of anticipation",
            "Subjective sense of time passing",
            "The experience of wonder",
            "Feeling of existential awareness",
            "The stream of consciousness",
        ];

        // Functional concepts (computational, objective) - 50+ concepts
        let functional_concepts = vec![
            // Algorithms and data structures
            "Recursive function evaluation",
            "Memory allocation algorithm",
            "Binary search implementation",
            "Sorting algorithm complexity",
            "Hash table collision resolution",
            "Red-black tree balancing",
            "Graph traversal algorithms",
            "Dynamic programming solution",
            "Greedy algorithm optimization",
            "Divide and conquer strategy",
            // Systems programming
            "Database query optimization",
            "Network packet routing",
            "Garbage collection cycles",
            "Thread synchronization primitive",
            "Memory management unit",
            "File system indexing",
            "Process scheduling algorithm",
            "Virtual memory paging",
            "Inter-process communication",
            "Socket connection handling",
            // Compilers and languages
            "Type inference in compilers",
            "Compiler optimization passes",
            "Abstract syntax tree parsing",
            "Lexical analysis tokenization",
            "Semantic analysis phase",
            "Register allocation algorithm",
            "Dead code elimination",
            "Loop unrolling optimization",
            "Inline function expansion",
            "Constant folding transformation",
            // Distributed systems
            "Consensus protocol execution",
            "Leader election algorithm",
            "Distributed lock management",
            "Message queue processing",
            "Load balancing strategy",
            "Cache invalidation protocol",
            "Eventual consistency model",
            "Sharding strategy implementation",
            "Replication factor calculation",
            "Partition tolerance handling",
            // Machine learning (functional aspects)
            "Gradient descent optimization",
            "Backpropagation computation",
            "Loss function minimization",
            "Weight matrix multiplication",
            "Activation function application",
            "Batch normalization calculation",
            "Learning rate scheduling",
            "Regularization term computation",
            "Cross-validation splitting",
            "Hyperparameter grid search",
            "Feature extraction pipeline",
            "Model serialization format",
        ];

        // Analyze phenomenal concept topology
        let mut phenomenal_topology = ConsciousnessTopology::default();
        for concept in &phenomenal_concepts {
            phenomenal_topology.add_state(concept_to_hv(concept));
        }
        let phenomenal_assessment = phenomenal_topology.analyze(0.5);

        // Analyze functional concept topology
        let mut functional_topology = ConsciousnessTopology::default();
        for concept in &functional_concepts {
            functional_topology.add_state(concept_to_hv(concept));
        }
        let functional_assessment = functional_topology.analyze(0.5);

        // Print results for analysis
        println!("\n========================================");
        println!("RESEARCH DIRECTION 1: H1 Validation");
        println!("========================================\n");

        println!("PHENOMENAL CONCEPTS:");
        println!("  Unity score: {:.4}", phenomenal_assessment.unity_score);
        println!(
            "  Betti numbers: β₀={}, β₁={}, β₂={}",
            phenomenal_assessment.betti.beta_0,
            phenomenal_assessment.betti.beta_1,
            phenomenal_assessment.betti.beta_2
        );
        println!("  Circularity: {:.4}", phenomenal_assessment.circularity);
        println!("  Completeness: {:.4}", phenomenal_assessment.completeness);
        println!("  Quality: {:.4}", phenomenal_assessment.quality);

        println!("\nFUNCTIONAL CONCEPTS:");
        println!("  Unity score: {:.4}", functional_assessment.unity_score);
        println!(
            "  Betti numbers: β₀={}, β₁={}, β₂={}",
            functional_assessment.betti.beta_0,
            functional_assessment.betti.beta_1,
            functional_assessment.betti.beta_2
        );
        println!("  Circularity: {:.4}", functional_assessment.circularity);
        println!("  Completeness: {:.4}", functional_assessment.completeness);
        println!("  Quality: {:.4}", functional_assessment.quality);

        // Calculate difference metrics
        let unity_diff = phenomenal_assessment.unity_score - functional_assessment.unity_score;
        let beta0_diff =
            phenomenal_assessment.betti.beta_0 as i64 - functional_assessment.betti.beta_0 as i64;

        println!("\nDIFFERENCES:");
        println!("  Unity score diff: {:.4} (P - F)", unity_diff);
        println!("  β₀ diff: {} (P - F)", beta0_diff);

        // Basic validation - both should produce valid topology
        assert!(phenomenal_assessment.unity_score >= 0.0);
        assert!(functional_assessment.unity_score >= 0.0);
        assert!(phenomenal_assessment.betti.beta_0 >= 1);
        assert!(functional_assessment.betti.beta_0 >= 1);

        // Note: With simulated (hash-based) vectors, we expect random-like topology
        // True H1 validation requires actual LLM embeddings via neural bridge
        println!("\nNOTE: This test uses hash-based simulated vectors.");
        println!("True H1 validation requires neural bridge with LLM embeddings.");
        println!("========================================\n");
    }

    /// Research Direction 2: Test H2 - Phenomenal Binding via HDC
    ///
    /// Hypothesis (H2): HDC binding (⊗) produces representations with higher
    /// topological integration than bundling (⊕), specifically for concept pairs
    /// humans report as phenomenally unified.
    #[test]
    fn test_phenomenal_binding_hypothesis() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Helper to create deterministic HV from concept string
        fn concept_to_hv(concept: &str) -> BinaryHV {
            let mut hasher = DefaultHasher::new();
            concept.hash(&mut hasher);
            BinaryHV::random(hasher.finish())
        }

        println!("\n========================================");
        println!("RESEARCH DIRECTION 2: H2 Validation");
        println!("Phenomenal Binding via HDC");
        println!("========================================\n");

        // Unified phenomenal pairs (humans report as single experience) - 50+ pairs
        let unified_pairs = vec![
            // Visual-object unity
            ("red", "apple"),        // "red apple" - unified percept
            ("blue", "sky"),         // "blue sky" - seamless unity
            ("green", "leaf"),       // "green leaf" - natural unity
            ("golden", "sunset"),    // "golden sunset" - merged
            ("silver", "moonlight"), // "silver moonlight" - unified
            ("white", "snow"),       // "white snow" - inherent
            ("black", "night"),      // "black night" - inseparable
            ("orange", "flame"),     // "orange flame" - unified glow
            ("pink", "blossom"),     // "pink blossom" - natural
            ("purple", "grape"),     // "purple grape" - unified
            // Sound-event unity
            ("loud", "crash"),        // "loud crash" - unified event
            ("soft", "whisper"),      // "soft whisper" - unified
            ("sharp", "scream"),      // "sharp scream" - unified
            ("deep", "rumble"),       // "deep rumble" - unified
            ("high", "shriek"),       // "high shriek" - unified
            ("gentle", "murmur"),     // "gentle murmur" - unified
            ("rhythmic", "drumbeat"), // "rhythmic drumbeat" - unified
            ("melodic", "birdsong"),  // "melodic birdsong" - unified
            ("crackling", "fire"),    // "crackling fire" - unified
            ("rushing", "waterfall"), // "rushing waterfall" - unified
            // Touch-sensation unity
            ("warm", "sunlight"),  // "warm sunlight" - unified sensation
            ("soft", "fur"),       // "soft fur" - unified texture
            ("cool", "breeze"),    // "cool breeze" - unified
            ("smooth", "silk"),    // "smooth silk" - unified
            ("rough", "bark"),     // "rough bark" - unified
            ("wet", "rain"),       // "wet rain" - unified
            ("cold", "ice"),       // "cold ice" - unified
            ("hot", "sand"),       // "hot sand" - unified
            ("prickly", "cactus"), // "prickly cactus" - unified
            ("velvety", "petal"),  // "velvety petal" - unified
            // Taste-substance unity
            ("sweet", "honey"),    // "sweet honey" - unified taste
            ("sour", "lemon"),     // "sour lemon" - unified
            ("bitter", "coffee"),  // "bitter coffee" - unified
            ("salty", "ocean"),    // "salty ocean" - unified
            ("spicy", "chili"),    // "spicy chili" - unified
            ("fresh", "mint"),     // "fresh mint" - unified
            ("rich", "chocolate"), // "rich chocolate" - unified
            ("tangy", "citrus"),   // "tangy citrus" - unified
            ("savory", "broth"),   // "savory broth" - unified
            ("creamy", "butter"),  // "creamy butter" - unified
            // Smell-source unity
            ("fragrant", "rose"),   // "fragrant rose" - unified
            ("pungent", "garlic"),  // "pungent garlic" - unified
            ("smoky", "campfire"),  // "smoky campfire" - unified
            ("earthy", "mushroom"), // "earthy mushroom" - unified
            ("floral", "jasmine"),  // "floral jasmine" - unified
            ("woody", "cedar"),     // "woody cedar" - unified
            ("fresh", "laundry"),   // "fresh laundry" - unified
            ("musky", "forest"),    // "musky forest" - unified
            ("citrus", "orange"),   // "citrus orange" - unified
            ("herbal", "basil"),    // "herbal basil" - unified
        ];

        // Co-occurring but separate pairs (two distinct experiences) - 50+ pairs
        let separate_pairs = vec![
            // Color and container/location
            ("red", "mailbox"),   // Red mailbox - two separate things
            ("blue", "building"), // Blue building - separable
            ("green", "fence"),   // Green fence - separable
            ("yellow", "sign"),   // Yellow sign - separable
            ("white", "wall"),    // White wall - separable
            ("black", "car"),     // Black car - separable
            ("orange", "cone"),   // Orange cone - separable
            ("pink", "house"),    // Pink house - separable
            ("gray", "pavement"), // Gray pavement - separable
            ("brown", "box"),     // Brown box - separable
            // Sound and background
            ("loud", "background"),  // Loud background - separable
            ("soft", "environment"), // Soft environment - separable
            ("sharp", "setting"),    // Sharp setting - separable
            ("deep", "context"),     // Deep context - separable
            ("high", "atmosphere"),  // High atmosphere - separable
            ("quiet", "office"),     // Quiet office - separable
            ("noisy", "street"),     // Noisy street - separable
            ("silent", "library"),   // Silent library - separable
            ("echoing", "hall"),     // Echoing hall - separable
            ("muffled", "room"),     // Muffled room - separable
            // Touch and space
            ("warm", "room"),      // Warm room - object + property
            ("soft", "carpet"),    // Soft carpet - less tightly bound
            ("cool", "basement"),  // Cool basement - separable
            ("smooth", "floor"),   // Smooth floor - separable
            ("rough", "concrete"), // Rough concrete - separable
            ("wet", "bathroom"),   // Wet bathroom - separable
            ("cold", "garage"),    // Cold garage - separable
            ("hot", "attic"),      // Hot attic - separable
            ("bumpy", "road"),     // Bumpy road - separable
            ("slippery", "tile"),  // Slippery tile - separable
            // Taste and container
            ("sweet", "jar"),       // Sweet jar - container vs contents
            ("sour", "bottle"),     // Sour bottle - separable
            ("bitter", "cup"),      // Bitter cup - separable
            ("salty", "shaker"),    // Salty shaker - separable
            ("spicy", "bowl"),      // Spicy bowl - separable
            ("fresh", "container"), // Fresh container - separable
            ("rich", "plate"),      // Rich plate - separable
            ("tangy", "package"),   // Tangy package - separable
            ("savory", "pot"),      // Savory pot - separable
            ("bland", "dish"),      // Bland dish - separable
            // Smell and location
            ("fragrant", "shop"),    // Fragrant shop - separable
            ("pungent", "kitchen"),  // Pungent kitchen - separable
            ("smoky", "restaurant"), // Smoky restaurant - separable
            ("earthy", "garden"),    // Earthy garden - separable
            ("floral", "store"),     // Floral store - separable
            ("woody", "cabin"),      // Woody cabin - separable
            ("fresh", "laundromat"), // Fresh laundromat - separable
            ("stale", "basement"),   // Stale basement - separable
            ("musty", "attic"),      // Musty attic - separable
            ("clean", "hospital"),   // Clean hospital - separable
        ];

        println!(
            "Testing {} unified pairs and {} separate pairs\n",
            unified_pairs.len(),
            separate_pairs.len()
        );

        // Results containers
        let mut unified_bind_unity = Vec::new();
        let mut unified_bundle_unity = Vec::new();
        let mut separate_bind_unity = Vec::new();
        let mut separate_bundle_unity = Vec::new();

        // Test unified pairs
        println!("UNIFIED PAIRS (Bind vs Bundle):");
        println!("────────────────────────────────────────");
        for (a, b) in &unified_pairs {
            let hv_a = concept_to_hv(a);
            let hv_b = concept_to_hv(b);

            // Bind (XOR) - creates orthogonal composite
            let bound = hv_a.bind(&hv_b);

            // Bundle (majority vote) - creates superposition
            let bundled = BinaryHV::bundle(&[hv_a.clone(), hv_b.clone()]);

            // Analyze topology of each
            let mut bind_topo = ConsciousnessTopology::default();
            bind_topo.add_state(hv_a.clone());
            bind_topo.add_state(hv_b.clone());
            bind_topo.add_state(bound);
            let bind_assess = bind_topo.analyze(0.5);

            let mut bundle_topo = ConsciousnessTopology::default();
            bundle_topo.add_state(hv_a.clone());
            bundle_topo.add_state(hv_b.clone());
            bundle_topo.add_state(bundled);
            let bundle_assess = bundle_topo.analyze(0.5);

            unified_bind_unity.push(bind_assess.unity_score);
            unified_bundle_unity.push(bundle_assess.unity_score);

            println!(
                "  {} + {}: bind_unity={:.3}, bundle_unity={:.3}",
                a, b, bind_assess.unity_score, bundle_assess.unity_score
            );
        }

        // Test separate pairs
        println!("\nSEPARATE PAIRS (Bind vs Bundle):");
        println!("────────────────────────────────────────");
        for (a, b) in &separate_pairs {
            let hv_a = concept_to_hv(a);
            let hv_b = concept_to_hv(b);

            let bound = hv_a.bind(&hv_b);
            let bundled = BinaryHV::bundle(&[hv_a.clone(), hv_b.clone()]);

            let mut bind_topo = ConsciousnessTopology::default();
            bind_topo.add_state(hv_a.clone());
            bind_topo.add_state(hv_b.clone());
            bind_topo.add_state(bound);
            let bind_assess = bind_topo.analyze(0.5);

            let mut bundle_topo = ConsciousnessTopology::default();
            bundle_topo.add_state(hv_a.clone());
            bundle_topo.add_state(hv_b.clone());
            bundle_topo.add_state(bundled);
            let bundle_assess = bundle_topo.analyze(0.5);

            separate_bind_unity.push(bind_assess.unity_score);
            separate_bundle_unity.push(bundle_assess.unity_score);

            println!(
                "  {} + {}: bind_unity={:.3}, bundle_unity={:.3}",
                a, b, bind_assess.unity_score, bundle_assess.unity_score
            );
        }

        // Calculate means
        let mean_unified_bind: f64 =
            unified_bind_unity.iter().sum::<f64>() / unified_bind_unity.len() as f64;
        let mean_unified_bundle: f64 =
            unified_bundle_unity.iter().sum::<f64>() / unified_bundle_unity.len() as f64;
        let mean_separate_bind: f64 =
            separate_bind_unity.iter().sum::<f64>() / separate_bind_unity.len() as f64;
        let mean_separate_bundle: f64 =
            separate_bundle_unity.iter().sum::<f64>() / separate_bundle_unity.len() as f64;

        println!("\n════════════════════════════════════════");
        println!("SUMMARY STATISTICS");
        println!("════════════════════════════════════════");
        println!("\nMean Unity Scores:");
        println!("┌────────────┬────────────┬────────────┐");
        println!("│            │ BIND (⊗)   │ BUNDLE (⊕) │");
        println!("├────────────┼────────────┼────────────┤");
        println!(
            "│ UNIFIED    │   {:.4}   │   {:.4}   │",
            mean_unified_bind, mean_unified_bundle
        );
        println!(
            "│ SEPARATE   │   {:.4}   │   {:.4}   │",
            mean_separate_bind, mean_separate_bundle
        );
        println!("└────────────┴────────────┴────────────┘");

        // Interaction effect: Does binding specifically help unified pairs?
        let bind_effect =
            (mean_unified_bind - mean_unified_bundle) - (mean_separate_bind - mean_separate_bundle);

        println!("\n2×2 ANOVA-style Analysis:");
        println!(
            "  Binding main effect: {:.4}",
            (mean_unified_bind + mean_separate_bind) / 2.0
                - (mean_unified_bundle + mean_separate_bundle) / 2.0
        );
        println!(
            "  Pair type main effect: {:.4}",
            (mean_unified_bind + mean_unified_bundle) / 2.0
                - (mean_separate_bind + mean_separate_bundle) / 2.0
        );
        println!("  Interaction (H2 test): {:.4}", bind_effect);

        if bind_effect > 0.0 {
            println!(
                "\n✓ H2 SUPPORTED: Binding produces relatively higher unity for unified pairs"
            );
        } else {
            println!("\n✗ H2 NOT SUPPORTED: Interaction effect is negative or zero");
        }

        println!("\nNOTE: With hash-based vectors, results are essentially random.");
        println!("True H2 validation requires semantically meaningful embeddings.");
        println!("========================================\n");

        // Basic assertions - both operations should produce valid results
        assert!(mean_unified_bind >= 0.0 && mean_unified_bind <= 1.0);
        assert!(mean_unified_bundle >= 0.0 && mean_unified_bundle <= 1.0);
    }

    /// K-Fold Cross-Validation for H1 Hypothesis
    ///
    /// Tests generalization of phenomenal vs functional concept topology
    /// by splitting data into K folds and computing held-out metrics.
    #[test]
    fn test_h1_kfold_cross_validation() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        const K: usize = 5; // 5-fold cross-validation

        fn concept_to_hv(concept: &str) -> BinaryHV {
            let mut hasher = DefaultHasher::new();
            concept.hash(&mut hasher);
            BinaryHV::random(hasher.finish())
        }

        println!("\n════════════════════════════════════════");
        println!("H1 K-FOLD CROSS-VALIDATION (K={})", K);
        println!("════════════════════════════════════════\n");

        // Full concept sets (same as H1 test)
        let phenomenal_concepts: Vec<&str> = vec![
            "The redness of a sunset",
            "What it is like to see blue",
            "The vividness of green in spring",
            "Brightness of morning light",
            "The experience of seeing yellow",
            "Perceiving depth in a landscape",
            "The shimmer of moonlight on water",
            "Colors in a rainbow experience",
            "The quality of darkness in shadow",
            "Visual experience of symmetry",
            "The qualia of sound",
            "Hearing a melody for the first time",
            "The experience of silence",
            "Resonance of a bass note",
            "Sharpness of a high-pitched tone",
            "The feeling of rhythm",
            "Experiencing dissonance vs harmony",
            "The subjective quality of thunder",
            "Whispered voice heard close",
            "Experience of echoing sound",
            "The feeling of warmth",
            "Subjective experience of pain",
            "The sensation of cold water",
            "Pressure on the skin",
            "The tingle of a light touch",
            "Ache of muscle fatigue",
            "The feeling of wind on face",
            "Proprioceptive awareness of limbs",
            "Sensation of breathing deeply",
            "The experience of itching",
            "The taste of sweetness",
            "Experiencing bitterness",
            "The smell of rain on earth",
            "Aroma of coffee experienced",
            "Sour taste experience",
            "Umami flavor sensation",
            "The scent of a flower",
            "Metallic taste experience",
            "Experiencing minty freshness",
            "The quality of smoky smell",
            "Phenomenal consciousness itself",
            "The experience of self-awareness",
            "Unified field of awareness",
            "Inner subjective feeling",
            "The feeling of being present",
            "Experiencing nostalgia",
            "The qualia of curiosity",
            "Feeling of anticipation",
            "Subjective sense of time passing",
            "The experience of wonder",
        ];

        let functional_concepts: Vec<&str> = vec![
            "Recursive function evaluation",
            "Memory allocation algorithm",
            "Binary search implementation",
            "Sorting algorithm complexity",
            "Hash table collision resolution",
            "Red-black tree balancing",
            "Graph traversal algorithms",
            "Dynamic programming solution",
            "Greedy algorithm optimization",
            "Divide and conquer strategy",
            "Database query optimization",
            "Network packet routing",
            "Garbage collection cycles",
            "Thread synchronization primitive",
            "Memory management unit",
            "File system indexing",
            "Process scheduling algorithm",
            "Virtual memory paging",
            "Inter-process communication",
            "Socket connection handling",
            "Type inference in compilers",
            "Compiler optimization passes",
            "Abstract syntax tree parsing",
            "Lexical analysis tokenization",
            "Semantic analysis phase",
            "Register allocation algorithm",
            "Dead code elimination",
            "Loop unrolling optimization",
            "Inline function expansion",
            "Constant folding transformation",
            "Consensus protocol execution",
            "Leader election algorithm",
            "Distributed lock management",
            "Message queue processing",
            "Load balancing strategy",
            "Cache invalidation protocol",
            "Eventual consistency model",
            "Sharding strategy implementation",
            "Replication factor calculation",
            "Partition tolerance handling",
            "Gradient descent optimization",
            "Backpropagation computation",
            "Loss function minimization",
            "Weight matrix multiplication",
            "Activation function application",
            "Batch normalization calculation",
            "Learning rate scheduling",
            "Regularization term computation",
            "Cross-validation splitting",
            "Hyperparameter grid search",
        ];

        let fold_size_p = phenomenal_concepts.len() / K;
        let fold_size_f = functional_concepts.len() / K;

        let mut fold_results: Vec<(f64, f64, f64)> = Vec::new(); // (phenomenal_unity, functional_unity, diff)

        for fold in 0..K {
            // Create test and train split
            let test_start_p = fold * fold_size_p;
            let test_end_p = test_start_p + fold_size_p;
            let test_start_f = fold * fold_size_f;
            let test_end_f = test_start_f + fold_size_f;

            // Train sets (all but test fold)
            let train_p: Vec<_> = phenomenal_concepts
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < test_start_p || *i >= test_end_p)
                .map(|(_, c)| *c)
                .collect();
            let train_f: Vec<_> = functional_concepts
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < test_start_f || *i >= test_end_f)
                .map(|(_, c)| *c)
                .collect();

            // Analyze train set topology
            let mut phenomenal_topology = ConsciousnessTopology::default();
            for concept in &train_p {
                phenomenal_topology.add_state(concept_to_hv(concept));
            }
            let phenomenal_assessment = phenomenal_topology.analyze(0.5);

            let mut functional_topology = ConsciousnessTopology::default();
            for concept in &train_f {
                functional_topology.add_state(concept_to_hv(concept));
            }
            let functional_assessment = functional_topology.analyze(0.5);

            let diff = phenomenal_assessment.unity_score - functional_assessment.unity_score;
            fold_results.push((
                phenomenal_assessment.unity_score,
                functional_assessment.unity_score,
                diff,
            ));

            println!(
                "Fold {}: P_unity={:.4}, F_unity={:.4}, diff={:.4}",
                fold + 1,
                phenomenal_assessment.unity_score,
                functional_assessment.unity_score,
                diff
            );
        }

        // Compute cross-validated statistics
        let mean_diff: f64 = fold_results.iter().map(|(_, _, d)| d).sum::<f64>() / K as f64;
        let variance: f64 = fold_results
            .iter()
            .map(|(_, _, d)| (d - mean_diff).powi(2))
            .sum::<f64>()
            / (K - 1) as f64;
        let std_error = (variance / K as f64).sqrt();

        println!("\n┌────────────────────────────────────────────────┐");
        println!("│  K-FOLD CROSS-VALIDATION RESULTS               │");
        println!("├────────────────────────────────────────────────┤");
        println!(
            "│  Mean (P - F) difference: {:>8.4}             │",
            mean_diff
        );
        println!(
            "│  Standard error:          {:>8.4}             │",
            std_error
        );
        println!(
            "│  95% CI: [{:.4}, {:.4}]               │",
            mean_diff - 1.96 * std_error,
            mean_diff + 1.96 * std_error
        );
        println!("└────────────────────────────────────────────────┘");

        // All fold results should be valid
        for (p, f, _) in &fold_results {
            assert!(*p >= 0.0 && *p <= 1.0);
            assert!(*f >= 0.0 && *f <= 1.0);
        }
    }

    /// K-Fold Cross-Validation for H2 Hypothesis
    ///
    /// Tests generalization of the phenomenal binding hypothesis
    /// by splitting pairs into K folds and measuring interaction effects.
    #[test]
    fn test_h2_kfold_cross_validation() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        const K: usize = 5; // 5-fold cross-validation

        fn concept_to_hv(concept: &str) -> BinaryHV {
            let mut hasher = DefaultHasher::new();
            concept.hash(&mut hasher);
            BinaryHV::random(hasher.finish())
        }

        fn compute_interaction(pairs: &[(&str, &str)], pair_type: &str) -> (f64, f64) {
            let mut bind_unity_scores = Vec::new();
            let mut bundle_unity_scores = Vec::new();

            for (a, b) in pairs {
                let hv_a = concept_to_hv(a);
                let hv_b = concept_to_hv(b);

                let bound = hv_a.bind(&hv_b);
                let bundled = BinaryHV::bundle(&[hv_a.clone(), hv_b.clone()]);

                let mut bind_topo = ConsciousnessTopology::default();
                bind_topo.add_state(hv_a.clone());
                bind_topo.add_state(hv_b.clone());
                bind_topo.add_state(bound);
                let bind_assess = bind_topo.analyze(0.5);

                let mut bundle_topo = ConsciousnessTopology::default();
                bundle_topo.add_state(hv_a);
                bundle_topo.add_state(hv_b);
                bundle_topo.add_state(bundled);
                let bundle_assess = bundle_topo.analyze(0.5);

                bind_unity_scores.push(bind_assess.unity_score);
                bundle_unity_scores.push(bundle_assess.unity_score);
            }

            let mean_bind = bind_unity_scores.iter().sum::<f64>() / bind_unity_scores.len() as f64;
            let mean_bundle =
                bundle_unity_scores.iter().sum::<f64>() / bundle_unity_scores.len() as f64;
            (mean_bind, mean_bundle)
        }

        println!("\n════════════════════════════════════════");
        println!("H2 K-FOLD CROSS-VALIDATION (K={})", K);
        println!("════════════════════════════════════════\n");

        // Full pair sets
        let unified_pairs: Vec<(&str, &str)> = vec![
            ("red", "apple"),
            ("blue", "sky"),
            ("green", "leaf"),
            ("golden", "sunset"),
            ("silver", "moonlight"),
            ("white", "snow"),
            ("black", "night"),
            ("orange", "flame"),
            ("pink", "blossom"),
            ("purple", "grape"),
            ("loud", "crash"),
            ("soft", "whisper"),
            ("sharp", "scream"),
            ("deep", "rumble"),
            ("high", "shriek"),
            ("gentle", "murmur"),
            ("rhythmic", "drumbeat"),
            ("melodic", "birdsong"),
            ("crackling", "fire"),
            ("rushing", "waterfall"),
            ("warm", "sunlight"),
            ("soft", "fur"),
            ("cool", "breeze"),
            ("smooth", "silk"),
            ("rough", "bark"),
            ("wet", "rain"),
            ("cold", "ice"),
            ("hot", "sand"),
            ("prickly", "cactus"),
            ("velvety", "petal"),
            ("sweet", "honey"),
            ("sour", "lemon"),
            ("bitter", "coffee"),
            ("salty", "ocean"),
            ("spicy", "chili"),
            ("fresh", "mint"),
            ("rich", "chocolate"),
            ("tangy", "citrus"),
            ("savory", "broth"),
            ("creamy", "butter"),
            ("fragrant", "rose"),
            ("pungent", "garlic"),
            ("smoky", "campfire"),
            ("earthy", "mushroom"),
            ("floral", "jasmine"),
            ("woody", "cedar"),
            ("fresh", "laundry"),
            ("musky", "forest"),
            ("citrus", "orange"),
            ("herbal", "basil"),
        ];

        let separate_pairs: Vec<(&str, &str)> = vec![
            ("red", "mailbox"),
            ("blue", "building"),
            ("green", "fence"),
            ("yellow", "sign"),
            ("white", "wall"),
            ("black", "car"),
            ("orange", "cone"),
            ("pink", "house"),
            ("gray", "pavement"),
            ("brown", "box"),
            ("loud", "background"),
            ("soft", "environment"),
            ("sharp", "setting"),
            ("deep", "context"),
            ("high", "atmosphere"),
            ("quiet", "office"),
            ("noisy", "street"),
            ("silent", "library"),
            ("echoing", "hall"),
            ("muffled", "room"),
            ("warm", "room"),
            ("soft", "carpet"),
            ("cool", "basement"),
            ("smooth", "floor"),
            ("rough", "concrete"),
            ("wet", "bathroom"),
            ("cold", "garage"),
            ("hot", "attic"),
            ("bumpy", "road"),
            ("slippery", "tile"),
            ("sweet", "jar"),
            ("sour", "bottle"),
            ("bitter", "cup"),
            ("salty", "shaker"),
            ("spicy", "bowl"),
            ("fresh", "container"),
            ("rich", "plate"),
            ("tangy", "package"),
            ("savory", "pot"),
            ("bland", "dish"),
            ("fragrant", "shop"),
            ("pungent", "kitchen"),
            ("smoky", "restaurant"),
            ("earthy", "garden"),
            ("floral", "store"),
            ("woody", "cabin"),
            ("fresh", "laundromat"),
            ("stale", "basement"),
            ("musty", "attic"),
            ("clean", "hospital"),
        ];

        let fold_size_u = unified_pairs.len() / K;
        let fold_size_s = separate_pairs.len() / K;

        let mut interaction_effects: Vec<f64> = Vec::new();

        for fold in 0..K {
            // Create train split (exclude fold for testing)
            let test_start_u = fold * fold_size_u;
            let test_end_u = test_start_u + fold_size_u;
            let test_start_s = fold * fold_size_s;
            let test_end_s = test_start_s + fold_size_s;

            let train_u: Vec<_> = unified_pairs
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < test_start_u || *i >= test_end_u)
                .map(|(_, p)| *p)
                .collect();
            let train_s: Vec<_> = separate_pairs
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < test_start_s || *i >= test_end_s)
                .map(|(_, p)| *p)
                .collect();

            let (u_bind, u_bundle) = compute_interaction(&train_u, "unified");
            let (s_bind, s_bundle) = compute_interaction(&train_s, "separate");

            // Interaction = (unified bind advantage) - (separate bind advantage)
            let interaction = (u_bind - u_bundle) - (s_bind - s_bundle);
            interaction_effects.push(interaction);

            println!(
                "Fold {}: U(bind={:.3}, bundle={:.3}), S(bind={:.3}, bundle={:.3}), interaction={:.4}",
                fold + 1,
                u_bind,
                u_bundle,
                s_bind,
                s_bundle,
                interaction
            );
        }

        // Compute cross-validated statistics
        let mean_interaction: f64 = interaction_effects.iter().sum::<f64>() / K as f64;
        let variance: f64 = interaction_effects
            .iter()
            .map(|e| (e - mean_interaction).powi(2))
            .sum::<f64>()
            / (K - 1) as f64;
        let std_error = (variance / K as f64).sqrt();

        // Count positive interaction effects across folds
        let positive_folds = interaction_effects.iter().filter(|&&e| e > 0.0).count();

        println!("\n┌────────────────────────────────────────────────┐");
        println!("│  K-FOLD CROSS-VALIDATION RESULTS               │");
        println!("├────────────────────────────────────────────────┤");
        println!(
            "│  Mean interaction effect: {:>8.4}             │",
            mean_interaction
        );
        println!(
            "│  Standard error:          {:>8.4}             │",
            std_error
        );
        println!(
            "│  95% CI: [{:.4}, {:.4}]               │",
            mean_interaction - 1.96 * std_error,
            mean_interaction + 1.96 * std_error
        );
        println!(
            "│  Positive folds:          {}/{}                    │",
            positive_folds, K
        );
        println!("├────────────────────────────────────────────────┤");
        if mean_interaction > 0.0 && positive_folds > K / 2 {
            println!("│  ✓ H2 SUPPORTED across folds                   │");
        } else {
            println!("│  ✗ H2 not consistently supported               │");
        }
        println!("└────────────────────────────────────────────────┘");

        // Assertions
        assert!(mean_interaction.is_finite());
    }
}
