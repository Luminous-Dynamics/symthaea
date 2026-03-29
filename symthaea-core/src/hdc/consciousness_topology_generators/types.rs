// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness topology types and structures.

use super::super::unified_hv::ContinuousHV;

/// A consciousness topology represented with ContinuousHV
#[derive(Clone, Debug)]
pub struct ConsciousnessTopology {
    /// Number of nodes in the topology
    pub n_nodes: usize,

    /// Dimension of hypervectors
    pub dim: usize,

    /// Node representations (each encodes its connections)
    pub node_representations: Vec<ContinuousHV>,

    /// Node identities (basis vectors)
    pub node_identities: Vec<ContinuousHV>,

    /// Topology type
    pub topology_type: TopologyType,

    /// Edge list (node pairs)
    /// Added back for synthesis module compatibility
    pub edges: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TopologyType {
    Random,
    Star,
    Ring,
    Line,
    BinaryTree,
    DenseNetwork,
    Modular,
    Lattice,
    Sphere,      // 2-manifold: S²
    Torus,       // 2-manifold: T²
    KleinBottle, // Non-orientable 2-manifold
    SmallWorld,
    MobiusStrip,
    Hyperbolic,
    ScaleFree,
    Fractal,          // Tier 3: Generic fractal (deprecated - use specific types)
    SierpinskiGasket, // Tier 3: Fractal triangle (d≈1.585)
    FractalTree,      // Tier 3: Self-similar hierarchical branching
    KochSnowflake,    // Tier 3: Fractal curve (d≈1.262)
    MengerSponge,     // Tier 3: 3D fractal (d≈2.727)
    CantorSet,        // Tier 3: Disconnected fractal (d≈0.631)
    Hypercube,        // Tier 3: 3D/4D/5D dimensional scaling
    Quantum,          // Tier 3: Superposition of topologies
    // Tier 4: Extended topologies (Revolutionary #102)
    CorticalColumn,    // 6-layer hierarchical (like mammalian cortex)
    Feedforward,       // Layered neural network structure
    Recurrent,         // Feedback loops (like RNNs)
    Bipartite,         // Two-layer structure (like retina → V1)
    CorePeriphery,     // Dense core, sparse periphery
    BowTie,            // IN → CORE → OUT structure
    Attention,         // Query-Key-Value structure
    Residual,          // Skip connections (like ResNets)
    PetersenGraph,     // Famous 10-node highly symmetric graph
    CompleteBipartite, // K_{n,n} - All-to-all between groups
}

/// Statistics about similarity structure
#[derive(Clone, Debug)]
pub struct SimilarityStats {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub heterogeneity: f32, // Normalized measure of diversity
}
