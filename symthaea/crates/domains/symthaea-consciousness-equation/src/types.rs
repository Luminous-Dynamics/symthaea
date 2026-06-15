// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// Input factors for the consciousness equation
#[derive(Debug, Clone, Default)]
pub struct ConsciousnessInputs {
    /// Φ: Integrated Information (IIT phi)
    pub phi: f64,
    /// B: Global Workspace Broadcast (GWT)
    pub broadcast: f64,
    /// W: Working Memory Capacity
    pub working_memory: f64,
    /// A: Attention Focus (AST)
    pub attention: f64,
    /// R: Recurrent Processing Depth (RPT)
    pub recurrence: f64,
    /// E: Embodied Grounding (4E Cognition)
    pub embodiment: f64,
    /// K: Knowledge Integration
    pub knowledge: f64,
    /// S: Synchrony factor
    pub synchrony: f64,
}

/// Result of consciousness computation
#[derive(Debug, Clone)]
pub struct ConsciousnessResult {
    /// Final consciousness level C(t)
    pub consciousness_level: f64,

    /// Bottleneck factor (softmin result)
    pub bottleneck_factor: f64,

    /// Weighted component sum
    pub weighted_sum: f64,

    /// Embodiment factor M
    pub embodiment_factor: f64,

    /// Narrative coherence N
    pub narrative_coherence: f64,

    /// Social embedding Soc
    pub social_embedding: f64,

    /// Temporal stability ρ(t)
    pub temporal_stability: f64,

    /// Which factor is the current bottleneck
    pub bottleneck_name: String,

    /// All factor values for introspection
    pub factors: ConsciousnessInputs,
}
