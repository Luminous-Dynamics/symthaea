// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Monitor Tier: groups optional consciousness analyzer fields.

use crate::consciousness::consciousness_resonance::ResonanceAnalyzer;
use crate::consciousness::consciousness_thermodynamics::ConsciousnessThermodynamicsAnalyzer;
use crate::consciousness::embodied_cognition::EmbodiedConsciousnessAnalyzer;
use crate::consciousness::hierarchical_free_energy::HierarchicalFreeEnergy;
use crate::consciousness::phenomenal_binding::TemporalSynchronizationAnalyzer;
use crate::consciousness::quantum_coherence::QuantumCoherenceAnalyzer;
use crate::consciousness::temporal_consciousness::TemporalConsciousnessAnalyzer;

/// Groups 7 optional consciousness analyzer CLS fields into a single tier.
///
/// All fields are `Option<T>` gated by config `enable_*` flags.
/// Reduces CLS field count by 6 (7 fields → 1).
#[derive(Default)]
pub(crate) struct ConsciousnessMonitorTier {
    /// Harmonic mode extraction from Phi time-series.
    pub resonance: Option<ResonanceAnalyzer>,

    /// CfC hidden state superposition richness.
    pub quantum_coherence: Option<QuantumCoherenceAnalyzer>,

    /// Phi trajectory, continuity, Husserlian time, temporal identity coherence.
    pub temporal: Option<TemporalConsciousnessAnalyzer>,

    /// Virtual body → body schema → sensorimotor coupling.
    pub embodied: Option<EmbodiedConsciousnessAnalyzer>,

    /// 7D consciousness [Φ, B, W, A, R, E, K] entropy/FE analysis.
    pub thermodynamics: Option<ConsciousnessThermodynamicsAnalyzer>,

    /// Phase coherence across consciousness dimensions.
    pub phenomenal_binding: Option<TemporalSynchronizationAnalyzer>,

    /// Multi-level variational free energy hierarchy.
    pub hierarchical_free_energy: Option<HierarchicalFreeEnergy>,
}
