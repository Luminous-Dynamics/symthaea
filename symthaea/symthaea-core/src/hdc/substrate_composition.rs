// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Substrate Composition: Weighted Mixtures of Substrates
//
// Models heterogeneous systems where consciousness runs on a weighted
// combination of substrate types (e.g., 60% silicon + 40% neuromorphic).
//
// Science: Hybrid substrate composition extends Putnam's (1967) multiple
// realizability to mixed-media systems where different components of the
// conscious process run on different physical substrates simultaneously.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use super::substrate_independence::{SubstrateRequirements, SubstrateType};

/// A weighted composition of multiple substrate types.
///
/// Models a hybrid system where consciousness emerges from a mixture
/// of different physical substrates, each contributing a fraction of
/// the total computation.
///
/// # Example
///
/// ```
/// use symthaea_core::hdc::substrate_composition::SubstrateComposition;
/// use symthaea_core::hdc::substrate_independence::SubstrateType;
///
/// let comp = SubstrateComposition::new(
///     "neural-silicon hybrid".into(),
///     vec![
///         (SubstrateType::BiologicalNeurons, 0.5),
///         (SubstrateType::SiliconDigital, 0.5),
///     ],
/// ).unwrap();
///
/// // Feasibility is a weighted blend of component feasibilities
/// let f = comp.feasibility();
/// assert!(f > 0.5 && f < 1.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateComposition {
    /// Human-readable name for this composition.
    pub name: String,
    /// Weights per substrate type (must sum to ~1.0).
    pub weights: HashMap<SubstrateType, f32>,
}

impl SubstrateComposition {
    /// Create a new substrate composition from a list of (type, weight) pairs.
    ///
    /// Validates that:
    /// - All weights are non-negative.
    /// - Weights sum to approximately 1.0 (±0.01).
    /// - At least one substrate is present.
    ///
    /// Aliases are canonicalized (e.g., `Silicon` → `SiliconDigital`).
    pub fn new(name: String, components: Vec<(SubstrateType, f32)>) -> Result<Self, String> {
        if components.is_empty() {
            return Err("Composition must have at least one substrate".into());
        }

        let mut weights = HashMap::new();
        for (substrate, weight) in components {
            if weight < 0.0 {
                return Err(format!(
                    "Weight for {:?} must be non-negative, got {weight}",
                    substrate
                ));
            }
            let canonical = substrate.canonical();
            *weights.entry(canonical).or_insert(0.0) += weight;
        }

        let total: f32 = weights.values().sum();
        if (total - 1.0).abs() > 0.01 {
            return Err(format!("Weights must sum to ~1.0 (±0.01), got {total:.4}"));
        }

        Ok(Self { name, weights })
    }

    /// Compute the weighted consciousness feasibility (0.0–1.0).
    ///
    /// Each substrate's feasibility is computed from its 9-dimensional
    /// requirements profile, then combined as a weighted average.
    pub fn feasibility(&self) -> f64 {
        let mut total = 0.0f64;
        for (&substrate, &weight) in &self.weights {
            let req = requirements_for(substrate);
            total += req.consciousness_feasibility() * weight as f64;
        }
        total
    }

    /// Compute weighted interpolation of all 9 requirement dimensions.
    ///
    /// Useful for understanding the composite profile across dimensions.
    pub fn per_dimension_requirements(&self) -> SubstrateRequirements {
        let mut causality = 0.0f64;
        let mut integration = 0.0f64;
        let mut dynamics = 0.0f64;
        let mut recurrence = 0.0f64;
        let mut binding = 0.0f64;
        let mut attention = 0.0f64;
        let mut workspace = 0.0f64;
        let mut hot = 0.0f64;
        let mut quantum = 0.0f64;

        for (&substrate, &weight) in &self.weights {
            let w = weight as f64;
            let req = requirements_for(substrate);
            causality += req.causality * w;
            integration += req.integration_capacity * w;
            dynamics += req.temporal_dynamics * w;
            recurrence += req.recurrence * w;
            binding += req.binding_capability * w;
            attention += req.attention_capability * w;
            workspace += req.workspace_capability * w;
            hot += req.hot_capability * w;
            quantum += req.quantum_support * w;
        }

        SubstrateRequirements {
            causality,
            integration_capacity: integration,
            temporal_dynamics: dynamics,
            recurrence,
            binding_capability: binding,
            attention_capability: attention,
            workspace_capability: workspace,
            hot_capability: hot,
            quantum_support: quantum,
        }
    }

    /// Get the substrate with the highest weight.
    pub fn dominant_substrate(&self) -> SubstrateType {
        self.weights
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(&st, _)| st)
            .unwrap_or(SubstrateType::SiliconDigital)
    }

    /// Get the minimum workspace capability across all components.
    ///
    /// Workspace is a critical bottleneck for consciousness (Global Workspace Theory).
    /// In a hybrid system the weakest link in workspace broadcasting limits the whole.
    pub fn workspace_bottleneck(&self) -> f64 {
        self.weights
            .keys()
            .map(|&st| requirements_for(st).workspace_capability)
            .fold(f64::INFINITY, f64::min)
    }
}

impl fmt::Display for SubstrateComposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<_> = self.weights.iter().filter(|&(_, &w)| w > 0.0).collect();
        // Sort by weight descending for consistent display
        parts.sort_by(|a, b| b.1.total_cmp(a.1));

        let desc: Vec<String> = parts
            .iter()
            .map(|(st, w)| format!("{:.0}% {}", *w * 100.0, st.name()))
            .collect();

        write!(f, "{}", desc.join(" + "))
    }
}

// ── Predefined Compositions ──────────────────────────────────────────

/// 50/50 biological neurons + silicon digital.
///
/// Models a brain-computer interface where half the processing
/// happens in wetware and half in silicon.
pub fn hybrid_neural_silicon() -> SubstrateComposition {
    SubstrateComposition::new(
        "Neural-Silicon Hybrid".into(),
        vec![
            (SubstrateType::BiologicalNeurons, 0.5),
            (SubstrateType::SiliconDigital, 0.5),
        ],
    )
    .expect("predefined composition is valid")
}

/// 80% silicon digital + 20% quantum computer.
///
/// Models a classical system enhanced with a quantum co-processor
/// for entanglement-based binding.
pub fn quantum_enhanced_classical() -> SubstrateComposition {
    SubstrateComposition::new(
        "Quantum-Enhanced Classical".into(),
        vec![
            (SubstrateType::SiliconDigital, 0.8),
            (SubstrateType::QuantumComputer, 0.2),
        ],
    )
    .expect("predefined composition is valid")
}

/// 60% neuromorphic chip + 40% photonic processor.
///
/// Models a spike-based processor paired with ultra-fast
/// photonic attention circuits.
pub fn neuromorphic_photonic() -> SubstrateComposition {
    SubstrateComposition::new(
        "Neuromorphic-Photonic".into(),
        vec![
            (SubstrateType::NeuromorphicChip, 0.6),
            (SubstrateType::PhotonicProcessor, 0.4),
        ],
    )
    .expect("predefined composition is valid")
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Get the requirements profile for a substrate type.
fn requirements_for(substrate: SubstrateType) -> SubstrateRequirements {
    match substrate.canonical() {
        SubstrateType::BiologicalNeurons => SubstrateRequirements::biological_neurons(),
        SubstrateType::SiliconDigital => SubstrateRequirements::silicon_digital(),
        SubstrateType::QuantumComputer => SubstrateRequirements::quantum_computer(),
        SubstrateType::PhotonicProcessor => SubstrateRequirements::photonic_processor(),
        SubstrateType::NeuromorphicChip => SubstrateRequirements::neuromorphic_chip(),
        SubstrateType::BiochemicalComputer => SubstrateRequirements::biochemical_computer(),
        SubstrateType::HybridSystem => SubstrateRequirements::hybrid_system(),
        SubstrateType::ExoticSubstrate => SubstrateRequirements::exotic_substrate(),
        SubstrateType::SpacecraftComputer => SubstrateRequirements::spacecraft_computer(),
        _ => SubstrateRequirements::silicon_digital(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction_valid() {
        let comp = SubstrateComposition::new(
            "test".into(),
            vec![
                (SubstrateType::SiliconDigital, 0.7),
                (SubstrateType::QuantumComputer, 0.3),
            ],
        );
        assert!(comp.is_ok());
    }

    #[test]
    fn test_weight_validation_negative() {
        let result =
            SubstrateComposition::new("bad".into(), vec![(SubstrateType::SiliconDigital, -0.5)]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("non-negative"));
    }

    #[test]
    fn test_weight_validation_sum() {
        let result = SubstrateComposition::new(
            "bad".into(),
            vec![
                (SubstrateType::SiliconDigital, 0.5),
                (SubstrateType::QuantumComputer, 0.3),
            ],
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("sum to ~1.0"));
    }

    #[test]
    fn test_feasibility_weighted() {
        let bio_only =
            SubstrateComposition::new("bio".into(), vec![(SubstrateType::BiologicalNeurons, 1.0)])
                .unwrap();

        let silicon_only =
            SubstrateComposition::new("silicon".into(), vec![(SubstrateType::SiliconDigital, 1.0)])
                .unwrap();

        let hybrid = SubstrateComposition::new(
            "mix".into(),
            vec![
                (SubstrateType::BiologicalNeurons, 0.5),
                (SubstrateType::SiliconDigital, 0.5),
            ],
        )
        .unwrap();

        let bio_f = bio_only.feasibility();
        let sil_f = silicon_only.feasibility();
        let hyb_f = hybrid.feasibility();

        // Hybrid should be between the two pure substrates
        assert!(hyb_f > sil_f.min(bio_f) - 0.01);
        assert!(hyb_f < bio_f.max(sil_f) + 0.01);

        // Check it's approximately the average
        let expected = (bio_f + sil_f) / 2.0;
        assert!(
            (hyb_f - expected).abs() < 0.01,
            "50/50 hybrid feasibility {hyb_f:.4} should be near average {expected:.4}"
        );
    }

    #[test]
    fn test_per_dimension_interpolation() {
        let comp = SubstrateComposition::new(
            "mix".into(),
            vec![
                (SubstrateType::BiologicalNeurons, 0.5),
                (SubstrateType::SiliconDigital, 0.5),
            ],
        )
        .unwrap();

        let req = comp.per_dimension_requirements();

        // Bio causality=1.0, Silicon causality=1.0 → blend=1.0
        assert!((req.causality - 1.0).abs() < 0.01);

        // Bio binding=1.0, Silicon binding=0.7 → blend=0.85
        assert!(
            (req.binding_capability - 0.85).abs() < 0.01,
            "Blended binding should be ~0.85, got {:.4}",
            req.binding_capability
        );
    }

    #[test]
    fn test_workspace_bottleneck() {
        let comp = SubstrateComposition::new(
            "bottleneck test".into(),
            vec![
                (SubstrateType::SiliconDigital, 0.5),  // workspace=0.9
                (SubstrateType::QuantumComputer, 0.5), // workspace=0.6
            ],
        )
        .unwrap();

        let bottleneck = comp.workspace_bottleneck();
        assert!(
            (bottleneck - 0.6).abs() < 0.01,
            "Workspace bottleneck should be 0.6 (quantum), got {bottleneck:.4}"
        );
    }

    #[test]
    fn test_predefined_compositions() {
        let ns = hybrid_neural_silicon();
        assert!(ns.feasibility() > 0.5);
        assert!(ns.feasibility() < 1.0);

        let qec = quantum_enhanced_classical();
        assert!(qec.feasibility() > 0.5);

        let np = neuromorphic_photonic();
        assert!(np.feasibility() > 0.5);
    }

    #[test]
    fn test_display_format() {
        let comp = SubstrateComposition::new(
            "display test".into(),
            vec![
                (SubstrateType::SiliconDigital, 0.6),
                (SubstrateType::NeuromorphicChip, 0.4),
            ],
        )
        .unwrap();

        let display = format!("{comp}");
        assert!(display.contains("60%"), "Should show 60%: {display}");
        assert!(display.contains("40%"), "Should show 40%: {display}");
        assert!(
            display.contains("Silicon"),
            "Should mention Silicon: {display}"
        );
        assert!(
            display.contains("Neuromorphic"),
            "Should mention Neuromorphic: {display}"
        );
    }

    #[test]
    fn test_dominant_substrate() {
        let comp = SubstrateComposition::new(
            "dominant test".into(),
            vec![
                (SubstrateType::SiliconDigital, 0.3),
                (SubstrateType::NeuromorphicChip, 0.7),
            ],
        )
        .unwrap();

        assert_eq!(comp.dominant_substrate(), SubstrateType::NeuromorphicChip);
    }
}
