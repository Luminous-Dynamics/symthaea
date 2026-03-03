//! Substrate independence manager.
//!
//! Consolidates the 6 substrate-related fields from CognitiveLoopService
//! into a single cohesive manager. Handles feasibility computation,
//! validation overlays, speed/scale modulation, and runtime reconfiguration.

use symthaea_core::hdc::substrate_independence::{SubstrateRequirements, SubstrateType};

use super::config::CognitiveLoopConfig;

/// Default honest confidence for substrates not in the validation framework.
/// Matches EvidenceLevel::Theoretical.confidence() = 0.10.
const THEORETICAL_CONFIDENCE: f64 = 0.10;

/// Manages substrate-dependent consciousness feasibility, validation overlays,
/// and speed/scale modulation factors.
#[derive(Debug, Clone)]
pub(crate) struct SubstrateManager {
    /// Pre-computed substrate feasibility [0,1] from config.substrate_type.
    /// Scales Equation V2 consciousness to reflect substrate limitations.
    pub(crate) feasibility: f64,

    /// Pending substrate transition description for telemetry.
    /// Populated by `reconfigure_substrate()`/`reconfigure_composition()`,
    /// drained into `CycleMetadata.substrate_transition` once per cycle.
    pub(crate) pending_transition: Option<String>,

    /// Honest evidence confidence for the current substrate (0.0–0.95).
    /// From SubstrateValidationFramework: biological=0.95, silicon=0.10, etc.
    pub(crate) honest_confidence: f64,

    /// Effective feasibility after validation overlay blending.
    /// When overlay disabled: equals feasibility.
    /// When enabled: feasibility × (floor + (1 − floor) × honest_confidence).
    pub(crate) effective_feasibility: f64,

    /// CfC tau factor from substrate speed modulation [0.5, 2.0].
    /// 1.0 when speed modulation is disabled.
    pub(crate) tau_factor: f32,

    /// Scale pressure: log10(substrate_max_scale / bio_max_scale).
    /// Telemetry-only. 0.0 when speed modulation is disabled.
    pub(crate) scale_pressure: f32,
}

impl SubstrateManager {
    /// Create a new substrate manager from config.
    pub fn new(config: &CognitiveLoopConfig) -> Self {
        let feasibility = if let Some(ref comp) = config.substrate_composition {
            comp.feasibility()
        } else {
            Self::requirements_for(&config.substrate_type).consciousness_feasibility()
        };

        let mut mgr = Self {
            feasibility,
            pending_transition: None,
            honest_confidence: 0.0,
            effective_feasibility: feasibility,
            tau_factor: 1.0,
            scale_pressure: 0.0,
        };
        mgr.recompute_effective_feasibility(config);
        mgr.recompute_substrate_dynamics(config);
        mgr
    }

    /// Switch substrate type at runtime, recomputing consciousness feasibility.
    ///
    /// Returns (old_feasibility, new_feasibility) for telemetry.
    pub fn reconfigure_substrate(
        &mut self,
        config: &mut CognitiveLoopConfig,
        substrate: SubstrateType,
    ) -> (f64, f64) {
        let old = self.feasibility;
        let old_type = config.substrate_type;
        let canonical = substrate.canonical();
        self.feasibility = Self::requirements_for(&canonical).consciousness_feasibility();
        config.substrate_type = canonical;
        // Clear any stale composition — single-substrate mode now.
        config.substrate_composition = None;
        self.pending_transition = Some(format!(
            "{:?} -> {:?} ({:.3} -> {:.3})",
            old_type, canonical, old, self.feasibility
        ));
        self.recompute_effective_feasibility(config);
        self.recompute_substrate_dynamics(config);
        (old, self.feasibility)
    }

    /// Switch to a substrate composition at runtime, recomputing feasibility.
    pub fn reconfigure_composition(
        &mut self,
        config: &mut CognitiveLoopConfig,
        composition: symthaea_core::hdc::substrate_composition::SubstrateComposition,
    ) {
        let old_feas = self.feasibility;
        self.feasibility = composition.feasibility();
        self.pending_transition = Some(format!(
            "-> {} ({:.3} -> {:.3})",
            composition.name, old_feas, self.feasibility
        ));
        config.substrate_composition = Some(composition);
        self.recompute_effective_feasibility(config);
        self.recompute_substrate_dynamics(config);
    }

    /// Recompute effective feasibility from raw feasibility × validation overlay.
    /// Called after any substrate/composition change and at startup.
    pub fn recompute_effective_feasibility(&mut self, config: &CognitiveLoopConfig) {
        let framework =
            symthaea_core::hdc::substrate_validation::SubstrateValidationFramework::new();
        self.honest_confidence = if let Some(ref comp) = config.substrate_composition {
            let mut blended = 0.0f64;
            for (sub, &weight) in &comp.weights {
                let conf = match Self::substrate_validation_key(sub) {
                    Some(k) => framework.honest_feasibility(k),
                    None => THEORETICAL_CONFIDENCE,
                };
                blended += conf * weight as f64;
            }
            blended
        } else {
            match Self::substrate_validation_key(&config.substrate_type) {
                Some(k) => framework.honest_feasibility(k),
                None => THEORETICAL_CONFIDENCE,
            }
        };

        if config.enable_validation_overlay {
            let floor = config.validation_skepticism_floor;
            let confidence = self.honest_confidence;
            self.effective_feasibility =
                self.feasibility * (floor + (1.0 - floor) * confidence);
        } else {
            self.effective_feasibility = self.feasibility;
        }
    }

    /// Recompute substrate speed/scale modulation factors.
    /// Called after any substrate change and at startup.
    /// When a composition is set, weight-blends speed/scale from all components.
    pub fn recompute_substrate_dynamics(&mut self, config: &CognitiveLoopConfig) {
        if !config.enable_substrate_speed_modulation {
            self.tau_factor = 1.0;
            self.scale_pressure = 0.0;
            return;
        }

        let bio_speed = SubstrateType::BiologicalNeurons.operation_speed();
        let bio_scale = SubstrateType::BiologicalNeurons.max_scale();

        let (sub_speed, sub_scale) = if let Some(ref comp) = config.substrate_composition {
            // Geometric mean (log-space blend) — physically meaningful when speeds
            // span 12 orders of magnitude.  exp(Σ wᵢ·ln(sᵢ)) avoids the slowest
            // component dominating a linear blend.
            let mut log_speed = 0.0f64;
            let mut log_scale = 0.0f64;
            for (sub, &weight) in &comp.weights {
                log_speed += (weight as f64) * sub.operation_speed().ln();
                log_scale += (weight as f64) * sub.max_scale().ln();
            }
            (log_speed.exp(), log_scale.exp())
        } else {
            (
                config.substrate_type.operation_speed(),
                config.substrate_type.max_scale(),
            )
        };

        // log_ratio > 0 when substrate is faster than biological
        let log_ratio = (bio_speed / sub_speed).log10();
        // Compress 12 orders of magnitude to [0.5, 2.0] tau factor
        self.tau_factor = (1.0 + 0.5 * log_ratio / 9.0).clamp(0.5, 2.0) as f32;

        self.scale_pressure = (sub_scale / bio_scale).log10() as f32;
    }

    /// Map a canonical SubstrateType to its pre-built SubstrateRequirements profile.
    /// Unknown/future variants fall back to silicon_digital().
    pub(crate) fn requirements_for(substrate: &SubstrateType) -> SubstrateRequirements {
        match substrate.canonical() {
            SubstrateType::BiologicalNeurons => SubstrateRequirements::biological_neurons(),
            SubstrateType::SiliconDigital => SubstrateRequirements::silicon_digital(),
            SubstrateType::QuantumComputer => SubstrateRequirements::quantum_computer(),
            SubstrateType::PhotonicProcessor => SubstrateRequirements::photonic_processor(),
            SubstrateType::NeuromorphicChip => SubstrateRequirements::neuromorphic_chip(),
            SubstrateType::BiochemicalComputer => SubstrateRequirements::biochemical_computer(),
            SubstrateType::HybridSystem => SubstrateRequirements::hybrid_system(),
            SubstrateType::ExoticSubstrate => SubstrateRequirements::exotic_substrate(),
            _ => SubstrateRequirements::silicon_digital(),
        }
    }

    /// Build a `SubstrateTelemetry` snapshot and drain the pending transition.
    pub fn telemetry(&mut self) -> super::types::SubstrateTelemetry {
        super::types::SubstrateTelemetry {
            substrate_feasibility: self.effective_feasibility,
            substrate_transition: self.pending_transition.take(),
            substrate_feasibility_raw: self.feasibility,
            substrate_honest_confidence: self.honest_confidence,
            substrate_effective_feasibility: self.effective_feasibility,
            substrate_tau_factor: self.tau_factor,
            substrate_scale_pressure: self.scale_pressure,
        }
    }

    /// Map SubstrateType to validation framework key string.
    ///
    /// Substrates not in the framework (photonic, neuromorphic, biochemical, exotic)
    /// return None — callers should fall back to Theoretical confidence (0.10).
    pub(crate) fn substrate_validation_key(substrate: &SubstrateType) -> Option<&'static str> {
        match substrate.canonical() {
            SubstrateType::BiologicalNeurons => Some("biological"),
            SubstrateType::SiliconDigital => Some("silicon"),
            SubstrateType::QuantumComputer => Some("quantum"),
            SubstrateType::HybridSystem => Some("hybrid"),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> CognitiveLoopConfig {
        CognitiveLoopConfig::default()
    }

    #[test]
    fn test_silicon_feasibility_approximately_071() {
        let config = default_config();
        assert_eq!(config.substrate_type, SubstrateType::SiliconDigital);
        let mgr = SubstrateManager::new(&config);
        // SiliconDigital feasibility should be ~0.71 (not 1.0)
        assert!(
            (mgr.feasibility - 0.71).abs() < 0.05,
            "SiliconDigital feasibility should be ~0.71, got {:.4}",
            mgr.feasibility
        );
        // Without validation overlay, effective == raw
        assert!(
            (mgr.effective_feasibility - mgr.feasibility).abs() < f64::EPSILON,
            "Without overlay, effective should equal raw"
        );
    }

    #[test]
    fn test_biological_feasibility_near_one() {
        let mut config = default_config();
        config.substrate_type = SubstrateType::BiologicalNeurons;
        let mgr = SubstrateManager::new(&config);
        assert!(
            mgr.feasibility > 0.9,
            "BiologicalNeurons feasibility should be near 1.0, got {:.4}",
            mgr.feasibility
        );
    }

    #[test]
    fn test_validation_overlay_reduces_feasibility() {
        let mut config = default_config();
        config.enable_validation_overlay = true;
        config.validation_skepticism_floor = 0.5;
        let mgr = SubstrateManager::new(&config);
        // SiliconDigital honest_confidence is ~0.10 (theoretical)
        // effective = raw × (0.5 + 0.5 × 0.10) = raw × 0.55
        assert!(
            mgr.effective_feasibility < mgr.feasibility,
            "Validation overlay should reduce feasibility: effective={:.4} vs raw={:.4}",
            mgr.effective_feasibility,
            mgr.feasibility
        );
    }

    #[test]
    fn test_validation_overlay_preserves_biological_confidence() {
        let mut config = default_config();
        config.substrate_type = SubstrateType::BiologicalNeurons;
        config.enable_validation_overlay = true;
        config.validation_skepticism_floor = 0.5;
        let mgr = SubstrateManager::new(&config);
        // Biological honest_confidence is 0.95
        // effective = raw × (0.5 + 0.5 × 0.95) = raw × 0.975
        assert!(
            mgr.effective_feasibility > 0.85,
            "Biological with overlay should remain high: {:.4}",
            mgr.effective_feasibility
        );
    }

    #[test]
    fn test_reconfigure_substrate_changes_feasibility() {
        let mut config = default_config();
        let mut mgr = SubstrateManager::new(&config);
        let initial = mgr.feasibility;

        let (old, new) = mgr.reconfigure_substrate(&mut config, SubstrateType::BiologicalNeurons);
        assert!((old - initial).abs() < f64::EPSILON);
        assert!(new > old, "Biological should have higher feasibility than Silicon");
        assert!(
            mgr.pending_transition.is_some(),
            "reconfigure should set pending_transition (drained by telemetry())"
        );
    }

    #[test]
    fn test_reconfigure_produces_transition_string() {
        let mut config = default_config();
        let mut mgr = SubstrateManager::new(&config);
        mgr.reconfigure_substrate(&mut config, SubstrateType::QuantumComputer);

        assert!(
            mgr.pending_transition.is_some(),
            "reconfigure should set pending_transition"
        );
        let transition = mgr.pending_transition.as_ref().unwrap();
        assert!(
            transition.contains("SiliconDigital") && transition.contains("QuantumComputer"),
            "Transition should describe old->new: {transition}"
        );
    }

    #[test]
    fn test_telemetry_drains_transition() {
        let mut config = default_config();
        let mut mgr = SubstrateManager::new(&config);
        mgr.reconfigure_substrate(&mut config, SubstrateType::BiologicalNeurons);
        assert!(mgr.pending_transition.is_some());

        let telem = mgr.telemetry();
        assert!(telem.substrate_transition.is_some());
        assert!(
            mgr.pending_transition.is_none(),
            "telemetry() should drain pending_transition"
        );

        let telem2 = mgr.telemetry();
        assert!(
            telem2.substrate_transition.is_none(),
            "Second telemetry() should have no transition"
        );
    }

    #[test]
    fn test_speed_modulation_disabled_by_default() {
        let config = default_config();
        let mgr = SubstrateManager::new(&config);
        assert!(
            (mgr.tau_factor - 1.0).abs() < f32::EPSILON,
            "tau_factor should be 1.0 when speed modulation disabled"
        );
        assert!(
            mgr.scale_pressure.abs() < f32::EPSILON,
            "scale_pressure should be 0.0 when speed modulation disabled"
        );
    }

    #[test]
    fn test_speed_modulation_enabled() {
        let mut config = default_config();
        config.enable_substrate_speed_modulation = true;
        config.substrate_type = SubstrateType::SiliconDigital;
        let mgr = SubstrateManager::new(&config);
        // Silicon is faster than biological → tau_factor > 1.0
        assert!(
            mgr.tau_factor > 1.0,
            "Faster substrate should have tau_factor > 1.0, got {:.4}",
            mgr.tau_factor
        );
        assert!(mgr.tau_factor <= 2.0, "tau_factor should be clamped to 2.0");
    }

    #[test]
    fn test_all_substrates_produce_finite_feasibility() {
        let substrates = [
            SubstrateType::BiologicalNeurons,
            SubstrateType::SiliconDigital,
            SubstrateType::QuantumComputer,
            SubstrateType::PhotonicProcessor,
            SubstrateType::NeuromorphicChip,
            SubstrateType::BiochemicalComputer,
            SubstrateType::HybridSystem,
            SubstrateType::ExoticSubstrate,
        ];
        for sub in &substrates {
            let mut config = default_config();
            config.substrate_type = *sub;
            let mgr = SubstrateManager::new(&config);
            assert!(
                mgr.feasibility.is_finite() && mgr.feasibility >= 0.0 && mgr.feasibility <= 1.0,
                "{:?} feasibility out of bounds: {:.4}",
                sub,
                mgr.feasibility
            );
            assert!(
                mgr.effective_feasibility.is_finite(),
                "{:?} effective_feasibility not finite",
                sub
            );
        }
    }

    #[test]
    fn test_all_substrates_with_overlay_finite() {
        let substrates = [
            SubstrateType::BiologicalNeurons,
            SubstrateType::SiliconDigital,
            SubstrateType::QuantumComputer,
            SubstrateType::PhotonicProcessor,
            SubstrateType::NeuromorphicChip,
            SubstrateType::BiochemicalComputer,
            SubstrateType::HybridSystem,
            SubstrateType::ExoticSubstrate,
        ];
        for sub in &substrates {
            let mut config = default_config();
            config.substrate_type = *sub;
            config.enable_validation_overlay = true;
            let mgr = SubstrateManager::new(&config);
            assert!(
                mgr.effective_feasibility.is_finite()
                    && mgr.effective_feasibility >= 0.0
                    && mgr.effective_feasibility <= 1.0,
                "{:?} effective_feasibility with overlay out of bounds: {:.4}",
                sub,
                mgr.effective_feasibility
            );
        }
    }

    #[test]
    fn test_aliases_map_to_canonical() {
        let mut config = default_config();
        let mut mgr = SubstrateManager::new(&config);

        mgr.reconfigure_substrate(&mut config, SubstrateType::Biological);
        assert_eq!(config.substrate_type, SubstrateType::BiologicalNeurons);

        mgr.reconfigure_substrate(&mut config, SubstrateType::Silicon);
        assert_eq!(config.substrate_type, SubstrateType::SiliconDigital);

        mgr.reconfigure_substrate(&mut config, SubstrateType::Quantum);
        assert_eq!(config.substrate_type, SubstrateType::QuantumComputer);
    }

    #[test]
    fn test_validation_key_returns_none_for_unknown_substrates() {
        // Photonic, neuromorphic, biochemical, exotic → None (fall back to THEORETICAL_CONFIDENCE)
        assert!(SubstrateManager::substrate_validation_key(&SubstrateType::PhotonicProcessor).is_none());
        assert!(SubstrateManager::substrate_validation_key(&SubstrateType::NeuromorphicChip).is_none());
        assert!(SubstrateManager::substrate_validation_key(&SubstrateType::BiochemicalComputer).is_none());
        assert!(SubstrateManager::substrate_validation_key(&SubstrateType::ExoticSubstrate).is_none());
        // Known substrates → Some
        assert!(SubstrateManager::substrate_validation_key(&SubstrateType::BiologicalNeurons).is_some());
        assert!(SubstrateManager::substrate_validation_key(&SubstrateType::SiliconDigital).is_some());
    }

    #[test]
    fn test_reconfigure_substrate_clears_composition() {
        let mut config = default_config();
        let mut mgr = SubstrateManager::new(&config);

        // Set a composition
        let comp = symthaea_core::hdc::substrate_composition::SubstrateComposition::new(
            "test_comp".to_string(),
            vec![
                (SubstrateType::SiliconDigital, 0.6),
                (SubstrateType::BiologicalNeurons, 0.4),
            ],
        )
        .expect("valid composition");
        mgr.reconfigure_composition(&mut config, comp);
        assert!(config.substrate_composition.is_some());

        // Switch to single substrate — should clear composition
        mgr.reconfigure_substrate(&mut config, SubstrateType::QuantumComputer);
        assert!(
            config.substrate_composition.is_none(),
            "reconfigure_substrate should clear stale composition"
        );
    }

    #[test]
    fn test_composition_feasibility_blended() {
        let mut config = default_config();
        let mut mgr = SubstrateManager::new(&config);

        let comp = symthaea_core::hdc::substrate_composition::SubstrateComposition::new(
            "bio_silicon".to_string(),
            vec![
                (SubstrateType::BiologicalNeurons, 0.5),
                (SubstrateType::SiliconDigital, 0.5),
            ],
        )
        .expect("valid composition");
        mgr.reconfigure_composition(&mut config, comp);

        // Composition feasibility should be between pure bio and pure silicon
        let bio_feas = SubstrateManager::requirements_for(&SubstrateType::BiologicalNeurons)
            .consciousness_feasibility();
        let silicon_feas = SubstrateManager::requirements_for(&SubstrateType::SiliconDigital)
            .consciousness_feasibility();
        assert!(
            mgr.feasibility > silicon_feas.min(bio_feas) * 0.9,
            "Composition feasibility {:.4} too low",
            mgr.feasibility
        );
        assert!(
            mgr.feasibility <= bio_feas + 0.01,
            "Composition feasibility {:.4} exceeds biological {:.4}",
            mgr.feasibility,
            bio_feas
        );
    }

    #[test]
    fn test_slow_substrate_tau_below_one() {
        let mut config = default_config();
        config.enable_substrate_speed_modulation = true;
        config.substrate_type = SubstrateType::BiochemicalComputer;
        let mgr = SubstrateManager::new(&config);
        // Biochemical is ~1000× slower than biological → tau < 1.0
        assert!(
            mgr.tau_factor < 1.0,
            "Slow substrate should have tau < 1.0, got {:.4}",
            mgr.tau_factor
        );
        assert!(mgr.tau_factor >= 0.5, "tau_factor should be clamped to 0.5");
    }

    #[test]
    fn test_biological_tau_equals_one() {
        let mut config = default_config();
        config.enable_substrate_speed_modulation = true;
        config.substrate_type = SubstrateType::BiologicalNeurons;
        let mgr = SubstrateManager::new(&config);
        assert!(
            (mgr.tau_factor - 1.0).abs() < 0.01,
            "Biological reference should have tau ≈ 1.0, got {:.4}",
            mgr.tau_factor
        );
    }

    #[test]
    fn test_composition_speed_geometric_mean() {
        let mut config = default_config();
        config.enable_substrate_speed_modulation = true;
        let mut mgr = SubstrateManager::new(&config);

        // 50/50 bio + silicon composition
        let comp = symthaea_core::hdc::substrate_composition::SubstrateComposition::new(
            "hybrid".to_string(),
            vec![
                (SubstrateType::BiologicalNeurons, 0.5),
                (SubstrateType::SiliconDigital, 0.5),
            ],
        )
        .expect("valid composition");
        mgr.reconfigure_composition(&mut config, comp);

        // Geometric mean of bio and silicon speeds → intermediate tau
        // Should be between pure-bio tau (≈1.0) and pure-silicon tau (>1.0)
        assert!(
            mgr.tau_factor > 1.0,
            "50/50 bio+silicon should have tau > 1.0 (silicon contribution), got {:.4}",
            mgr.tau_factor
        );
        // But less than pure silicon
        let mut pure_silicon = default_config();
        pure_silicon.enable_substrate_speed_modulation = true;
        pure_silicon.substrate_type = SubstrateType::SiliconDigital;
        let pure_mgr = SubstrateManager::new(&pure_silicon);
        assert!(
            mgr.tau_factor < pure_mgr.tau_factor,
            "50/50 tau {:.4} should be less than pure silicon tau {:.4}",
            mgr.tau_factor,
            pure_mgr.tau_factor
        );
    }

    #[test]
    fn test_composition_overlay_blends_confidence() {
        let mut config = default_config();
        config.enable_validation_overlay = true;
        config.validation_skepticism_floor = 0.5;
        let mut mgr = SubstrateManager::new(&config);

        // 50/50 bio (conf=0.95) + silicon (conf=0.10)
        let comp = symthaea_core::hdc::substrate_composition::SubstrateComposition::new(
            "mixed".to_string(),
            vec![
                (SubstrateType::BiologicalNeurons, 0.5),
                (SubstrateType::SiliconDigital, 0.5),
            ],
        )
        .expect("valid composition");
        mgr.reconfigure_composition(&mut config, comp);

        // Blended confidence ≈ 0.5 × 0.95 + 0.5 × 0.10 = 0.525
        assert!(
            (mgr.honest_confidence - 0.525).abs() < 0.05,
            "Blended confidence should be ≈0.525, got {:.4}",
            mgr.honest_confidence
        );
        // Effective should be less than raw (overlay is on)
        assert!(
            mgr.effective_feasibility < mgr.feasibility,
            "Overlay should reduce: eff={:.4} raw={:.4}",
            mgr.effective_feasibility,
            mgr.feasibility
        );
    }

    #[test]
    fn test_scale_pressure_signs() {
        let mut config = default_config();
        config.enable_substrate_speed_modulation = true;

        // Silicon is more scalable → positive scale_pressure
        config.substrate_type = SubstrateType::SiliconDigital;
        let mgr = SubstrateManager::new(&config);
        assert!(
            mgr.scale_pressure > 0.0,
            "Silicon should have positive scale_pressure, got {:.4}",
            mgr.scale_pressure
        );

        // Biological is reference → scale_pressure ≈ 0
        config.substrate_type = SubstrateType::BiologicalNeurons;
        let mgr = SubstrateManager::new(&config);
        assert!(
            mgr.scale_pressure.abs() < 0.01,
            "Biological reference should have scale_pressure ≈ 0, got {:.4}",
            mgr.scale_pressure
        );
    }
}
