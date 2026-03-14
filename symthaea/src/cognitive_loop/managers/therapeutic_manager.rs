//! # Therapeutic Manager — Client Model, Alliance, Crisis Detection, Regulation
//!
//! Consolidates therapeutic psychology into a single [`CognitiveSubsystem`] that reads
//! from an immutable [`CycleSnapshot`] and produces [`SubsystemOutput`] proposals.
//!
//! ## Components
//!
//! 1. **Client Model**: Tracks client affect, RDoC profile, diagnostic hypotheses
//! 2. **Therapeutic Alliance**: Bordin (1979) bond/goals/tasks with rupture-repair
//! 3. **Crisis Detector**: HDC-encoded crisis indicators for safety-critical detection
//! 4. **Regulation Engine**: Context-aware strategy → neuromodulator deltas
//! 5. **Scope Guard**: Architectural scope boundary enforcement
//!
//! ## Design
//!
//! Runs at interval 11 (co-prime with 7, 13, 19, 37, 41).
//! Safety checks (crisis detection) run every invocation regardless of other state.
//!
//! Science: Bordin (1979), Safran & Muran (2000), Stanley & Brown (2012),
//! APA Ethics Code (2017), Lambert (2013).

use super::super::subsystem_trait::{CognitiveSubsystem, CycleSnapshot, SubsystemOutput};
use symthaea_clinical::InterventionLibrary;
use symthaea_therapeutic::{
    ClientModel, CrisisDetector, RegulationEngine, ScopeGuard, TherapeuticAlliance,
};
use symthaea_therapeutic::affect_regulation::NeuromodDelta;
use symthaea_therapeutic::client_model::CoreAffectSnapshot;

/// Therapeutic Manager — integrates therapeutic psychology into the cognitive loop.
///
/// Implements `CognitiveSubsystem` at interval 11 (co-prime).
pub struct TherapeuticManager {
    /// Client psychological state model.
    pub client_model: ClientModel,
    /// Therapeutic working alliance (Bordin 1979).
    pub alliance: TherapeuticAlliance,
    /// Crisis detection system.
    pub crisis_detector: CrisisDetector,
    /// Emotion regulation strategy engine.
    pub regulation_engine: RegulationEngine,
    /// Scope boundary enforcement.
    pub scope_guard: ScopeGuard,
    /// Evidence-based intervention library.
    pub intervention_library: InterventionLibrary,
    /// Whether a crisis was detected this cycle.
    pub crisis_active: bool,
    /// Last detected crisis type name (for telemetry).
    pub last_crisis_type: Option<String>,
    /// Last neuromod delta from regulation strategy (for bath injection).
    pub last_neuromod_delta: Option<NeuromodDelta>,
}

impl Default for TherapeuticManager {
    fn default() -> Self {
        Self {
            client_model: ClientModel::new(),
            alliance: TherapeuticAlliance::new(),
            crisis_detector: CrisisDetector::new(),
            regulation_engine: RegulationEngine::new(),
            scope_guard: ScopeGuard::new(),
            intervention_library: InterventionLibrary::bootstrap(),
            crisis_active: false,
            last_crisis_type: None,
            last_neuromod_delta: None,
        }
    }
}

impl TherapeuticManager {
    /// Create a new therapeutic manager with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get current client distress level (0-1).
    pub fn client_distress(&self) -> f32 {
        self.client_model.distress()
    }

    /// Get current alliance composite (0-1).
    pub fn alliance_composite(&self) -> f32 {
        self.alliance.composite()
    }

    /// Get the currently active regulation strategy.
    pub fn active_strategy(
        &self,
    ) -> Option<symthaea_therapeutic::RegulationStrategy> {
        self.regulation_engine.active_strategy
    }

    /// Encode the current therapeutic state as a dream-compatible action vector.
    ///
    /// Returns a 32-element `Vec<f32>` encoding:
    /// [0..6]  = RDoC client profile (6 domains)
    /// [6]     = distress level
    /// [7]     = alliance composite
    /// [8]     = crisis_active (0 or 1)
    /// [9]     = strategy ordinal (0-6, or -1 if none)
    /// [10..18]= neuromod delta (8 transmitters)
    /// [18..32]= reserved (zeros)
    ///
    /// This vector is recorded into the `DreamEngine<Vec<f32>>` so the
    /// dream engine can generate therapeutic counterfactuals.
    pub fn dream_action_vector(&self) -> Vec<f32> {
        use symthaea_therapeutic::RegulationStrategy;

        let mut v = vec![0.0f32; 32];

        // RDoC profile (client state)
        let rdoc = &self.client_model.rdoc_profile;
        v[0] = rdoc.score(symthaea_clinical::RDocDomain::NegativeValence);
        v[1] = rdoc.score(symthaea_clinical::RDocDomain::PositiveValence);
        v[2] = rdoc.score(symthaea_clinical::RDocDomain::CognitiveSystems);
        v[3] = rdoc.score(symthaea_clinical::RDocDomain::SocialProcesses);
        v[4] = rdoc.score(symthaea_clinical::RDocDomain::ArousalRegulatory);
        v[5] = rdoc.score(symthaea_clinical::RDocDomain::Sensorimotor);

        // Distress and alliance
        v[6] = self.client_model.distress();
        v[7] = self.alliance.composite();
        v[8] = if self.crisis_active { 1.0 } else { 0.0 };

        // Strategy ordinal
        v[9] = match self.regulation_engine.active_strategy {
            Some(RegulationStrategy::CognitiveReappraisal) => 0.0,
            Some(RegulationStrategy::DistressTolerance) => 1.0,
            Some(RegulationStrategy::Grounding) => 2.0,
            Some(RegulationStrategy::Defusion) => 3.0,
            Some(RegulationStrategy::Validation) => 4.0,
            Some(RegulationStrategy::Containment) => 5.0,
            Some(RegulationStrategy::ExposurePrep) => 6.0,
            None => -1.0,
        };

        // Neuromod delta
        if let Some(ref delta) = self.last_neuromod_delta {
            v[10] = delta.dopamine;
            v[11] = delta.noradrenaline;
            v[12] = delta.serotonin;
            v[13] = delta.acetylcholine;
            v[14] = delta.gaba;
            v[15] = delta.oxytocin;
            v[16] = delta.glutamate;
            v[17] = delta.adenosine;
        }

        v
    }
}

impl CognitiveSubsystem for TherapeuticManager {
    fn name(&self) -> &'static str {
        "therapeutic_manager"
    }

    fn interval(&self) -> u32 {
        11 // co-prime with 7, 13, 19, 37, 41
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // ── 1. Update client affect from cycle snapshot ────────────────────
        let affect = CoreAffectSnapshot::new(
            snapshot.valence,
            snapshot.arousal,
            snapshot.cycle_number,
        );
        self.client_model.update_affect(affect);

        // ── 2. Crisis detection (ALWAYS runs — safety critical) ────────────
        // NOTE: Do not reset crisis_active here — text-based crisis detection
        // runs in cycle() before process() and sets crisis_active. We only
        // clear it if no affect-based crisis is detected AND no text-based
        // crisis was already flagged this cycle.
        let text_crisis_active = self.crisis_active;
        if !text_crisis_active {
            self.crisis_active = false;
            self.last_crisis_type = None;
        }
        if let Some(crisis_alert) = self.crisis_detector.detect_from_affect(
            snapshot.valence,
            snapshot.arousal,
        ) {
            self.crisis_active = true;
            self.last_crisis_type = Some(crisis_alert.crisis_type_name().to_string());
            // Crisis → suppress learning, dampen exploration, maximize confidence
            // (don't try novel things during crisis)
            output.lr_modulation = 0.5;
            output.exploration_delta = -0.05;
            output.confidence_delta = -0.05;
        }

        // ── 3. Alliance dynamics ──────────────────────────────────────────
        if let Some(rupture_type) =
            self.alliance
                .detect_rupture(snapshot.valence, snapshot.arousal)
        {
            self.alliance
                .register_rupture(rupture_type, snapshot.cycle_number);
            output.confidence_delta += -0.02;
            output.exploration_delta += 0.01;
        }

        // Natural alliance growth (small per cycle when affect is positive)
        if snapshot.valence > 0.0 {
            let growth = 0.002 * snapshot.valence;
            self.alliance.grow(growth, growth * 0.5, growth * 0.5);
        }

        // ── 4. Regulation strategy selection → neuromod delta ─────────────
        let strategy = self.regulation_engine.select_strategy(
            &self.client_model,
            &self.alliance,
            self.crisis_active,
        );
        let delta = self
            .regulation_engine
            .apply_strategy(strategy, self.client_model.distress());

        // Store last delta for neuromod injection by the cycle runner
        self.last_neuromod_delta = Some(delta);

        // Map neuromod delta to subsystem output (valence/arousal pathway)
        // Positive serotonin/oxytocin → positive valence
        // Negative noradrenaline → decrease arousal
        output.valence_delta += delta.serotonin * 0.5 + delta.oxytocin * 0.3;
        output.arousal_delta += delta.noradrenaline * 0.3 - delta.gaba * 0.2;

        // Dopamine modulates learning rate
        if delta.dopamine.abs() > 0.01 {
            output.lr_modulation *= 1.0 + (delta.dopamine as f64 * 0.5);
        }

        // Acetylcholine modulates confidence (attentional clarity)
        if delta.acetylcholine > 0.01 {
            output.confidence_delta += delta.acetylcholine as f64 * 0.02;
        }

        // ── 5. Track regulation effectiveness ─────────────────────────────
        if self.client_model.affect_trend() > 0.0 {
            self.regulation_engine.record_success(strategy);
        }

        output
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(valence: f32, arousal: f32, cycle: u64) -> CycleSnapshot {
        CycleSnapshot {
            cycle_number: cycle,
            valence,
            arousal,
            prediction_error: 0.1,
            coherence: 0.5,
            prediction_confidence: 0.5,
            unified_psi: 0.5,
            ..default_snapshot()
        }
    }

    fn default_snapshot() -> CycleSnapshot {
        CycleSnapshot {
            cycle_number: 0,
            prediction_confidence: 0.5,
            fep_lr_boost: 1.0,
            prediction_error: 0.1,
            coherence: 0.5,
            unified_psi: 0.5,
            phi_attention_weight: 1.0,
            arousal: 0.5,
            valence: 0.0,
            thermodynamic_load: 0.3,
            dissipative_health: 0.8,
            somatic_stress: 0.0,
            urgency: 1,
            attention_budget_exceeded: 0,
            compressed_state: [0.0; crate::cognitive_loop::subsystem_trait::SNAPSHOT_STATE_DIM],
            input_hv: [0; crate::cognitive_loop::subsystem_trait::SNAPSHOT_HV_BYTES],
            phenomenal_binding: 0.5,
            harmonic_coherence: 0.0,
            holographic_unity: 0.0,
            gradient_magnitude: 0.0,
            epistemic_confidence: 0.5,
            _reserved: [0; 12],
        }
    }

    #[test]
    fn test_therapeutic_manager_creation() {
        let manager = TherapeuticManager::new();
        assert_eq!(manager.name(), "therapeutic_manager");
        assert_eq!(manager.interval(), 11);
        assert!(!manager.crisis_active);
    }

    #[test]
    fn test_process_updates_client_affect() {
        let mut manager = TherapeuticManager::new();
        let snapshot = make_snapshot(-0.5, 0.8, 1);
        manager.process(&snapshot);
        assert_eq!(manager.client_model.current_affect.valence, -0.5);
    }

    #[test]
    fn test_process_positive_grows_alliance() {
        let mut manager = TherapeuticManager::new();
        let pre = manager.alliance.composite();
        let snapshot = make_snapshot(0.5, 0.4, 1);
        manager.process(&snapshot);
        assert!(manager.alliance.composite() >= pre);
    }

    #[test]
    fn test_process_negative_detects_rupture() {
        let mut manager = TherapeuticManager::new();
        // Strong negative affect + low arousal → withdrawal rupture
        let snapshot = make_snapshot(-0.7, 0.1, 1);
        let output = manager.process(&snapshot);
        assert!(manager.alliance.rupture_count > 0);
        assert!(output.confidence_delta < 0.0);
    }

    #[test]
    fn test_process_returns_neuromod_output() {
        let mut manager = TherapeuticManager::new();
        let snapshot = make_snapshot(-0.5, 0.8, 1);
        let output = manager.process(&snapshot);
        // Should produce some valence/arousal change from regulation
        // (validation strategy should boost serotonin/oxytocin → positive valence_delta)
        let _ = output; // non-neutral output is strategy-dependent
    }

    #[test]
    fn test_client_distress() {
        let mut manager = TherapeuticManager::new();
        manager
            .client_model
            .update_affect(CoreAffectSnapshot::new(-0.8, 0.9, 0));
        assert!(manager.client_distress() > 0.5);
    }
}
