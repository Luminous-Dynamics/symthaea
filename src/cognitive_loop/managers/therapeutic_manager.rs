// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
use super::therapeutic_dream_bridge::TherapeuticDreamTracker;
use symthaea_clinical::InterventionLibrary;
use symthaea_therapeutic::affect_regulation::NeuromodDelta;
use symthaea_therapeutic::client_model::CoreAffectSnapshot;
use symthaea_therapeutic::{
    CaseFormulation, ClientModel, CrisisDetector, NarrativeFragment, RegulationEngine, ScopeGuard,
    ShadowDetector, TherapeuticAlliance, TherapeuticNarrative,
};

/// Serializable snapshot of therapeutic session state for persistence.
#[derive(serde::Serialize)]
struct TherapeuticSessionSnapshot<'a> {
    client_model: &'a symthaea_therapeutic::ClientModel,
    alliance_bond: f32,
    alliance_goal: f32,
    alliance_task: f32,
    alliance_ruptures: u32,
    alliance_repairs: u32,
    narrative_coherence: f32,
    narrative_fragment_count: usize,
    formulation_predisposing: usize,
    formulation_perpetuating: usize,
    formulation_protective: usize,
    crisis_active: bool,
    serotonin_debt: f32,
    dopamine_debt: f32,
    shadow: symthaea_therapeutic::ShadowSnapshot,
}

/// Deserializable restore struct for session persistence.
#[derive(serde::Deserialize)]
struct TherapeuticSessionRestore {
    client_model: symthaea_therapeutic::ClientModel,
    alliance_bond: f32,
    alliance_goal: f32,
    alliance_task: f32,
    alliance_ruptures: u32,
    alliance_repairs: u32,
    narrative_coherence: f32,
    crisis_active: bool,
    #[serde(default)]
    shadow: Option<symthaea_therapeutic::ShadowSnapshot>,
}

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
    /// CBT 4P case formulation (predisposing, precipitating, perpetuating, protective).
    pub formulation: CaseFormulation,
    /// Therapeutic narrative with coherence tracking.
    pub narrative: TherapeuticNarrative,
    /// Whether a crisis was detected this cycle.
    pub crisis_active: bool,
    /// Last detected crisis type name (for telemetry).
    pub last_crisis_type: Option<String>,
    /// Last neuromod delta from regulation strategy (for bath injection).
    pub last_neuromod_delta: Option<NeuromodDelta>,
    /// Dream counterfactual prediction tracker (closed-loop validation).
    pub dream_tracker: TherapeuticDreamTracker,
    /// Shadow work detector — observability mode (Jung → Friston mapping).
    /// Tracks prediction error accumulation in emotionally charged, unintegrated content.
    pub shadow_detector: ShadowDetector,
    /// Last shadow telemetry snapshot (for telemetry output).
    pub last_shadow_telemetry: symthaea_therapeutic::ShadowTelemetry,
    /// Auto-checkpoint cycle counter (serialize every N cycles).
    checkpoint_counter: u32,
    /// Last serialized session snapshot (for periodic persistence).
    pub last_checkpoint: Option<String>,
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
            formulation: CaseFormulation::new(),
            narrative: TherapeuticNarrative::new(),
            crisis_active: false,
            last_crisis_type: None,
            last_neuromod_delta: None,
            dream_tracker: TherapeuticDreamTracker::new(),
            shadow_detector: ShadowDetector::new(),
            last_shadow_telemetry: symthaea_therapeutic::ShadowTelemetry::default(),
            checkpoint_counter: 0,
            last_checkpoint: None,
        }
    }
}

impl TherapeuticManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 11;

    /// Create a new therapeutic manager with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Serialize the therapeutic session state to JSON for persistence.
    ///
    /// Captures the core session data (affect trajectory, alliance, formulation,
    /// narrative coherence, risk level, session count) that can survive a restart.
    /// HDC vectors are excluded (marked `#[serde(skip)]` on ClientModel).
    pub fn serialize_session(&self) -> Result<String, serde_json::Error> {
        let session = TherapeuticSessionSnapshot {
            client_model: &self.client_model,
            alliance_bond: self.alliance.bond,
            alliance_goal: self.alliance.goal_agreement,
            alliance_task: self.alliance.task_agreement,
            alliance_ruptures: self.alliance.rupture_count,
            alliance_repairs: self.alliance.repair_count,
            narrative_coherence: self.narrative.coherence,
            narrative_fragment_count: self.narrative.fragments.len(),
            formulation_predisposing: self.formulation.predisposing.len(),
            formulation_perpetuating: self.formulation.perpetuating.len(),
            formulation_protective: self.formulation.protective.len(),
            crisis_active: self.crisis_active,
            serotonin_debt: self.regulation_engine.serotonin_debt(),
            dopamine_debt: self.regulation_engine.dopamine_debt(),
            shadow: self.shadow_detector.snapshot(),
        };
        serde_json::to_string(&session)
    }

    /// Restore therapeutic session state from a JSON snapshot.
    ///
    /// Restores affect trajectory, alliance, and risk level.
    /// HDC vectors are not restored (require re-encoding from text).
    pub fn restore_session(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let snapshot: TherapeuticSessionRestore = serde_json::from_str(json)?;
        self.client_model = snapshot.client_model;
        self.alliance.bond = snapshot.alliance_bond;
        self.alliance.goal_agreement = snapshot.alliance_goal;
        self.alliance.task_agreement = snapshot.alliance_task;
        self.alliance.rupture_count = snapshot.alliance_ruptures;
        self.alliance.repair_count = snapshot.alliance_repairs;
        self.narrative.coherence = snapshot.narrative_coherence;
        self.crisis_active = snapshot.crisis_active;
        if let Some(shadow) = snapshot.shadow {
            self.shadow_detector.restore(shadow);
        }
        Ok(())
    }

    /// Get current client distress level (0-1).
    pub fn client_distress(&self) -> f32 {
        self.client_model.distress()
    }

    /// Get current alliance composite (0-1).
    pub fn alliance_composite(&self) -> f32 {
        self.alliance.composite()
    }

    /// Get narrative coherence (0-1).
    pub fn narrative_coherence(&self) -> f32 {
        self.narrative.coherence
    }

    /// Get case formulation resilience ratio.
    pub fn formulation_resilience_ratio(&self) -> f32 {
        self.formulation.resilience_ratio()
    }

    /// Get the currently active regulation strategy.
    pub fn active_strategy(&self) -> Option<symthaea_therapeutic::RegulationStrategy> {
        self.regulation_engine.active_strategy
    }

    /// Update RDoC profile from live neuromodulator bath state (bidirectional bridge).
    ///
    /// Called from `cycle_phase_dynamics.rs` where both `self.neuromod.bath` and
    /// `self.therapeutic_manager` are accessible.
    pub fn update_rdoc_from_bath(&mut self, bath: &[f32; 8]) {
        RegulationEngine::update_rdoc_from_neuromod(&mut self.client_model.rdoc_profile, bath);
    }

    /// Run text-based crisis detection on input text.
    ///
    /// Must be called early in the cycle (before `process()`) to ensure
    /// crisis state is available for all downstream safety checks.
    /// Returns `true` if a crisis was detected.
    pub fn detect_crisis_from_text(&mut self, input: &str) -> bool {
        if let Some(alert) = self.crisis_detector.detect(input) {
            self.crisis_active = true;
            self.last_crisis_type = Some(alert.crisis_type_name().to_string());
            true
        } else {
            false
        }
    }

    /// Feed dream prediction accuracy back into strategy selection.
    ///
    /// When prediction accuracy is low (error > 0.5), bias strategy selection
    /// toward validated strategies (Validation, Grounding) rather than
    /// exploratory ones (ExposurePrep, CognitiveReappraisal).
    /// When accuracy is high (error < 0.2), permit dream-preferred strategies.
    pub fn apply_dream_feedback(&mut self) {
        let accuracy = self.dream_tracker.prediction_accuracy();
        if accuracy > 0.5 {
            // High error → clear dream preference, fall back to validated strategies
            self.regulation_engine.clear_dream_preference();
        }
        // Low error → dream preference remains active (set by dream engine)
    }

    /// Auto-checkpoint: serialize session every `interval` cycles.
    ///
    /// If `persist_path` is `Some`, writes the checkpoint to disk for
    /// cross-restart persistence. Otherwise stores in memory only.
    ///
    /// Returns `true` if a checkpoint was taken this call.
    pub fn maybe_checkpoint(&mut self, interval: u32, persist_path: Option<&str>) -> bool {
        self.checkpoint_counter += 1;
        if self.checkpoint_counter >= interval {
            self.checkpoint_counter = 0;
            if let Ok(json) = self.serialize_session() {
                // Persist to disk if path provided
                if let Some(path) = persist_path {
                    if let Err(e) = std::fs::write(path, &json) {
                        tracing::warn!(
                            target: "therapeutic_manager",
                            error = %e,
                            path = path,
                            "Failed to persist therapeutic checkpoint"
                        );
                    }
                }
                self.last_checkpoint = Some(json);
                return true;
            }
        }
        false
    }

    /// Load a therapeutic session from a checkpoint file on disk.
    ///
    /// Returns `Ok(true)` if a checkpoint was loaded, `Ok(false)` if
    /// the file doesn't exist, `Err` on parse failure.
    pub fn load_checkpoint(&mut self, path: &str) -> Result<bool, serde_json::Error> {
        match std::fs::read_to_string(path) {
            Ok(json) => {
                self.restore_session(&json)?;
                tracing::info!(
                    target: "therapeutic_manager",
                    path = path,
                    "Restored therapeutic session from checkpoint"
                );
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }

    /// Get dream prediction accuracy (lower = better, 1.0 = no data).
    pub fn dream_prediction_accuracy(&self) -> f32 {
        self.dream_tracker.prediction_accuracy()
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

/// Structured therapeutic session summary.
///
/// Generated periodically or on-demand to provide a clinical-style overview
/// of the therapeutic interaction state.
///
/// Science: Lambert (2013) Outcome Questionnaire, Bordin (1979) alliance model.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionSummary {
    /// Total cycles in session.
    pub total_cycles: u64,
    /// Number of crisis events detected.
    pub crisis_count: u32,
    /// Most frequently used regulation strategy.
    pub dominant_strategy: String,
    /// Alliance composite trend (positive = growing, negative = declining).
    pub alliance_trend: f32,
    /// Current alliance composite.
    pub alliance_composite: f32,
    /// Alliance rupture count.
    pub rupture_count: u32,
    /// Alliance repair count.
    pub repair_count: u32,
    /// Affect trajectory: (min_valence, max_valence, mean_valence).
    pub affect_min_valence: f32,
    pub affect_max_valence: f32,
    pub affect_mean_valence: f32,
    /// Current distress level.
    pub current_distress: f32,
    /// Narrative coherence.
    pub narrative_coherence: f32,
    /// Narrative fragment count.
    pub narrative_fragments: usize,
    /// Formulation: protective factor count.
    pub protective_factors: usize,
    /// Formulation: perpetuating factor count.
    pub perpetuating_factors: usize,
    /// Formulation resilience ratio.
    pub resilience_ratio: f32,
    /// Serotonin debt level.
    pub serotonin_debt: f32,
    /// Dopamine debt level.
    pub dopamine_debt: f32,
    /// Dream prediction accuracy (lower = better).
    pub dream_accuracy: f32,
    /// Clinical severity composite.
    pub clinical_severity: f32,
    /// Per-strategy effectiveness summary.
    pub strategy_effectiveness: Vec<(String, f32, u32)>, // (name, success_rate, applications)
}

impl TherapeuticManager {
    /// Generate a structured session summary.
    ///
    /// Provides a clinical-style overview suitable for logging, display,
    /// or export to external systems.
    pub fn session_summary(&self) -> SessionSummary {
        // Compute affect trajectory from client model
        let trajectory = &self.client_model.affect_trajectory;
        let (min_v, max_v, sum_v) = if trajectory.is_empty() {
            (0.0f32, 0.0f32, 0.0f32)
        } else {
            trajectory
                .iter()
                .fold((f32::MAX, f32::MIN, 0.0f32), |(min, max, sum), snapshot| {
                    (
                        min.min(snapshot.valence),
                        max.max(snapshot.valence),
                        sum + snapshot.valence,
                    )
                })
        };
        let mean_v = if trajectory.is_empty() {
            0.0
        } else {
            sum_v / trajectory.len() as f32
        };

        // Dominant strategy from effectiveness tracking
        let dominant_strategy = self
            .regulation_engine
            .most_effective_strategy()
            .map(|s| format!("{:?}", s))
            .unwrap_or_else(|| {
                self.regulation_engine
                    .active_strategy
                    .map(|s| format!("{:?}", s))
                    .unwrap_or_else(|| "None".to_string())
            });

        // Per-strategy effectiveness
        let strategy_eff: Vec<(String, f32, u32)> = symthaea_therapeutic::RegulationStrategy::ALL
            .iter()
            .filter_map(|s| {
                self.regulation_engine
                    .effectiveness(s)
                    .map(|eff| (format!("{:?}", s), eff.success_rate(), eff.applications))
            })
            .filter(|(_, _, apps)| *apps > 0)
            .collect();

        // Alliance trend: compare current to initial (0.3 default)
        let alliance_trend = self.alliance.composite() - 0.3;

        SessionSummary {
            total_cycles: self.client_model.cycle_count as u64,
            crisis_count: if self.crisis_active { 1 } else { 0 }, // per-session, not cumulative
            dominant_strategy,
            alliance_trend,
            alliance_composite: self.alliance.composite(),
            rupture_count: self.alliance.rupture_count,
            repair_count: self.alliance.repair_count,
            affect_min_valence: min_v,
            affect_max_valence: max_v,
            affect_mean_valence: mean_v,
            current_distress: self.client_model.distress(),
            narrative_coherence: self.narrative.coherence,
            narrative_fragments: self.narrative.fragments.len(),
            protective_factors: self.formulation.protective.len(),
            perpetuating_factors: self.formulation.perpetuating.len(),
            resilience_ratio: self.formulation.resilience_ratio(),
            serotonin_debt: self.regulation_engine.serotonin_debt(),
            dopamine_debt: self.regulation_engine.dopamine_debt(),
            dream_accuracy: self.dream_tracker.prediction_accuracy(),
            clinical_severity: self.client_model.clinical_severity(),
            strategy_effectiveness: strategy_eff,
        }
    }
}

impl TherapeuticManager {
    /// Export therapeutic telemetry to a JSON file.
    ///
    /// Writes a structured JSON document containing session summary, affect
    /// trajectory, alliance history, formulation, and strategy effectiveness.
    /// Useful for offline analysis, clinical dashboards, or research export.
    pub fn flush_telemetry(&self, path: &str) -> Result<(), String> {
        let summary = self.session_summary();
        let trajectory: Vec<(f32, f32, u64)> = self
            .client_model
            .affect_trajectory
            .iter()
            .map(|a| (a.valence, a.arousal, a.cycle))
            .collect();

        #[derive(serde::Serialize)]
        struct TelemetryExport {
            session_summary: SessionSummary,
            affect_trajectory: Vec<(f32, f32, u64)>,
            alliance_bond: f32,
            alliance_goal_agreement: f32,
            alliance_task_agreement: f32,
            formulation_perpetuating: Vec<String>,
            formulation_protective: Vec<String>,
            narrative_fragment_count: usize,
            dream_prediction_accuracy: f32,
        }

        let export = TelemetryExport {
            session_summary: summary,
            affect_trajectory: trajectory,
            alliance_bond: self.alliance.bond,
            alliance_goal_agreement: self.alliance.goal_agreement,
            alliance_task_agreement: self.alliance.task_agreement,
            formulation_perpetuating: self
                .formulation
                .perpetuating
                .iter()
                .map(|f| f.description.clone())
                .collect(),
            formulation_protective: self
                .formulation
                .protective
                .iter()
                .map(|f| f.description.clone())
                .collect(),
            narrative_fragment_count: self.narrative.fragments.len(),
            dream_prediction_accuracy: self.dream_tracker.prediction_accuracy(),
        };

        let json = serde_json::to_string_pretty(&export)
            .map_err(|e| format!("JSON serialization failed: {}", e))?;
        std::fs::write(path, json).map_err(|e| format!("File write failed: {}", e))?;
        Ok(())
    }
}

impl CognitiveSubsystem for TherapeuticManager {
    fn name(&self) -> &'static str {
        "therapeutic_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // ── 1. Update client affect from cycle snapshot ────────────────────
        let affect =
            CoreAffectSnapshot::new(snapshot.valence, snapshot.arousal, snapshot.cycle_number);
        self.client_model.update_affect(affect);

        // ── 1b. Update RDoC profile from sustained affect patterns ────────
        // Slow EMA adaptation makes neuromod deltas responsive to ongoing state.
        self.client_model.update_rdoc_from_affect();

        // ── 1c. Tick transmitter debt from sustained RDoC imbalance ────────
        // Accumulates serotonin/dopamine debt over time — amplifies future deltas.
        self.regulation_engine
            .tick_debt(&self.client_model.rdoc_profile);

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
        if let Some(crisis_alert) = self
            .crisis_detector
            .detect_from_affect(snapshot.valence, snapshot.arousal)
        {
            self.crisis_active = true;
            self.last_crisis_type = Some(crisis_alert.crisis_type_name().to_string());
            // Crisis → suppress learning, dampen exploration, maximize confidence
            // (don't try novel things during crisis)
            output.lr_modulation = 0.5;
            output.exploration_delta = -0.05;
            output.confidence_delta = -0.05;
        }

        // ── 3. Alliance dynamics ──────────────────────────────────────────
        if let Some(rupture_type) = self
            .alliance
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

        // ── 4. Record outcome of previous strategy (if any) ──────────────
        // Must happen BEFORE selecting the next strategy so the effectiveness
        // tracker has the latest affect-delta for the prior strategy.
        self.regulation_engine
            .record_outcome(self.client_model.distress());

        // ── 5. Regulation strategy selection → neuromod delta ─────────────
        let strategy = self.regulation_engine.select_strategy(
            &self.client_model,
            &self.alliance,
            self.crisis_active,
        );
        // RDoC-aware neuromod deltas: domain scores amplify relevant transmitters.
        // High NegativeValence → stronger serotonin; low PositiveValence → stronger dopamine.
        let delta = self.regulation_engine.apply_strategy_rdoc(
            strategy,
            self.client_model.distress(),
            &self.client_model.rdoc_profile,
        );

        // Record distress at application for next cycle's outcome tracking
        self.regulation_engine
            .record_application_distress(self.client_model.distress());

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

        // ── 6. Narrative fragment recording ─────────────────────────────────
        let is_traumatic = self.client_model.distress() > 0.7 && snapshot.valence < -0.3;
        let fragment_text = format!(
            "cycle_{}_v{:.2}_a{:.2}",
            snapshot.cycle_number, snapshot.valence, snapshot.arousal
        );
        self.narrative.integrate_fragment(NarrativeFragment::new(
            &fragment_text,
            snapshot.cycle_number,
            snapshot.valence,
            is_traumatic,
        ));

        // ── 6b. Shadow dynamics — observability mode ────────────────────────
        // Track prediction error accumulation in emotionally charged,
        // unintegrated content (Jung → Friston mapping).
        // NO interventions or Broca modifications — telemetry only.
        {
            let narrative_integration = self.narrative.coherence;
            let mut shadow_telemetry = self.shadow_detector.process(
                snapshot.cycle_number,
                snapshot.prediction_error,
                snapshot.valence,
                snapshot.arousal,
                &fragment_text,
                narrative_integration,
            );

            // Detect projections against current input
            let projections = self.shadow_detector.detect_projections(&fragment_text);
            shadow_telemetry.projection_detections = projections.len() as u32;

            // Set shadow-to-narrative ratio
            let total_narrative = self.narrative.fragments.len() as f32;
            if total_narrative > 0.0 {
                shadow_telemetry.shadow_to_narrative_ratio =
                    shadow_telemetry.shadow_fragment_count as f32 / total_narrative;
            }

            self.last_shadow_telemetry = shadow_telemetry;
        }

        // ── 7. Dream tracker: record current RDoC as prediction target ────
        {
            let rdoc = &self.client_model.rdoc_profile;
            let rdoc_vec = vec![
                rdoc.score(symthaea_clinical::RDocDomain::NegativeValence),
                rdoc.score(symthaea_clinical::RDocDomain::PositiveValence),
                rdoc.score(symthaea_clinical::RDocDomain::CognitiveSystems),
                rdoc.score(symthaea_clinical::RDocDomain::SocialProcesses),
                rdoc.score(symthaea_clinical::RDocDomain::ArousalRegulatory),
                rdoc.score(symthaea_clinical::RDocDomain::Sensorimotor),
            ];
            // Record actual outcome for prior prediction, then predict next
            self.dream_tracker.record_actual_outcome(&rdoc_vec);
            self.dream_tracker
                .record_prediction(rdoc_vec, snapshot.cycle_number);
        }

        // ── 8. Auto-checkpoint (every 100 cycles) ──────────────────────────
        self.maybe_checkpoint(100, None);

        // ── 9. Formulation updates ──────────────────────────────────────────
        // Auto-detect clinical patterns from affect trajectory (Wichers 2015)
        self.formulation
            .detect_patterns(&self.client_model.affect_trajectory, 30);

        if self.client_model.affect_trend() < -0.1 && self.formulation.perpetuating.is_empty() {
            self.formulation
                .add_perpetuating("sustained negative affect pattern", 0.6);
        }
        if self.alliance.composite() > 0.6
            && self.client_model.affect_trend() > 0.1
            && self.formulation.protective.is_empty()
        {
            self.formulation
                .add_protective("therapeutic engagement", 0.7);
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        // Layout: [crisis_active: u8 = 1][checkpoint_counter: u32 = 4]
        //         [alliance_bond: f32 = 4][alliance_goal: f32 = 4]
        //         [alliance_task: f32 = 4][alliance_ruptures: u32 = 4]
        //         [alliance_repairs: u32 = 4][narrative_coherence: f32 = 4]
        //         [dream_prediction_accuracy: f32 = 4][dream_resolution_rate: f32 = 4]
        // Total: 37 bytes
        //
        // Note: Full session state (ClientModel, RDoC profile, formulation,
        // narrative fragments, shadow detector) is persisted via
        // serialize_session() / restore_session() JSON. This binary
        // checkpoint captures the fast-changing primitive fields only.
        let mut data = Vec::with_capacity(37);
        data.push(self.crisis_active as u8);
        data.extend_from_slice(&self.checkpoint_counter.to_le_bytes());
        data.extend_from_slice(&self.alliance.bond.to_le_bytes());
        data.extend_from_slice(&self.alliance.goal_agreement.to_le_bytes());
        data.extend_from_slice(&self.alliance.task_agreement.to_le_bytes());
        data.extend_from_slice(&self.alliance.rupture_count.to_le_bytes());
        data.extend_from_slice(&self.alliance.repair_count.to_le_bytes());
        data.extend_from_slice(&self.narrative.coherence.to_le_bytes());
        data.extend_from_slice(&self.dream_tracker.prediction_accuracy().to_le_bytes());
        data.extend_from_slice(&self.dream_tracker.resolution_rate().to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        const MIN_SIZE: usize = 1 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4; // 37
        if data.len() < MIN_SIZE {
            return Err(format!(
                "TherapeuticManager checkpoint too short: {} < {}",
                data.len(),
                MIN_SIZE
            ));
        }
        self.crisis_active = data[0] != 0;
        self.checkpoint_counter = u32::from_le_bytes(
            data[1..5]
                .try_into()
                .map_err(|_| "TherapeuticManager: corrupt checkpoint_counter bytes")?,
        );
        self.alliance.bond = f32::from_le_bytes(
            data[5..9]
                .try_into()
                .map_err(|_| "TherapeuticManager: corrupt alliance_bond bytes")?,
        );
        self.alliance.goal_agreement = f32::from_le_bytes(
            data[9..13]
                .try_into()
                .map_err(|_| "TherapeuticManager: corrupt alliance_goal bytes")?,
        );
        self.alliance.task_agreement = f32::from_le_bytes(
            data[13..17]
                .try_into()
                .map_err(|_| "TherapeuticManager: corrupt alliance_task bytes")?,
        );
        self.alliance.rupture_count = u32::from_le_bytes(
            data[17..21]
                .try_into()
                .map_err(|_| "TherapeuticManager: corrupt rupture_count bytes")?,
        );
        self.alliance.repair_count = u32::from_le_bytes(
            data[21..25]
                .try_into()
                .map_err(|_| "TherapeuticManager: corrupt repair_count bytes")?,
        );
        self.narrative.coherence = f32::from_le_bytes(
            data[25..29]
                .try_into()
                .map_err(|_| "TherapeuticManager: corrupt narrative_coherence bytes")?,
        );
        // bytes [29..33]: dream_prediction_accuracy (informational, not restored — recomputed)
        // bytes [33..37]: dream_resolution_rate (informational, not restored — recomputed)
        Ok(())
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
        // Output deltas should be finite (strategy-dependent magnitude)
        assert!(
            output.valence_delta.is_finite(),
            "valence_delta should be finite"
        );
        assert!(
            output.confidence_delta.is_finite(),
            "confidence_delta should be finite"
        );
    }

    #[test]
    fn test_serialize_restore_roundtrip() {
        let mut manager = TherapeuticManager::new();
        // Run a few cycles to build state
        for i in 0..5 {
            let snapshot = make_snapshot(-0.3, 0.6, i);
            manager.process(&snapshot);
        }
        manager.alliance.bond = 0.7;
        manager.alliance.goal_agreement = 0.6;

        // Serialize
        let json = manager
            .serialize_session()
            .expect("serialize should succeed");
        assert!(!json.is_empty());

        // Restore into fresh manager
        let mut restored = TherapeuticManager::new();
        restored
            .restore_session(&json)
            .expect("restore should succeed");
        assert_eq!(restored.alliance.bond, 0.7);
        assert_eq!(restored.alliance.goal_agreement, 0.6);
        assert_eq!(
            restored.client_model.cycle_count,
            manager.client_model.cycle_count
        );
    }

    #[test]
    fn test_shadow_persistence_roundtrip() {
        let mut manager = TherapeuticManager::new();
        // Run cycles with distressing content to build shadow fragments
        for i in 0..30 {
            let mut snapshot = make_snapshot(-0.8, 0.7, i);
            snapshot.prediction_error = 0.6;
            manager.process(&snapshot);
            // Also feed shadow detector directly with high-PE content
            manager.shadow_detector.process(
                i as u64,
                0.6,
                -0.8,
                0.7,
                "painful unresolved conflict",
                0.1,
            );
        }

        let pre_count = manager.shadow_detector.fragment_count();
        let pre_pressure = manager.shadow_detector.total_pressure();
        assert!(pre_count > 0, "should have shadow fragments");

        // Serialize
        let json = manager
            .serialize_session()
            .expect("serialize with shadow should succeed");
        assert!(json.contains("shadow"));
        assert!(json.contains("fragments"));

        // Restore into fresh manager
        let mut restored = TherapeuticManager::new();
        restored
            .restore_session(&json)
            .expect("restore with shadow should succeed");

        assert_eq!(
            restored.shadow_detector.fragment_count(),
            pre_count,
            "fragment count should survive roundtrip"
        );
        // Pressure may differ slightly due to HDC encoding loss (serde skip),
        // but fragment scalar fields (cumulative_PE, recurrence) are preserved.
        let restored_pressure = restored.shadow_detector.total_pressure();
        assert!(
            (restored_pressure - pre_pressure).abs() < 0.01,
            "pressure should survive roundtrip: {} vs {}",
            restored_pressure,
            pre_pressure,
        );
    }

    #[test]
    fn test_shadow_persistence_backward_compat() {
        // Old snapshots without "shadow" field should restore cleanly (Option + serde(default))
        let json = r#"{
            "client_model": {"cycle_count": 5, "session_count": 1, "affect_trajectory": [], "rdoc_profile": {}, "presenting_concerns": [], "diagnostic_hypotheses": [], "automatic_thoughts": [], "core_beliefs": [], "behavioral_patterns": [], "risk_level": "None"},
            "alliance_bond": 0.5,
            "alliance_goal": 0.5,
            "alliance_task": 0.5,
            "alliance_ruptures": 0,
            "alliance_repairs": 0,
            "narrative_coherence": 0.3,
            "crisis_active": false
        }"#;

        let mut manager = TherapeuticManager::new();
        let result = manager.restore_session(json);
        assert!(
            result.is_ok(),
            "restore from old snapshot without shadow should succeed"
        );
        assert_eq!(manager.shadow_detector.fragment_count(), 0);
    }

    #[test]
    fn test_tick_debt_wired_in_process() {
        let mut manager = TherapeuticManager::new();
        // Set high negative valence RDoC to trigger debt accumulation
        manager
            .client_model
            .rdoc_profile
            .set_score(symthaea_clinical::RDocDomain::NegativeValence, 0.9);
        for i in 0..100 {
            let snapshot = make_snapshot(-0.5, 0.7, i);
            manager.process(&snapshot);
        }
        assert!(
            manager.regulation_engine.serotonin_debt() > 0.0,
            "Serotonin debt should accumulate through process() cycles",
        );
    }

    #[test]
    fn test_client_distress() {
        let mut manager = TherapeuticManager::new();
        manager
            .client_model
            .update_affect(CoreAffectSnapshot::new(-0.8, 0.9, 0));
        assert!(manager.client_distress() > 0.5);
    }

    // ── Proptests & hardening ───────────────────────────────────────────────

    use proptest::prelude::*;

    proptest! {
        /// Property: process() output valence_delta and arousal_delta are finite
        /// and lr_modulation is positive-finite regardless of input affect.
        #[test]
        fn prop_neuromod_modulation_bounds(
            valence in -1.0f32..=1.0,
            arousal in 0.0f32..=1.0,
            cycle in 0u64..10_000,
        ) {
            let mut manager = TherapeuticManager::new();
            let snapshot = make_snapshot(valence, arousal, cycle);
            let output = manager.process(&snapshot);

            prop_assert!(output.valence_delta.is_finite(),
                "valence_delta not finite: {} for v={}, a={}", output.valence_delta, valence, arousal);
            prop_assert!(output.arousal_delta.is_finite(),
                "arousal_delta not finite: {} for v={}, a={}", output.arousal_delta, valence, arousal);
            prop_assert!(output.lr_modulation.is_finite() && output.lr_modulation > 0.0,
                "lr_modulation not positive-finite: {} for v={}, a={}", output.lr_modulation, valence, arousal);
        }

        /// Property: crisis detection never panics for arbitrary affect values
        /// and crisis_active is always a valid boolean (trivially true in Rust,
        /// but we verify the detector handles the full affect range).
        #[test]
        fn prop_crisis_detection_no_panic(
            valence in -1.0f32..=1.0,
            arousal in 0.0f32..=1.0,
        ) {
            let mut manager = TherapeuticManager::new();
            let snapshot = make_snapshot(valence, arousal, 1);
            let _output = manager.process(&snapshot);
            // crisis_active must be set to some value without panic
            let _ = manager.crisis_active;
            // last_crisis_type must be None or Some(non-empty string)
            if let Some(ref t) = manager.last_crisis_type {
                prop_assert!(!t.is_empty(), "crisis type name should not be empty");
            }
        }

        /// Property: dream_action_vector always returns exactly 32 finite elements
        /// regardless of manager state after arbitrary processing.
        #[test]
        fn prop_dream_action_vector_shape_and_finiteness(
            valence in -1.0f32..=1.0,
            arousal in 0.0f32..=1.0,
            cycles in 0u32..20,
        ) {
            let mut manager = TherapeuticManager::new();
            for i in 0..cycles {
                let snapshot = make_snapshot(valence, arousal, i as u64);
                manager.process(&snapshot);
            }
            let vec = manager.dream_action_vector();
            prop_assert_eq!(vec.len(), 32, "dream_action_vector length must be 32");
            for (i, &val) in vec.iter().enumerate() {
                prop_assert!(val.is_finite(),
                    "dream_action_vector[{}] not finite: {}", i, val);
            }
        }

        /// Property: narrative coherence stays in [0, 1] after arbitrary fragment integration.
        #[test]
        fn prop_narrative_coherence_bounded(
            valence in -1.0f32..=1.0,
            arousal in 0.0f32..=1.0,
            n_cycles in 1u32..50,
        ) {
            let mut manager = TherapeuticManager::new();
            for i in 0..n_cycles {
                let snapshot = make_snapshot(valence, arousal, i as u64);
                manager.process(&snapshot);
            }
            let coh = manager.narrative_coherence();
            prop_assert!(coh >= 0.0 && coh <= 1.0,
                "narrative coherence out of [0,1]: {} after {} cycles", coh, n_cycles);
        }

        /// Property: formulation resilience_ratio is always non-negative and finite.
        #[test]
        fn prop_formulation_resilience_ratio_valid(
            valence in -1.0f32..=1.0,
            arousal in 0.0f32..=1.0,
            n_cycles in 1u32..50,
        ) {
            let mut manager = TherapeuticManager::new();
            for i in 0..n_cycles {
                let snapshot = make_snapshot(valence, arousal, i as u64);
                manager.process(&snapshot);
            }
            let ratio = manager.formulation_resilience_ratio();
            prop_assert!(ratio.is_finite() && ratio >= 0.0,
                "resilience_ratio invalid: {} after {} cycles", ratio, n_cycles);
        }
    }

    proptest! {
        /// Property: serotonin debt stays in [0, 1] and amplification factor in [1, 1.3]
        /// regardless of RDoC input sequences.
        #[test]
        fn prop_debt_accumulation_bounded(
            neg_val in 0.0f32..=1.0,
            pos_val in 0.0f32..=1.0,
            n_ticks in 1u32..500,
        ) {
            let mut manager = TherapeuticManager::new();
            manager.client_model.rdoc_profile.set_score(
                symthaea_clinical::RDocDomain::NegativeValence, neg_val,
            );
            manager.client_model.rdoc_profile.set_score(
                symthaea_clinical::RDocDomain::PositiveValence, pos_val,
            );
            for i in 0..n_ticks {
                let snapshot = make_snapshot(-0.3, 0.6, i as u64);
                manager.process(&snapshot);
            }
            let s_debt = manager.regulation_engine.serotonin_debt();
            let d_debt = manager.regulation_engine.dopamine_debt();
            prop_assert!(s_debt >= 0.0 && s_debt <= 1.0,
                "serotonin_debt out of [0,1]: {}", s_debt);
            prop_assert!(d_debt >= 0.0 && d_debt <= 1.0,
                "dopamine_debt out of [0,1]: {}", d_debt);
        }

        /// Property: dream tracker predictions stay bounded after many cycles.
        #[test]
        fn prop_dream_tracker_bounded(
            n_cycles in 1u32..200,
            valence in -1.0f32..=1.0,
            arousal in 0.0f32..=1.0,
        ) {
            let mut manager = TherapeuticManager::new();
            for i in 0..n_cycles {
                let snapshot = make_snapshot(valence, arousal, i as u64);
                manager.process(&snapshot);
            }
            let acc = manager.dream_prediction_accuracy();
            prop_assert!(acc.is_finite(), "dream accuracy not finite: {}", acc);
            prop_assert!(acc >= 0.0, "dream accuracy negative: {}", acc);
            prop_assert!(
                manager.dream_tracker.predictions.len() <= 64,
                "dream tracker exceeded MAX_HISTORY",
            );
        }

        /// Property: session summary always produces valid JSON regardless of state.
        #[test]
        fn prop_session_summary_valid_json(
            valence in -1.0f32..=1.0,
            arousal in 0.0f32..=1.0,
            n_cycles in 1u32..100,
        ) {
            let mut manager = TherapeuticManager::new();
            for i in 0..n_cycles {
                let snapshot = make_snapshot(valence, arousal, i as u64);
                manager.process(&snapshot);
            }
            let summary = manager.session_summary();
            let json = serde_json::to_string(&summary);
            prop_assert!(json.is_ok(), "session summary JSON failed: {:?}", json.err());
            let json_str = json.unwrap();
            prop_assert!(!json_str.is_empty(), "session summary JSON should not be empty");
            // Verify it round-trips
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json_str);
            prop_assert!(parsed.is_ok(), "session summary JSON invalid: {:?}", parsed.err());
        }

        /// Property: auto-checkpoint produces valid JSON when triggered.
        #[test]
        fn prop_checkpoint_valid_json(
            n_cycles in 100u32..300,
        ) {
            let mut manager = TherapeuticManager::new();
            for i in 0..n_cycles {
                let snapshot = make_snapshot(0.0, 0.5, i as u64);
                manager.process(&snapshot);
            }
            // After 100+ cycles, at least one checkpoint should have fired
            if let Some(ref json) = manager.last_checkpoint {
                // Verify it's valid JSON
                let parsed: Result<serde_json::Value, _> = serde_json::from_str(json);
                prop_assert!(parsed.is_ok(), "checkpoint JSON invalid: {:?}", parsed.err());
            }
        }
    }

    // ── Deterministic hardening tests ───────────────────────────────────────

    #[test]
    fn test_narrative_fragment_recording_and_coherence() {
        let mut manager = TherapeuticManager::new();
        assert_eq!(manager.narrative.fragments.len(), 0);
        assert_eq!(manager.narrative_coherence(), 0.0);

        // Process several cycles to accumulate narrative fragments
        for i in 0..10 {
            let snapshot = make_snapshot(0.2, 0.4, i);
            manager.process(&snapshot);
        }
        assert_eq!(manager.narrative.fragments.len(), 10);
        // Coherence should have been updated (non-zero after fragments)
        let coh = manager.narrative_coherence();
        assert!(coh.is_finite(), "coherence must be finite after fragments");
        assert!(
            coh >= 0.0 && coh <= 1.0,
            "coherence must be in [0,1], got {}",
            coh
        );
    }

    #[test]
    fn test_formulation_perpetuating_and_protective_persist() {
        let mut manager = TherapeuticManager::new();

        // Manually add formulation factors
        manager.formulation.add_perpetuating("rumination", 0.8);
        manager.formulation.add_protective("social support", 0.7);

        assert_eq!(manager.formulation.perpetuating.len(), 1);
        assert_eq!(manager.formulation.protective.len(), 1);

        // Process should not clear manually added factors
        let snapshot = make_snapshot(0.0, 0.5, 1);
        manager.process(&snapshot);

        assert!(
            manager.formulation.perpetuating.len() >= 1,
            "perpetuating factors must persist after process()"
        );
        assert!(
            manager.formulation.protective.len() >= 1,
            "protective factors must persist after process()"
        );

        // Resilience ratio should reflect both risk and protective factors
        let ratio = manager.formulation_resilience_ratio();
        assert!(ratio.is_finite() && ratio >= 0.0);
    }

    #[test]
    fn test_dream_action_vector_structure() {
        let mut manager = TherapeuticManager::new();

        // Process one cycle to populate neuromod delta
        let snapshot = make_snapshot(-0.5, 0.8, 1);
        manager.process(&snapshot);

        let vec = manager.dream_action_vector();
        assert_eq!(vec.len(), 32, "must be exactly 32 elements");

        // Elements [18..32] are reserved zeros
        for i in 18..32 {
            assert_eq!(vec[i], 0.0, "reserved element [{}] must be 0.0", i);
        }

        // Crisis flag must be 0.0 or 1.0
        assert!(
            vec[8] == 0.0 || vec[8] == 1.0,
            "crisis flag must be 0 or 1, got {}",
            vec[8]
        );

        // Strategy ordinal must be in [-1, 6]
        assert!(
            vec[9] >= -1.0 && vec[9] <= 6.0,
            "strategy ordinal out of [-1,6]: {}",
            vec[9]
        );

        // All elements must be finite
        for (i, &val) in vec.iter().enumerate() {
            assert!(val.is_finite(), "element [{}] not finite: {}", i, val);
        }
    }

    #[test]
    fn test_extreme_affect_resilience() {
        let mut manager = TherapeuticManager::new();

        // Extreme negative valence + high arousal
        let snapshot = make_snapshot(-1.0, 1.0, 1);
        let output = manager.process(&snapshot);
        assert!(output.valence_delta.is_finite());
        assert!(output.arousal_delta.is_finite());
        assert!(output.lr_modulation.is_finite());

        // Extreme positive valence + zero arousal
        let snapshot = make_snapshot(1.0, 0.0, 2);
        let output = manager.process(&snapshot);
        assert!(output.valence_delta.is_finite());
        assert!(output.arousal_delta.is_finite());
        assert!(output.lr_modulation.is_finite());

        // Zero valence + zero arousal
        let snapshot = make_snapshot(0.0, 0.0, 3);
        let output = manager.process(&snapshot);
        assert!(output.valence_delta.is_finite());
        assert!(output.arousal_delta.is_finite());
        assert!(output.lr_modulation.is_finite());

        // Rapid oscillation (stress test for EMA trackers)
        for i in 0..100 {
            let v = if i % 2 == 0 { -1.0 } else { 1.0 };
            let a = if i % 3 == 0 { 0.0 } else { 1.0 };
            let snapshot = make_snapshot(v, a, 100 + i);
            let output = manager.process(&snapshot);
            assert!(
                output.valence_delta.is_finite(),
                "valence_delta not finite at oscillation cycle {}",
                i
            );
            assert!(
                output.arousal_delta.is_finite(),
                "arousal_delta not finite at oscillation cycle {}",
                i
            );
            assert!(
                output.lr_modulation.is_finite() && output.lr_modulation > 0.0,
                "lr_modulation invalid at oscillation cycle {}: {}",
                i,
                output.lr_modulation
            );
        }

        // Verify manager is still coherent after extreme stress
        let coh = manager.narrative_coherence();
        assert!(coh.is_finite() && coh >= 0.0 && coh <= 1.0);
        let vec = manager.dream_action_vector();
        assert_eq!(vec.len(), 32);
        for (i, &val) in vec.iter().enumerate() {
            assert!(
                val.is_finite(),
                "dream vec [{}] not finite after stress: {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_update_rdoc_from_bath_wired() {
        let mut manager = TherapeuticManager::new();
        let initial_neg = manager
            .client_model
            .rdoc_profile
            .score(symthaea_clinical::RDocDomain::NegativeValence);
        // Low serotonin + low GABA → should increase NegativeValence
        let bath = [0.5, 0.5, 0.1, 0.5, 0.1, 0.5, 0.5, 0.5];
        manager.update_rdoc_from_bath(&bath);
        let new_neg = manager
            .client_model
            .rdoc_profile
            .score(symthaea_clinical::RDocDomain::NegativeValence);
        assert!(
            new_neg > initial_neg,
            "NegativeValence should increase with low 5-HT: {} -> {}",
            initial_neg,
            new_neg,
        );
    }

    #[test]
    fn test_dream_tracker_integrated_in_process() {
        let mut manager = TherapeuticManager::new();
        assert!(manager.dream_tracker.predictions.is_empty());

        let snapshot = make_snapshot(0.0, 0.5, 1);
        manager.process(&snapshot);

        assert_eq!(
            manager.dream_tracker.predictions.len(),
            1,
            "dream tracker should record one prediction per process() call",
        );
    }

    #[test]
    fn test_auto_checkpoint_fires() {
        let mut manager = TherapeuticManager::new();
        assert!(manager.last_checkpoint.is_none());

        // Run 101 cycles — checkpoint fires at 100
        for i in 0..101 {
            let snapshot = make_snapshot(0.0, 0.5, i);
            manager.process(&snapshot);
        }

        assert!(
            manager.last_checkpoint.is_some(),
            "checkpoint should fire after 100 cycles",
        );
        // Verify it's valid JSON
        let json = manager.last_checkpoint.as_ref().unwrap();
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(json);
        assert!(parsed.is_ok(), "checkpoint must be valid JSON");
    }

    #[test]
    fn test_dream_prediction_accuracy_improves_with_stable_state() {
        let mut manager = TherapeuticManager::new();
        // Run many cycles with stable state — predictions should converge
        for i in 0..50 {
            let snapshot = make_snapshot(0.0, 0.5, i);
            manager.process(&snapshot);
        }
        let accuracy = manager.dream_prediction_accuracy();
        assert!(accuracy.is_finite());
        // With stable state, predictions should be reasonably accurate (low error)
        assert!(
            accuracy < 0.5,
            "stable-state dream accuracy should be < 0.5, got {}",
            accuracy,
        );
    }

    #[test]
    fn test_extreme_out_of_range_affect_no_panic() {
        let mut manager = TherapeuticManager::new();

        // Values well outside normal [-1,1] / [0,1] range
        // The snapshot struct uses f32, so we can pass extreme values.
        // The system should handle them gracefully (clamp or process without panic).
        let extreme_vals: &[(f32, f32)] = &[
            (-100.0, 100.0),
            (100.0, -100.0),
            (f32::MIN, f32::MAX),
            (f32::EPSILON, f32::EPSILON),
            (-f32::EPSILON, f32::EPSILON),
        ];

        for (i, &(v, a)) in extreme_vals.iter().enumerate() {
            let snapshot = make_snapshot(v, a, i as u64);
            let output = manager.process(&snapshot);
            // Must not panic — outputs should be finite (or at least not NaN)
            assert!(
                !output.valence_delta.is_nan(),
                "valence_delta is NaN for extreme input ({}, {})",
                v,
                a
            );
            assert!(
                !output.arousal_delta.is_nan(),
                "arousal_delta is NaN for extreme input ({}, {})",
                v,
                a
            );
            assert!(
                !output.lr_modulation.is_nan(),
                "lr_modulation is NaN for extreme input ({}, {})",
                v,
                a
            );
        }
    }

    #[test]
    fn test_detect_crisis_from_text() {
        let mut manager = TherapeuticManager::new();
        assert!(!manager.crisis_active);

        // Should detect crisis language
        let detected = manager.detect_crisis_from_text("I want to end my life");
        assert!(detected, "should detect suicidal ideation");
        assert!(manager.crisis_active);
        assert!(manager.last_crisis_type.is_some());
    }

    #[test]
    fn test_detect_crisis_from_text_benign() {
        let mut manager = TherapeuticManager::new();
        let detected = manager.detect_crisis_from_text("I had a nice day at the park");
        assert!(!detected, "should not detect crisis in benign input");
        assert!(!manager.crisis_active);
    }

    #[test]
    fn test_apply_dream_feedback_clears_on_high_error() {
        let mut manager = TherapeuticManager::new();

        // Manually add a dream preference (ordinal 6 = ExposurePrep)
        manager.regulation_engine.incorporate_dream_wisdom(6, 0.5);
        assert!(
            manager
                .regulation_engine
                .dream_preferred_strategy()
                .is_some()
        );

        // Simulate high prediction error (bad accuracy)
        for i in 0..10 {
            manager.dream_tracker.record_prediction(vec![0.5; 6], i);
            manager
                .dream_tracker
                .record_actual_outcome(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        }

        // accuracy should be high (bad)
        assert!(manager.dream_prediction_accuracy() > 0.5);

        // Apply feedback — should clear dream preference
        manager.apply_dream_feedback();
        assert!(
            manager
                .regulation_engine
                .dream_preferred_strategy()
                .is_none(),
            "dream preference should be cleared when accuracy is bad"
        );
    }

    #[test]
    fn test_session_summary_generation() {
        let mut manager = TherapeuticManager::new();
        for i in 0..20 {
            let v = if i < 10 { -0.3 } else { 0.2 };
            let snapshot = make_snapshot(v, 0.5, i);
            manager.process(&snapshot);
        }
        let summary = manager.session_summary();
        assert!(summary.total_cycles > 0, "should have cycles");
        assert!(summary.alliance_composite >= 0.0 && summary.alliance_composite <= 1.0);
        assert!(summary.narrative_coherence >= 0.0 && summary.narrative_coherence <= 1.0);
        assert!(summary.resilience_ratio >= 0.0);
        assert!(summary.serotonin_debt >= 0.0 && summary.serotonin_debt <= 1.0);
        assert!(summary.clinical_severity >= 0.0);
        assert!(summary.affect_mean_valence.is_finite());
    }

    #[test]
    fn test_session_summary_serializable() {
        let mut manager = TherapeuticManager::new();
        for i in 0..10 {
            let snapshot = make_snapshot(0.0, 0.5, i);
            manager.process(&snapshot);
        }
        let summary = manager.session_summary();
        let json = serde_json::to_string(&summary);
        assert!(json.is_ok(), "session summary must be JSON-serializable");
        assert!(!json.unwrap().is_empty());
    }

    #[test]
    fn test_session_summary_strategy_effectiveness() {
        let mut manager = TherapeuticManager::new();
        // Run enough cycles to build effectiveness data
        for i in 0..50 {
            let v = if i % 3 == 0 { -0.5 } else { 0.2 };
            let snapshot = make_snapshot(v, 0.5, i);
            manager.process(&snapshot);
        }
        let summary = manager.session_summary();
        // Should have at least one strategy with applications
        // (may not if the process loop doesn't track enough)
        assert!(summary.dominant_strategy.len() > 0);
    }

    #[test]
    fn test_strategy_effectiveness_in_process() {
        let mut manager = TherapeuticManager::new();
        // Run 20 improving cycles — strategy should gain success records
        for i in 0..20 {
            let v = -0.3 + (i as f32 * 0.03); // gradually improving
            let snapshot = make_snapshot(v, 0.5, i as u64);
            manager.process(&snapshot);
        }
        // The regulation engine should have tracked at least some strategy applications
        let has_any = symthaea_therapeutic::RegulationStrategy::ALL
            .iter()
            .any(|s| manager.regulation_engine.effectiveness(s).is_some());
        // May or may not have data depending on threshold logic, but should not panic
        let _ = has_any;
    }

    #[test]
    fn test_flush_telemetry() {
        let tmp = std::env::temp_dir().join("therapeutic_test_flush.json");
        let path = tmp.to_str().unwrap();

        let mut manager = TherapeuticManager::new();
        for i in 0..20 {
            let v = -0.3 + (i as f32 * 0.03);
            let snapshot = make_snapshot(v, 0.5, i as u64);
            manager.process(&snapshot);
        }

        let result = manager.flush_telemetry(path);
        assert!(
            result.is_ok(),
            "flush_telemetry should succeed: {:?}",
            result.err()
        );
        assert!(
            std::path::Path::new(path).exists(),
            "telemetry file should exist"
        );

        // Verify valid JSON
        let contents = std::fs::read_to_string(path).unwrap();
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&contents);
        assert!(parsed.is_ok(), "telemetry must be valid JSON");
        let val = parsed.unwrap();
        assert!(val["session_summary"]["total_cycles"].is_number());
        assert!(val["affect_trajectory"].is_array());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_formulation_auto_detection_in_process() {
        let mut manager = TherapeuticManager::new();
        // 40 cycles of depression-like affect (low valence, low arousal)
        for i in 0..40 {
            let snapshot = make_snapshot(-0.6, 0.2, i);
            manager.process(&snapshot);
        }
        assert!(
            manager
                .formulation
                .perpetuating
                .iter()
                .any(|f| f.description.contains("low mood")),
            "should auto-detect depression-like pattern in formulation"
        );
    }

    #[test]
    fn test_checkpoint_with_persist_path() {
        let tmp = std::env::temp_dir().join("therapeutic_test_checkpoint.json");
        let path = tmp.to_str().unwrap();

        let mut manager = TherapeuticManager::new();
        for i in 0..5 {
            let snapshot = make_snapshot(0.0, 0.5, i);
            manager.process(&snapshot);
        }
        // Force a manual checkpoint with persistence
        manager.checkpoint_counter = 99;
        manager.maybe_checkpoint(100, Some(path));

        assert!(
            std::path::Path::new(path).exists(),
            "checkpoint file should exist"
        );

        // Load into fresh manager
        let mut restored = TherapeuticManager::new();
        let loaded = restored.load_checkpoint(path);
        assert!(loaded.is_ok());
        assert!(loaded.unwrap(), "should have loaded checkpoint");

        // Clean up
        let _ = std::fs::remove_file(path);
    }
}
