// Scaffolded for upcoming wiring — callers not yet connected
#![allow(dead_code)]

//! # FabricationManager — Manufacturing Consciousness Integration
//!
//! Translates Cincinnati quality events, ManufacturingTwin readings, and
//! DesignLoopTwin readings into embodied cognitive signals: neuromodulatory
//! nudges, confidence deltas, and learning rate modulation.
//!
//! ## Architecture
//!
//! Follows the [`GovernanceManager`] pattern:
//! - Event queue: inject / drain / process
//! - Neuromod coupling: phasic injections + baseline nudges
//! - CognitiveSubsystem trait implementation (interval 47, co-prime)
//!
//! ## Event Sources
//!
//! - **Cincinnati anomaly detection**: severity → NE burst (Aston-Jones 2005)
//! - **Print job completion**: quality → DA burst (Schultz 1997)
//! - **Safety level changes**: Red → NE+5-HT stress (Sapolsky 2004)
//! - **Quality trend**: improving → 5-HT rise (Crockett 2009)
//! - **PoGF scoring**: high score → oxytocin (Zak 2012)

use std::collections::VecDeque;

use super::super::subsystem_trait::{
    output_flags, CognitiveSubsystem, CycleSnapshot, SubsystemOutput,
};
use super::super::thresholds;

use symthaea_fabrication_kernel::{
    DesignLoopReading, DesignLoopTwin, ManufacturingReading, ManufacturingSafetyLevel,
    ManufacturingTwin,
};

// ═══════════════════════════════════════════════════════════════════════════════
// EVENT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// A fabrication event injected from external sources (Cincinnati monitors,
/// printer controllers, Mycelix bridge, etc.).
#[derive(Debug, Clone)]
pub struct FabricationEvent {
    pub kind: FabricationEventKind,
    pub timestamp_secs: u64,
}

/// The type of fabrication event.
#[derive(Debug, Clone)]
pub enum FabricationEventKind {
    /// Cincinnati in-situ anomaly detection.
    CincinnatiAnomaly {
        anomaly_type: String,
        severity: f32,
        layer: u32,
    },
    /// A print job has started.
    PrintJobStarted { job_id: String },
    /// A print job completed with quality metrics.
    PrintJobCompleted {
        job_id: String,
        quality_score: f32,
        pog_score: f32,
    },
    /// Manufacturing safety level changed.
    SafetyLevelChanged { level: ManufacturingSafetyLevel },
    /// ManufacturingTwin reading (periodic sensor data).
    TwinReading { reading: ManufacturingReading },
    /// DesignLoopTwin reading (design-manufacture feedback).
    DesignLoopReading { reading: DesignLoopReading },
}

// ═══════════════════════════════════════════════════════════════════════════════
// NEUROMOD QUEUE TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// A pending pharmacological injection (target transmitter, dose, half-life in cycles).
#[derive(Debug, Clone)]
pub(crate) struct PendingInjection {
    pub(crate) target: &'static str,
    pub(crate) dose: f32,
    pub(crate) half_life: u32,
}

/// A pending baseline nudge (target transmitter, delta).
#[derive(Debug, Clone)]
pub(crate) struct PendingBaseline {
    pub(crate) target: &'static str,
    pub(crate) nudge: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TELEMETRY
// ═══════════════════════════════════════════════════════════════════════════════

/// Fabrication telemetry snapshot for CycleMetadata / Pulse.
#[derive(Debug, Clone, Default)]
pub struct FabricationTelemetry {
    /// ManufacturingTwin free energy (0 = equilibrium, >0 = surprise).
    pub manufacturing_free_energy: f64,
    /// DesignLoopTwin free energy.
    pub design_loop_free_energy: f64,
    /// Current manufacturing safety level.
    pub safety_level: String,
    /// Total anomalies detected this cycle.
    pub anomaly_count: u32,
    /// EMA of anomaly severity.
    pub anomaly_ema: f32,
    /// Recommended manufacturing action.
    pub recommended_action: String,
    /// Mean prediction coherence across horizons.
    pub prediction_coherence: f32,
    /// EMA of PoGF scores.
    pub pog_score_ema: f32,
    /// Number of active print jobs.
    pub active_print_jobs: u32,
    /// Reward EMA for external consumption.
    pub reward_ema: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// FABRICATION MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/// Cognitive subsystem that translates manufacturing events and digital twin
/// readings into embodied cognitive signals: neuromodulatory nudges, confidence
/// deltas, and learning rate modulation.
pub struct FabricationManager {
    /// ManufacturingTwin for process monitoring.
    manufacturing_twin: ManufacturingTwin,
    /// DesignLoopTwin for design-manufacture feedback.
    design_loop_twin: DesignLoopTwin,

    /// Pending events (drained each `process()` call).
    pending_events: Vec<FabricationEvent>,

    /// Pending pharmacological injections to apply after process().
    pending_injections: Vec<PendingInjection>,
    /// Pending baseline nudges to apply after process().
    pending_baselines: Vec<PendingBaseline>,

    /// Per-harmony accumulator for fabrication feedback.
    harmonic_deltas: [f64; 8],

    /// Current manufacturing safety level.
    safety_level: ManufacturingSafetyLevel,
    /// EMA of anomaly severity.
    anomaly_ema: f32,
    /// Count of anomalies in current cycle.
    anomaly_count_this_cycle: u32,
    /// Total anomaly count (lifetime).
    total_anomaly_count: u64,

    /// Active print job count.
    active_print_jobs: u32,
    /// EMA of PoGF scores from completed prints.
    pog_score_ema: f32,

    /// Latest ManufacturingTwin free energy.
    last_manufacturing_fe: f64,
    /// Latest DesignLoopTwin free energy.
    last_design_loop_fe: f64,
    /// Latest recommended action string.
    last_recommended_action: String,
    /// Mean prediction coherence across horizons.
    last_prediction_coherence: f32,

    /// Exponential moving average of reward signal.
    reward_ema: f64,

    /// Current cycle number (set from snapshot each process()).
    current_cycle: u64,

    /// Accumulated confidence nudge from cross-coupling (drained in process()).
    confidence_nudge_acc: f64,
}

impl Default for FabricationManager {
    fn default() -> Self {
        Self {
            manufacturing_twin: ManufacturingTwin::new(),
            design_loop_twin: DesignLoopTwin::new(),
            pending_events: Vec::new(),
            pending_injections: Vec::new(),
            pending_baselines: Vec::new(),
            harmonic_deltas: [0.0; 8],
            safety_level: ManufacturingSafetyLevel::Green,
            anomaly_ema: 0.0,
            anomaly_count_this_cycle: 0,
            total_anomaly_count: 0,
            active_print_jobs: 0,
            pog_score_ema: 0.5,
            last_manufacturing_fe: 0.0,
            last_design_loop_fe: 0.0,
            last_recommended_action: String::new(),
            last_prediction_coherence: 1.0,
            reward_ema: 0.0,
            current_cycle: 0,
            confidence_nudge_acc: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════════

impl FabricationManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = thresholds::FABRICATION_MANAGER_INTERVAL;

    /// Maximum events processed per cycle.
    const MAX_EVENTS_PER_CYCLE: usize = thresholds::FAB_MAX_EVENTS_PER_CYCLE;

    /// Inject a fabrication event for processing in the next `process()` call.
    pub fn inject_event(&mut self, event: FabricationEvent) {
        self.pending_events.push(event);
    }

    /// Drain pending neuromod injections. Called by the accessor layer.
    pub fn drain_injections(&mut self) -> Vec<PendingInjection> {
        std::mem::take(&mut self.pending_injections)
    }

    /// Drain pending baseline nudges. Called by the accessor layer.
    pub fn drain_baselines(&mut self) -> Vec<PendingBaseline> {
        std::mem::take(&mut self.pending_baselines)
    }

    /// Current reward EMA (for external telemetry).
    pub fn reward_ema(&self) -> f64 {
        self.reward_ema
    }

    /// Current safety level.
    pub fn safety_level(&self) -> ManufacturingSafetyLevel {
        self.safety_level
    }

    /// Current anomaly EMA.
    pub fn anomaly_ema(&self) -> f32 {
        self.anomaly_ema
    }

    /// Current PoGF score EMA.
    pub fn pog_score_ema(&self) -> f32 {
        self.pog_score_ema
    }

    /// Active print jobs.
    pub fn active_print_jobs(&self) -> u32 {
        self.active_print_jobs
    }

    /// Accumulate a confidence nudge from cross-coupling (drained in process()).
    /// Clamped to [-0.1, 0.1] per call, NaN-guarded.
    pub fn nudge_confidence(&mut self, delta: f64) {
        if delta.is_finite() {
            self.confidence_nudge_acc =
                (self.confidence_nudge_acc + delta.clamp(-0.1, 0.1)).clamp(-0.2, 0.2);
        }
    }

    /// Get a telemetry snapshot for CycleMetadata / Pulse.
    pub fn telemetry(&self) -> FabricationTelemetry {
        FabricationTelemetry {
            manufacturing_free_energy: self.last_manufacturing_fe,
            design_loop_free_energy: self.last_design_loop_fe,
            safety_level: format!("{:?}", self.safety_level),
            anomaly_count: self.anomaly_count_this_cycle,
            anomaly_ema: self.anomaly_ema,
            recommended_action: self.last_recommended_action.clone(),
            prediction_coherence: self.last_prediction_coherence,
            pog_score_ema: self.pog_score_ema,
            active_print_jobs: self.active_print_jobs,
            reward_ema: self.reward_ema as f32,
        }
    }

    /// Per-harmony deltas accumulated during processing.
    pub fn harmonic_deltas(&self) -> &[f64; 8] {
        &self.harmonic_deltas
    }

    /// Reset harmonic deltas (called after they are consumed).
    pub fn reset_harmonic_deltas(&mut self) {
        self.harmonic_deltas = [0.0; 8];
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTERNAL EVENT PROCESSING
// ═══════════════════════════════════════════════════════════════════════════════

impl FabricationManager {
    /// Process Cincinnati anomaly — NE burst for attentional reorientation.
    fn process_anomaly(&mut self, severity: f32, output: &mut SubsystemOutput) {
        // Guard against NaN/Inf
        let sev = if severity.is_finite() {
            severity.clamp(0.0, 1.0)
        } else {
            return;
        };

        // Update anomaly EMA
        self.anomaly_ema = self.anomaly_ema * (1.0 - thresholds::FAB_ANOMALY_EMA_ALPHA)
            + sev * thresholds::FAB_ANOMALY_EMA_ALPHA;
        self.anomaly_count_this_cycle += 1;
        self.total_anomaly_count += 1;

        // NE phasic burst on significant anomaly
        if sev > thresholds::FAB_ANOMALY_SEVERITY_THRESHOLD {
            let dose = thresholds::FAB_ANOMALY_NE_DOSE * sev;
            self.queue_injection("norepinephrine", dose, thresholds::FAB_ANOMALY_NE_HALFLIFE);

            // Negative reward signal
            self.update_reward(-sev as f64 * 0.5);
        }

        // Flag for ANOMALY_DETECTED on high severity
        if sev > thresholds::FAB_ANOMALY_FLAG_THRESHOLD {
            output.flags |= output_flags::ANOMALY_DETECTED;
        }
    }

    /// Process print job started.
    fn process_print_started(&mut self) {
        self.active_print_jobs = self.active_print_jobs.saturating_add(1);
    }

    /// Process print job completed — DA burst for reward, quality-based LR modulation.
    fn process_print_completed(
        &mut self,
        quality_score: f32,
        pog_score: f32,
        output: &mut SubsystemOutput,
    ) {
        self.active_print_jobs = self.active_print_jobs.saturating_sub(1);

        let quality = if quality_score.is_finite() {
            quality_score.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let pog = if pog_score.is_finite() {
            pog_score.clamp(0.0, 1.0)
        } else {
            0.5
        };

        // Update PoGF EMA
        self.pog_score_ema = self.pog_score_ema * (1.0 - thresholds::FAB_POG_EMA_ALPHA)
            + pog * thresholds::FAB_POG_EMA_ALPHA;

        // DA phasic burst — successful print = positive reward prediction
        if quality > thresholds::FAB_LOW_QUALITY_THRESHOLD {
            self.queue_injection(
                "dopamine",
                thresholds::FAB_PRINT_SUCCESS_DA_DOSE * quality,
                thresholds::FAB_PRINT_SUCCESS_DA_HALFLIFE,
            );
            output.confidence_delta += thresholds::FAB_PRINT_SUCCESS_CONFIDENCE;
            self.update_reward(quality as f64);
        } else {
            output.confidence_delta += thresholds::FAB_PRINT_FAILURE_CONFIDENCE;
            self.update_reward(-0.3);
        }

        // LR dampening on poor quality
        if quality < thresholds::FAB_LOW_QUALITY_THRESHOLD {
            output.lr_modulation *= thresholds::FAB_LOW_QUALITY_LR_DAMPEN;
        }

        // Exploration boost on very poor quality
        if quality < thresholds::FAB_POOR_QUALITY_THRESHOLD {
            output.exploration_delta += thresholds::FAB_POOR_QUALITY_EXPLORE_DELTA;
        }

        // Oxytocin boost on high PoGF (prosocial reward)
        if pog > thresholds::FAB_POG_HIGH_THRESHOLD {
            self.queue_injection(
                "oxytocin",
                thresholds::FAB_HIGH_POG_OXY_DOSE,
                thresholds::FAB_HIGH_POG_OXY_HALFLIFE,
            );
        }

        // Quality trend: improving PoGF → 5-HT baseline rise
        if self.pog_score_ema > thresholds::FAB_POG_HIGH_THRESHOLD {
            self.queue_baseline("serotonin", thresholds::FAB_QUALITY_TREND_SHT_NUDGE);
        }
    }

    /// Process safety level change.
    fn process_safety_change(
        &mut self,
        level: ManufacturingSafetyLevel,
        output: &mut SubsystemOutput,
    ) {
        self.safety_level = level;

        match level {
            ManufacturingSafetyLevel::Red => {
                // Emergency halt — NE surge + 5-HT dip (Sapolsky 2004)
                self.queue_injection(
                    "norepinephrine",
                    thresholds::FAB_EMERGENCY_NE_DOSE,
                    thresholds::FAB_EMERGENCY_NE_HALFLIFE,
                );
                self.queue_baseline("serotonin", thresholds::FAB_EMERGENCY_SHT_NUDGE);
                output.flags |= output_flags::VETO_ACTION;
                output.flags |= output_flags::ESCALATE_URGENCY;
                output.arousal_delta += thresholds::FAB_EMERGENCY_AROUSAL;
                output.confidence_delta += thresholds::FAB_PRINT_FAILURE_CONFIDENCE;
                self.update_reward(-1.0);
            }
            ManufacturingSafetyLevel::Orange => {
                output.flags |= output_flags::ESCALATE_URGENCY;
                output.arousal_delta += thresholds::FAB_ORANGE_AROUSAL;
                self.update_reward(-0.3);
            }
            ManufacturingSafetyLevel::Yellow => {
                // Mild concern, no urgency escalation
                self.update_reward(-0.1);
            }
            ManufacturingSafetyLevel::Green => {
                // All clear
            }
        }
    }

    /// Process ManufacturingTwin reading — step the twin and capture output.
    fn process_twin_reading(&mut self, reading: &ManufacturingReading) {
        let output = self.manufacturing_twin.step(reading, 0.05); // ~20Hz = 50ms
        self.last_manufacturing_fe = output.free_energy;
        self.last_recommended_action = format!("{:?}", output.recommended_action);

        // Compute mean prediction coherence
        if !output.prediction_similarities.is_empty() {
            let sum: f32 = output.prediction_similarities.iter().map(|(_, s)| *s).sum();
            self.last_prediction_coherence = sum / output.prediction_similarities.len() as f32;
        }

        // Safety level from twin output
        if output.safety_level != self.safety_level {
            self.safety_level = output.safety_level;
        }
    }

    /// Process DesignLoopTwin reading.
    fn process_design_loop_reading(&mut self, reading: &DesignLoopReading) {
        let output = self.design_loop_twin.step(reading, 0.05);
        self.last_design_loop_fe = output.free_energy;
    }

    /// Update the reward EMA.
    fn update_reward(&mut self, reward: f64) {
        let r = if reward.is_finite() {
            reward.clamp(-1.0, 1.0)
        } else {
            return;
        };
        self.reward_ema = self.reward_ema * thresholds::FAB_REWARD_EMA_DECAY
            + r * (1.0 - thresholds::FAB_REWARD_EMA_DECAY);
    }

    /// Queue a neuromod injection with floor check.
    fn queue_injection(&mut self, target: &'static str, dose: f32, half_life: u32) {
        if dose.abs() >= thresholds::FAB_NEUROMOD_FLOOR && dose.is_finite() {
            self.pending_injections.push(PendingInjection {
                target,
                dose,
                half_life,
            });
        }
    }

    /// Queue a baseline nudge with floor check.
    fn queue_baseline(&mut self, target: &'static str, nudge: f32) {
        if nudge.abs() >= thresholds::FAB_NEUROMOD_FLOOR && nudge.is_finite() {
            self.pending_baselines
                .push(PendingBaseline { target, nudge });
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COGNITIVE SUBSYSTEM TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

impl CognitiveSubsystem for FabricationManager {
    fn name(&self) -> &'static str {
        "fabrication_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        self.current_cycle = snapshot.cycle_number;
        self.anomaly_count_this_cycle = 0;

        // Drain pending events (capped per cycle)
        let events: Vec<FabricationEvent> =
            if self.pending_events.len() > Self::MAX_EVENTS_PER_CYCLE {
                let rest = self.pending_events.split_off(Self::MAX_EVENTS_PER_CYCLE);
                let batch = std::mem::replace(&mut self.pending_events, rest);
                batch
            } else {
                std::mem::take(&mut self.pending_events)
            };

        // Early return if nothing to do
        if events.is_empty() {
            if self.confidence_nudge_acc.abs() > 1e-10 {
                output.confidence_delta += self.confidence_nudge_acc;
                self.confidence_nudge_acc = 0.0;
            }
            return output;
        }

        // Process each event
        for event in &events {
            match &event.kind {
                FabricationEventKind::CincinnatiAnomaly { severity, .. } => {
                    self.process_anomaly(*severity, &mut output);
                }
                FabricationEventKind::PrintJobStarted { .. } => {
                    self.process_print_started();
                }
                FabricationEventKind::PrintJobCompleted {
                    quality_score,
                    pog_score,
                    ..
                } => {
                    self.process_print_completed(*quality_score, *pog_score, &mut output);
                }
                FabricationEventKind::SafetyLevelChanged { level } => {
                    self.process_safety_change(*level, &mut output);
                }
                FabricationEventKind::TwinReading { reading } => {
                    self.process_twin_reading(reading);
                }
                FabricationEventKind::DesignLoopReading { reading } => {
                    self.process_design_loop_reading(reading);
                }
            }
        }

        // Drain cross-coupling nudge
        if self.confidence_nudge_acc.abs() > 1e-10 {
            output.confidence_delta += self.confidence_nudge_acc;
            self.confidence_nudge_acc = 0.0;
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        // Layout: [reward_ema: f64(8)] [anomaly_ema: f32(4)] [pog_score_ema: f32(4)] = 16 bytes
        let mut data = Vec::with_capacity(16);
        data.extend_from_slice(&self.reward_ema.to_le_bytes());
        data.extend_from_slice(&self.anomaly_ema.to_le_bytes());
        data.extend_from_slice(&self.pog_score_ema.to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 16 {
            return Err(format!(
                "FabricationManager checkpoint too short: {} < 16",
                data.len()
            ));
        }
        self.reward_ema = f64::from_le_bytes(
            data[0..8]
                .try_into()
                .map_err(|_| "FabricationManager: corrupt checkpoint bytes [0..8]".to_string())?,
        );
        self.anomaly_ema = f32::from_le_bytes(
            data[8..12]
                .try_into()
                .map_err(|_| "FabricationManager: corrupt checkpoint bytes [8..12]".to_string())?,
        );
        self.pog_score_ema =
            f32::from_le_bytes(data[12..16].try_into().map_err(|_| {
                "FabricationManager: corrupt checkpoint bytes [12..16]".to_string()
            })?);
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn default_snapshot() -> CycleSnapshot {
        CycleSnapshot::default()
    }

    fn anomaly_event(severity: f32) -> FabricationEvent {
        FabricationEvent {
            kind: FabricationEventKind::CincinnatiAnomaly {
                anomaly_type: "layer_delamination".into(),
                severity,
                layer: 42,
            },
            timestamp_secs: 0,
        }
    }

    fn print_completed_event(quality: f32, pog: f32) -> FabricationEvent {
        FabricationEvent {
            kind: FabricationEventKind::PrintJobCompleted {
                job_id: "job-001".into(),
                quality_score: quality,
                pog_score: pog,
            },
            timestamp_secs: 0,
        }
    }

    fn safety_event(level: ManufacturingSafetyLevel) -> FabricationEvent {
        FabricationEvent {
            kind: FabricationEventKind::SafetyLevelChanged { level },
            timestamp_secs: 0,
        }
    }

    #[test]
    fn test_neutral_without_events() {
        let mut mgr = FabricationManager::default();
        let output = mgr.process(&default_snapshot());
        assert!(output.is_neutral());
    }

    #[test]
    fn test_name() {
        let mgr = FabricationManager::default();
        assert_eq!(mgr.name(), "fabrication_manager");
    }

    #[test]
    fn test_interval_is_coprime() {
        let mgr = FabricationManager::default();
        assert_eq!(mgr.interval(), FabricationManager::INTERVAL);
        for interval in [7u32, 11, 13, 19, 23, 29, 37, 41, 43, 53, 67] {
            assert_ne!(
                FabricationManager::INTERVAL % interval,
                0,
                "{} should be co-prime with {}",
                FabricationManager::INTERVAL,
                interval
            );
        }
    }

    #[test]
    fn test_anomaly_triggers_ne_burst() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(anomaly_event(0.8));
        let _output = mgr.process(&default_snapshot());

        let injections = mgr.drain_injections();
        assert!(
            injections.iter().any(|i| i.target == "norepinephrine"),
            "High-severity anomaly should trigger NE injection"
        );
    }

    #[test]
    fn test_anomaly_below_threshold_no_ne() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(anomaly_event(0.3));
        let _output = mgr.process(&default_snapshot());

        let injections = mgr.drain_injections();
        assert!(
            !injections.iter().any(|i| i.target == "norepinephrine"),
            "Low-severity anomaly should not trigger NE"
        );
    }

    #[test]
    fn test_anomaly_flag_on_high_severity() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(anomaly_event(0.9));
        let output = mgr.process(&default_snapshot());
        assert!(output.has_flag(output_flags::ANOMALY_DETECTED));
    }

    #[test]
    fn test_print_success_triggers_da() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(print_completed_event(0.9, 0.8));
        let output = mgr.process(&default_snapshot());

        let injections = mgr.drain_injections();
        assert!(
            injections.iter().any(|i| i.target == "dopamine"),
            "Successful print should trigger DA injection"
        );
        assert!(
            output.confidence_delta > 0.0,
            "Success should boost confidence"
        );
    }

    #[test]
    fn test_poor_quality_dampens_lr() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(print_completed_event(0.3, 0.2));
        let output = mgr.process(&default_snapshot());

        assert!(output.lr_modulation < 1.0, "Poor quality should dampen LR");
        assert!(
            output.exploration_delta > 0.0,
            "Very poor quality should boost exploration"
        );
    }

    #[test]
    fn test_high_pog_triggers_oxytocin() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(print_completed_event(0.9, 0.9));
        let _output = mgr.process(&default_snapshot());

        let injections = mgr.drain_injections();
        assert!(
            injections.iter().any(|i| i.target == "oxytocin"),
            "High PoGF should trigger oxytocin"
        );
    }

    #[test]
    fn test_safety_red_veto_and_escalate() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(safety_event(ManufacturingSafetyLevel::Red));
        let output = mgr.process(&default_snapshot());

        assert!(output.has_flag(output_flags::VETO_ACTION));
        assert!(output.has_flag(output_flags::ESCALATE_URGENCY));
        assert!(output.arousal_delta > 0.0);

        let injections = mgr.drain_injections();
        assert!(injections.iter().any(|i| i.target == "norepinephrine"));
        let baselines = mgr.drain_baselines();
        assert!(baselines
            .iter()
            .any(|b| b.target == "serotonin" && b.nudge < 0.0));
    }

    #[test]
    fn test_safety_orange_escalate_no_veto() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(safety_event(ManufacturingSafetyLevel::Orange));
        let output = mgr.process(&default_snapshot());

        assert!(output.has_flag(output_flags::ESCALATE_URGENCY));
        assert!(!output.has_flag(output_flags::VETO_ACTION));
    }

    #[test]
    fn test_print_job_tracking() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(FabricationEvent {
            kind: FabricationEventKind::PrintJobStarted {
                job_id: "j1".into(),
            },
            timestamp_secs: 0,
        });
        mgr.inject_event(FabricationEvent {
            kind: FabricationEventKind::PrintJobStarted {
                job_id: "j2".into(),
            },
            timestamp_secs: 0,
        });
        let _output = mgr.process(&default_snapshot());
        assert_eq!(mgr.active_print_jobs(), 2);

        mgr.inject_event(print_completed_event(0.8, 0.7));
        let _output = mgr.process(&default_snapshot());
        assert_eq!(mgr.active_print_jobs(), 1);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut mgr = FabricationManager::default();
        mgr.reward_ema = 0.42;
        mgr.anomaly_ema = 0.15;
        mgr.pog_score_ema = 0.88;

        let data = mgr.checkpoint();
        let mut mgr2 = FabricationManager::default();
        mgr2.restore(&data).unwrap();

        assert!((mgr2.reward_ema - 0.42).abs() < 1e-10);
        assert!((mgr2.anomaly_ema - 0.15).abs() < 1e-6);
        assert!((mgr2.pog_score_ema - 0.88).abs() < 1e-6);
    }

    #[test]
    fn test_confidence_nudge_clamped() {
        let mut mgr = FabricationManager::default();
        mgr.nudge_confidence(0.5); // Should clamp to 0.1
        mgr.nudge_confidence(0.5); // Should clamp total to 0.2
        mgr.nudge_confidence(0.5); // Still 0.2

        let output = mgr.process(&default_snapshot());
        assert!(
            output.confidence_delta <= 0.2 + 1e-10,
            "Confidence nudge should be clamped to 0.2"
        );
    }

    #[test]
    fn test_nan_guarded_anomaly() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(anomaly_event(f32::NAN));
        let output = mgr.process(&default_snapshot());
        // Should not crash, anomaly_ema should remain unchanged
        assert!(mgr.anomaly_ema.is_finite());
        assert!(output.confidence_delta.is_finite());
    }

    #[test]
    fn test_telemetry_snapshot() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(anomaly_event(0.6));
        let _output = mgr.process(&default_snapshot());

        let telem = mgr.telemetry();
        assert!(telem.anomaly_ema > 0.0);
        assert_eq!(telem.anomaly_count, 1);
        assert_eq!(telem.safety_level, "Green");
    }

    #[test]
    fn test_twin_reading_updates_fe() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(FabricationEvent {
            kind: FabricationEventKind::TwinReading {
                reading: ManufacturingReading {
                    tolerance: 0.8,
                    surface_quality: 0.7,
                    throughput: 0.9,
                    energy_cost: 0.3,
                },
            },
            timestamp_secs: 0,
        });
        let _output = mgr.process(&default_snapshot());
        // After stepping, manufacturing FE should be updated (may be 0.0 on first step)
        assert!(mgr.last_manufacturing_fe.is_finite());
    }
}
