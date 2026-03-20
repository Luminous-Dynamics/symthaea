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
pub struct PendingInjection {
    pub target: &'static str,
    pub dose: f32,
    pub half_life: u32,
}

/// A pending baseline nudge (target transmitter, delta).
#[derive(Debug, Clone)]
pub struct PendingBaseline {
    pub target: &'static str,
    pub nudge: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TELEMETRY
// ═══════════════════════════════════════════════════════════════════════════════

/// Fabrication telemetry snapshot for CycleMetadata / Pulse.
#[derive(Debug, Clone, Default)]
pub struct FabricationTelemetry {
    pub manufacturing_free_energy: f64,
    pub design_loop_free_energy: f64,
    pub safety_level: String,
    pub anomaly_count: u32,
    pub anomaly_ema: f32,
    pub recommended_action: String,
    pub prediction_coherence: f32,
    pub pog_score_ema: f32,
    pub active_print_jobs: u32,
    pub reward_ema: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// FABRICATION MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/// Cognitive subsystem that translates manufacturing events and digital twin
/// readings into embodied cognitive signals.
pub struct FabricationManager {
    manufacturing_twin: ManufacturingTwin,
    design_loop_twin: DesignLoopTwin,
    pending_events: Vec<FabricationEvent>,
    pending_injections: Vec<PendingInjection>,
    pending_baselines: Vec<PendingBaseline>,
    harmonic_deltas: [f64; 8],
    safety_level: ManufacturingSafetyLevel,
    anomaly_ema: f32,
    anomaly_count_this_cycle: u32,
    total_anomaly_count: u64,
    active_print_jobs: u32,
    pog_score_ema: f32,
    last_manufacturing_fe: f64,
    last_design_loop_fe: f64,
    last_recommended_action: String,
    last_prediction_coherence: f32,
    reward_ema: f64,
    current_cycle: u64,
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
    pub const INTERVAL: u32 = thresholds::FABRICATION_MANAGER_INTERVAL;
    const MAX_EVENTS_PER_CYCLE: usize = thresholds::FAB_MAX_EVENTS_PER_CYCLE;

    pub fn inject_event(&mut self, event: FabricationEvent) {
        self.pending_events.push(event);
    }

    pub fn drain_injections(&mut self) -> Vec<PendingInjection> {
        std::mem::take(&mut self.pending_injections)
    }

    pub fn drain_baselines(&mut self) -> Vec<PendingBaseline> {
        std::mem::take(&mut self.pending_baselines)
    }

    pub fn reward_ema(&self) -> f64 {
        self.reward_ema
    }

    pub fn safety_level(&self) -> ManufacturingSafetyLevel {
        self.safety_level
    }

    pub fn anomaly_ema(&self) -> f32 {
        self.anomaly_ema
    }

    pub fn pog_score_ema(&self) -> f32 {
        self.pog_score_ema
    }

    pub fn active_print_jobs(&self) -> u32 {
        self.active_print_jobs
    }

    pub fn nudge_confidence(&mut self, delta: f64) {
        if delta.is_finite() {
            self.confidence_nudge_acc =
                (self.confidence_nudge_acc + delta.clamp(-0.1, 0.1)).clamp(-0.2, 0.2);
        }
    }

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

    pub fn harmonic_deltas(&self) -> &[f64; 8] {
        &self.harmonic_deltas
    }

    pub fn reset_harmonic_deltas(&mut self) {
        self.harmonic_deltas = [0.0; 8];
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTERNAL EVENT PROCESSING
// ═══════════════════════════════════════════════════════════════════════════════

impl FabricationManager {
    fn process_anomaly(&mut self, severity: f32, output: &mut SubsystemOutput) {
        let sev = if severity.is_finite() {
            severity.clamp(0.0, 1.0)
        } else {
            return;
        };
        self.anomaly_ema = self.anomaly_ema * (1.0 - thresholds::FAB_ANOMALY_EMA_ALPHA)
            + sev * thresholds::FAB_ANOMALY_EMA_ALPHA;
        self.anomaly_count_this_cycle += 1;
        self.total_anomaly_count += 1;

        if sev > thresholds::FAB_ANOMALY_SEVERITY_THRESHOLD {
            let dose = thresholds::FAB_ANOMALY_NE_DOSE * sev;
            self.queue_injection("norepinephrine", dose, thresholds::FAB_ANOMALY_NE_HALFLIFE);
            self.update_reward(-sev as f64 * 0.5);
        }
        if sev > thresholds::FAB_ANOMALY_FLAG_THRESHOLD {
            output.flags |= output_flags::ANOMALY_DETECTED;
        }
    }

    fn process_print_started(&mut self) {
        self.active_print_jobs = self.active_print_jobs.saturating_add(1);
    }

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
        self.pog_score_ema = self.pog_score_ema * (1.0 - thresholds::FAB_POG_EMA_ALPHA)
            + pog * thresholds::FAB_POG_EMA_ALPHA;

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
        if quality < thresholds::FAB_LOW_QUALITY_THRESHOLD {
            output.lr_modulation *= thresholds::FAB_LOW_QUALITY_LR_DAMPEN;
        }
        if quality < thresholds::FAB_POOR_QUALITY_THRESHOLD {
            output.exploration_delta += thresholds::FAB_POOR_QUALITY_EXPLORE_DELTA;
        }
        if pog > thresholds::FAB_POG_HIGH_THRESHOLD {
            self.queue_injection(
                "oxytocin",
                thresholds::FAB_HIGH_POG_OXY_DOSE,
                thresholds::FAB_HIGH_POG_OXY_HALFLIFE,
            );
        }
        if self.pog_score_ema > thresholds::FAB_POG_HIGH_THRESHOLD {
            self.queue_baseline("serotonin", thresholds::FAB_QUALITY_TREND_SHT_NUDGE);
        }
    }

    fn process_safety_change(
        &mut self,
        level: ManufacturingSafetyLevel,
        output: &mut SubsystemOutput,
    ) {
        self.safety_level = level;
        match level {
            ManufacturingSafetyLevel::Red => {
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
                self.update_reward(-0.1);
            }
            ManufacturingSafetyLevel::Green => {}
        }
    }

    fn process_twin_reading(&mut self, reading: &ManufacturingReading) {
        let output = self.manufacturing_twin.step(reading, 0.05);
        self.last_manufacturing_fe = output.free_energy;
        self.last_recommended_action = format!("{:?}", output.recommended_action);
        if !output.prediction_similarities.is_empty() {
            let sum: f32 = output.prediction_similarities.iter().map(|(_, s)| *s).sum();
            self.last_prediction_coherence =
                sum / output.prediction_similarities.len() as f32;
        }
        if output.safety_level != self.safety_level {
            self.safety_level = output.safety_level;
        }
    }

    fn process_design_loop_reading(&mut self, reading: &DesignLoopReading) {
        let output = self.design_loop_twin.step(reading, 0.05);
        self.last_design_loop_fe = output.free_energy;
    }

    fn update_reward(&mut self, reward: f64) {
        let r = if reward.is_finite() {
            reward.clamp(-1.0, 1.0)
        } else {
            return;
        };
        self.reward_ema = self.reward_ema * thresholds::FAB_REWARD_EMA_DECAY
            + r * (1.0 - thresholds::FAB_REWARD_EMA_DECAY);
    }

    fn queue_injection(&mut self, target: &'static str, dose: f32, half_life: u32) {
        if dose.abs() >= thresholds::FAB_NEUROMOD_FLOOR && dose.is_finite() {
            self.pending_injections.push(PendingInjection {
                target,
                dose,
                half_life,
            });
        }
    }

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

        let events: Vec<FabricationEvent> =
            if self.pending_events.len() > Self::MAX_EVENTS_PER_CYCLE {
                let rest = self.pending_events.split_off(Self::MAX_EVENTS_PER_CYCLE);
                let batch = std::mem::replace(&mut self.pending_events, rest);
                batch
            } else {
                std::mem::take(&mut self.pending_events)
            };

        if events.is_empty() {
            if self.confidence_nudge_acc.abs() > 1e-10 {
                output.confidence_delta += self.confidence_nudge_acc;
                self.confidence_nudge_acc = 0.0;
            }
            return output;
        }

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

        if self.confidence_nudge_acc.abs() > 1e-10 {
            output.confidence_delta += self.confidence_nudge_acc;
            self.confidence_nudge_acc = 0.0;
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
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
        self.pog_score_ema = f32::from_le_bytes(
            data[12..16].try_into().map_err(|_| {
                "FabricationManager: corrupt checkpoint bytes [12..16]".to_string()
            })?,
        );
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
        assert!(injections.iter().any(|i| i.target == "norepinephrine"));
    }

    #[test]
    fn test_anomaly_below_threshold_no_ne() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(anomaly_event(0.3));
        let _output = mgr.process(&default_snapshot());
        let injections = mgr.drain_injections();
        assert!(!injections.iter().any(|i| i.target == "norepinephrine"));
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
        assert!(injections.iter().any(|i| i.target == "dopamine"));
        assert!(output.confidence_delta > 0.0);
    }

    #[test]
    fn test_poor_quality_dampens_lr() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(print_completed_event(0.3, 0.2));
        let output = mgr.process(&default_snapshot());
        assert!(output.lr_modulation < 1.0);
        assert!(output.exploration_delta > 0.0);
    }

    #[test]
    fn test_high_pog_triggers_oxytocin() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(print_completed_event(0.9, 0.9));
        let _output = mgr.process(&default_snapshot());
        let injections = mgr.drain_injections();
        assert!(injections.iter().any(|i| i.target == "oxytocin"));
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
        mgr.nudge_confidence(0.5);
        mgr.nudge_confidence(0.5);
        mgr.nudge_confidence(0.5);
        let output = mgr.process(&default_snapshot());
        assert!(output.confidence_delta <= 0.2 + 1e-10);
    }

    #[test]
    fn test_nan_guarded_anomaly() {
        let mut mgr = FabricationManager::default();
        mgr.inject_event(anomaly_event(f32::NAN));
        let output = mgr.process(&default_snapshot());
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
        assert!(mgr.last_manufacturing_fe.is_finite());
    }
}
