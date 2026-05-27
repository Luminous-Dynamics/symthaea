// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Civic Crisis Detector — Prediction Error → Community Emergency Pipeline
//!
//! Monitors the cognitive loop's state for sustained anomalies that indicate
//! a community-level crisis, then produces structured `CivicCrisisEvent`s
//! that can be forwarded to the Mycelix civic emergency zomes.
//!
//! ## Detection Signals
//!
//! | Signal | Source | Threshold | Meaning |
//! |--------|--------|-----------|---------|
//! | Prediction error | FEP/Active Inference | > 0.7 sustained 10+ cycles | World model failing |
//! | Safety level | SafetyEnforcement | Orange/Red | Direct threat detected |
//! | Collective Phi drop | SwarmManager | < 0.3 after being > 0.5 | Network consciousness collapse |
//! | Arousal spike | Neuromodulator bath | > 0.8 sustained 5+ cycles | High-alert community state |
//!
//! ## Output
//!
//! `CivicCrisisEvent` is forwarded to Mycelix `emergency-incidents` via the
//! Symthaea conductor adapter (`symthaea-mycelix-conductor`):
//! - `severity` → `SeverityLevel` (Level1-Level5, FEMA-aligned)
//! - `crisis_type` → `DisasterType` (Infrastructure, CyberAttack, etc.)
//! - `description` → human-readable narrative from Broca
//!
//! Note: the Mycelix `declare_disaster` schema includes geospatial fields.
//! Until Symthaea emits reliable geo coordinates, the conductor adapter
//! publishes a transparent placeholder affected_area ("global/unknown").
//!
//! ## Design
//!
//! Conservative by design — false positives waste community resources.
//! Uses EMA smoothing and sustained-threshold gates to filter transient spikes.
//! Cooldown period (100 cycles) prevents rapid-fire declarations.
//!
//! ## References
//!
//! - Friston (2010) — Free energy principle: sustained prediction error = model failure
//! - Tononi (2004) — Phi drop = consciousness fragmentation
//! - FEMA National Incident Management System (NIMS) — severity levels

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// CRISIS EVENTS
// ═══════════════════════════════════════════════════════════════════════════════

/// A civic crisis event produced when sustained anomalies are detected.
///
/// This event is expanded by the conductor adapter into Mycelix
/// `DeclareDisasterInput` for `emergency_incidents::declare_disaster`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivicCrisisEvent {
    /// Crisis severity (FEMA-aligned, 1 = lowest, 5 = highest).
    pub severity: u8,
    /// What type of crisis (maps to Mycelix DisasterType).
    pub crisis_type: CrisisType,
    /// Human-readable description of detected anomaly.
    pub description: String,
    /// Confidence in the crisis detection (0.0-1.0).
    pub confidence: f64,
    /// The signals that triggered this event.
    pub trigger_signals: Vec<CrisisSignal>,
    /// Cycle at which the crisis was detected.
    pub detected_at_cycle: u64,
}

/// Types of crises detectable by the cognitive loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrisisType {
    /// Infrastructure anomaly (sustained prediction error in world model).
    Infrastructure,
    /// Network threat (safety level Orange/Red from sentinel).
    CyberAttack,
    /// Consciousness collapse (Phi drop across collective).
    ConsciousnessFailure,
    /// Community stress (sustained high arousal across peers).
    CommunityStress,
    /// Unknown anomaly (multiple weak signals converging).
    Unknown,
}

impl std::fmt::Display for CrisisType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Infrastructure => f.write_str("Infrastructure"),
            Self::CyberAttack => f.write_str("CyberAttack"),
            Self::ConsciousnessFailure => f.write_str("ConsciousnessFailure"),
            Self::CommunityStress => f.write_str("CommunityStress"),
            Self::Unknown => f.write_str("Unknown"),
        }
    }
}

/// Individual signals that contribute to crisis detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisisSignal {
    /// Signal name for audit trail.
    pub name: String,
    /// Current value.
    pub value: f64,
    /// Threshold that was exceeded.
    pub threshold: f64,
    /// How many consecutive cycles the threshold was exceeded.
    pub sustained_cycles: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// DETECTOR CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Sustained prediction error threshold for crisis detection.
/// Friston (2010): PE > 0.7 indicates severe model-world divergence.
const CRISIS_PE_THRESHOLD: f64 = 0.7;

/// Minimum sustained cycles of high PE before crisis triggers.
const CRISIS_PE_SUSTAINED_CYCLES: u64 = 10;

/// Collective Phi drop threshold (from > 0.5 to < 0.3).
/// Tononi (2004): Phi collapse = fragmented consciousness.
const CRISIS_PHI_DROP_THRESHOLD: f64 = 0.3;

/// Phi level that must have been achieved before a drop counts.
const CRISIS_PHI_BASELINE: f64 = 0.5;

/// Sustained arousal threshold for community stress detection.
const CRISIS_AROUSAL_THRESHOLD: f64 = 0.8;

/// Minimum sustained arousal cycles for stress detection.
const CRISIS_AROUSAL_SUSTAINED_CYCLES: u64 = 5;

/// Cooldown period between crisis events (cycles).
/// Prevents rapid-fire declarations that waste community resources.
const CRISIS_COOLDOWN_CYCLES: u64 = 100;

/// EMA smoothing factor for signal tracking.
const CRISIS_EMA_ALPHA: f64 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// DETECTOR STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Input snapshot for crisis detection (extracted from CycleMetadata).
#[derive(Debug, Clone, Default)]
pub struct CrisisDetectorInput {
    /// Current prediction error (from FEP module).
    pub prediction_error: f64,
    /// Current safety level (0=Green, 1=Yellow, 2=Orange, 3=Red).
    pub safety_level_ordinal: u8,
    /// Current consciousness level (Phi).
    pub consciousness_level: f64,
    /// Collective Phi from swarm (0.0 if no peers).
    pub collective_phi: f64,
    /// Current arousal level from neuromodulator bath.
    pub arousal: f64,
    /// Whether peers are connected (enables collective signals).
    pub has_peers: bool,
}

/// Civic crisis detector — monitors cognitive loop for community emergencies.
pub struct CivicCrisisDetector {
    /// EMA of prediction error.
    pe_ema: f64,
    /// Consecutive cycles above PE threshold.
    pe_sustained: u64,
    /// EMA of arousal.
    arousal_ema: f64,
    /// Consecutive cycles above arousal threshold.
    arousal_sustained: u64,
    /// Highest observed Phi (for drop detection).
    phi_high_water: f64,
    /// Whether a Phi drop has been detected.
    phi_drop_detected: bool,
    /// Cycles since last crisis event (cooldown tracking).
    cycles_since_last_event: u64,
    /// Total events emitted.
    total_events: u64,
    /// Whether the detector is enabled.
    enabled: bool,
}

impl Default for CivicCrisisDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl CivicCrisisDetector {
    /// Create a new crisis detector.
    pub fn new() -> Self {
        Self {
            pe_ema: 0.0,
            pe_sustained: 0,
            arousal_ema: 0.5,
            arousal_sustained: 0,
            phi_high_water: 0.0,
            phi_drop_detected: false,
            cycles_since_last_event: CRISIS_COOLDOWN_CYCLES, // start ready
            total_events: 0,
            enabled: true,
        }
    }

    /// Enable or disable the detector.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Whether the detector is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Total crisis events emitted.
    pub fn total_events(&self) -> u64 {
        self.total_events
    }

    /// Process one cycle's input and optionally emit a crisis event.
    ///
    /// Returns `Some(CivicCrisisEvent)` if a crisis is detected.
    /// Returns `None` if no crisis or if in cooldown.
    pub fn tick(&mut self, input: &CrisisDetectorInput, cycle: u64) -> Option<CivicCrisisEvent> {
        if !self.enabled {
            return None;
        }

        self.cycles_since_last_event += 1;

        // Update EMAs
        self.pe_ema =
            self.pe_ema * (1.0 - CRISIS_EMA_ALPHA) + input.prediction_error * CRISIS_EMA_ALPHA;
        self.arousal_ema =
            self.arousal_ema * (1.0 - CRISIS_EMA_ALPHA) + input.arousal * CRISIS_EMA_ALPHA;

        // Track PE sustained
        if self.pe_ema > CRISIS_PE_THRESHOLD {
            self.pe_sustained += 1;
        } else {
            self.pe_sustained = 0;
        }

        // Track arousal sustained
        if self.arousal_ema > CRISIS_AROUSAL_THRESHOLD {
            self.arousal_sustained += 1;
        } else {
            self.arousal_sustained = 0;
        }

        // Track Phi high water and drop
        let phi = if input.has_peers {
            input.collective_phi
        } else {
            input.consciousness_level
        };
        if phi > self.phi_high_water {
            self.phi_high_water = phi;
        }
        self.phi_drop_detected =
            self.phi_high_water >= CRISIS_PHI_BASELINE && phi < CRISIS_PHI_DROP_THRESHOLD;

        // Cooldown check
        if self.cycles_since_last_event < CRISIS_COOLDOWN_CYCLES {
            return None;
        }

        // Check for crisis triggers
        let mut signals = Vec::new();

        // Signal 1: Sustained prediction error (world model failing)
        if self.pe_sustained >= CRISIS_PE_SUSTAINED_CYCLES {
            signals.push(CrisisSignal {
                name: "prediction_error".to_string(),
                value: self.pe_ema,
                threshold: CRISIS_PE_THRESHOLD,
                sustained_cycles: self.pe_sustained,
            });
        }

        // Signal 2: Safety level escalation (direct threat)
        if input.safety_level_ordinal
            >= crate::cognitive_loop::thresholds::CRISIS_SAFETY_ORANGE_ORDINAL
        {
            // Orange or Red
            signals.push(CrisisSignal {
                name: "safety_level".to_string(),
                value: input.safety_level_ordinal as f64,
                threshold: crate::cognitive_loop::thresholds::CRISIS_SAFETY_ORANGE_ORDINAL as f64,
                sustained_cycles: 1,
            });
        }

        // Signal 3: Collective Phi collapse
        if self.phi_drop_detected {
            signals.push(CrisisSignal {
                name: "phi_collapse".to_string(),
                value: phi,
                threshold: CRISIS_PHI_DROP_THRESHOLD,
                sustained_cycles: 1,
            });
        }

        // Signal 4: Sustained community stress
        if self.arousal_sustained >= CRISIS_AROUSAL_SUSTAINED_CYCLES {
            signals.push(CrisisSignal {
                name: "community_arousal".to_string(),
                value: self.arousal_ema,
                threshold: CRISIS_AROUSAL_THRESHOLD,
                sustained_cycles: self.arousal_sustained,
            });
        }

        // Need at least one signal to declare crisis
        if signals.is_empty() {
            return None;
        }

        // Determine crisis type and severity from signals
        let (crisis_type, severity) = self.classify(&signals, input);
        let confidence = self.compute_confidence(&signals);
        let description = self.describe(&signals, &crisis_type);

        self.cycles_since_last_event = 0;
        self.total_events += 1;

        // Reset sustained counters after event
        self.pe_sustained = 0;
        self.arousal_sustained = 0;
        self.phi_drop_detected = false;

        Some(CivicCrisisEvent {
            severity,
            crisis_type,
            description,
            confidence,
            trigger_signals: signals,
            detected_at_cycle: cycle,
        })
    }

    /// Classify crisis type and severity from active signals.
    fn classify(&self, signals: &[CrisisSignal], input: &CrisisDetectorInput) -> (CrisisType, u8) {
        let has_safety = signals.iter().any(|s| s.name == "safety_level");
        let has_pe = signals.iter().any(|s| s.name == "prediction_error");
        let has_phi = signals.iter().any(|s| s.name == "phi_collapse");
        let has_arousal = signals.iter().any(|s| s.name == "community_arousal");

        let crisis_type = if has_safety
            && input.safety_level_ordinal
                >= crate::cognitive_loop::thresholds::CRISIS_SAFETY_RED_ORDINAL
        {
            CrisisType::CyberAttack // Red safety = active threat
        } else if has_phi {
            CrisisType::ConsciousnessFailure
        } else if has_pe {
            CrisisType::Infrastructure
        } else if has_arousal {
            CrisisType::CommunityStress
        } else {
            CrisisType::Unknown
        };

        // Severity: 1 (single weak signal) to 5 (multiple strong signals + Red safety)
        let signal_count = signals.len() as u8;
        let safety_boost = if input.safety_level_ordinal
            >= crate::cognitive_loop::thresholds::CRISIS_SAFETY_RED_ORDINAL
        {
            crate::cognitive_loop::thresholds::CRISIS_SEVERITY_BOOST_RED
        } else if input.safety_level_ordinal
            >= crate::cognitive_loop::thresholds::CRISIS_SAFETY_ORANGE_ORDINAL
        {
            crate::cognitive_loop::thresholds::CRISIS_SEVERITY_BOOST_ORANGE
        } else {
            0
        };
        let severity = (signal_count + safety_boost).clamp(
            crate::cognitive_loop::thresholds::CRISIS_SEVERITY_MIN,
            crate::cognitive_loop::thresholds::CRISIS_SEVERITY_MAX,
        );

        (crisis_type, severity)
    }

    /// Compute confidence from signal strengths.
    fn compute_confidence(&self, signals: &[CrisisSignal]) -> f64 {
        if signals.is_empty() {
            return 0.0;
        }
        // Average of (value / threshold) across signals, clamped to 1.0
        let total: f64 = signals
            .iter()
            .map(|s| {
                (s.value
                    / s.threshold
                        .max(crate::cognitive_loop::thresholds::CRISIS_CONFIDENCE_MIN_DENOMINATOR))
                .min(crate::cognitive_loop::thresholds::CRISIS_CONFIDENCE_MAX_SIGNAL)
            })
            .sum();
        (total / signals.len() as f64).min(1.0)
    }

    /// Generate human-readable description.
    fn describe(&self, signals: &[CrisisSignal], crisis_type: &CrisisType) -> String {
        let type_str = match crisis_type {
            CrisisType::Infrastructure => "Infrastructure anomaly",
            CrisisType::CyberAttack => "Network threat",
            CrisisType::ConsciousnessFailure => "Consciousness collapse",
            CrisisType::CommunityStress => "Community stress",
            CrisisType::Unknown => "Unknown anomaly",
        };

        let signal_list: Vec<String> = signals
            .iter()
            .map(|s| {
                format!(
                    "{}={:.2} (>{:.2}, {}cyc)",
                    s.name, s.value, s.threshold, s.sustained_cycles
                )
            })
            .collect();

        format!("{}: {}", type_str, signal_list.join(", "))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn default_input() -> CrisisDetectorInput {
        CrisisDetectorInput {
            prediction_error: 0.1,
            safety_level_ordinal: 0, // Green
            consciousness_level: 0.6,
            collective_phi: 0.0,
            arousal: 0.5,
            has_peers: false,
        }
    }

    #[test]
    fn test_no_crisis_under_normal_conditions() {
        let mut detector = CivicCrisisDetector::new();
        let input = default_input();

        for cycle in 0..200 {
            assert!(detector.tick(&input, cycle).is_none());
        }
    }

    #[test]
    fn test_sustained_pe_triggers_crisis() {
        let mut detector = CivicCrisisDetector::new();
        let mut input = default_input();
        input.prediction_error = 0.9; // High PE

        // Feed high PE until EMA crosses threshold and sustains
        let mut event = None;
        for cycle in 0..200 {
            if let Some(e) = detector.tick(&input, cycle) {
                event = Some(e);
                break;
            }
        }

        let event = event.expect("Should detect crisis from sustained high PE");
        assert_eq!(event.crisis_type, CrisisType::Infrastructure);
        assert!(event.severity >= 1);
        assert!(event.confidence > 0.0);
        assert!(!event.trigger_signals.is_empty());
    }

    #[test]
    fn test_safety_red_triggers_immediate_crisis() {
        let mut detector = CivicCrisisDetector::new();
        let mut input = default_input();
        input.safety_level_ordinal = 3; // Red

        let event = detector.tick(&input, 0);
        let event = event.expect("Red safety should trigger immediate crisis");
        assert_eq!(event.crisis_type, CrisisType::CyberAttack);
        assert!(event.severity >= 3, "Red safety should be high severity");
    }

    #[test]
    fn test_cooldown_prevents_rapid_fire() {
        let mut detector = CivicCrisisDetector::new();
        let mut input = default_input();
        input.safety_level_ordinal = 3; // Red

        // First event triggers
        assert!(detector.tick(&input, 0).is_some());

        // Subsequent ticks during cooldown should not trigger
        for cycle in 1..CRISIS_COOLDOWN_CYCLES {
            assert!(detector.tick(&input, cycle).is_none());
        }

        // After cooldown, should trigger again
        assert!(detector.tick(&input, CRISIS_COOLDOWN_CYCLES).is_some());
        assert_eq!(detector.total_events(), 2);
    }

    #[test]
    fn test_phi_collapse_detected() {
        let mut detector = CivicCrisisDetector::new();
        let mut input = default_input();
        input.consciousness_level = 0.7;

        // Build up high-water Phi
        for cycle in 0..20 {
            detector.tick(&input, cycle);
        }

        // Drop Phi below threshold
        input.consciousness_level = 0.1;
        // Also need safety escalation or sustained PE for the event to fire
        // (Phi drop alone is a signal, but needs compound trigger)
        input.safety_level_ordinal = 2; // Orange

        let event = detector.tick(&input, 20);
        let event = event.expect("Phi collapse + safety escalation should trigger");
        assert!(
            event
                .trigger_signals
                .iter()
                .any(|s| s.name == "phi_collapse")
        );
    }

    #[test]
    fn test_disabled_detector_never_fires() {
        let mut detector = CivicCrisisDetector::new();
        detector.set_enabled(false);

        let mut input = default_input();
        input.safety_level_ordinal = 3; // Red

        for cycle in 0..200 {
            assert!(detector.tick(&input, cycle).is_none());
        }
    }

    #[test]
    fn test_sustained_arousal_triggers_stress() {
        let mut detector = CivicCrisisDetector::new();
        let mut input = default_input();
        input.arousal = 0.95; // Very high arousal

        let mut event = None;
        for cycle in 0..200 {
            if let Some(e) = detector.tick(&input, cycle) {
                event = Some(e);
                break;
            }
        }

        let event = event.expect("Should detect community stress from sustained high arousal");
        assert_eq!(event.crisis_type, CrisisType::CommunityStress);
    }

    #[test]
    fn test_severity_scales_with_signal_count() {
        let mut detector = CivicCrisisDetector::new();
        let mut input = default_input();

        // All signals active: high PE, Red safety, Phi drop, high arousal
        input.prediction_error = 0.95;
        input.safety_level_ordinal = 3;
        input.consciousness_level = 0.1;
        input.arousal = 0.95;

        // Build Phi high water first
        let mut warmup = default_input();
        warmup.consciousness_level = 0.7;
        for cycle in 0..20 {
            detector.tick(&warmup, cycle);
        }

        // Now trigger with all signals
        // Need to feed high values long enough for EMAs to cross thresholds
        for cycle in 20..100 {
            if let Some(event) = detector.tick(&input, cycle) {
                assert!(
                    event.severity >= 3,
                    "Multiple signals should produce high severity, got {}",
                    event.severity
                );
                assert!(
                    event.trigger_signals.len() >= 2,
                    "Should have multiple signals"
                );
                return;
            }
        }
        panic!("Should have triggered a multi-signal crisis");
    }

    #[test]
    fn test_description_is_readable() {
        let mut detector = CivicCrisisDetector::new();
        let mut input = default_input();
        input.safety_level_ordinal = 2; // Orange

        let event = detector.tick(&input, 0).expect("Should trigger");
        assert!(
            event.description.contains("Network threat") || event.description.contains("anomaly")
        );
        assert!(event.description.contains("safety_level"));
    }

    #[test]
    fn test_confidence_bounded() {
        let mut detector = CivicCrisisDetector::new();
        let mut input = default_input();
        input.safety_level_ordinal = 3;
        input.prediction_error = 10.0; // Extreme

        let event = detector.tick(&input, 0).unwrap();
        assert!(event.confidence >= 0.0 && event.confidence <= 1.0);
    }
}
