// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Waste Manager — Circular Economy Cognitive Subsystem
//!
//! Consolidates waste classification, contamination detection, and circularity
//! tracking into a single [`CognitiveSubsystem`] that reads from an immutable
//! [`CycleSnapshot`] and produces [`SubsystemOutput`] proposals.
//!
//! ## Signals Modeled
//!
//! 1. **Classification confidence**: Waste stream identification quality → confidence
//!    (Ellen MacArthur Foundation 2015)
//! 2. **Contamination alerting**: Detected contaminants → arousal spike (Arnsten 2009)
//! 3. **Decomposition urgency**: Time-critical batch processing → exploration drive
//! 4. **Circularity health**: System-wide material circularity → valence (positive affect
//!    from healthy resource loops, McEwen 2007)
//! 5. **Safety escalation**: NE boost on Orange/Red safety (Sapolsky 2015)
//!
//! ## Design
//!
//! The manager maintains internal state (classification EMA, contamination severity,
//! waste totals) but does NOT mutate any `CognitiveLoopService` fields directly.
//! It proposes changes via `SubsystemOutput` which the `OutputCollector` integrates
//! via consensus averaging.

use super::super::subsystem_trait::{CognitiveSubsystem, CycleSnapshot, SubsystemOutput};

// ═══════════════════════════════════════════════════════════════════════════════
// WASTE EVENTS — injected from external circular economy layer
// ═══════════════════════════════════════════════════════════════════════════════

/// Events that the waste/circular economy layer injects into the manager.
#[derive(Debug, Clone)]
pub enum WasteEvent {
    /// A waste stream was classified by sensor or external system.
    ///
    /// Basis: Ellen MacArthur Foundation (2015) — accurate classification is
    /// the foundation of circular material flows.
    ClassificationReceived {
        /// Material category (e.g., "organic", "plastic", "metal", "ewaste").
        category: String,
        /// Classification confidence [0.0, 1.0].
        confidence: f32,
    },

    /// Contamination detected in a waste stream.
    ///
    /// Basis: Arnsten (2009) — acute stressor activates NE-mediated attention.
    ContaminationDetected {
        /// Severity of contamination [0.0, 1.0].
        severity: f32,
        /// Type of contaminant (e.g., "heavy_metal", "biohazard", "chemical").
        contaminant_type: String,
    },

    /// Progress update on a decomposition/processing batch.
    DecompositionUpdate {
        /// Unique batch identifier.
        batch_id: String,
        /// Completion percentage [0.0, 100.0].
        completion_pct: f32,
    },

    /// Report of material circularity potential and quantity.
    ///
    /// Basis: Ellen MacArthur Foundation (2015) — circularity potential
    /// measures how fully a material can re-enter productive cycles.
    CircularityReport {
        /// Material class (e.g., "aluminum", "PET", "compost").
        material_class: String,
        /// Circularity potential [0.0, 1.0] (1.0 = fully recyclable).
        circularity_potential: f32,
        /// Quantity in kilograms.
        quantity_kg: f32,
    },
}

// ═══════════════════════════════════════════════════════════════════════════════
// WASTE TELEMETRY
// ═══════════════════════════════════════════════════════════════════════════════

/// Waste telemetry snapshot for CycleMetadata.
#[derive(Debug, Clone, Default)]
pub struct WasteTelemetry {
    /// Classification confidence EMA [0.0, 1.0].
    pub classification_confidence: f32,
    /// Whether contamination is currently detected.
    pub contamination_detected: bool,
    /// Current safety level (0=Green, 1=Yellow, 2=Orange, 3=Red).
    pub safety_level: u8,
    /// Decomposition urgency [0.0, 1.0].
    pub decomposition_urgency: f32,
    /// Total waste events processed (monotonic counter).
    pub waste_events_processed: u32,
    /// Cumulative waste tracked (kg).
    pub total_waste_kg: f32,
    /// Running mean circularity [0.0, 1.0].
    pub mean_circularity: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// WASTE MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/// Waste Manager — circular economy signals → cognitive modulation.
///
/// Implements `CognitiveSubsystem` at interval 47 (co-prime with 7, 11, 13, 19, 37, 41, 43).
pub struct WasteManager {
    // ── Event queue ─────────────────────────────────────────────────────
    /// Pending events from the waste/circular economy layer (drained each process cycle).
    pending_events: Vec<WasteEvent>,

    // ── Classification state ────────────────────────────────────────────
    /// EMA of classification confidence [0.0, 1.0].
    classification_confidence: f32,

    // ── Contamination state ─────────────────────────────────────────────
    /// Whether contamination is currently detected.
    contamination_detected: bool,
    /// Severity of current contamination [0.0, 1.0].
    contamination_severity: f32,

    // ── Safety level ────────────────────────────────────────────────────
    /// 0=Green, 1=Yellow, 2=Orange, 3=Red.
    /// Escalates based on contamination severity.
    safety_level: u8,

    // ── Decomposition tracking ──────────────────────────────────────────
    /// Urgency of pending decomposition batches [0.0, 1.0].
    decomposition_urgency: f32,

    // ── Aggregate counters ──────────────────────────────────────────────
    /// Total waste events processed (monotonic).
    waste_events_processed: u32,
    /// Cumulative waste quantity tracked (kg).
    total_waste_kg: f32,
    /// Running mean circularity [0.0, 1.0].
    mean_circularity: f32,
    /// Number of circularity reports received (for running average).
    circularity_count: u32,

    // ── Telemetry ───────────────────────────────────────────────────────
    /// Latest telemetry snapshot.
    telemetry: WasteTelemetry,
}

impl Default for WasteManager {
    fn default() -> Self {
        Self {
            pending_events: Vec::new(),
            classification_confidence: 0.0,
            contamination_detected: false,
            contamination_severity: 0.0,
            safety_level: 0,
            decomposition_urgency: 0.0,
            waste_events_processed: 0,
            total_waste_kg: 0.0,
            mean_circularity: 0.0,
            circularity_count: 0,
            telemetry: WasteTelemetry::default(),
        }
    }
}

impl WasteManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 47;

    /// EMA alpha for classification confidence updates.
    /// Basis: Moderate smoothing to avoid oscillation from noisy classifiers.
    const CLASSIFICATION_EMA_ALPHA: f32 = 0.15;

    /// Confidence threshold above which classification boosts output confidence.
    /// Basis: Ellen MacArthur Foundation (2015) — reliable classification enables
    /// circular material routing.
    const CONFIDENCE_BOOST_THRESHOLD: f32 = 0.7;

    /// Confidence delta scale when classification confidence exceeds threshold.
    const CONFIDENCE_BOOST_SCALE: f64 = 0.15;

    /// Arousal delta scale for contamination severity.
    /// Basis: Arnsten (2009) — acute stressor activates NE-mediated arousal.
    const CONTAMINATION_AROUSAL_SCALE: f32 = 0.2;

    /// Decomposition urgency threshold for exploration drive.
    const DECOMPOSITION_URGENCY_THRESHOLD: f32 = 0.6;

    /// Exploration delta scale for decomposition urgency.
    const DECOMPOSITION_EXPLORATION_SCALE: f64 = 0.1;

    /// Circularity threshold for positive valence (system health signal).
    /// Basis: McEwen (2007) — positive feedback from homeostatic balance.
    const CIRCULARITY_VALENCE_THRESHOLD: f32 = 0.7;

    /// Valence boost scale for high circularity.
    const CIRCULARITY_VALENCE_SCALE: f32 = 0.05;

    /// NE arousal boost for Orange safety level.
    /// Basis: Sapolsky (2015) — moderate stress → NE-mediated vigilance.
    const SAFETY_ORANGE_AROUSAL: f32 = 0.15;

    /// NE arousal boost for Red safety level.
    /// Basis: Sapolsky (2015) — acute stress → full NE alarm response.
    const SAFETY_RED_AROUSAL: f32 = 0.25;

    /// DA suppression (negative valence) for Red safety level.
    const SAFETY_RED_VALENCE_SUPPRESS: f32 = -0.1;

    /// Contamination severity threshold for Yellow safety.
    const SAFETY_YELLOW_THRESHOLD: f32 = 0.3;

    /// Contamination severity threshold for Orange safety.
    const SAFETY_ORANGE_THRESHOLD: f32 = 0.6;

    /// Contamination severity threshold for Red safety.
    const SAFETY_RED_THRESHOLD: f32 = 0.85;

    /// Decomposition urgency decay per cycle (EMA toward 0).
    const DECOMPOSITION_DECAY: f32 = 0.95;

    /// Inject a waste event for processing in the next cycle.
    pub fn inject_event(&mut self, event: WasteEvent) {
        self.pending_events.push(event);
    }

    /// Current telemetry snapshot.
    pub fn telemetry(&self) -> &WasteTelemetry {
        &self.telemetry
    }

    /// Current classification confidence EMA.
    pub fn classification_confidence(&self) -> f32 {
        self.classification_confidence
    }

    /// Whether contamination is currently detected.
    pub fn contamination_detected(&self) -> bool {
        self.contamination_detected
    }

    /// Current contamination severity [0.0, 1.0].
    pub fn contamination_severity(&self) -> f32 {
        self.contamination_severity
    }

    /// Current safety level (0=Green, 1=Yellow, 2=Orange, 3=Red).
    pub fn safety_level(&self) -> u8 {
        self.safety_level
    }

    /// Decomposition urgency [0.0, 1.0].
    pub fn decomposition_urgency(&self) -> f32 {
        self.decomposition_urgency
    }

    /// Cumulative waste tracked (kg).
    pub fn total_waste_kg(&self) -> f32 {
        self.total_waste_kg
    }

    /// Running mean circularity [0.0, 1.0].
    pub fn mean_circularity(&self) -> f32 {
        self.mean_circularity
    }

    /// Number of waste events processed.
    pub fn waste_events_processed(&self) -> u32 {
        self.waste_events_processed
    }

    /// Drain and process all pending events, updating internal state.
    fn drain_events(&mut self) {
        let events: Vec<WasteEvent> = self.pending_events.drain(..).collect();

        for event in events {
            self.waste_events_processed = self.waste_events_processed.saturating_add(1);

            match event {
                WasteEvent::ClassificationReceived {
                    category: _,
                    confidence,
                } => {
                    let conf = Self::sanitize_f32(confidence).clamp(0.0, 1.0);
                    // EMA update: α * new + (1 - α) * old
                    self.classification_confidence = Self::CLASSIFICATION_EMA_ALPHA * conf
                        + (1.0 - Self::CLASSIFICATION_EMA_ALPHA) * self.classification_confidence;
                }

                WasteEvent::ContaminationDetected {
                    severity,
                    contaminant_type: _,
                } => {
                    let sev = Self::sanitize_f32(severity).clamp(0.0, 1.0);
                    self.contamination_detected = true;
                    // Take the max severity (worst case)
                    if sev > self.contamination_severity {
                        self.contamination_severity = sev;
                    }
                    self.update_safety_level();
                }

                WasteEvent::DecompositionUpdate {
                    batch_id: _,
                    completion_pct,
                } => {
                    let pct = Self::sanitize_f32(completion_pct).clamp(0.0, 100.0);
                    // Urgency inversely proportional to completion
                    // (incomplete batches are urgent)
                    let urgency = 1.0 - (pct / 100.0);
                    if urgency > self.decomposition_urgency {
                        self.decomposition_urgency = urgency;
                    }
                }

                WasteEvent::CircularityReport {
                    material_class: _,
                    circularity_potential,
                    quantity_kg,
                } => {
                    let circ = Self::sanitize_f32(circularity_potential).clamp(0.0, 1.0);
                    let kg = Self::sanitize_f32(quantity_kg).max(0.0);

                    self.total_waste_kg += kg;
                    self.circularity_count = self.circularity_count.saturating_add(1);

                    // Running average of circularity
                    let n = self.circularity_count as f32;
                    self.mean_circularity = self.mean_circularity * ((n - 1.0) / n) + circ / n;
                }
            }
        }
    }

    /// Update safety level based on contamination severity.
    fn update_safety_level(&mut self) {
        self.safety_level = if self.contamination_severity >= Self::SAFETY_RED_THRESHOLD {
            3 // Red
        } else if self.contamination_severity >= Self::SAFETY_ORANGE_THRESHOLD {
            2 // Orange
        } else if self.contamination_severity >= Self::SAFETY_YELLOW_THRESHOLD {
            1 // Yellow
        } else {
            0 // Green
        };
    }

    /// Update telemetry snapshot from current state.
    fn update_telemetry(&mut self) {
        self.telemetry = WasteTelemetry {
            classification_confidence: self.classification_confidence,
            contamination_detected: self.contamination_detected,
            safety_level: self.safety_level,
            decomposition_urgency: self.decomposition_urgency,
            waste_events_processed: self.waste_events_processed,
            total_waste_kg: self.total_waste_kg,
            mean_circularity: self.mean_circularity,
        };
    }

    /// Sanitize a float input — replace NaN/Inf with 0.0.
    #[inline]
    fn sanitize_f32(v: f32) -> f32 {
        if v.is_finite() { v } else { 0.0 }
    }
}

impl CognitiveSubsystem for WasteManager {
    fn name(&self) -> &'static str {
        "waste_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, _snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // ── 1. Drain and process events ─────────────────────────────────
        self.drain_events();

        // ── 2. Classification confidence → cognitive confidence ─────────
        // High classification confidence means the system can reliably
        // route waste streams → boost cognitive confidence.
        // Ellen MacArthur Foundation (2015): classification is the gateway
        // to circular material flows.
        if self.classification_confidence > Self::CONFIDENCE_BOOST_THRESHOLD {
            output.confidence_delta +=
                Self::CONFIDENCE_BOOST_SCALE * f64::from(self.classification_confidence);
        }

        // ── 3. Contamination → arousal spike ────────────────────────────
        // Contamination is an acute stressor requiring attention.
        // Arnsten (2009): moderate NE enhances PFC function; acute stress
        // shifts processing to habitual/reflexive pathways.
        if self.contamination_detected {
            output.arousal_delta += Self::CONTAMINATION_AROUSAL_SCALE * self.contamination_severity;
        }

        // ── 4. Safety level escalation → NE/DA modulation ──────────────
        // Sapolsky (2015): graduated stress response — moderate stress
        // enhances vigilance, acute stress suppresses reward seeking.
        match self.safety_level {
            2 => {
                // Orange — NE boost for vigilance
                output.arousal_delta += Self::SAFETY_ORANGE_AROUSAL;
            }
            3 => {
                // Red — NE boost + DA suppression (aversive)
                output.arousal_delta += Self::SAFETY_RED_AROUSAL;
                output.valence_delta += Self::SAFETY_RED_VALENCE_SUPPRESS;
            }
            _ => {}
        }

        // ── 5. Decomposition urgency → exploration drive ────────────────
        // Urgent batches need attention; exploration drive redirects
        // cognitive resources toward processing them.
        if self.decomposition_urgency > Self::DECOMPOSITION_URGENCY_THRESHOLD {
            output.exploration_delta += Self::DECOMPOSITION_EXPLORATION_SCALE
                * f64::from(self.decomposition_urgency - Self::DECOMPOSITION_URGENCY_THRESHOLD);
        }

        // Decay decomposition urgency over time (EMA toward 0)
        self.decomposition_urgency *= Self::DECOMPOSITION_DECAY;

        // ── 6. High circularity → positive valence ──────────────────────
        // System-level health signal: circular material flows are working.
        // McEwen (2007): allostatic balance → positive affect.
        if self.mean_circularity > Self::CIRCULARITY_VALENCE_THRESHOLD {
            output.valence_delta += Self::CIRCULARITY_VALENCE_SCALE
                * (self.mean_circularity - Self::CIRCULARITY_VALENCE_THRESHOLD);
        }

        // ── 7. Update telemetry ─────────────────────────────────────────
        self.update_telemetry();

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(32);
        data.extend_from_slice(&self.classification_confidence.to_le_bytes());
        data.push(self.contamination_detected as u8);
        data.extend_from_slice(&self.contamination_severity.to_le_bytes());
        data.push(self.safety_level);
        data.extend_from_slice(&self.decomposition_urgency.to_le_bytes());
        data.extend_from_slice(&self.waste_events_processed.to_le_bytes());
        data.extend_from_slice(&self.total_waste_kg.to_le_bytes());
        data.extend_from_slice(&self.mean_circularity.to_le_bytes());
        data.extend_from_slice(&self.circularity_count.to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 26 {
            return Err(format!(
                "WasteManager checkpoint too short: {} < 26",
                data.len()
            ));
        }
        self.classification_confidence = f32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| "WasteManager: corrupt bytes [0..4]".to_string())?,
        );
        self.contamination_detected = data[4] != 0;
        self.contamination_severity = f32::from_le_bytes(
            data[5..9]
                .try_into()
                .map_err(|_| "WasteManager: corrupt bytes [5..9]".to_string())?,
        );
        self.safety_level = data[9];
        self.decomposition_urgency = f32::from_le_bytes(
            data[10..14]
                .try_into()
                .map_err(|_| "WasteManager: corrupt bytes [10..14]".to_string())?,
        );
        self.waste_events_processed = u32::from_le_bytes(
            data[14..18]
                .try_into()
                .map_err(|_| "WasteManager: corrupt bytes [14..18]".to_string())?,
        );
        self.total_waste_kg = f32::from_le_bytes(
            data[18..22]
                .try_into()
                .map_err(|_| "WasteManager: corrupt bytes [18..22]".to_string())?,
        );
        self.mean_circularity = f32::from_le_bytes(
            data[22..26]
                .try_into()
                .map_err(|_| "WasteManager: corrupt bytes [22..26]".to_string())?,
        );
        if data.len() >= 30 {
            self.circularity_count = u32::from_le_bytes(
                data[26..30]
                    .try_into()
                    .map_err(|_| "WasteManager: corrupt bytes [26..30]".to_string())?,
            );
        }
        self.update_safety_level();
        self.update_telemetry();
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::subsystem_trait::CycleSnapshot;

    fn default_snapshot() -> CycleSnapshot {
        CycleSnapshot::default()
    }

    // ── 1. Default state ────────────────────────────────────────────────

    #[test]
    fn test_default_state() {
        let wm = WasteManager::default();
        assert_eq!(wm.classification_confidence, 0.0);
        assert!(!wm.contamination_detected);
        assert_eq!(wm.contamination_severity, 0.0);
        assert_eq!(wm.safety_level, 0); // Green
        assert_eq!(wm.decomposition_urgency, 0.0);
        assert_eq!(wm.waste_events_processed, 0);
        assert_eq!(wm.total_waste_kg, 0.0);
        assert_eq!(wm.mean_circularity, 0.0);
    }

    // ── 2. Name and interval ────────────────────────────────────────────

    #[test]
    fn test_name_and_interval() {
        let wm = WasteManager::default();
        assert_eq!(wm.name(), "waste_manager");
        assert_eq!(wm.interval(), 47);
    }

    // ── 3. Classification event updates confidence ──────────────────────

    #[test]
    fn test_classification_updates_confidence() {
        let mut wm = WasteManager::default();
        wm.inject_event(WasteEvent::ClassificationReceived {
            category: "plastic".to_string(),
            confidence: 0.9,
        });
        wm.process(&default_snapshot());

        // EMA: 0.15 * 0.9 + 0.85 * 0.0 = 0.135
        assert!(
            (wm.classification_confidence - 0.135).abs() < 1e-5,
            "Expected ~0.135, got {}",
            wm.classification_confidence
        );
    }

    // ── 4. Contamination event sets fields ──────────────────────────────

    #[test]
    fn test_contamination_event() {
        let mut wm = WasteManager::default();
        wm.inject_event(WasteEvent::ContaminationDetected {
            severity: 0.7,
            contaminant_type: "heavy_metal".to_string(),
        });
        wm.process(&default_snapshot());

        assert!(wm.contamination_detected);
        assert!((wm.contamination_severity - 0.7).abs() < 1e-5);
        assert_eq!(wm.safety_level, 2); // Orange (0.6 <= 0.7 < 0.85)
    }

    // ── 5. High confidence boosts output ────────────────────────────────

    #[test]
    fn test_high_confidence_boosts_output() {
        let mut wm = WasteManager::default();
        // Inject many events to bring EMA above threshold
        for _ in 0..50 {
            wm.inject_event(WasteEvent::ClassificationReceived {
                category: "metal".to_string(),
                confidence: 0.95,
            });
        }
        let output = wm.process(&default_snapshot());

        assert!(
            output.confidence_delta > 0.0,
            "High classification confidence should boost output confidence: {}",
            output.confidence_delta
        );
    }

    // ── 6. Contamination boosts arousal ─────────────────────────────────

    #[test]
    fn test_contamination_boosts_arousal() {
        let mut wm = WasteManager::default();
        wm.inject_event(WasteEvent::ContaminationDetected {
            severity: 0.8,
            contaminant_type: "biohazard".to_string(),
        });
        let output = wm.process(&default_snapshot());

        assert!(
            output.arousal_delta > 0.0,
            "Contamination should boost arousal: {}",
            output.arousal_delta
        );
    }

    // ── 7. Safety level escalation ──────────────────────────────────────

    #[test]
    fn test_safety_level_escalation() {
        let mut wm = WasteManager::default();

        // Green -> Yellow
        wm.inject_event(WasteEvent::ContaminationDetected {
            severity: 0.35,
            contaminant_type: "chemical".to_string(),
        });
        wm.process(&default_snapshot());
        assert_eq!(wm.safety_level, 1, "Expected Yellow (1)");

        // Yellow -> Red (max severity wins)
        wm.inject_event(WasteEvent::ContaminationDetected {
            severity: 0.9,
            contaminant_type: "biohazard".to_string(),
        });
        wm.process(&default_snapshot());
        assert_eq!(wm.safety_level, 3, "Expected Red (3)");
    }

    // ── 8. Orange safety → NE arousal boost ─────────────────────────────

    #[test]
    fn test_orange_safety_arousal_boost() {
        let mut wm = WasteManager::default();
        wm.inject_event(WasteEvent::ContaminationDetected {
            severity: 0.7,
            contaminant_type: "chemical".to_string(),
        });
        let output = wm.process(&default_snapshot());

        // Should have contamination arousal + orange safety arousal
        let expected_min = WasteManager::SAFETY_ORANGE_AROUSAL;
        assert!(
            output.arousal_delta >= expected_min,
            "Orange safety should add NE arousal: {} >= {}",
            output.arousal_delta,
            expected_min
        );
    }

    // ── 9. Red safety → NE boost + DA suppress ─────────────────────────

    #[test]
    fn test_red_safety_ne_and_da() {
        let mut wm = WasteManager::default();
        wm.inject_event(WasteEvent::ContaminationDetected {
            severity: 0.95,
            contaminant_type: "radioactive".to_string(),
        });
        let output = wm.process(&default_snapshot());

        assert!(
            output.arousal_delta >= WasteManager::SAFETY_RED_AROUSAL,
            "Red safety should boost NE arousal"
        );
        assert!(
            output.valence_delta < 0.0,
            "Red safety should suppress DA (negative valence): {}",
            output.valence_delta
        );
    }

    // ── 10. Mean circularity running average ────────────────────────────

    #[test]
    fn test_mean_circularity_running_average() {
        let mut wm = WasteManager::default();

        wm.inject_event(WasteEvent::CircularityReport {
            material_class: "aluminum".to_string(),
            circularity_potential: 0.9,
            quantity_kg: 10.0,
        });
        wm.inject_event(WasteEvent::CircularityReport {
            material_class: "PET".to_string(),
            circularity_potential: 0.5,
            quantity_kg: 20.0,
        });
        wm.process(&default_snapshot());

        // Average of 0.9 and 0.5 = 0.7
        assert!(
            (wm.mean_circularity - 0.7).abs() < 1e-5,
            "Expected mean circularity ~0.7, got {}",
            wm.mean_circularity
        );
    }

    // ── 11. Total waste kg accumulates ──────────────────────────────────

    #[test]
    fn test_total_waste_kg_accumulates() {
        let mut wm = WasteManager::default();

        wm.inject_event(WasteEvent::CircularityReport {
            material_class: "glass".to_string(),
            circularity_potential: 0.8,
            quantity_kg: 15.0,
        });
        wm.inject_event(WasteEvent::CircularityReport {
            material_class: "paper".to_string(),
            circularity_potential: 0.6,
            quantity_kg: 25.0,
        });
        wm.process(&default_snapshot());

        assert!(
            (wm.total_waste_kg - 40.0).abs() < 1e-5,
            "Expected 40.0 kg, got {}",
            wm.total_waste_kg
        );
    }

    // ── 12. Events drain after process ──────────────────────────────────

    #[test]
    fn test_events_drain_after_process() {
        let mut wm = WasteManager::default();
        wm.inject_event(WasteEvent::ClassificationReceived {
            category: "organic".to_string(),
            confidence: 0.8,
        });
        wm.inject_event(WasteEvent::ContaminationDetected {
            severity: 0.5,
            contaminant_type: "chemical".to_string(),
        });

        assert_eq!(wm.pending_events.len(), 2);
        wm.process(&default_snapshot());
        assert_eq!(wm.pending_events.len(), 0, "Events should be drained");
    }

    // ── 13. Multiple events aggregate ───────────────────────────────────

    #[test]
    fn test_multiple_events_aggregate() {
        let mut wm = WasteManager::default();

        wm.inject_event(WasteEvent::ClassificationReceived {
            category: "metal".to_string(),
            confidence: 0.8,
        });
        wm.inject_event(WasteEvent::ClassificationReceived {
            category: "plastic".to_string(),
            confidence: 0.9,
        });
        wm.inject_event(WasteEvent::ContaminationDetected {
            severity: 0.4,
            contaminant_type: "dust".to_string(),
        });
        wm.inject_event(WasteEvent::CircularityReport {
            material_class: "steel".to_string(),
            circularity_potential: 0.85,
            quantity_kg: 50.0,
        });

        let output = wm.process(&default_snapshot());

        assert_eq!(wm.waste_events_processed, 4);
        assert!(wm.contamination_detected);
        assert!((wm.total_waste_kg - 50.0).abs() < 1e-5);
        // Contamination detected → arousal should be positive
        assert!(output.arousal_delta > 0.0);
    }

    // ── 14. NaN/Inf inputs clamped ──────────────────────────────────────

    #[test]
    fn test_nan_inf_inputs_clamped() {
        let mut wm = WasteManager::default();

        wm.inject_event(WasteEvent::ClassificationReceived {
            category: "unknown".to_string(),
            confidence: f32::NAN,
        });
        wm.inject_event(WasteEvent::ContaminationDetected {
            severity: f32::INFINITY,
            contaminant_type: "test".to_string(),
        });
        wm.inject_event(WasteEvent::CircularityReport {
            material_class: "x".to_string(),
            circularity_potential: f32::NEG_INFINITY,
            quantity_kg: f32::NAN,
        });

        wm.process(&default_snapshot());

        assert!(
            wm.classification_confidence.is_finite(),
            "NaN confidence should be sanitized"
        );
        assert!(
            wm.contamination_severity.is_finite() && wm.contamination_severity <= 1.0,
            "Inf severity should be clamped: {}",
            wm.contamination_severity
        );
        assert!(
            wm.mean_circularity.is_finite(),
            "NaN circularity should be sanitized"
        );
        assert!(
            wm.total_waste_kg.is_finite() && wm.total_waste_kg >= 0.0,
            "NaN kg should be sanitized"
        );
    }

    // ── 15. Zero events → neutral output ────────────────────────────────

    #[test]
    fn test_zero_events_neutral_output() {
        let mut wm = WasteManager::default();
        let output = wm.process(&default_snapshot());

        assert!(
            output.is_neutral(),
            "No events should produce neutral output"
        );
    }

    // ── 16. Decomposition urgency → exploration ─────────────────────────

    #[test]
    fn test_decomposition_urgency_exploration() {
        let mut wm = WasteManager::default();
        // Batch at 10% completion → urgency = 0.9
        wm.inject_event(WasteEvent::DecompositionUpdate {
            batch_id: "batch-001".to_string(),
            completion_pct: 10.0,
        });
        let output = wm.process(&default_snapshot());

        assert!(
            output.exploration_delta > 0.0,
            "High decomposition urgency should drive exploration: {}",
            output.exploration_delta
        );
    }

    // ── 17. High circularity → positive valence ─────────────────────────

    #[test]
    fn test_high_circularity_positive_valence() {
        let mut wm = WasteManager::default();
        // Inject many high-circularity reports to bring mean above threshold
        for _ in 0..20 {
            wm.inject_event(WasteEvent::CircularityReport {
                material_class: "aluminum".to_string(),
                circularity_potential: 0.95,
                quantity_kg: 5.0,
            });
        }
        let output = wm.process(&default_snapshot());

        assert!(
            output.valence_delta > 0.0,
            "High circularity should boost valence: {}",
            output.valence_delta
        );
    }

    // ── 18. Checkpoint roundtrip ────────────────────────────────────────

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut wm = WasteManager::default();
        wm.inject_event(WasteEvent::ClassificationReceived {
            category: "metal".to_string(),
            confidence: 0.8,
        });
        wm.inject_event(WasteEvent::ContaminationDetected {
            severity: 0.5,
            contaminant_type: "chemical".to_string(),
        });
        wm.inject_event(WasteEvent::CircularityReport {
            material_class: "glass".to_string(),
            circularity_potential: 0.8,
            quantity_kg: 10.0,
        });
        wm.process(&default_snapshot());

        let data = wm.checkpoint();
        let mut wm2 = WasteManager::default();
        wm2.restore(&data).unwrap();

        assert!((wm2.classification_confidence - wm.classification_confidence).abs() < 1e-6);
        assert_eq!(wm2.contamination_detected, wm.contamination_detected);
        assert!((wm2.contamination_severity - wm.contamination_severity).abs() < 1e-6);
        assert_eq!(wm2.safety_level, wm.safety_level);
        assert!((wm2.total_waste_kg - wm.total_waste_kg).abs() < 1e-6);
        assert!((wm2.mean_circularity - wm.mean_circularity).abs() < 1e-6);
    }

    // ── 19. Telemetry updated after process ─────────────────────────────

    #[test]
    fn test_telemetry_updated() {
        let mut wm = WasteManager::default();
        wm.inject_event(WasteEvent::ContaminationDetected {
            severity: 0.7,
            contaminant_type: "lead".to_string(),
        });
        wm.process(&default_snapshot());

        let telem = wm.telemetry();
        assert!(telem.contamination_detected);
        assert_eq!(telem.safety_level, 2); // Orange
        assert_eq!(telem.waste_events_processed, 1);
    }

    // ── 20. Decomposition urgency decays ────────────────────────────────

    #[test]
    fn test_decomposition_urgency_decays() {
        let mut wm = WasteManager::default();
        wm.inject_event(WasteEvent::DecompositionUpdate {
            batch_id: "batch-001".to_string(),
            completion_pct: 10.0,
        });
        wm.process(&default_snapshot());
        let urgency_after_event = wm.decomposition_urgency;

        // Process again with no new events — urgency should decay
        wm.process(&default_snapshot());
        assert!(
            wm.decomposition_urgency < urgency_after_event,
            "Urgency should decay: {} < {}",
            wm.decomposition_urgency,
            urgency_after_event
        );
    }
}
