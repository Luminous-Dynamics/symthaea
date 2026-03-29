// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};

/// Self-reflection - meta-learning through introspection
///
/// Periodically analyzes the system's own performance and adjusts
/// internal thresholds to optimize behavior. This enables the system
/// to learn about itself and improve over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SelfReflection {
    // ===== Adaptive Thresholds =====
    /// Flow entry error threshold (adjusted based on flow frequency)
    pub flow_error_threshold: f32,

    /// Flow entry coherence threshold
    pub flow_coherence_threshold: f32,

    /// Boredom detection threshold (for curiosity drive)
    pub boredom_threshold: f32,

    /// Confidence threshold for trusting predictions
    pub trust_threshold: f32,

    // ===== Adaptive Integration Thresholds (Phase 11) =====
    /// Coherence gate threshold for resonator recall (default 0.3)
    pub coherence_gate_threshold: f32,

    /// Diversity lower bound for exploration governor (default 0.3)
    pub diversity_low_threshold: f32,

    /// Diversity upper bound for exploitation governor (default 0.7)
    pub diversity_high_threshold: f32,

    /// FEP surprise threshold for replay boost + exploration (default 0.8)
    pub surprise_threshold: f32,

    // ===== Meta-Statistics =====
    /// Total reflection cycles performed
    pub reflection_count: u64,

    /// Cycles since last reflection
    cycles_since_reflection: u32,

    /// Reflection interval (cycles between reflections)
    reflection_interval: u32,

    /// Historical flow entry rate (EMA)
    flow_entry_rate: f32,

    /// Historical exploration rate (EMA)
    exploration_rate: f32,

    /// Historical average error (EMA)
    historical_error: f32,

    /// Historical average confidence (EMA)
    historical_confidence: f32,

    /// Learning rate effectiveness score
    learning_effectiveness: f32,

    // ===== Adjustment History =====
    /// Number of threshold adjustments made
    pub adjustments_made: u32,

    /// Last adjustment direction for flow threshold (-1, 0, 1)
    last_flow_adjustment: i8,

    /// Last adjustment direction for boredom threshold
    last_boredom_adjustment: i8,

    // ===== Insights =====
    /// Current self-assessment
    pub self_assessment: SelfAssessment,

    /// Recommended actions based on reflection
    pub recommendations: Vec<Recommendation>,
}

/// Self-assessment of system state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelfAssessment {
    /// System is performing optimally
    Optimal,
    /// System is learning effectively
    Learning,
    /// System is stagnating (needs stimulation)
    Stagnating,
    /// System is struggling (high error, low confidence)
    Struggling,
    /// System is overconfident (low error but not learning)
    Overconfident,
    /// System is in exploration mode
    Exploring,
    /// System needs calibration
    NeedsCalibration,
}

impl SelfAssessment {
    /// Static string matching Debug output — avoids `format!("{:?}")` on hot path.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Optimal => "Optimal",
            Self::Learning => "Learning",
            Self::Stagnating => "Stagnating",
            Self::Struggling => "Struggling",
            Self::Overconfident => "Overconfident",
            Self::Exploring => "Exploring",
            Self::NeedsCalibration => "NeedsCalibration",
        }
    }
}

/// Recommendation from self-reflection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// What to adjust
    pub target: RecommendationTarget,
    /// Direction of adjustment
    pub direction: AdjustmentDirection,
    /// Confidence in this recommendation (0.0 to 1.0)
    pub confidence: f32,
    /// Reason for recommendation
    pub reason: String,
}

/// What the recommendation targets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationTarget {
    FlowThreshold,
    BoredomThreshold,
    TrustThreshold,
    LearningRate,
    ExplorationFactor,
    ReflectionInterval,
    CoherenceGateThreshold,
    SurpriseThreshold,
}

/// Direction of adjustment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdjustmentDirection {
    Increase,
    Decrease,
    NoChange,
}

impl Default for SelfReflection {
    fn default() -> Self {
        Self {
            // Initial thresholds (will be adjusted)
            flow_error_threshold: 0.25,
            flow_coherence_threshold: 0.6,
            boredom_threshold: 0.1,
            trust_threshold: 0.4,

            // Adaptive integration thresholds
            coherence_gate_threshold: 0.3,
            diversity_low_threshold: 0.3,
            diversity_high_threshold: 0.7,
            surprise_threshold: 0.8,

            // Meta-statistics
            reflection_count: 0,
            cycles_since_reflection: 0,
            reflection_interval: 50, // Reflect every 50 cycles
            flow_entry_rate: 0.0,
            exploration_rate: 0.0,
            historical_error: 0.5,
            historical_confidence: 0.5,
            learning_effectiveness: 0.5,

            // Adjustment tracking
            adjustments_made: 0,
            last_flow_adjustment: 0,
            last_boredom_adjustment: 0,

            // Initial state
            self_assessment: SelfAssessment::Learning,
            recommendations: Vec::new(),
        }
    }
}

impl SelfReflection {
    /// Minimum reflection interval
    const MIN_INTERVAL: u32 = 20;
    /// Maximum reflection interval
    const MAX_INTERVAL: u32 = 200;
    /// Threshold adjustment step size
    const ADJUSTMENT_STEP: f32 = 0.02;

    /// Record a cycle's metrics (called every cycle)
    pub fn record_cycle(
        &mut self,
        prediction_error: f32,
        in_flow: bool,
        exploring: bool,
        confidence: f32,
    ) {
        self.cycles_since_reflection += 1;

        // Update EMAs
        let alpha = 0.05;
        self.historical_error = self.historical_error * (1.0 - alpha) + prediction_error * alpha;
        self.historical_confidence =
            self.historical_confidence * (1.0 - alpha) + confidence * alpha;

        // Track flow and exploration rates
        let flow_val = if in_flow { 1.0 } else { 0.0 };
        let explore_val = if exploring { 1.0 } else { 0.0 };
        self.flow_entry_rate = self.flow_entry_rate * (1.0 - alpha) + flow_val * alpha;
        self.exploration_rate = self.exploration_rate * (1.0 - alpha) + explore_val * alpha;
    }

    /// Check if it's time to reflect
    pub fn should_reflect(&self) -> bool {
        self.cycles_since_reflection >= self.reflection_interval
    }

    /// Perform self-reflection and adjust thresholds
    pub fn reflect(&mut self) -> Vec<Recommendation> {
        self.reflection_count += 1;
        self.cycles_since_reflection = 0;
        self.recommendations.clear();

        // Analyze current state
        self.analyze_state();

        // Generate recommendations based on analysis
        self.generate_recommendations();

        // Apply automatic adjustments
        self.apply_adjustments();

        // Adjust reflection interval based on stability
        self.adjust_interval();

        self.recommendations.clone()
    }

    /// Analyze current system state
    fn analyze_state(&mut self) {
        // Determine self-assessment based on metrics
        self.self_assessment = if self.flow_entry_rate > 0.3 && self.historical_error < 0.2 {
            SelfAssessment::Optimal
        } else if self.exploration_rate > 0.3 {
            SelfAssessment::Exploring
        } else if self.historical_error < 0.15 && self.flow_entry_rate < 0.1 {
            SelfAssessment::Overconfident
        } else if self.historical_error > 0.5 && self.historical_confidence < 0.4 {
            SelfAssessment::Struggling
        } else if self.historical_error < 0.2 && self.exploration_rate < 0.05 {
            SelfAssessment::Stagnating
        } else if self.adjustments_made > 10 && self.flow_entry_rate < 0.05 {
            SelfAssessment::NeedsCalibration
        } else {
            SelfAssessment::Learning
        };

        // Update learning effectiveness
        // Good learning = moderate error (not too high, not too low) + improving trend
        let optimal_error = 0.25;
        let error_quality = 1.0 - (self.historical_error - optimal_error).abs() * 2.0;
        self.learning_effectiveness = error_quality.clamp(0.0, 1.0);
    }

    /// Generate recommendations based on current state
    fn generate_recommendations(&mut self) {
        match self.self_assessment {
            SelfAssessment::Stagnating => {
                // Lower boredom threshold to trigger exploration sooner
                self.recommendations.push(Recommendation {
                    target: RecommendationTarget::BoredomThreshold,
                    direction: AdjustmentDirection::Decrease,
                    confidence: 0.8,
                    reason:
                        "System is stagnating; lower boredom threshold to encourage exploration"
                            .into(),
                });
                // Increase exploration factor
                self.recommendations.push(Recommendation {
                    target: RecommendationTarget::ExplorationFactor,
                    direction: AdjustmentDirection::Increase,
                    confidence: 0.7,
                    reason: "Need more exploration to break stagnation".into(),
                });
            }
            SelfAssessment::Struggling => {
                // Raise trust threshold (be more cautious)
                self.recommendations.push(Recommendation {
                    target: RecommendationTarget::TrustThreshold,
                    direction: AdjustmentDirection::Increase,
                    confidence: 0.7,
                    reason: "High error rate; increase trust threshold for caution".into(),
                });
                // Lower learning rate if errors are very high
                if self.historical_error > 0.6 {
                    self.recommendations.push(Recommendation {
                        target: RecommendationTarget::LearningRate,
                        direction: AdjustmentDirection::Decrease,
                        confidence: 0.6,
                        reason: "Very high errors; reduce learning rate for stability".into(),
                    });
                }
            }
            SelfAssessment::Overconfident => {
                // Lower flow threshold (make flow harder to achieve)
                self.recommendations.push(Recommendation {
                    target: RecommendationTarget::FlowThreshold,
                    direction: AdjustmentDirection::Decrease,
                    confidence: 0.7,
                    reason: "Predictions too easy; tighten flow entry criteria".into(),
                });
                // Raise boredom threshold
                self.recommendations.push(Recommendation {
                    target: RecommendationTarget::BoredomThreshold,
                    direction: AdjustmentDirection::Decrease,
                    confidence: 0.6,
                    reason: "System may be bored but not detecting it".into(),
                });
            }
            SelfAssessment::NeedsCalibration => {
                // Reset to more moderate thresholds
                self.recommendations.push(Recommendation {
                    target: RecommendationTarget::FlowThreshold,
                    direction: AdjustmentDirection::NoChange,
                    confidence: 0.5,
                    reason: "Consider manual threshold review".into(),
                });
                // Extend reflection interval to allow stabilization
                self.recommendations.push(Recommendation {
                    target: RecommendationTarget::ReflectionInterval,
                    direction: AdjustmentDirection::Increase,
                    confidence: 0.8,
                    reason: "Allow more time between adjustments".into(),
                });
            }
            SelfAssessment::Optimal | SelfAssessment::Learning => {
                // Fine-tune based on specific metrics
                if self.flow_entry_rate < 0.1 && self.historical_error < 0.3 {
                    self.recommendations.push(Recommendation {
                        target: RecommendationTarget::FlowThreshold,
                        direction: AdjustmentDirection::Increase,
                        confidence: 0.5,
                        reason: "Good performance but rarely in flow; relax flow criteria".into(),
                    });
                }
            }
            SelfAssessment::Exploring => {
                // Don't interfere with active exploration
                // Maybe shorten reflection interval to monitor
                if self.exploration_rate > 0.5 {
                    self.recommendations.push(Recommendation {
                        target: RecommendationTarget::ReflectionInterval,
                        direction: AdjustmentDirection::Decrease,
                        confidence: 0.4,
                        reason: "High exploration; monitor more frequently".into(),
                    });
                }
            }
        }
    }

    /// Apply automatic threshold adjustments
    fn apply_adjustments(&mut self) {
        for rec in &self.recommendations {
            if rec.confidence < 0.5 {
                continue; // Skip low-confidence recommendations
            }

            match rec.target {
                RecommendationTarget::FlowThreshold => {
                    match rec.direction {
                        AdjustmentDirection::Increase => {
                            self.flow_error_threshold =
                                (self.flow_error_threshold + Self::ADJUSTMENT_STEP).min(0.4);
                            self.last_flow_adjustment = 1;
                        }
                        AdjustmentDirection::Decrease => {
                            self.flow_error_threshold =
                                (self.flow_error_threshold - Self::ADJUSTMENT_STEP).max(0.1);
                            self.last_flow_adjustment = -1;
                        }
                        AdjustmentDirection::NoChange => {
                            self.last_flow_adjustment = 0;
                        }
                    }
                    self.adjustments_made += 1;
                }
                RecommendationTarget::BoredomThreshold => {
                    match rec.direction {
                        AdjustmentDirection::Increase => {
                            self.boredom_threshold =
                                (self.boredom_threshold + Self::ADJUSTMENT_STEP).min(0.3);
                            self.last_boredom_adjustment = 1;
                        }
                        AdjustmentDirection::Decrease => {
                            self.boredom_threshold =
                                (self.boredom_threshold - Self::ADJUSTMENT_STEP).max(0.05);
                            self.last_boredom_adjustment = -1;
                        }
                        AdjustmentDirection::NoChange => {
                            self.last_boredom_adjustment = 0;
                        }
                    }
                    self.adjustments_made += 1;
                }
                RecommendationTarget::TrustThreshold => {
                    match rec.direction {
                        AdjustmentDirection::Increase => {
                            self.trust_threshold =
                                (self.trust_threshold + Self::ADJUSTMENT_STEP).min(0.7);
                        }
                        AdjustmentDirection::Decrease => {
                            self.trust_threshold =
                                (self.trust_threshold - Self::ADJUSTMENT_STEP).max(0.2);
                        }
                        AdjustmentDirection::NoChange => {}
                    }
                    self.adjustments_made += 1;
                }
                RecommendationTarget::CoherenceGateThreshold => {
                    match rec.direction {
                        AdjustmentDirection::Increase => {
                            self.coherence_gate_threshold =
                                (self.coherence_gate_threshold + Self::ADJUSTMENT_STEP).min(0.6);
                        }
                        AdjustmentDirection::Decrease => {
                            self.coherence_gate_threshold =
                                (self.coherence_gate_threshold - Self::ADJUSTMENT_STEP).max(0.1);
                        }
                        AdjustmentDirection::NoChange => {}
                    }
                    self.adjustments_made += 1;
                }
                RecommendationTarget::SurpriseThreshold => {
                    match rec.direction {
                        AdjustmentDirection::Increase => {
                            self.surprise_threshold =
                                (self.surprise_threshold + Self::ADJUSTMENT_STEP).min(0.95);
                        }
                        AdjustmentDirection::Decrease => {
                            self.surprise_threshold =
                                (self.surprise_threshold - Self::ADJUSTMENT_STEP).max(0.5);
                        }
                        AdjustmentDirection::NoChange => {}
                    }
                    self.adjustments_made += 1;
                }
                _ => {} // Other targets handled externally
            }
        }
    }

    /// Adjust reflection interval based on system stability
    fn adjust_interval(&mut self) {
        // If making many adjustments, reflect more often
        // If stable, reflect less often
        // Safe cast via f64 to prevent precision loss on large values
        let recent_adjustment_rate =
            (self.adjustments_made as f64 / self.reflection_count.max(1) as f64) as f32;

        if recent_adjustment_rate > 0.8 {
            // Lots of adjustments = unstable, reflect more
            self.reflection_interval = (self.reflection_interval - 5).max(Self::MIN_INTERVAL);
        } else if recent_adjustment_rate < 0.2 && self.self_assessment == SelfAssessment::Optimal {
            // Stable and optimal, reflect less
            self.reflection_interval = (self.reflection_interval + 10).min(Self::MAX_INTERVAL);
        }
    }

    /// Get current thresholds for use by other components
    pub fn get_thresholds(&self) -> ReflectionThresholds {
        ReflectionThresholds {
            flow_error: self.flow_error_threshold,
            flow_coherence: self.flow_coherence_threshold,
            boredom: self.boredom_threshold,
            trust: self.trust_threshold,
            coherence_gate: self.coherence_gate_threshold,
            diversity_low: self.diversity_low_threshold,
            diversity_high: self.diversity_high_threshold,
            surprise: self.surprise_threshold,
        }
    }

    /// Get a human-readable summary of current state
    pub fn summary(&self) -> ReflectionSummary {
        ReflectionSummary {
            reflection_count: self.reflection_count,
            adjustments_made: self.adjustments_made,
            learning_effectiveness: self.learning_effectiveness,
            next_reflection_in: self
                .reflection_interval
                .saturating_sub(self.cycles_since_reflection),
        }
    }

    /// Get learning effectiveness score
    pub fn learning_effectiveness(&self) -> f32 {
        self.learning_effectiveness
    }

    /// Reset self-reflection state
    pub fn reset(&mut self) {
        // Keep learned thresholds but reset statistics
        let thresholds = (
            self.flow_error_threshold,
            self.flow_coherence_threshold,
            self.boredom_threshold,
            self.trust_threshold,
        );
        *self = Self::default();
        self.flow_error_threshold = thresholds.0;
        self.flow_coherence_threshold = thresholds.1;
        self.boredom_threshold = thresholds.2;
        self.trust_threshold = thresholds.3;
    }

    /// Force an immediate reflection (triggered by motor commands)
    ///
    /// This bypasses the normal interval and triggers reflection now.
    pub fn force_reflection(&mut self) {
        // Set cycles to trigger immediate reflection
        self.cycles_since_reflection = self.reflection_interval;
    }
}

/// Thresholds from self-reflection for use by other components
#[derive(Debug, Clone, Copy)]
pub struct ReflectionThresholds {
    pub flow_error: f32,
    pub flow_coherence: f32,
    pub boredom: f32,
    pub trust: f32,
    pub coherence_gate: f32,
    pub diversity_low: f32,
    pub diversity_high: f32,
    pub surprise: f32,
}

/// Summary of self-reflection state
#[derive(Debug, Clone)]
pub struct ReflectionSummary {
    pub reflection_count: u64,
    pub adjustments_made: u32,
    pub learning_effectiveness: f32,
    pub next_reflection_in: u32,
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn reflection_default_state() {
        let sr = SelfReflection::default();
        assert_eq!(sr.self_assessment, SelfAssessment::Learning);
        assert_eq!(sr.reflection_count, 0);
        assert_eq!(sr.adjustments_made, 0);
    }

    #[test]
    fn reflection_should_reflect_at_interval() {
        let mut sr = SelfReflection::default();
        assert!(!sr.should_reflect());
        // Record enough cycles to reach interval
        for _ in 0..50 {
            sr.record_cycle(0.3, false, false, 0.5);
        }
        assert!(sr.should_reflect());
    }

    #[test]
    fn reflection_resets_counter_after_reflect() {
        let mut sr = SelfReflection::default();
        for _ in 0..50 {
            sr.record_cycle(0.3, false, false, 0.5);
        }
        sr.reflect();
        assert!(!sr.should_reflect());
        assert_eq!(sr.reflection_count, 1);
    }

    #[test]
    fn reflection_struggling_state() {
        let mut sr = SelfReflection::default();
        // High error + low confidence -> Struggling
        for _ in 0..50 {
            sr.record_cycle(0.7, false, false, 0.2);
        }
        let recs = sr.reflect();
        assert_eq!(sr.self_assessment, SelfAssessment::Struggling);
        assert!(
            !recs.is_empty(),
            "should produce recommendations when struggling"
        );
    }

    #[test]
    fn reflection_optimal_state() {
        let mut sr = SelfReflection::default();
        // Low error + high flow -> Optimal
        for _ in 0..50 {
            sr.record_cycle(0.1, true, false, 0.8);
        }
        sr.reflect();
        assert_eq!(sr.self_assessment, SelfAssessment::Optimal);
    }

    #[test]
    fn reflection_stagnating_state() {
        let mut sr = SelfReflection::default();
        // Need historical_error in [0.15, 0.2) and exploration_rate < 0.05
        // EMA alpha=0.05, starts at 0.5 -- needs ~120 cycles to converge
        // Error=0.17 converges to ~0.17 (between Overconfident < 0.15 and Stagnating < 0.2)
        for _ in 0..150 {
            sr.record_cycle(0.17, false, false, 0.8);
        }
        sr.reflect();
        assert_eq!(sr.self_assessment, SelfAssessment::Stagnating);
    }

    #[test]
    fn reflection_threshold_adjustments_bounded() {
        let mut sr = SelfReflection::default();
        // Run many reflection cycles to stress-test bounds
        for round in 0..20 {
            for _ in 0..50 {
                sr.record_cycle(if round % 2 == 0 { 0.1 } else { 0.7 }, false, false, 0.3);
            }
            sr.reflect();
        }
        assert!(
            sr.flow_error_threshold >= 0.1 && sr.flow_error_threshold <= 0.4,
            "flow threshold out of bounds: {}",
            sr.flow_error_threshold
        );
        assert!(
            sr.boredom_threshold >= 0.05 && sr.boredom_threshold <= 0.3,
            "boredom threshold out of bounds: {}",
            sr.boredom_threshold
        );
        assert!(
            sr.trust_threshold >= 0.2 && sr.trust_threshold <= 0.7,
            "trust threshold out of bounds: {}",
            sr.trust_threshold
        );
        assert!(
            sr.coherence_gate_threshold >= 0.1 && sr.coherence_gate_threshold <= 0.6,
            "coherence gate out of bounds: {}",
            sr.coherence_gate_threshold
        );
        assert!(
            sr.surprise_threshold >= 0.5 && sr.surprise_threshold <= 0.95,
            "surprise threshold out of bounds: {}",
            sr.surprise_threshold
        );
    }

    #[test]
    fn reflection_interval_adapts() {
        let mut sr = SelfReflection::default();
        let initial_interval = sr.reflection_interval;
        // Optimal state with few adjustments -> interval should grow
        for _ in 0..3 {
            for _ in 0..sr.reflection_interval {
                sr.record_cycle(0.1, true, false, 0.8);
            }
            sr.reflect();
        }
        assert!(
            sr.reflection_interval >= initial_interval,
            "interval should not shrink when optimal"
        );
    }

    #[test]
    fn reflection_force_triggers_immediate() {
        let mut sr = SelfReflection::default();
        sr.record_cycle(0.3, false, false, 0.5);
        assert!(!sr.should_reflect());
        sr.force_reflection();
        assert!(sr.should_reflect());
    }

    #[test]
    fn reflection_reset_preserves_thresholds() {
        let mut sr = SelfReflection::default();
        sr.flow_error_threshold = 0.35;
        sr.boredom_threshold = 0.15;
        sr.trust_threshold = 0.6;
        sr.reflection_count = 100;
        sr.reset();
        assert_eq!(sr.flow_error_threshold, 0.35);
        assert_eq!(sr.boredom_threshold, 0.15);
        assert_eq!(sr.trust_threshold, 0.6);
        assert_eq!(sr.reflection_count, 0);
    }

    #[test]
    fn reflection_get_thresholds() {
        let sr = SelfReflection::default();
        let t = sr.get_thresholds();
        assert_eq!(t.flow_error, sr.flow_error_threshold);
        assert_eq!(t.boredom, sr.boredom_threshold);
        assert_eq!(t.trust, sr.trust_threshold);
        assert_eq!(t.coherence_gate, sr.coherence_gate_threshold);
    }

    #[test]
    fn reflection_summary() {
        let sr = SelfReflection::default();
        let s = sr.summary();
        assert_eq!(s.reflection_count, 0);
        assert_eq!(s.adjustments_made, 0);
        assert_eq!(s.next_reflection_in, 50);
    }

    #[test]
    fn reflection_learning_effectiveness_bounded() {
        let mut sr = SelfReflection::default();
        // Various error levels
        for error in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            for _ in 0..50 {
                sr.record_cycle(error, false, false, 0.5);
            }
            sr.reflect();
            assert!(
                sr.learning_effectiveness >= 0.0 && sr.learning_effectiveness <= 1.0,
                "effectiveness out of bounds for error {}: {}",
                error,
                sr.learning_effectiveness
            );
        }
    }
}
