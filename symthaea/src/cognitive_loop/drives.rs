//! Cognitive drives - Emotion contagion, curiosity, and self-reflection
//!
//! These subsystems modulate the cognitive loop by providing emotional context,
//! novelty-seeking behavior, and meta-learning through introspection.

use serde::{Deserialize, Serialize};

use crate::dynamics::temporal_signatures::ConsciousnessPattern;

/// Emotion contagion - emotional content influences consciousness state
///
/// Detects emotional valence in input and nudges consciousness patterns:
/// - Positive emotions → Excited, Focused
/// - Negative emotions → Contemplative
/// - Neutral → no influence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EmotionContagion {
    /// Current emotional valence (-1.0 to 1.0)
    pub valence: f32,

    /// Arousal level (0.0 to 1.0)
    /// High arousal = excited/angry, Low arousal = calm/sad
    pub arousal: f32,

    /// Emotional influence strength (0.0 to 1.0)
    /// How much emotion affects consciousness pattern
    pub influence_strength: f32,

    /// Smoothed valence (EMA)
    smoothed_valence: f32,

    /// Smoothed arousal (EMA)
    smoothed_arousal: f32,
}

impl Default for EmotionContagion {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            influence_strength: 0.3,
            smoothed_valence: 0.0,
            smoothed_arousal: 0.5,
        }
    }
}

impl EmotionContagion {
    /// Positive emotion indicators
    const POSITIVE_WORDS: &'static [&'static str] = &[
        "happy",
        "joy",
        "love",
        "great",
        "wonderful",
        "excellent",
        "amazing",
        "beautiful",
        "fantastic",
        "good",
        "perfect",
        "brilliant",
        "awesome",
        "delighted",
        "excited",
        "pleased",
        "thrilled",
        "grateful",
        "hope",
        "success",
        "win",
        "celebrate",
        "smile",
        "laugh",
        "fun",
        "enjoy",
    ];

    /// Negative emotion indicators
    const NEGATIVE_WORDS: &'static [&'static str] = &[
        "sad",
        "angry",
        "fear",
        "hate",
        "terrible",
        "awful",
        "horrible",
        "bad",
        "wrong",
        "fail",
        "lost",
        "pain",
        "hurt",
        "worry",
        "anxious",
        "stressed",
        "frustrated",
        "disappointed",
        "regret",
        "sorry",
        "grief",
        "cry",
        "suffer",
        "struggle",
        "difficult",
        "problem",
        "error",
    ];

    /// High arousal indicators (excitement/intensity)
    const HIGH_AROUSAL: &'static [&'static str] = &[
        "!",
        "amazing",
        "incredible",
        "urgent",
        "now",
        "immediately",
        "excited",
        "thrilled",
        "furious",
        "terrified",
        "ecstatic",
    ];

    /// Analyze text for emotional content
    pub fn analyze(&mut self, text: &str) {
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        // Safe cast: use f64 intermediate to prevent precision loss on large word counts
        let word_count = (words.len().max(1) as f64) as f32;

        // Count emotional indicators (safe casts via f64)
        let positive_count = (Self::POSITIVE_WORDS
            .iter()
            .filter(|w| text_lower.contains(*w))
            .count() as f64) as f32;

        let negative_count = (Self::NEGATIVE_WORDS
            .iter()
            .filter(|w| text_lower.contains(*w))
            .count() as f64) as f32;

        let arousal_count = (Self::HIGH_AROUSAL
            .iter()
            .filter(|w| text_lower.contains(*w))
            .count() as f64) as f32;

        // Compute raw valence (-1 to 1)
        let total_emotional = positive_count + negative_count;
        let raw_valence = if total_emotional > 0.0 {
            (positive_count - negative_count) / total_emotional
        } else {
            0.0
        };

        // Compute intensity based on proportion of emotional words
        let emotional_density = total_emotional / word_count;
        let intensity = (emotional_density * 3.0).min(1.0); // Scale up, cap at 1

        // Compute arousal (base + exclamation points + high-arousal words)
        // Safe cast via f64 to handle large match counts
        let exclamation_boost = (text.matches('!').count() as f64 * 0.1) as f32;
        let raw_arousal = (0.5 + arousal_count * 0.1 + exclamation_boost).min(1.0);

        // Apply intensity to valence
        self.valence = raw_valence * intensity;

        // Update arousal
        self.arousal = raw_arousal;

        // Smooth with EMA
        let alpha = 0.3;
        self.smoothed_valence = self.smoothed_valence * (1.0 - alpha) + self.valence * alpha;
        self.smoothed_arousal = self.smoothed_arousal * (1.0 - alpha) + self.arousal * alpha;
    }

    /// Get suggested pattern nudge based on emotional state
    /// Returns (pattern_suggestion, strength) where strength is 0-1
    pub fn pattern_nudge(&self) -> (Option<ConsciousnessPattern>, f32) {
        let valence = self.smoothed_valence;
        let arousal = self.smoothed_arousal;

        // Only nudge if emotion is significant
        if valence.abs() < 0.2 {
            return (None, 0.0);
        }

        let strength = valence.abs() * self.influence_strength;

        let suggested_pattern = if valence > 0.3 && arousal > 0.6 {
            // High positive + high arousal → Excited
            Some(ConsciousnessPattern::Excited)
        } else if valence > 0.2 && arousal < 0.5 {
            // Positive + calm → Focused
            Some(ConsciousnessPattern::Focused)
        } else if valence < -0.3 {
            // Negative → Contemplative (processing/reflecting)
            Some(ConsciousnessPattern::Contemplative)
        } else if valence > 0.2 {
            // Mildly positive → Exploratory
            Some(ConsciousnessPattern::Exploratory)
        } else {
            None
        };

        (suggested_pattern, strength)
    }

    /// Get emotional valence for voice prosody
    pub fn prosody_valence(&self) -> f32 {
        self.smoothed_valence
    }

    /// Get arousal for voice prosody
    pub fn prosody_arousal(&self) -> f32 {
        self.smoothed_arousal
    }

    /// Get smoothed valence (EMA-filtered)
    pub fn smoothed_valence(&self) -> f32 {
        self.smoothed_valence
    }

    /// Get smoothed arousal (EMA-filtered)
    pub fn smoothed_arousal(&self) -> f32 {
        self.smoothed_arousal
    }

    /// Reset emotional state
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Curiosity drive - novelty-seeking mechanism to prevent stagnation
///
/// When predictions become too accurate/predictable, curiosity triggers
/// exploration mode to discover new patterns and prevent cognitive stagnation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CuriosityDrive {
    /// Recent prediction errors (rolling window).
    /// Capacity bound: HISTORY_SIZE (50) elements — evict before push via pop_front.
    error_history: std::collections::VecDeque<f32>,

    /// Boredom level (0.0 to 1.0)
    /// High when predictions are too accurate for too long
    pub boredom: f32,

    /// Curiosity level (0.0 to 1.0)
    /// High boredom triggers high curiosity
    pub curiosity: f32,

    /// Exploration urge (0.0 to 1.0)
    /// Direct measure of desire to explore new patterns
    pub exploration_urge: f32,

    /// Novelty bonus for learning rate
    /// Higher when exploring new territory
    pub novelty_bonus: f32,

    /// Consecutive cycles of low error (boredom buildup)
    low_error_streak: u32,

    /// Threshold below which error is "too good"
    boredom_threshold: f32,
}

impl Default for CuriosityDrive {
    fn default() -> Self {
        Self {
            error_history: std::collections::VecDeque::with_capacity(50),
            boredom: 0.0,
            curiosity: 0.3, // Start with some curiosity
            exploration_urge: 0.0,
            novelty_bonus: 1.0,
            low_error_streak: 0,
            boredom_threshold: 0.1,
        }
    }
}

impl CuriosityDrive {
    /// Window size for error history
    const HISTORY_SIZE: usize = 50;
    /// Boredom streak threshold
    const BOREDOM_STREAK: u32 = 10;
    /// Maximum novelty bonus
    const MAX_NOVELTY_BONUS: f32 = 1.5;

    /// Update curiosity drive based on prediction error
    pub fn update(&mut self, prediction_error: f32) {
        // Track error history — evict before push to prevent transient over-capacity
        if self.error_history.len() >= Self::HISTORY_SIZE {
            self.error_history.pop_front();
        }
        self.error_history.push_back(prediction_error);

        // Compute average error (safe division with max(1))
        let avg_error = if !self.error_history.is_empty() {
            self.error_history.iter().sum::<f32>() / self.error_history.len().max(1) as f32
        } else {
            0.5
        };

        // Detect boredom (consistently low error)
        if prediction_error < self.boredom_threshold {
            self.low_error_streak += 1;
        } else {
            self.low_error_streak = self.low_error_streak.saturating_sub(2);
        }

        // Boredom grows with low error streak (safe cast via f64)
        let streak_factor =
            ((self.low_error_streak as f64 / Self::BOREDOM_STREAK.max(1) as f64) as f32).min(1.0);
        self.boredom = 0.9 * self.boredom + 0.1 * streak_factor;

        // Curiosity is inverse of average error (interesting when things are predictable)
        let error_curiosity = (1.0 - avg_error.min(1.0)).max(0.0);
        self.curiosity = 0.8 * self.curiosity + 0.2 * error_curiosity;

        // Exploration urge triggered by high boredom + high curiosity
        if self.boredom > 0.5 && self.curiosity > 0.5 {
            self.exploration_urge = (self.boredom * self.curiosity).min(1.0);
        } else {
            self.exploration_urge *= 0.9; // Decay
        }

        // Novelty bonus: higher when exploring after boredom
        // This encourages the system to seek novel inputs
        if self.exploration_urge > 0.3 {
            self.novelty_bonus = 1.0 + (Self::MAX_NOVELTY_BONUS - 1.0) * self.exploration_urge;
        } else if prediction_error > 0.5 {
            // High error = novel situation, boost learning
            self.novelty_bonus = 1.0 + 0.3 * (prediction_error - 0.5);
        } else {
            self.novelty_bonus = 1.0;
        }
    }

    /// Check if exploration should be triggered
    pub fn should_explore(&self) -> bool {
        self.exploration_urge > 0.4 || (self.boredom > 0.7 && self.curiosity > 0.6)
    }

    /// Get effective learning rate with novelty bonus
    pub fn effective_learning_rate(&self, base_rate: f32) -> f32 {
        base_rate * self.novelty_bonus
    }

    /// Reset curiosity drive
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Set boredom threshold from self-reflection
    ///
    /// This allows the meta-learning system to adjust when boredom triggers.
    pub fn set_boredom_threshold(&mut self, threshold: f32) {
        self.boredom_threshold = threshold.clamp(0.05, 0.3);
    }

    /// Get current boredom threshold
    pub fn get_boredom_threshold(&self) -> f32 {
        self.boredom_threshold
    }
}

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

    // ═══════════════════════════════════════════════════════════════════════
    // EmotionContagion tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn emotion_default_neutral() {
        let ec = EmotionContagion::default();
        assert_eq!(ec.valence, 0.0);
        assert_eq!(ec.arousal, 0.5);
        assert_eq!(ec.smoothed_valence, 0.0);
        assert_eq!(ec.smoothed_arousal, 0.5);
    }

    #[test]
    fn emotion_positive_text_increases_valence() {
        let mut ec = EmotionContagion::default();
        ec.analyze("I am so happy and excited, this is wonderful!");
        assert!(
            ec.valence > 0.0,
            "valence should be positive: {}",
            ec.valence
        );
        assert!(ec.smoothed_valence > 0.0);
    }

    #[test]
    fn emotion_negative_text_decreases_valence() {
        let mut ec = EmotionContagion::default();
        ec.analyze("I am sad and angry, everything is terrible and awful");
        assert!(
            ec.valence < 0.0,
            "valence should be negative: {}",
            ec.valence
        );
        assert!(ec.smoothed_valence < 0.0);
    }

    #[test]
    fn emotion_neutral_text_near_zero() {
        let mut ec = EmotionContagion::default();
        ec.analyze("twelve purple chairs remain quite still");
        assert!(
            ec.valence.abs() < 0.01,
            "neutral text valence: {}",
            ec.valence
        );
    }

    #[test]
    fn emotion_exclamation_boosts_arousal() {
        let mut ec = EmotionContagion::default();
        ec.analyze("Now!!! Immediately!!!");
        assert!(
            ec.arousal > 0.5,
            "arousal should be elevated: {}",
            ec.arousal
        );
    }

    #[test]
    fn emotion_valence_bounded() {
        let mut ec = EmotionContagion::default();
        // Extreme positive
        ec.analyze("happy joy love great wonderful excellent amazing beautiful fantastic good perfect brilliant awesome");
        assert!(
            ec.valence >= -1.0 && ec.valence <= 1.0,
            "valence out of bounds: {}",
            ec.valence
        );
        // Extreme negative
        ec.analyze("sad angry fear hate terrible awful horrible bad wrong fail");
        assert!(ec.valence >= -1.0 && ec.valence <= 1.0);
    }

    #[test]
    fn emotion_smoothing_lags() {
        let mut ec = EmotionContagion::default();
        ec.analyze("I am incredibly happy and excited!");
        let raw = ec.valence;
        let smoothed = ec.smoothed_valence;
        // First analysis: smoothed should lag behind raw (EMA from 0)
        assert!(
            smoothed.abs() < raw.abs(),
            "smoothed {} should lag raw {}",
            smoothed,
            raw
        );
    }

    #[test]
    fn emotion_pattern_nudge_significant_positive() {
        let mut ec = EmotionContagion::default();
        // Build up enough smoothed valence
        for _ in 0..5 {
            ec.analyze("I am so happy and excited, wonderful amazing day!");
        }
        let (pattern, strength) = ec.pattern_nudge();
        assert!(pattern.is_some(), "should suggest a pattern");
        assert!(strength > 0.0, "strength should be positive");
    }

    #[test]
    fn emotion_pattern_nudge_weak_returns_none() {
        let ec = EmotionContagion::default();
        let (pattern, strength) = ec.pattern_nudge();
        assert!(
            pattern.is_none(),
            "neutral state should not suggest a pattern"
        );
        assert_eq!(strength, 0.0);
    }

    #[test]
    fn emotion_reset_restores_default() {
        let mut ec = EmotionContagion::default();
        ec.analyze("I am very happy!");
        ec.reset();
        assert_eq!(ec.valence, 0.0);
        assert_eq!(ec.smoothed_valence, 0.0);
    }

    #[test]
    fn emotion_prosody_accessors() {
        let mut ec = EmotionContagion::default();
        ec.analyze("happy wonderful great");
        assert_eq!(ec.prosody_valence(), ec.smoothed_valence);
        assert_eq!(ec.prosody_arousal(), ec.smoothed_arousal);
    }

    #[test]
    fn emotion_empty_input() {
        let mut ec = EmotionContagion::default();
        ec.analyze("");
        // Should not panic, valence stays near zero
        assert!(ec.valence.abs() < 0.01);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CuriosityDrive tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn curiosity_default_state() {
        let cd = CuriosityDrive::default();
        assert_eq!(cd.boredom, 0.0);
        assert!((cd.curiosity - 0.3).abs() < f32::EPSILON);
        assert_eq!(cd.exploration_urge, 0.0);
        assert_eq!(cd.novelty_bonus, 1.0);
    }

    #[test]
    fn curiosity_low_error_builds_boredom() {
        let mut cd = CuriosityDrive::default();
        // Feed many low-error updates
        for _ in 0..30 {
            cd.update(0.01);
        }
        assert!(cd.boredom > 0.3, "boredom should build: {}", cd.boredom);
    }

    #[test]
    fn curiosity_high_error_resets_streak() {
        let mut cd = CuriosityDrive::default();
        for _ in 0..20 {
            cd.update(0.01);
        }
        let boredom_before = cd.boredom;
        // High error should reduce low_error_streak
        for _ in 0..10 {
            cd.update(0.5);
        }
        assert!(
            cd.boredom < boredom_before + 0.1,
            "boredom should not keep growing after high error"
        );
    }

    #[test]
    fn curiosity_should_explore_after_boredom() {
        let mut cd = CuriosityDrive::default();
        // Build up enough boredom + curiosity
        for _ in 0..100 {
            cd.update(0.01); // Very low error → boredom + high curiosity
        }
        assert!(
            cd.should_explore(),
            "should want to explore after prolonged low error"
        );
    }

    #[test]
    fn curiosity_effective_lr_baseline() {
        let cd = CuriosityDrive::default();
        let lr = cd.effective_learning_rate(0.01);
        assert!(
            (lr - 0.01).abs() < 1e-6,
            "baseline novelty_bonus should be 1.0"
        );
    }

    #[test]
    fn curiosity_effective_lr_bounded() {
        let mut cd = CuriosityDrive::default();
        // Drive exploration to max
        for _ in 0..200 {
            cd.update(0.01);
        }
        let lr = cd.effective_learning_rate(1.0);
        assert!(
            lr <= CuriosityDrive::MAX_NOVELTY_BONUS,
            "LR bonus {} exceeds max {}",
            lr,
            CuriosityDrive::MAX_NOVELTY_BONUS
        );
        assert!(lr >= 1.0, "LR bonus should be >= 1.0");
    }

    #[test]
    fn curiosity_high_error_gives_novelty_bonus() {
        let mut cd = CuriosityDrive::default();
        cd.update(0.8); // High error = novel situation
        assert!(
            cd.novelty_bonus > 1.0,
            "high error should boost novelty: {}",
            cd.novelty_bonus
        );
    }

    #[test]
    fn curiosity_history_bounded() {
        let mut cd = CuriosityDrive::default();
        for i in 0..200 {
            cd.update(i as f32 * 0.005);
        }
        assert!(cd.error_history.len() <= CuriosityDrive::HISTORY_SIZE);
    }

    #[test]
    fn curiosity_set_boredom_threshold_clamped() {
        let mut cd = CuriosityDrive::default();
        cd.set_boredom_threshold(0.0);
        assert!((cd.get_boredom_threshold() - 0.05).abs() < f32::EPSILON);
        cd.set_boredom_threshold(1.0);
        assert!((cd.get_boredom_threshold() - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn curiosity_reset_restores_default() {
        let mut cd = CuriosityDrive::default();
        for _ in 0..50 {
            cd.update(0.01);
        }
        cd.reset();
        assert_eq!(cd.boredom, 0.0);
        assert_eq!(cd.exploration_urge, 0.0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SelfReflection tests
    // ═══════════════════════════════════════════════════════════════════════

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
        // High error + low confidence → Struggling
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
        // Low error + high flow → Optimal
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
        // EMA alpha=0.05, starts at 0.5 — needs ~120 cycles to converge
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
        // Optimal state with few adjustments → interval should grow
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
