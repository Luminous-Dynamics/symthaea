//! Thalamic routing and active inference bridge
//!
//! Determines cognitive depth (Reflex/Cortical/DeepThought) based on input
//! characteristics, and tracks prediction-outcome coupling quality.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::dynamics::temporal_signatures::ConsciousnessPattern;

use super::thresholds::{
    THALAMIC_COMPLEXITY_CORTICAL, THALAMIC_CORTICAL_BASE_RATE, THALAMIC_EMOTIONAL_BOOST_BASE,
    THALAMIC_EMOTIONAL_BOOST_SCALE, THALAMIC_EMOTIONAL_DAMPENING, THALAMIC_FACTOR_FLOOR,
    THALAMIC_FAMILIARITY_THRESHOLD, THALAMIC_INPUT_OFFSET, THALAMIC_NOVELTY_THRESHOLD,
    THALAMIC_URGENCY_THRESHOLD,
};

// ═══════════════════════════════════════════════════════════════════════════════
// CODE TASK DETECTION - Identifies code-related inputs
// ═══════════════════════════════════════════════════════════════════════════════

/// Type of code task detected from natural language input
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeTaskType {
    /// Write new code from scratch
    Create,
    /// Fix or debug existing code
    Debug,
    /// Improve existing code structure
    Refactor,
    /// Explain code concepts
    Explain,
    /// Not a code task
    None,
}

/// Detects whether natural language input is a code-related task
///
/// Uses keyword-based heuristics to classify input text as code tasks
/// with a confidence score.
#[derive(Debug, Clone)]
pub struct CodeTaskDetector {
    /// Keywords that strongly indicate code tasks
    code_keywords: Vec<&'static str>,
    /// Keywords that indicate debugging
    debug_keywords: Vec<&'static str>,
    /// Keywords that indicate refactoring
    refactor_keywords: Vec<&'static str>,
}

impl CodeTaskDetector {
    /// Create a new code task detector
    pub fn new() -> Self {
        Self {
            code_keywords: vec![
                "function",
                "code",
                "implement",
                "write",
                "create",
                "class",
                "struct",
                "module",
                "program",
                "script",
                "algorithm",
                "compile",
                "rust",
                "python",
                "javascript",
                "typescript",
                "java",
                "```",
                "fn ",
                "def ",
                "import",
                "crate",
            ],
            debug_keywords: vec![
                "debug", "fix", "bug", "error", "crash", "panic", "fail", "broken", "issue",
            ],
            refactor_keywords: vec![
                "refactor",
                "improve",
                "optimize",
                "clean",
                "restructure",
                "simplify",
                "efficient",
            ],
        }
    }

    /// Detect whether input text is a code task
    ///
    /// Returns `(is_code_task, confidence)` where confidence is 0.0 to 1.0
    pub fn detect(&self, input: &str) -> (bool, f32) {
        let lower = input.to_lowercase();
        let mut score = 0.0f32;

        for &kw in &self.code_keywords {
            if lower.contains(kw) {
                score += super::thresholds::CODE_TASK_KEYWORD_WEIGHT;
            }
        }
        for &kw in &self.debug_keywords {
            if lower.contains(kw) {
                score += super::thresholds::CODE_TASK_DEBUG_WEIGHT;
            }
        }
        for &kw in &self.refactor_keywords {
            if lower.contains(kw) {
                score += super::thresholds::CODE_TASK_REFACTOR_WEIGHT;
            }
        }

        let confidence = score.min(1.0);
        (
            confidence >= super::thresholds::CODE_TASK_CONFIDENCE_THRESHOLD,
            confidence,
        )
    }

    /// Detect the specific type of code task
    pub fn detect_task_type(&self, input: &str) -> CodeTaskType {
        let lower = input.to_lowercase();

        if self.debug_keywords.iter().any(|&kw| lower.contains(kw)) {
            return CodeTaskType::Debug;
        }
        if self.refactor_keywords.iter().any(|&kw| lower.contains(kw)) {
            return CodeTaskType::Refactor;
        }

        let has_code_kw = self.code_keywords.iter().any(|&kw| lower.contains(kw));

        if lower.contains("explain") || lower.contains("how does") || lower.contains("what is") {
            if has_code_kw {
                return CodeTaskType::Explain;
            }
            return CodeTaskType::None;
        }

        if has_code_kw {
            return CodeTaskType::Create;
        }

        CodeTaskType::None
    }
}

impl Default for CodeTaskDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// THALAMIC ROUTING - Cognitive Depth Selection
// ═══════════════════════════════════════════════════════════════════════════════

/// Cognitive depth determines how much processing to apply
///
/// Based on the Thalamus architecture (ARCHITECTURAL_EVOLUTION_SUMMARY.md):
/// - Reflex: Pattern matching only, <10ms response
/// - Cortical: Standard cognitive cycle, 50-200ms
/// - DeepThought: Full deliberation with causal reasoning, 200ms+
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CognitiveDepth {
    /// Fast pattern matching, minimal processing
    /// Used for: Familiar inputs, low novelty, low urgency
    Reflex,

    /// Standard cognitive cycle with prediction and learning
    /// Used for: Normal conversation, moderate complexity
    #[default]
    Cortical,

    /// Deep deliberation with causal reasoning and counterfactuals
    /// Used for: Novel situations, high stakes, complex reasoning
    DeepThought,
}

impl CognitiveDepth {
    /// Static string matching Debug output — avoids `format!("{:?}")` on hot path.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reflex => "Reflex",
            Self::Cortical => "Cortical",
            Self::DeepThought => "DeepThought",
        }
    }
}

/// Thalamic router - determines cognitive depth before processing
///
/// Implements the 3-path routing from Architecture V2:
/// - High novelty/urgency → DeepThought
/// - Normal → Cortical
/// - Familiar/low stakes → Reflex
#[derive(Debug, Clone)]
pub struct ThalamicRouter {
    /// Novelty threshold for DeepThought (0.0-1.0)
    pub novelty_threshold: f32,

    /// Urgency threshold for DeepThought (0.0-1.0)
    pub urgency_threshold: f32,

    /// Familiarity threshold for Reflex (0.0-1.0)
    pub familiarity_threshold: f32,

    /// Recent routing decisions for pattern analysis
    routing_history: VecDeque<CognitiveDepth>,

    /// Maximum history size
    max_history: usize,
}

impl Default for ThalamicRouter {
    fn default() -> Self {
        Self {
            novelty_threshold: THALAMIC_NOVELTY_THRESHOLD,
            urgency_threshold: THALAMIC_URGENCY_THRESHOLD,
            familiarity_threshold: THALAMIC_FAMILIARITY_THRESHOLD,
            routing_history: VecDeque::with_capacity(100),
            max_history: 100,
        }
    }
}

impl ThalamicRouter {
    /// Create a new ThalamicRouter with default thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Route based on input characteristics
    ///
    /// # Arguments
    /// * `novelty` - How novel/surprising the input is (0.0-1.0)
    /// * `urgency` - How urgent the response needs to be (0.0-1.0)
    /// * `complexity` - Estimated complexity of the input (0.0-1.0)
    /// * `emotional_intensity` - Emotional intensity of input (0.0-1.0)
    pub fn route(
        &mut self,
        novelty: f32,
        urgency: f32,
        complexity: f32,
        emotional_intensity: f32,
    ) -> CognitiveDepth {
        let depth = if novelty > self.novelty_threshold
            || urgency > self.urgency_threshold
            || complexity > super::thresholds::THALAMIC_COMPLEXITY_DEEP_THRESHOLD
            || emotional_intensity > super::thresholds::THALAMIC_EMOTIONAL_DEEP_THRESHOLD
        {
            // High stakes - use deep thought
            CognitiveDepth::DeepThought
        } else if novelty < self.familiarity_threshold
            && complexity < super::thresholds::THALAMIC_COMPLEXITY_REFLEX_THRESHOLD
            && urgency < super::thresholds::THALAMIC_URGENCY_REFLEX_THRESHOLD
        {
            // Familiar, simple, not urgent - use reflex
            CognitiveDepth::Reflex
        } else {
            // Default to standard cortical processing
            CognitiveDepth::Cortical
        };

        // Record history
        if self.routing_history.len() >= self.max_history {
            self.routing_history.pop_front();
        }
        self.routing_history.push_back(depth);

        depth
    }

    /// Route using factor graph belief propagation (probabilistic alternative).
    ///
    /// Builds a 3-variable factor graph with factors encoding compatibility
    /// between input features (novelty, urgency, complexity) and cognitive depths.
    /// Runs loopy BP for 5 iterations and returns the depth with highest marginal.
    ///
    /// This produces smoother routing decisions than the threshold-based `route()`,
    /// handling ambiguous cases (e.g., moderate novelty + moderate complexity)
    /// more naturally by integrating all evidence simultaneously.
    pub fn route_probabilistic(
        &mut self,
        novelty: f32,
        urgency: f32,
        complexity: f32,
        emotional_intensity: f32,
    ) -> CognitiveDepth {
        use crate::consciousness::factor_graph::{BPConfig, FactorGraph, MessageSchedule};

        let mut fg = FactorGraph::new();

        // Three variables, each with 3 states: {Reflex=0, Cortical=1, DeepThought=2}
        let v_novelty = fg.add_variable("novelty_depth", 3);
        let v_urgency = fg.add_variable("urgency_depth", 3);
        let v_complexity = fg.add_variable("complexity_depth", 3);

        // Unary factors: encode how each signal prefers each depth
        // Factor tables are [state0_prob, state1_prob, state2_prob]

        // Novelty factor: low novelty → Reflex, high → DeepThought
        let n = novelty.clamp(0.0, 1.0);
        let novelty_table = vec![
            (1.0 - n as f64).max(THALAMIC_FACTOR_FLOOR),
            THALAMIC_CORTICAL_BASE_RATE,
            (n as f64 + THALAMIC_INPUT_OFFSET).min(1.0),
        ];
        fg.add_factor(&[v_novelty], novelty_table);

        // Urgency factor: high urgency → DeepThought, low → Reflex
        let u = urgency.clamp(0.0, 1.0);
        let urgency_table = vec![
            (1.0 - u as f64).max(THALAMIC_FACTOR_FLOOR),
            THALAMIC_CORTICAL_BASE_RATE,
            (u as f64 + THALAMIC_INPUT_OFFSET).min(1.0),
        ];
        fg.add_factor(&[v_urgency], urgency_table);

        // Complexity factor: low → Reflex, high → DeepThought
        let c = complexity.clamp(0.0, 1.0);
        let complexity_table = vec![
            (1.0 - c as f64).max(THALAMIC_FACTOR_FLOOR),
            THALAMIC_COMPLEXITY_CORTICAL,
            (c as f64 + THALAMIC_INPUT_OFFSET).min(1.0),
        ];
        fg.add_factor(&[v_complexity], complexity_table);

        // Emotional intensity boosts DeepThought
        let e = emotional_intensity.clamp(0.0, 1.0);
        let emotional_boost = vec![
            (1.0 - e as f64 * THALAMIC_EMOTIONAL_DAMPENING).max(THALAMIC_FACTOR_FLOOR),
            THALAMIC_CORTICAL_BASE_RATE,
            (THALAMIC_EMOTIONAL_BOOST_BASE + e as f64 * THALAMIC_EMOTIONAL_BOOST_SCALE).min(1.0),
        ];
        // Apply as additional factor on novelty variable (acts as a prior boost)
        fg.add_factor(&[v_novelty], emotional_boost);

        // Pairwise consistency factor: encourage agreement between variables
        // Table: 3x3 (row = v_novelty state, col = v_urgency state)
        let agreement_table = vec![
            1.0, 0.3, 0.1, // novelty=Reflex: strongly prefer urgency=Reflex
            0.3, 1.0, 0.3, // novelty=Cortical: prefer urgency=Cortical
            0.1, 0.3, 1.0, // novelty=DeepThought: prefer urgency=DeepThought
        ];
        fg.add_factor(&[v_novelty, v_urgency], agreement_table.clone());
        fg.add_factor(&[v_urgency, v_complexity], agreement_table);

        // Run belief propagation
        let bp_config = BPConfig {
            max_iterations: 5,
            tolerance: 1e-4,
            damping: 0.5,
            schedule: MessageSchedule::Sequential,
        };
        let result = fg.run_bp(&bp_config);

        // Aggregate beliefs: average across all 3 variables
        let mut depth_scores = [0.0f64; 3];
        for belief in &result.beliefs {
            for (i, &b) in belief.iter().enumerate().take(3) {
                depth_scores[i] += b;
            }
        }

        // Select the depth with highest aggregated marginal
        let depth = if depth_scores[2] >= depth_scores[1] && depth_scores[2] >= depth_scores[0] {
            CognitiveDepth::DeepThought
        } else if depth_scores[0] >= depth_scores[1] {
            CognitiveDepth::Reflex
        } else {
            CognitiveDepth::Cortical
        };

        // Record history
        if self.routing_history.len() >= self.max_history {
            self.routing_history.pop_front();
        }
        self.routing_history.push_back(depth);

        depth
    }

    /// Route based on prediction error and pattern
    pub fn route_from_cycle(
        &mut self,
        prediction_error: f32,
        pattern: ConsciousnessPattern,
        emotional_valence: f32,
    ) -> CognitiveDepth {
        // Novelty from prediction error (high error = novel)
        let novelty = prediction_error.min(1.0);

        // Complexity from pattern
        let complexity = match pattern {
            ConsciousnessPattern::Uncertain => 0.8,
            ConsciousnessPattern::Transitioning => 0.7,
            ConsciousnessPattern::Exploratory => 0.6,
            ConsciousnessPattern::Contemplative => 0.5,
            ConsciousnessPattern::Focused => 0.4,
            ConsciousnessPattern::Excited => 0.4,
            ConsciousnessPattern::Resting => 0.2,
        };

        // Urgency from pattern (uncertain/transitioning = urgent)
        let urgency = match pattern {
            ConsciousnessPattern::Uncertain => 0.8,
            ConsciousnessPattern::Transitioning => 0.6,
            ConsciousnessPattern::Excited => 0.5,
            _ => 0.3,
        };

        // Emotional intensity from absolute valence
        let emotional_intensity = emotional_valence.abs();

        self.route(novelty, urgency, complexity, emotional_intensity)
    }

    /// Get statistics on routing patterns
    pub fn routing_stats(&self) -> (f32, f32, f32) {
        if self.routing_history.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        // Safe cast via f64 to prevent precision loss on large counts
        let total = self.routing_history.len().max(1) as f64;
        let reflex = (self
            .routing_history
            .iter()
            .filter(|d| **d == CognitiveDepth::Reflex)
            .count() as f64
            / total) as f32;
        let cortical = (self
            .routing_history
            .iter()
            .filter(|d| **d == CognitiveDepth::Cortical)
            .count() as f64
            / total) as f32;
        let deep = (self
            .routing_history
            .iter()
            .filter(|d| **d == CognitiveDepth::DeepThought)
            .count() as f64
            / total) as f32;

        (reflex, cortical, deep)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTIVE INFERENCE BRIDGE - Precision-Weighted Prediction Tracking
// ═══════════════════════════════════════════════════════════════════════════════

/// Quality of prediction-outcome coupling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouplingQuality {
    /// Not enough data to assess coupling
    InsufficientData,
    /// No meaningful coupling (MI < 0.1)
    NoCoupling,
    /// Weak coupling (MI 0.1-0.3)
    WeakCoupling,
    /// Moderate coupling (MI 0.3-0.6)
    ModerateCoupling,
    /// Strong coupling (MI > 0.6)
    StrongCoupling,
}

impl CouplingQuality {
    /// Is coupling meaningful?
    pub fn is_meaningful(&self) -> bool {
        matches!(
            self,
            Self::WeakCoupling | Self::ModerateCoupling | Self::StrongCoupling
        )
    }

    /// Human-readable label for telemetry/dashboard display.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InsufficientData => "InsufficientData",
            Self::NoCoupling => "NoCoupling",
            Self::WeakCoupling => "WeakCoupling",
            Self::ModerateCoupling => "ModerateCoupling",
            Self::StrongCoupling => "StrongCoupling",
        }
    }
}

/// Simplified Active Inference Bridge for prediction-outcome coupling
///
/// Tracks the relationship between prediction confidence and actual outcomes
/// using a simplified Phase-Amplitude Coupling (PAC) approach.
#[derive(Debug, Clone)]
pub struct ActiveInferenceBridge {
    /// Recent confidence values (phase signal)
    confidence_history: VecDeque<f64>,

    /// Recent outcomes (amplitude signal)
    outcome_history: VecDeque<f64>,

    /// Window size for coupling computation
    window_size: usize,

    /// Minimum data points before coupling is meaningful
    min_data_points: usize,

    /// Total observations
    total_observations: usize,
}

impl Default for ActiveInferenceBridge {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl ActiveInferenceBridge {
    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self {
            confidence_history: VecDeque::with_capacity(100),
            outcome_history: VecDeque::with_capacity(100),
            window_size: 100,
            min_data_points: 10,
            total_observations: 0,
        }
    }

    /// Observe a prediction resolution
    ///
    /// * `confidence`: The predicted confidence (0.0-1.0)
    /// * `success`: Whether the prediction was correct
    pub fn observe_resolution(&mut self, confidence: f64, success: bool) {
        self.total_observations += 1;

        // Track confidence
        if self.confidence_history.len() >= self.window_size {
            self.confidence_history.pop_front();
        }
        self.confidence_history.push_back(confidence);

        // Track outcome
        let outcome = if success { 1.0 } else { 0.0 };
        if self.outcome_history.len() >= self.window_size {
            self.outcome_history.pop_front();
        }
        self.outcome_history.push_back(outcome);
    }

    /// Compute the Modulation Index (simplified PAC)
    ///
    /// Returns a value in [0, 1] where:
    /// - 0.0 = No coupling (predictions don't inform outcomes)
    /// - 1.0 = Perfect coupling (confidence perfectly predicts success)
    pub fn modulation_index(&self) -> Option<f64> {
        if self.confidence_history.len() < self.min_data_points {
            return None;
        }

        // Compute correlation between confidence and success
        // Safe cast (already f64, just ensure non-zero)
        let n = self.confidence_history.len().max(1) as f64;
        let conf_mean: f64 = self.confidence_history.iter().sum::<f64>() / n;
        let out_mean: f64 = self.outcome_history.iter().sum::<f64>() / n;

        let mut covariance = 0.0;
        let mut conf_variance = 0.0;
        let mut out_variance = 0.0;

        for (c, o) in self
            .confidence_history
            .iter()
            .zip(self.outcome_history.iter())
        {
            let c_diff = c - conf_mean;
            let o_diff = o - out_mean;
            covariance += c_diff * o_diff;
            conf_variance += c_diff * c_diff;
            out_variance += o_diff * o_diff;
        }

        // Pearson correlation, normalized to [0, 1]
        let denom = (conf_variance * out_variance).sqrt();
        if denom < 1e-10 {
            return Some(0.0);
        }

        let correlation = covariance / denom;
        // Map [-1, 1] to [0, 1], favoring positive correlations
        Some((correlation.max(0.0)).clamp(0.0, 1.0))
    }

    /// Get the current coupling quality assessment
    pub fn coupling_quality(&self) -> CouplingQuality {
        match self.modulation_index() {
            None => CouplingQuality::InsufficientData,
            Some(mi) if mi < super::thresholds::COUPLING_NO_COUPLING_THRESHOLD as f64 => {
                CouplingQuality::NoCoupling
            }
            Some(mi) if mi < super::thresholds::COUPLING_WEAK_THRESHOLD as f64 => {
                CouplingQuality::WeakCoupling
            }
            Some(mi) if mi < super::thresholds::COUPLING_MODERATE_THRESHOLD as f64 => {
                CouplingQuality::ModerateCoupling
            }
            Some(_) => CouplingQuality::StrongCoupling,
        }
    }

    /// Get average prediction error (from recent history)
    pub fn average_prediction_error(&self) -> Option<f64> {
        if self.outcome_history.is_empty() {
            return None;
        }
        // Error = 1 - success rate (safe division with max(1))
        let success_rate: f64 =
            self.outcome_history.iter().sum::<f64>() / self.outcome_history.len().max(1) as f64;
        Some(1.0 - success_rate)
    }

    /// Get statistics
    pub fn statistics(&self) -> ActiveInferenceBridgeStats {
        ActiveInferenceBridgeStats {
            modulation_index: self.modulation_index(),
            coupling_quality: self.coupling_quality(),
            average_prediction_error: self.average_prediction_error(),
        }
    }

    /// Reset the bridge
    pub fn reset(&mut self) {
        self.confidence_history.clear();
        self.outcome_history.clear();
        self.total_observations = 0;
    }
}

/// Statistics from the Active Inference bridge
#[derive(Debug, Clone)]
pub struct ActiveInferenceBridgeStats {
    /// Current Modulation Index
    pub modulation_index: Option<f64>,
    /// Current coupling quality
    pub coupling_quality: CouplingQuality,
    /// Average prediction error (recent)
    pub average_prediction_error: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probabilistic_routing() {
        let mut router = ThalamicRouter::new();

        // Extreme case: very high novelty should route to DeepThought
        let depth = router.route_probabilistic(0.95, 0.9, 0.8, 0.5);
        assert_eq!(
            depth,
            CognitiveDepth::DeepThought,
            "High novelty+urgency+complexity should route to DeepThought"
        );

        // Extreme case: very low everything should route to Reflex
        let depth = router.route_probabilistic(0.05, 0.05, 0.05, 0.0);
        assert_eq!(
            depth,
            CognitiveDepth::Reflex,
            "Very low inputs should route to Reflex"
        );

        // Probabilistic routing should agree with threshold routing on extreme cases
        let mut threshold_router = ThalamicRouter::new();
        let threshold_depth = threshold_router.route(0.95, 0.9, 0.8, 0.5);
        assert_eq!(
            depth,
            CognitiveDepth::Reflex, // This tests the second call
        );
        // The high-novelty case should agree:
        let mut r2 = ThalamicRouter::new();
        let bp_high = r2.route_probabilistic(0.95, 0.9, 0.8, 0.5);
        assert_eq!(
            bp_high, threshold_depth,
            "BP routing should agree with threshold routing for extreme high inputs"
        );
    }
}
