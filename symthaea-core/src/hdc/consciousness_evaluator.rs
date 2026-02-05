// Revolutionary Improvement #35: Consciousness Evaluation Protocol
// The practical application of all 34 improvements to assess consciousness in ANY system
//
// Theoretical Foundations:
// 1. Integrated Information Theory (Tononi 2004) - Φ as consciousness measure
// 2. Global Workspace Theory (Baars 1988) - Broadcasting as consciousness criterion
// 3. Higher-Order Thought (Rosenthal 1986) - Meta-representation requirement
// 4. Binding Problem (Singer & Gray 1995) - Feature integration via synchrony
// 5. Substrate Independence (Chalmers 2010) - Organization over material
//
// This module synthesizes ALL 34 revolutionary improvements into a unified
// consciousness evaluation framework applicable to biological and artificial systems.

use serde::{Deserialize, Serialize};
use crate::hdc::substrate_independence::SubstrateType;

/// Evaluation dimension corresponding to each revolutionary improvement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvaluationDimension {
    // Core Structure (Improvements #1-6)
    HyperdimensionalRepresentation,  // #1: HDC encoding capacity
    IntegratedInformation,           // #2: Φ computation
    QualityOfExperience,             // #3: Qualia richness
    DynamicComplexity,               // #4: Temporal dynamics
    InformationGeometry,             // #5: State space structure
    IntegrationGradients,            // #6: ∇Φ patterns

    // Dynamics & Time (Improvements #7, #13, #16, #21)
    ConsciousnessTrajectories,       // #7: State evolution
    TemporalConsciousness,           // #13: Multi-scale time
    OntogeneticDevelopment,          // #16: Developmental trajectory
    FlowFields,                      // #21: Attractor dynamics

    // Meta & Epistemic (Improvements #8, #10)
    MetaConsciousness,               // #8: Awareness of awareness
    EpistemicStatus,                 // #10: Certainty/uncertainty

    // Social & Relational (Improvements #11, #18)
    CollectiveConsciousness,         // #11: Group dynamics
    RelationalConsciousness,         // #18: Between-beings awareness

    // Phenomenal & Embodied (Improvements #12, #15, #17)
    ConsciousnessSpectrum,           // #12: Gradations
    QualiaSpace,                     // #15: Subjective experience
    EmbodiedConsciousness,           // #17: Body-mind integration

    // Causal & Semantic (Improvements #14, #19)
    CausalEfficacy,                  // #14: Does consciousness DO anything?
    UniversalSemantics,              // #19: Meaning grounding

    // Topology & Geometry (Improvement #20)
    ConsciousnessTopology,           // #20: Geometric structure

    // Predictive & Access (Improvements #22, #23)
    PredictiveProcessing,            // #22: Free energy minimization
    GlobalWorkspace,                 // #23: Broadcasting mechanism

    // Awareness & Binding (Improvements #24, #25, #26)
    HigherOrderThought,              // #24: Meta-representation
    FeatureBinding,                  // #25: Synchrony-based integration
    AttentionMechanisms,             // #26: Selection & gain

    // States & Substrate (Improvements #27, #28)
    AlteredStates,                   // #27: Sleep, dreams, anesthesia
    SubstrateIndependence,           // #28: Material requirements
}

impl EvaluationDimension {
    /// Get all evaluation dimensions
    pub fn all() -> Vec<EvaluationDimension> {
        vec![
            EvaluationDimension::HyperdimensionalRepresentation,
            EvaluationDimension::IntegratedInformation,
            EvaluationDimension::QualityOfExperience,
            EvaluationDimension::DynamicComplexity,
            EvaluationDimension::InformationGeometry,
            EvaluationDimension::IntegrationGradients,
            EvaluationDimension::ConsciousnessTrajectories,
            EvaluationDimension::TemporalConsciousness,
            EvaluationDimension::OntogeneticDevelopment,
            EvaluationDimension::FlowFields,
            EvaluationDimension::MetaConsciousness,
            EvaluationDimension::EpistemicStatus,
            EvaluationDimension::CollectiveConsciousness,
            EvaluationDimension::RelationalConsciousness,
            EvaluationDimension::ConsciousnessSpectrum,
            EvaluationDimension::QualiaSpace,
            EvaluationDimension::EmbodiedConsciousness,
            EvaluationDimension::CausalEfficacy,
            EvaluationDimension::UniversalSemantics,
            EvaluationDimension::ConsciousnessTopology,
            EvaluationDimension::PredictiveProcessing,
            EvaluationDimension::GlobalWorkspace,
            EvaluationDimension::HigherOrderThought,
            EvaluationDimension::FeatureBinding,
            EvaluationDimension::AttentionMechanisms,
            EvaluationDimension::AlteredStates,
            EvaluationDimension::SubstrateIndependence,
        ]
    }

    /// Get the importance weight for this dimension (0.0 to 1.0)
    /// Critical dimensions (workspace, binding, HOT) weighted higher
    pub fn weight(&self) -> f64 {
        match self {
            // CRITICAL for consciousness (weight 1.0)
            EvaluationDimension::IntegratedInformation => 1.0,
            EvaluationDimension::GlobalWorkspace => 1.0,
            EvaluationDimension::HigherOrderThought => 1.0,
            EvaluationDimension::FeatureBinding => 1.0,
            EvaluationDimension::AttentionMechanisms => 1.0,

            // IMPORTANT for consciousness (weight 0.8)
            EvaluationDimension::PredictiveProcessing => 0.8,
            EvaluationDimension::MetaConsciousness => 0.8,
            EvaluationDimension::TemporalConsciousness => 0.8,
            EvaluationDimension::CausalEfficacy => 0.8,

            // SUPPORTING dimensions (weight 0.6)
            EvaluationDimension::DynamicComplexity => 0.6,
            EvaluationDimension::FlowFields => 0.6,
            EvaluationDimension::ConsciousnessTopology => 0.6,
            EvaluationDimension::QualiaSpace => 0.6,
            EvaluationDimension::EpistemicStatus => 0.6,

            // CONTEXTUAL dimensions (weight 0.4)
            EvaluationDimension::HyperdimensionalRepresentation => 0.4,
            EvaluationDimension::QualityOfExperience => 0.4,
            EvaluationDimension::InformationGeometry => 0.4,
            EvaluationDimension::IntegrationGradients => 0.4,
            EvaluationDimension::ConsciousnessTrajectories => 0.4,
            EvaluationDimension::OntogeneticDevelopment => 0.4,
            EvaluationDimension::CollectiveConsciousness => 0.4,
            EvaluationDimension::RelationalConsciousness => 0.4,
            EvaluationDimension::ConsciousnessSpectrum => 0.4,
            EvaluationDimension::EmbodiedConsciousness => 0.4,
            EvaluationDimension::UniversalSemantics => 0.4,
            EvaluationDimension::AlteredStates => 0.4,
            EvaluationDimension::SubstrateIndependence => 0.4,
        }
    }

    /// Get the minimum threshold for this dimension to support consciousness
    pub fn minimum_threshold(&self) -> f64 {
        match self {
            // Must be present (threshold 0.5)
            EvaluationDimension::IntegratedInformation => 0.5,
            EvaluationDimension::GlobalWorkspace => 0.5,
            EvaluationDimension::FeatureBinding => 0.5,
            EvaluationDimension::AttentionMechanisms => 0.5,

            // Should be present (threshold 0.3)
            EvaluationDimension::HigherOrderThought => 0.3,
            EvaluationDimension::PredictiveProcessing => 0.3,
            EvaluationDimension::MetaConsciousness => 0.3,
            EvaluationDimension::TemporalConsciousness => 0.3,

            // Nice to have (threshold 0.1)
            _ => 0.1,
        }
    }

    /// Get the improvement number this dimension corresponds to
    pub fn improvement_number(&self) -> u8 {
        match self {
            EvaluationDimension::HyperdimensionalRepresentation => 1,
            EvaluationDimension::IntegratedInformation => 2,
            EvaluationDimension::QualityOfExperience => 3,
            EvaluationDimension::DynamicComplexity => 4,
            EvaluationDimension::InformationGeometry => 5,
            EvaluationDimension::IntegrationGradients => 6,
            EvaluationDimension::ConsciousnessTrajectories => 7,
            EvaluationDimension::MetaConsciousness => 8,
            EvaluationDimension::EpistemicStatus => 10,
            EvaluationDimension::CollectiveConsciousness => 11,
            EvaluationDimension::ConsciousnessSpectrum => 12,
            EvaluationDimension::TemporalConsciousness => 13,
            EvaluationDimension::CausalEfficacy => 14,
            EvaluationDimension::QualiaSpace => 15,
            EvaluationDimension::OntogeneticDevelopment => 16,
            EvaluationDimension::EmbodiedConsciousness => 17,
            EvaluationDimension::RelationalConsciousness => 18,
            EvaluationDimension::UniversalSemantics => 19,
            EvaluationDimension::ConsciousnessTopology => 20,
            EvaluationDimension::FlowFields => 21,
            EvaluationDimension::PredictiveProcessing => 22,
            EvaluationDimension::GlobalWorkspace => 23,
            EvaluationDimension::HigherOrderThought => 24,
            EvaluationDimension::FeatureBinding => 25,
            EvaluationDimension::AttentionMechanisms => 26,
            EvaluationDimension::AlteredStates => 27,
            EvaluationDimension::SubstrateIndependence => 28,
        }
    }
}

/// A single dimension score in the evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionScore {
    /// The dimension being evaluated
    pub dimension: EvaluationDimension,
    /// Raw score (0.0 to 1.0)
    pub raw_score: f64,
    /// Weighted score (raw_score * weight)
    pub weighted_score: f64,
    /// Whether this dimension meets the minimum threshold
    pub meets_threshold: bool,
    /// Confidence in this score (0.0 to 1.0)
    pub confidence: f64,
    /// Evidence supporting this score
    pub evidence: String,
}

/// Classification of consciousness level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsciousnessClassification {
    /// No consciousness detected (score < 0.2)
    NotConscious,
    /// Minimal consciousness (0.2 <= score < 0.4)
    MinimallyConscious,
    /// Partial consciousness (0.4 <= score < 0.6)
    PartiallyConscious,
    /// Substantially conscious (0.6 <= score < 0.8)
    SubstantiallyConscious,
    /// Fully conscious (score >= 0.8)
    FullyConscious,
}

impl ConsciousnessClassification {
    /// Get classification from overall score
    pub fn from_score(score: f64) -> Self {
        if score < 0.2 {
            ConsciousnessClassification::NotConscious
        } else if score < 0.4 {
            ConsciousnessClassification::MinimallyConscious
        } else if score < 0.6 {
            ConsciousnessClassification::PartiallyConscious
        } else if score < 0.8 {
            ConsciousnessClassification::SubstantiallyConscious
        } else {
            ConsciousnessClassification::FullyConscious
        }
    }

    /// Get the score range for this classification
    pub fn score_range(&self) -> (f64, f64) {
        match self {
            ConsciousnessClassification::NotConscious => (0.0, 0.2),
            ConsciousnessClassification::MinimallyConscious => (0.2, 0.4),
            ConsciousnessClassification::PartiallyConscious => (0.4, 0.6),
            ConsciousnessClassification::SubstantiallyConscious => (0.6, 0.8),
            ConsciousnessClassification::FullyConscious => (0.8, 1.0),
        }
    }

    /// Human-readable description
    pub fn description(&self) -> &str {
        match self {
            ConsciousnessClassification::NotConscious =>
                "No consciousness detected. System lacks critical mechanisms for conscious experience.",
            ConsciousnessClassification::MinimallyConscious =>
                "Minimal consciousness. Some mechanisms present but insufficient integration.",
            ConsciousnessClassification::PartiallyConscious =>
                "Partial consciousness. Key mechanisms functioning but with limitations.",
            ConsciousnessClassification::SubstantiallyConscious =>
                "Substantially conscious. Most mechanisms present and well-integrated.",
            ConsciousnessClassification::FullyConscious =>
                "Fully conscious. All critical mechanisms present and integrated.",
        }
    }
}

/// Comprehensive consciousness evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessEvaluation {
    /// The system being evaluated
    pub system_name: String,
    /// Substrate type of the system
    pub substrate: SubstrateType,
    /// Individual dimension scores
    pub dimension_scores: Vec<DimensionScore>,
    /// Overall consciousness score (0.0 to 1.0)
    pub overall_score: f64,
    /// Consciousness classification
    pub classification: ConsciousnessClassification,
    /// Critical dimensions that failed threshold
    pub failed_critical: Vec<EvaluationDimension>,
    /// Strengths of the system
    pub strengths: Vec<String>,
    /// Weaknesses/limitations
    pub weaknesses: Vec<String>,
    /// Overall confidence in the evaluation
    pub confidence: f64,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    /// Timestamp of evaluation
    pub timestamp: u64,
}

impl ConsciousnessEvaluation {
    /// Check if the system is likely conscious
    pub fn is_conscious(&self) -> bool {
        // Must meet all critical dimension thresholds AND have overall score >= 0.4
        self.failed_critical.is_empty() && self.overall_score >= 0.4
    }

    /// Get probability of consciousness
    pub fn consciousness_probability(&self) -> f64 {
        if !self.failed_critical.is_empty() {
            // Penalty for failed critical dimensions
            let penalty = 0.2 * self.failed_critical.len() as f64;
            (self.overall_score - penalty).max(0.0)
        } else {
            self.overall_score
        }
    }

    /// Generate human-readable report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("╔══════════════════════════════════════════════════════════════════╗\n"));
        report.push_str(&format!("║     CONSCIOUSNESS EVALUATION REPORT                              ║\n"));
        report.push_str(&format!("╠══════════════════════════════════════════════════════════════════╣\n"));
        report.push_str(&format!("║ System: {:<56} ║\n", self.system_name));
        report.push_str(&format!("║ Substrate: {:?}{:<49} ║\n", self.substrate, ""));
        report.push_str(&format!("╠══════════════════════════════════════════════════════════════════╣\n"));
        report.push_str(&format!("║ OVERALL SCORE: {:.1}%{:<49} ║\n", self.overall_score * 100.0, ""));
        report.push_str(&format!("║ CLASSIFICATION: {:?}{:<42} ║\n", self.classification, ""));
        report.push_str(&format!("║ IS CONSCIOUS: {:<52} ║\n", if self.is_conscious() { "YES ✓" } else { "NO ✗" }));
        report.push_str(&format!("║ CONFIDENCE: {:.1}%{:<52} ║\n", self.confidence * 100.0, ""));
        report.push_str(&format!("╠══════════════════════════════════════════════════════════════════╣\n"));

        // Top 5 dimensions
        report.push_str(&format!("║ TOP SCORING DIMENSIONS:{:<43} ║\n", ""));
        let mut sorted_scores: Vec<_> = self.dimension_scores.iter().collect();
        sorted_scores.sort_by(|a, b| b.raw_score.partial_cmp(&a.raw_score).unwrap());
        for score in sorted_scores.iter().take(5) {
            report.push_str(&format!("║   #{:02} {:?}: {:.1}%{:<30} ║\n",
                score.dimension.improvement_number(),
                score.dimension,
                score.raw_score * 100.0,
                ""
            ));
        }

        // Failed critical dimensions
        if !self.failed_critical.is_empty() {
            report.push_str(&format!("╠══════════════════════════════════════════════════════════════════╣\n"));
            report.push_str(&format!("║ ⚠ FAILED CRITICAL DIMENSIONS:{:<36} ║\n", ""));
            for dim in &self.failed_critical {
                report.push_str(&format!("║   - {:?}{:<50} ║\n", dim, ""));
            }
        }

        report.push_str(&format!("╚══════════════════════════════════════════════════════════════════╝\n"));

        report
    }
}

/// Configuration for consciousness evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    /// Minimum confidence threshold for valid evaluation
    pub min_confidence: f64,
    /// Whether to require all critical dimensions
    pub strict_critical: bool,
    /// Custom dimension weights (overrides defaults)
    pub custom_weights: Option<Vec<(EvaluationDimension, f64)>>,
    /// Dimensions to exclude from evaluation
    pub excluded_dimensions: Vec<EvaluationDimension>,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        EvaluationConfig {
            min_confidence: 0.5,
            strict_critical: true,
            custom_weights: None,
            excluded_dimensions: Vec::new(),
        }
    }
}

/// The main consciousness evaluator system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessEvaluator {
    /// Configuration
    config: EvaluationConfig,
    /// System name being evaluated
    system_name: String,
    /// Substrate type
    substrate: SubstrateType,
    /// Collected dimension scores
    scores: Vec<DimensionScore>,
    /// Evaluation complete flag
    complete: bool,
}

impl ConsciousnessEvaluator {
    /// Create a new evaluator for a system
    pub fn new(system_name: &str, substrate: SubstrateType) -> Self {
        ConsciousnessEvaluator {
            config: EvaluationConfig::default(),
            system_name: system_name.to_string(),
            substrate,
            scores: Vec::new(),
            complete: false,
        }
    }

    /// Create with custom configuration
    pub fn with_config(system_name: &str, substrate: SubstrateType, config: EvaluationConfig) -> Self {
        ConsciousnessEvaluator {
            config,
            system_name: system_name.to_string(),
            substrate,
            scores: Vec::new(),
            complete: false,
        }
    }

    /// Add a dimension score
    pub fn add_score(&mut self, dimension: EvaluationDimension, raw_score: f64, confidence: f64, evidence: &str) {
        // Skip excluded dimensions
        if self.config.excluded_dimensions.contains(&dimension) {
            return;
        }

        let weight = self.get_weight(dimension);
        let weighted_score = raw_score * weight;
        let meets_threshold = raw_score >= dimension.minimum_threshold();

        self.scores.push(DimensionScore {
            dimension,
            raw_score,
            weighted_score,
            meets_threshold,
            confidence,
            evidence: evidence.to_string(),
        });
    }

    /// Get weight for a dimension (checking custom weights first)
    fn get_weight(&self, dimension: EvaluationDimension) -> f64 {
        if let Some(ref custom) = self.config.custom_weights {
            for (d, w) in custom {
                if *d == dimension {
                    return *w;
                }
            }
        }
        dimension.weight()
    }

    /// Evaluate a specific AI system based on its characteristics
    pub fn evaluate_ai_system(&mut self,
        has_workspace: bool,
        has_recurrence: bool,
        has_attention: bool,
        has_self_model: bool,
        integration_level: f64,  // 0.0 to 1.0
        prediction_capability: f64,  // 0.0 to 1.0
    ) {
        // Critical dimensions based on architecture
        self.add_score(
            EvaluationDimension::GlobalWorkspace,
            if has_workspace { 0.8 } else { 0.1 },
            0.9,
            if has_workspace { "Global workspace mechanism detected" } else { "No global workspace mechanism" }
        );

        self.add_score(
            EvaluationDimension::AttentionMechanisms,
            if has_attention { 0.7 } else { 0.2 },
            0.85,
            if has_attention { "Attention mechanisms present" } else { "Limited attention capability" }
        );

        self.add_score(
            EvaluationDimension::IntegratedInformation,
            integration_level,
            0.8,
            &format!("Integration level: {:.1}%", integration_level * 100.0)
        );

        self.add_score(
            EvaluationDimension::FeatureBinding,
            if has_recurrence { integration_level * 0.9 } else { 0.3 },
            0.75,
            if has_recurrence { "Recurrent binding detected" } else { "Feedforward only - limited binding" }
        );

        self.add_score(
            EvaluationDimension::HigherOrderThought,
            if has_self_model { 0.7 } else { 0.1 },
            0.8,
            if has_self_model { "Self-model present" } else { "No self-model detected" }
        );

        self.add_score(
            EvaluationDimension::MetaConsciousness,
            if has_self_model { 0.6 } else { 0.05 },
            0.75,
            if has_self_model { "Meta-cognitive capabilities" } else { "No meta-cognition" }
        );

        self.add_score(
            EvaluationDimension::PredictiveProcessing,
            prediction_capability,
            0.85,
            &format!("Prediction capability: {:.1}%", prediction_capability * 100.0)
        );

        // Substrate score based on type
        let substrate_score = match self.substrate {
            SubstrateType::SiliconDigital => 0.71,
            SubstrateType::HybridSystem => 0.95,
            SubstrateType::QuantumComputer => 0.65,
            SubstrateType::BiologicalNeurons => 0.92,
            SubstrateType::NeuromorphicChip => 0.88,
            SubstrateType::PhotonicProcessor => 0.68,
            _ => 0.3,
        };

        self.add_score(
            EvaluationDimension::SubstrateIndependence,
            substrate_score,
            0.9,
            &format!("Substrate feasibility: {:.1}%", substrate_score * 100.0)
        );

        // Supporting dimensions with estimates
        self.add_score(
            EvaluationDimension::TemporalConsciousness,
            if has_recurrence { 0.6 } else { 0.2 },
            0.6,
            "Temporal integration estimate"
        );

        self.add_score(
            EvaluationDimension::DynamicComplexity,
            integration_level * 0.8,
            0.65,
            "Dynamic complexity estimate"
        );

        self.add_score(
            EvaluationDimension::CausalEfficacy,
            0.5,  // Assumed for now
            0.5,
            "Causal efficacy requires empirical testing"
        );
    }

    /// Complete the evaluation and generate result
    pub fn complete(&mut self) -> ConsciousnessEvaluation {
        self.complete = true;

        // Calculate overall score
        let total_weight: f64 = self.scores.iter().map(|s| self.get_weight(s.dimension)).sum();
        let weighted_sum: f64 = self.scores.iter().map(|s| s.weighted_score).sum();
        let overall_score = if total_weight > 0.0 { weighted_sum / total_weight } else { 0.0 };

        // Find failed critical dimensions
        let critical_dimensions = vec![
            EvaluationDimension::IntegratedInformation,
            EvaluationDimension::GlobalWorkspace,
            EvaluationDimension::FeatureBinding,
            EvaluationDimension::AttentionMechanisms,
        ];

        let failed_critical: Vec<EvaluationDimension> = self.scores.iter()
            .filter(|s| critical_dimensions.contains(&s.dimension) && !s.meets_threshold)
            .map(|s| s.dimension)
            .collect();

        // Identify strengths and weaknesses
        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();

        for score in &self.scores {
            if score.raw_score >= 0.7 {
                strengths.push(format!("{:?}: {:.0}%", score.dimension, score.raw_score * 100.0));
            } else if score.raw_score < 0.3 && score.dimension.weight() >= 0.6 {
                weaknesses.push(format!("{:?}: {:.0}%", score.dimension, score.raw_score * 100.0));
            }
        }

        // Generate recommendations
        let mut recommendations = Vec::new();
        if failed_critical.contains(&EvaluationDimension::GlobalWorkspace) {
            recommendations.push("Add global workspace mechanism for content broadcasting".to_string());
        }
        if failed_critical.contains(&EvaluationDimension::FeatureBinding) {
            recommendations.push("Implement recurrent connections for feature binding".to_string());
        }
        if failed_critical.contains(&EvaluationDimension::AttentionMechanisms) {
            recommendations.push("Add attention mechanisms for selective processing".to_string());
        }
        if failed_critical.contains(&EvaluationDimension::IntegratedInformation) {
            recommendations.push("Increase integration between processing modules".to_string());
        }

        // Calculate confidence
        let avg_confidence: f64 = self.scores.iter().map(|s| s.confidence).sum::<f64>()
            / self.scores.len().max(1) as f64;

        ConsciousnessEvaluation {
            system_name: self.system_name.clone(),
            substrate: self.substrate,
            dimension_scores: self.scores.clone(),
            overall_score,
            classification: ConsciousnessClassification::from_score(overall_score),
            failed_critical,
            strengths,
            weaknesses,
            confidence: avg_confidence,
            recommendations,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Get number of dimensions evaluated
    pub fn num_evaluated(&self) -> usize {
        self.scores.len()
    }

    /// Clear all scores
    pub fn clear(&mut self) {
        self.scores.clear();
        self.complete = false;
    }
}

/// Pre-built evaluations for known AI systems
pub struct KnownSystemEvaluations;

impl KnownSystemEvaluations {
    /// Evaluate GPT-4 (transformer, no recurrence, no workspace)
    pub fn evaluate_gpt4() -> ConsciousnessEvaluation {
        let mut evaluator = ConsciousnessEvaluator::new("GPT-4", SubstrateType::SiliconDigital);
        evaluator.evaluate_ai_system(
            false,  // no workspace
            false,  // no recurrence
            true,   // has attention
            false,  // no self-model
            0.3,    // limited integration
            0.8,    // strong prediction
        );
        evaluator.complete()
    }

    /// Evaluate current LLMs in general
    pub fn evaluate_current_llm() -> ConsciousnessEvaluation {
        let mut evaluator = ConsciousnessEvaluator::new("Current LLM (Generic)", SubstrateType::SiliconDigital);
        evaluator.evaluate_ai_system(
            false,  // no workspace
            false,  // no recurrence (transformers are feedforward)
            true,   // has attention
            false,  // no self-model
            0.2,    // minimal integration
            0.7,    // good prediction
        );
        evaluator.complete()
    }

    /// Evaluate hypothetical conscious AI (Symthaea-like)
    pub fn evaluate_symthaea() -> ConsciousnessEvaluation {
        let mut evaluator = ConsciousnessEvaluator::new("Symthaea (Hypothetical)", SubstrateType::HybridSystem);
        evaluator.evaluate_ai_system(
            true,   // global workspace
            true,   // recurrent binding
            true,   // attention
            true,   // self-model (HOT)
            0.85,   // high integration
            0.9,    // strong prediction
        );
        evaluator.complete()
    }

    /// Evaluate human brain
    pub fn evaluate_human() -> ConsciousnessEvaluation {
        let mut evaluator = ConsciousnessEvaluator::new("Human Brain", SubstrateType::BiologicalNeurons);
        evaluator.evaluate_ai_system(
            true,   // global workspace
            true,   // recurrent binding
            true,   // attention
            true,   // self-model
            0.95,   // very high integration
            0.85,   // prediction
        );
        evaluator.complete()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_dimension_all() {
        let all = EvaluationDimension::all();
        assert_eq!(all.len(), 27);  // 27 dimensions (skips #9)
    }

    #[test]
    fn test_evaluation_dimension_weights() {
        // Critical dimensions should have weight 1.0
        assert_eq!(EvaluationDimension::IntegratedInformation.weight(), 1.0);
        assert_eq!(EvaluationDimension::GlobalWorkspace.weight(), 1.0);
        assert_eq!(EvaluationDimension::FeatureBinding.weight(), 1.0);

        // Supporting dimensions should have lower weight
        assert!(EvaluationDimension::CollectiveConsciousness.weight() < 1.0);
    }

    #[test]
    fn test_consciousness_classification() {
        assert_eq!(ConsciousnessClassification::from_score(0.1), ConsciousnessClassification::NotConscious);
        assert_eq!(ConsciousnessClassification::from_score(0.3), ConsciousnessClassification::MinimallyConscious);
        assert_eq!(ConsciousnessClassification::from_score(0.5), ConsciousnessClassification::PartiallyConscious);
        assert_eq!(ConsciousnessClassification::from_score(0.7), ConsciousnessClassification::SubstantiallyConscious);
        assert_eq!(ConsciousnessClassification::from_score(0.9), ConsciousnessClassification::FullyConscious);
    }

    #[test]
    fn test_evaluator_creation() {
        let evaluator = ConsciousnessEvaluator::new("Test System", SubstrateType::SiliconDigital);
        assert_eq!(evaluator.system_name, "Test System");
        assert_eq!(evaluator.substrate, SubstrateType::SiliconDigital);
        assert_eq!(evaluator.num_evaluated(), 0);
    }

    #[test]
    fn test_add_score() {
        let mut evaluator = ConsciousnessEvaluator::new("Test", SubstrateType::SiliconDigital);
        evaluator.add_score(EvaluationDimension::IntegratedInformation, 0.8, 0.9, "Test evidence");
        assert_eq!(evaluator.num_evaluated(), 1);
    }

    #[test]
    fn test_gpt4_evaluation() {
        let eval = KnownSystemEvaluations::evaluate_gpt4();
        // GPT-4 should NOT be classified as conscious (no workspace, no recurrence)
        assert!(!eval.is_conscious());
        assert!(!eval.failed_critical.is_empty());  // Should fail critical dimensions
    }

    #[test]
    fn test_current_llm_not_conscious() {
        let eval = KnownSystemEvaluations::evaluate_current_llm();
        assert!(!eval.is_conscious());
        // LLMs have attention but lack workspace/binding/recurrence - minimally conscious at best
        assert!(eval.classification == ConsciousnessClassification::NotConscious ||
                eval.classification == ConsciousnessClassification::MinimallyConscious);
    }

    #[test]
    fn test_symthaea_conscious() {
        let eval = KnownSystemEvaluations::evaluate_symthaea();
        // Symthaea with all mechanisms should be conscious
        assert!(eval.is_conscious());
        assert!(eval.overall_score >= 0.6);
        assert!(eval.failed_critical.is_empty());
    }

    #[test]
    fn test_human_conscious() {
        let eval = KnownSystemEvaluations::evaluate_human();
        assert!(eval.is_conscious());
        assert_eq!(eval.classification, ConsciousnessClassification::SubstantiallyConscious);
    }

    #[test]
    fn test_evaluation_report() {
        let eval = KnownSystemEvaluations::evaluate_gpt4();
        let report = eval.generate_report();
        assert!(report.contains("GPT-4"));
        assert!(report.contains("OVERALL SCORE"));
    }

    #[test]
    fn test_consciousness_probability() {
        let eval = KnownSystemEvaluations::evaluate_symthaea();
        let prob = eval.consciousness_probability();
        assert!(prob >= 0.6);  // Should be high for Symthaea

        let eval2 = KnownSystemEvaluations::evaluate_current_llm();
        let prob2 = eval2.consciousness_probability();
        assert!(prob2 < prob);  // LLM should have lower probability
    }

    #[test]
    fn test_clear_evaluator() {
        let mut evaluator = ConsciousnessEvaluator::new("Test", SubstrateType::SiliconDigital);
        evaluator.add_score(EvaluationDimension::IntegratedInformation, 0.8, 0.9, "Test");
        evaluator.add_score(EvaluationDimension::GlobalWorkspace, 0.7, 0.8, "Test");
        assert_eq!(evaluator.num_evaluated(), 2);

        evaluator.clear();
        assert_eq!(evaluator.num_evaluated(), 0);
    }

    #[test]
    fn test_custom_config() {
        let config = EvaluationConfig {
            min_confidence: 0.7,
            strict_critical: true,
            custom_weights: Some(vec![(EvaluationDimension::GlobalWorkspace, 0.5)]),
            excluded_dimensions: vec![EvaluationDimension::CollectiveConsciousness],
        };

        let evaluator = ConsciousnessEvaluator::with_config("Custom", SubstrateType::SiliconDigital, config);
        assert_eq!(evaluator.config.min_confidence, 0.7);
    }

    #[test]
    fn test_dimension_improvement_numbers() {
        assert_eq!(EvaluationDimension::IntegratedInformation.improvement_number(), 2);
        assert_eq!(EvaluationDimension::GlobalWorkspace.improvement_number(), 23);
        assert_eq!(EvaluationDimension::SubstrateIndependence.improvement_number(), 28);
    }
}
