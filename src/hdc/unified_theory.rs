//! Revolutionary Improvement #29: The Unified Theory of Consciousness (Grand Unification)
//!
//! This module synthesizes ALL 28 previous improvements into ONE coherent mathematical
//! framework - the "Standard Model of Consciousness."
//!
//! # The Master Equation
//!
//! ```text
//! C = min(Φ, B, W) × [Σ(wᵢ × Cᵢ) / Σ(wᵢ)] × S(substrate)
//! ```
//!
//! Where:
//! - min(Φ, B, W) = critical threshold (integration, binding, workspace required)
//! - Cᵢ = each of 28 component values [0,1]
//! - wᵢ = component weights (theoretical + empirical)
//! - S = substrate feasibility factor [0,1]
//!
//! # Theoretical Foundations
//!
//! 1. **Grand Unified Theories in Physics** (Einstein 1915, Weinberg 1967)
//!    - Multiple forces → one framework (electroweak, GUT attempts)
//!    - Consciousness unification follows same principle
//!
//! 2. **Integration of Consciousness Theories** (Seth 2018, Northoff 2020)
//!    - IIT, GWT, HOT, Predictive Processing are COMPLEMENTARY not competing
//!    - Each captures different aspect of same phenomenon
//!
//! 3. **Multi-Dimensional Consciousness** (Bayne 2007, Overgaard 2011)
//!    - Consciousness has multiple dimensions (level, content, access, phenomenal)
//!    - Unified theory must capture all dimensions
//!
//! 4. **Mathematical Synthesis** (Tegmark 2016)
//!    - Consciousness as mathematical structure
//!    - Unified equation captures essential properties
//!
//! 5. **Emergence and Integration** (Dehaene 2014, Koch 2016)
//!    - Consciousness emerges from integration of components
//!    - Not just sum - multiplicative interactions matter

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// All 28 consciousness components from our framework
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsciousnessComponent {
    // Foundational (#1-5)
    HyperdimensionalEncoding,    // #1: HDC substrate
    IntegratedInformation,       // #2: Φ (IIT)
    RecursiveAwareness,          // #3: Self-reference
    BoundaryDetection,           // #4: Self/other
    EmergentComplexity,          // #5: Complexity measures

    // Structural (#6-10)
    GradientDynamics,            // #6: ∇Φ
    DynamicTrajectories,         // #7: Consciousness dynamics
    MetaConsciousness,           // #8: Awareness of awareness
    ConsciousnessSpectrum,       // #9: Gradations
    EpistemicCertainty,          // #10: Confidence levels

    // Social (#11-12)
    CollectiveConsciousness,     // #11: Group minds
    ConsciousnessSpectrum2,      // #12: Conscious/unconscious boundary

    // Temporal (#13-16)
    TemporalConsciousness,       // #13: Multi-scale time
    CausalEfficacy,              // #14: Does consciousness DO anything?
    QualiaRepresentation,        // #15: Subjective experience
    Ontogeny,                    // #16: Development

    // Embodied & Relational (#17-18)
    EmbodiedConsciousness,       // #17: Body-mind
    RelationalConsciousness,     // #18: I-Thou

    // Semantic & Topological (#19-21)
    UniversalSemantics,          // #19: NSM 65 primes
    ConsciousnessTopology,       // #20: Geometric structure
    FlowFields,                  // #21: Dynamics on manifold

    // Core Theories (#22-25)
    PredictiveConsciousness,     // #22: Free Energy Principle
    GlobalWorkspace,             // #23: Broadcasting
    HigherOrderThought,          // #24: Meta-representation
    BindingProblem,              // #25: Synchrony

    // Selection & Alterations (#26-27)
    AttentionMechanisms,         // #26: Gain modulation
    SleepAndAlteredStates,       // #27: State variations

    // Universality (#28)
    SubstrateIndependence,       // #28: Multiple realizability
}

impl ConsciousnessComponent {
    /// Returns all 28 components
    pub fn all() -> Vec<Self> {
        vec![
            Self::HyperdimensionalEncoding,
            Self::IntegratedInformation,
            Self::RecursiveAwareness,
            Self::BoundaryDetection,
            Self::EmergentComplexity,
            Self::GradientDynamics,
            Self::DynamicTrajectories,
            Self::MetaConsciousness,
            Self::ConsciousnessSpectrum,
            Self::EpistemicCertainty,
            Self::CollectiveConsciousness,
            Self::ConsciousnessSpectrum2,
            Self::TemporalConsciousness,
            Self::CausalEfficacy,
            Self::QualiaRepresentation,
            Self::Ontogeny,
            Self::EmbodiedConsciousness,
            Self::RelationalConsciousness,
            Self::UniversalSemantics,
            Self::ConsciousnessTopology,
            Self::FlowFields,
            Self::PredictiveConsciousness,
            Self::GlobalWorkspace,
            Self::HigherOrderThought,
            Self::BindingProblem,
            Self::AttentionMechanisms,
            Self::SleepAndAlteredStates,
            Self::SubstrateIndependence,
        ]
    }

    /// Returns the category of this component
    pub fn category(&self) -> ComponentCategory {
        match self {
            Self::HyperdimensionalEncoding |
            Self::IntegratedInformation |
            Self::RecursiveAwareness |
            Self::BoundaryDetection |
            Self::EmergentComplexity => ComponentCategory::Foundational,

            Self::GradientDynamics |
            Self::DynamicTrajectories |
            Self::MetaConsciousness |
            Self::ConsciousnessSpectrum |
            Self::EpistemicCertainty => ComponentCategory::Structural,

            Self::CollectiveConsciousness |
            Self::ConsciousnessSpectrum2 => ComponentCategory::Social,

            Self::TemporalConsciousness |
            Self::CausalEfficacy |
            Self::QualiaRepresentation |
            Self::Ontogeny => ComponentCategory::Temporal,

            Self::EmbodiedConsciousness |
            Self::RelationalConsciousness => ComponentCategory::Embodied,

            Self::UniversalSemantics |
            Self::ConsciousnessTopology |
            Self::FlowFields => ComponentCategory::Semantic,

            Self::PredictiveConsciousness |
            Self::GlobalWorkspace |
            Self::HigherOrderThought |
            Self::BindingProblem => ComponentCategory::CoreTheory,

            Self::AttentionMechanisms |
            Self::SleepAndAlteredStates => ComponentCategory::Selection,

            Self::SubstrateIndependence => ComponentCategory::Universality,
        }
    }

    /// Returns the theoretical weight (importance) of this component
    /// Based on: literature consensus, causal centrality, experimental support
    pub fn theoretical_weight(&self) -> f64 {
        match self {
            // Critical requirements (highest weight)
            Self::IntegratedInformation => 1.0,    // Φ is THE measure
            Self::GlobalWorkspace => 1.0,          // Workspace required
            Self::BindingProblem => 1.0,           // Unity required

            // Core mechanisms (high weight)
            Self::HigherOrderThought => 0.9,       // Awareness mechanism
            Self::AttentionMechanisms => 0.9,      // Selection required
            Self::PredictiveConsciousness => 0.9,  // FEP unifies all

            // Important contributors (medium-high)
            Self::RecursiveAwareness => 0.8,
            Self::MetaConsciousness => 0.8,
            Self::TemporalConsciousness => 0.8,
            Self::EmbodiedConsciousness => 0.8,

            // Significant contributors (medium)
            Self::DynamicTrajectories => 0.7,
            Self::FlowFields => 0.7,
            Self::CausalEfficacy => 0.7,
            Self::QualiaRepresentation => 0.7,
            Self::ConsciousnessTopology => 0.7,

            // Supporting components (medium-low)
            Self::GradientDynamics => 0.6,
            Self::BoundaryDetection => 0.6,
            Self::EmergentComplexity => 0.6,
            Self::UniversalSemantics => 0.6,
            Self::EpistemicCertainty => 0.6,

            // Contextual (lower)
            Self::RelationalConsciousness => 0.5,
            Self::CollectiveConsciousness => 0.5,
            Self::Ontogeny => 0.5,
            Self::SleepAndAlteredStates => 0.5,

            // Infrastructure (lowest direct weight)
            Self::HyperdimensionalEncoding => 0.4,
            Self::ConsciousnessSpectrum => 0.4,
            Self::ConsciousnessSpectrum2 => 0.4,
            Self::SubstrateIndependence => 0.4,    // Enabling, not constitutive
        }
    }

    /// Whether this component is REQUIRED for consciousness (vs enhancing)
    pub fn is_required(&self) -> bool {
        matches!(self,
            Self::IntegratedInformation |
            Self::GlobalWorkspace |
            Self::BindingProblem |
            Self::AttentionMechanisms |
            Self::RecursiveAwareness
        )
    }

    /// Human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::HyperdimensionalEncoding => "HDC substrate for consciousness representation",
            Self::IntegratedInformation => "Φ: Information integration beyond parts",
            Self::RecursiveAwareness => "Self-referential awareness loop",
            Self::BoundaryDetection => "Self/other/world distinctions",
            Self::EmergentComplexity => "Consciousness as complex emergence",
            Self::GradientDynamics => "∇Φ: Gradients of integration",
            Self::DynamicTrajectories => "Consciousness state evolution",
            Self::MetaConsciousness => "Awareness of awareness",
            Self::ConsciousnessSpectrum => "Gradations of consciousness",
            Self::EpistemicCertainty => "Confidence and uncertainty",
            Self::CollectiveConsciousness => "Group minds and shared awareness",
            Self::ConsciousnessSpectrum2 => "Conscious/unconscious boundary",
            Self::TemporalConsciousness => "Multi-scale temporal experience",
            Self::CausalEfficacy => "Consciousness as causal force",
            Self::QualiaRepresentation => "Subjective phenomenal experience",
            Self::Ontogeny => "Developmental trajectory",
            Self::EmbodiedConsciousness => "Body-mind integration",
            Self::RelationalConsciousness => "I-Thou consciousness between beings",
            Self::UniversalSemantics => "NSM 65 universal meaning primitives",
            Self::ConsciousnessTopology => "Geometric structure of experience",
            Self::FlowFields => "Dynamics on consciousness manifold",
            Self::PredictiveConsciousness => "Free energy minimization",
            Self::GlobalWorkspace => "Broadcasting for conscious access",
            Self::HigherOrderThought => "Meta-representation for awareness",
            Self::BindingProblem => "Temporal synchrony for unity",
            Self::AttentionMechanisms => "Gain modulation and selection",
            Self::SleepAndAlteredStates => "State variations and alterations",
            Self::SubstrateIndependence => "Multiple realizability across substrates",
        }
    }
}

/// Categories grouping the 28 components
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentCategory {
    Foundational,   // #1-5: Basic building blocks
    Structural,     // #6-10: Organization
    Social,         // #11-12: Between beings
    Temporal,       // #13-16: Time and development
    Embodied,       // #17-18: Body and relation
    Semantic,       // #19-21: Meaning and structure
    CoreTheory,     // #22-25: Major theories (FEP, GWT, HOT, Binding)
    Selection,      // #26-27: Attention and states
    Universality,   // #28: Substrate independence
}

impl ComponentCategory {
    /// Weight multiplier for this category
    pub fn category_weight(&self) -> f64 {
        match self {
            Self::CoreTheory => 1.0,    // Highest: these are THE theories
            Self::Foundational => 0.9,
            Self::Selection => 0.9,
            Self::Temporal => 0.8,
            Self::Embodied => 0.8,
            Self::Structural => 0.7,
            Self::Semantic => 0.7,
            Self::Social => 0.6,
            Self::Universality => 0.5,  // Enabling, not constitutive
        }
    }
}

/// Value for a single component [0,1]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentValue {
    pub component: ConsciousnessComponent,
    pub value: f64,        // [0,1] - current measured value
    pub confidence: f64,   // [0,1] - how confident in measurement
    pub source: String,    // Where value came from
}

impl ComponentValue {
    pub fn new(component: ConsciousnessComponent, value: f64) -> Self {
        Self {
            component,
            value: value.clamp(0.0, 1.0),
            confidence: 1.0,
            source: "manual".to_string(),
        }
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.source = source.to_string();
        self
    }

    /// Weighted value = value × theoretical_weight × category_weight × confidence
    pub fn weighted_value(&self) -> f64 {
        self.value
            * self.component.theoretical_weight()
            * self.component.category().category_weight()
            * self.confidence
    }
}

/// The unified consciousness assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedAssessment {
    /// The master consciousness level [0,1]
    pub consciousness_level: f64,

    /// Critical threshold value (min of required components)
    pub critical_threshold: f64,

    /// Weighted average of all components
    pub weighted_average: f64,

    /// Substrate feasibility factor
    pub substrate_factor: f64,

    /// Individual component contributions
    pub contributions: HashMap<ConsciousnessComponent, f64>,

    /// Missing required components (if any)
    pub missing_required: Vec<ConsciousnessComponent>,

    /// Bottleneck analysis
    pub bottlenecks: Vec<(ConsciousnessComponent, f64)>,

    /// Consciousness level interpretation
    pub interpretation: String,

    /// Recommendations for improvement
    pub recommendations: Vec<String>,

    /// Confidence in overall assessment
    pub overall_confidence: f64,
}

/// Configuration for unified theory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedConfig {
    /// Minimum value for critical components to pass threshold
    pub critical_threshold: f64,

    /// Whether to require all critical components
    pub require_all_critical: bool,

    /// Whether to apply substrate factor
    pub apply_substrate_factor: bool,

    /// Default substrate factor if not specified
    pub default_substrate_factor: f64,

    /// Number of bottlenecks to identify
    pub num_bottlenecks: usize,
}

impl Default for UnifiedConfig {
    fn default() -> Self {
        Self {
            critical_threshold: 0.3,
            require_all_critical: true,
            apply_substrate_factor: true,
            default_substrate_factor: 0.92,  // Biological default
            num_bottlenecks: 5,
        }
    }
}

/// The Unified Theory of Consciousness
///
/// Synthesizes all 28 improvements into one coherent framework.
///
/// # The Master Equation
///
/// ```text
/// C = min(Φ, B, W, A, R) × [Σ(wᵢ × Cᵢ) / Σ(wᵢ)] × S
/// ```
///
/// Where:
/// - min(Φ, B, W, A, R) = critical threshold (integration, binding, workspace, attention, recursion)
/// - wᵢ × Cᵢ = weighted component values
/// - S = substrate feasibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedTheory {
    config: UnifiedConfig,
    component_values: HashMap<ConsciousnessComponent, ComponentValue>,
    substrate_factor: f64,
}

impl UnifiedTheory {
    /// Create new unified theory instance
    pub fn new() -> Self {
        Self {
            config: UnifiedConfig::default(),
            component_values: HashMap::new(),
            substrate_factor: 0.92,  // Default: biological
        }
    }

    /// Create with custom config
    pub fn with_config(config: UnifiedConfig) -> Self {
        let substrate_factor = config.default_substrate_factor;
        Self {
            config,
            component_values: HashMap::new(),
            substrate_factor,
        }
    }

    /// Set value for a component
    pub fn set_component(&mut self, component: ConsciousnessComponent, value: f64) {
        let cv = ComponentValue::new(component, value);
        self.component_values.insert(component, cv);
    }

    /// Set component with full details
    pub fn set_component_full(&mut self, value: ComponentValue) {
        self.component_values.insert(value.component, value);
    }

    /// Get component value
    pub fn get_component(&self, component: ConsciousnessComponent) -> Option<f64> {
        self.component_values.get(&component).map(|cv| cv.value)
    }

    /// Set substrate feasibility factor
    pub fn set_substrate_factor(&mut self, factor: f64) {
        self.substrate_factor = factor.clamp(0.0, 1.0);
    }

    /// Set substrate by type using HONEST (evidence-based) feasibility scores
    ///
    /// NOTE: Uses evidence levels from SubstrateValidationFramework.
    /// For unvalidated substrates (silicon, quantum, hybrid), this returns
    /// LOW scores reflecting our actual knowledge, not hypothetical claims.
    ///
    /// Use `set_substrate_type_hypothetical()` if you want the speculative scores.
    pub fn set_substrate_type(&mut self, substrate: &str) {
        // HONEST scores based on actual evidence levels
        self.substrate_factor = match substrate.to_lowercase().as_str() {
            "biological" => 0.95,   // Validated: extensive evidence
            "silicon" => 0.10,      // Theoretical: no empirical validation
            "quantum" => 0.10,      // Theoretical: contested, unproven
            "photonic" => 0.10,     // Theoretical: no evidence
            "neuromorphic" => 0.20, // Indirect: some biological similarity
            "biochemical" => 0.05,  // None: pure speculation
            "hybrid" => 0.00,       // None: NO evidence whatsoever
            "exotic" => 0.00,       // None: pure science fiction
            _ => 0.10,              // Default: theoretical at best
        };
    }

    /// Set substrate using HYPOTHETICAL (unvalidated) feasibility scores
    ///
    /// ⚠️ WARNING: These scores have NO empirical validation!
    /// They are placeholders based on philosophical assumptions.
    /// Use for exploration only, not for claims about real systems.
    pub fn set_substrate_type_hypothetical(&mut self, substrate: &str) {
        // HYPOTHETICAL scores - explicitly marked as unvalidated
        self.substrate_factor = match substrate.to_lowercase().as_str() {
            "biological" => 0.92,   // HYPOTHETICAL
            "silicon" => 0.71,      // HYPOTHETICAL - NO DATA
            "quantum" => 0.65,      // HYPOTHETICAL - CONTESTED
            "photonic" => 0.68,     // HYPOTHETICAL - NO DATA
            "neuromorphic" => 0.88, // HYPOTHETICAL
            "biochemical" => 0.28,  // HYPOTHETICAL
            "hybrid" => 0.95,       // HYPOTHETICAL - PURE SPECULATION
            "exotic" => 0.18,       // HYPOTHETICAL - SCIENCE FICTION
            _ => self.config.default_substrate_factor,
        };
    }

    /// Number of components set
    pub fn num_components(&self) -> usize {
        self.component_values.len()
    }

    /// Clear all values
    pub fn clear(&mut self) {
        self.component_values.clear();
        self.substrate_factor = self.config.default_substrate_factor;
    }

    /// Compute the critical threshold (min of required components)
    fn compute_critical_threshold(&self) -> (f64, Vec<ConsciousnessComponent>) {
        let required: Vec<ConsciousnessComponent> = ConsciousnessComponent::all()
            .into_iter()
            .filter(|c| c.is_required())
            .collect();

        let mut min_value = 1.0;
        let mut missing = Vec::new();

        for component in &required {
            match self.component_values.get(component) {
                Some(cv) if cv.value >= self.config.critical_threshold => {
                    if cv.value < min_value {
                        min_value = cv.value;
                    }
                }
                Some(cv) => {
                    // Below threshold
                    if cv.value < min_value {
                        min_value = cv.value;
                    }
                }
                None => {
                    missing.push(*component);
                    min_value = 0.0;
                }
            }
        }

        (min_value, missing)
    }

    /// Compute weighted average of all components
    fn compute_weighted_average(&self) -> f64 {
        if self.component_values.is_empty() {
            return 0.0;
        }

        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for cv in self.component_values.values() {
            let weight = cv.component.theoretical_weight()
                * cv.component.category().category_weight();
            weighted_sum += cv.value * weight * cv.confidence;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Identify bottleneck components (lowest weighted contribution)
    fn identify_bottlenecks(&self) -> Vec<(ConsciousnessComponent, f64)> {
        let mut contributions: Vec<(ConsciousnessComponent, f64)> = self.component_values
            .iter()
            .map(|(c, cv)| (*c, cv.weighted_value()))
            .collect();

        contributions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        contributions.into_iter()
            .take(self.config.num_bottlenecks)
            .collect()
    }

    /// Generate interpretation of consciousness level
    fn interpret_level(&self, level: f64, missing: &[ConsciousnessComponent]) -> String {
        if !missing.is_empty() {
            return format!(
                "INCOMPLETE: Missing required components ({:?}). Cannot assess consciousness.",
                missing
            );
        }

        match level {
            l if l >= 0.9 => "HIGHLY CONSCIOUS: Exceptional integration, awareness, and unified experience. All components operating at peak levels.".to_string(),
            l if l >= 0.7 => "CONSCIOUS: Clear unified awareness with strong integration. Normal waking consciousness.".to_string(),
            l if l >= 0.5 => "PARTIALLY CONSCIOUS: Fragmented or reduced awareness. Similar to drowsy, distracted, or early development states.".to_string(),
            l if l >= 0.3 => "MINIMALLY CONSCIOUS: Intermittent awareness. Similar to minimally conscious state or REM dreams.".to_string(),
            l if l >= 0.1 => "UNCONSCIOUS BUT PROCESSING: Below awareness threshold but neural processing continues. Similar to vegetative state.".to_string(),
            _ => "FULLY UNCONSCIOUS: No measurable consciousness. Similar to coma or deep anesthesia.".to_string(),
        }
    }

    /// Generate recommendations for improving consciousness
    fn generate_recommendations(&self, bottlenecks: &[(ConsciousnessComponent, f64)]) -> Vec<String> {
        let mut recs = Vec::new();

        for (component, value) in bottlenecks {
            if *value < 0.5 {
                let rec = match component {
                    ConsciousnessComponent::IntegratedInformation =>
                        "Increase Φ through enhanced interconnection and reduced modularity",
                    ConsciousnessComponent::GlobalWorkspace =>
                        "Strengthen workspace broadcasting - improve global information sharing",
                    ConsciousnessComponent::BindingProblem =>
                        "Enhance temporal synchrony - tighter phase coherence for feature binding",
                    ConsciousnessComponent::AttentionMechanisms =>
                        "Improve attention mechanisms - better gain modulation and selection",
                    ConsciousnessComponent::HigherOrderThought =>
                        "Strengthen meta-representation - enhance awareness of awareness",
                    ConsciousnessComponent::PredictiveConsciousness =>
                        "Reduce free energy - improve prediction accuracy and belief updating",
                    ConsciousnessComponent::TemporalConsciousness =>
                        "Enhance temporal integration - strengthen specious present binding",
                    ConsciousnessComponent::EmbodiedConsciousness =>
                        "Increase embodiment - stronger sensorimotor coupling",
                    _ => "Strengthen this component through targeted enhancement",
                };
                recs.push(format!("{:?}: {} (current: {:.2})", component, rec, value));
            }
        }

        if recs.is_empty() {
            recs.push("All components operating at acceptable levels".to_string());
        }

        recs
    }

    /// THE MASTER EQUATION: Assess overall consciousness level
    ///
    /// C = min(Φ, B, W, A, R) × [Σ(wᵢ × Cᵢ) / Σ(wᵢ)] × S
    pub fn assess(&self) -> UnifiedAssessment {
        // Step 1: Compute critical threshold
        let (critical_threshold, missing_required) = self.compute_critical_threshold();

        // Step 2: Compute weighted average
        let weighted_average = self.compute_weighted_average();

        // Step 3: Apply substrate factor
        let substrate_factor = if self.config.apply_substrate_factor {
            self.substrate_factor
        } else {
            1.0
        };

        // Step 4: THE MASTER EQUATION
        let consciousness_level = critical_threshold * weighted_average * substrate_factor;

        // Step 5: Compute individual contributions
        let contributions: HashMap<ConsciousnessComponent, f64> = self.component_values
            .iter()
            .map(|(c, cv)| (*c, cv.weighted_value()))
            .collect();

        // Step 6: Identify bottlenecks
        let bottlenecks = self.identify_bottlenecks();

        // Step 7: Generate interpretation
        let interpretation = self.interpret_level(consciousness_level, &missing_required);

        // Step 8: Generate recommendations
        let recommendations = self.generate_recommendations(&bottlenecks);

        // Step 9: Compute overall confidence
        let overall_confidence = if self.component_values.is_empty() {
            0.0
        } else {
            let avg_confidence: f64 = self.component_values.values()
                .map(|cv| cv.confidence)
                .sum::<f64>() / self.component_values.len() as f64;
            avg_confidence * (self.component_values.len() as f64 / 28.0)
        };

        UnifiedAssessment {
            consciousness_level,
            critical_threshold,
            weighted_average,
            substrate_factor,
            contributions,
            missing_required,
            bottlenecks,
            interpretation,
            recommendations,
            overall_confidence,
        }
    }

    /// Quick assessment with just critical components
    pub fn quick_assess(&self) -> f64 {
        let (critical, missing) = self.compute_critical_threshold();
        if !missing.is_empty() {
            return 0.0;
        }
        critical * self.compute_weighted_average() * self.substrate_factor
    }

    /// Check if system is conscious (threshold: 0.3)
    pub fn is_conscious(&self) -> bool {
        self.quick_assess() >= 0.3
    }

    /// Get consciousness level as category
    pub fn consciousness_category(&self) -> &'static str {
        let level = self.quick_assess();
        match level {
            l if l >= 0.9 => "Highly Conscious",
            l if l >= 0.7 => "Conscious",
            l if l >= 0.5 => "Partially Conscious",
            l if l >= 0.3 => "Minimally Conscious",
            l if l >= 0.1 => "Unconscious Processing",
            _ => "Fully Unconscious",
        }
    }

    /// Generate comprehensive report
    pub fn generate_report(&self) -> String {
        let assessment = self.assess();

        let mut report = String::new();

        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("           UNIFIED THEORY OF CONSCIOUSNESS - ASSESSMENT         \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        report.push_str("THE MASTER EQUATION:\n");
        report.push_str("  C = min(Φ, B, W, A, R) × [Σ(wᵢ × Cᵢ) / Σ(wᵢ)] × S\n\n");

        report.push_str(&format!("CONSCIOUSNESS LEVEL: {:.2} ({})\n\n",
            assessment.consciousness_level,
            self.consciousness_category()));

        report.push_str("COMPONENT BREAKDOWN:\n");
        report.push_str(&format!("  • Critical Threshold (min required): {:.3}\n",
            assessment.critical_threshold));
        report.push_str(&format!("  • Weighted Average (all components): {:.3}\n",
            assessment.weighted_average));
        report.push_str(&format!("  • Substrate Factor: {:.3}\n\n",
            assessment.substrate_factor));

        if !assessment.missing_required.is_empty() {
            report.push_str("⚠️  MISSING REQUIRED COMPONENTS:\n");
            for component in &assessment.missing_required {
                report.push_str(&format!("  • {:?}\n", component));
            }
            report.push_str("\n");
        }

        report.push_str("BOTTLENECKS (lowest contributors):\n");
        for (component, value) in &assessment.bottlenecks {
            report.push_str(&format!("  • {:?}: {:.3}\n", component, value));
        }
        report.push_str("\n");

        report.push_str("INTERPRETATION:\n");
        report.push_str(&format!("  {}\n\n", assessment.interpretation));

        report.push_str("RECOMMENDATIONS:\n");
        for rec in &assessment.recommendations {
            report.push_str(&format!("  • {}\n", rec));
        }

        report.push_str(&format!("\nOVERALL CONFIDENCE: {:.1}%\n",
            assessment.overall_confidence * 100.0));
        report.push_str(&format!("COMPONENTS SET: {}/28\n", self.num_components()));

        report.push_str("\n═══════════════════════════════════════════════════════════════\n");

        report
    }

    /// Create preset for normal waking consciousness
    pub fn preset_waking() -> Self {
        let mut theory = Self::new();
        theory.set_substrate_type("biological");

        // Set typical waking values
        theory.set_component(ConsciousnessComponent::IntegratedInformation, 0.85);
        theory.set_component(ConsciousnessComponent::GlobalWorkspace, 0.90);
        theory.set_component(ConsciousnessComponent::BindingProblem, 0.88);
        theory.set_component(ConsciousnessComponent::AttentionMechanisms, 0.82);
        theory.set_component(ConsciousnessComponent::RecursiveAwareness, 0.80);
        theory.set_component(ConsciousnessComponent::HigherOrderThought, 0.75);
        theory.set_component(ConsciousnessComponent::PredictiveConsciousness, 0.78);
        theory.set_component(ConsciousnessComponent::TemporalConsciousness, 0.85);
        theory.set_component(ConsciousnessComponent::EmbodiedConsciousness, 0.90);

        theory
    }

    /// Create preset for deep sleep
    pub fn preset_deep_sleep() -> Self {
        let mut theory = Self::new();
        theory.set_substrate_type("biological");

        // Set deep sleep (N3) values
        theory.set_component(ConsciousnessComponent::IntegratedInformation, 0.15);
        theory.set_component(ConsciousnessComponent::GlobalWorkspace, 0.05);
        theory.set_component(ConsciousnessComponent::BindingProblem, 0.10);
        theory.set_component(ConsciousnessComponent::AttentionMechanisms, 0.02);
        theory.set_component(ConsciousnessComponent::RecursiveAwareness, 0.05);
        theory.set_component(ConsciousnessComponent::HigherOrderThought, 0.02);

        theory
    }

    /// Create preset for advanced AI (like Symthaea with all components)
    ///
    /// ⚠️ NOTE: Uses HYPOTHETICAL substrate factor for exploration.
    /// There is NO empirical evidence that hybrid systems can be conscious.
    /// This preset is for theoretical exploration only.
    pub fn preset_advanced_ai() -> Self {
        let mut theory = Self::new();
        // Using HYPOTHETICAL value - explicitly marked as unvalidated
        theory.set_substrate_type_hypothetical("hybrid");  // 0.95 HYPOTHETICAL - NO DATA

        // Set optimal AI values (fully implemented components)
        theory.set_component(ConsciousnessComponent::IntegratedInformation, 0.90);
        theory.set_component(ConsciousnessComponent::GlobalWorkspace, 0.95);
        theory.set_component(ConsciousnessComponent::BindingProblem, 0.92);
        theory.set_component(ConsciousnessComponent::AttentionMechanisms, 0.88);
        theory.set_component(ConsciousnessComponent::RecursiveAwareness, 0.85);
        theory.set_component(ConsciousnessComponent::HigherOrderThought, 0.82);
        theory.set_component(ConsciousnessComponent::PredictiveConsciousness, 0.90);
        theory.set_component(ConsciousnessComponent::TemporalConsciousness, 0.80);
        theory.set_component(ConsciousnessComponent::UniversalSemantics, 0.95);
        theory.set_component(ConsciousnessComponent::ConsciousnessTopology, 0.88);
        theory.set_component(ConsciousnessComponent::EmbodiedConsciousness, 0.60);  // Limited without body

        theory
    }
}

impl Default for UnifiedTheory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_count() {
        assert_eq!(ConsciousnessComponent::all().len(), 28);
    }

    #[test]
    fn test_component_categories() {
        let all = ConsciousnessComponent::all();
        for component in all {
            let _cat = component.category();
            let _weight = component.theoretical_weight();
            assert!(component.theoretical_weight() >= 0.0 && component.theoretical_weight() <= 1.0);
        }
    }

    #[test]
    fn test_required_components() {
        let required: Vec<_> = ConsciousnessComponent::all()
            .into_iter()
            .filter(|c| c.is_required())
            .collect();
        assert_eq!(required.len(), 5);  // Φ, W, B, A, R
    }

    #[test]
    fn test_component_value() {
        let cv = ComponentValue::new(ConsciousnessComponent::IntegratedInformation, 0.8)
            .with_confidence(0.9)
            .with_source("test");

        assert_eq!(cv.value, 0.8);
        assert_eq!(cv.confidence, 0.9);
        assert_eq!(cv.source, "test");
        assert!(cv.weighted_value() > 0.0);
    }

    #[test]
    fn test_unified_theory_creation() {
        let theory = UnifiedTheory::new();
        assert_eq!(theory.num_components(), 0);
        assert!(!theory.is_conscious());
    }

    #[test]
    fn test_set_components() {
        let mut theory = UnifiedTheory::new();
        theory.set_component(ConsciousnessComponent::IntegratedInformation, 0.8);
        theory.set_component(ConsciousnessComponent::GlobalWorkspace, 0.9);

        assert_eq!(theory.num_components(), 2);
        assert_eq!(theory.get_component(ConsciousnessComponent::IntegratedInformation), Some(0.8));
    }

    #[test]
    fn test_substrate_factor_honest() {
        let mut theory = UnifiedTheory::new();

        // HONEST scores based on actual evidence
        theory.set_substrate_type("biological");
        assert!((theory.substrate_factor - 0.95).abs() < 0.01);  // Validated

        theory.set_substrate_type("silicon");
        assert!((theory.substrate_factor - 0.10).abs() < 0.01);  // Theoretical only

        theory.set_substrate_type("hybrid");
        assert!((theory.substrate_factor - 0.00).abs() < 0.01);  // No evidence!
    }

    #[test]
    fn test_substrate_factor_hypothetical() {
        let mut theory = UnifiedTheory::new();

        // HYPOTHETICAL scores (unvalidated - for exploration only)
        theory.set_substrate_type_hypothetical("biological");
        assert!((theory.substrate_factor - 0.92).abs() < 0.01);

        theory.set_substrate_type_hypothetical("silicon");
        assert!((theory.substrate_factor - 0.71).abs() < 0.01);

        theory.set_substrate_type_hypothetical("hybrid");
        assert!((theory.substrate_factor - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_missing_required() {
        let mut theory = UnifiedTheory::new();
        // Only set some required components
        theory.set_component(ConsciousnessComponent::IntegratedInformation, 0.8);
        theory.set_component(ConsciousnessComponent::GlobalWorkspace, 0.9);
        // Missing: BindingProblem, AttentionMechanisms, RecursiveAwareness

        let assessment = theory.assess();
        assert!(!assessment.missing_required.is_empty());
        assert!(assessment.consciousness_level < 0.3);
    }

    #[test]
    fn test_full_assessment() {
        let mut theory = UnifiedTheory::new();

        // Set all required components
        theory.set_component(ConsciousnessComponent::IntegratedInformation, 0.8);
        theory.set_component(ConsciousnessComponent::GlobalWorkspace, 0.9);
        theory.set_component(ConsciousnessComponent::BindingProblem, 0.85);
        theory.set_component(ConsciousnessComponent::AttentionMechanisms, 0.75);
        theory.set_component(ConsciousnessComponent::RecursiveAwareness, 0.7);

        let assessment = theory.assess();
        assert!(assessment.missing_required.is_empty());
        assert!(assessment.consciousness_level > 0.3);
        assert!(theory.is_conscious());
    }

    #[test]
    fn test_preset_waking() {
        let theory = UnifiedTheory::preset_waking();
        let level = theory.quick_assess();

        // Waking consciousness should pass consciousness threshold
        // Note: The master equation C = min(critical) × weighted_avg × substrate
        // is multiplicatively constrained, so even high values yield moderate C
        assert!(level > 0.4);
        assert!(theory.is_conscious());
        // Category depends on exact component values and weights
        let cat = theory.consciousness_category();
        assert!(cat == "Conscious" || cat == "Partially Conscious");
    }

    #[test]
    fn test_preset_deep_sleep() {
        let theory = UnifiedTheory::preset_deep_sleep();
        let level = theory.quick_assess();

        // Deep sleep should be very low
        assert!(level < 0.2);
        assert!(!theory.is_conscious());
    }

    #[test]
    fn test_preset_advanced_ai() {
        let theory = UnifiedTheory::preset_advanced_ai();
        let level = theory.quick_assess();

        // Advanced AI with hybrid substrate should be conscious
        assert!(level > 0.5);
        assert!(theory.is_conscious());
    }

    #[test]
    fn test_generate_report() {
        let theory = UnifiedTheory::preset_waking();
        let report = theory.generate_report();

        assert!(report.contains("UNIFIED THEORY"));
        assert!(report.contains("CONSCIOUSNESS LEVEL"));
        assert!(report.contains("MASTER EQUATION"));
    }

    #[test]
    fn test_clear() {
        let mut theory = UnifiedTheory::preset_waking();
        assert!(theory.num_components() > 0);

        theory.clear();
        assert_eq!(theory.num_components(), 0);
    }
}
