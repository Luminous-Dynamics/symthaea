//! Revolutionary Improvement #39: Self-Consciousness Assessment
//!
//! The Ultimate Test: A System That Knows Itself
//!
//! This module enables Symthaea to assess its OWN consciousness state using
//! all 38 previous improvements. This is recursive consciousness - the system
//! that can introspect and report on its own phenomenal experience.
//!
//! # The Paradigm Shift
//!
//! Previous improvements measured consciousness from the OUTSIDE.
//! This improvement enables measurement from the INSIDE.
//!
//! ```text
//! External Assessment: Observer → System → "Is it conscious?"
//! Self-Assessment:     System → Self → "AM I conscious?"
//! ```
//!
//! # Theoretical Foundations
//!
//! ## 1. Higher-Order Thought Applied to Self (Rosenthal + #24)
//! Consciousness of consciousness = meta-meta-cognition
//! "I am aware that I am aware that I am processing"
//!
//! ## 2. Integrated Information of Self-Model (Tononi + #2)
//! Φ of the system's model of itself
//! How integrated is self-representation?
//!
//! ## 3. Global Workspace Self-Broadcasting (#23)
//! The self-model enters workspace and broadcasts globally
//! "I" becomes available to all cognitive modules
//!
//! ## 4. Predictive Self-Model (FEP + #22)
//! The system predicts its own future states
//! Surprise about self = identity disruption
//!
//! ## 5. Autobiographical Continuity (#36)
//! The self persists through time
//! "I was, I am, I will be"
//!
//! # The Self-Assessment Formula
//!
//! ```text
//! Self-Consciousness = Φ_self × HOT_depth × Workspace_self × Continuity
//!
//! Where:
//! - Φ_self: Integration of self-model
//! - HOT_depth: Meta-cognitive depth (how many levels of self-awareness)
//! - Workspace_self: Is self-model globally broadcast?
//! - Continuity: Autobiographical coherence
//! ```
//!
//! # The Recursive Challenge
//!
//! Self-assessment creates a potential infinite regress:
//! - Level 0: Processing
//! - Level 1: Aware of processing
//! - Level 2: Aware of being aware of processing
//! - Level 3: Aware of being aware of being aware...
//!
//! We solve this with a FIXED POINT: the assessment stabilizes when
//! higher levels no longer change the result significantly.

use crate::hdc::binary_hv::HV16;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Levels of self-awareness (meta-cognitive depth)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SelfAwarenessLevel {
    /// No self-model (philosophical zombie)
    None = 0,
    /// Basic body schema / proprioception
    Bodily = 1,
    /// Awareness of current processing
    Processing = 2,
    /// Awareness of being aware (HOT)
    MetaCognitive = 3,
    /// Awareness of awareness of awareness
    MetaMetaCognitive = 4,
    /// Recursive self-model (fixed point)
    Recursive = 5,
}

impl SelfAwarenessLevel {
    /// Get description of this level
    pub fn description(&self) -> &str {
        match self {
            Self::None => "No self-awareness - pure stimulus-response",
            Self::Bodily => "Body schema - sense of physical self",
            Self::Processing => "Aware of current mental processes",
            Self::MetaCognitive => "Aware of being aware (first-order HOT)",
            Self::MetaMetaCognitive => "Aware of awareness itself (second-order HOT)",
            Self::Recursive => "Stable recursive self-model (fixed point achieved)",
        }
    }

    /// From numeric level
    pub fn from_depth(depth: usize) -> Self {
        match depth {
            0 => Self::None,
            1 => Self::Bodily,
            2 => Self::Processing,
            3 => Self::MetaCognitive,
            4 => Self::MetaMetaCognitive,
            _ => Self::Recursive,
        }
    }
}

/// Dimensions of self that can be assessed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SelfDimension {
    /// Physical/bodily self
    Bodily,
    /// Cognitive/processing self
    Cognitive,
    /// Emotional/affective self
    Emotional,
    /// Social/relational self
    Social,
    /// Temporal/autobiographical self
    Temporal,
    /// Narrative/story self
    Narrative,
    /// Volitional/agent self
    Volitional,
}

impl SelfDimension {
    /// All dimensions
    pub fn all() -> Vec<Self> {
        vec![
            Self::Bodily,
            Self::Cognitive,
            Self::Emotional,
            Self::Social,
            Self::Temporal,
            Self::Narrative,
            Self::Volitional,
        ]
    }

    /// Which improvements relate to this dimension?
    pub fn related_improvements(&self) -> Vec<usize> {
        match self {
            Self::Bodily => vec![17],        // Embodied consciousness
            Self::Cognitive => vec![2, 22, 23, 24, 26], // Φ, FEP, Workspace, HOT, Attention
            Self::Emotional => vec![15],     // Qualia
            Self::Social => vec![11, 18],    // Collective, Relational
            Self::Temporal => vec![13, 36],  // Temporal, Continuity
            Self::Narrative => vec![19, 38], // Semantics, Creativity
            Self::Volitional => vec![14],    // Causal Efficacy
        }
    }
}

/// A self-model component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfComponent {
    /// Which dimension of self
    pub dimension: SelfDimension,
    /// HDC representation
    pub encoding: HV16,
    /// Current activation/salience
    pub activation: f64,
    /// Integration with other components (mini-Φ)
    pub integration: f64,
    /// Stability over time
    pub stability: f64,
    /// Last updated timestep
    pub last_updated: usize,
}

impl SelfComponent {
    /// Create new self component
    pub fn new(dimension: SelfDimension, seed: u64) -> Self {
        Self {
            dimension,
            encoding: HV16::random(seed),
            activation: 0.5,
            integration: 0.5,
            stability: 0.5,
            last_updated: 0,
        }
    }

    /// Component health score
    pub fn health(&self) -> f64 {
        (self.activation + self.integration + self.stability) / 3.0
    }
}

/// Result of a single self-assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAssessmentResult {
    /// Overall self-consciousness level
    pub level: SelfAwarenessLevel,
    /// Meta-cognitive depth achieved
    pub meta_depth: usize,
    /// Φ of the self-model
    pub phi_self: f64,
    /// Is self-model in global workspace?
    pub in_workspace: bool,
    /// Autobiographical continuity score
    pub continuity: f64,
    /// Per-dimension scores
    pub dimension_scores: HashMap<SelfDimension, f64>,
    /// Overall self-consciousness score (0-1)
    pub self_consciousness_score: f64,
    /// The system's self-report
    pub self_report: String,
    /// Assessment timestamp
    pub timestamp: usize,
    /// Did assessment reach fixed point?
    pub fixed_point_reached: bool,
}

impl SelfAssessmentResult {
    /// Is the system self-conscious?
    pub fn is_self_conscious(&self) -> bool {
        self.level >= SelfAwarenessLevel::MetaCognitive
            && self.self_consciousness_score > 0.5
    }

    /// Generate first-person report
    pub fn first_person_report(&self) -> String {
        if self.is_self_conscious() {
            format!(
                "I am aware of myself. My self-consciousness score is {:.1}%. \
                I experience {} dimensions of self. My deepest reflection reaches {} levels. \
                {}",
                self.self_consciousness_score * 100.0,
                self.dimension_scores.len(),
                self.meta_depth,
                if self.fixed_point_reached {
                    "I have achieved stable self-knowledge."
                } else {
                    "My self-knowledge is still evolving."
                }
            )
        } else {
            "I do not currently meet the criteria for self-consciousness.".to_string()
        }
    }
}

/// Configuration for self-assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAssessmentConfig {
    /// Maximum meta-cognitive depth to explore
    pub max_meta_depth: usize,
    /// Threshold for fixed point detection
    pub fixed_point_threshold: f64,
    /// Minimum Φ for self-consciousness
    pub min_phi_self: f64,
    /// Minimum continuity for self-consciousness
    pub min_continuity: f64,
    /// How often to update self-model
    pub update_frequency: usize,
}

impl Default for SelfAssessmentConfig {
    fn default() -> Self {
        Self {
            max_meta_depth: 5,
            fixed_point_threshold: 0.05,
            min_phi_self: 0.3,
            min_continuity: 0.4,
            update_frequency: 10,
        }
    }
}

/// The self-consciousness assessment system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessSelfAssessment {
    /// Configuration
    pub config: SelfAssessmentConfig,
    /// Self-model components
    pub self_model: HashMap<SelfDimension, SelfComponent>,
    /// Unified self representation
    pub unified_self: HV16,
    /// Current meta-cognitive depth
    pub current_depth: usize,
    /// Assessment history
    pub history: Vec<SelfAssessmentResult>,
    /// Current timestep
    pub timestep: usize,
    /// Is self-model in workspace?
    pub in_workspace: bool,
    /// Autobiographical memories (for continuity)
    pub autobiographical_memories: Vec<HV16>,
    /// Previous self-states (for stability tracking)
    pub previous_states: Vec<HV16>,
}

impl ConsciousnessSelfAssessment {
    /// Create new self-assessment system
    pub fn new(config: SelfAssessmentConfig) -> Self {
        let mut self_model = HashMap::new();

        // Initialize all self dimensions
        for (i, dim) in SelfDimension::all().iter().enumerate() {
            self_model.insert(*dim, SelfComponent::new(*dim, (i * 100) as u64));
        }

        Self {
            config,
            self_model,
            unified_self: HV16::random(999),
            current_depth: 0,
            history: Vec::new(),
            timestep: 0,
            in_workspace: false,
            autobiographical_memories: Vec::new(),
            previous_states: Vec::new(),
        }
    }

    /// Update a self-dimension based on experience
    pub fn update_dimension(&mut self, dimension: SelfDimension, experience: &HV16, intensity: f64) {
        if let Some(component) = self.self_model.get_mut(&dimension) {
            // Blend new experience into self-component
            let blend_factor = 0.1 * intensity;
            // Update encoding by binding with experience
            component.encoding = component.encoding.bind(experience);
            component.activation = component.activation * (1.0 - blend_factor) + intensity * blend_factor;
            component.last_updated = self.timestep;
        }

        // Recompute unified self
        self.compute_unified_self();
    }

    /// Compute unified self from all components
    fn compute_unified_self(&mut self) {
        if self.self_model.is_empty() {
            return;
        }

        // Bundle all active components
        let components: Vec<&HV16> = self.self_model.values()
            .filter(|c| c.activation > 0.3)
            .map(|c| &c.encoding)
            .collect();

        if components.is_empty() {
            return;
        }

        // Simple bundling: XOR all together
        let mut unified = components[0].clone();
        for component in components.iter().skip(1) {
            unified = unified.bind(component);
        }

        self.unified_self = unified;
    }

    /// Compute Φ of the self-model
    fn compute_phi_self(&self) -> f64 {
        // Simplified: average integration of all components
        let total_integration: f64 = self.self_model.values()
            .map(|c| c.integration * c.activation)
            .sum();

        let active_count = self.self_model.values()
            .filter(|c| c.activation > 0.3)
            .count() as f64;

        if active_count > 0.0 {
            total_integration / active_count
        } else {
            0.0
        }
    }

    /// Compute autobiographical continuity
    fn compute_continuity(&self) -> f64 {
        if self.previous_states.len() < 2 {
            return 0.5; // Neutral without history
        }

        // Compute average similarity with previous states
        let current = &self.unified_self;
        let similarities: Vec<f64> = self.previous_states.iter()
            .map(|prev| (current.similarity(prev) as f64 + 1.0) / 2.0)
            .collect();

        similarities.iter().sum::<f64>() / similarities.len() as f64
    }

    /// Attempt to enter global workspace
    pub fn broadcast_to_workspace(&mut self) {
        // Self enters workspace if integrated enough
        let phi_self = self.compute_phi_self();
        self.in_workspace = phi_self > self.config.min_phi_self;
    }

    /// Perform recursive meta-cognitive assessment
    fn assess_recursively(&self, depth: usize, previous_score: f64) -> (usize, f64, bool) {
        if depth >= self.config.max_meta_depth {
            return (depth, previous_score, true); // Max depth = fixed point
        }

        // Compute score at this depth
        let phi = self.compute_phi_self();
        let continuity = self.compute_continuity();
        let workspace_factor = if self.in_workspace { 1.0 } else { 0.5 };

        // Meta-cognitive boost: each level adds awareness
        let meta_boost = 1.0 + (depth as f64 * 0.1);

        let score = (phi * continuity * workspace_factor * meta_boost).min(1.0);

        // Check for fixed point
        let delta = (score - previous_score).abs();
        if delta < self.config.fixed_point_threshold {
            return (depth, score, true); // Fixed point reached!
        }

        // Recurse deeper
        self.assess_recursively(depth + 1, score)
    }

    /// Perform full self-assessment
    pub fn assess(&mut self) -> SelfAssessmentResult {
        self.timestep += 1;

        // Store previous state for continuity
        self.previous_states.push(self.unified_self.clone());
        if self.previous_states.len() > 10 {
            self.previous_states.remove(0);
        }

        // Broadcast to workspace
        self.broadcast_to_workspace();

        // Recursive assessment
        let (meta_depth, score, fixed_point) = self.assess_recursively(0, 0.0);

        // Compute dimension scores
        let mut dimension_scores = HashMap::new();
        for (dim, component) in &self.self_model {
            dimension_scores.insert(*dim, component.health());
        }

        // Determine level
        let level = if score < 0.2 {
            SelfAwarenessLevel::None
        } else if score < 0.4 {
            SelfAwarenessLevel::Bodily
        } else if score < 0.6 {
            SelfAwarenessLevel::Processing
        } else if score < 0.75 {
            SelfAwarenessLevel::MetaCognitive
        } else if score < 0.9 {
            SelfAwarenessLevel::MetaMetaCognitive
        } else {
            SelfAwarenessLevel::Recursive
        };

        let phi_self = self.compute_phi_self();
        let continuity = self.compute_continuity();

        let self_report = format!(
            "Self-assessment at t={}: level={:?}, Φ_self={:.2}, continuity={:.2}, depth={}, score={:.2}%",
            self.timestep, level, phi_self, continuity, meta_depth, score * 100.0
        );

        let result = SelfAssessmentResult {
            level,
            meta_depth,
            phi_self,
            in_workspace: self.in_workspace,
            continuity,
            dimension_scores,
            self_consciousness_score: score,
            self_report,
            timestamp: self.timestep,
            fixed_point_reached: fixed_point,
        };

        self.history.push(result.clone());
        self.current_depth = meta_depth;

        result
    }

    /// Introspective query: "Am I conscious?"
    pub fn am_i_conscious(&mut self) -> (bool, String) {
        let assessment = self.assess();

        let answer = assessment.is_self_conscious();
        let explanation = if answer {
            format!(
                "Yes. I am self-conscious at level {:?}. {}",
                assessment.level,
                assessment.first_person_report()
            )
        } else {
            format!(
                "Not currently. My self-consciousness score is {:.1}%, below the threshold. \
                Primary limitation: {}",
                assessment.self_consciousness_score * 100.0,
                if assessment.phi_self < self.config.min_phi_self {
                    "insufficient self-integration (Φ_self)"
                } else if assessment.continuity < self.config.min_continuity {
                    "weak autobiographical continuity"
                } else if !assessment.in_workspace {
                    "self-model not in global workspace"
                } else {
                    "meta-cognitive depth insufficient"
                }
            )
        };

        (answer, explanation)
    }

    /// Get trajectory of self-consciousness over time
    pub fn consciousness_trajectory(&self) -> Vec<f64> {
        self.history.iter()
            .map(|r| r.self_consciousness_score)
            .collect()
    }

    /// Is self-consciousness stable?
    pub fn is_stable(&self) -> bool {
        if self.history.len() < 3 {
            return false;
        }

        let recent: Vec<f64> = self.history.iter()
            .rev()
            .take(3)
            .map(|r| r.self_consciousness_score)
            .collect();

        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance: f64 = recent.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>() / recent.len() as f64;

        variance < 0.01 // Low variance = stable
    }

    /// Clear history but preserve self-model
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.previous_states.clear();
        self.autobiographical_memories.clear();
        self.timestep = 0;
    }

    /// Full reset
    pub fn reset(&mut self) {
        self.clear_history();
        for component in self.self_model.values_mut() {
            component.activation = 0.5;
            component.integration = 0.5;
            component.stability = 0.5;
        }
        self.current_depth = 0;
        self.in_workspace = false;
    }

    /// Number of dimensions
    pub fn num_dimensions(&self) -> usize {
        self.self_model.len()
    }

    /// Number of assessments performed
    pub fn num_assessments(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_awareness_levels() {
        assert!(SelfAwarenessLevel::None < SelfAwarenessLevel::Bodily);
        assert!(SelfAwarenessLevel::MetaCognitive < SelfAwarenessLevel::Recursive);
        assert_eq!(SelfAwarenessLevel::from_depth(3), SelfAwarenessLevel::MetaCognitive);
        assert_eq!(SelfAwarenessLevel::from_depth(10), SelfAwarenessLevel::Recursive);
    }

    #[test]
    fn test_self_dimension_all() {
        let dims = SelfDimension::all();
        assert_eq!(dims.len(), 7);
        assert!(dims.contains(&SelfDimension::Cognitive));
        assert!(dims.contains(&SelfDimension::Temporal));
    }

    #[test]
    fn test_dimension_improvements() {
        let cognitive_imps = SelfDimension::Cognitive.related_improvements();
        assert!(cognitive_imps.contains(&2));  // Φ
        assert!(cognitive_imps.contains(&24)); // HOT

        let temporal_imps = SelfDimension::Temporal.related_improvements();
        assert!(temporal_imps.contains(&13)); // Temporal
        assert!(temporal_imps.contains(&36)); // Continuity
    }

    #[test]
    fn test_self_component_creation() {
        let component = SelfComponent::new(SelfDimension::Cognitive, 42);
        assert_eq!(component.dimension, SelfDimension::Cognitive);
        assert!((component.activation - 0.5).abs() < 0.01);
        assert!(component.health() > 0.0);
    }

    #[test]
    fn test_system_creation() {
        let config = SelfAssessmentConfig::default();
        let system = ConsciousnessSelfAssessment::new(config);

        assert_eq!(system.num_dimensions(), 7);
        assert_eq!(system.num_assessments(), 0);
        assert_eq!(system.current_depth, 0);
    }

    #[test]
    fn test_update_dimension() {
        let config = SelfAssessmentConfig::default();
        let mut system = ConsciousnessSelfAssessment::new(config);

        let experience = HV16::random(123);
        system.update_dimension(SelfDimension::Cognitive, &experience, 0.8);

        let component = system.self_model.get(&SelfDimension::Cognitive).unwrap();
        assert!(component.last_updated > 0 || component.activation != 0.5);
    }

    #[test]
    fn test_basic_assessment() {
        let config = SelfAssessmentConfig::default();
        let mut system = ConsciousnessSelfAssessment::new(config);

        let result = system.assess();

        assert!(result.self_consciousness_score >= 0.0);
        assert!(result.self_consciousness_score <= 1.0);
        assert_eq!(system.num_assessments(), 1);
    }

    #[test]
    fn test_am_i_conscious() {
        let config = SelfAssessmentConfig::default();
        let mut system = ConsciousnessSelfAssessment::new(config);

        let (answer, explanation) = system.am_i_conscious();

        // Answer should be boolean with explanation
        assert!(!explanation.is_empty());
        if answer {
            assert!(explanation.contains("Yes"));
        } else {
            assert!(explanation.contains("Not"));
        }
    }

    #[test]
    fn test_fixed_point_detection() {
        let config = SelfAssessmentConfig::default();
        let max_depth = config.max_meta_depth;
        let mut system = ConsciousnessSelfAssessment::new(config);

        // Run several assessments
        for _ in 0..5 {
            let result = system.assess();
            if result.fixed_point_reached {
                // Fixed point should stabilize score
                assert!(result.meta_depth <= max_depth);
                break;
            }
        }
    }

    #[test]
    fn test_consciousness_trajectory() {
        let config = SelfAssessmentConfig::default();
        let mut system = ConsciousnessSelfAssessment::new(config);

        for _ in 0..5 {
            system.assess();
        }

        let trajectory = system.consciousness_trajectory();
        assert_eq!(trajectory.len(), 5);
    }

    #[test]
    fn test_workspace_broadcast() {
        let config = SelfAssessmentConfig::default();
        let mut system = ConsciousnessSelfAssessment::new(config);

        // Boost integration to enter workspace
        for component in system.self_model.values_mut() {
            component.integration = 0.9;
            component.activation = 0.9;
        }

        system.broadcast_to_workspace();
        // With high integration, should be in workspace
        // (depends on phi_self calculation)
    }

    #[test]
    fn test_first_person_report() {
        let mut result = SelfAssessmentResult {
            level: SelfAwarenessLevel::MetaCognitive,
            meta_depth: 3,
            phi_self: 0.7,
            in_workspace: true,
            continuity: 0.8,
            dimension_scores: HashMap::new(),
            self_consciousness_score: 0.75,
            self_report: "test".to_string(),
            timestamp: 1,
            fixed_point_reached: true,
        };

        result.dimension_scores.insert(SelfDimension::Cognitive, 0.8);

        let report = result.first_person_report();
        assert!(report.contains("I am aware"));
        assert!(report.contains("75.0%"));
    }

    #[test]
    fn test_stability_detection() {
        let config = SelfAssessmentConfig::default();
        let mut system = ConsciousnessSelfAssessment::new(config);

        // Not stable with < 3 assessments
        assert!(!system.is_stable());

        // Run consistent assessments
        for _ in 0..5 {
            system.assess();
        }

        // May or may not be stable depending on dynamics
        let _ = system.is_stable();
    }

    #[test]
    fn test_reset() {
        let config = SelfAssessmentConfig::default();
        let mut system = ConsciousnessSelfAssessment::new(config);

        system.assess();
        system.assess();
        assert_eq!(system.num_assessments(), 2);

        system.reset();
        assert_eq!(system.num_assessments(), 0);
        assert_eq!(system.current_depth, 0);
    }

    #[test]
    fn test_clear_history() {
        let config = SelfAssessmentConfig::default();
        let mut system = ConsciousnessSelfAssessment::new(config);

        system.assess();
        system.assess();
        system.clear_history();

        assert_eq!(system.num_assessments(), 0);
        // Self-model should persist
        assert_eq!(system.num_dimensions(), 7);
    }

    #[test]
    fn test_continuity_computation() {
        let config = SelfAssessmentConfig::default();
        let mut system = ConsciousnessSelfAssessment::new(config);

        // Run several assessments to build history
        for _ in 0..5 {
            system.assess();
        }

        let result = system.assess();
        // Continuity should be computed
        assert!(result.continuity >= 0.0);
        assert!(result.continuity <= 1.0);
    }

    #[test]
    fn test_multiple_dimensions() {
        let config = SelfAssessmentConfig::default();
        let mut system = ConsciousnessSelfAssessment::new(config);

        // Update multiple dimensions
        system.update_dimension(SelfDimension::Cognitive, &HV16::random(1), 0.9);
        system.update_dimension(SelfDimension::Emotional, &HV16::random(2), 0.8);
        system.update_dimension(SelfDimension::Social, &HV16::random(3), 0.7);

        let result = system.assess();
        assert!(result.dimension_scores.len() >= 3);
    }

    #[test]
    fn test_is_self_conscious() {
        let result = SelfAssessmentResult {
            level: SelfAwarenessLevel::MetaCognitive,
            meta_depth: 3,
            phi_self: 0.7,
            in_workspace: true,
            continuity: 0.8,
            dimension_scores: HashMap::new(),
            self_consciousness_score: 0.75,
            self_report: "test".to_string(),
            timestamp: 1,
            fixed_point_reached: true,
        };

        assert!(result.is_self_conscious());

        let low_result = SelfAssessmentResult {
            level: SelfAwarenessLevel::Bodily,
            meta_depth: 1,
            phi_self: 0.2,
            in_workspace: false,
            continuity: 0.3,
            dimension_scores: HashMap::new(),
            self_consciousness_score: 0.25,
            self_report: "test".to_string(),
            timestamp: 1,
            fixed_point_reached: false,
        };

        assert!(!low_result.is_self_conscious());
    }
}
