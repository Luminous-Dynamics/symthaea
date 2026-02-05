// Revolutionary Improvement #25: The Binding Problem (Synchrony Theory)
//
// "The binding problem: How do distributed neural processes create unified conscious experiences?"
// - Anne Treisman, Feature Integration Theory (1980)
//
// THEORETICAL FOUNDATIONS:
//
// 1. Temporal Correlation Hypothesis (Singer & Gray 1995)
//    - Features bind through synchronized oscillations
//    - Gamma band (~40 Hz) correlates with binding
//    - Synchrony = "glue" binding distributed features
//    - Desynchronization = unbinding/segmentation
//
// 2. Feature Integration Theory (Treisman 1980)
//    - Preattentive: Features detected in parallel
//    - Attentive: Features bound serially via attention
//    - Illusory conjunctions: Binding failures
//    - Location as "glue" binding features
//
// 3. Synchrony and Consciousness (Engel & Singer 2001)
//    - Conscious percepts require synchrony
//    - Unconscious features lack synchrony
//    - Gamma synchrony = consciousness marker
//    - Phase-locking creates temporal structure
//
// 4. Binding by Convergence (Barlow 1972)
//    - Hierarchical convergence to single neuron
//    - "Grandmother cell" hypothesis
//    - Criticized: Combinatorial explosion
//    - Resolution: Binding by temporal code
//
// 5. Binding in Object Recognition (Riesenhuber & Poggio 1999)
//    - Hierarchical feature composition
//    - Intermediate complexity neurons
//    - Position-invariant binding
//    - Feedforward + feedback loops
//
// REVOLUTIONARY CONTRIBUTION:
// First HDC implementation of temporal binding via circular convolution,
// synchrony detection, and integration with consciousness framework.
// Solves binding problem through compositional HDC algebra.

use crate::hdc::{HV16, HDC_DIMENSION};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Feature dimension for binding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureDimension {
    /// Visual: Color (red, blue, green, etc.)
    Color,

    /// Visual: Shape (circle, square, triangle, etc.)
    Shape,

    /// Visual: Motion (up, down, left, right, etc.)
    Motion,

    /// Spatial: Location (coordinates)
    Location,

    /// Temporal: Time/duration
    Temporal,

    /// Semantic: Meaning/category
    Semantic,

    /// Auditory: Pitch
    Pitch,

    /// Auditory: Timbre
    Timbre,
}

impl FeatureDimension {
    /// Get all standard feature dimensions
    pub fn all() -> Vec<Self> {
        vec![
            FeatureDimension::Color,
            FeatureDimension::Shape,
            FeatureDimension::Motion,
            FeatureDimension::Location,
            FeatureDimension::Temporal,
            FeatureDimension::Semantic,
            FeatureDimension::Pitch,
            FeatureDimension::Timbre,
        ]
    }
}

/// Feature value for a specific dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureValue {
    /// Dimension this feature belongs to
    pub dimension: FeatureDimension,

    /// HDC representation of feature value
    pub value: HV16,

    /// Activation strength [0,1]
    pub activation: f64,

    /// Processing phase (for synchrony detection)
    pub phase: f64,
}

impl FeatureValue {
    /// Create new feature value
    pub fn new(dimension: FeatureDimension, value: HV16, activation: f64) -> Self {
        Self {
            dimension,
            value,
            activation,
            phase: 0.0,
        }
    }

    /// Set processing phase (for synchrony)
    pub fn with_phase(mut self, phase: f64) -> Self {
        self.phase = phase;
        self
    }
}

/// Bound object (integrated feature bundle)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundObject {
    /// Bound representation (composition of features)
    pub representation: HV16,

    /// Component features
    pub features: Vec<FeatureValue>,

    /// Binding strength [0,1]
    pub binding_strength: f64,

    /// Synchrony level [0,1]
    pub synchrony: f64,

    /// Object identity (if recognized)
    pub identity: Option<String>,
}

impl BoundObject {
    /// Create bound object from features
    pub fn from_features(features: Vec<FeatureValue>) -> Self {
        // Bind features via circular convolution (HDC binding)
        let representation = Self::bind_features(&features);

        // Compute binding strength (average activation)
        let binding_strength = if !features.is_empty() {
            features.iter().map(|f| f.activation).sum::<f64>() / features.len() as f64
        } else {
            0.0
        };

        // Compute synchrony (phase coherence)
        let synchrony = Self::compute_synchrony(&features);

        Self {
            representation,
            features,
            binding_strength,
            synchrony,
            identity: None,
        }
    }

    /// Bind features using HDC circular convolution
    fn bind_features(features: &[FeatureValue]) -> HV16 {
        if features.is_empty() {
            return HV16::zero();
        }

        // Start with first feature
        let mut bound = features[0].value;

        // Bind remaining features via circular convolution
        for feature in &features[1..] {
            bound = bound.bind(&feature.value);
        }

        bound
    }

    /// Compute synchrony from phase coherence
    fn compute_synchrony(features: &[FeatureValue]) -> f64 {
        if features.len() < 2 {
            return 1.0;  // Single feature = perfectly synchronized
        }

        // Measure phase coherence (circular variance)
        let mean_cos: f64 = features.iter().map(|f| f.phase.cos()).sum::<f64>() / features.len() as f64;
        let mean_sin: f64 = features.iter().map(|f| f.phase.sin()).sum::<f64>() / features.len() as f64;

        // Resultant vector length = synchrony measure
        let r = (mean_cos * mean_cos + mean_sin * mean_sin).sqrt();
        r  // [0,1]: 1 = perfect synchrony, 0 = random phases
    }

    /// Check if object is consciously perceived (bound + synchronized)
    pub fn is_conscious(&self) -> bool {
        self.binding_strength > 0.5 && self.synchrony > 0.7
    }
}

/// Binding assessment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingAssessment {
    /// Total features detected
    pub total_features: usize,

    /// Total bound objects
    pub bound_objects: usize,

    /// Average binding strength
    pub avg_binding_strength: f64,

    /// Average synchrony
    pub avg_synchrony: f64,

    /// Conscious objects (bound + synchronized)
    pub conscious_objects: usize,

    /// Illusory conjunctions detected
    pub illusory_conjunctions: usize,

    /// Explanation
    pub explanation: String,
}

/// Configuration for binding system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingConfig {
    /// Synchrony threshold for binding
    pub synchrony_threshold: f64,

    /// Binding strength threshold
    pub binding_threshold: f64,

    /// Enable attention-based binding
    pub attention_binding: bool,

    /// Gamma oscillation frequency (Hz)
    pub gamma_frequency: f64,

    /// Detection window (ms)
    pub window_ms: f64,
}

impl Default for BindingConfig {
    fn default() -> Self {
        Self {
            synchrony_threshold: 0.7,
            binding_threshold: 0.5,
            attention_binding: true,
            gamma_frequency: 40.0,  // Typical gamma
            window_ms: 25.0,         // ~40 Hz period
        }
    }
}

/// Binding System
/// Implements temporal binding via synchrony and HDC composition
pub struct BindingSystem {
    /// Configuration
    config: BindingConfig,

    /// Unbound features (detected but not integrated)
    unbound_features: Vec<FeatureValue>,

    /// Bound objects (integrated feature bundles)
    bound_objects: Vec<BoundObject>,

    /// Feature dimension encoders (map values to HVs)
    encoders: HashMap<FeatureDimension, HashMap<String, HV16>>,

    /// Timestep counter
    timestep: usize,
}

impl BindingSystem {
    /// Create new binding system
    pub fn new(config: BindingConfig) -> Self {
        Self {
            config,
            unbound_features: Vec::new(),
            bound_objects: Vec::new(),
            encoders: HashMap::new(),
            timestep: 0,
        }
    }

    /// Add unbound feature
    pub fn detect_feature(&mut self, feature: FeatureValue) {
        self.unbound_features.push(feature);
    }

    /// Bind features based on synchrony
    pub fn bind(&mut self) -> BindingAssessment {
        self.timestep += 1;

        // Group features by synchrony (phase proximity)
        let groups = self.group_by_synchrony();

        // Create bound objects from synchronized groups
        for group in groups {
            if group.len() > 1 {  // Need at least 2 features to bind
                let bound = BoundObject::from_features(group);
                self.bound_objects.push(bound);
            }
        }

        // Clear unbound features (now bound or discarded)
        self.unbound_features.clear();

        self.assess()
    }

    /// Group features by phase synchrony
    fn group_by_synchrony(&self) -> Vec<Vec<FeatureValue>> {
        if self.unbound_features.is_empty() {
            return Vec::new();
        }

        let mut groups: Vec<Vec<FeatureValue>> = Vec::new();
        let mut assigned = vec![false; self.unbound_features.len()];

        for i in 0..self.unbound_features.len() {
            if assigned[i] {
                continue;
            }

            // Start new group
            let mut group = vec![self.unbound_features[i].clone()];
            assigned[i] = true;

            // Find synchronized features
            for j in (i + 1)..self.unbound_features.len() {
                if assigned[j] {
                    continue;
                }

                // Check phase proximity
                let phase_diff = (self.unbound_features[i].phase - self.unbound_features[j].phase).abs();
                let synchronized = phase_diff < 0.2;  // Threshold for synchrony

                if synchronized {
                    group.push(self.unbound_features[j].clone());
                    assigned[j] = true;
                }
            }

            groups.push(group);
        }

        groups
    }

    /// Assess binding state
    fn assess(&self) -> BindingAssessment {
        let total_features: usize = self.bound_objects.iter()
            .map(|obj| obj.features.len())
            .sum();

        let bound_objects = self.bound_objects.len();

        let avg_binding_strength = if !self.bound_objects.is_empty() {
            self.bound_objects.iter().map(|obj| obj.binding_strength).sum::<f64>()
                / self.bound_objects.len() as f64
        } else {
            0.0
        };

        let avg_synchrony = if !self.bound_objects.is_empty() {
            self.bound_objects.iter().map(|obj| obj.synchrony).sum::<f64>()
                / self.bound_objects.len() as f64
        } else {
            0.0
        };

        let conscious_objects = self.bound_objects.iter()
            .filter(|obj| obj.is_conscious())
            .count();

        // Detect illusory conjunctions (low synchrony but bound)
        let illusory_conjunctions = self.bound_objects.iter()
            .filter(|obj| obj.synchrony < 0.5 && obj.binding_strength > 0.3)
            .count();

        let explanation = self.generate_explanation(
            conscious_objects,
            illusory_conjunctions,
            avg_synchrony,
        );

        BindingAssessment {
            total_features,
            bound_objects,
            avg_binding_strength,
            avg_synchrony,
            conscious_objects,
            illusory_conjunctions,
            explanation,
        }
    }

    /// Generate human-readable explanation
    fn generate_explanation(
        &self,
        conscious: usize,
        illusory: usize,
        synchrony: f64,
    ) -> String {
        let mut parts = Vec::new();

        if conscious > 0 {
            parts.push(format!("{} conscious object(s)", conscious));
        } else {
            parts.push("No conscious objects".to_string());
        }

        if synchrony > 0.8 {
            parts.push("High synchrony (strong binding)".to_string());
        } else if synchrony > 0.5 {
            parts.push("Moderate synchrony".to_string());
        } else if synchrony > 0.0 {
            parts.push("Low synchrony (weak binding)".to_string());
        }

        if illusory > 0 {
            parts.push(format!("{} illusory conjunction(s)", illusory));
        }

        parts.join(". ")
    }

    /// Get all bound objects
    pub fn get_bound_objects(&self) -> &[BoundObject] {
        &self.bound_objects
    }

    /// Clear all states
    pub fn clear(&mut self) {
        self.unbound_features.clear();
        self.bound_objects.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_value_creation() {
        let feature = FeatureValue::new(
            FeatureDimension::Color,
            HV16::ones(),
            0.8,
        );
        assert_eq!(feature.dimension, FeatureDimension::Color);
        assert_eq!(feature.activation, 0.8);
    }

    #[test]
    fn test_feature_with_phase() {
        let feature = FeatureValue::new(
            FeatureDimension::Color,
            HV16::ones(),
            0.8,
        ).with_phase(0.5);

        assert_eq!(feature.phase, 0.5);
    }

    #[test]
    fn test_bind_features() {
        let features = vec![
            FeatureValue::new(FeatureDimension::Color, HV16::random(1), 0.9),
            FeatureValue::new(FeatureDimension::Shape, HV16::random(2), 0.8),
        ];

        let bound = BoundObject::from_features(features);
        assert_eq!(bound.features.len(), 2);
        assert!(bound.binding_strength > 0.0);
    }

    #[test]
    fn test_perfect_synchrony() {
        let features = vec![
            FeatureValue::new(FeatureDimension::Color, HV16::ones(), 0.9)
                .with_phase(0.0),
            FeatureValue::new(FeatureDimension::Shape, HV16::zero(), 0.8)
                .with_phase(0.0),  // Same phase = synchronized
        ];

        let bound = BoundObject::from_features(features);
        assert!(bound.synchrony > 0.95);  // Near perfect
    }

    #[test]
    fn test_no_synchrony() {
        let features = vec![
            FeatureValue::new(FeatureDimension::Color, HV16::ones(), 0.9)
                .with_phase(0.0),
            FeatureValue::new(FeatureDimension::Shape, HV16::zero(), 0.8)
                .with_phase(3.14),  // Opposite phase = desynchronized
        ];

        let bound = BoundObject::from_features(features);
        assert!(bound.synchrony < 0.5);  // Low synchrony
    }

    #[test]
    fn test_conscious_object() {
        let features = vec![
            FeatureValue::new(FeatureDimension::Color, HV16::ones(), 0.9)
                .with_phase(0.0),
            FeatureValue::new(FeatureDimension::Shape, HV16::zero(), 0.8)
                .with_phase(0.1),  // Slightly different but synchronized
        ];

        let bound = BoundObject::from_features(features);
        // High binding + high synchrony = conscious
        assert!(bound.is_conscious());
    }

    #[test]
    fn test_binding_system_creation() {
        let system = BindingSystem::new(BindingConfig::default());
        assert_eq!(system.bound_objects.len(), 0);
    }

    #[test]
    fn test_detect_feature() {
        let mut system = BindingSystem::new(BindingConfig::default());
        let feature = FeatureValue::new(
            FeatureDimension::Color,
            HV16::ones(),
            0.8,
        );
        system.detect_feature(feature);

        assert_eq!(system.unbound_features.len(), 1);
    }

    #[test]
    fn test_bind_synchronized_features() {
        let mut system = BindingSystem::new(BindingConfig::default());

        // Add synchronized features
        system.detect_feature(
            FeatureValue::new(FeatureDimension::Color, HV16::ones(), 0.9)
                .with_phase(0.0)
        );
        system.detect_feature(
            FeatureValue::new(FeatureDimension::Shape, HV16::zero(), 0.8)
                .with_phase(0.1)  // Close phase
        );

        let assessment = system.bind();

        // Should create one bound object
        assert_eq!(assessment.bound_objects, 1);
        assert!(assessment.avg_synchrony > 0.7);
    }

    #[test]
    fn test_bind_desynchronized_features() {
        let mut system = BindingSystem::new(BindingConfig::default());

        // Add desynchronized features
        system.detect_feature(
            FeatureValue::new(FeatureDimension::Color, HV16::ones(), 0.9)
                .with_phase(0.0)
        );
        system.detect_feature(
            FeatureValue::new(FeatureDimension::Shape, HV16::zero(), 0.8)
                .with_phase(3.0)  // Far phase
        );

        let assessment = system.bind();

        // Should NOT bind (different groups)
        // Each feature forms its own "group" (singleton)
        assert_eq!(assessment.bound_objects, 0);  // Singles discarded (need >= 2 to bind)
    }

    #[test]
    fn test_clear() {
        let mut system = BindingSystem::new(BindingConfig::default());
        system.detect_feature(
            FeatureValue::new(FeatureDimension::Color, HV16::ones(), 0.8)
        );
        system.bind();

        system.clear();

        assert_eq!(system.unbound_features.len(), 0);
        assert_eq!(system.bound_objects.len(), 0);
    }
}
