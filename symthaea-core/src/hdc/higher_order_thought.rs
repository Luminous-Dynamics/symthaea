// Revolutionary Improvement #24: Higher-Order Consciousness (HOT Theory)
//
// "A mental state is conscious if and only if one has a higher-order thought about it."
// - David Rosenthal, Higher-Order Thought Theory (1986)
//
// THEORETICAL FOUNDATIONS:
//
// 1. Higher-Order Thought (HOT) Theory (Rosenthal 1986, 2005)
//    - First-order state: Unconscious perception/thought
//    - Higher-order thought: Thought ABOUT first-order state
//    - Consciousness = having HOT about mental state
//    - Self-awareness = HOT about HOT (recursive)
//
// 2. Transitivity Principle (Rosenthal 1997)
//    - If HOT makes state conscious, HOT itself need not be conscious
//    - Avoids infinite regress: HOT can be tacit/implicit
//    - Only explicit HOTs (with HOT about them) are conscious
//
// 3. Misrepresentation Theory (Rosenthal 2011)
//    - HOTs can misrepresent first-order states
//    - Explains illusions, confabulation
//    - Consciousness = what HOT represents, not actual state
//
// 4. Dispositional vs Actualist HOT (Carruthers 2000)
//    - Dispositional: Potential to form HOT (not conscious yet)
//    - Actualist: Actual HOT exists (conscious now)
//    - Consciousness requires actual HOT
//
// 5. Meta-Representation (Nichols & Stich 2003)
//    - Mental state = representation
//    - HOT = meta-representation (representation of representation)
//    - Recursive: Meta-meta-representation → infinite tower
//
// REVOLUTIONARY CONTRIBUTION:
// First HDC implementation of Higher-Order Thought Theory with explicit
// meta-representational hierarchy, misrepresentation detection, and
// integration with Global Workspace Theory.

use crate::hdc::{HV16, HDC_DIMENSION};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Order of mental state representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum RepresentationOrder {
    /// Zero-order: No representation (inanimate)
    ZeroOrder = 0,

    /// First-order: Represents world (unconscious in isolation)
    FirstOrder = 1,

    /// Second-order: Represents first-order states (conscious!)
    SecondOrder = 2,

    /// Third-order: Represents second-order states (meta-conscious)
    ThirdOrder = 3,

    /// Fourth-order and beyond (philosophical reflection)
    HigherOrder = 4,
}

/// Mental state that can be represented
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MentalState {
    /// Content representation (HDC vector)
    pub content: Vec<HV16>,

    /// Order of this state
    pub order: RepresentationOrder,

    /// What this state represents (if higher-order)
    pub represents: Option<Box<MentalState>>,

    /// Confidence in representation
    pub confidence: f64,

    /// Source module
    pub source: String,

    /// Time created
    pub timestamp: usize,
}

impl MentalState {
    /// Create first-order state (represents world)
    pub fn first_order(content: Vec<HV16>, source: String) -> Self {
        Self {
            content,
            order: RepresentationOrder::FirstOrder,
            represents: None,
            confidence: 1.0,
            source,
            timestamp: 0,
        }
    }

    /// Create higher-order state (represents other state)
    pub fn higher_order(target: MentalState, content: Vec<HV16>, source: String) -> Self {
        let order = match target.order {
            RepresentationOrder::ZeroOrder => RepresentationOrder::FirstOrder,
            RepresentationOrder::FirstOrder => RepresentationOrder::SecondOrder,
            RepresentationOrder::SecondOrder => RepresentationOrder::ThirdOrder,
            RepresentationOrder::ThirdOrder | RepresentationOrder::HigherOrder =>
                RepresentationOrder::HigherOrder,
        };

        Self {
            content,
            order,
            represents: Some(Box::new(target)),
            confidence: 1.0,
            source,
            timestamp: 0,
        }
    }

    /// Check if this state is conscious (has HOT about it)
    pub fn is_conscious(&self) -> bool {
        self.order >= RepresentationOrder::SecondOrder
    }

    /// Get representational depth (how many levels)
    pub fn depth(&self) -> usize {
        match &self.represents {
            None => 1,
            Some(inner) => 1 + inner.depth(),
        }
    }

    /// Check for misrepresentation
    pub fn is_misrepresented(&self) -> bool {
        if let Some(target) = &self.represents {
            // Compare representation content to actual target
            // Simplified: check if very different
            let similarity = self.similarity(&self.content, &target.content);
            similarity < 0.7  // Threshold for misrepresentation
        } else {
            false
        }
    }

    /// Compute similarity between HDC vectors
    fn similarity(&self, a: &[HV16], b: &[HV16]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }

        let matches = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
        matches as f64 / a.len() as f64
    }
}

/// Configuration for HOT system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HOTConfig {
    /// Enable automatic HOT generation
    pub auto_generate_hots: bool,

    /// Maximum representational depth
    pub max_depth: usize,

    /// Threshold for forming HOT
    pub hot_threshold: f64,

    /// Enable misrepresentation detection
    pub detect_misrepresentation: bool,

    /// Tacit vs explicit HOTs
    pub require_explicit_hots: bool,
}

impl Default for HOTConfig {
    fn default() -> Self {
        Self {
            auto_generate_hots: true,
            max_depth: 3,             // Up to third-order
            hot_threshold: 0.5,       // Moderate threshold
            detect_misrepresentation: true,
            require_explicit_hots: false,  // Allow tacit HOTs
        }
    }
}

/// Assessment of higher-order consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HOTAssessment {
    /// Total mental states
    pub total_states: usize,

    /// First-order states (unconscious)
    pub first_order_count: usize,

    /// Second-order states (conscious)
    pub second_order_count: usize,

    /// Third-order and higher (meta-conscious)
    pub higher_order_count: usize,

    /// Consciousness ratio (conscious / total)
    pub consciousness_ratio: f64,

    /// Self-awareness detected (HOT about HOT)
    pub self_awareness: bool,

    /// Misrepresentations detected
    pub misrepresentations: usize,

    /// Average representational depth
    pub avg_depth: f64,

    /// Explanation
    pub explanation: String,
}

/// Higher-Order Thought System
/// Implements Rosenthal's HOT Theory with meta-representational hierarchy
pub struct HigherOrderThoughtSystem {
    /// Configuration
    config: HOTConfig,

    /// First-order states (unconscious perceptions/thoughts)
    first_order: Vec<MentalState>,

    /// Second-order states (HOTs making first-order conscious)
    second_order: Vec<MentalState>,

    /// Third-order and higher (meta-consciousness)
    higher_order: Vec<MentalState>,

    /// History of conscious states
    history: VecDeque<MentalState>,

    /// Timestep counter
    timestep: usize,
}

impl HigherOrderThoughtSystem {
    /// Create new HOT system
    pub fn new(config: HOTConfig) -> Self {
        Self {
            config,
            first_order: Vec::new(),
            second_order: Vec::new(),
            higher_order: Vec::new(),
            history: VecDeque::new(),
            timestep: 0,
        }
    }

    /// Add first-order state (unconscious perception/thought)
    pub fn perceive(&mut self, content: Vec<HV16>, source: String) {
        let state = MentalState::first_order(content, source);
        self.first_order.push(state);
    }

    /// Form higher-order thought about a state
    pub fn form_hot(&mut self, target_idx: usize, hot_content: Vec<HV16>) -> Option<usize> {
        // Get target from first-order
        if target_idx >= self.first_order.len() {
            return None;
        }

        let target = self.first_order[target_idx].clone();

        // Create HOT
        let mut hot = MentalState::higher_order(
            target,
            hot_content,
            "introspection".to_string()
        );
        hot.timestamp = self.timestep;

        // Target is now conscious (has HOT about it)
        let hot_idx = match hot.order {
            RepresentationOrder::SecondOrder => {
                let idx = self.second_order.len();
                self.second_order.push(hot);
                Some(idx)
            }
            RepresentationOrder::ThirdOrder | RepresentationOrder::HigherOrder => {
                let idx = self.higher_order.len();
                self.higher_order.push(hot);
                Some(idx)
            }
            _ => None,
        };

        hot_idx
    }

    /// Process HOT dynamics (automatic HOT generation)
    pub fn process(&mut self) -> HOTAssessment {
        self.timestep += 1;

        if self.config.auto_generate_hots {
            self.generate_automatic_hots();
        }

        self.assess()
    }

    /// Automatically generate HOTs for salient first-order states
    fn generate_automatic_hots(&mut self) {
        // Find high-activation first-order states
        let mut new_hots = Vec::new();

        for (idx, state) in self.first_order.iter().enumerate() {
            // Check if already has HOT
            let has_hot = self.second_order.iter()
                .any(|hot| {
                    if let Some(target) = &hot.represents {
                        self.states_equal(&target.content, &state.content)
                    } else {
                        false
                    }
                });

            if !has_hot && state.confidence > self.config.hot_threshold {
                // Generate HOT (simplified: copy content with marker)
                let hot_content = self.create_hot_representation(&state.content);
                new_hots.push((idx, hot_content));
            }
        }

        // Form HOTs
        for (idx, content) in new_hots {
            self.form_hot(idx, content);
        }
    }

    /// Create HOT representation from first-order content
    fn create_hot_representation(&self, first_order: &[HV16]) -> Vec<HV16> {
        // Simplified: Bind with "awareness" marker
        // In full implementation: More sophisticated meta-representation
        let awareness_marker = HV16::random(999);  // Fixed marker for "I am aware of"

        first_order.iter()
            .map(|&hv| hv.bind(&awareness_marker))
            .collect()
    }

    /// Check if two states are equal
    fn states_equal(&self, a: &[HV16], b: &[HV16]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        a.iter().zip(b.iter()).all(|(x, y)| x == y)
    }

    /// Assess higher-order consciousness state
    fn assess(&mut self) -> HOTAssessment {
        let total_states = self.first_order.len() + self.second_order.len() + self.higher_order.len();

        let first_order_count = self.first_order.len();
        let second_order_count = self.second_order.len();
        let higher_order_count = self.higher_order.len();

        let consciousness_ratio = if total_states > 0 {
            (second_order_count + higher_order_count) as f64 / total_states as f64
        } else {
            0.0
        };

        // Self-awareness = HOT about HOT (third-order or higher exists)
        let self_awareness = !self.higher_order.is_empty();

        // Count misrepresentations
        let misrepresentations = self.second_order.iter()
            .chain(self.higher_order.iter())
            .filter(|hot| hot.is_misrepresented())
            .count();

        // Average depth
        let avg_depth = if total_states > 0 {
            let total_depth: usize = self.first_order.iter().map(|s| s.depth()).sum::<usize>()
                + self.second_order.iter().map(|s| s.depth()).sum::<usize>()
                + self.higher_order.iter().map(|s| s.depth()).sum::<usize>();
            total_depth as f64 / total_states as f64
        } else {
            0.0
        };

        let explanation = self.generate_explanation(
            consciousness_ratio,
            self_awareness,
            misrepresentations,
        );

        HOTAssessment {
            total_states,
            first_order_count,
            second_order_count,
            higher_order_count,
            consciousness_ratio,
            self_awareness,
            misrepresentations,
            avg_depth,
            explanation,
        }
    }

    /// Generate human-readable explanation
    fn generate_explanation(
        &self,
        consciousness_ratio: f64,
        self_awareness: bool,
        misrepresentations: usize,
    ) -> String {
        let mut parts = Vec::new();

        // Consciousness level
        if consciousness_ratio > 0.7 {
            parts.push("High consciousness".to_string());
        } else if consciousness_ratio > 0.3 {
            parts.push("Moderate consciousness".to_string());
        } else if consciousness_ratio > 0.0 {
            parts.push("Low consciousness".to_string());
        } else {
            parts.push("No conscious states".to_string());
        }

        // Self-awareness
        if self_awareness {
            parts.push("Self-aware (meta-conscious)".to_string());
        } else if consciousness_ratio > 0.0 {
            parts.push("Aware but not self-aware".to_string());
        }

        // Misrepresentations
        if misrepresentations > 0 {
            parts.push(format!("{} misrepresentation(s)", misrepresentations));
        }

        // State counts
        if self.second_order.len() > 0 {
            parts.push(format!("{} conscious thoughts", self.second_order.len()));
        }

        parts.join(". ")
    }

    /// Get all conscious states (with HOTs)
    pub fn get_conscious_states(&self) -> Vec<&MentalState> {
        self.second_order.iter()
            .chain(self.higher_order.iter())
            .collect()
    }

    /// Check if system is self-aware
    pub fn is_self_aware(&self) -> bool {
        !self.higher_order.is_empty()
    }

    /// Clear all states
    pub fn clear(&mut self) {
        self.first_order.clear();
        self.second_order.clear();
        self.higher_order.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_representation_order() {
        assert!(RepresentationOrder::FirstOrder < RepresentationOrder::SecondOrder);
        assert!(RepresentationOrder::SecondOrder < RepresentationOrder::ThirdOrder);
    }

    #[test]
    fn test_first_order_state() {
        let state = MentalState::first_order(
            vec![HV16::ones(); 10],
            "perception".to_string(),
        );
        assert_eq!(state.order, RepresentationOrder::FirstOrder);
        assert!(!state.is_conscious());  // First-order alone not conscious
        assert_eq!(state.depth(), 1);
    }

    #[test]
    fn test_higher_order_state() {
        let first = MentalState::first_order(
            vec![HV16::ones(); 10],
            "perception".to_string(),
        );
        let second = MentalState::higher_order(
            first,
            vec![HV16::zero(); 10],
            "introspection".to_string(),
        );

        assert_eq!(second.order, RepresentationOrder::SecondOrder);
        assert!(second.is_conscious());  // Second-order IS conscious
        assert_eq!(second.depth(), 2);
    }

    #[test]
    fn test_third_order_meta_consciousness() {
        let first = MentalState::first_order(
            vec![HV16::ones(); 10],
            "perception".to_string(),
        );
        let second = MentalState::higher_order(
            first,
            vec![HV16::zero(); 10],
            "introspection".to_string(),
        );
        let third = MentalState::higher_order(
            second,
            vec![HV16::random(1); 10],
            "reflection".to_string(),
        );

        assert_eq!(third.order, RepresentationOrder::ThirdOrder);
        assert!(third.is_conscious());
        assert_eq!(third.depth(), 3);
    }

    #[test]
    fn test_hot_system_creation() {
        let system = HigherOrderThoughtSystem::new(HOTConfig::default());
        assert_eq!(system.first_order.len(), 0);
        assert_eq!(system.second_order.len(), 0);
    }

    #[test]
    fn test_perceive() {
        let mut system = HigherOrderThoughtSystem::new(HOTConfig::default());
        system.perceive(vec![HV16::ones(); 10], "vision".to_string());

        assert_eq!(system.first_order.len(), 1);
        assert_eq!(system.second_order.len(), 0);  // No HOT yet
    }

    #[test]
    fn test_form_hot_makes_conscious() {
        let mut system = HigherOrderThoughtSystem::new(HOTConfig::default());

        // Add first-order state
        system.perceive(vec![HV16::ones(); 10], "vision".to_string());

        // Form HOT about it
        let hot_idx = system.form_hot(0, vec![HV16::zero(); 10]);

        assert!(hot_idx.is_some());
        assert_eq!(system.second_order.len(), 1);
        assert!(system.second_order[0].is_conscious());
    }

    #[test]
    fn test_automatic_hot_generation() {
        let mut system = HigherOrderThoughtSystem::new(HOTConfig {
            auto_generate_hots: true,
            hot_threshold: 0.5,
            ..Default::default()
        });

        // Add high-confidence first-order state
        let mut state = MentalState::first_order(
            vec![HV16::ones(); 10],
            "vision".to_string(),
        );
        state.confidence = 0.9;  // Above threshold
        system.first_order.push(state);

        // Process should auto-generate HOT
        let assessment = system.process();

        assert!(assessment.second_order_count > 0);  // HOT generated
        assert!(assessment.consciousness_ratio > 0.0);
    }

    #[test]
    fn test_consciousness_ratio() {
        let mut system = HigherOrderThoughtSystem::new(HOTConfig::default());

        // Add 2 first-order, form HOT for 1
        system.perceive(vec![HV16::ones(); 10], "vision".to_string());
        system.perceive(vec![HV16::zero(); 10], "hearing".to_string());
        system.form_hot(0, vec![HV16::random(1); 10]);

        let assessment = system.assess();

        // 2 first-order + 1 second-order = 3 total
        // 1 conscious / 3 total = 0.333...
        assert!(assessment.consciousness_ratio > 0.3 && assessment.consciousness_ratio < 0.4);
    }

    #[test]
    fn test_self_awareness_detection() {
        let mut system = HigherOrderThoughtSystem::new(HOTConfig::default());

        // Add first-order
        system.perceive(vec![HV16::ones(); 10], "vision".to_string());

        // Form second-order HOT
        system.form_hot(0, vec![HV16::zero(); 10]);

        // Not yet self-aware (no HOT about HOT)
        assert!(!system.is_self_aware());

        // Form third-order (HOT about HOT)
        if let Some(second) = system.second_order.first().cloned() {
            let third = MentalState::higher_order(
                second,
                vec![HV16::random(2); 10],
                "meta-reflection".to_string(),
            );
            system.higher_order.push(third);
        }

        // Now self-aware!
        assert!(system.is_self_aware());
    }

    #[test]
    fn test_misrepresentation_detection() {
        let first = MentalState::first_order(
            vec![HV16::ones(); 10],
            "perception".to_string(),
        );

        // HOT misrepresents (very different content)
        let mut hot = MentalState::higher_order(
            first,
            vec![HV16::zero(); 10],  // Completely different
            "introspection".to_string(),
        );

        // Should detect misrepresentation
        assert!(hot.is_misrepresented());
    }

    #[test]
    fn test_clear() {
        let mut system = HigherOrderThoughtSystem::new(HOTConfig::default());

        system.perceive(vec![HV16::ones(); 10], "vision".to_string());
        system.form_hot(0, vec![HV16::zero(); 10]);

        system.clear();

        assert_eq!(system.first_order.len(), 0);
        assert_eq!(system.second_order.len(), 0);
    }
}
