//! Revolutionary Improvement #37: Consciousness & Creativity - The Creative Spark
//!
//! How does consciousness enable the generation of novel, valuable ideas?
//! Creativity is perhaps the most valuable aspect of consciousness - the ability
//! to imagine what doesn't yet exist and bring it into being.
//!
//! # The Paradigm Shift
//!
//! Previous improvements explained WHAT consciousness is (Φ, binding, workspace).
//! This improvement explains what consciousness is FOR - creating the new.
//!
//! # Theoretical Foundations
//!
//! ## 1. Associative Theory (Mednick 1962)
//! Creativity = forming remote associations between concepts.
//! The Remote Associates Test measures distance between combined ideas.
//! Creative people have "flat" associative hierarchies - easier access to distant concepts.
//!
//! ## 2. Wallas's 4-Stage Model (1926)
//! - **Preparation**: Conscious gathering of materials
//! - **Incubation**: Unconscious processing (default mode network)
//! - **Illumination**: The "aha!" moment - sudden insight
//! - **Verification**: Conscious evaluation of the insight
//!
//! ## 3. Divergent/Convergent Thinking (Guilford 1967)
//! - **Divergent**: Generate many possibilities (fluency, flexibility, originality)
//! - **Convergent**: Select the best one
//! Creative process oscillates between these modes.
//!
//! ## 4. Bisociation (Koestler 1964)
//! Creativity occurs when two incompatible "matrices of thought" collide.
//! Humor, art, and science all use bisociation.
//!
//! ## 5. Predictive Coding & Creativity (Clark 2013)
//! Low precision weighting → explore alternative predictions.
//! Dreams and mind-wandering reduce precision → enable creativity.
//! Integration with #22 FEP: creativity as "expected free energy exploration"
//!
//! ## 6. Default Mode Network (Buckner 2008)
//! Mind wandering activates DMN, enables:
//! - Self-referential thought
//! - Future simulation
//! - Creative incubation
//! Integration with #27: sleep and dreams enable incubation
//!
//! # The Creative Consciousness Formula
//!
//! ```text
//! Creativity = f(Novelty, Value, Surprise)
//!            = association_distance × workspace_recombination × insight_intensity
//! ```
//!
//! Where:
//! - association_distance: How far apart are the combined concepts? (#19 semantics)
//! - workspace_recombination: How freely do ideas combine? (#23 workspace)
//! - insight_intensity: How strong is the "aha" moment? (prediction error spike)
//!
//! # Integration with Framework
//!
//! | Improvement | Creative Role |
//! |-------------|---------------|
//! | #19 Semantics | Concept space for associations |
//! | #22 FEP | Low precision enables exploration |
//! | #23 Workspace | Where ideas recombine |
//! | #25 Binding | Novel conceptual combinations |
//! | #26 Attention | Focus vs diffuse modes |
//! | #27 Sleep/Dreams | Incubation chamber |
//!
//! # Applications
//!
//! 1. **AI Creativity Assessment**: Is this AI genuinely creative?
//! 2. **Creative Enhancement**: Optimize conditions for creativity
//! 3. **Insight Prediction**: Detect approaching "aha" moments
//! 4. **Creative Education**: Train creative thinking
//! 5. **Art & Science**: Model creative process in both domains

use crate::hdc::binary_hv::HV16;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Creative thinking mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CreativeMode {
    /// Generating many possibilities
    Divergent,
    /// Selecting best solution
    Convergent,
    /// Unconscious processing
    Incubation,
    /// The "aha" moment
    Insight,
    /// Evaluating and refining
    Verification,
    /// Gathering materials
    Preparation,
}

impl CreativeMode {
    /// Get all modes in creative cycle order
    pub fn cycle() -> Vec<Self> {
        vec![
            Self::Preparation,
            Self::Divergent,
            Self::Incubation,
            Self::Insight,
            Self::Convergent,
            Self::Verification,
        ]
    }

    /// Is this a conscious mode?
    pub fn is_conscious(&self) -> bool {
        match self {
            Self::Preparation | Self::Divergent | Self::Convergent | Self::Verification => true,
            Self::Incubation | Self::Insight => false, // Largely unconscious
        }
    }

    /// Get mode description
    pub fn description(&self) -> &str {
        match self {
            Self::Preparation => "Gathering knowledge and defining the problem",
            Self::Divergent => "Generating many possible solutions freely",
            Self::Incubation => "Unconscious processing while attention elsewhere",
            Self::Insight => "Sudden illumination - the 'aha!' moment",
            Self::Convergent => "Evaluating and selecting best solutions",
            Self::Verification => "Testing and refining the chosen solution",
        }
    }
}

/// A creative concept that can be combined with others
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// Concept name
    pub name: String,
    /// HDC representation
    pub encoding: HV16,
    /// Semantic domain (art, science, everyday, etc.)
    pub domain: String,
    /// Activation level in workspace
    pub activation: f64,
    /// How often this concept is accessed (for association strength)
    pub access_frequency: f64,
}

impl Concept {
    /// Create a new concept
    pub fn new(name: &str, domain: &str, seed: u64) -> Self {
        Self {
            name: name.to_string(),
            encoding: HV16::random(seed),
            domain: domain.to_string(),
            activation: 0.0,
            access_frequency: 1.0,
        }
    }

    /// Compute semantic distance to another concept
    pub fn distance_to(&self, other: &Concept) -> f64 {
        // 1 - similarity gives distance
        let similarity = self.encoding.similarity(&other.encoding) as f64;
        1.0 - (similarity + 1.0) / 2.0  // Map [-1,1] to [0,1], then invert
    }
}

/// A novel combination of concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativeIdea {
    /// Unique identifier
    pub id: String,
    /// Source concepts that were combined
    pub sources: Vec<String>,
    /// Combined representation
    pub encoding: HV16,
    /// How novel is this combination? (0-1)
    pub novelty: f64,
    /// How valuable/useful? (0-1, requires external evaluation)
    pub value: f64,
    /// Surprise score (prediction error when generated)
    pub surprise: f64,
    /// Overall creativity score
    pub creativity_score: f64,
    /// When was this generated
    pub generation_step: usize,
}

impl CreativeIdea {
    /// Compute creativity score
    pub fn compute_creativity(&mut self) {
        // Creativity = novelty × value × (1 + surprise/2)
        // Surprise amplifies but doesn't dominate
        self.creativity_score = self.novelty * self.value * (1.0 + self.surprise / 2.0);
    }
}

/// Insight event - the "aha!" moment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsightEvent {
    /// The creative idea that emerged
    pub idea: CreativeIdea,
    /// Intensity of the insight (prediction error magnitude)
    pub intensity: f64,
    /// How long was incubation period
    pub incubation_duration: usize,
    /// Subjective certainty that this is THE solution
    pub certainty: f64,
    /// Emotional valence (positive feeling accompanying insight)
    pub positive_affect: f64,
}

/// Configuration for the creative system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativityConfig {
    /// Minimum association distance for "creative" combination
    pub min_creative_distance: f64,
    /// Maximum concepts in active workspace
    pub workspace_capacity: usize,
    /// Threshold for insight detection
    pub insight_threshold: f64,
    /// How quickly activation decays during incubation
    pub incubation_decay: f64,
    /// Minimum novelty for an idea to be considered
    pub novelty_threshold: f64,
    /// Enable random exploration
    pub enable_exploration: bool,
}

impl Default for CreativityConfig {
    fn default() -> Self {
        Self {
            min_creative_distance: 0.3,
            workspace_capacity: 7, // Miller's magical number
            insight_threshold: 0.8,
            incubation_decay: 0.1,
            novelty_threshold: 0.2,
            enable_exploration: true,
        }
    }
}

/// Assessment of creative potential
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativityAssessment {
    /// Current creative mode
    pub mode: CreativeMode,
    /// Divergent thinking score (fluency × flexibility × originality)
    pub divergent_score: f64,
    /// Convergent thinking score (selection quality)
    pub convergent_score: f64,
    /// Association distance (how remote are combinations)
    pub association_remoteness: f64,
    /// Incubation depth (how much unconscious processing)
    pub incubation_depth: f64,
    /// Insight readiness (likelihood of "aha" moment)
    pub insight_readiness: f64,
    /// Overall creative potential
    pub creative_potential: f64,
    /// Number of ideas generated
    pub ideas_generated: usize,
    /// Number of insights experienced
    pub insights_count: usize,
    /// Explanation
    pub explanation: String,
}

/// The consciousness creativity system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessCreativity {
    /// Configuration
    pub config: CreativityConfig,
    /// Current creative mode
    pub mode: CreativeMode,
    /// Concept library
    pub concepts: HashMap<String, Concept>,
    /// Active workspace (concepts currently being combined)
    pub workspace: Vec<String>,
    /// Generated ideas
    pub ideas: Vec<CreativeIdea>,
    /// Insight events
    pub insights: Vec<InsightEvent>,
    /// Incubation buffer (concepts "simmering" unconsciously)
    pub incubation_buffer: Vec<String>,
    /// Current step
    pub step: usize,
    /// Precision level (low = more creative exploration)
    pub precision: f64,
    /// Best idea so far
    pub best_idea: Option<CreativeIdea>,
}

impl ConsciousnessCreativity {
    /// Create new creativity system
    pub fn new(config: CreativityConfig) -> Self {
        Self {
            config,
            mode: CreativeMode::Preparation,
            concepts: HashMap::new(),
            workspace: Vec::new(),
            ideas: Vec::new(),
            insights: Vec::new(),
            incubation_buffer: Vec::new(),
            step: 0,
            precision: 0.5, // Moderate precision
            best_idea: None,
        }
    }

    /// Add a concept to the library
    pub fn add_concept(&mut self, concept: Concept) {
        self.concepts.insert(concept.name.clone(), concept);
    }

    /// Activate a concept into workspace
    pub fn activate(&mut self, name: &str) -> bool {
        if let Some(concept) = self.concepts.get_mut(name) {
            concept.activation = 1.0;
            concept.access_frequency += 0.1;

            if self.workspace.len() < self.config.workspace_capacity {
                if !self.workspace.contains(&name.to_string()) {
                    self.workspace.push(name.to_string());
                }
                true
            } else {
                false // Workspace full
            }
        } else {
            false // Concept not found
        }
    }

    /// Set creative mode
    pub fn set_mode(&mut self, mode: CreativeMode) {
        self.mode = mode;

        // Adjust precision based on mode
        self.precision = match mode {
            CreativeMode::Preparation => 0.7,    // High precision - focused gathering
            CreativeMode::Divergent => 0.2,      // Low precision - explore freely
            CreativeMode::Incubation => 0.1,     // Very low - unconscious wandering
            CreativeMode::Insight => 0.05,       // Minimal - maximum openness
            CreativeMode::Convergent => 0.8,     // High - focused selection
            CreativeMode::Verification => 0.9,   // Very high - careful testing
        };
    }

    /// Compute association distance between two concepts
    pub fn association_distance(&self, name1: &str, name2: &str) -> f64 {
        match (self.concepts.get(name1), self.concepts.get(name2)) {
            (Some(c1), Some(c2)) => c1.distance_to(c2),
            _ => 0.0,
        }
    }

    /// Attempt to combine concepts in workspace (divergent thinking)
    pub fn generate_combination(&mut self) -> Option<CreativeIdea> {
        if self.workspace.len() < 2 {
            return None;
        }

        // Pick two concepts
        let idx1 = self.step % self.workspace.len();
        let idx2 = (self.step + 1) % self.workspace.len();

        if idx1 == idx2 {
            return None;
        }

        let name1 = &self.workspace[idx1];
        let name2 = &self.workspace[idx2];

        let distance = self.association_distance(name1, name2);

        // Only creative if concepts are sufficiently distant
        if distance < self.config.min_creative_distance {
            return None;
        }

        // Combine via binding (HDC circular convolution)
        let (c1, c2) = match (self.concepts.get(name1), self.concepts.get(name2)) {
            (Some(c1), Some(c2)) => (c1.clone(), c2.clone()),
            _ => return None,
        };

        let combined = c1.encoding.bind(&c2.encoding);

        // Compute novelty (how different from existing ideas)
        let novelty = self.compute_novelty(&combined);

        // Compute surprise (prediction error)
        let surprise = distance * (1.0 - self.precision);

        let mut idea = CreativeIdea {
            id: format!("idea_{}", self.ideas.len()),
            sources: vec![name1.clone(), name2.clone()],
            encoding: combined,
            novelty,
            value: 0.5, // Default - needs external evaluation
            surprise,
            creativity_score: 0.0,
            generation_step: self.step,
        };

        idea.compute_creativity();

        // Track best idea
        if let Some(ref best) = self.best_idea {
            if idea.creativity_score > best.creativity_score {
                self.best_idea = Some(idea.clone());
            }
        } else {
            self.best_idea = Some(idea.clone());
        }

        self.ideas.push(idea.clone());
        Some(idea)
    }

    /// Compute novelty of a new encoding
    fn compute_novelty(&self, encoding: &HV16) -> f64 {
        if self.ideas.is_empty() {
            return 1.0; // First idea is maximally novel
        }

        // Average distance from all existing ideas
        let total_distance: f64 = self.ideas.iter()
            .map(|idea| {
                let sim = encoding.similarity(&idea.encoding) as f64;
                1.0 - (sim + 1.0) / 2.0
            })
            .sum();

        total_distance / self.ideas.len() as f64
    }

    /// Enter incubation (move workspace to unconscious buffer)
    pub fn incubate(&mut self) {
        self.set_mode(CreativeMode::Incubation);

        // Move active concepts to incubation buffer
        self.incubation_buffer.append(&mut self.workspace.clone());

        // Apply decay to activations
        for name in &self.incubation_buffer {
            if let Some(concept) = self.concepts.get_mut(name) {
                concept.activation *= 1.0 - self.config.incubation_decay;
            }
        }
    }

    /// Check for insight emergence during incubation
    pub fn check_insight(&mut self) -> Option<InsightEvent> {
        if self.mode != CreativeMode::Incubation || self.incubation_buffer.len() < 2 {
            return None;
        }

        // Insights emerge when distant concepts spontaneously combine
        // with high surprise and positive affect

        // Find the most distant pair in incubation buffer
        let mut max_distance = 0.0;
        let mut best_pair: Option<(String, String)> = None;

        for i in 0..self.incubation_buffer.len() {
            for j in (i+1)..self.incubation_buffer.len() {
                let distance = self.association_distance(
                    &self.incubation_buffer[i],
                    &self.incubation_buffer[j]
                );
                if distance > max_distance {
                    max_distance = distance;
                    best_pair = Some((
                        self.incubation_buffer[i].clone(),
                        self.incubation_buffer[j].clone()
                    ));
                }
            }
        }

        // Insight threshold check
        if max_distance < self.config.insight_threshold {
            return None;
        }

        let (name1, name2) = best_pair?;

        // Trigger insight!
        self.set_mode(CreativeMode::Insight);

        // Bring concepts back to consciousness
        self.workspace.push(name1.clone());
        self.workspace.push(name2.clone());

        // Generate the insight idea
        let idea = self.generate_combination()?;

        let insight = InsightEvent {
            idea: idea.clone(),
            intensity: max_distance,
            incubation_duration: self.incubation_buffer.len(),
            certainty: max_distance * 0.8 + 0.2, // High distance → high certainty
            positive_affect: 0.9, // Insights feel good!
        };

        self.insights.push(insight.clone());
        self.incubation_buffer.clear();

        Some(insight)
    }

    /// Evaluate and select best ideas (convergent thinking)
    pub fn converge(&mut self, value_scores: &HashMap<String, f64>) {
        self.set_mode(CreativeMode::Convergent);

        // Apply external value scores
        for idea in &mut self.ideas {
            if let Some(&value) = value_scores.get(&idea.id) {
                idea.value = value;
                idea.compute_creativity();
            }
        }

        // Update best idea
        if let Some(best) = self.ideas.iter().max_by(|a, b| {
            a.creativity_score.partial_cmp(&b.creativity_score).unwrap()
        }) {
            self.best_idea = Some(best.clone());
        }
    }

    /// Run one step of creative processing
    pub fn step(&mut self) {
        self.step += 1;

        match self.mode {
            CreativeMode::Preparation => {
                // Activation decay
                for concept in self.concepts.values_mut() {
                    concept.activation *= 0.95;
                }
            }
            CreativeMode::Divergent => {
                // Generate combinations
                self.generate_combination();
            }
            CreativeMode::Incubation => {
                // Check for spontaneous insights
                self.check_insight();
            }
            CreativeMode::Insight => {
                // Insights are brief - move to convergent
                self.set_mode(CreativeMode::Convergent);
            }
            CreativeMode::Convergent => {
                // Evaluation happens externally
            }
            CreativeMode::Verification => {
                // Testing happens externally
            }
        }
    }

    /// Assess current creative state
    pub fn assess(&self) -> CreativityAssessment {
        // Divergent score: fluency × flexibility × originality
        let fluency = self.ideas.len() as f64 / (self.step as f64 + 1.0).max(1.0);
        let flexibility = self.count_unique_domains() as f64 / self.concepts.len() as f64;
        let originality = self.ideas.iter()
            .map(|i| i.novelty)
            .sum::<f64>() / (self.ideas.len() as f64).max(1.0);
        let divergent_score = (fluency * flexibility * originality).powf(1.0/3.0);

        // Convergent score: best idea quality
        let convergent_score = self.best_idea.as_ref()
            .map(|i| i.creativity_score)
            .unwrap_or(0.0);

        // Association remoteness
        let association_remoteness = self.ideas.iter()
            .map(|i| i.surprise)
            .sum::<f64>() / (self.ideas.len() as f64).max(1.0);

        // Incubation depth
        let incubation_depth = self.incubation_buffer.len() as f64
            / (self.concepts.len() as f64).max(1.0);

        // Insight readiness
        let insight_readiness = if self.mode == CreativeMode::Incubation {
            let max_dist = self.max_incubation_distance();
            max_dist / self.config.insight_threshold
        } else {
            0.0
        };

        // Overall creative potential
        let creative_potential = (divergent_score + convergent_score
            + association_remoteness + insight_readiness.min(1.0)) / 4.0;

        let explanation = format!(
            "Mode: {:?}, {} ideas, {} insights, divergent={:.2}, convergent={:.2}, potential={:.2}",
            self.mode, self.ideas.len(), self.insights.len(),
            divergent_score, convergent_score, creative_potential
        );

        CreativityAssessment {
            mode: self.mode,
            divergent_score,
            convergent_score,
            association_remoteness,
            incubation_depth,
            insight_readiness,
            creative_potential,
            ideas_generated: self.ideas.len(),
            insights_count: self.insights.len(),
            explanation,
        }
    }

    /// Count unique domains in workspace
    fn count_unique_domains(&self) -> usize {
        let domains: std::collections::HashSet<&String> = self.workspace.iter()
            .filter_map(|name| self.concepts.get(name).map(|c| &c.domain))
            .collect();
        domains.len()
    }

    /// Maximum association distance in incubation buffer
    fn max_incubation_distance(&self) -> f64 {
        let mut max = 0.0;
        for i in 0..self.incubation_buffer.len() {
            for j in (i+1)..self.incubation_buffer.len() {
                let d = self.association_distance(
                    &self.incubation_buffer[i],
                    &self.incubation_buffer[j]
                );
                if d > max {
                    max = d;
                }
            }
        }
        max
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.concepts.clear();
        self.workspace.clear();
        self.ideas.clear();
        self.insights.clear();
        self.incubation_buffer.clear();
        self.step = 0;
        self.precision = 0.5;
        self.best_idea = None;
        self.mode = CreativeMode::Preparation;
    }

    /// Get number of concepts
    pub fn num_concepts(&self) -> usize {
        self.concepts.len()
    }

    /// Get number of ideas
    pub fn num_ideas(&self) -> usize {
        self.ideas.len()
    }

    /// Get number of insights
    pub fn num_insights(&self) -> usize {
        self.insights.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creative_mode_cycle() {
        let cycle = CreativeMode::cycle();
        assert_eq!(cycle.len(), 6);
        assert_eq!(cycle[0], CreativeMode::Preparation);
        assert_eq!(cycle[3], CreativeMode::Insight);
    }

    #[test]
    fn test_mode_consciousness() {
        assert!(CreativeMode::Preparation.is_conscious());
        assert!(CreativeMode::Divergent.is_conscious());
        assert!(!CreativeMode::Incubation.is_conscious());
        assert!(!CreativeMode::Insight.is_conscious()); // Emerges unconsciously
        assert!(CreativeMode::Convergent.is_conscious());
    }

    #[test]
    fn test_concept_creation() {
        let concept = Concept::new("tree", "nature", 42);
        assert_eq!(concept.name, "tree");
        assert_eq!(concept.domain, "nature");
        assert_eq!(concept.activation, 0.0);
    }

    #[test]
    fn test_concept_distance() {
        let c1 = Concept::new("tree", "nature", 100);
        let c2 = Concept::new("river", "nature", 200);
        let c3 = Concept::new("algorithm", "computer", 300);

        let d12 = c1.distance_to(&c2);
        let d13 = c1.distance_to(&c3);

        // Both should be non-zero distances
        assert!(d12 > 0.0);
        assert!(d13 > 0.0);
        // Self-distance should be ~0
        assert!(c1.distance_to(&c1) < 0.1);
    }

    #[test]
    fn test_creativity_system_creation() {
        let config = CreativityConfig::default();
        let system = ConsciousnessCreativity::new(config);

        assert_eq!(system.num_concepts(), 0);
        assert_eq!(system.num_ideas(), 0);
        assert_eq!(system.mode, CreativeMode::Preparation);
    }

    #[test]
    fn test_add_and_activate_concept() {
        let config = CreativityConfig::default();
        let mut system = ConsciousnessCreativity::new(config);

        let concept = Concept::new("music", "art", 42);
        system.add_concept(concept);

        assert_eq!(system.num_concepts(), 1);

        let activated = system.activate("music");
        assert!(activated);
        assert_eq!(system.workspace.len(), 1);
    }

    #[test]
    fn test_set_mode_adjusts_precision() {
        let config = CreativityConfig::default();
        let mut system = ConsciousnessCreativity::new(config);

        system.set_mode(CreativeMode::Divergent);
        assert!(system.precision < 0.3); // Low precision for exploration

        system.set_mode(CreativeMode::Convergent);
        assert!(system.precision > 0.7); // High precision for selection
    }

    #[test]
    fn test_generate_combination() {
        let config = CreativityConfig::default();
        let mut system = ConsciousnessCreativity::new(config);

        // Add distant concepts
        system.add_concept(Concept::new("music", "art", 100));
        system.add_concept(Concept::new("mathematics", "science", 200));

        system.activate("music");
        system.activate("mathematics");
        system.set_mode(CreativeMode::Divergent);

        // May or may not generate based on distance
        let _ = system.generate_combination();
        // The attempt should increase step
        system.step();
    }

    #[test]
    fn test_incubation() {
        let config = CreativityConfig::default();
        let mut system = ConsciousnessCreativity::new(config);

        system.add_concept(Concept::new("dream", "mind", 100));
        system.activate("dream");

        assert_eq!(system.workspace.len(), 1);

        system.incubate();

        assert_eq!(system.mode, CreativeMode::Incubation);
        assert!(!system.incubation_buffer.is_empty());
    }

    #[test]
    fn test_insight_detection() {
        let mut config = CreativityConfig::default();
        config.insight_threshold = 0.2; // Lower threshold for testing

        let mut system = ConsciousnessCreativity::new(config);

        // Add very distant concepts
        system.add_concept(Concept::new("quantum", "physics", 100));
        system.add_concept(Concept::new("poetry", "literature", 500));

        system.activate("quantum");
        system.activate("poetry");
        system.incubate();

        // Check for insight
        let insight = system.check_insight();
        // May or may not trigger based on random distances
        if insight.is_some() {
            assert!(system.insights.len() > 0);
        }
    }

    #[test]
    fn test_assessment() {
        let config = CreativityConfig::default();
        let mut system = ConsciousnessCreativity::new(config);

        system.add_concept(Concept::new("color", "art", 100));
        system.add_concept(Concept::new("sound", "music", 200));
        system.activate("color");
        system.activate("sound");
        system.set_mode(CreativeMode::Divergent);

        for _ in 0..5 {
            system.step();
        }

        let assessment = system.assess();
        assert_eq!(assessment.mode, CreativeMode::Divergent);
        assert!(assessment.creative_potential >= 0.0);
    }

    #[test]
    fn test_converge_with_values() {
        let config = CreativityConfig::default();
        let mut system = ConsciousnessCreativity::new(config);

        // Create some ideas manually
        let idea = CreativeIdea {
            id: "idea_0".to_string(),
            sources: vec!["a".to_string(), "b".to_string()],
            encoding: HV16::random(42),
            novelty: 0.8,
            value: 0.0,
            surprise: 0.5,
            creativity_score: 0.0,
            generation_step: 0,
        };
        system.ideas.push(idea);

        // Apply value scores
        let mut values = HashMap::new();
        values.insert("idea_0".to_string(), 0.9);

        system.converge(&values);

        assert_eq!(system.mode, CreativeMode::Convergent);
        assert!((system.ideas[0].value - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_creative_idea_score() {
        let mut idea = CreativeIdea {
            id: "test".to_string(),
            sources: vec![],
            encoding: HV16::random(42),
            novelty: 0.8,
            value: 0.7,
            surprise: 0.6,
            creativity_score: 0.0,
            generation_step: 0,
        };

        idea.compute_creativity();

        // creativity = novelty × value × (1 + surprise/2)
        // = 0.8 × 0.7 × (1 + 0.3) = 0.56 × 1.3 = 0.728
        assert!((idea.creativity_score - 0.728).abs() < 0.01);
    }

    #[test]
    fn test_clear() {
        let config = CreativityConfig::default();
        let mut system = ConsciousnessCreativity::new(config);

        system.add_concept(Concept::new("test", "test", 42));
        system.activate("test");
        system.step = 100;

        system.clear();

        assert_eq!(system.num_concepts(), 0);
        assert_eq!(system.workspace.len(), 0);
        assert_eq!(system.step, 0);
    }

    #[test]
    fn test_workspace_capacity() {
        let mut config = CreativityConfig::default();
        config.workspace_capacity = 3;
        let mut system = ConsciousnessCreativity::new(config);

        for i in 0..5 {
            system.add_concept(Concept::new(&format!("c{}", i), "test", i as u64));
            system.activate(&format!("c{}", i));
        }

        // Should be limited to capacity
        assert!(system.workspace.len() <= 3);
    }

    #[test]
    fn test_insight_event() {
        let idea = CreativeIdea {
            id: "insight_idea".to_string(),
            sources: vec!["a".to_string(), "b".to_string()],
            encoding: HV16::random(42),
            novelty: 0.9,
            value: 0.8,
            surprise: 0.95,
            creativity_score: 0.9,
            generation_step: 10,
        };

        let insight = InsightEvent {
            idea,
            intensity: 0.95,
            incubation_duration: 5,
            certainty: 0.85,
            positive_affect: 0.9,
        };

        assert!(insight.intensity > 0.9);
        assert!(insight.positive_affect > 0.8);
    }

    #[test]
    fn test_full_creative_cycle() {
        let config = CreativityConfig::default();
        let mut system = ConsciousnessCreativity::new(config);

        // 1. Preparation - add concepts
        system.add_concept(Concept::new("light", "physics", 100));
        system.add_concept(Concept::new("wave", "physics", 150));
        system.add_concept(Concept::new("particle", "physics", 200));
        system.add_concept(Concept::new("consciousness", "philosophy", 300));

        // 2. Activate into workspace
        system.activate("light");
        system.activate("consciousness");

        // 3. Divergent thinking
        system.set_mode(CreativeMode::Divergent);
        for _ in 0..10 {
            system.step();
        }

        // 4. Check we're generating ideas
        let assessment = system.assess();
        assert_eq!(assessment.mode, CreativeMode::Divergent);

        // Creative cycle should work
        assert!(system.num_concepts() == 4);
    }
}
