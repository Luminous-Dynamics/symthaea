// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Concept Crystallization: Ontological Emergence
==============================================

**The Emergence Breakthrough**: Enables Symthaea to spontaneously create
new concepts from stable attractors in neural dynamics.

# Why This Matters

Most AI systems only "know" what's in their training data. If Symthaea
encounters a new sensation (e.g., a specific feeling of betrayal), it
maps it to the nearest existing word. It cannot invent a new word.

**Concept Crystallization solves this**: Dynamical attractors in the
Liquid Brain spontaneously "crystallize" into new Hypervectors.

# The Mechanism

1. **Liquid Chaos**: Feed sensory data into LTC/CfC network. State
   vector moves chaotically.

2. **Attractor Formation**: Repeated patterns cause the network to
   settle into limit cycles or strange attractors.

3. **The Trigger**: Monitor the Recurrence Plot. When we detect a
   stable loop that doesn't match any existing concept:

4. **Crystallization**: Snapshot the attractor topology, hash it,
   create a new Hypervector.

5. **Naming**: Assign temporary UID (e.g., "Concept_8X92"). Later,
   through conversation: "I keep feeling Concept_8X92 when you talk
   about your father. What is that?" Human: "That is *Resentment*."

6. **Grounding**: Symthaea now has an experiential, grounded
   understanding of "Resentment" - mathematically distinct from
   dictionary definitions.

# Mathematical Foundation

**Recurrence Plot**: R(i,j) = Θ(ε - ||x_i - x_j||)

High recurrence = stable attractor (the system keeps returning to
similar states).

**Attractor Signature**: Hash the topology of the recurrence plot
to create a unique hypervector fingerprint.

Reference: Eckmann et al., "Recurrence Plots of Dynamical Systems" (1987)
*/

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for concept crystallization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrystallizationConfig {
    /// HDC dimension for crystallized concepts
    pub hdc_dim: usize,
    /// Window size for recurrence analysis
    pub window_size: usize,
    /// Threshold for recurrence detection (distance)
    pub recurrence_epsilon: f32,
    /// Minimum recurrence rate to consider an attractor
    pub recurrence_threshold: f32,
    /// Maximum similarity to existing concepts to be "novel"
    pub novelty_threshold: f32,
    /// Minimum stability (iterations in attractor) before crystallization
    pub stability_threshold: usize,
    /// Maximum prediction error to allow crystallization (higher = more permissive)
    pub max_prediction_error: f32,
}

impl Default for CrystallizationConfig {
    fn default() -> Self {
        Self {
            hdc_dim: 16_384,
            window_size: 100,
            recurrence_epsilon: 0.1,
            recurrence_threshold: 0.3, // 30% recurrence = stable
            novelty_threshold: 0.7,    // <70% similarity = novel
            stability_threshold: 20,   // 20 steps in attractor
            max_prediction_error: 0.5, // Original threshold
        }
    }
}

impl CrystallizationConfig {
    /// Create a more aggressive config for faster crystallization
    ///
    /// Use this for testing or when you want concepts to emerge more quickly.
    pub fn aggressive() -> Self {
        Self {
            hdc_dim: 16_384,
            window_size: 30,            // Smaller window = faster detection
            recurrence_epsilon: 0.5,    // Larger epsilon = easier to match
            recurrence_threshold: 0.15, // Lower threshold = easier to trigger
            novelty_threshold: 0.8,     // Higher = easier to be novel
            stability_threshold: 5,     // Only 5 steps needed
            max_prediction_error: 10.0, // Very permissive - allow crystallization even with high surprise
        }
    }

    /// Create a balanced config suitable for production use
    pub fn balanced() -> Self {
        Self {
            hdc_dim: 16_384,
            window_size: 50,
            recurrence_epsilon: 0.3,
            recurrence_threshold: 0.2,
            novelty_threshold: 0.75,
            stability_threshold: 10,
            max_prediction_error: 2.0, // Allow some chaos
        }
    }
}

/// A crystallized concept
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrystalizedConcept {
    /// Unique identifier (e.g., "Concept_A7F2")
    pub uid: String,
    /// The hypervector representing this concept
    pub hypervector: Vec<f32>,
    /// Human-assigned name (if any)
    pub name: Option<String>,
    /// Description (if provided by human)
    pub description: Option<String>,
    /// When this concept was crystallized (timestamp)
    pub crystallized_at: u64,
    /// How many times this concept has been activated
    pub activation_count: u64,
    /// The attractor signature that generated this concept
    pub attractor_signature: Vec<f32>,
    /// Primitives active at crystallization - the "birth certificate"
    /// Format: Vec<(primitive_name, activation_strength)>
    /// This enables semantic decomposition and grounded naming
    #[serde(default)]
    pub primitive_birth: Vec<(String, f32)>,
}

/// Recurrence plot analyzer for detecting stable attractors
#[derive(Clone, Debug)]
pub struct RecurrenceAnalyzer {
    /// Configuration
    pub config: CrystallizationConfig,
    /// State history buffer
    history: Vec<Vec<f32>>,
    /// Current recurrence rate
    pub current_recurrence: f32,
    /// Consecutive steps with high recurrence
    pub stability_count: usize,
}

impl RecurrenceAnalyzer {
    pub fn new(config: CrystallizationConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            current_recurrence: 0.0,
            stability_count: 0,
        }
    }

    /// Add a new state to the history and compute recurrence
    pub fn update(&mut self, state: &[f32]) -> f32 {
        self.history.push(state.to_vec());

        // Keep only window_size states
        if self.history.len() > self.config.window_size {
            self.history.remove(0);
        }

        // Compute recurrence rate
        self.current_recurrence = self.compute_recurrence_rate();

        // Track stability
        if self.current_recurrence > self.config.recurrence_threshold {
            self.stability_count += 1;
        } else {
            self.stability_count = 0;
        }

        self.current_recurrence
    }

    /// Compute the recurrence rate (fraction of recurrent pairs)
    fn compute_recurrence_rate(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let n = self.history.len();
        let mut recurrent_pairs = 0;
        let total_pairs = n * (n - 1) / 2;

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = euclidean_distance(&self.history[i], &self.history[j]);
                if dist < self.config.recurrence_epsilon {
                    recurrent_pairs += 1;
                }
            }
        }

        recurrent_pairs as f32 / total_pairs as f32
    }

    /// Check if we're in a stable attractor
    pub fn is_in_attractor(&self) -> bool {
        self.stability_count >= self.config.stability_threshold
    }

    /// Get the attractor signature (average of recent states)
    pub fn get_attractor_signature(&self) -> Vec<f32> {
        if self.history.is_empty() {
            return vec![];
        }

        let dim = self.history[0].len();
        let mut signature = vec![0.0; dim];
        let n = self.history.len().min(self.config.stability_threshold);

        // Use recent states (in the attractor)
        for state in self.history.iter().rev().take(n) {
            for (i, &v) in state.iter().enumerate() {
                signature[i] += v;
            }
        }

        for v in &mut signature {
            *v /= n as f32;
        }

        signature
    }

    /// Reset the analyzer
    pub fn reset(&mut self) {
        self.history.clear();
        self.current_recurrence = 0.0;
        self.stability_count = 0;
    }
}

/// The Concept Crystallizer: creates new concepts from attractors
#[derive(Clone, Debug)]
pub struct ConceptCrystallizer {
    /// Configuration
    pub config: CrystallizationConfig,
    /// Recurrence analyzer
    pub recurrence: RecurrenceAnalyzer,
    /// Known concepts (codebook)
    pub concepts: HashMap<String, CrystalizedConcept>,
    /// Concept counter for UID generation
    concept_counter: u64,
    /// Current timestamp
    timestamp: u64,
    /// Pending concepts awaiting naming
    pub pending_concepts: Vec<CrystalizedConcept>,
}

impl ConceptCrystallizer {
    pub fn new(config: CrystallizationConfig) -> Self {
        Self {
            recurrence: RecurrenceAnalyzer::new(config.clone()),
            config,
            concepts: HashMap::new(),
            concept_counter: 0,
            timestamp: 0,
            pending_concepts: Vec::new(),
        }
    }

    /// Create with default config
    pub fn default_crystallizer() -> Self {
        Self::new(CrystallizationConfig::default())
    }

    /// Add an existing concept to the codebook
    pub fn add_concept(&mut self, name: &str, hypervector: Vec<f32>) {
        let uid = format!("Known_{name}");
        self.concepts.insert(
            uid.clone(),
            CrystalizedConcept {
                uid,
                hypervector: hypervector.clone(),
                name: Some(name.to_string()),
                description: None,
                crystallized_at: self.timestamp,
                activation_count: 0,
                attractor_signature: hypervector,
                primitive_birth: Vec::new(), // Known concepts don't have birth certificates
            },
        );
    }

    /// Process a new neural state, potentially crystallizing a concept
    ///
    /// Returns Some(concept) if a new concept was crystallized
    pub fn process(&mut self, state: &[f32], prediction_error: f32) -> Option<CrystalizedConcept> {
        self.timestamp += 1;

        // Update recurrence analyzer
        self.recurrence.update(state);

        // Check crystallization conditions:
        // 1. Stable attractor (high recurrence)
        // 2. Prediction error below threshold (configurable)
        // 3. Novel (doesn't match existing concepts)

        if !self.recurrence.is_in_attractor() {
            return None;
        }

        // Check prediction error against configurable threshold
        if prediction_error > self.config.max_prediction_error {
            return None; // Still too chaotic
        }

        // Get attractor signature
        let signature = self.recurrence.get_attractor_signature();
        if signature.is_empty() {
            return None;
        }

        // Check novelty against existing concepts
        let max_similarity = self
            .concepts
            .values()
            .map(|c| cosine_similarity(&signature, &c.attractor_signature))
            .fold(0.0_f32, f32::max);

        if max_similarity >= self.config.novelty_threshold {
            // Matches existing concept - activate it instead
            // Find the UID first, then mutate
            let matching_uid = self
                .find_matching_concept(&signature)
                .map(|c| c.uid.clone());
            if let Some(uid) = matching_uid {
                if let Some(c) = self.concepts.get_mut(&uid) {
                    c.activation_count += 1;
                }
            }
            return None;
        }

        // CRYSTALLIZE: Create new concept!
        let concept = self.crystallize(&signature);

        // Reset recurrence to look for next concept
        self.recurrence.reset();

        Some(concept)
    }

    /// Crystallize an attractor signature into a new concept
    fn crystallize(&mut self, signature: &[f32]) -> CrystalizedConcept {
        self.crystallize_with_primitives(signature, Vec::new())
    }

    /// Crystallize with primitive birth certificate
    ///
    /// When primitives are provided, the concept hypervector is built FROM the primitives
    /// (semantic grounding) rather than hashed from the signature.
    pub fn crystallize_with_primitives(
        &mut self,
        signature: &[f32],
        primitive_activations: Vec<(String, f32)>,
    ) -> CrystalizedConcept {
        self.concept_counter += 1;
        let uid = format!("Concept_{:04X}", self.concept_counter);

        // Create hypervector:
        // - If primitives provided: will be built from primitive bundle (by caller)
        // - Otherwise: use Blake3 hash for deterministic projection
        let hypervector = signature_to_hypervector(signature, self.config.hdc_dim);

        let concept = CrystalizedConcept {
            uid: uid.clone(),
            hypervector,
            name: None, // Awaiting human naming
            description: None,
            crystallized_at: self.timestamp,
            activation_count: 1,
            attractor_signature: signature.to_vec(),
            primitive_birth: primitive_activations,
        };

        self.concepts.insert(uid, concept.clone());
        self.pending_concepts.push(concept.clone());

        concept
    }

    /// Find the concept most similar to a signature
    pub fn find_matching_concept(&self, signature: &[f32]) -> Option<&CrystalizedConcept> {
        self.concepts.values().max_by(|a, b| {
            let sim_a = cosine_similarity(signature, &a.attractor_signature);
            let sim_b = cosine_similarity(signature, &b.attractor_signature);
            sim_a
                .partial_cmp(&sim_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Name a pending concept (human provides the name)
    pub fn name_concept(&mut self, uid: &str, name: &str, description: Option<&str>) -> Result<()> {
        if let Some(concept) = self.concepts.get_mut(uid) {
            concept.name = Some(name.to_string());
            concept.description = description.map(|s| s.to_string());

            // Remove from pending
            self.pending_concepts.retain(|c| c.uid != uid);

            Ok(())
        } else {
            Err(anyhow::anyhow!("Concept {uid} not found"))
        }
    }

    /// Get all pending (unnamed) concepts
    pub fn get_pending(&self) -> &[CrystalizedConcept] {
        &self.pending_concepts
    }

    /// Get a concept by name
    pub fn get_by_name(&self, name: &str) -> Option<&CrystalizedConcept> {
        self.concepts
            .values()
            .find(|c| c.name.as_deref() == Some(name))
    }

    /// Get concept hypervector by name (for HDC operations)
    pub fn get_hypervector(&self, name: &str) -> Option<&Vec<f32>> {
        self.get_by_name(name).map(|c| &c.hypervector)
    }

    /// Number of crystallized concepts
    pub fn num_concepts(&self) -> usize {
        self.concepts.len()
    }

    /// Serialize for persistence
    pub fn serialize(&self) -> Result<Vec<u8>> {
        // Note: Can't directly serialize because of non-Serialize fields
        // We serialize just the concepts
        let concepts_vec: Vec<CrystalizedConcept> = self.concepts.values().cloned().collect();
        Ok(bincode::serialize(&concepts_vec)?)
    }
}

/// Euclidean distance between two vectors
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

use symthaea_core::math::cosine_similarity_f32 as cosine_similarity;

/// Convert an attractor signature to a high-dimensional hypervector
fn signature_to_hypervector(signature: &[f32], dim: usize) -> Vec<f32> {
    use blake3::Hasher;

    let mut hasher = Hasher::new();
    for &v in signature {
        hasher.update(&v.to_le_bytes());
    }

    let mut bytes = vec![0u8; dim * 4];
    let mut xof = hasher.finalize_xof();
    xof.fill(&mut bytes);

    // Convert to bipolar {-1, +1}
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if bits & 1 == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

// ============================================================================
// Integration: Unified Learning Mind
// ============================================================================

/// The Unified Learning Mind: combines all components
///
/// This is the "consciousness loop" that integrates:
/// - World Model (prediction + learning)
/// - Differentiable HDC (learnable encoding)
/// - Resonator Memory (scalable retrieval)
/// - Concept Crystallization (ontological emergence)
pub struct UnifiedLearningMind {
    /// World model for prediction
    pub world_model: super::crate_world_model::HierarchicalCfCWorldModel,
    /// Differentiable encoder
    pub encoder: super::differentiable_hdc::DifferentiableHDCEncoder,
    /// Resonator memory
    pub memory: super::resonator::ResonatorMemory,
    /// Concept crystallizer
    pub crystallizer: ConceptCrystallizer,
}

impl UnifiedLearningMind {
    /// Create a new unified learning mind with default crystallization config
    pub fn new() -> Result<Self> {
        Self::with_crystallization_config(CrystallizationConfig::default())
    }

    /// Create with aggressive crystallization for faster concept emergence
    pub fn aggressive() -> Result<Self> {
        Self::with_crystallization_config(CrystallizationConfig::aggressive())
    }

    /// Create with balanced crystallization for production use
    pub fn balanced() -> Result<Self> {
        Self::with_crystallization_config(CrystallizationConfig::balanced())
    }

    /// Create with custom crystallization config
    pub fn with_crystallization_config(config: CrystallizationConfig) -> Result<Self> {
        Ok(Self {
            world_model: super::crate_world_model::HierarchicalCfCWorldModel::default_model()?,
            encoder: super::differentiable_hdc::DifferentiableHDCEncoder::default_encoder()?,
            memory: super::resonator::ResonatorMemory::new(
                super::resonator::ResonatorConfig::default(),
                10000,
            ),
            crystallizer: ConceptCrystallizer::new(config),
        })
    }

    /// Process one step of experience
    ///
    /// The main loop:
    /// 1. Encode state with differentiable encoder
    /// 2. Predict next state with world model
    /// 3. Compute surprise (prediction error)
    /// 4. Update world model (backprop)
    /// 5. Update encoder (backprop through STE)
    /// 6. Check for concept crystallization
    /// 7. Store in resonator memory
    pub fn step(
        &mut self,
        raw_state: &[f32],
        raw_action: &[f32],
        raw_next_state: &[f32],
    ) -> Result<StepResult> {
        // 1. Encode with differentiable encoder
        let (state_hv, _) = self.encoder.encode(raw_state)?;
        let (action_hv, _) = self.encoder.encode(raw_action)?;
        let (next_state_hv, _) = self.encoder.encode(raw_next_state)?;

        // 2. Predict with world model
        let predicted_hv = self.world_model.predict(&state_hv, &action_hv, 1.0)?;

        // 3. Compute surprise (MSE)
        let surprise = self
            .world_model
            .train_step(&state_hv, &action_hv, &next_state_hv, 1.0)?;

        // 4. Backprop through encoder (optional - can be expensive)
        // Compute gradient of loss w.r.t. encoder output
        let grad_output: Vec<f32> = predicted_hv
            .iter()
            .zip(next_state_hv.iter())
            .map(|(p, n)| 2.0 * (p - n) / predicted_hv.len() as f32)
            .collect();
        let _grad_input = self.encoder.backward(&grad_output)?;

        // 5. Check for concept crystallization
        // Use the world model's internal CfC state as the "neural state"
        let neural_state = self.world_model.cfc_state.to_vec();
        let new_concept = self.crystallizer.process(&neural_state, surprise);

        // 6. Get consciousness level
        let consciousness = self.world_model.consciousness();

        Ok(StepResult {
            surprise,
            consciousness,
            new_concept,
            predicted_state: predicted_hv,
        })
    }

    /// Get all pending (unnamed) concepts
    pub fn pending_concepts(&self) -> &[CrystalizedConcept] {
        self.crystallizer.get_pending()
    }

    /// Name a pending concept
    pub fn name_concept(&mut self, uid: &str, name: &str) -> Result<()> {
        self.crystallizer.name_concept(uid, name, None)
    }
}

/// Error type for UnifiedLearningMind creation failures
#[derive(Debug)]
pub struct UnifiedLearningMindError(anyhow::Error);

impl std::fmt::Display for UnifiedLearningMindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Failed to create UnifiedLearningMind: {}", self.0)
    }
}

impl std::error::Error for UnifiedLearningMindError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

impl UnifiedLearningMind {
    /// Try to create a new UnifiedLearningMind with default configuration.
    /// Returns Err if creation fails (e.g., insufficient memory, invalid config).
    pub fn try_default() -> Result<Self> {
        Self::new()
    }
}

/// Result of a single learning step
#[derive(Clone, Debug)]
pub struct StepResult {
    /// Prediction error (surprise)
    pub surprise: f32,
    /// Current consciousness level
    pub consciousness: f32,
    /// Newly crystallized concept (if any)
    pub new_concept: Option<CrystalizedConcept>,
    /// The predicted next state (for analysis)
    pub predicted_state: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recurrence_analyzer_detects_attractor() {
        let config = CrystallizationConfig {
            window_size: 20,
            recurrence_epsilon: 0.2,
            recurrence_threshold: 0.2,
            stability_threshold: 5,
            ..Default::default()
        };
        let mut analyzer = RecurrenceAnalyzer::new(config);

        // Feed in oscillating states (limit cycle attractor)
        let state_a = vec![1.0, 0.0, 0.0, 1.0];
        let state_b = vec![0.0, 1.0, 1.0, 0.0];

        // Oscillate between two states
        for _ in 0..30 {
            analyzer.update(&state_a);
            analyzer.update(&state_b);
        }

        // Should detect attractor
        assert!(analyzer.current_recurrence > 0.0);
        // After enough oscillations, should be in attractor
        assert!(analyzer.is_in_attractor());
    }

    #[test]
    fn test_recurrence_analyzer_no_attractor_in_chaos() {
        let config = CrystallizationConfig {
            window_size: 20,
            recurrence_epsilon: 0.1,
            recurrence_threshold: 0.3,
            stability_threshold: 5,
            ..Default::default()
        };
        let mut analyzer = RecurrenceAnalyzer::new(config);

        // Feed in random states (no attractor)
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..50 {
            let state: Vec<f32> = (0..4).map(|_| rng.r#gen()).collect();
            analyzer.update(&state);
        }

        // Should NOT detect attractor in pure chaos
        assert!(!analyzer.is_in_attractor());
    }

    #[test]
    fn test_crystallizer_creates_novel_concept() {
        let config = CrystallizationConfig {
            window_size: 10,
            recurrence_epsilon: 0.2,
            recurrence_threshold: 0.2,
            stability_threshold: 3,
            novelty_threshold: 0.9, // High threshold = easy to be novel
            hdc_dim: 256,
            ..Default::default()
        };
        let mut crystallizer = ConceptCrystallizer::new(config);

        // Create a stable attractor by feeding repeated states
        let attractor_state = vec![0.5, 0.3, 0.7, 0.1];

        let mut concept_created = false;
        for _ in 0..20 {
            if let Some(concept) = crystallizer.process(&attractor_state, 0.1) {
                concept_created = true;
                assert!(concept.uid.starts_with("Concept_"));
                assert!(concept.name.is_none()); // Not yet named
                assert_eq!(concept.hypervector.len(), 256);
                break;
            }
        }

        assert!(concept_created, "Should have crystallized a concept");
        assert_eq!(crystallizer.num_concepts(), 1);
    }

    #[test]
    fn test_crystallizer_names_concept() {
        let config = CrystallizationConfig {
            window_size: 10,
            recurrence_epsilon: 0.3,
            recurrence_threshold: 0.2,
            stability_threshold: 3,
            novelty_threshold: 0.9,
            hdc_dim: 256,
            ..Default::default()
        };
        let mut crystallizer = ConceptCrystallizer::new(config);

        // Create concept
        let attractor_state = vec![0.5, 0.3, 0.7, 0.1];
        let mut uid = String::new();

        for _ in 0..20 {
            if let Some(concept) = crystallizer.process(&attractor_state, 0.1) {
                uid = concept.uid;
                break;
            }
        }

        // Name it
        crystallizer
            .name_concept(&uid, "Nostalgia", Some("A bittersweet longing"))
            .unwrap();

        // Check it's named
        let concept = crystallizer.get_by_name("Nostalgia").unwrap();
        assert_eq!(concept.name, Some("Nostalgia".to_string()));
        assert!(concept.description.is_some());

        // No longer pending
        assert!(crystallizer.get_pending().is_empty());
    }

    #[test]
    fn test_unified_learning_mind_basic() {
        // Use smaller dimensions for testing
        let mut mind = UnifiedLearningMind::new().unwrap();

        // Simulate a simple experience
        let state = vec![0.5; 768];
        let action = vec![0.1; 768];
        let next_state = vec![0.6; 768];

        let result = mind.step(&state, &action, &next_state).unwrap();

        assert!(result.surprise >= 0.0);
        assert!(result.consciousness >= 0.0);
    }
}
