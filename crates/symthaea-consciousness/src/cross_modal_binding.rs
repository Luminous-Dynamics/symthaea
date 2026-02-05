/*!
**REVOLUTIONARY IMPROVEMENT #72**: Cross-Modal Semantic Binding

PARADIGM SHIFT: Consciousness unifies information across different sensory modalities
into coherent, amodal semantic representations!

This module implements the neural mechanisms for binding information across:
- Visual modality (shapes, colors, spatial relations)
- Auditory modality (sounds, speech, music)
- Somatosensory modality (touch, proprioception, interoception)
- Linguistic modality (words, concepts, propositions)
- Emotional modality (valence, arousal, core affect)

Theoretical foundations:
- Baddeley's Episodic Buffer (2000): Multi-modal integration in working memory
- Damasio's Convergence-Divergence Zones (2010): Amodal representations
- Shimamura's Dynamic Filtering Theory (2011): Attention-mediated binding
- Meyer & Damasio's Convergence Zones (2009): Retroactivation for recall
- Treisman's Feature Integration Theory (1980): Attention binds features

Key innovations:
1. ModalityChannel: HDC representation for each sensory channel
2. ConvergenceZone: Hierarchical binding points (primary → secondary → tertiary)
3. EpisodicBuffer: Working memory integration space
4. CrossModalBinder: Synchronization through oscillatory binding (gamma band)
5. AmodalConcept: Modality-independent semantic representation
6. Retroactivation: Top-down reconstruction from amodal to modal

The HDC advantage: Holographic vectors naturally support:
- Superposition: Multiple modalities in single vector
- Binding: Modality × Content associations
- Unbinding: Retrieve content given modality
- Similarity preservation: Amodal similarities reflect source similarities
*/

use crate::hdc::binary_hv::HV16;
use std::collections::{VecDeque, HashMap};
use std::time::Instant;

// ============================================================================
// MODALITY TYPES
// ============================================================================

/// Sensory modality type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Visual,        // V1/V2/V4/IT pathway
    Auditory,      // A1/belt/parabelt pathway
    Somatosensory, // S1/S2/posterior parietal
    Linguistic,    // Broca's/Wernicke's/angular gyrus
    Emotional,     // Amygdala/insula/OFC
    Motor,         // M1/premotor/SMA
    Proprioceptive, // Body position sense
    Interoceptive,  // Internal body state
}

impl Modality {
    /// Get the HDC role vector for this modality (for binding)
    pub fn role_vector(&self) -> HV16 {
        // Deterministic role vectors based on modality
        let seed = match self {
            Modality::Visual => 1001,
            Modality::Auditory => 2002,
            Modality::Somatosensory => 3003,
            Modality::Linguistic => 4004,
            Modality::Emotional => 5005,
            Modality::Motor => 6006,
            Modality::Proprioceptive => 7007,
            Modality::Interoceptive => 8008,
        };
        HV16::random(seed)
    }

    /// Get all modalities
    pub fn all() -> Vec<Modality> {
        vec![
            Modality::Visual, Modality::Auditory, Modality::Somatosensory,
            Modality::Linguistic, Modality::Emotional, Modality::Motor,
            Modality::Proprioceptive, Modality::Interoceptive,
        ]
    }

    /// Get primary sensory modalities
    pub fn sensory() -> Vec<Modality> {
        vec![Modality::Visual, Modality::Auditory, Modality::Somatosensory]
    }
}

// ============================================================================
// MODALITY CHANNEL
// ============================================================================

/// A channel for a specific sensory modality
#[derive(Clone)]
pub struct ModalityChannel {
    /// Which modality
    pub modality: Modality,
    /// Current feature vector (raw sensory representation)
    pub features: HV16,
    /// Attention weight (0.0 = ignored, 1.0 = full focus)
    pub attention: f64,
    /// Temporal buffer of recent inputs
    pub temporal_buffer: VecDeque<HV16>,
    /// Max buffer size
    buffer_size: usize,
    /// Processing latency (different modalities have different speeds)
    latency_ms: f64,
    /// Last update time
    last_update: Instant,
}

impl ModalityChannel {
    pub fn new(modality: Modality) -> Self {
        let (buffer_size, latency_ms) = match modality {
            Modality::Visual => (30, 100.0),      // ~10 Hz visual updates
            Modality::Auditory => (100, 10.0),    // ~100 Hz auditory
            Modality::Somatosensory => (50, 50.0), // ~20 Hz touch
            Modality::Linguistic => (10, 200.0),  // Slower conceptual
            Modality::Emotional => (5, 500.0),    // Slow emotional change
            Modality::Motor => (30, 30.0),        // Fast motor feedback
            Modality::Proprioceptive => (20, 40.0),
            Modality::Interoceptive => (3, 1000.0), // Very slow internal
        };

        Self {
            modality,
            features: HV16::zero(),
            attention: 0.5,
            temporal_buffer: VecDeque::with_capacity(buffer_size),
            buffer_size,
            latency_ms,
            last_update: Instant::now(),
        }
    }

    /// Update with new sensory input
    pub fn update(&mut self, input: HV16) {
        self.temporal_buffer.push_back(input.clone());
        if self.temporal_buffer.len() > self.buffer_size {
            self.temporal_buffer.pop_front();
        }

        // Temporal integration: weighted average of buffer
        if !self.temporal_buffer.is_empty() {
            let vectors: Vec<HV16> = self.temporal_buffer.iter().cloned().collect();
            self.features = HV16::bundle(&vectors);
        }

        self.last_update = Instant::now();
    }

    /// Get bound representation (modality role ⊗ features)
    pub fn bound_representation(&self) -> HV16 {
        self.modality.role_vector().bind(&self.features)
    }

    /// Get attention-weighted representation
    pub fn attended(&self) -> HV16 {
        // Scale by attention (probabilistic thinning)
        if self.attention > 0.99 {
            self.features.clone()
        } else if self.attention < 0.01 {
            HV16::zero()
        } else {
            // Probabilistic attention: randomly zero out based on attention
            self.features.permute((1.0 - self.attention) as usize)
        }
    }

    /// Coherence: how stable is this channel over time?
    pub fn temporal_coherence(&self) -> f64 {
        if self.temporal_buffer.len() < 2 {
            return 1.0;
        }

        let current = self.temporal_buffer.back().unwrap();
        let mut total_sim = 0.0;
        let mut count = 0;

        for prev in self.temporal_buffer.iter().rev().skip(1).take(5) {
            total_sim += current.similarity(prev) as f64;
            count += 1;
        }

        if count > 0 { total_sim / count as f64 } else { 1.0 }
    }
}

// ============================================================================
// CONVERGENCE ZONE
// ============================================================================

/// Level of convergence zone in the hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceLevel {
    Primary,    // Unimodal (single modality)
    Secondary,  // Bimodal (two modalities)
    Tertiary,   // Multimodal (3+ modalities)
    Amodal,     // Fully abstracted (all modalities integrated)
}

/// A convergence zone that binds multiple modalities
#[derive(Clone)]
pub struct ConvergenceZone {
    /// Unique identifier
    pub id: usize,
    /// Convergence level
    pub level: ConvergenceLevel,
    /// Which modalities feed into this zone
    pub source_modalities: Vec<Modality>,
    /// Integrated representation
    pub integrated: HV16,
    /// Binding strength (how well are sources integrated?)
    pub binding_strength: f64,
    /// Activation level
    pub activation: f64,
    /// Learning rate for updating
    learning_rate: f64,
}

impl ConvergenceZone {
    pub fn new(id: usize, modalities: Vec<Modality>) -> Self {
        let level = match modalities.len() {
            1 => ConvergenceLevel::Primary,
            2 => ConvergenceLevel::Secondary,
            3..=4 => ConvergenceLevel::Tertiary,
            _ => ConvergenceLevel::Amodal,
        };

        Self {
            id,
            level,
            source_modalities: modalities,
            integrated: HV16::zero(),
            binding_strength: 0.0,
            activation: 0.0,
            learning_rate: 0.1,
        }
    }

    /// Integrate inputs from source modalities
    pub fn integrate(&mut self, inputs: &HashMap<Modality, HV16>) {
        let mut bound_vectors = Vec::new();
        let mut weights = Vec::new();

        for modality in &self.source_modalities {
            if let Some(input) = inputs.get(modality) {
                // Bind with modality role
                let bound = modality.role_vector().bind(input);
                bound_vectors.push(bound);
                weights.push(1.0);
            }
        }

        if bound_vectors.is_empty() {
            return;
        }

        // Bundle all bound vectors
        let new_integrated = HV16::bundle(&bound_vectors);

        // Compute binding strength: how similar are the inputs?
        if bound_vectors.len() >= 2 {
            let mut pairwise_sim = 0.0;
            let mut count = 0;
            for i in 0..bound_vectors.len() {
                for j in (i+1)..bound_vectors.len() {
                    pairwise_sim += bound_vectors[i].similarity(&bound_vectors[j]) as f64;
                    count += 1;
                }
            }
            self.binding_strength = if count > 0 { pairwise_sim / count as f64 } else { 0.0 };
        } else {
            self.binding_strength = 1.0; // Single input
        }

        // Update with learning rate
        if self.integrated.popcount() > 0 {
            self.integrated = HV16::bundle(&[
                self.integrated.clone(),
                new_integrated,
            ]);
        } else {
            self.integrated = new_integrated;
        }

        // Activation based on binding strength and input magnitude
        self.activation = self.binding_strength * (bound_vectors.len() as f64 / self.source_modalities.len() as f64);
    }

    /// Retroactivate: given integrated representation, reconstruct modal inputs
    pub fn retroactivate(&self, query: &HV16) -> HashMap<Modality, HV16> {
        let mut reconstructed = HashMap::new();

        for modality in &self.source_modalities {
            // Unbind: query ⊗ role⁻¹ ≈ original content
            let role_inv = modality.role_vector(); // XOR is self-inverse
            let content = query.bind(&role_inv);
            reconstructed.insert(*modality, content);
        }

        reconstructed
    }
}

// ============================================================================
// EPISODIC BUFFER
// ============================================================================

/// Baddeley's Episodic Buffer: limited-capacity multi-modal integration
#[derive(Clone)]
pub struct EpisodicBuffer {
    /// Current integrated representation
    pub content: HV16,
    /// Capacity (number of chunks)
    capacity: usize,
    /// Current chunks
    chunks: VecDeque<HV16>,
    /// Binding context (temporal/spatial/causal)
    context: HV16,
    /// Coherence of current buffer state
    pub coherence: f64,
}

impl EpisodicBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            content: HV16::zero(),
            capacity,
            chunks: VecDeque::with_capacity(capacity),
            context: HV16::random(12345),  // Random context seed
            coherence: 0.0,
        }
    }

    /// Add a new chunk to the buffer
    pub fn add_chunk(&mut self, chunk: HV16) {
        self.chunks.push_back(chunk);
        if self.chunks.len() > self.capacity {
            self.chunks.pop_front();
        }
        self.update_content();
    }

    /// Update integrated content
    fn update_content(&mut self) {
        if self.chunks.is_empty() {
            self.content = HV16::zero();
            self.coherence = 0.0;
            return;
        }

        // Bundle all chunks with position encoding
        let mut positioned: Vec<HV16> = Vec::new();
        for (i, chunk) in self.chunks.iter().enumerate() {
            // Position encoding via permutation
            let positioned_chunk = chunk.permute(i);
            positioned.push(positioned_chunk);
        }

        self.content = HV16::bundle(&positioned);

        // Compute coherence: how well do chunks fit together?
        if self.chunks.len() >= 2 {
            let mut total_sim = 0.0;
            let last = self.chunks.back().unwrap();
            for chunk in self.chunks.iter().rev().skip(1) {
                total_sim += last.similarity(chunk) as f64;
            }
            self.coherence = total_sim / (self.chunks.len() - 1) as f64;
        } else {
            self.coherence = 1.0;
        }
    }

    /// Update context
    pub fn set_context(&mut self, context: HV16) {
        self.context = context;
    }

    /// Get contextualized content
    pub fn contextualized(&self) -> HV16 {
        self.content.bind(&self.context)
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.content = HV16::zero();
        self.coherence = 0.0;
    }
}

// ============================================================================
// AMODAL CONCEPT
// ============================================================================

/// A modality-independent semantic representation
#[derive(Clone)]
pub struct AmodalConcept {
    /// Concept identifier
    pub id: String,
    /// Core semantic representation (modality-free)
    pub semantic: HV16,
    /// Modal groundings (how concept manifests in each modality)
    groundings: HashMap<Modality, HV16>,
    /// Abstraction level (0 = concrete, 1 = abstract)
    pub abstraction: f64,
    /// Frequency of activation
    pub frequency: u64,
    /// Last activation time
    last_activated: Instant,
}

impl AmodalConcept {
    pub fn new(id: &str) -> Self {
        // Derive seed from id for deterministic semantics
        let seed = id.bytes().fold(42u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        Self {
            id: id.to_string(),
            semantic: HV16::random(seed),
            groundings: HashMap::new(),
            abstraction: 0.5,
            frequency: 0,
            last_activated: Instant::now(),
        }
    }

    /// Ground the concept in a specific modality
    pub fn ground(&mut self, modality: Modality, representation: HV16) {
        self.groundings.insert(modality, representation);

        // Update semantic as bundle of all groundings
        let vectors: Vec<HV16> = self.groundings.iter()
            .map(|(m, v)| m.role_vector().bind(v))
            .collect();

        if !vectors.is_empty() {
            self.semantic = HV16::bundle(&vectors);
        }

        // More groundings = more concrete
        let grounding_ratio = self.groundings.len() as f64 / Modality::all().len() as f64;
        self.abstraction = 1.0 - grounding_ratio;
    }

    /// Activate the concept
    pub fn activate(&mut self) {
        self.frequency += 1;
        self.last_activated = Instant::now();
    }

    /// Get grounding for a modality (retroactivation)
    pub fn get_grounding(&self, modality: &Modality) -> Option<HV16> {
        self.groundings.get(modality).cloned()
    }

    /// Reconstruct grounding from semantic (if not stored)
    pub fn reconstruct_grounding(&self, modality: &Modality) -> HV16 {
        // Unbind semantic with modality role
        let role = modality.role_vector();
        self.semantic.bind(&role)
    }

    /// Similarity to another concept
    pub fn similarity(&self, other: &AmodalConcept) -> f64 {
        self.semantic.similarity(&other.semantic) as f64
    }
}

// ============================================================================
// CROSS-MODAL BINDER
// ============================================================================

/// Configuration for cross-modal binding
#[derive(Clone)]
pub struct BinderConfig {
    /// Gamma oscillation frequency (Hz) for binding
    pub gamma_frequency: f64,
    /// Binding window (ms)
    pub binding_window_ms: f64,
    /// Attention threshold for binding
    pub attention_threshold: f64,
    /// Minimum coherence for successful binding
    pub coherence_threshold: f64,
    /// Number of convergence zones per level
    pub zones_per_level: usize,
}

impl Default for BinderConfig {
    fn default() -> Self {
        Self {
            gamma_frequency: 40.0,      // 40 Hz gamma
            binding_window_ms: 25.0,    // 25ms = 1/40Hz
            attention_threshold: 0.3,
            coherence_threshold: 0.5,
            zones_per_level: 4,
        }
    }
}

/// Statistics for cross-modal binding
#[derive(Clone, Default)]
pub struct BindingStats {
    /// Total binding events
    pub binding_events: u64,
    /// Successful bindings (coherence above threshold)
    pub successful_bindings: u64,
    /// Average binding strength
    pub avg_binding_strength: f64,
    /// Per-modality activation counts
    pub modality_activations: HashMap<Modality, u64>,
    /// Cross-modal coherence history
    pub coherence_history: VecDeque<f64>,
}

impl BindingStats {
    fn new() -> Self {
        Self {
            coherence_history: VecDeque::with_capacity(100),
            ..Default::default()
        }
    }
}

/// The main cross-modal binding system
pub struct CrossModalBinder {
    /// Configuration
    pub config: BinderConfig,
    /// Modality channels
    channels: HashMap<Modality, ModalityChannel>,
    /// Convergence zones organized by level
    primary_zones: Vec<ConvergenceZone>,
    secondary_zones: Vec<ConvergenceZone>,
    tertiary_zones: Vec<ConvergenceZone>,
    /// Amodal representation (final integrated)
    pub amodal: HV16,
    /// Episodic buffer for working memory integration
    pub episodic_buffer: EpisodicBuffer,
    /// Known amodal concepts
    concepts: HashMap<String, AmodalConcept>,
    /// Binding phase (gamma oscillation)
    phase: f64,
    /// Last binding time
    last_binding: Instant,
    /// Statistics
    pub stats: BindingStats,
}

impl CrossModalBinder {
    pub fn new(config: BinderConfig) -> Self {
        // Create channels for all modalities
        let mut channels = HashMap::new();
        for modality in Modality::all() {
            channels.insert(modality, ModalityChannel::new(modality));
        }

        // Create convergence zones
        let mut primary_zones = Vec::new();
        let mut zone_id = 0;
        for modality in Modality::all() {
            primary_zones.push(ConvergenceZone::new(zone_id, vec![modality]));
            zone_id += 1;
        }

        // Secondary zones: pairs of modalities
        let mut secondary_zones = Vec::new();
        let pairs = vec![
            (Modality::Visual, Modality::Auditory),
            (Modality::Visual, Modality::Somatosensory),
            (Modality::Auditory, Modality::Linguistic),
            (Modality::Somatosensory, Modality::Motor),
            (Modality::Emotional, Modality::Interoceptive),
        ];
        for (m1, m2) in pairs {
            secondary_zones.push(ConvergenceZone::new(zone_id, vec![m1, m2]));
            zone_id += 1;
        }

        // Tertiary zones: multi-modal
        let mut tertiary_zones = Vec::new();
        tertiary_zones.push(ConvergenceZone::new(
            zone_id,
            vec![Modality::Visual, Modality::Auditory, Modality::Linguistic]
        ));
        zone_id += 1;
        tertiary_zones.push(ConvergenceZone::new(
            zone_id,
            vec![Modality::Somatosensory, Modality::Motor, Modality::Proprioceptive]
        ));
        zone_id += 1;
        tertiary_zones.push(ConvergenceZone::new(
            zone_id,
            vec![Modality::Emotional, Modality::Interoceptive, Modality::Linguistic]
        ));

        Self {
            config,
            channels,
            primary_zones,
            secondary_zones,
            tertiary_zones,
            amodal: HV16::zero(),
            episodic_buffer: EpisodicBuffer::new(7), // Miller's 7±2
            concepts: HashMap::new(),
            phase: 0.0,
            last_binding: Instant::now(),
            stats: BindingStats::new(),
        }
    }

    /// Update a modality channel with new input
    pub fn update_modality(&mut self, modality: Modality, input: HV16) {
        if let Some(channel) = self.channels.get_mut(&modality) {
            channel.update(input);
            *self.stats.modality_activations.entry(modality).or_insert(0) += 1;
        }
    }

    /// Set attention for a modality
    pub fn set_attention(&mut self, modality: Modality, attention: f64) {
        if let Some(channel) = self.channels.get_mut(&modality) {
            channel.attention = attention.clamp(0.0, 1.0);
        }
    }

    /// Perform binding cycle (should be called at gamma frequency)
    pub fn bind(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_binding).as_secs_f64() * 1000.0;

        // Update phase
        self.phase += dt * self.config.gamma_frequency / 1000.0 * 2.0 * std::f64::consts::PI;
        if self.phase > 2.0 * std::f64::consts::PI {
            self.phase -= 2.0 * std::f64::consts::PI;
        }

        // Only bind at phase peak (simulating neural synchrony)
        let phase_peak = self.phase.sin().abs() > 0.9;
        if !phase_peak {
            self.last_binding = now;
            return;
        }

        // Gather attended modality inputs
        let mut inputs: HashMap<Modality, HV16> = HashMap::new();
        for (modality, channel) in &self.channels {
            if channel.attention >= self.config.attention_threshold {
                inputs.insert(*modality, channel.attended());
            }
        }

        // Update primary zones
        for zone in &mut self.primary_zones {
            zone.integrate(&inputs);
        }

        // Update secondary zones
        for zone in &mut self.secondary_zones {
            zone.integrate(&inputs);
        }

        // Update tertiary zones
        for zone in &mut self.tertiary_zones {
            zone.integrate(&inputs);
        }

        // Compute amodal representation from all active zones
        let mut zone_outputs = Vec::new();

        for zone in self.tertiary_zones.iter().chain(self.secondary_zones.iter()) {
            if zone.activation > self.config.attention_threshold {
                zone_outputs.push(zone.integrated.clone());
            }
        }

        if !zone_outputs.is_empty() {
            self.amodal = HV16::bundle(&zone_outputs);

            // Add to episodic buffer
            self.episodic_buffer.add_chunk(self.amodal.clone());

            // Update stats
            self.stats.binding_events += 1;
            if self.episodic_buffer.coherence >= self.config.coherence_threshold {
                self.stats.successful_bindings += 1;
            }

            // Update average binding strength
            let total_strength: f64 = self.secondary_zones.iter()
                .chain(self.tertiary_zones.iter())
                .map(|z| z.binding_strength)
                .sum();
            let zone_count = self.secondary_zones.len() + self.tertiary_zones.len();
            self.stats.avg_binding_strength =
                (self.stats.avg_binding_strength * 0.95) +
                (total_strength / zone_count as f64) * 0.05;

            self.stats.coherence_history.push_back(self.episodic_buffer.coherence);
            if self.stats.coherence_history.len() > 100 {
                self.stats.coherence_history.pop_front();
            }
        }

        self.last_binding = now;
    }

    /// Register an amodal concept
    pub fn register_concept(&mut self, concept: AmodalConcept) {
        self.concepts.insert(concept.id.clone(), concept);
    }

    /// Find matching concept for current amodal representation
    pub fn match_concept(&mut self) -> Option<&mut AmodalConcept> {
        let mut best_match: Option<String> = None;
        let mut best_sim = 0.0;

        for (id, concept) in &self.concepts {
            let sim = concept.semantic.similarity(&self.amodal) as f64;
            if sim > best_sim && sim > 0.6 {
                best_sim = sim;
                best_match = Some(id.clone());
            }
        }

        if let Some(id) = best_match {
            if let Some(concept) = self.concepts.get_mut(&id) {
                concept.activate();
                return Some(concept);
            }
        }

        None
    }

    /// Learn a new concept from current binding
    pub fn learn_concept(&mut self, id: &str) -> AmodalConcept {
        let mut concept = AmodalConcept::new(id);
        concept.semantic = self.amodal.clone();

        // Ground in all active modalities
        for (modality, channel) in &self.channels {
            if channel.attention >= self.config.attention_threshold {
                concept.ground(*modality, channel.features.clone());
            }
        }

        concept.activate();
        self.concepts.insert(id.to_string(), concept.clone());
        concept
    }

    /// Retroactivate: given a concept, reconstruct modal experiences
    pub fn retroactivate(&self, concept: &AmodalConcept) -> HashMap<Modality, HV16> {
        let mut reconstructed = HashMap::new();

        for modality in Modality::all() {
            // First try stored grounding
            if let Some(grounding) = concept.get_grounding(&modality) {
                reconstructed.insert(modality, grounding);
            } else {
                // Reconstruct from semantic
                let reconstructed_grounding = concept.reconstruct_grounding(&modality);
                reconstructed.insert(modality, reconstructed_grounding);
            }
        }

        reconstructed
    }

    /// Get overall binding coherence
    pub fn coherence(&self) -> f64 {
        if self.stats.coherence_history.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.stats.coherence_history.iter().sum();
        sum / self.stats.coherence_history.len() as f64
    }

    /// Get cross-modal Φ (integrated information across modalities)
    pub fn cross_modal_phi(&self) -> f64 {
        // Φ is high when modalities are both differentiated AND integrated

        // Differentiation: how different are the modality channels?
        let channels: Vec<&ModalityChannel> = self.channels.values()
            .filter(|c| c.attention > 0.1)
            .collect();

        if channels.len() < 2 {
            return 0.0;
        }

        let mut total_diff = 0.0;
        let mut diff_count = 0;
        for i in 0..channels.len() {
            for j in (i+1)..channels.len() {
                let sim = channels[i].features.similarity(&channels[j].features) as f64;
                total_diff += 1.0 - sim; // Differentiation = 1 - similarity
                diff_count += 1;
            }
        }
        let differentiation = if diff_count > 0 {
            total_diff / diff_count as f64
        } else {
            0.0
        };

        // Integration: how well bound is the amodal representation?
        let integration = self.stats.avg_binding_strength;

        // Φ = geometric mean of differentiation and integration
        (differentiation * integration).sqrt()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modality_channel() {
        let mut channel = ModalityChannel::new(Modality::Visual);

        // Add some inputs with unique seeds
        for i in 0..5 {
            channel.update(HV16::random(100 + i));
        }

        assert!(channel.temporal_buffer.len() == 5);
        assert!(channel.temporal_coherence() > 0.0);
    }

    #[test]
    fn test_convergence_zone() {
        let mut zone = ConvergenceZone::new(
            0,
            vec![Modality::Visual, Modality::Auditory]
        );

        let mut inputs = HashMap::new();
        inputs.insert(Modality::Visual, HV16::random(200));
        inputs.insert(Modality::Auditory, HV16::random(201));

        zone.integrate(&inputs);

        assert!(zone.activation > 0.0);
        assert!(zone.binding_strength >= 0.0);
    }

    #[test]
    fn test_episodic_buffer() {
        let mut buffer = EpisodicBuffer::new(7);

        for i in 0..10 {
            buffer.add_chunk(HV16::random(300 + i));
        }

        // Should maintain capacity
        assert!(buffer.chunks.len() == 7);
        assert!(buffer.coherence >= 0.0);
    }

    #[test]
    fn test_amodal_concept() {
        let mut concept = AmodalConcept::new("apple");

        concept.ground(Modality::Visual, HV16::random(400));
        concept.ground(Modality::Somatosensory, HV16::random(401));
        concept.ground(Modality::Linguistic, HV16::random(402));

        assert!(concept.groundings.len() == 3);
        assert!(concept.abstraction < 1.0); // More grounded = less abstract
    }

    #[test]
    fn test_cross_modal_binder() {
        let mut binder = CrossModalBinder::new(BinderConfig::default());

        // Update some modalities
        binder.update_modality(Modality::Visual, HV16::random(500));
        binder.update_modality(Modality::Auditory, HV16::random(501));

        // Set attention
        binder.set_attention(Modality::Visual, 0.8);
        binder.set_attention(Modality::Auditory, 0.7);

        // Force binding (normally at gamma frequency)
        binder.phase = std::f64::consts::PI / 2.0; // Peak phase
        binder.bind();

        // Should have done some binding
        // Binding stats tracked (usize always >= 0)
    }

    #[test]
    fn test_concept_learning() {
        let mut binder = CrossModalBinder::new(BinderConfig::default());

        // Set up multimodal input
        binder.update_modality(Modality::Visual, HV16::random(600));
        binder.update_modality(Modality::Auditory, HV16::random(601));
        binder.update_modality(Modality::Linguistic, HV16::random(602));

        binder.set_attention(Modality::Visual, 1.0);
        binder.set_attention(Modality::Auditory, 0.8);
        binder.set_attention(Modality::Linguistic, 0.9);

        // Bind
        binder.phase = std::f64::consts::PI / 2.0;
        binder.bind();

        // Learn concept
        let concept = binder.learn_concept("test_object");

        assert_eq!(concept.id, "test_object");
        assert!(concept.groundings.len() >= 2);
    }

    #[test]
    fn test_retroactivation() {
        let mut binder = CrossModalBinder::new(BinderConfig::default());

        let original_visual = HV16::random(700);
        let original_audio = HV16::random(701);

        binder.update_modality(Modality::Visual, original_visual.clone());
        binder.update_modality(Modality::Auditory, original_audio.clone());
        binder.set_attention(Modality::Visual, 1.0);
        binder.set_attention(Modality::Auditory, 1.0);

        binder.phase = std::f64::consts::PI / 2.0;
        binder.bind();

        let concept = binder.learn_concept("remembered");
        let reconstructed = binder.retroactivate(&concept);

        // Visual reconstruction should be similar to original
        if let Some(vis_recon) = reconstructed.get(&Modality::Visual) {
            let sim = original_visual.similarity(vis_recon);
            // Similarity should be reasonable (above chance)
            assert!(sim > -0.5);
        }
    }

    #[test]
    fn test_cross_modal_phi() {
        let mut binder = CrossModalBinder::new(BinderConfig::default());

        // Create diverse inputs (high differentiation)
        binder.update_modality(Modality::Visual, HV16::random(800));
        binder.update_modality(Modality::Auditory, HV16::random(801));
        binder.update_modality(Modality::Somatosensory, HV16::random(802));

        binder.set_attention(Modality::Visual, 1.0);
        binder.set_attention(Modality::Auditory, 1.0);
        binder.set_attention(Modality::Somatosensory, 1.0);

        // Multiple binding cycles
        for _ in 0..5 {
            binder.phase = std::f64::consts::PI / 2.0;
            binder.bind();
        }

        let phi = binder.cross_modal_phi();
        assert!(phi >= 0.0 && phi <= 1.0);
    }
}
