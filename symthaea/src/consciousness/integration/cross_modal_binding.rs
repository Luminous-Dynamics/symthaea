// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cross-Modal Binding: Multi-Sensory Integration
//!
//! Provides mechanisms for binding information across different modalities
//! (vision, audio, text, etc.) into unified conscious representations.

use crate::hdc::primitive_system::PrimitiveSystem;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::binary_hv::BinaryHV;

/// Types of sensory modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Visual information
    Visual,
    /// Auditory information
    Auditory,
    /// Textual/linguistic information (alias: Linguistic)
    Textual,
    /// Linguistic information (alias for Textual)
    Linguistic,
    /// Proprioceptive (body sense)
    Proprioceptive,
    /// Somatosensory (touch, pressure, temperature)
    Somatosensory,
    /// Motor control and action
    Motor,
    /// Temporal/sequential
    Temporal,
    /// Spatial
    Spatial,
    /// Emotional/affective
    Affective,
    /// Emotional (alias for Affective)
    Emotional,
    /// Interoceptive (internal body state)
    Interoceptive,
    /// Abstract/conceptual
    Abstract,
}

impl Modality {
    /// Get all modality variants
    pub fn all() -> Vec<Modality> {
        vec![
            Modality::Visual,
            Modality::Auditory,
            Modality::Linguistic,
            Modality::Somatosensory,
            Modality::Motor,
            Modality::Emotional,
            Modality::Interoceptive,
        ]
    }

    /// Get sensory modalities (excluding motor and interoceptive)
    pub fn sensory() -> Vec<Modality> {
        vec![
            Modality::Visual,
            Modality::Auditory,
            Modality::Linguistic,
            Modality::Somatosensory,
        ]
    }
}

// =============================================================================
// MODALITY CHANNEL - For multi-modal integration
// =============================================================================

/// A channel for processing a specific modality
#[derive(Debug, Clone)]
pub struct ModalityChannel {
    /// The modality this channel processes
    pub modality: Modality,
    /// Current feature representation
    pub features: BinaryHV,
    /// Attention weight for this channel (0.0-1.0)
    pub attention: f64,
    /// Temporal buffer for coherence computation
    pub temporal_buffer: VecDeque<BinaryHV>,
    /// Maximum buffer size
    buffer_capacity: usize,
}

impl ModalityChannel {
    /// Create a new modality channel
    pub fn new(modality: Modality) -> Self {
        Self {
            modality,
            features: BinaryHV::zero(),
            attention: 0.5,
            temporal_buffer: VecDeque::with_capacity(8),
            buffer_capacity: 8,
        }
    }

    /// Update channel with new features
    pub fn update(&mut self, features: BinaryHV) {
        // Add to temporal buffer
        if self.temporal_buffer.len() >= self.buffer_capacity {
            self.temporal_buffer.pop_front();
        }
        self.temporal_buffer.push_back(features);
        self.features = features;
    }

    /// Get attended representation (features scaled by attention)
    pub fn attended(&self) -> BinaryHV {
        // For binary HV, attention modulates by probabilistic bit flipping
        // Higher attention = less noise, lower = more noise
        if self.attention >= 0.99 {
            return self.features;
        }
        // Add noise inversely proportional to attention
        let noise_level = (1.0 - self.attention) as f32 * 0.3;
        self.features.add_noise(noise_level, self.modality as u64)
    }

    /// Compute temporal coherence (similarity between recent representations)
    pub fn temporal_coherence(&self) -> f64 {
        if self.temporal_buffer.len() < 2 {
            return 1.0;
        }

        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..self.temporal_buffer.len() - 1 {
            let sim = self.temporal_buffer[i].similarity(&self.temporal_buffer[i + 1]);
            total_sim += sim as f64;
            count += 1;
        }

        if count > 0 {
            total_sim / count as f64
        } else {
            1.0
        }
    }
}

// =============================================================================
// CONVERGENCE ZONE - Hierarchical multi-modal binding
// =============================================================================

/// Level of convergence in the hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConvergenceLevel {
    /// Primary: single modality
    Primary,
    /// Secondary: two modalities
    Secondary,
    /// Tertiary: three modalities
    Tertiary,
    /// Amodal: all sensory modalities
    Amodal,
}

/// A convergence zone that integrates multiple modalities
#[derive(Debug, Clone)]
pub struct ConvergenceZone {
    /// Unique identifier
    pub id: usize,
    /// Source modalities
    pub source_modalities: Vec<Modality>,
    /// Current integrated representation
    pub integrated: BinaryHV,
    /// Binding strength (0.0-1.0)
    pub binding_strength: f64,
    /// Activation level (0.0-1.0)
    pub activation: f64,
    /// Level in hierarchy
    pub level: ConvergenceLevel,
}

impl ConvergenceZone {
    /// Create a new convergence zone
    pub fn new(id: usize, source_modalities: Vec<Modality>) -> Self {
        let level = match source_modalities.len() {
            1 => ConvergenceLevel::Primary,
            2 => ConvergenceLevel::Secondary,
            3 => ConvergenceLevel::Tertiary,
            _ => ConvergenceLevel::Amodal,
        };

        Self {
            id,
            source_modalities,
            integrated: BinaryHV::zero(),
            binding_strength: 0.0,
            activation: 0.0,
            level,
        }
    }

    /// Integrate inputs from source modalities
    pub fn integrate(&mut self, inputs: &HashMap<Modality, BinaryHV>) {
        let mut vectors = Vec::new();
        let mut found_count = 0;

        for modality in &self.source_modalities {
            if let Some(hv) = inputs.get(modality) {
                vectors.push(*hv);
                found_count += 1;
            }
        }

        if vectors.is_empty() {
            self.activation = 0.0;
            return;
        }

        // Integrate via bundling
        self.integrated = BinaryHV::bundle(&vectors);

        // Compute binding strength (average pairwise similarity)
        self.binding_strength = self.compute_binding_strength(&vectors);

        // Activation based on how many modalities are present
        self.activation = found_count as f64 / self.source_modalities.len() as f64;
    }

    /// Compute binding strength from input vectors
    fn compute_binding_strength(&self, vectors: &[BinaryHV]) -> f64 {
        if vectors.len() < 2 {
            return 1.0;
        }

        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..vectors.len() {
            for j in (i + 1)..vectors.len() {
                let sim = vectors[i].similarity(&vectors[j]);
                total_sim += sim as f64;
                count += 1;
            }
        }

        if count > 0 {
            total_sim / count as f64
        } else {
            1.0
        }
    }
}

// =============================================================================
// EPISODIC BUFFER - Short-term multi-modal memory
// =============================================================================

/// Episodic buffer for short-term multi-modal memory
#[derive(Debug, Clone)]
pub struct EpisodicBuffer {
    /// Buffer capacity (Miller's number ~7)
    capacity: usize,
    /// Stored chunks (most recent at back)
    chunks: VecDeque<BinaryHV>,
}

impl EpisodicBuffer {
    /// Create a new episodic buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            chunks: VecDeque::with_capacity(capacity),
        }
    }

    /// Add a chunk to the buffer
    pub fn add_chunk(&mut self, chunk: BinaryHV) {
        if self.chunks.len() >= self.capacity {
            self.chunks.pop_front();
        }
        self.chunks.push_back(chunk);
    }

    /// Get all chunks
    pub fn chunks(&self) -> &VecDeque<BinaryHV> {
        &self.chunks
    }

    /// Get the most recent chunk
    pub fn most_recent(&self) -> Option<&BinaryHV> {
        self.chunks.back()
    }

    /// Compute a bundled representation of buffer contents
    pub fn bundled(&self) -> BinaryHV {
        if self.chunks.is_empty() {
            return BinaryHV::zero();
        }
        let vec: Vec<BinaryHV> = self.chunks.iter().cloned().collect();
        BinaryHV::bundle(&vec)
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Get buffer length
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.chunks.clear();
    }
}

/// Configuration for cross-modal binding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalBindingConfig {
    /// Hypervector dimension
    pub dimension: usize,
    /// Binding strength threshold
    pub binding_threshold: f32,
    /// Decay rate for temporal binding
    pub temporal_decay: f32,
    /// Maximum bindings per modality
    pub max_bindings: usize,
    /// Whether to use attention weighting
    pub use_attention: bool,
}

impl Default for CrossModalBindingConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            binding_threshold: 0.5,
            temporal_decay: 0.1,
            max_bindings: 100,
            use_attention: true,
        }
    }
}

/// A modal representation
#[derive(Debug, Clone)]
pub struct ModalRepresentation {
    /// The modality
    pub modality: Modality,
    /// Hypervector representation
    pub hv: ContinuousHV,
    /// Confidence in this representation
    pub confidence: f32,
    /// Timestamp
    pub timestamp: u64,
    /// Source identifier
    pub source: String,
    /// Attention weight
    pub attention: f32,
}

impl ModalRepresentation {
    /// Create a new modal representation
    pub fn new(
        modality: Modality,
        hv: ContinuousHV,
        confidence: f32,
        source: impl Into<String>,
    ) -> Self {
        Self {
            modality,
            hv,
            confidence,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            source: source.into(),
            attention: 1.0,
        }
    }

    /// Update attention weight
    pub fn with_attention(mut self, attention: f32) -> Self {
        self.attention = attention.clamp(0.0, 1.0);
        self
    }
}

/// Result of a binding operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingResult {
    /// The bound representation (as vector for serialization)
    pub bound_hv: Vec<f32>,
    /// Modalities involved
    pub modalities: Vec<Modality>,
    /// Binding strength
    pub strength: f32,
    /// Coherence measure
    pub coherence: f32,
    /// Individual modal contributions
    pub contributions: HashMap<String, f32>,
}

/// The cross-modal binder
#[derive(Debug)]
pub struct CrossModalBinder {
    /// Configuration
    config: CrossModalBindingConfig,
    /// Current modal representations per modality
    representations: HashMap<Modality, Vec<ModalRepresentation>>,
    /// Current bound state
    current_binding: Option<ContinuousHV>,
    /// Binding history
    binding_history: Vec<BindingResult>,
    /// Statistics
    stats: BinderStats,
}

/// Statistics for the binder
#[derive(Debug, Clone, Default)]
pub struct BinderStats {
    /// Total bindings performed
    pub total_bindings: u64,
    /// Average binding strength
    pub avg_strength: f32,
    /// Average coherence
    pub avg_coherence: f32,
    /// Modality counts
    pub modality_counts: HashMap<Modality, u64>,
}

impl CrossModalBinder {
    /// Create a new cross-modal binder
    pub fn new(config: CrossModalBindingConfig) -> Self {
        Self {
            config,
            representations: HashMap::new(),
            current_binding: None,
            binding_history: Vec::new(),
            stats: BinderStats::default(),
        }
    }

    /// Add a modal representation
    pub fn add_representation(&mut self, repr: ModalRepresentation) {
        let modality = repr.modality;

        let reps = self.representations.entry(modality).or_default();

        // Maintain max bindings
        if reps.len() >= self.config.max_bindings {
            reps.remove(0);
        }

        reps.push(repr);

        // Update stats
        *self.stats.modality_counts.entry(modality).or_insert(0) += 1;
    }

    /// Perform cross-modal binding
    pub fn bind(&mut self) -> Option<BindingResult> {
        if self.representations.is_empty() {
            return None;
        }

        // Collect all representations with attention weighting
        let mut weighted_hvs = Vec::new();
        let mut modalities = Vec::new();
        let mut contributions = HashMap::new();

        for (modality, reps) in &self.representations {
            if let Some(rep) = reps.last() {
                let weight = if self.config.use_attention {
                    rep.attention * rep.confidence
                } else {
                    rep.confidence
                };

                weighted_hvs.push((rep.hv.clone(), weight));
                modalities.push(*modality);
                contributions.insert(format!("{modality:?}"), weight);
            }
        }

        if weighted_hvs.is_empty() {
            return None;
        }

        // Compute weighted bundle
        let _total_weight: f32 = weighted_hvs.iter().map(|(_, w)| w).sum();
        let bound_hv = if weighted_hvs.len() == 1 {
            weighted_hvs[0].0.clone()
        } else {
            let hvs: Vec<&ContinuousHV> = weighted_hvs.iter().map(|(hv, _)| hv).collect();
            ContinuousHV::bundle(&hvs)
        };

        // Calculate binding strength (average pairwise similarity)
        let strength = self.calculate_binding_strength(&weighted_hvs);

        // Calculate coherence
        let coherence = self.calculate_coherence(&weighted_hvs);

        // Update current binding
        self.current_binding = Some(bound_hv.clone());

        // Update statistics
        self.stats.total_bindings += 1;
        let n = self.stats.total_bindings as f32;
        self.stats.avg_strength = (self.stats.avg_strength * (n - 1.0) + strength) / n;
        self.stats.avg_coherence = (self.stats.avg_coherence * (n - 1.0) + coherence) / n;

        let result = BindingResult {
            bound_hv: bound_hv.as_slice().to_vec(),
            modalities,
            strength,
            coherence,
            contributions,
        };

        self.binding_history.push(result.clone());
        Some(result)
    }

    /// Calculate binding strength
    fn calculate_binding_strength(&self, weighted_hvs: &[(ContinuousHV, f32)]) -> f32 {
        if weighted_hvs.len() < 2 {
            return 1.0;
        }

        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..weighted_hvs.len() {
            for j in (i + 1)..weighted_hvs.len() {
                let sim = weighted_hvs[i].0.similarity(&weighted_hvs[j].0);
                total_sim += sim;
                count += 1;
            }
        }

        if count > 0 {
            total_sim / count as f32
        } else {
            1.0
        }
    }

    /// Calculate coherence measure
    fn calculate_coherence(&self, weighted_hvs: &[(ContinuousHV, f32)]) -> f32 {
        if weighted_hvs.is_empty() {
            return 0.0;
        }

        // Coherence based on weight consistency and similarity
        let total_weight: f32 = weighted_hvs.iter().map(|(_, w)| w).sum();
        let avg_weight = total_weight / weighted_hvs.len() as f32;

        let weight_variance: f32 = weighted_hvs
            .iter()
            .map(|(_, w)| (w - avg_weight).powi(2))
            .sum::<f32>()
            / weighted_hvs.len() as f32;

        // Lower variance = higher coherence
        1.0 / (1.0 + weight_variance.sqrt())
    }

    /// Query current binding against a probe
    pub fn query(&self, probe: &ContinuousHV) -> Option<f32> {
        self.current_binding
            .as_ref()
            .map(|binding| binding.similarity(probe))
    }

    /// Unbind a specific modality from current binding
    pub fn unbind(&mut self, modality: Modality) -> Option<ContinuousHV> {
        let current = self.current_binding.as_ref()?;
        let modal_rep = self.representations.get(&modality)?.last()?;

        // Unbind by binding with inverse
        let unbound = current.bind(&modal_rep.hv);
        Some(unbound)
    }

    /// Apply temporal decay
    pub fn decay(&mut self, dt: f32) {
        let decay_factor = (1.0 - self.config.temporal_decay * dt).max(0.0);

        for reps in self.representations.values_mut() {
            for rep in reps.iter_mut() {
                rep.attention *= decay_factor;
            }
            // Remove representations with very low attention
            reps.retain(|r| r.attention > 0.01);
        }
    }

    /// Get current bound state
    pub fn current_binding(&self) -> Option<&ContinuousHV> {
        self.current_binding.as_ref()
    }

    /// Get representations for a modality
    pub fn get_representations(&self, modality: Modality) -> Option<&Vec<ModalRepresentation>> {
        self.representations.get(&modality)
    }

    /// Get statistics
    pub fn stats(&self) -> &BinderStats {
        &self.stats
    }

    /// Clear all representations
    pub fn clear(&mut self) {
        self.representations.clear();
        self.current_binding = None;
    }

    /// Update a modality with a new BinaryHV representation.
    ///
    /// Converts BinaryHV to ContinuousHV (bipolar) and adds/replaces the representation.
    pub fn update_modality(&mut self, modality: Modality, hv: BinaryHV) {
        let bipolar = hv.to_bipolar();
        let continuous = ContinuousHV::from_vec(bipolar);
        let repr = ModalRepresentation::new(modality, continuous, 1.0, "update_modality");
        self.add_representation(repr);
    }

    /// Set attention weight for all representations of a modality.
    pub fn set_attention(&mut self, modality: Modality, attention: f32) {
        if let Some(reps) = self.representations.get_mut(&modality) {
            for rep in reps.iter_mut() {
                rep.attention = attention.clamp(0.0, 1.0);
            }
        }
    }

    /// Set attention weight for ALL modalities (global precision gating).
    /// Science: Talsma (2015) — high free energy reduces binding precision
    pub fn set_attention_weight(&mut self, weight: f32) {
        let w = weight.clamp(0.0, 1.0);
        for reps in self.representations.values_mut() {
            for rep in reps.iter_mut() {
                rep.attention = w;
            }
        }
    }

    /// Compute a cross-modal integration metric (analogous to Φ).
    ///
    /// Measures how much information is integrated across modalities by computing
    /// average pairwise similarity of modal representations in the bound space.
    pub fn cross_modal_psi(&self) -> f64 {
        let modality_hvs: Vec<&ContinuousHV> = self
            .representations
            .values()
            .filter_map(|reps| reps.last().map(|r| &r.hv))
            .collect();

        if modality_hvs.len() < 2 {
            return 0.0;
        }

        // Average pairwise similarity as integration proxy
        let mut total_sim = 0.0;
        let mut count = 0;
        for i in 0..modality_hvs.len() {
            for j in (i + 1)..modality_hvs.len() {
                total_sim += modality_hvs[i].similarity(modality_hvs[j]).abs() as f64;
                count += 1;
            }
        }

        if count > 0 {
            total_sim / count as f64
        } else {
            0.0
        }
    }
}

impl Default for CrossModalBinder {
    fn default() -> Self {
        Self::new(CrossModalBindingConfig::default())
    }
}

// ==============================================================================
// NSM PRIMITIVE GROUNDING (Wierzbicka's Natural Semantic Metalanguage)
// ==============================================================================

/// NSM grounding for sensory modalities
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct ModalityPrimitiveGrounding {
    /// The modality being grounded
    pub modality: Modality,
    /// NSM primitive composition
    pub nsm_primitives: Vec<String>,
    /// HDC encoding via primitive binding
    pub primitive_encoding: BinaryHV,
    /// Whether this is a sensory input modality
    pub is_sensory: bool,
    /// Whether this involves body awareness
    pub is_embodied: bool,
}

#[allow(dead_code)]
impl ModalityPrimitiveGrounding {
    pub(crate) fn new(modality: Modality, system: &PrimitiveSystem) -> Self {
        let (primitives, sensory, embodied) = Self::nsm_mapping(modality);
        let encoding = encode_primitives(&primitives, system);

        Self {
            modality,
            nsm_primitives: primitives,
            primitive_encoding: encoding,
            is_sensory: sensory,
            is_embodied: embodied,
        }
    }

    fn nsm_mapping(modality: Modality) -> (Vec<String>, bool, bool) {
        match modality {
            // Visual: seeing things
            Modality::Visual => (
                vec![
                    "SEE".into(),
                    "THING".into(),
                    "BECAUSE".into(),
                    "LOOK".into(),
                ],
                true,
                false,
            ),
            // Auditory: hearing sounds
            Modality::Auditory => (
                vec!["HEAR".into(), "SOMETHING".into(), "BECAUSE".into()],
                true,
                false,
            ),
            // Textual/Linguistic: language and words
            Modality::Textual | Modality::Linguistic => (
                vec!["SAY".into(), "WORDS".into(), "KNOW".into(), "THINK".into()],
                false,
                false,
            ),
            // Proprioceptive: body position sense
            Modality::Proprioceptive => (
                vec!["FEEL".into(), "BODY".into(), "WHERE".into(), "MOVE".into()],
                true,
                true,
            ),
            // Somatosensory: touch and physical sensation
            Modality::Somatosensory => (
                vec![
                    "FEEL".into(),
                    "TOUCH".into(),
                    "BODY".into(),
                    "SOMETHING".into(),
                ],
                true,
                true,
            ),
            // Motor: movement and action
            Modality::Motor => (
                vec!["DO".into(), "MOVE".into(), "BODY".into(), "WANT".into()],
                false,
                true,
            ),
            // Temporal: time and sequence
            Modality::Temporal => (
                vec!["TIME".into(), "BEFORE".into(), "AFTER".into(), "NOW".into()],
                false,
                false,
            ),
            // Spatial: space and location
            Modality::Spatial => (
                vec!["WHERE".into(), "PLACE".into(), "NEAR".into(), "FAR".into()],
                false,
                false,
            ),
            // Affective/Emotional: feelings and emotions
            Modality::Affective | Modality::Emotional => (
                vec!["FEEL".into(), "GOOD".into(), "BAD".into(), "BECAUSE".into()],
                false,
                true,
            ),
            // Interoceptive: internal body states
            Modality::Interoceptive => (
                vec![
                    "FEEL".into(),
                    "INSIDE".into(),
                    "BODY".into(),
                    "SOMETHING".into(),
                ],
                true,
                true,
            ),
            // Abstract: conceptual thinking
            Modality::Abstract => (
                vec![
                    "THINK".into(),
                    "SOMETHING".into(),
                    "NOT".into(),
                    "SEE".into(),
                ],
                false,
                false,
            ),
        }
    }
}

/// NSM grounding for convergence levels (integration hierarchy)
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct ConvergenceLevelPrimitiveGrounding {
    /// The convergence level being grounded
    pub level: ConvergenceLevel,
    /// NSM primitive composition
    pub nsm_primitives: Vec<String>,
    /// HDC encoding via primitive binding
    pub primitive_encoding: BinaryHV,
    /// Integration breadth (how many modalities)
    pub integration_breadth: u8,
}

#[allow(dead_code)]
impl ConvergenceLevelPrimitiveGrounding {
    pub(crate) fn new(level: ConvergenceLevel, system: &PrimitiveSystem) -> Self {
        let (primitives, breadth) = Self::nsm_mapping(level);
        let encoding = encode_primitives(&primitives, system);

        Self {
            level,
            nsm_primitives: primitives,
            primitive_encoding: encoding,
            integration_breadth: breadth,
        }
    }

    fn nsm_mapping(level: ConvergenceLevel) -> (Vec<String>, u8) {
        match level {
            // Primary: single modality only
            ConvergenceLevel::Primary => (vec!["ONE".into(), "KIND".into(), "ONLY".into()], 1),
            // Secondary: two modalities combined
            ConvergenceLevel::Secondary => {
                (vec!["TWO".into(), "KIND".into(), "TOGETHER".into()], 2)
            }
            // Tertiary: three modalities combined
            ConvergenceLevel::Tertiary => (
                vec![
                    "SOME".into(),
                    "KIND".into(),
                    "TOGETHER".into(),
                    "MORE".into(),
                ],
                3,
            ),
            // Amodal: all modalities unified
            ConvergenceLevel::Amodal => (
                vec!["ALL".into(), "KIND".into(), "ONE".into(), "SAME".into()],
                255, // represents "all"
            ),
        }
    }
}

/// Unified NSM grounding system for cross-modal binding
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct CrossModalNSMGrounding {
    /// Modality groundings
    pub modalities: HashMap<Modality, ModalityPrimitiveGrounding>,
    /// Convergence level groundings
    pub convergence_levels: HashMap<ConvergenceLevel, ConvergenceLevelPrimitiveGrounding>,
}

#[allow(dead_code)]
impl CrossModalNSMGrounding {
    /// Create complete grounding system from primitive system
    pub(crate) fn new(system: &PrimitiveSystem) -> Self {
        let mut modalities = HashMap::new();
        let mut convergence_levels = HashMap::new();

        // Ground all modalities
        for modality in &[
            Modality::Visual,
            Modality::Auditory,
            Modality::Textual,
            Modality::Linguistic,
            Modality::Proprioceptive,
            Modality::Somatosensory,
            Modality::Motor,
            Modality::Temporal,
            Modality::Spatial,
            Modality::Affective,
            Modality::Emotional,
            Modality::Interoceptive,
            Modality::Abstract,
        ] {
            modalities.insert(
                *modality,
                ModalityPrimitiveGrounding::new(*modality, system),
            );
        }

        // Ground all convergence levels
        for level in &[
            ConvergenceLevel::Primary,
            ConvergenceLevel::Secondary,
            ConvergenceLevel::Tertiary,
            ConvergenceLevel::Amodal,
        ] {
            convergence_levels.insert(
                *level,
                ConvergenceLevelPrimitiveGrounding::new(*level, system),
            );
        }

        Self {
            modalities,
            convergence_levels,
        }
    }

    /// Query modalities by semantic similarity
    pub(crate) fn query_modalities(
        &self,
        query: &BinaryHV,
        threshold: f32,
    ) -> Vec<(&Modality, f32)> {
        let mut results: Vec<_> = self
            .modalities
            .iter()
            .map(|(m, g)| (m, g.primitive_encoding.similarity(query)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Query convergence levels by semantic similarity
    pub(crate) fn query_levels(
        &self,
        query: &BinaryHV,
        threshold: f32,
    ) -> Vec<(&ConvergenceLevel, f32)> {
        let mut results: Vec<_> = self
            .convergence_levels
            .iter()
            .map(|(l, g)| (l, g.primitive_encoding.similarity(query)))
            .filter(|(_, sim)| *sim >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get sensory modalities
    pub(crate) fn sensory_modalities(&self) -> Vec<&Modality> {
        self.modalities
            .iter()
            .filter(|(_, g)| g.is_sensory)
            .map(|(m, _)| m)
            .collect()
    }

    /// Get embodied modalities (involving body awareness)
    pub(crate) fn embodied_modalities(&self) -> Vec<&Modality> {
        self.modalities
            .iter()
            .filter(|(_, g)| g.is_embodied)
            .map(|(m, _)| m)
            .collect()
    }
}

/// Encode NSM primitives into HDC vector via sequential binding
#[allow(dead_code)]
fn encode_primitives(primitives: &[String], system: &PrimitiveSystem) -> BinaryHV {
    let vectors: Vec<BinaryHV> = primitives
        .iter()
        .map(|name| {
            if let Some(p) = system.get(name) {
                p.encoding
            } else if let Some(p) = system.get(&name.to_lowercase()) {
                p.encoding
            } else {
                // Fallback: deterministic random for unknown primitives
                let seed = name
                    .bytes()
                    .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
                BinaryHV::random(seed)
            }
        })
        .collect();

    if vectors.is_empty() {
        return BinaryHV::random(0);
    }

    // Bind sequentially with position encoding for order preservation
    let mut result = vectors[0];
    for (i, v) in vectors.iter().enumerate().skip(1) {
        let position_hv = BinaryHV::random(i as u64 * 1000);
        let positioned = v.bind(&position_hv);
        result = result.bind(&positioned);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binder_creation() {
        let binder = CrossModalBinder::default();
        assert!(binder.current_binding().is_none());
    }

    #[test]
    fn test_modal_representation() {
        let hv = ContinuousHV::random(512, 42);
        let repr = ModalRepresentation::new(Modality::Visual, hv, 0.9, "camera");
        assert_eq!(repr.modality, Modality::Visual);
        assert_eq!(repr.confidence, 0.9);
    }

    #[test]
    fn test_binding() {
        let mut binder = CrossModalBinder::default();

        let visual = ModalRepresentation::new(
            Modality::Visual,
            ContinuousHV::random(512, 42),
            0.9,
            "camera",
        );
        let audio = ModalRepresentation::new(
            Modality::Auditory,
            ContinuousHV::random(512, 42),
            0.8,
            "microphone",
        );

        binder.add_representation(visual);
        binder.add_representation(audio);

        let result = binder.bind();
        assert!(result.is_some());

        let binding = result.unwrap();
        assert_eq!(binding.modalities.len(), 2);
        assert!(binding.strength >= 0.0 && binding.strength <= 1.0);
    }

    // ==========================================================================
    // Tests for new types added for multi_modal_integration support
    // ==========================================================================

    #[test]
    fn test_modality_all() {
        let all = Modality::all();
        assert!(!all.is_empty());
        assert!(all.contains(&Modality::Visual));
        assert!(all.contains(&Modality::Auditory));
        assert!(all.contains(&Modality::Linguistic));
    }

    #[test]
    fn test_modality_sensory() {
        let sensory = Modality::sensory();
        assert!(sensory.contains(&Modality::Visual));
        assert!(sensory.contains(&Modality::Auditory));
        // Motor should not be in sensory
        assert!(!sensory.contains(&Modality::Motor));
    }

    #[test]
    fn test_modality_channel_creation() {
        let channel = ModalityChannel::new(Modality::Visual);
        assert_eq!(channel.modality, Modality::Visual);
        assert_eq!(channel.attention, 0.5);
        assert!(channel.temporal_buffer.is_empty());
    }

    #[test]
    fn test_modality_channel_update() {
        let mut channel = ModalityChannel::new(Modality::Auditory);
        let features = BinaryHV::random(42);

        channel.update(features);

        assert_eq!(channel.temporal_buffer.len(), 1);
        assert_eq!(channel.features, features);
    }

    #[test]
    fn test_modality_channel_temporal_coherence() {
        let mut channel = ModalityChannel::new(Modality::Visual);

        // Single item should have coherence 1.0
        channel.update(BinaryHV::random(1));
        assert_eq!(channel.temporal_coherence(), 1.0);

        // Add more items
        channel.update(BinaryHV::random(2));
        channel.update(BinaryHV::random(3));

        // Should have valid coherence
        let coherence = channel.temporal_coherence();
        assert!((0.0..=1.0).contains(&coherence));
    }

    #[test]
    fn test_convergence_level() {
        let primary = ConvergenceZone::new(0, vec![Modality::Visual]);
        assert_eq!(primary.level, ConvergenceLevel::Primary);

        let secondary = ConvergenceZone::new(1, vec![Modality::Visual, Modality::Auditory]);
        assert_eq!(secondary.level, ConvergenceLevel::Secondary);

        let tertiary = ConvergenceZone::new(
            2,
            vec![Modality::Visual, Modality::Auditory, Modality::Linguistic],
        );
        assert_eq!(tertiary.level, ConvergenceLevel::Tertiary);

        let amodal = ConvergenceZone::new(3, Modality::sensory());
        assert_eq!(amodal.level, ConvergenceLevel::Amodal);
    }

    #[test]
    fn test_convergence_zone_integrate() {
        let mut zone = ConvergenceZone::new(0, vec![Modality::Visual, Modality::Auditory]);

        let mut inputs = HashMap::new();
        inputs.insert(Modality::Visual, BinaryHV::random(1));
        inputs.insert(Modality::Auditory, BinaryHV::random(2));

        zone.integrate(&inputs);

        assert!(zone.activation > 0.0);
        assert!(zone.binding_strength >= 0.0 && zone.binding_strength <= 1.0);
    }

    #[test]
    fn test_convergence_zone_partial_input() {
        let mut zone = ConvergenceZone::new(0, vec![Modality::Visual, Modality::Auditory]);

        let mut inputs = HashMap::new();
        inputs.insert(Modality::Visual, BinaryHV::random(1));
        // Only one modality provided

        zone.integrate(&inputs);

        // Activation should be 0.5 (1 of 2 modalities)
        assert_eq!(zone.activation, 0.5);
    }

    #[test]
    fn test_episodic_buffer_creation() {
        let buffer = EpisodicBuffer::new(7);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_episodic_buffer_add_chunk() {
        let mut buffer = EpisodicBuffer::new(3);

        buffer.add_chunk(BinaryHV::random(1));
        assert_eq!(buffer.len(), 1);

        buffer.add_chunk(BinaryHV::random(2));
        buffer.add_chunk(BinaryHV::random(3));
        assert_eq!(buffer.len(), 3);

        // Should evict oldest when over capacity
        buffer.add_chunk(BinaryHV::random(4));
        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn test_episodic_buffer_bundled() {
        let mut buffer = EpisodicBuffer::new(5);

        buffer.add_chunk(BinaryHV::random(1));
        buffer.add_chunk(BinaryHV::random(2));
        buffer.add_chunk(BinaryHV::random(3));

        let bundled = buffer.bundled();
        // Bundled should be similar to all inputs
        for chunk in buffer.chunks() {
            let sim = bundled.similarity(chunk);
            assert!(sim > 0.4, "Bundled should be similar to components");
        }
    }

    #[test]
    fn test_episodic_buffer_clear() {
        let mut buffer = EpisodicBuffer::new(5);
        buffer.add_chunk(BinaryHV::random(1));
        buffer.add_chunk(BinaryHV::random(2));

        buffer.clear();

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }
}
