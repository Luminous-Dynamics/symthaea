// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cross-Modal Binding: Multi-Sensory Integration
//!
//! Provides modality channels, convergence zones, short-term episodic bundling,
//! and a ContinuousHV cross-modal binder.
//!
//! The binder treats confidence and attention as evidence weights. A weight is
//! not merely reported as metadata: it changes the actual weighted bundle. Any
//! mutation that can change the bundle invalidates the cached `current_binding`
//! so queries never silently operate on stale evidence.

use super::modality_identity::{modality_seed, modality_sort_key};
use crate::hdc::primitive_system::PrimitiveSystem;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::binary_hv::BinaryHV;

/// Types of sensory modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Visual,
    Auditory,
    Textual,
    Linguistic,
    Proprioceptive,
    Somatosensory,
    Motor,
    Temporal,
    Spatial,
    Affective,
    Emotional,
    Interoceptive,
    Abstract,
}

impl Modality {
    /// Modalities currently initialized by the root multi-modal integrator.
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

    /// Sensory modalities used by the current amodal convergence zone.
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
// MODALITY CHANNEL
// =============================================================================

#[derive(Debug, Clone)]
pub struct ModalityChannel {
    pub modality: Modality,
    pub features: BinaryHV,
    pub attention: f64,
    pub temporal_buffer: VecDeque<BinaryHV>,
    buffer_capacity: usize,
}

impl ModalityChannel {
    pub fn new(modality: Modality) -> Self {
        Self {
            modality,
            features: BinaryHV::zero(),
            attention: 0.5,
            temporal_buffer: VecDeque::with_capacity(8),
            buffer_capacity: 8,
        }
    }

    pub fn update(&mut self, features: BinaryHV) {
        if self.temporal_buffer.len() >= self.buffer_capacity {
            self.temporal_buffer.pop_front();
        }
        self.temporal_buffer.push_back(features);
        self.features = features;
    }

    /// Representation after attention-dependent noise. Non-finite attention is
    /// treated pessimistically as zero precision.
    pub fn attended(&self) -> BinaryHV {
        let attention = if self.attention.is_finite() {
            self.attention.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if attention >= 0.99 {
            return self.features;
        }
        let noise_level = (1.0 - attention) as f32 * 0.3;
        self.features.add_noise(noise_level, modality_seed(self.modality))
    }

    pub fn temporal_coherence(&self) -> f64 {
        if self.temporal_buffer.len() < 2 {
            return 1.0;
        }

        let mut total_sim = 0.0;
        let mut count = 0usize;
        for index in 0..self.temporal_buffer.len() - 1 {
            total_sim += self.temporal_buffer[index]
                .similarity(&self.temporal_buffer[index + 1]) as f64;
            count += 1;
        }

        if count == 0 {
            1.0
        } else {
            total_sim / count as f64
        }
    }
}

// =============================================================================
// CONVERGENCE ZONE
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConvergenceLevel {
    Primary,
    Secondary,
    Tertiary,
    Amodal,
}

#[derive(Debug, Clone)]
pub struct ConvergenceZone {
    pub id: usize,
    pub source_modalities: Vec<Modality>,
    pub integrated: BinaryHV,
    pub binding_strength: f64,
    pub activation: f64,
    pub level: ConvergenceLevel,
}

impl ConvergenceZone {
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

    /// Integrate only the modalities present in `inputs`.
    ///
    /// Empty input explicitly clears the zone's derived state. This prevents a
    /// prior-cycle vector from surviving after its evidence disappears.
    pub fn integrate(&mut self, inputs: &HashMap<Modality, BinaryHV>) {
        let vectors: Vec<BinaryHV> = self
            .source_modalities
            .iter()
            .filter_map(|modality| inputs.get(modality).copied())
            .collect();

        if vectors.is_empty() {
            self.integrated = BinaryHV::zero();
            self.binding_strength = 0.0;
            self.activation = 0.0;
            return;
        }

        self.integrated = BinaryHV::bundle(&vectors);
        self.binding_strength = Self::compute_binding_strength(&vectors);
        self.activation = vectors.len() as f64 / self.source_modalities.len() as f64;
    }

    fn compute_binding_strength(vectors: &[BinaryHV]) -> f64 {
        if vectors.len() < 2 {
            return 1.0;
        }

        let mut total_sim = 0.0;
        let mut count = 0usize;
        for i in 0..vectors.len() {
            for j in (i + 1)..vectors.len() {
                total_sim += vectors[i].similarity(&vectors[j]) as f64;
                count += 1;
            }
        }

        if count == 0 {
            1.0
        } else {
            (total_sim / count as f64).clamp(0.0, 1.0)
        }
    }
}

// =============================================================================
// EPISODIC BUFFER
// =============================================================================

#[derive(Debug, Clone)]
pub struct EpisodicBuffer {
    capacity: usize,
    chunks: VecDeque<BinaryHV>,
}

impl EpisodicBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            chunks: VecDeque::with_capacity(capacity),
        }
    }

    pub fn add_chunk(&mut self, chunk: BinaryHV) {
        if self.capacity == 0 {
            return;
        }
        if self.chunks.len() >= self.capacity {
            self.chunks.pop_front();
        }
        self.chunks.push_back(chunk);
    }

    pub fn chunks(&self) -> &VecDeque<BinaryHV> {
        &self.chunks
    }

    pub fn most_recent(&self) -> Option<&BinaryHV> {
        self.chunks.back()
    }

    pub fn bundled(&self) -> BinaryHV {
        if self.chunks.is_empty() {
            return BinaryHV::zero();
        }
        let vectors: Vec<BinaryHV> = self.chunks.iter().copied().collect();
        BinaryHV::bundle(&vectors)
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
    }
}

// =============================================================================
// CONTINUOUS CROSS-MODAL BINDER
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalBindingConfig {
    pub dimension: usize,
    pub binding_threshold: f32,
    pub temporal_decay: f32,
    pub max_bindings: usize,
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

#[derive(Debug, Clone)]
pub struct ModalRepresentation {
    pub modality: Modality,
    pub hv: ContinuousHV,
    pub confidence: f32,
    pub timestamp: u64,
    pub source: String,
    pub attention: f32,
}

impl ModalRepresentation {
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

    pub fn with_attention(mut self, attention: f32) -> Self {
        self.attention = if attention.is_finite() {
            attention.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self
    }

    fn effective_weight(&self, use_attention: bool) -> f32 {
        let confidence = if self.confidence.is_finite() {
            self.confidence.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let attention = if self.attention.is_finite() {
            self.attention.clamp(0.0, 1.0)
        } else {
            0.0
        };

        if use_attention {
            confidence * attention
        } else {
            confidence
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingResult {
    pub bound_hv: Vec<f32>,
    pub modalities: Vec<Modality>,
    pub strength: f32,
    pub coherence: f32,
    /// Normalized relative contribution of each included modality. Values sum
    /// to approximately 1.0 for a non-empty binding.
    pub contributions: HashMap<String, f32>,
}

#[derive(Debug)]
pub struct CrossModalBinder {
    config: CrossModalBindingConfig,
    representations: HashMap<Modality, Vec<ModalRepresentation>>,
    current_binding: Option<ContinuousHV>,
    binding_history: Vec<BindingResult>,
    stats: BinderStats,
}

#[derive(Debug, Clone, Default)]
pub struct BinderStats {
    pub total_bindings: u64,
    pub avg_strength: f32,
    pub avg_coherence: f32,
    pub modality_counts: HashMap<Modality, u64>,
}

impl CrossModalBinder {
    pub fn new(config: CrossModalBindingConfig) -> Self {
        Self {
            config,
            representations: HashMap::new(),
            current_binding: None,
            binding_history: Vec::new(),
            stats: BinderStats::default(),
        }
    }

    /// Add evidence and invalidate any binding derived from the prior evidence
    /// set. The new binding is computed lazily by [`bind`](Self::bind).
    pub fn add_representation(&mut self, repr: ModalRepresentation) {
        let modality = repr.modality;
        let reps = self.representations.entry(modality).or_default();

        if self.config.max_bindings == 0 {
            reps.clear();
        } else {
            if reps.len() >= self.config.max_bindings {
                reps.remove(0);
            }
            reps.push(repr);
        }

        *self.stats.modality_counts.entry(modality).or_insert(0) += 1;
        self.current_binding = None;
    }

    /// Perform confidence/attention-weighted cross-modal bundling.
    ///
    /// Modalities with zero or non-finite effective weight are present in the
    /// history but do not enter the binding. If no trustworthy modality remains,
    /// the result and cached binding are both `None`.
    pub fn bind(&mut self) -> Option<BindingResult> {
        let mut entries: Vec<(Modality, &ModalRepresentation, f32)> = self
            .representations
            .iter()
            .filter_map(|(modality, reps)| {
                let rep = reps.last()?;
                let weight = rep.effective_weight(self.config.use_attention);
                (weight > 0.0).then_some((*modality, rep, weight))
            })
            .collect();

        entries.sort_by_key(|(modality, _, _)| modality_sort_key(*modality));
        if entries.is_empty() {
            self.current_binding = None;
            return None;
        }

        let hvs: Vec<&ContinuousHV> = entries.iter().map(|(_, rep, _)| &rep.hv).collect();
        let weights: Vec<f32> = entries.iter().map(|(_, _, weight)| *weight).collect();
        let total_weight: f32 = weights.iter().sum();
        if !total_weight.is_finite() || total_weight <= f32::EPSILON {
            self.current_binding = None;
            return None;
        }

        let bound_hv = if hvs.len() == 1 {
            hvs[0].clone()
        } else {
            ContinuousHV::weighted_bundle(&hvs, &weights)
        };

        let strength = Self::calculate_binding_strength(&entries);
        let coherence = Self::calculate_coherence(&entries, strength);

        let modalities: Vec<Modality> = entries.iter().map(|(modality, _, _)| *modality).collect();
        let contributions: HashMap<String, f32> = entries
            .iter()
            .map(|(modality, _, weight)| {
                (format!("{modality:?}"), *weight / total_weight)
            })
            .collect();

        self.current_binding = Some(bound_hv.clone());
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

    fn calculate_binding_strength(entries: &[(Modality, &ModalRepresentation, f32)]) -> f32 {
        if entries.len() < 2 {
            return 1.0;
        }

        let mut weighted_similarity = 0.0f32;
        let mut pair_weight_sum = 0.0f32;
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let pair_weight = entries[i].2 * entries[j].2;
                let similarity = entries[i]
                    .1
                    .hv
                    .similarity(&entries[j].1.hv)
                    .clamp(0.0, 1.0);
                weighted_similarity += similarity * pair_weight;
                pair_weight_sum += pair_weight;
            }
        }

        if pair_weight_sum <= f32::EPSILON {
            0.0
        } else {
            (weighted_similarity / pair_weight_sum).clamp(0.0, 1.0)
        }
    }

    fn calculate_coherence(
        entries: &[(Modality, &ModalRepresentation, f32)],
        strength: f32,
    ) -> f32 {
        if entries.is_empty() {
            return 0.0;
        }
        if entries.len() == 1 {
            return 1.0;
        }

        let avg_weight = entries.iter().map(|(_, _, w)| *w).sum::<f32>() / entries.len() as f32;
        let variance = entries
            .iter()
            .map(|(_, _, weight)| (*weight - avg_weight).powi(2))
            .sum::<f32>()
            / entries.len() as f32;
        let balance = 1.0 / (1.0 + variance.sqrt());
        (strength * balance).clamp(0.0, 1.0)
    }

    pub fn query(&self, probe: &ContinuousHV) -> Option<f32> {
        self.current_binding
            .as_ref()
            .map(|binding| binding.similarity(probe))
    }

    /// Approximate legacy unbinding helper.
    ///
    /// The current state is a *bundle*, not a pure associative bind, so exact
    /// algebraic recovery of one modality is not defined here. This method is
    /// retained for API compatibility and should not be interpreted as exact
    /// inversion of the weighted superposition.
    pub fn unbind(&mut self, modality: Modality) -> Option<ContinuousHV> {
        let current = self.current_binding.as_ref()?;
        let modal_rep = self.representations.get(&modality)?.last()?;
        Some(current.bind(&modal_rep.hv))
    }

    /// Apply temporal attention decay and invalidate the cached binding because
    /// its weights have changed.
    pub fn decay(&mut self, dt: f32) {
        if !dt.is_finite() || dt < 0.0 {
            return;
        }
        let rate = if self.config.temporal_decay.is_finite() {
            self.config.temporal_decay.max(0.0)
        } else {
            0.0
        };
        let decay_factor = (1.0 - rate * dt).clamp(0.0, 1.0);

        for reps in self.representations.values_mut() {
            for rep in reps.iter_mut() {
                rep.attention *= decay_factor;
            }
            reps.retain(|rep| rep.attention.is_finite() && rep.attention > 0.01);
        }
        self.representations.retain(|_, reps| !reps.is_empty());
        self.current_binding = None;
    }

    pub fn current_binding(&self) -> Option<&ContinuousHV> {
        self.current_binding.as_ref()
    }

    pub fn get_representations(&self, modality: Modality) -> Option<&Vec<ModalRepresentation>> {
        self.representations.get(&modality)
    }

    pub fn stats(&self) -> &BinderStats {
        &self.stats
    }

    pub fn clear(&mut self) {
        self.representations.clear();
        self.current_binding = None;
    }

    pub fn update_modality(&mut self, modality: Modality, hv: BinaryHV) {
        let continuous = ContinuousHV::from_vec(hv.to_bipolar());
        self.add_representation(ModalRepresentation::new(
            modality,
            continuous,
            1.0,
            "update_modality",
        ));
    }

    pub fn set_attention(&mut self, modality: Modality, attention: f32) {
        let weight = if attention.is_finite() {
            attention.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if let Some(reps) = self.representations.get_mut(&modality) {
            for rep in reps.iter_mut() {
                rep.attention = weight;
            }
        }
        self.current_binding = None;
    }

    /// Set attention weight for all modalities (global precision gating).
    pub fn set_attention_weight(&mut self, weight: f32) {
        let weight = if weight.is_finite() {
            weight.clamp(0.0, 1.0)
        } else {
            0.0
        };
        for reps in self.representations.values_mut() {
            for rep in reps.iter_mut() {
                rep.attention = weight;
            }
        }
        self.current_binding = None;
    }

    /// Weighted cross-modal integration proxy based on pairwise absolute cosine
    /// similarity. Zero-confidence/attention modalities do not contribute.
    pub fn cross_modal_psi(&self) -> f64 {
        let entries: Vec<(&ContinuousHV, f32)> = self
            .representations
            .values()
            .filter_map(|reps| {
                let rep = reps.last()?;
                let weight = rep.effective_weight(self.config.use_attention);
                (weight > 0.0).then_some((&rep.hv, weight))
            })
            .collect();

        if entries.len() < 2 {
            return 0.0;
        }

        let mut total = 0.0f64;
        let mut total_weight = 0.0f64;
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let pair_weight = (entries[i].1 * entries[j].1) as f64;
                total += entries[i].0.similarity(entries[j].0).abs() as f64 * pair_weight;
                total_weight += pair_weight;
            }
        }

        if total_weight <= f64::EPSILON {
            0.0
        } else {
            (total / total_weight).clamp(0.0, 1.0)
        }
    }
}

impl Default for CrossModalBinder {
    fn default() -> Self {
        Self::new(CrossModalBindingConfig::default())
    }
}

// =============================================================================
// NSM PRIMITIVE GROUNDING
// =============================================================================

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct ModalityPrimitiveGrounding {
    pub modality: Modality,
    pub nsm_primitives: Vec<String>,
    pub primitive_encoding: BinaryHV,
    pub is_sensory: bool,
    pub is_embodied: bool,
}

#[allow(dead_code)]
impl ModalityPrimitiveGrounding {
    pub(crate) fn new(modality: Modality, system: &PrimitiveSystem) -> Self {
        let (primitives, sensory, embodied) = Self::nsm_mapping(modality);
        let primitive_encoding = encode_primitives(&primitives, system);
        Self {
            modality,
            nsm_primitives: primitives,
            primitive_encoding,
            is_sensory: sensory,
            is_embodied: embodied,
        }
    }

    fn nsm_mapping(modality: Modality) -> (Vec<String>, bool, bool) {
        match modality {
            Modality::Visual => (
                vec!["SEE".into(), "THING".into(), "BECAUSE".into(), "LOOK".into()],
                true,
                false,
            ),
            Modality::Auditory => (
                vec!["HEAR".into(), "SOMETHING".into(), "BECAUSE".into()],
                true,
                false,
            ),
            Modality::Textual | Modality::Linguistic => (
                vec!["SAY".into(), "WORDS".into(), "KNOW".into(), "THINK".into()],
                false,
                false,
            ),
            Modality::Proprioceptive => (
                vec!["FEEL".into(), "BODY".into(), "WHERE".into(), "MOVE".into()],
                true,
                true,
            ),
            Modality::Somatosensory => (
                vec!["FEEL".into(), "TOUCH".into(), "BODY".into(), "SOMETHING".into()],
                true,
                true,
            ),
            Modality::Motor => (
                vec!["DO".into(), "MOVE".into(), "BODY".into(), "WANT".into()],
                false,
                true,
            ),
            Modality::Temporal => (
                vec!["TIME".into(), "BEFORE".into(), "AFTER".into(), "NOW".into()],
                false,
                false,
            ),
            Modality::Spatial => (
                vec!["WHERE".into(), "PLACE".into(), "NEAR".into(), "FAR".into()],
                false,
                false,
            ),
            Modality::Affective | Modality::Emotional => (
                vec!["FEEL".into(), "GOOD".into(), "BAD".into(), "BECAUSE".into()],
                false,
                true,
            ),
            Modality::Interoceptive => (
                vec!["FEEL".into(), "INSIDE".into(), "BODY".into(), "SOMETHING".into()],
                true,
                true,
            ),
            Modality::Abstract => (
                vec!["THINK".into(), "SOMETHING".into(), "NOT".into(), "SEE".into()],
                false,
                false,
            ),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct ConvergenceLevelPrimitiveGrounding {
    pub level: ConvergenceLevel,
    pub nsm_primitives: Vec<String>,
    pub primitive_encoding: BinaryHV,
    pub integration_breadth: u8,
}

#[allow(dead_code)]
impl ConvergenceLevelPrimitiveGrounding {
    pub(crate) fn new(level: ConvergenceLevel, system: &PrimitiveSystem) -> Self {
        let (primitives, integration_breadth) = Self::nsm_mapping(level);
        let primitive_encoding = encode_primitives(&primitives, system);
        Self {
            level,
            nsm_primitives: primitives,
            primitive_encoding,
            integration_breadth,
        }
    }

    fn nsm_mapping(level: ConvergenceLevel) -> (Vec<String>, u8) {
        match level {
            ConvergenceLevel::Primary => (vec!["ONE".into(), "KIND".into(), "ONLY".into()], 1),
            ConvergenceLevel::Secondary => {
                (vec!["TWO".into(), "KIND".into(), "TOGETHER".into()], 2)
            }
            ConvergenceLevel::Tertiary => (
                vec!["SOME".into(), "KIND".into(), "TOGETHER".into(), "MORE".into()],
                3,
            ),
            ConvergenceLevel::Amodal => (
                vec!["ALL".into(), "KIND".into(), "ONE".into(), "SAME".into()],
                255,
            ),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct CrossModalNSMGrounding {
    pub modalities: HashMap<Modality, ModalityPrimitiveGrounding>,
    pub convergence_levels: HashMap<ConvergenceLevel, ConvergenceLevelPrimitiveGrounding>,
}

#[allow(dead_code)]
impl CrossModalNSMGrounding {
    pub(crate) fn new(system: &PrimitiveSystem) -> Self {
        let mut modalities = HashMap::new();
        let mut convergence_levels = HashMap::new();

        for modality in [
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
            modalities.insert(modality, ModalityPrimitiveGrounding::new(modality, system));
        }

        for level in [
            ConvergenceLevel::Primary,
            ConvergenceLevel::Secondary,
            ConvergenceLevel::Tertiary,
            ConvergenceLevel::Amodal,
        ] {
            convergence_levels.insert(
                level,
                ConvergenceLevelPrimitiveGrounding::new(level, system),
            );
        }

        Self {
            modalities,
            convergence_levels,
        }
    }

    pub(crate) fn query_modalities(
        &self,
        query: &BinaryHV,
        threshold: f32,
    ) -> Vec<(&Modality, f32)> {
        let mut results: Vec<_> = self
            .modalities
            .iter()
            .map(|(modality, grounding)| {
                (modality, grounding.primitive_encoding.similarity(query))
            })
            .filter(|(_, similarity)| *similarity >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub(crate) fn query_levels(
        &self,
        query: &BinaryHV,
        threshold: f32,
    ) -> Vec<(&ConvergenceLevel, f32)> {
        let mut results: Vec<_> = self
            .convergence_levels
            .iter()
            .map(|(level, grounding)| (level, grounding.primitive_encoding.similarity(query)))
            .filter(|(_, similarity)| *similarity >= threshold)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub(crate) fn sensory_modalities(&self) -> Vec<&Modality> {
        self.modalities
            .iter()
            .filter(|(_, grounding)| grounding.is_sensory)
            .map(|(modality, _)| modality)
            .collect()
    }

    pub(crate) fn embodied_modalities(&self) -> Vec<&Modality> {
        self.modalities
            .iter()
            .filter(|(_, grounding)| grounding.is_embodied)
            .map(|(modality, _)| modality)
            .collect()
    }
}

#[allow(dead_code)]
fn encode_primitives(primitives: &[String], system: &PrimitiveSystem) -> BinaryHV {
    let vectors: Vec<BinaryHV> = primitives
        .iter()
        .map(|name| {
            if let Some(primitive) = system.get(name) {
                primitive.encoding
            } else if let Some(primitive) = system.get(&name.to_lowercase()) {
                primitive.encoding
            } else {
                let seed = name
                    .bytes()
                    .fold(0u64, |acc, byte| acc.wrapping_mul(31).wrapping_add(byte as u64));
                BinaryHV::random(seed)
            }
        })
        .collect();

    if vectors.is_empty() {
        return BinaryHV::random(0);
    }

    let mut result = vectors[0];
    for (index, vector) in vectors.iter().enumerate().skip(1) {
        let position_hv = BinaryHV::random(index as u64 * 1000);
        result = result.bind(&vector.bind(&position_hv));
    }
    result
}

// =============================================================================
// TESTS
// =============================================================================

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
    fn confidence_weights_change_actual_bundle() {
        let visual_hv = ContinuousHV::random(512, 42);
        let audio_hv = ContinuousHV::random(512, 43);
        let mut binder = CrossModalBinder::default();
        binder.add_representation(ModalRepresentation::new(
            Modality::Visual,
            visual_hv.clone(),
            1.0,
            "camera",
        ));
        binder.add_representation(ModalRepresentation::new(
            Modality::Auditory,
            audio_hv.clone(),
            0.05,
            "microphone",
        ));

        let result = binder.bind().unwrap();
        let bound = ContinuousHV::from_vec(result.bound_hv.clone());
        assert!(bound.similarity(&visual_hv) > bound.similarity(&audio_hv));
        assert!(result.contributions["Visual"] > result.contributions["Auditory"]);
        let sum: f32 = result.contributions.values().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn zero_confidence_evidence_does_not_enter_binding() {
        let mut binder = CrossModalBinder::default();
        binder.add_representation(ModalRepresentation::new(
            Modality::Visual,
            ContinuousHV::random(512, 42),
            1.0,
            "camera",
        ));
        binder.add_representation(ModalRepresentation::new(
            Modality::Auditory,
            ContinuousHV::random(512, 43),
            0.0,
            "microphone",
        ));

        let result = binder.bind().unwrap();
        assert_eq!(result.modalities, vec![Modality::Visual]);
        assert!(!result.contributions.contains_key("Auditory"));
    }

    #[test]
    fn all_untrusted_evidence_produces_no_binding() {
        let mut binder = CrossModalBinder::default();
        binder.add_representation(ModalRepresentation::new(
            Modality::Visual,
            ContinuousHV::random(512, 42),
            f32::NAN,
            "camera",
        ));
        assert!(binder.bind().is_none());
        assert!(binder.current_binding().is_none());
    }

    #[test]
    fn new_evidence_invalidates_cached_binding() {
        let mut binder = CrossModalBinder::default();
        binder.add_representation(ModalRepresentation::new(
            Modality::Visual,
            ContinuousHV::random(512, 42),
            1.0,
            "camera",
        ));
        assert!(binder.bind().is_some());
        assert!(binder.current_binding().is_some());

        binder.add_representation(ModalRepresentation::new(
            Modality::Auditory,
            ContinuousHV::random(512, 43),
            1.0,
            "microphone",
        ));
        assert!(binder.current_binding().is_none());
    }

    #[test]
    fn attention_change_invalidates_cached_binding() {
        let mut binder = CrossModalBinder::default();
        binder.add_representation(ModalRepresentation::new(
            Modality::Visual,
            ContinuousHV::random(512, 42),
            1.0,
            "camera",
        ));
        binder.bind().unwrap();
        binder.set_attention(Modality::Visual, 0.5);
        assert!(binder.current_binding().is_none());
    }

    #[test]
    fn convergence_zone_clears_stale_state_when_evidence_disappears() {
        let mut zone = ConvergenceZone::new(0, vec![Modality::Visual, Modality::Auditory]);
        let mut inputs = HashMap::new();
        inputs.insert(Modality::Visual, BinaryHV::random(1));
        zone.integrate(&inputs);
        assert!(zone.activation > 0.0);

        zone.integrate(&HashMap::new());
        assert_eq!(zone.activation, 0.0);
        assert_eq!(zone.binding_strength, 0.0);
        assert_eq!(zone.integrated, BinaryHV::zero());
    }

    #[test]
    fn decay_invalidates_and_removes_expired_representations() {
        let mut config = CrossModalBindingConfig::default();
        config.temporal_decay = 1.0;
        let mut binder = CrossModalBinder::new(config);
        binder.add_representation(ModalRepresentation::new(
            Modality::Visual,
            ContinuousHV::random(512, 42),
            1.0,
            "camera",
        ));
        binder.bind().unwrap();
        binder.decay(1.0);
        assert!(binder.current_binding().is_none());
        assert!(binder.get_representations(Modality::Visual).is_none());
    }

    #[test]
    fn test_modality_all() {
        let all = Modality::all();
        assert!(all.contains(&Modality::Visual));
        assert!(all.contains(&Modality::Auditory));
        assert!(all.contains(&Modality::Linguistic));
    }

    #[test]
    fn test_modality_sensory() {
        let sensory = Modality::sensory();
        assert!(sensory.contains(&Modality::Visual));
        assert!(sensory.contains(&Modality::Auditory));
        assert!(!sensory.contains(&Modality::Motor));
    }

    #[test]
    fn test_modality_channel_update_and_coherence() {
        let mut channel = ModalityChannel::new(Modality::Auditory);
        let features = BinaryHV::random(42);
        channel.update(features);
        assert_eq!(channel.temporal_buffer.len(), 1);
        assert_eq!(channel.features, features);
        assert_eq!(channel.temporal_coherence(), 1.0);

        channel.update(BinaryHV::random(43));
        let coherence = channel.temporal_coherence();
        assert!((0.0..=1.0).contains(&coherence));
    }

    #[test]
    fn test_convergence_levels_and_partial_activation() {
        assert_eq!(
            ConvergenceZone::new(0, vec![Modality::Visual]).level,
            ConvergenceLevel::Primary
        );
        let mut secondary =
            ConvergenceZone::new(1, vec![Modality::Visual, Modality::Auditory]);
        assert_eq!(secondary.level, ConvergenceLevel::Secondary);
        let mut inputs = HashMap::new();
        inputs.insert(Modality::Visual, BinaryHV::random(1));
        secondary.integrate(&inputs);
        assert_eq!(secondary.activation, 0.5);
    }

    #[test]
    fn episodic_buffer_respects_capacity_and_clear() {
        let mut buffer = EpisodicBuffer::new(2);
        buffer.add_chunk(BinaryHV::random(1));
        buffer.add_chunk(BinaryHV::random(2));
        buffer.add_chunk(BinaryHV::random(3));
        assert_eq!(buffer.len(), 2);
        assert!(buffer.most_recent().is_some());
        assert_ne!(buffer.bundled(), BinaryHV::zero());
        buffer.clear();
        assert!(buffer.is_empty());
    }
}
