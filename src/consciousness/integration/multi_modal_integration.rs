// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Multi-Modal Consciousness Integration
//!
//! Integrates multi-modal perception with consciousness processing while
//! preserving a critical epistemic distinction: a modality that did not
//! provide evidence in the current cycle is absent, not a zero-valued percept.
//!
//! 1. **Modal Fusion**: Combines currently-present inputs from vision, audio,
//!    language, somatosensation, and other modalities.
//! 2. **Phi-Guided Binding**: Uses the existing Φ heuristic over active zones.
//! 3. **Primitive Routing**: Routes the current unified representation.
//! 4. **Consciousness Streaming**: Emits multi-modal awareness events.

use crate::consciousness::cross_modal_binding::{
    ConvergenceLevel, ConvergenceZone, EpisodicBuffer, Modality, ModalityChannel,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::primitive_system::PrimitiveSystem;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for multi-modal integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Enable Phi-guided binding optimization.
    pub phi_guided_binding: bool,
    /// Attention weight update rate.
    pub attention_learning_rate: f64,
    /// Minimum binding strength to consider coherent.
    pub coherence_threshold: f64,
    /// Enable consciousness event streaming.
    pub enable_streaming: bool,
    /// Maximum modal channels to process per cycle.
    pub max_channels_per_cycle: usize,
    /// Episodic buffer capacity.
    pub buffer_capacity: usize,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            phi_guided_binding: true,
            attention_learning_rate: 0.1,
            coherence_threshold: 0.3,
            enable_streaming: true,
            max_channels_per_cycle: 8,
            buffer_capacity: 7,
        }
    }
}

// =============================================================================
// MODAL INPUT
// =============================================================================

/// Input from a specific modality.
#[derive(Debug, Clone)]
pub struct ModalInput {
    /// The modality.
    pub modality: Modality,
    /// Feature encoding.
    pub features: BinaryHV,
    /// Confidence in this input (nominally 0.0-1.0).
    pub confidence: f64,
    /// Source timestamp.
    pub timestamp: Duration,
    /// Source identifier.
    pub source: String,
}

impl ModalInput {
    /// Create a new modal input.
    pub fn new(modality: Modality, features: BinaryHV, confidence: f64) -> Self {
        Self {
            modality,
            features,
            confidence,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO),
            source: String::new(),
        }
    }

    /// Create with source identifier.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }
}

// =============================================================================
// INTEGRATION RESULT
// =============================================================================

/// Result of multi-modal integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationResult {
    /// Unified representation across modalities that supplied trusted evidence
    /// in this cycle. Zero when no modality supplied trusted evidence.
    pub unified_representation: BinaryHV,
    /// Integrated Phi heuristic over currently active convergence zones.
    pub integrated_phi: f64,
    /// Binding coherence across currently trusted modalities.
    pub binding_coherence: f64,
    /// Effective current-cycle weights for modalities that were present.
    ///
    /// A present but zero-confidence modality appears with weight 0.0; an absent
    /// modality has no entry. This intentionally preserves the distinction
    /// between "observed but untrusted" and "not observed".
    pub attention_weights: HashMap<String, f64>,
    /// Dominant trusted modality in the current cycle.
    pub dominant_modality: Option<String>,
    /// Active convergence zones.
    pub active_zones: Vec<ConvergenceZoneInfo>,
    /// Processing time.
    pub processing_time_us: u64,
}

/// Info about an active convergence zone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceZoneInfo {
    /// Zone level.
    pub level: String,
    /// Source modalities configured for this zone.
    pub sources: Vec<String>,
    /// Binding strength.
    pub binding_strength: f64,
    /// Fraction of configured modalities represented by trusted current-cycle
    /// evidence.
    pub activation: f64,
}

// =============================================================================
// INTEGRATION EVENT
// =============================================================================

/// Event emitted during integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationEvent {
    /// Event type.
    pub event_type: IntegrationEventType,
    /// Timestamp.
    pub timestamp: u64,
    /// Associated result, when applicable.
    pub result: Option<IntegrationResult>,
}

/// Types of integration events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrationEventType {
    /// New modal input received.
    ModalInput,
    /// Binding updated.
    BindingUpdated,
    /// Coherent state achieved.
    CoherentState,
    /// Attention shifted.
    AttentionShift,
    /// Integration cycle completed.
    CycleCompleted,
}

// =============================================================================
// MULTI-MODAL INTEGRATOR
// =============================================================================

/// Integrates multi-modal perception with consciousness processing.
///
/// `channels` retain temporal history across cycles, while `active_modalities`
/// and `current_confidence` describe evidence from the current cycle only. This
/// prevents historical channel state from masquerading as fresh sensory input.
pub struct MultiModalIntegrator {
    config: IntegrationConfig,
    channels: HashMap<Modality, ModalityChannel>,
    convergence_zones: Vec<ConvergenceZone>,
    episodic_buffer: EpisodicBuffer,
    unified: BinaryHV,
    current_phi: f64,
    event_sender: Option<Sender<IntegrationEvent>>,
    stats: IntegrationStats,
    last_integration: Instant,
    active_modalities: HashSet<Modality>,
    current_confidence: HashMap<Modality, f64>,
}

/// Statistics for integration.
#[derive(Debug, Clone, Default)]
pub struct IntegrationStats {
    pub total_cycles: u64,
    pub total_inputs: u64,
    pub avg_phi: f64,
    pub peak_phi: f64,
    pub avg_coherence: f64,
    pub events_emitted: u64,
}

impl MultiModalIntegrator {
    /// Create a new multi-modal integrator.
    pub fn new(config: IntegrationConfig) -> Self {
        let mut channels = HashMap::new();
        for modality in Modality::all() {
            channels.insert(modality, ModalityChannel::new(modality));
        }

        Self {
            config: config.clone(),
            channels,
            convergence_zones: Self::create_convergence_hierarchy(),
            episodic_buffer: EpisodicBuffer::new(config.buffer_capacity),
            unified: BinaryHV::zero(),
            current_phi: 0.0,
            event_sender: None,
            stats: IntegrationStats::default(),
            last_integration: Instant::now(),
            active_modalities: HashSet::new(),
            current_confidence: HashMap::new(),
        }
    }

    fn create_convergence_hierarchy() -> Vec<ConvergenceZone> {
        let mut zones = Vec::new();
        let mut id = 0;

        for modality in Modality::all() {
            zones.push(ConvergenceZone::new(id, vec![modality]));
            id += 1;
        }

        let bimodal_pairs = [
            (Modality::Visual, Modality::Linguistic),
            (Modality::Visual, Modality::Auditory),
            (Modality::Auditory, Modality::Linguistic),
            (Modality::Somatosensory, Modality::Motor),
            (Modality::Emotional, Modality::Interoceptive),
        ];
        for (m1, m2) in bimodal_pairs {
            zones.push(ConvergenceZone::new(id, vec![m1, m2]));
            id += 1;
        }

        zones.push(ConvergenceZone::new(
            id,
            vec![Modality::Visual, Modality::Auditory, Modality::Linguistic],
        ));
        id += 1;

        zones.push(ConvergenceZone::new(id, Modality::sensory()));
        zones
    }

    /// Enable event streaming.
    pub fn enable_streaming(&mut self) -> Receiver<IntegrationEvent> {
        let (sender, receiver) = channel();
        self.event_sender = Some(sender);
        receiver
    }

    /// Modalities for which an observation was supplied in the current cycle,
    /// regardless of whether that observation was trusted enough to fuse.
    pub fn active_modalities(&self) -> &HashSet<Modality> {
        &self.active_modalities
    }

    /// Sanitized current-cycle confidence for one present modality.
    pub fn current_confidence(&self, modality: Modality) -> Option<f64> {
        self.current_confidence.get(&modality).copied()
    }

    fn sanitized_confidence(confidence: f64) -> f64 {
        if confidence.is_finite() {
            confidence.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Process multiple modal inputs.
    ///
    /// Current-cycle presence is rebuilt on every call. Channel history is kept
    /// for temporal coherence, but an old feature never participates in a later
    /// cycle unless that modality supplies a fresh input again.
    pub fn integrate(&mut self, inputs: &[ModalInput]) -> IntegrationResult {
        let start = Instant::now();
        self.stats.total_cycles += 1;
        self.active_modalities.clear();
        self.current_confidence.clear();

        for input in inputs.iter().take(self.config.max_channels_per_cycle) {
            let confidence = Self::sanitized_confidence(input.confidence);
            self.active_modalities.insert(input.modality);
            // Last observation for a modality wins, matching the feature update.
            self.current_confidence.insert(input.modality, confidence);

            if let Some(channel) = self.channels.get_mut(&input.modality) {
                channel.update(input.features);
                channel.attention = (channel.attention
                    + confidence * self.config.attention_learning_rate)
                    .clamp(0.0, 1.0);
            }
            self.stats.total_inputs += 1;
        }

        self.emit_event(IntegrationEventType::ModalInput, None);

        // Only current-cycle, non-zero-confidence evidence can enter fusion.
        // Confidence modulates the actual BinaryHV through the same noise-based
        // attention mechanism used by ModalityChannel rather than merely being
        // reported as metadata.
        let channel_inputs: HashMap<Modality, BinaryHV> = self
            .channels
            .iter()
            .filter_map(|(modality, channel)| {
                let confidence = *self.current_confidence.get(modality)?;
                if confidence <= 0.0 {
                    return None;
                }
                let effective_attention = (channel.attention * confidence).clamp(0.0, 1.0);
                let attended = if effective_attention >= 0.99 {
                    channel.features
                } else {
                    let noise_level = (1.0 - effective_attention) as f32 * 0.3;
                    channel.features.add_noise(noise_level, *modality as u64)
                };
                Some((*modality, attended))
            })
            .collect();

        for zone in &mut self.convergence_zones {
            zone.integrate(&channel_inputs);
        }

        self.emit_event(IntegrationEventType::BindingUpdated, None);

        // Inactive high-level zones are excluded. With no trusted evidence, the
        // unified representation is explicitly zero rather than a bundle of
        // historical or placeholder zone vectors.
        let high_level_zones: Vec<_> = self
            .convergence_zones
            .iter()
            .filter(|zone| {
                zone.activation > 0.0
                    && matches!(
                        zone.level,
                        ConvergenceLevel::Amodal | ConvergenceLevel::Tertiary
                    )
            })
            .collect();

        self.unified = if high_level_zones.is_empty() {
            BinaryHV::zero()
        } else {
            let vectors: Vec<BinaryHV> = high_level_zones
                .iter()
                .map(|zone| zone.integrated)
                .collect();
            BinaryHV::bundle(&vectors)
        };

        if !channel_inputs.is_empty() {
            self.episodic_buffer.add_chunk(self.unified);
        }

        let phi = self.compute_integrated_phi();
        self.current_phi = phi;

        let prev_sum = self.stats.avg_phi * (self.stats.total_cycles - 1) as f64;
        self.stats.avg_phi = (prev_sum + phi) / self.stats.total_cycles as f64;
        if phi > self.stats.peak_phi {
            self.stats.peak_phi = phi;
        }

        let coherence = self.compute_binding_coherence();
        let prev_coh_sum = self.stats.avg_coherence * (self.stats.total_cycles - 1) as f64;
        self.stats.avg_coherence =
            (prev_coh_sum + coherence) / self.stats.total_cycles as f64;

        if coherence >= self.config.coherence_threshold && !channel_inputs.is_empty() {
            self.emit_event(IntegrationEventType::CoherentState, None);
        }

        let mut attention_weights = HashMap::new();
        let mut dominant_modality = None;
        let mut max_weight = 0.0;

        for modality in &self.active_modalities {
            let confidence = self.current_confidence.get(modality).copied().unwrap_or(0.0);
            if let Some(channel) = self.channels.get(modality) {
                let effective_weight = (channel.attention * confidence).clamp(0.0, 1.0);
                let key = format!("{modality:?}");
                attention_weights.insert(key.clone(), effective_weight);
                if effective_weight > max_weight {
                    max_weight = effective_weight;
                    dominant_modality = Some(key);
                }
            }
        }

        let active_zones: Vec<_> = self
            .convergence_zones
            .iter()
            .filter(|zone| zone.activation > 0.0)
            .map(|zone| ConvergenceZoneInfo {
                level: format!("{:?}", zone.level),
                sources: zone
                    .source_modalities
                    .iter()
                    .map(|modality| format!("{modality:?}"))
                    .collect(),
                binding_strength: zone.binding_strength,
                activation: zone.activation,
            })
            .collect();

        let processing_time_us = start.elapsed().as_micros() as u64;
        self.last_integration = Instant::now();

        let result = IntegrationResult {
            unified_representation: self.unified,
            integrated_phi: phi,
            binding_coherence: coherence,
            attention_weights,
            dominant_modality,
            active_zones,
            processing_time_us,
        };

        self.emit_event(IntegrationEventType::CycleCompleted, Some(result.clone()));
        result
    }

    /// Compute the existing integrated-Φ heuristic over active zones only.
    ///
    /// An absent modality therefore cannot lower or inflate the score merely by
    /// existing in the enum or convergence hierarchy.
    fn compute_integrated_phi(&self) -> f64 {
        let mut phi_sum = 0.0;
        let mut weight_sum = 0.0;

        for zone in &self.convergence_zones {
            if zone.activation <= 0.0 {
                continue;
            }
            let weight = match zone.level {
                ConvergenceLevel::Primary => 0.5,
                ConvergenceLevel::Secondary => 1.0,
                ConvergenceLevel::Tertiary => 2.0,
                ConvergenceLevel::Amodal => 3.0,
            };
            phi_sum += zone.binding_strength * zone.activation * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            (phi_sum / weight_sum).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Compute temporal coherence only for trusted modalities observed in the
    /// current cycle. Historical channel buffers may inform those modalities,
    /// but absent channels do not participate.
    fn compute_binding_coherence(&self) -> f64 {
        let active_channels: Vec<_> = self
            .channels
            .iter()
            .filter_map(|(modality, channel)| {
                let confidence = self.current_confidence.get(modality).copied()?;
                (confidence > 0.0).then_some(channel)
            })
            .collect();

        if active_channels.is_empty() {
            return 0.0;
        }

        active_channels
            .iter()
            .map(|channel| channel.temporal_coherence())
            .sum::<f64>()
            / active_channels.len() as f64
    }

    /// Route unified representation to primitives.
    pub fn route_to_primitives(&self, system: &PrimitiveSystem) -> Vec<(String, f64)> {
        let mut matches = Vec::new();

        for primitive in system.all_primitives() {
            let similarity = self.unified.similarity(&primitive.encoding) as f64;
            if similarity > 0.3 {
                matches.push((primitive.name.clone(), similarity));
            }
        }

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(10);
        matches
    }

    pub fn unified_representation(&self) -> &BinaryHV {
        &self.unified
    }

    pub fn current_phi(&self) -> f64 {
        self.current_phi
    }

    pub fn stats(&self) -> &IntegrationStats {
        &self.stats
    }

    /// Get current-cycle modal channel feature HVs as ContinuousHV for IIT4
    /// analysis. Absent and zero-confidence modalities are excluded.
    #[cfg(feature = "iit4")]
    pub fn component_hvs_for_iit4(&self) -> Vec<symthaea_core::hdc::ContinuousHV> {
        self.channels
            .iter()
            .filter_map(|(modality, channel)| {
                let confidence = self.current_confidence.get(modality).copied()?;
                if confidence <= 0.0 {
                    return None;
                }

                let mut values = Vec::with_capacity(16384);
                for &byte in channel.features.0.iter() {
                    for bit in 0..8 {
                        values.push(if (byte >> bit) & 1 == 1 { 1.0 } else { -1.0 });
                    }
                }
                Some(symthaea_core::hdc::ContinuousHV::from_vec(values))
            })
            .collect()
    }

    fn emit_event(&mut self, event_type: IntegrationEventType, result: Option<IntegrationResult>) {
        if let Some(ref sender) = self.event_sender {
            let event = IntegrationEvent {
                event_type,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|duration| duration.as_millis() as u64)
                    .unwrap_or(0),
                result,
            };
            if sender.send(event).is_ok() {
                self.stats.events_emitted += 1;
            }
        }
    }

    /// Reset all temporal and current-cycle state.
    pub fn reset(&mut self) {
        for channel in self.channels.values_mut() {
            channel.features = BinaryHV::zero();
            channel.attention = 0.5;
            channel.temporal_buffer.clear();
        }
        self.active_modalities.clear();
        self.current_confidence.clear();
        self.episodic_buffer.clear();
        self.unified = BinaryHV::zero();
        self.current_phi = 0.0;
    }
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

pub fn visual_input(features: BinaryHV, confidence: f64) -> ModalInput {
    ModalInput::new(Modality::Visual, features, confidence).with_source("visual_perception")
}

pub fn auditory_input(features: BinaryHV, confidence: f64) -> ModalInput {
    ModalInput::new(Modality::Auditory, features, confidence).with_source("auditory_perception")
}

pub fn linguistic_input(features: BinaryHV, confidence: f64) -> ModalInput {
    ModalInput::new(Modality::Linguistic, features, confidence).with_source("language_processing")
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modal_input_creation() {
        let features = BinaryHV::random(42);
        let input = ModalInput::new(Modality::Visual, features, 0.9).with_source("test");

        assert_eq!(input.modality, Modality::Visual);
        assert_eq!(input.confidence, 0.9);
        assert_eq!(input.source, "test");
    }

    #[test]
    fn test_integrator_creation() {
        let integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        assert_eq!(integrator.channels.len(), Modality::all().len());
        assert!(!integrator.convergence_zones.is_empty());
        assert!(integrator.active_modalities().is_empty());
    }

    #[test]
    fn test_integration_cycle() {
        let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let inputs = vec![
            visual_input(BinaryHV::random(1), 0.8),
            auditory_input(BinaryHV::random(2), 0.7),
            linguistic_input(BinaryHV::random(3), 0.9),
        ];

        let result = integrator.integrate(&inputs);

        assert!((0.0..=1.0).contains(&result.integrated_phi));
        assert_eq!(result.attention_weights.len(), 3);
        assert_eq!(integrator.stats().total_cycles, 1);
        assert_eq!(integrator.stats().total_inputs, 3);
    }

    #[test]
    fn empty_cycle_is_true_absence_and_clears_previous_presence() {
        let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let first = integrator.integrate(&[visual_input(BinaryHV::random(1), 0.9)]);
        assert!(!first.active_zones.is_empty());
        assert_eq!(integrator.active_modalities().len(), 1);

        let empty = integrator.integrate(&[]);
        assert!(integrator.active_modalities().is_empty());
        assert!(empty.active_zones.is_empty());
        assert!(empty.attention_weights.is_empty());
        assert!(empty.dominant_modality.is_none());
        assert_eq!(empty.integrated_phi, 0.0);
        assert_eq!(empty.binding_coherence, 0.0);
        assert_eq!(empty.unified_representation, BinaryHV::zero());
    }

    #[test]
    fn zero_confidence_is_present_but_does_not_fuse() {
        let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let result = integrator.integrate(&[visual_input(BinaryHV::random(7), 0.0)]);

        assert!(integrator.active_modalities().contains(&Modality::Visual));
        assert_eq!(integrator.current_confidence(Modality::Visual), Some(0.0));
        assert_eq!(result.attention_weights.get("Visual"), Some(&0.0));
        assert!(result.active_zones.is_empty());
        assert_eq!(result.unified_representation, BinaryHV::zero());
        assert!(result.dominant_modality.is_none());
    }

    #[test]
    fn non_finite_confidence_is_pessimistically_untrusted() {
        let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let result = integrator.integrate(&[visual_input(BinaryHV::random(7), f64::NAN)]);

        assert_eq!(integrator.current_confidence(Modality::Visual), Some(0.0));
        assert_eq!(result.attention_weights.get("Visual"), Some(&0.0));
        assert!(result.active_zones.is_empty());
    }

    #[test]
    fn absent_modalities_do_not_appear_as_primary_zones() {
        let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let result = integrator.integrate(&[visual_input(BinaryHV::random(7), 0.9)]);

        let primary_sources: Vec<_> = result
            .active_zones
            .iter()
            .filter(|zone| zone.level == "Primary")
            .flat_map(|zone| zone.sources.iter().cloned())
            .collect();
        assert_eq!(primary_sources, vec!["Visual".to_string()]);
    }

    #[test]
    fn confidence_changes_actual_fused_representation() {
        let feature = BinaryHV::random(42);
        let audio = BinaryHV::random(43);

        let mut high = MultiModalIntegrator::new(IntegrationConfig::default());
        let high_result = high.integrate(&[
            visual_input(feature, 1.0),
            auditory_input(audio, 1.0),
        ]);

        let mut low = MultiModalIntegrator::new(IntegrationConfig::default());
        let low_result = low.integrate(&[
            visual_input(feature, 0.05),
            auditory_input(audio, 1.0),
        ]);

        assert_ne!(
            high_result.unified_representation,
            low_result.unified_representation,
            "current confidence must affect the actual fused vector, not only metadata"
        );
    }

    #[test]
    fn test_coherent_state() {
        let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let features = BinaryHV::random(42);
        let mut last_coherence = 0.0;
        for _ in 0..5 {
            let result = integrator.integrate(&[visual_input(features, 0.9)]);
            last_coherence = result.binding_coherence;
        }
        assert!((0.0..=1.0).contains(&last_coherence));
    }

    #[test]
    fn test_primitive_routing() {
        let integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let system = PrimitiveSystem::new();
        let matches = integrator.route_to_primitives(&system);
        let prim_count = system.all_primitives().count();
        assert!(matches.len() <= prim_count);
    }

    #[test]
    fn test_streaming_events() {
        let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let receiver = integrator.enable_streaming();
        let _ = integrator.integrate(&[visual_input(BinaryHV::random(1), 0.8)]);

        std::thread::sleep(Duration::from_millis(10));
        let mut event_count = 0;
        while receiver.try_recv().is_ok() {
            event_count += 1;
        }
        assert!(event_count >= 1);
    }

    #[test]
    fn test_reset() {
        let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let _ = integrator.integrate(&[visual_input(BinaryHV::random(1), 0.8)]);

        integrator.reset();
        assert_eq!(integrator.current_phi(), 0.0);
        assert!(integrator.active_modalities().is_empty());
        assert!(integrator.current_confidence.is_empty());
        assert!(integrator.episodic_buffer.is_empty());
    }
}
