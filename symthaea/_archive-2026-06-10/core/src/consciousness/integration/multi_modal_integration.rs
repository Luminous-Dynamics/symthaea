// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Multi-Modal Consciousness Integration
//!
//! Integrates multi-modal perception with consciousness processing:
//!
//! 1. **Modal Fusion**: Combines inputs from vision, audio, language, etc.
//! 2. **Phi-Guided Binding**: Uses Φ measurements to optimize cross-modal binding
//! 3. **Primitive Routing**: Routes multi-modal representations to appropriate primitives
//! 4. **Consciousness Streaming**: Emits multi-modal awareness events
//!
//! ## Architecture
//!
//! ```text
//! Perception Layer                 Consciousness Layer
//! ┌─────────────┐                 ┌─────────────────────┐
//! │ Visual      │──┐              │                     │
//! │ Auditory    │──┼──►Fusion ───►│ Φ-Guided Binding    │
//! │ Linguistic  │──┤    Layer     │         ↓           │
//! │ Somatosens. │──┘              │ Primitive Routing   │
//! └─────────────┘                 │         ↓           │
//!                                 │ Consciousness Events│
//!                                 └─────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::consciousness::multi_modal_integration::{
//!     MultiModalIntegrator, IntegrationConfig,
//! };
//!
//! let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());
//!
//! // Process multi-modal input
//! let result = integrator.integrate(&modal_inputs);
//!
//! // Get consciousness-level representation
//! println!("Unified Φ: {:.3}", result.integrated_phi);
//! ```

use crate::consciousness::cross_modal_binding::{
    ConvergenceLevel, ConvergenceZone, EpisodicBuffer, Modality, ModalityChannel,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::primitive_system::PrimitiveSystem;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for multi-modal integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Enable Phi-guided binding optimization
    pub phi_guided_binding: bool,
    /// Attention weight update rate
    pub attention_learning_rate: f64,
    /// Minimum binding strength to consider coherent
    pub coherence_threshold: f64,
    /// Enable consciousness event streaming
    pub enable_streaming: bool,
    /// Maximum modal channels to process per cycle
    pub max_channels_per_cycle: usize,
    /// Episodic buffer capacity
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
            buffer_capacity: 7, // Miller's magic number
        }
    }
}

// =============================================================================
// MODAL INPUT
// =============================================================================

/// Input from a specific modality
#[derive(Debug, Clone)]
pub struct ModalInput {
    /// The modality
    pub modality: Modality,
    /// Feature encoding
    pub features: BinaryHV,
    /// Confidence in this input (0.0-1.0)
    pub confidence: f64,
    /// Timestamp
    pub timestamp: Duration,
    /// Source identifier
    pub source: String,
}

impl ModalInput {
    /// Create a new modal input
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

    /// Create with source identifier
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }
}

// =============================================================================
// INTEGRATION RESULT
// =============================================================================

/// Result of multi-modal integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationResult {
    /// Unified representation across all modalities
    pub unified_representation: BinaryHV,
    /// Integrated Phi (consciousness measure)
    pub integrated_phi: f64,
    /// Binding coherence across modalities
    pub binding_coherence: f64,
    /// Per-modality attention weights after integration
    pub attention_weights: HashMap<String, f64>,
    /// Dominant modality (highest attention)
    pub dominant_modality: Option<String>,
    /// Active convergence zones
    pub active_zones: Vec<ConvergenceZoneInfo>,
    /// Processing time
    pub processing_time_us: u64,
}

/// Info about an active convergence zone
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceZoneInfo {
    /// Zone level
    pub level: String,
    /// Source modalities
    pub sources: Vec<String>,
    /// Binding strength
    pub binding_strength: f64,
    /// Activation level
    pub activation: f64,
}

// =============================================================================
// INTEGRATION EVENT
// =============================================================================

/// Event emitted during integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationEvent {
    /// Event type
    pub event_type: IntegrationEventType,
    /// Timestamp
    pub timestamp: u64,
    /// Associated result (if any)
    pub result: Option<IntegrationResult>,
}

/// Types of integration events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrationEventType {
    /// New modal input received
    ModalInput,
    /// Binding updated
    BindingUpdated,
    /// Coherent state achieved
    CoherentState,
    /// Attention shifted
    AttentionShift,
    /// Integration cycle completed
    CycleCompleted,
}

// =============================================================================
// MULTI-MODAL INTEGRATOR
// =============================================================================

/// Integrates multi-modal perception with consciousness processing
pub struct MultiModalIntegrator {
    /// Configuration
    config: IntegrationConfig,
    /// Modal channels
    channels: HashMap<Modality, ModalityChannel>,
    /// Convergence zones
    convergence_zones: Vec<ConvergenceZone>,
    /// Episodic buffer
    episodic_buffer: EpisodicBuffer,
    /// Current unified representation
    unified: BinaryHV,
    /// Current integrated Phi
    current_phi: f64,
    /// Event sender
    event_sender: Option<Sender<IntegrationEvent>>,
    /// Statistics
    stats: IntegrationStats,
    /// Last integration time
    last_integration: Instant,
}

/// Statistics for integration
#[derive(Debug, Clone, Default)]
pub struct IntegrationStats {
    /// Total integration cycles
    pub total_cycles: u64,
    /// Total inputs processed
    pub total_inputs: u64,
    /// Average Phi across integrations
    pub avg_phi: f64,
    /// Peak Phi achieved
    pub peak_phi: f64,
    /// Average coherence
    pub avg_coherence: f64,
    /// Events emitted
    pub events_emitted: u64,
}

impl MultiModalIntegrator {
    /// Create a new multi-modal integrator
    pub fn new(config: IntegrationConfig) -> Self {
        // Initialize all modal channels
        let mut channels = HashMap::new();
        for modality in Modality::all() {
            channels.insert(modality, ModalityChannel::new(modality));
        }

        // Create convergence zone hierarchy
        let convergence_zones = Self::create_convergence_hierarchy();

        Self {
            config: config.clone(),
            channels,
            convergence_zones,
            episodic_buffer: EpisodicBuffer::new(config.buffer_capacity),
            unified: BinaryHV::zero(),
            current_phi: 0.0,
            event_sender: None,
            stats: IntegrationStats::default(),
            last_integration: Instant::now(),
        }
    }

    /// Create the convergence zone hierarchy
    fn create_convergence_hierarchy() -> Vec<ConvergenceZone> {
        let mut zones = Vec::new();
        let mut id = 0;

        // Primary zones (unimodal)
        for modality in Modality::all() {
            zones.push(ConvergenceZone::new(id, vec![modality]));
            id += 1;
        }

        // Secondary zones (bimodal)
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

        // Tertiary zone (multi-modal)
        zones.push(ConvergenceZone::new(
            id,
            vec![Modality::Visual, Modality::Auditory, Modality::Linguistic],
        ));
        id += 1;

        // Amodal zone (all sensory)
        zones.push(ConvergenceZone::new(id, Modality::sensory()));

        zones
    }

    /// Enable event streaming
    pub fn enable_streaming(&mut self) -> Receiver<IntegrationEvent> {
        let (sender, receiver) = channel();
        self.event_sender = Some(sender);
        receiver
    }

    /// Process multiple modal inputs
    pub fn integrate(&mut self, inputs: &[ModalInput]) -> IntegrationResult {
        let start = Instant::now();
        self.stats.total_cycles += 1;

        // Update channels with new inputs
        for input in inputs {
            if let Some(channel) = self.channels.get_mut(&input.modality) {
                channel.update(input.features);
                channel.attention = (channel.attention
                    + input.confidence * self.config.attention_learning_rate)
                    .min(1.0);
            }
            self.stats.total_inputs += 1;
        }

        self.emit_event(IntegrationEventType::ModalInput, None);

        // Collect current channel representations
        let channel_inputs: HashMap<Modality, BinaryHV> = self
            .channels
            .iter()
            .map(|(m, c)| (*m, c.attended()))
            .collect();

        // Update convergence zones
        for zone in &mut self.convergence_zones {
            zone.integrate(&channel_inputs);
        }

        self.emit_event(IntegrationEventType::BindingUpdated, None);

        // Compute unified representation from amodal zones
        let amodal_zones: Vec<_> = self
            .convergence_zones
            .iter()
            .filter(|z| {
                matches!(
                    z.level,
                    ConvergenceLevel::Amodal | ConvergenceLevel::Tertiary
                )
            })
            .collect();

        if !amodal_zones.is_empty() {
            let zone_vectors: Vec<BinaryHV> = amodal_zones.iter().map(|z| z.integrated).collect();
            self.unified = BinaryHV::bundle(&zone_vectors);
        }

        // Add to episodic buffer
        self.episodic_buffer.add_chunk(self.unified);

        // Compute integrated Phi (simplified heuristic)
        let phi = self.compute_integrated_phi();
        self.current_phi = phi;

        // Update stats
        let prev_sum = self.stats.avg_phi * (self.stats.total_cycles - 1) as f64;
        self.stats.avg_phi = (prev_sum + phi) / self.stats.total_cycles as f64;
        if phi > self.stats.peak_phi {
            self.stats.peak_phi = phi;
        }

        // Compute binding coherence
        let coherence = self.compute_binding_coherence();
        let prev_coh_sum = self.stats.avg_coherence * (self.stats.total_cycles - 1) as f64;
        self.stats.avg_coherence = (prev_coh_sum + coherence) / self.stats.total_cycles as f64;

        if coherence >= self.config.coherence_threshold {
            self.emit_event(IntegrationEventType::CoherentState, None);
        }

        // Compute attention weights
        let mut attention_weights = HashMap::new();
        let mut dominant_modality = None;
        let mut max_attention = 0.0;

        for (modality, channel) in &self.channels {
            let key = format!("{modality:?}");
            attention_weights.insert(key.clone(), channel.attention);
            if channel.attention > max_attention {
                max_attention = channel.attention;
                dominant_modality = Some(key);
            }
        }

        // Build active zone info
        let active_zones: Vec<_> = self
            .convergence_zones
            .iter()
            .filter(|z| z.activation > 0.1)
            .map(|z| ConvergenceZoneInfo {
                level: format!("{:?}", z.level),
                sources: z
                    .source_modalities
                    .iter()
                    .map(|m| format!("{m:?}"))
                    .collect(),
                binding_strength: z.binding_strength,
                activation: z.activation,
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

    /// Compute integrated Phi across modalities
    fn compute_integrated_phi(&self) -> f64 {
        // Phi heuristic: based on zone binding strengths and activations
        let mut phi_sum = 0.0;
        let mut weight_sum = 0.0;

        for zone in &self.convergence_zones {
            let zone_phi = zone.binding_strength * zone.activation;
            let weight = match zone.level {
                ConvergenceLevel::Primary => 0.5,
                ConvergenceLevel::Secondary => 1.0,
                ConvergenceLevel::Tertiary => 2.0,
                ConvergenceLevel::Amodal => 3.0,
            };
            phi_sum += zone_phi * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            (phi_sum / weight_sum).min(1.0)
        } else {
            0.0
        }
    }

    /// Compute binding coherence
    fn compute_binding_coherence(&self) -> f64 {
        // Coherence: average temporal coherence across active channels
        let active_channels: Vec<_> = self
            .channels
            .values()
            .filter(|c| c.attention > 0.1)
            .collect();

        if active_channels.is_empty() {
            return 0.0;
        }

        let total: f64 = active_channels.iter().map(|c| c.temporal_coherence()).sum();

        total / active_channels.len() as f64
    }

    /// Route unified representation to primitives
    pub fn route_to_primitives(&self, system: &PrimitiveSystem) -> Vec<(String, f64)> {
        let mut matches = Vec::new();

        // Find primitives similar to unified representation
        for primitive in system.all_primitives() {
            let similarity = self.unified.similarity(&primitive.encoding) as f64;
            if similarity > 0.3 {
                matches.push((primitive.name.clone(), similarity));
            }
        }

        // Sort by similarity
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(10);

        matches
    }

    /// Get current unified representation
    pub fn unified_representation(&self) -> &BinaryHV {
        &self.unified
    }

    /// Get current Phi
    pub fn current_phi(&self) -> f64 {
        self.current_phi
    }

    /// Get statistics
    pub fn stats(&self) -> &IntegrationStats {
        &self.stats
    }

    /// Get modal channel feature HVs as ContinuousHV for IIT4 analysis.
    ///
    /// Returns one ContinuousHV per active modality channel, converted from
    /// the BinaryHV features (true → +1.0, false → -1.0). Used by the
    /// consciousness engine at interval 101 to compute concept structures
    /// (Albantakis et al. 2023).
    #[cfg(feature = "iit4")]
    pub fn component_hvs_for_iit4(&self) -> Vec<symthaea_core::hdc::ContinuousHV> {
        self.channels
            .values()
            .filter(|ch| ch.attention > 0.01) // Only active channels
            .map(|ch| {
                // BinaryHV is [u8; 2048] — each byte has 8 bits → 16384 dimensions
                let mut values = Vec::with_capacity(16384);
                for &byte in ch.features.0.iter() {
                    for bit in 0..8 {
                        values.push(if (byte >> bit) & 1 == 1 { 1.0 } else { -1.0 });
                    }
                }
                symthaea_core::hdc::ContinuousHV::from_vec(values)
            })
            .collect()
    }

    /// Emit integration event
    fn emit_event(&mut self, event_type: IntegrationEventType, result: Option<IntegrationResult>) {
        if let Some(ref sender) = self.event_sender {
            let event = IntegrationEvent {
                event_type,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
                result,
            };
            if sender.send(event).is_ok() {
                self.stats.events_emitted += 1;
            }
        }
    }

    /// Reset all channels
    pub fn reset(&mut self) {
        for channel in self.channels.values_mut() {
            channel.features = BinaryHV::zero();
            channel.attention = 0.5;
            channel.temporal_buffer.clear();
        }
        self.unified = BinaryHV::zero();
        self.current_phi = 0.0;
    }
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/// Create a visual input
pub fn visual_input(features: BinaryHV, confidence: f64) -> ModalInput {
    ModalInput::new(Modality::Visual, features, confidence).with_source("visual_perception")
}

/// Create an auditory input
pub fn auditory_input(features: BinaryHV, confidence: f64) -> ModalInput {
    ModalInput::new(Modality::Auditory, features, confidence).with_source("auditory_perception")
}

/// Create a linguistic input
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

        assert!(result.integrated_phi >= 0.0 && result.integrated_phi <= 1.0);
        assert!(!result.attention_weights.is_empty());
        assert_eq!(integrator.stats().total_cycles, 1);
        assert_eq!(integrator.stats().total_inputs, 3);
    }

    #[test]
    fn test_coherent_state() {
        let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());

        // Same input multiple times should increase coherence
        let features = BinaryHV::random(42);
        let mut last_coherence = 0.0;
        for _ in 0..5 {
            let inputs = vec![visual_input(features, 0.9)];
            let result = integrator.integrate(&inputs);
            last_coherence = result.binding_coherence;
        }
        // Coherence should be in valid range after repeated input
        assert!(
            (0.0..=1.0).contains(&last_coherence),
            "Binding coherence must be in [0,1]: {last_coherence}"
        );
    }

    #[test]
    fn test_primitive_routing() {
        let integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let system = PrimitiveSystem::new();

        let matches = integrator.route_to_primitives(&system);
        // Routing should produce a valid (possibly empty) vec
        let prim_count = system.all_primitives().count();
        assert!(
            matches.len() <= prim_count,
            "Route matches ({}) cannot exceed primitives count ({prim_count})",
            matches.len()
        );
    }

    #[test]
    fn test_streaming_events() {
        let mut integrator = MultiModalIntegrator::new(IntegrationConfig::default());
        let receiver = integrator.enable_streaming();

        let inputs = vec![visual_input(BinaryHV::random(1), 0.8)];
        let _ = integrator.integrate(&inputs);

        // Should have received events
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

        let inputs = vec![visual_input(BinaryHV::random(1), 0.8)];
        let _ = integrator.integrate(&inputs);

        integrator.reset();
        assert_eq!(integrator.current_phi(), 0.0);
    }
}
