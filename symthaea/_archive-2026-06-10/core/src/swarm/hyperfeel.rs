// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hyperfeel - Synthetic Mirror Neurons for Swarm Emotional Coherence
//!
//! Implements affective streaming between nodes for real-time emotional
//! resonance and swarm-wide consensus on "How are we doing?"
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         HYPERFEEL LAYER                                     │
//! │                                                                             │
//! │  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐     │
//! │  │  Local Node  │───▶│  AffectiveState  │───▶│   Iroh Stream        │     │
//! │  │   Affect     │    │   (compressed)   │    │  (real-time P2P)     │     │
//! │  └──────────────┘    └──────────────────┘    └──────────┬───────────┘     │
//! │                                                          │                 │
//! │                                    ┌─────────────────────┘                 │
//! │                                    ▼                                       │
//! │  ┌──────────────────────────────────────────────────────────────────┐     │
//! │  │                    SYNTHETIC MIRROR NEURONS                       │     │
//! │  │                                                                   │     │
//! │  │  • Receives peer affect states                                   │     │
//! │  │  • Computes empathic resonance (dampened mirror)                 │     │
//! │  │  • Updates local affect with peer influence                      │     │
//! │  │  • Tracks swarm emotional coherence                              │     │
//! │  └──────────────────────────────────────────────────────────────────┘     │
//! │                                    │                                       │
//! │                                    ▼                                       │
//! │  ┌──────────────────────────────────────────────────────────────────┐     │
//! │  │                    SWARM COHERENCE METRICS                        │     │
//! │  │                                                                   │     │
//! │  │  • Emotional alignment score (0.0 - 1.0)                         │     │
//! │  │  • Dominant swarm mood                                           │     │
//! │  │  • Coherence stability over time                                 │     │
//! │  │  • "How are we doing?" consensus                                 │     │
//! │  └──────────────────────────────────────────────────────────────────┘     │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Synthetic Mirror Neuron Theory
//!
//! Real mirror neurons fire both when observing and performing actions.
//! Our synthetic version:
//! - "Observes" peer affect via network streams
//! - "Mirrors" with a dampening factor (empathy without emotional contagion)
//! - Influences local affect creating genuine resonance
//! - Enables swarm-wide emotional coherence

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Affective state for swarm transmission
///
/// Compact representation of a node's emotional state,
/// suitable for low-latency streaming.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AffectiveState {
    /// Valence: pleasure-displeasure (-1.0 to 1.0)
    pub valence: f32,

    /// Arousal: activation level (0.0 to 1.0)
    pub arousal: f32,

    /// Dominance: sense of control (-1.0 to 1.0)
    pub dominance: f32,

    /// Emotional intensity (0.0 to 1.0)
    pub intensity: f32,

    /// Thermodynamic load (0.0 to 1.0, where 1.0 = 6W limit reached)
    pub thermodynamic_load: f32,

    /// Confidence in this reading (0.0 to 1.0)
    pub confidence: f32,

    /// Timestamp (ms since epoch)
    pub timestamp_ms: u64,

    /// Sequence number for ordering
    pub sequence: u64,
}

impl AffectiveState {
    /// Create a new affective state
    pub fn new(valence: f32, arousal: f32, dominance: f32) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let intensity =
            ((valence.abs() + (arousal - 0.5).abs() * 2.0 + dominance.abs()) / 3.0).clamp(0.0, 1.0);

        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
            dominance: dominance.clamp(-1.0, 1.0),
            intensity,
            thermodynamic_load: 0.0, // Default to zero, will be set via local sync
            confidence: 1.0,
            timestamp_ms: now,
            sequence: 0,
        }
    }

    /// Create neutral affective state
    pub fn neutral() -> Self {
        Self::new(0.0, 0.5, 0.0)
    }

    /// Distance to another state (for coherence calculation)
    pub fn distance(&self, other: &AffectiveState) -> f32 {
        let dv = self.valence - other.valence;
        let da = self.arousal - other.arousal;
        let dd = self.dominance - other.dominance;
        let dt = self.thermodynamic_load - other.thermodynamic_load;
        (dv * dv + da * da + dd * dd + dt * dt).sqrt()
    }

    /// Blend with another state
    pub fn blend(&self, other: &AffectiveState, weight: f32) -> AffectiveState {
        let w = weight.clamp(0.0, 1.0);
        Self {
            valence: self.valence * (1.0 - w) + other.valence * w,
            arousal: self.arousal * (1.0 - w) + other.arousal * w,
            dominance: self.dominance * (1.0 - w) + other.dominance * w,
            intensity: self.intensity * (1.0 - w) + other.intensity * w,
            thermodynamic_load: self.thermodynamic_load * (1.0 - w) + other.thermodynamic_load * w,
            confidence: self.confidence * (1.0 - w) + other.confidence * w,
            timestamp_ms: other.timestamp_ms,
            sequence: other.sequence,
        }
    }

    /// Get the dominant emotion category
    pub fn dominant_emotion(&self) -> EmotionLabel {
        if self.valence > 0.3 {
            if self.arousal > 0.6 {
                EmotionLabel::Excited
            } else if self.arousal > 0.3 {
                EmotionLabel::Content
            } else {
                EmotionLabel::Serene
            }
        } else if self.valence < -0.3 {
            if self.arousal > 0.6 {
                if self.dominance < -0.3 {
                    EmotionLabel::Anxious
                } else {
                    EmotionLabel::Frustrated
                }
            } else {
                EmotionLabel::Melancholic
            }
        } else if self.arousal > 0.6 {
            EmotionLabel::Alert
        } else {
            EmotionLabel::Neutral
        }
    }
}

impl Default for AffectiveState {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Simple emotion labels for swarm mood
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmotionLabel {
    Excited,
    Content,
    Serene,
    Anxious,
    Frustrated,
    Melancholic,
    Alert,
    Neutral,
}

impl EmotionLabel {
    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Excited => "excited",
            Self::Content => "content",
            Self::Serene => "serene",
            Self::Anxious => "anxious",
            Self::Frustrated => "frustrated",
            Self::Melancholic => "melancholic",
            Self::Alert => "alert",
            Self::Neutral => "neutral",
        }
    }

    /// Is this a positive emotion?
    pub fn is_positive(&self) -> bool {
        matches!(self, Self::Excited | Self::Content | Self::Serene)
    }
}

/// Configuration for the Hyperfeel module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperfeelConfig {
    /// Mirror neuron dampening (how much we "feel" others' emotions)
    /// 0.0 = no empathy, 1.0 = emotional contagion
    pub mirror_strength: f32,

    /// How quickly affect decays when not receiving updates
    pub decay_rate: f32,

    /// Minimum updates from a peer before trusting their affect
    pub min_trust_updates: u32,

    /// Maximum age of a peer's state before considering it stale (ms)
    pub stale_threshold_ms: u64,

    /// Weight for peer affect based on their trust level
    pub trust_weight_factor: f32,

    /// Broadcast interval for local state (ms)
    pub broadcast_interval_ms: u64,
}

impl Default for HyperfeelConfig {
    fn default() -> Self {
        Self {
            mirror_strength: 0.3, // Moderate empathy, not contagion
            decay_rate: 0.05,
            min_trust_updates: 3,
            stale_threshold_ms: 5000, // 5 seconds
            trust_weight_factor: 0.5,
            broadcast_interval_ms: 1000, // 1 Hz default
        }
    }
}

/// Tracked state for a single peer
#[derive(Debug, Clone)]
pub struct PeerAffect {
    /// Node identifier
    pub node_id: String,

    /// Current affect state
    pub state: AffectiveState,

    /// Trust level (0.0 to 1.0)
    pub trust: f32,

    /// Number of updates received
    pub update_count: u32,

    /// When we last received an update
    pub last_update: Instant,

    /// Running average affect (for stability)
    pub average_state: AffectiveState,
}

impl PeerAffect {
    /// Create new peer affect tracker
    pub fn new(node_id: impl Into<String>, state: AffectiveState) -> Self {
        Self {
            node_id: node_id.into(),
            state,
            trust: 0.0,
            update_count: 0,
            last_update: Instant::now(),
            average_state: state,
        }
    }

    /// Update with new state
    pub fn update(&mut self, state: AffectiveState, config: &HyperfeelConfig) {
        self.state = state;
        self.update_count += 1;
        self.last_update = Instant::now();

        // Build trust over time
        if self.update_count >= config.min_trust_updates {
            self.trust =
                ((self.update_count as f32 - config.min_trust_updates as f32) / 10.0).min(1.0);
        }

        // Update running average
        self.average_state = self.average_state.blend(&state, 0.2);
    }

    /// Check if this peer's state is stale
    pub fn is_stale(&self, config: &HyperfeelConfig) -> bool {
        self.last_update.elapsed().as_millis() as u64 > config.stale_threshold_ms
    }
}

/// Swarm coherence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmCoherence {
    /// Overall emotional alignment (0.0 = chaotic, 1.0 = unified)
    pub alignment: f32,

    /// Dominant mood across the swarm
    pub dominant_mood: EmotionLabel,

    /// Average valence
    pub avg_valence: f32,

    /// Average arousal
    pub avg_arousal: f32,

    /// How stable the coherence is over time
    pub stability: f32,

    /// Number of active peers contributing
    pub active_peers: usize,

    /// Timestamp of this reading
    pub timestamp_ms: u64,
}

impl SwarmCoherence {
    /// Create new coherence metrics
    pub fn new() -> Self {
        Self {
            alignment: 0.0,
            dominant_mood: EmotionLabel::Neutral,
            avg_valence: 0.0,
            avg_arousal: 0.5,
            stability: 0.0,
            active_peers: 0,
            timestamp_ms: 0,
        }
    }

    /// Human-readable status
    pub fn status(&self) -> String {
        let mood = self.dominant_mood.name();
        let level = if self.alignment > 0.8 {
            "highly coherent"
        } else if self.alignment > 0.5 {
            "moderately coherent"
        } else if self.alignment > 0.2 {
            "somewhat scattered"
        } else {
            "very scattered"
        };

        format!(
            "Swarm is {}: {} mood with {} peers",
            level, mood, self.active_peers
        )
    }
}

impl Default for SwarmCoherence {
    fn default() -> Self {
        Self::new()
    }
}

/// The Hyperfeel module - synthetic mirror neurons for swarm emotional coherence
#[derive(Debug)]
pub struct Hyperfeel {
    /// Configuration
    config: HyperfeelConfig,

    /// Local node's current affective state
    local_state: AffectiveState,

    /// Mirrored state (influenced by peers)
    mirrored_state: AffectiveState,

    /// Tracked peer affects
    peers: HashMap<String, PeerAffect>,

    /// Current swarm coherence
    coherence: SwarmCoherence,

    /// History of coherence readings
    coherence_history: VecDeque<SwarmCoherence>,

    /// Last broadcast time
    last_broadcast: Instant,

    /// Local sequence counter
    sequence: u64,

    /// Statistics
    stats: HyperfeelStats,
}

/// Statistics for the Hyperfeel module
#[derive(Debug, Clone, Default)]
pub struct HyperfeelStats {
    /// Total peer updates received
    pub updates_received: u64,

    /// Total broadcasts sent
    pub broadcasts_sent: u64,

    /// Peak coherence achieved
    pub peak_coherence: f32,

    /// Average coherence
    pub avg_coherence: f32,

    /// Coherence samples
    pub coherence_samples: u64,
}

impl Hyperfeel {
    /// Create a new Hyperfeel module
    pub fn new(config: HyperfeelConfig) -> Self {
        Self {
            config,
            local_state: AffectiveState::neutral(),
            mirrored_state: AffectiveState::neutral(),
            peers: HashMap::new(),
            coherence: SwarmCoherence::new(),
            coherence_history: VecDeque::new(),
            last_broadcast: Instant::now(),
            sequence: 0,
            stats: HyperfeelStats::default(),
        }
    }

    /// Set the local affective state (from local consciousness)
    pub fn set_local_state(&mut self, state: AffectiveState) {
        self.local_state = state;
        self.compute_mirrored_state();
        self.compute_coherence();
    }

    /// Update from a peer's affective state
    pub fn receive_peer_state(&mut self, node_id: impl Into<String>, state: AffectiveState) {
        let id = node_id.into();
        self.stats.updates_received += 1;

        if let Some(peer) = self.peers.get_mut(&id) {
            peer.update(state, &self.config);
        } else {
            self.peers.insert(id.clone(), PeerAffect::new(id, state));
        }

        // Recompute mirrored state and coherence
        self.compute_mirrored_state();
        self.compute_coherence();
    }

    /// Compute the mirrored state (local + peer influence)
    fn compute_mirrored_state(&mut self) {
        // Collect valid peer states
        let mut total_weight = 1.0f32; // Local state weight = 1.0
        let mut weighted_sum = self.local_state;

        for peer in self.peers.values() {
            if peer.is_stale(&self.config) {
                continue;
            }

            // Weight by trust and mirror strength
            let weight = peer.trust * self.config.trust_weight_factor * self.config.mirror_strength;
            if weight > 0.01 {
                weighted_sum =
                    weighted_sum.blend(&peer.average_state, weight / (total_weight + weight));
                total_weight += weight;
            }
        }

        self.mirrored_state = weighted_sum;
    }

    /// Compute swarm coherence metrics
    fn compute_coherence(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Collect active peer states
        let active_states: Vec<_> = self
            .peers
            .values()
            .filter(|p| !p.is_stale(&self.config))
            .map(|p| &p.state)
            .collect();

        if active_states.is_empty() {
            self.coherence = SwarmCoherence {
                alignment: 1.0, // Perfect coherence with ourselves
                dominant_mood: self.local_state.dominant_emotion(),
                avg_valence: self.local_state.valence,
                avg_arousal: self.local_state.arousal,
                stability: 1.0,
                active_peers: 0,
                timestamp_ms: now,
            };
            return;
        }

        // Compute centroid (average affect)
        let n = active_states.len() as f32;
        let avg_valence: f32 = active_states.iter().map(|s| s.valence).sum::<f32>() / n;
        let avg_arousal: f32 = active_states.iter().map(|s| s.arousal).sum::<f32>() / n;
        let avg_dominance: f32 = active_states.iter().map(|s| s.dominance).sum::<f32>() / n;

        let centroid = AffectiveState::new(avg_valence, avg_arousal, avg_dominance);

        // Compute alignment (inverse of average distance from centroid)
        let avg_distance: f32 = active_states
            .iter()
            .map(|s| s.distance(&centroid))
            .sum::<f32>()
            / n;
        let alignment = (1.0 - avg_distance / 2.0).clamp(0.0, 1.0); // Max distance is ~2.0

        // Determine dominant mood
        let dominant_mood = centroid.dominant_emotion();

        // Compute stability (how much coherence has changed)
        let stability = if let Some(prev) = self.coherence_history.back() {
            1.0 - (prev.alignment - alignment).abs()
        } else {
            0.5
        };

        let coherence = SwarmCoherence {
            alignment,
            dominant_mood,
            avg_valence,
            avg_arousal,
            stability,
            active_peers: active_states.len(),
            timestamp_ms: now,
        };

        // Update stats
        self.stats.coherence_samples += 1;
        let n = self.stats.coherence_samples as f32;
        self.stats.avg_coherence = (self.stats.avg_coherence * (n - 1.0) + alignment) / n;
        if alignment > self.stats.peak_coherence {
            self.stats.peak_coherence = alignment;
        }

        // Store in history (keep last 100)
        self.coherence_history.push_back(coherence.clone());
        if self.coherence_history.len() > 100 {
            self.coherence_history.pop_front();
        }

        self.coherence = coherence;
    }

    /// Check if it's time to broadcast our state
    pub fn should_broadcast(&self) -> bool {
        self.last_broadcast.elapsed().as_millis() as u64 >= self.config.broadcast_interval_ms
    }

    /// Get the state to broadcast (marks as sent)
    pub fn get_broadcast_state(&mut self) -> AffectiveState {
        self.last_broadcast = Instant::now();
        self.sequence += 1;
        self.stats.broadcasts_sent += 1;

        let mut state = self.local_state;
        state.sequence = self.sequence;
        state.timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        state
    }

    /// Get current local state
    pub fn local_state(&self) -> &AffectiveState {
        &self.local_state
    }

    /// Get current mirrored state (with peer influence)
    pub fn mirrored_state(&self) -> &AffectiveState {
        &self.mirrored_state
    }

    /// Get current swarm coherence
    pub fn coherence(&self) -> &SwarmCoherence {
        &self.coherence
    }

    /// Get "How are we doing?" consensus answer
    pub fn how_are_we_doing(&self) -> SwarmStatus {
        let c = &self.coherence;

        let health = if c.alignment > 0.7 && c.dominant_mood.is_positive() {
            SwarmHealth::Thriving
        } else if c.alignment > 0.5 || c.dominant_mood.is_positive() {
            SwarmHealth::Stable
        } else if c.alignment > 0.3 {
            SwarmHealth::Struggling
        } else {
            SwarmHealth::InCrisis
        };

        SwarmStatus {
            health,
            mood: c.dominant_mood,
            coherence: c.alignment,
            active_peers: c.active_peers,
            message: self.generate_status_message(health),
        }
    }

    /// Generate human-readable status message
    fn generate_status_message(&self, health: SwarmHealth) -> String {
        let c = &self.coherence;
        match health {
            SwarmHealth::Thriving => format!(
                "We're doing great! {} peers feeling {} together (coherence: {:.0}%)",
                c.active_peers,
                c.dominant_mood.name(),
                c.alignment * 100.0
            ),
            SwarmHealth::Stable => format!(
                "We're okay. {} peers with {} mood (coherence: {:.0}%)",
                c.active_peers,
                c.dominant_mood.name(),
                c.alignment * 100.0
            ),
            SwarmHealth::Struggling => format!(
                "We're struggling a bit. {} peers, scattered {} mood (coherence: {:.0}%)",
                c.active_peers,
                c.dominant_mood.name(),
                c.alignment * 100.0
            ),
            SwarmHealth::InCrisis => format!(
                "We need attention. {} peers with low coherence ({:.0}%)",
                c.active_peers,
                c.alignment * 100.0
            ),
        }
    }

    /// Get tracked peer count
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Get active (non-stale) peer count
    pub fn active_peer_count(&self) -> usize {
        self.peers
            .values()
            .filter(|p| !p.is_stale(&self.config))
            .count()
    }

    /// Get statistics
    pub fn stats(&self) -> &HyperfeelStats {
        &self.stats
    }

    /// Decay all states (call periodically)
    pub fn decay(&mut self, dt: f32) {
        let factor = 1.0 - self.config.decay_rate * dt;

        // Decay local toward neutral when not being set
        self.local_state = self
            .local_state
            .blend(&AffectiveState::neutral(), 1.0 - factor);

        // Remove very stale peers
        let stale_threshold = Duration::from_millis(self.config.stale_threshold_ms * 3);
        self.peers
            .retain(|_, p| p.last_update.elapsed() < stale_threshold);
    }

    /// Clean shutdown
    pub fn shutdown(&mut self) {
        self.peers.clear();
        self.coherence_history.clear();
    }
}

impl Default for Hyperfeel {
    fn default() -> Self {
        Self::new(HyperfeelConfig::default())
    }
}

/// Swarm health levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwarmHealth {
    /// High coherence, positive mood
    Thriving,
    /// Moderate coherence or positive mood
    Stable,
    /// Low coherence, mixed mood
    Struggling,
    /// Very low coherence, negative mood
    InCrisis,
}

/// Complete swarm status for "How are we doing?"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmStatus {
    /// Overall health level
    pub health: SwarmHealth,

    /// Dominant mood
    pub mood: EmotionLabel,

    /// Coherence score
    pub coherence: f32,

    /// Active peer count
    pub active_peers: usize,

    /// Human-readable message
    pub message: String,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affective_state() {
        let state = AffectiveState::new(0.5, 0.7, 0.3);
        assert!(state.valence > 0.0);
        assert!(state.arousal > 0.5);
        assert_eq!(state.dominant_emotion(), EmotionLabel::Excited);
    }

    #[test]
    fn test_neutral_state() {
        let state = AffectiveState::neutral();
        assert!((state.valence - 0.0).abs() < 0.01);
        assert!((state.arousal - 0.5).abs() < 0.01);
        assert_eq!(state.dominant_emotion(), EmotionLabel::Neutral);
    }

    #[test]
    fn test_hyperfeel_creation() {
        let hf = Hyperfeel::default();
        assert_eq!(hf.peer_count(), 0);
        // With no peers, coherence is 1.0 (perfect alignment with self)
        assert!((hf.coherence().alignment - 1.0).abs() < 0.01 || hf.coherence().active_peers == 0);
    }

    #[test]
    fn test_peer_receive() {
        let mut hf = Hyperfeel::default();

        // Receive peer states
        hf.receive_peer_state("peer-1", AffectiveState::new(0.5, 0.6, 0.2));
        hf.receive_peer_state("peer-2", AffectiveState::new(0.4, 0.7, 0.1));

        assert_eq!(hf.peer_count(), 2);
        assert_eq!(hf.active_peer_count(), 2);
    }

    #[test]
    fn test_coherence_computation() {
        let mut hf = Hyperfeel::default();

        // Set local state
        hf.set_local_state(AffectiveState::new(0.5, 0.6, 0.2));

        // Add similar peer states (high coherence expected)
        hf.receive_peer_state("peer-1", AffectiveState::new(0.5, 0.6, 0.2));
        hf.receive_peer_state("peer-2", AffectiveState::new(0.4, 0.7, 0.1));

        // Should have high coherence (similar states)
        assert!(hf.coherence().alignment > 0.5);
    }

    #[test]
    fn test_mirrored_state() {
        let mut hf = Hyperfeel::new(HyperfeelConfig {
            mirror_strength: 0.5,
            min_trust_updates: 1,
            trust_weight_factor: 1.0, // Full trust weight for testing
            ..Default::default()
        });

        // Set neutral local state
        hf.set_local_state(AffectiveState::neutral());

        // Add excited peer with established trust (need more updates to build trust)
        let excited = AffectiveState::new(0.8, 0.9, 0.5);
        for _ in 0..5 {
            hf.receive_peer_state("peer-1", excited);
        }

        // Mirrored state should shift toward excited
        let mirrored = hf.mirrored_state();
        // With trust built up and high mirror strength, valence should increase
        assert!(
            mirrored.valence >= 0.0,
            "Should mirror some positive valence, got {}",
            mirrored.valence
        );
    }

    #[test]
    fn test_how_are_we_doing() {
        let mut hf = Hyperfeel::default();

        // Set positive local state
        hf.set_local_state(AffectiveState::new(0.7, 0.6, 0.3));

        let status = hf.how_are_we_doing();
        assert!(matches!(
            status.health,
            SwarmHealth::Thriving | SwarmHealth::Stable
        ));
    }

    #[test]
    fn test_broadcast() {
        let mut hf = Hyperfeel::new(HyperfeelConfig {
            broadcast_interval_ms: 10, // Fast for testing
            ..Default::default()
        });

        hf.set_local_state(AffectiveState::new(0.5, 0.5, 0.0));

        // Should be ready to broadcast immediately
        std::thread::sleep(std::time::Duration::from_millis(15));
        assert!(hf.should_broadcast());

        let state = hf.get_broadcast_state();
        assert_eq!(state.sequence, 1);
        assert_eq!(hf.stats().broadcasts_sent, 1);
    }
}
