// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Resonance Graph — Consciousness-Aligned Content Discovery
//!
//! Replaces engagement-based content ranking with resonance-based routing.
//! Content is ranked by HDV Hamming similarity between the content's embedding
//! and the reader's consciousness state, with a diversity bonus that peaks
//! at moderate distance (0.3-0.7 Hamming) to prevent echo chambers.
//!
//! # Key Insight
//! Engagement-optimized feeds maximize dopamine spikes (addiction).
//! Resonance-optimized feeds maximize meaningful connection (growth).
//!
//! # References
//! - Woolley, A. et al. (2010). Evidence for a collective intelligence factor

use std::collections::{HashMap, VecDeque};
use symthaea_core::hdc::BinaryHV;

/// HDC dimension in bits (16,384 = 2048 bytes * 8).
const HDC_DIMENSION: u32 = 16_384;

/// Peak diversity bonus range (Hamming distance where bonus is maximized).
const DIVERSITY_PEAK_LOW: f64 = 0.3;
const DIVERSITY_PEAK_HIGH: f64 = 0.7;

/// Maximum diversity bonus multiplier.
const DIVERSITY_MAX_BONUS: f64 = 0.3;

/// Echo chamber warning threshold: fraction of content from high-resonance peers.
const ECHO_CHAMBER_THRESHOLD: f64 = 0.85;

/// Maximum content items tracked per peer.
const MAX_CONTENT_PER_PEER: usize = 100;

/// Maximum peers in the resonance graph.
const MAX_RESONANCE_PEERS: usize = 256;

/// A reference to a piece of content in the mesh.
#[derive(Debug, Clone)]
pub struct ContentRef {
    /// Peer who created/shared this content.
    pub source_peer: String,
    /// BLAKE3 hash of the content.
    pub content_hash: [u8; 32],
    /// HDV embedding of the content's semantic meaning.
    pub hdv_embedding: BinaryHV,
    /// Content domain (e.g., "water", "governance", "knowledge").
    pub domain: String,
    /// Creation timestamp (Unix seconds).
    pub created_at: u64,
}

/// A ranked content item with resonance score.
#[derive(Debug, Clone)]
pub struct RankedContent {
    /// The content reference.
    pub content: ContentRef,
    /// Raw resonance score (Hamming similarity).
    pub resonance: f64,
    /// Diversity bonus applied.
    pub diversity_bonus: f64,
    /// Final composite score.
    pub score: f64,
}

/// Resonance graph telemetry.
#[derive(Debug, Clone, Default)]
pub struct SocialFabricTelemetry {
    /// Mean resonance across recent content.
    pub resonance_mean: f64,
    /// Diversity metric (0=echo chamber, 1=maximally diverse).
    pub diversity: f64,
    /// Echo chamber risk score (0-1, >0.85 triggers warning).
    pub echo_chamber_risk: f64,
    /// Number of unique peers with content.
    pub peer_reach: usize,
    /// Total content items tracked.
    pub content_count: usize,
}

/// Peer resonance state.
#[derive(Debug, Clone)]
struct PeerResonance {
    /// Recent content from this peer.
    content: VecDeque<ContentRef>,
    /// Base resonance with this peer (EMA of Hamming similarities).
    resonance_ema: f64,
    /// Last interaction timestamp.
    last_seen: u64,
}

/// Resonance-based content routing graph.
pub struct ResonanceGraph {
    /// Per-peer resonance tracking.
    peers: HashMap<String, PeerResonance>,
    /// Our consciousness state embedding (for resonance computation).
    our_state: Option<BinaryHV>,
    /// Recent content interactions for echo chamber detection.
    recent_sources: VecDeque<String>,
    /// Configuration.
    max_recent_sources: usize,
}

impl ResonanceGraph {
    /// Create a new resonance graph.
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
            our_state: None,
            recent_sources: VecDeque::new(),
            max_recent_sources: 100,
        }
    }

    /// Update our consciousness state embedding.
    pub fn set_our_state(&mut self, state: BinaryHV) {
        self.our_state = Some(state);
    }

    /// Add content from a peer.
    pub fn add_content(&mut self, content: ContentRef) {
        if self.peers.len() >= MAX_RESONANCE_PEERS && !self.peers.contains_key(&content.source_peer)
        {
            // Evict least-recently-seen peer
            if let Some(lru_key) = self
                .peers
                .iter()
                .min_by_key(|(_, v)| v.last_seen)
                .map(|(k, _)| k.clone())
            {
                self.peers.remove(&lru_key);
            }
        }

        let peer = self
            .peers
            .entry(content.source_peer.clone())
            .or_insert_with(|| PeerResonance {
                content: VecDeque::new(),
                resonance_ema: 0.5,
                last_seen: 0,
            });

        peer.last_seen = content.created_at;

        // Update resonance EMA if we have a state
        if let Some(ref our_state) = self.our_state {
            let similarity = hamming_similarity(our_state, &content.hdv_embedding);
            peer.resonance_ema = peer.resonance_ema * 0.9 + similarity * 0.1;
        }

        // Cap content per peer
        if peer.content.len() >= MAX_CONTENT_PER_PEER {
            peer.content.pop_front();
        }
        peer.content.push_back(content);
    }

    /// Rank content by resonance with diversity bonus.
    ///
    /// This is the core ranking algorithm that replaces engagement-based sorting.
    pub fn rank_content(&self, items: &[ContentRef], limit: usize) -> Vec<RankedContent> {
        let our_state = match &self.our_state {
            Some(s) => s,
            None => {
                // Without state, return items in original order
                return items
                    .iter()
                    .take(limit)
                    .map(|c| RankedContent {
                        content: c.clone(),
                        resonance: 0.5,
                        diversity_bonus: 0.0,
                        score: 0.5,
                    })
                    .collect();
            }
        };

        let mut ranked: Vec<RankedContent> = items
            .iter()
            .map(|c| {
                let resonance = hamming_similarity(our_state, &c.hdv_embedding);
                let div_bonus = diversity_bonus(resonance);
                let score = resonance + div_bonus;
                RankedContent {
                    content: c.clone(),
                    resonance,
                    diversity_bonus: div_bonus,
                    score,
                }
            })
            .collect();

        ranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.truncate(limit);
        ranked
    }

    /// Detect echo chamber risk.
    ///
    /// Returns fraction of recent content from high-resonance peers (>0.7).
    pub fn echo_chamber_risk(&self) -> f64 {
        if self.recent_sources.is_empty() {
            return 0.0;
        }
        let high_resonance_count = self
            .recent_sources
            .iter()
            .filter(|source| {
                self.peers
                    .get(*source)
                    .map(|p| p.resonance_ema > 0.7)
                    .unwrap_or(false)
            })
            .count();
        high_resonance_count as f64 / self.recent_sources.len() as f64
    }

    /// Record that content from a source was consumed (for echo chamber tracking).
    pub fn record_consumption(&mut self, source_peer: &str) {
        if self.recent_sources.len() >= self.max_recent_sources {
            self.recent_sources.pop_front();
        }
        self.recent_sources.push_back(source_peer.to_string());
    }

    /// Mean resonance across all tracked peers.
    pub fn mean_resonance(&self) -> f64 {
        if self.peers.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.peers.values().map(|p| p.resonance_ema).sum();
        sum / self.peers.len() as f64
    }

    /// Diversity metric: variance of resonance values (higher = more diverse).
    pub fn diversity(&self) -> f64 {
        if self.peers.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_resonance();
        let variance: f64 = self
            .peers
            .values()
            .map(|p| (p.resonance_ema - mean).powi(2))
            .sum::<f64>()
            / self.peers.len() as f64;
        // Normalize: max variance for [0,1] values is 0.25
        (variance / 0.25).clamp(0.0, 1.0)
    }

    /// Number of unique peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Total content items tracked.
    pub fn content_count(&self) -> usize {
        self.peers.values().map(|p| p.content.len()).sum()
    }

    /// Generate telemetry snapshot.
    pub fn telemetry(&self) -> SocialFabricTelemetry {
        SocialFabricTelemetry {
            resonance_mean: self.mean_resonance(),
            diversity: self.diversity(),
            echo_chamber_risk: self.echo_chamber_risk(),
            peer_reach: self.peer_count(),
            content_count: self.content_count(),
        }
    }
}

impl Default for ResonanceGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute Hamming similarity between two BinaryHVs.
///
/// Returns a value in [0.0, 1.0] where 1.0 = identical.
fn hamming_similarity(a: &BinaryHV, b: &BinaryHV) -> f64 {
    let dist = a.hamming_distance(b);
    1.0 - (dist as f64 / BinaryHV::DIM as f64)
}

/// Anti-echo-chamber diversity bonus.
///
/// Peaks at 0.3-0.7 Hamming distance (moderate difference = most growth).
/// Returns 0.0 for very similar or very different content.
fn diversity_bonus(resonance: f64) -> f64 {
    let distance = 1.0 - resonance;
    if distance >= DIVERSITY_PEAK_LOW && distance <= DIVERSITY_PEAK_HIGH {
        DIVERSITY_MAX_BONUS
    } else if distance < DIVERSITY_PEAK_LOW {
        DIVERSITY_MAX_BONUS * (distance / DIVERSITY_PEAK_LOW)
    } else {
        DIVERSITY_MAX_BONUS * ((1.0 - distance) / (1.0 - DIVERSITY_PEAK_HIGH))
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_content(peer: &str, domain: &str) -> ContentRef {
        ContentRef {
            source_peer: peer.to_string(),
            content_hash: [0; 32],
            hdv_embedding: BinaryHV::random(rand::random::<u64>()),
            domain: domain.to_string(),
            created_at: 1000,
        }
    }

    #[test]
    fn test_empty_graph() {
        let g = ResonanceGraph::new();
        assert_eq!(g.peer_count(), 0);
        assert_eq!(g.content_count(), 0);
    }

    #[test]
    fn test_add_content() {
        let mut g = ResonanceGraph::new();
        g.add_content(make_content("peer1", "water"));
        assert_eq!(g.peer_count(), 1);
        assert_eq!(g.content_count(), 1);
    }

    #[test]
    fn test_resonance_ranking() {
        let mut g = ResonanceGraph::new();
        g.set_our_state(BinaryHV::random(42));
        let items: Vec<ContentRef> = (0..10)
            .map(|i| make_content(&format!("peer_{}", i), "test"))
            .collect();
        let ranked = g.rank_content(&items, 5);
        assert_eq!(ranked.len(), 5);
        // Scores should be monotonically non-increasing
        for w in ranked.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_diversity_bonus_peak() {
        // At peak distance (0.5 resonance = 0.5 distance), bonus should be max
        let bonus = diversity_bonus(0.5);
        assert!((bonus - DIVERSITY_MAX_BONUS).abs() < 0.01);
    }

    #[test]
    fn test_diversity_bonus_edges() {
        // Very similar content (resonance ~1.0) -> low bonus
        assert!(diversity_bonus(0.95) < DIVERSITY_MAX_BONUS * 0.5);
        // Very different content (resonance ~0.0) -> low bonus
        assert!(diversity_bonus(0.05) < DIVERSITY_MAX_BONUS * 0.5);
    }

    #[test]
    fn test_echo_chamber_detection() {
        let mut g = ResonanceGraph::new();
        // Add a high-resonance peer
        let peer = PeerResonance {
            content: VecDeque::new(),
            resonance_ema: 0.9, // High resonance
            last_seen: 1000,
        };
        g.peers.insert("echo_peer".to_string(), peer);
        // All recent content from this peer
        for _ in 0..10 {
            g.record_consumption("echo_peer");
        }
        assert!(g.echo_chamber_risk() > ECHO_CHAMBER_THRESHOLD);
    }

    #[test]
    fn test_mean_resonance() {
        let mut g = ResonanceGraph::new();
        g.peers.insert(
            "a".to_string(),
            PeerResonance {
                content: VecDeque::new(),
                resonance_ema: 0.8,
                last_seen: 0,
            },
        );
        g.peers.insert(
            "b".to_string(),
            PeerResonance {
                content: VecDeque::new(),
                resonance_ema: 0.4,
                last_seen: 0,
            },
        );
        assert!((g.mean_resonance() - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_telemetry() {
        let g = ResonanceGraph::new();
        let t = g.telemetry();
        assert_eq!(t.peer_reach, 0);
        assert_eq!(t.content_count, 0);
    }

    #[test]
    fn test_hamming_similarity_identical() {
        let hv = BinaryHV::random(99);
        assert!((hamming_similarity(&hv, &hv) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_max_peers_eviction() {
        let mut g = ResonanceGraph::new();
        for i in 0..MAX_RESONANCE_PEERS + 10 {
            g.add_content(ContentRef {
                source_peer: format!("peer_{}", i),
                content_hash: [0; 32],
                hdv_embedding: BinaryHV::random(i as u64),
                domain: "test".to_string(),
                created_at: i as u64,
            });
        }
        assert!(g.peer_count() <= MAX_RESONANCE_PEERS);
    }

    #[test]
    fn test_ranking_without_state() {
        let g = ResonanceGraph::new();
        let items = vec![make_content("p", "d")];
        let ranked = g.rank_content(&items, 10);
        assert_eq!(ranked.len(), 1);
        assert!((ranked[0].resonance - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_default() {
        let g = ResonanceGraph::default();
        assert_eq!(g.peer_count(), 0);
    }

    #[test]
    fn test_diversity_metric() {
        let mut g = ResonanceGraph::new();
        // All same resonance = 0 diversity
        for i in 0..5 {
            g.peers.insert(
                format!("p{}", i),
                PeerResonance {
                    content: VecDeque::new(),
                    resonance_ema: 0.5,
                    last_seen: 0,
                },
            );
        }
        assert!(g.diversity() < 0.01);

        // Max spread = high diversity
        g.peers.clear();
        g.peers.insert(
            "a".to_string(),
            PeerResonance {
                content: VecDeque::new(),
                resonance_ema: 0.0,
                last_seen: 0,
            },
        );
        g.peers.insert(
            "b".to_string(),
            PeerResonance {
                content: VecDeque::new(),
                resonance_ema: 1.0,
                last_seen: 0,
            },
        );
        assert!(g.diversity() > 0.5);
    }

    #[test]
    fn test_content_per_peer_cap() {
        let mut g = ResonanceGraph::new();
        for i in 0..MAX_CONTENT_PER_PEER + 10 {
            g.add_content(ContentRef {
                source_peer: "same_peer".to_string(),
                content_hash: [i as u8; 32],
                hdv_embedding: BinaryHV::random(i as u64),
                domain: "test".to_string(),
                created_at: i as u64,
            });
        }
        assert!(g.peers.get("same_peer").unwrap().content.len() <= MAX_CONTENT_PER_PEER);
    }
}
