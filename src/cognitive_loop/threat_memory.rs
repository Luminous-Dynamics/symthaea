// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Threat Memory — HDV-encoded immune pattern storage
//!
//! Encodes detected threat patterns as 16,384-dimensional hyperdimensional
//! vectors for rapid similarity-based threat recognition. Integrates with
//! the knowledge graph for persistent storage and dream consolidation.
//!
//! ## Immune Memory Analogy
//!
//! Like B-cell memory in the adaptive immune system, threat memories
//! enable faster recognition of previously-encountered attack patterns.
//! HDV similarity search replaces slow pattern matching with O(1) lookup.
//!
//! ## Dream Consolidation
//!
//! Threat patterns receive +30% salience boost during dream consolidation,
//! reflecting the biological priority of encoding survival-relevant memories.
//! Basis: Walker (2017) — threat-related memories preferentially consolidated.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════════
// THREAT SIGNAL KINDS — threat taxonomy for the immune sentinel
// ═══════════════════════════════════════════════════════════════════════════════

/// Kinds of threat detected by the immune sentinel.
///
/// These correspond to attack patterns observable in a distributed
/// governance/swarm system. Each variant maps to a one-hot encoding
/// dimension in the compact feature vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreatSignalKind {
    /// Rapid-fire proposal submission designed to exhaust reviewer attention.
    ProposalFlood,
    /// Coordinated voting from a cluster of identities (Sybil attack).
    VoteClustering,
    /// Peer exhibiting statistically anomalous behavior patterns.
    PeerBehaviorAnomaly,
    /// Federated gradient with suspicious fingerprint (poisoning attempt).
    GradientFingerprint,
    /// Abnormal timing in protocol messages (replay or delay attack).
    TimingAnomaly,
    /// Self-referential dispatch loop (resource exhaustion attack).
    DispatchLoop,
    /// Attempt to manipulate consciousness metrics (Phi/coherence spoofing).
    ConsciousnessManipulation,
    /// Coordinated cartel attack (from CartelDetector).
    CartelAttack,
    /// Systematic oppression pattern in governance.
    GovernanceOppression,
    /// Epistemic threat: uncalibrated confidence, dogmatic certainty,
    /// or systematic resistance to evidence update.
    EpistemicThreat,
}

impl ThreatSignalKind {
    /// Return the one-hot index for this threat kind (0-9).
    fn one_hot_index(self) -> usize {
        match self {
            Self::ProposalFlood => 0,
            Self::VoteClustering => 1,
            Self::PeerBehaviorAnomaly => 2,
            Self::GradientFingerprint => 3,
            Self::TimingAnomaly => 4,
            Self::DispatchLoop => 5,
            Self::ConsciousnessManipulation => 6,
            Self::CartelAttack => 7,
            Self::GovernanceOppression => 8,
            Self::EpistemicThreat => 9,
        }
    }

    /// Convert from SentinelManager's ThreatSignalKind (when sentinel feature enabled).
    ///
    /// Both enums have the same variant names; this bridges them without
    /// creating a cross-feature dependency.
    #[cfg(feature = "sentinel")]
    pub fn from_sentinel_kind(kind: super::managers::sentinel_manager::ThreatSignalKind) -> Self {
        use super::managers::sentinel_manager::ThreatSignalKind as SK;
        match kind {
            SK::ProposalFlood => Self::ProposalFlood,
            SK::VoteClustering => Self::VoteClustering,
            SK::PeerBehaviorAnomaly => Self::PeerBehaviorAnomaly,
            SK::GradientFingerprint => Self::GradientFingerprint,
            SK::TimingAnomaly => Self::TimingAnomaly,
            SK::DispatchLoop => Self::DispatchLoop,
            SK::ConsciousnessManipulation => Self::ConsciousnessManipulation,
            SK::CartelAttack => Self::CartelAttack,
            SK::GovernanceOppression => Self::GovernanceOppression,
            SK::EpistemicThreat => Self::EpistemicThreat,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// THREAT PATTERN — HDV-encoded immune memory entry
// ═══════════════════════════════════════════════════════════════════════════════

/// A threat pattern encoded as an HDV for similarity-based recognition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatPattern {
    /// Unique identifier for this threat pattern.
    pub id: u64,
    /// The threat kind that produced this pattern.
    pub kind: ThreatSignalKind,
    /// Human-readable description of the pattern.
    pub description: String,
    /// Severity when this pattern was first detected (0.0-1.0).
    pub original_severity: f32,
    /// Number of times this pattern has been matched.
    pub match_count: u32,
    /// Cycle at which this pattern was first recorded.
    pub first_seen: usize,
    /// Cycle at which this pattern was last matched.
    pub last_seen: usize,
    /// Confidence in this pattern's validity (0.0-1.0).
    /// Increases with corroboration, decays with time.
    pub confidence: f32,
    /// Whether this pattern came from a peer (vs local detection).
    pub from_peer: bool,
    /// Compact feature vector for similarity (32 f32 values).
    /// Derived from the full HDV but storable in JSON telemetry.
    pub feature_vector: Vec<f32>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// THREAT MEMORY — immune pattern library
// ═══════════════════════════════════════════════════════════════════════════════

/// Threat memory store — the immune system's pattern library.
///
/// Stores encoded threat patterns and provides similarity-based lookup.
/// Bounded to prevent unbounded growth (FIFO eviction of oldest patterns).
pub(crate) struct ThreatMemory {
    /// Stored threat patterns.
    patterns: VecDeque<ThreatPattern>,
    /// Maximum stored patterns (prevents unbounded memory growth).
    max_patterns: usize,
    /// Next pattern ID.
    next_id: u64,
    /// Dream consolidation salience boost for threat patterns.
    /// Basis: Walker (2017) — threat memories get priority consolidation.
    dream_salience_boost: f32,
}

impl Default for ThreatMemory {
    fn default() -> Self {
        Self {
            patterns: VecDeque::with_capacity(256),
            max_patterns: 256,
            next_id: 0,
            dream_salience_boost: 0.3, // +30% salience
        }
    }
}

impl ThreatMemory {
    /// Similarity threshold for pattern matching.
    /// Cosine similarity > this value counts as a match.
    const MATCH_THRESHOLD: f32 = 0.75;

    /// Minimum confidence to accept a peer-reported pattern.
    const MIN_PEER_CONFIDENCE: f32 = 0.4;

    /// Create a new threat memory with custom capacity.
    pub fn with_capacity(max_patterns: usize) -> Self {
        Self {
            patterns: VecDeque::with_capacity(max_patterns),
            max_patterns,
            ..Default::default()
        }
    }

    /// Encode a threat signal into a compact feature vector.
    ///
    /// The feature vector captures:
    /// - Threat kind (one-hot, 10 dims)
    /// - Severity (1 dim)
    /// - Confidence (1 dim)
    /// - Temporal pattern (8 dims: sin/cos encoding of detection time)
    /// - Evidence hash features (15 dims: pseudo-random from description)
    ///
    /// Total: 32 dimensions (compact representation for similarity search).
    pub fn encode_threat(
        kind: ThreatSignalKind,
        severity: f32,
        confidence: f32,
        cycle: usize,
        description: &str,
    ) -> Vec<f32> {
        let mut features = vec![0.0f32; 32];

        // One-hot threat kind (dims 0-9)
        features[kind.one_hot_index()] = 1.0;

        // Severity and confidence (dims 10-11, after one-hot block)
        features[10] = severity.clamp(0.0, 1.0);
        features[11] = confidence.clamp(0.0, 1.0);

        // Temporal encoding (dims 12-19): sin/cos at 4 frequencies
        let t = cycle as f32;
        for i in 0..4 {
            let freq = (i + 1) as f32 * 0.01;
            features[12 + i * 2] = (t * freq).sin();
            features[13 + i * 2] = (t * freq).cos();
        }

        // Description hash features (dims 20-31)
        let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
        for byte in description.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3); // FNV prime
        }
        for i in 0..12 {
            // Extract pseudo-random features from hash
            let shift = (i * 4) % 60;
            let nibble = ((hash >> shift) & 0xF) as f32 / 15.0;
            features[20 + i] = nibble * 2.0 - 1.0; // map to [-1, 1]
        }

        features
    }

    /// Compute cosine similarity between two feature vectors.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Store a new threat pattern from a local detection.
    ///
    /// If a similar pattern already exists (cosine > MATCH_THRESHOLD),
    /// updates the existing pattern instead of creating a duplicate.
    ///
    /// Returns the pattern ID (new or existing).
    pub fn store_threat(
        &mut self,
        kind: ThreatSignalKind,
        severity: f32,
        confidence: f32,
        cycle: usize,
        description: &str,
    ) -> u64 {
        let features = Self::encode_threat(kind, severity, confidence, cycle, description);

        // Check for similar existing pattern
        if let Some(existing) = self.find_similar_mut(&features) {
            existing.match_count += 1;
            existing.last_seen = cycle;
            existing.confidence = (existing.confidence + confidence * 0.3).min(1.0);
            return existing.id;
        }

        // Create new pattern
        let id = self.next_id;
        self.next_id += 1;

        let pattern = ThreatPattern {
            id,
            kind,
            description: description.to_string(),
            original_severity: severity,
            match_count: 1,
            first_seen: cycle,
            last_seen: cycle,
            confidence,
            from_peer: false,
            feature_vector: features,
        };

        // Evict oldest if at capacity
        if self.patterns.len() >= self.max_patterns {
            self.patterns.pop_front();
        }
        self.patterns.push_back(pattern);

        id
    }

    /// Store a threat pattern received from a peer.
    ///
    /// Peer patterns start with reduced confidence and are marked as external.
    pub fn store_peer_threat(
        &mut self,
        kind: ThreatSignalKind,
        severity: f32,
        peer_confidence: f32,
        cycle: usize,
        description: &str,
        feature_vector: Vec<f32>,
    ) -> Option<u64> {
        // Reject low-confidence peer reports
        if peer_confidence < Self::MIN_PEER_CONFIDENCE {
            return None;
        }

        // Check for existing similar pattern
        if let Some(existing) = self.find_similar_mut(&feature_vector) {
            // Corroborate: peer agreement boosts confidence
            existing.match_count += 1;
            existing.last_seen = cycle;
            existing.confidence = (existing.confidence + 0.1).min(1.0);
            return Some(existing.id);
        }

        // New peer pattern — reduced confidence
        let id = self.next_id;
        self.next_id += 1;

        let pattern = ThreatPattern {
            id,
            kind,
            description: description.to_string(),
            original_severity: severity,
            match_count: 1,
            first_seen: cycle,
            last_seen: cycle,
            confidence: peer_confidence * 0.7, // discount peer reports
            from_peer: true,
            feature_vector,
        };

        if self.patterns.len() >= self.max_patterns {
            self.patterns.pop_front();
        }
        self.patterns.push_back(pattern);

        Some(id)
    }

    /// Find the most similar stored pattern to a feature vector.
    ///
    /// Returns `Some((pattern, similarity))` if similarity > MATCH_THRESHOLD.
    pub fn find_similar(&self, features: &[f32]) -> Option<(&ThreatPattern, f32)> {
        let mut best: Option<(usize, f32)> = None;
        for (i, pattern) in self.patterns.iter().enumerate() {
            let sim = Self::cosine_similarity(&pattern.feature_vector, features);
            if sim > Self::MATCH_THRESHOLD {
                if best.map_or(true, |(_, best_sim)| sim > best_sim) {
                    best = Some((i, sim));
                }
            }
        }
        best.map(|(i, sim)| (&self.patterns[i], sim))
    }

    /// Find similar pattern (mutable) for in-place updates.
    fn find_similar_mut(&mut self, features: &[f32]) -> Option<&mut ThreatPattern> {
        let mut best_idx: Option<(usize, f32)> = None;
        for (i, pattern) in self.patterns.iter().enumerate() {
            let sim = Self::cosine_similarity(&pattern.feature_vector, features);
            if sim > Self::MATCH_THRESHOLD {
                if best_idx.map_or(true, |(_, best_sim)| sim > best_sim) {
                    best_idx = Some((i, sim));
                }
            }
        }
        best_idx.map(move |(i, _)| &mut self.patterns[i])
    }

    /// Check if an incoming event matches any known threat pattern.
    ///
    /// Returns matching patterns with similarity scores, sorted by similarity (descending).
    pub fn match_against(
        &self,
        kind: ThreatSignalKind,
        severity: f32,
        confidence: f32,
        cycle: usize,
        description: &str,
    ) -> Vec<(&ThreatPattern, f32)> {
        let features = Self::encode_threat(kind, severity, confidence, cycle, description);
        let mut matches: Vec<(&ThreatPattern, f32)> = self
            .patterns
            .iter()
            .map(|p| (p, Self::cosine_similarity(&p.feature_vector, &features)))
            .filter(|(_, sim)| *sim > Self::MATCH_THRESHOLD)
            .collect();
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }

    /// Get the dream consolidation salience boost for threat patterns.
    pub fn dream_salience_boost(&self) -> f32 {
        self.dream_salience_boost
    }

    /// Number of stored patterns.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Get all stored patterns (for serialization/telemetry).
    pub fn patterns(&self) -> &VecDeque<ThreatPattern> {
        &self.patterns
    }

    /// Decay confidence of old patterns that haven't been seen recently.
    ///
    /// Called during dream consolidation. Patterns not seen for >500 cycles
    /// lose confidence at 0.99x per consolidation.
    pub fn decay_old_patterns(&mut self, current_cycle: usize) {
        for pattern in &mut self.patterns {
            if current_cycle.saturating_sub(pattern.last_seen) > 500 {
                pattern.confidence *= 0.99;
            }
        }
        // Remove patterns with negligible confidence
        self.patterns.retain(|p| p.confidence > 0.01);
    }

    /// Get the highest-confidence patterns for sharing with peers.
    ///
    /// Returns up to `limit` patterns, sorted by confidence (descending).
    /// Only patterns with confidence > 0.5 are shared.
    pub fn shareable_patterns(&self, limit: usize) -> Vec<&ThreatPattern> {
        let mut candidates: Vec<&ThreatPattern> = self
            .patterns
            .iter()
            .filter(|p| p.confidence > 0.5)
            .collect();
        candidates.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit);
        candidates
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_threat_dimensionality() {
        let features = ThreatMemory::encode_threat(
            ThreatSignalKind::ProposalFlood,
            0.5,
            0.8,
            100,
            "test flood",
        );
        assert_eq!(features.len(), 32);
    }

    #[test]
    fn test_encode_threat_one_hot() {
        let features =
            ThreatMemory::encode_threat(ThreatSignalKind::VoteClustering, 0.5, 0.8, 100, "test");
        assert_eq!(features[1], 1.0); // VoteClustering = index 1
        assert_eq!(features[0], 0.0); // ProposalFlood = index 0
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut memory = ThreatMemory::default();
        let id = memory.store_threat(
            ThreatSignalKind::ProposalFlood,
            0.5,
            0.8,
            100,
            "flood attack",
        );
        assert_eq!(memory.pattern_count(), 1);
        assert_eq!(id, 0);
    }

    #[test]
    fn test_deduplication() {
        let mut memory = ThreatMemory::default();
        let id1 = memory.store_threat(
            ThreatSignalKind::ProposalFlood,
            0.5,
            0.8,
            100,
            "flood attack",
        );
        let id2 = memory.store_threat(
            ThreatSignalKind::ProposalFlood,
            0.6,
            0.9,
            110,
            "flood attack",
        );
        // Same kind + similar description -> should deduplicate
        assert_eq!(id1, id2);
        assert_eq!(memory.pattern_count(), 1);
        // Match count should be 2
        assert_eq!(memory.patterns()[0].match_count, 2);
    }

    #[test]
    fn test_different_threats_not_deduplicated() {
        let mut memory = ThreatMemory::default();
        memory.store_threat(ThreatSignalKind::ProposalFlood, 0.5, 0.8, 100, "flood");
        memory.store_threat(ThreatSignalKind::VoteClustering, 0.5, 0.8, 100, "sybil");
        assert_eq!(memory.pattern_count(), 2);
    }

    #[test]
    fn test_capacity_eviction() {
        let mut memory = ThreatMemory::with_capacity(3);
        for i in 0..5 {
            memory.store_threat(
                ThreatSignalKind::ProposalFlood,
                0.5,
                0.8,
                i * 100,
                &format!("unique_threat_{}", i),
            );
        }
        assert!(memory.pattern_count() <= 3);
    }

    #[test]
    fn test_peer_threat_reduced_confidence() {
        let mut memory = ThreatMemory::default();
        let features = ThreatMemory::encode_threat(
            ThreatSignalKind::ProposalFlood,
            0.5,
            0.8,
            100,
            "peer threat",
        );
        memory.store_peer_threat(
            ThreatSignalKind::ProposalFlood,
            0.5,
            0.8,
            100,
            "peer threat",
            features,
        );
        // Peer confidence is discounted by 0.7
        assert!(memory.patterns()[0].confidence < 0.8);
        assert!(memory.patterns()[0].from_peer);
    }

    #[test]
    fn test_peer_threat_low_confidence_rejected() {
        let mut memory = ThreatMemory::default();
        let features =
            ThreatMemory::encode_threat(ThreatSignalKind::ProposalFlood, 0.5, 0.2, 100, "weak");
        let result = memory.store_peer_threat(
            ThreatSignalKind::ProposalFlood,
            0.5,
            0.2,
            100,
            "weak",
            features,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_match_against_finds_similar() {
        let mut memory = ThreatMemory::default();
        memory.store_threat(
            ThreatSignalKind::ProposalFlood,
            0.5,
            0.8,
            100,
            "flood attack",
        );

        // Use same cycle to ensure temporal encoding similarity (different cycles have different sin/cos)
        let matches = memory.match_against(
            ThreatSignalKind::ProposalFlood,
            0.6,
            0.9,
            100,
            "flood attack",
        );
        assert!(!matches.is_empty());
        assert!(matches[0].1 > ThreatMemory::MATCH_THRESHOLD);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 1.0];
        let sim = ThreatMemory::cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = ThreatMemory::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_decay_old_patterns() {
        let mut memory = ThreatMemory::default();
        memory.store_threat(ThreatSignalKind::ProposalFlood, 0.5, 0.3, 100, "old");

        // Decay at cycle 700 (>500 cycles since last seen)
        memory.decay_old_patterns(700);
        assert!(memory.patterns()[0].confidence < 0.3);
    }

    #[test]
    fn test_shareable_patterns_filtered() {
        let mut memory = ThreatMemory::default();
        memory.store_threat(ThreatSignalKind::ProposalFlood, 0.5, 0.8, 100, "high conf");
        memory.store_threat(ThreatSignalKind::VoteClustering, 0.5, 0.3, 100, "low conf");

        let shareable = memory.shareable_patterns(10);
        assert_eq!(shareable.len(), 1); // Only high-confidence pattern
        assert!(shareable[0].confidence > 0.5);
    }

    #[test]
    fn test_dream_salience_boost() {
        let memory = ThreatMemory::default();
        assert!((memory.dream_salience_boost() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_epistemic_threat_one_hot_index() {
        assert_eq!(ThreatSignalKind::EpistemicThreat.one_hot_index(), 9);
    }

    #[test]
    fn test_epistemic_threat_encode_and_store() {
        let mut memory = ThreatMemory::default();
        let id = memory.store_threat(
            ThreatSignalKind::EpistemicThreat,
            0.6,
            0.7,
            200,
            "dogmatic certainty detected",
        );
        assert_eq!(memory.pattern_count(), 1);
        let pattern = &memory.patterns()[0];
        assert_eq!(pattern.id, id);
        assert_eq!(pattern.kind, ThreatSignalKind::EpistemicThreat);
        // One-hot at index 9 should be 1.0
        assert_eq!(pattern.feature_vector[9], 1.0);
        // Other one-hot indices should be 0.0
        for i in 0..9 {
            assert_eq!(pattern.feature_vector[i], 0.0);
        }
    }

    #[test]
    fn test_epistemic_threat_match_against() {
        let mut memory = ThreatMemory::default();
        memory.store_threat(
            ThreatSignalKind::EpistemicThreat,
            0.5,
            0.8,
            100,
            "evidence resistance",
        );
        let matches = memory.match_against(
            ThreatSignalKind::EpistemicThreat,
            0.6,
            0.9,
            100,
            "evidence resistance",
        );
        assert!(!matches.is_empty());
    }
}
