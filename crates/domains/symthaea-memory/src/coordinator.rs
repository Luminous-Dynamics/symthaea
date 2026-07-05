// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Memory Coordinator: Cross-Tier Memory Integration
//!
//! Bridges episodic and semantic memory with shared consciousness signals
//! (Phi, coherence, retrieval frequency) so each tier can inform the others.
//!
//! The coordinator also accepts graduation events from working memory
//! (which lives in a separate crate) and processes them into episodic storage.
//!
//! ## Cross-Tier Signals
//!
//! ```text
//! ┌─────────────┐      ┌──────────────────┐      ┌───────────────┐
//! │   Working    │─grad─►  Memory           │─sig──►  Episodic     │
//! │   Memory     │      │  Coordinator      │      │  Memory       │
//! └─────────────┘      │                    │      └───────────────┘
//!                      │  Φ, coherence,     │
//!                      │  retrieval freq    │      ┌───────────────┐
//!                      │                    │─sig──►  Semantic     │
//!                      └──────────────────┘      │  Memory       │
//!                                                  └───────────────┘
//! ```

use crate::episodic_replay::{Episode, EpisodicMemory};
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::ContinuousHV;

// ═══════════════════════════════════════════════════════════════════════════════
// SHARED SIGNALS
// ═══════════════════════════════════════════════════════════════════════════════

/// Shared signal state across all memory tiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySignals {
    /// Psi — Current consciousness estimate from consciousness processing
    pub psi: f64,
    /// Current coherence level (from coherence tracker)
    pub coherence: f64,
    /// Σ (Sigma) — Synergistic integration (Layer 2, computed every N cycles).
    /// When available, provides higher-fidelity consciousness assessment than Psi.
    pub sigma: Option<f64>,
    /// Global retrieval frequency (exponential moving average)
    pub retrieval_rate: f64,
    /// Step counter for temporal coordination
    pub step: u64,
}

impl Default for MemorySignals {
    fn default() -> Self {
        Self {
            psi: 0.0,
            coherence: 0.0,
            sigma: None,
            retrieval_rate: 0.0,
            step: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GRADUATION EVENTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Event from working memory signaling an item is ready for consolidation.
///
/// When a working memory item has persisted through enough decay cycles
/// and is being evicted, it can be "graduated" to episodic memory.
#[derive(Debug, Clone)]
pub struct GraduationEvent {
    /// HDC content vector of the item
    pub content: ContinuousHV,
    /// Label/description of the item
    pub label: String,
    /// How many decay steps the item survived in working memory
    pub steps_survived: u64,
    /// Final activation level when evicted/graduated
    pub final_activation: f64,
    /// Psi at the time of graduation
    pub psi_at_graduation: f64,
    /// Coherence at the time of graduation
    pub coherence_at_graduation: f64,
    /// Source of the memory
    pub source: MemorySource,
    /// Whether the information has been verified (e.g., via NIX_BUILD or EpistemicVerifier)
    pub is_verified: bool,
}

/// Source of a memory item
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemorySource {
    /// Internal cognitive process
    #[default]
    Internal,
    /// Autonomous web research
    WebResearch,
    /// Direct user interaction
    UserInteraction,
    /// Action outcome (verification signal)
    ActionFeedback,
    /// Evicted from semantic ring buffer (survived full rotation)
    SemanticEviction,
    /// Received via social relay from a peer agent
    Social,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the memory coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    /// Minimum steps in working memory before graduation to episodic
    pub min_wm_steps_for_graduation: u64,
    /// How much retrieval boosts consolidation strength (log-scaled)
    pub retrieval_consolidation_boost: f64,
    /// Weight of semantic retrieval frequency in enriched episodic priority
    pub semantic_frequency_weight: f64,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            min_wm_steps_for_graduation: 3,
            retrieval_consolidation_boost: 0.1,
            semantic_frequency_weight: 0.15,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COORDINATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Coordinates across memory tiers with shared Phi/coherence signals.
///
/// Provides:
/// - Shared signal broadcast (Phi, coherence, retrieval rate)
/// - Graduation processing (working memory → episodic)
/// - Enriched priority scoring (base priority + cross-tier signals)
/// - Retrieval tracking for reconsolidation
pub struct MemoryCoordinator {
    /// Shared signal state
    signals: MemorySignals,
    /// Configuration
    config: CoordinatorConfig,
    /// Per-memory retrieval counts (keyed by content hash)
    retrieval_counts: std::collections::HashMap<u64, u32>,
    /// Pending graduation events from working memory
    graduation_queue: Vec<GraduationEvent>,
    /// Statistics
    pub stats: CoordinatorStats,
}

/// Statistics for the coordinator
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoordinatorStats {
    /// Number of graduation events successfully processed
    pub graduations_processed: u64,
    /// Number of graduation events rejected (too few steps)
    pub graduations_rejected: u64,
    /// Total cross-tier retrieval events recorded
    pub cross_tier_retrievals: u64,
    /// Total signal update calls
    pub signal_updates: u64,
}

impl MemoryCoordinator {
    /// Create a new memory coordinator with the given configuration
    pub fn new(config: CoordinatorConfig) -> Self {
        Self {
            signals: MemorySignals::default(),
            config,
            retrieval_counts: std::collections::HashMap::new(),
            graduation_queue: Vec::new(),
            stats: CoordinatorStats::default(),
        }
    }

    /// Update shared signals from consciousness processing.
    ///
    /// Call this each cognitive cycle with the current Phi and coherence values.
    pub fn update_signals(&mut self, phi: f64, coherence: f64) {
        self.update_signals_with_sigma(phi, coherence, None);
    }

    /// Update shared signals with optional Σ (Layer 2 synergistic integration).
    ///
    /// When sigma is `Some`, it modulates graduation quality bonuses.
    pub fn update_signals_with_sigma(&mut self, phi: f64, coherence: f64, sigma: Option<f64>) {
        self.signals.psi = phi;
        self.signals.coherence = coherence;
        self.signals.sigma = sigma;
        self.signals.step += 1;
        self.stats.signal_updates += 1;

        // Update retrieval rate (exponential moving average)
        let instantaneous = if self.signals.step > 1 {
            self.stats.cross_tier_retrievals as f64 / self.signals.step as f64
        } else {
            0.0
        };
        self.signals.retrieval_rate = self.signals.retrieval_rate * 0.95 + instantaneous * 0.05;
    }

    /// Get current shared signals (for other tiers to read)
    pub fn signals(&self) -> &MemorySignals {
        &self.signals
    }

    /// Queue a graduation event from working memory.
    ///
    /// Graduation events are processed lazily via `process_graduations()`.
    pub fn queue_graduation(&mut self, event: GraduationEvent) {
        self.graduation_queue.push(event);
    }

    /// Process pending graduations into episodic memory.
    ///
    /// Returns the number of episodes successfully stored.
    pub fn process_graduations(&mut self, episodic: &mut EpisodicMemory) -> usize {
        let events: Vec<GraduationEvent> = self.graduation_queue.drain(..).collect();
        let mut stored = 0;

        for event in events {
            // Only graduate if sufficient time in working memory
            if event.steps_survived < self.config.min_wm_steps_for_graduation {
                self.stats.graduations_rejected += 1;
                continue;
            }

            // Persistence in WM boosts effective phi.
            // Verified information gets a significant boost to its consolidation strength.
            let persistence_bonus = (event.steps_survived as f64 / 10.0).min(0.2);
            let sigma_bonus = self.signals.sigma.map_or(0.0, |s| (s * 0.15).min(0.1));

            let verification_bonus = if event.is_verified {
                match event.source {
                    MemorySource::ActionFeedback => 0.3, // "Proven" by reality
                    MemorySource::WebResearch => 0.2,    // "Verified" by multiple sources
                    _ => 0.1,
                }
            } else {
                0.0
            };

            let adjusted_phi =
                (event.psi_at_graduation + persistence_bonus + sigma_bonus + verification_bonus)
                    .min(1.0);

            // Create episode from graduation event
            let episode = Episode::with_metadata(
                event.content.clone(),
                event.content, // Self-referential output for graduated items
                adjusted_phi,
                self.signals.step,
                1.0 - event.final_activation as f32, // Low activation → high surprise
                if event.is_verified { 0.5 } else { 0.0 }, // Verified = positive valence
                event.coherence_at_graduation as f32,
            );

            if episodic.store_if_significant(episode) {
                stored += 1;
            }
            self.stats.graduations_processed += 1;
        }

        stored
    }

    /// The Art of Forgetting: Causal Pruning.
    ///
    /// Periodically cleanses the episodic memory of low-value noise.
    /// Threshold is modulated by current coherence and metabolic load.
    pub fn prune_memories(&mut self, episodic: &mut EpisodicMemory, metabolic_load: f32) -> usize {
        // Higher metabolic load (stress) increases the pruning threshold (survival of the fittest)
        // Base threshold (0.2) + stress adjustment
        let threshold = 0.2 + (metabolic_load as f64 * 0.2);
        episodic.prune(threshold)
    }

    /// Record a retrieval event for consolidation tracking.
    ///
    /// Call this whenever a memory is retrieved from any tier. The content_hash
    /// is a lightweight identifier for the memory (e.g., first 8 bytes of the HDC vector).
    pub fn record_retrieval(&mut self, content_hash: u64) {
        *self.retrieval_counts.entry(content_hash).or_insert(0) += 1;
        self.stats.cross_tier_retrievals += 1;
    }

    /// Get the retrieval count for a memory (by content hash)
    pub fn retrieval_count(&self, content_hash: u64) -> u32 {
        self.retrieval_counts
            .get(&content_hash)
            .copied()
            .unwrap_or(0)
    }

    /// Return the top-k most-retrieved memories, sorted descending by retrieval count.
    ///
    /// Returns a `Vec<(content_hash, retrieval_count)>` of at most `k` entries.
    /// Entries with zero retrievals are never included.
    ///
    /// Used by the dream consolidation phase to identify candidate memories for
    /// episodic→semantic promotion: memories retrieved ≥ N times across dream
    /// replays are considered "well-consolidated" and eligible for semantic encoding.
    pub fn most_replayed(&self, k: usize) -> Vec<(u64, u32)> {
        if k == 0 {
            return Vec::new();
        }
        let mut pairs: Vec<(u64, u32)> = self
            .retrieval_counts
            .iter()
            .filter(|&(_, &count)| count > 0)
            .map(|(&hash, &count)| (hash, count))
            .collect();
        // Sort descending by count (stable so ties preserve HashMap order)
        pairs.sort_unstable_by_key(|p| std::cmp::Reverse(p.1));
        pairs.truncate(k);
        pairs
    }

    /// Compute enriched episodic priority score incorporating cross-tier signals.
    ///
    /// Augments the base priority with:
    /// - Retrieval frequency bonus (frequently retrieved = more important)
    /// - Coherence bonus (episodes stored during high coherence are more valuable)
    pub fn enriched_priority(&self, base_priority: f64, content_hash: u64) -> f64 {
        let retrieval_freq = self.retrieval_count(content_hash) as f64;
        let retrieval_bonus = retrieval_freq.ln_1p() * self.config.semantic_frequency_weight;
        let coherence_bonus = self.signals.coherence * 0.1;

        base_priority + retrieval_bonus + coherence_bonus
    }

    /// Get consolidation boost for a memory based on its retrieval history.
    ///
    /// More frequently retrieved memories get a logarithmic boost, modeling
    /// biological reconsolidation where retrieval strengthens memory traces.
    pub fn consolidation_boost(&self, content_hash: u64) -> f64 {
        let count = self.retrieval_count(content_hash);
        if count == 0 {
            return 0.0;
        }
        (count as f64).ln_1p() * self.config.retrieval_consolidation_boost
    }

    /// Get current Phi for use by other memory tiers
    pub fn current_phi(&self) -> f64 {
        self.signals.psi
    }

    /// Get current coherence for use by other memory tiers
    pub fn current_coherence(&self) -> f64 {
        self.signals.coherence
    }

    /// Get current step
    pub fn current_step(&self) -> u64 {
        self.signals.step
    }
}

impl Default for MemoryCoordinator {
    fn default() -> Self {
        Self::new(CoordinatorConfig::default())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTENT HASHING UTILITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute a cheap content hash from a ContinuousHV for retrieval tracking.
///
/// Uses the first 8 dimensions of the vector as a fingerprint.
pub fn content_hash(hv: &ContinuousHV) -> u64 {
    let data = hv.as_slice();
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &val in data.iter().take(8) {
        let bits = val.to_bits();
        hash ^= bits as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let coord = MemoryCoordinator::default();
        assert_eq!(coord.signals().step, 0);
        assert!((coord.signals().psi - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_signal_updates() {
        let mut coord = MemoryCoordinator::default();
        coord.update_signals(0.7, 0.8);
        assert!((coord.signals().psi - 0.7).abs() < 1e-6);
        assert!((coord.signals().coherence - 0.8).abs() < 1e-6);
        assert_eq!(coord.signals().step, 1);
    }

    #[test]
    fn test_retrieval_tracking() {
        let mut coord = MemoryCoordinator::default();
        coord.record_retrieval(42);
        coord.record_retrieval(42);
        coord.record_retrieval(99);

        assert_eq!(coord.retrieval_count(42), 2);
        assert_eq!(coord.retrieval_count(99), 1);
        assert_eq!(coord.retrieval_count(0), 0);
        assert_eq!(coord.stats.cross_tier_retrievals, 3);
    }

    #[test]
    fn test_enriched_priority() {
        let mut coord = MemoryCoordinator::default();
        coord.update_signals(0.5, 0.8);

        // No retrievals: should add only coherence bonus
        let p1 = coord.enriched_priority(1.0, 42);
        assert!(p1 > 1.0, "Coherence bonus should increase priority");

        // With retrievals: should add retrieval bonus on top
        coord.record_retrieval(42);
        coord.record_retrieval(42);
        coord.record_retrieval(42);
        let p2 = coord.enriched_priority(1.0, 42);
        assert!(p2 > p1, "Retrieval bonus should stack");
    }

    #[test]
    fn test_consolidation_boost() {
        let mut coord = MemoryCoordinator::default();
        assert_eq!(coord.consolidation_boost(42), 0.0);

        coord.record_retrieval(42);
        let boost1 = coord.consolidation_boost(42);
        assert!(boost1 > 0.0);

        coord.record_retrieval(42);
        coord.record_retrieval(42);
        let boost3 = coord.consolidation_boost(42);
        assert!(boost3 > boost1, "More retrievals = more boost");
    }

    #[test]
    fn test_graduation_filtering() {
        let mut coord = MemoryCoordinator::new(CoordinatorConfig {
            min_wm_steps_for_graduation: 3,
            ..CoordinatorConfig::default()
        });
        coord.update_signals(0.5, 0.6);

        // Too few steps - should be rejected
        coord.queue_graduation(GraduationEvent {
            content: ContinuousHV::random(1024, 1),
            label: "short-lived".into(),
            steps_survived: 1,
            final_activation: 0.3,
            psi_at_graduation: 0.5,
            coherence_at_graduation: 0.6,
            source: MemorySource::Internal,
            is_verified: false,
        });

        // Enough steps - should be processed
        coord.queue_graduation(GraduationEvent {
            content: ContinuousHV::random(1024, 2),
            label: "persisted".into(),
            steps_survived: 5,
            final_activation: 0.4,
            psi_at_graduation: 0.6,
            coherence_at_graduation: 0.7,
            source: MemorySource::WebResearch,
            is_verified: true,
        });

        let config = crate::episodic_replay::EpisodicReplayConfig {
            psi_threshold: 0.1, // Low threshold so graduation works
            ..Default::default()
        };
        let mut episodic = EpisodicMemory::new(config);
        let stored = coord.process_graduations(&mut episodic);

        assert_eq!(coord.stats.graduations_rejected, 1);
        assert_eq!(coord.stats.graduations_processed, 1);
        assert_eq!(stored, 1);
    }

    #[test]
    fn test_content_hash_deterministic() {
        let hv = ContinuousHV::random(1024, 42);
        let h1 = content_hash(&hv);
        let h2 = content_hash(&hv);
        assert_eq!(h1, h2, "Same vector should produce same hash");
    }

    #[test]
    fn test_content_hash_different() {
        let hv1 = ContinuousHV::random(1024, 1);
        let hv2 = ContinuousHV::random(1024, 2);
        let h1 = content_hash(&hv1);
        let h2 = content_hash(&hv2);
        assert_ne!(h1, h2, "Different vectors should produce different hashes");
    }

    // ═══════════════════════════════════════════════════════════════════
    // Phase 6: Cross-Tier Coordinator Integration Tests
    // ═══════════════════════════════════════════════════════════════════

    fn make_graduation(seed: u64, steps: u64, psi: f64, verified: bool) -> GraduationEvent {
        GraduationEvent {
            content: ContinuousHV::random(256, seed),
            label: format!("grad-{seed}"),
            steps_survived: steps,
            final_activation: 0.4,
            psi_at_graduation: psi,
            coherence_at_graduation: 0.6,
            source: MemorySource::Internal,
            is_verified: verified,
        }
    }

    #[test]
    fn test_coordinator_store_and_retrieve_via_graduation() {
        let mut coord = MemoryCoordinator::default();
        coord.update_signals(0.7, 0.8);

        let config = crate::episodic_replay::EpisodicReplayConfig {
            psi_threshold: 0.1,
            ..Default::default()
        };
        let mut episodic = EpisodicMemory::new(config);

        // Queue and process a graduation
        coord.queue_graduation(make_graduation(42, 5, 0.6, false));
        let stored = coord.process_graduations(&mut episodic);
        assert_eq!(stored, 1, "Should store one episode");

        // Verify the episode is in episodic memory
        assert_eq!(episodic.len(), 1);
        let top = episodic.get_top_episodes(1);
        assert!(!top.is_empty());
        assert!(top[0].psi > 0.5, "Episode should have adjusted Phi");

        // Record retrieval and check enrichment
        let hash = content_hash(&top[0].input);
        coord.record_retrieval(hash);
        let base = top[0].priority_score(10, 0.2);
        let enriched = coord.enriched_priority(base, hash);
        assert!(
            enriched > base,
            "Enriched priority should exceed base: {enriched} vs {base}"
        );
    }

    #[test]
    fn test_coordinator_working_to_episodic_consolidation() {
        let mut coord = MemoryCoordinator::new(CoordinatorConfig {
            min_wm_steps_for_graduation: 3,
            ..CoordinatorConfig::default()
        });
        coord.update_signals(0.5, 0.7);

        let config = crate::episodic_replay::EpisodicReplayConfig {
            psi_threshold: 0.1,
            ..Default::default()
        };
        let mut episodic = EpisodicMemory::new(config);

        // Simulate multiple WM items graduating at different times
        coord.queue_graduation(make_graduation(1, 5, 0.4, false));
        coord.queue_graduation(make_graduation(2, 8, 0.7, true)); // Verified = higher adjusted Phi
        coord.queue_graduation(make_graduation(3, 2, 0.6, false)); // Below min steps

        let stored = coord.process_graduations(&mut episodic);
        assert_eq!(stored, 2, "Two of three should pass graduation filter");
        assert_eq!(coord.stats.graduations_rejected, 1);
        assert_eq!(coord.stats.graduations_processed, 2);
        assert_eq!(episodic.len(), 2);

        // Verified item should have higher Phi than unverified (due to verification bonus)
        let top = episodic.get_top_episodes(2);
        assert!(
            top[0].psi > top[1].psi,
            "Verified item should have higher adjusted Phi: {} vs {}",
            top[0].psi,
            top[1].psi
        );
    }

    #[test]
    fn test_coordinator_retrieval_boosts_consolidation() {
        let mut coord = MemoryCoordinator::default();
        coord.update_signals(0.6, 0.7);

        let hv = ContinuousHV::random(256, 99);
        let hash = content_hash(&hv);

        // No retrievals → no consolidation boost
        assert_eq!(coord.consolidation_boost(hash), 0.0);

        // Repeated retrievals → growing boost (logarithmic)
        for _ in 0..5 {
            coord.record_retrieval(hash);
        }
        let boost5 = coord.consolidation_boost(hash);
        assert!(boost5 > 0.0, "5 retrievals should give positive boost");

        for _ in 0..10 {
            coord.record_retrieval(hash);
        }
        let boost15 = coord.consolidation_boost(hash);
        assert!(
            boost15 > boost5,
            "15 retrievals should give more boost than 5: {boost15} vs {boost5}"
        );

        // Logarithmic growth: boost15 should be less than 3× boost5
        assert!(
            boost15 < boost5 * 3.0,
            "Boost should grow sublinearly: {boost15} vs {boost5}"
        );
    }

    #[test]
    fn test_coordinator_prune_lifecycle() {
        let mut coord = MemoryCoordinator::default();
        coord.update_signals(0.5, 0.6);

        let config = crate::episodic_replay::EpisodicReplayConfig {
            psi_threshold: 0.0,
            capacity: 200,
            ..Default::default()
        };
        let mut episodic = EpisodicMemory::new(config);

        // Store episodes directly with varying Phi at timestamp 0
        // (Episode::new gives consolidation_strength=1.0, no prediction_error/valence)
        for i in 0..8 {
            let psi = i as f64 * 0.1; // 0.0, 0.1, ..., 0.7
            let input = ContinuousHV::random(256, i + 100);
            let output = ContinuousHV::random(256, i + 200);
            let ep = Episode::new(input, output, psi, 0);
            episodic.store_if_significant(ep);
        }

        // Advance the clock far into the future so recency bonus decays to ~0
        // (store a high-timestamp episode to advance current_cycle)
        let future_ep = Episode::new(
            ContinuousHV::random(256, 999),
            ContinuousHV::random(256, 1000),
            0.9,
            100_000,
        );
        episodic.store_if_significant(future_ep);
        let initial_count = episodic.len();
        assert_eq!(initial_count, 9);

        // Prune with high metabolic load → threshold = 0.2 + 0.9*0.2 = 0.38
        // Episodes with psi*0.5 + consolidation(0.1) + replay_penalty(0.05) < 0.38
        // → psi < 0.46 should be pruned (those aged out with 0 recency bonus)
        let pruned = coord.prune_memories(&mut episodic, 0.9);
        assert!(
            pruned > 0,
            "High metabolic load should prune some low-quality episodes"
        );
        assert!(
            episodic.len() < initial_count,
            "Episodic count should decrease: {} should be < {}",
            episodic.len(),
            initial_count
        );

        // The recent high-psi episode (0.9) should survive
        let remaining = episodic.get_top_episodes(1);
        assert!(
            remaining[0].psi >= 0.7,
            "High-Phi episode should survive: {}",
            remaining[0].psi
        );
    }
}
