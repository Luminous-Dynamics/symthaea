// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cultural Memory — self-authored artistic history + imitation-of-self.
//!
//! **Requires**: `feature = "creative"` + `feature = "social-fabric"`.
//!
//! ## What this is (and is not)
//!
//! This is the first concrete slice of
//! `ART_CULTURE_REVIEW_AND_PLAN_2026-07-06.md` Phase 4 ("Culture layer").
//! The 2026-07-06 review found that Symthaea can create art but has **no
//! mechanism for art to be remembered-and-built-upon over time, let alone
//! shared across agents** ("Cultural transmission: Absent" in the
//! capability scorecard). This module builds the *local* half of that gap:
//!
//! - **Self-authored publishing**: every artifact `CreativeManager` produces
//!   is registered into this agent's own [`SocialFabricManager`] /
//!   [`ResonanceGraph`] via the exact same `SocialEvent::ContentReceived`
//!   path real peer content would take over the mesh — so the resonance /
//!   diversity / echo-chamber ranking math is exercised identically
//!   regardless of source.
//! - **Imitation-of-self**: before generating new visual art, the agent can
//!   query its own resonance graph for a high-ranked past self-authored
//!   piece and use it as the *basis* for the next generation (real
//!   structural mutation for visual art via `symthaea_atelier::iterate::
//!   mutate_scene`; seed-level perturbation only for music/poetry, since
//!   those engines don't expose a scene-graph-equivalent to mutate).
//! - **Retention / "artistic canon"**: a small persisted JSON file tracks
//!   the top-N highest-scoring self-authored artifacts across sessions —
//!   tradition formation groundwork: what persists is what scored well, not
//!   everything.
//!
//! **This is NOT yet cross-agent culture.** Two things are still required
//! for that, both deliberately *not* done in this pass:
//!
//! 1. **A live mesh-send call.** `ContentAnnounce::encode()`
//!    (`src/swarm/mesh/content_packet.rs`) has zero callers anywhere in the
//!    codebase — nothing has ever pushed a `ContentAnnounce` onto the
//!    outbound mesh queue. The wiring pattern already exists for other
//!    payload types — see `Mind::emit_affective()` at `src/mind/tick.rs:1209`,
//!    which builds a `WisdomPacket`, signs it (`self.sign_mesh_packet`), and
//!    pushes `crate::swarm::mesh::MeshOutbound { packet }` onto
//!    `self.mesh_outbox` — but an equivalent `emit_content_announce()` does
//!    not exist and this pass does not add one.
//! 2. **The mesh-authentication sign-off.** Mesh packets have no real
//!    peer-identity authentication today (flagged in
//!    `symthaea_improvement_plan_july2026`, cross-session memory note).
//!    Broadcasting self-authored art to real peers before that gap is
//!    closed (or explicitly accepted fail-open) is a product/security
//!    decision for the project owner, not something to do unilaterally
//!    here.
//!
//! Until both land, "culture" here means: one agent's own artistic history,
//! ranked and retained by the same math that will rank cross-agent content
//! once publishing goes live.
//!
//! ## Embedding caveat
//!
//! Artifacts don't have a native full-dimension semantic HDC embedding
//! available at the point `CreativeManager` produces them (the muse/atelier
//! engines are driven by `CognitiveSnapshot.thought_vector`, a small
//! `Vec<f32>`, not a `BinaryHV`). Rather than inventing new HDC math, this
//! module derives each artifact's `hdv_embedding` deterministically from its
//! **content hash**: `BinaryHV::random(seed_from(blake3(bytes)))`. This
//! makes the embedding a reproducible content-addressed fingerprint (same
//! bytes -> same vector, forever), which is enough for `ResonanceGraph` to
//! treat republished/identical content consistently — but it is **not** a
//! semantic embedding of the artifact's visual/musical qualities, so
//! Hamming-similarity between two *different* artifacts' fingerprints
//! carries no aesthetic meaning today (it's close to uniform noise). Actual
//! aesthetic quality is tracked separately and honestly via
//! `aesthetic_score` in the retained canon (see [`CanonEntry`]) — that is
//! the signal imitation should ultimately prefer; wiring a true semantic
//! artifact encoder is future work.

use std::collections::HashMap;
use std::path::PathBuf;

use symthaea_core::hdc::BinaryHV;

use super::managers::social_fabric_manager::{SocialEvent, SocialFabricManager};
use super::subsystem_trait::CognitiveSubsystem;
use crate::swarm::resonance_graph::{ContentRef, ResonanceGraph};

/// Domain tag for self-authored visual (atelier) artifacts.
pub const DOMAIN_VISUAL: &str = "art:visual";
/// Domain tag for self-authored musical (muse) artifacts.
pub const DOMAIN_MUSIC: &str = "art:music";
/// Domain tag for self-authored poetry (Broca creative_mode) artifacts.
pub const DOMAIN_POETRY: &str = "art:poetry";

/// Source-peer identity used for self-authored content.
///
/// Deliberately not a real peer ID — there is no live network send yet
/// (see module docs), so "self" documents honestly that this content has
/// never left the local agent.
pub const SELF_PEER: &str = "self";

/// Default path for the persisted lineage + canon file.
const CULTURAL_CANON_PATH: &str = ".claude/artistic_canon.json";

/// Top-N cutoff for the persisted "artistic canon" (Phase 4 item 2:
/// tradition/canon formation — retention dynamics over the shared store).
const CANON_SIZE: usize = 20;

/// Maximum lineage entries kept in memory / persisted (bounded so a
/// long-running session doesn't grow this file unboundedly). Oldest
/// entries (by insertion order) are evicted first once the cap is hit.
const MAX_LINEAGE_ENTRIES: usize = 500;

/// Seed + domain lineage for one published artifact.
///
/// `ContentRef` (the resonance-graph wire type) has no room for
/// generation parameters — it's deliberately a thin, transport-agnostic
/// content reference. This side-table is the honest way to remember
/// "what seed produced this hash" so imitation can retrieve a real
/// generation basis rather than just a hash.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LineageEntry {
    /// The generation seed used to produce this artifact.
    pub seed: u64,
    /// Domain tag (`DOMAIN_VISUAL` / `DOMAIN_MUSIC` / `DOMAIN_POETRY`).
    pub domain: String,
}

/// One retained entry in the artistic canon: a self-authored artifact that
/// scored well enough to be worth remembering across sessions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CanonEntry {
    /// BLAKE3 content hash, hex-encoded (JSON object keys/values must be
    /// strings; raw `[u8; 32]` doesn't round-trip through `serde_json` as a
    /// map key without a custom encoding).
    pub content_hash_hex: String,
    /// Domain tag.
    pub domain: String,
    /// Aesthetic score at time of publishing (0.0-1.0 composite).
    pub aesthetic_score: f32,
    /// Resonance against the agent's consciousness state at time of
    /// ranking (0.0-1.0; 0.5 if no state was set yet — see
    /// `ResonanceGraph::rank_content`).
    pub resonance: f64,
    /// Generation seed (duplicated from the lineage table so the canon
    /// file is self-contained even if lineage entries get evicted).
    pub seed: u64,
    /// Creation timestamp (Unix seconds).
    pub created_at: u64,
    /// How many times this artifact has been selected as an imitation basis
    /// by [`CulturalMemoryManager::best_seed_for_domain`] (Phase 4 item 2:
    /// tradition/canon formation — what gets *built upon*, not just what
    /// scored well once, is what should persist). `#[serde(default)]` so
    /// canon files written before this field existed still deserialize.
    #[serde(default)]
    pub reuse_count: u32,
}

impl CanonEntry {
    /// Ranking score combining raw aesthetic quality with how much this
    /// artifact has been built upon since. `aesthetic_score` alone would let
    /// a one-off high scorer permanently outrank something the agent keeps
    /// returning to and refining; `reuse_count` alone would let a mediocre
    /// but frequently-selected artifact entrench regardless of quality.
    /// `ln(1 + reuse_count)` gives a diminishing-returns bonus (the 10th
    /// reuse matters much less than the 1st) so raw quality still dominates
    /// for lightly-reused work, while genuine tradition — repeated
    /// imitation — visibly outranks a single good score over time.
    pub fn effective_score(&self) -> f32 {
        self.aesthetic_score + REINFORCEMENT_BONUS_WEIGHT * (1.0 + self.reuse_count as f32).ln()
    }
}

/// Weight on the reinforcement bonus in [`CanonEntry::effective_score`].
/// Modest by design: a handful of reuses (ln(1+3)*0.05 ≈ 0.07) should nudge
/// ranking, not override a large aesthetic-score gap; only sustained,
/// repeated imitation (ln(1+20)*0.05 ≈ 0.15) meaningfully outweighs a
/// single-digit-percent quality difference.
const REINFORCEMENT_BONUS_WEIGHT: f32 = 0.05;

/// On-disk shape of the cultural memory file: lineage map (all tracked
/// self-authored artifacts, bounded) + canon (top-N by aesthetic score).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct CulturalMemoryFile {
    /// hex(content_hash) -> LineageEntry
    lineage: HashMap<String, LineageEntry>,
    /// Insertion order of `lineage` keys, oldest first — used to bound
    /// memory/file size via FIFO eviction. Kept separate from the map
    /// itself since `HashMap` has no stable iteration order.
    lineage_order: Vec<String>,
    canon: Vec<CanonEntry>,
}

/// Manages this agent's self-authored artistic history: publishing into a
/// private `SocialFabricManager`/`ResonanceGraph`, a seed-lineage table for
/// imitation, and a persisted top-N "artistic canon".
pub struct CulturalMemoryManager {
    /// Private resonance graph + neuromod bridge for self-authored content.
    ///
    /// This is a **separate instance** from any `SocialFabricManager` the
    /// wider `CognitiveLoopService` may run for real mesh peer content —
    /// wiring the two together (so self-authored and peer content share one
    /// graph) is natural future work once live mesh-send exists, but would
    /// require threading a shared handle from `CognitiveLoopService` into
    /// `CreativeManager`, a larger structural change than this pass's scope.
    social: SocialFabricManager,
    /// content_hash -> (seed, domain), all tracked self-authored artifacts
    /// (bounded, FIFO-evicted).
    lineage: HashMap<[u8; 32], LineageEntry>,
    /// Insertion order for FIFO eviction.
    lineage_order: std::collections::VecDeque<[u8; 32]>,
    /// Top-N retained artifacts by aesthetic score.
    canon: Vec<CanonEntry>,
    /// Path used for persistence.
    path: PathBuf,
}

impl CulturalMemoryManager {
    /// Create a new manager, loading persisted lineage + canon from the
    /// default path (`.claude/artistic_canon.json`) if present.
    pub fn new() -> Self {
        Self::new_with_path(None)
    }

    /// Create a new manager with an explicit persistence path.
    pub fn new_with_path(path: Option<PathBuf>) -> Self {
        let path = path.unwrap_or_else(|| PathBuf::from(CULTURAL_CANON_PATH));
        let file = Self::load_file(&path);

        let mut lineage = HashMap::new();
        let mut lineage_order = std::collections::VecDeque::new();
        for hex in &file.lineage_order {
            if let Some(entry) = file.lineage.get(hex) {
                if let Some(hash) = hex_to_hash(hex) {
                    lineage.insert(hash, entry.clone());
                    lineage_order.push_back(hash);
                }
            }
        }

        Self {
            social: SocialFabricManager::new(true),
            lineage,
            lineage_order,
            canon: file.canon,
            path,
        }
    }

    fn load_file(path: &std::path::Path) -> CulturalMemoryFile {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Persist lineage + canon to disk. Silently no-ops on write failure
    /// (mirrors `AestheticMemory::save`'s convention elsewhere in the
    /// creative pipeline).
    pub fn save(&self) {
        let lineage: HashMap<String, LineageEntry> = self
            .lineage
            .iter()
            .map(|(hash, entry)| (hash_to_hex(hash), entry.clone()))
            .collect();
        let lineage_order: Vec<String> = self.lineage_order.iter().map(hash_to_hex).collect();
        let file = CulturalMemoryFile {
            lineage,
            lineage_order,
            canon: self.canon.clone(),
        };
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(&file) {
            let _ = std::fs::write(&self.path, json);
        }
    }

    /// Access the private resonance graph (read-only).
    pub fn graph(&self) -> &ResonanceGraph {
        self.social.graph()
    }

    /// Publish a self-authored artifact.
    ///
    /// - Hashes `bytes` (BLAKE3) to get a stable content identity.
    /// - Derives a deterministic (non-semantic — see module docs)
    ///   `hdv_embedding` from that hash.
    /// - Computes resonance against the agent's current consciousness state
    ///   (if any has been set on the graph) via the real
    ///   `ResonanceGraph::rank_content` ranking math.
    /// - Records seed lineage and updates the retained canon.
    /// - Injects `SocialEvent::ContentReceived` into the private
    ///   `SocialFabricManager` — the same event variant real peer content
    ///   arrives as (see `cycle_phase_dynamics/mod.rs`) — so the content
    ///   only actually lands in the graph once `tick_social` (or an
    ///   equivalent `process()` call) drains pending events, exactly like
    ///   the live receive path.
    ///
    /// Returns the `ContentRef` that was queued, mostly for test/telemetry
    /// convenience.
    pub fn publish(
        &mut self,
        bytes: &[u8],
        domain: &str,
        seed: u64,
        aesthetic_score: f32,
        created_at: u64,
    ) -> ContentRef {
        let hash = blake3::hash(bytes);
        let content_hash: [u8; 32] = *hash.as_bytes();
        let hdv_embedding = derive_embedding(&content_hash);

        let content_ref = ContentRef {
            source_peer: SELF_PEER.to_string(),
            content_hash,
            hdv_embedding,
            domain: domain.to_string(),
            created_at,
        };

        // Resonance against current state, via the real ranking math.
        // (Doesn't require the item to already be in the graph.)
        let resonance = self
            .social
            .graph()
            .rank_content(std::slice::from_ref(&content_ref), 1)
            .first()
            .map(|r| r.resonance)
            .unwrap_or(0.5);

        self.record_lineage(content_hash, seed, domain);
        self.update_canon(CanonEntry {
            content_hash_hex: hash_to_hex(&content_hash),
            domain: domain.to_string(),
            aesthetic_score,
            resonance,
            seed,
            created_at,
            reuse_count: 0,
        });

        self.social
            .inject_event(SocialEvent::ContentReceived(content_ref.clone()));

        content_ref
    }

    /// Drain pending social events into the resonance graph and run the
    /// usual `SocialFabricManager` neuromod telemetry step.
    ///
    /// This is the same `CognitiveSubsystem::process` call the wider
    /// cognitive loop would make each cycle; exposed here so tests (and,
    /// once wired, `CreativeManager`) can force a self-authored `publish`
    /// to actually land in the graph.
    pub fn tick_social(
        &mut self,
        snapshot: &super::subsystem_trait::CycleSnapshot,
    ) -> super::subsystem_trait::SubsystemOutput {
        self.social.process(snapshot)
    }

    fn record_lineage(&mut self, content_hash: [u8; 32], seed: u64, domain: &str) {
        if !self.lineage.contains_key(&content_hash) && self.lineage.len() >= MAX_LINEAGE_ENTRIES {
            if let Some(oldest) = self.lineage_order.pop_front() {
                self.lineage.remove(&oldest);
            }
        }
        if self
            .lineage
            .insert(
                content_hash,
                LineageEntry {
                    seed,
                    domain: domain.to_string(),
                },
            )
            .is_none()
        {
            self.lineage_order.push_back(content_hash);
        }
    }

    fn update_canon(&mut self, entry: CanonEntry) {
        self.canon.push(entry);
        self.resort_canon();
        self.canon.truncate(CANON_SIZE);
    }

    /// Re-sort the canon by [`CanonEntry::effective_score`] (aesthetic
    /// quality + reinforcement bonus), descending.
    fn resort_canon(&mut self) {
        self.canon.sort_by(|a, b| {
            b.effective_score()
                .partial_cmp(&a.effective_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// The persisted top-N artistic canon (read-only).
    pub fn canon(&self) -> &[CanonEntry] {
        &self.canon
    }

    /// Find the best past self-authored artifact in `domain` to use as an
    /// imitation basis, returning its generation seed.
    ///
    /// Two-tier selection:
    /// 1. **Preferred**: the canon (`Self::canon`) restricted to `domain`,
    ///    ranked by [`CanonEntry::effective_score`] — the meaningful
    ///    "what's actually good, and what's been built upon" signal.
    ///    Selecting a canon entry this way **reinforces** it: its
    ///    `reuse_count` increments, so an artifact the agent keeps
    ///    returning to becomes progressively more entrenched — this is the
    ///    actual tradition-formation mechanic (Phase 4 item 2). Persisted on
    ///    the next `save()`/`Drop`, not forced immediately (matches this
    ///    struct's existing save-on-drop convention).
    /// 2. **Fallback**: when no canon entry exists for `domain` (e.g. this
    ///    domain has never scored well enough to make the top-N, or has no
    ///    history at all), fall back to ranking the FULL lineage table via
    ///    `ResonanceGraph::rank_content` over content-hash-derived
    ///    fingerprints. See the module-level embedding caveat: with those
    ///    fingerprints, this fallback ranking is not yet a meaningful
    ///    aesthetic-similarity signal when more than one candidate exists —
    ///    it mainly proves out the mechanism end-to-end for domains the
    ///    canon doesn't cover yet. Fallback selections are not reinforced
    ///    (they aren't tracked in the canon to reinforce).
    pub fn best_seed_for_domain(&mut self, domain: &str) -> Option<u64> {
        if let Some(idx) = self
            .canon
            .iter()
            .enumerate()
            .filter(|(_, e)| e.domain == domain)
            .max_by(|(_, a), (_, b)| {
                a.effective_score()
                    .partial_cmp(&b.effective_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
        {
            self.canon[idx].reuse_count += 1;
            let seed = self.canon[idx].seed;
            self.resort_canon();
            return Some(seed);
        }

        let candidates: Vec<ContentRef> = self
            .lineage
            .iter()
            .filter(|(_, entry)| entry.domain == domain)
            .map(|(hash, entry)| ContentRef {
                source_peer: SELF_PEER.to_string(),
                content_hash: *hash,
                hdv_embedding: derive_embedding(hash),
                domain: entry.domain.clone(),
                created_at: 0,
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        let ranked = self.social.graph().rank_content(&candidates, 1);
        ranked
            .first()
            .and_then(|r| self.lineage.get(&r.content.content_hash))
            .map(|e| e.seed)
    }
}

impl Default for CulturalMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CulturalMemoryManager {
    fn drop(&mut self) {
        self.save();
    }
}

/// Derive a deterministic, non-semantic `BinaryHV` fingerprint from a
/// content hash (see module-level embedding caveat).
fn derive_embedding(content_hash: &[u8; 32]) -> BinaryHV {
    let seed = u64::from_le_bytes(content_hash[0..8].try_into().expect("8 bytes"));
    BinaryHV::random(seed)
}

fn hash_to_hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_to_hash(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "symthaea_cultural_memory_test_{name}_{}.json",
            std::process::id()
        ))
    }

    #[test]
    fn publish_lands_in_graph_after_tick() {
        let path = temp_path("publish");
        let _ = std::fs::remove_file(&path);
        let mut mgr = CulturalMemoryManager::new_with_path(Some(path.clone()));

        assert_eq!(mgr.graph().content_count(), 0);
        let content_ref = mgr.publish(b"svg-bytes-1", DOMAIN_VISUAL, 42, 0.8, 1000);
        assert_ne!(content_ref.hdv_embedding, BinaryHV::zero());

        // Not yet landed — publish only enqueues, same as the live receive path.
        assert_eq!(mgr.graph().content_count(), 0);

        mgr.tick_social(&super::super::subsystem_trait::CycleSnapshot::default());
        assert_eq!(mgr.graph().content_count(), 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn different_content_yields_different_embeddings() {
        let e1 = derive_embedding(blake3::hash(b"artifact one").as_bytes());
        let e2 = derive_embedding(blake3::hash(b"artifact two").as_bytes());
        assert_ne!(e1, e2);
        // Deterministic: same bytes -> same embedding.
        let e1_again = derive_embedding(blake3::hash(b"artifact one").as_bytes());
        assert_eq!(e1, e1_again);
    }

    #[test]
    fn lineage_returns_expected_parent_seed() {
        let path = temp_path("lineage");
        let _ = std::fs::remove_file(&path);
        let mut mgr = CulturalMemoryManager::new_with_path(Some(path.clone()));

        mgr.publish(b"visual-artifact-a", DOMAIN_VISUAL, 111, 0.4, 1000);
        mgr.publish(b"visual-artifact-b", DOMAIN_VISUAL, 222, 0.9, 2000);
        mgr.publish(b"music-artifact-a", DOMAIN_MUSIC, 333, 0.95, 3000);

        // Both visual artifacts fit in the canon (well under CANON_SIZE), so
        // selection is the deterministic canon path: the higher-scoring one
        // (222, score 0.9) must win over the lower-scoring one (111, 0.4).
        let seed = mgr.best_seed_for_domain(DOMAIN_VISUAL);
        assert_eq!(seed, Some(222), "higher aesthetic_score should win");

        let music_seed = mgr.best_seed_for_domain(DOMAIN_MUSIC);
        assert_eq!(music_seed, Some(333));

        // No poetry ever published -> no seed to imitate from.
        assert_eq!(mgr.best_seed_for_domain(DOMAIN_POETRY), None);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn canon_retains_only_top_n_by_score() {
        let path = temp_path("canon");
        let _ = std::fs::remove_file(&path);
        let mut mgr = CulturalMemoryManager::new_with_path(Some(path.clone()));

        // Publish more than CANON_SIZE artifacts with varying scores.
        let total = CANON_SIZE + 15;
        for i in 0..total {
            let score = (i as f32) / (total as f32); // strictly increasing
            let bytes = format!("artifact-{i}");
            mgr.publish(bytes.as_bytes(), DOMAIN_VISUAL, i as u64, score, i as u64);
        }

        assert_eq!(mgr.canon().len(), CANON_SIZE);
        // The highest scores should be the last `total-1`, `total-2`, ...
        let top_score = mgr.canon()[0].aesthetic_score;
        assert!((top_score - (total - 1) as f32 / total as f32).abs() < 1e-6);
        // Canon is sorted descending by score.
        for w in mgr.canon().windows(2) {
            assert!(w[0].aesthetic_score >= w[1].aesthetic_score);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn canon_persists_and_reloads() {
        let path = temp_path("persist");
        let _ = std::fs::remove_file(&path);
        {
            let mut mgr = CulturalMemoryManager::new_with_path(Some(path.clone()));
            mgr.publish(b"persisted-artifact", DOMAIN_VISUAL, 7, 0.77, 5000);
            mgr.save();
        }

        let mut reloaded = CulturalMemoryManager::new_with_path(Some(path.clone()));
        assert_eq!(reloaded.canon().len(), 1);
        assert_eq!(reloaded.canon()[0].seed, 7);
        assert_eq!(reloaded.best_seed_for_domain(DOMAIN_VISUAL), Some(7));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn hex_roundtrip() {
        let hash = *blake3::hash(b"roundtrip").as_bytes();
        let hex = hash_to_hex(&hash);
        assert_eq!(hex.len(), 64);
        assert_eq!(hex_to_hash(&hex), Some(hash));
    }

    // ── Tradition formation: reinforcement (Phase 4 item 2 follow-up) ──────

    #[test]
    fn imitation_reinforces_the_selected_canon_entry() {
        let path = temp_path("reinforce");
        let _ = std::fs::remove_file(&path);
        let mut mgr = CulturalMemoryManager::new_with_path(Some(path.clone()));

        mgr.publish(b"only-visual-artifact", DOMAIN_VISUAL, 99, 0.5, 1000);
        assert_eq!(mgr.canon()[0].reuse_count, 0);

        for _ in 0..3 {
            assert_eq!(mgr.best_seed_for_domain(DOMAIN_VISUAL), Some(99));
        }
        assert_eq!(
            mgr.canon()[0].reuse_count,
            3,
            "each selection as an imitation basis should increment reuse_count"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn repeated_reuse_lets_a_lower_scorer_overtake_a_higher_one() {
        // The actual tradition-formation claim: a lower-scoring artifact
        // that keeps getting imitated should eventually rank above a
        // higher-scoring one nobody has ever returned to.
        let path = temp_path("overtake");
        let _ = std::fs::remove_file(&path);
        let mut mgr = CulturalMemoryManager::new_with_path(Some(path.clone()));

        mgr.publish(b"never-reused", DOMAIN_VISUAL, 1, 0.6, 1000);
        mgr.publish(b"heavily-reused", DOMAIN_VISUAL, 2, 0.55, 2000);

        // Before any reuse, the higher raw score (seed 1) wins.
        assert_eq!(mgr.best_seed_for_domain(DOMAIN_VISUAL), Some(1));

        // Manually reinforce seed 2 well past what the first query above
        // did to seed 1 (that query itself reinforced seed 1 once — offset
        // by reinforcing seed 2 enough times to overtake both the score gap
        // AND that one reinforcement).
        for _ in 0..20 {
            let entry = mgr
                .canon
                .iter_mut()
                .find(|e| e.seed == 2)
                .expect("seed 2 must be in canon");
            entry.reuse_count += 1;
        }
        mgr.resort_canon();

        assert!(
            mgr.canon[0].seed == 2,
            "heavily-reused (lower raw score) should now rank first: {:?}",
            mgr.canon
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn effective_score_matches_hand_computed_formula() {
        let entry = CanonEntry {
            content_hash_hex: "irrelevant".to_string(),
            domain: DOMAIN_VISUAL.to_string(),
            aesthetic_score: 0.5,
            resonance: 0.5,
            seed: 1,
            created_at: 0,
            reuse_count: 4,
        };
        let expected = 0.5 + REINFORCEMENT_BONUS_WEIGHT * 5.0f32.ln();
        assert!((entry.effective_score() - expected).abs() < 1e-6);
    }

    #[test]
    fn reuse_count_defaults_to_zero_when_deserializing_old_canon_files() {
        // A canon file written before `reuse_count` existed has no such key.
        // #[serde(default)] must let it load anyway, at 0 (never reinforced).
        let json = r#"{
            "content_hash_hex": "aabb",
            "domain": "art:visual",
            "aesthetic_score": 0.7,
            "resonance": 0.5,
            "seed": 42,
            "created_at": 1000
        }"#;
        let entry: CanonEntry =
            serde_json::from_str(json).expect("must deserialize without reuse_count");
        assert_eq!(entry.reuse_count, 0);
    }
}
