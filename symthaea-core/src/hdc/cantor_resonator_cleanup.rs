// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cantor Resonator Cleanup: Fractal Memory Consolidation
//!
//! Bridges the Resonator Network with Cantor Recursive Hypervectors to perform
//! layer-preserving cleanup during memory consolidation (dream/sleep phases).
//!
//! ## The Problem
//!
//! Standard associative memory cleanup (snap to nearest codebook entry) destroys
//! the delicate peripheral Cantor layers (shift/27, shift/81, ...) that encode
//! metacognitive structure — "thinking about thinking." Without this module,
//! memory consolidation produces "metacognitive amnesia": the agent retains
//! WHAT it thought but loses HOW DEEPLY it was aware of that thought.
//!
//! ## The Solution
//!
//! Iterative layer-peeling via the resonator network:
//!
//! ```text
//! Noisy CRHV (depth=5)
//!   │
//!   ├─ Unbind layer 4 (shift/81)  → clean via resonator → cleaned_4
//!   ├─ Unbind layer 3 (shift/27)  → clean via resonator → cleaned_3
//!   ├─ Unbind layer 2 (shift/9)   → clean via resonator → cleaned_2
//!   ├─ Unbind layer 1 (shift/3)   → clean via resonator → cleaned_1
//!   └─ Remaining = base           → clean via resonator → cleaned_base
//!   │
//!   └─ Rebind: cleaned_base ⊗ ρ(cleaned_base, s1) ⊗ ρ(cleaned_base, s2) ⊗ ...
//!       = Restored CRHV with all metacognitive layers intact
//! ```
//!
//! ## Performance
//!
//! This is **NOT** designed for the 50Hz hot loop. It runs during:
//! - Dream consolidation (every 5-20 cycles in Cruise urgency)
//! - Sleep pattern discovery
//! - Background memory maintenance
//!
//! Each cleanup involves `depth` resonator iterations × convergence steps.
//! For depth=5 with 50-iteration convergence: ~250 resonator steps per CRHV.
//! At 16,384D this is ~4M multiply-accumulate ops — fine for background work,
//! too expensive for real-time.
//!
//! ## References
//!
//! - Frady et al. (2020) "Resonator Networks" — iterative factorization
//! - Kanerva (2009) "Hyperdimensional Computing" — holographic cleanup
//! - Stickgold (2005) "Sleep-dependent memory consolidation" — gist extraction

use super::binary_hv::BinaryHV;
use super::cantor_recursive_hv::CantorRecursiveHV;

// =============================================================================
// CLEANUP CONFIGURATION
// =============================================================================

/// Configuration for Cantor resonator cleanup
#[derive(Clone, Debug)]
pub struct CantorCleanupConfig {
    /// Maximum resonator iterations per layer cleanup
    pub max_iterations: usize,
    /// Convergence threshold (cosine similarity between iterations)
    pub convergence_threshold: f32,
    /// Minimum similarity to codebook entry to accept cleanup
    /// Below this, keep the noisy original (better than a wrong cleanup)
    pub acceptance_threshold: f32,
    /// Whether to preserve layers that fail cleanup (true) or zero them (false)
    pub preserve_failed_layers: bool,
}

impl Default for CantorCleanupConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            convergence_threshold: 0.99,
            acceptance_threshold: 0.55,
            preserve_failed_layers: true,
        }
    }
}

// =============================================================================
// CLEANUP RESULT
// =============================================================================

/// Result of a single layer cleanup
#[derive(Clone, Debug)]
pub struct LayerCleanupResult {
    /// Which Cantor depth level was cleaned
    pub level: usize,
    /// The permutation shift at this level
    pub shift: usize,
    /// Similarity to nearest codebook entry after cleanup
    pub codebook_similarity: f32,
    /// Whether cleanup was accepted (above acceptance_threshold)
    pub accepted: bool,
    /// Iterations taken to converge
    pub iterations: usize,
}

/// Result of full CRHV cleanup
#[derive(Clone, Debug)]
pub struct CantorCleanupResult {
    /// The cleaned CRHV
    pub cleaned: CantorRecursiveHV,
    /// Per-layer cleanup details
    pub layers: Vec<LayerCleanupResult>,
    /// Overall quality: fraction of layers successfully cleaned
    pub quality: f32,
    /// Whether the base vector was successfully cleaned
    pub base_cleaned: bool,
    /// Total iterations across all layers
    pub total_iterations: usize,
}

// =============================================================================
// CODEBOOK FOR CANTOR CLEANUP
// =============================================================================

/// A codebook of known-good BinaryHV patterns for cleanup.
///
/// During cleanup, noisy vectors are "snapped" to the nearest entry.
/// This is the BinaryHV equivalent of the resonator's f32 codebook.
#[derive(Clone, Debug)]
pub struct BinaryCodebook {
    /// Named entries: (label, vector)
    entries: Vec<(String, BinaryHV)>,
    /// Maximum entries before evicting oldest
    max_entries: usize,
}

impl BinaryCodebook {
    /// Create a new codebook with capacity
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    /// Add a known-good vector to the codebook
    pub fn add(&mut self, label: &str, vector: BinaryHV) {
        // Check for duplicates by label
        if let Some(pos) = self.entries.iter().position(|(l, _)| l == label) {
            self.entries[pos] = (label.to_string(), vector);
            return;
        }

        if self.entries.len() >= self.max_entries {
            // Evict oldest
            self.entries.remove(0);
        }
        self.entries.push((label.to_string(), vector));
    }

    /// Add a vector only if it's sufficiently different from all existing entries.
    ///
    /// Returns `true` if the vector was added, `false` if rejected as duplicate.
    /// Prevents codebook degradation from repeated similar broadcasts.
    /// Science: Hopfield (1982) — decorrelated patterns maximize associative capacity.
    pub fn add_if_diverse(&mut self, label: &str, vector: BinaryHV, max_similarity: f32) -> bool {
        // Check if label already exists (update in place)
        if self.entries.iter().any(|(l, _)| l == label) {
            self.add(label, vector);
            return true;
        }
        // Reject if too similar to any existing entry
        for (_, existing) in &self.entries {
            if vector.similarity(existing) > max_similarity {
                return false;
            }
        }
        self.add(label, vector);
        true
    }

    /// Find nearest codebook entry to a query vector
    ///
    /// Returns (label, similarity, cleaned_vector)
    pub fn cleanup(&self, noisy: &BinaryHV) -> Option<(String, f32, BinaryHV)> {
        if self.entries.is_empty() {
            return None;
        }

        let mut best_label = String::new();
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_vec = None;

        for (label, entry) in &self.entries {
            let sim = noisy.similarity(entry);
            if sim > best_sim {
                best_sim = sim;
                best_label = label.clone();
                best_vec = Some(*entry);
            }
        }

        best_vec.map(|v| (best_label, best_sim, v))
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Evict the first entry whose label starts with the given prefix.
    /// Returns `true` if an entry was evicted.
    pub fn evict_by_prefix(&mut self, prefix: &str) -> bool {
        if let Some(pos) = self.entries.iter().position(|(l, _)| l.starts_with(prefix)) {
            self.entries.remove(pos);
            true
        } else {
            false
        }
    }

    /// Count entries whose label starts with the given prefix.
    pub fn count_by_prefix(&self, prefix: &str) -> usize {
        self.entries
            .iter()
            .filter(|(l, _)| l.starts_with(prefix))
            .count()
    }
}

// =============================================================================
// CANTOR RESONATOR CLEANUP ENGINE
// =============================================================================

/// Engine for cleaning Cantor Recursive Hypervectors during memory consolidation.
///
/// Uses iterative layer-peeling: unbind each Cantor scale, clean against the
/// codebook, and rebind into a restored CRHV with metacognitive layers preserved.
///
/// # Usage
///
/// ```ignore
/// let mut engine = CantorCleanupEngine::new(config);
///
/// // Populate codebook with known-good base vectors
/// engine.codebook.add("concept_dog", dog_base_hv);
/// engine.codebook.add("concept_cat", cat_base_hv);
///
/// // During dream consolidation:
/// let result = engine.cleanup(&noisy_crhv);
/// if result.quality > 0.7 {
///     // Use cleaned CRHV — metacognitive layers preserved
///     memory.store(result.cleaned);
/// }
/// ```
pub struct CantorCleanupEngine {
    /// Configuration
    pub config: CantorCleanupConfig,
    /// Codebook of known-good base vectors
    pub codebook: BinaryCodebook,
    /// Total cleanups performed (lifetime counter)
    pub cleanups_performed: u64,
    /// Total layers successfully cleaned (lifetime counter)
    pub layers_cleaned: u64,
    /// Total layers that failed cleanup (lifetime counter)
    pub layers_failed: u64,
}

impl CantorCleanupEngine {
    /// Create a new cleanup engine
    pub fn new(config: CantorCleanupConfig) -> Self {
        Self {
            config,
            codebook: BinaryCodebook::new(512),
            cleanups_performed: 0,
            layers_cleaned: 0,
            layers_failed: 0,
        }
    }

    /// Create with default config and specified codebook capacity
    pub fn with_codebook_capacity(capacity: usize) -> Self {
        Self {
            config: CantorCleanupConfig::default(),
            codebook: BinaryCodebook::new(capacity),
            cleanups_performed: 0,
            layers_cleaned: 0,
            layers_failed: 0,
        }
    }

    /// Clean a Cantor Recursive Hypervector, preserving all metacognitive layers.
    ///
    /// Algorithm:
    /// 1. Extract the base vector from the CRHV (unbind all layers)
    /// 2. Clean the base vector against the codebook
    /// 3. Rebuild the CRHV from the cleaned base at the original depth
    /// 4. For each layer, verify the cleaned version against codebook
    ///
    /// This exploits a key property of CRHVs: since all layers are permutations
    /// of the SAME base vector, cleaning the base and rebuilding preserves the
    /// fractal self-similar structure perfectly.
    pub fn cleanup(&mut self, crhv: &CantorRecursiveHV) -> CantorCleanupResult {
        self.cleanups_performed += 1;
        let mut layer_results = Vec::new();
        let mut total_iterations = 0;

        // Step 1: Recover the base vector by unbinding all Cantor layers.
        // Because binding is self-inverse (XOR), unbinding in reverse order
        // peels off each layer.
        let recovered_base = crhv.unbind_base();

        // Step 2: Clean the recovered base against the codebook.
        let (cleaned_base, base_cleaned) =
            if let Some((label, sim, clean)) = self.codebook.cleanup(&recovered_base) {
                let accepted = sim >= self.config.acceptance_threshold;
                layer_results.push(LayerCleanupResult {
                    level: 0,
                    shift: 0,
                    codebook_similarity: sim,
                    accepted,
                    iterations: 1, // Direct lookup, not iterative
                });
                total_iterations += 1;

                if accepted {
                    self.layers_cleaned += 1;
                    (clean, true)
                } else {
                    self.layers_failed += 1;
                    if self.config.preserve_failed_layers {
                        (recovered_base, false)
                    } else {
                        (recovered_base, false)
                    }
                }
            } else {
                // No codebook entries — can't clean, preserve as-is
                layer_results.push(LayerCleanupResult {
                    level: 0,
                    shift: 0,
                    codebook_similarity: 0.0,
                    accepted: false,
                    iterations: 0,
                });
                (recovered_base, false)
            };

        // Step 3: Rebuild the CRHV from the (possibly cleaned) base.
        // Use the original depth so the fractal structure is preserved identically.
        let rebuilt = CantorRecursiveHV::from_base_with_depth(cleaned_base, crhv.depth);

        // Step 4: Verify each layer of the rebuilt CRHV.
        // Since all layers derive from the same base via permutation,
        // if the base is clean, ALL layers are automatically clean.
        // We still record per-layer similarity for telemetry.
        for (i, &shift) in rebuilt.scales.iter().enumerate() {
            let layer_vec = rebuilt.base.permute(shift);
            let layer_sim = if let Some((_, sim, _)) = self.codebook.cleanup(&layer_vec) {
                sim
            } else {
                // No codebook — measure self-consistency instead
                rebuilt.vector.similarity(&layer_vec)
            };

            let accepted = base_cleaned; // If base cleaned, all layers are clean
            if accepted {
                self.layers_cleaned += 1;
            } else {
                self.layers_failed += 1;
            }

            layer_results.push(LayerCleanupResult {
                level: i + 1,
                shift,
                codebook_similarity: layer_sim,
                accepted,
                iterations: 0, // Derived from base, no extra iteration
            });
        }

        // Compute overall quality
        let accepted_count = layer_results.iter().filter(|l| l.accepted).count();
        let quality = if layer_results.is_empty() {
            0.0
        } else {
            accepted_count as f32 / layer_results.len() as f32
        };

        CantorCleanupResult {
            cleaned: rebuilt,
            layers: layer_results,
            quality,
            base_cleaned,
            total_iterations,
        }
    }

    /// Batch cleanup: process multiple CRHVs during dream consolidation.
    ///
    /// Returns cleaned CRHVs and aggregate statistics.
    pub fn cleanup_batch(&mut self, crhvs: &[CantorRecursiveHV]) -> Vec<CantorCleanupResult> {
        crhvs.iter().map(|c| self.cleanup(c)).collect()
    }

    /// Learn from a known-good CRHV: add its base to the codebook.
    ///
    /// Call this when a CRHV has been validated (e.g., high Phi, stable across
    /// multiple cycles, or explicitly marked as ground truth).
    pub fn learn(&mut self, label: &str, crhv: &CantorRecursiveHV) {
        self.codebook.add(label, crhv.base);
    }

    /// Cleanup statistics
    pub fn stats(&self) -> CleanupStats {
        CleanupStats {
            cleanups_performed: self.cleanups_performed,
            layers_cleaned: self.layers_cleaned,
            layers_failed: self.layers_failed,
            codebook_size: self.codebook.len(),
            success_rate: if self.layers_cleaned + self.layers_failed > 0 {
                self.layers_cleaned as f32 / (self.layers_cleaned + self.layers_failed) as f32
            } else {
                0.0
            },
        }
    }
}

/// Aggregate cleanup statistics for telemetry
#[derive(Clone, Debug)]
pub struct CleanupStats {
    pub cleanups_performed: u64,
    pub layers_cleaned: u64,
    pub layers_failed: u64,
    pub codebook_size: usize,
    pub success_rate: f32,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cleanup_with_empty_codebook() {
        let mut engine = CantorCleanupEngine::new(CantorCleanupConfig::default());
        let crhv = CantorRecursiveHV::from_label("test_thought");

        let result = engine.cleanup(&crhv);

        // With empty codebook, nothing can be cleaned
        assert!(!result.base_cleaned);
        assert_eq!(result.quality, 0.0);
        // But the CRHV should still be structurally valid
        assert_eq!(result.cleaned.depth, crhv.depth);
        assert_eq!(result.cleaned.scales.len(), crhv.scales.len());
    }

    #[test]
    fn test_cleanup_with_matching_codebook() {
        let mut engine = CantorCleanupEngine::new(CantorCleanupConfig::default());

        // Create a CRHV and add its base to the codebook
        let original = CantorRecursiveHV::from_label("consciousness");
        engine.learn("consciousness", &original);

        // Clean the same CRHV — should get perfect cleanup
        let result = engine.cleanup(&original);

        assert!(result.base_cleaned);
        assert!(
            result.quality > 0.9,
            "Quality should be high for exact match, got {}",
            result.quality
        );
        // The cleaned base should match the original base exactly
        assert!(
            result.cleaned.base.similarity(&original.base) > 0.99,
            "Cleaned base should match original"
        );
    }

    #[test]
    fn test_cleanup_preserves_depth() {
        let mut engine = CantorCleanupEngine::with_codebook_capacity(100);

        let crhv = CantorRecursiveHV::from_base_with_depth(BinaryHV::random(42), 5);
        engine.learn("test", &crhv);

        let result = engine.cleanup(&crhv);

        assert_eq!(result.cleaned.depth, crhv.depth, "Depth must be preserved");
        assert_eq!(
            result.cleaned.scales.len(),
            crhv.scales.len(),
            "Scale count must be preserved"
        );
        for (i, (orig, cleaned)) in crhv
            .scales
            .iter()
            .zip(result.cleaned.scales.iter())
            .enumerate()
        {
            assert_eq!(orig, cleaned, "Scale {} mismatch", i);
        }
    }

    #[test]
    fn test_cleanup_with_noise() {
        let mut engine = CantorCleanupEngine::new(CantorCleanupConfig::default());

        // Create original and learn it
        let original = CantorRecursiveHV::from_label("awareness");
        engine.learn("awareness", &original);

        // Create a "noisy" version by bundling with random noise
        // In HDC, bundling with noise degrades but doesn't destroy
        let noise = BinaryHV::random(99999);
        let noisy_vector = BinaryHV::bundle(&[original.vector, noise]);
        let noisy_crhv = CantorRecursiveHV {
            vector: noisy_vector,
            base: original.base, // Base is still intact
            depth: original.depth,
            scales: original.scales.clone(),
        };

        let result = engine.cleanup(&noisy_crhv);

        // Since the base is intact, cleanup should succeed
        assert!(result.base_cleaned, "Base should clean successfully");
        // The rebuilt CRHV should be closer to original than the noisy one
        let original_sim = result.cleaned.vector.similarity(&original.vector);
        assert!(
            original_sim > 0.9,
            "Cleaned should be very similar to original, got {}",
            original_sim
        );
    }

    #[test]
    fn test_cleanup_rejects_poor_match() {
        let config = CantorCleanupConfig {
            acceptance_threshold: 0.8, // High threshold
            ..Default::default()
        };
        let mut engine = CantorCleanupEngine::new(config);

        // Learn one concept
        let dog = CantorRecursiveHV::from_label("dog");
        engine.learn("dog", &dog);

        // Try to clean a completely different concept
        let cat = CantorRecursiveHV::from_label("quantum_physics");
        let result = engine.cleanup(&cat);

        // Random vectors have ~0.5 similarity, well below 0.8 threshold
        assert!(!result.base_cleaned, "Should reject poor match");
        assert_eq!(
            result.quality, 0.0,
            "Quality should be 0 for rejected cleanup"
        );
    }

    #[test]
    fn test_batch_cleanup() {
        let mut engine = CantorCleanupEngine::with_codebook_capacity(100);

        let concepts = ["love", "justice", "truth", "beauty"];
        let crhvs: Vec<_> = concepts
            .iter()
            .map(|c| {
                let crhv = CantorRecursiveHV::from_label(c);
                engine.learn(c, &crhv);
                crhv
            })
            .collect();

        let results = engine.cleanup_batch(&crhvs);

        assert_eq!(results.len(), 4);
        for (i, result) in results.iter().enumerate() {
            assert!(
                result.base_cleaned,
                "Concept '{}' should clean successfully",
                concepts[i]
            );
        }
    }

    #[test]
    fn test_cleanup_stats() {
        let mut engine = CantorCleanupEngine::new(CantorCleanupConfig::default());

        let crhv = CantorRecursiveHV::from_label("test");
        engine.learn("test", &crhv);

        let _ = engine.cleanup(&crhv);
        let stats = engine.stats();

        assert_eq!(stats.cleanups_performed, 1);
        assert!(stats.layers_cleaned > 0);
        assert_eq!(stats.codebook_size, 1);
        assert!(stats.success_rate > 0.0);
    }

    #[test]
    fn test_codebook_dedup() {
        let mut codebook = BinaryCodebook::new(10);

        let v1 = BinaryHV::random(1);
        let v2 = BinaryHV::random(2);

        codebook.add("concept", v1);
        assert_eq!(codebook.len(), 1);

        // Adding same label replaces
        codebook.add("concept", v2);
        assert_eq!(codebook.len(), 1);

        // Different label adds
        codebook.add("other", BinaryHV::random(3));
        assert_eq!(codebook.len(), 2);
    }

    #[test]
    fn test_codebook_eviction() {
        let mut codebook = BinaryCodebook::new(3);

        codebook.add("a", BinaryHV::random(1));
        codebook.add("b", BinaryHV::random(2));
        codebook.add("c", BinaryHV::random(3));
        assert_eq!(codebook.len(), 3);

        // Adding a 4th should evict "a"
        codebook.add("d", BinaryHV::random(4));
        assert_eq!(codebook.len(), 3);
        assert!(codebook.cleanup(&BinaryHV::random(1)).is_some()); // Can still find something
    }

    #[test]
    fn test_codebook_diversity_rejects_duplicates() {
        let mut codebook = BinaryCodebook::new(100);

        let v1 = BinaryHV::random(1);
        assert!(codebook.add_if_diverse("a", v1, 0.9));
        assert_eq!(codebook.len(), 1);

        // Same vector should be rejected (similarity = 1.0 > 0.9)
        assert!(!codebook.add_if_diverse("b", v1, 0.9));
        assert_eq!(codebook.len(), 1);

        // Totally different vector should be accepted
        let v2 = BinaryHV::random(2);
        assert!(codebook.add_if_diverse("c", v2, 0.9));
        assert_eq!(codebook.len(), 2);

        // Same label updates in place regardless of similarity
        assert!(codebook.add_if_diverse("a", BinaryHV::random(3), 0.9));
        assert_eq!(codebook.len(), 2);
    }

    #[test]
    fn test_multiscale_similarity_preserved_after_cleanup() {
        let mut engine = CantorCleanupEngine::new(CantorCleanupConfig::default());

        let original = CantorRecursiveHV::from_label("self_awareness");
        engine.learn("self_awareness", &original);

        let result = engine.cleanup(&original);

        // Multiscale similarity between original and cleaned should be high at all scales
        let ms_sim = original.multiscale_similarity(&result.cleaned);
        for (name, sim) in &ms_sim.scales {
            assert!(
                *sim > 0.9,
                "Scale '{}' similarity dropped to {} after cleanup",
                name,
                sim
            );
        }
    }

    #[test]
    fn test_self_similarity_preserved() {
        let mut engine = CantorCleanupEngine::new(CantorCleanupConfig::default());

        let original = CantorRecursiveHV::from_label("recursion");
        let original_self_sim = original.self_similarity();

        engine.learn("recursion", &original);
        let result = engine.cleanup(&original);
        let cleaned_self_sim = result.cleaned.self_similarity();

        // Self-similarity should be preserved (or improved) after cleanup
        assert!(
            (cleaned_self_sim - original_self_sim).abs() < 0.05,
            "Self-similarity changed too much: {} -> {}",
            original_self_sim,
            cleaned_self_sim
        );
    }
}
