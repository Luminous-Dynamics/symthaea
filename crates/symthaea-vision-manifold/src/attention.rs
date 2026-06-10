// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Surprise-driven spatial attention via per-patch free energy tracking.
//!
//! Each patch position accumulates prediction error between consecutive frames.
//! High-surprise patches indicate scene regions with unexpected change —
//! the manifold's analog of active inference "foraging."

use symthaea_core::hdc::ContinuousHV;

use crate::types::{AttentionMap, PatchGrid};

/// Tracks per-patch surprise (free energy proxy) over time.
pub struct SurpriseMap {
    grid: PatchGrid,
    surprise: Vec<f32>,
    decay: f32,
    threshold: f32,
}

impl SurpriseMap {
    pub fn new(grid: PatchGrid, decay: f32, threshold: f32) -> Self {
        let n = grid.num_patches();
        Self {
            grid,
            surprise: vec![0.0; n],
            decay,
            threshold,
        }
    }

    /// Update surprise from current vs previous per-patch HVs.
    ///
    /// Surprise at each patch = 1 - cosine_similarity(current, previous).
    /// Accumulated with exponential decay. Bounded to the theoretical
    /// steady-state maximum of `1.0 / (1.0 - decay)` to prevent runaway
    /// accumulation from pathological inputs while preserving normal dynamics.
    pub fn update(&mut self, current_patches: &[ContinuousHV], previous_patches: &[ContinuousHV]) {
        // Decay existing surprise
        for s in &mut self.surprise {
            *s *= self.decay;
        }

        if previous_patches.is_empty() {
            return;
        }

        let n = current_patches
            .len()
            .min(previous_patches.len())
            .min(self.surprise.len());

        // Soft cap at theoretical steady-state maximum
        let max_surprise = 1.0 / (1.0 - self.decay).max(0.01);

        for i in 0..n {
            let sim = current_patches[i].similarity(&previous_patches[i]);
            let patch_surprise = (1.0 - sim).max(0.0);
            self.surprise[i] += patch_surprise;
            self.surprise[i] = self.surprise[i].min(max_surprise);
        }
    }

    /// Get the current attention map snapshot.
    pub fn attention_map(&self) -> AttentionMap {
        AttentionMap {
            values: self.surprise.clone(),
            grid: self.grid.clone(),
        }
    }

    /// Return salient patch positions (above threshold).
    pub fn salient_patches(&self) -> Vec<(usize, usize, f32)> {
        self.attention_map().salient_patches(self.threshold)
    }

    /// Mean surprise across all patches.
    pub fn mean_surprise(&self) -> f32 {
        if self.surprise.is_empty() {
            return 0.0;
        }
        self.surprise.iter().sum::<f32>() / self.surprise.len() as f32
    }

    /// Maximum surprise across all patches.
    pub fn max_surprise(&self) -> f32 {
        self.surprise.iter().copied().fold(0.0f32, f32::max)
    }

    /// Dampen surprise at a specific grid position by a multiplicative factor.
    ///
    /// Used for top-down priming: once a region has been recognized by the
    /// ventral stream, its surprise is reduced so it won't trigger repeated
    /// foveation (biological analog: you don't re-read "STOP" every frame).
    ///
    /// # Arguments
    /// * `row`, `col` — Grid coordinates of the patch to dampen.
    /// * `factor` — Multiplicative factor (0.0 = full suppression, 1.0 = no change).
    pub fn dampen(&mut self, row: usize, col: usize, factor: f32) {
        let idx = self.grid.patch_index(row, col);
        if let Some(s) = self.surprise.get_mut(idx) {
            *s *= factor.clamp(0.0, 1.0);
        }
    }

    /// Access the underlying grid.
    pub fn grid(&self) -> &PatchGrid {
        &self.grid
    }

    /// Inject cross-scale predictive errors as additional free-energy signal.
    ///
    /// Called after `update()` when a `PredictiveCodingHierarchy` is active.
    /// Adds `weight * error[i]` to each patch's accumulated surprise, capped at
    /// the same steady-state maximum as the temporal surprise.
    ///
    /// This makes the surprise map reflect both:
    /// - **Temporal change** (frame-to-frame): from `update()`
    /// - **Scale inconsistency** (coarse fails to predict fine): from this call
    ///
    /// # Arguments
    /// * `patch_errors` — Per-patch cross-scale prediction error (0.0–1.0). Length
    ///   may differ from the grid count; excess entries are ignored, missing ones
    ///   receive no injection.
    /// * `weight` — Mixing weight for the injected signal (0.0 = no injection,
    ///   1.0 = full cross-scale signal). Typical value: 0.3.
    pub fn inject_cross_scale_error(&mut self, patch_errors: &[f32], weight: f32) {
        if weight < 1e-6 || patch_errors.is_empty() {
            return;
        }
        let max_surprise = 1.0 / (1.0 - self.decay).max(0.01);
        let n = patch_errors.len().min(self.surprise.len());
        for i in 0..n {
            self.surprise[i] += weight * patch_errors[i].clamp(0.0, 1.0);
            self.surprise[i] = self.surprise[i].min(max_surprise);
        }
    }

    /// Reset all surprise to zero.
    pub fn reset(&mut self) {
        self.surprise.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid() -> PatchGrid {
        PatchGrid::new(32, 32, 8) // 4×4 = 16 patches
    }

    fn random_patch_hvs(n: usize, seed_base: u64) -> Vec<ContinuousHV> {
        (0..n)
            .map(|i| ContinuousHV::random(16_384, seed_base + i as u64))
            .collect()
    }

    #[test]
    fn test_identical_frames_no_surprise() {
        let grid = make_grid();
        let mut sm = SurpriseMap::new(grid, 0.9, 0.3);

        let patches = random_patch_hvs(16, 1000);
        sm.update(&patches, &patches);

        assert!(
            sm.max_surprise() < 0.05,
            "Identical frames should produce near-zero surprise, got {}",
            sm.max_surprise()
        );
    }

    #[test]
    fn test_different_frames_high_surprise() {
        let grid = make_grid();
        let mut sm = SurpriseMap::new(grid, 0.9, 0.3);

        let prev = random_patch_hvs(16, 1000);
        let curr = random_patch_hvs(16, 2000); // completely different
        sm.update(&curr, &prev);

        assert!(
            sm.mean_surprise() > 0.5,
            "Completely different frames should produce high surprise, got {}",
            sm.mean_surprise()
        );
    }

    #[test]
    fn test_single_patch_change_localized() {
        let grid = make_grid();
        let mut sm = SurpriseMap::new(grid, 0.9, 0.3);

        let prev = random_patch_hvs(16, 1000);
        let mut curr = prev.clone();
        // Replace patch 5 with a completely different vector
        curr[5] = ContinuousHV::random(16_384, 9999);

        sm.update(&curr, &prev);

        // Patch 5 should have the highest surprise
        let max_idx = sm
            .surprise
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 5, "Changed patch should have highest surprise");
    }

    #[test]
    fn test_surprise_decays() {
        let grid = make_grid();
        let mut sm = SurpriseMap::new(grid, 0.5, 0.3); // aggressive decay

        let prev = random_patch_hvs(16, 1000);
        let curr = random_patch_hvs(16, 2000);
        sm.update(&curr, &prev);
        let surprise_after_change = sm.mean_surprise();

        // Feed identical frames → surprise decays
        sm.update(&curr, &curr);
        let surprise_after_same = sm.mean_surprise();

        assert!(
            surprise_after_same < surprise_after_change,
            "Surprise should decay: {surprise_after_same} < {surprise_after_change}"
        );
    }

    #[test]
    fn test_reset() {
        let grid = make_grid();
        let mut sm = SurpriseMap::new(grid, 0.9, 0.3);

        let prev = random_patch_hvs(16, 1000);
        let curr = random_patch_hvs(16, 2000);
        sm.update(&curr, &prev);
        assert!(sm.max_surprise() > 0.0);

        sm.reset();
        assert_eq!(sm.max_surprise(), 0.0);
    }

    #[test]
    fn test_dampen_reduces_surprise() {
        let grid = make_grid();
        let mut sm = SurpriseMap::new(grid, 0.9, 0.3);

        let prev = random_patch_hvs(16, 1000);
        let curr = random_patch_hvs(16, 2000);
        sm.update(&curr, &prev);

        let before = sm.surprise[5];
        assert!(before > 0.0);

        sm.dampen(1, 1, 0.5); // patch index 5 in 4×4 grid
        let after = sm.surprise[5];
        assert!(
            (after - before * 0.5).abs() < 1e-6,
            "Dampen(0.5) should halve surprise: before={before}, after={after}"
        );
    }

    #[test]
    fn test_dampen_full_suppression() {
        let grid = make_grid();
        let mut sm = SurpriseMap::new(grid, 0.9, 0.3);

        let prev = random_patch_hvs(16, 1000);
        let curr = random_patch_hvs(16, 2000);
        sm.update(&curr, &prev);
        assert!(sm.surprise[0] > 0.0);

        sm.dampen(0, 0, 0.0);
        assert_eq!(sm.surprise[0], 0.0);
    }

    #[test]
    fn test_dampen_clamps_factor() {
        let grid = make_grid();
        let mut sm = SurpriseMap::new(grid, 0.9, 0.3);

        let prev = random_patch_hvs(16, 1000);
        let curr = random_patch_hvs(16, 2000);
        sm.update(&curr, &prev);
        let before = sm.surprise[3];

        // Factor > 1.0 should be clamped to 1.0 (no change)
        sm.dampen(0, 3, 2.0);
        assert!((sm.surprise[3] - before).abs() < 1e-6);
    }

    #[test]
    fn test_dampen_out_of_bounds_is_noop() {
        let grid = make_grid();
        let mut sm = SurpriseMap::new(grid, 0.9, 0.3);

        let prev = random_patch_hvs(16, 1000);
        let curr = random_patch_hvs(16, 2000);
        sm.update(&curr, &prev);

        // Out of bounds — should not panic
        sm.dampen(100, 100, 0.5);
    }

    #[test]
    fn test_grid_accessor() {
        let grid = make_grid();
        let sm = SurpriseMap::new(grid, 0.9, 0.3);
        assert_eq!(sm.grid().cols, 4);
        assert_eq!(sm.grid().rows, 4);
        assert_eq!(sm.grid().patch_size, 8);
    }

    // === P2-A: Cross-Scale Predictive Error Injection ===

    #[test]
    fn test_inject_cross_scale_increases_surprise() {
        let grid = make_grid();
        let mut sm = SurpriseMap::new(grid, 0.9, 0.3);

        // Start with zero surprise
        assert_eq!(sm.max_surprise(), 0.0);

        // Inject 1.0 error at all 16 patches with weight 0.5
        let errors = vec![1.0f32; 16];
        sm.inject_cross_scale_error(&errors, 0.5);

        // Each patch should now have 0.5 surprise
        for &s in &sm.attention_map().values {
            assert!(
                (s - 0.5).abs() < 1e-6,
                "Each patch should have 0.5 surprise, got {s}"
            );
        }
    }

    #[test]
    fn test_inject_cross_scale_zero_weight_is_noop() {
        let grid = make_grid();
        let mut sm = SurpriseMap::new(grid, 0.9, 0.3);

        let errors = vec![1.0f32; 16];
        sm.inject_cross_scale_error(&errors, 0.0);
        assert_eq!(sm.max_surprise(), 0.0);
    }

    #[test]
    fn test_inject_cross_scale_bounded_by_steady_state() {
        let decay = 0.9;
        let grid = make_grid();
        let mut sm = SurpriseMap::new(grid, decay, 0.3);
        let max_allowed = 1.0 / (1.0 - decay);

        // Inject many times to try to overflow
        let errors = vec![1.0f32; 16];
        for _ in 0..200 {
            sm.inject_cross_scale_error(&errors, 1.0);
        }

        for &s in &sm.attention_map().values {
            assert!(
                s <= max_allowed + 1e-6,
                "Surprise should be bounded: {s} > {max_allowed}"
            );
        }
    }

    #[test]
    fn test_inject_cross_scale_partial_patch_count() {
        let grid = make_grid(); // 16 patches
        let mut sm = SurpriseMap::new(grid, 0.9, 0.3);

        // Only 4 error values — should update first 4 patches, leave others at 0
        let errors = vec![1.0f32; 4];
        sm.inject_cross_scale_error(&errors, 0.5);

        let values = sm.attention_map().values;
        for &s in &values[..4] {
            assert!((s - 0.5).abs() < 1e-6);
        }
        for &s in &values[4..] {
            assert_eq!(s, 0.0, "Patches beyond error count should be untouched");
        }
    }

    #[test]
    fn test_surprise_bounded_under_constant_change() {
        let grid = make_grid();
        let decay = 0.9;
        let mut sm = SurpriseMap::new(grid, decay, 0.3);
        let max_surprise = 1.0 / (1.0 - decay);

        // Feed 100 frames of maximally different patches
        for i in 0..100u64 {
            let prev = random_patch_hvs(16, 1000 + i * 100);
            let curr = random_patch_hvs(16, 2000 + i * 100);
            sm.update(&curr, &prev);
        }

        // All surprise values should be bounded
        for &s in &sm.surprise {
            assert!(
                s <= max_surprise + 1e-6,
                "Surprise {s} exceeds theoretical max {max_surprise}"
            );
        }
    }
}
