// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Surprise-driven spatial attention via per-patch free energy tracking.
//!
//! Each patch position accumulates prediction error between consecutive frames.
//! High-surprise patches indicate scene regions with unexpected change —
//! the manifold's analog of active inference "foraging."

use symthaea_core::hdc::ContinuousHV;

use crate::types::{AttentionMap, PatchGrid, SurpriseMapState};

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
        let _ = self.dampen_checked(row, col, factor);
    }

    /// Checked variant of [`Self::dampen`] that rejects malformed policy input
    /// before mutating the accumulated surprise map.
    pub fn dampen_checked(&mut self, row: usize, col: usize, factor: f32) -> Result<(), String> {
        if !factor.is_finite() || !(0.0..=1.0).contains(&factor) {
            return Err(format!(
                "dampen factor must be finite and in [0, 1], got {factor}"
            ));
        }
        if row >= self.grid.rows || col >= self.grid.cols {
            return Err(format!(
                "surprise coordinate ({row}, {col}) is outside {}x{} grid",
                self.grid.rows, self.grid.cols
            ));
        }
        let idx = self.grid.patch_index(row, col);
        self.surprise[idx] *= factor;
        Ok(())
    }

    /// Update the temporal persistence of accumulated surprise.
    ///
    /// Larger values retain evidence longer. The value is clamped below one so
    /// the steady-state cap remains finite.
    pub fn set_decay(&mut self, decay: f32) {
        let _ = self.set_decay_checked(decay);
    }

    /// Checked surprise-persistence update.
    pub fn set_decay_checked(&mut self, decay: f32) -> Result<(), String> {
        if !decay.is_finite() || !(0.001..=0.999).contains(&decay) {
            return Err(format!(
                "surprise decay must be finite and in [0.001, 0.999], got {decay}"
            ));
        }
        self.decay = decay;
        Ok(())
    }

    /// Current temporal surprise persistence.
    pub fn decay(&self) -> f32 {
        self.decay
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
        let _ = self.inject_cross_scale_error_checked(patch_errors, weight);
    }

    /// Checked cross-scale evidence injection.
    ///
    /// All supplied values are validated before any patch is updated, so one
    /// malformed value cannot leave the attention map partially mutated.
    pub fn inject_cross_scale_error_checked(
        &mut self,
        patch_errors: &[f32],
        weight: f32,
    ) -> Result<(), String> {
        if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
            return Err(format!(
                "cross-scale weight must be finite and in [0, 1], got {weight}"
            ));
        }
        if let Some((index, value)) = patch_errors
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(format!(
                "cross-scale error at index {index} must be finite and in [0, 1], got {value}"
            ));
        }
        if weight < 1e-6 || patch_errors.is_empty() {
            return Ok(());
        }
        let max_surprise = 1.0 / (1.0 - self.decay).max(0.01);
        let n = patch_errors.len().min(self.surprise.len());
        for (surprise, error) in self.surprise[..n].iter_mut().zip(&patch_errors[..n]) {
            *surprise = (*surprise + weight * *error).min(max_surprise);
        }
        Ok(())
    }

    pub(crate) fn save_state(&self) -> SurpriseMapState {
        SurpriseMapState {
            values: self.surprise.clone(),
            decay: self.decay,
            threshold: self.threshold,
            cols: self.grid.cols,
            rows: self.grid.rows,
            patch_size: self.grid.patch_size,
            frame_width: self.grid.frame_width,
            frame_height: self.grid.frame_height,
        }
    }

    pub(crate) fn validate_state(
        state: &SurpriseMapState,
        expected_grid: &PatchGrid,
    ) -> Result<(), String> {
        if state.cols != expected_grid.cols
            || state.rows != expected_grid.rows
            || state.patch_size != expected_grid.patch_size
            || state.frame_width != expected_grid.frame_width
            || state.frame_height != expected_grid.frame_height
        {
            return Err(format!(
                "surprise grid mismatch: saved={}x{}@{} for {}x{}, expected={}x{}@{} for {}x{}",
                state.cols,
                state.rows,
                state.patch_size,
                state.frame_width,
                state.frame_height,
                expected_grid.cols,
                expected_grid.rows,
                expected_grid.patch_size,
                expected_grid.frame_width,
                expected_grid.frame_height
            ));
        }
        if state.values.len() != expected_grid.num_patches() {
            return Err(format!(
                "surprise value count mismatch: saved={}, expected={}",
                state.values.len(),
                expected_grid.num_patches()
            ));
        }
        if !state.values.iter().all(|value| value.is_finite()) {
            return Err("surprise state contains non-finite values".to_string());
        }
        if !(0.0..1.0).contains(&state.decay) {
            return Err(format!("invalid surprise decay: {}", state.decay));
        }
        if !state.threshold.is_finite() || state.threshold <= 0.0 {
            return Err(format!("invalid surprise threshold: {}", state.threshold));
        }
        Ok(())
    }

    pub(crate) fn load_state(&mut self, state: &SurpriseMapState) {
        self.surprise = state.values.clone();
        self.decay = state.decay;
        self.threshold = state.threshold;
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
    fn test_runtime_decay_update_changes_retention() {
        let grid = make_grid();
        let prev = random_patch_hvs(16, 1000);
        let curr = random_patch_hvs(16, 2000);

        let mut short_memory = SurpriseMap::new(grid.clone(), 0.5, 0.3);
        let mut long_memory = SurpriseMap::new(grid, 0.5, 0.3);
        short_memory.update(&curr, &prev);
        long_memory.update(&curr, &prev);
        long_memory.set_decay(0.95);

        short_memory.update(&curr, &curr);
        long_memory.update(&curr, &curr);

        assert_eq!(long_memory.decay(), 0.95);
        assert!(
            long_memory.mean_surprise() > short_memory.mean_surprise(),
            "higher decay should retain surprise evidence longer"
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
    fn checked_surprise_policy_rejects_non_finite_input_atomically() {
        let grid = make_grid();
        let mut sm = SurpriseMap::new(grid, 0.9, 0.3);
        sm.inject_cross_scale_error(&vec![0.5; 16], 0.5);
        let before = sm.attention_map().values;
        let decay_before = sm.decay();

        assert!(sm.dampen_checked(0, 0, f32::NAN).is_err());
        assert!(sm.set_decay_checked(f32::INFINITY).is_err());
        let mut malformed = vec![0.25; 16];
        malformed[7] = f32::NAN;
        assert!(
            sm.inject_cross_scale_error_checked(&malformed, 0.5)
                .is_err()
        );

        assert_eq!(sm.attention_map().values, before);
        assert_eq!(sm.decay(), decay_before);
    }

    #[test]
    fn checked_surprise_policy_rejects_out_of_range_input() {
        let grid = make_grid();
        let mut sm = SurpriseMap::new(grid, 0.9, 0.3);
        assert!(sm.dampen_checked(4, 0, 0.5).is_err());
        assert!(sm.dampen_checked(0, 0, 1.1).is_err());
        assert!(sm.set_decay_checked(1.0).is_err());
        assert!(sm.inject_cross_scale_error_checked(&[0.5], -0.1).is_err());
        assert_eq!(sm.max_surprise(), 0.0);
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
