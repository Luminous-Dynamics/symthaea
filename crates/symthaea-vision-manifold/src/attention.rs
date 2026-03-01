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
    /// Accumulated with exponential decay.
    pub fn update(
        &mut self,
        current_patches: &[ContinuousHV],
        previous_patches: &[ContinuousHV],
    ) {
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
        for i in 0..n {
            let sim = current_patches[i].similarity(&previous_patches[i]);
            let patch_surprise = (1.0 - sim).max(0.0);
            self.surprise[i] += patch_surprise;
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
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
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
}
