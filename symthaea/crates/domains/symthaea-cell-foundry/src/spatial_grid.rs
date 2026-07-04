// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Uniform 3D spatial grid for O(1)-amortized radius queries.
//!
//! Every neighbour/distance-based subsystem in this crate (chemical
//! diffusion's `neighbours()`, `form_synapses`, `form_gap_junctions`) used
//! to do a brute-force O(n) scan per query — O(n^2) total per day, the
//! wall that capped the cancer-defection experiment at 3,000 cells / 14
//! days this session. This grid buckets cells once per day and answers
//! "everything within radius R of this point" by scanning a fixed 27
//! neighbouring buckets instead of every other cell.
//!
//! `cell_size` must be >= every radius the grid is queried with — see
//! `crate::morphogenetic_consciousness::SPATIAL_GRID_CELL_SIZE`, which is
//! kept at the largest of the three radii actually in use. Given that,
//! a fixed 3x3x3 bucket scan is always sufficient: any point within
//! `radius <= cell_size` of the query centre must fall in the centre
//! bucket or one of its 26 direct neighbours.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct SpatialGrid {
    cell_size: f32,
    buckets: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    /// Build a grid from a position snapshot. `cell_size` must be >= every
    /// radius this grid will later be queried with.
    pub(crate) fn build(positions: &[[f32; 3]], cell_size: f32) -> Self {
        let mut buckets: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (i, pos) in positions.iter().enumerate() {
            buckets
                .entry(Self::bucket_key(pos, cell_size))
                .or_default()
                .push(i);
        }
        Self { cell_size, buckets }
    }

    /// The `cell_size` this grid was built with — callers should check
    /// `radius <= grid.cell_size()` before querying.
    pub(crate) fn cell_size(&self) -> f32 {
        self.cell_size
    }

    fn bucket_key(pos: &[f32; 3], cell_size: f32) -> (i32, i32, i32) {
        (
            (pos[0] / cell_size).floor() as i32,
            (pos[1] / cell_size).floor() as i32,
            (pos[2] / cell_size).floor() as i32,
        )
    }

    /// Indices of every point within `radius` of `center` (excluding
    /// `exclude`, if given). `radius` must be `<= self.cell_size()`.
    pub(crate) fn query_radius(
        &self,
        positions: &[[f32; 3]],
        center: &[f32; 3],
        radius: f32,
        exclude: Option<usize>,
    ) -> Vec<usize> {
        debug_assert!(
            radius <= self.cell_size,
            "query radius {radius} exceeds grid cell size {}",
            self.cell_size
        );
        let (cx, cy, cz) = Self::bucket_key(center, self.cell_size);
        let mut result = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(indices) = self.buckets.get(&(cx + dx, cy + dy, cz + dz)) else {
                        continue;
                    };
                    for &idx in indices {
                        if Some(idx) == exclude {
                            continue;
                        }
                        let p = &positions[idx];
                        let ddx = p[0] - center[0];
                        let ddy = p[1] - center[1];
                        let ddz = p[2] - center[2];
                        if (ddx * ddx + ddy * ddy + ddz * ddz).sqrt() < radius {
                            result.push(idx);
                        }
                    }
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn brute_force(positions: &[[f32; 3]], idx: usize, radius: f32) -> Vec<usize> {
        let pos = &positions[idx];
        let mut out: Vec<usize> = (0..positions.len())
            .filter(|&j| {
                if j == idx {
                    return false;
                }
                let p = &positions[j];
                let dx = p[0] - pos[0];
                let dy = p[1] - pos[1];
                let dz = p[2] - pos[2];
                (dx * dx + dy * dy + dz * dz).sqrt() < radius
            })
            .collect();
        out.sort_unstable();
        out
    }

    #[test]
    fn grid_matches_brute_force_across_random_configurations() {
        let mut rng = StdRng::seed_from_u64(42);
        for trial in 0..6 {
            let n = 40 + trial * 30;
            let positions: Vec<[f32; 3]> = (0..n)
                .map(|_| {
                    [
                        rng.gen_range(-1.0..1.0f32),
                        rng.gen_range(-1.0..1.0f32),
                        rng.gen_range(-1.0..1.0f32),
                    ]
                })
                .collect();
            let grid = SpatialGrid::build(&positions, 0.4);
            for &radius in &[0.1f32, 0.15, 0.4] {
                for idx in 0..n {
                    let mut grid_result =
                        grid.query_radius(&positions, &positions[idx], radius, Some(idx));
                    grid_result.sort_unstable();
                    let brute = brute_force(&positions, idx, radius);
                    assert_eq!(
                        grid_result, brute,
                        "mismatch at n={n}, radius={radius}, idx={idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn empty_grid_returns_empty() {
        let positions: Vec<[f32; 3]> = vec![];
        let grid = SpatialGrid::build(&positions, 0.4);
        let result = grid.query_radius(&positions, &[0.0, 0.0, 0.0], 0.1, None);
        assert!(result.is_empty());
    }

    #[test]
    fn query_excludes_self_when_asked() {
        let positions = vec![[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]];
        let grid = SpatialGrid::build(&positions, 0.4);
        let with_exclude = grid.query_radius(&positions, &positions[0], 0.4, Some(0));
        assert_eq!(with_exclude, vec![1]);
        let without_exclude = grid.query_radius(&positions, &positions[0], 0.4, None);
        let mut sorted = without_exclude.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1]);
    }
}
