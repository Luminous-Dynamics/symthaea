// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grid-based spatial hash for O(n×k) neighbor queries.
//!
//! Replaces O(n²) brute-force iteration for harmony resonance and
//! FEP gradient computation. Cell size equals harmony_range so
//! only a 3^D neighborhood needs checking.

use nalgebra::SVector;
use std::collections::HashMap;

/// Grid-based spatial hash for D-dimensional space.
///
/// Each cell is `cell_size` wide. Neighbors within `cell_size` distance
/// are guaranteed to be in the same cell or an adjacent cell (3^D total).
pub struct SpatialHash<const D: usize> {
    /// Cell width (used for external queries).
    pub cell_size: f64,
    inv_cell_size: f64,
    cells: HashMap<[i64; D], Vec<usize>>,
}

impl<const D: usize> SpatialHash<D> {
    /// Create a new spatial hash.
    ///
    /// `cell_size` should equal the maximum interaction range (e.g., `harmony_range`).
    /// This guarantees all potential neighbors are in adjacent cells.
    pub fn new(cell_size: f64) -> Self {
        assert!(cell_size > 0.0, "cell_size must be positive");
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            cells: HashMap::new(),
        }
    }

    /// Clear all entries. Call at the start of each tick before re-inserting.
    pub fn clear(&mut self) {
        self.cells.clear();
    }

    /// Insert an agent at the given position.
    pub fn insert(&mut self, index: usize, position: &SVector<f64, D>) {
        let key = self.cell_key(position);
        self.cells.entry(key).or_default().push(index);
    }

    /// Query all agent indices within the cell neighborhood of `position`.
    ///
    /// Returns indices in the same cell and all 3^D - 1 adjacent cells.
    /// Caller must still check actual distance — this is a broadphase filter.
    pub fn query_neighbors(&self, position: &SVector<f64, D>) -> Vec<usize> {
        let center = self.cell_key(position);
        let mut result = Vec::new();

        // Iterate over 3^D neighborhood
        self.visit_neighborhood(&center, 0, &mut [0i64; D], &mut result);
        result
    }

    /// Number of occupied cells.
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// Total number of stored entries.
    pub fn len(&self) -> usize {
        self.cells.values().map(|v| v.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    fn cell_key(&self, position: &SVector<f64, D>) -> [i64; D] {
        let mut key = [0i64; D];
        for i in 0..D {
            key[i] = (position[i] * self.inv_cell_size).floor() as i64;
        }
        key
    }

    fn visit_neighborhood(
        &self,
        center: &[i64; D],
        dim: usize,
        offset: &mut [i64; D],
        result: &mut Vec<usize>,
    ) {
        if dim == D {
            let mut key = *center;
            for i in 0..D {
                key[i] += offset[i];
            }
            if let Some(indices) = self.cells.get(&key) {
                result.extend_from_slice(indices);
            }
            return;
        }
        for d in -1..=1i64 {
            offset[dim] = d;
            self.visit_neighborhood(center, dim + 1, offset, result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_query_same_cell() {
        let mut hash = SpatialHash::<2>::new(10.0);
        hash.insert(0, &SVector::from([5.0, 5.0]));
        hash.insert(1, &SVector::from([7.0, 3.0]));

        let neighbors = hash.query_neighbors(&SVector::from([5.0, 5.0]));
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
    }

    #[test]
    fn query_adjacent_cells() {
        let mut hash = SpatialHash::<2>::new(10.0);
        // Agent at (5, 5) — cell (0, 0)
        hash.insert(0, &SVector::from([5.0, 5.0]));
        // Agent at (15, 5) — cell (1, 0), adjacent
        hash.insert(1, &SVector::from([15.0, 5.0]));

        let neighbors = hash.query_neighbors(&SVector::from([9.0, 5.0]));
        assert!(neighbors.contains(&0), "should find agent in same cell");
        assert!(neighbors.contains(&1), "should find agent in adjacent cell");
    }

    #[test]
    fn distant_agents_not_found() {
        let mut hash = SpatialHash::<2>::new(10.0);
        hash.insert(0, &SVector::from([5.0, 5.0]));
        // Agent at (55, 55) — cell (5, 5), far away
        hash.insert(1, &SVector::from([55.0, 55.0]));

        let neighbors = hash.query_neighbors(&SVector::from([5.0, 5.0]));
        assert!(neighbors.contains(&0));
        assert!(!neighbors.contains(&1), "distant agent should not appear");
    }

    #[test]
    fn clear_removes_all() {
        let mut hash = SpatialHash::<2>::new(10.0);
        hash.insert(0, &SVector::from([5.0, 5.0]));
        assert_eq!(hash.len(), 1);
        hash.clear();
        assert_eq!(hash.len(), 0);
        assert!(hash.is_empty());
    }

    #[test]
    fn works_in_3d() {
        let mut hash = SpatialHash::<3>::new(10.0);
        hash.insert(0, &SVector::from([5.0, 5.0, 5.0]));
        hash.insert(1, &SVector::from([15.0, 5.0, 5.0]));
        hash.insert(2, &SVector::from([55.0, 55.0, 55.0]));

        let neighbors = hash.query_neighbors(&SVector::from([5.0, 5.0, 5.0]));
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
        assert!(!neighbors.contains(&2));
    }

    #[test]
    fn negative_coordinates() {
        let mut hash = SpatialHash::<2>::new(10.0);
        hash.insert(0, &SVector::from([-5.0, -5.0]));
        hash.insert(1, &SVector::from([-15.0, -5.0]));

        let neighbors = hash.query_neighbors(&SVector::from([-5.0, -5.0]));
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
    }

    #[test]
    fn large_population() {
        let mut hash = SpatialHash::<2>::new(40.0);
        for i in 0..1000 {
            let x = (i as f64 * 7.3) % 200.0 - 100.0;
            let y = (i as f64 * 11.1) % 200.0 - 100.0;
            hash.insert(i, &SVector::from([x, y]));
        }
        assert_eq!(hash.len(), 1000);

        // Query should return a subset, not all 1000
        let neighbors = hash.query_neighbors(&SVector::from([0.0, 0.0]));
        assert!(neighbors.len() < 1000, "should not return all agents");
        assert!(!neighbors.is_empty(), "should find some agents near origin");
    }
}
