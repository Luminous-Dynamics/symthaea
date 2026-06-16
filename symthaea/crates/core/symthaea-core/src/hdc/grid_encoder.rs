// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC grid encoder for 2D spatial reasoning.
//!
//! Encodes 2D grids (like ARC puzzles) into hyperdimensional vectors by binding
//! row, column, and color basis vectors. Supports rule extraction via input/output
//! binding and transformation algebra for common grid operations.
//!
//! Reference: Chollet (2019), "On the Measure of Intelligence", arXiv:1911.01547

use super::ContinuousHV;

/// Encodes 2D color grids into hyperdimensional vectors.
///
/// Uses pre-computed basis HVs for rows, columns, and colors.
/// Cell encoding: `bind(row_hv, col_hv, color_hv)`.
/// Grid encoding: `bundle(all_cells)`, L2-normalized.
pub struct GridEncoder {
    dim: usize,
    color_hvs: Vec<ContinuousHV>,
    row_hvs: Vec<ContinuousHV>,
    col_hvs: Vec<ContinuousHV>,
}

impl GridEncoder {
    /// Create a new grid encoder with pre-computed basis HVs.
    ///
    /// - `dim`: hypervector dimensionality
    /// - `max_rows`/`max_cols`: maximum grid dimensions
    /// - `num_colors`: number of distinct colors (0..num_colors)
    /// - `seed`: base RNG seed for deterministic HV generation
    pub fn new(dim: usize, max_rows: usize, max_cols: usize, num_colors: usize, seed: u64) -> Self {
        let color_hvs: Vec<ContinuousHV> = (0..num_colors)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(50_000 + i as u64)))
            .collect();
        let row_hvs: Vec<ContinuousHV> = (0..max_rows)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(60_000 + i as u64)))
            .collect();
        let col_hvs: Vec<ContinuousHV> = (0..max_cols)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(70_000 + i as u64)))
            .collect();
        Self {
            dim,
            color_hvs,
            row_hvs,
            col_hvs,
        }
    }

    /// Encode a single cell at (row, col) with the given color index.
    pub fn encode_cell(&self, row: usize, col: usize, color: usize) -> ContinuousHV {
        let row_hv = &self.row_hvs[row];
        let col_hv = &self.col_hvs[col];
        let color_hv = &self.color_hvs[color];
        row_hv.bind(col_hv).bind(color_hv)
    }

    /// Encode a full grid into a single HV. Bundles all cell encodings, then L2-normalizes.
    pub fn encode_grid(&self, grid: &[Vec<u8>]) -> ContinuousHV {
        let cells: Vec<ContinuousHV> = grid
            .iter()
            .enumerate()
            .flat_map(|(r, row)| {
                row.iter()
                    .enumerate()
                    .map(move |(c, &color)| self.encode_cell(r, c, color as usize))
            })
            .collect();
        let refs: Vec<&ContinuousHV> = cells.iter().collect();
        ContinuousHV::bundle(&refs).normalize()
    }

    /// Query the color information at a specific position by unbinding the position.
    ///
    /// ContinuousHV bind is approximately self-inverse, so binding again
    /// with the position HVs extracts the color component.
    pub fn query_position(&self, grid_hv: &ContinuousHV, row: usize, col: usize) -> ContinuousHV {
        let row_hv = &self.row_hvs[row];
        let col_hv = &self.col_hvs[col];
        grid_hv.bind(row_hv).bind(col_hv)
    }

    /// Identify the most likely color at a given position by querying and comparing
    /// against all color basis vectors.
    ///
    /// Returns `(color_index, similarity)`.
    pub fn identify_color(&self, grid_hv: &ContinuousHV, row: usize, col: usize) -> (usize, f32) {
        let query = self.query_position(grid_hv, row, col);
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, chv) in self.color_hvs.iter().enumerate() {
            let sim = query.similarity(chv);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
        (best_idx, best_sim)
    }

    /// Encode a transformation rule by binding input and output grid HVs.
    pub fn encode_rule(&self, input_hv: &ContinuousHV, output_hv: &ContinuousHV) -> ContinuousHV {
        input_hv.bind(output_hv)
    }

    /// Apply a rule to an input grid HV by binding the input with the rule.
    pub fn apply_rule(&self, input_hv: &ContinuousHV, rule_hv: &ContinuousHV) -> ContinuousHV {
        input_hv.bind(rule_hv)
    }

    /// Bundle multiple rule HVs into a consensus rule, then L2-normalize.
    pub fn bundle_rules(&self, rules: &[ContinuousHV]) -> ContinuousHV {
        let refs: Vec<&ContinuousHV> = rules.iter().collect();
        ContinuousHV::bundle(&refs).normalize()
    }

    /// The dimensionality of vectors produced by this encoder.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// The number of colors this encoder supports.
    pub fn num_colors(&self) -> usize {
        self.color_hvs.len()
    }

    /// Access a color basis HV by index.
    pub fn color_hv(&self, idx: usize) -> &ContinuousHV {
        &self.color_hvs[idx]
    }

    // ── Transformation algebra (pure grid operations) ──

    /// Translate a grid by (dx, dy), filling exposed cells with `fill` color.
    pub fn translate_grid(grid: &[Vec<u8>], dx: i32, dy: i32, fill: u8) -> Vec<Vec<u8>> {
        let rows = grid.len();
        if rows == 0 {
            return vec![];
        }
        let cols = grid[0].len();
        let mut out = vec![vec![fill; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                let sr = r as i32 - dy;
                let sc = c as i32 - dx;
                if sr >= 0 && sr < rows as i32 && sc >= 0 && sc < cols as i32 {
                    out[r][c] = grid[sr as usize][sc as usize];
                }
            }
        }
        out
    }

    /// Horizontal mirror (reflect across vertical axis).
    pub fn reflect_x(grid: &[Vec<u8>]) -> Vec<Vec<u8>> {
        grid.iter()
            .map(|row| row.iter().rev().copied().collect())
            .collect()
    }

    /// Vertical mirror (reflect across horizontal axis).
    pub fn reflect_y(grid: &[Vec<u8>]) -> Vec<Vec<u8>> {
        grid.iter().rev().cloned().collect()
    }

    /// Clockwise 90-degree rotation.
    pub fn rotate_90(grid: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let rows = grid.len();
        if rows == 0 {
            return vec![];
        }
        let cols = grid[0].len();
        let mut out = vec![vec![0u8; rows]; cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c][rows - 1 - r] = grid[r][c];
            }
        }
        out
    }

    /// Replace all occurrences of `from` color with `to` color.
    pub fn color_replace(grid: &[Vec<u8>], from: u8, to: u8) -> Vec<Vec<u8>> {
        grid.iter()
            .map(|row| {
                row.iter()
                    .map(|&c| if c == from { to } else { c })
                    .collect()
            })
            .collect()
    }

    /// Fill a rectangular region with a given color (inclusive bounds).
    pub fn fill_region(
        grid: &[Vec<u8>],
        top: usize,
        left: usize,
        bottom: usize,
        right: usize,
        color: u8,
    ) -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = grid.to_vec();
        for r in top..=bottom.min(out.len().saturating_sub(1)) {
            for c in left..=right.min(out[r].len().saturating_sub(1)) {
                out[r][c] = color;
            }
        }
        out
    }

    /// Experimental: pure algebraic translation operator as an HV.
    ///
    /// Constructs an operator that represents the shift (dx, dy) by binding
    /// permuted row/col basis vectors. This is a research-frontier approach —
    /// the operator is valid but recovery fidelity depends on dimensionality.
    pub fn translation_operator(&self, dx: i32, dy: i32) -> ContinuousHV {
        // Build operator by binding shifted-position pairs.
        // For each (r, c) that maps to (r+dy, c+dx) within bounds,
        // encode the source→destination binding.
        let max_rows = self.row_hvs.len();
        let max_cols = self.col_hvs.len();
        let mut pairs: Vec<ContinuousHV> = Vec::new();
        for r in 0..max_rows {
            let dr = r as i32 + dy;
            if dr < 0 || dr >= max_rows as i32 {
                continue;
            }
            for c in 0..max_cols {
                let dc = c as i32 + dx;
                if dc < 0 || dc >= max_cols as i32 {
                    continue;
                }
                // bind(source_pos, dest_pos)
                let src = self.row_hvs[r].bind(&self.col_hvs[c]);
                let dst = self.row_hvs[dr as usize].bind(&self.col_hvs[dc as usize]);
                pairs.push(src.bind(&dst));
            }
        }
        if pairs.is_empty() {
            return ContinuousHV::random(self.dim, 0);
        }
        let refs: Vec<&ContinuousHV> = pairs.iter().collect();
        ContinuousHV::bundle(&refs).normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoder() -> GridEncoder {
        GridEncoder::new(256, 5, 5, 6, 12345)
    }

    fn sample_grid() -> Vec<Vec<u8>> {
        vec![
            vec![0, 1, 2, 0, 0],
            vec![0, 0, 0, 0, 0],
            vec![3, 3, 3, 0, 0],
            vec![0, 0, 0, 4, 4],
            vec![0, 0, 0, 0, 5],
        ]
    }

    #[test]
    fn test_self_similarity() {
        let enc = make_encoder();
        let grid = sample_grid();
        let hv = enc.encode_grid(&grid);
        let sim = hv.similarity(&hv);
        assert!(
            (sim - 1.0).abs() < 0.01,
            "Self-similarity should be ~1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_different_grids_dissimilar() {
        let enc = make_encoder();
        let g1 = sample_grid();
        let g2 = vec![
            vec![5, 4, 3, 2, 1],
            vec![1, 2, 3, 4, 5],
            vec![0, 0, 0, 0, 0],
            vec![5, 5, 5, 5, 5],
            vec![1, 1, 1, 1, 1],
        ];
        let hv1 = enc.encode_grid(&g1);
        let hv2 = enc.encode_grid(&g2);
        let sim = hv1.similarity(&hv2);
        // At dim=256, bundled grids with shared colors can have moderate similarity.
        // Key property: different grids are less similar than identical ones.
        let self_sim = hv1.similarity(&hv1);
        assert!(
            sim < self_sim - 0.1,
            "Different grids should be less similar than self: got {} vs self {}",
            sim,
            self_sim
        );
    }

    #[test]
    fn test_single_cell_change_moderate_similarity() {
        let enc = make_encoder();
        let g1 = sample_grid();
        let mut g2 = g1.clone();
        g2[2][2] = 5; // change one cell
        let hv1 = enc.encode_grid(&g1);
        let hv2 = enc.encode_grid(&g2);
        let sim = hv1.similarity(&hv2);
        assert!(
            sim > 0.5 && sim < 1.0,
            "Single-cell change should give moderate similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_cell_encoding_determinism() {
        let enc = make_encoder();
        let c1 = enc.encode_cell(1, 2, 3);
        let c2 = enc.encode_cell(1, 2, 3);
        let sim = c1.similarity(&c2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Same cell should produce identical HV, got similarity {}",
            sim
        );
    }

    #[test]
    fn test_position_query_recovers_color() {
        let enc = make_encoder();
        let grid = vec![vec![3]]; // single cell at (0,0) with color 3
        let hv = enc.encode_grid(&grid);
        let (color, sim) = enc.identify_color(&hv, 0, 0);
        assert_eq!(color, 3, "Should recover color 3, got {}", color);
        assert!(
            sim > 0.0,
            "Color similarity should be positive, got {}",
            sim
        );
    }

    #[test]
    fn test_rule_consistency() {
        let enc = make_encoder();
        let input = sample_grid();
        let output = GridEncoder::color_replace(&input, 0, 1);
        let in_hv = enc.encode_grid(&input);
        let out_hv = enc.encode_grid(&output);
        let rule1 = enc.encode_rule(&in_hv, &out_hv);
        let rule2 = enc.encode_rule(&in_hv, &out_hv);
        let sim = rule1.similarity(&rule2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Same rule should be deterministic, got similarity {}",
            sim
        );
    }

    #[test]
    fn test_translate() {
        let grid = vec![vec![1, 2], vec![3, 4]];
        let result = GridEncoder::translate_grid(&grid, 1, 0, 0);
        assert_eq!(result, vec![vec![0, 1], vec![0, 3]]);
    }

    #[test]
    fn test_reflect_x() {
        let grid = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let result = GridEncoder::reflect_x(&grid);
        assert_eq!(result, vec![vec![3, 2, 1], vec![6, 5, 4]]);
    }

    #[test]
    fn test_reflect_y() {
        let grid = vec![vec![1, 2], vec![3, 4]];
        let result = GridEncoder::reflect_y(&grid);
        assert_eq!(result, vec![vec![3, 4], vec![1, 2]]);
    }

    #[test]
    fn test_rotate_90() {
        let grid = vec![vec![1, 2], vec![3, 4]];
        let result = GridEncoder::rotate_90(&grid);
        // Clockwise 90: row 0 becomes last col, row 1 becomes first col
        assert_eq!(result, vec![vec![3, 1], vec![4, 2]]);
    }

    #[test]
    fn test_color_replace() {
        let grid = vec![vec![0, 1, 0], vec![2, 0, 2]];
        let result = GridEncoder::color_replace(&grid, 0, 5);
        assert_eq!(result, vec![vec![5, 1, 5], vec![2, 5, 2]]);
    }

    #[test]
    fn test_fill_region() {
        let grid = vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]];
        let result = GridEncoder::fill_region(&grid, 0, 1, 1, 2, 7);
        assert_eq!(result, vec![vec![0, 7, 7], vec![0, 7, 7], vec![0, 0, 0]]);
    }

    #[test]
    fn test_bundle_rules_normalization() {
        let enc = make_encoder();
        let g1 = sample_grid();
        let g2 = GridEncoder::color_replace(&g1, 0, 1);
        let g3 = GridEncoder::color_replace(&g1, 0, 2);
        let hv1 = enc.encode_grid(&g1);
        let hv2 = enc.encode_grid(&g2);
        let hv3 = enc.encode_grid(&g3);
        let r1 = enc.encode_rule(&hv1, &hv2);
        let r2 = enc.encode_rule(&hv1, &hv3);
        let bundled = enc.bundle_rules(&[r1, r2]);
        let self_sim = bundled.similarity(&bundled);
        assert!(
            (self_sim - 1.0).abs() < 0.01,
            "Bundled rules should be normalized, self-sim = {}",
            self_sim
        );
    }

    #[test]
    fn test_translation_operator_valid() {
        let enc = make_encoder();
        let op = enc.translation_operator(1, 0);
        let self_sim = op.similarity(&op);
        assert!(
            (self_sim - 1.0).abs() < 0.01,
            "Translation operator should be a valid normalized HV, self-sim = {}",
            self_sim
        );
    }

    #[test]
    fn test_dimension_accessor() {
        let enc = GridEncoder::new(512, 3, 3, 4, 42);
        assert_eq!(enc.dimension(), 512);
    }
}
