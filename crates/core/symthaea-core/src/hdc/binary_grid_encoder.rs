// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Binary HDC grid encoder for 2D spatial reasoning.
//!
//! Uses BinaryHV (XOR binding) instead of ContinuousHV for rule extraction.
//! Key advantage: XOR is self-inverse, so `unbind(bind(A, B), B) = A` exactly.
//! This enables clean rule generalization across different inputs — the critical
//! operation for ARC-style abstract reasoning tasks.
//!
//! The ContinuousHV grid encoder fails at rule transfer because element-wise
//! multiply is NOT self-inverse: `A * (A * B) = A² * B ≠ B`.

use super::BinaryHV;

/// An input-output grid pair: `(input_grid, output_grid)` where each grid is `Vec<Vec<u8>>`.
pub type GridPair = (Vec<Vec<u8>>, Vec<Vec<u8>>);

/// Encodes 2D color grids into BinaryHV vectors using XOR binding.
///
/// Cell encoding: `row_hv ⊕ col_hv ⊕ color_hv`.
/// Grid encoding: `majority_vote(all_cells)`.
/// Rule encoding: `input_hv ⊕ output_hv` (self-inverse).
pub struct BinaryGridEncoder {
    color_hvs: Vec<BinaryHV>,
    row_hvs: Vec<BinaryHV>,
    col_hvs: Vec<BinaryHV>,
}

impl BinaryGridEncoder {
    /// Create a new binary grid encoder with pre-computed basis HVs.
    ///
    /// - `max_rows`/`max_cols`: maximum grid dimensions
    /// - `num_colors`: number of distinct colors (0..num_colors)
    /// - `seed`: base RNG seed for deterministic HV generation
    pub fn new(max_rows: usize, max_cols: usize, num_colors: usize, seed: u64) -> Self {
        let color_hvs: Vec<BinaryHV> = (0..num_colors)
            .map(|i| BinaryHV::random(seed.wrapping_add(50_000 + i as u64)))
            .collect();
        let row_hvs: Vec<BinaryHV> = (0..max_rows)
            .map(|i| BinaryHV::random(seed.wrapping_add(60_000 + i as u64)))
            .collect();
        let col_hvs: Vec<BinaryHV> = (0..max_cols)
            .map(|i| BinaryHV::random(seed.wrapping_add(70_000 + i as u64)))
            .collect();
        Self {
            color_hvs,
            row_hvs,
            col_hvs,
        }
    }

    /// Encode a single cell at (row, col) with the given color index.
    pub fn encode_cell(&self, row: usize, col: usize, color: usize) -> BinaryHV {
        self.row_hvs[row]
            .bind(&self.col_hvs[col])
            .bind(&self.color_hvs[color])
    }

    /// Encode a full grid into a single BinaryHV via majority-vote bundling.
    pub fn encode_grid(&self, grid: &[Vec<u8>]) -> BinaryHV {
        let cells: Vec<BinaryHV> = grid
            .iter()
            .enumerate()
            .flat_map(|(r, row)| {
                row.iter()
                    .enumerate()
                    .map(move |(c, &color)| self.encode_cell(r, c, color as usize))
            })
            .collect();
        BinaryHV::bundle(&cells)
    }

    /// Encode a transformation rule by XOR-binding input and output grid HVs.
    ///
    /// Because XOR is self-inverse: `rule ⊕ input = output` and `rule ⊕ output = input`.
    pub fn encode_rule(&self, input_hv: &BinaryHV, output_hv: &BinaryHV) -> BinaryHV {
        input_hv.bind(output_hv)
    }

    /// Apply a rule to an input grid HV: `input ⊕ rule`.
    ///
    /// With XOR binding: `input ⊕ (input ⊕ output) = output` (exact recovery).
    pub fn apply_rule(&self, input_hv: &BinaryHV, rule_hv: &BinaryHV) -> BinaryHV {
        input_hv.bind(rule_hv)
    }

    /// Infer the input from an output and a rule: `output ⊕ rule`.
    ///
    /// With XOR binding: `output ⊕ (input ⊕ output) = input` (exact recovery).
    /// This is the key operation for abductive reasoning.
    pub fn infer_input(&self, output_hv: &BinaryHV, rule_hv: &BinaryHV) -> BinaryHV {
        output_hv.bind(rule_hv)
    }

    /// Bundle multiple rule HVs into a consensus rule via majority vote.
    pub fn bundle_rules(&self, rules: &[BinaryHV]) -> BinaryHV {
        BinaryHV::bundle(rules)
    }

    /// The number of colors this encoder supports.
    pub fn num_colors(&self) -> usize {
        self.color_hvs.len()
    }

    /// Access a color basis HV by index.
    pub fn color_hv(&self, idx: usize) -> &BinaryHV {
        &self.color_hvs[idx]
    }

    /// Decode the color at a single cell from a (possibly bundled) grid HV.
    ///
    /// Standard VSA "cleanup memory" retrieval: unbind the cell's row/col
    /// keys from the grid HV — `grid_hv ⊕ row_hv[r] ⊕ col_hv[c]` — which
    /// recovers a noisy version of that cell's color HV (noisy because the
    /// grid HV is a majority-vote bundle over every cell, not a single
    /// binding), then cleans it up via nearest-neighbor against the color
    /// codebook. Ties (or an empty codebook) resolve to color 0.
    pub fn decode_cell(&self, grid_hv: &BinaryHV, row: usize, col: usize) -> usize {
        let unbound = grid_hv.bind(&self.row_hvs[row]).bind(&self.col_hvs[col]);
        self.color_hvs
            .iter()
            .enumerate()
            .map(|(i, c)| (i, unbound.similarity(c)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Decode a full `rows`×`cols` grid from a grid HV via per-cell cleanup.
    ///
    /// This is genuinely lossy: majority-vote bundling degrades as the
    /// number of cells grows, so per-cell decode accuracy falls with grid
    /// size — this is the honest mechanism behind "strict" (pixel-perfect)
    /// scoring being much harder than 2-AFC similarity scoring.
    pub fn decode_grid(&self, grid_hv: &BinaryHV, rows: usize, cols: usize) -> Vec<Vec<u8>> {
        (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| self.decode_cell(grid_hv, r, c) as u8)
                    .collect()
            })
            .collect()
    }

    /// Fraction of cells where `predicted` and `actual` agree.
    /// Grids are compared cell-by-cell up to the smaller of the two shapes;
    /// mismatched dimensions count every extra cell as wrong.
    pub fn grid_accuracy(predicted: &[Vec<u8>], actual: &[Vec<u8>]) -> f32 {
        let rows = predicted.len().max(actual.len());
        let mut total = 0usize;
        let mut correct = 0usize;
        for r in 0..rows {
            let p_row = predicted.get(r).map(Vec::as_slice).unwrap_or(&[]);
            let a_row = actual.get(r).map(Vec::as_slice).unwrap_or(&[]);
            let cols = p_row.len().max(a_row.len());
            for c in 0..cols {
                total += 1;
                if p_row.get(c) == a_row.get(c) {
                    correct += 1;
                }
            }
        }
        if total == 0 {
            1.0
        } else {
            correct as f32 / total as f32
        }
    }

    /// Whether `predicted` and `actual` match exactly, cell for cell.
    pub fn grid_exact_match(predicted: &[Vec<u8>], actual: &[Vec<u8>]) -> bool {
        predicted == actual
    }

    // ========================================================================
    // Rule Discovery (Program Synthesis)
    // ========================================================================

    /// Discover a transformation rule from multiple input/output examples.
    ///
    /// Unlike `encode_rule` (which memorizes a single pair), this method
    /// extracts a *generalizable* rule by:
    /// 1. Computing per-pair rules via XOR
    /// 2. Bundling them via majority vote (noise reduction)
    /// 3. Measuring rule consistency across pairs
    /// 4. Testing generalization on held-out examples
    ///
    /// Returns a `DiscoveredRule` with the consensus rule, consistency score,
    /// and generalization estimate.
    pub fn discover_rule(&self, examples: &[GridPair]) -> DiscoveredRule {
        if examples.is_empty() {
            return DiscoveredRule::empty();
        }
        if examples.len() == 1 {
            let hv_in = self.encode_grid(&examples[0].0);
            let hv_out = self.encode_grid(&examples[0].1);
            let rule = self.encode_rule(&hv_in, &hv_out);
            return DiscoveredRule {
                rule_hv: rule,
                consistency: 1.0,
                generalization_score: 0.0, // can't estimate from 1 example
                num_examples: 1,
            };
        }

        // 1. Compute per-pair rules
        let pair_rules: Vec<BinaryHV> = examples
            .iter()
            .map(|(input, output)| {
                let hv_in = self.encode_grid(input);
                let hv_out = self.encode_grid(output);
                self.encode_rule(&hv_in, &hv_out)
            })
            .collect();

        // 2. Consensus rule via majority vote
        let consensus = self.bundle_rules(&pair_rules);

        // 3. Rule consistency: average pairwise similarity of individual rules
        let mut consistency_sum = 0.0f32;
        let mut consistency_count = 0u32;
        for i in 0..pair_rules.len() {
            for j in (i + 1)..pair_rules.len() {
                consistency_sum += pair_rules[i].similarity(&pair_rules[j]);
                consistency_count += 1;
            }
        }
        let consistency = if consistency_count > 0 {
            consistency_sum / consistency_count as f32
        } else {
            1.0
        };

        // 4. Leave-one-out generalization estimate
        let mut gen_score = 0.0f32;
        for hold_out in 0..examples.len() {
            // Train on all except hold_out
            let train_rules: Vec<BinaryHV> = pair_rules
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != hold_out)
                .map(|(_, r)| *r)
                .collect();
            let train_consensus = self.bundle_rules(&train_rules);

            // Test: apply to held-out input, compare to held-out output
            let hv_test_in = self.encode_grid(&examples[hold_out].0);
            let hv_test_out = self.encode_grid(&examples[hold_out].1);
            let predicted = self.apply_rule(&hv_test_in, &train_consensus);
            gen_score += predicted.similarity(&hv_test_out);
        }
        let generalization_score = gen_score / examples.len() as f32;

        DiscoveredRule {
            rule_hv: consensus,
            consistency,
            generalization_score,
            num_examples: examples.len(),
        }
    }

    /// Classify which transformation type best matches a set of examples.
    ///
    /// Given a library of named prototype rules (learned from prior tasks),
    /// find the closest matching rule via HDC similarity. This enables
    /// transfer learning: "this looks like a color replacement" even on
    /// novel inputs.
    pub fn classify_rule(
        &self,
        observed_rule: &BinaryHV,
        prototypes: &[(String, BinaryHV)],
    ) -> Option<(String, f32)> {
        prototypes
            .iter()
            .map(|(name, proto)| (name.clone(), observed_rule.similarity(proto)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Apply a discovered rule to a novel input and measure confidence.
    ///
    /// Returns (predicted_output_hv, confidence) where confidence is
    /// the rule's generalization_score weighted by how similar the novel
    /// input is to the training distribution.
    pub fn apply_discovered_rule(
        &self,
        input: &[Vec<u8>],
        rule: &DiscoveredRule,
        training_inputs: &[Vec<Vec<u8>>],
    ) -> (BinaryHV, f32) {
        let hv_in = self.encode_grid(input);
        let predicted = self.apply_rule(&hv_in, &rule.rule_hv);

        // Confidence: generalization score × input similarity to training distribution
        let input_similarity = if training_inputs.is_empty() {
            0.5 // no prior → moderate confidence
        } else {
            let train_hvs: Vec<BinaryHV> = training_inputs
                .iter()
                .map(|g| self.encode_grid(g))
                .collect();
            let centroid = BinaryHV::bundle(&train_hvs);
            hv_in.similarity(&centroid).max(0.0)
        };

        let confidence = rule.generalization_score * (0.5 + 0.5 * input_similarity);
        (predicted, confidence)
    }
}

/// Result of rule discovery from multiple examples.
#[derive(Debug, Clone)]
pub struct DiscoveredRule {
    /// The consensus rule HV (majority-vote of per-pair rules).
    pub rule_hv: BinaryHV,
    /// Consistency score: how similar are individual pair rules to each other?
    /// High consistency (>0.7) suggests a single consistent transformation.
    /// Low consistency (<0.3) suggests multiple different transformations mixed.
    pub consistency: f32,
    /// Leave-one-out generalization estimate: how well does the rule
    /// transfer to held-out examples? This is the honest measure of
    /// whether the rule genuinely generalizes vs just memorizes.
    pub generalization_score: f32,
    /// Number of training examples used.
    pub num_examples: usize,
}

impl DiscoveredRule {
    /// Empty rule (no examples).
    fn empty() -> Self {
        Self {
            rule_hv: BinaryHV::random(0),
            consistency: 0.0,
            generalization_score: 0.0,
            num_examples: 0,
        }
    }

    /// Whether this rule is likely to generalize (heuristic threshold).
    pub fn is_reliable(&self) -> bool {
        self.num_examples >= 2 && self.consistency > 0.4 && self.generalization_score > 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoder() -> BinaryGridEncoder {
        BinaryGridEncoder::new(5, 5, 6, 12345)
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
        assert!(
            sim < 0.75,
            "Different grids should have moderate-to-low similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_xor_rule_self_inverse() {
        let enc = make_encoder();
        let g1 = sample_grid();
        let g2 = vec![
            vec![1, 2, 3, 1, 1],
            vec![1, 1, 1, 1, 1],
            vec![4, 4, 4, 1, 1],
            vec![1, 1, 1, 5, 5],
            vec![1, 1, 1, 1, 0],
        ];
        let hv1 = enc.encode_grid(&g1);
        let hv2 = enc.encode_grid(&g2);

        // Rule = input ⊕ output
        let rule = enc.encode_rule(&hv1, &hv2);

        // Apply rule to input should recover output
        let predicted_output = enc.apply_rule(&hv1, &rule);
        let sim = predicted_output.similarity(&hv2);
        assert!(
            sim > 0.9,
            "XOR rule applied to same input should recover output: sim={}",
            sim
        );

        // Infer input from output should recover input
        let inferred_input = enc.infer_input(&hv2, &rule);
        let sim_in = inferred_input.similarity(&hv1);
        assert!(
            sim_in > 0.9,
            "XOR rule inversion should recover input: sim={}",
            sim_in
        );
    }

    #[test]
    fn test_rule_generalizes_across_inputs() {
        // This is the critical test: learn a rule from pair (A→B),
        // apply it to novel input C → should approximate D.
        let enc = make_encoder();

        // Training pair: color_replace(0 → 1)
        let g_in = sample_grid();
        let g_out: Vec<Vec<u8>> = g_in
            .iter()
            .map(|row| row.iter().map(|&c| if c == 0 { 1 } else { c }).collect())
            .collect();
        let hv_in = enc.encode_grid(&g_in);
        let hv_out = enc.encode_grid(&g_out);
        let rule = enc.encode_rule(&hv_in, &hv_out);

        // Novel input: different grid with same transform
        let g_test = vec![vec![0, 2, 0], vec![3, 0, 3], vec![0, 4, 0]];
        let g_expected: Vec<Vec<u8>> = g_test
            .iter()
            .map(|row| row.iter().map(|&c| if c == 0 { 1 } else { c }).collect())
            .collect();
        // Note: BinaryGridEncoder needs to handle different grid sizes via the
        // same max_rows/max_cols. The test grid is smaller, so it encodes fine.
        let enc3 = BinaryGridEncoder::new(3, 3, 6, 12345);
        let hv_test = enc3.encode_grid(&g_test);
        let hv_expected = enc3.encode_grid(&g_expected);

        // Learn rule from 3x3 pair
        let hv_in3: Vec<Vec<u8>> = vec![vec![0, 1, 2], vec![3, 0, 0], vec![0, 5, 0]];
        let hv_out3: Vec<Vec<u8>> = hv_in3
            .iter()
            .map(|row| row.iter().map(|&c| if c == 0 { 1 } else { c }).collect())
            .collect();
        let enc3_in = enc3.encode_grid(&hv_in3);
        let enc3_out = enc3.encode_grid(&hv_out3);
        let rule3 = enc3.encode_rule(&enc3_in, &enc3_out);

        // Apply rule to test input
        let predicted = enc3.apply_rule(&hv_test, &rule3);
        let sim = predicted.similarity(&hv_expected);

        // With BinaryHV XOR, the rule from one pair won't perfectly generalize
        // to another (bundling noise), but should show positive signal
        // compared to a random baseline.
        let random_hv = BinaryHV::random(99999);
        let random_sim = random_hv.similarity(&hv_expected);
        assert!(
            sim > random_sim,
            "Rule application should beat random baseline: predicted_sim={}, random_sim={}",
            sim,
            random_sim
        );
    }

    #[test]
    fn test_bundle_rules_consensus() {
        let enc = make_encoder();
        let g1 = sample_grid();
        let g2: Vec<Vec<u8>> = g1
            .iter()
            .map(|row| row.iter().map(|&c| if c == 0 { 1 } else { c }).collect())
            .collect();
        let hv1 = enc.encode_grid(&g1);
        let hv2 = enc.encode_grid(&g2);

        let r1 = enc.encode_rule(&hv1, &hv2);
        let r2 = enc.encode_rule(&hv1, &hv2); // same rule
        let bundled = enc.bundle_rules(&[r1, r2]);

        // Bundled identical rules should equal either one
        let sim = bundled.similarity(&r1);
        assert!(
            sim > 0.99,
            "Bundle of identical rules should match: sim={}",
            sim
        );
    }

    #[test]
    fn test_abductive_inference() {
        let enc = make_encoder();
        let g_in = sample_grid();
        let g_out: Vec<Vec<u8>> = g_in
            .iter()
            .map(|row| row.iter().map(|&c| if c == 0 { 1 } else { c }).collect())
            .collect();
        let hv_in = enc.encode_grid(&g_in);
        let hv_out = enc.encode_grid(&g_out);
        let rule = enc.encode_rule(&hv_in, &hv_out);

        // Infer input from output (abduction)
        let inferred = enc.infer_input(&hv_out, &rule);
        let sim = inferred.similarity(&hv_in);
        assert!(
            sim > 0.9,
            "Abductive inference should recover input: sim={}",
            sim
        );
    }

    #[test]
    fn test_decode_grid_recovers_small_grid() {
        // Small grid, few cells -> majority-vote bundling noise is low ->
        // decode should recover most/all cells correctly.
        let enc = BinaryGridEncoder::new(3, 3, 6, 777);
        let grid = vec![vec![0, 1, 2], vec![3, 0, 1], vec![2, 3, 0]];
        let hv = enc.encode_grid(&grid);
        let decoded = enc.decode_grid(&hv, 3, 3);
        let acc = BinaryGridEncoder::grid_accuracy(&decoded, &grid);
        assert!(
            acc > 0.8,
            "Small-grid decode accuracy should be high, got {}",
            acc
        );
    }

    #[test]
    fn test_decode_accuracy_degrades_with_grid_size() {
        // More cells bundled together -> more majority-vote noise -> lower
        // per-cell decode accuracy. This is the honest mechanism behind
        // "strict scoring is much harder than 2-AFC" claims.
        let small_enc = BinaryGridEncoder::new(3, 3, 6, 42);
        let small_grid = vec![vec![1, 2, 3], vec![4, 5, 0], vec![1, 2, 3]];
        let small_hv = small_enc.encode_grid(&small_grid);
        let small_acc =
            BinaryGridEncoder::grid_accuracy(&small_enc.decode_grid(&small_hv, 3, 3), &small_grid);

        let big_enc = BinaryGridEncoder::new(15, 15, 6, 42);
        let big_grid: Vec<Vec<u8>> = (0..15)
            .map(|r| (0..15).map(|c| ((r * 15 + c) % 6) as u8).collect())
            .collect();
        let big_hv = big_enc.encode_grid(&big_grid);
        let big_acc =
            BinaryGridEncoder::grid_accuracy(&big_enc.decode_grid(&big_hv, 15, 15), &big_grid);

        assert!(
            small_acc >= big_acc,
            "Decode accuracy should not improve with more cells: small={}, big={}",
            small_acc,
            big_acc
        );
    }

    #[test]
    fn test_grid_accuracy_identical() {
        let grid = sample_grid();
        assert_eq!(BinaryGridEncoder::grid_accuracy(&grid, &grid), 1.0);
    }

    #[test]
    fn test_grid_accuracy_partial() {
        let a = vec![vec![0u8, 1], vec![2, 3]];
        let b = vec![vec![0u8, 1], vec![9, 3]]; // 1 of 4 cells differ
        assert!((BinaryGridEncoder::grid_accuracy(&a, &b) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_grid_exact_match() {
        let grid = sample_grid();
        assert!(BinaryGridEncoder::grid_exact_match(&grid, &grid));
        let mut other = grid.clone();
        other[0][0] = 9;
        assert!(!BinaryGridEncoder::grid_exact_match(&grid, &other));
    }
}
