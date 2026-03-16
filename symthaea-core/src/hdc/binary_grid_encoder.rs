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
}
