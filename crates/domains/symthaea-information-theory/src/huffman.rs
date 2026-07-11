// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Huffman source coding — the optimal prefix code, and Shannon's source-coding
//! bound `H ≤ L < H + 1`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

struct Node {
    weight: u64,
    left: Option<usize>,
    right: Option<usize>,
    leaf: Option<usize>,
}

/// The Huffman code length for each symbol, given its frequency/weight. A single
/// symbol gets length 1 by convention.
pub fn code_lengths(freqs: &[u64]) -> Vec<usize> {
    let n = freqs.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1];
    }
    let mut nodes: Vec<Node> = freqs
        .iter()
        .enumerate()
        .map(|(i, &w)| Node {
            weight: w,
            left: None,
            right: None,
            leaf: Some(i),
        })
        .collect();
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> =
        (0..n).map(|i| Reverse((nodes[i].weight, i))).collect();
    while heap.len() > 1 {
        let Reverse((w1, a)) = heap.pop().unwrap();
        let Reverse((w2, b)) = heap.pop().unwrap();
        let parent = nodes.len();
        nodes.push(Node {
            weight: w1 + w2,
            left: Some(a),
            right: Some(b),
            leaf: None,
        });
        heap.push(Reverse((w1 + w2, parent)));
    }
    let Reverse((_, root)) = heap.pop().unwrap();
    let mut lengths = vec![0usize; n];
    let mut stack = vec![(root, 0usize)];
    while let Some((idx, depth)) = stack.pop() {
        if let Some(leaf) = nodes[idx].leaf {
            lengths[leaf] = depth;
        } else {
            if let Some(l) = nodes[idx].left {
                stack.push((l, depth + 1));
            }
            if let Some(r) = nodes[idx].right {
                stack.push((r, depth + 1));
            }
        }
    }
    lengths
}

/// Expected code length `L = Σ pᵢ·lᵢ` (bits/symbol) for the given frequencies
/// and code lengths.
pub fn average_length(freqs: &[u64], lengths: &[usize]) -> f64 {
    let total: u64 = freqs.iter().sum();
    if total == 0 {
        return 0.0;
    }
    freqs
        .iter()
        .zip(lengths)
        .map(|(&f, &l)| f as f64 * l as f64)
        .sum::<f64>()
        / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::entropy;

    fn probs(freqs: &[u64]) -> Vec<f64> {
        let total: u64 = freqs.iter().sum();
        freqs.iter().map(|&f| f as f64 / total as f64).collect()
    }

    #[test]
    fn uniform_meets_entropy_exactly() {
        // Four equiprobable symbols → all length 2, L = H = 2.
        let freqs = [1, 1, 1, 1];
        let lens = code_lengths(&freqs);
        assert_eq!(lens, vec![2, 2, 2, 2]);
        assert!((average_length(&freqs, &lens) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn shannon_source_coding_bound() {
        // For any distribution, H ≤ L < H + 1.
        for freqs in [
            vec![5, 2, 1, 1],
            vec![8, 4, 2, 1, 1],
            vec![10, 1, 1, 1, 1, 1],
        ] {
            let lens = code_lengths(&freqs);
            let l = average_length(&freqs, &lens);
            let h = entropy(&probs(&freqs));
            assert!(l >= h - 1e-9, "L={l} < H={h}");
            assert!(l < h + 1.0, "L={l} ≥ H+1={}", h + 1.0);
        }
    }

    #[test]
    fn kraft_equality_for_complete_code() {
        // A Huffman code is a complete prefix code: Σ 2^{−lᵢ} = 1.
        let lens = code_lengths(&[5, 2, 1, 1]);
        let kraft: f64 = lens.iter().map(|&l| 2f64.powi(-(l as i32))).sum();
        assert!((kraft - 1.0).abs() < 1e-12, "kraft={kraft}");
    }
}
