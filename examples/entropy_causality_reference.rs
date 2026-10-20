//! A dependency-free finite-state reference experiment for entropy vs. causal structure.
//!
//! Run with: cargo run --example entropy_causality_reference --release
//! Test with: cargo test --example entropy_causality_reference
//!
//! This is deliberately NOT an IIT Phi implementation. It computes Shannon
//! statistics and a transparent causal-cut sensitivity diagnostic so that
//! entropy-only and interaction-sensitive baselines can be compared on exact
//! small transition models before any scalable HDC proxy is trusted.

use std::f64::consts::LN_2;

const MAX_NODES: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Copy { from: usize },
    Xor { left: usize, right: usize },
    And { left: usize, right: usize },
    Constant(bool),
}

impl Rule {
    fn sources(self) -> [Option<usize>; 2] {
        match self {
            Rule::Copy { from } => [Some(from), None],
            Rule::Xor { left, right } | Rule::And { left, right } => {
                [Some(left), Some(right)]
            }
            Rule::Constant(_) => [None, None],
        }
    }

    fn evaluate(self, state: usize) -> bool {
        let bit = |index: usize| ((state >> index) & 1) != 0;
        match self {
            Rule::Copy { from } => bit(from),
            Rule::Xor { left, right } => bit(left) ^ bit(right),
            Rule::And { left, right } => bit(left) & bit(right),
            Rule::Constant(value) => value,
        }
    }
}

#[derive(Clone, Debug)]
struct Network {
    /// One synchronous update rule per target node.
    rules: Vec<Rule>,
    /// Independent per-node output-flip probability, constrained to [0, 0.5].
    flip_probability: f64,
}

impl Network {
    fn new(rules: Vec<Rule>, flip_probability: f64) -> Self {
        assert!(!rules.is_empty(), "network must have at least one node");
        assert!(rules.len() <= MAX_NODES, "exhaustive benchmark is capped at {MAX_NODES} nodes");
        assert!(
            flip_probability.is_finite() && (0.0..=0.5).contains(&flip_probability),
            "flip probability must be finite and in [0, 0.5]"
        );

        for (target, rule) in rules.iter().copied().enumerate() {
            for source in rule.sources().into_iter().flatten() {
                assert!(source < rules.len(), "node {target} references invalid source {source}");
            }
        }

        Self { rules, flip_probability }
    }

    fn node_count(&self) -> usize {
        self.rules.len()
    }

    fn state_count(&self) -> usize {
        1usize << self.node_count()
    }

    fn probability_of_one(&self, previous_state: usize, target: usize, cut_mask: Option<usize>) -> f64 {
        let rule = self.rules[target];
        let crosses_cut = cut_mask.is_some_and(|mask| {
            let target_side = (mask >> target) & 1;
            rule.sources()
                .into_iter()
                .flatten()
                .any(|source| ((mask >> source) & 1) != target_side)
        });

        // A severed cross-partition dependency is replaced by an independent
        // fair binary input. This is a diagnostic intervention, not IIT's
        // formal partition operation.
        let unnoised_probability = if crosses_cut {
            0.5
        } else if rule.evaluate(previous_state) {
            1.0
        } else {
            0.0
        };

        let noise = self.flip_probability;
        noise + (1.0 - 2.0 * noise) * unnoised_probability
    }

    /// Exact conditional distribution P(X[t+1] = next | X[t] = previous).
    fn transition_row(&self, previous: usize, cut_mask: Option<usize>) -> Vec<f64> {
        let states = self.state_count();
        let mut row = vec![0.0; states];

        for next in 0..states {
            let mut probability = 1.0;
            for target in 0..self.node_count() {
                let p_one = self.probability_of_one(previous, target, cut_mask);
                probability *= if ((next >> target) & 1) != 0 {
                    p_one
                } else {
                    1.0 - p_one
                };
            }
            row[next] = probability;
        }

        row
    }

    fn transition_matrix(&self, cut_mask: Option<usize>) -> Vec<Vec<f64>> {
        (0..self.state_count())
            .map(|previous| self.transition_row(previous, cut_mask))
            .collect()
    }

    fn relabel(&self, old_to_new: &[usize]) -> Self {
        assert_eq!(old_to_new.len(), self.node_count());
        let mut seen = vec![false; self.node_count()];
        for &new in old_to_new {
            assert!(new < self.node_count() && !seen[new], "mapping must be a permutation");
            seen[new] = true;
        }

        let remap = |source: usize| old_to_new[source];
        let mut new_rules = vec![Rule::Constant(false); self.node_count()];
        for (old_target, rule) in self.rules.iter().copied().enumerate() {
            let new_target = old_to_new[old_target];
            new_rules[new_target] = match rule {
                Rule::Copy { from } => Rule::Copy { from: remap(from) },
                Rule::Xor { left, right } => Rule::Xor { left: remap(left), right: remap(right) },
                Rule::And { left, right } => Rule::And { left: remap(left), right: remap(right) },
                Rule::Constant(value) => Rule::Constant(value),
            };
        }
        Self::new(new_rules, self.flip_probability)
    }
}

#[derive(Clone, Copy, Debug)]
struct Metrics {
    nodes: usize,
    /// H(X[t+1]) under a uniform distribution over X[t].
    state_entropy_bits: f64,
    /// H(X[t+1] | X[t]) under a uniform distribution over X[t].
    transition_entropy_bits: f64,
    /// I(X[t]; X[t+1]) = H(X[t+1]) - H(X[t+1] | X[t]).
    mutual_information_bits: f64,
    /// Minimum over nontrivial cuts of mean conditional-distribution TV distance.
    /// This is a proposed causal-cut sensitivity proxy, NOT IIT Phi.
    min_cut_sensitivity: f64,
}

fn entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .copied()
        .filter(|&p| p > 0.0)
        .map(|p| -p * (p.ln() / LN_2))
        .sum()
}

fn mean_output_distribution(tpm: &[Vec<f64>]) -> Vec<f64> {
    let rows = tpm.len();
    let states = tpm.first().map_or(0, Vec::len);
    let mut marginal = vec![0.0; states];
    for row in tpm {
        for (next, probability) in row.iter().copied().enumerate() {
            marginal[next] += probability / rows as f64;
        }
    }
    marginal
}

fn mean_row_entropy(tpm: &[Vec<f64>]) -> f64 {
    tpm.iter().map(|row| entropy_bits(row)).sum::<f64>() / tpm.len() as f64
}

fn mean_row_total_variation(left: &[Vec<f64>], right: &[Vec<f64>]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            assert_eq!(a.len(), b.len());
            0.5 * a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>()
        })
        .sum::<f64>()
        / left.len() as f64
}

fn measure(network: &Network) -> Metrics {
    let original_tpm = network.transition_matrix(None);
    let output_marginal = mean_output_distribution(&original_tpm);
    let state_entropy_bits = entropy_bits(&output_marginal);
    let transition_entropy_bits = mean_row_entropy(&original_tpm);
    let mutual_information_bits = (state_entropy_bits - transition_entropy_bits).max(0.0);

    let full_mask = network.state_count() - 1;
    let mut min_cut_sensitivity = f64::INFINITY;

    // A mask describes side A. Requiring node 0 in A removes duplicate
    // A|B versus B|A cuts. The all-nodes mask is the trivial cut and is skipped.
    for cut_mask in 1..full_mask {
        if cut_mask & 1 == 0 {
            continue;
        }
        let cut_tpm = network.transition_matrix(Some(cut_mask));
        min_cut_sensitivity = min_cut_sensitivity.min(mean_row_total_variation(&original_tpm, &cut_tpm));
    }

    if !min_cut_sensitivity.is_finite() {
        min_cut_sensitivity = 0.0;
    }

    Metrics {
        nodes: network.node_count(),
        state_entropy_bits,
        transition_entropy_bits,
        mutual_information_bits,
        min_cut_sensitivity,
    }
}

fn self_memory(nodes: usize, noise: f64) -> Network {
    Network::new((0..nodes).map(|i| Rule::Copy { from: i }).collect(), noise)
}

fn directed_ring(nodes: usize, noise: f64) -> Network {
    assert!(nodes >= 2);
    Network::new(
        (0..nodes).map(|i| Rule::Copy { from: (i + 1) % nodes }).collect(),
        noise,
    )
}

fn common_driver(nodes: usize, noise: f64) -> Network {
    assert!(nodes >= 2);
    Network::new((0..nodes).map(|_| Rule::Copy { from: 0 }).collect(), noise)
}

fn xor_synergy(noise: f64) -> Network {
    Network::new(
        vec![
            Rule::Copy { from: 0 },
            Rule::Copy { from: 1 },
            Rule::Xor { left: 0, right: 1 },
        ],
        noise,
    )
}

fn main() {
    let cases = [
        ("independent_max_noise", self_memory(3, 0.5)),
        ("recurrent_ring_no_noise", directed_ring(3, 0.0)),
        ("common_driver_no_noise", common_driver(3, 0.0)),
        ("xor_synergy_no_noise", xor_synergy(0.0)),
        ("ring_noise_10pct", directed_ring(3, 0.1)),
    ];

    println!("network\tnodes\tH_next_bits\tH_next_given_prev_bits\tI_prev_next_bits\tmin_cut_sensitivity_tv");
    for (name, network) in cases {
        let m = measure(&network);
        println!(
            "{name}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            m.nodes,
            m.state_entropy_bits,
            m.transition_entropy_bits,
            m.mutual_information_bits,
            m.min_cut_sensitivity,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) {
        assert!((a - b).abs() <= tolerance, "{a} != {b} within {tolerance}");
    }

    #[test]
    fn entropy_has_correct_binary_extremes() {
        close(entropy_bits(&[1.0, 0.0]), 0.0, 1e-12);
        close(entropy_bits(&[0.5, 0.5]), 1.0, 1e-12);
        close(entropy_bits(&[0.25, 0.25, 0.25, 0.25]), 2.0, 1e-12);
    }

    #[test]
    fn every_transition_row_is_a_probability_distribution() {
        let network = directed_ring(3, 0.13);
        for row in network.transition_matrix(None) {
            assert!(row.iter().all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0));
            close(row.iter().sum(), 1.0, 1e-12);
        }
    }

    #[test]
    fn maximal_state_entropy_does_not_imply_causal_integration() {
        let independent_noise = measure(&self_memory(3, 0.5));
        let deterministic_ring = measure(&directed_ring(3, 0.0));

        // Both have maximal H(X[t+1]) = 3 bits under the uniform input.
        close(independent_noise.state_entropy_bits, 3.0, 1e-12);
        close(deterministic_ring.state_entropy_bits, 3.0, 1e-12);

        // Yet one has no information about its prior state and no cross-unit
        // dependencies, while the deterministic ring has both temporal
        // information and measurable sensitivity to cross-partition cuts.
        close(independent_noise.mutual_information_bits, 0.0, 1e-12);
        close(independent_noise.min_cut_sensitivity, 0.0, 1e-12);
        close(deterministic_ring.transition_entropy_bits, 0.0, 1e-12);
        assert!(deterministic_ring.mutual_information_bits > 2.9);
        assert!(deterministic_ring.min_cut_sensitivity > 0.0);
    }

    #[test]
    fn maximal_transition_noise_erases_cut_sensitivity() {
        let metrics = measure(&directed_ring(4, 0.5));
        close(metrics.state_entropy_bits, 4.0, 1e-12);
        close(metrics.transition_entropy_bits, 4.0, 1e-12);
        close(metrics.mutual_information_bits, 0.0, 1e-12);
        close(metrics.min_cut_sensitivity, 0.0, 1e-12);
    }

    #[test]
    fn xor_circuit_exposes_joint_dependence() {
        let metrics = measure(&xor_synergy(0.0));
        close(metrics.state_entropy_bits, 2.0, 1e-12);
        close(metrics.transition_entropy_bits, 0.0, 1e-12);
        close(metrics.mutual_information_bits, 2.0, 1e-12);
        assert!(metrics.min_cut_sensitivity > 0.0);
    }

    #[test]
    fn results_are_invariant_to_node_relabeling() {
        let network = directed_ring(4, 0.07);
        let permuted = network.relabel(&[2, 0, 3, 1]);
        let original = measure(&network);
        let relabeled = measure(&permuted);

        close(original.state_entropy_bits, relabeled.state_entropy_bits, 1e-12);
        close(original.transition_entropy_bits, relabeled.transition_entropy_bits, 1e-12);
        close(original.mutual_information_bits, relabeled.mutual_information_bits, 1e-12);
        close(original.min_cut_sensitivity, relabeled.min_cut_sensitivity, 1e-12);
    }

    #[test]
    fn cut_metric_is_zero_when_no_rule_depends_on_another_node() {
        let network = Network::new(
            vec![Rule::Copy { from: 0 }, Rule::Copy { from: 1 }, Rule::Copy { from: 2 }],
            0.17,
        );
        close(measure(&network).min_cut_sensitivity, 0.0, 1e-12);
    }

    #[test]
    fn noise_changes_dynamics_without_being_misreported_as_entropy_alone() {
        let clean = measure(&directed_ring(3, 0.0));
        let noisy = measure(&directed_ring(3, 0.2));
        assert!(noisy.transition_entropy_bits > clean.transition_entropy_bits);
        assert!(noisy.mutual_information_bits < clean.mutual_information_bits);
        assert!(noisy.min_cut_sensitivity < clean.min_cut_sensitivity);
    }
}
