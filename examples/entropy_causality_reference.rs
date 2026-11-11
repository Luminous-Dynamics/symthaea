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

const MAX_NODES: usize = 8;

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

    fn canonical_specification(&self) -> String {
        let rules = self
            .rules
            .iter()
            .map(|rule| match rule {
                Rule::Copy { from } => format!("copy:{from}"),
                Rule::Xor { left, right } => format!("xor:{left}:{right}"),
                Rule::And { left, right } => format!("and:{left}:{right}"),
                Rule::Constant(value) => format!("constant:{}", if *value { 1 } else { 0 }),
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("schema=boolean-network-v1;rules={rules};flip={:.6};flip_bits={:016x}", self.flip_probability, self.flip_probability.to_bits())
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
    /// H(X[t]) under the explicitly supplied distribution over previous states.
    input_state_entropy_bits: f64,
    /// H(X[t+1]) under the explicitly supplied distribution over previous states.
    state_entropy_bits: f64,
    /// H(X[t+1] | X[t]) under the explicitly supplied distribution over previous states.
    transition_entropy_bits: f64,
    /// I(X[t]; X[t+1]) = H(X[t+1]) - H(X[t+1] | X[t]).
    mutual_information_bits: f64,
    /// Minimum over nontrivial cuts of prior-weighted conditional-row TV distance.
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

fn weighted_output_distribution(
    tpm: &[Vec<f64>],
    previous_state_distribution: &[f64],
) -> Vec<f64> {
    assert_eq!(tpm.len(), previous_state_distribution.len());
    let states = tpm.first().map_or(0, Vec::len);
    let mut marginal = vec![0.0; states];
    for (row, weight) in tpm.iter().zip(previous_state_distribution) {
        for (next, probability) in row.iter().copied().enumerate() {
            marginal[next] += weight * probability;
        }
    }
    marginal
}

fn weighted_mean_row_entropy(
    tpm: &[Vec<f64>],
    previous_state_distribution: &[f64],
) -> f64 {
    assert_eq!(tpm.len(), previous_state_distribution.len());
    tpm.iter()
        .zip(previous_state_distribution)
        .map(|(row, weight)| weight * entropy_bits(row))
        .sum()
}

fn weighted_mean_row_total_variation(
    left: &[Vec<f64>],
    right: &[Vec<f64>],
    previous_state_distribution: &[f64],
) -> f64 {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), previous_state_distribution.len());
    left.iter()
        .zip(right)
        .zip(previous_state_distribution)
        .map(|((a, b), weight)| {
            assert_eq!(a.len(), b.len());
            weight * 0.5 * a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>()
        })
        .sum()
}

fn uniform_state_distribution(network: &Network) -> Vec<f64> {
    vec![1.0 / network.state_count() as f64; network.state_count()]
}

fn canonical_prior_specification(probabilities: &[f64]) -> String {
    let entries = probabilities
        .iter()
        .enumerate()
        .map(|(state, probability)| format!("{state}:{probability:.6}:{:016x}", probability.to_bits()))
        .collect::<Vec<_>>()
        .join(",");
    format!("schema=state-prior-v1;prior={entries}")
}

fn measure(network: &Network) -> Metrics {
    measure_with_state_distribution(network, &uniform_state_distribution(network))
}

fn measure_with_state_distribution(
    network: &Network,
    previous_state_distribution: &[f64],
) -> Metrics {
    assert_eq!(
        previous_state_distribution.len(),
        network.state_count(),
        "state distribution must have one probability per previous state"
    );
    assert!(
        previous_state_distribution
            .iter()
            .all(|p| p.is_finite() && *p >= 0.0),
        "state distribution values must be finite and nonnegative"
    );
    let probability_sum: f64 = previous_state_distribution.iter().sum();
    assert!(
        (probability_sum - 1.0).abs() <= 1e-12,
        "state distribution must sum to 1 (got {probability_sum})"
    );

    let original_tpm = network.transition_matrix(None);
    let output_marginal =
        weighted_output_distribution(&original_tpm, previous_state_distribution);
    let state_entropy_bits = entropy_bits(&output_marginal);
    let input_state_entropy_bits = entropy_bits(previous_state_distribution);
    let transition_entropy_bits =
        weighted_mean_row_entropy(&original_tpm, previous_state_distribution);
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
        min_cut_sensitivity = min_cut_sensitivity.min(weighted_mean_row_total_variation(
            &original_tpm,
            &cut_tpm,
            previous_state_distribution,
        ));
    }

    if !min_cut_sensitivity.is_finite() {
        min_cut_sensitivity = 0.0;
    }

    Metrics {
        nodes: network.node_count(),
        input_state_entropy_bits,
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

fn and_circuit(noise: f64) -> Network {
    Network::new(
        vec![
            Rule::Copy { from: 0 },
            Rule::Copy { from: 1 },
            Rule::And { left: 0, right: 1 },
        ],
        noise,
    )
}

fn feed_forward_chain(nodes: usize, noise: f64) -> Network {
    assert!(nodes >= 2);
    Network::new(
        (0..nodes)
            .map(|i| Rule::Copy { from: if i == 0 { 0 } else { i - 1 } })
            .collect(),
        noise,
    )
}

fn main() {
    let mut cases: Vec<(String, Network, Vec<f64>)> = Vec::new();

    let independent = self_memory(3, 0.5);
    cases.push((
        "independent_max_noise".to_string(),
        independent.clone(),
        uniform_state_distribution(&independent),
    ));

    let clean_ring = directed_ring(3, 0.0);
    cases.push((
        "recurrent_ring_uniform_prior".to_string(),
        clean_ring.clone(),
        uniform_state_distribution(&clean_ring),
    ));
    let mut skewed_prior = vec![0.0; clean_ring.state_count()];
    skewed_prior[0] = 0.7;
    skewed_prior[1] = 0.3;
    cases.push((
        "recurrent_ring_skewed_prior".to_string(),
        clean_ring,
        skewed_prior,
    ));

    let common = common_driver(3, 0.0);
    cases.push((
        "common_driver_no_noise".to_string(),
        common.clone(),
        uniform_state_distribution(&common),
    ));
    let chain = feed_forward_chain(4, 0.0);
    cases.push((
        "feed_forward_chain_no_noise".to_string(),
        chain.clone(),
        uniform_state_distribution(&chain),
    ));
    let xor = xor_synergy(0.0);
    cases.push((
        "xor_synergy_no_noise".to_string(),
        xor.clone(),
        uniform_state_distribution(&xor),
    ));
    let and = and_circuit(0.0);
    cases.push((
        "and_synergy_no_noise".to_string(),
        and.clone(),
        uniform_state_distribution(&and),
    ));

    // Sweep noise explicitly instead of assuming entropy and integration share
    // one monotonic relationship.
    for noise in [0.0, 0.05, 0.10, 0.20, 0.35, 0.50] {
        let ring = directed_ring(3, noise);
        cases.push((
            format!("ring_noise_{:.2}", noise),
            ring.clone(),
            uniform_state_distribution(&ring),
        ));
    }

    println!("network\tnodes\tmodel_spec\tprior_spec\tH_prev_bits\tH_next_bits\tH_next_given_prev_bits\tI_prev_next_bits\tmin_cut_sensitivity_tv");
    for (name, network, prior) in cases {
        let m = measure_with_state_distribution(&network, &prior);
        println!(
            "{name}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            m.nodes,
            network.canonical_specification(),
            canonical_prior_specification(&prior),
            m.input_state_entropy_bits,
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
    fn canonical_manifest_matches_rust_model_and_prior_encodings() {
        let manifest = include_str!("../validation/fixtures/entropy_causality_models.tsv");
        let mut cases = 0usize;
        for (line_index, raw_line) in manifest.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            assert_eq!(fields.len(), 5, "manifest line {}", line_index + 1);
            let (name, node_count, rules_text, noise_hex, prior_text) =
                (fields[0], fields[1], fields[2], fields[3], fields[4]);
            let node_count: usize = node_count.parse().expect("valid node count");
            let noise_bits = u64::from_str_radix(noise_hex, 16).expect("valid noise f64 bits");
            let noise = f64::from_bits(noise_bits);
            let rules: Vec<Rule> = rules_text
                .split(',')
                .map(|token| {
                    let parts: Vec<&str> = token.split(':').collect();
                    match parts.as_slice() {
                        ["copy", from] => Rule::Copy {
                            from: from.parse().expect("copy source"),
                        },
                        ["xor", left, right] => Rule::Xor {
                            left: left.parse().expect("xor left source"),
                            right: right.parse().expect("xor right source"),
                        },
                        ["and", left, right] => Rule::And {
                            left: left.parse().expect("and left source"),
                            right: right.parse().expect("and right source"),
                        },
                        ["constant", "0"] => Rule::Constant(false),
                        ["constant", "1"] => Rule::Constant(true),
                        _ => panic!("unsupported manifest rule at line {}: {token}", line_index + 1),
                    }
                })
                .collect();
            assert_eq!(rules.len(), node_count, "rule count for {name}");
            let network = Network::new(rules, noise);
            assert_eq!(
                network.canonical_specification(),
                format!(
                    "schema=boolean-network-v1;rules={rules_text};flip={:.6};flip_bits={:016x}",
                    noise, noise_bits
                ),
                "canonical model mismatch for {name}"
            );

            let prior = if prior_text == "uniform" {
                uniform_state_distribution(&network)
            } else {
                let entries = prior_text
                    .strip_prefix("sparse:")
                    .expect("supported prior spec");
                let mut prior = vec![0.0; network.state_count()];
                for entry in entries.split(',') {
                    let (state_text, probability_hex) = entry.split_once(':').expect("prior entry");
                    let state: usize = state_text.parse().expect("prior state index");
                    let probability_bits =
                        u64::from_str_radix(probability_hex, 16).expect("prior f64 bits");
                    assert!(state < prior.len(), "prior state outside model for {name}");
                    prior[state] = f64::from_bits(probability_bits);
                }
                prior
            };
            assert_eq!(prior.len(), 1usize << node_count, "prior length for {name}");
            assert!(
                (prior.iter().sum::<f64>() - 1.0).abs() <= 1e-12,
                "prior mass for {name}"
            );
            assert_eq!(
                canonical_prior_specification(&prior),
                {
                    let entries = prior
                        .iter()
                        .enumerate()
                        .map(|(state, probability)| {
                            format!("{state}:{probability:.6}:{:016x}", probability.to_bits())
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("schema=state-prior-v1;prior={entries}")
                },
                "canonical prior mismatch for {name}"
            );
            cases += 1;
        }
        assert_eq!(cases, 13, "manifest fixture count changed; update this assertion deliberately");
    }

    #[test]
    fn entropy_has_correct_binary_extremes() {
        close(entropy_bits(&[1.0, 0.0]), 0.0, 1e-12);
        close(entropy_bits(&[0.5, 0.5]), 1.0, 1e-12);
        close(entropy_bits(&[0.25, 0.25, 0.25, 0.25]), 2.0, 1e-12);
    }

    #[test]
    fn canonical_specification_includes_rule_order_and_noise() {
        assert_eq!(
            directed_ring(3, 0.1).canonical_specification(),
            "schema=boolean-network-v1;rules=copy:1,copy:2,copy:0;flip=0.100000;flip_bits=3fb999999999999a"
        );
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
    fn common_driver_has_dependence_without_mutual_causation_between_children() {
        let metrics = measure(&common_driver(3, 0.0));
        close(metrics.state_entropy_bits, 1.0, 1e-12);
        close(metrics.transition_entropy_bits, 0.0, 1e-12);
        close(metrics.mutual_information_bits, 1.0, 1e-12);
        close(metrics.min_cut_sensitivity, 0.5, 1e-12);
    }

    #[test]
    fn feed_forward_chain_and_and_circuit_are_enumerated_exactly() {
        let chain = measure(&feed_forward_chain(4, 0.0));
        close(chain.state_entropy_bits, 3.0, 1e-12);
        close(chain.transition_entropy_bits, 0.0, 1e-12);
        close(chain.mutual_information_bits, 3.0, 1e-12);

        let and = measure(&and_circuit(0.0));
        close(and.state_entropy_bits, 2.0, 1e-12);
        close(and.transition_entropy_bits, 0.0, 1e-12);
        close(and.mutual_information_bits, 2.0, 1e-12);
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
    fn same_transition_rules_change_entropy_under_a_skewed_state_ensemble() {
        let network = directed_ring(3, 0.0);
        let uniform = uniform_state_distribution(&network);
        let mut skewed = vec![0.0; network.state_count()];
        skewed[0] = 0.7;
        skewed[1] = 0.3;

        let uniform_metrics = measure_with_state_distribution(&network, &uniform);
        let skewed_metrics = measure_with_state_distribution(&network, &skewed);

        close(uniform_metrics.input_state_entropy_bits, 3.0, 1e-12);
        close(uniform_metrics.state_entropy_bits, 3.0, 1e-12);
        close(skewed_metrics.input_state_entropy_bits, entropy_bits(&[0.7, 0.3]), 1e-12);
        close(skewed_metrics.state_entropy_bits, entropy_bits(&[0.7, 0.3]), 1e-12);
        close(skewed_metrics.transition_entropy_bits, 0.0, 1e-12);
        close(skewed_metrics.mutual_information_bits, entropy_bits(&[0.7, 0.3]), 1e-12);
        assert!(skewed_metrics.state_entropy_bits < uniform_metrics.state_entropy_bits);
        // The transition rules are identical: only the prior over states changed.
        assert_eq!(network.canonical_specification(), directed_ring(3, 0.0).canonical_specification());
    }

    #[test]
    #[should_panic(expected = "state distribution must sum to 1")]
    fn invalid_state_ensemble_fails_closed() {
        let network = directed_ring(3, 0.0);
        let _ = measure_with_state_distribution(&network, &[1.0; 8]);
    }

    #[test]
    fn noise_sweep_preserves_marginal_entropy_but_erases_temporal_information() {
        let probabilities = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50];
        let metrics: Vec<_> = probabilities
            .iter()
            .map(|noise| measure(&directed_ring(3, *noise)))
            .collect();

        for point in &metrics {
            close(point.state_entropy_bits, 3.0, 1e-12);
        }
        close(metrics[0].transition_entropy_bits, 0.0, 1e-12);
        close(metrics[0].mutual_information_bits, 3.0, 1e-12);
        close(metrics[0].min_cut_sensitivity, 0.75, 1e-12);
        close(metrics[5].transition_entropy_bits, 3.0, 1e-12);
        close(metrics[5].mutual_information_bits, 0.0, 1e-12);
        close(metrics[5].min_cut_sensitivity, 0.0, 1e-12);

        for pair in metrics.windows(2) {
            assert!(pair[1].transition_entropy_bits > pair[0].transition_entropy_bits);
            assert!(pair[1].mutual_information_bits < pair[0].mutual_information_bits);
            assert!(pair[1].min_cut_sensitivity < pair[0].min_cut_sensitivity);
        }
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
