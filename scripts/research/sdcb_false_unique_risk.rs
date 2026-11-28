//! Independent finite-cohort reference for false-unique scientific-claim risk.
//! Evaluator-side known-answer oracle only; not production policy or SOWRC code.

use std::collections::{BTreeSet, HashSet};

const FIXTURE: &str = include_str!("../../docs/research/SDCB_FALSE_UNIQUE_AUDIT_FIXTURE_V1.tsv");
const Q_DENOMINATOR: u32 = 65_536;
const OTHER_TOKENS: [&str; 2] = ["OTHER_UNRESOLVED", "OTHER_FAMILY"];

#[derive(Clone, Debug)]
struct Episode {
    id: String,
    q_numerator: u32,
    generating_mechanism: String,
    generating_family: String,
    g_struct: BTreeSet<String>,
    g_budget: BTreeSet<String>,
    g_family: BTreeSet<String>,
    mechanism_sets: Vec<BTreeSet<String>>,
    family_sets: Vec<BTreeSet<String>>,
}

#[derive(Clone, Copy, Debug)]
enum Endpoint {
    Structural,
    Budget,
    Family,
}

#[derive(Debug)]
struct EndpointStats {
    n: usize,
    singleton_count: usize,
    false_unique_count: usize,
    compatibility_miss_count: usize,
    generating_covered_count: usize,
}

impl EndpointStats {
    fn commitment_rate(&self) -> f64 {
        self.singleton_count as f64 / self.n as f64
    }

    fn joint_false_unique_risk(&self) -> f64 {
        self.false_unique_count as f64 / self.n as f64
    }

    fn conditional_false_unique_risk(&self) -> Option<f64> {
        if self.singleton_count == 0 {
            None
        } else {
            Some(self.false_unique_count as f64 / self.singleton_count as f64)
        }
    }
}

struct PointStats {
    structural: EndpointStats,
    budget: EndpointStats,
    family: EndpointStats,
}

fn parse_set(raw: &str) -> BTreeSet<String> {
    if raw == "-" || raw.is_empty() {
        BTreeSet::new()
    } else {
        raw.split(',').map(str::to_owned).collect()
    }
}

fn parse_fixture() -> Vec<Episode> {
    let mut episodes = Vec::new();
    for (line_number, line) in FIXTURE.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('|').collect();
        assert_eq!(fields.len(), 13, "fixture line {} must have 13 fields", line_number + 1);
        episodes.push(Episode {
            id: fields[0].to_owned(),
            q_numerator: fields[1].parse().expect("q numerator must be an integer"),
            generating_mechanism: fields[2].to_owned(),
            generating_family: fields[3].to_owned(),
            g_struct: parse_set(fields[4]),
            g_budget: parse_set(fields[5]),
            g_family: parse_set(fields[6]),
            mechanism_sets: vec![parse_set(fields[7]), parse_set(fields[8]), parse_set(fields[9])],
            family_sets: vec![parse_set(fields[10]), parse_set(fields[11]), parse_set(fields[12])],
        });
    }
    validate_fixture(&episodes).expect("committed fixture must validate");
    episodes
}

fn is_other_token(value: &str) -> bool {
    OTHER_TOKENS.contains(&value)
}

fn is_substantive_singleton(candidate: &BTreeSet<String>) -> bool {
    candidate.len() == 1 && !candidate.iter().any(|value| is_other_token(value))
}

fn validate_fixture(episodes: &[Episode]) -> Result<(), String> {
    if episodes.is_empty() || episodes.len() > 20 {
        return Err("fixture must contain between 1 and 20 episodes".to_owned());
    }
    let mut ids = HashSet::new();
    for episode in episodes {
        if !ids.insert(episode.id.as_str()) {
            return Err(format!("duplicate episode id: {}", episode.id));
        }
        if episode.q_numerator == 0 || episode.q_numerator > Q_DENOMINATOR {
            return Err(format!("episode {} has q outside (0,1]", episode.id));
        }
        if episode.g_struct.is_empty() || episode.g_budget.is_empty() || episode.g_family.is_empty() {
            return Err(format!("episode {} has an empty evaluator compatibility set", episode.id));
        }
        if episode.mechanism_sets.len() != 3 || episode.family_sets.len() != 3 {
            return Err(format!("episode {} must have three operating points", episode.id));
        }
        for sets in [&episode.mechanism_sets, &episode.family_sets] {
            for pair in sets.windows(2) {
                if !pair[0].is_subset(&pair[1]) {
                    return Err(format!("episode {} has a non-nested set family", episode.id));
                }
            }
        }
    }
    Ok(())
}

fn candidate_set(episode: &Episode, endpoint: Endpoint, point: usize) -> &BTreeSet<String> {
    match endpoint {
        Endpoint::Structural | Endpoint::Budget => &episode.mechanism_sets[point],
        Endpoint::Family => &episode.family_sets[point],
    }
}

fn evaluator_set(episode: &Episode, endpoint: Endpoint) -> &BTreeSet<String> {
    match endpoint {
        Endpoint::Structural => &episode.g_struct,
        Endpoint::Budget => &episode.g_budget,
        Endpoint::Family => &episode.g_family,
    }
}

fn generating_id(episode: &Episode, endpoint: Endpoint) -> &str {
    match endpoint {
        Endpoint::Structural | Endpoint::Budget => &episode.generating_mechanism,
        Endpoint::Family => &episode.generating_family,
    }
}

fn endpoint_stats(episodes: &[Episode], point: usize, endpoint: Endpoint) -> EndpointStats {
    let mut stats = EndpointStats {
        n: episodes.len(),
        singleton_count: 0,
        false_unique_count: 0,
        compatibility_miss_count: 0,
        generating_covered_count: 0,
    };
    for episode in episodes {
        let candidate = candidate_set(episode, endpoint, point);
        let evaluator = evaluator_set(episode, endpoint);
        let singleton = is_substantive_singleton(candidate);
        let miss = !evaluator.is_subset(candidate);
        if singleton {
            stats.singleton_count += 1;
            if miss {
                stats.false_unique_count += 1;
            }
        }
        if miss {
            stats.compatibility_miss_count += 1;
        }
        if candidate.contains(generating_id(episode, endpoint)) {
            stats.generating_covered_count += 1;
        }
    }
    stats
}

fn evaluate_point(episodes: &[Episode], point: usize) -> PointStats {
    PointStats {
        structural: endpoint_stats(episodes, point, Endpoint::Structural),
        budget: endpoint_stats(episodes, point, Endpoint::Budget),
        family: endpoint_stats(episodes, point, Endpoint::Family),
    }
}

fn joint_loss(episode: &Episode, point: usize, endpoint: Endpoint) -> bool {
    let candidate = candidate_set(episode, endpoint, point);
    is_substantive_singleton(candidate) && !evaluator_set(episode, endpoint).is_subset(candidate)
}

fn exact_mask_weight(episodes: &[Episode], mask: u64) -> u128 {
    episodes.iter().enumerate().fold(1_u128, |weight, (index, episode)| {
        let selected = (mask & (1_u64 << index)) != 0;
        let factor = if selected {
            episode.q_numerator
        } else {
            Q_DENOMINATOR - episode.q_numerator
        };
        weight * factor as u128
    })
}

fn enumerate_expected_ht(
    episodes: &[Episode],
    point: usize,
    endpoint: Endpoint,
) -> (f64, f64, u128, u128) {
    assert!(episodes.len() <= 20);
    let n = episodes.len();
    let total_weight = (Q_DENOMINATOR as u128).pow(n as u32);
    let mut weight_sum = 0_u128;
    let mut expected_estimate = 0.0_f64;

    for mask in 0..(1_u64 << n) {
        let weight = exact_mask_weight(episodes, mask);
        weight_sum += weight;
        let mut ht_total = 0.0_f64;
        for (index, episode) in episodes.iter().enumerate() {
            if (mask & (1_u64 << index)) != 0 && joint_loss(episode, point, endpoint) {
                let q = episode.q_numerator as f64 / Q_DENOMINATOR as f64;
                ht_total += 1.0 / q;
            }
        }
        let estimate = ht_total / n as f64;
        expected_estimate += (weight as f64 / total_weight as f64) * estimate;
    }

    let true_count = episodes.iter().filter(|e| joint_loss(e, point, endpoint)).count();
    (true_count as f64 / n as f64, expected_estimate, weight_sum, total_weight)
}

fn print_report(episodes: &[Episode]) {
    for point in 0..3 {
        let stats = evaluate_point(episodes, point);
        println!("lambda_index={point}");
        for (name, endpoint) in [
            ("structural", &stats.structural),
            ("budget", &stats.budget),
            ("family", &stats.family),
        ] {
            match endpoint.conditional_false_unique_risk() {
                Some(risk) => println!(
                    "  {name}: singleton={} commitment={:.6} false_unique={}/{} conditional_risk={:.6} joint_risk={:.6}",
                    endpoint.singleton_count,
                    endpoint.commitment_rate(),
                    endpoint.false_unique_count,
                    endpoint.singleton_count,
                    risk,
                    endpoint.joint_false_unique_risk()
                ),
                None => println!("  {name}: NO_SINGLETON_COHORT commitment=0 false_unique_bound=WITHHELD"),
            }
        }
    }
    println!("audit_mask_count={}", 1_u64 << episodes.len());
    println!("audit_probability_denominator_bits={}", 16 * episodes.len());
}

fn main() {
    print_report(&parse_fixture());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<Episode> {
        parse_fixture()
    }

    #[test]
    fn fixture_has_positive_exact_dyadic_support_and_nested_sets() {
        let episodes = fixture();
        assert_eq!(episodes.len(), 6);
        assert!(episodes.iter().all(|e| e.q_numerator > 0 && e.q_numerator <= Q_DENOMINATOR));
        assert!(validate_fixture(&episodes).is_ok());
    }

    #[test]
    fn fixed_operating_point_keeps_three_estimands_and_denominators_separate() {
        let stats = evaluate_point(&fixture(), 0);
        assert_eq!((stats.structural.singleton_count, stats.structural.false_unique_count), (4, 3));
        assert!((stats.structural.conditional_false_unique_risk().unwrap() - 0.75).abs() < 1e-12);
        assert_eq!((stats.budget.singleton_count, stats.budget.false_unique_count), (4, 4));
        assert!((stats.budget.conditional_false_unique_risk().unwrap() - 1.0).abs() < 1e-12);
        assert_eq!((stats.family.singleton_count, stats.family.false_unique_count), (3, 3));
        assert!((stats.family.conditional_false_unique_risk().unwrap() - 1.0).abs() < 1e-12);
        assert_ne!(stats.structural.singleton_count, stats.family.singleton_count);
    }

    #[test]
    fn nominal_pointwise_truth_does_not_hide_budget_or_family_ambiguity() {
        let e2 = &fixture()[1];
        assert!(e2.g_struct.is_subset(&e2.mechanism_sets[0]));
        assert!(!e2.g_budget.is_subset(&e2.mechanism_sets[0]));
        assert!(!e2.g_family.is_subset(&e2.family_sets[0]));
    }

    #[test]
    fn compatibility_miss_is_monotone_but_direct_false_unique_need_not_be() {
        let episodes = fixture();
        let e6 = &episodes[5];
        let direct: Vec<bool> = (0..3).map(|p| joint_loss(e6, p, Endpoint::Structural)).collect();
        assert_eq!(direct, vec![false, true, false]);
        for endpoint in [Endpoint::Structural, Endpoint::Budget, Endpoint::Family] {
            let misses: Vec<bool> = (0..3)
                .map(|p| !evaluator_set(e6, endpoint).is_subset(candidate_set(e6, endpoint, p)))
                .collect();
            assert_eq!(misses, vec![true, true, false]);
        }
    }

    #[test]
    fn zero_commitment_is_not_a_conditional_risk_pass() {
        let stats = evaluate_point(&fixture(), 2);
        assert_eq!(stats.structural.singleton_count, 0);
        assert_eq!(stats.budget.singleton_count, 0);
        assert_eq!(stats.family.singleton_count, 0);
        assert!(stats.structural.conditional_false_unique_risk().is_none());
        assert!(stats.budget.conditional_false_unique_risk().is_none());
        assert!(stats.family.conditional_false_unique_risk().is_none());
    }

    #[test]
    fn family_overlap_is_not_inferred_from_a_sampled_parameter_grid() {
        let e2 = &fixture()[1];
        assert_eq!(e2.g_family, parse_set("F1,F2"));
        // The public ID set represents a supplied exact witness, not a grid search.
    }

    #[test]
    fn unresolved_other_singleton_is_not_counted_as_a_substantive_claim() {
        let e4 = &fixture()[3];
        assert_eq!(e4.g_struct, parse_set("OTHER_UNRESOLVED"));
        assert!(joint_loss(e4, 0, Endpoint::Structural));
        assert!(!is_substantive_singleton(&parse_set("OTHER_UNRESOLVED")));
    }

    #[test]
    fn complete_audit_ht_estimate_equals_full_cohort_joint_risk() {
        let episodes = fixture();
        for endpoint in [Endpoint::Structural, Endpoint::Budget, Endpoint::Family] {
            let (truth, expected, _, _) = enumerate_expected_ht(&episodes, 0, endpoint);
            assert!((truth - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn all_64_bernoulli_masks_have_exact_total_dyadic_weight_and_ht_is_unbiased() {
        let episodes = fixture();
        for point in 0..3 {
            for endpoint in [Endpoint::Structural, Endpoint::Budget, Endpoint::Family] {
                let (truth, expected, weight_sum, total_weight) =
                    enumerate_expected_ht(&episodes, point, endpoint);
                assert_eq!(weight_sum, total_weight);
                assert!((truth - expected).abs() < 1e-12, "{endpoint:?} at point {point}");
            }
        }
    }
}
