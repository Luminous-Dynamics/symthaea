//! Known-ground-truth synthetic search benchmark for architecture-search policy validation.
//!
//! This module is an implementation test, not a surrogate for neural-network training.
//! The objective landscapes are synthetic and fully enumerable. Their only purpose is to
//! expose search/evaluation bugs, measure finite-budget behavior, and test proxy-failure
//! controls before expensive model experiments are started.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::paired_evaluation::{pareto_front, ParetoPoint};

const LEVELS: u8 = 4;
const PROFILE_SHA256: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const OPTIMUM_TARGET: f64 = 0.95;

/// Synthetic landscapes have exhaustive, deterministic ground truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticLandscape {
    /// A smooth, single-basin quality surface with resource costs.
    SmoothTradeoff,
    /// A planted local trap separated from a higher global optimum by a valley.
    Deceptive,
}

/// Search policy controls which candidates are fully evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticSearchPolicy {
    Random,
    LocalSearch,
    ProxyRanked,
}

/// Proxy controls are deliberately explicit; only the synthetic environment knows the truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticProxyMode {
    /// Feature-only complexity prior with a small deterministic perturbation.
    Correlated,
    /// Assign the same proxy values to permuted candidate IDs.
    Shuffled,
    /// Oracle-derived adversarial control: reverses the true quality ordering.
    /// This mode is available only for testing proxy failure, never as a usable prior.
    AntiCorrelated,
}

/// A discrete architecture configuration. Each gene is in 0..4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SyntheticCandidate {
    pub width: u8,
    pub depth: u8,
    pub attention: u8,
    pub recurrence: u8,
}

impl SyntheticCandidate {
    pub fn id(self) -> String {
        format!(
            "w{}-d{}-a{}-r{}",
            self.width, self.depth, self.attention, self.recurrence
        )
    }

    fn genes(self) -> [u8; 4] {
        [self.width, self.depth, self.attention, self.recurrence]
    }
}

/// Known objective values for one synthetic configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticObjectives {
    /// Normalized task quality; higher is better.
    pub quality: f64,
    /// Relative latency units, not milliseconds; lower is better.
    pub latency: f64,
    /// Relative energy units, not measured joules; lower is better.
    pub energy: f64,
    /// Relative memory units, not measured bytes; lower is better.
    pub memory: u64,
}

/// Exhaustive oracle for a finite synthetic candidate space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticOracle {
    pub landscape: SyntheticLandscape,
    pub candidate_count: usize,
    pub global_optimum_ids: Vec<String>,
    pub global_optimum_quality: f64,
    pub top_k_ids: Vec<String>,
    pub pareto_front_ids: Vec<String>,
    pub exact_quality_latency_hypervolume: f64,
}

/// Result of one policy run under one fixed evaluation budget and seed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticSearchRun {
    pub landscape: SyntheticLandscape,
    pub policy: SyntheticSearchPolicy,
    pub proxy_mode: SyntheticProxyMode,
    pub seed: u64,
    pub budget: usize,
    pub evaluated_ids: Vec<String>,
    pub best_evaluated_id: String,
    pub best_evaluated_quality: f64,
    pub simple_regret: f64,
    pub exact_optimum_hit: bool,
    pub top_k_recall: f64,
    pub pareto_front_recall: f64,
    pub normalized_hypervolume: f64,
    /// Spearman correlation between the synthetic proxy's full-space ranking and true quality.
    /// This is diagnostic-only; the oracle is available solely because this is a synthetic test.
    pub proxy_quality_spearman: f64,
    pub evaluations_to_target: Option<usize>,
}

/// Aggregate across independent seeds. Standard deviation is sample SD (n - 1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticSearchAggregate {
    pub landscape: SyntheticLandscape,
    pub policy: SyntheticSearchPolicy,
    pub proxy_mode: SyntheticProxyMode,
    pub budget: usize,
    pub seed_count: usize,
    pub mean_simple_regret: f64,
    pub simple_regret_sample_sd: f64,
    pub optimum_hit_rate: f64,
    pub mean_top_k_recall: f64,
    pub mean_pareto_front_recall: f64,
    pub target_hit_rate: f64,
    pub mean_normalized_hypervolume: f64,
    pub mean_proxy_quality_spearman: f64,
    /// Mean over only the runs that reached the target; use target_hit_rate alongside it.
    pub mean_evaluations_to_target: Option<f64>,
}

/// Enumerate the complete 4×4×4×4 (256-candidate) space in canonical ID order.
pub fn enumerate_candidates() -> Vec<SyntheticCandidate> {
    let mut candidates = Vec::with_capacity((LEVELS as usize).pow(4));
    for width in 0..LEVELS {
        for depth in 0..LEVELS {
            for attention in 0..LEVELS {
                for recurrence in 0..LEVELS {
                    candidates.push(SyntheticCandidate {
                        width,
                        depth,
                        attention,
                        recurrence,
                    });
                }
            }
        }
    }
    candidates.sort_by_key(|candidate| candidate.id());
    candidates
}

/// Evaluate one candidate under a fully specified synthetic landscape.
pub fn evaluate_candidate(
    candidate: SyntheticCandidate,
    landscape: SyntheticLandscape,
) -> SyntheticObjectives {
    let width = candidate.width as f64;
    let depth = candidate.depth as f64;
    let attention = candidate.attention as f64;
    let recurrence = candidate.recurrence as f64;

    let global_distance = 0.12 * (3.0 - width).abs()
        + 0.11 * (2.0 - depth).abs()
        + 0.12 * (3.0 - attention).abs()
        + 0.10 * (1.0 - recurrence).abs();
    let smooth_quality = (1.0 - global_distance).clamp(0.0, 1.0);

    let quality = match landscape {
        SyntheticLandscape::SmoothTradeoff => smooth_quality,
        SyntheticLandscape::Deceptive => {
            let decoy_distance = 0.15 * width
                + 0.14 * depth
                + 0.13 * attention
                + 0.12 * (3.0 - recurrence).abs();
            let decoy_quality = (0.86 - decoy_distance).clamp(0.0, 0.86);
            smooth_quality.max(decoy_quality)
        }
    };

    // Monotonic but differently weighted costs ensure quality/cost trade-offs.
    let latency = 1.0 + 0.40 * width + 0.35 * depth + 0.45 * attention + 0.12 * recurrence;
    let energy = 0.20 + 0.08 * width + 0.06 * depth + 0.10 * attention + 0.02 * recurrence;
    let memory = ((candidate.width as u64 + 1)
        * (candidate.depth as u64 + 1)
        * 128
        * (2 + candidate.attention as u64))
        + (candidate.recurrence as u64 * 16);

    SyntheticObjectives {
        quality,
        latency,
        energy,
        memory,
    }
}

/// Compute exhaustive truth independently from any search policy.
pub fn exhaustive_oracle(landscape: SyntheticLandscape, top_k: usize) -> SyntheticOracle {
    let candidates = enumerate_candidates();
    let evaluated: Vec<(SyntheticCandidate, SyntheticObjectives)> = candidates
        .iter()
        .copied()
        .map(|candidate| (candidate, evaluate_candidate(candidate, landscape)))
        .collect();

    let best_quality = evaluated
        .iter()
        .map(|(_, objectives)| objectives.quality)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut global_optimum_ids: Vec<String> = evaluated
        .iter()
        .filter(|(_, objectives)| (objectives.quality - best_quality).abs() <= f64::EPSILON)
        .map(|(candidate, _)| candidate.id())
        .collect();
    global_optimum_ids.sort();

    let mut ranked = evaluated.clone();
    ranked.sort_by(|(candidate_a, objectives_a), (candidate_b, objectives_b)| {
        objectives_b
            .quality
            .total_cmp(&objectives_a.quality)
            .then_with(|| candidate_a.id().cmp(&candidate_b.id()))
    });
    let top_k_ids = ranked
        .iter()
        .take(top_k.min(ranked.len()))
        .map(|(candidate, _)| candidate.id())
        .collect();

    let pareto_points: Vec<ParetoPoint> = evaluated
        .iter()
        .map(|(candidate, objectives)| ParetoPoint {
            candidate_id: candidate.id(),
            comparison_profile_sha256: PROFILE_SHA256.to_owned(),
            quality: objectives.quality,
            latency_ms: objectives.latency,
            energy_joules_per_token: objectives.energy,
            peak_memory_bytes: objectives.memory,
        })
        .collect();
    let pareto_front_ids = pareto_front(&pareto_points)
        .expect("the exhaustive synthetic profile is finite and internally consistent");

    let min_latency = evaluated
        .iter()
        .map(|(_, objectives)| objectives.latency)
        .fold(f64::INFINITY, f64::min);
    let max_latency = evaluated
        .iter()
        .map(|(_, objectives)| objectives.latency)
        .fold(f64::NEG_INFINITY, f64::max);
    let exact_quality_latency_hypervolume = hypervolume_2d(
        &evaluated
            .iter()
            .map(|(_, objectives)| {
                (
                    objectives.quality,
                    normalize_latency_utility(objectives.latency, min_latency, max_latency),
                )
            })
            .collect::<Vec<_>>(),
    );

    SyntheticOracle {
        landscape,
        candidate_count: candidates.len(),
        global_optimum_ids,
        global_optimum_quality: best_quality,
        top_k_ids,
        pareto_front_ids,
        exact_quality_latency_hypervolume,
    }
}

/// Run one search policy with a hard budget on full objective evaluations.
///
/// The correlated and shuffled proxy modes are based only on candidate features. The
/// AntiCorrelated mode is an oracle-derived adversarial control that reverses true quality;
/// it is deliberately not an implementable prior. This lets tests verify that a search
/// policy can be harmed by corrupted signals without giving the policy direct objective access.
pub fn run_synthetic_search(
    input_candidates: &[SyntheticCandidate],
    landscape: SyntheticLandscape,
    policy: SyntheticSearchPolicy,
    proxy_mode: SyntheticProxyMode,
    budget: usize,
    seed: u64,
) -> Result<SyntheticSearchRun, String> {
    if input_candidates.is_empty() {
        return Err("candidate space must not be empty".to_owned());
    }
    let mut candidates = input_candidates.to_vec();
    candidates.sort_by_key(|candidate| candidate.id());
    let mut seen = HashSet::new();
    for candidate in &candidates {
        if candidate.width >= LEVELS
            || candidate.depth >= LEVELS
            || candidate.attention >= LEVELS
            || candidate.recurrence >= LEVELS
        {
            return Err(format!("candidate {:?} has a gene outside 0..{LEVELS}", candidate.id()));
        }
        if !seen.insert(candidate.id()) {
            return Err(format!("duplicate candidate ID {:?}", candidate.id()));
        }
    }
    if candidates.len() != (LEVELS as usize).pow(4) {
        return Err(format!(
            "synthetic search requires the complete {}-candidate grid, found {}",
            (LEVELS as usize).pow(4),
            candidates.len()
        ));
    }
    if budget == 0 || budget > candidates.len() {
        return Err(format!("budget must be in 1..={}", candidates.len()));
    }

    let truth = exhaustive_oracle(landscape, 10);
    let candidate_by_id: HashMap<String, SyntheticCandidate> = candidates
        .iter()
        .copied()
        .map(|candidate| (candidate.id(), candidate))
        .collect();
    let proxy_by_id = make_proxy_scores(&candidates, landscape, proxy_mode, seed);
    let min_latency = candidates
        .iter()
        .map(|candidate| evaluate_candidate(*candidate, landscape).latency)
        .fold(f64::INFINITY, f64::min);
    let max_latency = candidates
        .iter()
        .map(|candidate| evaluate_candidate(*candidate, landscape).latency)
        .fold(f64::NEG_INFINITY, f64::max);

    let evaluated_ids = match policy {
        SyntheticSearchPolicy::Random => random_order(&candidates, seed)
            .into_iter()
            .take(budget)
            .map(SyntheticCandidate::id)
            .collect(),
        SyntheticSearchPolicy::ProxyRanked => {
            let mut ranked: Vec<SyntheticCandidate> = candidates.clone();
            ranked.sort_by(|a, b| {
                proxy_by_id[&b.id()]
                    .total_cmp(&proxy_by_id[&a.id()])
                    .then_with(|| a.id().cmp(&b.id()))
            });
            ranked
                .into_iter()
                .take(budget)
                .map(SyntheticCandidate::id)
                .collect()
        }
        SyntheticSearchPolicy::LocalSearch => local_search(&candidates, landscape, budget, seed),
    };

    let evaluated_set: HashSet<&str> = evaluated_ids.iter().map(String::as_str).collect();
    let mut best_candidate_id = String::new();
    let mut best_quality = f64::NEG_INFINITY;
    let mut objectives_by_id = HashMap::new();
    let mut evaluations_to_target = None;
    let mut running_best = f64::NEG_INFINITY;
    for (index, id) in evaluated_ids.iter().enumerate() {
        let candidate = candidate_by_id[id];
        let objectives = evaluate_candidate(candidate, landscape);
        if objectives.quality > best_quality {
            best_quality = objectives.quality;
            best_candidate_id.clone_from(id);
        }
        running_best = running_best.max(objectives.quality);
        if evaluations_to_target.is_none() && running_best >= OPTIMUM_TARGET {
            evaluations_to_target = Some(index + 1);
        }
        objectives_by_id.insert(id.clone(), objectives);
    }

    let exact_optimum_hit = truth
        .global_optimum_ids
        .iter()
        .any(|id| evaluated_set.contains(id.as_str()));
    let top_k_hits = truth
        .top_k_ids
        .iter()
        .filter(|id| evaluated_set.contains(id.as_str()))
        .count();
    let top_k_recall = if truth.top_k_ids.is_empty() {
        0.0
    } else {
        top_k_hits as f64 / truth.top_k_ids.len() as f64
    };
    let pareto_hits = truth
        .pareto_front_ids
        .iter()
        .filter(|id| evaluated_set.contains(id.as_str()))
        .count();
    let pareto_front_recall = if truth.pareto_front_ids.is_empty() {
        0.0
    } else {
        pareto_hits as f64 / truth.pareto_front_ids.len() as f64
    };
    let proxy_quality_spearman = spearman_proxy_quality(&candidates, landscape, &proxy_by_id);
    let normalized_hypervolume = hypervolume_2d(
        &objectives_by_id
            .values()
            .map(|objectives| {
                (
                    objectives.quality,
                    normalize_latency_utility(objectives.latency, min_latency, max_latency),
                )
            })
            .collect::<Vec<_>>(),
    );

    Ok(SyntheticSearchRun {
        landscape,
        policy,
        proxy_mode,
        seed,
        budget,
        evaluated_ids,
        best_evaluated_id: best_candidate_id,
        best_evaluated_quality: best_quality,
        simple_regret: (truth.global_optimum_quality - best_quality).max(0.0),
        exact_optimum_hit,
        top_k_recall,
        pareto_front_recall,
        normalized_hypervolume,
        proxy_quality_spearman,
        evaluations_to_target,
    })
}

/// Compare a policy across the supplied independent seeds using identical search budgets.
pub fn aggregate_runs(runs: &[SyntheticSearchRun]) -> Result<SyntheticSearchAggregate, String> {
    let Some(first) = runs.first() else {
        return Err("at least one run is required".to_owned());
    };
    let mut seeds = HashSet::new();
    for run in runs {
        if run.landscape != first.landscape
            || run.policy != first.policy
            || run.proxy_mode != first.proxy_mode
            || run.budget != first.budget
        {
            return Err("aggregate requires one landscape, policy, proxy mode, and budget".to_owned());
        }
        if !seeds.insert(run.seed) {
            return Err(format!("duplicate seed {}", run.seed));
        }
    }
    let count = runs.len() as f64;
    let mean_regret = runs.iter().map(|run| run.simple_regret).sum::<f64>() / count;
    let variance = if runs.len() < 2 {
        0.0
    } else {
        runs.iter()
            .map(|run| (run.simple_regret - mean_regret).powi(2))
            .sum::<f64>()
            / (runs.len() - 1) as f64
    };
    let target_runs: Vec<f64> = runs
        .iter()
        .filter_map(|run| run.evaluations_to_target.map(|value| value as f64))
        .collect();

    Ok(SyntheticSearchAggregate {
        landscape: first.landscape,
        policy: first.policy,
        proxy_mode: first.proxy_mode,
        budget: first.budget,
        seed_count: runs.len(),
        mean_simple_regret: mean_regret,
        simple_regret_sample_sd: variance.sqrt(),
        optimum_hit_rate: runs.iter().filter(|run| run.exact_optimum_hit).count() as f64 / count,
        mean_top_k_recall: runs.iter().map(|run| run.top_k_recall).sum::<f64>() / count,
        mean_pareto_front_recall: runs
            .iter()
            .map(|run| run.pareto_front_recall)
            .sum::<f64>()
            / count,
        target_hit_rate: runs
            .iter()
            .filter(|run| run.evaluations_to_target.is_some())
            .count() as f64
            / count,
        mean_normalized_hypervolume: runs
            .iter()
            .map(|run| run.normalized_hypervolume)
            .sum::<f64>()
            / count,
        mean_proxy_quality_spearman: runs
            .iter()
            .map(|run| run.proxy_quality_spearman)
            .sum::<f64>()
            / count,
        mean_evaluations_to_target: if target_runs.is_empty() {
            None
        } else {
            Some(target_runs.iter().sum::<f64>() / target_runs.len() as f64)
        },
    })
}

fn local_search(
    candidates: &[SyntheticCandidate],
    landscape: SyntheticLandscape,
    budget: usize,
    seed: u64,
) -> Vec<String> {
    let by_id: HashMap<String, SyntheticCandidate> = candidates
        .iter()
        .copied()
        .map(|candidate| (candidate.id(), candidate))
        .collect();
    let order = random_order(candidates, seed);
    let mut evaluated: Vec<String> = Vec::with_capacity(budget);
    let mut evaluated_set = HashSet::new();
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut next_restart = 0usize;
    let mut current_id = order[0].id();

    while evaluated.len() < budget {
        if !evaluated_set.contains(&current_id) {
            let candidate = by_id[&current_id];
            let quality = evaluate_candidate(candidate, landscape).quality;
            evaluated.push(current_id.clone());
            evaluated_set.insert(current_id.clone());
            scores.insert(current_id.clone(), quality);
        }
        if evaluated.len() >= budget {
            break;
        }

        let current = by_id[&current_id];
        let mut neighbors = neighbor_candidates(current);
        neighbors.retain(|candidate| !evaluated_set.contains(&candidate.id()));
        neighbors.sort_by_key(|candidate| candidate.id());
        if neighbors.is_empty() {
            current_id = next_unvisited(&order, &evaluated_set, &mut next_restart)
                .expect("unvisited candidate exists while budget is not exhausted")
                .id();
            continue;
        }

        let mut best_neighbor: Option<(String, f64)> = None;
        for neighbor in neighbors {
            if evaluated.len() >= budget {
                break;
            }
            let id = neighbor.id();
            if evaluated_set.contains(&id) {
                continue;
            }
            let quality = evaluate_candidate(neighbor, landscape).quality;
            evaluated.push(id.clone());
            evaluated_set.insert(id.clone());
            scores.insert(id.clone(), quality);
            if best_neighbor
                .as_ref()
                .is_none_or(|(_, best_score)| quality > *best_score)
            {
                best_neighbor = Some((id, quality));
            }
        }

        let current_score = scores[&current_id];
        if let Some((neighbor_id, neighbor_score)) = best_neighbor {
            if neighbor_score > current_score {
                current_id = neighbor_id;
                continue;
            }
        }
        if evaluated.len() < budget {
            current_id = next_unvisited(&order, &evaluated_set, &mut next_restart)
                .expect("unvisited candidate exists while budget is not exhausted")
                .id();
        }
    }
    evaluated
}

fn neighbor_candidates(candidate: SyntheticCandidate) -> Vec<SyntheticCandidate> {
    let genes = candidate.genes();
    let mut neighbors = Vec::with_capacity(8);
    for coordinate in 0..genes.len() {
        for step in [-1i8, 1i8] {
            let next_value = genes[coordinate] as i8 + step;
            if !(0..LEVELS as i8).contains(&next_value) {
                continue;
            }
            let mut next = genes;
            next[coordinate] = next_value as u8;
            neighbors.push(SyntheticCandidate {
                width: next[0],
                depth: next[1],
                attention: next[2],
                recurrence: next[3],
            });
        }
    }
    neighbors
}

fn next_unvisited<'a>(
    order: &'a [SyntheticCandidate],
    evaluated: &HashSet<String>,
    cursor: &mut usize,
) -> Option<&'a SyntheticCandidate> {
    while *cursor < order.len() {
        let candidate = &order[*cursor];
        *cursor += 1;
        if !evaluated.contains(&candidate.id()) {
            return Some(candidate);
        }
    }
    None
}

fn random_order(
    candidates: &[SyntheticCandidate],
    seed: u64,
) -> Vec<SyntheticCandidate> {
    let mut order = candidates.to_vec();
    let mut state = seed;
    for index in (1..order.len()).rev() {
        let selected = (next_random(&mut state) as usize) % (index + 1);
        order.swap(index, selected);
    }
    order
}

fn make_proxy_scores(
    candidates: &[SyntheticCandidate],
    landscape: SyntheticLandscape,
    mode: SyntheticProxyMode,
    seed: u64,
) -> HashMap<String, f64> {
    let mut base_scores = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        // Intentionally imperfect complexity prior, not a read of the true objective:
        // it favors width/attention and penalizes recurrence, while being weak on depth.
        let prior = 0.30 * candidate.width as f64
            + 0.25 * candidate.attention as f64
            + 0.15 * candidate.depth as f64
            - 0.10 * candidate.recurrence as f64;
        let noise = deterministic_noise(seed ^ candidate_key(candidate));
        base_scores.push((prior + noise * 0.20, candidate.id()));
    }

    match mode {
        SyntheticProxyMode::Correlated => base_scores
            .into_iter()
            .map(|(score, id)| (id, score))
            .collect(),
        SyntheticProxyMode::AntiCorrelated => candidates
            .iter()
            .map(|candidate| {
                (
                    candidate.id(),
                    -evaluate_candidate(*candidate, landscape).quality,
                )
            })
            .collect(),
        SyntheticProxyMode::Shuffled => {
            let ids: Vec<String> = base_scores.iter().map(|(_, id)| id.clone()).collect();
            let mut scores: Vec<f64> = base_scores.iter().map(|(score, _)| *score).collect();
            let mut state = seed ^ 0x6a09_e667_f3bc_c909;
            for index in (1..scores.len()).rev() {
                let selected = (next_random(&mut state) as usize) % (index + 1);
                scores.swap(index, selected);
            }
            ids.into_iter().zip(scores).map(|(id, score)| (id, score)).collect()
        }
    }
}

fn candidate_key(candidate: &SyntheticCandidate) -> u64 {
    let genes = candidate.genes();
    genes.iter().fold(0u64, |acc, gene| acc * 4 + *gene as u64)
}

fn deterministic_noise(seed: u64) -> f64 {
    let value = splitmix64(seed);
    let unit = (value >> 11) as f64 / ((1u64 << 53) as f64);
    2.0 * unit - 1.0
}

fn next_random(state: &mut u64) -> u64 {
    *state = splitmix64(*state);
    *state
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Diagnostic-only rank correlation of the candidate-only proxy against the known oracle.
/// The proxy still does not guide itself using this value; it is computed only for synthetic
/// benchmark reporting after proxy values have been generated.
fn spearman_proxy_quality(
    candidates: &[SyntheticCandidate],
    landscape: SyntheticLandscape,
    proxy_by_id: &HashMap<String, f64>,
) -> f64 {
    let proxy_scores: Vec<f64> = candidates
        .iter()
        .map(|candidate| proxy_by_id[&candidate.id()])
        .collect();
    let true_scores: Vec<f64> = candidates
        .iter()
        .map(|candidate| evaluate_candidate(*candidate, landscape).quality)
        .collect();
    pearson_correlation(&average_ranks(&proxy_scores), &average_ranks(&true_scores))
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|a, b| values[*a].total_cmp(&values[*b]).then_with(|| a.cmp(b)));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < indices.len() {
        let mut end = start + 1;
        while end < indices.len()
            && values[indices[start]].total_cmp(&values[indices[end]]).is_eq()
        {
            end += 1;
        }
        let average_rank = ((start + 1) as f64 + end as f64) / 2.0;
        for position in start..end {
            ranks[indices[position]] = average_rank;
        }
        start = end;
    }
    ranks
}

fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let count = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / count;
    let mean_b = b.iter().sum::<f64>() / count;
    let covariance = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum::<f64>();
    let variance_a = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>();
    let variance_b = b.iter().map(|y| (y - mean_b).powi(2)).sum::<f64>();
    let denominator = (variance_a * variance_b).sqrt();
    if denominator <= f64::EPSILON {
        0.0
    } else {
        (covariance / denominator).clamp(-1.0, 1.0)
    }
}

fn normalize_latency_utility(latency: f64, min_latency: f64, max_latency: f64) -> f64 {
    if (max_latency - min_latency).abs() <= f64::EPSILON {
        1.0
    } else {
        (1.0 - (latency - min_latency) / (max_latency - min_latency)).clamp(0.0, 1.0)
    }
}

/// Exact 2-D hypervolume for unit-normalized maximization points, with reference (0, 0).
fn hypervolume_2d(points: &[(f64, f64)]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let mut xs: Vec<f64> = points
        .iter()
        .map(|(quality, _)| quality.clamp(0.0, 1.0))
        .collect();
    xs.push(0.0);
    xs.sort_by(f64::total_cmp);
    xs.dedup_by(|left, right| (*left - *right).abs() <= f64::EPSILON);

    let mut volume = 0.0;
    for interval in xs.windows(2) {
        let left = interval[0];
        let right = interval[1];
        let maximum_y = points
            .iter()
            .filter(|(quality, _)| *quality >= right)
            .map(|(_, utility)| utility.clamp(0.0, 1.0))
            .fold(0.0, f64::max);
        volume += (right - left) * maximum_y;
    }
    volume.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerates_exactly_256_unique_candidates() {
        let candidates = enumerate_candidates();
        assert_eq!(candidates.len(), 256);
        assert_eq!(
            candidates.iter().map(|candidate| candidate.id()).collect::<HashSet<_>>().len(),
            256
        );
    }

    #[test]
    fn both_landscapes_have_a_unique_known_global_optimum() {
        for landscape in [SyntheticLandscape::SmoothTradeoff, SyntheticLandscape::Deceptive] {
            let oracle = exhaustive_oracle(landscape, 10);
            assert_eq!(oracle.candidate_count, 256);
            assert_eq!(oracle.global_optimum_quality, 1.0);
            assert_eq!(oracle.global_optimum_ids, vec!["w3-d2-a3-r1".to_owned()]);
            assert_eq!(oracle.global_optimum_ids.len(), 1);
            assert!(!oracle.pareto_front_ids.is_empty());
            assert!(oracle.exact_pareto_hypervolume > 0.0);
        }
    }

    #[test]
    fn deceptive_landscape_has_a_strict_local_trap() {
        let trap = SyntheticCandidate {
            width: 0,
            depth: 0,
            attention: 0,
            recurrence: 3,
        };
        let trap_quality = evaluate_candidate(trap, SyntheticLandscape::Deceptive).quality;
        assert!((trap_quality - 0.86).abs() < f64::EPSILON);
        for neighbor in neighbor_candidates(trap) {
            assert!(
                evaluate_candidate(neighbor, SyntheticLandscape::Deceptive).quality < trap_quality
            );
        }
        let global = exhaustive_oracle(SyntheticLandscape::Deceptive, 10);
        assert_eq!(global.global_optimum_quality, 1.0);
        assert!(!global.global_optimum_ids.contains(&trap.id()));
    }

    #[test]
    fn every_search_policy_obeys_the_exact_evaluation_budget() {
        let candidates = enumerate_candidates();
        for policy in [
            SyntheticSearchPolicy::Random,
            SyntheticSearchPolicy::LocalSearch,
            SyntheticSearchPolicy::ProxyRanked,
        ] {
            let result = run_synthetic_search(
                &candidates,
                SyntheticLandscape::SmoothTradeoff,
                policy,
                SyntheticProxyMode::Correlated,
                32,
                101,
            )
            .expect("valid synthetic search");
            assert_eq!(result.evaluated_ids.len(), 32);
            assert_eq!(result.budget, 32);
            assert!(result.simple_regret >= 0.0);
        }
    }

    #[test]
    fn every_policy_is_reproducible_for_the_same_seed() {
        let candidates = enumerate_candidates();
        for (policy, proxy_mode) in [
            (SyntheticSearchPolicy::Random, SyntheticProxyMode::Correlated),
            (SyntheticSearchPolicy::LocalSearch, SyntheticProxyMode::Correlated),
            (SyntheticSearchPolicy::ProxyRanked, SyntheticProxyMode::Correlated),
            (SyntheticSearchPolicy::ProxyRanked, SyntheticProxyMode::Shuffled),
            (SyntheticSearchPolicy::ProxyRanked, SyntheticProxyMode::AntiCorrelated),
        ] {
            let a = run_synthetic_search(
                &candidates,
                SyntheticLandscape::Deceptive,
                policy,
                proxy_mode,
                64,
                0x1234,
            )
            .unwrap();
            let b = run_synthetic_search(
                &candidates,
                SyntheticLandscape::Deceptive,
                policy,
                proxy_mode,
                64,
                0x1234,
            )
            .unwrap();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn candidate_input_order_does_not_change_deterministic_results() {
        let mut candidates = enumerate_candidates();
        let forward = run_synthetic_search(
            &candidates,
            SyntheticLandscape::SmoothTradeoff,
            SyntheticSearchPolicy::ProxyRanked,
            SyntheticProxyMode::Correlated,
            32,
            77,
        )
        .unwrap();
        candidates.reverse();
        let reverse = run_synthetic_search(
            &candidates,
            SyntheticLandscape::SmoothTradeoff,
            SyntheticSearchPolicy::ProxyRanked,
            SyntheticProxyMode::Correlated,
            32,
            77,
        )
        .unwrap();
        assert_eq!(forward.evaluated_ids, reverse.evaluated_ids);
        assert_eq!(forward.simple_regret, reverse.simple_regret);
    }

    #[test]
    fn proxy_modes_produce_distinct_rankings_without_changing_ground_truth() {
        let candidates = enumerate_candidates();
        let oracle = exhaustive_oracle(SyntheticLandscape::SmoothTradeoff, 10);
        let correlated = run_synthetic_search(
            &candidates,
            SyntheticLandscape::SmoothTradeoff,
            SyntheticSearchPolicy::ProxyRanked,
            SyntheticProxyMode::Correlated,
            32,
            17,
        )
        .unwrap();
        let shuffled = run_synthetic_search(
            &candidates,
            SyntheticLandscape::SmoothTradeoff,
            SyntheticSearchPolicy::ProxyRanked,
            SyntheticProxyMode::Shuffled,
            32,
            17,
        )
        .unwrap();
        let anti = run_synthetic_search(
            &candidates,
            SyntheticLandscape::SmoothTradeoff,
            SyntheticSearchPolicy::ProxyRanked,
            SyntheticProxyMode::AntiCorrelated,
            32,
            17,
        )
        .unwrap();
        assert_ne!(correlated.evaluated_ids, shuffled.evaluated_ids);
        assert_ne!(correlated.evaluated_ids, anti.evaluated_ids);
        assert!(anti.best_evaluated_quality < correlated.best_evaluated_quality);
        assert!(correlated.simple_regret.is_finite());
        assert_eq!(oracle.global_optimum_quality, 1.0);
    }

    #[test]
    fn proxy_rank_correlation_diagnostic_identifies_shuffled_and_adversarial_controls() {
        let candidates = enumerate_candidates();
        let correlated = run_synthetic_search(
            &candidates,
            SyntheticLandscape::SmoothTradeoff,
            SyntheticSearchPolicy::ProxyRanked,
            SyntheticProxyMode::Correlated,
            32,
            17,
        )
        .unwrap();
        let shuffled = run_synthetic_search(
            &candidates,
            SyntheticLandscape::SmoothTradeoff,
            SyntheticSearchPolicy::ProxyRanked,
            SyntheticProxyMode::Shuffled,
            32,
            17,
        )
        .unwrap();
        let anti = run_synthetic_search(
            &candidates,
            SyntheticLandscape::SmoothTradeoff,
            SyntheticSearchPolicy::ProxyRanked,
            SyntheticProxyMode::AntiCorrelated,
            32,
            17,
        )
        .unwrap();
        assert!(correlated.proxy_quality_spearman > 0.4);
        assert!(shuffled.proxy_quality_spearman.abs() < 0.2);
        assert!(anti.proxy_quality_spearman < -0.99);
    }

    #[test]
    fn aggregation_rejects_mixed_budgets_or_duplicate_seeds() {
        let candidates = enumerate_candidates();
        let first = run_synthetic_search(
            &candidates,
            SyntheticLandscape::SmoothTradeoff,
            SyntheticSearchPolicy::Random,
            SyntheticProxyMode::Correlated,
            16,
            1,
        )
        .unwrap();
        let second = run_synthetic_search(
            &candidates,
            SyntheticLandscape::SmoothTradeoff,
            SyntheticSearchPolicy::Random,
            SyntheticProxyMode::Correlated,
            16,
            2,
        )
        .unwrap();
        let aggregate = aggregate_runs(&[first.clone(), second.clone()]).unwrap();
        assert_eq!(aggregate.seed_count, 2);
        assert!(aggregate.simple_regret_sample_sd >= 0.0);

        assert!(aggregate_runs(&[first.clone(), first.clone()]).is_err());
        let mut mixed_budget = second;
        mixed_budget.budget = 32;
        assert!(aggregate_runs(&[first, mixed_budget]).is_err());
    }

    #[test]
    fn hypervolume_is_monotone_when_a_new_nondominated_point_is_added() {
        let base = hypervolume_2d(&[(0.4, 0.8), (0.7, 0.3)]);
        let extended = hypervolume_2d(&[(0.4, 0.8), (0.7, 0.3), (0.9, 0.9)]);
        assert!(extended > base);
    }

    #[test]
    fn invalid_budgets_and_out_of_space_candidates_are_rejected() {
        let candidates = enumerate_candidates();
        assert!(run_synthetic_search(
            &candidates,
            SyntheticLandscape::SmoothTradeoff,
            SyntheticSearchPolicy::Random,
            SyntheticProxyMode::Correlated,
            0,
            1
        )
        .is_err());
        assert!(run_synthetic_search(
            &candidates,
            SyntheticLandscape::SmoothTradeoff,
            SyntheticSearchPolicy::Random,
            SyntheticProxyMode::Correlated,
            257,
            1
        )
        .is_err());
        let invalid = vec![SyntheticCandidate {
            width: 9,
            depth: 0,
            attention: 0,
            recurrence: 0,
        }];
        assert!(run_synthetic_search(
            &invalid,
            SyntheticLandscape::SmoothTradeoff,
            SyntheticSearchPolicy::Random,
            SyntheticProxyMode::Correlated,
            1,
            1
        )
        .is_err());
    }
}
