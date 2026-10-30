//! NAS-Bench-201 proxy-local-search trajectory archive and promotion planner.
//!
//! This module implements a target-free candidate generation/retention primitive inspired by
//! R-MF-NAS. It preserves first-improvement and 2-opt escape candidates rather than only
//! passing the final proxy-ranked elite to successive halving.
//!
//! It does NOT claim to reproduce the paper's full Search Trajectory Network protocol, nor does
//! it use target labels. Promotion quotas are explicit and can be passed to the multi-fidelity
//! query firewall for controlled evaluations.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt;

use sha2::{Digest, Sha256};

use super::tabular_nas_firewall::ArchitectureFeatures;

const NB201_EDGES: usize = 6;
const NB201_OPERATIONS: u8 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyDirection {
    Maximize,
    Minimize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrajectorySearchConfig {
    pub proxy_id: String,
    pub target_metric_id: String,
    pub proxy_direction: ProxyDirection,
    pub seed: u64,
    pub max_walks: usize,
    pub max_unique_visits: usize,
    pub max_moves_per_walk: usize,
    pub max_escapes_per_walk: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectoryEventKind {
    Start,
    FirstImprovement,
    LocalOptimum,
    TwoOptEscape,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryEvent {
    pub event_index: usize,
    pub walk_index: usize,
    pub kind: TrajectoryEventKind,
    pub architecture_id: String,
    pub from_architecture_id: Option<String>,
    pub proxy_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchTrajectoryArchive {
    pub config: TrajectorySearchConfig,
    /// True only when all 5^6 NAS-Bench-201 topologies are present. Otherwise local-optimum
    /// events are relative to the supplied catalog, not the complete search space.
    pub catalog_is_complete: bool,
    pub proxy_id: String,
    pub proxy_direction: ProxyDirection,
    pub seed: u64,
    pub feature_catalog_sha256: String,
    pub visited_architecture_ids: Vec<String>,
    pub start_architecture_ids: Vec<String>,
    pub first_improvement_architecture_ids: Vec<String>,
    pub local_optimum_architecture_ids: Vec<String>,
    pub escape_architecture_ids: Vec<String>,
    pub events: Vec<TrajectoryEvent>,
    pub archive_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrajectorySearchError {
    pub field: String,
    pub message: String,
}

impl TrajectorySearchError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for TrajectorySearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for TrajectorySearchError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionSource {
    ProxyElite,
    FirstImprovement,
    Escape,
    LocalOptimum,
    TrajectoryFallback,
    TopologyDiversity,
    RankedFallback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionCandidate {
    pub architecture_id: String,
    pub source: PromotionSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionMix {
    pub promotion_count: usize,
    pub proxy_elite_quota: usize,
    pub trajectory_quota: usize,
    pub topology_diversity_quota: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialPoolStrategy {
    Random,
    SingleProxyRanked,
    TrajectoryPortfolio,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitialCandidatePool {
    pub strategy: InitialPoolStrategy,
    pub candidate_ids: Vec<String>,
    pub seed: Option<u64>,
    pub source_evidence_sha256: String,
    pub pool_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionPlan {
    pub proxy_id: String,
    pub archive_sha256: String,
    pub candidates: Vec<PromotionCandidate>,
    pub proxy_elite_count: usize,
    pub trajectory_count: usize,
    pub diversity_count: usize,
    pub fallback_count: usize,
    pub plan_sha256: String,
}

/// Run deterministic first-improvement local search with bounded 2-opt escapes.
///
/// A proxy score is used only as a target-free search objective. No final-accuracy or training
/// label is accepted by this function. Topology encodings must be six NAS-Bench-201 operation
/// indices in 0..5, matching the canonical architecture-engineering import.
pub fn build_trajectory_archive(
    features: &[ArchitectureFeatures],
    config: &TrajectorySearchConfig,
) -> Result<SearchTrajectoryArchive, TrajectorySearchError> {
    validate_search_inputs(features, config)?;

    let mut by_id: HashMap<String, &ArchitectureFeatures> =
        HashMap::with_capacity(features.len());
    let mut by_topology: HashMap<Vec<u8>, String> =
        HashMap::with_capacity(features.len());
    let mut all_ids: Vec<String> = Vec::with_capacity(features.len());

    for feature in features {
        let topology = feature.topology_encoding.clone();
        if by_topology
            .insert(topology, feature.architecture_id.clone())
            .is_some()
        {
            return Err(TrajectorySearchError::new(
                "features.topology_encoding",
                "duplicate topology encoding is not allowed",
            ));
        }
        by_id.insert(feature.architecture_id.clone(), feature);
        all_ids.push(feature.architecture_id.clone());
    }
    all_ids.sort();

    let mut rng = SplitMix64::new(config.seed);
    let mut globally_visited = HashSet::with_capacity(config.max_unique_visits);
    let mut visited_order = Vec::with_capacity(config.max_unique_visits.min(features.len()));
    let mut start_ids = Vec::new();
    let mut first_improvement_ids = Vec::new();
    let mut local_optima_ids = Vec::new();
    let mut escape_ids = Vec::new();
    let mut events = Vec::new();

    for walk_index in 0..config.max_walks {
        if globally_visited.len() >= config.max_unique_visits.min(features.len()) {
            break;
        }

        let unvisited_starts: Vec<&String> = all_ids
            .iter()
            .filter(|architecture_id| !globally_visited.contains(architecture_id.as_str()))
            .collect();
        if unvisited_starts.is_empty() {
            break;
        }

        let start_index = (rng.next_u64() as usize) % unvisited_starts.len();
        let mut current_id = unvisited_starts[start_index].clone().clone();
        let mut path_seen: HashSet<String> = HashSet::new();
        let mut moves_this_walk = 0usize;
        let mut escapes_this_walk = 0usize;

        path_seen.insert(current_id.clone());
        push_visited(
            &current_id,
            &mut globally_visited,
            &mut visited_order,
            config.max_unique_visits.min(features.len()),
        );
        start_ids.push(current_id.clone());
        push_event(
            &mut events,
            walk_index,
            TrajectoryEventKind::Start,
            current_id.clone(),
            None,
            proxy_score(by_id[&current_id], &config.proxy_id),
        );

        while moves_this_walk < config.max_moves_per_walk
            && globally_visited.len() < config.max_unique_visits.min(features.len())
        {
            let current_score = proxy_score(by_id[&current_id], &config.proxy_id);
            let one_opt = one_opt_neighbors(
                by_id[&current_id],
                &by_topology,
                &mut rng,
            );
            let first_improvement = one_opt.into_iter().find(|candidate_id| {
                if path_seen.contains(candidate_id) {
                    return false;
                }
                let candidate_score = proxy_score(by_id[candidate_id], &config.proxy_id);
                is_better(candidate_score, current_score, config.proxy_direction)
            });

            if let Some(next_id) = first_improvement {
                let next_score = proxy_score(by_id[&next_id], &config.proxy_id);
                let from_id = current_id.clone();
                current_id = next_id.clone();
                path_seen.insert(next_id.clone());
                push_visited(
                    &next_id,
                    &mut globally_visited,
                    &mut visited_order,
                    config.max_unique_visits.min(features.len()),
                );
                first_improvement_ids.push(next_id.clone());
                push_event(
                    &mut events,
                    walk_index,
                    TrajectoryEventKind::FirstImprovement,
                    next_id,
                    Some(from_id),
                    next_score,
                );
                moves_this_walk += 1;
                continue;
            }

            local_optima_ids.push(current_id.clone());
            push_event(
                &mut events,
                walk_index,
                TrajectoryEventKind::LocalOptimum,
                current_id.clone(),
                None,
                current_score,
            );

            if escapes_this_walk >= config.max_escapes_per_walk {
                break;
            }

            let two_opt = two_opt_neighbors(
                by_id[&current_id],
                &by_topology,
                &mut rng,
            );
            let escape = two_opt
                .into_iter()
                .find(|candidate_id| !path_seen.contains(candidate_id));

            let Some(escape_id) = escape else {
                break;
            };
            let from_id = current_id.clone();
            current_id = escape_id.clone();
            path_seen.insert(escape_id.clone());
            push_visited(
                &escape_id,
                &mut globally_visited,
                &mut visited_order,
                config.max_unique_visits.min(features.len()),
            );
            escape_ids.push(escape_id.clone());
            push_event(
                &mut events,
                walk_index,
                TrajectoryEventKind::TwoOptEscape,
                escape_id.clone(),
                Some(from_id),
                proxy_score(by_id[&escape_id], &config.proxy_id),
            );
            escapes_this_walk += 1;
            moves_this_walk += 1;
        }
    }

    let start_ids = deduplicate_in_order(start_ids);
    let first_improvement_ids = deduplicate_in_order(first_improvement_ids);
    let local_optima_ids = deduplicate_in_order(local_optima_ids);
    let escape_ids = deduplicate_in_order(escape_ids);
    let catalog_is_complete =
        features.len() == (NB201_OPERATIONS as usize).pow(NB201_EDGES as u32);
    let feature_catalog_sha256 = digest_feature_catalog(features, &config.proxy_id);
    let archive_sha256 = digest_archive(
        config,
        catalog_is_complete,
        &feature_catalog_sha256,
        &visited_order,
        &start_ids,
        &first_improvement_ids,
        &local_optima_ids,
        &escape_ids,
        &events,
    );
    Ok(SearchTrajectoryArchive {
        config: config.clone(),
        catalog_is_complete,
        proxy_id: config.proxy_id.clone(),
        proxy_direction: config.proxy_direction,
        seed: config.seed,
        feature_catalog_sha256,
        visited_architecture_ids: visited_order,
        start_architecture_ids: start_ids,
        first_improvement_architecture_ids: first_improvement_ids,
        local_optimum_architecture_ids: local_optima_ids,
        escape_architecture_ids: escape_ids,
        events,
        archive_sha256,
    })
}

/// Construct a reproducible random candidate pool from public IDs/topologies only.
/// No proxy score, target label or training observation is read.
pub fn random_candidate_pool(
    candidates: &[ArchitectureFeatures],
    pool_size: usize,
    seed: u64,
) -> Result<InitialCandidatePool, TrajectorySearchError> {
    validate_public_catalog(candidates, pool_size)?;
    let mut candidate_ids: Vec<String> = candidates
        .iter()
        .map(|candidate| candidate.architecture_id.clone())
        .collect();
    candidate_ids.sort();
    let source_evidence_sha256 = digest_public_catalog(candidates);
    let mut rng = SplitMix64::new(seed);
    shuffle(&mut candidate_ids, &mut rng);
    candidate_ids.truncate(pool_size);
    let pool_sha256 = digest_initial_pool(
        InitialPoolStrategy::Random,
        &candidate_ids,
        Some(seed),
        &source_evidence_sha256,
    );
    Ok(InitialCandidatePool {
        strategy: InitialPoolStrategy::Random,
        candidate_ids,
        seed: Some(seed),
        source_evidence_sha256,
        pool_sha256,
    })
}

/// Build a deterministic top-k candidate pool using one declared proxy. The target metric is
/// explicitly supplied so it cannot be accidentally used as a ranking feature.
pub fn proxy_ranked_candidate_pool(
    candidates: &[ArchitectureFeatures],
    proxy_id: &str,
    target_metric_id: &str,
    direction: ProxyDirection,
    pool_size: usize,
) -> Result<InitialCandidatePool, TrajectorySearchError> {
    validate_public_catalog(candidates, pool_size)?;
    if proxy_id.trim().is_empty() || proxy_id == target_metric_id {
        return Err(TrajectorySearchError::new(
            "proxy_id",
            "proxy must be non-empty and distinct from target metric",
        ));
    }
    for (index, candidate) in candidates.iter().enumerate() {
        if candidate.proxy_scores.contains_key(target_metric_id)
            || candidate.proxy_times_seconds.contains_key(target_metric_id)
        {
            return Err(TrajectorySearchError::new(
                format!("candidates[{index}]"),
                "target metric cannot appear in a target-free initial pool feature table",
            ));
        }
        let Some(score) = candidate.proxy_scores.get(proxy_id) else {
            return Err(TrajectorySearchError::new(
                format!("candidates[{index}].proxy_scores"),
                format!("missing proxy {proxy_id:?}"),
            ));
        };
        if !score.is_finite() {
            return Err(TrajectorySearchError::new(
                format!("candidates[{index}].proxy_scores.{proxy_id}"),
                "proxy score must be finite",
            ));
        }
        let Some(runtime) = candidate.proxy_times_seconds.get(proxy_id) else {
            return Err(TrajectorySearchError::new(
                format!("candidates[{index}].proxy_times_seconds"),
                format!("missing runtime for proxy {proxy_id:?}"),
            ));
        };
        if !runtime.is_finite() || *runtime < 0.0 {
            return Err(TrajectorySearchError::new(
                format!("candidates[{index}].proxy_times_seconds.{proxy_id}"),
                "proxy runtime must be finite and non-negative",
            ));
        }
    }
    let mut ranked: Vec<&ArchitectureFeatures> = candidates.iter().collect();
    ranked.sort_by(|left, right| {
        let left_score = left.proxy_scores[proxy_id];
        let right_score = right.proxy_scores[proxy_id];
        let order = match direction {
            ProxyDirection::Maximize => right_score.total_cmp(&left_score),
            ProxyDirection::Minimize => left_score.total_cmp(&right_score),
        };
        order.then_with(|| left.architecture_id.cmp(&right.architecture_id))
    });
    let candidate_ids: Vec<String> = ranked
        .into_iter()
        .take(pool_size)
        .map(|candidate| candidate.architecture_id.clone())
        .collect();
    let source_evidence_sha256 = digest_single_proxy_catalog(candidates, proxy_id, direction);
    let pool_sha256 = digest_initial_pool(
        InitialPoolStrategy::SingleProxyRanked,
        &candidate_ids,
        None,
        &source_evidence_sha256,
    );
    Ok(InitialCandidatePool {
        strategy: InitialPoolStrategy::SingleProxyRanked,
        candidate_ids,
        seed: None,
        source_evidence_sha256,
        pool_sha256,
    })
}

/// Convert a validated quota-based promotion plan to the common initial-pool interface.
pub fn trajectory_candidate_pool(
    plan: &PromotionPlan,
) -> Result<InitialCandidatePool, TrajectorySearchError> {
    if plan.candidates.is_empty() || plan.plan_sha256.len() != 64 {
        return Err(TrajectorySearchError::new(
            "promotion_plan",
            "must contain candidates and a 64-character plan digest",
        ));
    }
    let mut seen = HashSet::with_capacity(plan.candidates.len());
    let candidate_ids: Vec<String> = plan
        .candidates
        .iter()
        .map(|candidate| candidate.architecture_id.clone())
        .collect();
    if candidate_ids
        .iter()
        .any(|candidate_id| candidate_id.trim().is_empty() || !seen.insert(candidate_id.as_str()))
    {
        return Err(TrajectorySearchError::new(
            "promotion_plan.candidates",
            "candidate IDs must be non-empty and unique",
        ));
    }
    let source_evidence_sha256 = plan.plan_sha256.clone();
    let pool_sha256 = digest_initial_pool(
        InitialPoolStrategy::TrajectoryPortfolio,
        &candidate_ids,
        None,
        &source_evidence_sha256,
    );
    Ok(InitialCandidatePool {
        strategy: InitialPoolStrategy::TrajectoryPortfolio,
        candidate_ids,
        seed: None,
        source_evidence_sha256,
        pool_sha256,
    })
}

fn validate_public_catalog(
    candidates: &[ArchitectureFeatures],
    pool_size: usize,
) -> Result<(), TrajectorySearchError> {
    if candidates.is_empty() || pool_size == 0 || pool_size > candidates.len() {
        return Err(TrajectorySearchError::new(
            "pool_size",
            "candidate catalog must be non-empty and pool_size must be in 1..=candidate_count",
        ));
    }
    let mut ids = HashSet::with_capacity(candidates.len());
    let mut topologies = HashSet::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        if candidate.architecture_id.trim().is_empty()
            || !ids.insert(candidate.architecture_id.as_str())
        {
            return Err(TrajectorySearchError::new(
                format!("candidates[{index}].architecture_id"),
                "candidate IDs must be non-empty and unique",
            ));
        }
        if candidate.topology_encoding.len() != NB201_EDGES
            || candidate.topology_encoding.iter().any(|operation| *operation >= NB201_OPERATIONS)
        {
            return Err(TrajectorySearchError::new(
                format!("candidates[{index}].topology_encoding"),
                "must contain six NAS-Bench-201 operation indices in 0..5",
            ));
        }
        if !topologies.insert(candidate.topology_encoding.as_slice()) {
            return Err(TrajectorySearchError::new(
                format!("candidates[{index}].topology_encoding"),
                "topology encodings must be unique",
            ));
        }
    }
    Ok(())
}

fn digest_public_catalog(candidates: &[ArchitectureFeatures]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.nasbench201.public-candidate-catalog.v0.1");
    let mut ordered: Vec<&ArchitectureFeatures> = candidates.iter().collect();
    ordered.sort_by(|left, right| left.architecture_id.cmp(&right.architecture_id));
    hasher.update((ordered.len() as u64).to_be_bytes());
    for candidate in ordered {
        update_string(&mut hasher, &candidate.architecture_id);
        hasher.update((candidate.topology_encoding.len() as u64).to_be_bytes());
        hasher.update(&candidate.topology_encoding);
    }
    hex_digest(hasher.finalize().as_slice())
}

fn digest_single_proxy_catalog(
    candidates: &[ArchitectureFeatures],
    proxy_id: &str,
    direction: ProxyDirection,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.nasbench201.single-proxy-catalog.v0.1");
    update_string(&mut hasher, proxy_id);
    hasher.update([direction_byte(direction)]);
    let mut ordered: Vec<&ArchitectureFeatures> = candidates.iter().collect();
    ordered.sort_by(|left, right| left.architecture_id.cmp(&right.architecture_id));
    hasher.update((ordered.len() as u64).to_be_bytes());
    for candidate in ordered {
        update_string(&mut hasher, &candidate.architecture_id);
        hasher.update(candidate.proxy_scores[proxy_id].to_bits().to_be_bytes());
        hasher.update(candidate.proxy_times_seconds[proxy_id].to_bits().to_be_bytes());
        hasher.update((candidate.topology_encoding.len() as u64).to_be_bytes());
        hasher.update(&candidate.topology_encoding);
    }
    hex_digest(hasher.finalize().as_slice())
}

fn digest_initial_pool(
    strategy: InitialPoolStrategy,
    candidate_ids: &[String],
    seed: Option<u64>,
    source_digest: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.nasbench201.initial-candidate-pool.v0.1");
    hasher.update([match strategy {
        InitialPoolStrategy::Random => 0,
        InitialPoolStrategy::SingleProxyRanked => 1,
        InitialPoolStrategy::TrajectoryPortfolio => 2,
    }]);
    match seed {
        Some(value) => {
            hasher.update([1]);
            hasher.update(value.to_be_bytes());
        }
        None => hasher.update([0]),
    }
    update_string(&mut hasher, source_digest);
    hasher.update((candidate_ids.len() as u64).to_be_bytes());
    for candidate_id in candidate_ids {
        update_string(&mut hasher, candidate_id);
    }
    hex_digest(hasher.finalize().as_slice())
}

/// Build a final-fidelity promotion set from proxy elites, trajectory memory and topology
/// diversity. The quotas are explicit and auditable; any unfilled quota is filled from the
/// remaining proxy ranking and identified as fallback rather than silently re-labelled.
pub fn plan_promotions(
    features: &[ArchitectureFeatures],
    archive: &SearchTrajectoryArchive,
    mix: &PromotionMix,
) -> Result<PromotionPlan, TrajectorySearchError> {
    if features.is_empty() {
        return Err(TrajectorySearchError::new(
            "features",
            "at least one architecture is required",
        ));
    }
    if archive.proxy_id.trim().is_empty() {
        return Err(TrajectorySearchError::new("archive.proxy_id", "must be non-empty"));
    }
    if mix.promotion_count == 0 || mix.promotion_count > features.len() {
        return Err(TrajectorySearchError::new(
            "mix.promotion_count",
            "must be in 1..=architecture_count",
        ));
    }
    let reserved = mix
        .proxy_elite_quota
        .checked_add(mix.trajectory_quota)
        .and_then(|value| value.checked_add(mix.topology_diversity_quota))
        .ok_or_else(|| TrajectorySearchError::new("mix", "quota sum overflow"))?;
    if reserved > mix.promotion_count {
        return Err(TrajectorySearchError::new(
            "mix",
            "explicit quotas cannot exceed promotion_count",
        ));
    }
    if archive.proxy_id.is_empty() {
        return Err(TrajectorySearchError::new(
            "archive.proxy_id",
            "must be non-empty",
        ));
    }
    if archive.proxy_id != archive.config.proxy_id
        || archive.proxy_direction != archive.config.proxy_direction
        || archive.seed != archive.config.seed
    {
        return Err(TrajectorySearchError::new(
            "archive",
            "duplicated archive metadata does not match the frozen search configuration",
        ));
    }
    let supplied_catalog_sha256 = digest_feature_catalog(features, &archive.proxy_id);
    let supplied_catalog_is_complete =
        features.len() == (NB201_OPERATIONS as usize).pow(NB201_EDGES as u32);
    if supplied_catalog_is_complete != archive.catalog_is_complete {
        return Err(TrajectorySearchError::new(
            "archive.catalog_is_complete",
            "catalog-completeness flag does not match supplied topology catalog",
        ));
    }
    if supplied_catalog_sha256 != archive.feature_catalog_sha256 {
        return Err(TrajectorySearchError::new(
            "archive.feature_catalog_sha256",
            "promotion features do not match the catalog used to generate the trajectory",
        ));
    }
    let expected_archive_sha256 = digest_archive(
        &archive.config,
        archive.catalog_is_complete,
        &archive.feature_catalog_sha256,
        &archive.visited_architecture_ids,
        &archive.start_architecture_ids,
        &archive.first_improvement_architecture_ids,
        &archive.local_optimum_architecture_ids,
        &archive.escape_architecture_ids,
        &archive.events,
    );
    if expected_archive_sha256 != archive.archive_sha256 {
        return Err(TrajectorySearchError::new(
            "archive.archive_sha256",
            "trajectory archive content does not match its digest",
        ));
    }

    let mut by_id: HashMap<String, &ArchitectureFeatures> =
        HashMap::with_capacity(features.len());
    for (index, feature) in features.iter().enumerate() {
        if feature.architecture_id.trim().is_empty() {
            return Err(TrajectorySearchError::new(
                format!("features[{index}].architecture_id"),
                "must be non-empty",
            ));
        }
        if feature.topology_encoding.len() != NB201_EDGES
            || feature.topology_encoding.iter().any(|operation| *operation >= NB201_OPERATIONS)
        {
            return Err(TrajectorySearchError::new(
                format!("features[{index}].topology_encoding"),
                "must contain six NAS-Bench-201 operation indices in 0..5",
            ));
        }
        let Some(score) = feature.proxy_scores.get(&archive.proxy_id) else {
            return Err(TrajectorySearchError::new(
                format!("features[{index}].proxy_scores"),
                format!("missing proxy {:?}", archive.proxy_id),
            ));
        };
        if !score.is_finite() {
            return Err(TrajectorySearchError::new(
                format!("features[{index}].proxy_scores.{}", archive.proxy_id),
                "must be finite",
            ));
        }
        let Some(runtime) = feature.proxy_times_seconds.get(&archive.proxy_id) else {
            return Err(TrajectorySearchError::new(
                format!("features[{index}].proxy_times_seconds"),
                format!("missing runtime for proxy {:?}", archive.proxy_id),
            ));
        };
        if !runtime.is_finite() || *runtime < 0.0 {
            return Err(TrajectorySearchError::new(
                format!("features[{index}].proxy_times_seconds.{}", archive.proxy_id),
                "runtime must be finite and non-negative",
            ));
        }
        if by_id
            .insert(feature.architecture_id.clone(), feature)
            .is_some()
        {
            return Err(TrajectorySearchError::new(
                format!("features[{index}].architecture_id"),
                "duplicate architecture ID",
            ));
        }
    }

    for architecture_id in archive
        .visited_architecture_ids
        .iter()
        .chain(archive.start_architecture_ids.iter())
        .chain(archive.first_improvement_architecture_ids.iter())
        .chain(archive.local_optimum_architecture_ids.iter())
        .chain(archive.escape_architecture_ids.iter())
    {
        if !by_id.contains_key(architecture_id) {
            return Err(TrajectorySearchError::new(
                "archive",
                format!("unknown architecture ID {architecture_id:?}"),
            ));
        }
    }

    let mut ranked: Vec<&ArchitectureFeatures> = features.iter().collect();
    ranked.sort_by(|left, right| {
        let left_score = left.proxy_scores[&archive.proxy_id];
        let right_score = right.proxy_scores[&archive.proxy_id];
        let order = match archive.proxy_direction {
            ProxyDirection::Maximize => right_score.total_cmp(&left_score),
            ProxyDirection::Minimize => left_score.total_cmp(&right_score),
        };
        order.then_with(|| left.architecture_id.cmp(&right.architecture_id))
    });

    let mut candidates = Vec::with_capacity(mix.promotion_count);
    let mut selected = HashSet::with_capacity(mix.promotion_count);
    let mut proxy_elite_count = 0usize;
    let mut trajectory_count = 0usize;
    let mut diversity_count = 0usize;
    let mut fallback_count = 0usize;

    for feature in ranked.iter().take(mix.proxy_elite_quota) {
        if add_candidate(
            feature.architecture_id.clone(),
            PromotionSource::ProxyElite,
            &mut candidates,
            &mut selected,
        ) {
            proxy_elite_count += 1;
        }
    }

    let trajectory_order = interleave_trajectory_ids(archive);
    let mut trajectory_added = 0usize;
    for (architecture_id, source) in trajectory_order {
        if trajectory_added >= mix.trajectory_quota {
            break;
        }
        if add_candidate(
            architecture_id,
            source,
            &mut candidates,
            &mut selected,
        ) {
            trajectory_added += 1;
            trajectory_count += 1;
        }
    }
    while trajectory_added < mix.trajectory_quota {
        let next = ranked
            .iter()
            .find(|feature| !selected.contains(feature.architecture_id.as_str()));
        let Some(feature) = next else {
            break;
        };
        if add_candidate(
            feature.architecture_id.clone(),
            PromotionSource::TrajectoryFallback,
            &mut candidates,
            &mut selected,
        ) {
            trajectory_added += 1;
            fallback_count += 1;
        }
    }

    while diversity_count < mix.topology_diversity_quota {
        let next = ranked
            .iter()
            .filter(|feature| !selected.contains(feature.architecture_id.as_str()))
            .max_by(|left, right| {
                let left_distance = min_hamming_distance(
                    &left.topology_encoding,
                    &candidates,
                    &by_id,
                );
                let right_distance = min_hamming_distance(
                    &right.topology_encoding,
                    &candidates,
                    &by_id,
                );
                left_distance
                    .cmp(&right_distance)
                    .then_with(|| left.architecture_id.cmp(&right.architecture_id))
            });
        let Some(feature) = next else {
            break;
        };
        if add_candidate(
            feature.architecture_id.clone(),
            PromotionSource::TopologyDiversity,
            &mut candidates,
            &mut selected,
        ) {
            diversity_count += 1;
        } else {
            break;
        }
    }

    for feature in &ranked {
        if candidates.len() >= mix.promotion_count {
            break;
        }
        if add_candidate(
            feature.architecture_id.clone(),
            PromotionSource::RankedFallback,
            &mut candidates,
            &mut selected,
        ) {
            fallback_count += 1;
        }
    }

    if candidates.len() != mix.promotion_count {
        return Err(TrajectorySearchError::new(
            "promotion_plan",
            "could not fill requested number of unique promotions",
        ));
    }

    let plan_sha256 = digest_plan(archive, mix, &candidates);
    Ok(PromotionPlan {
        proxy_id: archive.proxy_id.clone(),
        archive_sha256: archive.archive_sha256.clone(),
        candidates,
        proxy_elite_count,
        trajectory_count,
        diversity_count,
        fallback_count,
        plan_sha256,
    })
}

fn validate_search_inputs(
    features: &[ArchitectureFeatures],
    config: &TrajectorySearchConfig,
) -> Result<(), TrajectorySearchError> {
    if features.is_empty() {
        return Err(TrajectorySearchError::new(
            "features",
            "at least one architecture is required",
        ));
    }
    if config.proxy_id.trim().is_empty() || config.target_metric_id.trim().is_empty() {
        return Err(TrajectorySearchError::new("proxy_id", "proxy and target metric IDs must be non-empty"));
    }
    if config.proxy_id == config.target_metric_id {
        return Err(TrajectorySearchError::new(
            "proxy_id",
            "the target metric cannot be reused as a proxy feature",
        ));
    }
    if config.max_walks == 0 || config.max_unique_visits == 0 || config.max_moves_per_walk == 0 {
        return Err(TrajectorySearchError::new(
            "budget",
            "walk, unique-visit and move budgets must be positive",
        ));
    }
    if !features.iter().any(|feature| feature.proxy_scores.contains_key(&config.proxy_id)) {
        return Err(TrajectorySearchError::new(
            "proxy_id",
            format!("proxy {:?} is not present", config.proxy_id),
        ));
    }

    let mut architecture_ids = HashSet::with_capacity(features.len());
    let mut topology_encodings = HashSet::with_capacity(features.len());
    for (index, feature) in features.iter().enumerate() {
        if feature.architecture_id.trim().is_empty()
            || !architecture_ids.insert(feature.architecture_id.as_str())
        {
            return Err(TrajectorySearchError::new(
                format!("features[{index}].architecture_id"),
                "architecture IDs must be non-empty and unique",
            ));
        }
        if feature.topology_encoding.len() != NB201_EDGES
            || feature
                .topology_encoding
                .iter()
                .any(|operation| *operation >= NB201_OPERATIONS)
        {
            return Err(TrajectorySearchError::new(
                format!("features[{index}].topology_encoding"),
                "must contain six operation indices in 0..5",
            ));
        }
        if !topology_encodings.insert(feature.topology_encoding.as_slice()) {
            return Err(TrajectorySearchError::new(
                format!("features[{index}].topology_encoding"),
                "duplicate topology encoding",
            ));
        }
        let Some(score) = feature.proxy_scores.get(&config.proxy_id) else {
            return Err(TrajectorySearchError::new(
                format!("features[{index}].proxy_scores"),
                "every candidate must have the selected proxy",
            ));
        };
        if !score.is_finite() {
            return Err(TrajectorySearchError::new(
                format!("features[{index}].proxy_scores.{}", config.proxy_id),
                "proxy score must be finite",
            ));
        }
        let Some(runtime) = feature.proxy_times_seconds.get(&config.proxy_id) else {
            return Err(TrajectorySearchError::new(
                format!("features[{index}].proxy_times_seconds"),
                "every candidate must have a runtime for the selected proxy",
            ));
        };
        if !runtime.is_finite() || *runtime < 0.0 {
            return Err(TrajectorySearchError::new(
                format!("features[{index}].proxy_times_seconds.{}", config.proxy_id),
                "proxy runtime must be finite and non-negative",
            ));
        }
    }
    Ok(())
}

fn proxy_score(feature: &ArchitectureFeatures, proxy_id: &str) -> f64 {
    feature.proxy_scores[proxy_id]
}

fn is_better(candidate: f64, current: f64, direction: ProxyDirection) -> bool {
    match direction {
        ProxyDirection::Maximize => candidate > current,
        ProxyDirection::Minimize => candidate < current,
    }
}

fn one_opt_neighbors(
    current: &ArchitectureFeatures,
    by_topology: &HashMap<Vec<u8>, String>,
    rng: &mut SplitMix64,
) -> Vec<String> {
    let mut neighbors = Vec::with_capacity(NB201_EDGES * (NB201_OPERATIONS as usize - 1));
    for edge in 0..NB201_EDGES {
        for operation in 0..NB201_OPERATIONS {
            if operation == current.topology_encoding[edge] {
                continue;
            }
            let mut topology = current.topology_encoding.clone();
            topology[edge] = operation;
            if let Some(architecture_id) = by_topology.get(&topology) {
                neighbors.push(architecture_id.clone());
            }
        }
    }
    shuffle(&mut neighbors, rng);
    neighbors
}

fn two_opt_neighbors(
    current: &ArchitectureFeatures,
    by_topology: &HashMap<Vec<u8>, String>,
    rng: &mut SplitMix64,
) -> Vec<String> {
    let mut neighbors = Vec::new();
    for left_edge in 0..NB201_EDGES {
        for right_edge in (left_edge + 1)..NB201_EDGES {
            for left_op in 0..NB201_OPERATIONS {
                if left_op == current.topology_encoding[left_edge] {
                    continue;
                }
                for right_op in 0..NB201_OPERATIONS {
                    if right_op == current.topology_encoding[right_edge] {
                        continue;
                    }
                    let mut topology = current.topology_encoding.clone();
                    topology[left_edge] = left_op;
                    topology[right_edge] = right_op;
                    if let Some(architecture_id) = by_topology.get(&topology) {
                        neighbors.push(architecture_id.clone());
                    }
                }
            }
        }
    }
    shuffle(&mut neighbors, rng);
    neighbors
}

fn push_visited(
    architecture_id: &str,
    visited: &mut HashSet<String>,
    order: &mut Vec<String>,
    limit: usize,
) {
    if order.len() < limit && visited.insert(architecture_id.to_owned()) {
        order.push(architecture_id.to_owned());
    }
}

fn push_event(
    events: &mut Vec<TrajectoryEvent>,
    walk_index: usize,
    kind: TrajectoryEventKind,
    architecture_id: String,
    from_architecture_id: Option<String>,
    proxy_score: f64,
) {
    let event_index = events.len();
    events.push(TrajectoryEvent {
        event_index,
        walk_index,
        kind,
        architecture_id,
        from_architecture_id,
        proxy_score,
    });
}

fn deduplicate_in_order(values: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::with_capacity(values.len());
    values
        .into_iter()
        .filter(|value| seen.insert(value.clone()))
        .collect()
}

fn interleave_trajectory_ids(
    archive: &SearchTrajectoryArchive,
) -> Vec<(String, PromotionSource)> {
    let groups = [
        (
            archive.first_improvement_architecture_ids.as_slice(),
            PromotionSource::FirstImprovement,
        ),
        (
            archive.escape_architecture_ids.as_slice(),
            PromotionSource::Escape,
        ),
        (
            archive.local_optimum_architecture_ids.as_slice(),
            PromotionSource::LocalOptimum,
        ),
    ];
    let mut positions = [0usize; 3];
    let mut output = Vec::new();
    loop {
        let mut emitted = false;
        for (index, (values, source)) in groups.iter().enumerate() {
            if let Some(value) = values.get(positions[index]) {
                output.push((value.clone(), *source));
                positions[index] += 1;
                emitted = true;
            }
        }
        if !emitted {
            break;
        }
    }
    output
}

fn add_candidate(
    architecture_id: String,
    source: PromotionSource,
    candidates: &mut Vec<PromotionCandidate>,
    selected: &mut HashSet<String>,
) -> bool {
    if selected.insert(architecture_id.clone()) {
        candidates.push(PromotionCandidate {
            architecture_id,
            source,
        });
        true
    } else {
        false
    }
}

fn min_hamming_distance(
    topology: &[u8],
    selected: &[PromotionCandidate],
    by_id: &HashMap<String, &ArchitectureFeatures>,
) -> usize {
    if selected.is_empty() {
        return NB201_EDGES;
    }
    selected
        .iter()
        .filter_map(|candidate| by_id.get(&candidate.architecture_id))
        .map(|feature| {
            topology
                .iter()
                .zip(feature.topology_encoding.iter())
                .filter(|(left, right)| left != right)
                .count()
        })
        .min()
        .unwrap_or(NB201_EDGES)
}

fn digest_feature_catalog(
    features: &[ArchitectureFeatures],
    proxy_id: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.nasbench201.proxy-feature-catalog.v0.1");
    update_string(&mut hasher, proxy_id);
    let mut ordered: Vec<&ArchitectureFeatures> = features.iter().collect();
    ordered.sort_by(|left, right| left.architecture_id.cmp(&right.architecture_id));
    hasher.update((ordered.len() as u64).to_be_bytes());
    for feature in ordered {
        update_string(&mut hasher, &feature.architecture_id);
        hasher.update((feature.topology_encoding.len() as u64).to_be_bytes());
        hasher.update(&feature.topology_encoding);
        let score = feature.proxy_scores.get(proxy_id).copied().unwrap_or(f64::NAN);
        let runtime = feature
            .proxy_times_seconds
            .get(proxy_id)
            .copied()
            .unwrap_or(f64::NAN);
        hasher.update(score.to_bits().to_be_bytes());
        hasher.update(runtime.to_bits().to_be_bytes());
    }
    hex_digest(hasher.finalize().as_slice())
}

fn digest_archive(
    config: &TrajectorySearchConfig,
    catalog_is_complete: bool,
    feature_catalog_sha256: &str,
    visited: &[String],
    starts: &[String],
    first_improvements: &[String],
    local_optima: &[String],
    escapes: &[String],
    events: &[TrajectoryEvent],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.nasbench201.trajectory-archive.v0.1");
    update_string(&mut hasher, &config.proxy_id);
    update_string(&mut hasher, &config.target_metric_id);
    hasher.update([direction_byte(config.proxy_direction)]);
    hasher.update(config.seed.to_be_bytes());
    hasher.update((config.max_walks as u64).to_be_bytes());
    hasher.update((config.max_unique_visits as u64).to_be_bytes());
    hasher.update((config.max_moves_per_walk as u64).to_be_bytes());
    hasher.update((config.max_escapes_per_walk as u64).to_be_bytes());
    hasher.update([catalog_is_complete as u8]);
    update_string(&mut hasher, feature_catalog_sha256);
    hasher.update((visited.len() as u64).to_be_bytes());
    for architecture_id in visited {
        update_string(&mut hasher, architecture_id);
    }
    for values in [starts, first_improvements, local_optima, escapes] {
        hasher.update((values.len() as u64).to_be_bytes());
        for architecture_id in values {
            update_string(&mut hasher, architecture_id);
        }
    }
    hasher.update((events.len() as u64).to_be_bytes());
    for event in events {
        hasher.update((event.event_index as u64).to_be_bytes());
        hasher.update((event.walk_index as u64).to_be_bytes());
        hasher.update([event_kind_byte(event.kind)]);
        update_string(&mut hasher, &event.architecture_id);
        match &event.from_architecture_id {
            Some(from_id) => {
                hasher.update([1]);
                update_string(&mut hasher, from_id);
            }
            None => hasher.update([0]),
        }
        hasher.update(event.proxy_score.to_bits().to_be_bytes());
    }
    hex_digest(hasher.finalize().as_slice())
}

fn digest_plan(
    archive: &SearchTrajectoryArchive,
    mix: &PromotionMix,
    candidates: &[PromotionCandidate],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.nasbench201.trajectory-promotion-plan.v0.1");
    update_string(&mut hasher, &archive.archive_sha256);
    hasher.update((mix.promotion_count as u64).to_be_bytes());
    hasher.update((mix.proxy_elite_quota as u64).to_be_bytes());
    hasher.update((mix.trajectory_quota as u64).to_be_bytes());
    hasher.update((mix.topology_diversity_quota as u64).to_be_bytes());
    hasher.update((candidates.len() as u64).to_be_bytes());
    for candidate in candidates {
        update_string(&mut hasher, &candidate.architecture_id);
        hasher.update([promotion_source_byte(candidate.source)]);
    }
    hex_digest(hasher.finalize().as_slice())
}

fn direction_byte(direction: ProxyDirection) -> u8 {
    match direction {
        ProxyDirection::Maximize => 0,
        ProxyDirection::Minimize => 1,
    }
}

fn event_kind_byte(kind: TrajectoryEventKind) -> u8 {
    match kind {
        TrajectoryEventKind::Start => 0,
        TrajectoryEventKind::FirstImprovement => 1,
        TrajectoryEventKind::LocalOptimum => 2,
        TrajectoryEventKind::TwoOptEscape => 3,
    }
}

fn promotion_source_byte(source: PromotionSource) -> u8 {
    match source {
        PromotionSource::ProxyElite => 0,
        PromotionSource::FirstImprovement => 1,
        PromotionSource::Escape => 2,
        PromotionSource::LocalOptimum => 3,
        PromotionSource::TrajectoryFallback => 4,
        PromotionSource::TopologyDiversity => 5,
        PromotionSource::RankedFallback => 6,
    }
}

fn update_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes.iter().copied() {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 15) as usize] as char);
    }
    output
}

fn shuffle<T>(values: &mut [T], rng: &mut SplitMix64) {
    for index in (1..values.len()).rev() {
        let swap_index = (rng.next_u64() as usize) % (index + 1);
        values.swap(index, swap_index);
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_catalog() -> Vec<ArchitectureFeatures> {
        let mut features = Vec::with_capacity(64);
        for mask in 0u8..64 {
            let mut topology = Vec::with_capacity(NB201_EDGES);
            let mut ones = 0usize;
            for edge in 0..NB201_EDGES {
                let operation = (mask >> edge) & 1;
                topology.push(operation);
                ones += operation as usize;
            }
            let score = -((ones as isize - 1).abs() as f64);
            let mut proxy_scores = BTreeMap::new();
            proxy_scores.insert("landscape".into(), score);
            let mut proxy_times_seconds = BTreeMap::new();
            proxy_times_seconds.insert("landscape".into(), 0.001);
            features.push(ArchitectureFeatures {
                architecture_id: format!("arch-{mask:02}"),
                proxy_scores,
                proxy_times_seconds,
                topology_encoding: topology,
            });
        }
        features
    }

    fn config(seed: u64) -> TrajectorySearchConfig {
        TrajectorySearchConfig {
            proxy_id: "landscape".into(),
            target_metric_id: "validation_accuracy".into(),
            proxy_direction: ProxyDirection::Maximize,
            seed,
            max_walks: 4,
            max_unique_visits: 64,
            max_moves_per_walk: 20,
            max_escapes_per_walk: 2,
        }
    }

    #[test]
    fn random_initial_pool_is_reproducible_and_uses_only_public_catalog() {
        let mut metadata_only = binary_catalog();
        for feature in &mut metadata_only {
            feature.proxy_scores.clear();
            feature.proxy_times_seconds.clear();
        }
        let first = random_candidate_pool(&metadata_only, 12, 77).unwrap();
        let second = random_candidate_pool(&metadata_only, 12, 77).unwrap();
        assert_eq!(first.candidate_ids, second.candidate_ids);
        assert_eq!(first.pool_sha256, second.pool_sha256);
        assert_eq!(first.strategy, InitialPoolStrategy::Random);
        assert_eq!(first.candidate_ids.len(), 12);
    }

    #[test]
    fn single_proxy_candidate_pool_is_direction_aware_and_rejects_target() {
        let candidates = binary_catalog();
        let highest = proxy_ranked_candidate_pool(
            &candidates,
            "landscape",
            "validation_accuracy",
            ProxyDirection::Maximize,
            3,
        )
        .unwrap();
        assert_eq!(highest.candidate_ids.len(), 3);
        assert_eq!(highest.strategy, InitialPoolStrategy::SingleProxyRanked);
        assert!(proxy_ranked_candidate_pool(
            &candidates,
            "validation_accuracy",
            "validation_accuracy",
            ProxyDirection::Maximize,
            3,
        ).is_err());
    }

    #[test]
    fn trajectory_promotion_plan_converts_to_common_initial_pool() {
        let features = binary_catalog();
        let archive = build_trajectory_archive(&features, &config(43)).unwrap();
        let mix = PromotionMix {
            promotion_count: 12,
            proxy_elite_quota: 3,
            trajectory_quota: 5,
            topology_diversity_quota: 4,
        };
        let plan = plan_promotions(&features, &archive, &mix).unwrap();
        let pool = trajectory_candidate_pool(&plan).unwrap();
        assert_eq!(pool.strategy, InitialPoolStrategy::TrajectoryPortfolio);
        assert_eq!(
            pool.candidate_ids,
            plan.candidates
                .iter()
                .map(|candidate| candidate.architecture_id.clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(pool.source_evidence_sha256, plan.plan_sha256);
        assert_eq!(pool.pool_sha256.len(), 64);
    }

    #[test]
    fn proxy_ranked_pool_rejects_target_metric_and_missing_runtime() {
        let mut candidates = binary_catalog();
        candidates[0].proxy_scores.insert("validation_accuracy".into(), 1.0);
        candidates[0].proxy_times_seconds.insert("validation_accuracy".into(), 0.0);
        assert!(proxy_ranked_candidate_pool(
            &candidates,
            "landscape",
            "validation_accuracy",
            ProxyDirection::Maximize,
            3,
        ).is_err());

        let mut candidates = binary_catalog();
        candidates[0].proxy_times_seconds.clear();
        assert!(proxy_ranked_candidate_pool(
            &candidates,
            "landscape",
            "validation_accuracy",
            ProxyDirection::Maximize,
            3,
        ).is_err());
    }

    #[test]
    fn archive_replay_is_deterministic_for_same_seed_and_input() {
        let features = binary_catalog();
        let first = build_trajectory_archive(&features, &config(17)).unwrap();
        let second = build_trajectory_archive(&features, &config(17)).unwrap();
        assert_eq!(first.archive_sha256, second.archive_sha256);
        assert_eq!(first.events, second.events);
        assert_eq!(first.visited_architecture_ids, second.visited_architecture_ids);
    }

    #[test]
    fn archive_records_first_improvements_and_local_optima() {
        let features = binary_catalog();
        let archive = build_trajectory_archive(&features, &config(17)).unwrap();
        assert!(!archive.first_improvement_architecture_ids.is_empty());
        assert!(!archive.local_optimum_architecture_ids.is_empty());
        for event in &archive.events {
            if event.kind == TrajectoryEventKind::FirstImprovement {
                let from_id = event.from_architecture_id.as_ref().unwrap();
                let from_score = features
                    .iter()
                    .find(|feature| &feature.architecture_id == from_id)
                    .unwrap()
                    .proxy_scores["landscape"];
                assert!(event.proxy_score > from_score);
            }
        }
    }

    #[test]
    fn every_escape_is_exactly_two_topology_edits_from_its_origin() {
        let features = binary_catalog();
        let archive = build_trajectory_archive(&features, &config(17)).unwrap();
        let by_id: HashMap<&str, &ArchitectureFeatures> = features
            .iter()
            .map(|feature| (feature.architecture_id.as_str(), feature))
            .collect();
        for event in archive.events.iter().filter(|event| event.kind == TrajectoryEventKind::TwoOptEscape) {
            let from = by_id[event.from_architecture_id.as_deref().unwrap()];
            let to = by_id[event.architecture_id.as_str()];
            let distance = from
                .topology_encoding
                .iter()
                .zip(to.topology_encoding.iter())
                .filter(|(left, right)| left != right)
                .count();
            assert_eq!(distance, 2);
        }
    }

    #[test]
    fn trajectory_archive_is_bounded_by_unique_visit_budget() {
        let features = binary_catalog();
        let mut bounded = config(17);
        bounded.max_unique_visits = 9;
        let archive = build_trajectory_archive(&features, &bounded).unwrap();
        assert!(archive.visited_architecture_ids.len() <= 9);
    }

    #[test]
    fn promotion_plan_has_distinct_sources_and_fixed_size() {
        let features = binary_catalog();
        let archive = build_trajectory_archive(&features, &config(29)).unwrap();
        let mix = PromotionMix {
            promotion_count: 16,
            proxy_elite_quota: 4,
            trajectory_quota: 6,
            topology_diversity_quota: 6,
        };
        let plan = plan_promotions(&features, &archive, &mix).unwrap();
        assert_eq!(plan.candidates.len(), 16);
        let unique: HashSet<&str> = plan
            .candidates
            .iter()
            .map(|candidate| candidate.architecture_id.as_str())
            .collect();
        assert_eq!(unique.len(), 16);
        assert_eq!(plan.proxy_elite_count, 4);
        assert_eq!(plan.trajectory_count, 6);
        assert_eq!(plan.diversity_count, 6);
        assert_eq!(plan.plan_sha256.len(), 64);
    }

    #[test]
    fn promotion_plan_is_reproducible_and_bound_to_archive() {
        let features = binary_catalog();
        let archive_a = build_trajectory_archive(&features, &config(17)).unwrap();
        let archive_b = build_trajectory_archive(&features, &config(17)).unwrap();
        let mix = PromotionMix {
            promotion_count: 12,
            proxy_elite_quota: 4,
            trajectory_quota: 4,
            topology_diversity_quota: 4,
        };
        let plan_a = plan_promotions(&features, &archive_a, &mix).unwrap();
        let plan_b = plan_promotions(&features, &archive_b, &mix).unwrap();
        assert_eq!(plan_a.plan_sha256, plan_b.plan_sha256);
        assert_eq!(plan_a.candidates, plan_b.candidates);
        assert_eq!(plan_a.archive_sha256, archive_a.archive_sha256);
    }

    #[test]
    fn partial_catalog_is_explicit_and_cannot_claim_full_space_optima() {
        let features = binary_catalog();
        let mut archive = build_trajectory_archive(&features, &config(17)).unwrap();
        assert!(!archive.catalog_is_complete);
        archive.catalog_is_complete = true;
        let mix = PromotionMix {
            promotion_count: 8,
            proxy_elite_quota: 2,
            trajectory_quota: 3,
            topology_diversity_quota: 3,
        };
        assert!(plan_promotions(&features, &archive, &mix).is_err());
    }

    #[test]
    fn planner_rejects_quota_overcommit_and_unknown_archive_candidate() {
        let features = binary_catalog();
        let mut archive = build_trajectory_archive(&features, &config(17)).unwrap();
        let overcommitted = PromotionMix {
            promotion_count: 8,
            proxy_elite_quota: 4,
            trajectory_quota: 3,
            topology_diversity_quota: 2,
        };
        assert!(plan_promotions(&features, &archive, &overcommitted).is_err());
        archive.escape_architecture_ids.push("unseen-architecture".into());
        let valid_mix = PromotionMix {
            promotion_count: 8,
            proxy_elite_quota: 2,
            trajectory_quota: 3,
            topology_diversity_quota: 3,
        };
        assert!(plan_promotions(&features, &archive, &valid_mix).is_err());
    }

    #[test]
    fn search_rejects_target_metric_smuggled_as_proxy() {
        let features = binary_catalog();
        let mut invalid = config(1);
        invalid.target_metric_id = invalid.proxy_id.clone();
        assert!(build_trajectory_archive(&features, &invalid).is_err());
    }

    #[test]
    fn search_rejects_invalid_topology_missing_proxy_and_nan() {
        let mut invalid = binary_catalog();
        invalid[0].topology_encoding.pop();
        assert!(build_trajectory_archive(&invalid, &config(1)).is_err());

        let mut missing = binary_catalog();
        missing[0].proxy_scores.clear();
        assert!(build_trajectory_archive(&missing, &config(1)).is_err());

        let mut non_finite = binary_catalog();
        non_finite[0].proxy_scores.insert("landscape".into(), f64::NAN);
        assert!(build_trajectory_archive(&non_finite, &config(1)).is_err());

        let mut missing_runtime = binary_catalog();
        missing_runtime[0].proxy_times_seconds.clear();
        assert!(build_trajectory_archive(&missing_runtime, &config(1)).is_err());
    }
}
