//! Cluster-bootstrap stability for cost-sensitive MIS Pareto frontiers.
//!
//! The caller supplies one already-aggregated point per graph-cluster and policy.
//! Repeated trials should be aggregated within each graph according to the frozen
//! analysis plan before they reach this API. This module resamples graphs, not rows,
//! and reports how often each policy belongs to the non-dominated frontier. The
//! resulting membership frequencies are stability diagnostics, not confidence levels.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::mis_cost_risk::MisCostMetric;
use crate::mis_pareto::{
    MisNormalizedCostValue, MisParetoError, MisParetoPoint, mis_cost_pareto_frontier,
};
use crate::rng::XorShift64;

/// One policy's already-aggregated outcome for one independent graph instance.
#[derive(Debug, Clone, PartialEq)]
pub struct MisParetoClusterObservation {
    /// Stable graph/instance cluster identity shared by policies for this input graph.
    pub cluster_id: String,
    /// Normalized and aggregated policy result for this graph.
    pub point: MisParetoPoint,
}

/// Configuration for graph-cluster bootstrap estimation of frontier stability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MisParetoStabilityConfig {
    /// Number of cluster bootstrap resamples; must be positive.
    pub replicates: usize,
    /// Seed for deterministic cluster resampling.
    pub seed: u64,
    /// Dominance tolerance passed unchanged to the existing Pareto frontier function.
    pub dominance_tolerance: f64,
}

/// Point estimate and bootstrap stability for one policy.
#[derive(Debug, Clone, PartialEq)]
pub struct MisParetoStabilityEntry {
    /// Stable policy identifier.
    pub policy_id: String,
    /// Evidence class retained from the point-estimate aggregation.
    pub evidence_class: crate::mis_cost_risk::MisEvidenceClass,
    /// Aggregate point from the original cohort, before resampling.
    pub point_estimate: MisParetoPoint,
    /// Whether the policy is on the frontier from the original point estimates.
    pub point_frontier_member: bool,
    /// Number of bootstrap resamples in which the policy belonged to the frontier.
    pub frontier_membership_count: usize,
    /// Membership count divided by bootstrap replicate count.
    ///
    /// This is a stability frequency, not a confidence probability that the policy is
    /// truly Pareto-optimal.
    pub frontier_membership_frequency: f64,
}

/// Result of cluster-bootstrap Pareto-frontier stability analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct MisParetoStabilityReport {
    /// Shared frozen cohort identity.
    pub cohort_id: String,
    /// Number of independent graph clusters in the input cohort.
    pub cluster_count: usize,
    /// Total number of cluster-policy observations supplied.
    pub observation_count: usize,
    /// Bootstrap replicates completed.
    pub replicates: usize,
    /// Seed used for resampling.
    pub seed: u64,
    /// Policy IDs on the original point-estimate frontier, sorted by policy ID.
    pub point_frontier_policy_ids: Vec<String>,
    /// Per-policy point estimate and frontier membership frequency, sorted by policy ID.
    pub entries: Vec<MisParetoStabilityEntry>,
}

/// Invalid cluster-stability input.
#[derive(Debug, Clone, PartialEq)]
pub enum MisParetoStabilityError {
    /// No cluster observations were supplied.
    EmptyObservations,
    /// The cohort ID or a graph-cluster ID is empty.
    EmptyCohortOrClusterId,
    /// Fewer than two independent graph clusters were supplied.
    InsufficientClusters { cluster_count: usize },
    /// At least one policy is repeated within a graph cluster.
    DuplicatePolicyInCluster { cluster_id: String, policy_id: String },
    /// Graph clusters do not all provide exactly the same policy set.
    MismatchedPolicySet { cluster_id: String },
    /// One policy silently mixes evidence classes across graph clusters.
    MixedEvidenceClasses { policy_id: String },
    /// A selected normalized cost dimension is missing, duplicated, or otherwise invalid.
    InvalidParetoPoint(MisParetoError),
    /// Monetary units differ across cluster records, even if each cluster is internally consistent.
    CrossClusterCurrencyMismatch,
    /// Bootstrap replicate count must be positive.
    ZeroReplicates,
}

impl fmt::Display for MisParetoStabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyObservations => write!(f, "at least one graph-policy observation is required"),
            Self::EmptyCohortOrClusterId => write!(f, "cohort and graph-cluster IDs must be nonempty"),
            Self::InsufficientClusters { cluster_count } => {
                write!(f, "Pareto stability needs at least two independent graph clusters, got {cluster_count}")
            }
            Self::DuplicatePolicyInCluster { cluster_id, policy_id } => {
                write!(f, "policy {policy_id} is duplicated within graph cluster {cluster_id}")
            }
            Self::MismatchedPolicySet { cluster_id } => {
                write!(f, "graph cluster {cluster_id} does not contain the exact shared policy set")
            }
            Self::MixedEvidenceClasses { policy_id } => {
                write!(f, "policy {policy_id} mixes evidence classes across graph clusters")
            }
            Self::InvalidParetoPoint(error) => write!(f, "invalid per-cluster Pareto point: {error}"),
            Self::CrossClusterCurrencyMismatch => {
                write!(f, "monetary cost uses different currencies across graph clusters")
            }
            Self::ZeroReplicates => write!(f, "Pareto stability bootstrap replicate count must be positive"),
        }
    }
}

impl std::error::Error for MisParetoStabilityError {}

/// Numerically stable arithmetic mean for finite nonnegative or signed measurements.
fn mean(values: &[f64]) -> f64 {
    let scale = values.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return 0.0;
    }
    let normalized = values.iter().map(|value| value / scale).sum::<f64>() / values.len() as f64;
    scale * normalized.clamp(-1.0, 1.0)
}

fn aggregate_resampled_clusters(
    cluster_indices: &[usize],
    clusters: &[BTreeMap<String, MisParetoPoint>],
    policy_ids: &[String],
    metrics: &[MisCostMetric],
    cohort_id: &str,
) -> Vec<MisParetoPoint> {
    policy_ids
        .iter()
        .map(|policy_id| {
            let selected: Vec<&MisParetoPoint> = cluster_indices
                .iter()
                .map(|&cluster_index| {
                    clusters[cluster_index]
                        .get(policy_id)
                        .expect("cluster policy sets are validated before aggregation")
                })
                .collect();
            let first = selected[0];
            let normalized_quality_loss = mean(
                &selected.iter().map(|point| point.normalized_quality_loss).collect::<Vec<_>>(),
            );
            let infeasible_rate = mean(
                &selected.iter().map(|point| point.infeasible_rate).collect::<Vec<_>>(),
            );
            let target_miss_rate = mean(
                &selected.iter().map(|point| point.target_miss_rate).collect::<Vec<_>>(),
            );
            let costs = metrics
                .iter()
                .map(|&metric| {
                    let values: Vec<&MisNormalizedCostValue> = selected
                        .iter()
                        .map(|point| {
                            point.costs.iter().find(|cost| cost.metric == metric)
                                .expect("selected metric sets are validated before aggregation")
                        })
                        .collect();
                    MisNormalizedCostValue {
                        metric,
                        normalized_mean_amount: mean(
                            &values.iter().map(|cost| cost.normalized_mean_amount).collect::<Vec<_>>(),
                        ),
                        currency_code: values[0].currency_code.clone(),
                    }
                })
                .collect();
            MisParetoPoint {
                cohort_id: cohort_id.to_owned(),
                policy_id: policy_id.clone(),
                evidence_class: first.evidence_class,
                normalized_quality_loss,
                infeasible_rate,
                target_miss_rate,
                costs,
            }
        })
        .collect()
}

/// Estimate how stable a cost-sensitive Pareto frontier is under graph-cluster resampling.
///
/// Input points must already aggregate repeated trials within each graph, with the same
/// declared aggregation rule for all policies. Each graph must include exactly one point
/// for every policy, and a policy's evidence class must remain homogeneous across graphs.
/// Metric scales must be predeclared and shared; monetary currency must agree across all
/// clusters. The original-cohort point estimates are aggregated with equal graph weight.
///
/// Each bootstrap replicate samples the original number of graph clusters with replacement
/// and recomputes policy aggregates and the Pareto frontier. Membership frequency is the
/// fraction of replicates whose frontier includes a policy. It is a diagnostic of frontier
/// stability; it is not the probability that a policy is truly Pareto-optimal. The method
/// cannot repair graph-family dependence; if graphs are nested within a few independent
/// families, use family-level clusters or a hierarchical model instead.
pub fn mis_cost_pareto_frontier_stability(
    observations: &[MisParetoClusterObservation],
    metrics: &[MisCostMetric],
    config: MisParetoStabilityConfig,
) -> Result<MisParetoStabilityReport, MisParetoStabilityError> {
    if observations.is_empty() {
        return Err(MisParetoStabilityError::EmptyObservations);
    }
    if config.replicates == 0 {
        return Err(MisParetoStabilityError::ZeroReplicates);
    }

    let cohort_id = observations[0].point.cohort_id.trim();
    if cohort_id.is_empty() || observations.iter().any(|observation| {
        observation.cluster_id.trim().is_empty() || observation.point.cohort_id.trim() != cohort_id
    }) {
        return Err(MisParetoStabilityError::EmptyCohortOrClusterId);
    }

    let mut clusters: BTreeMap<String, BTreeMap<String, MisParetoPoint>> = BTreeMap::new();
    let mut policy_evidence = BTreeMap::new();
    let mut currency_by_metric: HashMap<MisCostMetric, Option<String>> = HashMap::new();
    for observation in observations {
        let cluster = clusters.entry(observation.cluster_id.clone()).or_default();
        let policy_id = observation.point.policy_id.clone();
        if cluster.contains_key(&policy_id) {
            return Err(MisParetoStabilityError::DuplicatePolicyInCluster {
                cluster_id: observation.cluster_id.clone(),
                policy_id,
            });
        }
        match policy_evidence.get(&observation.point.policy_id) {
            Some(evidence) if *evidence != observation.point.evidence_class => {
                return Err(MisParetoStabilityError::MixedEvidenceClasses {
                    policy_id: observation.point.policy_id.clone(),
                });
            }
            None => {
                policy_evidence.insert(
                    observation.point.policy_id.clone(),
                    observation.point.evidence_class,
                );
            }
            _ => {}
        }
        for cost in &observation.point.costs {
            if cost.metric == MisCostMetric::MoneyMicrounits {
                match currency_by_metric.get(&cost.metric) {
                    Some(currency) if currency != &cost.currency_code => {
                        return Err(MisParetoStabilityError::CrossClusterCurrencyMismatch);
                    }
                    None => {
                        currency_by_metric.insert(cost.metric, cost.currency_code.clone());
                    }
                    _ => {}
                }
            }
        }
        cluster.insert(policy_id, observation.point.clone());
    }

    if clusters.len() < 2 {
        return Err(MisParetoStabilityError::InsufficientClusters {
            cluster_count: clusters.len(),
        });
    }

    let first_cluster = clusters.values().next().expect("nonempty cluster map");
    let policy_ids: Vec<String> = first_cluster.keys().cloned().collect();
    let expected_policy_set: BTreeSet<String> = policy_ids.iter().cloned().collect();
    for (cluster_id, cluster) in &clusters {
        let observed_policy_set: BTreeSet<String> = cluster.keys().cloned().collect();
        if observed_policy_set != expected_policy_set {
            return Err(MisParetoStabilityError::MismatchedPolicySet {
                cluster_id: cluster_id.clone(),
            });
        }
        let points: Vec<MisParetoPoint> = policy_ids
            .iter()
            .map(|policy_id| cluster.get(policy_id).expect("set equality was checked").clone())
            .collect();
        mis_cost_pareto_frontier(&points, metrics, config.dominance_tolerance)
            .map_err(MisParetoStabilityError::InvalidParetoPoint)?;
    }

    let ordered_clusters: Vec<BTreeMap<String, MisParetoPoint>> = clusters.into_values().collect();
    let all_cluster_indices: Vec<usize> = (0..ordered_clusters.len()).collect();
    let point_estimates = aggregate_resampled_clusters(
        &all_cluster_indices,
        &ordered_clusters,
        &policy_ids,
        metrics,
        cohort_id,
    );
    let point_frontier_indices = mis_cost_pareto_frontier(
        &point_estimates,
        metrics,
        config.dominance_tolerance,
    ).map_err(MisParetoStabilityError::InvalidParetoPoint)?;
    let point_frontier_set: BTreeSet<usize> = point_frontier_indices.iter().copied().collect();
    let mut membership_counts = vec![0usize; policy_ids.len()];
    let mut rng = XorShift64::new(config.seed);
    for _ in 0..config.replicates {
        let sampled_indices: Vec<usize> = (0..ordered_clusters.len())
            .map(|_| rng.next_usize(ordered_clusters.len()).unwrap_or(0))
            .collect();
        let bootstrap_points = aggregate_resampled_clusters(
            &sampled_indices,
            &ordered_clusters,
            &policy_ids,
            metrics,
            cohort_id,
        );
        let frontier = mis_cost_pareto_frontier(
            &bootstrap_points,
            metrics,
            config.dominance_tolerance,
        ).map_err(MisParetoStabilityError::InvalidParetoPoint)?;
        for index in frontier {
            membership_counts[index] += 1;
        }
    }

    let entries = point_estimates
        .into_iter()
        .enumerate()
        .map(|(index, point_estimate)| MisParetoStabilityEntry {
            policy_id: point_estimate.policy_id.clone(),
            evidence_class: point_estimate.evidence_class,
            point_estimate,
            point_frontier_member: point_frontier_set.contains(&index),
            frontier_membership_count: membership_counts[index],
            frontier_membership_frequency: membership_counts[index] as f64 / config.replicates as f64,
        })
        .collect();
    Ok(MisParetoStabilityReport {
        cohort_id: cohort_id.to_owned(),
        cluster_count: ordered_clusters.len(),
        observation_count: observations.len(),
        replicates: config.replicates,
        seed: config.seed,
        point_frontier_policy_ids: point_frontier_indices.iter()
            .map(|&index| policy_ids[index].clone())
            .collect(),
        entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mis_cost_risk::MisEvidenceClass;

    fn cost(metric: MisCostMetric, amount: f64) -> MisNormalizedCostValue {
        MisNormalizedCostValue {
            metric,
            normalized_mean_amount: amount,
            currency_code: None,
        }
    }

    fn point(
        policy_id: &str,
        loss: f64,
        infeasible: f64,
        miss: f64,
        cost_amount: f64,
    ) -> MisParetoPoint {
        MisParetoPoint {
            cohort_id: "cohort-001".to_owned(),
            policy_id: policy_id.to_owned(),
            evidence_class: MisEvidenceClass::ClassicalSolver,
            normalized_quality_loss: loss,
            infeasible_rate: infeasible,
            target_miss_rate: miss,
            costs: vec![cost(MisCostMetric::PhysicalShots, cost_amount)],
        }
    }

    fn observation(cluster_id: &str, point: MisParetoPoint) -> MisParetoClusterObservation {
        MisParetoClusterObservation {
            cluster_id: cluster_id.to_owned(),
            point,
        }
    }

    fn metrics() -> [MisCostMetric; 1] {
        [MisCostMetric::PhysicalShots]
    }

    #[test]
    fn frontier_membership_is_replayable_and_identifies_stably_dominated_policy() {
        let observations = vec![
            observation("g1", point("dominant", 0.1, 0.0, 0.0, 0.2)),
            observation("g1", point("dominated", 0.8, 0.0, 0.5, 0.9)),
            observation("g2", point("dominant", 0.2, 0.0, 0.0, 0.3)),
            observation("g2", point("dominated", 0.7, 0.0, 0.4, 0.8)),
            observation("g3", point("dominant", 0.15, 0.0, 0.0, 0.25)),
            observation("g3", point("dominated", 0.75, 0.0, 0.6, 0.85)),
        ];
        let config = MisParetoStabilityConfig {
            replicates: 250,
            seed: 77,
            dominance_tolerance: 0.0,
        };
        let first = mis_cost_pareto_frontier_stability(&observations, &metrics(), config).unwrap();
        let second = mis_cost_pareto_frontier_stability(&observations, &metrics(), config).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.cluster_count, 3);
        assert_eq!(first.observation_count, 6);
        assert_eq!(first.point_frontier_policy_ids, vec!["dominant".to_owned()]);
        let dominant = first.entries.iter().find(|entry| entry.policy_id == "dominant").unwrap();
        let dominated = first.entries.iter().find(|entry| entry.policy_id == "dominated").unwrap();
        assert_eq!(dominant.frontier_membership_frequency, 1.0);
        assert_eq!(dominated.frontier_membership_frequency, 0.0);
    }

    #[test]
    fn rejects_zero_replicates_mixed_evidence_and_cross_cluster_currency() {
        let observations = vec![
            observation("g1", point("a", 0.1, 0.0, 0.0, 0.2)),
            observation("g2", point("a", 0.2, 0.0, 0.0, 0.3)),
        ];
        assert_eq!(
            mis_cost_pareto_frontier_stability(
                &observations,
                &metrics(),
                MisParetoStabilityConfig { replicates: 0, seed: 1, dominance_tolerance: 0.0 },
            ),
            Err(MisParetoStabilityError::ZeroReplicates)
        );
        let mut mixed_evidence = observations.clone();
        mixed_evidence[1].point.evidence_class = MisEvidenceClass::IdealQuantumSimulator;
        assert_eq!(
            mis_cost_pareto_frontier_stability(&mixed_evidence, &metrics(), MisParetoStabilityConfig { replicates: 10, seed: 1, dominance_tolerance: 0.0 }),
            Err(MisParetoStabilityError::MixedEvidenceClasses { policy_id: "a".to_owned() })
        );

        let mut usd = point("a", 0.1, 0.0, 0.0, 0.2);
        usd.costs = vec![MisNormalizedCostValue {
            metric: MisCostMetric::MoneyMicrounits,
            normalized_mean_amount: 0.2,
            currency_code: Some("USD".to_owned()),
        }];
        let mut eur = point("a", 0.2, 0.0, 0.0, 0.3);
        eur.costs = vec![MisNormalizedCostValue {
            metric: MisCostMetric::MoneyMicrounits,
            normalized_mean_amount: 0.3,
            currency_code: Some("EUR".to_owned()),
        }];
        assert_eq!(
            mis_cost_pareto_frontier_stability(
                &[observation("g1", usd), observation("g2", eur)],
                &[MisCostMetric::MoneyMicrounits],
                MisParetoStabilityConfig { replicates: 10, seed: 1, dominance_tolerance: 0.0 },
            ),
            Err(MisParetoStabilityError::CrossClusterCurrencyMismatch)
        );
    }

    #[test]
    fn frontier_membership_frequency_exposes_graph_composition_sensitivity() {
        // On g1, policy a dominates b; on g2, policy b dominates a. When the
        // two graphs are equally weighted, their point estimates tie and both
        // appear on the point frontier. A cluster bootstrap should show that
        // either policy falls off the frontier when one graph is resampled twice.
        let observations = vec![
            observation("g1", point("a", 0.1, 0.0, 0.0, 0.1)),
            observation("g1", point("b", 0.9, 0.0, 0.0, 0.9)),
            observation("g2", point("a", 0.9, 0.0, 0.0, 0.9)),
            observation("g2", point("b", 0.1, 0.0, 0.0, 0.1)),
        ];
        let config = MisParetoStabilityConfig {
            replicates: 1_000,
            seed: 2026,
            dominance_tolerance: 0.0,
        };
        let first = mis_cost_pareto_frontier_stability(&observations, &metrics(), config).unwrap();
        let replay = mis_cost_pareto_frontier_stability(&observations, &metrics(), config).unwrap();
        assert_eq!(first, replay);
        assert_eq!(first.point_frontier_policy_ids, vec!["a".to_owned(), "b".to_owned()]);
        for entry in &first.entries {
            assert!(entry.frontier_membership_frequency > 0.60);
            assert!(entry.frontier_membership_frequency < 0.90);
        }
    }

    #[test]
    fn rejects_missing_policy_duplicate_policy_and_single_cluster() {
        let cfg = MisParetoStabilityConfig { replicates: 10, seed: 9, dominance_tolerance: 0.0 };
        assert_eq!(
            mis_cost_pareto_frontier_stability(&[], &metrics(), cfg),
            Err(MisParetoStabilityError::EmptyObservations)
        );
        assert_eq!(
            mis_cost_pareto_frontier_stability(&[observation("g1", point("a", 0.1, 0.0, 0.0, 0.2))], &metrics(), cfg),
            Err(MisParetoStabilityError::InsufficientClusters { cluster_count: 1 })
        );
        let duplicate = vec![
            observation("g1", point("a", 0.1, 0.0, 0.0, 0.2)),
            observation("g1", point("a", 0.2, 0.0, 0.0, 0.3)),
            observation("g2", point("a", 0.2, 0.0, 0.0, 0.2)),
        ];
        assert_eq!(
            mis_cost_pareto_frontier_stability(&duplicate, &metrics(), cfg),
            Err(MisParetoStabilityError::DuplicatePolicyInCluster { cluster_id: "g1".to_owned(), policy_id: "a".to_owned() })
        );
        let mismatch = vec![
            observation("g1", point("a", 0.1, 0.0, 0.0, 0.2)),
            observation("g1", point("b", 0.2, 0.0, 0.0, 0.3)),
            observation("g2", point("a", 0.2, 0.0, 0.0, 0.2)),
        ];
        assert_eq!(
            mis_cost_pareto_frontier_stability(&mismatch, &metrics(), cfg),
            Err(MisParetoStabilityError::MismatchedPolicySet { cluster_id: "g2".to_owned() })
        );
    }
}
