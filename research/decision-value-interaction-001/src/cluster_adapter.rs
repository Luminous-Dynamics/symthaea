//! Hierarchical adapter from paired per-world regret certificates to independent-cluster deltas.
//!
//! The downstream VIS-DVPC-014 analyzer owns bootstrap intervals, sign-flip
//! diagnostics and family-wise adjustment. This module only creates one
//! equally weighted delta per independent cluster and predeclared contrast.
//!
//! Aggregation order is deliberate:
//! 1. average hidden-world counterfactuals within each episode;
//! 2. average episode means within each preregistered independent cluster;
//! 3. emit one row per cluster and contrast.
//!
//! This prevents episodes with more enumerated worlds from dominating a
//! cluster, and prevents correlated frames/rows from being treated as
//! independent statistical replications.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::{DecisionValueInteractionCertificate, PairedRunIdentity};

const MIN_INDEPENDENT_CLUSTERS: usize = 5;
const COMPARISON_FAMILY_SIZE: usize = 4;
const MIN_BOOTSTRAP_REPLICATES: u64 = 1_000;
const MAX_BOOTSTRAP_REPLICATES: u64 = 1_000_000;
const ZERO_DIGEST: [u8; 32] = [0; 32];

/// An evaluator-produced per-episode/per-hidden-world certificate, assigned
/// to an independently sampled cluster by a preregistered evaluator manifest.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusteredInteractionOutcome {
    pub cluster_id: String,
    pub certificate: DecisionValueInteractionCertificate,
}

/// Frozen metadata consumed by the VIS-DVPC-014 cluster inference analyzer.
///
/// Both digests must come from independently frozen artifacts. The bundle
/// lock covers the evaluator/data evidence; the cluster-assignment digest
/// commits to the canonical episode-to-cluster manifest. This adapter checks
/// the supplied assignments against outcomes, while the downstream CLI
/// independently checks the manifest digest supplied by the custodian.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterInferenceMetadata {
    pub experiment_id: String,
    pub dataset_id: String,
    pub dataset_revision: String,
    pub split: String,
    pub cluster_unit: String,
    pub primary_metric: String,
    pub bundle_lock_sha256: String,
    pub cluster_assignment_manifest_sha256: String,
    pub bootstrap_seed: u64,
    pub bootstrap_replicates: u64,
    pub familywise_confidence: f64,
}

/// One frozen episode-to-cluster assignment from the independently
/// preregistered cluster manifest. The initial-state digest prevents a
/// manifest row from being rebound to a different episode state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusterAssignment {
    pub episode_id: u64,
    pub cluster_id: String,
    pub initial_state_sha256: [u8; 32],
}

/// Canonical TSV ready for the VIS-DVPC-014 standalone cluster analyzer.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterDeltaExport {
    pub tsv: String,
    pub independent_cluster_count: usize,
    pub episode_count: usize,
    pub hidden_world_observation_count: usize,
    pub comparison_count: usize,
}

/// Validation or aggregation failure. No partial TSV is returned on error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClusterAdapterError(pub String);

impl fmt::Display for ClusterAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ClusterAdapterError {}

fn fail(message: impl Into<String>) -> ClusterAdapterError {
    ClusterAdapterError(message.into())
}

fn valid_token(value: &str, field: &str) -> Result<(), ClusterAdapterError> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(fail(format!(
            "{field} must be a non-empty token without whitespace or paths"
        )));
    }
    Ok(())
}

fn valid_digest(value: &str, field: &str) -> Result<String, ClusterAdapterError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(fail(format!(
            "{field} must be a 64-character hexadecimal SHA-256 digest"
        )));
    }
    Ok(value.to_ascii_lowercase())
}

fn digest_is_empty(digest: &[u8; 32]) -> bool {
    *digest == ZERO_DIGEST
}

fn validate_metadata(
    metadata: &ClusterInferenceMetadata,
) -> Result<(String, String), ClusterAdapterError> {
    for (field, value) in [
        ("experiment_id", metadata.experiment_id.as_str()),
        ("dataset_id", metadata.dataset_id.as_str()),
        ("dataset_revision", metadata.dataset_revision.as_str()),
        ("cluster_unit", metadata.cluster_unit.as_str()),
        ("primary_metric", metadata.primary_metric.as_str()),
    ] {
        valid_token(value, field)?;
    }
    if !matches!(metadata.split.as_str(), "test" | "private_test") {
        return Err(fail(
            "confirmatory cluster inference accepts only test or private_test split",
        ));
    }
    if matches!(
        metadata.cluster_unit.as_str(),
        "frame" | "image" | "crop" | "row" | "sample" | "timestamp" | "detection"
    ) {
        return Err(fail(
            "cluster_unit is too granular and would create pseudoreplication",
        ));
    }
    let bundle_lock = valid_digest(&metadata.bundle_lock_sha256, "bundle_lock_sha256")?;
    let cluster_assignment_manifest = valid_digest(
        &metadata.cluster_assignment_manifest_sha256,
        "cluster_assignment_manifest_sha256",
    )?;
    if !(MIN_BOOTSTRAP_REPLICATES..=MAX_BOOTSTRAP_REPLICATES)
        .contains(&metadata.bootstrap_replicates)
    {
        return Err(fail(format!(
            "bootstrap_replicates must be in {MIN_BOOTSTRAP_REPLICATES}..={MAX_BOOTSTRAP_REPLICATES}"
        )));
    }
    if !metadata.familywise_confidence.is_finite()
        || !(0.5..1.0).contains(&metadata.familywise_confidence)
    {
        return Err(fail("familywise_confidence must be finite and in [0.5, 1)"));
    }
    let per_tail = (1.0 - metadata.familywise_confidence) / (2.0 * COMPARISON_FAMILY_SIZE as f64);
    if metadata.bootstrap_replicates as f64 * per_tail < 20.0 {
        return Err(fail(
            "bootstrap_replicates provides fewer than 20 draws in a Bonferroni-adjusted tail",
        ));
    }
    Ok((bundle_lock, cluster_assignment_manifest))
}

fn validate_identity(identity: &PairedRunIdentity) -> Result<(), ClusterAdapterError> {
    let digests = [
        ("hidden_world_sha256", &identity.hidden_world_sha256),
        ("initial_state_sha256", &identity.initial_state_sha256),
        ("task_sha256", &identity.task_sha256),
        ("loss_scale_sha256", &identity.loss_scale_sha256),
        (
            "evaluator_profile_sha256",
            &identity.evaluator_profile_sha256,
        ),
        (
            "execution_environment_sha256",
            &identity.execution_environment_sha256,
        ),
        ("action_domain_sha256", &identity.action_domain_sha256),
        ("resource_budget_sha256", &identity.resource_budget_sha256),
        (
            "model_configuration_sha256",
            &identity.model_configuration_sha256,
        ),
        ("candidate_policy_sha256", &identity.candidate_policy_sha256),
    ];
    for (field, digest) in digests {
        if digest_is_empty(digest) {
            return Err(fail(format!("{field} must not be zero-filled")));
        }
    }
    Ok(())
}

/// Fields that must remain fixed for the whole confirmatory experiment.
/// Episode, initial-state, hidden-world and policy-seed identity are allowed to
/// vary only under the explicit hierarchy validated below.
fn same_experiment_context(a: &PairedRunIdentity, b: &PairedRunIdentity) -> bool {
    a.task_sha256 == b.task_sha256
        && a.loss_scale_sha256 == b.loss_scale_sha256
        && a.evaluator_profile_sha256 == b.evaluator_profile_sha256
        && a.execution_environment_sha256 == b.execution_environment_sha256
        && a.action_domain_sha256 == b.action_domain_sha256
        && a.resource_budget_sha256 == b.resource_budget_sha256
        && a.model_configuration_sha256 == b.model_configuration_sha256
        && a.candidate_policy_sha256 == b.candidate_policy_sha256
}

/// Recompute all four effects from the four regrets; do not trust a serialized
/// or candidate-supplied effect value.
fn validate_certificate(
    certificate: &DecisionValueInteractionCertificate,
) -> Result<[i64; 4], ClusterAdapterError> {
    let identity = &certificate.pairing_identity;
    validate_identity(identity)?;
    if certificate.episode_id != identity.episode_id
        || certificate.hidden_world_id != identity.hidden_world_id
        || certificate.hidden_world_sha256 != identity.hidden_world_sha256
    {
        return Err(fail(
            "certificate episode/hidden-world identity disagrees with paired identity",
        ));
    }
    if certificate
        .realized_regret_ppm
        .iter()
        .any(|regret| *regret > super::DECISION_REGRET_PPM)
    {
        return Err(fail("certificate regret is outside the frozen PPM scale"));
    }
    let [r00, r10, r01, r11] = certificate.realized_regret_ppm;
    let expected = [
        i64::from(r00) - i64::from(r10),
        i64::from(r00) - i64::from(r01),
        i64::from(r00) - i64::from(r11),
        i64::from(r10) + i64::from(r01) - i64::from(r00) - i64::from(r11),
    ];
    let actual = [
        certificate.compute_benefit_ppm,
        certificate.observation_benefit_ppm,
        certificate.joint_benefit_ppm,
        certificate.interaction_ppm,
    ];
    if expected != actual {
        return Err(fail(
            "certificate signed effects do not recompute from the four arm regrets",
        ));
    }
    let expected_sign = if expected[3] > 0 {
        super::InteractionSign::Complementary
    } else if expected[3] < 0 {
        super::InteractionSign::Antagonistic
    } else {
        super::InteractionSign::AdditiveOrNeutral
    };
    if certificate.interaction_sign != expected_sign {
        return Err(fail(
            "certificate interaction sign disagrees with recomputed interaction",
        ));
    }
    Ok(expected)
}

#[derive(Clone, Debug)]
struct EpisodeAccumulator {
    cluster_id: String,
    initial_state_sha256: [u8; 32],
    policy_seed: u64,
    hidden_worlds: BTreeMap<u64, [u8; 32]>,
    metric_sums: [i128; 4],
    world_count: usize,
}

impl EpisodeAccumulator {
    fn new(cluster_id: String, identity: &PairedRunIdentity) -> Self {
        Self {
            cluster_id,
            initial_state_sha256: identity.initial_state_sha256,
            policy_seed: identity.policy_seed,
            hidden_worlds: BTreeMap::new(),
            metric_sums: [0; 4],
            world_count: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ClusterMean {
    episode_count: usize,
    hidden_world_observation_count: usize,
    metric_means: [f64; 4],
}

fn fmt_ppm(value: f64) -> String {
    if value == 0.0 {
        "0.000000".to_owned()
    } else {
        format!("{value:.6}")
    }
}

/// Convert evaluator-side certificates into cluster-level deltas for
/// VIS-DVPC-014. The adapter performs no confidence interval or p-value
/// calculation; that remains exclusively owned by issue #8578.
///
/// It emits four preregistered comparisons (compute, observation, joint and
/// compute×observation interaction). Each episode is averaged equally within
/// its cluster, after averaging its balanced hidden-world cohort. The
/// downstream analyzer then weights independent clusters equally.
pub fn export_cluster_deltas(
    outcomes: &[ClusteredInteractionOutcome],
    assignments: &[ClusterAssignment],
    metadata: &ClusterInferenceMetadata,
) -> Result<ClusterDeltaExport, ClusterAdapterError> {
    let (bundle_lock, cluster_assignment_manifest) = validate_metadata(metadata)?;
    if outcomes.is_empty() {
        return Err(fail(
            "at least one paired interaction certificate is required",
        ));
    }

    let mut assignment_by_episode: BTreeMap<u64, &ClusterAssignment> = BTreeMap::new();
    for assignment in assignments {
        valid_token(&assignment.cluster_id, "cluster_id")?;
        if digest_is_empty(&assignment.initial_state_sha256) {
            return Err(fail("cluster assignment initial-state digest must not be zero-filled"));
        }
        if assignment_by_episode.insert(assignment.episode_id, assignment).is_some() {
            return Err(fail("duplicate episode_id in frozen cluster-assignment manifest"));
        }
    }
    if assignment_by_episode.is_empty() {
        return Err(fail("frozen cluster-assignment manifest is empty"));
    }

    let mut used_assignments = BTreeSet::new();
    let mut experiment_context: Option<PairedRunIdentity> = None;
    let mut episodes: BTreeMap<u64, EpisodeAccumulator> = BTreeMap::new();
    let mut episode_clusters: BTreeMap<u64, String> = BTreeMap::new();
    let mut seen_pairs: BTreeSet<(String, u64, u64)> = BTreeSet::new();

    for row in outcomes {
        valid_token(&row.cluster_id, "cluster_id")?;
        let certificate = &row.certificate;
        let identity = &certificate.pairing_identity;
        let effects = validate_certificate(certificate)?;

        let assignment = assignment_by_episode
            .get(&certificate.episode_id)
            .ok_or_else(|| fail("episode is absent from frozen cluster-assignment manifest"))?;
        if assignment.cluster_id != row.cluster_id {
            return Err(fail("outcome cluster_id disagrees with frozen cluster-assignment manifest"));
        }
        if assignment.initial_state_sha256 != identity.initial_state_sha256 {
            return Err(fail("episode initial-state digest disagrees with frozen cluster-assignment manifest"));
        }
        used_assignments.insert(certificate.episode_id);

        if let Some(reference) = &experiment_context {
            if !same_experiment_context(reference, identity) {
                return Err(fail(
                    "task/loss/evaluator/environment/action/resource/model/policy context changed within the experiment",
                ));
            }
        } else {
            experiment_context = Some(identity.clone());
        }

        if let Some(previous_cluster) = episode_clusters.get(&certificate.episode_id) {
            if previous_cluster != &row.cluster_id {
                return Err(fail(
                    "one episode_id is assigned to more than one independent cluster",
                ));
            }
        } else {
            episode_clusters.insert(certificate.episode_id, row.cluster_id.clone());
        }

        let pair_key = (
            row.cluster_id.clone(),
            certificate.episode_id,
            certificate.hidden_world_id,
        );
        if !seen_pairs.insert(pair_key) {
            return Err(fail(
                "duplicate (cluster, episode, hidden-world) certificate",
            ));
        }

        let episode = episodes
            .entry(certificate.episode_id)
            .or_insert_with(|| EpisodeAccumulator::new(row.cluster_id.clone(), identity));
        if episode.cluster_id != row.cluster_id {
            return Err(fail("episode cluster assignment changed"));
        }
        if episode.initial_state_sha256 != identity.initial_state_sha256
            || episode.policy_seed != identity.policy_seed
        {
            return Err(fail(
                "initial-state or policy-seed identity changed within an episode",
            ));
        }
        if episode
            .hidden_worlds
            .insert(certificate.hidden_world_id, certificate.hidden_world_sha256)
            .is_some()
        {
            return Err(fail("duplicate hidden-world ID within an episode"));
        }
        for (sum, effect) in episode.metric_sums.iter_mut().zip(effects) {
            *sum = sum
                .checked_add(i128::from(effect))
                .ok_or_else(|| fail("cluster adapter integer accumulation overflowed"))?;
        }
        episode.world_count += 1;
    }

    if used_assignments.len() != assignment_by_episode.len() {
        return Err(fail(
            "frozen cluster-assignment manifest contains episodes with no evaluator outcomes",
        ));
    }

    let mut cluster_episodes: BTreeMap<String, Vec<EpisodeAccumulator>> = BTreeMap::new();
    for (_, episode) in episodes {
        cluster_episodes
            .entry(episode.cluster_id.clone())
            .or_default()
            .push(episode);
    }
    if cluster_episodes.len() < MIN_INDEPENDENT_CLUSTERS {
        return Err(fail(format!(
            "at least {MIN_INDEPENDENT_CLUSTERS} independent clusters are required"
        )));
    }

    let mut cluster_means: BTreeMap<String, ClusterMean> = BTreeMap::new();
    let mut total_episodes = 0_usize;
    let mut total_world_observations = 0_usize;

    for (cluster_id, mut episode_rows) in cluster_episodes {
        episode_rows.sort_by_key(|episode| episode.initial_state_sha256);
        let reference_cohort = episode_rows
            .first()
            .map(|episode| &episode.hidden_worlds)
            .ok_or_else(|| fail("cluster contains no episodes"))?;
        if reference_cohort.is_empty() {
            return Err(fail("episode contains no hidden-world outcomes"));
        }

        let mut cluster_sums = [0.0_f64; 4];
        let mut cluster_world_count = 0_usize;
        for episode in &episode_rows {
            if &episode.hidden_worlds != reference_cohort {
                return Err(fail(
                    "hidden-world cohort or digest changed across episodes within a cluster",
                ));
            }
            if episode.world_count != reference_cohort.len() {
                return Err(fail(
                    "episode world count disagrees with its frozen cluster cohort",
                ));
            }
            for (sum, effect_sum) in cluster_sums.iter_mut().zip(episode.metric_sums) {
                *sum += effect_sum as f64 / episode.world_count as f64;
            }
            cluster_world_count += episode.world_count;
        }
        let episode_count = episode_rows.len();
        for value in &mut cluster_sums {
            *value /= episode_count as f64;
            if !value.is_finite() {
                return Err(fail("cluster mean is non-finite"));
            }
        }
        total_episodes += episode_count;
        total_world_observations += cluster_world_count;
        cluster_means.insert(
            cluster_id,
            ClusterMean {
                episode_count,
                hidden_world_observation_count: cluster_world_count,
                metric_means: cluster_sums,
            },
        );
    }

    const COMPARISONS: [(&str, &str, &str, usize); 4] = [
        ("compute_benefit", "compute_only", "baseline", 0),
        ("observation_benefit", "observation_only", "baseline", 1),
        ("joint_benefit", "joint", "baseline", 2),
        (
            "compute_observation_interaction",
            "joint_over_additive",
            "additive_expectation",
            3,
        ),
    ];

    let mut tsv = String::new();
    use fmt::Write as _;
    writeln!(tsv, "META\tformat\tVIS-DVPC-CLUSTER-DELTAS-001-v1").unwrap();
    writeln!(tsv, "META\texperiment_id\t{}", metadata.experiment_id).unwrap();
    writeln!(tsv, "META\tdataset_id\t{}", metadata.dataset_id).unwrap();
    writeln!(tsv, "META\tdataset_revision\t{}", metadata.dataset_revision).unwrap();
    writeln!(tsv, "META\tsplit\t{}", metadata.split).unwrap();
    writeln!(tsv, "META\tcluster_unit\t{}", metadata.cluster_unit).unwrap();
    writeln!(tsv, "META\tprimary_metric\t{}", metadata.primary_metric).unwrap();
    writeln!(tsv, "META\tpositive_is_better\ttrue").unwrap();
    writeln!(tsv, "META\tbundle_lock_sha256\t{bundle_lock}").unwrap();
    writeln!(
        tsv,
        "META\tcluster_assignment_manifest_sha256\t{cluster_assignment_manifest}"
    )
    .unwrap();
    writeln!(
        tsv,
        "META\tcomparison_family_size\t{COMPARISON_FAMILY_SIZE}"
    )
    .unwrap();
    writeln!(tsv, "META\tbootstrap_seed\t{}", metadata.bootstrap_seed).unwrap();
    writeln!(
        tsv,
        "META\tbootstrap_replicates\t{}",
        metadata.bootstrap_replicates
    )
    .unwrap();
    writeln!(
        tsv,
        "META\tfamilywise_confidence\t{:.10}",
        metadata.familywise_confidence
    )
    .unwrap();
    writeln!(
        tsv,
        "DELTA_HEADER\tcomparison_id\tcandidate_id\tbaseline_id\tcluster_id\tdelta"
    )
    .unwrap();

    for (comparison_id, candidate_id, baseline_id, metric_index) in COMPARISONS {
        for (cluster_id, means) in &cluster_means {
            writeln!(
                tsv,
                "DELTA\t{comparison_id}\t{candidate_id}\t{baseline_id}\t{cluster_id}\t{}",
                fmt_ppm(means.metric_means[metric_index])
            )
            .unwrap();
        }
    }

    Ok(ClusterDeltaExport {
        tsv,
        independent_cluster_count: cluster_means.len(),
        episode_count: total_episodes,
        hidden_world_observation_count: total_world_observations,
        comparison_count: COMPARISON_FAMILY_SIZE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ApplicationDisposition, ArmOutcome, PairedRunIdentity, TreatmentArm,
        evaluate_decision_value_interaction,
    };

    fn digest(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn identity(episode_id: u64, world_id: u64) -> PairedRunIdentity {
        PairedRunIdentity {
            episode_id,
            hidden_world_id: world_id,
            hidden_world_sha256: digest(world_id as u8),
            initial_state_sha256: digest(episode_id as u8),
            task_sha256: digest(3),
            loss_scale_sha256: digest(4),
            evaluator_profile_sha256: digest(5),
            execution_environment_sha256: digest(8),
            action_domain_sha256: digest(9),
            resource_budget_sha256: digest(10),
            model_configuration_sha256: digest(6),
            candidate_policy_sha256: digest(7),
            policy_seed: episode_id + 99,
        }
    }

    fn certificate(
        episode_id: u64,
        world_id: u64,
        regrets: [u32; 4],
    ) -> DecisionValueInteractionCertificate {
        let arms = [
            TreatmentArm::Baseline,
            TreatmentArm::ComputeOnly,
            TreatmentArm::ObservationOnly,
            TreatmentArm::Joint,
        ];
        let id = identity(episode_id, world_id);
        let outcomes = arms
            .into_iter()
            .enumerate()
            .map(|(index, arm)| ArmOutcome {
                identity: id.clone(),
                arm,
                realized_regret_ppm: regrets[index],
                disposition: ApplicationDisposition::Applied {
                    at_us: 100 + index as u64 * 10,
                },
            })
            .collect::<Vec<_>>();
        evaluate_decision_value_interaction(&outcomes).expect("valid four-arm certificate")
    }

    fn fixture() -> Vec<ClusteredInteractionOutcome> {
        let mut rows = Vec::new();
        for cluster in 0..6_u64 {
            for episode_offset in 0..2_u64 {
                let episode_id = cluster * 10 + episode_offset + 1;
                rows.push(ClusteredInteractionOutcome {
                    cluster_id: format!("scene-{cluster:02}"),
                    certificate: certificate(episode_id, 1, [400_000, 350_000, 380_000, 100_000]),
                });
                rows.push(ClusteredInteractionOutcome {
                    cluster_id: format!("scene-{cluster:02}"),
                    certificate: certificate(episode_id, 2, [400_000, 300_000, 250_000, 150_000]),
                });
            }
        }
        rows
    }

    fn assignments() -> Vec<ClusterAssignment> {
        (0..6_u64)
            .flat_map(|cluster| {
                (0..2_u64).map(move |episode_offset| {
                    let episode_id = cluster * 10 + episode_offset + 1;
                    ClusterAssignment {
                        episode_id,
                        cluster_id: format!("scene-{cluster:02}"),
                        initial_state_sha256: digest(episode_id as u8),
                    }
                })
            })
            .collect()
    }

    fn metadata() -> ClusterInferenceMetadata {
        ClusterInferenceMetadata {
            experiment_id: "compute-observation-v1".to_owned(),
            dataset_id: "robo-dyna".to_owned(),
            dataset_revision: "frozen-commit-a1b2c3d".to_owned(),
            split: "test".to_owned(),
            cluster_unit: "scene".to_owned(),
            primary_metric: "signed_decision_regret_effect_ppm".to_owned(),
            bundle_lock_sha256: "a".repeat(64),
            cluster_assignment_manifest_sha256:
                "8257074ad9d509a144c807fe1eeca8c0f29fb9c5101589a3e220d207236d0aab".to_owned(),
            bootstrap_seed: 42,
            bootstrap_replicates: 10_000,
            familywise_confidence: 0.95,
        }
    }

    #[test]
    fn exports_one_row_per_independent_cluster_and_contrast() {
        let result = export_cluster_deltas(&fixture(), &assignments(), &metadata()).expect("valid cluster export");
        assert_eq!(result.independent_cluster_count, 6);
        assert_eq!(result.episode_count, 12);
        assert_eq!(result.hidden_world_observation_count, 24);
        assert_eq!(result.comparison_count, 4);
        assert_eq!(
            result
                .tsv
                .lines()
                .filter(|line| line.starts_with("DELTA\t"))
                .count(),
            24
        );
        assert!(result.tsv.contains("META\tcomparison_family_size\t4"));
        assert!(result.tsv.contains("DELTA\tcompute_observation_interaction\tjoint_over_additive\tadditive_expectation\tscene-00\t115000.000000"));
    }

    #[test]
    fn export_is_invariant_to_input_row_order() {
        let rows = fixture();
        let mut reversed = rows.clone();
        reversed.reverse();
        assert_eq!(
            export_cluster_deltas(&rows, &assignments(), &metadata()).unwrap(),
            export_cluster_deltas(&reversed, &assignments(), &metadata()).unwrap()
        );
    }

    #[test]
    fn duplicate_cluster_episode_world_pair_is_rejected() {
        let mut rows = fixture();
        rows.push(rows[0].clone());
        assert!(
            export_cluster_deltas(&rows, &assignments(), &metadata())
                .unwrap_err()
                .to_string()
                .contains("duplicate (cluster, episode, hidden-world)")
        );
    }

    #[test]
    fn episode_cannot_be_assigned_to_two_clusters() {
        let mut rows = fixture();
        let mut duplicate = rows[0].clone();
        duplicate.cluster_id = "different-scene".to_owned();
        rows.push(duplicate);
        assert!(
            export_cluster_deltas(&rows, &assignments(), &metadata())
                .unwrap_err()
                .to_string()
                .contains("more than one independent cluster")
        );
    }

    #[test]
    fn episode_cluster_reassignment_is_rejected_against_frozen_manifest() {
        let mut rows = fixture();
        rows[0].cluster_id = "scene-01".to_owned();
        assert!(
            export_cluster_deltas(&rows, &assignments(), &metadata())
                .unwrap_err()
                .to_string()
                .contains("disagrees with frozen cluster-assignment manifest")
        );
    }

    #[test]
    fn initial_state_rebinding_is_rejected_against_frozen_manifest() {
        let mut rows = fixture();
        rows[0].certificate.pairing_identity.initial_state_sha256 = digest(200);
        assert!(
            export_cluster_deltas(&rows, &assignments(), &metadata())
                .unwrap_err()
                .to_string()
                .contains("initial-state digest disagrees")
        );
    }

    #[test]
    fn missing_and_extra_assignment_manifest_entries_are_rejected() {
        let mut missing = assignments();
        missing.pop();
        let missing_error = export_cluster_deltas(&fixture(), &missing, &metadata())
            .unwrap_err()
            .to_string();
        assert!(
            missing_error.contains("absent from frozen cluster-assignment manifest")
                || missing_error.contains("manifest contains episodes with no evaluator outcomes")
        );

        let mut extra = assignments();
        extra.push(ClusterAssignment {
            episode_id: 999,
            cluster_id: "scene-99".to_owned(),
            initial_state_sha256: digest(99),
        });
        assert!(
            export_cluster_deltas(&fixture(), &extra, &metadata())
                .unwrap_err()
                .to_string()
                .contains("manifest contains episodes with no evaluator outcomes")
        );
    }

    #[test]
    fn duplicate_episode_assignment_is_rejected() {
        let mut frozen = assignments();
        frozen.push(frozen[0].clone());
        assert!(
            export_cluster_deltas(&fixture(), &frozen, &metadata())
                .unwrap_err()
                .to_string()
                .contains("duplicate episode_id")
        );
    }

    #[test]
    fn hidden_world_cohort_must_be_identical_across_episodes_in_cluster() {
        let mut rows = fixture();
        rows.retain(|row| row.certificate.episode_id != 2 || row.certificate.hidden_world_id != 2);
        assert!(
            export_cluster_deltas(&rows, &assignments(), &metadata())
                .unwrap_err()
                .to_string()
                .contains("hidden-world cohort or digest changed")
        );
    }

    #[test]
    fn execution_resource_and_task_context_must_remain_fixed() {
        let mut rows = fixture();
        rows[0].certificate.pairing_identity.resource_budget_sha256 = digest(99);
        assert!(
            export_cluster_deltas(&rows, &assignments(), &metadata())
                .unwrap_err()
                .to_string()
                .contains("context changed")
        );
    }

    #[test]
    fn too_few_clusters_fail_closed() {
        let rows: Vec<_> = fixture()
            .into_iter()
            .filter(|row| row.cluster_id.as_str() <= "scene-02")
            .collect();
        assert!(
            export_cluster_deltas(&rows, &assignments(), &metadata())
                .unwrap_err()
                .to_string()
                .contains("at least 5 independent clusters")
        );
    }

    #[test]
    fn pseudoreplicated_cluster_unit_is_rejected() {
        let mut config = metadata();
        config.cluster_unit = "frame".to_owned();
        assert!(
            export_cluster_deltas(&fixture(), &assignments(), &config)
                .unwrap_err()
                .to_string()
                .contains("too granular")
        );
    }

    #[test]
    fn mismatched_certificate_identity_and_effects_are_rejected() {
        let mut rows = fixture();
        rows[0].certificate.interaction_ppm += 1;
        assert!(
            export_cluster_deltas(&rows, &assignments(), &metadata())
                .unwrap_err()
                .to_string()
                .contains("signed effects do not recompute")
        );
    }

    #[test]
    fn invalid_bundle_lock_and_underpowered_bootstrap_config_are_rejected() {
        let mut config = metadata();
        config.bundle_lock_sha256 = "not-a-digest".to_owned();
        assert!(
            export_cluster_deltas(&fixture(), &assignments(), &config)
                .unwrap_err()
                .to_string()
                .contains("64-character")
        );
        let mut config = metadata();
        config.cluster_assignment_manifest_sha256 = "not-a-digest".to_owned();
        assert!(
            export_cluster_deltas(&fixture(), &assignments(), &config)
                .unwrap_err()
                .to_string()
                .contains("64-character")
        );
        let mut config = metadata();
        config.bootstrap_replicates = 1_000;
        assert!(
            export_cluster_deltas(&fixture(), &assignments(), &config)
                .unwrap_err()
                .to_string()
                .contains("fewer than 20 draws")
        );
    }
}
