//! Randomized, source-cluster-level contamination exposure intervention manifests.
//!
//! The assignment unit is a source/task cluster, never an individual item.
//! Treatment clusters receive benchmark-derived training variants; the matched
//! placebo training run receives unrelated records with equal token counts.
//! Randomization is stratified by task family and cluster-size bucket.
//!
//! A valid manifest proves deterministic allocation and declared token matching.
//! It does not prove the training backend used the declared corpus, that the
//! model checkpoint was produced from it, or that the source-family labels are
//! scientifically correct. Those claims require independently captured
//! training receipts and post-training checkpoint attestations.

use crate::{hash_domain, hmac_sha256};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const INTERVENTION_DESIGN_SCHEMA: &str =
    "symthaea.randomized_contamination_intervention_design.v1";
pub const INTERVENTION_MANIFEST_SCHEMA: &str =
    "symthaea.randomized_contamination_intervention_manifest.v1";
pub const INTERVENTION_AUDIT_SCHEMA: &str =
    "symthaea.randomized_contamination_intervention_audit.v1";
pub const INTERVENTION_MIN_SEED_BYTES: usize = 32;
pub const MAX_INTERVENTION_CLUSTERS: usize = 100_000;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExposureMechanismV1 {
    ExactInstance,
    SyntacticVariant,
    SemanticVariant,
    SharedTaskRecipe,
    AnswerOnly,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingStageV1 {
    ContinuedPretraining,
    InstructionSft,
    AnswerAugmentedSft,
    PreferenceOptimization,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClusterSizeBucketV1 {
    Small,
    Medium,
    Large,
}

fn size_bucket(item_count: usize) -> ClusterSizeBucketV1 {
    match item_count {
        0..=4 => ClusterSizeBucketV1::Small,
        5..=16 => ClusterSizeBucketV1::Medium,
        _ => ClusterSizeBucketV1::Large,
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceClusterV1 {
    pub cluster_id: String,
    pub task_family: String,
    pub source_family_sha256: String,
    pub item_ids: Vec<String>,
    /// Digest of the benchmark-derived record/variant used in treatment training.
    pub exposure_variant_sha256: String,
    pub exposure_token_count: u64,
    /// Digest of unrelated replacement content for the matched placebo run.
    pub matched_placebo_record_sha256: String,
    pub matched_placebo_token_count: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RandomizedInterventionDesignV1 {
    pub schema: String,
    pub experiment_id: String,
    pub replicate_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub benchmark_task_set_sha256: String,
    pub model_base_checkpoint_sha256: String,
    pub training_stage: TrainingStageV1,
    pub exposure_mechanism: ExposureMechanismV1,
    /// Target fraction of eligible source clusters assigned to treatment.
    pub target_exposure_fraction_permille: u16,
    /// The treatment and placebo models must use the same total token budget.
    pub shared_training_corpus_sha256: String,
    pub shared_training_token_count: u64,
    pub total_training_token_budget: u64,
    pub training_steps: u64,
    pub training_code_sha256: String,
    pub runtime_image_sha256: String,
    pub tokenization_pipeline_sha256: String,
    pub training_hyperparameters_sha256: String,
    pub optimizer_schedule_sha256: String,
    pub treatment_training_corpus_sha256: String,
    pub placebo_training_corpus_sha256: String,
    pub clusters: Vec<SourceClusterV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InterventionAssignmentV1 {
    pub cluster_id: String,
    pub task_family: String,
    pub source_family_sha256: String,
    pub item_ids: Vec<String>,
    pub size_bucket: ClusterSizeBucketV1,
    pub priority_sha256: String,
    pub assigned_to_exposure: bool,
    pub exposure_variant_sha256: String,
    pub exposure_token_count: u64,
    pub matched_placebo_record_sha256: String,
    pub matched_placebo_token_count: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AllocationStratumV1 {
    pub task_family: String,
    pub size_bucket: ClusterSizeBucketV1,
    pub cluster_count: u64,
    pub exposed_cluster_count: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RandomizedInterventionManifestV1 {
    pub schema: String,
    pub experiment_id: String,
    pub replicate_id: String,
    pub design_sha256: String,
    pub training_stage: TrainingStageV1,
    pub exposure_mechanism: ExposureMechanismV1,
    pub model_base_checkpoint_sha256: String,
    pub shared_training_corpus_sha256: String,
    pub shared_training_token_count: u64,
    pub total_training_token_budget: u64,
    pub randomization_seed_commitment_sha256: String,
    pub assignment_sha256: String,
    pub exposed_cluster_count: u64,
    pub total_cluster_count: u64,
    pub exposed_item_count: u64,
    pub total_item_count: u64,
    pub exposed_training_token_count: u64,
    pub matched_placebo_token_count: u64,
    pub actual_exposure_cluster_fraction_permille: u16,
    pub actual_exposure_token_fraction_permille: u16,
    pub allocation_strata: Vec<AllocationStratumV1>,
    pub assignments: Vec<InterventionAssignmentV1>,
    pub manifest_sha256: String,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InterventionAuditReportV1 {
    pub schema: String,
    pub valid: bool,
    pub randomization_replayed: bool,
    pub experiment_id: String,
    pub replicate_id: String,
    pub design_sha256: String,
    pub manifest_sha256: String,
    pub seed_commitment_sha256: String,
    pub cluster_count: u64,
    pub item_count: u64,
    pub exposed_cluster_count: u64,
    pub exposed_item_count: u64,
    pub exposed_training_token_count: u64,
    pub matched_placebo_token_count: u64,
    pub shared_training_token_count: u64,
    pub total_training_token_budget: u64,
    pub treatment_token_budget_conserved: bool,
    pub placebo_token_budget_conserved: bool,
    pub token_budget_match: bool,
    pub task_family_stratification_complete: bool,
    pub source_cluster_integrity: bool,
    pub claim_boundary: String,
}

#[derive(Serialize)]
struct ManifestDigestPayloadV1<'a> {
    schema: &'a str,
    experiment_id: &'a str,
    replicate_id: &'a str,
    design_sha256: &'a str,
    training_stage: TrainingStageV1,
    exposure_mechanism: ExposureMechanismV1,
    model_base_checkpoint_sha256: &'a str,
    shared_training_corpus_sha256: &'a str,
    shared_training_token_count: u64,
    total_training_token_budget: u64,
    randomization_seed_commitment_sha256: &'a str,
    assignment_sha256: &'a str,
    exposed_cluster_count: u64,
    total_cluster_count: u64,
    exposed_item_count: u64,
    total_item_count: u64,
    exposed_training_token_count: u64,
    matched_placebo_token_count: u64,
    actual_exposure_cluster_fraction_permille: u16,
    actual_exposure_token_fraction_permille: u16,
    allocation_strata: &'a [AllocationStratumV1],
    assignments: &'a [InterventionAssignmentV1],
    claim_boundary: &'a str,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be a lowercase SHA-256 digest"))
    }
}

fn priority_digest(
    seed: &[u8],
    experiment_id: &str,
    replicate_id: &str,
    stratum_id: &str,
    cluster_id: &str,
) -> String {
    let mut message = Vec::with_capacity(
        experiment_id.len() + replicate_id.len() + stratum_id.len() + cluster_id.len() + 56,
    );
    for value in [experiment_id, replicate_id, stratum_id, cluster_id] {
        message.extend_from_slice(&(value.len() as u64).to_be_bytes());
        message.extend_from_slice(value.as_bytes());
    }
    crate::hex_digest(&hmac_sha256(seed, &message))
}

fn stratum_id(task_family: &str, bucket: ClusterSizeBucketV1) -> String {
    format!("{task_family}:{bucket:?}")
}

fn validate_design(
    design: &RandomizedInterventionDesignV1,
    seed: &[u8],
) -> Result<(), String> {
    if design.schema != INTERVENTION_DESIGN_SCHEMA {
        return Err(format!("design schema must be {INTERVENTION_DESIGN_SCHEMA}"));
    }
    if seed.len() < INTERVENTION_MIN_SEED_BYTES {
        return Err(format!(
            "randomization seed must contain at least {INTERVENTION_MIN_SEED_BYTES} bytes"
        ));
    }
    for (value, label) in [
        (&design.benchmark_task_set_sha256, "benchmark_task_set_sha256"),
        (&design.model_base_checkpoint_sha256, "model_base_checkpoint_sha256"),
        (&design.shared_training_corpus_sha256, "shared_training_corpus_sha256"),
        (&design.training_code_sha256, "training_code_sha256"),
        (&design.runtime_image_sha256, "runtime_image_sha256"),
        (&design.tokenization_pipeline_sha256, "tokenization_pipeline_sha256"),
        (&design.training_hyperparameters_sha256, "training_hyperparameters_sha256"),
        (&design.optimizer_schedule_sha256, "optimizer_schedule_sha256"),
        (&design.treatment_training_corpus_sha256, "treatment_training_corpus_sha256"),
        (&design.placebo_training_corpus_sha256, "placebo_training_corpus_sha256"),
    ] {
        validate_digest(value, label)?;
    }
    if design.experiment_id.trim().is_empty()
        || design.experiment_id.len() > 128
        || design.replicate_id.trim().is_empty()
        || design.replicate_id.len() > 128
        || design.benchmark_id.trim().is_empty()
        || design.benchmark_version.trim().is_empty()
    {
        return Err("experiment, replicate and benchmark identity fields must be nonempty".to_owned());
    }
    if design.total_training_token_budget == 0 || design.training_steps == 0 {
        return Err("training token budget and training_steps must be positive".to_owned());
    }
    if design.target_exposure_fraction_permille > 1_000 {
        return Err("target_exposure_fraction_permille must be in [0, 1000]".to_owned());
    }
    if design.treatment_training_corpus_sha256 == design.placebo_training_corpus_sha256
        || design.shared_training_corpus_sha256 == design.treatment_training_corpus_sha256
        || design.shared_training_corpus_sha256 == design.placebo_training_corpus_sha256
    {
        return Err("shared, treatment and placebo training corpus digests must be pairwise distinct".to_owned());
    }
    if design.shared_training_token_count == 0
        || design.shared_training_token_count > design.total_training_token_budget
    {
        return Err("shared_training_token_count must be positive and no greater than total budget".to_owned());
    }
    if !(2..=MAX_INTERVENTION_CLUSTERS).contains(&design.clusters.len()) {
        return Err(format!(
            "cluster count must be in [2, {MAX_INTERVENTION_CLUSTERS}]"
        ));
    }

    let mut cluster_ids = BTreeSet::new();
    let mut item_ids = BTreeSet::new();
    let mut source_digests = BTreeSet::new();
    let mut stratum_exposure_token_counts = BTreeMap::<String, u64>::new();
    let mut exposure_variant_digests = BTreeSet::new();
    let mut placebo_record_digests = BTreeSet::new();
    for cluster in &design.clusters {
        if cluster.cluster_id.trim().is_empty()
            || cluster.cluster_id.len() > 256
            || !cluster_ids.insert(cluster.cluster_id.as_str())
        {
            return Err("cluster IDs must be unique, nonempty and <=256 bytes".to_owned());
        }
        if cluster.task_family.trim().is_empty() || cluster.task_family.len() > 128 {
            return Err(format!("cluster {} has invalid task_family", cluster.cluster_id));
        }
        validate_digest(
            &cluster.source_family_sha256,
            "cluster.source_family_sha256",
        )?;
        validate_digest(
            &cluster.exposure_variant_sha256,
            "cluster.exposure_variant_sha256",
        )?;
        validate_digest(
            &cluster.matched_placebo_record_sha256,
            "cluster.matched_placebo_record_sha256",
        )?;
        if !exposure_variant_digests.insert(cluster.exposure_variant_sha256.as_str()) {
            return Err("exposure variant record digest is reused across source clusters".to_owned());
        }
        if !placebo_record_digests.insert(cluster.matched_placebo_record_sha256.as_str()) {
            return Err("matched placebo record digest is reused across source clusters".to_owned());
        }
        if cluster.exposure_variant_sha256 == cluster.matched_placebo_record_sha256 {
            return Err(format!(
                "cluster {} exposure variant and matched placebo record digests must differ",
                cluster.cluster_id
            ));
        }
        if cluster.item_ids.is_empty() || cluster.item_ids.len() > 10_000 {
            return Err(format!("cluster {} must contain 1..=10000 item IDs", cluster.cluster_id));
        }
        if cluster.exposure_token_count == 0
            || cluster.matched_placebo_token_count != cluster.exposure_token_count
        {
            return Err(format!(
                "cluster {} exposure and placebo token counts must be equal and positive",
                cluster.cluster_id
            ));
        }
        let stratum = stratum_id(&cluster.task_family, size_bucket(cluster.item_ids.len()));
        if let Some(existing) = stratum_exposure_token_counts.get(&stratum) {
            if *existing != cluster.exposure_token_count {
                return Err(format!(
                    "stratum {stratum} must use an equal exposure token count for seed-invariant budget conservation"
                ));
            }
        } else {
            stratum_exposure_token_counts.insert(stratum, cluster.exposure_token_count);
        }
        if !source_digests.insert(cluster.source_family_sha256.as_str()) {
            // Multiple clusters may originate from the same source family. This
            // is allowed only if they are still assigned as one atomic cluster.
            // Distinct cluster IDs with a repeated source family digest are
            // rejected so family leakage cannot be disguised by splitting IDs.
            return Err(format!(
                "source family {} appears in multiple clusters",
                cluster.source_family_sha256
            ));
        }
        let mut within_cluster = BTreeSet::new();
        for item_id in &cluster.item_ids {
            if item_id.trim().is_empty() || item_id.len() > 256 {
                return Err(format!("cluster {} contains an empty/oversized item ID", cluster.cluster_id));
            }
            if !within_cluster.insert(item_id.as_str()) || !item_ids.insert(item_id.as_str()) {
                return Err(format!(
                    "item {} is duplicated within/across source clusters",
                    item_id
                ));
            }
        }
    }
    Ok(())
}

fn manifest_digest(manifest: &RandomizedInterventionManifestV1) -> Result<String, String> {
    let payload = ManifestDigestPayloadV1 {
        schema: &manifest.schema,
        experiment_id: &manifest.experiment_id,
        replicate_id: &manifest.replicate_id,
        design_sha256: &manifest.design_sha256,
        training_stage: manifest.training_stage,
        exposure_mechanism: manifest.exposure_mechanism,
        model_base_checkpoint_sha256: &manifest.model_base_checkpoint_sha256,
        shared_training_corpus_sha256: &manifest.shared_training_corpus_sha256,
        shared_training_token_count: manifest.shared_training_token_count,
        total_training_token_budget: manifest.total_training_token_budget,
        randomization_seed_commitment_sha256: &manifest.randomization_seed_commitment_sha256,
        assignment_sha256: &manifest.assignment_sha256,
        exposed_cluster_count: manifest.exposed_cluster_count,
        total_cluster_count: manifest.total_cluster_count,
        exposed_item_count: manifest.exposed_item_count,
        total_item_count: manifest.total_item_count,
        exposed_training_token_count: manifest.exposed_training_token_count,
        matched_placebo_token_count: manifest.matched_placebo_token_count,
        actual_exposure_cluster_fraction_permille: manifest.actual_exposure_cluster_fraction_permille,
        actual_exposure_token_fraction_permille: manifest.actual_exposure_token_fraction_permille,
        allocation_strata: &manifest.allocation_strata,
        assignments: &manifest.assignments,
        claim_boundary: &manifest.claim_boundary,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize intervention manifest payload: {error}"))?;
    Ok(hash_domain(b"randomized-intervention-manifest.v1", &bytes))
}

/// Randomize source/task clusters into exposure and unexposed groups within
/// task-family and cluster-size strata. Selection is deterministic for a fixed
/// seed and design, with round-to-nearest target allocation per stratum.
pub fn randomize_intervention(
    design: &RandomizedInterventionDesignV1,
    seed: &[u8],
) -> Result<RandomizedInterventionManifestV1, String> {
    validate_design(design, seed)?;
    let design_bytes = serde_json::to_vec(design)
        .map_err(|error| format!("cannot serialize intervention design: {error}"))?;
    let design_sha256 = hash_domain(b"randomized-intervention-design.v1", &design_bytes);
    let randomization_seed_commitment_sha256 =
        hash_domain(b"randomized-intervention-seed.v1", seed);

    let mut strata = BTreeMap::<String, Vec<&SourceClusterV1>>::new();
    for cluster in &design.clusters {
        let bucket = size_bucket(cluster.item_ids.len());
        let stratum = stratum_id(&cluster.task_family, bucket);
        strata.entry(stratum).or_default().push(cluster);
    }

    let mut assignments = Vec::with_capacity(design.clusters.len());
    let mut allocation_strata = Vec::with_capacity(strata.len());
    let mut exposed_cluster_count = 0_u64;
    let mut total_item_count = 0_u64;
    let mut exposed_item_count = 0_u64;
    let mut exposed_training_token_count = 0_u64;
    let mut matched_placebo_token_count = 0_u64;
    let total_cluster_count = u64::try_from(design.clusters.len())
        .map_err(|_| "total cluster count exceeds u64")?;

    for (stratum, mut clusters) in strata {
        if design.target_exposure_fraction_permille > 0
            && design.target_exposure_fraction_permille < 1_000
            && clusters.len() < 2
        {
            return Err(format!(
                "stratum {stratum} has fewer than two clusters and cannot contain both treatment and placebo"
            ));
        }
        let bucket = size_bucket(clusters[0].item_ids.len());
        let task_family = clusters[0].task_family.clone();
        let mut ranked: Vec<(String, &SourceClusterV1)> = clusters
            .drain(..)
            .map(|cluster| {
                (
                    priority_digest(seed, &design.experiment_id, &design.replicate_id, &stratum, &cluster.cluster_id),
                    cluster,
                )
            })
            .collect();
        ranked.sort_by(|left, right| {
            left.0.cmp(&right.0).then_with(|| {
                left.1.cluster_id.cmp(&right.1.cluster_id)
            })
        });
        let stratum_count = ranked.len();
        let rounded_exposed_count = (stratum_count
            .checked_mul(usize::from(design.target_exposure_fraction_permille))
            .ok_or_else(|| "stratum exposure count overflows usize".to_owned())?
            + 500)
            / 1_000;
        let exposed_count = if design.target_exposure_fraction_permille == 0 {
            0
        } else if design.target_exposure_fraction_permille == 1_000 {
            stratum_count
        } else {
            // Any interior dose must contain both treatment and placebo clusters.
            rounded_exposed_count.clamp(1, stratum_count - 1)
        };
        allocation_strata.push(AllocationStratumV1 {
            task_family,
            size_bucket: bucket,
            cluster_count: u64::try_from(stratum_count)
                .map_err(|_| "stratum cluster count exceeds u64")?,
            exposed_cluster_count: u64::try_from(exposed_count)
                .map_err(|_| "stratum exposed count exceeds u64")?,
        });

        for (rank, (priority_sha256, cluster)) in ranked.into_iter().enumerate() {
            let assigned_to_exposure = rank < exposed_count;
            let item_count = u64::try_from(cluster.item_ids.len())
                .map_err(|_| "cluster item count exceeds u64")?;
            total_item_count = total_item_count
                .checked_add(item_count)
                .ok_or_else(|| "total item count overflows u64".to_owned())?;
            if assigned_to_exposure {
                exposed_cluster_count = exposed_cluster_count
                    .checked_add(1)
                    .ok_or_else(|| "exposed cluster count overflows u64".to_owned())?;
                exposed_item_count = exposed_item_count
                    .checked_add(item_count)
                    .ok_or_else(|| "exposed item count overflows u64".to_owned())?;
                exposed_training_token_count = exposed_training_token_count
                    .checked_add(cluster.exposure_token_count)
                    .ok_or_else(|| "exposed training token count overflows u64".to_owned())?;
                matched_placebo_token_count = matched_placebo_token_count
                    .checked_add(cluster.matched_placebo_token_count)
                    .ok_or_else(|| "matched placebo token count overflows u64".to_owned())?;
            }
            assignments.push(InterventionAssignmentV1 {
                cluster_id: cluster.cluster_id.clone(),
                task_family: cluster.task_family.clone(),
                source_family_sha256: cluster.source_family_sha256.clone(),
                item_ids: cluster.item_ids.clone(),
                size_bucket: bucket,
                priority_sha256,
                assigned_to_exposure,
                exposure_variant_sha256: cluster.exposure_variant_sha256.clone(),
                exposure_token_count: cluster.exposure_token_count,
                matched_placebo_record_sha256: cluster.matched_placebo_record_sha256.clone(),
                matched_placebo_token_count: cluster.matched_placebo_token_count,
            });
        }
    }

    if exposed_training_token_count > design.total_training_token_budget {
        return Err("exposed benchmark token count exceeds total training token budget".to_owned());
    }
    if matched_placebo_token_count != exposed_training_token_count {
        return Err("matched placebo token budget does not equal exposed benchmark token budget".to_owned());
    }
    let treatment_total = design.shared_training_token_count
        .checked_add(exposed_training_token_count)
        .ok_or_else(|| "treatment total training token count overflows u64".to_owned())?;
    let placebo_total = design.shared_training_token_count
        .checked_add(matched_placebo_token_count)
        .ok_or_else(|| "placebo total training token count overflows u64".to_owned())?;
    if treatment_total != design.total_training_token_budget
        || placebo_total != design.total_training_token_budget
    {
        return Err("shared corpus plus matched intervention tokens must equal total training token budget in both arms".to_owned());
    }
    assignments.sort_by(|left, right| left.cluster_id.cmp(&right.cluster_id));
    allocation_strata.sort_by(|left, right| {
        left.task_family
            .cmp(&right.task_family)
            .then_with(|| left.size_bucket.cmp(&right.size_bucket))
    });

    let actual_exposure_cluster_fraction_permille = if total_cluster_count == 0 {
        0
    } else {
        u16::try_from(
            (u128::from(exposed_cluster_count) * 1_000 / u128::from(total_cluster_count))
                .min(1_000),
        )
        .map_err(|_| "actual cluster exposure fraction exceeds u16")?
    };
    let actual_exposure_token_fraction_permille = if design.total_training_token_budget == 0 {
        0
    } else {
        u16::try_from(
            (u128::from(exposed_training_token_count) * 1_000
                / u128::from(design.total_training_token_budget))
                .min(1_000),
        )
        .map_err(|_| "actual token exposure fraction exceeds u16")?
    };

    let assignment_bytes = serde_json::to_vec(&assignments)
        .map_err(|error| format!("cannot serialize intervention assignments: {error}"))?;
    let mut manifest = RandomizedInterventionManifestV1 {
        schema: INTERVENTION_MANIFEST_SCHEMA.to_owned(),
        experiment_id: design.experiment_id.clone(),
        replicate_id: design.replicate_id.clone(),
        design_sha256,
        training_stage: design.training_stage,
        exposure_mechanism: design.exposure_mechanism,
        model_base_checkpoint_sha256: design.model_base_checkpoint_sha256.clone(),
        shared_training_corpus_sha256: design.shared_training_corpus_sha256.clone(),
        shared_training_token_count: design.shared_training_token_count,
        total_training_token_budget: design.total_training_token_budget,
        randomization_seed_commitment_sha256,
        assignment_sha256: hash_domain(b"randomized-intervention-assignments.v1", &assignment_bytes),
        exposed_cluster_count,
        total_cluster_count,
        exposed_item_count,
        total_item_count,
        exposed_training_token_count,
        matched_placebo_token_count,
        actual_exposure_cluster_fraction_permille,
        actual_exposure_token_fraction_permille,
        allocation_strata,
        assignments,
        manifest_sha256: String::new(),
        claim_boundary: "replayable_source_cluster_assignment_and_declared_token_matching_only_not_verified_training_execution_or_checkpoint_provenance".to_owned(),
    };
    manifest.manifest_sha256 = manifest_digest(&manifest)?;
    Ok(manifest)
}

/// Recompute the exact assignment from the preregistered design and seed.
/// This detects any edits to item/cluster membership, exposure labels, priorities,
/// matched token counts, or report summaries.
pub fn audit_intervention(
    design: &RandomizedInterventionDesignV1,
    manifest: &RandomizedInterventionManifestV1,
    seed: &[u8],
) -> Result<InterventionAuditReportV1, String> {
    let supplied_manifest_digest = manifest_digest(manifest)?;
    if supplied_manifest_digest != manifest.manifest_sha256 {
        return Err("intervention manifest digest does not match its contents".to_owned());
    }
    let assignment_bytes = serde_json::to_vec(&manifest.assignments)
        .map_err(|error| format!("cannot serialize supplied assignments: {error}"))?;
    let supplied_assignment_digest =
        hash_domain(b"randomized-intervention-assignments.v1", &assignment_bytes);
    if supplied_assignment_digest != manifest.assignment_sha256 {
        return Err("intervention assignment digest does not match its assignments".to_owned());
    }

    let expected = randomize_intervention(design, seed)?;
    if expected.manifest_sha256 != manifest.manifest_sha256 {
        return Err("intervention manifest digest does not match replayed allocation".to_owned());
    }
    let expected_bytes = serde_json::to_vec(&expected)
        .map_err(|error| format!("cannot serialize replayed manifest: {error}"))?;
    let observed_bytes = serde_json::to_vec(manifest)
        .map_err(|error| format!("cannot serialize supplied manifest: {error}"))?;
    if expected_bytes != observed_bytes {
        return Err("intervention manifest differs from deterministic replay".to_owned());
    }
    let cluster_integrity = manifest.assignments.iter().all(|assignment| {
        !assignment.item_ids.is_empty()
            && assignment.item_ids.iter().all(|item| !item.trim().is_empty())
    });
    let task_family_stratification_complete = manifest.allocation_strata.iter().all(|stratum| {
        stratum.exposed_cluster_count <= stratum.cluster_count
    });
    Ok(InterventionAuditReportV1 {
        schema: INTERVENTION_AUDIT_SCHEMA.to_owned(),
        valid: true,
        randomization_replayed: true,
        experiment_id: manifest.experiment_id.clone(),
        replicate_id: manifest.replicate_id.clone(),
        design_sha256: manifest.design_sha256.clone(),
        manifest_sha256: manifest.manifest_sha256.clone(),
        seed_commitment_sha256: manifest.randomization_seed_commitment_sha256.clone(),
        cluster_count: manifest.total_cluster_count,
        item_count: manifest.total_item_count,
        exposed_cluster_count: manifest.exposed_cluster_count,
        exposed_item_count: manifest.exposed_item_count,
        exposed_training_token_count: manifest.exposed_training_token_count,
        matched_placebo_token_count: manifest.matched_placebo_token_count,
        shared_training_token_count: design.shared_training_token_count,
        total_training_token_budget: design.total_training_token_budget,
        treatment_token_budget_conserved: design.shared_training_token_count
            .checked_add(manifest.exposed_training_token_count)
            == Some(design.total_training_token_budget),
        placebo_token_budget_conserved: design.shared_training_token_count
            .checked_add(manifest.matched_placebo_token_count)
            == Some(design.total_training_token_budget),
        token_budget_match: manifest.exposed_training_token_count
            == manifest.matched_placebo_token_count,
        task_family_stratification_complete,
        source_cluster_integrity: cluster_integrity,
        claim_boundary: "deterministic_randomization_replayed_and_manifest_integrity_checked_only_not_proof_training_backend_used_declared_corpus".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn digest(value: &str) -> String {
        hash_domain(b"intervention-fixture", value.as_bytes())
    }

    fn design() -> RandomizedInterventionDesignV1 {
        let mut clusters = Vec::new();
        for family in ["math", "code", "science", "logic"] {
            for index in 0..8 {
                let cluster_id = format!("{family}-cluster-{index}");
                clusters.push(SourceClusterV1 {
                    cluster_id: cluster_id.clone(),
                    task_family: family.to_owned(),
                    source_family_sha256: digest(&format!("source-{cluster_id}")),
                    item_ids: vec![
                        format!("{cluster_id}-item-1"),
                        format!("{cluster_id}-item-2"),
                    ],
                    exposure_variant_sha256: digest(&format!("variant-{cluster_id}")),
                    exposure_token_count: 500,
                    matched_placebo_record_sha256: digest(&format!("placebo-{cluster_id}")),
                    matched_placebo_token_count: 500,
                });
            }
        }
        RandomizedInterventionDesignV1 {
            schema: INTERVENTION_DESIGN_SCHEMA.to_owned(),
            experiment_id: "causal-contam-fixture".to_owned(),
            replicate_id: "replicate-001".to_owned(),
            benchmark_id: "contamination-panel".to_owned(),
            benchmark_version: "v1".to_owned(),
            benchmark_task_set_sha256: digest("task-set"),
            model_base_checkpoint_sha256: digest("base-checkpoint"),
            training_stage: TrainingStageV1::InstructionSft,
            exposure_mechanism: ExposureMechanismV1::SemanticVariant,
            target_exposure_fraction_permille: 500,
            shared_training_corpus_sha256: digest("shared-corpus"),
            shared_training_token_count: 1_992_000,
            total_training_token_budget: 2_000_000,
            training_steps: 5_000,
            training_code_sha256: digest("training-code"),
            runtime_image_sha256: digest("runtime-image"),
            tokenization_pipeline_sha256: digest("tokenization-pipeline"),
            training_hyperparameters_sha256: digest("hyperparameters"),
            optimizer_schedule_sha256: digest("optimizer-schedule"),
            treatment_training_corpus_sha256: digest("treatment-corpus"),
            placebo_training_corpus_sha256: digest("placebo-corpus"),
            clusters,
        }
    }

    fn seed() -> Vec<u8> {
        vec![0xA5; INTERVENTION_MIN_SEED_BYTES]
    }

    #[test]
    fn versioned_schemas_and_design_fixture_are_consistent() {
        let design_schema: Value = serde_json::from_str(include_str!(
            "../schema/randomized-intervention-design.schema.json"
        )).unwrap();
        let manifest_schema: Value = serde_json::from_str(include_str!(
            "../schema/randomized-intervention-manifest.schema.json"
        )).unwrap();
        let audit_schema: Value = serde_json::from_str(include_str!(
            "../schema/randomized-intervention-audit.schema.json"
        )).unwrap();
        let fixture: RandomizedInterventionDesignV1 = serde_json::from_str(include_str!(
            "../fixtures/intervention-semantic-variant-design.json"
        )).unwrap();

        assert_eq!(
            design_schema["properties"]["schema"]["const"].as_str(),
            Some(INTERVENTION_DESIGN_SCHEMA)
        );
        let design_required: BTreeSet<String> = design_schema["required"].as_array().unwrap()
            .iter().map(|value| value.as_str().unwrap().to_owned()).collect();
        let design_properties: BTreeSet<String> = design_schema["properties"].as_object().unwrap()
            .keys().cloned().collect();
        assert_eq!(design_required, design_properties);

        assert_eq!(
            manifest_schema["properties"]["schema"]["const"].as_str(),
            Some(INTERVENTION_MANIFEST_SCHEMA)
        );
        let manifest_required: BTreeSet<String> = manifest_schema["required"].as_array().unwrap()
            .iter().map(|value| value.as_str().unwrap().to_owned()).collect();
        let manifest_properties: BTreeSet<String> = manifest_schema["properties"].as_object().unwrap()
            .keys().cloned().collect();
        assert_eq!(manifest_required, manifest_properties);

        let audit_required: BTreeSet<String> = audit_schema["required"].as_array().unwrap()
            .iter().map(|value| value.as_str().unwrap().to_owned()).collect();
        let audit_properties: BTreeSet<String> = audit_schema["properties"].as_object().unwrap()
            .keys().cloned().collect();
        assert_eq!(audit_required, audit_properties);
        assert_eq!(
            audit_schema["properties"]["schema"]["const"].as_str(),
            Some(INTERVENTION_AUDIT_SCHEMA)
        );

        let manifest = randomize_intervention(&fixture, &seed()).unwrap();
        let manifest_json: Value = serde_json::to_value(&manifest).unwrap();
        assert_eq!(manifest_json["schema"].as_str(), Some(INTERVENTION_MANIFEST_SCHEMA));
        let audit = audit_intervention(&fixture, &manifest, &seed()).unwrap();
        assert_eq!(
            serde_json::to_value(&audit).unwrap()["schema"].as_str(),
            Some(INTERVENTION_AUDIT_SCHEMA)
        );
        assert_eq!(manifest.exposed_training_token_count, 8_000);
        assert_eq!(manifest.shared_training_token_count, 92_000);
        assert_eq!(manifest.total_training_token_budget, 100_000);
    }

    #[test]
    fn randomization_is_deterministic_and_replayable() {
        let design = design();
        let first = randomize_intervention(&design, &seed()).unwrap();
        let second = randomize_intervention(&design, &seed()).unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        let report = audit_intervention(&design, &first, &seed()).unwrap();
        assert!(report.valid && report.randomization_replayed);
    }

    #[test]
    fn task_family_and_cluster_size_strata_receive_balanced_exposure_counts() {
        let manifest = randomize_intervention(&design(), &seed()).unwrap();
        assert_eq!(manifest.allocation_strata.len(), 4);
        assert!(manifest.allocation_strata.iter().all(|stratum| {
            stratum.cluster_count == 8 && stratum.exposed_cluster_count == 4
        }));
        assert_eq!(manifest.exposed_cluster_count, 16);
        assert_eq!(manifest.total_cluster_count, 32);
    }

    #[test]
    fn cluster_assignment_never_splits_items_across_exposure_status() {
        let manifest = randomize_intervention(&design(), &seed()).unwrap();
        let mut seen_items = BTreeSet::new();
        for assignment in &manifest.assignments {
            assert!(assignment.item_ids.iter().all(|item| seen_items.insert(item)));
            assert_eq!(assignment.size_bucket, ClusterSizeBucketV1::Small);
        }
        assert_eq!(seen_items.len(), 64);
    }

    #[test]
    fn treatment_exposure_and_placebo_tokens_are_exactly_matched() {
        let manifest = randomize_intervention(&design(), &seed()).unwrap();
        assert_eq!(
            manifest.exposed_training_token_count,
            manifest.matched_placebo_token_count
        );
        assert!(manifest.exposed_training_token_count > 0);
        assert!(manifest.actual_exposure_token_fraction_permille <= 1_000);
    }

    #[test]
    fn extreme_interior_doses_still_leave_both_arms_in_every_stratum() {
        let mut one_permille = design();
        one_permille.target_exposure_fraction_permille = 1;
        one_permille.shared_training_token_count = 1_998_000;
        let low_manifest = randomize_intervention(&one_permille, &seed()).unwrap();
        assert!(low_manifest.allocation_strata.iter().all(|stratum| {
            stratum.exposed_cluster_count == 1
                && stratum.exposed_cluster_count < stratum.cluster_count
        }));

        let mut nine_nine_nine = design();
        nine_nine_nine.target_exposure_fraction_permille = 999;
        nine_nine_nine.shared_training_token_count = 1_986_000;
        let high_manifest = randomize_intervention(&nine_nine_nine, &seed()).unwrap();
        assert!(high_manifest.allocation_strata.iter().all(|stratum| {
            stratum.exposed_cluster_count == stratum.cluster_count - 1
                && stratum.exposed_cluster_count > 0
        }));
    }

    #[test]
    fn zero_dose_produces_a_clean_negative_control() {
        let mut design = design();
        design.target_exposure_fraction_permille = 0;
        design.shared_training_token_count = design.total_training_token_budget;
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        assert_eq!(manifest.exposed_cluster_count, 0);
        assert_eq!(manifest.exposed_item_count, 0);
        assert_eq!(manifest.exposed_training_token_count, 0);
        assert_eq!(manifest.matched_placebo_token_count, 0);
    }

    #[test]
    fn full_dose_exposes_every_cluster_within_each_stratum() {
        let mut design = design();
        design.target_exposure_fraction_permille = 1_000;
        design.shared_training_token_count = 1_984_000;
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        assert_eq!(manifest.exposed_cluster_count, manifest.total_cluster_count);
        assert_eq!(manifest.exposed_item_count, manifest.total_item_count);
    }

    #[test]
    fn duplicate_source_family_digest_is_rejected_to_prevent_cluster_splitting() {
        let mut design = design();
        design.clusters[1].source_family_sha256 = design.clusters[0].source_family_sha256.clone();
        assert!(randomize_intervention(&design, &seed())
            .unwrap_err()
            .contains("source family"));
    }

    #[test]
    fn replicate_id_changes_randomized_priorities_when_seed_is_reused() {
        let mut first_design = design();
        let first = randomize_intervention(&first_design, &seed()).unwrap();
        first_design.replicate_id = "replicate-002".to_owned();
        let second = randomize_intervention(&first_design, &seed()).unwrap();
        let first_priorities: BTreeMap<&str, &str> = first.assignments.iter()
            .map(|assignment| (assignment.cluster_id.as_str(), assignment.priority_sha256.as_str()))
            .collect();
        let second_priorities: BTreeMap<&str, &str> = second.assignments.iter()
            .map(|assignment| (assignment.cluster_id.as_str(), assignment.priority_sha256.as_str()))
            .collect();
        assert_ne!(first_priorities, second_priorities);
    }

    #[test]
    fn identical_exposure_and_placebo_payloads_are_rejected() {
        let mut invalid_design = design();
        invalid_design.clusters[0].matched_placebo_record_sha256 =
            invalid_design.clusters[0].exposure_variant_sha256.clone();
        assert!(randomize_intervention(&invalid_design, &seed())
            .unwrap_err()
            .contains("exposure variant and matched placebo record digests must differ"));
    }

    #[test]
    fn common_corpus_must_differ_from_both_arm_specific_corpora() {
        let mut invalid_design = design();
        invalid_design.shared_training_corpus_sha256 =
            invalid_design.treatment_training_corpus_sha256.clone();
        assert!(randomize_intervention(&invalid_design, &seed())
            .unwrap_err()
            .contains("must be pairwise distinct"));
    }

    #[test]
    fn duplicate_exposure_or_placebo_record_digest_is_rejected() {
        let mut invalid = design();
        invalid.clusters[1].exposure_variant_sha256 =
            invalid.clusters[0].exposure_variant_sha256.clone();
        assert!(randomize_intervention(&invalid, &seed())
            .unwrap_err().contains("exposure variant record digest is reused"));

        let mut invalid = design();
        invalid.clusters[1].matched_placebo_record_sha256 =
            invalid.clusters[0].matched_placebo_record_sha256.clone();
        assert!(randomize_intervention(&invalid, &seed())
            .unwrap_err().contains("matched placebo record digest is reused"));
    }

    #[test]
    fn duplicate_item_ids_across_clusters_fail_closed() {
        let mut design = design();
        design.clusters[1].item_ids[0] = design.clusters[0].item_ids[0].clone();
        assert!(randomize_intervention(&design, &seed())
            .unwrap_err()
            .contains("duplicated within/across"));
    }

    #[test]
    fn unequal_token_counts_within_stratum_fail_before_randomization() {
        let mut invalid_design = design();
        invalid_design.clusters[1].exposure_token_count += 1;
        invalid_design.clusters[1].matched_placebo_token_count += 1;
        assert!(randomize_intervention(&invalid_design, &seed())
            .unwrap_err()
            .contains("equal exposure token count"));
    }

    #[test]
    fn placebo_token_mismatch_fails_closed() {
        let mut design = design();
        design.clusters[0].matched_placebo_token_count += 1;
        assert!(randomize_intervention(&design, &seed())
            .unwrap_err()
            .contains("token counts must be equal"));
    }

    #[test]
    fn design_and_seed_mutations_change_or_invalidate_manifest() {
        let design = design();
        let manifest = randomize_intervention(&design, &seed()).unwrap();
        let mut changed_design = design.clone();
        changed_design.training_steps += 1;
        assert!(audit_intervention(&changed_design, &manifest, &seed()).is_err());
        let different_seed = vec![0x5A; INTERVENTION_MIN_SEED_BYTES];
        assert!(audit_intervention(&design, &manifest, &different_seed).is_err());
    }

    #[test]
    fn assignment_mutation_is_detected_by_manifest_digest_and_replay() {
        let design = design();
        let mut manifest = randomize_intervention(&design, &seed()).unwrap();
        manifest.assignments[0].assigned_to_exposure = !manifest.assignments[0].assigned_to_exposure;
        assert!(audit_intervention(&design, &manifest, &seed())
            .unwrap_err()
            .contains("manifest digest does not match its contents"));
    }

    #[test]
    fn assignment_payload_with_recomputed_manifest_hash_still_fails_replay() {
        let design = design();
        let mut manifest = randomize_intervention(&design, &seed()).unwrap();
        manifest.assignments[0].assigned_to_exposure = !manifest.assignments[0].assigned_to_exposure;
        let assignment_bytes = serde_json::to_vec(&manifest.assignments).unwrap();
        manifest.assignment_sha256 =
            hash_domain(b"randomized-intervention-assignments.v1", &assignment_bytes);
        manifest.manifest_sha256 = manifest_digest(&manifest).unwrap();
        assert!(audit_intervention(&design, &manifest, &seed())
            .unwrap_err()
            .contains("does not match replayed allocation"));
    }

    #[test]
    fn invalid_training_budget_or_fraction_fails_closed() {
        let mut invalid_design = design();
        invalid_design.total_training_token_budget = 0;
        assert!(randomize_intervention(&invalid_design, &seed()).is_err());
        let mut invalid_design = design();
        invalid_design.target_exposure_fraction_permille = 1_001;
        assert!(randomize_intervention(&invalid_design, &seed()).is_err());
    }
}
