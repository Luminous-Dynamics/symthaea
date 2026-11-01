//! Causal decomposition of training-exposure effects by evaluation-task relation.
//!
//! This module separates exact replay, surface/semantic retention, same-recipe
//! generalization, novel-recipe skill transfer, and unrelated controls.
//! The independent uncertainty unit remains the independently trained
//! treatment/placebo model pair. Items and difficulty cells are nested.
//!
//! Relation and lineage data are evaluator-only. Digests establish content
//! consistency, not identity, custody, training consumption, or truth.
//! Confirmatory eligibility is therefore hard-false in this version.

use crate::{
    causal_analysis::{
        analyze_replicated_causal_study, mean_interval, MeanIntervalV1,
        ReplicatedCausalStudyV1, ScoreCountV1,
    },
    hash_domain,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const EXPOSURE_TRANSFER_STUDY_SCHEMA: &str =
    "symthaea.exposure_transfer_study.v1";
pub const EXPOSURE_TRANSFER_REPORT_SCHEMA: &str =
    "symthaea.exposure_transfer_report.v1";
pub const MIN_TRANSFER_FAMILIES: usize = 2;
pub const MAX_TRANSFER_REPLICATES: usize = 1_000;
pub const MAX_TRANSFER_ROWS_PER_REPLICATE: usize = 10_000;
pub const MAX_ITEM_IDS_PER_ROW: usize = 100_000;
pub const MAX_ITEM_IDS_PER_REPLICATE: usize = 500_000;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum TransferRelationV1 {
    ExactReplay,
    SurfaceVariant,
    SemanticEquivalent,
    FreshSameRecipe,
    NovelRecipeSameSkill,
    UnrelatedControl,
}

impl TransferRelationV1 {
    pub const ALL: [Self; 6] = [
        Self::ExactReplay,
        Self::SurfaceVariant,
        Self::SemanticEquivalent,
        Self::FreshSameRecipe,
        Self::NovelRecipeSameSkill,
        Self::UnrelatedControl,
    ];
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RelationAdjudicationDecisionV1 {
    Accepted,
    Indeterminate,
    Rejected,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransferStratumOutcomeV1 {
    pub task_family: String,
    pub difficulty_band: String,
    pub relation: TransferRelationV1,
    pub cluster_count: u64,
    /// Opaque evaluator-side IDs; never copy task text or answer content here.
    pub item_ids: Vec<String>,
    pub item_set_sha256: String,
    pub source_lineage_manifest_sha256: String,
    pub recipe_lineage_manifest_sha256: String,
    pub adjudication_report_sha256: String,
    pub adjudication_reviewer_ids: Vec<String>,
    pub minimum_item_agreement_permille: u16,
    pub adjudication_decision: RelationAdjudicationDecisionV1,
    pub base: ScoreCountV1,
    pub treatment: ScoreCountV1,
    pub placebo: ScoreCountV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransferReplicateV1 {
    pub replicate_id: String,
    /// Domain-separated digest of the paired causal study's EvaluationEvidenceV1.
    pub evaluation_evidence_sha256: String,
    pub training_receipt_audit_sha256: String,
    pub strata: Vec<TransferStratumOutcomeV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExposureTransferStudyV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub primary_metric_id: String,
    pub primary_metric_definition_sha256: String,
    pub relation_protocol_sha256: String,
    pub evaluator_relation_manifest_sha256: String,
    pub replicates: Vec<TransferReplicateV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransferReplicateRelationEffectV1 {
    pub replicate_id: String,
    pub base_score_permille: f64,
    pub treatment_score_permille: f64,
    pub placebo_score_permille: f64,
    pub treatment_minus_placebo_permille: f64,
    pub treatment_minus_base_permille: f64,
    pub placebo_minus_base_permille: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransferFamilyRelationEffectV1 {
    pub task_family: String,
    pub difficulty_band_count: u64,
    pub item_count_per_replicate: u64,
    pub cluster_count_per_replicate: u64,
    pub base_score_permille: MeanIntervalV1,
    pub treatment_score_permille: MeanIntervalV1,
    pub placebo_score_permille: MeanIntervalV1,
    pub treatment_minus_placebo_permille: MeanIntervalV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransferRelationEffectV1 {
    pub relation: TransferRelationV1,
    pub family_count: u64,
    pub difficulty_band_count_per_replicate: u64,
    pub item_count_per_replicate: u64,
    pub cluster_count_per_replicate: u64,
    /// Every interval is across independent model-pair replicates.
    pub base_score_permille: MeanIntervalV1,
    pub treatment_score_permille: MeanIntervalV1,
    pub placebo_score_permille: MeanIntervalV1,
    pub treatment_minus_placebo_permille: MeanIntervalV1,
    pub treatment_minus_base_permille: MeanIntervalV1,
    pub placebo_minus_base_permille: MeanIntervalV1,
    pub worst_family_treatment_minus_placebo_permille: f64,
    pub family_effects: Vec<TransferFamilyRelationEffectV1>,
    pub replicate_effects: Vec<TransferReplicateRelationEffectV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransferContrastV1 {
    pub contrast_id: String,
    /// The contrast is effect(positive_relation) minus effect(reference_relation).
    pub positive_relation: TransferRelationV1,
    pub reference_relation: TransferRelationV1,
    pub treatment_minus_placebo_difference_permille: MeanIntervalV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExposureTransferReportV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub primary_metric_id: String,
    pub base_causal_study_sha256: String,
    pub transfer_study_sha256: String,
    pub relation_protocol_sha256: String,
    pub evaluator_relation_manifest_sha256: String,
    pub independent_model_pair_replicate_count: u64,
    pub replicate_count_meets_base_minimum: bool,
    pub family_count: u64,
    pub transfer_relation_effects: Vec<TransferRelationEffectV1>,
    pub predeclared_contrasts: Vec<TransferContrastV1>,
    pub contrast_intervals_are_multiplicity_adjusted: bool,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub analysis_status: String,
    pub issues: Vec<String>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Copy)]
struct RateTriplet {
    base: f64,
    treatment: f64,
    placebo: f64,
}

#[derive(Debug, Clone)]
struct FamilyRelationValues {
    rates: RateTriplet,
    item_count: u64,
    cluster_count: u64,
    difficulty_band_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CellBindingSignature {
    item_set_sha256: String,
    source_lineage_manifest_sha256: String,
    recipe_lineage_manifest_sha256: String,
    adjudication_report_sha256: String,
    item_count: u64,
    cluster_count: u64,
    adjudication_reviewer_ids: Vec<String>,
    minimum_item_agreement_permille: u16,
    adjudication_decision: RelationAdjudicationDecisionV1,
}

#[derive(Debug, Clone)]
struct PairRelationValues {
    rates: RateTriplet,
    item_count: u64,
    cluster_count: u64,
    difficulty_band_count: u64,
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

fn mean(values: &[f64], label: &str) -> Result<f64, String> {
    if values.is_empty() {
        return Err(format!("{label} cannot be estimated from an empty set"));
    }
    let result = values.iter().sum::<f64>() / values.len() as f64;
    if result.is_finite() {
        Ok(result)
    } else {
        Err(format!("{label} is non-finite"))
    }
}

fn sum_u64(values: impl IntoIterator<Item = u64>, label: &str) -> Result<u64, String> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total.checked_add(value).ok_or_else(|| format!("{label} overflows"))
    })
}

fn score_rate(score: &ScoreCountV1, expected_count: u64, label: &str) -> Result<f64, String> {
    if expected_count == 0
        || score.observed_count != expected_count
        || score.success_count > score.observed_count
    {
        return Err(format!(
            "{label} must have success_count <= observed_count = item_ids length > 0"
        ));
    }
    let value = score.success_count as f64 * 1_000.0 / score.observed_count as f64;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{label} rate is non-finite"))
    }
}

fn canonical_item_set_digest(item_ids: &[String]) -> Result<String, String> {
    if item_ids.is_empty() || item_ids.len() > MAX_ITEM_IDS_PER_ROW {
        return Err(format!(
            "item_ids length must be in 1..={MAX_ITEM_IDS_PER_ROW}"
        ));
    }
    let mut sorted = item_ids.to_vec();
    if sorted.iter().any(|item| {
        item.trim().is_empty()
            || item.len() > 128
            || !item
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    }) {
        return Err("item IDs must be 1..=128 ASCII letters/digits or . _ : -".to_owned());
    }
    sorted.sort();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("item IDs must be unique within an item-set commitment".to_owned());
    }
    let bytes = serde_json::to_vec(&sorted)
        .map_err(|error| format!("cannot serialize canonical item IDs: {error}"))?;
    Ok(hash_domain(b"exposure-transfer-item-set.v1", &bytes))
}

fn evaluation_evidence_digest(
    evidence: &crate::causal_analysis::EvaluationEvidenceV1,
) -> Result<String, String> {
    let bytes = serde_json::to_vec(evidence)
        .map_err(|error| format!("cannot serialize evaluation evidence: {error}"))?;
    Ok(hash_domain(
        b"exposure-transfer-evaluation-evidence.v1",
        &bytes,
    ))
}

fn family_band_rate_map(
    replicate: &TransferReplicateV1,
) -> Result<
    BTreeMap<(String, TransferRelationV1), FamilyRelationValues>,
    String,
> {
    let mut by_family_relation =
        BTreeMap::<(String, TransferRelationV1), Vec<(RateTriplet, u64, u64)>>::new();

    for row in &replicate.strata {
        let item_count = u64::try_from(row.item_ids.len())
            .map_err(|_| "item count exceeds u64".to_owned())?;
        let rates = RateTriplet {
            base: score_rate(&row.base, item_count, "base score")?,
            treatment: score_rate(&row.treatment, item_count, "treatment score")?,
            placebo: score_rate(&row.placebo, item_count, "placebo score")?,
        };
        by_family_relation
            .entry((row.task_family.clone(), row.relation))
            .or_default()
            .push((rates, item_count, row.cluster_count));
    }

    let mut output = BTreeMap::new();
    for (key, rows) in by_family_relation {
        let base: Vec<f64> = rows.iter().map(|row| row.0.base).collect();
        let treatment: Vec<f64> = rows.iter().map(|row| row.0.treatment).collect();
        let placebo: Vec<f64> = rows.iter().map(|row| row.0.placebo).collect();
        output.insert(
            key,
            FamilyRelationValues {
                rates: RateTriplet {
                    base: mean(&base, "family base rate")?,
                    treatment: mean(&treatment, "family treatment rate")?,
                    placebo: mean(&placebo, "family placebo rate")?,
                },
                item_count: sum_u64(rows.iter().map(|row| row.1), "family item count")?,
                cluster_count: sum_u64(rows.iter().map(|row| row.2), "family cluster count")?,
                difficulty_band_count: u64::try_from(rows.len())
                    .map_err(|_| "difficulty-band count exceeds u64".to_owned())?,
            },
        );
    }
    Ok(output)
}

fn validate_replicate_matrix(
    replicate: &TransferReplicateV1,
    expected_families: Option<&BTreeSet<String>>,
    expected_cells: &mut BTreeMap<
        (String, String, TransferRelationV1),
        CellBindingSignature,
    >,
    expected_matrix_keys: &mut Option<BTreeSet<(String, String, TransferRelationV1)>>,
) -> Result<BTreeSet<String>, String> {
    if replicate.replicate_id.trim().is_empty() || replicate.replicate_id.len() > 128 {
        return Err("transfer replicate_id must be nonempty and at most 128 bytes".to_owned());
    }
    validate_digest(
        &replicate.evaluation_evidence_sha256,
        "evaluation_evidence_sha256",
    )?;
    validate_digest(
        &replicate.training_receipt_audit_sha256,
        "training_receipt_audit_sha256",
    )?;
    if replicate.strata.is_empty()
        || replicate.strata.len() > MAX_TRANSFER_ROWS_PER_REPLICATE
    {
        return Err(format!(
            "strata length must be in 1..={MAX_TRANSFER_ROWS_PER_REPLICATE}"
        ));
    }

    let total_item_ids = replicate.strata.iter().try_fold(0_usize, |total, row| {
        total.checked_add(row.item_ids.len())
            .ok_or_else(|| "replicate item-ID count overflows".to_owned())
    })?;
    if total_item_ids > MAX_ITEM_IDS_PER_REPLICATE {
        return Err(format!(
            "replicate item-ID count exceeds {MAX_ITEM_IDS_PER_REPLICATE}"
        ));
    }

    let mut families = BTreeSet::new();
    let mut cells = BTreeMap::<(String, String), BTreeMap<TransferRelationV1, &TransferStratumOutcomeV1>>::new();
    let mut global_items = BTreeSet::new();
    let mut matrix_keys = BTreeSet::new();

    for row in &replicate.strata {
        if row.task_family.trim().is_empty()
            || row.task_family.len() > 64
            || row.difficulty_band.trim().is_empty()
            || row.difficulty_band.len() > 64
        {
            return Err("task_family and difficulty_band must be nonempty and at most 64 bytes".to_owned());
        }
        if row.cluster_count == 0 {
            return Err("cluster_count must be positive".to_owned());
        }
        let computed_item_digest = canonical_item_set_digest(&row.item_ids)?;
        validate_digest(&row.item_set_sha256, "item_set_sha256")?;
        if computed_item_digest != row.item_set_sha256 {
            return Err(format!(
                "item-set digest mismatch for {}/{}/{:?}",
                row.task_family, row.difficulty_band, row.relation
            ));
        }
        validate_digest(
            &row.source_lineage_manifest_sha256,
            "source_lineage_manifest_sha256",
        )?;
        validate_digest(
            &row.recipe_lineage_manifest_sha256,
            "recipe_lineage_manifest_sha256",
        )?;
        validate_digest(
            &row.adjudication_report_sha256,
            "adjudication_report_sha256",
        )?;
        if row.adjudication_reviewer_ids.len() < 3
            || row.adjudication_reviewer_ids.len() > 20
        {
            return Err("each relation cell requires 3..=20 independent adjudication reviewers".to_owned());
        }
        let reviewer_ids: BTreeSet<&str> = row
            .adjudication_reviewer_ids
            .iter()
            .map(String::as_str)
            .collect();
        if reviewer_ids.len() != row.adjudication_reviewer_ids.len()
            || row.adjudication_reviewer_ids.iter().any(|reviewer| {
                reviewer.trim().is_empty()
                    || reviewer.len() > 64
                    || !reviewer
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
            })
        {
            return Err("adjudication reviewer IDs must be distinct opaque ASCII identifiers".to_owned());
        }
        if row.adjudication_decision != RelationAdjudicationDecisionV1::Accepted {
            return Err("indeterminate or rejected relation adjudications cannot enter a scored stratum".to_owned());
        }
        if !(800..=1_000).contains(&row.minimum_item_agreement_permille) {
            return Err("minimum_item_agreement_permille must be in [800, 1000]".to_owned());
        }
        if matches!(
            row.relation,
            TransferRelationV1::SemanticEquivalent
                | TransferRelationV1::NovelRecipeSameSkill
        ) && row.minimum_item_agreement_permille != 1_000
        {
            return Err("semantic-equivalence and same-skill strata require unanimous per-item adjudication".to_owned());
        }
        let item_count = u64::try_from(row.item_ids.len())
            .map_err(|_| "item count exceeds u64".to_owned())?;
        score_rate(&row.base, item_count, "base score")?;
        score_rate(&row.treatment, item_count, "treatment score")?;
        score_rate(&row.placebo, item_count, "placebo score")?;

        for item_id in &row.item_ids {
            if !global_items.insert(item_id.as_str()) {
                return Err("an item ID occurs in more than one relation/family/difficulty cell".to_owned());
            }
        }

        let key = (row.task_family.clone(), row.difficulty_band.clone());
        matrix_keys.insert((row.task_family.clone(), row.difficulty_band.clone(), row.relation));
        let relation_map = cells.entry(key).or_default();
        if relation_map.insert(row.relation, row).is_some() {
            return Err("duplicate task_family/difficulty_band/relation cell".to_owned());
        }
        families.insert(row.task_family.clone());
    }

    if families.len() < MIN_TRANSFER_FAMILIES {
        return Err(format!(
            "at least {MIN_TRANSFER_FAMILIES} distinct task families are required"
        ));
    }
    if let Some(expected) = expected_families {
        if &families != expected {
            return Err("task-family set differs across independent model-pair replicates".to_owned());
        }
    }

    let mut family_bands = BTreeMap::<String, BTreeSet<String>>::new();
    for ((family, band), relation_map) in &cells {
        if relation_map.len() != TransferRelationV1::ALL.len()
            || TransferRelationV1::ALL.iter().any(|relation| !relation_map.contains_key(relation))
        {
            return Err(format!(
                "family {family} difficulty band {band} must contain all six relation strata"
            ));
        }
        let first = relation_map
            .get(&TransferRelationV1::ExactReplay)
            .ok_or_else(|| "exact-replay reference cell is missing".to_owned())?;
        let first_count = first.item_ids.len();
        let first_clusters = first.cluster_count;
        for row in relation_map.values() {
            if row.item_ids.len() != first_count || row.cluster_count != first_clusters {
                return Err(format!(
                    "all six relations in family {family} difficulty band {band} must have matched item and cluster counts"
                ));
            }
            let key = (family.clone(), band.clone(), row.relation);
            let signature = CellBindingSignature {
                item_set_sha256: row.item_set_sha256.clone(),
                source_lineage_manifest_sha256: row.source_lineage_manifest_sha256.clone(),
                recipe_lineage_manifest_sha256: row.recipe_lineage_manifest_sha256.clone(),
                adjudication_report_sha256: row.adjudication_report_sha256.clone(),
                item_count: u64::try_from(row.item_ids.len())
                    .map_err(|_| "item count exceeds u64".to_owned())?,
                cluster_count: row.cluster_count,
                adjudication_reviewer_ids: row.adjudication_reviewer_ids.clone(),
                minimum_item_agreement_permille: row.minimum_item_agreement_permille,
                adjudication_decision: row.adjudication_decision,
            };
            if let Some(previous) = expected_cells.get(&key) {
                if previous != &signature {
                    return Err("item, lineage or adjudication bindings differ across model-pair replicates".to_owned());
                }
            } else {
                expected_cells.insert(key, signature);
            }
        }
        family_bands.entry(family.clone()).or_default().insert(band.clone());
    }

    let first_bands = family_bands
        .values()
        .next()
        .ok_or_else(|| "no task-family difficulty bands were supplied".to_owned())?;
    if family_bands.values().any(|bands| bands != first_bands) {
        return Err("every task family must use the same matched difficulty-band set".to_owned());
    }
    if let Some(expected) = expected_matrix_keys.as_ref() {
        if expected != &matrix_keys {
            return Err("the complete task-family/difficulty/relation matrix must match across model-pair replicates".to_owned());
        }
    } else {
        *expected_matrix_keys = Some(matrix_keys);
    }
    Ok(families)
}

fn relation_pair_values(
    family_values: &BTreeMap<(String, TransferRelationV1), FamilyRelationValues>,
    families: &BTreeSet<String>,
    relation: TransferRelationV1,
) -> Result<PairRelationValues, String> {
    let mut base = Vec::new();
    let mut treatment = Vec::new();
    let mut placebo = Vec::new();
    let mut item_counts = Vec::new();
    let mut cluster_counts = Vec::new();
    let mut band_counts = Vec::new();
    for family in families {
        let value = family_values
            .get(&(family.clone(), relation))
            .ok_or_else(|| format!("family {family} is missing relation {relation:?}"))?;
        base.push(value.rates.base);
        treatment.push(value.rates.treatment);
        placebo.push(value.rates.placebo);
        item_counts.push(value.item_count);
        cluster_counts.push(value.cluster_count);
        band_counts.push(value.difficulty_band_count);
    }
    Ok(PairRelationValues {
        rates: RateTriplet {
            base: mean(&base, "family-macro base rate")?,
            treatment: mean(&treatment, "family-macro treatment rate")?,
            placebo: mean(&placebo, "family-macro placebo rate")?,
        },
        item_count: sum_u64(item_counts, "relation item count")?,
        cluster_count: sum_u64(cluster_counts, "relation cluster count")?,
        difficulty_band_count: sum_u64(band_counts, "relation difficulty-band count")?,
    })
}

fn relation_contrasts() -> [(&'static str, TransferRelationV1, TransferRelationV1); 6] {
    [
        (
            "exact_memorization_premium",
            TransferRelationV1::ExactReplay,
            TransferRelationV1::NovelRecipeSameSkill,
        ),
        (
            "surface_retention_premium",
            TransferRelationV1::SurfaceVariant,
            TransferRelationV1::NovelRecipeSameSkill,
        ),
        (
            "semantic_retention_premium",
            TransferRelationV1::SemanticEquivalent,
            TransferRelationV1::NovelRecipeSameSkill,
        ),
        (
            "same_recipe_transfer",
            TransferRelationV1::FreshSameRecipe,
            TransferRelationV1::UnrelatedControl,
        ),
        (
            "cross_recipe_skill_transfer",
            TransferRelationV1::NovelRecipeSameSkill,
            TransferRelationV1::UnrelatedControl,
        ),
        (
            "exact_to_same_recipe_gap",
            TransferRelationV1::ExactReplay,
            TransferRelationV1::FreshSameRecipe,
        ),
    ]
}

/// Analyze a transfer-ladder outcome file against the existing, independently
/// replicated randomized training study. The base analysis is revalidated
/// first; transfer rows must bind to the same model-pair IDs, checkpoint and
/// training-receipt evidence.
pub fn analyze_exposure_transfer(
    base_study: &ReplicatedCausalStudyV1,
    study: &ExposureTransferStudyV1,
) -> Result<ExposureTransferReportV1, String> {
    let base_report = analyze_replicated_causal_study(base_study)?;
    if study.schema != EXPOSURE_TRANSFER_STUDY_SCHEMA {
        return Err(format!(
            "study schema must be {EXPOSURE_TRANSFER_STUDY_SCHEMA}"
        ));
    }
    if study.study_id != base_study.study_id
        || study.benchmark_id != base_study.benchmark_id
        || study.benchmark_version != base_study.benchmark_version
        || study.primary_metric_id != base_study.primary_metric_id
        || study.primary_metric_definition_sha256
            != base_study.primary_metric_definition_sha256
    {
        return Err("transfer study identity and metric must exactly match the base causal study".to_owned());
    }
    validate_digest(
        &study.primary_metric_definition_sha256,
        "primary_metric_definition_sha256",
    )?;
    validate_digest(
        &study.relation_protocol_sha256,
        "relation_protocol_sha256",
    )?;
    validate_digest(
        &study.evaluator_relation_manifest_sha256,
        "evaluator_relation_manifest_sha256",
    )?;
    if study.replicates.len() < 2
        || study.replicates.len() > MAX_TRANSFER_REPLICATES
        || study.replicates.len() != base_study.replicates.len()
    {
        return Err(format!(
            "transfer replicate count must match the base study and be in 2..={MAX_TRANSFER_REPLICATES}"
        ));
    }

    let mut base_by_id = BTreeMap::new();
    for base in &base_study.replicates {
        let id = base.design.replicate_id.as_str();
        if base_by_id.insert(id, base).is_some() {
            return Err("base causal study contains duplicate model-pair replicate IDs".to_owned());
        }
    }

    let mut transfer_ids = BTreeSet::new();
    let mut expected_cells = BTreeMap::new();
    let mut expected_matrix_keys = None;
    let mut expected_families: Option<BTreeSet<String>> = None;
    let mut pair_values = BTreeMap::<
        (String, TransferRelationV1),
        PairRelationValues,
    >::new();
    let mut family_values = BTreeMap::<
        (TransferRelationV1, String),
        Vec<RateTriplet>,
    >::new();
    let mut family_cell_metadata = BTreeMap::<
        (TransferRelationV1, String),
        (u64, u64, u64),
    >::new();

    for transfer_replicate in &study.replicates {
        if !transfer_ids.insert(transfer_replicate.replicate_id.as_str()) {
            return Err("transfer model-pair replicate IDs must be unique".to_owned());
        }
        let base = base_by_id
            .get(transfer_replicate.replicate_id.as_str())
            .ok_or_else(|| "transfer replicate has no matching base causal replicate".to_owned())?;
        if transfer_replicate.training_receipt_audit_sha256
            != base.training_receipt_audit_sha256
        {
            return Err("transfer replicate training receipt does not match the base causal replicate".to_owned());
        }
        let expected_evidence = evaluation_evidence_digest(&base.evaluation_evidence)?;
        if transfer_replicate.evaluation_evidence_sha256 != expected_evidence {
            return Err("transfer evaluation evidence digest does not match base checkpoint/evaluator bindings".to_owned());
        }

        let families = validate_replicate_matrix(
            transfer_replicate,
            expected_families.as_ref(),
            &mut expected_cells,
            &mut expected_matrix_keys,
        )?;
        if let Some(existing) = &expected_families {
            if existing != &families {
                return Err("task-family universe differs across model-pair replicates".to_owned());
            }
        } else {
            expected_families = Some(families.clone());
        }

        let family_rates = family_band_rate_map(transfer_replicate)?;
        for ((family, relation), value) in &family_rates {
            family_values
                .entry((*relation, family.clone()))
                .or_default()
                .push(value.rates);
            let metadata = (
                value.difficulty_band_count,
                value.item_count,
                value.cluster_count,
            );
            if let Some(previous) = family_cell_metadata.get(&(*relation, family.clone())) {
                if previous != &metadata {
                    return Err("family/relation counts differ across independent model-pair replicates".to_owned());
                }
            } else {
                family_cell_metadata.insert((*relation, family.clone()), metadata);
            }
        }

        let families = expected_families
            .as_ref()
            .ok_or_else(|| "no task families were supplied".to_owned())?;
        for relation in TransferRelationV1::ALL {
            let value = relation_pair_values(&family_rates, families, relation)?;
            pair_values.insert(
                (transfer_replicate.replicate_id.clone(), relation),
                value,
            );
        }
    }

    if transfer_ids.len() != base_by_id.len() {
        return Err("transfer study must contain exactly one row for every base model-pair replicate".to_owned());
    }

    let families = expected_families
        .ok_or_else(|| "no task families were supplied".to_owned())?;
    let mut relation_reports = Vec::new();
    let mut replicate_by_relation =
        BTreeMap::<TransferRelationV1, Vec<TransferReplicateRelationEffectV1>>::new();

    for relation in TransferRelationV1::ALL {
        let mut base_values = Vec::new();
        let mut treatment_values = Vec::new();
        let mut placebo_values = Vec::new();
        let mut treatment_placebo_values = Vec::new();
        let mut treatment_base_values = Vec::new();
        let mut placebo_base_values = Vec::new();
        let mut relation_item_count = None;
        let mut relation_cluster_count = None;
        let mut relation_band_count = None;
        let mut replicate_effects = Vec::new();

        for transfer_replicate in &study.replicates {
            let value = pair_values
                .get(&(transfer_replicate.replicate_id.clone(), relation))
                .ok_or_else(|| "missing relation summary for model-pair replicate".to_owned())?;
            let base = value.rates.base;
            let treatment = value.rates.treatment;
            let placebo = value.rates.placebo;
            base_values.push(base);
            treatment_values.push(treatment);
            placebo_values.push(placebo);
            treatment_placebo_values.push(treatment - placebo);
            treatment_base_values.push(treatment - base);
            placebo_base_values.push(placebo - base);
            relation_item_count.get_or_insert(value.item_count);
            relation_cluster_count.get_or_insert(value.cluster_count);
            relation_band_count.get_or_insert(value.difficulty_band_count);
            if relation_item_count != Some(value.item_count)
                || relation_cluster_count != Some(value.cluster_count)
                || relation_band_count != Some(value.difficulty_band_count)
            {
                return Err("relation totals differ across independent model-pair replicates".to_owned());
            }
            replicate_effects.push(TransferReplicateRelationEffectV1 {
                replicate_id: transfer_replicate.replicate_id.clone(),
                base_score_permille: base,
                treatment_score_permille: treatment,
                placebo_score_permille: placebo,
                treatment_minus_placebo_permille: treatment - placebo,
                treatment_minus_base_permille: treatment - base,
                placebo_minus_base_permille: placebo - base,
            });
        }

        replicate_by_relation.insert(relation, replicate_effects.clone());

        let mut family_effects = Vec::new();
        let mut family_means = Vec::new();
        for family in &families {
            let values = family_values
                .get(&(relation, family.clone()))
                .ok_or_else(|| format!("family {family} is missing relation {relation:?}"))?;
            let family_base: Vec<f64> = values.iter().map(|value| value.base).collect();
            let family_treatment: Vec<f64> = values.iter().map(|value| value.treatment).collect();
            let family_placebo: Vec<f64> = values.iter().map(|value| value.placebo).collect();
            let family_effect: Vec<f64> = values
                .iter()
                .map(|value| value.treatment - value.placebo)
                .collect();
            let metadata = family_cell_metadata
                .get(&(relation, family.clone()))
                .ok_or_else(|| "family/relation metadata is missing".to_owned())?;
            let effect_interval = mean_interval(&family_effect)?;
            family_means.push(effect_interval.mean);
            family_effects.push(TransferFamilyRelationEffectV1 {
                task_family: family.clone(),
                difficulty_band_count: metadata.0,
                item_count_per_replicate: metadata.1,
                cluster_count_per_replicate: metadata.2,
                base_score_permille: mean_interval(&family_base)?,
                treatment_score_permille: mean_interval(&family_treatment)?,
                placebo_score_permille: mean_interval(&family_placebo)?,
                treatment_minus_placebo_permille: effect_interval,
            });
        }

        relation_reports.push(TransferRelationEffectV1 {
            relation,
            family_count: u64::try_from(families.len())
                .map_err(|_| "family count exceeds u64".to_owned())?,
            difficulty_band_count_per_replicate: relation_band_count
                .ok_or_else(|| "relation difficulty-band count is missing".to_owned())?,
            item_count_per_replicate: relation_item_count
                .ok_or_else(|| "relation item count is missing".to_owned())?,
            cluster_count_per_replicate: relation_cluster_count
                .ok_or_else(|| "relation cluster count is missing".to_owned())?,
            base_score_permille: mean_interval(&base_values)?,
            treatment_score_permille: mean_interval(&treatment_values)?,
            placebo_score_permille: mean_interval(&placebo_values)?,
            treatment_minus_placebo_permille: mean_interval(&treatment_placebo_values)?,
            treatment_minus_base_permille: mean_interval(&treatment_base_values)?,
            placebo_minus_base_permille: mean_interval(&placebo_base_values)?,
            worst_family_treatment_minus_placebo_permille: family_means
                .into_iter()
                .fold(f64::INFINITY, f64::min),
            family_effects,
            replicate_effects,
        });
    }

    let mut predeclared_contrasts = Vec::new();
    for (contrast_id, positive, reference) in relation_contrasts() {
        let positive_effects = replicate_by_relation
            .get(&positive)
            .ok_or_else(|| "positive relation contrast is missing".to_owned())?;
        let reference_effects = replicate_by_relation
            .get(&reference)
            .ok_or_else(|| "reference relation contrast is missing".to_owned())?;
        if positive_effects.len() != reference_effects.len() {
            return Err("contrast arms have different replicate counts".to_owned());
        }
        let mut differences = Vec::with_capacity(positive_effects.len());
        for (left, right) in positive_effects.iter().zip(reference_effects) {
            if left.replicate_id != right.replicate_id {
                return Err("contrast arms are not paired by model-pair replicate ID".to_owned());
            }
            differences.push(
                left.treatment_minus_placebo_permille
                    - right.treatment_minus_placebo_permille,
            );
        }
        predeclared_contrasts.push(TransferContrastV1 {
            contrast_id: contrast_id.to_owned(),
            positive_relation: positive,
            reference_relation: reference,
            treatment_minus_placebo_difference_permille: mean_interval(&differences)?,
        });
    }

    let transfer_bytes = serde_json::to_vec(study)
        .map_err(|error| format!("cannot serialize transfer study for digest: {error}"))?;
    let transfer_study_sha256 =
        hash_domain(b"exposure-transfer-study.v1", &transfer_bytes);
    let issues = vec![
        "training and checkpoint receipts are content-bound but this analyzer has no independent trust-root verifier".to_owned(),
        "semantic-equivalence and same-skill labels require external blinded expert adjudication".to_owned(),
        "relation contrasts are descriptive predeclared intervals; no multiplicity adjustment is applied".to_owned(),
        "global model updates can spill over across all task strata; stratum contrasts are not pure direct effects under unrestricted interference".to_owned(),
    ];

    Ok(ExposureTransferReportV1 {
        schema: EXPOSURE_TRANSFER_REPORT_SCHEMA.to_owned(),
        study_id: study.study_id.clone(),
        benchmark_id: study.benchmark_id.clone(),
        benchmark_version: study.benchmark_version.clone(),
        primary_metric_id: study.primary_metric_id.clone(),
        base_causal_study_sha256: base_report.study_sha256,
        transfer_study_sha256,
        relation_protocol_sha256: study.relation_protocol_sha256.clone(),
        evaluator_relation_manifest_sha256: study.evaluator_relation_manifest_sha256.clone(),
        independent_model_pair_replicate_count: study.replicates.len() as u64,
        replicate_count_meets_base_minimum: study.replicates.len()
            >= base_study.minimum_confirmatory_replicates as usize,
        family_count: families.len() as u64,
        transfer_relation_effects: relation_reports,
        predeclared_contrasts,
        contrast_intervals_are_multiplicity_adjusted: false,
        independently_authenticated: false,
        confirmatory_eligible: false,
        analysis_status: "exploratory_only_unqualified".to_owned(),
        issues,
        claim_boundary: "This report decomposes observed treatment/placebo score differences across evaluator-declared task-relation strata. It does not authenticate exposure, checkpoint identity, evaluator custody, semantic equivalence or skill mapping; it does not establish that training exposure caused a specific item's answer; it does not prove absence from training; and it does not establish FDR control, arbitrary-interference direct effects, external validity or SOTA superiority. Uncertainty is across independent model-pair training replicates, not questions.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> String {
        value.to_owned()
    }

    #[test]
    fn canonical_item_digest_is_order_independent() {
        let first = vec![id("item-b"), id("item-a")];
        let second = vec![id("item-a"), id("item-b")];
        assert_eq!(
            canonical_item_set_digest(&first).unwrap(),
            canonical_item_set_digest(&second).unwrap()
        );
    }

    #[test]
    fn canonical_item_digest_rejects_duplicate_and_blank_ids() {
        assert!(canonical_item_set_digest(&[id("same"), id("same")]).is_err());
        assert!(canonical_item_set_digest(&[id(" ")]).is_err());
        assert!(canonical_item_set_digest(&[]).is_err());
    }

    #[test]
    fn score_counts_require_exact_item_denominator() {
        let valid = ScoreCountV1 {
            success_count: 7,
            observed_count: 10,
        };
        assert_eq!(score_rate(&valid, 10, "test").unwrap(), 700.0);
        assert!(score_rate(&valid, 9, "test").is_err());
        assert!(score_rate(
            &ScoreCountV1 {
                success_count: 11,
                observed_count: 10,
            },
            10,
            "test"
        )
        .is_err());
    }

    #[test]
    fn transfer_ladder_contains_six_distinct_relations() {
        let unique: BTreeSet<_> = TransferRelationV1::ALL.into_iter().collect();
        assert_eq!(unique.len(), 6);
    }

    fn sample_transfer_replicate() -> TransferReplicateV1 {
        let mut strata = Vec::new();
        for family in ["mathematics", "code"] {
            for relation in TransferRelationV1::ALL {
                let item_ids = vec![format!("{family}_{relation:?}")];
                let item_set_sha256 = canonical_item_set_digest(&item_ids).unwrap();
                let score = ScoreCountV1 {
                    success_count: 1,
                    observed_count: 1,
                };
                strata.push(TransferStratumOutcomeV1 {
                    task_family: family.to_owned(),
                    difficulty_band: "medium".to_owned(),
                    relation,
                    cluster_count: 1,
                    item_ids,
                    item_set_sha256,
                    source_lineage_manifest_sha256: "a".repeat(64),
                    recipe_lineage_manifest_sha256: "b".repeat(64),
                    adjudication_report_sha256: "c".repeat(64),
                    adjudication_reviewer_ids: vec![
                        "rater-a".to_owned(),
                        "rater-b".to_owned(),
                        "rater-c".to_owned(),
                    ],
                    minimum_item_agreement_permille: 1_000,
                    adjudication_decision: RelationAdjudicationDecisionV1::Accepted,
                    base: score.clone(),
                    treatment: score.clone(),
                    placebo: score,
                });
            }
        }
        TransferReplicateV1 {
            replicate_id: "pair-001".to_owned(),
            evaluation_evidence_sha256: "d".repeat(64),
            training_receipt_audit_sha256: "e".repeat(64),
            strata,
        }
    }

    #[test]
    fn complete_six_relation_matrix_is_accepted() {
        let replicate = sample_transfer_replicate();
        let mut expected_cells = BTreeMap::new();
        let mut expected_matrix_keys = None;
        let families = validate_replicate_matrix(
            &replicate,
            None,
            &mut expected_cells,
            &mut expected_matrix_keys,
        )
        .unwrap();
        assert_eq!(families.len(), 2);
        assert_eq!(expected_matrix_keys.unwrap().len(), 12);
    }

    #[test]
    fn incomplete_relation_matrix_is_rejected() {
        let mut replicate = sample_transfer_replicate();
        replicate.strata.pop();
        let mut expected_cells = BTreeMap::new();
        let mut expected_matrix_keys = None;
        assert!(validate_replicate_matrix(
            &replicate,
            None,
            &mut expected_cells,
            &mut expected_matrix_keys,
        )
        .is_err());
    }

    #[test]
    fn duplicate_item_across_relation_strata_is_rejected() {
        let mut replicate = sample_transfer_replicate();
        replicate.strata[1].item_ids = replicate.strata[0].item_ids.clone();
        replicate.strata[1].item_set_sha256 =
            canonical_item_set_digest(&replicate.strata[1].item_ids).unwrap();
        let mut expected_cells = BTreeMap::new();
        let mut expected_matrix_keys = None;
        assert!(validate_replicate_matrix(
            &replicate,
            None,
            &mut expected_cells,
            &mut expected_matrix_keys,
        )
        .is_err());
    }

    #[test]
    fn cell_matrix_must_match_across_replicates() {
        let first = sample_transfer_replicate();
        let mut second = sample_transfer_replicate();
        second.replicate_id = "pair-002".to_owned();
        for row in &mut second.strata {
            row.difficulty_band = "hard".to_owned();
        }
        let mut expected_cells = BTreeMap::new();
        let mut expected_matrix_keys = None;
        validate_replicate_matrix(
            &first,
            None,
            &mut expected_cells,
            &mut expected_matrix_keys,
        )
        .unwrap();
        assert!(validate_replicate_matrix(
            &second,
            None,
            &mut expected_cells,
            &mut expected_matrix_keys,
        )
        .is_err());
    }

    #[test]
    fn treatment_placebo_contrast_direction_is_explicit() {
        let high = RateTriplet {
            base: 500.0,
            treatment: 800.0,
            placebo: 600.0,
        };
        let low = RateTriplet {
            base: 500.0,
            treatment: 550.0,
            placebo: 600.0,
        };
        let difference = high.treatment - high.placebo
            - (low.treatment - low.placebo);
        assert_eq!(difference, 250.0);
    }
}
