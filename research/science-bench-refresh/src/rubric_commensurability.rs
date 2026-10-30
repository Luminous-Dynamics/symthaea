//! Structural commensurability audit for distinct rubric measurement instruments.
//!
//! This module asks whether two rubric instruments define comparable constructs,
//! scales and weights before their scores or model rankings are compared. It does
//! not infer equivalence from wording, validate construct truth, fit a latent
//! variable model, establish human alignment or grant scoring eligibility.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const RUBRIC_COMMENSURABILITY_PLAN_SCHEMA: &str =
    "symthaea.science_bench_refresh_rubric_commensurability_plan.v1";
pub const RUBRIC_COMMENSURABILITY_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_rubric_commensurability_report.v1";

const PLAN_DOMAIN: &[u8] = b"symthaea.rubric_commensurability.plan.v1\0";
const REPORT_DOMAIN: &[u8] = b"symthaea.rubric_commensurability.report.v1\0";
const MAX_CONSTRUCTS: usize = 256;
const MAX_CRITERIA: usize = 512;
const MAX_SCALE_POINTS: usize = 64;
const MAX_ANCHORS: usize = 100_000;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CommensurabilityPurposeV1 {
    ExampleOnly,
    QualifiedRun,
}

impl CommensurabilityPurposeV1 {
    fn qualified(self) -> bool {
        self == Self::QualifiedRun
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScoreDirectionV1 {
    HigherRawIsBetter,
    LowerRawIsBetter,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConstructRelationV1 {
    Equivalent,
    NarrowerThan,
    BroaderThan,
    Overlaps,
    Orthogonal,
    Incompatible,
    Unmapped,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConstructCommensurabilityV1 {
    DirectlyCommensurable,
    CalibrationRequired,
    PartiallyMapped,
    Incompatible,
    Unmapped,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OverallCommensurabilityV1 {
    FullyCommensurable,
    PartiallyCommensurable,
    CalibrationRequired,
    NonCommensurable,
    InsufficientEvidence,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConstructDefinitionV1 {
    pub construct_id: String,
    pub definition_sha256: String,
    pub scope_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScaleAnchorV1 {
    pub raw_value: i64,
    pub normalized_score_millionths: u32,
    pub semantic_anchor_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CriterionConstructMappingV1 {
    pub construct_id: Option<String>,
    pub relation: ConstructRelationV1,
    pub mapping_evidence_sha256: String,
    pub adjudicator_id_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RubricCriterionV1 {
    pub criterion_id: String,
    pub definition_sha256: String,
    /// Commits to the criterion's intended construct, scope, threshold semantics
    /// and scoring interpretation, not merely its display name.
    pub measurement_spec_sha256: String,
    pub score_min_raw: i64,
    pub score_max_raw: i64,
    pub score_direction: ScoreDirectionV1,
    pub scale_anchors: Vec<ScaleAnchorV1>,
    pub weight_millionths: u32,
    pub construct_mapping: CriterionConstructMappingV1,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RubricInstrumentV1 {
    pub rubric_id: String,
    pub rubric_revision: String,
    pub rubric_sha256: String,
    pub criteria: Vec<RubricCriterionV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SharedAnchorItemV1 {
    pub anchor_id: String,
    pub construct_id: String,
    pub semantics_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RubricCommensurabilityPlanFieldsV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: CommensurabilityPurposeV1,
    pub comparison_id: String,
    pub construct_ontology_sha256: String,
    pub constructs: Vec<ConstructDefinitionV1>,
    pub left_instrument: RubricInstrumentV1,
    pub right_instrument: RubricInstrumentV1,
    pub shared_anchor_items: Vec<SharedAnchorItemV1>,
    pub minimum_anchor_items_per_construct: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RubricCommensurabilityPlanV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: CommensurabilityPurposeV1,
    pub comparison_id: String,
    pub construct_ontology_sha256: String,
    pub constructs: Vec<ConstructDefinitionV1>,
    pub left_instrument: RubricInstrumentV1,
    pub right_instrument: RubricInstrumentV1,
    pub shared_anchor_items: Vec<SharedAnchorItemV1>,
    pub minimum_anchor_items_per_construct: u32,
    pub plan_commitment_sha256: String,
}

impl RubricCommensurabilityPlanV1 {
    fn fields(&self) -> RubricCommensurabilityPlanFieldsV1 {
        RubricCommensurabilityPlanFieldsV1 {
            schema: self.schema.clone(),
            analyzer_source_revision: self.analyzer_source_revision.clone(),
            purpose: self.purpose,
            comparison_id: self.comparison_id.clone(),
            construct_ontology_sha256: self.construct_ontology_sha256.clone(),
            constructs: self.constructs.clone(),
            left_instrument: self.left_instrument.clone(),
            right_instrument: self.right_instrument.clone(),
            shared_anchor_items: self.shared_anchor_items.clone(),
            minimum_anchor_items_per_construct: self.minimum_anchor_items_per_construct,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ConstructCommensurabilityResultV1 {
    pub construct_id: String,
    pub status: ConstructCommensurabilityV1,
    pub left_criterion_ids: Vec<String>,
    pub right_criterion_ids: Vec<String>,
    pub left_relations: Vec<ConstructRelationV1>,
    pub right_relations: Vec<ConstructRelationV1>,
    pub left_weight_millionths: u64,
    pub right_weight_millionths: u64,
    pub weight_difference_millionths: i64,
    pub left_measurement_spec_sha256: Vec<String>,
    pub right_measurement_spec_sha256: Vec<String>,
    pub exact_measurement_spec_match: bool,
    pub normalized_scale_anchor_match: bool,
    pub shared_anchor_item_count: u64,
    pub required_anchor_item_count: u64,
    pub direct_score_comparison_allowed: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RubricCommensurabilityReportFieldsV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub analyzer_worktree_clean: bool,
    pub comparison_id: String,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub left_rubric_id: String,
    pub right_rubric_id: String,
    pub overall_status: OverallCommensurabilityV1,
    pub construct_count: u64,
    pub directly_commensurable_construct_count: u64,
    pub calibration_required_construct_count: u64,
    pub partially_mapped_construct_count: u64,
    pub incompatible_construct_count: u64,
    pub unmapped_construct_count: u64,
    pub insufficient_anchor_construct_count: u64,
    pub left_criterion_count: u64,
    pub right_criterion_count: u64,
    pub left_unmapped_criterion_count: u64,
    pub right_unmapped_criterion_count: u64,
    pub left_weight_total_millionths: u64,
    pub right_weight_total_millionths: u64,
    pub left_right_mapping_adjudicator_overlap_count: u64,
    pub mapping_adjudicator_ids_disjoint: bool,
    pub mapping_panel_independence_proven: bool,
    pub shared_anchor_item_count: u64,
    pub minimum_anchor_items_per_construct: u32,
    pub direct_score_comparison_allowed: bool,
    pub model_ranking_comparison_allowed: bool,
    pub construct_validity_proven: bool,
    pub human_alignment_proven: bool,
    pub generalized_accuracy_estimated: bool,
    pub scoring_eligible_declared: bool,
    pub external_anchor_required: bool,
    pub construct_results: Vec<ConstructCommensurabilityResultV1>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RubricCommensurabilityReportV1 {
    #[serde(flatten)]
    pub fields: RubricCommensurabilityReportFieldsV1,
    pub report_commitment_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn domain_digest(domain: &[u8], payload: &[u8]) -> Result<String, String> {
    let length = u64::try_from(payload.len())
        .map_err(|_| "domain-separated payload exceeds u64 length".to_owned())?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(length.to_be_bytes());
    hasher.update(payload);
    Ok(sha256_hex(&hasher.finalize()))
}

fn valid_digest(value: &str, allow_zero: bool) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && (allow_zero || value.bytes().any(|byte| byte != b'0'))
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

fn nonempty(value: &str, max: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max
}

fn validate_scale(criterion: &RubricCriterionV1, allow_zero: bool, label: &str) -> Vec<String> {
    let mut issues = Vec::new();
    if criterion.score_min_raw >= criterion.score_max_raw {
        issues.push(format!("{label} requires score_min_raw < score_max_raw"));
    }
    if criterion.scale_anchors.len() < 2 || criterion.scale_anchors.len() > MAX_SCALE_POINTS {
        issues.push(format!("{label} must contain 2..={MAX_SCALE_POINTS} scale anchors"));
        return issues;
    }
    let mut raw_values = BTreeSet::new();
    let mut semantic_digests = BTreeSet::new();
    let mut previous_raw: Option<i64> = None;
    let mut previous_normalized: Option<u32> = None;
    for anchor in &criterion.scale_anchors {
        if anchor.raw_value < criterion.score_min_raw || anchor.raw_value > criterion.score_max_raw {
            issues.push(format!("{label} scale anchor raw value is outside its declared range"));
        }
        if anchor.normalized_score_millionths > 1_000_000 {
            issues.push(format!("{label} normalized scale anchor exceeds 1,000,000"));
        }
        if !valid_digest(&anchor.semantic_anchor_sha256, allow_zero) {
            issues.push(format!("{label} has an invalid semantic scale-anchor digest"));
        }
        if !raw_values.insert(anchor.raw_value) || !semantic_digests.insert(anchor.semantic_anchor_sha256.as_str()) {
            issues.push(format!("{label} has duplicate raw values or semantic scale anchors"));
        }
        if let Some(previous) = previous_raw {
            if anchor.raw_value <= previous {
                issues.push(format!("{label} scale anchors must be ordered by strictly increasing raw value"));
            }
        }
        if let Some(previous) = previous_normalized {
            let monotonic = match criterion.score_direction {
                ScoreDirectionV1::HigherRawIsBetter => anchor.normalized_score_millionths > previous,
                ScoreDirectionV1::LowerRawIsBetter => anchor.normalized_score_millionths < previous,
            };
            if !monotonic {
                issues.push(format!("{label} normalized scores are not strictly monotonic with score_direction"));
            }
        }
        previous_raw = Some(anchor.raw_value);
        previous_normalized = Some(anchor.normalized_score_millionths);
    }
    let first = criterion.scale_anchors.first().unwrap();
    let last = criterion.scale_anchors.last().unwrap();
    if first.raw_value != criterion.score_min_raw || last.raw_value != criterion.score_max_raw {
        issues.push(format!("{label} scale anchors must cover both declared raw endpoints"));
    }
    match criterion.score_direction {
        ScoreDirectionV1::HigherRawIsBetter => {
            if first.normalized_score_millionths != 0 || last.normalized_score_millionths != 1_000_000 {
                issues.push(format!("{label} higher-is-better scale must map raw min/max to normalized 0/1,000,000"));
            }
        }
        ScoreDirectionV1::LowerRawIsBetter => {
            if first.normalized_score_millionths != 1_000_000 || last.normalized_score_millionths != 0 {
                issues.push(format!("{label} lower-is-better scale must map raw min/max to normalized 1,000,000/0"));
            }
        }
    }
    issues
}

fn validate_instrument(
    instrument: &RubricInstrumentV1,
    construct_ids: &BTreeSet<&str>,
    allow_zero: bool,
    label: &str,
) -> Vec<String> {
    let mut issues = Vec::new();
    if !nonempty(&instrument.rubric_id, 128) || !nonempty(&instrument.rubric_revision, 128) {
        issues.push(format!("{label} rubric identity/revision must be non-empty and bounded"));
    }
    if !valid_digest(&instrument.rubric_sha256, allow_zero) {
        issues.push(format!("{label} has an invalid rubric_sha256"));
    }
    if instrument.criteria.is_empty() || instrument.criteria.len() > MAX_CRITERIA {
        issues.push(format!("{label} criteria must contain 1..={MAX_CRITERIA} rows"));
    }
    let mut criterion_ids = BTreeSet::new();
    let mut weight_total = 0_u64;
    for criterion in &instrument.criteria {
        let criterion_label = format!("{label}.{}", criterion.criterion_id);
        if !nonempty(&criterion.criterion_id, 128) || !criterion_ids.insert(criterion.criterion_id.as_str()) {
            issues.push(format!("{criterion_label} criterion ID must be non-empty, bounded and unique"));
        }
        if !valid_digest(&criterion.definition_sha256, allow_zero)
            || !valid_digest(&criterion.measurement_spec_sha256, allow_zero)
        {
            issues.push(format!("{criterion_label} definition/measurement digest is invalid"));
        }
        weight_total = match weight_total.checked_add(u64::from(criterion.weight_millionths)) {
            Some(total) => total,
            None => {
                issues.push(format!("{label} criterion weight sum overflow"));
                u64::MAX
            }
        };
        if criterion.weight_millionths == 0 || criterion.weight_millionths > 1_000_000 {
            issues.push(format!("{criterion_label} weight must be in 1..=1,000,000"));
        }
        issues.extend(validate_scale(criterion, allow_zero, &criterion_label));
        let mapping = &criterion.construct_mapping;
        if !valid_digest(&mapping.mapping_evidence_sha256, allow_zero)
            || !valid_digest(&mapping.adjudicator_id_sha256, allow_zero)
        {
            issues.push(format!("{criterion_label} requires valid mapping evidence and adjudicator digests"));
        }
        match (&mapping.construct_id, mapping.relation) {
            (Some(id), relation) => {
                if !construct_ids.contains(id.as_str()) {
                    issues.push(format!("{criterion_label} maps to an unknown construct"));
                }
                if relation == ConstructRelationV1::Unmapped {
                    issues.push(format!("{criterion_label} has a construct ID but is marked unmapped"));
                }
            }
            (None, ConstructRelationV1::Unmapped) => {}
            (None, _) => issues.push(format!("{criterion_label} requires a construct ID unless relation is unmapped")),
        }
    }
    if weight_total != 1_000_000 {
        issues.push(format!("{label} criterion weights must sum exactly to 1,000,000 (found {weight_total})"));
    }
    issues
}

pub fn validate_rubric_commensurability_plan_fields(
    fields: &RubricCommensurabilityPlanFieldsV1,
) -> Result<(), Vec<String>> {
    let mut issues = Vec::new();
    let allow_zero = !fields.purpose.qualified();
    if fields.schema != RUBRIC_COMMENSURABILITY_PLAN_SCHEMA {
        issues.push("unsupported rubric commensurability plan schema".to_owned());
    }
    if !valid_revision(&fields.analyzer_source_revision)
        || fields.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
    {
        issues.push("analyzer_source_revision must match the exact compiled validator revision".to_owned());
    }
    if fields.purpose.qualified() && crate::BUILT_GENERATOR_WORKTREE_CLEAN != "true" {
        issues.push("qualified commensurability plans require a clean analyzer worktree".to_owned());
    }
    if !nonempty(&fields.comparison_id, 128) {
        issues.push("comparison_id must be non-empty and bounded".to_owned());
    }
    if !valid_digest(&fields.construct_ontology_sha256, allow_zero) {
        issues.push("construct_ontology_sha256 is invalid".to_owned());
    }
    if fields.constructs.is_empty() || fields.constructs.len() > MAX_CONSTRUCTS {
        issues.push(format!("constructs must contain 1..={MAX_CONSTRUCTS} rows"));
    }
    if fields.shared_anchor_items.len() > MAX_ANCHORS {
        issues.push(format!("shared_anchor_items exceeds {MAX_ANCHORS}"));
    }
    if fields.minimum_anchor_items_per_construct == 0 {
        issues.push("minimum_anchor_items_per_construct must be positive".to_owned());
    }

    let mut construct_ids = BTreeSet::new();
    for construct in &fields.constructs {
        if !nonempty(&construct.construct_id, 128) || !construct_ids.insert(construct.construct_id.as_str()) {
            issues.push("construct IDs must be non-empty, bounded and unique".to_owned());
        }
        if !valid_digest(&construct.definition_sha256, allow_zero)
            || !valid_digest(&construct.scope_sha256, allow_zero)
        {
            issues.push(format!("construct {} has invalid definition/scope digest", construct.construct_id));
        }
    }
    let construct_id_set: BTreeSet<&str> = construct_ids.iter().copied().collect();
    issues.extend(validate_instrument(&fields.left_instrument, &construct_id_set, allow_zero, "left_instrument"));
    issues.extend(validate_instrument(&fields.right_instrument, &construct_id_set, allow_zero, "right_instrument"));
    if fields.left_instrument.rubric_id == fields.right_instrument.rubric_id {
        issues.push("left and right rubric instruments must have distinct rubric IDs".to_owned());
    }

    let mut anchor_ids = BTreeSet::new();
    let mut anchor_semantics = BTreeSet::new();
    for anchor in &fields.shared_anchor_items {
        if !nonempty(&anchor.anchor_id, 128) || !anchor_ids.insert(anchor.anchor_id.as_str()) {
            issues.push("shared anchor IDs must be non-empty, bounded and unique".to_owned());
        }
        if !construct_id_set.contains(anchor.construct_id.as_str()) {
            issues.push(format!("anchor {} references an unknown construct", anchor.anchor_id));
        }
        if !valid_digest(&anchor.semantics_sha256, allow_zero) {
            issues.push(format!("anchor {} has invalid semantics digest", anchor.anchor_id));
        }
        if !anchor_semantics.insert((anchor.construct_id.as_str(), anchor.semantics_sha256.as_str())) {
            issues.push(format!("anchor {} duplicates semantic content for its construct", anchor.anchor_id));
        }
    }
    if issues.is_empty() { Ok(()) } else { Err(issues) }
}

fn canonical_anchor_map(criterion: &RubricCriterionV1) -> Vec<(String, u32)> {
    let mut result: Vec<(String, u32)> = criterion.scale_anchors.iter()
        .map(|anchor| (anchor.semantic_anchor_sha256.clone(), anchor.normalized_score_millionths))
        .collect();
    result.sort();
    result
}

fn instrument_construct_map<'a>(
    instrument: &'a RubricInstrumentV1,
) -> BTreeMap<&'a str, Vec<&'a RubricCriterionV1>> {
    let mut result: BTreeMap<&str, Vec<&RubricCriterionV1>> = BTreeMap::new();
    for criterion in &instrument.criteria {
        if let Some(construct_id) = criterion.construct_mapping.construct_id.as_deref() {
            result.entry(construct_id).or_default().push(criterion);
        }
    }
    result
}

fn criterion_ids(criteria: &[&RubricCriterionV1]) -> Vec<String> {
    let mut ids: Vec<String> = criteria.iter().map(|criterion| criterion.criterion_id.clone()).collect();
    ids.sort();
    ids
}

fn relation_list(criteria: &[&RubricCriterionV1]) -> Vec<ConstructRelationV1> {
    criteria.iter().map(|criterion| criterion.construct_mapping.relation).collect()
}

fn score_weight(criteria: &[&RubricCriterionV1]) -> u64 {
    criteria.iter().map(|criterion| u64::from(criterion.weight_millionths)).sum()
}

fn measurement_digests(criteria: &[&RubricCriterionV1]) -> Vec<String> {
    let mut digests: Vec<String> = criteria.iter().map(|criterion| criterion.measurement_spec_sha256.clone()).collect();
    digests.sort();
    digests
}

fn shared_anchors_by_construct(
    fields: &RubricCommensurabilityPlanFieldsV1,
) -> BTreeMap<&str, u64> {
    let mut counts = BTreeMap::new();
    for anchor in &fields.shared_anchor_items {
        *counts.entry(anchor.construct_id.as_str()).or_insert(0_u64) += 1;
    }
    counts
}

fn construct_result(
    construct: &ConstructDefinitionV1,
    left: &[&RubricCriterionV1],
    right: &[&RubricCriterionV1],
    anchor_count: u64,
    required_anchors: u64,
    mapping_adjudicators_disjoint: bool,
) -> ConstructCommensurabilityResultV1 {
    let left_relations = relation_list(left);
    let right_relations = relation_list(right);
    let left_measurement = measurement_digests(left);
    let right_measurement = measurement_digests(right);
    let exact_measurement_spec_match = !left_measurement.is_empty()
        && left_measurement == right_measurement
        && left.len() == right.len();
    let mut left_scales: Vec<Vec<(String, u32)>> = left.iter().map(|criterion| canonical_anchor_map(criterion)).collect();
    let mut right_scales: Vec<Vec<(String, u32)>> = right.iter().map(|criterion| canonical_anchor_map(criterion)).collect();
    left_scales.sort();
    right_scales.sort();
    let normalized_scale_anchor_match = !left_scales.is_empty()
        && left_scales == right_scales
        && left.len() == right.len();
    let left_weight = score_weight(left);
    let right_weight = score_weight(right);
    let weight_difference = i64::try_from(right_weight).unwrap_or(i64::MAX)
        - i64::try_from(left_weight).unwrap_or(i64::MAX);
    let all_equivalent = !left.is_empty()
        && !right.is_empty()
        && left_relations.iter().all(|relation| *relation == ConstructRelationV1::Equivalent)
        && right_relations.iter().all(|relation| *relation == ConstructRelationV1::Equivalent);

    let mut reasons = Vec::new();
    let status = if left.is_empty() || right.is_empty() {
        reasons.push("one rubric has no criterion mapped to this construct".to_owned());
        ConstructCommensurabilityV1::Unmapped
    } else if left_relations.iter().any(|relation| *relation == ConstructRelationV1::Incompatible)
        || right_relations.iter().any(|relation| *relation == ConstructRelationV1::Incompatible)
        || left_relations.iter().any(|relation| *relation == ConstructRelationV1::Orthogonal)
        || right_relations.iter().any(|relation| *relation == ConstructRelationV1::Orthogonal)
    {
        reasons.push("at least one adjudicated mapping is incompatible or orthogonal".to_owned());
        ConstructCommensurabilityV1::Incompatible
    } else if all_equivalent && exact_measurement_spec_match && normalized_scale_anchor_match
        && left_weight == right_weight && anchor_count >= required_anchors
        && mapping_adjudicators_disjoint
    {
        ConstructCommensurabilityV1::DirectlyCommensurable
    } else if all_equivalent {
        if !exact_measurement_spec_match {
            reasons.push("equivalent construct labels have different measurement specifications".to_owned());
        }
        if !normalized_scale_anchor_match {
            reasons.push("semantic score-scale anchors do not match".to_owned());
        }
        if left_weight != right_weight {
            reasons.push("criterion weight totals differ for this construct".to_owned());
        }
        if anchor_count < required_anchors {
            reasons.push("shared anchor-item coverage is below the preregistered minimum".to_owned());
        }
        if !mapping_adjudicators_disjoint {
            reasons.push("left and right rubric mapping panels share adjudicator identity commitments".to_owned());
        }
        ConstructCommensurabilityV1::CalibrationRequired
    } else {
        reasons.push("criterion mappings are narrower, broader, overlapping or otherwise non-equivalent".to_owned());
        ConstructCommensurabilityV1::PartiallyMapped
    };

    if status == ConstructCommensurabilityV1::DirectlyCommensurable && reasons.is_empty() {
        reasons.push("criterion definitions, measurement specifications, normalized scale anchors, weights and anchor coverage match".to_owned());
    }
    ConstructCommensurabilityResultV1 {
        construct_id: construct.construct_id.clone(),
        status,
        left_criterion_ids: criterion_ids(left),
        right_criterion_ids: criterion_ids(right),
        left_relations,
        right_relations,
        left_weight_millionths: left_weight,
        right_weight_millionths: right_weight,
        weight_difference_millionths: weight_difference,
        left_measurement_spec_sha256: left_measurement,
        right_measurement_spec_sha256: right_measurement,
        exact_measurement_spec_match,
        normalized_scale_anchor_match,
        shared_anchor_item_count: anchor_count,
        required_anchor_item_count: required_anchors,
        direct_score_comparison_allowed: status == ConstructCommensurabilityV1::DirectlyCommensurable,
        reasons,
    }
}

fn plan_fields_commitment(fields: &RubricCommensurabilityPlanFieldsV1) -> Result<String, String> {
    let payload = serde_json::to_vec(fields)
        .map_err(|error| format!("cannot serialize commensurability plan fields: {error}"))?;
    domain_digest(PLAN_DOMAIN, &payload)
}

pub fn compute_rubric_commensurability_plan_commitment(fields_bytes: &[u8]) -> Result<String, String> {
    let fields: RubricCommensurabilityPlanFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| format!("commensurability plan fields invalid: {error}"))?;
    validate_rubric_commensurability_plan_fields(&fields)
        .map_err(|issues| issues.join("; "))?;
    plan_fields_commitment(&fields)
}

pub fn verify_rubric_commensurability_plan(
    plan_bytes: &[u8],
) -> Result<(RubricCommensurabilityPlanV1, String), String> {
    let plan: RubricCommensurabilityPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("commensurability plan invalid: {error}"))?;
    let fields = plan.fields();
    validate_rubric_commensurability_plan_fields(&fields)
        .map_err(|issues| issues.join("; "))?;
    let expected = plan_fields_commitment(&fields)?;
    if !valid_digest(&plan.plan_commitment_sha256, !plan.purpose.qualified())
        || plan.plan_commitment_sha256 != expected
    {
        return Err("rubric commensurability plan commitment mismatch".to_owned());
    }
    Ok((plan, sha256_hex(plan_bytes)))
}

pub fn audit_rubric_commensurability(
    plan_bytes: &[u8],
) -> Result<RubricCommensurabilityReportV1, String> {
    let (plan, plan_sha256) = verify_rubric_commensurability_plan(plan_bytes)?;
    let fields = plan.fields();
    let left = instrument_construct_map(&plan.left_instrument);
    let right = instrument_construct_map(&plan.right_instrument);
    let anchors = shared_anchors_by_construct(&fields);
    let left_adjudicators: BTreeSet<&str> = plan.left_instrument.criteria.iter()
        .map(|criterion| criterion.construct_mapping.adjudicator_id_sha256.as_str())
        .collect();
    let right_adjudicators: BTreeSet<&str> = plan.right_instrument.criteria.iter()
        .map(|criterion| criterion.construct_mapping.adjudicator_id_sha256.as_str())
        .collect();
    let adjudicator_overlap: BTreeSet<&str> = left_adjudicators.intersection(&right_adjudicators).copied().collect();
    let mapping_adjudicators_disjoint = adjudicator_overlap.is_empty();
    let mut results = Vec::new();
    for construct in &plan.constructs {
        let left_rows = left.get(construct.construct_id.as_str()).cloned().unwrap_or_default();
        let right_rows = right.get(construct.construct_id.as_str()).cloned().unwrap_or_default();
        results.push(construct_result(
            construct,
            &left_rows,
            &right_rows,
            anchors.get(construct.construct_id.as_str()).copied().unwrap_or(0),
            u64::from(plan.minimum_anchor_items_per_construct),
            mapping_adjudicators_disjoint,
        ));
    }

    let count_status = |status| results.iter().filter(|row| row.status == status).count() as u64;
    let direct = count_status(ConstructCommensurabilityV1::DirectlyCommensurable);
    let calibration = count_status(ConstructCommensurabilityV1::CalibrationRequired);
    let partial = count_status(ConstructCommensurabilityV1::PartiallyMapped);
    let incompatible = count_status(ConstructCommensurabilityV1::Incompatible);
    let unmapped = count_status(ConstructCommensurabilityV1::Unmapped);
    let insufficient_anchor = results.iter()
        .filter(|row| row.shared_anchor_item_count < row.required_anchor_item_count)
        .count() as u64;
    let left_unmapped = plan.left_instrument.criteria.iter()
        .filter(|criterion| criterion.construct_mapping.relation == ConstructRelationV1::Unmapped).count() as u64;
    let right_unmapped = plan.right_instrument.criteria.iter()
        .filter(|criterion| criterion.construct_mapping.relation == ConstructRelationV1::Unmapped).count() as u64;
    let left_weight = score_weight(&plan.left_instrument.criteria.iter().collect::<Vec<_>>());
    let right_weight = score_weight(&plan.right_instrument.criteria.iter().collect::<Vec<_>>());

    let overall = if incompatible > 0 {
        OverallCommensurabilityV1::NonCommensurable
    } else if unmapped > 0 || insufficient_anchor > 0 {
        OverallCommensurabilityV1::InsufficientEvidence
    } else if calibration > 0 {
        OverallCommensurabilityV1::CalibrationRequired
    } else if partial > 0 || direct < plan.constructs.len() as u64 {
        if direct == 0 { OverallCommensurabilityV1::NonCommensurable }
        else { OverallCommensurabilityV1::PartiallyCommensurable }
    } else {
        OverallCommensurabilityV1::FullyCommensurable
    };

    let direct_allowed = overall == OverallCommensurabilityV1::FullyCommensurable
        && direct == plan.constructs.len() as u64
        && left_unmapped == 0
        && right_unmapped == 0
        && mapping_adjudicators_disjoint;
    let report_fields = RubricCommensurabilityReportFieldsV1 {
        schema: RUBRIC_COMMENSURABILITY_REPORT_SCHEMA.to_owned(),
        analyzer_source_revision: plan.analyzer_source_revision,
        analyzer_worktree_clean: crate::BUILT_GENERATOR_WORKTREE_CLEAN == "true",
        comparison_id: plan.comparison_id,
        plan_sha256,
        plan_commitment_sha256: plan.plan_commitment_sha256,
        left_rubric_id: plan.left_instrument.rubric_id,
        right_rubric_id: plan.right_instrument.rubric_id,
        overall_status: overall,
        construct_count: plan.constructs.len() as u64,
        directly_commensurable_construct_count: direct,
        calibration_required_construct_count: calibration,
        partially_mapped_construct_count: partial,
        incompatible_construct_count: incompatible,
        unmapped_construct_count: unmapped,
        insufficient_anchor_construct_count: insufficient_anchor,
        left_criterion_count: plan.left_instrument.criteria.len() as u64,
        right_criterion_count: plan.right_instrument.criteria.len() as u64,
        left_unmapped_criterion_count: left_unmapped,
        right_unmapped_criterion_count: right_unmapped,
        left_weight_total_millionths: left_weight,
        right_weight_total_millionths: right_weight,
        left_right_mapping_adjudicator_overlap_count: adjudicator_overlap.len() as u64,
        mapping_adjudicator_ids_disjoint: mapping_adjudicators_disjoint,
        mapping_panel_independence_proven: false,
        shared_anchor_item_count: plan.shared_anchor_items.len() as u64,
        minimum_anchor_items_per_construct: plan.minimum_anchor_items_per_construct,
        direct_score_comparison_allowed: direct_allowed,
        model_ranking_comparison_allowed: direct_allowed,
        construct_validity_proven: false,
        human_alignment_proven: false,
        generalized_accuracy_estimated: false,
        scoring_eligible_declared: false,
        external_anchor_required: true,
        construct_results: results,
        claim_boundary: "Structural rubric commensurability audit only. Declared mapping, measurement-specification and score-anchor digests are checked, and adjudicator identity commitments are required to be disjoint across instruments for direct comparison; identity disjointness does not prove actual reviewer independence. The analyzer does not establish the truth of declarations, human construct validity, human alignment, generalized accuracy or scoring eligibility.".to_owned(),
    };
    let payload = serde_json::to_vec(&report_fields)
        .map_err(|error| format!("cannot serialize commensurability report: {error}"))?;
    let commitment = domain_digest(REPORT_DOMAIN, &payload)?;
    Ok(RubricCommensurabilityReportV1 {
        fields: report_fields,
        report_commitment_sha256: commitment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &str) -> String {
        sha256_hex(value.as_bytes())
    }

    fn criterion(
        id: &str,
        construct: &str,
        weight: u32,
        measurement: &str,
        mapping: ConstructRelationV1,
        direction: ScoreDirectionV1,
    ) -> RubricCriterionV1 {
        let anchors = match direction {
            ScoreDirectionV1::HigherRawIsBetter => vec![
                ScaleAnchorV1 { raw_value: 0, normalized_score_millionths: 0, semantic_anchor_sha256: digest("poor") },
                ScaleAnchorV1 { raw_value: 100, normalized_score_millionths: 500_000, semantic_anchor_sha256: digest("adequate") },
                ScaleAnchorV1 { raw_value: 200, normalized_score_millionths: 1_000_000, semantic_anchor_sha256: digest("excellent") },
            ],
            ScoreDirectionV1::LowerRawIsBetter => vec![
                ScaleAnchorV1 { raw_value: 0, normalized_score_millionths: 1_000_000, semantic_anchor_sha256: digest("excellent") },
                ScaleAnchorV1 { raw_value: 10, normalized_score_millionths: 500_000, semantic_anchor_sha256: digest("adequate") },
                ScaleAnchorV1 { raw_value: 20, normalized_score_millionths: 0, semantic_anchor_sha256: digest("poor") },
            ],
        };
        RubricCriterionV1 {
            criterion_id: id.to_owned(),
            definition_sha256: digest(&format!("definition-{id}")),
            measurement_spec_sha256: digest(measurement),
            score_min_raw: 0,
            score_max_raw: if direction == ScoreDirectionV1::HigherRawIsBetter { 200 } else { 20 },
            score_direction: direction,
            scale_anchors: anchors,
            weight_millionths: weight,
            construct_mapping: CriterionConstructMappingV1 {
                construct_id: Some(construct.to_owned()),
                relation: mapping,
                mapping_evidence_sha256: digest(&format!("mapping-evidence-{id}")),
                adjudicator_id_sha256: digest(&format!("adjudicator-{id}")),
            },
        }
    }

    fn make_fields() -> RubricCommensurabilityPlanFieldsV1 {
        RubricCommensurabilityPlanFieldsV1 {
            schema: RUBRIC_COMMENSURABILITY_PLAN_SCHEMA.to_owned(),
            analyzer_source_revision: crate::BUILT_GENERATOR_SOURCE_REVISION.to_owned(),
            purpose: CommensurabilityPurposeV1::QualifiedRun,
            comparison_id: "rubric-commensurability-example".to_owned(),
            construct_ontology_sha256: digest("ontology-v1"),
            constructs: vec![
                ConstructDefinitionV1 {
                    construct_id: "factual_accuracy".to_owned(),
                    definition_sha256: digest("factual-accuracy-definition"),
                    scope_sha256: digest("factual-accuracy-scope"),
                },
                ConstructDefinitionV1 {
                    construct_id: "evidence_support".to_owned(),
                    definition_sha256: digest("evidence-support-definition"),
                    scope_sha256: digest("evidence-support-scope"),
                },
            ],
            left_instrument: RubricInstrumentV1 {
                rubric_id: "rubric-A".to_owned(),
                rubric_revision: "rev-1".to_owned(),
                rubric_sha256: digest("rubric-A"),
                criteria: vec![
                    criterion("accuracy-A", "factual_accuracy", 600_000, "accuracy-spec-v1", ConstructRelationV1::Equivalent, ScoreDirectionV1::HigherRawIsBetter),
                    criterion("evidence-A", "evidence_support", 400_000, "evidence-spec-v1", ConstructRelationV1::Equivalent, ScoreDirectionV1::HigherRawIsBetter),
                ],
            },
            right_instrument: RubricInstrumentV1 {
                rubric_id: "rubric-B".to_owned(),
                rubric_revision: "rev-7".to_owned(),
                rubric_sha256: digest("rubric-B"),
                criteria: vec![
                    criterion("accuracy-B", "factual_accuracy", 600_000, "accuracy-spec-v1", ConstructRelationV1::Equivalent, ScoreDirectionV1::HigherRawIsBetter),
                    criterion("evidence-B", "evidence_support", 400_000, "evidence-spec-v1", ConstructRelationV1::Equivalent, ScoreDirectionV1::HigherRawIsBetter),
                ],
            },
            shared_anchor_items: vec![
                SharedAnchorItemV1 { anchor_id: "anchor-1".to_owned(), construct_id: "factual_accuracy".to_owned(), semantics_sha256: digest("anchor-1") },
                SharedAnchorItemV1 { anchor_id: "anchor-2".to_owned(), construct_id: "factual_accuracy".to_owned(), semantics_sha256: digest("anchor-2") },
                SharedAnchorItemV1 { anchor_id: "anchor-3".to_owned(), construct_id: "evidence_support".to_owned(), semantics_sha256: digest("anchor-3") },
                SharedAnchorItemV1 { anchor_id: "anchor-4".to_owned(), construct_id: "evidence_support".to_owned(), semantics_sha256: digest("anchor-4") },
            ],
            minimum_anchor_items_per_construct: 2,
        }
    }

    fn committed_plan() -> (RubricCommensurabilityPlanV1, Vec<u8>) {
        let fields = make_fields();
        let commitment = plan_fields_commitment(&fields).unwrap();
        let plan = RubricCommensurabilityPlanV1 {
            schema: fields.schema.clone(),
            analyzer_source_revision: fields.analyzer_source_revision.clone(),
            purpose: fields.purpose,
            comparison_id: fields.comparison_id.clone(),
            construct_ontology_sha256: fields.construct_ontology_sha256.clone(),
            constructs: fields.constructs.clone(),
            left_instrument: fields.left_instrument.clone(),
            right_instrument: fields.right_instrument.clone(),
            shared_anchor_items: fields.shared_anchor_items.clone(),
            minimum_anchor_items_per_construct: fields.minimum_anchor_items_per_construct,
            plan_commitment_sha256: commitment,
        };
        let bytes = serde_json::to_vec_pretty(&plan).unwrap();
        (plan, bytes)
    }

    #[test]
    fn identical_measurement_instruments_are_structurally_commensurable() {
        let (_, bytes) = committed_plan();
        let report = audit_rubric_commensurability(&bytes).unwrap();
        assert_eq!(report.fields.overall_status, OverallCommensurabilityV1::FullyCommensurable);
        assert_eq!(report.fields.directly_commensurable_construct_count, 2);
        assert!(report.fields.direct_score_comparison_allowed);
        assert!(report.fields.model_ranking_comparison_allowed);
        assert!(!report.fields.construct_validity_proven);
        assert!(!report.fields.scoring_eligible_declared);
    }

    #[test]
    fn equivalent_labels_do_not_hide_different_measurement_specs() {
        let mut fields = make_fields();
        fields.right_instrument.criteria[0].measurement_spec_sha256 = digest("different specification");
        let issues = validate_rubric_commensurability_plan_fields(&fields);
        assert!(issues.is_ok());
        let (_, bytes) = {
            let commitment = plan_fields_commitment(&fields).unwrap();
            let plan = RubricCommensurabilityPlanV1 {
                schema: fields.schema.clone(),
                analyzer_source_revision: fields.analyzer_source_revision.clone(),
                purpose: fields.purpose,
                comparison_id: fields.comparison_id.clone(),
                construct_ontology_sha256: fields.construct_ontology_sha256.clone(),
                constructs: fields.constructs.clone(),
                left_instrument: fields.left_instrument.clone(),
                right_instrument: fields.right_instrument.clone(),
                shared_anchor_items: fields.shared_anchor_items.clone(),
                minimum_anchor_items_per_construct: fields.minimum_anchor_items_per_construct,
                plan_commitment_sha256: commitment,
            };
            (plan.clone(), serde_json::to_vec_pretty(&plan).unwrap())
        };
        let report = audit_rubric_commensurability(&bytes).unwrap();
        assert_eq!(report.fields.overall_status, OverallCommensurabilityV1::CalibrationRequired);
        assert!(!report.fields.direct_score_comparison_allowed);
    }

    #[test]
    fn shared_mapping_adjudicators_require_calibration_instead_of_direct_comparison() {
        let mut fields = make_fields();
        fields.right_instrument.criteria[0].construct_mapping.adjudicator_id_sha256 =
            fields.left_instrument.criteria[0].construct_mapping.adjudicator_id_sha256.clone();
        let commitment = plan_fields_commitment(&fields).unwrap();
        let plan = RubricCommensurabilityPlanV1 {
            schema: fields.schema.clone(),
            analyzer_source_revision: fields.analyzer_source_revision.clone(),
            purpose: fields.purpose,
            comparison_id: fields.comparison_id.clone(),
            construct_ontology_sha256: fields.construct_ontology_sha256.clone(),
            constructs: fields.constructs.clone(),
            left_instrument: fields.left_instrument.clone(),
            right_instrument: fields.right_instrument.clone(),
            shared_anchor_items: fields.shared_anchor_items.clone(),
            minimum_anchor_items_per_construct: fields.minimum_anchor_items_per_construct,
            plan_commitment_sha256: commitment,
        };
        let report = audit_rubric_commensurability(&serde_json::to_vec(&plan).unwrap()).unwrap();
        assert_eq!(report.fields.overall_status, OverallCommensurabilityV1::CalibrationRequired);
        assert!(!report.fields.mapping_adjudicator_ids_disjoint);
        assert_eq!(report.fields.left_right_mapping_adjudicator_overlap_count, 1);
        assert!(!report.fields.mapping_panel_independence_proven);
        assert!(!report.fields.direct_score_comparison_allowed);
    }

    #[test]
    fn duplicate_semantic_anchor_content_cannot_inflate_anchor_coverage() {
        let mut fields = make_fields();
        fields.shared_anchor_items[1].semantics_sha256 =
            fields.shared_anchor_items[0].semantics_sha256.clone();
        assert!(validate_rubric_commensurability_plan_fields(&fields).is_err());
    }

    #[test]
    fn detects_incompatible_and_unmapped_constructs() {
        let mut fields = make_fields();
        fields.right_instrument.criteria[0].construct_mapping.relation = ConstructRelationV1::Incompatible;
        fields.right_instrument.criteria[1].construct_mapping.construct_id = None;
        fields.right_instrument.criteria[1].construct_mapping.relation = ConstructRelationV1::Unmapped;
        let commitment = plan_fields_commitment(&fields).unwrap();
        let plan = RubricCommensurabilityPlanV1 {
            schema: fields.schema.clone(),
            analyzer_source_revision: fields.analyzer_source_revision.clone(),
            purpose: fields.purpose,
            comparison_id: fields.comparison_id.clone(),
            construct_ontology_sha256: fields.construct_ontology_sha256.clone(),
            constructs: fields.constructs.clone(),
            left_instrument: fields.left_instrument.clone(),
            right_instrument: fields.right_instrument.clone(),
            shared_anchor_items: fields.shared_anchor_items.clone(),
            minimum_anchor_items_per_construct: fields.minimum_anchor_items_per_construct,
            plan_commitment_sha256: commitment,
        };
        let report = audit_rubric_commensurability(&serde_json::to_vec(&plan).unwrap()).unwrap();
        assert_eq!(report.fields.overall_status, OverallCommensurabilityV1::NonCommensurable);
        assert!(!report.fields.direct_score_comparison_allowed);
        assert_eq!(report.fields.right_unmapped_criterion_count, 1);
    }

    #[test]
    fn insufficient_anchor_coverage_blocks_direct_comparison() {
        let mut fields = make_fields();
        fields.shared_anchor_items.pop();
        let commitment = plan_fields_commitment(&fields).unwrap();
        let plan = RubricCommensurabilityPlanV1 {
            schema: fields.schema.clone(),
            analyzer_source_revision: fields.analyzer_source_revision.clone(),
            purpose: fields.purpose,
            comparison_id: fields.comparison_id.clone(),
            construct_ontology_sha256: fields.construct_ontology_sha256.clone(),
            constructs: fields.constructs.clone(),
            left_instrument: fields.left_instrument.clone(),
            right_instrument: fields.right_instrument.clone(),
            shared_anchor_items: fields.shared_anchor_items.clone(),
            minimum_anchor_items_per_construct: fields.minimum_anchor_items_per_construct,
            plan_commitment_sha256: commitment,
        };
        let report = audit_rubric_commensurability(&serde_json::to_vec(&plan).unwrap()).unwrap();
        assert_eq!(report.fields.overall_status, OverallCommensurabilityV1::InsufficientEvidence);
        assert!(!report.fields.direct_score_comparison_allowed);
    }

    #[test]
    fn rejects_weights_that_do_not_sum_to_one() {
        let mut fields = make_fields();
        fields.left_instrument.criteria[0].weight_millionths = 599_999;
        assert!(validate_rubric_commensurability_plan_fields(&fields).is_err());
    }

    #[test]
    fn rejects_nonmonotonic_or_incomplete_score_scales() {
        let mut fields = make_fields();
        fields.left_instrument.criteria[0].scale_anchors[1].normalized_score_millionths = 0;
        assert!(validate_rubric_commensurability_plan_fields(&fields).is_err());
        let mut fields = make_fields();
        fields.right_instrument.criteria[0].scale_anchors.pop();
        assert!(validate_rubric_commensurability_plan_fields(&fields).is_err());
    }

    #[test]
    fn rejects_unknown_constructs_and_duplicate_anchor_ids() {
        let mut fields = make_fields();
        fields.left_instrument.criteria[0].construct_mapping.construct_id = Some("invented".to_owned());
        assert!(validate_rubric_commensurability_plan_fields(&fields).is_err());
        let mut fields = make_fields();
        fields.shared_anchor_items[1].anchor_id = fields.shared_anchor_items[0].anchor_id.clone();
        assert!(validate_rubric_commensurability_plan_fields(&fields).is_err());
    }

    #[test]
    fn plan_commitment_rejects_post_hoc_mapping_mutation() {
        let (_, bytes) = committed_plan();
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        value["right_instrument"]["criteria"][0]["construct_mapping"]["relation"] =
            serde_json::json!("overlaps");
        assert!(verify_rubric_commensurability_plan(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn analysis_is_deterministic_for_exact_plan_bytes() {
        let (_, bytes) = committed_plan();
        let first = serde_json::to_vec(&audit_rubric_commensurability(&bytes).unwrap()).unwrap();
        let second = serde_json::to_vec(&audit_rubric_commensurability(&bytes).unwrap()).unwrap();
        assert_eq!(first, second);
    }
}
