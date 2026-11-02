//! Quantify adjudication attrition and bound selection from disputed transfer labels.
//!
//! The candidate universe is frozen before adjudication. Every candidate receives
//! exactly one terminal decision. Accepted candidates must match the transfer
//! study's scored item set; all other candidates remain in the denominator.
//!
//! For binary outcomes, an excluded candidate's unknown treatment-minus-placebo
//! effect is bounded by [-1, +1]. The resulting cell bounds are intentionally
//! conservative and are macro-aggregated across difficulty bands and families.
//! This is partial identification, not an imputation model.

use crate::{
    causal_analysis::{
        MeanIntervalV1, ReplicatedCausalStudyV1, analyze_replicated_causal_study, mean_interval,
    },
    exposure_transfer::{
        ExposureTransferStudyV1, RelationAdjudicationDecisionV1, TransferRelationV1,
    },
    hash_domain,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const ATTRITION_STUDY_SCHEMA: &str = "symthaea.adjudication_attrition_study.v1";
pub const ATTRITION_REPORT_SCHEMA: &str = "symthaea.adjudication_attrition_report.v1";
pub const CANDIDATE_MANIFEST_SCHEMA: &str = "symthaea.adjudication_candidate_manifest.v1";
pub const MAX_CANDIDATES: usize = 1_000_000;
pub const MAX_ADJUDICATION_RECORDS: usize = 1_000_000;
pub const MIN_TASK_FAMILIES: usize = 2;
pub const MAX_REVIEWERS_PER_CANDIDATE: usize = 20;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TerminalDecisionV1 {
    Accepted,
    Indeterminate,
    Rejected,
    Duplicate,
    ProtocolInvalid,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateManifestItemV1 {
    pub candidate_id: String,
    pub task_family: String,
    pub difficulty_band: String,
    pub relation: TransferRelationV1,
    /// Per-candidate source and recipe lineage commitments. These are evaluator-only.
    pub source_lineage_sha256: String,
    pub recipe_lineage_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateManifestV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    /// Strict UTC timestamp, seconds precision: YYYY-MM-DDTHH:MM:SSZ.
    pub frozen_at_utc: String,
    pub candidates: Vec<CandidateManifestItemV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateAdjudicationV1 {
    pub candidate_id: String,
    pub terminal_decision: TerminalDecisionV1,
    pub reason_code: String,
    pub reviewer_ids: Vec<String>,
    /// Per-item agreement among the recorded reviewers, in permille.
    pub agreement_permille: Option<u16>,
    /// Strict UTC timestamp, seconds precision.
    pub decided_at_utc: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdjudicationAttritionStudyV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    /// Domain-separated digest of the exact ExposureTransferStudyV1 JSON.
    pub transfer_study_sha256: String,
    pub candidate_manifest: CandidateManifestV1,
    pub candidate_manifest_sha256: String,
    pub adjudication_started_at_utc: String,
    pub adjudications: Vec<CandidateAdjudicationV1>,
    pub adjudication_report_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttritionReplicateBoundsV1 {
    pub replicate_id: String,
    pub accepted_subset_effect_permille: Option<f64>,
    pub full_candidate_lower_effect_permille: f64,
    pub full_candidate_upper_effect_permille: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttritionCellReportV1 {
    pub task_family: String,
    pub difficulty_band: String,
    pub relation: TransferRelationV1,
    pub candidate_count: u64,
    pub accepted_count: u64,
    pub indeterminate_count: u64,
    pub rejected_count: u64,
    pub duplicate_count: u64,
    pub protocol_invalid_count: u64,
    pub excluded_count: u64,
    pub acceptance_rate_permille: f64,
    pub adjudication_agreement_mean_permille: Option<f64>,
    pub accepted_subset_effect_permille: Option<MeanIntervalV1>,
    pub full_candidate_lower_effect_permille: MeanIntervalV1,
    pub full_candidate_upper_effect_permille: MeanIntervalV1,
    pub per_replicate_bounds: Vec<AttritionReplicateBoundsV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttritionRelationReportV1 {
    pub relation: TransferRelationV1,
    pub candidate_count: u64,
    pub accepted_count: u64,
    pub indeterminate_count: u64,
    pub rejected_count: u64,
    pub duplicate_count: u64,
    pub protocol_invalid_count: u64,
    pub excluded_count: u64,
    /// Pooled candidate-volume coverage, descriptive only.
    pub acceptance_rate_permille: f64,
    /// Equal difficulty-band weight within family, then equal family weight.
    pub family_macro_acceptance_rate_permille: f64,
    pub full_candidate_lower_effect_permille: MeanIntervalV1,
    pub full_candidate_upper_effect_permille: MeanIntervalV1,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdjudicationAttritionReportV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub base_causal_study_sha256: String,
    pub transfer_study_sha256: String,
    pub candidate_manifest_sha256: String,
    pub adjudication_report_sha256: String,
    pub attrition_study_sha256: String,
    pub independent_model_pair_replicate_count: u64,
    pub candidate_count: u64,
    pub accepted_count: u64,
    pub excluded_count: u64,
    pub overall_acceptance_rate_permille: f64,
    pub relation_reports: Vec<AttritionRelationReportV1>,
    pub cell_reports: Vec<AttritionCellReportV1>,
    pub intervals_multiplicity_adjusted: bool,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub analysis_status: String,
    pub issues: Vec<String>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CellKey {
    task_family: String,
    difficulty_band: String,
    relation: TransferRelationV1,
}

#[derive(Debug, Clone, Copy, Default)]
struct DecisionCounts {
    accepted: u64,
    indeterminate: u64,
    rejected: u64,
    duplicate: u64,
    protocol_invalid: u64,
}

impl DecisionCounts {
    fn excluded(self) -> u64 {
        self.indeterminate + self.rejected + self.duplicate + self.protocol_invalid
    }

    fn total(self) -> u64 {
        self.accepted + self.excluded()
    }

    fn increment(&mut self, decision: &TerminalDecisionV1) -> Result<(), String> {
        let counter = match decision {
            TerminalDecisionV1::Accepted => &mut self.accepted,
            TerminalDecisionV1::Indeterminate => &mut self.indeterminate,
            TerminalDecisionV1::Rejected => &mut self.rejected,
            TerminalDecisionV1::Duplicate => &mut self.duplicate,
            TerminalDecisionV1::ProtocolInvalid => &mut self.protocol_invalid,
        };
        *counter = counter
            .checked_add(1)
            .ok_or_else(|| "adjudication decision count overflows".to_owned())?;
        Ok(())
    }
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be a lowercase SHA-256 digest"))
    }
}

fn valid_opaque_id(value: &str, max_len: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= max_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
}

fn parse_utc_seconds(value: &str, label: &str) -> Result<(i32, u32, u32, u32, u32, u32), String> {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return Err(format!("{label} must use YYYY-MM-DDTHH:MM:SSZ"));
    }
    let parse = |range: std::ops::Range<usize>| -> Result<u32, String> {
        let part = std::str::from_utf8(&bytes[range])
            .map_err(|_| format!("{label} contains non-ASCII digits"))?;
        if !part.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!("{label} contains non-ASCII digits"));
        }
        part.parse::<u32>()
            .map_err(|_| format!("{label} contains an invalid number"))
    };
    let year = parse(0..4)? as i32;
    let month = parse(5..7)?;
    let day = parse(8..10)?;
    let hour = parse(11..13)?;
    let minute = parse(14..16)?;
    let second = parse(17..19)?;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 0,
    };
    if year == 0 || day == 0 || day > days || hour > 23 || minute > 59 || second > 59 {
        return Err(format!("{label} is not a valid UTC calendar timestamp"));
    }
    Ok((year, month, day, hour, minute, second))
}

fn canonical_json<T: Serialize>(value: &T, label: &str) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|error| format!("cannot serialize {label}: {error}"))
}

fn canonical_candidate_manifest(manifest: &CandidateManifestV1) -> Result<Vec<u8>, String> {
    let mut normalized = manifest.clone();
    normalized
        .candidates
        .sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    canonical_json(&normalized, "candidate manifest")
}

fn canonical_adjudication_report(records: &[CandidateAdjudicationV1]) -> Result<Vec<u8>, String> {
    let mut normalized = records.to_vec();
    normalized.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    for record in &mut normalized {
        record.reviewer_ids.sort();
    }
    canonical_json(&normalized, "adjudication records")
}

fn canonical_ids_digest(domain: &[u8], ids: &[String]) -> Result<String, String> {
    let mut sorted = ids.to_vec();
    sorted.sort();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("candidate item IDs must be unique within each cell".to_owned());
    }
    let bytes = canonical_json(&sorted, "canonical item IDs")?;
    Ok(hash_domain(domain, &bytes))
}

fn validate_reviewer_ids(reviewers: &[String], min_count: usize) -> Result<(), String> {
    if reviewers.len() < min_count || reviewers.len() > MAX_REVIEWERS_PER_CANDIDATE {
        return Err(format!(
            "reviewer count must be in {min_count}..={MAX_REVIEWERS_PER_CANDIDATE}"
        ));
    }
    let unique: BTreeSet<&str> = reviewers.iter().map(String::as_str).collect();
    if unique.len() != reviewers.len()
        || reviewers
            .iter()
            .any(|reviewer| !valid_opaque_id(reviewer, 64))
    {
        return Err("reviewer IDs must be distinct opaque ASCII identifiers".to_owned());
    }
    Ok(())
}

fn accepted_cell_adjudication_digest(
    records: &[&CandidateAdjudicationV1],
) -> Result<String, String> {
    let mut normalized: Vec<CandidateAdjudicationV1> =
        records.iter().map(|record| (*record).clone()).collect();
    normalized.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    for record in &mut normalized {
        record.reviewer_ids.sort();
    }
    let bytes = canonical_json(&normalized, "accepted cell adjudication records")?;
    Ok(hash_domain(
        b"exposure-transfer-cell-adjudication.v1",
        &bytes,
    ))
}

fn cell_lineage_digest(
    candidates: &[&CandidateManifestItemV1],
    source: bool,
) -> Result<String, String> {
    let mut tuples: Vec<(&str, &str)> = candidates
        .iter()
        .map(|item| {
            (
                item.candidate_id.as_str(),
                if source {
                    item.source_lineage_sha256.as_str()
                } else {
                    item.recipe_lineage_sha256.as_str()
                },
            )
        })
        .collect();
    tuples.sort();
    let bytes = canonical_json(&tuples, "cell lineage tuples")?;
    Ok(hash_domain(
        if source {
            b"exposure-transfer-source-lineage-cell.v1"
        } else {
            b"exposure-transfer-recipe-lineage-cell.v1"
        },
        &bytes,
    ))
}

fn outcome_digest(
    evidence: &crate::causal_analysis::EvaluationEvidenceV1,
) -> Result<String, String> {
    let bytes = canonical_json(evidence, "evaluation evidence")?;
    Ok(hash_domain(
        b"exposure-transfer-evaluation-evidence.v1",
        &bytes,
    ))
}

fn transfer_digest(study: &ExposureTransferStudyV1) -> Result<String, String> {
    let bytes = canonical_json(study, "transfer study")?;
    Ok(hash_domain(b"exposure-transfer-study.v1", &bytes))
}

fn item_set_digest(ids: &[String]) -> Result<String, String> {
    canonical_ids_digest(b"exposure-transfer-item-set.v1", ids)
}

/// Canonical domain-separated digest of a frozen candidate manifest.
pub fn compute_candidate_manifest_sha256(manifest: &CandidateManifestV1) -> Result<String, String> {
    let bytes = canonical_candidate_manifest(manifest)?;
    Ok(hash_domain(b"adjudication-candidate-manifest.v1", &bytes))
}

/// Canonical domain-separated digest of all terminal adjudication decisions.
pub fn compute_adjudication_report_sha256(
    records: &[CandidateAdjudicationV1],
) -> Result<String, String> {
    let bytes = canonical_adjudication_report(records)?;
    Ok(hash_domain(b"adjudication-attrition-report.v1", &bytes))
}

/// Canonical digest for a cell's accepted per-candidate decisions only.
/// The caller supplies the already cell-filtered accepted records.
pub fn compute_accepted_cell_adjudication_sha256(
    records: &[CandidateAdjudicationV1],
) -> Result<String, String> {
    if records.is_empty()
        || records
            .iter()
            .any(|record| record.terminal_decision != TerminalDecisionV1::Accepted)
    {
        return Err("accepted-cell digest requires one or more accepted records only".to_owned());
    }
    let refs: Vec<&CandidateAdjudicationV1> = records.iter().collect();
    accepted_cell_adjudication_digest(&refs)
}

/// Canonical source or recipe lineage commitment for one candidate cell.
/// Every candidate must belong to the same task-family/difficulty/relation cell.
pub fn compute_candidate_cell_lineage_sha256(
    candidates: &[CandidateManifestItemV1],
    source: bool,
) -> Result<String, String> {
    if candidates.is_empty() {
        return Err("candidate lineage cell cannot be empty".to_owned());
    }
    let first = &candidates[0];
    let mut ids = BTreeSet::new();
    for candidate in candidates {
        validate_candidate(candidate)?;
        if candidate.task_family != first.task_family
            || candidate.difficulty_band != first.difficulty_band
            || candidate.relation != first.relation
        {
            return Err(
                "candidate lineage digest input must contain exactly one relation cell".to_owned(),
            );
        }
        if !ids.insert(candidate.candidate_id.as_str()) {
            return Err("candidate lineage digest input contains duplicate IDs".to_owned());
        }
    }
    let refs: Vec<&CandidateManifestItemV1> = candidates.iter().collect();
    cell_lineage_digest(&refs, source)
}

/// Canonical digest of the exact parsed transfer-study artifact.
pub fn compute_transfer_study_sha256(
    transfer_study: &ExposureTransferStudyV1,
) -> Result<String, String> {
    transfer_digest(transfer_study)
}

fn validate_candidate(candidate: &CandidateManifestItemV1) -> Result<(), String> {
    if !valid_opaque_id(&candidate.candidate_id, 128)
        || !valid_opaque_id(&candidate.task_family, 64)
        || !valid_opaque_id(&candidate.difficulty_band, 64)
    {
        return Err(
            "candidate ID, task family and difficulty band must be nonempty opaque IDs".to_owned(),
        );
    }
    require_digest(&candidate.source_lineage_sha256, "source_lineage_sha256")?;
    require_digest(&candidate.recipe_lineage_sha256, "recipe_lineage_sha256")?;
    Ok(())
}

fn validate_adjudication_record(
    candidate: &CandidateManifestItemV1,
    record: &CandidateAdjudicationV1,
) -> Result<(), String> {
    if record.candidate_id != candidate.candidate_id {
        return Err("adjudication candidate ID does not match manifest".to_owned());
    }
    if !valid_opaque_id(&record.reason_code, 128) {
        return Err("adjudication reason_code must be a nonempty opaque ASCII ID".to_owned());
    }
    parse_utc_seconds(&record.decided_at_utc, "decided_at_utc")?;
    let min_reviewers = match record.terminal_decision {
        TerminalDecisionV1::Accepted
        | TerminalDecisionV1::Indeterminate
        | TerminalDecisionV1::Rejected => 3,
        TerminalDecisionV1::Duplicate | TerminalDecisionV1::ProtocolInvalid => 1,
    };
    validate_reviewer_ids(&record.reviewer_ids, min_reviewers)?;
    match record.terminal_decision {
        TerminalDecisionV1::Accepted => {
            let agreement = record
                .agreement_permille
                .ok_or_else(|| "accepted candidate requires agreement_permille".to_owned())?;
            if !(800..=1_000).contains(&agreement) {
                return Err(
                    "accepted candidate agreement must be in [800, 1000] permille".to_owned(),
                );
            }
            if matches!(
                candidate.relation,
                TransferRelationV1::SemanticEquivalent | TransferRelationV1::NovelRecipeSameSkill
            ) && agreement != 1_000
            {
                return Err(
                    "semantic-equivalent and same-skill candidates require unanimous agreement"
                        .to_owned(),
                );
            }
        }
        TerminalDecisionV1::Indeterminate | TerminalDecisionV1::Rejected => {
            if record.agreement_permille.is_none_or(|value| value > 1_000) {
                return Err(
                    "indeterminate/rejected candidates require a valid recorded agreement"
                        .to_owned(),
                );
            }
        }
        TerminalDecisionV1::Duplicate | TerminalDecisionV1::ProtocolInvalid => {
            if record.agreement_permille.is_some() {
                return Err(
                    "duplicate/protocol-invalid records must not claim semantic agreement"
                        .to_owned(),
                );
            }
        }
    }
    Ok(())
}

fn score_success_difference(
    treatment: &crate::causal_analysis::ScoreCountV1,
    placebo: &crate::causal_analysis::ScoreCountV1,
    expected_count: u64,
    label: &str,
) -> Result<i128, String> {
    if treatment.observed_count != expected_count
        || placebo.observed_count != expected_count
        || treatment.success_count > expected_count
        || placebo.success_count > expected_count
    {
        return Err(format!(
            "{label} score counts do not match the accepted item denominator"
        ));
    }
    Ok(i128::from(treatment.success_count) - i128::from(placebo.success_count))
}

fn effect_bounds_permille(
    accepted_success_difference: i128,
    accepted_count: u64,
    candidate_count: u64,
) -> Result<(Option<f64>, f64, f64), String> {
    if candidate_count == 0 || accepted_count > candidate_count {
        return Err(
            "candidate_count must be positive and accepted_count cannot exceed it".to_owned(),
        );
    }
    let excluded = i128::from(candidate_count - accepted_count);
    let denominator = candidate_count as f64;
    let lower = (accepted_success_difference - excluded) as f64 * 1_000.0 / denominator;
    let upper = (accepted_success_difference + excluded) as f64 * 1_000.0 / denominator;
    let accepted = if accepted_count == 0 {
        None
    } else {
        Some(accepted_success_difference as f64 * 1_000.0 / accepted_count as f64)
    };
    if !lower.is_finite()
        || !upper.is_finite()
        || lower < -1_000.0 - f64::EPSILON
        || upper > 1_000.0 + f64::EPSILON
        || accepted.is_some_and(|value| !value.is_finite())
    {
        return Err("partial-identification bounds are invalid".to_owned());
    }
    Ok((accepted, lower.max(-1_000.0), upper.min(1_000.0)))
}

/// Analyze adjudication attrition against the same randomized training and
/// transfer studies. Candidates are frozen before adjudication; every candidate
/// remains in the denominator. Missing outcomes for excluded candidates receive
/// the worst-case binary treatment-effect interval [-1, +1].
pub fn analyze_adjudication_attrition(
    base_study: &ReplicatedCausalStudyV1,
    transfer_study: &ExposureTransferStudyV1,
    study: &AdjudicationAttritionStudyV1,
) -> Result<AdjudicationAttritionReportV1, String> {
    let base_report = analyze_replicated_causal_study(base_study)?;
    if study.schema != ATTRITION_STUDY_SCHEMA {
        return Err(format!("study schema must be {ATTRITION_STUDY_SCHEMA}"));
    }
    if study.study_id != base_study.study_id
        || study.study_id != transfer_study.study_id
        || study.benchmark_id != base_study.benchmark_id
        || study.benchmark_id != transfer_study.benchmark_id
        || study.benchmark_version != base_study.benchmark_version
        || study.benchmark_version != transfer_study.benchmark_version
    {
        return Err("attrition, transfer and causal study identities must match".to_owned());
    }
    if study.candidate_manifest.schema != CANDIDATE_MANIFEST_SCHEMA
        || study.candidate_manifest.study_id != study.study_id
        || study.candidate_manifest.benchmark_id != study.benchmark_id
        || study.candidate_manifest.benchmark_version != study.benchmark_version
    {
        return Err("candidate manifest identity/schema must match the study".to_owned());
    }
    if study.candidate_manifest.candidates.is_empty()
        || study.candidate_manifest.candidates.len() > MAX_CANDIDATES
        || study.adjudications.len() != study.candidate_manifest.candidates.len()
        || study.adjudications.len() > MAX_ADJUDICATION_RECORDS
    {
        return Err(
            "candidate manifest and adjudication record counts must match and be bounded"
                .to_owned(),
        );
    }
    let frozen = parse_utc_seconds(&study.candidate_manifest.frozen_at_utc, "frozen_at_utc")?;
    let started = parse_utc_seconds(
        &study.adjudication_started_at_utc,
        "adjudication_started_at_utc",
    )?;
    if started <= frozen {
        return Err(
            "adjudication must start strictly after the candidate manifest is frozen".to_owned(),
        );
    }

    let manifest_bytes = canonical_candidate_manifest(&study.candidate_manifest)?;
    let computed_manifest_sha = hash_domain(b"adjudication-candidate-manifest.v1", &manifest_bytes);
    require_digest(
        &study.candidate_manifest_sha256,
        "candidate_manifest_sha256",
    )?;
    if study.candidate_manifest_sha256 != computed_manifest_sha {
        return Err("candidate manifest digest mismatch".to_owned());
    }
    let adjudication_bytes = canonical_adjudication_report(&study.adjudications)?;
    let computed_adjudication_sha =
        hash_domain(b"adjudication-attrition-report.v1", &adjudication_bytes);
    require_digest(
        &study.adjudication_report_sha256,
        "adjudication_report_sha256",
    )?;
    if study.adjudication_report_sha256 != computed_adjudication_sha {
        return Err("adjudication report digest mismatch".to_owned());
    }
    let computed_transfer_sha = transfer_digest(transfer_study)?;
    require_digest(&study.transfer_study_sha256, "transfer_study_sha256")?;
    if study.transfer_study_sha256 != computed_transfer_sha {
        return Err("transfer study digest does not match the supplied transfer study".to_owned());
    }

    let mut candidates = BTreeMap::<String, &CandidateManifestItemV1>::new();
    let mut cell_candidates = BTreeMap::<CellKey, Vec<&CandidateManifestItemV1>>::new();
    for candidate in &study.candidate_manifest.candidates {
        validate_candidate(candidate)?;
        if candidates
            .insert(candidate.candidate_id.clone(), candidate)
            .is_some()
        {
            return Err("candidate IDs must be globally unique".to_owned());
        }
        let key = CellKey {
            task_family: candidate.task_family.clone(),
            difficulty_band: candidate.difficulty_band.clone(),
            relation: candidate.relation,
        };
        cell_candidates.entry(key).or_default().push(candidate);
    }

    let mut decisions = BTreeMap::<String, &CandidateAdjudicationV1>::new();
    let mut counts = BTreeMap::<CellKey, DecisionCounts>::new();
    let mut accepted_records_by_cell = BTreeMap::<CellKey, Vec<&CandidateAdjudicationV1>>::new();
    let mut agreement_by_cell = BTreeMap::<CellKey, Vec<f64>>::new();
    for record in &study.adjudications {
        let candidate = candidates.get(&record.candidate_id).ok_or_else(|| {
            "adjudication contains a candidate absent from the frozen manifest".to_owned()
        })?;
        validate_adjudication_record(candidate, record)?;
        if record.decided_at_utc < study.adjudication_started_at_utc {
            return Err("adjudication decision timestamp precedes adjudication start".to_owned());
        }
        if decisions
            .insert(record.candidate_id.clone(), record)
            .is_some()
        {
            return Err("each candidate must have exactly one terminal decision".to_owned());
        }
        let key = CellKey {
            task_family: candidate.task_family.clone(),
            difficulty_band: candidate.difficulty_band.clone(),
            relation: candidate.relation,
        };
        counts
            .entry(key.clone())
            .or_default()
            .increment(&record.terminal_decision)?;
        if record.terminal_decision == TerminalDecisionV1::Accepted {
            accepted_records_by_cell
                .entry(key.clone())
                .or_default()
                .push(record);
        }
        if let Some(agreement) = record.agreement_permille {
            agreement_by_cell
                .entry(key)
                .or_default()
                .push(f64::from(agreement));
        }
    }
    if decisions.len() != candidates.len() {
        return Err(
            "every frozen candidate must have exactly one terminal adjudication record".to_owned(),
        );
    }

    let mut family_bands = BTreeMap::<String, BTreeSet<String>>::new();
    for key in cell_candidates.keys() {
        family_bands
            .entry(key.task_family.clone())
            .or_default()
            .insert(key.difficulty_band.clone());
    }
    if family_bands.len() < MIN_TASK_FAMILIES {
        return Err(format!(
            "at least {MIN_TASK_FAMILIES} task families are required"
        ));
    }
    let first_bands = family_bands
        .values()
        .next()
        .ok_or_else(|| "no task families supplied".to_owned())?;
    if family_bands.values().any(|bands| bands != first_bands) {
        return Err("all task families must use the same difficulty-band set".to_owned());
    }
    for (family, band) in {
        let mut cells = BTreeSet::new();
        for key in cell_candidates.keys() {
            cells.insert((key.task_family.clone(), key.difficulty_band.clone()));
        }
        cells
    } {
        for relation in TransferRelationV1::ALL {
            let key = CellKey {
                task_family: family.clone(),
                difficulty_band: band.clone(),
                relation,
            };
            if !cell_candidates.contains_key(&key) {
                return Err(format!(
                    "candidate manifest is missing {family}/{band}/{relation:?}"
                ));
            }
        }
    }

    let computed_base_ids: BTreeSet<&str> = base_study
        .replicates
        .iter()
        .map(|replicate| replicate.design.replicate_id.as_str())
        .collect();
    if computed_base_ids.len() != base_study.replicates.len()
        || transfer_study.replicates.len() != base_study.replicates.len()
    {
        return Err(
            "transfer study must have exactly one replicate per unique causal model pair"
                .to_owned(),
        );
    }
    if transfer_study.schema != crate::exposure_transfer::EXPOSURE_TRANSFER_STUDY_SCHEMA
        || transfer_study.primary_metric_id != base_study.primary_metric_id
        || transfer_study.primary_metric_definition_sha256
            != base_study.primary_metric_definition_sha256
    {
        return Err("transfer schema/metric does not match causal study".to_owned());
    }

    let mut transfer_ids = BTreeSet::new();
    let mut transfer_cell_signatures =
        BTreeMap::<CellKey, (String, String, String, String, Vec<String>)>::new();
    let mut pair_bounds = BTreeMap::<CellKey, Vec<AttritionReplicateBoundsV1>>::new();

    for transfer_replicate in &transfer_study.replicates {
        if !transfer_ids.insert(transfer_replicate.replicate_id.as_str()) {
            return Err("transfer replicate IDs must be unique".to_owned());
        }
        let base = base_study
            .replicates
            .iter()
            .find(|replicate| replicate.design.replicate_id == transfer_replicate.replicate_id)
            .ok_or_else(|| "transfer replicate has no matching causal replicate".to_owned())?;
        if transfer_replicate.training_receipt_audit_sha256 != base.training_receipt_audit_sha256 {
            return Err(
                "transfer training receipt digest differs from causal replicate".to_owned(),
            );
        }
        if transfer_replicate.evaluation_evidence_sha256
            != outcome_digest(&base.evaluation_evidence)?
        {
            return Err(
                "transfer evaluation evidence digest differs from causal replicate".to_owned(),
            );
        }

        let mut seen_transfer_cells = BTreeSet::new();
        for row in &transfer_replicate.strata {
            let key = CellKey {
                task_family: row.task_family.clone(),
                difficulty_band: row.difficulty_band.clone(),
                relation: row.relation,
            };
            if !seen_transfer_cells.insert(key.clone()) {
                return Err("transfer replicate contains duplicate relation cells".to_owned());
            }
            let cell_items = cell_candidates.get(&key).ok_or_else(|| {
                "transfer row is absent from the frozen candidate universe".to_owned()
            })?;
            let accepted_ids: Vec<String> = cell_items
                .iter()
                .filter(|candidate| {
                    decisions
                        .get(&candidate.candidate_id)
                        .is_some_and(|record| record.terminal_decision == TerminalDecisionV1::Accepted)
                })
                .map(|candidate| candidate.candidate_id.clone())
                .collect();
            if accepted_ids.is_empty() {
                return Err(
                    "a transfer score row exists for a cell with no accepted candidates".to_owned(),
                );
            }
            let mut sorted_row_ids = row.item_ids.clone();
            sorted_row_ids.sort();
            let mut sorted_accepted_ids = accepted_ids.clone();
            sorted_accepted_ids.sort();
            let row_ids_digest = item_set_digest(&row.item_ids)?;
            let accepted_ids_digest = item_set_digest(&accepted_ids)?;
            if row.item_set_sha256 != row_ids_digest
                || row_ids_digest != accepted_ids_digest
                || sorted_row_ids != sorted_accepted_ids
            {
                return Err("transfer row item set must exactly equal all accepted candidate IDs for the cell".to_owned());
            }
            let expected_source = cell_lineage_digest(cell_items, true)?;
            let expected_recipe = cell_lineage_digest(cell_items, false)?;
            if row.source_lineage_manifest_sha256 != expected_source
                || row.recipe_lineage_manifest_sha256 != expected_recipe
            {
                return Err("transfer source/recipe lineage digests do not bind the frozen candidate manifest".to_owned());
            }
            if row.cluster_count == 0
                || row.base.observed_count != accepted_ids.len() as u64
                || row.treatment.observed_count != accepted_ids.len() as u64
                || row.placebo.observed_count != accepted_ids.len() as u64
                || row.base.success_count > row.base.observed_count
                || row.treatment.success_count > row.treatment.observed_count
                || row.placebo.success_count > row.placebo.observed_count
            {
                return Err(
                    "transfer score counts must match the accepted candidate denominator"
                        .to_owned(),
                );
            }
            if row.adjudication_decision != RelationAdjudicationDecisionV1::Accepted {
                return Err("transfer relation row must have accepted adjudication".to_owned());
            }
            let accepted_records: Vec<&CandidateAdjudicationV1> = accepted_records_by_cell
                .get(&key)
                .cloned()
                .unwrap_or_default();
            let expected_adjudication_digest =
                accepted_cell_adjudication_digest(&accepted_records)?;
            if row.adjudication_report_sha256 != expected_adjudication_digest {
                return Err(
                    "transfer adjudication digest does not bind accepted per-candidate decisions"
                        .to_owned(),
                );
            }
            let mut reviewers = accepted_records
                .first()
                .ok_or_else(|| "accepted candidate records are missing".to_owned())?
                .reviewer_ids
                .clone();
            reviewers.sort();
            let minimum_agreement = accepted_records
                .iter()
                .filter_map(|record| record.agreement_permille)
                .min()
                .ok_or_else(|| "accepted records have no agreement values".to_owned())?;
            let mut row_reviewers = row.adjudication_reviewer_ids.clone();
            row_reviewers.sort();
            if row_reviewers != reviewers
                || accepted_records.iter().any(|record| {
                    let mut record_reviewers = record.reviewer_ids.clone();
                    record_reviewers.sort();
                    record_reviewers != reviewers
                })
                || row.minimum_item_agreement_permille != minimum_agreement
            {
                return Err(
                    "transfer cell reviewer/agreement fields differ from accepted item records"
                        .to_owned(),
                );
            }
            let signature = (
                row.item_set_sha256.clone(),
                row.source_lineage_manifest_sha256.clone(),
                row.recipe_lineage_manifest_sha256.clone(),
                row.adjudication_report_sha256.clone(),
                sorted_row_ids,
            );
            if let Some(previous) = transfer_cell_signatures.get(&key) {
                if previous.0 != signature.0
                    || previous.1 != signature.1
                    || previous.2 != signature.2
                    || previous.3 != signature.3
                    || previous.4 != signature.4
                {
                    return Err(
                        "accepted transfer cell bindings differ across model-pair replicates"
                            .to_owned(),
                    );
                }
            } else {
                transfer_cell_signatures.insert(key.clone(), signature);
            }
            let success_difference = score_success_difference(
                &row.treatment,
                &row.placebo,
                accepted_ids.len() as u64,
                "accepted treatment/placebo",
            )?;
            let candidate_count = cell_items.len() as u64;
            let (accepted_effect, lower, upper) = effect_bounds_permille(
                success_difference,
                accepted_ids.len() as u64,
                candidate_count,
            )?;
            pair_bounds
                .entry(key)
                .or_default()
                .push(AttritionReplicateBoundsV1 {
                    replicate_id: transfer_replicate.replicate_id.clone(),
                    accepted_subset_effect_permille: accepted_effect,
                    full_candidate_lower_effect_permille: lower,
                    full_candidate_upper_effect_permille: upper,
                });
        }
    }
    if transfer_ids != computed_base_ids {
        return Err(
            "transfer replicate set must exactly match causal model-pair replicate set".to_owned(),
        );
    }

    let mut cell_reports = Vec::new();
    for (key, cell_items) in &cell_candidates {
        let decision_counts = *counts
            .get(key)
            .ok_or_else(|| "candidate cell has no adjudication counts".to_owned())?;
        let accepted_ids: Vec<&str> = cell_items
            .iter()
            .filter(|candidate| {
                decisions
                    .get(&candidate.candidate_id)
                    .is_some_and(|record| {
                            record.terminal_decision == TerminalDecisionV1::Accepted
                        })
            })
            .map(|candidate| candidate.candidate_id.as_str())
            .collect();
        let per_replicate = pair_bounds.get(key).cloned().unwrap_or_default();
        if per_replicate.len() != transfer_study.replicates.len() {
            if !accepted_ids.is_empty() {
                return Err(
                    "accepted cell must have one scored outcome for every model-pair replicate"
                        .to_owned(),
                );
            }
        }
        let mut lower_values = Vec::new();
        let mut upper_values = Vec::new();
        let mut accepted_effect_values = Vec::new();
        for bound in &per_replicate {
            lower_values.push(bound.full_candidate_lower_effect_permille);
            upper_values.push(bound.full_candidate_upper_effect_permille);
            if let Some(value) = bound.accepted_subset_effect_permille {
                accepted_effect_values.push(value);
            }
        }
        let agreement_values = agreement_by_cell.get(key).cloned().unwrap_or_default();
        let n = decision_counts.total();
        if n != cell_items.len() as u64 {
            return Err(
                "adjudication outcome counts do not reconcile to candidate manifest".to_owned(),
            );
        }
        let accepted_effect_interval =
            if accepted_effect_values.len() == transfer_study.replicates.len() {
                Some(mean_interval(&accepted_effect_values)?)
            } else {
                None
            };
        let (lower_interval, upper_interval) = if per_replicate.is_empty() {
            // No accepted items: every candidate outcome is unknown. The
            // identified effect is the full binary interval [-1000, +1000].
            (
                mean_interval(&vec![-1_000.0; transfer_study.replicates.len()])?,
                mean_interval(&vec![1_000.0; transfer_study.replicates.len()])?,
            )
        } else {
            (mean_interval(&lower_values)?, mean_interval(&upper_values)?)
        };
        cell_reports.push(AttritionCellReportV1 {
            task_family: key.task_family.clone(),
            difficulty_band: key.difficulty_band.clone(),
            relation: key.relation,
            candidate_count: n,
            accepted_count: decision_counts.accepted,
            indeterminate_count: decision_counts.indeterminate,
            rejected_count: decision_counts.rejected,
            duplicate_count: decision_counts.duplicate,
            protocol_invalid_count: decision_counts.protocol_invalid,
            excluded_count: decision_counts.excluded(),
            acceptance_rate_permille: decision_counts.accepted as f64 * 1_000.0 / n as f64,
            adjudication_agreement_mean_permille: if agreement_values.is_empty() {
                None
            } else {
                Some(agreement_values.iter().sum::<f64>() / agreement_values.len() as f64)
            },
            accepted_subset_effect_permille: accepted_effect_interval,
            full_candidate_lower_effect_permille: lower_interval,
            full_candidate_upper_effect_permille: upper_interval,
            per_replicate_bounds: per_replicate,
        });
    }

    let families: BTreeSet<String> = family_bands.keys().cloned().collect();
    let mut relation_reports = Vec::new();
    for relation in TransferRelationV1::ALL {
        let relation_cells: Vec<&AttritionCellReportV1> = cell_reports
            .iter()
            .filter(|cell| cell.relation == relation)
            .collect();
        let candidate_count = relation_cells
            .iter()
            .map(|cell| cell.candidate_count)
            .sum::<u64>();
        let accepted_count = relation_cells
            .iter()
            .map(|cell| cell.accepted_count)
            .sum::<u64>();
        let indeterminate_count = relation_cells
            .iter()
            .map(|cell| cell.indeterminate_count)
            .sum::<u64>();
        let rejected_count = relation_cells
            .iter()
            .map(|cell| cell.rejected_count)
            .sum::<u64>();
        let duplicate_count = relation_cells
            .iter()
            .map(|cell| cell.duplicate_count)
            .sum::<u64>();
        let protocol_invalid_count = relation_cells
            .iter()
            .map(|cell| cell.protocol_invalid_count)
            .sum::<u64>();
        let excluded_count = relation_cells
            .iter()
            .map(|cell| cell.excluded_count)
            .sum::<u64>();
        let acceptance_rate = accepted_count as f64 * 1_000.0 / candidate_count as f64;
        let mut family_acceptance_rates = Vec::new();
        for family in &families {
            let mut band_rates = Vec::new();
            for band in family_bands
                .get(family)
                .ok_or_else(|| "family difficulty-band set missing".to_owned())?
            {
                let cell = relation_cells
                    .iter()
                    .find(|cell| cell.task_family == *family && cell.difficulty_band == *band)
                    .ok_or_else(|| {
                        "relation cell missing from macro acceptance calculation".to_owned()
                    })?;
                band_rates.push(cell.acceptance_rate_permille);
            }
            family_acceptance_rates.push(mean_simple(&band_rates)?);
        }
        let family_macro_acceptance = mean_simple(&family_acceptance_rates)?;

        let mut lower_by_replicate = Vec::new();
        let mut upper_by_replicate = Vec::new();
        for transfer_replicate in &transfer_study.replicates {
            let mut family_lower = Vec::new();
            let mut family_upper = Vec::new();
            for family in &families {
                let mut band_lower = Vec::new();
                let mut band_upper = Vec::new();
                for band in family_bands
                    .get(family)
                    .ok_or_else(|| "family difficulty-band set missing".to_owned())?
                {
                    let key = CellKey {
                        task_family: family.clone(),
                        difficulty_band: band.clone(),
                        relation,
                    };
                    let cell_bounds = pair_bounds.get(&key);
                    if let Some(bounds) = cell_bounds {
                        let pair = bounds
                            .iter()
                            .find(|bound| bound.replicate_id == transfer_replicate.replicate_id)
                            .ok_or_else(|| {
                                "replicate bound missing for relation cell".to_owned()
                            })?;
                        band_lower.push(pair.full_candidate_lower_effect_permille);
                        band_upper.push(pair.full_candidate_upper_effect_permille);
                    } else {
                        let count = cell_candidates
                            .get(&key)
                            .ok_or_else(|| "candidate relation cell missing".to_owned())?
                            .len();
                        if count == 0 {
                            return Err("empty candidate cell cannot be included in macro effect"
                                .to_owned());
                        }
                        band_lower.push(-1_000.0);
                        band_upper.push(1_000.0);
                    }
                }
                family_lower.push(mean_simple(&band_lower)?);
                family_upper.push(mean_simple(&band_upper)?);
            }
            lower_by_replicate.push(mean_simple(&family_lower)?);
            upper_by_replicate.push(mean_simple(&family_upper)?);
        }

        relation_reports.push(AttritionRelationReportV1 {
            relation,
            candidate_count,
            accepted_count,
            indeterminate_count,
            rejected_count,
            duplicate_count,
            protocol_invalid_count,
            excluded_count,
            acceptance_rate_permille: acceptance_rate,
            family_macro_acceptance_rate_permille: family_macro_acceptance,
            full_candidate_lower_effect_permille: mean_interval(&lower_by_replicate)?,
            full_candidate_upper_effect_permille: mean_interval(&upper_by_replicate)?,
        });
    }

    let candidate_count = cell_reports
        .iter()
        .map(|cell| cell.candidate_count)
        .sum::<u64>();
    let accepted_count = cell_reports
        .iter()
        .map(|cell| cell.accepted_count)
        .sum::<u64>();
    let excluded_count = cell_reports
        .iter()
        .map(|cell| cell.excluded_count)
        .sum::<u64>();
    let attrition_study_bytes = canonical_json(study, "adjudication attrition study")?;
    let attrition_study_sha256 =
        hash_domain(b"adjudication-attrition-study.v1", &attrition_study_bytes);
    let issues = vec![
        "worst-case bounds assign each excluded candidate an unknown treatment-minus-placebo effect in [-1, +1]; bounds may be wide".to_owned(),
        "accepted-subset effects can be selection-biased when adjudication attrition depends on difficulty or task properties".to_owned(),
        "reviewer pseudonyms and content digests are self-declared consistency evidence, not authenticated identities or custody".to_owned(),
        "partial-identification bounds cover the frozen pre-adjudication candidate-hypothesis universe, not a latent true-relation population".to_owned(),
        "95% intervals are replicate-level descriptive intervals; they are not simultaneous or multiplicity-adjusted".to_owned(),
        "partial-identification bounds do not repair incorrect semantic or skill labels among accepted candidates".to_owned(),
    ];

    Ok(AdjudicationAttritionReportV1 {
        schema: ATTRITION_REPORT_SCHEMA.to_owned(),
        study_id: study.study_id.clone(),
        benchmark_id: study.benchmark_id.clone(),
        benchmark_version: study.benchmark_version.clone(),
        base_causal_study_sha256: base_report.study_sha256,
        transfer_study_sha256: study.transfer_study_sha256.clone(),
        candidate_manifest_sha256: study.candidate_manifest_sha256.clone(),
        adjudication_report_sha256: study.adjudication_report_sha256.clone(),
        attrition_study_sha256,
        independent_model_pair_replicate_count: transfer_study.replicates.len() as u64,
        candidate_count,
        accepted_count,
        excluded_count,
        overall_acceptance_rate_permille: accepted_count as f64 * 1_000.0 / candidate_count as f64,
        relation_reports,
        cell_reports,
        intervals_multiplicity_adjusted: false,
        independently_authenticated: false,
        confirmatory_eligible: false,
        analysis_status: "exploratory_partial_identification".to_owned(),
        issues,
        claim_boundary: "This report quantifies adjudication attrition and computes worst-case treatment-minus-placebo effect bounds for candidates excluded from the accepted transfer cohort. It assumes each excluded binary item effect lies in [-1, +1], includes every frozen candidate in the denominator, and macro-averages difficulty bands within families then families equally. It bounds the frozen pre-adjudication candidate-hypothesis universe, not a latent population of correctly labeled relations. It does not prove reviewer independence, semantic truth, skill equivalence, training exposure, or absence of global interference; it does not impute missing outcomes or establish a causal direct effect, FDR control, external validity or SOTA superiority.".to_owned(),
    })
}

fn mean_simple(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("cannot calculate a mean from empty or non-finite values".to_owned());
    }
    let result = values.iter().sum::<f64>() / values.len() as f64;
    if result.is_finite() {
        Ok(result)
    } else {
        Err("mean is non-finite".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_candidate(id: &str, relation: TransferRelationV1) -> CandidateManifestItemV1 {
        CandidateManifestItemV1 {
            candidate_id: id.to_owned(),
            task_family: "mathematics".to_owned(),
            difficulty_band: "medium".to_owned(),
            relation,
            source_lineage_sha256: "a".repeat(64),
            recipe_lineage_sha256: "b".repeat(64),
        }
    }

    #[test]
    fn excluded_candidates_produce_worst_case_binary_bounds() {
        // One accepted item has treatment=1, placebo=0; one excluded item is
        // completely unknown. Full-candidate effect is therefore in [0, 1].
        let (accepted, lower, upper) = effect_bounds_permille(1, 1, 2).unwrap();
        assert_eq!(accepted, Some(1_000.0));
        assert_eq!(lower, 0.0);
        assert_eq!(upper, 1_000.0);
    }

    #[test]
    fn excluded_candidates_can_reverse_an_accepted_subset_effect() {
        // Accepted treatment advantage can be fully offset by excluded items.
        let (accepted, lower, upper) = effect_bounds_permille(2, 2, 6).unwrap();
        assert_eq!(accepted, Some(1_000.0));
        assert!((lower - (-333.3333333333333)).abs() < 1.0e-9);
        assert!((upper - 1_000.0).abs() < 1.0e-9);
    }

    #[test]
    fn fully_unadjudicated_cell_has_full_binary_bound() {
        let (accepted, lower, upper) = effect_bounds_permille(0, 0, 7).unwrap();
        assert_eq!(accepted, None);
        assert_eq!(lower, -1_000.0);
        assert_eq!(upper, 1_000.0);
    }

    #[test]
    fn candidate_manifest_digest_is_order_independent() {
        let a = CandidateManifestV1 {
            schema: CANDIDATE_MANIFEST_SCHEMA.to_owned(),
            study_id: "study".to_owned(),
            benchmark_id: "benchmark".to_owned(),
            benchmark_version: "v1".to_owned(),
            frozen_at_utc: "2026-11-01T10:00:00Z".to_owned(),
            candidates: vec![
                test_candidate("b", TransferRelationV1::ExactReplay),
                test_candidate("a", TransferRelationV1::SurfaceVariant),
            ],
        };
        let mut b = a.clone();
        b.candidates.reverse();
        let hash_a = hash_domain(
            b"adjudication-candidate-manifest.v1",
            &canonical_candidate_manifest(&a).unwrap(),
        );
        let hash_b = hash_domain(
            b"adjudication-candidate-manifest.v1",
            &canonical_candidate_manifest(&b).unwrap(),
        );
        assert_eq!(hash_a, hash_b);
    }

    #[test]
    fn adjudication_digest_is_order_independent_including_reviewer_order() {
        let first = CandidateAdjudicationV1 {
            candidate_id: "candidate-a".to_owned(),
            terminal_decision: TerminalDecisionV1::Accepted,
            reason_code: "consensus".to_owned(),
            reviewer_ids: vec![
                "rater-b".to_owned(),
                "rater-a".to_owned(),
                "rater-c".to_owned(),
            ],
            agreement_permille: Some(1_000),
            decided_at_utc: "2026-11-01T10:05:00Z".to_owned(),
        };
        let second = CandidateAdjudicationV1 {
            candidate_id: "candidate-b".to_owned(),
            terminal_decision: TerminalDecisionV1::Rejected,
            reason_code: "not_equivalent".to_owned(),
            reviewer_ids: vec![
                "rater-a".to_owned(),
                "rater-b".to_owned(),
                "rater-c".to_owned(),
            ],
            agreement_permille: Some(667),
            decided_at_utc: "2026-11-01T10:06:00Z".to_owned(),
        };
        let mut reordered_first = first.clone();
        reordered_first.reviewer_ids.reverse();
        let digest_a = hash_domain(
            b"adjudication-attrition-report.v1",
            &canonical_adjudication_report(&[first.clone(), second.clone()]).unwrap(),
        );
        let digest_b = hash_domain(
            b"adjudication-attrition-report.v1",
            &canonical_adjudication_report(&[second, reordered_first]).unwrap(),
        );
        assert_eq!(digest_a, digest_b);
    }

    #[test]
    fn terminal_decision_counts_keep_each_exclusion_category_separate() {
        let mut counts = DecisionCounts::default();
        for decision in [
            TerminalDecisionV1::Accepted,
            TerminalDecisionV1::Indeterminate,
            TerminalDecisionV1::Rejected,
            TerminalDecisionV1::Duplicate,
            TerminalDecisionV1::ProtocolInvalid,
        ] {
            counts.increment(&decision).unwrap();
        }
        assert_eq!(counts.total(), 5);
        assert_eq!(counts.accepted, 1);
        assert_eq!(counts.excluded(), 4);
    }

    #[test]
    fn timestamp_parser_rejects_impossible_dates_and_clocks() {
        assert!(parse_utc_seconds("2026-02-29T12:00:00Z", "test").is_err());
        assert!(parse_utc_seconds("2026-11-01T25:00:00Z", "test").is_err());
        assert!(parse_utc_seconds("2026-11-01T12:60:00Z", "test").is_err());
        assert!(parse_utc_seconds("2026-11-01T12:00:00Z", "test").is_ok());
    }

    #[test]
    fn accepted_semantic_and_skill_labels_require_unanimity() {
        for relation in [
            TransferRelationV1::SemanticEquivalent,
            TransferRelationV1::NovelRecipeSameSkill,
        ] {
            let candidate = test_candidate("candidate-1", relation);
            let record = CandidateAdjudicationV1 {
                candidate_id: "candidate-1".to_owned(),
                terminal_decision: TerminalDecisionV1::Accepted,
                reason_code: "consensus".to_owned(),
                reviewer_ids: vec!["r1".to_owned(), "r2".to_owned(), "r3".to_owned()],
                agreement_permille: Some(999),
                decided_at_utc: "2026-11-01T10:05:00Z".to_owned(),
            };
            assert!(validate_adjudication_record(&candidate, &record).is_err());
        }
    }

    #[test]
    fn indeterminate_decision_remains_a_terminal_denominator_outcome() {
        let candidate = test_candidate("candidate-1", TransferRelationV1::SemanticEquivalent);
        let record = CandidateAdjudicationV1 {
            candidate_id: "candidate-1".to_owned(),
            terminal_decision: TerminalDecisionV1::Indeterminate,
            reason_code: "semantic_disagreement".to_owned(),
            reviewer_ids: vec!["r1".to_owned(), "r2".to_owned(), "r3".to_owned()],
            agreement_permille: Some(667),
            decided_at_utc: "2026-11-01T10:05:00Z".to_owned(),
        };
        assert!(validate_adjudication_record(&candidate, &record).is_ok());
    }
}
