// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic, bounded spatial arrangement search over explicit placement choices.
//!
//! This is not a route planner, evacuation simulator, stability solver, or vessel
//! qualification authority. Each candidate is checked by the existing full validator.

use crate::modular_interior::{
    validate_configuration, verify_validation_receipt, Configuration, Finding, Placement,
    PositionMm, ValidationReceipt,
};
use crate::modular_interior_impact::{analyze_configuration_change, ImpactReport};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const PLACEMENT_SEARCH_VERSION: &str = "marine-interior-placement-search-v1";
pub const HARD_MAX_PLACEMENT_DOMAINS: usize = 50;
pub const HARD_MAX_CHOICES_PER_DOMAIN: usize = 32;
pub const HARD_MAX_TOTAL_CHOICES: usize = 256;
pub const HARD_MAX_PLACEMENT_CANDIDATES: usize = 10_000;
pub const HARD_MAX_MOVED_INSTANCES: usize = 4;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PlacementSearchLimits {
    pub max_domains: usize,
    pub max_choices_per_domain: usize,
    pub max_total_choices: usize,
    pub max_candidates: usize,
    pub max_moved_instances: usize,
}

impl Default for PlacementSearchLimits {
    fn default() -> Self {
        Self {
            max_domains: 20,
            max_choices_per_domain: 8,
            max_total_choices: 128,
            max_candidates: 1_000,
            max_moved_instances: 3,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementSearchError {
    pub code: String,
    pub message: String,
}

impl PlacementSearchError {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self { code: code.to_owned(), message: message.into() }
    }
}

impl fmt::Display for PlacementSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for PlacementSearchError {}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementChoice {
    pub choice_id: String,
    pub deck_id: String,
    pub position_mm: PositionMm,
    pub rotation_z_deg: i16,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementDomain {
    pub instance_id: String,
    pub choices: Vec<PlacementChoice>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlacementCandidateDisposition {
    RejectedByValidator,
    ReviewRequired,
    CurrentChecksClear,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlacementCandidate {
    pub candidate_id: String,
    pub selected_choice_ids: Vec<String>,
    pub moved_instance_ids: Vec<String>,
    pub disposition: PlacementCandidateDisposition,
    /// Findings absent from the baseline and each constituent one-move candidate.
    /// This is an interaction diagnostic, not causal proof.
    pub interaction_findings: Vec<Finding>,
    pub configuration_sha256: String,
    pub configuration: Configuration,
    pub validation_receipt: ValidationReceipt,
    pub impact_report: ImpactReport,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlacementSearchReportBody {
    pub search_version: String,
    pub baseline_configuration_id: String,
    pub baseline_configuration_sha256: String,
    pub baseline_validation_receipt: ValidationReceipt,
    pub input_manifest_sha256: String,
    pub limits: PlacementSearchLimits,
    pub candidate_budget: usize,
    pub candidate_count_lower_bound: usize,
    pub search_complete: bool,
    pub truncated_by_candidate_cap: bool,
    pub candidates: Vec<PlacementCandidate>,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PlacementSearchReport {
    pub body: PlacementSearchReportBody,
    pub report_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct ChoiceFingerprint {
    choice_id: String,
    deck_id: String,
    x_mm: i64,
    y_mm: i64,
    z_mm: i64,
    rotation_z_deg: i16,
}

#[derive(Clone, Debug, Serialize)]
struct DomainFingerprint {
    instance_id: String,
    choices: Vec<ChoiceFingerprint>,
}

#[derive(Clone, Debug)]
struct IndexedChoice {
    domain_index: usize,
    choice_index: usize,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    Ok(sha256_hex(&serde_json::to_vec(value)?))
}

fn serialization_error(error: serde_json::Error) -> PlacementSearchError {
    PlacementSearchError::new("SERIALIZATION_FAILED", error.to_string())
}

fn stable_id(id: &str) -> bool {
    (3..=80).contains(&id.len())
        && id.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
        })
}

fn placement_matches_choice(placement: &Placement, choice: &PlacementChoice) -> bool {
    placement.deck_id == choice.deck_id
        && placement.position_mm.x_mm == choice.position_mm.x_mm
        && placement.position_mm.y_mm == choice.position_mm.y_mm
        && placement.position_mm.z_mm == choice.position_mm.z_mm
        && placement.rotation_z_deg == choice.rotation_z_deg
}

fn choice_fingerprint(choice: &PlacementChoice) -> ChoiceFingerprint {
    ChoiceFingerprint {
        choice_id: choice.choice_id.clone(),
        deck_id: choice.deck_id.clone(),
        x_mm: choice.position_mm.x_mm,
        y_mm: choice.position_mm.y_mm,
        z_mm: choice.position_mm.z_mm,
        rotation_z_deg: choice.rotation_z_deg,
    }
}

fn normalize_domains(
    baseline: &Configuration,
    domains: &[PlacementDomain],
    limits: &PlacementSearchLimits,
) -> Result<(Vec<PlacementDomain>, String), PlacementSearchError> {
    if limits.max_domains == 0 || limits.max_domains > HARD_MAX_PLACEMENT_DOMAINS {
        return Err(PlacementSearchError::new(
            "INVALID_DOMAIN_LIMIT",
            format!("max_domains must be in 1..={HARD_MAX_PLACEMENT_DOMAINS}"),
        ));
    }
    if domains.is_empty() || domains.len() > limits.max_domains {
        return Err(PlacementSearchError::new(
            "DOMAIN_COUNT_OUT_OF_RANGE",
            format!("received {} placement domains; configured maximum is {}", domains.len(), limits.max_domains),
        ));
    }
    if limits.max_choices_per_domain == 0
        || limits.max_choices_per_domain > HARD_MAX_CHOICES_PER_DOMAIN
    {
        return Err(PlacementSearchError::new(
            "INVALID_CHOICES_PER_DOMAIN_LIMIT",
            format!("max_choices_per_domain must be in 1..={HARD_MAX_CHOICES_PER_DOMAIN}"),
        ));
    }
    if limits.max_total_choices == 0 || limits.max_total_choices > HARD_MAX_TOTAL_CHOICES {
        return Err(PlacementSearchError::new(
            "INVALID_TOTAL_CHOICE_LIMIT",
            format!("max_total_choices must be in 1..={HARD_MAX_TOTAL_CHOICES}"),
        ));
    }
    if limits.max_candidates == 0 || limits.max_candidates > HARD_MAX_PLACEMENT_CANDIDATES {
        return Err(PlacementSearchError::new(
            "INVALID_CANDIDATE_BUDGET",
            format!("max_candidates must be in 1..={HARD_MAX_PLACEMENT_CANDIDATES}"),
        ));
    }
    if limits.max_moved_instances == 0 || limits.max_moved_instances > HARD_MAX_MOVED_INSTANCES {
        return Err(PlacementSearchError::new(
            "INVALID_MOVE_CARDINALITY",
            format!("max_moved_instances must be in 1..={HARD_MAX_MOVED_INSTANCES}"),
        ));
    }

    let mut normalized = Vec::with_capacity(domains.len());
    let mut seen_instances = BTreeSet::new();
    let mut seen_choices = BTreeSet::new();
    let mut total_choices = 0usize;

    for domain in domains {
        if !stable_id(&domain.instance_id) {
            return Err(PlacementSearchError::new(
                "INVALID_INSTANCE_ID",
                format!("instance ID {:?} is not a stable identifier", domain.instance_id),
            ));
        }
        if !seen_instances.insert(domain.instance_id.clone()) {
            return Err(PlacementSearchError::new(
                "DUPLICATE_INSTANCE_DOMAIN",
                format!("instance {:?} has multiple placement domains", domain.instance_id),
            ));
        }
        let matches: Vec<&Placement> = baseline.placements.iter()
            .filter(|placement| placement.instance_id == domain.instance_id).collect();
        if matches.len() != 1 {
            return Err(PlacementSearchError::new(
                "INSTANCE_NOT_UNIQUE",
                format!("baseline instance {:?} resolves to {} placements", domain.instance_id, matches.len()),
            ));
        }
        if domain.choices.is_empty() || domain.choices.len() > limits.max_choices_per_domain {
            return Err(PlacementSearchError::new(
                "CHOICE_COUNT_OUT_OF_RANGE",
                format!("instance {:?} has {} choices; configured maximum is {}", domain.instance_id, domain.choices.len(), limits.max_choices_per_domain),
            ));
        }
        total_choices = total_choices.saturating_add(domain.choices.len());
        let current = matches[0];
        let mut choices = domain.choices.clone();
        let mut seen_positions = BTreeSet::new();
        for choice in &choices {
            if !stable_id(&choice.choice_id) {
                return Err(PlacementSearchError::new(
                    "INVALID_CHOICE_ID",
                    format!("choice ID {:?} is not a stable identifier", choice.choice_id),
                ));
            }
            if !seen_choices.insert(choice.choice_id.clone()) {
                return Err(PlacementSearchError::new(
                    "DUPLICATE_CHOICE_ID",
                    format!("choice ID {:?} appears more than once", choice.choice_id),
                ));
            }
            let fingerprint = (
                choice.deck_id.clone(),
                choice.position_mm.x_mm,
                choice.position_mm.y_mm,
                choice.position_mm.z_mm,
                choice.rotation_z_deg,
            );
            if !seen_positions.insert(fingerprint) {
                return Err(PlacementSearchError::new(
                    "DUPLICATE_PLACEMENT_CHOICE",
                    format!("instance {:?} has duplicate destination placements", domain.instance_id),
                ));
            }
            if placement_matches_choice(current, choice) {
                return Err(PlacementSearchError::new(
                    "NO_OP_PLACEMENT_CHOICE",
                    format!("choice {:?} repeats the baseline placement", choice.choice_id),
                ));
            }
        }
        choices.sort_by(|a, b| a.choice_id.cmp(&b.choice_id)
            .then(a.deck_id.cmp(&b.deck_id))
            .then(a.position_mm.x_mm.cmp(&b.position_mm.x_mm))
            .then(a.position_mm.y_mm.cmp(&b.position_mm.y_mm))
            .then(a.position_mm.z_mm.cmp(&b.position_mm.z_mm))
            .then(a.rotation_z_deg.cmp(&b.rotation_z_deg)));
        normalized.push(PlacementDomain {
            instance_id: domain.instance_id.clone(),
            choices,
        });
    }
    if total_choices > limits.max_total_choices {
        return Err(PlacementSearchError::new(
            "TOTAL_CHOICE_LIMIT_EXCEEDED",
            format!("received {total_choices} choices; configured maximum is {}", limits.max_total_choices),
        ));
    }
    normalized.sort_by(|a, b| a.instance_id.cmp(&b.instance_id));

    let fingerprints: Vec<DomainFingerprint> = normalized.iter().map(|domain| DomainFingerprint {
        instance_id: domain.instance_id.clone(),
        choices: domain.choices.iter().map(choice_fingerprint).collect(),
    }).collect();
    let manifest = digest(&fingerprints).map_err(serialization_error)?;
    Ok((normalized, manifest))
}

fn enumerate_exact_size(
    domains: &[PlacementDomain],
    desired_size: usize,
    start_domain: usize,
    selected: &mut Vec<IndexedChoice>,
    output: &mut Vec<Vec<IndexedChoice>>,
    stop_after: usize,
) {
    if output.len() >= stop_after {
        return;
    }
    if selected.len() == desired_size {
        output.push(selected.clone());
        return;
    }
    let remaining = desired_size - selected.len();
    if domains.len().saturating_sub(start_domain) < remaining {
        return;
    }
    for domain_index in start_domain..domains.len() {
        if output.len() >= stop_after {
            return;
        }
        for choice_index in 0..domains[domain_index].choices.len() {
            selected.push(IndexedChoice { domain_index, choice_index });
            enumerate_exact_size(
                domains, desired_size, domain_index + 1, selected, output, stop_after,
            );
            selected.pop();
            if output.len() >= stop_after {
                return;
            }
        }
    }
}

fn finding_key(finding: &Finding) -> (String, String) {
    (finding.code.clone(), finding.subject.clone())
}

fn finding_keys(receipt: &ValidationReceipt) -> BTreeSet<(String, String)> {
    receipt.body.findings.iter().map(finding_key).collect()
}

fn interaction_findings(
    combined: &ValidationReceipt,
    baseline_keys: &BTreeSet<(String, String)>,
    constituent_keys: &[BTreeSet<(String, String)>],
) -> Vec<Finding> {
    let mut seen = BTreeSet::new();
    let mut output: Vec<Finding> = combined.body.findings.iter().filter(|finding| {
        let key = finding_key(finding);
        !baseline_keys.contains(&key)
            && !constituent_keys.iter().any(|keys| keys.contains(&key))
            && seen.insert(key)
    }).cloned().collect();
    output.sort_by(|a, b| a.code.cmp(&b.code).then(a.subject.cmp(&b.subject)));
    output
}

/// Searches finite placement alternatives for distinct instances. Every materialized
/// combination is revalidated; a "current checks clear" result is not safety approval.
pub fn search_placements(
    baseline: &Configuration,
    input_domains: &[PlacementDomain],
    limits: PlacementSearchLimits,
) -> Result<PlacementSearchReport, PlacementSearchError> {
    let (domains, input_manifest_sha256) =
        normalize_domains(baseline, input_domains, &limits)?;
    let baseline_sha256 = digest(baseline).map_err(serialization_error)?;
    let baseline_validation_receipt =
        validate_configuration(baseline).map_err(serialization_error)?;
    if !verify_validation_receipt(baseline, &baseline_validation_receipt) {
        return Err(PlacementSearchError::new(
            "BASELINE_RECEIPT_INVALID",
            "baseline validation receipt failed self-verification",
        ));
    }

    let mut single_reports = BTreeMap::<String, ValidationReceipt>::new();
    for domain in &domains {
        for choice in &domain.choices {
            let mut candidate = baseline.clone();
            let Some(placement) = candidate.placements.iter_mut()
                .find(|placement| placement.instance_id == domain.instance_id)
            else {
                return Err(PlacementSearchError::new(
                    "INSTANCE_DISAPPEARED",
                    format!("instance {:?} disappeared after domain validation", domain.instance_id),
                ));
            };
            placement.deck_id = choice.deck_id.clone();
            placement.position_mm = choice.position_mm.clone();
            placement.rotation_z_deg = choice.rotation_z_deg;
            let receipt = validate_configuration(&candidate).map_err(serialization_error)?;
            single_reports.insert(choice.choice_id.clone(), receipt);
        }
    }

    let stop_after = limits.max_candidates.saturating_add(1);
    let mut all_combinations = Vec::new();
    for size in 1..=limits.max_moved_instances.min(domains.len()) {
        enumerate_exact_size(
            &domains, size, 0, &mut Vec::new(), &mut all_combinations, stop_after,
        );
        if all_combinations.len() >= stop_after {
            break;
        }
    }
    let truncated_by_candidate_cap = all_combinations.len() > limits.max_candidates;
    let candidate_count_lower_bound = if truncated_by_candidate_cap {
        limits.max_candidates.saturating_add(1)
    } else {
        all_combinations.len()
    };
    if truncated_by_candidate_cap {
        all_combinations.truncate(limits.max_candidates);
    }

    let baseline_keys = finding_keys(&baseline_validation_receipt);
    let mut candidates = Vec::with_capacity(all_combinations.len());
    for combination in all_combinations {
        let mut selected_choice_ids = Vec::with_capacity(combination.len());
        let mut moved_instance_ids = Vec::with_capacity(combination.len());
        let mut choice_digests = Vec::with_capacity(combination.len());
        for indexed in &combination {
            let domain = &domains[indexed.domain_index];
            let choice = &domain.choices[indexed.choice_index];
            selected_choice_ids.push(choice.choice_id.clone());
            moved_instance_ids.push(domain.instance_id.clone());
            choice_digests.push((choice.choice_id.clone(), digest(choice).map_err(serialization_error)?));
        }

        let seed = digest(&(baseline_sha256.clone(), choice_digests))
            .map_err(serialization_error)?;
        let candidate_id = format!("place-{seed}");
        let config_suffix = &seed[..48];
        let candidate_configuration_id = format!("LD-PLACEMENT-{config_suffix}");
        let mut candidate = baseline.clone();
        candidate.configuration_id = candidate_configuration_id.clone();
        candidate.configuration_name = format!("{} / placement search", baseline.configuration_name);
        candidate.customization_intent = format!(
            "{} Placement candidate generated by bounded search; complete vessel-specific review remains mandatory.",
            baseline.customization_intent
        );
        for indexed in &combination {
            let domain = &domains[indexed.domain_index];
            let choice = &domain.choices[indexed.choice_index];
            let Some(placement) = candidate.placements.iter_mut()
                .find(|placement| placement.instance_id == domain.instance_id)
            else {
                return Err(PlacementSearchError::new(
                    "INSTANCE_DISAPPEARED",
                    format!("instance {:?} disappeared during materialization", domain.instance_id),
                ));
            };
            placement.deck_id = choice.deck_id.clone();
            placement.position_mm = choice.position_mm.clone();
            placement.rotation_z_deg = choice.rotation_z_deg;
        }

        let configuration_sha256 = digest(&candidate).map_err(serialization_error)?;
        let validation_receipt = validate_configuration(&candidate).map_err(serialization_error)?;
        let impact_report = analyze_configuration_change(baseline, &candidate)
            .map_err(serialization_error)?;
        let constituent_keys: Vec<BTreeSet<(String, String)>> = selected_choice_ids.iter()
            .filter_map(|choice_id| single_reports.get(choice_id))
            .map(finding_keys)
            .collect();
        let interactions = interaction_findings(&validation_receipt, &baseline_keys, &constituent_keys);
        let disposition = match validation_receipt.body.overall_disposition.as_str() {
            "rejected" => PlacementCandidateDisposition::RejectedByValidator,
            "indeterminate" => PlacementCandidateDisposition::ReviewRequired,
            "candidate_only" => PlacementCandidateDisposition::CurrentChecksClear,
            _ => PlacementCandidateDisposition::ReviewRequired,
        };
        candidates.push(PlacementCandidate {
            candidate_id,
            selected_choice_ids,
            moved_instance_ids,
            disposition,
            interaction_findings: interactions,
            configuration_sha256,
            configuration: candidate,
            validation_receipt,
            impact_report,
        });
    }

    let search_complete = !truncated_by_candidate_cap;
    let body = PlacementSearchReportBody {
        search_version: PLACEMENT_SEARCH_VERSION.to_owned(),
        baseline_configuration_id: baseline.configuration_id.clone(),
        baseline_configuration_sha256: baseline_sha256,
        baseline_validation_receipt,
        input_manifest_sha256,
        limits: limits.clone(),
        candidate_budget: limits.max_candidates,
        candidate_count_lower_bound,
        search_complete,
        truncated_by_candidate_cap,
        candidates,
        claim_ceiling: concat!(
            "Bounded search over explicit placements of declared module instances. ",
            "Geometry and declared constraints are screened by the current validator; this does ",
            "not compute service routes, evacuation performance, hydrostatics, intact/damage stability, ",
            "structural qualification, regulatory compliance, seaworthiness, or operating authority."
        ).to_owned(),
    };
    let report_sha256 = digest(&body).map_err(serialization_error)?;
    Ok(PlacementSearchReport { body, report_sha256 })
}

/// Recomputes the exact bounded search. Digest equality checks integrity/reproducibility,
/// not signer identity, external timestamps, or provenance authenticity.
pub fn verify_placement_search_report(
    baseline: &Configuration,
    domains: &[PlacementDomain],
    limits: &PlacementSearchLimits,
    report: &PlacementSearchReport,
) -> bool {
    if digest(&report.body).ok().as_deref() != Some(report.report_sha256.as_str()) {
        return false;
    }
    let Ok(recomputed) = search_placements(baseline, domains, limits.clone()) else {
        return false;
    };
    recomputed.report_sha256 == report.report_sha256
        && digest(&recomputed.body).ok() == digest(&report.body).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIVE: &str = include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");
    const ORACLE: &str = include_str!("../fixtures/modular-interiors/placement-search-oracle-v1.json");

    #[derive(Deserialize)]
    struct PlacementOracle {
        cases: Vec<PlacementOracleCase>,
    }

    #[derive(Deserialize)]
    struct PlacementOracleCase {
        case_id: String,
        expected_candidate_count: usize,
        expected_search_complete: bool,
        expected_candidate_sets: Vec<Vec<String>>,
        expected_pair_interaction_findings: Vec<String>,
    }

    fn live() -> Configuration {
        serde_json::from_str(LIVE).expect("live fixture parses")
    }

    fn choice(choice_id: &str, x_mm: i64, y_mm: i64, rotation_z_deg: i16) -> PlacementChoice {
        PlacementChoice {
            choice_id: choice_id.to_owned(),
            deck_id: "MAIN".to_owned(),
            position_mm: PositionMm { x_mm, y_mm, z_mm: 0 },
            rotation_z_deg,
        }
    }

    fn domains() -> Vec<PlacementDomain> {
        vec![
            PlacementDomain {
                instance_id: "CABIN-01".to_owned(),
                choices: vec![choice("CABIN-X600", 600, 1100, 0)],
            },
            PlacementDomain {
                instance_id: "WETROOM-01".to_owned(),
                choices: vec![choice("WETROOM-X2900", 2900, 1100, 0)],
            },
        ]
    }

    fn limits() -> PlacementSearchLimits {
        PlacementSearchLimits {
            max_domains: 8,
            max_choices_per_domain: 8,
            max_total_choices: 32,
            max_candidates: 100,
            max_moved_instances: 3,
        }
    }

    #[test]
    fn deterministic_under_domain_and_choice_permutation() {
        let baseline = live();
        let mut original = domains();
        original[0].choices.push(choice("CABIN-ROT90", 400, 1100, 90));
        let mut permuted = original.clone();
        permuted.reverse();
        for domain in &mut permuted { domain.choices.reverse(); }
        let first = search_placements(&baseline, &original, limits()).unwrap();
        let second = search_placements(&baseline, &permuted, limits()).unwrap();
        assert_eq!(first.report_sha256, second.report_sha256);
        assert!(verify_placement_search_report(&baseline, &original, &limits(), &first));
    }

    #[test]
    fn pairwise_placement_overlap_is_found_only_after_composition() {
        let baseline = live();
        let report = search_placements(&baseline, &domains(), limits()).unwrap();
        assert_eq!(report.body.candidates.len(), 3);
        let singles: Vec<&PlacementCandidate> = report.body.candidates.iter()
            .filter(|candidate| candidate.selected_choice_ids.len() == 1).collect();
        assert!(singles.iter().all(|candidate| !candidate.validation_receipt.body.findings
            .iter().any(|finding| finding.code == "MODULE_PHYSICAL_OVERLAP")));
        let pair = report.body.candidates.iter()
            .find(|candidate| candidate.selected_choice_ids.len() == 2).expect("pair candidate exists");
        assert!(pair.interaction_findings.iter().any(|finding|
            finding.code == "MODULE_PHYSICAL_OVERLAP" && finding.subject == "CABIN-01 / WETROOM-01"));
        assert_eq!(pair.disposition, PlacementCandidateDisposition::RejectedByValidator);
    }

    #[test]
    fn cap_is_explicit_and_reports_a_candidate_count_lower_bound() {
        let baseline = live();
        let mut inputs = domains();
        inputs[0].choices.push(choice("CABIN-X700", 700, 1100, 0));
        let mut cap = limits();
        cap.max_candidates = 2;
        let report = search_placements(&baseline, &inputs, cap).unwrap();
        assert_eq!(report.body.candidates.len(), 2);
        assert_eq!(report.body.candidate_count_lower_bound, 3);
        assert!(!report.body.search_complete);
        assert!(report.body.truncated_by_candidate_cap);
    }

    #[test]
    fn duplicate_domains_and_noop_choices_fail_closed() {
        let baseline = live();
        let mut inputs = domains();
        inputs.push(inputs[0].clone());
        assert_eq!(search_placements(&baseline, &inputs, limits()).unwrap_err().code,
            "DUPLICATE_INSTANCE_DOMAIN");

        let noop = PlacementDomain {
            instance_id: "CABIN-01".to_owned(),
            choices: vec![choice("CABIN-NOOP", 400, 1100, 0)],
        };
        assert_eq!(search_placements(&baseline, &[noop], limits()).unwrap_err().code,
            "NO_OP_PLACEMENT_CHOICE");
    }

    #[test]
    fn invalid_rotation_fails_in_the_canonical_validator() {
        let baseline = live();
        let invalid = vec![PlacementDomain {
            instance_id: "FLEX-01".to_owned(),
            choices: vec![choice("FLEX-ROT45", 7400, 1100, 45)],
        }];
        let report = search_placements(&baseline, &invalid, limits()).unwrap();
        assert_eq!(report.body.candidates[0].disposition, PlacementCandidateDisposition::RejectedByValidator);
        assert!(report.body.candidates[0].validation_receipt.body.findings.iter()
            .any(|finding| finding.code == "INVALID_PLACEMENT_GEOMETRY"));
    }

    #[test]
    fn frozen_placement_oracle_matches_candidate_sets_and_interaction() {
        let oracle: PlacementOracle = serde_json::from_str(ORACLE).expect("placement oracle parses");
        for case in oracle.cases {
            assert_eq!(case.case_id, "TWO-MODULE-CLOSING-GAP-001");
            let report = search_placements(&live(), &domains(), limits()).unwrap();
            assert_eq!(report.body.candidates.len(), case.expected_candidate_count);
            assert_eq!(report.body.search_complete, case.expected_search_complete);
            let mut observed: Vec<Vec<String>> = report.body.candidates.iter().map(|candidate| {
                let mut ids = candidate.selected_choice_ids.clone();
                ids.sort();
                ids
            }).collect();
            let mut expected = case.expected_candidate_sets;
            for ids in &mut expected { ids.sort(); }
            observed.sort();
            expected.sort();
            assert_eq!(observed, expected);
            let pair = report.body.candidates.iter()
                .find(|candidate| candidate.selected_choice_ids.len() == 2).unwrap();
            let observed_findings: Vec<String> = pair.interaction_findings.iter()
                .map(|finding| format!("{}|{}", finding.code, finding.subject)).collect();
            assert_eq!(observed_findings, case.expected_pair_interaction_findings);
        }
    }
}
