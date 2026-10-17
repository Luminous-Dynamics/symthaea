// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded, deterministic combinations of modular-interior alternatives.
//!
//! This is a concept-search and audit layer, not a vessel-safety authority.
//! Every materialized combination is revalidated as a whole. A partial search
//! states that it is partial; it never claims exhaustive coverage.

use crate::modular_interior::{
    validate_configuration, verify_validation_receipt, Configuration, Finding,
    ValidationReceipt,
};
use crate::modular_interior_impact::{analyze_configuration_change, ImpactReport};
use crate::modular_interior_variants::{
    generate_drop_in_variants, ModuleVariant, VariantDisposition,
    VariantFinding, VariantGenerationReport,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const COMBINATION_SEARCH_VERSION: &str = "marine-interior-combinations-v1";
pub const HARD_MAX_INPUT_VARIANTS: usize = 128;
pub const HARD_MAX_CANDIDATES: usize = 10_000;
pub const HARD_MAX_CHANGED_MODULES: usize = 8;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SearchLimits {
    pub max_input_variants: usize,
    pub max_candidates: usize,
    pub max_changed_modules: usize,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            max_input_variants: 128,
            max_candidates: 1_000,
            max_changed_modules: 3,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CombinationSearchError {
    pub code: String,
    pub message: String,
}

impl CombinationSearchError {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self { code: code.to_owned(), message: message.into() }
    }
}

impl fmt::Display for CombinationSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for CombinationSearchError {}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CombinationDisposition {
    RejectedPreflight,
    RequiresReconfiguration,
    RejectedByValidator,
    ReviewRequired,
    CurrentChecksClear,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CombinationCandidate {
    pub combination_id: String,
    pub candidate_configuration_id: String,
    pub selected_variant_ids: Vec<String>,
    pub target_module_ids: Vec<String>,
    pub disposition: CombinationDisposition,
    pub preflight_findings: Vec<VariantFinding>,
    /// Findings absent from the baseline and every constituent single-change
    /// validation. This is a diagnostic interaction signal, not causal proof.
    pub interaction_findings: Vec<Finding>,
    pub configuration_sha256: Option<String>,
    pub configuration: Option<Configuration>,
    pub validation_receipt: Option<ValidationReceipt>,
    pub impact_report: Option<ImpactReport>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CombinationSearchReportBody {
    pub search_version: String,
    pub baseline_configuration_id: String,
    pub baseline_configuration_sha256: String,
    pub variant_input_manifest_sha256: String,
    pub limits: SearchLimits,
    pub baseline_validation_receipt: ValidationReceipt,
    pub candidate_count_lower_bound: usize,
    pub candidates: Vec<CombinationCandidate>,
    pub candidate_budget: usize,
    /// True only when all combinations within the declared cardinality scope
    /// were enumerated. False means results are a deterministic prefix only.
    pub search_complete: bool,
    pub truncated_by_candidate_cap: bool,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CombinationSearchReport {
    pub body: CombinationSearchReportBody,
    pub report_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct VariantFingerprint {
    variant_id: String,
    target_module_id: String,
    replacement_sha256: String,
}

#[derive(Clone, Debug)]
struct IndexedCombination {
    indexes: Vec<usize>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    Ok(sha256_hex(&serde_json::to_vec(value)?))
}

fn search_error_from_serialization(error: serde_json::Error) -> CombinationSearchError {
    CombinationSearchError::new("SERIALIZATION_FAILED", error.to_string())
}

fn validate_limits(
    limits: &SearchLimits,
    input_len: usize,
) -> Result<(), CombinationSearchError> {
    if limits.max_input_variants == 0 || limits.max_input_variants > HARD_MAX_INPUT_VARIANTS {
        return Err(CombinationSearchError::new(
            "INVALID_INPUT_VARIANT_LIMIT",
            format!("max_input_variants must be in 1..={HARD_MAX_INPUT_VARIANTS}"),
        ));
    }
    if input_len > limits.max_input_variants {
        return Err(CombinationSearchError::new(
            "INPUT_VARIANT_LIMIT_EXCEEDED",
            format!("received {input_len} variants; configured limit is {}", limits.max_input_variants),
        ));
    }
    if limits.max_candidates == 0 || limits.max_candidates > HARD_MAX_CANDIDATES {
        return Err(CombinationSearchError::new(
            "INVALID_CANDIDATE_BUDGET",
            format!("max_candidates must be in 1..={HARD_MAX_CANDIDATES}"),
        ));
    }
    if limits.max_changed_modules == 0 || limits.max_changed_modules > HARD_MAX_CHANGED_MODULES {
        return Err(CombinationSearchError::new(
            "INVALID_COMBINATION_CARDINALITY",
            format!("max_changed_modules must be in 1..={HARD_MAX_CHANGED_MODULES}"),
        ));
    }
    Ok(())
}

fn sorted_variants(
    variants: &[ModuleVariant],
) -> Result<(Vec<ModuleVariant>, String), CombinationSearchError> {
    let mut identified = Vec::with_capacity(variants.len());
    let mut seen_ids = BTreeSet::new();
    for variant in variants {
        if !seen_ids.insert(variant.variant_id.clone()) {
            return Err(CombinationSearchError::new(
                "DUPLICATE_VARIANT_ID",
                format!("variant ID {:?} occurs more than once", variant.variant_id),
            ));
        }
        let replacement_sha256 = digest(&variant.replacement).map_err(search_error_from_serialization)?;
        identified.push((variant.clone(), replacement_sha256));
    }
    identified.sort_by(|(a, a_sha), (b, b_sha)| {
        a.replacement.module_id.cmp(&b.replacement.module_id)
            .then(a.variant_id.cmp(&b.variant_id))
            .then(a_sha.cmp(b_sha))
    });
    let fingerprints: Vec<VariantFingerprint> = identified.iter().map(|(variant, replacement_sha256)| {
        VariantFingerprint {
            variant_id: variant.variant_id.clone(),
            target_module_id: variant.replacement.module_id.clone(),
            replacement_sha256: replacement_sha256.clone(),
        }
    }).collect();
    let manifest = digest(&fingerprints).map_err(search_error_from_serialization)?;
    Ok((identified.into_iter().map(|(variant, _)| variant).collect(), manifest))
}

fn enumerate_exact_size(
    variants: &[ModuleVariant],
    desired_size: usize,
    start: usize,
    selected: &mut Vec<usize>,
    used_targets: &mut BTreeSet<String>,
    output: &mut Vec<IndexedCombination>,
    stop_after: usize,
) {
    if output.len() >= stop_after {
        return;
    }
    if selected.len() == desired_size {
        output.push(IndexedCombination { indexes: selected.clone() });
        return;
    }
    let remaining = desired_size - selected.len();
    if variants.len().saturating_sub(start) < remaining {
        return;
    }
    for index in start..variants.len() {
        if output.len() >= stop_after {
            return;
        }
        let target = variants[index].replacement.module_id.clone();
        // A combination may select only one alternative for any module type.
        if used_targets.contains(&target) {
            continue;
        }
        used_targets.insert(target.clone());
        selected.push(index);
        enumerate_exact_size(
            variants, desired_size, index + 1, selected, used_targets, output, stop_after,
        );
        selected.pop();
        used_targets.remove(&target);
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
    let mut findings: Vec<Finding> = combined.body.findings.iter().filter(|finding| {
        let key = finding_key(finding);
        !baseline_keys.contains(&key)
            && !constituent_keys.iter().any(|keys| keys.contains(&key))
            && seen.insert(key)
    }).cloned().collect();
    findings.sort_by(|a, b| a.code.cmp(&b.code).then(a.subject.cmp(&b.subject)));
    findings
}

fn combination_id(
    baseline_sha256: &str,
    variant_ids: &[String],
    replacements: &[ModuleVariant],
) -> Result<String, CombinationSearchError> {
    let replacement_fingerprints: Vec<(String, String)> = replacements.iter().map(|variant| {
        Ok((variant.variant_id.clone(), digest(&variant.replacement)?))
    }).collect::<Result<_, serde_json::Error>>().map_err(search_error_from_serialization)?;
    let seed = digest(&(baseline_sha256, variant_ids, replacement_fingerprints))
        .map_err(search_error_from_serialization)?;
    Ok(format!("marine-combo-{seed}"))
}

fn disposition_for_preflight(reports: &[VariantGenerationReport]) -> Option<CombinationDisposition> {
    if reports.iter().any(|report| report.body.variants.iter().any(|item|
        matches!(item.disposition, VariantDisposition::RejectedPreflight | VariantDisposition::RejectedByValidator)
            && item.configuration.is_none()
    )) {
        return Some(CombinationDisposition::RejectedPreflight);
    }
    if reports.iter().any(|report| report.body.variants.iter().any(|item| {
        matches!(item.disposition, VariantDisposition::RequiresReconfiguration)
            || item.configuration.is_none()
    })) {
        return Some(CombinationDisposition::RequiresReconfiguration);
    }
    None
}

/// Runs deterministic, cardinality-first search over the provided drop-in variants.
///
/// Only one alternative per target module is allowed in each combination. Individual
/// variants must first pass the existing drop-in interface contract. Each materialized
/// combination then receives a fresh full-configuration validation and impact analysis.
/// The search is bounded by hard input/candidate/cardinality caps.
pub fn search_combinations(
    baseline: &Configuration,
    variants: &[ModuleVariant],
    limits: SearchLimits,
) -> Result<CombinationSearchReport, CombinationSearchError> {
    validate_limits(&limits, variants.len())?;
    let (variants, variant_input_manifest_sha256) = sorted_variants(variants)?;
    let baseline_sha256 = digest(baseline).map_err(search_error_from_serialization)?;
    let baseline_validation_receipt =
        validate_configuration(baseline).map_err(search_error_from_serialization)?;
    if !verify_validation_receipt(baseline, &baseline_validation_receipt) {
        return Err(CombinationSearchError::new(
            "BASELINE_RECEIPT_INVALID",
            "baseline validation receipt failed self-verification",
        ));
    }

    let mut single_reports = BTreeMap::<String, VariantGenerationReport>::new();
    for variant in &variants {
        let report = generate_drop_in_variants(baseline, std::slice::from_ref(variant))
            .map_err(search_error_from_serialization)?;
        single_reports.insert(variant.variant_id.clone(), report);
    }

    let mut all_combinations = Vec::new();
    let stop_after = limits.max_candidates.saturating_add(1);
    let max_cardinality = limits.max_changed_modules.min(variants.len());
    for size in 1..=max_cardinality {
        enumerate_exact_size(
            &variants,
            size,
            0,
            &mut Vec::new(),
            &mut BTreeSet::new(),
            &mut all_combinations,
            stop_after,
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
    let search_complete = !truncated_by_candidate_cap;

    let baseline_keys = finding_keys(&baseline_validation_receipt);
    let mut candidates = Vec::with_capacity(all_combinations.len());
    for combination in all_combinations {
        let chosen: Vec<ModuleVariant> = combination.indexes.iter()
            .map(|index| variants[*index].clone()).collect();
        let chosen_ids: Vec<String> = chosen.iter().map(|variant| variant.variant_id.clone()).collect();
        let mut target_module_ids: Vec<String> =
            chosen.iter().map(|variant| variant.replacement.module_id.clone()).collect();
        target_module_ids.sort();
        target_module_ids.dedup();

        let combo_id = combination_id(&baseline_sha256, &chosen_ids, &chosen)?;
        let candidate_configuration_id = format!(
            "LD-MODCOMBO-{}",
            &combo_id["marine-combo-".len()..combo_id.len().min("marine-combo-".len() + 48)]
        );

        let constituent_reports: Vec<VariantGenerationReport> = chosen_ids.iter()
            .filter_map(|id| single_reports.get(id).cloned()).collect();
        let preflight_findings: Vec<VariantFinding> = constituent_reports.iter()
            .flat_map(|report| report.body.variants.iter())
            .flat_map(|item| item.preflight_findings.clone()).collect();

        if let Some(disposition) = disposition_for_preflight(&constituent_reports) {
            candidates.push(CombinationCandidate {
                combination_id: combo_id,
                candidate_configuration_id,
                selected_variant_ids: chosen_ids,
                target_module_ids,
                disposition,
                preflight_findings,
                interaction_findings: Vec::new(),
                configuration_sha256: None,
                configuration: None,
                validation_receipt: None,
                impact_report: None,
            });
            continue;
        }

        let mut candidate = baseline.clone();
        for variant in &chosen {
            let matches: Vec<usize> = candidate.module_catalog.iter().enumerate()
                .filter_map(|(index, module)| (module.module_id == variant.replacement.module_id).then_some(index))
                .collect();
            if matches.len() != 1 {
                return Err(CombinationSearchError::new(
                    "TARGET_MODULE_AMBIGUOUS",
                    format!("target module {:?} resolved to {} catalog entries", variant.replacement.module_id, matches.len()),
                ));
            }
            candidate.module_catalog[matches[0]] = variant.replacement.clone();
            for placement in &mut candidate.placements {
                if placement.module_id == variant.replacement.module_id {
                    placement.module_revision = variant.replacement.revision;
                }
            }
        }
        candidate.configuration_id = candidate_configuration_id.clone();
        candidate.configuration_name = format!("{} / bounded combination", baseline.configuration_name);
        candidate.customization_intent = format!(
            "{} Bounded combination candidate; all modules require integrated vessel-specific revalidation.",
            baseline.customization_intent
        );

        let candidate_sha256 = digest(&candidate).map_err(search_error_from_serialization)?;
        let validation_receipt = validate_configuration(&candidate)
            .map_err(search_error_from_serialization)?;
        let impact_report = analyze_configuration_change(baseline, &candidate)
            .map_err(search_error_from_serialization)?;

        // An interaction finding is new relative to the baseline AND every
        // constituent single-change result. Its absence is not proof of no interaction.
        let constituent_keys: Vec<BTreeSet<(String, String)>> = chosen_ids.iter()
            .filter_map(|id| single_reports.get(id))
            .filter_map(|report| report.body.variants.iter().find_map(|item| item.validation_receipt.as_ref()))
            .map(finding_keys)
            .collect();
        let interactions = interaction_findings(&validation_receipt, &baseline_keys, &constituent_keys);
        let disposition = match validation_receipt.body.overall_disposition.as_str() {
            "rejected" => CombinationDisposition::RejectedByValidator,
            "indeterminate" => CombinationDisposition::ReviewRequired,
            "candidate_only" => CombinationDisposition::CurrentChecksClear,
            _ => CombinationDisposition::ReviewRequired,
        };
        candidates.push(CombinationCandidate {
            combination_id: combo_id,
            candidate_configuration_id,
            selected_variant_ids: chosen_ids,
            target_module_ids,
            disposition,
            preflight_findings,
            interaction_findings: interactions,
            configuration_sha256: Some(candidate_sha256),
            configuration: Some(candidate),
            validation_receipt: Some(validation_receipt),
            impact_report: Some(impact_report),
        });
    }

    let body = CombinationSearchReportBody {
        search_version: COMBINATION_SEARCH_VERSION.to_owned(),
        baseline_configuration_id: baseline.configuration_id.clone(),
        baseline_configuration_sha256: baseline_sha256,
        variant_input_manifest_sha256,
        limits: limits.clone(),
        candidate_budget: limits.max_candidates,
        candidates,
        search_complete,
        truncated_by_candidate_cap,
        claim_ceiling: concat!(
            "Bounded search over declared drop-in module alternatives only. ",
            "Interaction findings are differences from baseline and tested constituent singles, ",
            "not causal proof. Current-check dispositions do not qualify vessel safety, structural ",
            "integrity, stability, seaworthiness, regulatory compliance, or operating authority."
        ).to_owned(),
    };
    let report_sha256 = digest(&body).map_err(search_error_from_serialization)?;
    Ok(CombinationSearchReport { body, report_sha256 })
}

/// Recomputes the complete bounded search from its exact inputs and compares the body digest.
pub fn verify_combination_search_report(
    baseline: &Configuration,
    variants: &[ModuleVariant],
    limits: &SearchLimits,
    report: &CombinationSearchReport,
) -> bool {
    if digest(&report.body).ok().as_deref() != Some(report.report_sha256.as_str()) {
        return false;
    }
    let Ok(recomputed) = search_combinations(baseline, variants, limits.clone()) else {
        return false;
    };
    recomputed.report_sha256 == report.report_sha256
        && digest(&recomputed.body).ok() == digest(&report.body).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular_interior_variants::ModuleVariant;

    const LIVE: &str = include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");
    const ORACLE: &str = include_str!("../fixtures/modular-interiors/combination-search-oracle-v1.json");

    #[derive(serde::Deserialize)]
    struct SearchOracle {
        cases: Vec<SearchOracleCase>,
    }

    #[derive(serde::Deserialize)]
    struct SearchOracleCase {
        case_id: String,
        input_variant_ids: Vec<String>,
        expected_candidate_count: usize,
        expected_search_complete: bool,
        expected_candidate_sets: Vec<Vec<String>>,
        expected_pair_interaction_findings: Vec<String>,
    }

    fn live() -> Configuration {
        serde_json::from_str(LIVE).expect("live fixture parses")
    }

    fn variant(config: &Configuration, module_id: &str, variant_id: &str, ac_increase: f64) -> ModuleVariant {
        let mut replacement = config.module_catalog.iter()
            .find(|module| module.module_id == module_id)
            .expect("fixture contains requested module")
            .clone();
        replacement.revision += 1;
        replacement.name = format!("{} combination-search test variant", replacement.name);
        if ac_increase != 0.0 {
            let port = replacement.ports.iter_mut()
                .find(|port| port.port_id == "AC_IN")
                .expect("selected module has AC_IN");
            port.rating.value = Some(port.rating.value.expect("AC rating is populated") + ac_increase);
        }
        ModuleVariant { variant_id: variant_id.to_owned(), replacement }
    }

    fn default_limits() -> SearchLimits {
        SearchLimits { max_input_variants: 16, max_candidates: 100, max_changed_modules: 3 }
    }

    #[test]
    fn cardinality_first_search_is_deterministic_under_input_permutation() {
        let baseline = live();
        let a = variant(&baseline, "SLEEP-POD", "SLEEP-V2", 0.0);
        let b = variant(&baseline, "MODULE-STORAGE", "STORAGE-V2", 0.0);
        let limits = default_limits();
        let first = search_combinations(&baseline, &[a.clone(), b.clone()], limits.clone()).unwrap();
        let second = search_combinations(&baseline, &[b.clone(), a.clone()], limits.clone()).unwrap();
        assert_eq!(first.report_sha256, second.report_sha256);
        assert!(first.body.search_complete);
        assert_eq!(first.body.candidates.len(), 3);
        assert_eq!(first.body.candidates[0].selected_variant_ids.len(), 1);
        assert_eq!(first.body.candidates[1].selected_variant_ids.len(), 1);
        assert_eq!(first.body.candidates[2].selected_variant_ids.len(), 2);
        assert!(verify_combination_search_report(&baseline, &[a, b], &limits, &first));
    }

    #[test]
    fn candidate_budget_marks_prefix_incomplete_instead_of_claiming_exhaustion() {
        let baseline = live();
        let a = variant(&baseline, "SLEEP-POD", "SLEEP-V2", 0.0);
        let b = variant(&baseline, "MODULE-STORAGE", "STORAGE-V2", 0.0);
        let mut limits = default_limits();
        limits.max_candidates = 2;
        let report = search_combinations(&baseline, &[a, b], limits).unwrap();
        assert_eq!(report.body.candidates.len(), 2);
        assert_eq!(report.body.candidate_count_lower_bound, 3);
        assert!(!report.body.search_complete);
        assert!(report.body.truncated_by_candidate_cap);
    }

    #[test]
    fn combination_never_selects_two_variants_for_the_same_target_module() {
        let baseline = live();
        let a = variant(&baseline, "SLEEP-POD", "SLEEP-V2-A", 0.0);
        let b = variant(&baseline, "SLEEP-POD", "SLEEP-V2-B", 0.0);
        let report = search_combinations(&baseline, &[a, b], default_limits()).unwrap();
        assert!(report.body.candidates.iter().all(|candidate| candidate.target_module_ids.len() == 1));
        assert!(report.body.candidates.iter().all(|candidate| candidate.selected_variant_ids.len() == 1));
    }

    #[test]
    fn interface_incompatible_variant_is_not_materialized_as_a_drop_in() {
        let baseline = live();
        let mut incompatible = variant(&baseline, "SLEEP-POD", "SLEEP-PORT-CHANGE", 0.0);
        incompatible.replacement.ports[0].connector_profile = "UNAPPROVED-ALTERNATE-PROFILE".to_owned();
        let report = search_combinations(&baseline, &[incompatible], default_limits()).unwrap();
        assert_eq!(report.body.candidates[0].disposition, CombinationDisposition::RequiresReconfiguration);
        assert!(report.body.candidates[0].configuration.is_none());
    }

    #[test]
    fn combined_service_overload_is_reported_as_a_new_interaction() {
        let baseline = live();
        // Synthetic-only control: each +2,000 W change stays below the 8,000 W
        // assumed AC-bus capacity by itself; together they should exceed it.
        let galley = variant(&baseline, "COMPACT-GALLEY", "GALLEY-HIGH-LOAD", 2_000.0);
        let workspace = variant(&baseline, "FLEX-WORKSPACE", "WORKSPACE-HIGH-LOAD", 2_000.0);
        let limits = default_limits();
        let single_a = search_combinations(&baseline, &[galley.clone()], limits.clone()).unwrap();
        let single_b = search_combinations(&baseline, &[workspace.clone()], limits.clone()).unwrap();
        assert!(single_a.body.candidates[0].interaction_findings.is_empty());
        assert!(single_b.body.candidates[0].interaction_findings.is_empty());
        let combined = search_combinations(&baseline, &[galley, workspace], limits).unwrap();
        let pair = combined.body.candidates.iter()
            .find(|candidate| candidate.selected_variant_ids.len() == 2)
            .expect("pair combination generated");
        assert!(pair.validation_receipt.as_ref().expect("whole configuration validated")
            .body.findings.iter().any(|finding| finding.code == "SERVICE_CAPACITY_EXCEEDED"));
        assert!(pair.interaction_findings.iter().any(|finding|
            finding.code == "SERVICE_CAPACITY_EXCEEDED" && finding.subject == "AC-BUS"));
        assert_eq!(pair.disposition, CombinationDisposition::RejectedByValidator);
    }

    #[test]
    fn verifier_rejects_mutated_receipts_and_bad_limits_fail_closed() {
        let baseline = live();
        let a = variant(&baseline, "SLEEP-POD", "SLEEP-V2", 0.0);
        let limits = default_limits();
        let mut report = search_combinations(&baseline, &[a.clone()], limits.clone()).unwrap();
        assert!(verify_combination_search_report(&baseline, &[a.clone()], &limits, &report));
        report.body.search_complete = false;
        assert!(!verify_combination_search_report(&baseline, &[a.clone()], &limits, &report));
        let invalid_limits = SearchLimits { max_input_variants: 129, max_candidates: 1, max_changed_modules: 1 };
        assert!(search_combinations(&baseline, &[a], invalid_limits).is_err());
    }


    #[test]
    fn frozen_combination_oracle_matches_enumeration_and_interactions() {
        let oracle: SearchOracle = serde_json::from_str(ORACLE).expect("oracle fixture parses");
        for case in oracle.cases {
            let baseline = live();
            let variants = match case.case_id.as_str() {
                "CARDINALITY-FIRST-TWO-TARGETS-001" => vec![
                    variant(&baseline, "SLEEP-POD", "SLEEP-V2", 0.0),
                    variant(&baseline, "MODULE-STORAGE", "STORAGE-V2", 0.0),
                ],
                "COMBINED-AC-SERVICE-OVERLOAD-002" => vec![
                    variant(&baseline, "COMPACT-GALLEY", "GALLEY-HIGH-LOAD", 2_000.0),
                    variant(&baseline, "FLEX-WORKSPACE", "WORKSPACE-HIGH-LOAD", 2_000.0),
                ],
                other => panic!("unknown frozen oracle case: {other}"),
            };
            let ids: Vec<String> = variants.iter().map(|item| item.variant_id.clone()).collect();
            assert_eq!(ids.into_iter().collect::<BTreeSet<_>>(),
                case.input_variant_ids.into_iter().collect::<BTreeSet<_>>());
            let report = search_combinations(&baseline, &variants, default_limits()).unwrap();
            assert_eq!(report.body.candidates.len(), case.expected_candidate_count, "{}", case.case_id);
            assert_eq!(report.body.search_complete, case.expected_search_complete, "{}", case.case_id);

            let mut observed_sets: Vec<Vec<String>> = report.body.candidates.iter()
                .map(|candidate| {
                    let mut ids = candidate.selected_variant_ids.clone();
                    ids.sort();
                    ids
                }).collect();
            let mut expected_sets = case.expected_candidate_sets.clone();
            for set in &mut expected_sets { set.sort(); }
            observed_sets.sort();
            expected_sets.sort();
            assert_eq!(observed_sets, expected_sets, "{}", case.case_id);

            let observed_interactions: Vec<String> = report.body.candidates.iter()
                .find(|candidate| candidate.selected_variant_ids.len() == 2)
                .expect("oracle has a two-variant combination")
                .interaction_findings.iter()
                .map(|finding| format!("{}|{}", finding.code, finding.subject))
                .collect();
            assert_eq!(observed_interactions, case.expected_pair_interaction_findings, "{}", case.case_id);
        }
    }

    #[test]
    fn duplicate_variant_ids_are_rejected_before_search() {
        let baseline = live();
        let a = variant(&baseline, "SLEEP-POD", "DUPLICATE-ID", 0.0);
        let b = variant(&baseline, "MODULE-STORAGE", "DUPLICATE-ID", 0.0);
        let error = search_combinations(&baseline, &[a, b], default_limits()).unwrap_err();
        assert_eq!(error.code, "DUPLICATE_VARIANT_ID");
    }
}
