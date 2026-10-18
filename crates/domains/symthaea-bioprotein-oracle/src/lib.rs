// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synthetic-first decision oracle for the BIOPROTEIN research benchmark.
//!
//! This crate validates benchmark structure and declared evidence-gate semantics only.
//! It does not calculate diets, authenticate sources, establish food safety, predict
//! biological outcomes, grant regulatory authorization, or validate lifecycle claims.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

pub const RUNNER_VERSION: &str = "0.1.0";
pub const EXPECTED_CORPUS_ID: &str = "BIOPROTEIN-GATE-CORPUS-V1";
pub const EXPECTED_CORPUS_SCHEMA_VERSION: &str = "1.0.0";
const CLAIM_CEILING: &str = concat!(
    "Synthetic decision-semantics only; no biological outcome, nutritional adequacy, safety, ",
    "regulatory clearance, real cost, or lifecycle superiority is established."
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Claim {
    EvidenceEligibility,
    NutritionFeasibility,
    LcaComparison,
    ScenarioFeasibility,
    CostComparison,
    AnimalOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntendedUse {
    IngredientScreen,
    ComplementaryFoodScreen,
    CompleteDietScreen,
    FeedScreen,
    CultivatedProcessScreen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IngredientClass {
    Conventional,
    Microbial,
    Fungal,
    Algae,
    FermentationDerived,
    CultivatedCell,
    Hybrid,
    SyntheticControl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FreshnessState {
    Current,
    Stale,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    EligibleForDeclaredComparison,
    Blocked,
    Indeterminate,
    NotComparable,
    InvalidEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightsStatus {
    Permitted,
    AttributionRequired,
    Restricted,
    Unknown,
    Expired,
    Revoked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchState {
    Match,
    Mismatch,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BasisState {
    Normalized,
    MixedWithoutConversion,
    Unknown,
    IncompatibleUnits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NutritionGate {
    Pass,
    Fail,
    Unknown,
    NotRequested,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NutritionEvidenceState {
    Complete,
    Partial,
    MissingRequired,
    NotAssessed,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafetyGate {
    Pass,
    Fail,
    Unknown,
    NotAssessed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorizationGate {
    Pass,
    Fail,
    Unknown,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Availability {
    Available,
    Unavailable,
    SeasonalUnknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    Measured,
    Estimated,
    Synthetic,
    Simulated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeEvidenceState {
    NotRequested,
    Observed,
    Modeled,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LcaState {
    Complete,
    Incomplete,
    NotRequested,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceEvidence {
    pub source_id: String,
    pub rights_status: RightsStatus,
    pub permission_evidence_present: bool,
    pub attribution_required: bool,
    pub attribution_present: bool,
    pub permission_evidence_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostValue {
    pub value: f64,
    pub currency_or_unit: String,
    pub basis: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub candidate_id: String,
    pub ingredient_class: IngredientClass,
    pub source: SourceEvidence,
    pub identity_match: MatchState,
    pub target_match: MatchState,
    pub source_freshness: FreshnessState,
    pub basis_state: BasisState,
    pub nutrition_gate: NutritionGate,
    pub nutrition_evidence_state: NutritionEvidenceState,
    pub safety_gate: SafetyGate,
    pub authorization_gate: AuthorizationGate,
    pub availability: Availability,
    pub evidence_kind: EvidenceKind,
    pub reported_as: Option<EvidenceKind>,
    pub outcome_evidence_state: OutcomeEvidenceState,
    pub cost: CostValue,
    pub lca_state: Option<LcaState>,
    pub scenario_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub target_species: String,
    pub life_stage: String,
    pub intended_use: IntendedUse,
    pub requested_claim: Claim,
    pub scenario_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expected {
    pub disposition: Disposition,
    pub reason_codes: Vec<String>,
    #[serde(default)]
    pub claims_allowed: Vec<Claim>,
    #[serde(default)]
    pub claims_not_established: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseFixture {
    pub case_id: String,
    pub request: Request,
    pub candidate: Candidate,
    pub expected: Expected,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Corpus {
    pub corpus_id: String,
    pub schema_version: String,
    pub status: String,
    pub created_at: String,
    pub disclaimer: String,
    pub schema_file: String,
    pub case_schema_file: String,
    pub case_count: usize,
    pub cases: Vec<CaseFixture>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaseResult {
    pub case_id: String,
    pub status: RunStatus,
    pub expected_disposition: Disposition,
    pub actual_disposition: Disposition,
    pub expected_reason_codes: Vec<String>,
    pub actual_reason_codes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunCounts {
    pub attempted: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunReceipt {
    pub receipt_schema_version: String,
    pub runner_version: String,
    pub corpus_id: Option<String>,
    pub corpus_schema_version: Option<String>,
    pub corpus_status: Option<String>,
    pub corpus_disclaimer: Option<String>,
    pub claim_ceiling: String,
    pub git_head: Option<String>,
    pub expected_git_head: Option<String>,
    pub corpus_sha256: String,
    pub counts: RunCounts,
    pub status: RunStatus,
    pub errors: Vec<String>,
    pub cases: Vec<CaseResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleOutcome {
    pub disposition: Disposition,
    pub reason_codes: Vec<String>,
}

fn outcome(disposition: Disposition, reason: &str) -> OracleOutcome {
    OracleOutcome {
        disposition,
        reason_codes: vec![reason.to_owned()],
    }
}

fn eligible() -> OracleOutcome {
    outcome(
        Disposition::EligibleForDeclaredComparison,
        "ALL_DECLARED_SYNTHETIC_GATES_PASS",
    )
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn check_source_rights(source: &SourceEvidence) -> Option<OracleOutcome> {
    match source.rights_status {
        RightsStatus::Restricted => {
            Some(outcome(Disposition::Blocked, "RIGHTS_RESTRICTED"))
        }
        RightsStatus::Unknown => Some(outcome(Disposition::Blocked, "RIGHTS_UNKNOWN")),
        RightsStatus::Expired => Some(outcome(Disposition::Blocked, "RIGHTS_EXPIRED")),
        RightsStatus::Revoked => Some(outcome(Disposition::Blocked, "RIGHTS_REVOKED")),
        RightsStatus::Permitted => {
            if !source.permission_evidence_present
                || source
                    .permission_evidence_ref
                    .as_deref()
                    .is_none_or(|value| !nonempty(value))
            {
                Some(outcome(Disposition::Blocked, "RIGHTS_EVIDENCE_MISSING"))
            } else if source.attribution_required && !source.attribution_present {
                Some(outcome(Disposition::Blocked, "ATTRIBUTION_MISSING"))
            } else {
                None
            }
        }
        RightsStatus::AttributionRequired => {
            if !source.permission_evidence_present
                || source
                    .permission_evidence_ref
                    .as_deref()
                    .is_none_or(|value| !nonempty(value))
            {
                Some(outcome(Disposition::Blocked, "RIGHTS_EVIDENCE_MISSING"))
            } else if !source.attribution_required || !source.attribution_present {
                Some(outcome(Disposition::Blocked, "ATTRIBUTION_MISSING"))
            } else {
                None
            }
        }
    }
}

fn validate_case_shell(fixture: &CaseFixture) -> Vec<String> {
    let mut errors = Vec::new();
    if !nonempty(&fixture.case_id) {
        errors.push("CASE_ID_EMPTY".to_owned());
    }
    if !nonempty(&fixture.request.target_species)
        || !nonempty(&fixture.request.life_stage)
        || !nonempty(&fixture.request.intended_use)
        || !nonempty(&fixture.request.scenario_digest)
    {
        errors.push("REQUEST_FIELD_EMPTY".to_owned());
    }
    if !nonempty(&fixture.candidate.candidate_id)
        || !nonempty(&fixture.candidate.source.source_id)
    {
        errors.push("CANDIDATE_OR_SOURCE_ID_EMPTY".to_owned());
    }
    if !fixture.candidate.cost.value.is_finite() || fixture.candidate.cost.value < 0.0 {
        errors.push("COST_VALUE_INVALID".to_owned());
    }
    if !nonempty(&fixture.candidate.cost.currency_or_unit) {
        errors.push("COST_UNIT_EMPTY".to_owned());
    }
    if !nonempty(&fixture.candidate.scenario_digest) {
        errors.push("CANDIDATE_SCENARIO_DIGEST_EMPTY".to_owned());
    }
    if fixture.expected.reason_codes.is_empty() {
        errors.push("EXPECTED_REASON_CODES_EMPTY".to_owned());
    }
    errors
}

/// Validates metadata and structural invariants before evaluation.
/// Parsing via the typed Serde structs also rejects unknown fields and invalid enum values.
pub fn validate_corpus(corpus: &Corpus) -> Vec<String> {
    let mut errors = Vec::new();
    if corpus.corpus_id != EXPECTED_CORPUS_ID {
        errors.push("CORPUS_ID_MISMATCH".to_owned());
    }
    if corpus.schema_version != EXPECTED_CORPUS_SCHEMA_VERSION {
        errors.push("CORPUS_SCHEMA_VERSION_MISMATCH".to_owned());
    }
    if corpus.status != "synthetic_design_fixture_only" {
        errors.push("CORPUS_NOT_DECLARED_SYNTHETIC_ONLY".to_owned());
    }
    if corpus.schema_file != "bioprotein-gate-corpus-schema-v1.json" {
        errors.push("CORPUS_SCHEMA_POINTER_MISMATCH".to_owned());
    }
    if corpus.case_schema_file != "bioprotein-gate-schema-v1.json" {
        errors.push("CASE_SCHEMA_POINTER_MISMATCH".to_owned());
    }
    if corpus.cases.len() != corpus.case_count {
        errors.push("CASE_COUNT_MISMATCH".to_owned());
    }
    if corpus.cases.is_empty() {
        errors.push("CORPUS_EMPTY".to_owned());
    }

    let mut case_ids = BTreeSet::new();
    for fixture in &corpus.cases {
        if !case_ids.insert(fixture.case_id.clone()) {
            errors.push(format!("DUPLICATE_CASE_ID:{}", fixture.case_id));
        }
        let case_id_bytes = fixture.case_id.as_bytes();
        let valid_case_id = case_id_bytes.len() == "BIOP-GATE0-001".len()
            && fixture.case_id.starts_with("BIOP-GATE")
            && case_id_bytes.get(9).is_some_and(u8::is_ascii_digit)
            && case_id_bytes.get(10) == Some(&b'-')
            && case_id_bytes[11..].iter().all(u8::is_ascii_digit);
        if !valid_case_id {
            errors.push(format!("CASE_ID_FORMAT_INVALID:{}", fixture.case_id));
        }
        errors.extend(
            validate_case_shell(fixture)
                .into_iter()
                .map(|code| format!("{}:{}", fixture.case_id, code)),
        );
    }
    errors.sort();
    errors.dedup();
    errors
}

/// Applies explicit claim-specific rules. It is not a nutrient calculator or safety model.
pub fn evaluate_case(fixture: &CaseFixture) -> OracleOutcome {
    let candidate = &fixture.candidate;
    let request = &fixture.request;

    if let Some(reported) = &candidate.reported_as {
        if reported != &candidate.evidence_kind {
            return outcome(
                Disposition::InvalidEvidence,
                if candidate.evidence_kind == EvidenceKind::Simulated
                    && reported == &EvidenceKind::Measured
                {
                    "SIMULATED_REPORTED_AS_MEASURED"
                } else {
                    "EVIDENCE_KIND_MISMATCH"
                },
            );
        }
    }

    if let Some(result) = check_source_rights(&candidate.source) {
        return result;
    }

    match &candidate.source_freshness {
        FreshnessState::Current => {}
        FreshnessState::Stale => {
            return outcome(Disposition::Indeterminate, "SOURCE_STALE")
        }
        FreshnessState::Unknown => {
            return outcome(Disposition::Indeterminate, "SOURCE_FRESHNESS_UNKNOWN")
        }
    }

    match candidate.identity_match {
        MatchState::Mismatch => {
            return outcome(Disposition::Blocked, "INGREDIENT_IDENTITY_MISMATCH")
        }
        MatchState::Unknown => {
            return outcome(Disposition::Indeterminate, "INGREDIENT_IDENTITY_UNKNOWN")
        }
        MatchState::Match => {}
    }
    match candidate.target_match {
        MatchState::Mismatch => {
            return outcome(Disposition::Blocked, "TARGET_SPECIES_MISMATCH")
        }
        MatchState::Unknown => {
            return outcome(Disposition::Indeterminate, "TARGET_SPECIES_UNKNOWN")
        }
        MatchState::Match => {}
    }

    match request.requested_claim {
        Claim::EvidenceEligibility => eligible(),
        Claim::NutritionFeasibility => evaluate_nutrition(candidate),
        Claim::ScenarioFeasibility => evaluate_scenario(candidate, request),
        Claim::LcaComparison => evaluate_lca(candidate, request),
        Claim::CostComparison => evaluate_cost(candidate, request),
        Claim::AnimalOutcome => evaluate_animal_outcome(candidate),
    }
}

fn evaluate_nutrition(candidate: &Candidate) -> OracleOutcome {
    match candidate.basis_state {
        BasisState::MixedWithoutConversion => {
            return outcome(Disposition::NotComparable, "BASIS_MIXED_WITHOUT_CONVERSION")
        }
        BasisState::IncompatibleUnits => {
            return outcome(Disposition::NotComparable, "UNITS_INCOMPATIBLE")
        }
        BasisState::Unknown => return outcome(Disposition::Indeterminate, "BASIS_UNKNOWN"),
        BasisState::Normalized => {}
    }

    match candidate.nutrition_evidence_state {
        NutritionEvidenceState::MissingRequired => {
            return outcome(Disposition::Indeterminate, "REQUIRED_NUTRITION_UNKNOWN")
        }
        NutritionEvidenceState::Partial => {
            return outcome(Disposition::Indeterminate, "NUTRITION_EVIDENCE_PARTIAL")
        }
        NutritionEvidenceState::NotAssessed => {
            return outcome(Disposition::Indeterminate, "NUTRITION_EVIDENCE_NOT_ASSESSED")
        }
        NutritionEvidenceState::NotApplicable => {
            return outcome(Disposition::Indeterminate, "NUTRITION_NOT_APPLICABLE")
        }
        NutritionEvidenceState::Complete => {}
    }

    match candidate.nutrition_gate {
        NutritionGate::Fail => {
            return outcome(Disposition::Blocked, "NUTRITION_GATE_FAILED")
        }
        NutritionGate::Unknown => {
            return outcome(Disposition::Indeterminate, "NUTRITION_GATE_UNKNOWN")
        }
        NutritionGate::NotRequested => {
            return outcome(Disposition::Indeterminate, "NUTRITION_GATE_NOT_REQUESTED")
        }
        NutritionGate::Pass => {}
    }

    match candidate.safety_gate {
        SafetyGate::Fail => return outcome(Disposition::Blocked, "SAFETY_GATE_FAILED"),
        SafetyGate::NotAssessed => {
            return outcome(Disposition::Indeterminate, "SAFETY_NOT_ASSESSED")
        }
        SafetyGate::Unknown => {
            return outcome(Disposition::Indeterminate, "SAFETY_GATE_UNKNOWN")
        }
        SafetyGate::Pass => {}
    }

    match candidate.authorization_gate {
        AuthorizationGate::Fail => {
            return outcome(Disposition::Blocked, "AUTHORIZATION_FAILED")
        }
        AuthorizationGate::Unknown => {
            return outcome(Disposition::Indeterminate, "AUTHORIZATION_UNKNOWN")
        }
        AuthorizationGate::Pass | AuthorizationGate::NotApplicable => {}
    }

    match candidate.availability {
        Availability::Unavailable => {
            return outcome(Disposition::Blocked, "RESOURCE_UNAVAILABLE")
        }
        Availability::SeasonalUnknown => {
            return outcome(Disposition::Indeterminate, "SEASONAL_AVAILABILITY_UNKNOWN")
        }
        Availability::Available => {}
    }

    eligible()
}

fn evaluate_scenario(candidate: &Candidate, request: &Request) -> OracleOutcome {
    if candidate.scenario_digest != request.scenario_digest {
        return outcome(Disposition::NotComparable, "SCENARIO_DIGEST_MISMATCH");
    }
    match candidate.safety_gate {
        SafetyGate::Fail => return outcome(Disposition::Blocked, "SAFETY_GATE_FAILED"),
        SafetyGate::NotAssessed => {
            return outcome(Disposition::Indeterminate, "SAFETY_NOT_ASSESSED")
        }
        SafetyGate::Unknown => {
            return outcome(Disposition::Indeterminate, "SAFETY_GATE_UNKNOWN")
        }
        SafetyGate::Pass => {}
    }
    match candidate.authorization_gate {
        AuthorizationGate::Fail => {
            return outcome(Disposition::Blocked, "AUTHORIZATION_FAILED")
        }
        AuthorizationGate::Unknown => {
            return outcome(Disposition::Indeterminate, "AUTHORIZATION_UNKNOWN")
        }
        AuthorizationGate::Pass | AuthorizationGate::NotApplicable => {}
    }
    match candidate.availability {
        Availability::Unavailable => {
            return outcome(Disposition::Blocked, "RESOURCE_UNAVAILABLE")
        }
        Availability::SeasonalUnknown => {
            return outcome(Disposition::Indeterminate, "SEASONAL_AVAILABILITY_UNKNOWN")
        }
        Availability::Available => {}
    }
    eligible()
}

fn evaluate_lca(candidate: &Candidate, request: &Request) -> OracleOutcome {
    if candidate.scenario_digest != request.scenario_digest {
        return outcome(Disposition::NotComparable, "SCENARIO_DIGEST_MISMATCH");
    }
    match candidate.lca_state {
        Some(LcaState::Complete) => eligible(),
        Some(LcaState::Incomplete) | Some(LcaState::NotRequested) | None => {
            outcome(Disposition::NotComparable, "LCA_EVIDENCE_INCOMPLETE")
        }
    }
}

fn evaluate_cost(candidate: &Candidate, request: &Request) -> OracleOutcome {
    if candidate.scenario_digest != request.scenario_digest {
        return outcome(Disposition::NotComparable, "SCENARIO_DIGEST_MISMATCH");
    }
    if !nonempty(&candidate.cost.currency_or_unit) || !nonempty(&candidate.cost.basis) {
        return outcome(Disposition::InvalidEvidence, "INVALID_COST_BASIS");
    }
    if !candidate.cost.value.is_finite() || candidate.cost.value < 0.0 {
        return outcome(Disposition::InvalidEvidence, "COST_VALUE_INVALID");
    }
    eligible()
}

fn evaluate_animal_outcome(candidate: &Candidate) -> OracleOutcome {
    match candidate.outcome_evidence_state {
        OutcomeEvidenceState::Observed if candidate.evidence_kind == EvidenceKind::Measured => {
            eligible()
        }
        OutcomeEvidenceState::Modeled => {
            outcome(Disposition::Indeterminate, "ANIMAL_OUTCOME_NOT_MEASURED")
        }
        OutcomeEvidenceState::Missing => {
            outcome(Disposition::Indeterminate, "ANIMAL_OUTCOME_MISSING")
        }
        OutcomeEvidenceState::NotRequested => {
            outcome(Disposition::Indeterminate, "ANIMAL_OUTCOME_NOT_PROVIDED")
        }
        OutcomeEvidenceState::Observed => {
            outcome(Disposition::Indeterminate, "ANIMAL_OUTCOME_NOT_MEASURED")
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn incomplete_receipt(
    raw: &[u8],
    actual_head: Option<&str>,
    expected_head: Option<&str>,
    error: String,
) -> RunReceipt {
    RunReceipt {
        receipt_schema_version: "1.0.0".to_owned(),
        runner_version: RUNNER_VERSION.to_owned(),
        corpus_id: None,
        corpus_schema_version: None,
        corpus_status: None,
        corpus_disclaimer: None,
        claim_ceiling: CLAIM_CEILING.to_owned(),
        git_head: actual_head.map(str::to_owned),
        expected_git_head: expected_head.map(str::to_owned),
        corpus_sha256: sha256_hex(raw),
        counts: RunCounts { attempted: 0, passed: 0, failed: 0, skipped: 0 },
        status: RunStatus::Incomplete,
        errors: vec![error],
        cases: Vec::new(),
    }
}

/// Run the corpus with exact corpus bytes and caller-supplied Git HEAD identities.
/// Both HEADs are required to report pass; mismatches report fail, missing identity incomplete.
pub fn run_corpus(
    raw: &[u8],
    actual_head: Option<&str>,
    expected_head: Option<&str>,
) -> RunReceipt {
    let corpus: Corpus = match serde_json::from_slice(raw) {
        Ok(value) => value,
        Err(error) => {
            return incomplete_receipt(
                raw,
                actual_head,
                expected_head,
                format!("CORPUS_PARSE_FAILED:{error}"),
            );
        }
    };
    let validation_errors = validate_corpus(&corpus);
    if !validation_errors.is_empty() {
        return RunReceipt {
            receipt_schema_version: "1.0.0".to_owned(),
            runner_version: RUNNER_VERSION.to_owned(),
            corpus_id: Some(corpus.corpus_id),
            corpus_schema_version: Some(corpus.schema_version),
            corpus_status: Some(corpus.status),
            corpus_disclaimer: Some(corpus.disclaimer),
            claim_ceiling: CLAIM_CEILING.to_owned(),
            git_head: actual_head.map(str::to_owned),
            expected_git_head: expected_head.map(str::to_owned),
            corpus_sha256: sha256_hex(raw),
            counts: RunCounts {
                attempted: 0,
                passed: 0,
                failed: 0,
                skipped: corpus.cases.len(),
            },
            status: RunStatus::Incomplete,
            errors: validation_errors,
            cases: Vec::new(),
        };
    }

    let mut results = Vec::with_capacity(corpus.cases.len());
    for fixture in &corpus.cases {
        let actual = evaluate_case(fixture);
        let pass = actual.disposition == fixture.expected.disposition
            && actual.reason_codes == fixture.expected.reason_codes;
        results.push(CaseResult {
            case_id: fixture.case_id.clone(),
            status: if pass { RunStatus::Pass } else { RunStatus::Fail },
            expected_disposition: fixture.expected.disposition.clone(),
            actual_disposition: actual.disposition,
            expected_reason_codes: fixture.expected.reason_codes.clone(),
            actual_reason_codes: actual.reason_codes,
        });
    }
    results.sort_by(|a, b| a.case_id.cmp(&b.case_id));

    let passed = results.iter().filter(|item| item.status == RunStatus::Pass).count();
    let failed = results.len() - passed;
    let mut errors = Vec::new();
    if failed > 0 {
        errors.push(format!("EXPECTED_ACTUAL_MISMATCHES:{failed}"));
    }

    let head_mismatch = matches!(
        (actual_head, expected_head),
        (Some(actual), Some(expected)) if actual != expected
    );
    let head_missing = actual_head.is_none() || expected_head.is_none();
    if head_mismatch {
        errors.push("GIT_HEAD_MISMATCH".to_owned());
    }
    if head_missing {
        errors.push("GIT_HEAD_IDENTITY_MISSING".to_owned());
    }

    let status = if failed > 0 || head_mismatch {
        RunStatus::Fail
    } else if head_missing {
        RunStatus::Incomplete
    } else {
        RunStatus::Pass
    };

    RunReceipt {
        receipt_schema_version: "1.0.0".to_owned(),
        runner_version: RUNNER_VERSION.to_owned(),
        corpus_id: Some(corpus.corpus_id),
        corpus_schema_version: Some(corpus.schema_version),
        corpus_status: Some(corpus.status),
        corpus_disclaimer: Some(corpus.disclaimer),
        claim_ceiling: CLAIM_CEILING.to_owned(),
        git_head: actual_head.map(str::to_owned),
        expected_git_head: expected_head.map(str::to_owned),
        corpus_sha256: sha256_hex(raw),
        counts: RunCounts {
            attempted: results.len(),
            passed,
            failed,
            skipped: 0,
        },
        status,
        errors,
        cases: results,
    }
}

fn git_head() -> Option<String> {
    let output = Command::new("git").args(["rev-parse", "HEAD"]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|text| text.trim().to_owned())
        .filter(|text| !text.is_empty())
}

/// CLI invocation: cargo run -p symthaea-bioprotein-oracle -- --expected-head <sha>
pub fn run_cli() -> i32 {
    let mut args = std::env::args().skip(1);
    let mut corpus_path = PathBuf::from(
        "docs/research/bioprotein/fixtures/bioprotein-gate-corpus-v1.json",
    );
    let mut receipt_path: Option<PathBuf> = None;
    let mut expected_head: Option<String> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => match args.next() {
                Some(value) => corpus_path = PathBuf::from(value),
                None => {
                    eprintln!("missing value for --corpus");
                    return 2;
                }
            },
            "--receipt" => match args.next() {
                Some(value) => receipt_path = Some(PathBuf::from(value)),
                None => {
                    eprintln!("missing value for --receipt");
                    return 2;
                }
            },
            "--expected-head" => match args.next() {
                Some(value) => expected_head = Some(value),
                None => {
                    eprintln!("missing value for --expected-head");
                    return 2;
                }
            },
            "-h" | "--help" => {
                println!(
                    "Usage: symthaea-bioprotein-oracle --expected-head <sha> \
                     [--corpus <path>] [--receipt <path>]"
                );
                return 0;
            }
            other => {
                eprintln!("unknown argument: {other}");
                return 2;
            }
        }
    }

    let raw = match fs::read(&corpus_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("cannot read corpus {}: {error}", corpus_path.display());
            return 2;
        }
    };
    let actual_head = git_head();
    let receipt = run_corpus(&raw, actual_head.as_deref(), expected_head.as_deref());
    let output = match serde_json::to_string_pretty(&receipt) {
        Ok(json) => json + "\n",
        Err(error) => {
            eprintln!("cannot serialize receipt: {error}");
            return 2;
        }
    };

    if let Some(path) = receipt_path {
        if let Err(error) = fs::write(&path, output.as_bytes()) {
            eprintln!("cannot write receipt {}: {error}", path.display());
            return 2;
        }
        println!("receipt={}", path.display());
    } else {
        print!("{output}");
    }

    match receipt.status {
        RunStatus::Pass => 0,
        RunStatus::Fail => 1,
        RunStatus::Incomplete => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_CORPUS: &str = include_str!(
        "../../../../docs/research/bioprotein/fixtures/bioprotein-gate-corpus-v1.json"
    );

    fn fixture_corpus() -> Corpus {
        serde_json::from_str(FIXTURE_CORPUS).expect("typed synthetic corpus should parse")
    }

    #[test]
    fn frozen_corpus_dispositions_and_reason_codes_match() {
        let receipt = run_corpus(FIXTURE_CORPUS.as_bytes(), Some("head-a"), Some("head-a"));
        assert_eq!(receipt.status, RunStatus::Pass, "{:?}", receipt.errors);
        assert_eq!(receipt.counts.attempted, 29);
        assert_eq!(receipt.counts.passed, 29);
        assert_eq!(receipt.counts.failed, 0);
        assert_eq!(receipt.counts.skipped, 0);
        assert_eq!(receipt.corpus_status.as_deref(), Some("synthetic_design_fixture_only"));
        assert!(receipt.claim_ceiling.contains("no biological outcome"));
    }


    #[test]
    fn malformed_json_is_incomplete_not_pass() {
        let receipt = run_corpus(b"{", Some("head-a"), Some("head-a"));
        assert_eq!(receipt.status, RunStatus::Incomplete);
        assert_eq!(receipt.counts.attempted, 0);
        assert!(receipt.errors.iter().any(|error| error.starts_with("CORPUS_PARSE_FAILED:")));
    }

    #[test]
    fn schema_version_mismatch_is_incomplete_not_pass() {
        let mut corpus = fixture_corpus();
        corpus.schema_version = "99.0.0".to_owned();
        let raw = serde_json::to_vec(&corpus).unwrap();
        let receipt = run_corpus(&raw, Some("head-a"), Some("head-a"));
        assert_eq!(receipt.status, RunStatus::Incomplete);
        assert!(receipt.errors.iter().any(|error| error == "CORPUS_SCHEMA_VERSION_MISMATCH"));
    }

    #[test]
    fn missing_scoped_rights_evidence_fails_closed() {
        let mut corpus = fixture_corpus();
        let case = corpus.cases.iter_mut().find(|c| c.case_id == "BIOP-GATE0-001").unwrap();
        case.candidate.source.permission_evidence_present = false;
        case.candidate.source.permission_evidence_ref = None;
        let actual = evaluate_case(case);
        assert_eq!(actual.disposition, Disposition::Blocked);
        assert_eq!(actual.reason_codes, vec!["RIGHTS_EVIDENCE_MISSING".to_owned()]);
    }

    #[test]
    fn revoked_rights_override_other_positive_gates() {
        let mut corpus = fixture_corpus();
        let case = corpus.cases.iter_mut().find(|c| c.case_id == "BIOP-GATE0-001").unwrap();
        case.candidate.source.rights_status = RightsStatus::Revoked;
        let actual = evaluate_case(case);
        assert_eq!(actual.disposition, Disposition::Blocked);
        assert_eq!(actual.reason_codes, vec!["RIGHTS_REVOKED".to_owned()]);
    }

    #[test]
    fn unknown_nutrition_does_not_become_zero_or_pass() {
        let mut corpus = fixture_corpus();
        let case = corpus.cases.iter_mut().find(|c| c.case_id == "BIOP-GATE1-003").unwrap();
        let actual = evaluate_case(case);
        assert_eq!(actual.disposition, Disposition::Indeterminate);
        assert_eq!(actual.reason_codes, vec!["REQUIRED_NUTRITION_UNKNOWN".to_owned()]);
    }

    #[test]
    fn candidate_order_does_not_change_canonical_case_results() {
        let corpus = fixture_corpus();
        let original = serde_json::to_vec(&corpus).unwrap();
        let receipt_a = run_corpus(&original, Some("head-a"), Some("head-a"));
        let mut reversed = corpus;
        reversed.cases.reverse();
        let reordered = serde_json::to_vec(&reversed).unwrap();
        let receipt_b = run_corpus(&reordered, Some("head-a"), Some("head-a"));
        let ids_a: Vec<_> = receipt_a.cases.iter().map(|c| c.case_id.as_str()).collect();
        let ids_b: Vec<_> = receipt_b.cases.iter().map(|c| c.case_id.as_str()).collect();
        assert_eq!(ids_a, ids_b);
        assert_eq!(receipt_a.cases, receipt_b.cases);
        assert_eq!(receipt_a.counts, receipt_b.counts);
        assert_eq!(receipt_a.status, receipt_b.status);
    }

    #[test]
    fn wrong_head_never_passes() {
        let receipt = run_corpus(
            FIXTURE_CORPUS.as_bytes(),
            Some("actual-sha"),
            Some("expected-sha"),
        );
        assert_eq!(receipt.status, RunStatus::Fail);
        assert!(receipt.errors.iter().any(|error| error == "GIT_HEAD_MISMATCH"));
    }

    #[test]
    fn missing_head_never_passes() {
        let receipt = run_corpus(FIXTURE_CORPUS.as_bytes(), None, Some("expected-sha"));
        assert_eq!(receipt.status, RunStatus::Incomplete);
        assert!(receipt.errors.iter().any(|error| error == "GIT_HEAD_IDENTITY_MISSING"));
    }


    #[test]
    fn permitted_source_with_missing_required_attribution_is_blocked() {
        let mut corpus = fixture_corpus();
        let case = corpus
            .cases
            .iter_mut()
            .find(|c| c.case_id == "BIOP-GATE0-001")
            .unwrap();
        case.candidate.source.attribution_required = true;
        case.candidate.source.attribution_present = false;
        let actual = evaluate_case(case);
        assert_eq!(actual.disposition, Disposition::Blocked);
        assert_eq!(actual.reason_codes, vec!["ATTRIBUTION_MISSING".to_owned()]);
    }

    #[test]
    fn malformed_case_id_is_incomplete_not_pass() {
        let mut corpus = fixture_corpus();
        corpus.cases[0].case_id = "BIOP-GATEQ-01X".to_owned();
        let raw = serde_json::to_vec(&corpus).unwrap();
        let receipt = run_corpus(&raw, Some("head-a"), Some("head-a"));
        assert_eq!(receipt.status, RunStatus::Incomplete);
        assert!(
            receipt
                .errors
                .iter()
                .any(|error| error.starts_with("CASE_ID_FORMAT_INVALID:"))
        );
    }

    #[test]
    fn head_mismatch_is_reported_even_when_a_case_fails() {
        let mut corpus = fixture_corpus();
        corpus.cases[0].expected.reason_codes = vec!["WRONG_EXPECTED_REASON".to_owned()];
        let raw = serde_json::to_vec(&corpus).unwrap();
        let receipt = run_corpus(&raw, Some("actual-sha"), Some("expected-sha"));
        assert_eq!(receipt.status, RunStatus::Fail);
        assert!(receipt.errors.iter().any(|error| error == "GIT_HEAD_MISMATCH"));
        assert!(
            receipt
                .errors
                .iter()
                .any(|error| error == "EXPECTED_ACTUAL_MISMATCHES:1")
        );
    }

    #[test]
    fn duplicate_case_id_is_incomplete_not_pass() {
        let mut corpus = fixture_corpus();
        let mut duplicate = corpus.cases[0].clone();
        duplicate.case_id = corpus.cases[1].case_id.clone();
        corpus.cases.push(duplicate);
        corpus.case_count = corpus.cases.len();
        let raw = serde_json::to_vec(&corpus).unwrap();
        let receipt = run_corpus(&raw, Some("head-a"), Some("head-a"));
        assert_eq!(receipt.status, RunStatus::Incomplete);
        assert!(receipt.errors.iter().any(|error| error.starts_with("DUPLICATE_CASE_ID:")));
    }

    #[test]
    fn simulated_result_cannot_be_reported_as_measured() {
        let mut corpus = fixture_corpus();
        let case = corpus.cases.iter_mut().find(|c| c.case_id == "BIOP-GATE4-002").unwrap();
        case.candidate.evidence_kind = EvidenceKind::Simulated;
        case.candidate.reported_as = Some(EvidenceKind::Measured);
        let actual = evaluate_case(case);
        assert_eq!(actual.disposition, Disposition::InvalidEvidence);
        assert_eq!(actual.reason_codes, vec!["SIMULATED_REPORTED_AS_MEASURED".to_owned()]);
    }

    #[test]
    fn cost_basis_failure_is_scoped_to_cost_claim() {
        let mut corpus = fixture_corpus();
        let case = corpus.cases.iter_mut().find(|c| c.case_id == "BIOP-GATE1-006").unwrap();
        case.candidate.cost.basis.clear();
        let nutrition = evaluate_case(case);
        assert_eq!(nutrition.disposition, Disposition::EligibleForDeclaredComparison);
        let mut cost_case = case.clone();
        cost_case.request.requested_claim = Claim::CostComparison;
        let cost = evaluate_case(&cost_case);
        assert_eq!(cost.disposition, Disposition::InvalidEvidence);
        assert_eq!(cost.reason_codes, vec!["INVALID_COST_BASIS".to_owned()]);
    }


    #[test]
    fn cost_comparison_requires_scenario_alignment() {
        let mut corpus = fixture_corpus();
        let case = corpus
            .cases
            .iter_mut()
            .find(|c| c.case_id == "BIOP-GATE1-006")
            .unwrap();
        case.request.requested_claim = Claim::CostComparison;
        case.candidate.scenario_digest = "sha256:other-scenario".to_owned();
        let actual = evaluate_case(case);
        assert_eq!(actual.disposition, Disposition::NotComparable);
        assert_eq!(
            actual.reason_codes,
            vec!["SCENARIO_DIGEST_MISMATCH".to_owned()]
        );
    }

    #[test]
    fn missing_lca_does_not_change_the_distinct_nutrition_claim() {
        let mut corpus = fixture_corpus();
        let case = corpus.cases.iter_mut().find(|c| c.case_id == "BIOP-GATE1-006").unwrap();
        case.candidate.lca_state = Some(LcaState::Incomplete);
        let mut lca_case = case.clone();
        lca_case.request.requested_claim = Claim::LcaComparison;
        assert_eq!(evaluate_case(&lca_case).disposition, Disposition::NotComparable);
        assert_eq!(evaluate_case(case).disposition, Disposition::EligibleForDeclaredComparison);
    }
}
