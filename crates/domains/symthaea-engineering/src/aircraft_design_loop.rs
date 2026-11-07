//! Dry-run aircraft engineering study admission and evaluation receipt boundary.
//!
//! This module does not run a solver, score an aircraft, infer physical truth,
//! authorize a design, or establish airworthiness. It checks that a proposed
//! evaluation is bound to a frozen study, candidate, operating case, allow-listed
//! backend, model profile and explicit cost basis before a human may review it.
//! Completed, failed, aborted, out-of-domain and non-converged evaluations are
//! all representable; none is silently converted into a feasible aircraft claim.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// Versioned wire contract for an aircraft design study.
pub const AIRCRAFT_STUDY_SCHEMA_VERSION: &str = "aircraft-design-study/0.1.0";
/// Versioned wire contract for a proposed analysis or experiment.
pub const AIRCRAFT_EVALUATION_PROPOSAL_SCHEMA_VERSION: &str =
    "aircraft-evaluation-proposal/0.1.0";
/// Versioned wire contract for a returned evaluation record.
pub const AIRCRAFT_EVALUATION_RECEIPT_SCHEMA_VERSION: &str =
    "aircraft-evaluation-receipt/0.1.0";

/// Canonically serialize a value as UTF-8 JSON with recursively sorted object
/// keys, compact separators and preserved array order, then compute SHA-256.
/// This binds the serialized payload deterministically; it does not authenticate
/// artifact sources or establish physical truth.
pub fn canonical_sha256<T: Serialize>(value: &T) -> Result<String, String> {
    let value = serde_json::to_value(value)
        .map_err(|error| format!("cannot serialize canonical payload: {error}"))?;
    let canonical = canonicalize_json(value);
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| format!("cannot encode canonical JSON: {error}"))?;
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}")
            .map_err(|error| format!("cannot encode SHA-256 digest: {error}"))?;
    }
    Ok(encoded)
}

fn canonicalize_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(canonicalize_json).collect())
        }
        serde_json::Value::Object(object) => {
            let mut entries = object.into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut sorted = serde_json::Map::new();
            for (key, value) in entries {
                sorted.insert(key, canonicalize_json(value));
            }
            serde_json::Value::Object(sorted)
        }
        scalar => scalar,
    }
}

/// Immutable reference to a separately owned artifact.
///
/// The digest identifies the exact artifact bytes claimed by the caller. This
/// module validates digest syntax and equality; it does not fetch or
/// authenticate the artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub id: String,
    pub sha256: String,
}

impl ArtifactRef {
    pub fn validate(&self, field: &str) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err(format!("{field}.id must not be empty"));
        }
        validate_sha256(&self.sha256, &format!("{field}.sha256"))
    }

    pub fn matches_sha256(&self, digest: &str) -> bool {
        self.sha256.eq_ignore_ascii_case(digest)
    }
}

/// The method family is descriptive, not a universal fidelity ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationMethod {
    AnalyticalBound,
    LowOrderNumerical,
    ExternalSolver,
    PhysicalMeasurement,
}

/// Exact backend admission and the model profiles allowed for this study.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendAdmission {
    pub backend: ArtifactRef,
    pub allowed_methods: Vec<EvaluationMethod>,
    pub model_profiles: Vec<ArtifactRef>,
}

/// Frozen study manifest. Aircraft-specific physics and artifacts remain owned
/// by their existing domain work packages; this manifest references them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AircraftStudyManifest {
    pub schema_version: String,
    pub study_id: String,
    pub mission: ArtifactRef,
    pub candidate_configurations: Vec<ArtifactRef>,
    pub required_operating_cases: Vec<ArtifactRef>,
    pub decision_questions: Vec<ArtifactRef>,
    pub objective_profile: ArtifactRef,
    pub hard_constraint_profile: ArtifactRef,
    pub allowed_backends: Vec<BackendAdmission>,
    pub cost_basis: ArtifactRef,
    /// Integer cost units have meaning only under the exact cost_basis digest.
    pub total_budget_units: u64,
    pub consumed_budget_units: u64,
    pub claim_ceiling: String,
}

impl AircraftStudyManifest {
    /// Canonical identity of the frozen manifest and every referenced digest.
    pub fn canonical_sha256(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != AIRCRAFT_STUDY_SCHEMA_VERSION {
            return Err("unsupported aircraft study schema version".to_owned());
        }
        validate_nonempty(&self.study_id, "study_id")?;
        validate_nonempty(&self.claim_ceiling, "claim_ceiling")?;
        self.mission.validate("mission")?;
        self.objective_profile.validate("objective_profile")?;
        self.hard_constraint_profile
            .validate("hard_constraint_profile")?;
        self.cost_basis.validate("cost_basis")?;

        validate_artifact_collection(
            &self.candidate_configurations,
            "candidate_configurations",
            true,
        )?;
        validate_artifact_collection(
            &self.required_operating_cases,
            "required_operating_cases",
            true,
        )?;
        validate_artifact_collection(&self.decision_questions, "decision_questions", true)?;

        if self.allowed_backends.is_empty() {
            return Err("study must admit at least one backend".to_owned());
        }
        let mut backend_digests = BTreeSet::new();
        for (index, admission) in self.allowed_backends.iter().enumerate() {
            admission
                .backend
                .validate(&format!("allowed_backends[{index}].backend"))?;
            if !backend_digests.insert(admission.backend.sha256.to_ascii_lowercase()) {
                return Err("duplicate allowed backend digest".to_owned());
            }
            if admission.allowed_methods.is_empty() {
                return Err(format!(
                    "allowed_backends[{index}] must declare at least one evaluation method"
                ));
            }
            let method_count = admission.allowed_methods.iter().collect::<BTreeSet<_>>().len();
            if method_count != admission.allowed_methods.len() {
                return Err(format!(
                    "allowed_backends[{index}] contains duplicate evaluation methods"
                ));
            }
            validate_artifact_collection(
                &admission.model_profiles,
                &format!("allowed_backends[{index}].model_profiles"),
                true,
            )?;
        }

        if self.consumed_budget_units > self.total_budget_units {
            return Err("study consumed budget exceeds total budget".to_owned());
        }
        Ok(())
    }

    /// Validate a proposal without starting a solver or experiment.
    pub fn review_proposal(
        &self,
        proposal: &AircraftEvaluationProposal,
    ) -> Result<ProposalReview, String> {
        self.validate()?;
        proposal.validate_shape()?;
        let study_sha256 = self.canonical_sha256()?;
        let proposal_sha256 = proposal.canonical_sha256()?;

        if !self.mission.matches_sha256(&proposal.mission_sha256) {
            return Err("proposal mission digest does not match frozen study".to_owned());
        }
        if !contains_digest(&self.candidate_configurations, &proposal.candidate_sha256) {
            return Err("proposal candidate is not in the frozen study".to_owned());
        }
        if !contains_digest(&self.required_operating_cases, &proposal.operating_case_sha256) {
            return Err("proposal operating case is not required by the frozen study".to_owned());
        }
        if !contains_digest(&self.decision_questions, &proposal.decision_question_sha256) {
            return Err("proposal decision question is not in the frozen study".to_owned());
        }
        if !self.cost_basis.matches_sha256(&proposal.cost_basis_sha256) {
            return Err("proposal cost basis does not match the frozen study".to_owned());
        }

        let backend = self
            .allowed_backends
            .iter()
            .find(|item| item.backend.matches_sha256(&proposal.backend_sha256))
            .ok_or_else(|| "proposal backend is not allow-listed".to_owned())?;
        if !backend.allowed_methods.contains(&proposal.method) {
            return Err("proposal method is not admitted for this backend".to_owned());
        }
        if !contains_digest(&backend.model_profiles, &proposal.model_profile_sha256) {
            return Err("proposal model profile is not admitted for this backend".to_owned());
        }

        let remaining = self
            .total_budget_units
            .saturating_sub(self.consumed_budget_units);
        let disposition = if proposal.cost_units > remaining {
            ProposalDisposition::AbstainBudgetExceeded
        } else {
            ProposalDisposition::EligibleForHumanReview
        };

        Ok(ProposalReview {
            proposal_id: proposal.proposal_id.clone(),
            study_sha256,
            proposal_sha256,
            disposition,
            execution_started: false,
            remaining_budget_units: remaining,
            requested_cost_units: proposal.cost_units,
            note: match disposition {
                ProposalDisposition::EligibleForHumanReview => {
                    "Dry-run only: eligible for human review; no backend was invoked.".to_owned()
                }
                ProposalDisposition::AbstainBudgetExceeded => {
                    "Abstain: proposal cost exceeds the remaining frozen study budget.".to_owned()
                }
            },
        })
    }

    /// Validate and classify a returned receipt. Failed and out-of-domain runs
    /// are retained as records and their actual cost is not discarded.
    pub fn review_receipt(
        &self,
        proposal: &AircraftEvaluationProposal,
        receipt: &AircraftEvaluationReceipt,
    ) -> Result<ReceiptReview, String> {
        let proposal_review = self.review_proposal(proposal)?;
        if proposal_review.disposition != ProposalDisposition::EligibleForHumanReview {
            return Err("receipt references a proposal that was not eligible for review".to_owned());
        }
        receipt.validate_shape()?;
        let study_sha256 = proposal_review.study_sha256.clone();
        let proposal_sha256 = proposal_review.proposal_sha256.clone();
        let receipt_sha256 = receipt.canonical_sha256()?;
        if receipt.proposal_id != proposal.proposal_id {
            return Err("receipt proposal identity mismatch".to_owned());
        }
        if !receipt.request_sha256.eq_ignore_ascii_case(&proposal.request_sha256) {
            return Err("receipt request digest does not match proposal".to_owned());
        }
        for (name, observed, expected) in [
            ("mission", &receipt.mission_sha256, &proposal.mission_sha256),
            ("candidate", &receipt.candidate_sha256, &proposal.candidate_sha256),
            (
                "operating case",
                &receipt.operating_case_sha256,
                &proposal.operating_case_sha256,
            ),
            ("backend", &receipt.backend_sha256, &proposal.backend_sha256),
            (
                "model profile",
                &receipt.model_profile_sha256,
                &proposal.model_profile_sha256,
            ),
            (
                "cost basis",
                &receipt.cost_basis_sha256,
                &proposal.cost_basis_sha256,
            ),
        ] {
            if !observed.eq_ignore_ascii_case(expected) {
                return Err(format!("receipt {name} identity mismatch"));
            }
        }

        let actual_remaining = self
            .total_budget_units
            .saturating_sub(self.consumed_budget_units);
        let budget_overrun = receipt.cost_units_consumed > actual_remaining;
        let decision_eligible_within_declared_scope = receipt.outcome == RunOutcome::Completed
            && receipt.applicability == ApplicabilityDisposition::WithinDeclaredDomain;

        let disposition = if budget_overrun {
            ReceiptDisposition::RecordedWithBudgetOverrun
        } else if decision_eligible_within_declared_scope {
            ReceiptDisposition::RecordedWithinDeclaredScope
        } else {
            ReceiptDisposition::RecordedNotDecisionEligible
        };

        Ok(ReceiptReview {
            receipt_id: receipt.receipt_id.clone(),
            study_sha256,
            proposal_sha256,
            receipt_sha256,
            disposition,
            actual_cost_units: receipt.cost_units_consumed,
            budget_overrun_units: receipt.cost_units_consumed.saturating_sub(actual_remaining),
            decision_eligible_within_declared_scope,
            flight_approval: false,
            note: "Receipt classification is not a feasibility finding, design release, airworthiness finding, or flight approval.".to_owned(),
        })
    }
}

/// Evaluation proposals are requests for human review, not solver commands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AircraftEvaluationProposal {
    pub schema_version: String,
    pub proposal_id: String,
    pub mission_sha256: String,
    pub candidate_sha256: String,
    pub operating_case_sha256: String,
    pub backend_sha256: String,
    pub model_profile_sha256: String,
    pub method: EvaluationMethod,
    pub request_sha256: String,
    pub decision_question_sha256: String,
    pub cost_basis_sha256: String,
    pub cost_units: u64,
    pub rationale: String,
}

impl AircraftEvaluationProposal {
    /// Canonical identity of the exact proposal payload.
    pub fn canonical_sha256(&self) -> Result<String, String> {
        self.validate_shape()?;
        canonical_sha256(self)
    }

    fn validate_shape(&self) -> Result<(), String> {
        if self.schema_version != AIRCRAFT_EVALUATION_PROPOSAL_SCHEMA_VERSION {
            return Err("unsupported aircraft evaluation proposal schema version".to_owned());
        }
        validate_nonempty(&self.proposal_id, "proposal_id")?;
        validate_nonempty(&self.rationale, "rationale")?;
        if self.cost_units == 0 {
            return Err("proposal cost_units must be > 0".to_owned());
        }
        for (field, value) in [
            ("mission_sha256", &self.mission_sha256),
            ("candidate_sha256", &self.candidate_sha256),
            ("operating_case_sha256", &self.operating_case_sha256),
            ("backend_sha256", &self.backend_sha256),
            ("model_profile_sha256", &self.model_profile_sha256),
            ("request_sha256", &self.request_sha256),
            ("decision_question_sha256", &self.decision_question_sha256),
            ("cost_basis_sha256", &self.cost_basis_sha256),
        ] {
            validate_sha256(value, field)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalDisposition {
    EligibleForHumanReview,
    AbstainBudgetExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposalReview {
    pub proposal_id: String,
    pub study_sha256: String,
    pub proposal_sha256: String,
    pub disposition: ProposalDisposition,
    pub execution_started: bool,
    pub remaining_budget_units: u64,
    pub requested_cost_units: u64,
    pub note: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunOutcome {
    Completed,
    Failed,
    Aborted,
    OutOfDomain,
    NonConverged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApplicabilityDisposition {
    WithinDeclaredDomain,
    OutsideDeclaredDomain,
    NotEstablished,
    Incomplete,
}

/// Receipt for an observed evaluation. Every outcome, including failure, must
/// carry the exact request and a digest for the retained run record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AircraftEvaluationReceipt {
    pub schema_version: String,
    pub receipt_id: String,
    pub proposal_id: String,
    pub request_sha256: String,
    pub mission_sha256: String,
    pub candidate_sha256: String,
    pub operating_case_sha256: String,
    pub backend_sha256: String,
    pub model_profile_sha256: String,
    pub run_record_sha256: String,
    pub output_sha256: String,
    pub cost_basis_sha256: String,
    pub cost_units_consumed: u64,
    pub outcome: RunOutcome,
    pub applicability: ApplicabilityDisposition,
}

impl AircraftEvaluationReceipt {
    /// Canonical identity of the exact returned receipt payload.
    pub fn canonical_sha256(&self) -> Result<String, String> {
        self.validate_shape()?;
        canonical_sha256(self)
    }

    fn validate_shape(&self) -> Result<(), String> {
        if self.schema_version != AIRCRAFT_EVALUATION_RECEIPT_SCHEMA_VERSION {
            return Err("unsupported aircraft evaluation receipt schema version".to_owned());
        }
        validate_nonempty(&self.receipt_id, "receipt_id")?;
        validate_nonempty(&self.proposal_id, "proposal_id")?;
        if self.cost_units_consumed == 0 {
            return Err("receipt cost_units_consumed must be > 0".to_owned());
        }
        for (field, value) in [
            ("request_sha256", &self.request_sha256),
            ("mission_sha256", &self.mission_sha256),
            ("candidate_sha256", &self.candidate_sha256),
            ("operating_case_sha256", &self.operating_case_sha256),
            ("backend_sha256", &self.backend_sha256),
            ("model_profile_sha256", &self.model_profile_sha256),
            ("run_record_sha256", &self.run_record_sha256),
            ("output_sha256", &self.output_sha256),
            ("cost_basis_sha256", &self.cost_basis_sha256),
        ] {
            validate_sha256(value, field)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptDisposition {
    RecordedWithinDeclaredScope,
    RecordedNotDecisionEligible,
    RecordedWithBudgetOverrun,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptReview {
    pub receipt_id: String,
    pub study_sha256: String,
    pub proposal_sha256: String,
    pub receipt_sha256: String,
    pub disposition: ReceiptDisposition,
    pub actual_cost_units: u64,
    pub budget_overrun_units: u64,
    /// Means only completed and within the declared model domain; it does not
    /// mean that the model is validated or that the aircraft is feasible.
    pub decision_eligible_within_declared_scope: bool,
    pub flight_approval: bool,
    pub note: String,
}

fn validate_artifact_collection(
    artifacts: &[ArtifactRef],
    field: &str,
    require_nonempty_collection: bool,
) -> Result<(), String> {
    if require_nonempty_collection && artifacts.is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    let mut seen = BTreeSet::new();
    for (index, artifact) in artifacts.iter().enumerate() {
        artifact.validate(&format!("{field}[{index}]"))?;
        if !seen.insert(artifact.sha256.to_ascii_lowercase()) {
            return Err(format!("{field} contains a duplicate artifact digest"));
        }
    }
    Ok(())
}

fn contains_digest(artifacts: &[ArtifactRef], digest: &str) -> bool {
    artifacts.iter().any(|item| item.matches_sha256(digest))
}

fn validate_nonempty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{field} must be 64 hexadecimal characters"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MISSION: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const CONFIG_A: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const CONFIG_B: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const CASE: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const QUESTION: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const OBJECTIVE: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const CONSTRAINTS: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const BACKEND: &str = "2222222222222222222222222222222222222222222222222222222222222222";
    const MODEL: &str = "3333333333333333333333333333333333333333333333333333333333333333";
    const COST: &str = "4444444444444444444444444444444444444444444444444444444444444444";
    const REQUEST: &str = "5555555555555555555555555555555555555555555555555555555555555555";
    const RUN: &str = "6666666666666666666666666666666666666666666666666666666666666666";
    const OUTPUT: &str = "7777777777777777777777777777777777777777777777777777777777777777";

    fn artifact(id: &str, digest: &str) -> ArtifactRef {
        ArtifactRef {
            id: id.to_owned(),
            sha256: digest.to_owned(),
        }
    }

    fn manifest() -> AircraftStudyManifest {
        AircraftStudyManifest {
            schema_version: AIRCRAFT_STUDY_SCHEMA_VERSION.to_owned(),
            study_id: "AER-01-SYNTH-STUDY".to_owned(),
            mission: artifact("mission", MISSION),
            candidate_configurations: vec![
                artifact("candidate-a", CONFIG_A),
                artifact("candidate-b", CONFIG_B),
            ],
            required_operating_cases: vec![artifact("hot-high-takeoff", CASE)],
            decision_questions: vec![artifact("partial-span-vs-conventional", QUESTION)],
            objective_profile: artifact("mission-economics", OBJECTIVE),
            hard_constraint_profile: artifact("hard-limits", CONSTRAINTS),
            allowed_backends: vec![BackendAdmission {
                backend: artifact("pinned-aero-solver", BACKEND),
                allowed_methods: vec![
                    EvaluationMethod::LowOrderNumerical,
                    EvaluationMethod::ExternalSolver,
                ],
                model_profiles: vec![artifact("aero-profile-v1", MODEL)],
            }],
            cost_basis: artifact("engineering-compute-credit", COST),
            total_budget_units: 100,
            consumed_budget_units: 20,
            claim_ceiling: "synthetic_or_research_screening_only".to_owned(),
        }
    }

    fn proposal() -> AircraftEvaluationProposal {
        AircraftEvaluationProposal {
            schema_version: AIRCRAFT_EVALUATION_PROPOSAL_SCHEMA_VERSION.to_owned(),
            proposal_id: "proposal-001".to_owned(),
            mission_sha256: MISSION.to_owned(),
            candidate_sha256: CONFIG_A.to_owned(),
            operating_case_sha256: CASE.to_owned(),
            backend_sha256: BACKEND.to_owned(),
            model_profile_sha256: MODEL.to_owned(),
            method: EvaluationMethod::ExternalSolver,
            request_sha256: REQUEST.to_owned(),
            decision_question_sha256: QUESTION.to_owned(),
            cost_basis_sha256: COST.to_owned(),
            cost_units: 25,
            rationale: "Could resolve a binding hot/high climb-power uncertainty.".to_owned(),
        }
    }

    fn receipt(outcome: RunOutcome, applicability: ApplicabilityDisposition) -> AircraftEvaluationReceipt {
        AircraftEvaluationReceipt {
            schema_version: AIRCRAFT_EVALUATION_RECEIPT_SCHEMA_VERSION.to_owned(),
            receipt_id: "receipt-001".to_owned(),
            proposal_id: "proposal-001".to_owned(),
            request_sha256: REQUEST.to_owned(),
            mission_sha256: MISSION.to_owned(),
            candidate_sha256: CONFIG_A.to_owned(),
            operating_case_sha256: CASE.to_owned(),
            backend_sha256: BACKEND.to_owned(),
            model_profile_sha256: MODEL.to_owned(),
            run_record_sha256: RUN.to_owned(),
            output_sha256: OUTPUT.to_owned(),
            cost_basis_sha256: COST.to_owned(),
            cost_units_consumed: 22,
            outcome,
            applicability,
        }
    }

    #[test]
    fn dry_run_admits_only_frozen_candidate_case_backend_model_and_cost() {
        let study = manifest();
        let review = study.review_proposal(&proposal()).expect("valid proposal");
        assert_eq!(review.disposition, ProposalDisposition::EligibleForHumanReview);
        assert!(!review.execution_started);
        assert_eq!(review.remaining_budget_units, 80);
    }

    #[test]
    fn proposal_cannot_invent_a_candidate_or_operating_case() {
        let study = manifest();
        let mut request = proposal();
        request.candidate_sha256 = "9999999999999999999999999999999999999999999999999999999999999999".to_owned();
        assert!(study.review_proposal(&request).is_err());
        request = proposal();
        request.operating_case_sha256 = "9999999999999999999999999999999999999999999999999999999999999999".to_owned();
        assert!(study.review_proposal(&request).is_err());
    }

    #[test]
    fn proposal_cannot_bypass_backend_or_model_profile_allow_list() {
        let study = manifest();
        let mut request = proposal();
        request.backend_sha256 = "9999999999999999999999999999999999999999999999999999999999999999".to_owned();
        assert!(study.review_proposal(&request).is_err());
        request = proposal();
        request.model_profile_sha256 = "9999999999999999999999999999999999999999999999999999999999999999".to_owned();
        assert!(study.review_proposal(&request).is_err());
        request = proposal();
        request.method = EvaluationMethod::PhysicalMeasurement;
        assert!(study.review_proposal(&request).is_err());
    }

    #[test]
    fn budget_exhaustion_causes_abstention_not_a_free_evaluation() {
        let mut study = manifest();
        study.consumed_budget_units = 90;
        let review = study.review_proposal(&proposal()).expect("well-formed proposal");
        assert_eq!(review.disposition, ProposalDisposition::AbstainBudgetExceeded);
        assert!(!review.execution_started);
    }

    #[test]
    fn invalid_manifest_or_cost_basis_fails_closed() {
        let mut study = manifest();
        study.consumed_budget_units = study.total_budget_units + 1;
        assert!(study.validate().is_err());
        study = manifest();
        let mut request = proposal();
        request.cost_basis_sha256 = "9999999999999999999999999999999999999999999999999999999999999999".to_owned();
        assert!(study.review_proposal(&request).is_err());
    }

    #[test]
    fn completed_in_domain_receipt_is_recorded_but_never_grants_flight_approval() {
        let study = manifest();
        let review = study
            .review_receipt(
                &proposal(),
                &receipt(RunOutcome::Completed, ApplicabilityDisposition::WithinDeclaredDomain),
            )
            .expect("valid receipt");
        assert_eq!(review.disposition, ReceiptDisposition::RecordedWithinDeclaredScope);
        assert!(review.decision_eligible_within_declared_scope);
        assert!(!review.flight_approval);
    }

    #[test]
    fn failed_and_out_of_domain_receipts_are_retained_but_not_decision_eligible() {
        let study = manifest();
        for (outcome, applicability) in [
            (RunOutcome::Failed, ApplicabilityDisposition::WithinDeclaredDomain),
            (RunOutcome::Completed, ApplicabilityDisposition::OutsideDeclaredDomain),
            (RunOutcome::NonConverged, ApplicabilityDisposition::NotEstablished),
            (RunOutcome::Aborted, ApplicabilityDisposition::Incomplete),
            (RunOutcome::OutOfDomain, ApplicabilityDisposition::OutsideDeclaredDomain),
        ] {
            let review = study
                .review_receipt(&proposal(), &receipt(outcome, applicability))
                .expect("receipt must be recorded");
            assert_eq!(review.disposition, ReceiptDisposition::RecordedNotDecisionEligible);
            assert!(!review.decision_eligible_within_declared_scope);
            assert!(!review.flight_approval);
        }
    }

    #[test]
    fn actual_cost_overrun_is_recorded_not_hidden() {
        let study = manifest();
        let mut overrun = receipt(RunOutcome::Failed, ApplicabilityDisposition::Incomplete);
        overrun.cost_units_consumed = 81;
        let review = study.review_receipt(&proposal(), &overrun).expect("overrun is a record");
        assert_eq!(review.disposition, ReceiptDisposition::RecordedWithBudgetOverrun);
        assert_eq!(review.budget_overrun_units, 1);
        assert!(!review.flight_approval);
    }
}
