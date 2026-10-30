//! Adaptive-evaluation exposure policy for quantum benchmark claims.
//!
//! This module separates exploratory tuning from confirmatory evidence. It
//! authorizes development, budgeted validation, and one-shot confirmation on
//! a fresh sealed family set. It does not implement a reusable-holdout privacy
//! mechanism, secure storage, authentication, signatures, or sequential tests.
//! Callers must persist reservations atomically in an independently protected
//! append-only ledger before exposing a benchmark result.

/// Versioned schema label for adaptive evaluation reservations.
pub const ADAPTIVE_EVALUATION_SCHEMA: &str =
    "symthaea.quantum_comp.adaptive_evaluation.v1";

/// Evaluation lane; the lane determines what a result is permitted to support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationLane {
    /// Public or unrestricted exploration. Never confirmatory.
    Development,
    /// Controlled feedback with a finite query budget. Never confirmatory.
    Validation,
    /// One-shot evaluation of a frozen candidate on a never-before-used family set.
    Confirmatory,
    /// Sequential evaluation. Rejected until a specific anytime-valid protocol is implemented.
    Sequential,
}

/// Disposition returned by the policy gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationDisposition {
    /// The result may guide development, but cannot support a confirmatory claim.
    DevelopmentOnly,
    /// The result may guide tuning within the declared validation exposure budget.
    ValidationOnly,
    /// The request is eligible for one confirmatory evaluation; it is not a result or a pass.
    ConfirmatoryOneShotEligible,
}

/// Frozen comparison protocol and sealed-set policy for one claim family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveEvaluationPlan {
    /// Stable identifier for the complete preregistered claim family.
    pub claim_family_id: String,
    /// Digest of the complete comparison and statistical analysis protocol.
    pub comparison_protocol_digest: String,
    /// Digest of the fixed classical baseline and its tuning/resource budget.
    pub baseline_digest: String,
    /// Digest of the independent output-quality and feasibility verifier.
    pub acceptance_verifier_digest: String,
    /// Digest of the sealed independent-family set reserved for confirmation.
    pub confirmatory_family_set_digest: String,
    /// Digest of the frozen candidate manifest (code, configuration and parameters).
    pub frozen_candidate_manifest_digest: String,
    /// Maximum validation evaluations permitted per family-set digest.
    pub max_validation_queries_per_family_set: usize,
}

/// A proposed evaluation event. All digests are opaque identifiers; this module
/// does not compute or authenticate them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveEvaluationRequest {
    /// Unique identifier for this evaluation reservation.
    pub evaluation_id: String,
    /// Claim family to which this request belongs.
    pub claim_family_id: String,
    /// Requested evaluation lane.
    pub lane: EvaluationLane,
    /// Digest of the candidate manifest or preregistered candidate batch.
    pub candidate_manifest_digest: String,
    /// Optional parent candidate digest establishing the tuning lineage.
    pub parent_candidate_manifest_digest: Option<String>,
    /// Digest of independent workload-family IDs, not individual instance rows.
    pub benchmark_family_set_digest: String,
    /// Comparison protocol digest; must match the frozen plan.
    pub comparison_protocol_digest: String,
    /// Classical baseline and matched tuning-budget digest; must match the plan.
    pub baseline_digest: String,
    /// Independent acceptance-verifier digest; must match the plan.
    pub acceptance_verifier_digest: String,
    /// One-based ordinal for this lane, claim family and benchmark-family set.
    pub query_ordinal: usize,
}

/// Persisted reservation record. Append it before running the evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveEvaluationRecord {
    /// Original reservation request.
    pub request: AdaptiveEvaluationRequest,
}

/// Authorization returned by the deterministic policy gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveEvaluationAuthorization {
    /// Allowed use of the resulting evaluation.
    pub disposition: EvaluationDisposition,
    /// Ledger revision observed during authorization. Append must compare-and-swap
    /// against this revision to prevent concurrent double reservations.
    pub expected_ledger_revision: usize,
    /// Earlier reservations in the same lane, claim family and family set.
    pub prior_scope_queries: usize,
    /// True only for a confirmatory request with no prior exposure of its family set.
    pub confirmatory_family_set_unexposed: bool,
}

/// Fail-closed policy and history errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveEvaluationError {
    /// A required plan field is blank or its query budget is zero.
    InvalidPlan,
    /// A required identifier or digest is blank.
    EmptyIdentifier,
    /// The request does not match the frozen comparison protocol, baseline or verifier.
    ContractMismatch,
    /// The evaluation identifier has already been reserved.
    DuplicateEvaluationId,
    /// Query ordinal is not the next ordinal for this scope.
    InvalidQueryOrdinal,
    /// The validation query budget for this family set is exhausted.
    ValidationBudgetExhausted,
    /// A non-confirmatory request attempted to access the sealed confirmatory set.
    SealedFamilySetReserved,
    /// Confirmatory request references a family set other than the sealed set.
    ConfirmatoryFamilySetMismatch,
    /// The confirmatory family set was already touched by any prior evaluation.
    ConfirmatoryFamilySetAlreadyExposed,
    /// Candidate manifest differs from the preregistered frozen candidate manifest.
    CandidateManifestNotFrozen,
    /// Parent candidate digest is absent from the prior candidate history.
    UnknownCandidateParent,
    /// Sequential mode requires a registered anytime-valid inference method.
    SequentialProtocolNotRegistered,
    /// Historical reservation data is malformed.
    MalformedHistory,
}

impl std::fmt::Display for AdaptiveEvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidPlan => "adaptive evaluation plan is incomplete or invalid",
            Self::EmptyIdentifier => "required identifier or digest is blank",
            Self::ContractMismatch => "request differs from the frozen comparison contract",
            Self::DuplicateEvaluationId => "evaluation ID has already been reserved",
            Self::InvalidQueryOrdinal => "query ordinal is not the next ordinal for this scope",
            Self::ValidationBudgetExhausted => "validation query budget is exhausted",
            Self::SealedFamilySetReserved => "sealed confirmatory family set is reserved",
            Self::ConfirmatoryFamilySetMismatch => {
                "confirmatory request does not use the plan's sealed family set"
            }
            Self::ConfirmatoryFamilySetAlreadyExposed => {
                "confirmatory family set has prior evaluation exposure"
            }
            Self::CandidateManifestNotFrozen => {
                "candidate manifest differs from the frozen confirmation candidate"
            }
            Self::UnknownCandidateParent => {
                "candidate parent digest is absent from prior candidate history"
            }
            Self::SequentialProtocolNotRegistered => {
                "sequential evaluation requires a registered anytime-valid method"
            }
            Self::MalformedHistory => "adaptive evaluation history is malformed",
        };
        f.write_str(message)
    }
}

impl std::error::Error for AdaptiveEvaluationError {}

fn is_nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn validate_plan(plan: &AdaptiveEvaluationPlan) -> Result<(), AdaptiveEvaluationError> {
    if !is_nonempty(&plan.claim_family_id)
        || !is_nonempty(&plan.comparison_protocol_digest)
        || !is_nonempty(&plan.baseline_digest)
        || !is_nonempty(&plan.acceptance_verifier_digest)
        || !is_nonempty(&plan.confirmatory_family_set_digest)
        || !is_nonempty(&plan.frozen_candidate_manifest_digest)
        || plan.max_validation_queries_per_family_set == 0
    {
        return Err(AdaptiveEvaluationError::InvalidPlan);
    }
    Ok(())
}

fn validate_request_ids(
    request: &AdaptiveEvaluationRequest,
) -> Result<(), AdaptiveEvaluationError> {
    if !is_nonempty(&request.evaluation_id)
        || !is_nonempty(&request.claim_family_id)
        || !is_nonempty(&request.candidate_manifest_digest)
        || !is_nonempty(&request.benchmark_family_set_digest)
        || !is_nonempty(&request.comparison_protocol_digest)
        || !is_nonempty(&request.baseline_digest)
        || !is_nonempty(&request.acceptance_verifier_digest)
    {
        return Err(AdaptiveEvaluationError::EmptyIdentifier);
    }
    if let Some(parent) = &request.parent_candidate_manifest_digest {
        if !is_nonempty(parent) || parent == &request.candidate_manifest_digest {
            return Err(AdaptiveEvaluationError::EmptyIdentifier);
        }
    }
    if request.query_ordinal == 0 {
        return Err(AdaptiveEvaluationError::InvalidQueryOrdinal);
    }
    Ok(())
}

fn validate_history(
    history: &[AdaptiveEvaluationRecord],
) -> Result<(), AdaptiveEvaluationError> {
    let mut evaluation_ids = std::collections::HashSet::with_capacity(history.len());
    for record in history {
        validate_request_ids(&record.request)
            .map_err(|_| AdaptiveEvaluationError::MalformedHistory)?;
        if !evaluation_ids.insert(record.request.evaluation_id.as_str()) {
            return Err(AdaptiveEvaluationError::MalformedHistory);
        }
    }
    Ok(())
}

/// Authorize one proposed evaluation against the frozen plan and reservation history.
///
/// Development and validation results are explicitly non-confirmatory. Validation
/// queries are counted per claim-family/family-set pair. Confirmatory evaluation
/// requires the exact frozen candidate manifest and a family-set digest that has
/// never appeared in the supplied history, regardless of lane or claim family.
///
/// The caller must append the returned request as a reservation using an atomic
/// compare-and-swap at expected_ledger_revision before exposing the benchmark.
/// This pure function cannot prevent races, data disclosure, ledger rollback, or
/// a compromised evaluator. The caller must also ensure the family-set digest
/// faithfully commits to the true independent family IDs.
pub fn authorize_adaptive_evaluation(
    plan: &AdaptiveEvaluationPlan,
    history: &[AdaptiveEvaluationRecord],
    request: &AdaptiveEvaluationRequest,
) -> Result<AdaptiveEvaluationAuthorization, AdaptiveEvaluationError> {
    validate_plan(plan)?;
    validate_history(history)?;
    validate_request_ids(request)?;

    if request.claim_family_id != plan.claim_family_id
        || request.comparison_protocol_digest != plan.comparison_protocol_digest
        || request.baseline_digest != plan.baseline_digest
        || request.acceptance_verifier_digest != plan.acceptance_verifier_digest
    {
        return Err(AdaptiveEvaluationError::ContractMismatch);
    }

    if history
        .iter()
        .any(|record| record.request.evaluation_id == request.evaluation_id)
    {
        return Err(AdaptiveEvaluationError::DuplicateEvaluationId);
    }

    if let Some(parent) = &request.parent_candidate_manifest_digest {
        let parent_known = history.iter().any(|record| {
            record.request.claim_family_id == request.claim_family_id
                && record.request.candidate_manifest_digest == *parent
        });
        if !parent_known {
            return Err(AdaptiveEvaluationError::UnknownCandidateParent);
        }
    }

    let same_scope_count = history
        .iter()
        .filter(|record| {
            record.request.claim_family_id == request.claim_family_id
                && record.request.benchmark_family_set_digest
                    == request.benchmark_family_set_digest
                && record.request.lane == request.lane
        })
        .count();

    match request.lane {
        EvaluationLane::Development => {
            if request.benchmark_family_set_digest == plan.confirmatory_family_set_digest {
                return Err(AdaptiveEvaluationError::SealedFamilySetReserved);
            }
            if request.query_ordinal != same_scope_count + 1 {
                return Err(AdaptiveEvaluationError::InvalidQueryOrdinal);
            }
            Ok(AdaptiveEvaluationAuthorization {
                disposition: EvaluationDisposition::DevelopmentOnly,
                expected_ledger_revision: history.len(),
                prior_scope_queries: same_scope_count,
                confirmatory_family_set_unexposed: false,
            })
        }
        EvaluationLane::Validation => {
            if request.benchmark_family_set_digest == plan.confirmatory_family_set_digest {
                return Err(AdaptiveEvaluationError::SealedFamilySetReserved);
            }
            if same_scope_count >= plan.max_validation_queries_per_family_set {
                return Err(AdaptiveEvaluationError::ValidationBudgetExhausted);
            }
            if request.query_ordinal != same_scope_count + 1 {
                return Err(AdaptiveEvaluationError::InvalidQueryOrdinal);
            }
            Ok(AdaptiveEvaluationAuthorization {
                disposition: EvaluationDisposition::ValidationOnly,
                expected_ledger_revision: history.len(),
                prior_scope_queries: same_scope_count,
                confirmatory_family_set_unexposed: false,
            })
        }
        EvaluationLane::Confirmatory => {
            if request.benchmark_family_set_digest != plan.confirmatory_family_set_digest {
                return Err(AdaptiveEvaluationError::ConfirmatoryFamilySetMismatch);
            }
            if request.candidate_manifest_digest != plan.frozen_candidate_manifest_digest {
                return Err(AdaptiveEvaluationError::CandidateManifestNotFrozen);
            }
            if request.query_ordinal != 1 {
                return Err(AdaptiveEvaluationError::InvalidQueryOrdinal);
            }
            let family_set_seen = history.iter().any(|record| {
                record.request.benchmark_family_set_digest
                    == plan.confirmatory_family_set_digest
            });
            if family_set_seen {
                return Err(AdaptiveEvaluationError::ConfirmatoryFamilySetAlreadyExposed);
            }
            Ok(AdaptiveEvaluationAuthorization {
                disposition: EvaluationDisposition::ConfirmatoryOneShotEligible,
                expected_ledger_revision: history.len(),
                prior_scope_queries: same_scope_count,
                confirmatory_family_set_unexposed: true,
            })
        }
        EvaluationLane::Sequential => Err(AdaptiveEvaluationError::SequentialProtocolNotRegistered),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> AdaptiveEvaluationPlan {
        AdaptiveEvaluationPlan {
            claim_family_id: "qarch-017-claim-family".to_owned(),
            comparison_protocol_digest: "protocol-v1".to_owned(),
            baseline_digest: "baseline-v1".to_owned(),
            acceptance_verifier_digest: "verifier-v1".to_owned(),
            confirmatory_family_set_digest: "sealed-families-v1".to_owned(),
            frozen_candidate_manifest_digest: "frozen-candidate-v3".to_owned(),
            max_validation_queries_per_family_set: 2,
        }
    }

    fn request(
        id: &str,
        lane: EvaluationLane,
        set: &str,
        ordinal: usize,
    ) -> AdaptiveEvaluationRequest {
        AdaptiveEvaluationRequest {
            evaluation_id: id.to_owned(),
            claim_family_id: "qarch-017-claim-family".to_owned(),
            lane,
            candidate_manifest_digest: "candidate-v1".to_owned(),
            parent_candidate_manifest_digest: None,
            benchmark_family_set_digest: set.to_owned(),
            comparison_protocol_digest: "protocol-v1".to_owned(),
            baseline_digest: "baseline-v1".to_owned(),
            acceptance_verifier_digest: "verifier-v1".to_owned(),
            query_ordinal: ordinal,
        }
    }

    fn record(request: AdaptiveEvaluationRequest) -> AdaptiveEvaluationRecord {
        AdaptiveEvaluationRecord { request }
    }

    #[test]
    fn development_is_never_confirmatory() {
        let request = request("dev-1", EvaluationLane::Development, "public-v1", 1);
        let result = authorize_adaptive_evaluation(&plan(), &[], &request).unwrap();
        assert_eq!(result.disposition, EvaluationDisposition::DevelopmentOnly);
        assert!(!result.confirmatory_family_set_unexposed);
    }

    #[test]
    fn validation_is_non_confirmatory_and_budgeted() {
        let plan = plan();
        let first = request("val-1", EvaluationLane::Validation, "validation-v1", 1);
        let second = request("val-2", EvaluationLane::Validation, "validation-v1", 2);
        let first_auth = authorize_adaptive_evaluation(&plan, &[], &first).unwrap();
        assert_eq!(first_auth.disposition, EvaluationDisposition::ValidationOnly);
        let history = vec![record(first), record(second)];
        let third = request("val-3", EvaluationLane::Validation, "validation-v1", 3);
        assert_eq!(
            authorize_adaptive_evaluation(&plan, &history, &third),
            Err(AdaptiveEvaluationError::ValidationBudgetExhausted)
        );
    }

    #[test]
    fn validation_requires_monotone_query_ordinals() {
        let history = vec![record(request(
            "val-1",
            EvaluationLane::Validation,
            "validation-v1",
            1,
        ))];
        let skipped = request("val-3", EvaluationLane::Validation, "validation-v1", 3);
        assert_eq!(
            authorize_adaptive_evaluation(&plan(), &history, &skipped),
            Err(AdaptiveEvaluationError::InvalidQueryOrdinal)
        );
    }

    #[test]
    fn untouched_sealed_family_set_allows_frozen_candidate_once() {
        let mut candidate = request(
            "confirm-1",
            EvaluationLane::Confirmatory,
            "sealed-families-v1",
            1,
        );
        candidate.candidate_manifest_digest = "frozen-candidate-v3".to_owned();
        let auth = authorize_adaptive_evaluation(&plan(), &[], &candidate).unwrap();
        assert_eq!(
            auth.disposition,
            EvaluationDisposition::ConfirmatoryOneShotEligible
        );
        assert!(auth.confirmatory_family_set_unexposed);
        assert_eq!(auth.expected_ledger_revision, 0);
    }

    #[test]
    fn development_cannot_consume_the_sealed_confirmatory_set() {
        let request = request(
            "dev-sealed",
            EvaluationLane::Development,
            "sealed-families-v1",
            1,
        );
        assert_eq!(
            authorize_adaptive_evaluation(&plan(), &[], &request),
            Err(AdaptiveEvaluationError::SealedFamilySetReserved)
        );
    }

    #[test]
    fn validation_cannot_consume_the_sealed_confirmatory_set() {
        let request = request(
            "val-sealed",
            EvaluationLane::Validation,
            "sealed-families-v1",
            1,
        );
        assert_eq!(
            authorize_adaptive_evaluation(&plan(), &[], &request),
            Err(AdaptiveEvaluationError::SealedFamilySetReserved)
        );
    }

    #[test]
    fn changed_candidate_cannot_enter_confirmatory_lane() {
        let request = request(
            "confirm-changed",
            EvaluationLane::Confirmatory,
            "sealed-families-v1",
            1,
        );
        assert_eq!(
            authorize_adaptive_evaluation(&plan(), &[], &request),
            Err(AdaptiveEvaluationError::CandidateManifestNotFrozen)
        );
    }

    #[test]
    fn any_prior_use_exposes_a_confirmatory_family_set() {
        let prior = request(
            "dev-prior",
            EvaluationLane::Development,
            "sealed-families-v1",
            1,
        );
        let mut confirm = request(
            "confirm-after-exposure",
            EvaluationLane::Confirmatory,
            "sealed-families-v1",
            1,
        );
        confirm.candidate_manifest_digest = "frozen-candidate-v3".to_owned();
        assert_eq!(
            authorize_adaptive_evaluation(&plan(), &[record(prior)], &confirm),
            Err(AdaptiveEvaluationError::ConfirmatoryFamilySetAlreadyExposed)
        );
    }

    #[test]
    fn baseline_protocol_or_verifier_drift_is_rejected() {
        let mut changed = request("bad-contract", EvaluationLane::Validation, "validation-v1", 1);
        changed.baseline_digest = "retuned-baseline".to_owned();
        assert_eq!(
            authorize_adaptive_evaluation(&plan(), &[], &changed),
            Err(AdaptiveEvaluationError::ContractMismatch)
        );
    }

    #[test]
    fn candidate_parent_must_exist_in_prior_lineage() {
        let mut child = request("child", EvaluationLane::Validation, "validation-v1", 1);
        child.parent_candidate_manifest_digest = Some("unknown-parent".to_owned());
        assert_eq!(
            authorize_adaptive_evaluation(&plan(), &[], &child),
            Err(AdaptiveEvaluationError::UnknownCandidateParent)
        );
    }

    #[test]
    fn duplicate_evaluation_ids_are_rejected() {
        let prior = request("duplicate", EvaluationLane::Development, "public-v1", 1);
        let current = request("duplicate", EvaluationLane::Validation, "validation-v1", 1);
        assert_eq!(
            authorize_adaptive_evaluation(&plan(), &[record(prior)], &current),
            Err(AdaptiveEvaluationError::DuplicateEvaluationId)
        );
    }

    #[test]
    fn sequential_lane_fails_closed_without_registered_method() {
        let request = request("seq-1", EvaluationLane::Sequential, "validation-v1", 1);
        assert_eq!(
            authorize_adaptive_evaluation(&plan(), &[], &request),
            Err(AdaptiveEvaluationError::SequentialProtocolNotRegistered)
        );
    }

    #[test]
    fn malformed_history_fails_closed() {
        let mut malformed = request("bad-history", EvaluationLane::Development, "public-v1", 0);
        malformed.evaluation_id = " ".to_owned();
        let current = request("current", EvaluationLane::Development, "public-v1", 1);
        assert_eq!(
            authorize_adaptive_evaluation(&plan(), &[record(malformed)], &current),
            Err(AdaptiveEvaluationError::MalformedHistory)
        );
    }

    #[test]
    fn exact_illustrative_adaptive_overfit_fixture_is_reproducible() {
        // Under a fair null, 12 or more wins among 16 independent signs has
        // p = 2517 / 65536. Selecting the best of 50 independent candidates
        // makes the same nominal threshold fire with probability 1-(1-p)^50.
        let tail = 2517.0_f64 / 65_536.0;
        let familywise = 1.0 - (1.0 - tail).powi(50);
        assert!((tail - 0.038_406_372_070_312_5).abs() < 1e-15);
        assert!((familywise - 0.858_883_161_177_509_1).abs() < 1e-12);
    }
}
