//! Versioned, deterministic decision receipts for replayable ICCB runs.
//!
//! The input manifest is hashed byte-for-byte with its own declared encoding version.
//! This module does not canonicalize arbitrary manifest formats or claim its contents
//! are true. Its own versioned binary encoding applies to scenario, output and receipt
//! fields. Verification re-evaluates the supplied scenario using this evaluator. A
//! valid receipt is not spending authority.

use crate::{
    CandidateScenario, EvaluationReceipt, EvidenceState, OptionEvaluation, RejectionReason,
};
use crate::uncertainty_contract::{CompatibleAnalysisRoute, UncertaintyRoutingReceipt};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;

pub const DECISION_RECEIPT_SCHEMA: &str = "iccb-decision-receipt-v1";
pub const CANONICALIZATION_VERSION: &str = "iccb-canonical-binary-v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionRunContext {
    /// Pinned producer commit; accepts 40- or 64-character hexadecimal Git IDs.
    pub producer_commit: String,
    /// Stable evaluator/package version.
    pub evaluator_version: String,
    /// Version/encoding of the supplied manifest bytes; declared, not independently validated.
    pub input_manifest_encoding_version: String,
    /// Exact bytes of that manifest. These bytes are hashed verbatim.
    pub input_manifest_bytes: Vec<u8>,
    /// Must be derived from these same manifest bytes via UncertaintyAssessmentRef::route.
    pub routing: UncertaintyRoutingReceipt,
    /// Explicit ceiling on what the receipt's result may claim.
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionReceipt {
    pub schema_version: String,
    pub canonicalization_version: String,
    pub input_manifest_encoding_version: String,
    pub upstream_producer_version: String,
    pub scenario_id: String,
    pub evaluation_day: u32,
    pub producer_commit: String,
    pub evaluator_version: String,
    pub input_manifest_sha256: String,
    pub scenario_sha256: String,
    pub uncertainty_route: CompatibleAnalysisRoute,
    pub evidence_reference: String,
    pub output_sha256: String,
    pub claim_ceiling: String,
    pub limitation: String,
    /// Commitment to every preceding receipt field, excluding this field.
    pub receipt_sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionReceiptError {
    MissingScenarioId,
    MissingProducerCommit,
    MalformedProducerCommit,
    MissingEvaluatorVersion,
    EmptyInputManifest,
    MissingInputManifestEncodingVersion,
    MissingUpstreamProducerVersion,
    MissingClaimCeiling,
    RoutingScenarioMismatch,
    MissingRoutingEvidence,
    RoutingManifestDigestMismatch,
    EmptyOptionId,
    DuplicateOptionId,
    EmptyEvidenceId,
    DuplicateEvidenceId,
    DuplicateRejectionReason,
    ReceiptMismatch,
}

/// Return the lowercase hexadecimal SHA-256 digest of the exact supplied bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(64);
    for byte in digest {
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

fn valid_git_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn digest_matches(reported: &str, computed: &str) -> bool {
    reported
        .strip_prefix("sha256:")
        .unwrap_or(reported)
        .eq_ignore_ascii_case(computed)
}

/// Append a length-prefixed byte field. Fixed field order and a domain tag make
/// this encoding unambiguous without platform-sized integers.
fn put_bytes(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&(value.len() as u64).to_be_bytes());
    target.extend_from_slice(value);
}

fn put_str(target: &mut Vec<u8>, value: &str) {
    put_bytes(target, value.as_bytes());
}

fn put_u64(target: &mut Vec<u8>, value: u64) {
    target.extend_from_slice(&value.to_be_bytes());
}

fn put_i64(target: &mut Vec<u8>, value: i64) {
    target.extend_from_slice(&value.to_be_bytes());
}

fn put_bool(target: &mut Vec<u8>, value: bool) {
    target.push(u8::from(value));
}

fn evidence_state_tag(state: EvidenceState) -> u8 {
    match state {
        EvidenceState::Qualified => 0,
        EvidenceState::Stale => 1,
        EvidenceState::Conflicting => 2,
        EvidenceState::Unknown => 3,
    }
}

fn rejection_tag(reason: RejectionReason) -> u8 {
    match reason {
        RejectionReason::NegativeCapitalRequirement => 0,
        RejectionReason::BudgetExceeded => 1,
        RejectionReason::LiquidityFloorBreached => 2,
        RejectionReason::MaintenanceLimitExceeded => 3,
        RejectionReason::Unauthorized => 4,
        RejectionReason::RequiredEvidenceMissingOrUnusable => 5,
    }
}

fn route_tag(route: CompatibleAnalysisRoute) -> u8 {
    match route {
        CompatibleAnalysisRoute::ExpectedValueOrValueOfInformation => 0,
        CompatibleAnalysisRoute::RobustProbabilityRanking => 1,
        CompatibleAnalysisRoute::ConservativeCompatibilityPlanning => 2,
        CompatibleAnalysisRoute::DiscrepancyAwareAssessment => 3,
        CompatibleAnalysisRoute::Abstain => 4,
    }
}

fn validate_scenario(scenario: &CandidateScenario) -> Result<(), DecisionReceiptError> {
    if scenario.id.trim().is_empty() {
        return Err(DecisionReceiptError::MissingScenarioId);
    }

    let mut option_ids = BTreeSet::new();
    for option in &scenario.options {
        if option.id.trim().is_empty() {
            return Err(DecisionReceiptError::EmptyOptionId);
        }
        if !option_ids.insert(option.id.as_str()) {
            return Err(DecisionReceiptError::DuplicateOptionId);
        }

        let mut evidence_ids = BTreeSet::new();
        for claim in &option.evidence {
            if claim.id.trim().is_empty() {
                return Err(DecisionReceiptError::EmptyEvidenceId);
            }
            if !evidence_ids.insert(claim.id.as_str()) {
                return Err(DecisionReceiptError::DuplicateEvidenceId);
            }
        }
    }
    Ok(())
}

/// Canonical commitment to scenario inputs. Options, evidence claims and
/// dependency IDs are sorted; dependency IDs are deduplicated because this
/// evaluator counts distinct IDs only.
fn canonical_scenario_bytes(
    scenario: &CandidateScenario,
) -> Result<Vec<u8>, DecisionReceiptError> {
    validate_scenario(scenario)?;
    let mut bytes = Vec::new();
    put_str(&mut bytes, "ICCB-CANDIDATE-SCENARIO-V1");
    put_str(&mut bytes, &scenario.id);
    put_i64(&mut bytes, scenario.mandate.budget_limit_minor);
    put_i64(&mut bytes, scenario.mandate.minimum_liquidity_minor);
    put_i64(&mut bytes, scenario.mandate.maximum_maintenance_backlog_delta_minor);
    put_u64(&mut bytes, u64::from(scenario.mandate.evaluation_day));
    put_bool(&mut bytes, scenario.mandate.required_evidence);

    let mut options: Vec<_> = scenario.options.iter().collect();
    options.sort_by(|left, right| left.id.cmp(&right.id));
    put_u64(&mut bytes, options.len() as u64);

    for option in options {
        put_str(&mut bytes, &option.id);
        put_i64(&mut bytes, option.expected_net_cash_minor);
        put_i64(&mut bytes, option.reported_valuation_change_minor);
        put_i64(&mut bytes, option.required_capital_minor);
        put_i64(&mut bytes, option.available_liquidity_after_minor);
        put_i64(&mut bytes, option.productive_capacity_delta);
        put_i64(&mut bytes, option.maintenance_backlog_delta_minor);
        put_bool(&mut bytes, option.authority_granted);

        let mut dependencies: Vec<_> = option
            .dependency_ids
            .iter()
            .map(String::as_str)
            .collect();
        dependencies.sort_unstable();
        dependencies.dedup();
        put_u64(&mut bytes, dependencies.len() as u64);
        for dependency in dependencies {
            put_str(&mut bytes, dependency);
        }

        let mut claims: Vec<_> = option.evidence.iter().collect();
        claims.sort_by(|left, right| left.id.cmp(&right.id));
        put_u64(&mut bytes, claims.len() as u64);
        for claim in claims {
            put_str(&mut bytes, &claim.id);
            bytes.push(evidence_state_tag(claim.state));
            match &claim.source_id {
                Some(source) => {
                    bytes.push(1);
                    put_str(&mut bytes, source);
                }
                None => bytes.push(0),
            }
            match claim.observed_day {
                Some(day) => {
                    bytes.push(1);
                    put_u64(&mut bytes, u64::from(day));
                }
                None => bytes.push(0),
            }
            match claim.valid_through_day {
                Some(day) => {
                    bytes.push(1);
                    put_u64(&mut bytes, u64::from(day));
                }
                None => bytes.push(0),
            }
        }
    }
    Ok(bytes)
}

fn canonical_output_bytes(output: &EvaluationReceipt) -> Result<Vec<u8>, DecisionReceiptError> {
    if output.scenario_id.trim().is_empty() {
        return Err(DecisionReceiptError::MissingScenarioId);
    }

    let mut options: Vec<&OptionEvaluation> = output.evaluated_options.iter().collect();
    options.sort_by(|left, right| left.option_id.cmp(&right.option_id));
    if options.iter().any(|option| option.option_id.trim().is_empty()) {
        return Err(DecisionReceiptError::EmptyOptionId);
    }
    for pair in options.windows(2) {
        if pair[0].option_id == pair[1].option_id {
            return Err(DecisionReceiptError::DuplicateOptionId);
        }
    }

    let mut bytes = Vec::new();
    put_str(&mut bytes, "ICCB-EVALUATION-OUTPUT-V1");
    put_str(&mut bytes, &output.scenario_id);
    put_u64(&mut bytes, u64::from(output.evaluation_day));
    put_u64(&mut bytes, options.len() as u64);

    for option in options {
        put_str(&mut bytes, &option.option_id);
        put_bool(&mut bytes, option.accepted);
        let mut reasons: Vec<u8> = option
            .rejection_reasons
            .iter()
            .copied()
            .map(rejection_tag)
            .collect();
        reasons.sort_unstable();
        let reason_count = reasons.len();
        reasons.dedup();
        if reasons.len() != reason_count {
            return Err(DecisionReceiptError::DuplicateRejectionReason);
        }
        put_u64(&mut bytes, reasons.len() as u64);
        bytes.extend_from_slice(&reasons);
        put_i64(&mut bytes, option.outcomes.expected_net_cash_minor);
        put_i64(&mut bytes, option.outcomes.reported_valuation_change_minor);
        put_i64(&mut bytes, option.outcomes.available_liquidity_after_minor);
        put_i64(&mut bytes, option.outcomes.productive_capacity_delta);
        put_i64(&mut bytes, option.outcomes.maintenance_backlog_delta_minor);
        put_u64(&mut bytes, option.outcomes.declared_dependency_count as u64);
    }
    Ok(bytes)
}

fn canonical_receipt_bytes(receipt: &DecisionReceipt) -> Vec<u8> {
    let mut bytes = Vec::new();
    put_str(&mut bytes, &receipt.schema_version);
    put_str(&mut bytes, &receipt.canonicalization_version);
    put_str(&mut bytes, &receipt.input_manifest_encoding_version);
    put_str(&mut bytes, &receipt.upstream_producer_version);
    put_str(&mut bytes, &receipt.scenario_id);
    put_u64(&mut bytes, u64::from(receipt.evaluation_day));
    put_str(&mut bytes, &receipt.producer_commit);
    put_str(&mut bytes, &receipt.evaluator_version);
    put_str(&mut bytes, &receipt.input_manifest_sha256);
    put_str(&mut bytes, &receipt.scenario_sha256);
    bytes.push(route_tag(receipt.uncertainty_route));
    put_str(&mut bytes, &receipt.evidence_reference);
    put_str(&mut bytes, &receipt.output_sha256);
    put_str(&mut bytes, &receipt.claim_ceiling);
    put_str(&mut bytes, &receipt.limitation);
    bytes
}

fn validate_context(
    context: &DecisionRunContext,
    scenario: &CandidateScenario,
) -> Result<String, DecisionReceiptError> {
    if context.producer_commit.trim().is_empty() {
        return Err(DecisionReceiptError::MissingProducerCommit);
    }
    if !valid_git_id(&context.producer_commit) {
        return Err(DecisionReceiptError::MalformedProducerCommit);
    }
    if context.evaluator_version.trim().is_empty() {
        return Err(DecisionReceiptError::MissingEvaluatorVersion);
    }
    if context.input_manifest_bytes.is_empty() {
        return Err(DecisionReceiptError::EmptyInputManifest);
    }
    if context.input_manifest_encoding_version.trim().is_empty() {
        return Err(DecisionReceiptError::MissingInputManifestEncodingVersion);
    }
    if context.routing.producer_version.trim().is_empty() {
        return Err(DecisionReceiptError::MissingUpstreamProducerVersion);
    }
    if context.claim_ceiling.trim().is_empty() {
        return Err(DecisionReceiptError::MissingClaimCeiling);
    }
    if context.routing.scenario_id != scenario.id {
        return Err(DecisionReceiptError::RoutingScenarioMismatch);
    }
    if context.routing.evidence_reference.trim().is_empty()
        || context.routing.limitation.trim().is_empty()
    {
        return Err(DecisionReceiptError::MissingRoutingEvidence);
    }

    let manifest_digest = sha256_hex(&context.input_manifest_bytes);
    if !digest_matches(&context.routing.source_manifest_digest, &manifest_digest) {
        return Err(DecisionReceiptError::RoutingManifestDigestMismatch);
    }
    Ok(manifest_digest)
}

/// Build a receipt around the scenario's actual output. The manifest digest is
/// recomputed from exact supplied bytes. The scenario/output are hashed separately
/// with this module's canonical encoding, while the manifest's own encoding label
/// is preserved verbatim for later independent replay.
pub fn create_decision_receipt(
    context: &DecisionRunContext,
    scenario: &CandidateScenario,
) -> Result<DecisionReceipt, DecisionReceiptError> {
    let manifest_digest = validate_context(context, scenario)?;
    let scenario_digest = sha256_hex(&canonical_scenario_bytes(scenario)?);
    let output = scenario.evaluate();
    let output_digest = sha256_hex(&canonical_output_bytes(&output)?);

    let mut receipt = DecisionReceipt {
        schema_version: DECISION_RECEIPT_SCHEMA.to_owned(),
        canonicalization_version: CANONICALIZATION_VERSION.to_owned(),
        input_manifest_encoding_version: context.input_manifest_encoding_version.clone(),
        upstream_producer_version: context.routing.producer_version.clone(),
        scenario_id: scenario.id.clone(),
        evaluation_day: output.evaluation_day,
        producer_commit: context.producer_commit.clone(),
        evaluator_version: context.evaluator_version.clone(),
        input_manifest_sha256: manifest_digest,
        scenario_sha256: scenario_digest,
        uncertainty_route: context.routing.route,
        evidence_reference: context.routing.evidence_reference.clone(),
        output_sha256: output_digest,
        claim_ceiling: context.claim_ceiling.clone(),
        limitation: context.routing.limitation.clone(),
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = sha256_hex(&canonical_receipt_bytes(&receipt));
    Ok(receipt)
}

/// Re-evaluate the supplied scenario and compare every field against the record.
/// This establishes reproducibility for the pinned inputs and current evaluator,
/// not evidence truth, model validity, or authorization to act.
pub fn verify_decision_receipt(
    context: &DecisionRunContext,
    scenario: &CandidateScenario,
    recorded: &DecisionReceipt,
) -> Result<(), DecisionReceiptError> {
    let expected = create_decision_receipt(context, scenario)?;
    if &expected == recorded {
        Ok(())
    } else {
        Err(DecisionReceiptError::ReceiptMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uncertainty_contract::{UncertaintyAssessmentRef, UncertaintyEvidenceKind};
    use crate::{DecisionOption, EvidenceClaim, Mandate};

    fn scenario() -> CandidateScenario {
        let claim1 = EvidenceClaim {
            id: "evidence-1".into(),
            state: EvidenceState::Qualified,
            source_id: Some("fixture-source".into()),
            observed_day: Some(1),
            valid_through_day: Some(10),
        };
        let option1 = DecisionOption {
            id: "build".into(),
            expected_net_cash_minor: 120,
            reported_valuation_change_minor: 20,
            required_capital_minor: 100,
            available_liquidity_after_minor: 50,
            productive_capacity_delta: 5,
            maintenance_backlog_delta_minor: 2,
            dependency_ids: vec!["grid".into(), "water".into(), "grid".into()],
            evidence: vec![claim1],
            authority_granted: true,
        };
        let option2 = DecisionOption {
            id: "repair".into(),
            expected_net_cash_minor: 70,
            reported_valuation_change_minor: 0,
            required_capital_minor: 20,
            available_liquidity_after_minor: 80,
            productive_capacity_delta: 2,
            maintenance_backlog_delta_minor: -5,
            dependency_ids: vec!["water".into()],
            evidence: vec![EvidenceClaim {
                id: "evidence-2".into(),
                state: EvidenceState::Qualified,
                source_id: Some("fixture-source".into()),
                observed_day: Some(2),
                valid_through_day: Some(10),
            }],
            authority_granted: true,
        };
        CandidateScenario {
            id: "fixture-capex-1".into(),
            mandate: Mandate {
                budget_limit_minor: 1_000,
                minimum_liquidity_minor: 0,
                maximum_maintenance_backlog_delta_minor: 100,
                evaluation_day: 5,
                required_evidence: true,
            },
            options: vec![option1, option2],
        }
    }

    fn context(manifest: &[u8], scenario_id: &str) -> DecisionRunContext {
        let manifest_digest = sha256_hex(manifest);
        let routing = UncertaintyAssessmentRef {
            scenario_id: scenario_id.into(),
            producer_version: "fixture-producer-v1".into(),
            source_manifest_digest: format!("sha256:{manifest_digest}"),
            evidence_kind: UncertaintyEvidenceKind::BoundedStateProbabilities {
                bounds_receipt_id: "bounds-receipt-1".into(),
            },
        }
        .route()
        .unwrap();
        DecisionRunContext {
            producer_commit: "0123456789abcdef0123456789abcdef01234567".into(),
            evaluator_version: "iccb-evaluator-0.1.0".into(),
            input_manifest_encoding_version: "utf8-synthetic-fixture-v1".into(),
            input_manifest_bytes: manifest.to_vec(),
            routing,
            claim_ceiling: "synthetic rule evaluation; not a forecast or authorization".into(),
        }
    }

    #[test]
    fn sha256_matches_published_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn identical_pinned_inputs_produce_identical_receipts_and_replay() {
        let scenario = scenario();
        let context = context(b"canonical-manifest-v1", &scenario.id);
        let first = create_decision_receipt(&context, &scenario).unwrap();
        let second = create_decision_receipt(&context, &scenario).unwrap();
        assert_eq!(first, second);
        verify_decision_receipt(&context, &scenario, &first).unwrap();
    }

    #[test]
    fn source_manifest_mutation_invalidates_old_receipt() {
        let scenario = scenario();
        let first_context = context(b"canonical-manifest-v1", &scenario.id);
        let first = create_decision_receipt(&first_context, &scenario).unwrap();
        let changed_context = context(b"canonical-manifest-v2", &scenario.id);
        assert_eq!(
            verify_decision_receipt(&changed_context, &scenario, &first),
            Err(DecisionReceiptError::ReceiptMismatch)
        );
    }

    #[test]
    fn decision_relevant_scenario_mutation_invalidates_old_receipt() {
        let original = scenario();
        let context = context(b"canonical-manifest-v1", &original.id);
        let receipt = create_decision_receipt(&context, &original).unwrap();
        let mut changed = original.clone();
        changed.options[0].expected_net_cash_minor += 1;
        assert_eq!(
            verify_decision_receipt(&context, &changed, &receipt),
            Err(DecisionReceiptError::ReceiptMismatch)
        );
    }

    #[test]
    fn option_and_dependency_order_do_not_change_semantic_receipt() {
        let original = scenario();
        let context = context(b"canonical-manifest-v1", &original.id);
        let receipt = create_decision_receipt(&context, &original).unwrap();
        let mut reordered = original.clone();
        reordered.options.reverse();
        reordered.options[1].dependency_ids.reverse();
        let reordered_receipt = create_decision_receipt(&context, &reordered).unwrap();
        assert_eq!(receipt, reordered_receipt);
    }

    #[test]
    fn duplicate_option_ids_are_rejected() {
        let mut scenario = scenario();
        let duplicate_id = scenario.options[0].id.clone();
        scenario.options[1].id = duplicate_id;
        let context = context(b"canonical-manifest-v1", &scenario.id);
        assert_eq!(
            create_decision_receipt(&context, &scenario),
            Err(DecisionReceiptError::DuplicateOptionId)
        );
    }

    #[test]
    fn manifest_digest_must_match_routing_receipt() {
        let scenario = scenario();
        let mut context = context(b"canonical-manifest-v1", &scenario.id);
        context.routing.source_manifest_digest =
            "sha256:937a22712ea5c20073d2c8a226f1f4a9c2c30aba40a497f2412a34426f986535".into();
        assert_eq!(
            create_decision_receipt(&context, &scenario),
            Err(DecisionReceiptError::RoutingManifestDigestMismatch)
        );
    }

    #[test]
    fn duplicate_evidence_ids_are_rejected() {
        let mut scenario = scenario();
        let duplicate_claim = scenario.options[0].evidence[0].clone();
        scenario.options[0].evidence.push(duplicate_claim);
        let context = context(b"canonical-manifest-v1", &scenario.id);
        assert_eq!(
            create_decision_receipt(&context, &scenario),
            Err(DecisionReceiptError::DuplicateEvidenceId)
        );
    }

    #[test]
    fn evaluator_version_mutation_invalidates_old_receipt() {
        let scenario = scenario();
        let original_context = context(b"canonical-manifest-v1", &scenario.id);
        let receipt = create_decision_receipt(&original_context, &scenario).unwrap();
        let mut changed_context = original_context.clone();
        changed_context.evaluator_version = "iccb-evaluator-0.1.1".into();
        assert_eq!(
            verify_decision_receipt(&changed_context, &scenario, &receipt),
            Err(DecisionReceiptError::ReceiptMismatch)
        );
    }

    #[test]
    fn tampered_receipt_output_digest_is_rejected() {
        let scenario = scenario();
        let context = context(b"canonical-manifest-v1", &scenario.id);
        let mut receipt = create_decision_receipt(&context, &scenario).unwrap();
        receipt.output_sha256 = "0".repeat(64);
        assert_eq!(
            verify_decision_receipt(&context, &scenario, &receipt),
            Err(DecisionReceiptError::ReceiptMismatch)
        );
    }

    #[test]
    fn invalid_pinned_metadata_fails_closed() {
        let scenario = scenario();
        let mut context = context(b"canonical-manifest-v1", &scenario.id);
        context.producer_commit = "not-a-git-commit".into();
        assert_eq!(
            create_decision_receipt(&context, &scenario),
            Err(DecisionReceiptError::MalformedProducerCommit)
        );
        context.producer_commit = "0123456789abcdef0123456789abcdef01234567".into();
        context.input_manifest_bytes.clear();
        assert_eq!(
            create_decision_receipt(&context, &scenario),
            Err(DecisionReceiptError::EmptyInputManifest)
        );
    }
}
