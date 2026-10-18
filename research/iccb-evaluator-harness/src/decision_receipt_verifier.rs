//! Independent receipt conformance verifier.
//!
//! This package deliberately reimplements the V1 wire encoding instead of
//! importing the candidate crate's private canonicalizers. It verifies pinned
//! manifests, scenario/output digests and the receipt digest. It uses the
//! candidate's public deterministic evaluator to replay outputs; it is not an
//! independent financial/engineering model or proof of source truth.

use iccb_evaluator::decision_receipt::{DecisionReceipt, DecisionRunContext};

const SUPPORTED_RECEIPT_SCHEMA: &str = "iccb-decision-receipt-v1";
const SUPPORTED_CANONICALIZATION_VERSION: &str = "iccb-canonical-binary-v1";
use iccb_evaluator::uncertainty_contract::{
    CompatibleAnalysisRoute, UncertaintyEvidenceKind, UncertaintyRoutingReceipt,
};
use iccb_evaluator::{
    CandidateScenario, EvaluationReceipt, EvidenceState, OptionEvaluation, RejectionReason,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::Write as _;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiptVerificationError {
    UnsupportedReceiptSchema,
    UnsupportedCanonicalizationVersion,
    InvalidPinnedMetadata,
    InvalidUncertaintyAssessment,
    RoutingScenarioMismatch,
    RoutingManifestDigestMismatch,
    InputManifestDigestMismatch,
    InvalidScenario,
    ScenarioDigestMismatch,
    InvalidEvaluatorOutput,
    EvaluationOutputDigestMismatch,
    ReceiptMetadataMismatch,
    ReceiptDigestMismatch,
}

fn independent_sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(64);
    for byte in digest {
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

fn digest_matches(reported: &str, computed: &str) -> bool {
    reported.strip_prefix("sha256:").unwrap_or(reported).eq_ignore_ascii_case(computed)
}

fn valid_git_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

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

fn state_tag(state: EvidenceState) -> u8 {
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

fn put_uncertainty_kind(target: &mut Vec<u8>, kind: &UncertaintyEvidenceKind) {
    match kind {
        UncertaintyEvidenceKind::PointProbabilityDistribution {
            qualification_receipt_id,
            validation_corpus_digest,
        } => {
            target.push(0);
            put_str(target, qualification_receipt_id);
            put_str(target, validation_corpus_digest);
        }
        UncertaintyEvidenceKind::BoundedStateProbabilities { bounds_receipt_id } => {
            target.push(1);
            put_str(target, bounds_receipt_id);
        }
        UncertaintyEvidenceKind::OutcomeCompatibilitySet { compatibility_receipt_id } => {
            target.push(2);
            put_str(target, compatibility_receipt_id);
        }
        UncertaintyEvidenceKind::ModelDiscrepancyEnvelope { envelope_receipt_id } => {
            target.push(3);
            put_str(target, envelope_receipt_id);
        }
        UncertaintyEvidenceKind::OutOfDomain { reason } => {
            target.push(4);
            put_str(target, reason);
        }
    }
}

fn encode_scenario(scenario: &CandidateScenario) -> Result<Vec<u8>, ReceiptVerificationError> {
    if scenario.id.trim().is_empty() {
        return Err(ReceiptVerificationError::InvalidScenario);
    }
    let mut option_ids = BTreeSet::new();
    for option in &scenario.options {
        if option.id.trim().is_empty() || !option_ids.insert(option.id.as_str()) {
            return Err(ReceiptVerificationError::InvalidScenario);
        }
        let mut evidence_ids = BTreeSet::new();
        for claim in &option.evidence {
            if claim.id.trim().is_empty() || !evidence_ids.insert(claim.id.as_str()) {
                return Err(ReceiptVerificationError::InvalidScenario);
            }
        }
    }

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

        let mut dependencies: Vec<_> = option.dependency_ids.iter().map(String::as_str).collect();
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
            bytes.push(state_tag(claim.state));
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

fn encode_output(output: &EvaluationReceipt) -> Result<Vec<u8>, ReceiptVerificationError> {
    if output.scenario_id.trim().is_empty() {
        return Err(ReceiptVerificationError::InvalidEvaluatorOutput);
    }
    let mut options: Vec<&OptionEvaluation> = output.evaluated_options.iter().collect();
    options.sort_by(|left, right| left.option_id.cmp(&right.option_id));
    if options.iter().any(|option| option.option_id.trim().is_empty()) {
        return Err(ReceiptVerificationError::InvalidEvaluatorOutput);
    }
    for pair in options.windows(2) {
        if pair[0].option_id == pair[1].option_id {
            return Err(ReceiptVerificationError::InvalidEvaluatorOutput);
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
        let count = reasons.len();
        reasons.dedup();
        if count != reasons.len() {
            return Err(ReceiptVerificationError::InvalidEvaluatorOutput);
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

fn encode_receipt(receipt: &DecisionReceipt) -> Vec<u8> {
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
    put_uncertainty_kind(&mut bytes, &receipt.uncertainty_evidence_kind);
    put_str(&mut bytes, &receipt.evidence_reference);
    put_str(&mut bytes, &receipt.output_sha256);
    put_str(&mut bytes, &receipt.claim_ceiling);
    put_str(&mut bytes, &receipt.limitation);
    bytes
}

/// Independently verifies the V1 receipt encoding and replays the scenario.
///
/// The verifier uses a separately implemented canonical encoder and SHA-256
/// path. It does reuse the public candidate evaluator for the outcome calculation;
/// physical correctness and receipt-source authenticity are outside its claim.
pub fn verify_decision_receipt_independently(
    context: &DecisionRunContext,
    scenario: &CandidateScenario,
    recorded: &DecisionReceipt,
) -> Result<(), ReceiptVerificationError> {
    if recorded.schema_version != SUPPORTED_RECEIPT_SCHEMA {
        return Err(ReceiptVerificationError::UnsupportedReceiptSchema);
    }
    if recorded.canonicalization_version != SUPPORTED_CANONICALIZATION_VERSION {
        return Err(ReceiptVerificationError::UnsupportedCanonicalizationVersion);
    }
    if context.producer_commit.trim().is_empty()
        || !valid_git_id(&context.producer_commit)
        || context.evaluator_version.trim().is_empty()
        || context.input_manifest_encoding_version.trim().is_empty()
        || context.input_manifest_bytes.is_empty()
        || context.claim_ceiling.trim().is_empty()
    {
        return Err(ReceiptVerificationError::InvalidPinnedMetadata);
    }

    let route = context
        .uncertainty_assessment
        .route()
        .map_err(|_| ReceiptVerificationError::InvalidUncertaintyAssessment)?;
    if route.scenario_id != scenario.id {
        return Err(ReceiptVerificationError::RoutingScenarioMismatch);
    }
    let manifest_digest = independent_sha256_hex(&context.input_manifest_bytes);
    if !digest_matches(&route.source_manifest_digest, &manifest_digest) {
        return Err(ReceiptVerificationError::RoutingManifestDigestMismatch);
    }
    if recorded.input_manifest_sha256 != manifest_digest {
        return Err(ReceiptVerificationError::InputManifestDigestMismatch);
    }

    let scenario_digest = independent_sha256_hex(&encode_scenario(scenario)?);
    if recorded.scenario_sha256 != scenario_digest {
        return Err(ReceiptVerificationError::ScenarioDigestMismatch);
    }
    let output = scenario.evaluate();
    let output_digest = independent_sha256_hex(&encode_output(&output)?);
    if recorded.output_sha256 != output_digest {
        return Err(ReceiptVerificationError::EvaluationOutputDigestMismatch);
    }

    if recorded.scenario_id != scenario.id
        || recorded.evaluation_day != output.evaluation_day
        || recorded.producer_commit != context.producer_commit
        || recorded.evaluator_version != context.evaluator_version
        || recorded.input_manifest_encoding_version != context.input_manifest_encoding_version
        || recorded.upstream_producer_version != route.producer_version
        || recorded.uncertainty_route != route.route
        || recorded.uncertainty_evidence_kind != route.evidence_kind
        || recorded.evidence_reference != route.evidence_reference
        || recorded.claim_ceiling != context.claim_ceiling
        || recorded.limitation != route.limitation
    {
        return Err(ReceiptVerificationError::ReceiptMetadataMismatch);
    }

    let receipt_digest = independent_sha256_hex(&encode_receipt(recorded));
    if recorded.receipt_sha256 != receipt_digest {
        return Err(ReceiptVerificationError::ReceiptDigestMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use iccb_evaluator::uncertainty_contract::UncertaintyAssessmentRef;
    use iccb_evaluator::{DecisionOption, EvidenceClaim, Mandate};

    fn scenario() -> CandidateScenario {
        CandidateScenario {
            id: "fixture-capex-1".into(),
            mandate: Mandate {
                budget_limit_minor: 1_000,
                minimum_liquidity_minor: 0,
                maximum_maintenance_backlog_delta_minor: 100,
                evaluation_day: 5,
                required_evidence: true,
            },
            options: vec![
                DecisionOption {
                    id: "build".into(),
                    expected_net_cash_minor: 120,
                    reported_valuation_change_minor: 20,
                    required_capital_minor: 100,
                    available_liquidity_after_minor: 50,
                    productive_capacity_delta: 5,
                    maintenance_backlog_delta_minor: 2,
                    dependency_ids: vec!["grid".into(), "water".into(), "grid".into()],
                    evidence: vec![EvidenceClaim {
                        id: "evidence-1".into(),
                        state: EvidenceState::Qualified,
                        source_id: Some("fixture-source".into()),
                        observed_day: Some(1),
                        valid_through_day: Some(10),
                    }],
                    authority_granted: true,
                },
                DecisionOption {
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
                },
            ],
        }
    }

    fn context(manifest: &[u8], scenario_id: &str) -> DecisionRunContext {
        let digest = independent_sha256_hex(manifest);
        DecisionRunContext {
            producer_commit: "0123456789abcdef0123456789abcdef01234567".into(),
            evaluator_version: "iccb-evaluator-0.1.0".into(),
            input_manifest_encoding_version: "utf8-synthetic-fixture-v1".into(),
            input_manifest_bytes: manifest.to_vec(),
            uncertainty_assessment: UncertaintyAssessmentRef {
                scenario_id: scenario_id.into(),
                producer_version: "fixture-producer-v1".into(),
                source_manifest_digest: format!("sha256:{digest}"),
                evidence_kind: UncertaintyEvidenceKind::BoundedStateProbabilities {
                    bounds_receipt_id: "bounds-receipt-1".into(),
                },
            },
            claim_ceiling: "synthetic rule evaluation; not a forecast or authorization".into(),
        }
    }

    fn frozen_golden_receipt() -> DecisionReceipt {
        DecisionReceipt {
            schema_version: "iccb-decision-receipt-v1".into(),
            canonicalization_version: "iccb-canonical-binary-v1".into(),
            input_manifest_encoding_version: "utf8-synthetic-fixture-v1".into(),
            upstream_producer_version: "fixture-producer-v1".into(),
            scenario_id: "fixture-capex-1".into(),
            evaluation_day: 5,
            producer_commit: "0123456789abcdef0123456789abcdef01234567".into(),
            evaluator_version: "iccb-evaluator-0.1.0".into(),
            input_manifest_sha256:
                "118603a03fa97d3a84dfbf7bc3ec10fcd02b72abce596f74e70cbcdb93cc8980"
                    .into(),
            scenario_sha256:
                "b1cc79675e9d7c0b2d7fdf41ae90f134c7985053d024d4ad608b432974373a58"
                    .into(),
            uncertainty_route: CompatibleAnalysisRoute::RobustProbabilityRanking,
            uncertainty_evidence_kind: UncertaintyEvidenceKind::BoundedStateProbabilities {
                bounds_receipt_id: "bounds-receipt-1".into(),
            },
            evidence_reference: "bounds-receipt-1".into(),
            output_sha256:
                "5f05afb631afd0d8977742c232ddcf91f2ac59b0c571de0c2684be8f0a227e87"
                    .into(),
            claim_ceiling: "synthetic rule evaluation; not a forecast or authorization".into(),
            limitation:
                "Only probability-mass ambiguity is routed here; payoff/model discrepancy and out-of-domain states require separate treatment."
                    .into(),
            receipt_sha256:
                "28460a3267a62079c0aabd350b87c8050a58d10bcd40423514ed9d9e7856d5d4"
                    .into(),
        }
    }

    #[test]
    fn independent_verifier_accepts_frozen_golden_receipt() {
        let scenario = scenario();
        let context = context(b"canonical-manifest-v1", &scenario.id);
        verify_decision_receipt_independently(&context, &scenario, &frozen_golden_receipt())
            .unwrap();
    }

    #[test]
    fn independent_verifier_is_invariant_to_semantic_input_order() {
        let mut scenario = scenario();
        let context = context(b"canonical-manifest-v1", &scenario.id);
        let recorded = frozen_golden_receipt();
        scenario.options.reverse();
        scenario.options[1].dependency_ids.reverse();
        verify_decision_receipt_independently(&context, &scenario, &recorded).unwrap();
    }

    #[test]
    fn independent_verifier_rejects_changed_scenario() {
        let mut scenario = scenario();
        let context = context(b"canonical-manifest-v1", &scenario.id);
        let recorded = frozen_golden_receipt();
        scenario.options[0].expected_net_cash_minor += 1;
        assert_eq!(
            verify_decision_receipt_independently(&context, &scenario, &recorded),
            Err(ReceiptVerificationError::ScenarioDigestMismatch)
        );
    }

    #[test]
    fn independent_verifier_rejects_receipt_digest_tampering() {
        let scenario = scenario();
        let context = context(b"canonical-manifest-v1", &scenario.id);
        let mut recorded = frozen_golden_receipt();
        recorded.receipt_sha256 = "0".repeat(64);
        assert_eq!(
            verify_decision_receipt_independently(&context, &scenario, &recorded),
            Err(ReceiptVerificationError::ReceiptDigestMismatch)
        );
    }

    #[test]
    fn independent_verifier_rejects_manifest_mutation() {
        let scenario = scenario();
        let context = context(b"canonical-manifest-v2", &scenario.id);
        assert_eq!(
            verify_decision_receipt_independently(
                &context,
                &scenario,
                &frozen_golden_receipt()
            ),
            Err(ReceiptVerificationError::InputManifestDigestMismatch)
        );
    }
}
