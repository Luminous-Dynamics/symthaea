//! Conjunctive, proof-obligation underwriting decision certificate.
//!
//! The certificate assembler binds independently evaluated project claims to
//! exact project/evidence/rule/model/scenario/evaluator manifests. Critical
//! obligations are conjunctive: a strong financial result cannot compensate
//! for an unresolved grid right, market admission, physical limit or contract.
//!
//! This module validates the structure, dependency graph and temporal envelope
//! of supplied receipts. It does not authenticate source bytes, verify issuer
//! authority/signatures, rerun physical or financial models, or prove that an
//! upstream evaluator is correct. Each receipt must be independently verified
//! by its designated verifier before being marked Qualified.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObligationCriticality {
    Critical,
    Material,
    Informational,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofDisposition {
    Qualified,
    Conditional,
    Unresolved,
    Rejected,
    Invalidated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnderwritingDecision {
    SupportableWithinDeclaredScope,
    Conditional,
    NotSupportable,
    DoNotProceed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofReceiptRef<'a> {
    pub receipt_id: &'a str,
    pub receipt_sha256: &'a str,
    pub available_at_utc: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofObligation<'a> {
    pub obligation_id: &'a str,
    pub claim_id: &'a str,
    pub criticality: ObligationCriticality,
    pub disposition: ProofDisposition,
    /// Every dependency must be present in the same certificate.
    pub dependency_ids: &'a [&'a str],
    pub evidence_receipts: &'a [ProofReceiptRef<'a>],
    /// Receipt from the exact evaluator/model run that assessed this claim.
    pub evaluator_receipt: Option<ProofReceiptRef<'a>>,
    pub valid_from_utc: i64,
    pub valid_until_utc: Option<i64>,
    /// Explicit conditions that would suspend or invalidate this claim.
    pub invalidation_conditions: &'a [&'a str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnderwritingCertificateInputs<'a> {
    pub project_id: &'a str,
    pub project_revision: &'a str,
    pub project_revision_sha256: &'a str,
    pub decision_time_utc: i64,
    pub evidence_frontier_sha256: &'a str,
    pub rule_profile_id: &'a str,
    pub rule_profile_sha256: &'a str,
    pub model_suite_sha256: &'a str,
    pub scenario_census_sha256: &'a str,
    pub scenario_count: u32,
    /// Digest of the canonical serialized certificate input manifest, created
    /// by the caller's deterministic serializer.
    pub input_manifest_sha256: &'a str,
    pub obligations: &'a [ProofObligation<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnderwritingCertificateError {
    InvalidReference,
    InvalidDigest,
    EmptyScenarioCensus,
    EmptyObligationSet,
    DuplicateObligationId,
    DuplicateClaimId,
    MissingDependency,
    DependencyCycle,
    DependencyNotQualified,
    EvidenceKnownAfterDecision,
    EvaluatorReceiptMissing,
    QualifiedClaimWithoutEvidence,
    ConditionalClaimWithoutCondition,
    ClaimOutsideValidityWindow,
    ReceiptDigestInvalid,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObligationCertificateLine<'a> {
    pub obligation_id: &'a str,
    pub claim_id: &'a str,
    pub criticality: ObligationCriticality,
    pub disposition: ProofDisposition,
    pub dependency_count: u32,
    pub evidence_receipt_count: u32,
    pub evaluator_receipt_id: Option<&'a str>,
    pub invalidation_condition_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnderwritingDecisionCertificate<'a> {
    pub project_id: &'a str,
    pub project_revision: &'a str,
    pub project_revision_sha256: &'a str,
    pub decision_time_utc: i64,
    pub evidence_frontier_sha256: &'a str,
    pub rule_profile_id: &'a str,
    pub rule_profile_sha256: &'a str,
    pub model_suite_sha256: &'a str,
    pub scenario_census_sha256: &'a str,
    pub scenario_count: u32,
    pub input_manifest_sha256: &'a str,
    pub decision: UnderwritingDecision,
    pub qualified_critical_count: u32,
    pub conditional_critical_count: u32,
    pub unresolved_critical_count: u32,
    pub rejected_or_invalidated_critical_count: u32,
    pub unresolved_material_count: u32,
    pub conditional_material_count: u32,
    pub obligations: Vec<ObligationCertificateLine<'a>>,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_receipt(
    receipt: ProofReceiptRef<'_>,
    decision_time_utc: i64,
) -> Result<(), UnderwritingCertificateError> {
    if !nonempty(receipt.receipt_id) {
        return Err(UnderwritingCertificateError::InvalidReference);
    }
    if !valid_sha256(receipt.receipt_sha256) {
        return Err(UnderwritingCertificateError::ReceiptDigestInvalid);
    }
    if receipt.available_at_utc > decision_time_utc {
        return Err(UnderwritingCertificateError::EvidenceKnownAfterDecision);
    }
    Ok(())
}

fn visit_dependency_graph(
    index: usize,
    obligations: &[ProofObligation<'_>],
    states: &mut [u8],
) -> Result<(), UnderwritingCertificateError> {
    match states[index] {
        1 => return Err(UnderwritingCertificateError::DependencyCycle),
        2 => return Ok(()),
        _ => {}
    }
    states[index] = 1;
    for dependency_id in obligations[index].dependency_ids {
        let dependency_index = obligations
            .iter()
            .position(|obligation| obligation.obligation_id == *dependency_id)
            .ok_or(UnderwritingCertificateError::MissingDependency)?;
        visit_dependency_graph(dependency_index, obligations, states)?;
    }
    states[index] = 2;
    Ok(())
}

fn validate_dependencies(
    obligations: &[ProofObligation<'_>],
) -> Result<(), UnderwritingCertificateError> {
    let mut states = vec![0_u8; obligations.len()];
    for index in 0..obligations.len() {
        visit_dependency_graph(index, obligations, &mut states)?;
    }

    for obligation in obligations {
        for dependency_id in obligation.dependency_ids {
            let dependency = obligations
                .iter()
                .find(|candidate| candidate.obligation_id == *dependency_id)
                .ok_or(UnderwritingCertificateError::MissingDependency)?;
            if dependency.disposition != ProofDisposition::Qualified
                && obligation.disposition == ProofDisposition::Qualified
            {
                return Err(UnderwritingCertificateError::DependencyNotQualified);
            }
            if matches!(
                dependency.disposition,
                ProofDisposition::Rejected | ProofDisposition::Invalidated
            ) && obligation.disposition != ProofDisposition::Rejected
                && obligation.disposition != ProofDisposition::Invalidated
            {
                return Err(UnderwritingCertificateError::DependencyNotQualified);
            }
        }
    }
    Ok(())
}

/// Build the deterministic decision envelope after independently verified
/// receipts and evaluator dispositions have been supplied.
pub fn evaluate_underwriting_certificate<'a>(
    inputs: UnderwritingCertificateInputs<'a>,
) -> Result<UnderwritingDecisionCertificate<'a>, UnderwritingCertificateError> {
    for reference in [
        inputs.project_id,
        inputs.project_revision,
        inputs.rule_profile_id,
    ] {
        if !nonempty(reference) {
            return Err(UnderwritingCertificateError::InvalidReference);
        }
    }
    for digest in [
        inputs.project_revision_sha256,
        inputs.evidence_frontier_sha256,
        inputs.rule_profile_sha256,
        inputs.model_suite_sha256,
        inputs.scenario_census_sha256,
        inputs.input_manifest_sha256,
    ] {
        if !valid_sha256(digest) {
            return Err(UnderwritingCertificateError::InvalidDigest);
        }
    }
    if inputs.scenario_count == 0 {
        return Err(UnderwritingCertificateError::EmptyScenarioCensus);
    }
    if inputs.obligations.is_empty() {
        return Err(UnderwritingCertificateError::EmptyObligationSet);
    }

    for (index, obligation) in inputs.obligations.iter().enumerate() {
        if !nonempty(obligation.obligation_id) || !nonempty(obligation.claim_id) {
            return Err(UnderwritingCertificateError::InvalidReference);
        }
        for previous in &inputs.obligations[..index] {
            if previous.obligation_id == obligation.obligation_id {
                return Err(UnderwritingCertificateError::DuplicateObligationId);
            }
            if previous.claim_id == obligation.claim_id {
                return Err(UnderwritingCertificateError::DuplicateClaimId);
            }
        }
        if obligation.valid_from_utc > inputs.decision_time_utc
            || obligation
                .valid_until_utc
                .map(|end| inputs.decision_time_utc > end)
                .unwrap_or(false)
        {
            return Err(UnderwritingCertificateError::ClaimOutsideValidityWindow);
        }
        if obligation.disposition == ProofDisposition::Qualified
            && obligation.evidence_receipts.is_empty()
        {
            return Err(UnderwritingCertificateError::QualifiedClaimWithoutEvidence);
        }
        if obligation.disposition == ProofDisposition::Conditional
            && obligation.invalidation_conditions.is_empty()
        {
            return Err(UnderwritingCertificateError::ConditionalClaimWithoutCondition);
        }
        if matches!(
            obligation.disposition,
            ProofDisposition::Qualified
                | ProofDisposition::Conditional
                | ProofDisposition::Rejected
                | ProofDisposition::Invalidated
        ) && obligation.evaluator_receipt.is_none()
        {
            return Err(UnderwritingCertificateError::EvaluatorReceiptMissing);
        }
        for receipt in obligation.evidence_receipts {
            validate_receipt(*receipt, inputs.decision_time_utc)?;
        }
        if let Some(receipt) = obligation.evaluator_receipt {
            validate_receipt(receipt, inputs.decision_time_utc)?;
        }
        for condition in obligation.invalidation_conditions {
            if !nonempty(condition) {
                return Err(UnderwritingCertificateError::InvalidReference);
            }
        }
    }
    validate_dependencies(inputs.obligations)?;

    let mut qualified_critical = 0_u32;
    let mut conditional_critical = 0_u32;
    let mut unresolved_critical = 0_u32;
    let mut rejected_or_invalidated_critical = 0_u32;
    let mut unresolved_material = 0_u32;
    let mut conditional_material = 0_u32;
    let mut lines = Vec::with_capacity(inputs.obligations.len());

    for obligation in inputs.obligations {
        match (obligation.criticality, obligation.disposition) {
            (ObligationCriticality::Critical, ProofDisposition::Qualified) => {
                qualified_critical = qualified_critical
                    .checked_add(1)
                    .ok_or(UnderwritingCertificateError::ArithmeticOverflow)?;
            }
            (ObligationCriticality::Critical, ProofDisposition::Conditional) => {
                conditional_critical = conditional_critical
                    .checked_add(1)
                    .ok_or(UnderwritingCertificateError::ArithmeticOverflow)?;
            }
            (ObligationCriticality::Critical, ProofDisposition::Unresolved) => {
                unresolved_critical = unresolved_critical
                    .checked_add(1)
                    .ok_or(UnderwritingCertificateError::ArithmeticOverflow)?;
            }
            (
                ObligationCriticality::Critical,
                ProofDisposition::Rejected | ProofDisposition::Invalidated,
            ) => {
                rejected_or_invalidated_critical = rejected_or_invalidated_critical
                    .checked_add(1)
                    .ok_or(UnderwritingCertificateError::ArithmeticOverflow)?;
            }
            (ObligationCriticality::Material, ProofDisposition::Conditional) => {
                conditional_material = conditional_material
                    .checked_add(1)
                    .ok_or(UnderwritingCertificateError::ArithmeticOverflow)?;
            }
            (
                ObligationCriticality::Material,
                ProofDisposition::Unresolved
                | ProofDisposition::Rejected
                | ProofDisposition::Invalidated,
            ) => {
                unresolved_material = unresolved_material
                    .checked_add(1)
                    .ok_or(UnderwritingCertificateError::ArithmeticOverflow)?;
            }
            _ => {}
        }
        lines.push(ObligationCertificateLine {
            obligation_id: obligation.obligation_id,
            claim_id: obligation.claim_id,
            criticality: obligation.criticality,
            disposition: obligation.disposition,
            dependency_count: u32::try_from(obligation.dependency_ids.len())
                .map_err(|_| UnderwritingCertificateError::ArithmeticOverflow)?,
            evidence_receipt_count: u32::try_from(obligation.evidence_receipts.len())
                .map_err(|_| UnderwritingCertificateError::ArithmeticOverflow)?,
            evaluator_receipt_id: obligation
                .evaluator_receipt
                .map(|receipt| receipt.receipt_id),
            invalidation_condition_count: u32::try_from(obligation.invalidation_conditions.len())
                .map_err(|_| {
                UnderwritingCertificateError::ArithmeticOverflow
            })?,
        });
    }

    let decision = if rejected_or_invalidated_critical > 0 {
        UnderwritingDecision::DoNotProceed
    } else if unresolved_critical > 0 {
        UnderwritingDecision::NotSupportable
    } else if conditional_critical > 0 || unresolved_material > 0 || conditional_material > 0 {
        UnderwritingDecision::Conditional
    } else {
        UnderwritingDecision::SupportableWithinDeclaredScope
    };

    Ok(UnderwritingDecisionCertificate {
        project_id: inputs.project_id,
        project_revision: inputs.project_revision,
        project_revision_sha256: inputs.project_revision_sha256,
        decision_time_utc: inputs.decision_time_utc,
        evidence_frontier_sha256: inputs.evidence_frontier_sha256,
        rule_profile_id: inputs.rule_profile_id,
        rule_profile_sha256: inputs.rule_profile_sha256,
        model_suite_sha256: inputs.model_suite_sha256,
        scenario_census_sha256: inputs.scenario_census_sha256,
        scenario_count: inputs.scenario_count,
        input_manifest_sha256: inputs.input_manifest_sha256,
        decision,
        qualified_critical_count: qualified_critical,
        conditional_critical_count: conditional_critical,
        unresolved_critical_count: unresolved_critical,
        rejected_or_invalidated_critical_count: rejected_or_invalidated_critical,
        unresolved_material_count: unresolved_material,
        conditional_material_count: conditional_material,
        obligations: lines,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const DIGEST_B: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn receipt(id: &'static str, digest: &'static str) -> ProofReceiptRef<'static> {
        ProofReceiptRef {
            receipt_id: id,
            receipt_sha256: digest,
            available_at_utc: 90,
        }
    }

    fn base_obligations() -> [ProofObligation<'static>; 3] {
        [
            ProofObligation {
                obligation_id: "claim/financial",
                claim_id: "financial-feasibility",
                criticality: ObligationCriticality::Critical,
                disposition: ProofDisposition::Qualified,
                dependency_ids: &[],
                evidence_receipts: &[receipt("source/financial", DIGEST_A)],
                evaluator_receipt: Some(receipt("evaluator/financial", DIGEST_B)),
                valid_from_utc: 0,
                valid_until_utc: Some(200),
                invalidation_conditions: &["PPA or tariff changes"],
            },
            ProofObligation {
                obligation_id: "claim/grid",
                claim_id: "grid-rights",
                criticality: ObligationCriticality::Critical,
                disposition: ProofDisposition::Qualified,
                dependency_ids: &[],
                evidence_receipts: &[receipt("source/grid", DIGEST_A)],
                evaluator_receipt: Some(receipt("evaluator/grid", DIGEST_B)),
                valid_from_utc: 0,
                valid_until_utc: Some(200),
                invalidation_conditions: &["Connection agreement expires or is superseded"],
            },
            ProofObligation {
                obligation_id: "claim/physical",
                claim_id: "physical-capability",
                criticality: ObligationCriticality::Critical,
                disposition: ProofDisposition::Qualified,
                dependency_ids: &["claim/grid"],
                evidence_receipts: &[receipt("source/physical", DIGEST_A)],
                evaluator_receipt: Some(receipt("evaluator/physical", DIGEST_B)),
                valid_from_utc: 0,
                valid_until_utc: Some(200),
                invalidation_conditions: &["Battery capability moves outside calibrated surface"],
            },
        ]
    }

    fn inputs<'a>(obligations: &'a [ProofObligation<'a>]) -> UnderwritingCertificateInputs<'a> {
        UnderwritingCertificateInputs {
            project_id: "pv-bess/project-001",
            project_revision: "revision-7",
            project_revision_sha256: DIGEST_A,
            decision_time_utc: 100,
            evidence_frontier_sha256: DIGEST_B,
            rule_profile_id: "SAWEM-REV2.3/PROFILE-1",
            rule_profile_sha256: DIGEST_A,
            model_suite_sha256: DIGEST_B,
            scenario_census_sha256: DIGEST_A,
            scenario_count: 20,
            input_manifest_sha256: DIGEST_B,
            obligations,
        }
    }

    #[test]
    fn all_critical_obligations_qualified_yield_only_scoped_supportability() {
        let obligations = base_obligations();
        let result = evaluate_underwriting_certificate(inputs(&obligations)).unwrap();
        assert_eq!(
            result.decision,
            UnderwritingDecision::SupportableWithinDeclaredScope
        );
        assert_eq!(result.qualified_critical_count, 3);
        assert_eq!(result.scenario_count, 20);
        assert_eq!(result.obligations.len(), 3);
    }

    #[test]
    fn unresolved_critical_grid_rights_block_financially_strong_project() {
        let mut obligations = base_obligations();
        obligations[1].disposition = ProofDisposition::Unresolved;
        obligations[1].evidence_receipts = &[];
        obligations[1].evaluator_receipt = None;
        obligations[2].disposition = ProofDisposition::Unresolved;
        obligations[2].evidence_receipts = &[];
        obligations[2].evaluator_receipt = None;
        assert_eq!(
            evaluate_underwriting_certificate(inputs(&obligations))
                .unwrap()
                .decision,
            UnderwritingDecision::NotSupportable
        );
    }

    #[test]
    fn conditional_critical_obligation_never_becomes_unconditional_support() {
        let mut obligations = base_obligations();
        obligations[1].disposition = ProofDisposition::Conditional;
        obligations[1].invalidation_conditions = &["Pending signed POC allocation"];
        assert_eq!(
            evaluate_underwriting_certificate(inputs(&obligations))
                .unwrap()
                .decision,
            UnderwritingDecision::Conditional
        );
    }

    #[test]
    fn material_gap_produces_conditional_decision_but_informational_gap_does_not() {
        let mut obligations = base_obligations();
        let material = ProofObligation {
            obligation_id: "claim/warranty",
            claim_id: "warranty-eligibility",
            criticality: ObligationCriticality::Material,
            disposition: ProofDisposition::Unresolved,
            dependency_ids: &["claim/physical"],
            evidence_receipts: &[],
            evaluator_receipt: None,
            valid_from_utc: 0,
            valid_until_utc: None,
            invalidation_conditions: &[],
        };
        let all = [obligations[0], obligations[1], obligations[2], material];
        assert_eq!(
            evaluate_underwriting_certificate(inputs(&all))
                .unwrap()
                .decision,
            UnderwritingDecision::Conditional
        );

        obligations[0].criticality = ObligationCriticality::Informational;
        obligations[0].disposition = ProofDisposition::Unresolved;
        obligations[0].evidence_receipts = &[];
        obligations[0].evaluator_receipt = None;
        assert_eq!(
            evaluate_underwriting_certificate(inputs(&obligations))
                .unwrap()
                .decision,
            UnderwritingDecision::SupportableWithinDeclaredScope
        );
    }

    #[test]
    fn rejected_or_invalidated_critical_claim_produces_do_not_proceed() {
        let mut obligations = base_obligations();
        obligations[1].disposition = ProofDisposition::Invalidated;
        assert_eq!(
            evaluate_underwriting_certificate(inputs(&obligations))
                .unwrap()
                .decision,
            UnderwritingDecision::DoNotProceed
        );
    }

    #[test]
    fn future_known_receipts_and_expired_claims_fail_closed() {
        let mut obligations = base_obligations();
        obligations[0].evidence_receipts = &[ProofReceiptRef {
            available_at_utc: 101,
            ..receipt("source/future", DIGEST_A)
        }];
        assert_eq!(
            evaluate_underwriting_certificate(inputs(&obligations)),
            Err(UnderwritingCertificateError::EvidenceKnownAfterDecision)
        );

        obligations = base_obligations();
        obligations[1].valid_until_utc = Some(99);
        assert_eq!(
            evaluate_underwriting_certificate(inputs(&obligations)),
            Err(UnderwritingCertificateError::ClaimOutsideValidityWindow)
        );
    }

    #[test]
    fn dependency_cycles_missing_dependencies_and_unsupported_dependencies_fail_closed() {
        let mut obligations = base_obligations();
        obligations[1].dependency_ids = &["claim/physical"];
        assert_eq!(
            evaluate_underwriting_certificate(inputs(&obligations)),
            Err(UnderwritingCertificateError::DependencyCycle)
        );

        obligations = base_obligations();
        obligations[2].dependency_ids = &["claim/missing"];
        assert_eq!(
            evaluate_underwriting_certificate(inputs(&obligations)),
            Err(UnderwritingCertificateError::MissingDependency)
        );

        obligations = base_obligations();
        obligations[1].disposition = ProofDisposition::Conditional;
        obligations[1].invalidation_conditions = &["Await signed grid allocation"];
        assert_eq!(
            evaluate_underwriting_certificate(inputs(&obligations)),
            Err(UnderwritingCertificateError::DependencyNotQualified)
        );
    }

    #[test]
    fn qualified_claims_need_evidence_and_evaluator_receipts() {
        let mut obligations = base_obligations();
        obligations[0].evidence_receipts = &[];
        assert_eq!(
            evaluate_underwriting_certificate(inputs(&obligations)),
            Err(UnderwritingCertificateError::QualifiedClaimWithoutEvidence)
        );

        obligations = base_obligations();
        obligations[0].evaluator_receipt = None;
        assert_eq!(
            evaluate_underwriting_certificate(inputs(&obligations)),
            Err(UnderwritingCertificateError::EvaluatorReceiptMissing)
        );
    }

    #[test]
    fn duplicate_ids_and_invalid_manifest_digests_are_rejected() {
        let mut obligations = base_obligations();
        obligations[2].obligation_id = "claim/grid";
        assert_eq!(
            evaluate_underwriting_certificate(inputs(&obligations)),
            Err(UnderwritingCertificateError::DuplicateObligationId)
        );

        obligations = base_obligations();
        obligations[2].claim_id = "grid-rights";
        assert_eq!(
            evaluate_underwriting_certificate(inputs(&obligations)),
            Err(UnderwritingCertificateError::DuplicateClaimId)
        );

        let invalid = UnderwritingCertificateInputs {
            project_revision_sha256: "bad",
            obligations: &obligations,
            ..inputs(&obligations)
        };
        assert_eq!(
            evaluate_underwriting_certificate(invalid),
            Err(UnderwritingCertificateError::InvalidDigest)
        );
    }
}
