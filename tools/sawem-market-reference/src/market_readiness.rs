//! Source-qualified market readiness evaluation.
//!
//! Readiness is not a single launch-date boolean. Each prerequisite declares
//! its scope, responsible authority and minimum evidence class. Evidence is
//! evaluated at both valid-time and known-at time. The evaluator does not
//! infer the law, determine which prerequisites apply, authenticate upstream
//! source bytes, or establish a real participant's SAWEM admission.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessScope {
    InternalTesting,
    ExternalCommercial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessAuthority {
    Regulator,
    MarketOperator,
    GridOperator,
    Participant,
    Counterparty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessEvidenceClass {
    OfficialAuthorityRecord,
    OperatorParticipantRecord,
    ParticipantSpecificRecord,
    SecondaryReport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessEvidenceDisposition {
    Satisfied,
    NotSatisfied,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadinessRequirement<'a> {
    pub requirement_id: &'a str,
    pub scope: ReadinessScope,
    pub required_authority: ReadinessAuthority,
    pub required_evidence_class: ReadinessEvidenceClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadinessEvidence<'a> {
    pub evidence_id: &'a str,
    pub requirement_id: &'a str,
    pub scope: ReadinessScope,
    pub authority: ReadinessAuthority,
    pub evidence_class: ReadinessEvidenceClass,
    pub disposition: ReadinessEvidenceDisposition,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub published_at_utc: i64,
    pub available_at_utc: i64,
    pub valid_from_utc: i64,
    /// Exclusive end; None means no declared end date.
    pub valid_until_utc: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarketReadinessInputs<'a> {
    pub assessment_id: &'a str,
    pub scope: ReadinessScope,
    /// Reconstructs what could defensibly be known at this decision time.
    pub decision_time_utc: i64,
    /// Requirements are supplied by an explicitly versioned market/profile owner.
    pub requirement_profile_ref: &'a str,
    pub requirement_profile_sha256: &'a str,
    pub requirements: &'a [ReadinessRequirement<'a>],
    pub evidence: &'a [ReadinessEvidence<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketReadinessError {
    InvalidEvidenceReference,
    InvalidDigest,
    InvalidValidityWindow,
    InvalidPublicationChronology,
    DuplicateRequirement,
    DuplicateEvidenceId,
    ConflictingEvidence,
    NoRequirementsForScope,
    EmptyRequirementSet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketReadinessState {
    Ready,
    NotReady,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarketReadinessAssessment<'a> {
    pub assessment_id: &'a str,
    pub scope: ReadinessScope,
    pub decision_time_utc: i64,
    pub state: MarketReadinessState,
    pub satisfied_requirement_ids: Vec<&'a str>,
    pub blocked_requirement_ids: Vec<&'a str>,
    pub unresolved_requirement_ids: Vec<&'a str>,
    pub future_known_excluded_count: u32,
    pub scope_mismatch_excluded_count: u32,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn evidence_is_valid_at(evidence: ReadinessEvidence<'_>, decision_time_utc: i64) -> bool {
    evidence.valid_from_utc <= decision_time_utc
        && evidence
            .valid_until_utc
            .map(|end| decision_time_utc < end)
            .unwrap_or(true)
}

/// Evaluate one declared readiness scope using a frozen requirement profile.
/// Only exactly-authoritative, correctly-typed and temporally valid evidence
/// available by the decision time can satisfy or block a requirement.
/// Secondary reports may inform research, but cannot prove regulatory approval
/// or participant-specific admission where a stronger class is required.
pub fn assess_market_readiness<'a>(
    inputs: MarketReadinessInputs<'a>,
) -> Result<MarketReadinessAssessment<'a>, MarketReadinessError> {
    if !nonempty(inputs.assessment_id) || !nonempty(inputs.requirement_profile_ref) {
        return Err(MarketReadinessError::InvalidEvidenceReference);
    }
    if !valid_sha256(inputs.requirement_profile_sha256) {
        return Err(MarketReadinessError::InvalidDigest);
    }
    if inputs.requirements.is_empty() {
        return Err(MarketReadinessError::EmptyRequirementSet);
    }

    for (index, requirement) in inputs.requirements.iter().copied().enumerate() {
        if !nonempty(requirement.requirement_id) {
            return Err(MarketReadinessError::InvalidEvidenceReference);
        }
        for previous in &inputs.requirements[..index] {
            if previous.requirement_id == requirement.requirement_id
                && previous.scope == requirement.scope
            {
                return Err(MarketReadinessError::DuplicateRequirement);
            }
        }
    }

    if !inputs
        .requirements
        .iter()
        .any(|requirement| requirement.scope == inputs.scope)
    {
        return Err(MarketReadinessError::NoRequirementsForScope);
    }

    for (index, evidence) in inputs.evidence.iter().copied().enumerate() {
        if !nonempty(evidence.evidence_id)
            || !nonempty(evidence.requirement_id)
            || !nonempty(evidence.source_ref)
        {
            return Err(MarketReadinessError::InvalidEvidenceReference);
        }
        if !valid_sha256(evidence.source_sha256) {
            return Err(MarketReadinessError::InvalidDigest);
        }
        if evidence.valid_until_utc.map(|end| end <= evidence.valid_from_utc) == Some(true) {
            return Err(MarketReadinessError::InvalidValidityWindow);
        }
        if evidence.available_at_utc < evidence.published_at_utc {
            return Err(MarketReadinessError::InvalidPublicationChronology);
        }
        for previous in &inputs.evidence[..index] {
            if previous.evidence_id == evidence.evidence_id {
                return Err(MarketReadinessError::DuplicateEvidenceId);
            }
        }
    }

    let mut satisfied = Vec::new();
    let mut blocked = Vec::new();
    let mut unresolved = Vec::new();
    let mut future_known_excluded_count = 0_u32;
    let mut scope_mismatch_excluded_count = 0_u32;

    for requirement in inputs
        .requirements
        .iter()
        .copied()
        .filter(|requirement| requirement.scope == inputs.scope)
    {
        let mut selected: Option<ReadinessEvidence<'a>> = None;

        for evidence in inputs.evidence.iter().copied() {
            if evidence.requirement_id != requirement.requirement_id {
                continue;
            }
            if evidence.scope != inputs.scope {
                scope_mismatch_excluded_count =
                    scope_mismatch_excluded_count.saturating_add(1);
                continue;
            }
            if evidence.available_at_utc > inputs.decision_time_utc
                || evidence.published_at_utc > inputs.decision_time_utc
            {
                future_known_excluded_count =
                    future_known_excluded_count.saturating_add(1);
                continue;
            }
            if !evidence_is_valid_at(evidence, inputs.decision_time_utc) {
                continue;
            }
            if evidence.authority != requirement.required_authority
                || evidence.evidence_class != requirement.required_evidence_class
            {
                continue;
            }
            if selected.is_some() {
                return Err(MarketReadinessError::ConflictingEvidence);
            }
            selected = Some(evidence);
        }

        match selected.map(|evidence| evidence.disposition) {
            Some(ReadinessEvidenceDisposition::Satisfied) => {
                satisfied.push(requirement.requirement_id);
            }
            Some(ReadinessEvidenceDisposition::NotSatisfied) => {
                blocked.push(requirement.requirement_id);
            }
            Some(ReadinessEvidenceDisposition::Unknown) | None => {
                unresolved.push(requirement.requirement_id);
            }
        }
    }

    let state = if !blocked.is_empty() {
        MarketReadinessState::NotReady
    } else if !unresolved.is_empty() {
        MarketReadinessState::Indeterminate
    } else {
        MarketReadinessState::Ready
    };

    Ok(MarketReadinessAssessment {
        assessment_id: inputs.assessment_id,
        scope: inputs.scope,
        decision_time_utc: inputs.decision_time_utc,
        state,
        satisfied_requirement_ids: satisfied,
        blocked_requirement_ids: blocked,
        unresolved_requirement_ids: unresolved,
        future_known_excluded_count,
        scope_mismatch_excluded_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE_SHA: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const SOURCE_SHA: &str =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn requirement(
        requirement_id: &'static str,
        authority: ReadinessAuthority,
        evidence_class: ReadinessEvidenceClass,
    ) -> ReadinessRequirement<'static> {
        ReadinessRequirement {
            requirement_id,
            scope: ReadinessScope::ExternalCommercial,
            required_authority: authority,
            required_evidence_class: evidence_class,
        }
    }

    fn evidence(
        evidence_id: &'static str,
        requirement_id: &'static str,
        authority: ReadinessAuthority,
        evidence_class: ReadinessEvidenceClass,
        disposition: ReadinessEvidenceDisposition,
        available_at_utc: i64,
    ) -> ReadinessEvidence<'static> {
        ReadinessEvidence {
            evidence_id,
            requirement_id,
            scope: ReadinessScope::ExternalCommercial,
            authority,
            evidence_class,
            disposition,
            source_ref: evidence_id,
            source_sha256: SOURCE_SHA,
            published_at_utc: 90,
            available_at_utc,
            valid_from_utc: 95,
            valid_until_utc: None,
        }
    }

    fn inputs<'a>(
        requirements: &'a [ReadinessRequirement<'a>],
        evidence: &'a [ReadinessEvidence<'a>],
    ) -> MarketReadinessInputs<'a> {
        MarketReadinessInputs {
            assessment_id: "sawem/external-readiness/2026-11-06",
            scope: ReadinessScope::ExternalCommercial,
            decision_time_utc: 100,
            requirement_profile_ref: "profile/sawem-external-prerequisites/v1",
            requirement_profile_sha256: PROFILE_SHA,
            requirements,
            evidence,
        }
    }

    #[test]
    fn all_exact_authority_and_evidence_requirements_satisfied_returns_ready() {
        let requirements = [
            requirement(
                "market-code-approval",
                ReadinessAuthority::Regulator,
                ReadinessEvidenceClass::OfficialAuthorityRecord,
            ),
            requirement(
                "participant-admission",
                ReadinessAuthority::MarketOperator,
                ReadinessEvidenceClass::ParticipantSpecificRecord,
            ),
        ];
        let evidence = [
            evidence(
                "nersa-approval",
                "market-code-approval",
                ReadinessAuthority::Regulator,
                ReadinessEvidenceClass::OfficialAuthorityRecord,
                ReadinessEvidenceDisposition::Satisfied,
                98,
            ),
            evidence(
                "mo-admission",
                "participant-admission",
                ReadinessAuthority::MarketOperator,
                ReadinessEvidenceClass::ParticipantSpecificRecord,
                ReadinessEvidenceDisposition::Satisfied,
                99,
            ),
        ];
        let result = assess_market_readiness(inputs(&requirements, &evidence)).unwrap();
        assert_eq!(result.state, MarketReadinessState::Ready);
        assert_eq!(result.satisfied_requirement_ids.len(), 2);
        assert!(result.unresolved_requirement_ids.is_empty());
    }

    #[test]
    fn secondary_launch_reporting_cannot_satisfy_regulatory_approval() {
        let requirements = [requirement(
            "market-code-approval",
            ReadinessAuthority::Regulator,
            ReadinessEvidenceClass::OfficialAuthorityRecord,
        )];
        let evidence = [evidence(
            "industry-launch-article",
            "market-code-approval",
            ReadinessAuthority::Regulator,
            ReadinessEvidenceClass::SecondaryReport,
            ReadinessEvidenceDisposition::Satisfied,
            99,
        )];
        let result = assess_market_readiness(inputs(&requirements, &evidence)).unwrap();
        assert_eq!(result.state, MarketReadinessState::Indeterminate);
        assert_eq!(
            result.unresolved_requirement_ids,
            vec!["market-code-approval"]
        );
    }

    #[test]
    fn explicit_authoritative_not_approved_evidence_blocks_readiness() {
        let requirements = [requirement(
            "market-code-approval",
            ReadinessAuthority::Regulator,
            ReadinessEvidenceClass::OfficialAuthorityRecord,
        )];
        let evidence = [evidence(
            "nersa-status",
            "market-code-approval",
            ReadinessAuthority::Regulator,
            ReadinessEvidenceClass::OfficialAuthorityRecord,
            ReadinessEvidenceDisposition::NotSatisfied,
            99,
        )];
        let result = assess_market_readiness(inputs(&requirements, &evidence)).unwrap();
        assert_eq!(result.state, MarketReadinessState::NotReady);
        assert_eq!(result.blocked_requirement_ids, vec!["market-code-approval"]);
    }

    #[test]
    fn future_known_and_not_yet_effective_evidence_cannot_satisfy_requirement() {
        let requirements = [requirement(
            "market-code-approval",
            ReadinessAuthority::Regulator,
            ReadinessEvidenceClass::OfficialAuthorityRecord,
        )];
        let future_known = [evidence(
            "future-approval",
            "market-code-approval",
            ReadinessAuthority::Regulator,
            ReadinessEvidenceClass::OfficialAuthorityRecord,
            ReadinessEvidenceDisposition::Satisfied,
            101,
        )];
        let result = assess_market_readiness(inputs(&requirements, &future_known)).unwrap();
        assert_eq!(result.state, MarketReadinessState::Indeterminate);
        assert_eq!(result.future_known_excluded_count, 1);

        let not_yet_effective = [ReadinessEvidence {
            valid_from_utc: 101,
            ..evidence(
                "published-future-effective",
                "market-code-approval",
                ReadinessAuthority::Regulator,
                ReadinessEvidenceClass::OfficialAuthorityRecord,
                ReadinessEvidenceDisposition::Satisfied,
                99,
            )
        }];
        let result = assess_market_readiness(inputs(&requirements, &not_yet_effective)).unwrap();
        assert_eq!(result.state, MarketReadinessState::Indeterminate);
    }

    #[test]
    fn expired_evidence_and_wrong_authority_remain_unresolved() {
        let requirements = [requirement(
            "market-code-approval",
            ReadinessAuthority::Regulator,
            ReadinessEvidenceClass::OfficialAuthorityRecord,
        )];
        let expired = [ReadinessEvidence {
            valid_until_utc: Some(100),
            ..evidence(
                "expired-approval",
                "market-code-approval",
                ReadinessAuthority::Regulator,
                ReadinessEvidenceClass::OfficialAuthorityRecord,
                ReadinessEvidenceDisposition::Satisfied,
                99,
            )
        }];
        assert_eq!(
            assess_market_readiness(inputs(&requirements, &expired))
                .unwrap()
                .state,
            MarketReadinessState::Indeterminate
        );

        let wrong_authority = [evidence(
            "operator-assertion",
            "market-code-approval",
            ReadinessAuthority::MarketOperator,
            ReadinessEvidenceClass::OfficialAuthorityRecord,
            ReadinessEvidenceDisposition::Satisfied,
            99,
        )];
        assert_eq!(
            assess_market_readiness(inputs(&requirements, &wrong_authority))
                .unwrap()
                .state,
            MarketReadinessState::Indeterminate
        );
    }

    #[test]
    fn internal_testing_evidence_does_not_satisfy_external_commercial_requirement() {
        let requirements = [requirement(
            "participant-admission",
            ReadinessAuthority::MarketOperator,
            ReadinessEvidenceClass::ParticipantSpecificRecord,
        )];
        let internal = [ReadinessEvidence {
            scope: ReadinessScope::InternalTesting,
            ..evidence(
                "test-cohort-invitation",
                "participant-admission",
                ReadinessAuthority::MarketOperator,
                ReadinessEvidenceClass::ParticipantSpecificRecord,
                ReadinessEvidenceDisposition::Satisfied,
                99,
            )
        }];
        let result = assess_market_readiness(inputs(&requirements, &internal)).unwrap();
        assert_eq!(result.state, MarketReadinessState::Indeterminate);
        assert_eq!(result.scope_mismatch_excluded_count, 1);
    }

    #[test]
    fn duplicate_requirement_evidence_identity_and_source_are_rejected() {
        let duplicate_requirements = [
            requirement(
                "market-code-approval",
                ReadinessAuthority::Regulator,
                ReadinessEvidenceClass::OfficialAuthorityRecord,
            ),
            requirement(
                "market-code-approval",
                ReadinessAuthority::Regulator,
                ReadinessEvidenceClass::OfficialAuthorityRecord,
            ),
        ];
        assert_eq!(
            assess_market_readiness(inputs(&duplicate_requirements, &[])),
            Err(MarketReadinessError::DuplicateRequirement)
        );

        let requirements = [requirement(
            "market-code-approval",
            ReadinessAuthority::Regulator,
            ReadinessEvidenceClass::OfficialAuthorityRecord,
        )];
        let duplicate_ids = [
            evidence(
                "same-evidence",
                "market-code-approval",
                ReadinessAuthority::Regulator,
                ReadinessEvidenceClass::OfficialAuthorityRecord,
                ReadinessEvidenceDisposition::Satisfied,
                99,
            ),
            evidence(
                "same-evidence",
                "market-code-approval",
                ReadinessAuthority::Regulator,
                ReadinessEvidenceClass::OfficialAuthorityRecord,
                ReadinessEvidenceDisposition::NotSatisfied,
                99,
            ),
        ];
        assert_eq!(
            assess_market_readiness(inputs(&requirements, &duplicate_ids)),
            Err(MarketReadinessError::DuplicateEvidenceId)
        );

        let shared_source = [
            evidence(
                "evidence-one",
                "market-code-approval",
                ReadinessAuthority::Regulator,
                ReadinessEvidenceClass::OfficialAuthorityRecord,
                ReadinessEvidenceDisposition::Satisfied,
                99,
            ),
            ReadinessEvidence {
                requirement_id: "trading-rules-approval",
                source_ref: "evidence-one",
                ..evidence(
                    "evidence-two",
                    "trading-rules-approval",
                    ReadinessAuthority::Regulator,
                    ReadinessEvidenceClass::OfficialAuthorityRecord,
                    ReadinessEvidenceDisposition::Satisfied,
                    99,
                )
            },
        ];
        // One official instrument may support more than one separately scoped
        // requirement; evidence IDs remain unique, source identity may be shared.
        let other_requirement = requirement(
            "trading-rules-approval",
            ReadinessAuthority::Regulator,
            ReadinessEvidenceClass::OfficialAuthorityRecord,
        );
        assert_eq!(
            assess_market_readiness(inputs(&[requirements[0], other_requirement], &shared_source))
                .unwrap()
                .state,
            MarketReadinessState::Ready
        );
    }

    #[test]
    fn requirement_profile_must_contain_entries_for_requested_scope() {
        let internal_only = [ReadinessRequirement {
            scope: ReadinessScope::InternalTesting,
            ..requirement(
                "platform-test-access",
                ReadinessAuthority::MarketOperator,
                ReadinessEvidenceClass::OperatorParticipantRecord,
            )
        }];
        assert_eq!(
            assess_market_readiness(inputs(&internal_only, &[])),
            Err(MarketReadinessError::NoRequirementsForScope)
        );
    }

    #[test]
    fn conflicting_valid_primary_evidence_fails_instead_of_arbitrary_selection() {
        let requirements = [requirement(
            "market-code-approval",
            ReadinessAuthority::Regulator,
            ReadinessEvidenceClass::OfficialAuthorityRecord,
        )];
        let evidence = [
            evidence(
                "nersa-approved",
                "market-code-approval",
                ReadinessAuthority::Regulator,
                ReadinessEvidenceClass::OfficialAuthorityRecord,
                ReadinessEvidenceDisposition::Satisfied,
                99,
            ),
            evidence(
                "nersa-not-approved",
                "market-code-approval",
                ReadinessAuthority::Regulator,
                ReadinessEvidenceClass::OfficialAuthorityRecord,
                ReadinessEvidenceDisposition::NotSatisfied,
                99,
            ),
        ];
        assert_eq!(
            assess_market_readiness(inputs(&requirements, &evidence)),
            Err(MarketReadinessError::ConflictingEvidence)
        );
    }

    #[test]
    fn empty_requirement_profile_and_invalid_timestamps_or_digests_fail_closed() {
        let empty: [ReadinessRequirement<'static>; 0] = [];
        assert_eq!(
            assess_market_readiness(inputs(&empty, &[])),
            Err(MarketReadinessError::EmptyRequirementSet)
        );

        let requirements = [requirement(
            "market-code-approval",
            ReadinessAuthority::Regulator,
            ReadinessEvidenceClass::OfficialAuthorityRecord,
        )];
        let invalid_digest = [ReadinessEvidence {
            source_sha256: "not-a-digest",
            ..evidence(
                "bad-digest",
                "market-code-approval",
                ReadinessAuthority::Regulator,
                ReadinessEvidenceClass::OfficialAuthorityRecord,
                ReadinessEvidenceDisposition::Satisfied,
                99,
            )
        }];
        assert_eq!(
            assess_market_readiness(inputs(&requirements, &invalid_digest)),
            Err(MarketReadinessError::InvalidDigest)
        );

        let clock_inversion = [ReadinessEvidence {
            available_at_utc: 89,
            ..evidence(
                "clock-inversion",
                "market-code-approval",
                ReadinessAuthority::Regulator,
                ReadinessEvidenceClass::OfficialAuthorityRecord,
                ReadinessEvidenceDisposition::Satisfied,
                99,
            )
        }];
        assert_eq!(
            assess_market_readiness(inputs(&requirements, &clock_inversion)),
            Err(MarketReadinessError::InvalidPublicationChronology)
        );
    }
}
