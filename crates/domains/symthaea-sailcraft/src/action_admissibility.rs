//! Synthetic sailcraft encounter/action-admissibility contract.
//!
//! This is a domain adapter over externally qualified ODD, encounter/rule,
//! model-feasibility, authority and human-readiness assessments. It does not
//! interpret COLREGs, predict target trajectories, calculate CPA/TCPA, run a
//! vessel dynamics model, or issue helm/sheet/actuator commands.
//!
//! An ADMISSIBLE result means only that the supplied assessments and exact
//! evidence references satisfy this synthetic P0 contract. It is not a safety
//! guarantee, navigation authorization, or permission to operate.

use std::collections::BTreeSet;

/// State attached to every evidence reference consumed by this contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceState {
    Qualified,
    Missing,
    Stale,
    Unverified,
    Conflicted,
    Revoked,
}

/// A digest is an identity/integrity pointer, not authentication or proof of
/// physical validity. Freshness is evaluated against the declared timebase.
#[derive(Clone, Debug, PartialEq)]
pub struct QualifiedEvidenceRef {
    pub state: EvidenceState,
    pub source_sha256: [u8; 32],
    pub timebase_id: String,
    pub age_seconds: f64,
    pub maximum_age_seconds: f64,
}

impl QualifiedEvidenceRef {
    pub fn qualified(
        source_sha256: [u8; 32],
        timebase_id: impl Into<String>,
        age_seconds: f64,
        maximum_age_seconds: f64,
    ) -> Self {
        Self {
            state: EvidenceState::Qualified,
            source_sha256,
            timebase_id: timebase_id.into(),
            age_seconds,
            maximum_age_seconds,
        }
    }

    pub fn missing() -> Self {
        Self {
            state: EvidenceState::Missing,
            source_sha256: [0; 32],
            timebase_id: String::new(),
            age_seconds: 0.0,
            maximum_age_seconds: 0.0,
        }
    }

    pub fn stale(source_sha256: [u8; 32], timebase_id: impl Into<String>) -> Self {
        Self {
            state: EvidenceState::Stale,
            source_sha256,
            timebase_id: timebase_id.into(),
            age_seconds: 1.0,
            maximum_age_seconds: 0.0,
        }
    }

    pub fn conflicted(source_sha256: [u8; 32], timebase_id: impl Into<String>) -> Self {
        Self {
            state: EvidenceState::Conflicted,
            source_sha256,
            timebase_id: timebase_id.into(),
            age_seconds: 0.0,
            maximum_age_seconds: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ManeuverClass {
    HoldCourse,
    TackPort,
    TackStarboard,
    Gybe,
    HeaveTo,
    ReefOrFurl,
}

/// A candidate label is a planner proposal, not a command or actuator output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidateManeuver {
    pub maneuver: ManeuverClass,
    pub target_course_deg: f64,
}

/// A separately evaluated encounter/rule result. This crate deliberately
/// consumes the upstream result; it does not implement COLREG interpretation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleAssessment {
    ConsistentWithCandidate,
    ConflictsWithCandidate,
    Ambiguous,
    Unknown,
}

/// Applicability of the candidate trajectory to the vessel-specific ODD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OddAssessment {
    WithinDeclaredDomain,
    OutsideDeclaredDomain,
    Unknown,
}

/// Feasibility result from a named, configuration-bound sailcraft model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelAssessment {
    Feasible,
    Infeasible,
    OutOfModelDomain,
    Unknown,
}

/// Whether the pre-existing responsible human is actually ready to adjudicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HumanReadiness {
    Ready,
    NotReady,
    Unknown,
}

/// Only maneuvers already present in this independently supplied envelope may
/// be considered. This is a ceiling, never a grant of control authority.
#[derive(Clone, Debug, PartialEq)]
pub struct AuthorityCeiling {
    pub allowed_maneuvers: BTreeSet<ManeuverClass>,
    pub evidence: QualifiedEvidenceRef,
}

/// Human oversight is required for ambiguity; the contract cannot create the
/// human role, presume acknowledgement, or mint a fallback capability.
#[derive(Clone, Debug, PartialEq)]
pub struct HumanOversight {
    pub readiness: HumanReadiness,
    pub evidence: QualifiedEvidenceRef,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActionAdmissibilityInput {
    pub candidate: CandidateManeuver,
    pub common_timebase_id: String,
    pub route_evidence: QualifiedEvidenceRef,
    pub encounter_evidence: QualifiedEvidenceRef,
    pub odd_evidence: QualifiedEvidenceRef,
    pub model_evidence: QualifiedEvidenceRef,
    pub authority: AuthorityCeiling,
    pub rule_assessment: RuleAssessment,
    pub odd_assessment: OddAssessment,
    pub model_assessment: ModelAssessment,
    pub human_oversight: HumanOversight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionAdmissibilityDisposition {
    AdmissibleWithinDeclaredModel,
    RejectedModelInfeasible,
    RejectedRuleConflict,
    AmbiguousRequiresHuman,
    NotEvaluatedMissingEvidence,
    NotEvaluatedStaleEvidence,
    ConflictedInput,
    OutOfScope,
    FallbackRequired,
    RejectedAuthorityCeiling,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActionAdmissibilityResult {
    pub disposition: ActionAdmissibilityDisposition,
    pub reason: &'static str,
    /// This contract is decision support only. It never emits an executable action.
    pub actuator_command_emitted: bool,
    pub authority_expanded: bool,
}

impl ActionAdmissibilityResult {
    fn new(
        disposition: ActionAdmissibilityDisposition,
        reason: &'static str,
    ) -> Self {
        Self {
            disposition,
            reason,
            actuator_command_emitted: false,
            authority_expanded: false,
        }
    }
}

fn evidence_status(
    evidence: &QualifiedEvidenceRef,
    common_timebase_id: &str,
) -> Result<(), ActionAdmissibilityDisposition> {
    match evidence.state {
        EvidenceState::Conflicted => {
            return Err(ActionAdmissibilityDisposition::ConflictedInput);
        }
        EvidenceState::Revoked => {
            return Err(ActionAdmissibilityDisposition::FallbackRequired);
        }
        EvidenceState::Stale => {
            return Err(ActionAdmissibilityDisposition::NotEvaluatedStaleEvidence);
        }
        EvidenceState::Missing | EvidenceState::Unverified => {
            return Err(ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence);
        }
        EvidenceState::Qualified => {}
    }

    if evidence.source_sha256 == [0; 32]
        || evidence.timebase_id.trim().is_empty()
        || evidence.timebase_id != common_timebase_id
        || !evidence.age_seconds.is_finite()
        || !evidence.maximum_age_seconds.is_finite()
        || evidence.age_seconds < 0.0
        || evidence.maximum_age_seconds < 0.0
    {
        return Err(ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence);
    }
    if evidence.age_seconds > evidence.maximum_age_seconds {
        return Err(ActionAdmissibilityDisposition::NotEvaluatedStaleEvidence);
    }
    Ok(())
}

fn result_for_evidence_error(error: ActionAdmissibilityDisposition) -> ActionAdmissibilityResult {
    match error {
        ActionAdmissibilityDisposition::ConflictedInput => ActionAdmissibilityResult::new(
            error,
            "one or more required evidence sources report conflicting state",
        ),
        ActionAdmissibilityDisposition::NotEvaluatedStaleEvidence => ActionAdmissibilityResult::new(
            error,
            "one or more required evidence sources are stale",
        ),
        ActionAdmissibilityDisposition::FallbackRequired => ActionAdmissibilityResult::new(
            error,
            "a required authority or qualification reference has been revoked",
        ),
        _ => ActionAdmissibilityResult::new(
            ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence,
            "a required evidence reference is missing, unverified, malformed or on a different timebase",
        ),
    }
}

/// Evaluate one candidate against external, already-qualified assessments.
///
/// Precedence is conservative and deterministic:
/// evidence integrity/currentness -> candidate validity -> ODD -> model
/// feasibility -> external rule assessment -> authority ceiling -> admission.
/// No path creates an actuator command or increases the supplied authority.
pub fn evaluate_action_admissibility(
    input: &ActionAdmissibilityInput,
) -> ActionAdmissibilityResult {
    let timebase = input.common_timebase_id.trim();
    if timebase.is_empty() {
        return ActionAdmissibilityResult::new(
            ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence,
            "common timebase identity is missing",
        );
    }

    let evidence = [
        &input.route_evidence,
        &input.encounter_evidence,
        &input.odd_evidence,
        &input.model_evidence,
        &input.authority.evidence,
    ];
    for reference in evidence {
        if let Err(error) = evidence_status(reference, timebase) {
            return result_for_evidence_error(error);
        }
    }

    if !input.candidate.target_course_deg.is_finite()
        || !(0.0..360.0).contains(&input.candidate.target_course_deg)
    {
        return ActionAdmissibilityResult::new(
            ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence,
            "candidate course is outside the canonical finite bearing contract",
        );
    }

    match input.odd_assessment {
        OddAssessment::OutsideDeclaredDomain => {
            return ActionAdmissibilityResult::new(
                ActionAdmissibilityDisposition::OutOfScope,
                "candidate trajectory is outside the vessel-specific ODD",
            );
        }
        OddAssessment::Unknown => {
            return ActionAdmissibilityResult::new(
                ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence,
                "ODD assessment is unknown",
            );
        }
        OddAssessment::WithinDeclaredDomain => {}
    }

    match input.model_assessment {
        ModelAssessment::Infeasible => {
            return ActionAdmissibilityResult::new(
                ActionAdmissibilityDisposition::RejectedModelInfeasible,
                "qualified sailcraft model reports the candidate maneuver infeasible",
            );
        }
        ModelAssessment::OutOfModelDomain => {
            return ActionAdmissibilityResult::new(
                ActionAdmissibilityDisposition::OutOfScope,
                "candidate maneuver exceeds the qualified sailcraft model domain",
            );
        }
        ModelAssessment::Unknown => {
            return ActionAdmissibilityResult::new(
                ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence,
                "sailcraft model feasibility is unknown",
            );
        }
        ModelAssessment::Feasible => {}
    }

    match input.rule_assessment {
        RuleAssessment::ConflictsWithCandidate => {
            return ActionAdmissibilityResult::new(
                ActionAdmissibilityDisposition::RejectedRuleConflict,
                "external encounter/rule assessment conflicts with the proposed maneuver",
            );
        }
        RuleAssessment::Unknown => {
            return ActionAdmissibilityResult::new(
                ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence,
                "external encounter/rule applicability is unknown",
            );
        }
        RuleAssessment::Ambiguous => {
            let oversight = &input.human_oversight;
            match evidence_status(&oversight.evidence, timebase) {
                Ok(()) if oversight.readiness == HumanReadiness::Ready => {
                    return ActionAdmissibilityResult::new(
                        ActionAdmissibilityDisposition::AmbiguousRequiresHuman,
                        "rule applicability is ambiguous; route to the pre-existing responsible human",
                    );
                }
                Err(ActionAdmissibilityDisposition::ConflictedInput) => {
                    return result_for_evidence_error(ActionAdmissibilityDisposition::ConflictedInput);
                }
                Err(ActionAdmissibilityDisposition::NotEvaluatedStaleEvidence) => {
                    return result_for_evidence_error(ActionAdmissibilityDisposition::NotEvaluatedStaleEvidence);
                }
                _ => {
                    return ActionAdmissibilityResult::new(
                        ActionAdmissibilityDisposition::FallbackRequired,
                        "rule applicability is ambiguous and a ready, qualified human reviewer is unavailable",
                    );
                }
            }
        }
        RuleAssessment::ConsistentWithCandidate => {}
    }

    if !input.authority.allowed_maneuvers.contains(&input.candidate.maneuver) {
        return ActionAdmissibilityResult::new(
            ActionAdmissibilityDisposition::RejectedAuthorityCeiling,
            "candidate maneuver exceeds the independently supplied authority ceiling",
        );
    }

    ActionAdmissibilityResult::new(
        ActionAdmissibilityDisposition::AdmissibleWithinDeclaredModel,
        "all supplied evidence, model, ODD, rule and authority predicates admit this candidate within the declared synthetic contract",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLOCK: &str = "synthetic-common-timebase";

    fn ev(byte: u8) -> QualifiedEvidenceRef {
        QualifiedEvidenceRef::qualified([byte; 32], CLOCK, 0.0, 1.0)
    }

    fn input() -> ActionAdmissibilityInput {
        ActionAdmissibilityInput {
            candidate: CandidateManeuver {
                maneuver: ManeuverClass::TackPort,
                target_course_deg: 315.0,
            },
            common_timebase_id: CLOCK.to_string(),
            route_evidence: ev(1),
            encounter_evidence: ev(2),
            odd_evidence: ev(3),
            model_evidence: ev(4),
            authority: AuthorityCeiling {
                allowed_maneuvers: BTreeSet::from([
                    ManeuverClass::HoldCourse,
                    ManeuverClass::TackPort,
                    ManeuverClass::TackStarboard,
                ]),
                evidence: ev(5),
            },
            rule_assessment: RuleAssessment::ConsistentWithCandidate,
            odd_assessment: OddAssessment::WithinDeclaredDomain,
            model_assessment: ModelAssessment::Feasible,
            human_oversight: HumanOversight {
                readiness: HumanReadiness::Ready,
                evidence: ev(6),
            },
        }
    }

    fn disposition(input: &ActionAdmissibilityInput) -> ActionAdmissibilityDisposition {
        evaluate_action_admissibility(input).disposition
    }

    #[test]
    fn candidate_is_admitted_only_with_all_qualified_external_assessments() {
        let result = evaluate_action_admissibility(&input());
        assert_eq!(
            result.disposition,
            ActionAdmissibilityDisposition::AdmissibleWithinDeclaredModel
        );
        assert!(!result.actuator_command_emitted);
        assert!(!result.authority_expanded);
    }

    #[test]
    fn missing_route_evidence_fails_closed() {
        let mut value = input();
        value.route_evidence = QualifiedEvidenceRef::missing();
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence
        );
    }

    #[test]
    fn stale_encounter_evidence_fails_closed() {
        let mut value = input();
        value.encounter_evidence = QualifiedEvidenceRef::stale([7; 32], CLOCK);
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::NotEvaluatedStaleEvidence
        );
    }

    #[test]
    fn conflicted_odd_evidence_fails_closed() {
        let mut value = input();
        value.odd_evidence = QualifiedEvidenceRef::conflicted([8; 32], CLOCK);
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::ConflictedInput
        );
    }

    #[test]
    fn revoked_authority_reference_requires_fallback() {
        let mut value = input();
        value.authority.evidence.state = EvidenceState::Revoked;
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::FallbackRequired
        );
    }

    #[test]
    fn cross_timebase_evidence_is_not_combined() {
        let mut value = input();
        value.route_evidence.timebase_id = "other-clock".to_string();
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence
        );
    }

    #[test]
    fn expired_freshness_window_is_not_treated_as_zero_uncertainty() {
        let mut value = input();
        value.model_evidence.age_seconds = 2.0;
        value.model_evidence.maximum_age_seconds = 1.0;
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::NotEvaluatedStaleEvidence
        );
    }

    #[test]
    fn out_of_odd_candidate_is_not_admitted() {
        let mut value = input();
        value.odd_assessment = OddAssessment::OutsideDeclaredDomain;
        assert_eq!(disposition(&value), ActionAdmissibilityDisposition::OutOfScope);
    }

    #[test]
    fn unknown_odd_assessment_is_not_admitted() {
        let mut value = input();
        value.odd_assessment = OddAssessment::Unknown;
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence
        );
    }

    #[test]
    fn physically_infeasible_candidate_is_rejected() {
        let mut value = input();
        value.model_assessment = ModelAssessment::Infeasible;
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::RejectedModelInfeasible
        );
    }

    #[test]
    fn model_extrapolation_is_out_of_scope() {
        let mut value = input();
        value.model_assessment = ModelAssessment::OutOfModelDomain;
        assert_eq!(disposition(&value), ActionAdmissibilityDisposition::OutOfScope);
    }

    #[test]
    fn unknown_model_result_is_not_admitted() {
        let mut value = input();
        value.model_assessment = ModelAssessment::Unknown;
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence
        );
    }

    #[test]
    fn externally_assessed_rule_conflict_rejects_candidate() {
        let mut value = input();
        value.rule_assessment = RuleAssessment::ConflictsWithCandidate;
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::RejectedRuleConflict
        );
    }

    #[test]
    fn ambiguous_rule_result_requests_a_ready_human() {
        let mut value = input();
        value.rule_assessment = RuleAssessment::Ambiguous;
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::AmbiguousRequiresHuman
        );
    }

    #[test]
    fn ambiguous_rule_without_ready_human_requires_predeclared_fallback() {
        let mut value = input();
        value.rule_assessment = RuleAssessment::Ambiguous;
        value.human_oversight.readiness = HumanReadiness::NotReady;
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::FallbackRequired
        );
    }

    #[test]
    fn unknown_rule_assessment_is_not_admitted() {
        let mut value = input();
        value.rule_assessment = RuleAssessment::Unknown;
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence
        );
    }

    #[test]
    fn candidate_cannot_expand_the_authority_ceiling() {
        let mut value = input();
        value.candidate.maneuver = ManeuverClass::Gybe;
        let result = evaluate_action_admissibility(&value);
        assert_eq!(
            result.disposition,
            ActionAdmissibilityDisposition::RejectedAuthorityCeiling
        );
        assert!(!result.authority_expanded);
        assert!(!result.actuator_command_emitted);
    }

    #[test]
    fn invalid_course_bearing_is_not_admitted() {
        let mut value = input();
        value.candidate.target_course_deg = 360.0;
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence
        );
    }

    #[test]
    fn non_finite_course_bearing_is_not_admitted() {
        let mut value = input();
        value.candidate.target_course_deg = f64::NAN;
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence
        );
    }

    #[test]
    fn ambiguous_human_evidence_stale_or_missing_never_admits_candidate() {
        let mut stale = input();
        stale.rule_assessment = RuleAssessment::Ambiguous;
        stale.human_oversight.evidence =
            QualifiedEvidenceRef::stale([9; 32], CLOCK);
        assert_eq!(
            disposition(&stale),
            ActionAdmissibilityDisposition::FallbackRequired
        );

        let mut missing = input();
        missing.rule_assessment = RuleAssessment::Ambiguous;
        missing.human_oversight.evidence = QualifiedEvidenceRef::missing();
        assert_eq!(
            disposition(&missing),
            ActionAdmissibilityDisposition::FallbackRequired
        );
    }

    #[test]
    fn common_timebase_must_be_present() {
        let mut value = input();
        value.common_timebase_id.clear();
        assert_eq!(
            disposition(&value),
            ActionAdmissibilityDisposition::NotEvaluatedMissingEvidence
        );
    }

    #[test]
    fn every_disposition_remains_decision_support_only() {
        let mut cases = Vec::new();
        let mut a = input(); a.route_evidence = QualifiedEvidenceRef::missing(); cases.push(a);
        let mut b = input(); b.encounter_evidence = QualifiedEvidenceRef::stale([7; 32], CLOCK); cases.push(b);
        let mut c = input(); c.odd_evidence = QualifiedEvidenceRef::conflicted([8; 32], CLOCK); cases.push(c);
        let mut d = input(); d.authority.evidence.state = EvidenceState::Revoked; cases.push(d);
        let mut e = input(); e.odd_assessment = OddAssessment::OutsideDeclaredDomain; cases.push(e);
        let mut f = input(); f.model_assessment = ModelAssessment::Infeasible; cases.push(f);
        let mut g = input(); g.rule_assessment = RuleAssessment::ConflictsWithCandidate; cases.push(g);
        let mut h = input(); h.rule_assessment = RuleAssessment::Ambiguous; cases.push(h);
        let mut i = input(); i.candidate.maneuver = ManeuverClass::Gybe; cases.push(i);
        cases.push(input());

        for case in &cases {
            let result = evaluate_action_admissibility(case);
            assert!(!result.actuator_command_emitted);
            assert!(!result.authority_expanded);
        }
    }
}
