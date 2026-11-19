//! Deterministic, fail-closed selection across qualified plasma closure profiles.
//! This validates caller declarations only; it does not classify physics or prove profiles.

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationStatus { Qualified, Provisional, Withdrawn }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RightsStatus { Verified, Unverified, Missing }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission { Granted, Denied, Unknown }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RightsEvidence {
    pub status: RightsStatus,
    pub license_identifier: Option<String>,
    pub execute_research: Permission,
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PhysicalGate {
    Conservation,
    FrameSymmetry,
    EntropyAdmissibility,
    WellPosedness,
    SpuriousModeStability,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateDisposition {
    Pass,
    Fail,
    NotApplicable,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalGateEvidence {
    pub gate: PhysicalGate,
    pub disposition: GateDisposition,
    /// Digest of the regime-specific mathematical/physical gate contract.
    pub contract_sha256: String,
    /// Digest of the result evidence. Unknown may omit this until evidence exists.
    pub evidence_sha256: Option<String>,
    /// Required for NotApplicable and Unknown; explains the scoped disposition.
    pub explanation: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterEnvelope {
    pub name: String,
    /// Exact canonical unit string. v0.1 performs no unit conversion.
    pub unit: String,
    pub minimum: i64,
    pub maximum: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureProfile {
    pub profile_id: String,
    pub physics_model_id: String,
    pub regime_id: String,
    pub dimensions: u8,
    pub qualification_status: QualificationStatus,
    pub model_artifact_sha256: String,
    pub source_artifact_sha256: String,
    pub solver_build_sha256: String,
    pub tableau_sha256: String,
    /// Independently issued evidence that the stated envelope was qualified.
    pub qualification_receipt_sha256: String,
    pub rights: RightsEvidence,
    /// Five explicit physical qualification gates; NotApplicable requires evidence and rationale.
    pub physical_gates: Vec<PhysicalGateEvidence>,
    pub parameter_envelopes: Vec<ParameterEnvelope>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterObservation {
    pub name: String,
    pub unit: String,
    /// Conservative measurement interval. Both endpoints are required.
    pub lower_value: Option<i64>,
    pub upper_value: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalState {
    pub physics_model_id: String,
    pub regime_id: String,
    pub dimensions: u8,
    pub solver_build_sha256: String,
    pub tableau_sha256: String,
    pub parameter_observations: Vec<ParameterObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateIssue {
    EmptyPhysicsModelId,
    EmptyRegimeId,
    ZeroDimensions,
    InvalidSolverBuildDigest,
    InvalidTableauDigest,
    EmptyObservationName(usize),
    EmptyObservationUnit(String),
    DuplicateObservationName(String),
    ReversedObservationInterval(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectionReason {
    EmptyProfileId,
    DuplicateProfileId(String),
    EmptyPhysicsModelId,
    EmptyRegimeId,
    ZeroDimensions,
    ProfileNotQualified,
    InvalidModelArtifactDigest,
    InvalidSourceArtifactDigest,
    InvalidSolverBuildDigest,
    InvalidTableauDigest,
    InvalidQualificationReceiptDigest,
    RightsNotVerified,
    ExecutionNotPermitted,
    MissingLicenseIdentifier,
    InvalidRightsEvidenceDigest,
    MissingPhysicalGate(PhysicalGate),
    DuplicatePhysicalGate(PhysicalGate),
    InvalidPhysicalGateContractDigest(PhysicalGate),
    InvalidPhysicalGateEvidenceDigest(PhysicalGate),
    PhysicalGateFailed(PhysicalGate),
    PhysicalGateUnknown(PhysicalGate),
    PhysicalGateNotApplicableWithoutRationale(PhysicalGate),
    PhysicalGateUnknownWithoutRationale(PhysicalGate),
    EmptyParameterEnvelopes,
    EmptyParameterName,
    EmptyParameterUnit(String),
    DuplicateParameterName(String),
    ReversedParameterEnvelope(String),
    PhysicsModelMismatch,
    RegimeMismatch,
    DimensionMismatch,
    SolverBuildMismatch,
    TableauMismatch,
    MissingObservation(String),
    UnprofiledObservation(String),
    ParameterUnitMismatch(String),
    UnknownMeasurement(String),
    OutsideEnvelope(String),
    MeasurementUncertaintyCrossesEnvelope(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionDisposition { Selected, Unknown, Abstained, Ambiguous }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileDecision {
    pub profile_id: String,
    pub eligible: bool,
    pub uncertainty_blocker: bool,
    pub reasons: Vec<RejectionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionReport {
    pub disposition: SelectionDisposition,
    pub selected_profile_id: Option<String>,
    pub candidate_profile_ids: Vec<String>,
    pub decisions: Vec<ProfileDecision>,
    pub state_issues: Vec<StateIssue>,
}

fn non_empty(value: &str) -> bool { !value.trim().is_empty() }

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn push_unique<T: PartialEq>(items: &mut Vec<T>, item: T) {
    if !items.contains(&item) { items.push(item); }
}

fn validate_state(state: &LocalState) -> Vec<StateIssue> {
    let mut issues = Vec::new();
    if !non_empty(&state.physics_model_id) { push_unique(&mut issues, StateIssue::EmptyPhysicsModelId); }
    if !non_empty(&state.regime_id) { push_unique(&mut issues, StateIssue::EmptyRegimeId); }
    if state.dimensions == 0 { push_unique(&mut issues, StateIssue::ZeroDimensions); }
    if !is_sha256(&state.solver_build_sha256) { push_unique(&mut issues, StateIssue::InvalidSolverBuildDigest); }
    if !is_sha256(&state.tableau_sha256) { push_unique(&mut issues, StateIssue::InvalidTableauDigest); }

    let mut names = BTreeSet::new();
    for (index, observation) in state.parameter_observations.iter().enumerate() {
        if !non_empty(&observation.name) { push_unique(&mut issues, StateIssue::EmptyObservationName(index)); }
        if !non_empty(&observation.unit) {
            push_unique(&mut issues, StateIssue::EmptyObservationUnit(observation.name.clone()));
        }
        if !names.insert(observation.name.as_str()) {
            push_unique(&mut issues, StateIssue::DuplicateObservationName(observation.name.clone()));
        }
        if let (Some(lower), Some(upper)) = (observation.lower_value, observation.upper_value) {
            if lower > upper {
                push_unique(&mut issues, StateIssue::ReversedObservationInterval(observation.name.clone()));
            }
        }
    }
    issues
}

fn validate_profile_shape(profile: &ClosureProfile, reasons: &mut Vec<RejectionReason>) {
    if !non_empty(&profile.profile_id) { push_unique(reasons, RejectionReason::EmptyProfileId); }
    if !non_empty(&profile.physics_model_id) { push_unique(reasons, RejectionReason::EmptyPhysicsModelId); }
    if !non_empty(&profile.regime_id) { push_unique(reasons, RejectionReason::EmptyRegimeId); }
    if profile.dimensions == 0 { push_unique(reasons, RejectionReason::ZeroDimensions); }
    if profile.qualification_status != QualificationStatus::Qualified {
        push_unique(reasons, RejectionReason::ProfileNotQualified);
    }
    if !is_sha256(&profile.model_artifact_sha256) { push_unique(reasons, RejectionReason::InvalidModelArtifactDigest); }
    if !is_sha256(&profile.source_artifact_sha256) { push_unique(reasons, RejectionReason::InvalidSourceArtifactDigest); }
    if !is_sha256(&profile.solver_build_sha256) { push_unique(reasons, RejectionReason::InvalidSolverBuildDigest); }
    if !is_sha256(&profile.tableau_sha256) { push_unique(reasons, RejectionReason::InvalidTableauDigest); }
    if !is_sha256(&profile.qualification_receipt_sha256) {
        push_unique(reasons, RejectionReason::InvalidQualificationReceiptDigest);
    }
    if profile.rights.status != RightsStatus::Verified { push_unique(reasons, RejectionReason::RightsNotVerified); }
    if profile.rights.execute_research != Permission::Granted {
        push_unique(reasons, RejectionReason::ExecutionNotPermitted);
    }
    if !profile.rights.license_identifier.as_deref().is_some_and(non_empty) {
        push_unique(reasons, RejectionReason::MissingLicenseIdentifier);
    }
    if !is_sha256(&profile.rights.evidence_sha256) {
        push_unique(reasons, RejectionReason::InvalidRightsEvidenceDigest);
    }
    if profile.parameter_envelopes.is_empty() { push_unique(reasons, RejectionReason::EmptyParameterEnvelopes); }

    let required_gates = [
        PhysicalGate::Conservation,
        PhysicalGate::FrameSymmetry,
        PhysicalGate::EntropyAdmissibility,
        PhysicalGate::WellPosedness,
        PhysicalGate::SpuriousModeStability,
    ];
    let mut seen_gates = BTreeSet::new();
    for gate_evidence in &profile.physical_gates {
        if !seen_gates.insert(gate_evidence.gate) {
            push_unique(
                reasons,
                RejectionReason::DuplicatePhysicalGate(gate_evidence.gate),
            );
        }
        if !is_sha256(&gate_evidence.contract_sha256) {
            push_unique(
                reasons,
                RejectionReason::InvalidPhysicalGateContractDigest(gate_evidence.gate),
            );
        }
        match gate_evidence.disposition {
            GateDisposition::Pass | GateDisposition::Fail | GateDisposition::NotApplicable => {
                if !gate_evidence.evidence_sha256.as_deref().is_some_and(is_sha256) {
                    push_unique(
                        reasons,
                        RejectionReason::InvalidPhysicalGateEvidenceDigest(gate_evidence.gate),
                    );
                }
            }
            GateDisposition::Unknown => {
                if gate_evidence
                    .evidence_sha256
                    .as_deref()
                    .is_some_and(|digest| !is_sha256(digest))
                {
                    push_unique(
                        reasons,
                        RejectionReason::InvalidPhysicalGateEvidenceDigest(gate_evidence.gate),
                    );
                }
            }
        }
        match gate_evidence.disposition {
            GateDisposition::Pass => {}
            GateDisposition::Fail => {
                push_unique(reasons, RejectionReason::PhysicalGateFailed(gate_evidence.gate));
            }
            GateDisposition::NotApplicable => {
                if !gate_evidence.explanation.as_deref().is_some_and(non_empty) {
                    push_unique(
                        reasons,
                        RejectionReason::PhysicalGateNotApplicableWithoutRationale(
                            gate_evidence.gate,
                        ),
                    );
                }
            }
            GateDisposition::Unknown => {
                if gate_evidence.explanation.as_deref().is_some_and(non_empty) {
                    push_unique(reasons, RejectionReason::PhysicalGateUnknown(gate_evidence.gate));
                } else {
                    push_unique(
                        reasons,
                        RejectionReason::PhysicalGateUnknownWithoutRationale(
                            gate_evidence.gate,
                        ),
                    );
                }
            }
        }
    }
    for required_gate in required_gates {
        if !seen_gates.contains(&required_gate) {
            push_unique(reasons, RejectionReason::MissingPhysicalGate(required_gate));
        }
    }

    let mut names = BTreeSet::new();
    for envelope in &profile.parameter_envelopes {
        if !non_empty(&envelope.name) { push_unique(reasons, RejectionReason::EmptyParameterName); }
        if !non_empty(&envelope.unit) {
            push_unique(reasons, RejectionReason::EmptyParameterUnit(envelope.name.clone()));
        }
        if !names.insert(envelope.name.as_str()) {
            push_unique(reasons, RejectionReason::DuplicateParameterName(envelope.name.clone()));
        }
        if envelope.minimum > envelope.maximum {
            push_unique(reasons, RejectionReason::ReversedParameterEnvelope(envelope.name.clone()));
        }
    }
}

fn evaluate_profile(profile: &ClosureProfile, state: &LocalState) -> (ProfileDecision, bool) {
    let mut reasons = Vec::new();
    validate_profile_shape(profile, &mut reasons);

    if profile.physics_model_id != state.physics_model_id { push_unique(&mut reasons, RejectionReason::PhysicsModelMismatch); }
    if profile.regime_id != state.regime_id { push_unique(&mut reasons, RejectionReason::RegimeMismatch); }
    if profile.dimensions != state.dimensions { push_unique(&mut reasons, RejectionReason::DimensionMismatch); }
    if profile.solver_build_sha256 != state.solver_build_sha256 {
        push_unique(&mut reasons, RejectionReason::SolverBuildMismatch);
    }
    if profile.tableau_sha256 != state.tableau_sha256 { push_unique(&mut reasons, RejectionReason::TableauMismatch); }

    let base_compatible = reasons.iter().all(|reason| {
        matches!(
            reason,
            RejectionReason::MissingObservation(_)
                | RejectionReason::UnknownMeasurement(_)
                | RejectionReason::MeasurementUncertaintyCrossesEnvelope(_)
                | RejectionReason::MissingPhysicalGate(_)
                | RejectionReason::PhysicalGateUnknown(_)
        )
    });
    if base_compatible {
        let mut envelope_names = BTreeSet::new();
        for envelope in &profile.parameter_envelopes {
            envelope_names.insert(envelope.name.as_str());
            let Some(observation) = state.parameter_observations.iter()
                .find(|observation| observation.name == envelope.name) else {
                push_unique(&mut reasons, RejectionReason::MissingObservation(envelope.name.clone()));
                continue;
            };

            if observation.unit != envelope.unit {
                push_unique(&mut reasons, RejectionReason::ParameterUnitMismatch(envelope.name.clone()));
                continue;
            }
            let (Some(lower), Some(upper)) = (observation.lower_value, observation.upper_value) else {
                push_unique(&mut reasons, RejectionReason::UnknownMeasurement(envelope.name.clone()));
                continue;
            };
            if upper < envelope.minimum || lower > envelope.maximum {
                push_unique(&mut reasons, RejectionReason::OutsideEnvelope(envelope.name.clone()));
            } else if lower < envelope.minimum || upper > envelope.maximum {
                push_unique(&mut reasons, RejectionReason::MeasurementUncertaintyCrossesEnvelope(envelope.name.clone()));
            }
        }
        for observation in &state.parameter_observations {
            if !envelope_names.contains(observation.name.as_str()) {
                push_unique(&mut reasons, RejectionReason::UnprofiledObservation(observation.name.clone()));
            }
        }
    }

    let uncertainty_blocker = base_compatible && !reasons.is_empty() && reasons.iter().all(|reason| {
        matches!(reason,
            RejectionReason::MissingObservation(_)
            | RejectionReason::UnknownMeasurement(_)
            | RejectionReason::MeasurementUncertaintyCrossesEnvelope(_)
            | RejectionReason::MissingPhysicalGate(_)
            | RejectionReason::PhysicalGateUnknown(_))
    });
    let eligible = reasons.is_empty();
    (ProfileDecision {
        profile_id: profile.profile_id.clone(),
        eligible,
        uncertainty_blocker,
        reasons,
    }, uncertainty_blocker)
}

/// Select only a unique, fully qualified profile with exact solver identity.
///
/// Unknown measurements produce Unknown. No compatible profile produces
/// Abstained. Multiple compatible profiles produce Ambiguous; no score or
/// arbitrary tie-break is invented in v0.1.
pub fn select_closure(profiles: &[ClosureProfile], state: &LocalState) -> SelectionReport {
    let state_issues = validate_state(state);
    if !state_issues.is_empty() {
        return SelectionReport {
            disposition: SelectionDisposition::Unknown,
            selected_profile_id: None,
            candidate_profile_ids: Vec::new(),
            decisions: Vec::new(),
            state_issues,
        };
    }
    if profiles.is_empty() {
        return SelectionReport {
            disposition: SelectionDisposition::Abstained,
            selected_profile_id: None,
            candidate_profile_ids: Vec::new(),
            decisions: Vec::new(),
            state_issues,
        };
    }

    let mut profile_ids = BTreeSet::new();
    let mut decisions = Vec::with_capacity(profiles.len());
    let mut candidates = Vec::new();
    let mut has_unknown_candidate = false;
    let mut has_duplicate_profile_id = false;

    for profile in profiles {
        if !profile_ids.insert(profile.profile_id.as_str()) {
            has_duplicate_profile_id = true;
            decisions.push(ProfileDecision {
                profile_id: profile.profile_id.clone(),
                eligible: false,
                uncertainty_blocker: false,
                reasons: vec![RejectionReason::DuplicateProfileId(profile.profile_id.clone())],
            });
            continue;
        }
        let (decision, uncertainty_blocker) = evaluate_profile(profile, state);
        if decision.eligible { candidates.push(profile.profile_id.clone()); }
        has_unknown_candidate |= uncertainty_blocker;
        decisions.push(decision);
    }

    decisions.sort_by(|left, right| left.profile_id.cmp(&right.profile_id));
    candidates.sort();

    let (disposition, selected_profile_id) = if has_duplicate_profile_id {
        (SelectionDisposition::Abstained, None)
    } else if candidates.len() > 1 {
        (SelectionDisposition::Ambiguous, None)
    } else if candidates.len() == 1 && !has_unknown_candidate {
        (SelectionDisposition::Selected, candidates.first().cloned())
    } else if has_unknown_candidate {
        (SelectionDisposition::Unknown, None)
    } else {
        (SelectionDisposition::Abstained, None)
    };

    SelectionReport { disposition, selected_profile_id, candidate_profile_ids: candidates, decisions, state_issues }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn profile() -> ClosureProfile {
        ClosureProfile {
            profile_id: "landau-linear-fno-v1".into(),
            physics_model_id: "vlasov-poisson-1d-electrostatic".into(),
            regime_id: "linear-landau-damping".into(),
            dimensions: 1,
            qualification_status: QualificationStatus::Qualified,
            model_artifact_sha256: DIGEST_A.into(),
            source_artifact_sha256: DIGEST_B.into(),
            solver_build_sha256: DIGEST_A.into(),
            tableau_sha256: DIGEST_B.into(),
            qualification_receipt_sha256: DIGEST_A.into(),
            rights: RightsEvidence {
                status: RightsStatus::Verified,
                license_identifier: Some("CC-BY-4.0".into()),
                execute_research: Permission::Granted,
                evidence_sha256: DIGEST_B.into(),
            },
            physical_gates: vec![
                PhysicalGate::Conservation,
                PhysicalGate::FrameSymmetry,
                PhysicalGate::EntropyAdmissibility,
                PhysicalGate::WellPosedness,
                PhysicalGate::SpuriousModeStability,
            ].into_iter().map(|gate| PhysicalGateEvidence {
                gate,
                disposition: GateDisposition::Pass,
                contract_sha256: DIGEST_A.into(),
                evidence_sha256: Some(DIGEST_B.into()),
                explanation: None,
            }).collect(),
            parameter_envelopes: vec![
                ParameterEnvelope { name: "k_lambda_D_milli".into(), unit: "1e-3".into(), minimum: 100, maximum: 300 },
                ParameterEnvelope { name: "amplitude_micro".into(), unit: "1e-6".into(), minimum: 10, maximum: 80 },
            ],
        }
    }

    fn state() -> LocalState {
        LocalState {
            physics_model_id: "vlasov-poisson-1d-electrostatic".into(),
            regime_id: "linear-landau-damping".into(),
            dimensions: 1,
            solver_build_sha256: DIGEST_A.into(),
            tableau_sha256: DIGEST_B.into(),
            parameter_observations: vec![
                ParameterObservation { name: "k_lambda_D_milli".into(), unit: "1e-3".into(), lower_value: Some(200), upper_value: Some(200) },
                ParameterObservation { name: "amplitude_micro".into(), unit: "1e-6".into(), lower_value: Some(40), upper_value: Some(40) },
            ],
        }
    }

    #[test]
    fn uniquely_qualified_profile_is_selected() {
        let report = select_closure(&[profile()], &state());
        assert_eq!(report.disposition, SelectionDisposition::Selected);
        assert_eq!(report.selected_profile_id.as_deref(), Some("landau-linear-fno-v1"));
    }

    #[test]
    fn overlapping_qualified_profiles_are_ambiguous_not_arbitrarily_ranked() {
        let mut second = profile();
        second.profile_id = "landau-linear-fno-v2".into();
        second.model_artifact_sha256 = DIGEST_B.into();
        let report = select_closure(&[profile(), second], &state());
        assert_eq!(report.disposition, SelectionDisposition::Ambiguous);
        assert_eq!(report.candidate_profile_ids.len(), 2);
        assert!(report.selected_profile_id.is_none());
    }

    #[test]
    fn definite_out_of_envelope_state_abstains() {
        let mut current = state();
        current.parameter_observations[1].lower_value = Some(90);
        current.parameter_observations[1].upper_value = Some(90);
        let report = select_closure(&[profile()], &current);
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.decisions[0].reasons.contains(&RejectionReason::OutsideEnvelope("amplitude_micro".into())));
    }

    #[test]
    fn missing_measurement_returns_unknown() {
        let mut current = state();
        current.parameter_observations[1].lower_value = None;
        current.parameter_observations[1].upper_value = None;
        let report = select_closure(&[profile()], &current);
        assert_eq!(report.disposition, SelectionDisposition::Unknown);
        assert!(report.decisions[0].reasons.contains(&RejectionReason::UnknownMeasurement("amplitude_micro".into())));
    }

    #[test]
    fn uncertainty_interval_crossing_envelope_boundary_returns_unknown() {
        let mut current = state();
        current.parameter_observations[1].lower_value = Some(70);
        current.parameter_observations[1].upper_value = Some(90);
        let report = select_closure(&[profile()], &current);
        assert_eq!(report.disposition, SelectionDisposition::Unknown);
        assert!(report.decisions[0].reasons.contains(
            &RejectionReason::MeasurementUncertaintyCrossesEnvelope("amplitude_micro".into())));
    }

    #[test]
    fn unknown_possible_candidate_prevents_arbitrary_selection() {
        let mut second = profile();
        second.profile_id = "possibly-overlapping-fno".into();
        second.parameter_envelopes[1].minimum = 35;
        second.parameter_envelopes[1].maximum = 60;
        let mut current = state();
        current.parameter_observations[1].lower_value = None;
        current.parameter_observations[1].upper_value = None;
        let report = select_closure(&[profile(), second], &current);
        assert_eq!(report.disposition, SelectionDisposition::Unknown);
        assert!(report.candidate_profile_ids.is_empty());
    }

    #[test]
    fn unverified_dataset_rights_block_selection() {
        let mut candidate = profile();
        candidate.rights.status = RightsStatus::Unverified;
        let report = select_closure(&[candidate], &state());
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.decisions[0].reasons.contains(&RejectionReason::RightsNotVerified));
    }

    #[test]
    fn provisional_profile_is_not_selectable() {
        let mut candidate = profile();
        candidate.qualification_status = QualificationStatus::Provisional;
        let report = select_closure(&[candidate], &state());
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.decisions[0].reasons.contains(&RejectionReason::ProfileNotQualified));
    }

    #[test]
    fn solver_build_must_match_exactly() {
        let mut current = state();
        current.solver_build_sha256 = DIGEST_B.into();
        let report = select_closure(&[profile()], &current);
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.decisions[0].reasons.contains(&RejectionReason::SolverBuildMismatch));
    }

    #[test]
    fn tableau_must_match_exactly() {
        let mut current = state();
        current.tableau_sha256 = DIGEST_A.into();
        let report = select_closure(&[profile()], &current);
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.decisions[0].reasons.contains(&RejectionReason::TableauMismatch));
    }

    #[test]
    fn unit_mismatch_is_not_silently_converted() {
        let mut current = state();
        current.parameter_observations[0].unit = "dimensionless".into();
        let report = select_closure(&[profile()], &current);
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.decisions[0].reasons.contains(&RejectionReason::ParameterUnitMismatch("k_lambda_D_milli".into())));
    }

    #[test]
    fn reversed_measurement_interval_returns_unknown() {
        let mut current = state();
        current.parameter_observations[0].lower_value = Some(250);
        current.parameter_observations[0].upper_value = Some(200);
        let report = select_closure(&[profile()], &current);
        assert_eq!(report.disposition, SelectionDisposition::Unknown);
        assert!(report.state_issues.contains(&StateIssue::ReversedObservationInterval("k_lambda_D_milli".into())));
    }

    #[test]
    fn missing_rights_or_referee_digest_blocks_selection() {
        let mut candidate = profile();
        candidate.rights.evidence_sha256 = "missing".into();
        candidate.qualification_receipt_sha256 = "bad".into();
        let report = select_closure(&[candidate], &state());
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.decisions[0].reasons.contains(&RejectionReason::InvalidRightsEvidenceDigest));
        assert!(report.decisions[0].reasons.contains(&RejectionReason::InvalidQualificationReceiptDigest));
    }

    #[test]
    fn duplicate_profile_identity_abstains() {
        let candidate = profile();
        let report = select_closure(&[candidate.clone(), candidate], &state());
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.selected_profile_id.is_none());
        assert!(report.decisions.iter().any(|decision| decision.reasons.contains(
            &RejectionReason::DuplicateProfileId("landau-linear-fno-v1".into()))));
    }

    #[test]
    fn denied_or_unknown_execution_permission_blocks_selection() {
        for permission in [Permission::Denied, Permission::Unknown] {
            let mut candidate = profile();
            candidate.rights.execute_research = permission;
            let report = select_closure(&[candidate], &state());
            assert_eq!(report.disposition, SelectionDisposition::Abstained);
            assert!(report.decisions[0].reasons.contains(&RejectionReason::ExecutionNotPermitted));
        }
    }

    #[test]
    fn unprofiled_local_parameter_is_not_ignored() {
        let mut current = state();
        current.parameter_observations.push(ParameterObservation {
            name: "unmodeled_temperature".into(),
            unit: "kelvin".into(),
            lower_value: Some(10),
            upper_value: Some(10),
        });
        let report = select_closure(&[profile()], &current);
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.decisions[0].reasons.contains(
            &RejectionReason::UnprofiledObservation("unmodeled_temperature".into())
        ));
    }

    #[test]
    fn missing_profile_parameter_is_unknown_not_a_wildcard() {
        let mut current = state();
        current.parameter_observations.remove(1);
        let report = select_closure(&[profile()], &current);
        assert_eq!(report.disposition, SelectionDisposition::Unknown);
        assert!(report.decisions[0].reasons.contains(
            &RejectionReason::MissingObservation("amplitude_micro".into())
        ));
    }

    #[test]
    fn unknown_physical_gate_prevents_selection() {
        let mut candidate = profile();
        let gate = &mut candidate.physical_gates[2];
        gate.disposition = GateDisposition::Unknown;
        gate.evidence_sha256 = None;
        gate.explanation = Some("Entropy convention or evidence remains unresolved".into());
        let report = select_closure(&[candidate], &state());
        assert_eq!(report.disposition, SelectionDisposition::Unknown);
        assert!(report.decisions[0].reasons.contains(
            &RejectionReason::PhysicalGateUnknown(PhysicalGate::EntropyAdmissibility)
        ));
    }

    #[test]
    fn failed_physical_gate_abstains() {
        let mut candidate = profile();
        candidate.physical_gates[1].disposition = GateDisposition::Fail;
        let report = select_closure(&[candidate], &state());
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.decisions[0].reasons.contains(
            &RejectionReason::PhysicalGateFailed(PhysicalGate::FrameSymmetry)
        ));
    }

    #[test]
    fn missing_physical_gate_returns_unknown() {
        let mut candidate = profile();
        candidate.physical_gates.retain(|gate| gate.gate != PhysicalGate::WellPosedness);
        let report = select_closure(&[candidate], &state());
        assert_eq!(report.disposition, SelectionDisposition::Unknown);
        assert!(report.decisions[0].reasons.contains(
            &RejectionReason::MissingPhysicalGate(PhysicalGate::WellPosedness)
        ));
    }

    #[test]
    fn duplicate_physical_gate_identity_abstains() {
        let mut candidate = profile();
        candidate.physical_gates.push(candidate.physical_gates[0].clone());
        let report = select_closure(&[candidate], &state());
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.decisions[0].reasons.contains(
            &RejectionReason::DuplicatePhysicalGate(PhysicalGate::Conservation)
        ));
    }

    #[test]
    fn not_applicable_gate_requires_evidence_and_rationale() {
        let mut candidate = profile();
        candidate.physical_gates[1].disposition = GateDisposition::NotApplicable;
        candidate.physical_gates[1].explanation = None;
        let report = select_closure(&[candidate], &state());
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.decisions[0].reasons.contains(
            &RejectionReason::PhysicalGateNotApplicableWithoutRationale(PhysicalGate::FrameSymmetry)
        ));
    }

    #[test]
    fn justified_not_applicable_gate_can_be_selected() {
        let mut candidate = profile();
        candidate.physical_gates[1].disposition = GateDisposition::NotApplicable;
        candidate.physical_gates[1].evidence_sha256 = Some(DIGEST_B.into());
        candidate.physical_gates[1].explanation =
            Some("The scoped equation model does not admit this transformation family".into());
        assert_eq!(
            select_closure(&[candidate], &state()).disposition,
            SelectionDisposition::Selected
        );
    }

    #[test]
    fn pass_gate_requires_evidence_digest() {
        let mut candidate = profile();
        candidate.physical_gates[0].evidence_sha256 = None;
        let report = select_closure(&[candidate], &state());
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.decisions[0].reasons.contains(
            &RejectionReason::InvalidPhysicalGateEvidenceDigest(PhysicalGate::Conservation)
        ));
    }

    #[test]
    fn unknown_gate_requires_explanation() {
        let mut candidate = profile();
        candidate.physical_gates[3].disposition = GateDisposition::Unknown;
        candidate.physical_gates[3].evidence_sha256 = None;
        candidate.physical_gates[3].explanation = None;
        let report = select_closure(&[candidate], &state());
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.decisions[0].reasons.contains(
            &RejectionReason::PhysicalGateUnknownWithoutRationale(PhysicalGate::WellPosedness)
        ));
    }

    #[test]
    fn no_profile_is_explicit_abstention() {
        let report = select_closure(&[], &state());
        assert_eq!(report.disposition, SelectionDisposition::Abstained);
        assert!(report.selected_profile_id.is_none());
    }

    #[test]
    fn inclusive_parameter_boundaries_are_supported() {
        let mut current = state();
        current.parameter_observations[1].lower_value = Some(80);
        current.parameter_observations[1].upper_value = Some(80);
        assert_eq!(select_closure(&[profile()], &current).disposition, SelectionDisposition::Selected);
    }

    #[test]
    fn profile_order_does_not_change_selection_report() {
        let first = profile();
        let mut second = profile();
        second.profile_id = "landau-linear-fno-v2".into();
        second.regime_id = "unrelated-regime".into();
        let current = state();
        let expected = select_closure(&[first.clone(), second.clone()], &current);
        let actual = select_closure(&[second, first], &current);
        assert_eq!(actual, expected);
    }

    #[test]
    fn unknown_local_state_identity_never_selects() {
        let mut current = state();
        current.tableau_sha256 = "unknown".into();
        let report = select_closure(&[profile()], &current);
        assert_eq!(report.disposition, SelectionDisposition::Unknown);
        assert!(report.state_issues.contains(&StateIssue::InvalidTableauDigest));
    }
}
