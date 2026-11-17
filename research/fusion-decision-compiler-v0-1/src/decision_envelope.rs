//! Decision-stability envelope over a frozen set of matched solver outputs.
//!
//! This is a deterministic finite-model-set audit. It does not average outputs,
//! estimate a probability distribution, or treat ensemble spread as a
//! confidence interval. Out-of-domain and non-converged runs remain visible
//! and prevent a complete decision envelope.

use std::collections::BTreeSet;

use super::cross_solver_comparison::{ApplicabilityState, ConvergenceState};
use super::evidence_dependence::{
    ObservableDependenceAssessment, RunDependencyManifest, compare_observable_dependencies,
};

pub const DECISION_ENVELOPE_VERSION: &str = "fusion-model-decision-envelope/0.1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThresholdDirection {
    AtLeast,
    AtMost,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThresholdResult {
    Pass,
    Fail,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionEnvelopeDisposition {
    AllModelsPassObservedEnvelope,
    AllModelsFailObservedEnvelope,
    DecisionSplit,
    IncompleteOrNotComparable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpreadDisposition {
    WithinDeclaredTolerance,
    ExceedsDeclaredTolerance,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionEnvelopeDiagnostic {
    EmptyCaseIdentity,
    EmptyCandidateIdentity,
    EmptyCandidateRevision,
    InvalidGeometryDigest,
    EmptyServiceBoundary,
    EmptyObservable,
    EmptyCanonicalQuantity,
    EmptyUnit,
    EmptyFrame,
    EmptySignConvention,
    EmptyTimeBasis,
    EmptyNormalizationContract,
    NonFiniteThreshold,
    InvalidSpreadTolerance,
    InvalidReferenceScale,
    TooFewModels,
    EmptyRunIdentity,
    EmptySolverIdentity,
    EmptySolverRevision,
    CandidateRevisionMismatch,
    GeometryMismatch,
    CaseIdentityMismatch,
    ServiceBoundaryMismatch,
    ObservableMismatch,
    CanonicalQuantityMismatch,
    UnitMismatch,
    FrameMismatch,
    SignConventionMismatch,
    TimeBasisMismatch,
    NormalizationMismatch,
    InvalidOutputDigest,
    EmptyRunReceipt,
    EmptyOutputEvidence,
    NonFiniteOutput,
    RunNotInDeclaredDomain,
    RunNotConverged,
    DuplicateRunIdentity,
    DependencyManifestRunMismatch,
    DependencyInventoryInvalid,
    DependencyManifestMissingOutput,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionEnvelopeProfile {
    pub case_id: String,
    pub candidate_id: String,
    pub candidate_revision: String,
    pub geometry_sha256: String,
    pub service_boundary_ref: String,
    pub observable_id: String,
    pub canonical_quantity_ref: String,
    pub unit_ref: String,
    pub frame_ref: String,
    pub sign_convention_ref: String,
    pub time_basis_ref: String,
    pub normalization_contract_ref: String,
    pub threshold_value: f64,
    pub threshold_direction: ThresholdDirection,
    pub threshold_inclusive: bool,
    pub absolute_spread_tolerance: f64,
    /// Positive physical scale used only for a descriptive relative range.
    pub relative_spread_reference_scale: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelEnvelopeInput {
    pub run_id: String,
    pub solver_id: String,
    pub solver_revision: String,
    pub case_id: String,
    pub candidate_id: String,
    pub candidate_revision: String,
    pub geometry_sha256: String,
    pub service_boundary_ref: String,
    pub observable_id: String,
    pub canonical_quantity_ref: String,
    pub unit_ref: String,
    pub frame_ref: String,
    pub sign_convention_ref: String,
    pub time_basis_ref: String,
    pub normalization_contract_ref: String,
    pub normalized_value: f64,
    pub output_sha256: String,
    pub run_receipt_ref: String,
    pub output_evidence_ref: String,
    pub applicability: ApplicabilityState,
    pub convergence: ConvergenceState,
    pub dependencies: RunDependencyManifest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelDecision {
    pub run_id: String,
    pub solver_id: String,
    pub solver_revision: String,
    pub output_sha256: String,
    pub run_receipt_ref: String,
    pub output_evidence_ref: String,
    pub threshold_result: ThresholdResult,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairDependenceReport {
    pub first_run_id: String,
    pub second_run_id: String,
    pub assessment: ObservableDependenceAssessment,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionEnvelopeReport {
    pub contract_version: &'static str,
    pub case_id: String,
    pub candidate_id: String,
    pub observable_id: String,
    pub disposition: DecisionEnvelopeDisposition,
    pub spread_disposition: SpreadDisposition,
    pub minimum_value: Option<f64>,
    pub maximum_value: Option<f64>,
    pub absolute_range: Option<f64>,
    pub relative_range: Option<f64>,
    pub threshold_value: f64,
    pub threshold_direction: ThresholdDirection,
    pub threshold_inclusive: bool,
    pub absolute_spread_tolerance: f64,
    pub model_decisions: Vec<ModelDecision>,
    pub dependence_reports: Vec<PairDependenceReport>,
    pub diagnostics: Vec<DecisionEnvelopeDiagnostic>,
    pub claim_ceiling: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionEnvelopePlan {
    pub profile: DecisionEnvelopeProfile,
    pub models: Vec<ModelEnvelopeInput>,
}

/// Evaluate a finite, frozen set of matched solver outputs against one
/// engineering threshold. Invalid/OOD/non-converged members are not filtered
/// out: the threshold envelope becomes incomplete.
pub fn evaluate_decision_envelope(plan: &DecisionEnvelopePlan) -> DecisionEnvelopeReport {
    let mut diagnostics = validate_profile(&plan.profile);
    if plan.models.len() < 2 {
        diagnostics.push(DecisionEnvelopeDiagnostic::TooFewModels);
    }

    let mut seen_runs = BTreeSet::new();
    for model in &plan.models {
        validate_model(&plan.profile, model, &mut diagnostics);
        if !seen_runs.insert(model.run_id.as_str()) {
            diagnostics.push(DecisionEnvelopeDiagnostic::DuplicateRunIdentity);
        }
    }

    let mut dependence_reports = Vec::new();
    if diagnostics.is_empty() {
        for first_index in 0..plan.models.len() {
            for second_index in (first_index + 1)..plan.models.len() {
                let first = &plan.models[first_index];
                let second = &plan.models[second_index];
                match compare_observable_dependencies(
                    &first.dependencies,
                    &second.dependencies,
                    std::slice::from_ref(&plan.profile.observable_id),
                ) {
                    Ok(mut assessments) => {
                        if let Some(assessment) = assessments.pop() {
                            dependence_reports.push(PairDependenceReport {
                                first_run_id: first.run_id.clone(),
                                second_run_id: second.run_id.clone(),
                                assessment,
                            });
                        }
                    }
                    Err(_) => {
                        diagnostics.push(DecisionEnvelopeDiagnostic::DependencyInventoryInvalid);
                    }
                }
            }
        }
    }

    dependence_reports.sort_by(|left, right| {
        (&left.first_run_id, &left.second_run_id)
            .cmp(&(&right.first_run_id, &right.second_run_id))
    });

    let mut model_decisions = Vec::new();
    let mut minimum = None::<f64>;
    let mut maximum = None::<f64>;

    if diagnostics.is_empty() {
        let mut models: Vec<&ModelEnvelopeInput> = plan.models.iter().collect();
        models.sort_by(|left, right| left.run_id.cmp(&right.run_id));
        for model in models {
            minimum = Some(minimum.map_or(model.normalized_value, |value| {
                value.min(model.normalized_value)
            }));
            maximum = Some(maximum.map_or(model.normalized_value, |value| {
                value.max(model.normalized_value)
            }));
            model_decisions.push(ModelDecision {
                run_id: model.run_id.clone(),
                solver_id: model.solver_id.clone(),
                solver_revision: model.solver_revision.clone(),
                output_sha256: model.output_sha256.clone(),
                run_receipt_ref: model.run_receipt_ref.clone(),
                output_evidence_ref: model.output_evidence_ref.clone(),
                threshold_result: threshold_result(
                    model.normalized_value,
                    plan.profile.threshold_value,
                    plan.profile.threshold_direction,
                    plan.profile.threshold_inclusive,
                ),
            });
        }
    }

    let absolute_range = minimum.zip(maximum).map(|(min_value, max_value)| max_value - min_value);
    let relative_range = absolute_range.zip(plan.profile.relative_spread_reference_scale)
        .map(|(range, scale)| range / scale);

    let (disposition, spread_disposition) = if !diagnostics.is_empty() {
        (
            DecisionEnvelopeDisposition::IncompleteOrNotComparable,
            SpreadDisposition::Unavailable,
        )
    } else {
        let pass_count = model_decisions
            .iter()
            .filter(|model| model.threshold_result == ThresholdResult::Pass)
            .count();
        let decision = if pass_count == model_decisions.len() {
            DecisionEnvelopeDisposition::AllModelsPassObservedEnvelope
        } else if pass_count == 0 {
            DecisionEnvelopeDisposition::AllModelsFailObservedEnvelope
        } else {
            DecisionEnvelopeDisposition::DecisionSplit
        };
        let spread = if absolute_range
            .is_some_and(|range| range <= plan.profile.absolute_spread_tolerance)
        {
            SpreadDisposition::WithinDeclaredTolerance
        } else {
            SpreadDisposition::ExceedsDeclaredTolerance
        };
        (decision, spread)
    };

    DecisionEnvelopeReport {
        contract_version: DECISION_ENVELOPE_VERSION,
        case_id: plan.profile.case_id.clone(),
        candidate_id: plan.profile.candidate_id.clone(),
        observable_id: plan.profile.observable_id.clone(),
        disposition,
        spread_disposition,
        minimum_value: minimum,
        maximum_value: maximum,
        absolute_range,
        relative_range,
        threshold_value: plan.profile.threshold_value,
        threshold_direction: plan.profile.threshold_direction,
        threshold_inclusive: plan.profile.threshold_inclusive,
        absolute_spread_tolerance: plan.profile.absolute_spread_tolerance,
        model_decisions,
        dependence_reports,
        diagnostics,
        claim_ceiling: "finite matched solver-set threshold and spread audit only",
    }
}

fn validate_profile(profile: &DecisionEnvelopeProfile) -> Vec<DecisionEnvelopeDiagnostic> {
    let mut diagnostics = Vec::new();
    for (value, diagnostic) in [
        (&profile.case_id, DecisionEnvelopeDiagnostic::EmptyCaseIdentity),
        (&profile.candidate_id, DecisionEnvelopeDiagnostic::EmptyCandidateIdentity),
        (&profile.candidate_revision, DecisionEnvelopeDiagnostic::EmptyCandidateRevision),
        (&profile.service_boundary_ref, DecisionEnvelopeDiagnostic::EmptyServiceBoundary),
        (&profile.observable_id, DecisionEnvelopeDiagnostic::EmptyObservable),
        (&profile.canonical_quantity_ref, DecisionEnvelopeDiagnostic::EmptyCanonicalQuantity),
        (&profile.unit_ref, DecisionEnvelopeDiagnostic::EmptyUnit),
        (&profile.frame_ref, DecisionEnvelopeDiagnostic::EmptyFrame),
        (&profile.sign_convention_ref, DecisionEnvelopeDiagnostic::EmptySignConvention),
        (&profile.time_basis_ref, DecisionEnvelopeDiagnostic::EmptyTimeBasis),
        (&profile.normalization_contract_ref, DecisionEnvelopeDiagnostic::EmptyNormalizationContract),
    ] {
        if value.trim().is_empty() {
            diagnostics.push(diagnostic);
        }
    }
    if !is_sha256(&profile.geometry_sha256) {
        diagnostics.push(DecisionEnvelopeDiagnostic::InvalidGeometryDigest);
    }
    if !profile.threshold_value.is_finite() {
        diagnostics.push(DecisionEnvelopeDiagnostic::NonFiniteThreshold);
    }
    if !profile.absolute_spread_tolerance.is_finite() || profile.absolute_spread_tolerance < 0.0 {
        diagnostics.push(DecisionEnvelopeDiagnostic::InvalidSpreadTolerance);
    }
    if profile.relative_spread_reference_scale
        .is_some_and(|scale| !scale.is_finite() || scale <= 0.0)
    {
        diagnostics.push(DecisionEnvelopeDiagnostic::InvalidReferenceScale);
    }
    diagnostics
}

fn validate_model(
    profile: &DecisionEnvelopeProfile,
    model: &ModelEnvelopeInput,
    diagnostics: &mut Vec<DecisionEnvelopeDiagnostic>,
) {
    for (value, diagnostic) in [
        (&model.run_id, DecisionEnvelopeDiagnostic::EmptyRunIdentity),
        (&model.solver_id, DecisionEnvelopeDiagnostic::EmptySolverIdentity),
        (&model.solver_revision, DecisionEnvelopeDiagnostic::EmptySolverRevision),
        (&model.run_receipt_ref, DecisionEnvelopeDiagnostic::EmptyRunReceipt),
        (&model.output_evidence_ref, DecisionEnvelopeDiagnostic::EmptyOutputEvidence),
    ] {
        if value.trim().is_empty() {
            diagnostics.push(diagnostic);
        }
    }
    if model.case_id != profile.case_id || model.candidate_id != profile.candidate_id {
        diagnostics.push(DecisionEnvelopeDiagnostic::CaseIdentityMismatch);
    }
    if model.candidate_revision != profile.candidate_revision {
        diagnostics.push(DecisionEnvelopeDiagnostic::CandidateRevisionMismatch);
    }
    if model.geometry_sha256 != profile.geometry_sha256 {
        diagnostics.push(DecisionEnvelopeDiagnostic::GeometryMismatch);
    }
    if model.service_boundary_ref != profile.service_boundary_ref {
        diagnostics.push(DecisionEnvelopeDiagnostic::ServiceBoundaryMismatch);
    }
    if model.observable_id != profile.observable_id {
        diagnostics.push(DecisionEnvelopeDiagnostic::ObservableMismatch);
    }
    if model.canonical_quantity_ref != profile.canonical_quantity_ref {
        diagnostics.push(DecisionEnvelopeDiagnostic::CanonicalQuantityMismatch);
    }
    if model.unit_ref != profile.unit_ref {
        diagnostics.push(DecisionEnvelopeDiagnostic::UnitMismatch);
    }
    if model.frame_ref != profile.frame_ref {
        diagnostics.push(DecisionEnvelopeDiagnostic::FrameMismatch);
    }
    if model.sign_convention_ref != profile.sign_convention_ref {
        diagnostics.push(DecisionEnvelopeDiagnostic::SignConventionMismatch);
    }
    if model.time_basis_ref != profile.time_basis_ref {
        diagnostics.push(DecisionEnvelopeDiagnostic::TimeBasisMismatch);
    }
    if model.normalization_contract_ref != profile.normalization_contract_ref {
        diagnostics.push(DecisionEnvelopeDiagnostic::NormalizationMismatch);
    }
    if !is_sha256(&model.output_sha256) {
        diagnostics.push(DecisionEnvelopeDiagnostic::InvalidOutputDigest);
    }
    if !model.normalized_value.is_finite() {
        diagnostics.push(DecisionEnvelopeDiagnostic::NonFiniteOutput);
    }
    match model.applicability {
        ApplicabilityState::InDeclaredDomain => {}
        ApplicabilityState::Unknown | ApplicabilityState::OutOfDomain => {
            diagnostics.push(DecisionEnvelopeDiagnostic::RunNotInDeclaredDomain);
        }
    }
    match model.convergence {
        ConvergenceState::Converged => {}
        ConvergenceState::NonConverged
        | ConvergenceState::NotReported
        | ConvergenceState::NotExecuted => {
            diagnostics.push(DecisionEnvelopeDiagnostic::RunNotConverged);
        }
    }
    if model.dependencies.run_id != model.run_id {
        diagnostics.push(DecisionEnvelopeDiagnostic::DependencyManifestRunMismatch);
    }
    if !model.dependencies.output_observables.contains(&profile.observable_id) {
        diagnostics.push(DecisionEnvelopeDiagnostic::DependencyManifestMissingOutput);
    }
}

fn threshold_result(
    value: f64,
    threshold: f64,
    direction: ThresholdDirection,
    inclusive: bool,
) -> ThresholdResult {
    let passes = match (direction, inclusive) {
        (ThresholdDirection::AtLeast, true) => value >= threshold,
        (ThresholdDirection::AtLeast, false) => value > threshold,
        (ThresholdDirection::AtMost, true) => value <= threshold,
        (ThresholdDirection::AtMost, false) => value < threshold,
    };
    if passes { ThresholdResult::Pass } else { ThresholdResult::Fail }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_dependence::{
        DependencyIdentity, DependencyKind, ObservableDependency,
    };

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn profile() -> DecisionEnvelopeProfile {
        DecisionEnvelopeProfile {
            case_id: "matched-case-001".to_string(),
            candidate_id: "candidate-42".to_string(),
            candidate_revision: "candidate-r7".to_string(),
            geometry_sha256: sha('a'),
            service_boundary_ref: "plant-net-export-boundary-v1".to_string(),
            observable_id: "net-electric-export".to_string(),
            canonical_quantity_ref: "quantity-net-electric-power".to_string(),
            unit_ref: "MWe".to_string(),
            frame_ref: "plant-terminal".to_string(),
            sign_convention_ref: "net-export-positive".to_string(),
            time_basis_ref: "steady-state-60s".to_string(),
            normalization_contract_ref: "normalization-net-power-v1".to_string(),
            threshold_value: 100.0,
            threshold_direction: ThresholdDirection::AtLeast,
            threshold_inclusive: true,
            absolute_spread_tolerance: 10.0,
            relative_spread_reference_scale: Some(100.0),
        }
    }

    fn dependencies(run_id: &str, node_id: &str, complete: bool) -> RunDependencyManifest {
        RunDependencyManifest {
            run_id: run_id.to_string(),
            output_observables: vec!["net-electric-export".to_string()],
            complete_observables: if complete {
                vec!["net-electric-export".to_string()]
            } else {
                Vec::new()
            },
            dependencies: vec![ObservableDependency {
                observable_id: "net-electric-export".to_string(),
                dependency: DependencyIdentity {
                    node_id: node_id.to_string(),
                    revision: "rev-1".to_string(),
                    kind: DependencyKind::SharedScenarioAssumption,
                    content_sha256: Some(sha('c')),
                    source_evidence_ref: Some(format!("evidence-{node_id}")),
                },
            }],
        }
    }

    fn model(run_id: &str, solver_id: &str, value: f64) -> ModelEnvelopeInput {
        let profile = profile();
        ModelEnvelopeInput {
            run_id: run_id.to_string(),
            solver_id: solver_id.to_string(),
            solver_revision: format!("{solver_id}-rev-1"),
            case_id: profile.case_id,
            candidate_id: profile.candidate_id,
            candidate_revision: profile.candidate_revision,
            geometry_sha256: profile.geometry_sha256,
            service_boundary_ref: profile.service_boundary_ref,
            observable_id: profile.observable_id,
            canonical_quantity_ref: profile.canonical_quantity_ref,
            unit_ref: profile.unit_ref,
            frame_ref: profile.frame_ref,
            sign_convention_ref: profile.sign_convention_ref,
            time_basis_ref: profile.time_basis_ref,
            normalization_contract_ref: profile.normalization_contract_ref,
            normalized_value: value,
            output_sha256: sha(if solver_id == "PROCESS" { 'b' } else { 'd' }),
            run_receipt_ref: format!("run-receipt-{run_id}"),
            output_evidence_ref: format!("observable-output-{run_id}"),
            applicability: ApplicabilityState::InDeclaredDomain,
            convergence: ConvergenceState::Converged,
            dependencies: dependencies(run_id, solver_id, true),
        }
    }

    fn plan(values: &[(&str, f64)]) -> DecisionEnvelopePlan {
        DecisionEnvelopePlan {
            profile: profile(),
            models: values
                .iter()
                .map(|(run, value)| {
                    let solver = if run.contains("process") { "PROCESS" } else { "SARAS" };
                    model(run, solver, *value)
                })
                .collect(),
        }
    }

    #[test]
    fn all_models_passing_produces_an_all_pass_observed_envelope() {
        let report = evaluate_decision_envelope(&plan(&[("run-process", 120.0), ("run-saras", 130.0)]));
        assert_eq!(report.disposition, DecisionEnvelopeDisposition::AllModelsPassObservedEnvelope);
        assert_eq!(report.spread_disposition, SpreadDisposition::WithinDeclaredTolerance);
    }

    #[test]
    fn all_models_failing_produces_an_all_fail_observed_envelope() {
        let report = evaluate_decision_envelope(&plan(&[("run-process", 70.0), ("run-saras", 80.0)]));
        assert_eq!(report.disposition, DecisionEnvelopeDisposition::AllModelsFailObservedEnvelope);
    }

    #[test]
    fn threshold_split_is_explicit_and_not_averaged_away() {
        let report = evaluate_decision_envelope(&plan(&[("run-process", 90.0), ("run-saras", 110.0)]));
        assert_eq!(report.disposition, DecisionEnvelopeDisposition::DecisionSplit);
        assert_eq!(report.minimum_value, Some(90.0));
        assert_eq!(report.maximum_value, Some(110.0));
    }

    #[test]
    fn threshold_equality_obeys_inclusive_policy() {
        let mut input = plan(&[("run-process", 100.0), ("run-saras", 110.0)]);
        assert_eq!(
            evaluate_decision_envelope(&input).model_decisions[0].threshold_result,
            ThresholdResult::Pass
        );
        input.profile.threshold_inclusive = false;
        assert_eq!(
            evaluate_decision_envelope(&input).model_decisions[0].threshold_result,
            ThresholdResult::Fail
        );
    }

    #[test]
    fn same_side_but_large_spread_is_reported_separately() {
        let report = evaluate_decision_envelope(&plan(&[("run-process", 120.0), ("run-saras", 180.0)]));
        assert_eq!(report.disposition, DecisionEnvelopeDisposition::AllModelsPassObservedEnvelope);
        assert_eq!(report.spread_disposition, SpreadDisposition::ExceedsDeclaredTolerance);
        assert_eq!(report.absolute_range, Some(60.0));
        assert_eq!(report.relative_range, Some(0.6));
    }

    #[test]
    fn large_spread_far_above_threshold_is_not_a_decision_split() {
        let report = evaluate_decision_envelope(&plan(&[("run-process", 180.0), ("run-saras", 280.0)]));
        assert_eq!(report.disposition, DecisionEnvelopeDisposition::AllModelsPassObservedEnvelope);
        assert_eq!(report.spread_disposition, SpreadDisposition::ExceedsDeclaredTolerance);
    }

    #[test]
    fn identity_mismatch_is_not_comparable() {
        let mut input = plan(&[("run-process", 120.0), ("run-saras", 130.0)]);
        input.models[1].unit_ref = "MW".to_string();
        let report = evaluate_decision_envelope(&input);
        assert_eq!(report.disposition, DecisionEnvelopeDisposition::IncompleteOrNotComparable);
        assert!(report.diagnostics.contains(&DecisionEnvelopeDiagnostic::UnitMismatch));
    }

    #[test]
    fn out_of_domain_run_is_not_filtered_out() {
        let mut input = plan(&[("run-process", 120.0), ("run-saras", 130.0)]);
        input.models[1].applicability = ApplicabilityState::OutOfDomain;
        let report = evaluate_decision_envelope(&input);
        assert_eq!(report.disposition, DecisionEnvelopeDisposition::IncompleteOrNotComparable);
        assert!(report.model_decisions.is_empty());
    }

    #[test]
    fn nonconverged_run_is_not_filtered_out() {
        let mut input = plan(&[("run-process", 120.0), ("run-saras", 130.0)]);
        input.models[1].convergence = ConvergenceState::NonConverged;
        let report = evaluate_decision_envelope(&input);
        assert_eq!(report.disposition, DecisionEnvelopeDisposition::IncompleteOrNotComparable);
    }

    #[test]
    fn duplicate_run_identity_is_rejected() {
        let mut input = plan(&[("run-process", 120.0), ("run-saras", 130.0)]);
        input.models[1].run_id = input.models[0].run_id.clone();
        input.models[1].dependencies.run_id = input.models[1].run_id.clone();
        let report = evaluate_decision_envelope(&input);
        assert_eq!(report.disposition, DecisionEnvelopeDisposition::IncompleteOrNotComparable);
        assert!(report.diagnostics.contains(&DecisionEnvelopeDiagnostic::DuplicateRunIdentity));
    }

    #[test]
    fn stale_dependency_manifest_run_id_is_rejected() {
        let mut input = plan(&[("run-process", 120.0), ("run-saras", 130.0)]);
        input.models[1].dependencies.run_id = "stale-run".to_string();
        let report = evaluate_decision_envelope(&input);
        assert!(report.diagnostics.contains(&DecisionEnvelopeDiagnostic::DependencyManifestRunMismatch));
    }

    #[test]
    fn shared_assumptions_are_reported_as_common_mode() {
        let mut input = plan(&[("run-process", 120.0), ("run-saras", 130.0)]);
        input.models[1].dependencies.dependencies[0].dependency.node_id =
            input.models[0].dependencies.dependencies[0].dependency.node_id.clone();
        let report = evaluate_decision_envelope(&input);
        assert_eq!(
            report.dependence_reports[0].assessment.state,
            super::super::evidence_dependence::ObservableDependenceState::SharedDependencyFound
        );
    }

    #[test]
    fn incomplete_dependency_inventory_is_unknown_not_independent() {
        let mut input = plan(&[("run-process", 120.0), ("run-saras", 130.0)]);
        input.models[1].dependencies.complete_observables.clear();
        let report = evaluate_decision_envelope(&input);
        assert_eq!(
            report.dependence_reports[0].assessment.state,
            super::super::evidence_dependence::ObservableDependenceState::UnknownDependencyInventory
        );
    }

    #[test]
    fn input_order_does_not_change_canonical_decision_or_range() {
        let first = plan(&[("run-process", 90.0), ("run-saras", 110.0)]);
        let mut second = plan(&[("run-process", 90.0), ("run-saras", 110.0)]);
        second.models.reverse();
        assert_eq!(
            evaluate_decision_envelope(&first),
            evaluate_decision_envelope(&second)
        );
    }

    #[test]
    fn adding_an_extreme_model_extends_the_observed_envelope() {
        let base = evaluate_decision_envelope(&plan(&[("run-process", 120.0), ("run-saras", 130.0)]));
        let extended = evaluate_decision_envelope(&plan(&[
            ("run-process", 120.0),
            ("run-saras", 130.0),
            ("run-stellarator", 80.0),
        ]));
        assert_eq!(base.minimum_value, Some(120.0));
        assert_eq!(extended.minimum_value, Some(80.0));
        assert_eq!(extended.disposition, DecisionEnvelopeDisposition::DecisionSplit);
    }

    #[test]
    fn output_digests_and_applicability_are_part_of_the_gate() {
        let mut input = plan(&[("run-process", 120.0), ("run-saras", 130.0)]);
        input.models[0].output_sha256 = "bad".to_string();
        let report = evaluate_decision_envelope(&input);
        assert!(report.diagnostics.contains(&DecisionEnvelopeDiagnostic::InvalidOutputDigest));
    }

    #[test]
    fn at_most_threshold_is_supported() {
        let mut input = plan(&[("run-process", 20.0), ("run-saras", 30.0)]);
        input.profile.threshold_direction = ThresholdDirection::AtMost;
        input.profile.threshold_value = 25.0;
        let report = evaluate_decision_envelope(&input);
        assert_eq!(report.disposition, DecisionEnvelopeDisposition::DecisionSplit);
    }
}
