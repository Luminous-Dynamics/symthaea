//! Uncertainty-qualified net-electric export reserve gate.
//!
//! This module consumes a source-produced one-sided lower-bound receipt. It
//! does not estimate covariance, fit distributions or assume independence.
//! The receipt must bind to the exact solver run and service-boundary output.

use super::plant_service_gate::{
    PowerBalanceDisposition, PowerBalanceProfile, PowerBalanceReport, PowerBalanceRun,
};

pub const ROBUST_NET_EXPORT_VERSION: &str = "fusion-robust-net-export/0.1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NetExportBoundMethod {
    JointCovarianceContrast,
    SimultaneousComponentIntervals,
    DirectNetOutputInterval,
    ParametricPlusModelFormDiscrepancy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFormValidity {
    WithinDeclaredDomain,
    Unknown,
    OutOfDomain,
    MaterialDiscrepancy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NetExportEvidenceClass {
    SyntheticFixture,
    Assumption,
    ModelledOutput,
    MeasuredObservation,
    IndependentVerification,
    PhysicalQualification,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NetExportLowerBoundReceipt {
    pub receipt_id: String,
    pub run_id: String,
    pub output_sha256: String,
    pub schema_ref: String,
    pub schema_sha256: String,
    pub service_boundary_ref: String,
    pub unit_ref: String,
    pub averaging_window_ref: String,
    pub sign_convention_ref: String,
    pub lower_bound_mwe: f64,
    pub one_sided_coverage_level: f64,
    pub evidence_class: NetExportEvidenceClass,
    /// Identifies the single-test or campaign-family coverage scope.
    pub coverage_scope_ref: String,
    pub method: NetExportBoundMethod,
    pub method_ref: String,
    /// Authenticated method-specific evidence payload identity.
    pub method_evidence_sha256: String,
    pub method_run_id: String,
    pub method_output_sha256: String,
    /// Ordered source variables used by the method.
    pub method_variable_order: Vec<String>,
    pub distribution_assumption_ref: String,
    pub assumption_manifest_sha256: String,
    pub validity_domain_ref: String,
    pub model_form_validity: ModelFormValidity,
    pub model_form_evidence_ref: String,
    pub model_form_evidence_sha256: String,
    /// Required for JointCovarianceContrast; absent for other methods.
    pub covariance_receipt_ref: Option<String>,
    pub covariance_receipt_sha256: Option<String>,
    pub covariance_run_id: Option<String>,
    pub covariance_output_sha256: Option<String>,
    /// Exact ordered variables used in c^T Sigma c.
    pub covariance_variable_order: Vec<String>,
    /// Required for SimultaneousComponentIntervals.
    pub simultaneous_allocation_ref: Option<String>,
    /// Required for DirectNetOutputInterval.
    pub direct_net_analysis_ref: Option<String>,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RobustNetExportProfile {
    pub required_one_sided_coverage_level: f64,
    pub numerical_tolerance_mw: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RobustNetExportDisposition {
    /// A deterministic point estimate exists, but no valid uncertainty
    /// receipt supports a robust reserve decision.
    PointEstimateOnly,
    /// A valid one-sided lower bound does not clear the declared reserve.
    RobustlyBelowReserve,
    /// A valid one-sided lower bound clears the declared net-export reserve.
    RobustlyAboveReserve,
    /// Synthetic bounds can exercise the contract but never qualify service.
    SyntheticOnly,
    /// Assumption-only bounds can support sensitivity analysis, not robust qualification.
    AssumptionOnly,
    /// The parametric interval exists but model-form validity is unresolved.
    ModelFormUnresolved,
    /// The evaluated solver output lies outside its declared validity domain.
    OutOfDomain,
    /// A material structural/model-form discrepancy remains unresolved.
    ModelFormDiscrepancy,
    /// A supplied receipt or its source binding is invalid/stale/incompatible.
    NotEvaluable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RobustNetExportDiagnostic {
    InvalidRequiredCoverage,
    InvalidNumericalTolerance,
    MissingPointEstimate,
    PointBalanceNotAdmissible,
    RunIdentityMismatch,
    SolverIdentityMismatch,
    PointReportIdentityMismatch,
    OutputDigestMismatch,
    InvalidOutputDigest,
    SchemaReferenceMismatch,
    SchemaDigestMismatch,
    ServiceBoundaryMismatch,
    UnitMismatch,
    AveragingWindowMismatch,
    SignConventionMismatch,
    EmptyReceiptIdentity,
    NonFiniteLowerBound,
    LowerBoundAbovePointEstimate,
    InvalidCoverageLevel,
    CoverageBelowRequirement,
    EmptyCoverageScope,
    EmptyMethodReference,
    EmptyDistributionAssumption,
    InvalidAssumptionManifestDigest,
    EmptyValidityDomain,
    InvalidModelFormEvidence,
    MissingCovarianceReceipt,
    MissingSimultaneousAllocationReceipt,
    MethodEvidenceReceiptMismatch,
    MethodVariableOrderMismatch,
    MissingDirectNetAnalysisReceipt,
    UnexpectedMethodReceipt,
    EmptyEvidenceReference,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RobustNetExportReport {
    pub contract_version: &'static str,
    pub run_id: String,
    pub solver_id: String,
    pub point_estimate_disposition: PowerBalanceDisposition,
    pub disposition: RobustNetExportDisposition,
    pub point_estimate_net_mwe: Option<f64>,
    pub one_sided_lower_bound_mwe: Option<f64>,
    pub required_reserve_mwe: f64,
    pub robust_reserve_margin_mwe: Option<f64>,
    pub one_sided_coverage_level: Option<f64>,
    pub evidence_class: Option<NetExportEvidenceClass>,
    pub method: Option<NetExportBoundMethod>,
    pub method_ref: Option<String>,
    pub method_evidence_sha256: Option<String>,
    pub assumption_manifest_sha256: Option<String>,
    pub validity_domain_ref: Option<String>,
    pub model_form_validity: Option<ModelFormValidity>,
    pub model_form_evidence_ref: Option<String>,
    pub receipt_id: Option<String>,
    pub diagnostics: Vec<RobustNetExportDiagnostic>,
    pub evidence_refs: Vec<String>,
    pub claim_ceiling: &'static str,
}

/// Compare a source-produced one-sided net-export lower bound with the same
/// frozen power-balance profile used by the deterministic service gate.
///
/// A missing receipt yields PointEstimateOnly. A malformed/stale receipt is
/// NotEvaluable. The method-specific reference requirements prevent an
/// unknown cross-covariance from silently becoming zero or independence.
pub fn evaluate_robust_net_export(
    power_profile: &PowerBalanceProfile,
    power_run: &PowerBalanceRun,
    point_report: &PowerBalanceReport,
    robust_profile: &RobustNetExportProfile,
    receipt: Option<&NetExportLowerBoundReceipt>,
) -> RobustNetExportReport {
    let point = point_report.reported_net_electric_mw;
    let mut diagnostics = Vec::new();

    if !robust_profile.required_one_sided_coverage_level.is_finite()
        || robust_profile.required_one_sided_coverage_level <= 0.0
        || robust_profile.required_one_sided_coverage_level > 1.0
    {
        diagnostics.push(RobustNetExportDiagnostic::InvalidRequiredCoverage);
    }
    if !robust_profile.numerical_tolerance_mw.is_finite()
        || robust_profile.numerical_tolerance_mw < 0.0
    {
        diagnostics.push(RobustNetExportDiagnostic::InvalidNumericalTolerance);
    }
    if point.is_none_or(|value| !value.is_finite()) {
        diagnostics.push(RobustNetExportDiagnostic::MissingPointEstimate);
    }
    if point_report.run_id != power_run.run_id || point_report.solver_id != power_run.solver_id {
        diagnostics.push(RobustNetExportDiagnostic::PointReportIdentityMismatch);
    }
    let recomputed_point_report =
        super::plant_service_gate::evaluate_plant_power_balance(power_profile, power_run);
    if recomputed_point_report != *point_report {
        diagnostics.push(RobustNetExportDiagnostic::PointReportIdentityMismatch);
    }
    if matches!(
        point_report.disposition,
        PowerBalanceDisposition::NotEvaluable | PowerBalanceDisposition::BalanceInconsistent
    ) {
        diagnostics.push(RobustNetExportDiagnostic::PointBalanceNotAdmissible);
    }
    if !diagnostics.is_empty() {
        return report(
            power_run,
            point_report,
            point,
            None,
            RobustNetExportDisposition::NotEvaluable,
            diagnostics,
            Vec::new(),
        );
    }

    let Some(receipt) = receipt else {
        return report(
            power_run,
            point_report,
            point,
            None,
            RobustNetExportDisposition::PointEstimateOnly,
            Vec::new(),
            vec![power_run.receipt_ref.clone()],
        );
    };

    if receipt.receipt_id.trim().is_empty()
        || receipt.run_id.trim().is_empty()
        || receipt.method_ref.trim().is_empty()
    {
        diagnostics.push(RobustNetExportDiagnostic::EmptyReceiptIdentity);
    }
    if receipt.run_id != power_run.run_id {
        diagnostics.push(RobustNetExportDiagnostic::RunIdentityMismatch);
    }
    if receipt.output_sha256 != power_run.output_sha256 {
        diagnostics.push(RobustNetExportDiagnostic::OutputDigestMismatch);
    }
    if !is_sha256(&receipt.output_sha256) {
        diagnostics.push(RobustNetExportDiagnostic::InvalidOutputDigest);
    }
    if receipt.schema_ref != power_profile.schema_ref || receipt.schema_ref != power_run.schema_ref
    {
        diagnostics.push(RobustNetExportDiagnostic::SchemaReferenceMismatch);
    }
    if receipt.schema_sha256 != power_profile.schema_sha256
        || receipt.schema_sha256 != power_run.schema_sha256
        || !is_sha256(&receipt.schema_sha256)
    {
        diagnostics.push(RobustNetExportDiagnostic::SchemaDigestMismatch);
    }
    if receipt.service_boundary_ref != power_profile.service_boundary_ref
        || receipt.service_boundary_ref != power_run.service_boundary_ref
    {
        diagnostics.push(RobustNetExportDiagnostic::ServiceBoundaryMismatch);
    }
    if receipt.unit_ref != power_profile.unit_ref || receipt.unit_ref != power_run.unit_ref {
        diagnostics.push(RobustNetExportDiagnostic::UnitMismatch);
    }
    if receipt.averaging_window_ref != power_profile.averaging_window_ref
        || receipt.averaging_window_ref != power_run.averaging_window_ref
    {
        diagnostics.push(RobustNetExportDiagnostic::AveragingWindowMismatch);
    }
    if receipt.sign_convention_ref != power_profile.sign_convention_ref
        || receipt.sign_convention_ref != power_run.sign_convention_ref
    {
        diagnostics.push(RobustNetExportDiagnostic::SignConventionMismatch);
    }
    if !receipt.lower_bound_mwe.is_finite() {
        diagnostics.push(RobustNetExportDiagnostic::NonFiniteLowerBound);
    }
    if receipt.coverage_scope_ref.trim().is_empty() {
        diagnostics.push(RobustNetExportDiagnostic::EmptyCoverageScope);
    }
    if !receipt.one_sided_coverage_level.is_finite()
        || receipt.one_sided_coverage_level <= 0.0
        || receipt.one_sided_coverage_level > 1.0
    {
        diagnostics.push(RobustNetExportDiagnostic::InvalidCoverageLevel);
    } else if receipt.one_sided_coverage_level < robust_profile.required_one_sided_coverage_level {
        diagnostics.push(RobustNetExportDiagnostic::CoverageBelowRequirement);
    }
    if receipt.distribution_assumption_ref.trim().is_empty() {
        diagnostics.push(RobustNetExportDiagnostic::EmptyDistributionAssumption);
    }
    if !is_sha256(&receipt.assumption_manifest_sha256) {
        diagnostics.push(RobustNetExportDiagnostic::InvalidAssumptionManifestDigest);
    }
    if receipt.validity_domain_ref.trim().is_empty() {
        diagnostics.push(RobustNetExportDiagnostic::EmptyValidityDomain);
    }
    if receipt.model_form_evidence_ref.trim().is_empty()
        || !is_sha256(&receipt.model_form_evidence_sha256)
    {
        diagnostics.push(RobustNetExportDiagnostic::InvalidModelFormEvidence);
    }
    if receipt.evidence_ref.trim().is_empty() {
        diagnostics.push(RobustNetExportDiagnostic::EmptyEvidenceReference);
    }

    if !is_sha256(&receipt.method_evidence_sha256)
        || receipt.method_run_id != receipt.run_id
        || receipt.method_output_sha256 != receipt.output_sha256
    {
        diagnostics.push(RobustNetExportDiagnostic::MethodEvidenceReceiptMismatch);
    }
    let expected_method_order: &[&str] = match receipt.method {
        NetExportBoundMethod::JointCovarianceContrast
        | NetExportBoundMethod::SimultaneousComponentIntervals => {
            &["P_e_gross", "P_recirc_var", "P_recirc_fix"]
        }
        NetExportBoundMethod::DirectNetOutputInterval
        | NetExportBoundMethod::ParametricPlusModelFormDiscrepancy => &["P_elec"],
    };
    if receipt.method_variable_order
        != expected_method_order
            .iter()
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>()
    {
        diagnostics.push(RobustNetExportDiagnostic::MethodVariableOrderMismatch);
    }

    match receipt.method {
        NetExportBoundMethod::JointCovarianceContrast => {
            if receipt
                .covariance_receipt_ref
                .as_deref()
                .is_none_or(str::is_empty)
            {
                diagnostics.push(RobustNetExportDiagnostic::MissingCovarianceReceipt);
            }
            if receipt
                .covariance_receipt_sha256
                .as_deref()
                .is_none_or(|value| !is_sha256(value))
                || receipt.covariance_run_id.as_deref() != Some(receipt.run_id.as_str())
                || receipt.covariance_output_sha256.as_deref()
                    != Some(receipt.output_sha256.as_str())
                || receipt.covariance_variable_order
                    != ["P_e_gross", "P_recirc_var", "P_recirc_fix"]
            {
                diagnostics.push(RobustNetExportDiagnostic::MissingCovarianceReceipt);
            }
            if receipt.simultaneous_allocation_ref.is_some()
                || receipt.direct_net_analysis_ref.is_some()
            {
                diagnostics.push(RobustNetExportDiagnostic::UnexpectedMethodReceipt);
            }
        }
        NetExportBoundMethod::SimultaneousComponentIntervals => {
            if receipt
                .simultaneous_allocation_ref
                .as_deref()
                .is_none_or(str::is_empty)
            {
                diagnostics.push(RobustNetExportDiagnostic::MissingSimultaneousAllocationReceipt);
            }
            if receipt.covariance_receipt_ref.is_some()
                || receipt.direct_net_analysis_ref.is_some()
                || receipt.covariance_receipt_sha256.is_some()
                || receipt.covariance_run_id.is_some()
                || receipt.covariance_output_sha256.is_some()
                || !receipt.covariance_variable_order.is_empty()
            {
                diagnostics.push(RobustNetExportDiagnostic::UnexpectedMethodReceipt);
            }
        }
        NetExportBoundMethod::DirectNetOutputInterval => {
            if receipt
                .direct_net_analysis_ref
                .as_deref()
                .is_none_or(str::is_empty)
            {
                diagnostics.push(RobustNetExportDiagnostic::MissingDirectNetAnalysisReceipt);
            }
            if receipt.covariance_receipt_ref.is_some()
                || receipt.simultaneous_allocation_ref.is_some()
                || receipt.covariance_receipt_sha256.is_some()
                || receipt.covariance_run_id.is_some()
                || receipt.covariance_output_sha256.is_some()
                || !receipt.covariance_variable_order.is_empty()
            {
                diagnostics.push(RobustNetExportDiagnostic::UnexpectedMethodReceipt);
            }
        }
        NetExportBoundMethod::ParametricPlusModelFormDiscrepancy => {
            if receipt.model_form_validity != ModelFormValidity::WithinDeclaredDomain
                || receipt.model_form_evidence_ref.trim().is_empty()
                || !is_sha256(&receipt.model_form_evidence_sha256)
            {
                diagnostics.push(RobustNetExportDiagnostic::InvalidModelFormEvidence);
            }
            if receipt.covariance_receipt_ref.is_some()
                || receipt.simultaneous_allocation_ref.is_some()
                || receipt.direct_net_analysis_ref.is_some()
                || receipt.covariance_receipt_sha256.is_some()
                || receipt.covariance_run_id.is_some()
                || receipt.covariance_output_sha256.is_some()
                || !receipt.covariance_variable_order.is_empty()
            {
                diagnostics.push(RobustNetExportDiagnostic::UnexpectedMethodReceipt);
            }
        }
    }

    let point_value = point.unwrap_or(f64::NAN);
    if receipt.lower_bound_mwe.is_finite()
        && receipt.lower_bound_mwe > point_value + robust_profile.numerical_tolerance_mw
    {
        diagnostics.push(RobustNetExportDiagnostic::LowerBoundAbovePointEstimate);
    }

    let mut evidence_refs = vec![power_run.receipt_ref.clone(), receipt.evidence_ref.clone()];
    if let Some(reference) = &receipt.covariance_receipt_ref {
        evidence_refs.push(reference.clone());
    }
    if let Some(reference) = &receipt.simultaneous_allocation_ref {
        evidence_refs.push(reference.clone());
    }
    if let Some(reference) = &receipt.direct_net_analysis_ref {
        evidence_refs.push(reference.clone());
    }
    if !receipt.model_form_evidence_ref.trim().is_empty() {
        evidence_refs.push(receipt.model_form_evidence_ref.clone());
    }

    if !diagnostics.is_empty() {
        return report(
            power_run,
            point_report,
            point,
            Some(receipt),
            RobustNetExportDisposition::NotEvaluable,
            diagnostics,
            evidence_refs,
        );
    }

    let lower = receipt.lower_bound_mwe;
    let margin = lower - power_profile.required_net_export_reserve_mw;
    let disposition = match receipt.model_form_validity {
        ModelFormValidity::Unknown => RobustNetExportDisposition::ModelFormUnresolved,
        ModelFormValidity::OutOfDomain => RobustNetExportDisposition::OutOfDomain,
        ModelFormValidity::MaterialDiscrepancy => RobustNetExportDisposition::ModelFormDiscrepancy,
        ModelFormValidity::WithinDeclaredDomain => match receipt.evidence_class {
            NetExportEvidenceClass::SyntheticFixture => RobustNetExportDisposition::SyntheticOnly,
            NetExportEvidenceClass::Assumption => RobustNetExportDisposition::AssumptionOnly,
            NetExportEvidenceClass::ModelledOutput
            | NetExportEvidenceClass::MeasuredObservation
            | NetExportEvidenceClass::IndependentVerification
            | NetExportEvidenceClass::PhysicalQualification => {
                if power_profile.reserve_threshold_inclusive {
                    if margin >= 0.0 {
                        RobustNetExportDisposition::RobustlyAboveReserve
                    } else {
                        RobustNetExportDisposition::RobustlyBelowReserve
                    }
                } else if margin > 0.0 {
                    RobustNetExportDisposition::RobustlyAboveReserve
                } else {
                    RobustNetExportDisposition::RobustlyBelowReserve
                }
            }
        },
    };

    report(
        power_run,
        point_report,
        point,
        Some(receipt),
        disposition,
        diagnostics,
        evidence_refs,
    )
}

fn report(
    power_run: &PowerBalanceRun,
    point_report: &PowerBalanceReport,
    point: Option<f64>,
    receipt: Option<&NetExportLowerBoundReceipt>,
    disposition: RobustNetExportDisposition,
    diagnostics: Vec<RobustNetExportDiagnostic>,
    evidence_refs: Vec<String>,
) -> RobustNetExportReport {
    let lower = receipt.map(|value| value.lower_bound_mwe);
    let margin = lower.map(|value| value - point_report.required_net_export_reserve_mw);
    RobustNetExportReport {
        contract_version: ROBUST_NET_EXPORT_VERSION,
        run_id: power_run.run_id.clone(),
        solver_id: power_run.solver_id.clone(),
        point_estimate_disposition: point_report.disposition,
        disposition,
        point_estimate_net_mwe: point,
        one_sided_lower_bound_mwe: lower,
        required_reserve_mwe: point_report.required_net_export_reserve_mw,
        robust_reserve_margin_mwe: margin,
        one_sided_coverage_level: receipt.map(|value| value.one_sided_coverage_level),
        evidence_class: receipt.map(|value| value.evidence_class),
        method: receipt.map(|value| value.method),
        method_ref: receipt.map(|value| value.method_ref.clone()),
        method_evidence_sha256: receipt.map(|value| value.method_evidence_sha256.clone()),
        assumption_manifest_sha256: receipt.map(|value| value.assumption_manifest_sha256.clone()),
        validity_domain_ref: receipt.map(|value| value.validity_domain_ref.clone()),
        model_form_validity: receipt.map(|value| value.model_form_validity),
        model_form_evidence_ref: receipt.map(|value| value.model_form_evidence_ref.clone()),
        receipt_id: receipt.map(|value| value.receipt_id.clone()),
        diagnostics,
        evidence_refs,
        claim_ceiling: "uncertainty receipt bound to declared net-electric reserve only",
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plant_service_gate::{
        PowerObservable, PowerTermEvidence, evaluate_plant_power_balance,
    };

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn power_profile() -> PowerBalanceProfile {
        PowerBalanceProfile {
            schema_ref: "fusion-power-schema-v1".to_string(),
            schema_sha256: sha('a'),
            service_boundary_ref: "plant-net-export-v1".to_string(),
            unit_ref: "MWe".to_string(),
            averaging_window_ref: "steady-state-60s-v1".to_string(),
            sign_convention_ref: "net-export-positive-v1".to_string(),
            absolute_balance_tolerance_mw: 0.001,
            required_net_export_reserve_mw: 50.0,
            reserve_threshold_inclusive: true,
        }
    }

    fn power_run() -> PowerBalanceRun {
        let profile = power_profile();
        PowerBalanceRun {
            run_id: "run-42".to_string(),
            solver_id: "PROCESS".to_string(),
            solver_revision: "process-revision-1".to_string(),
            candidate_revision: "candidate-r7".to_string(),
            output_sha256: sha('b'),
            schema_ref: profile.schema_ref,
            schema_sha256: profile.schema_sha256,
            service_boundary_ref: profile.service_boundary_ref,
            unit_ref: profile.unit_ref,
            averaging_window_ref: profile.averaging_window_ref,
            sign_convention_ref: profile.sign_convention_ref,
            receipt_ref: "solver-run-receipt-42".to_string(),
            terms: vec![
                PowerTermEvidence {
                    observable: PowerObservable::GrossElectric,
                    value_mw: 300.0,
                    source_locator: "out[1]".to_string(),
                    evidence_ref: "gross-evidence".to_string(),
                },
                PowerTermEvidence {
                    observable: PowerObservable::VariableRecirculated,
                    value_mw: 100.0,
                    source_locator: "out[2]".to_string(),
                    evidence_ref: "variable-evidence".to_string(),
                },
                PowerTermEvidence {
                    observable: PowerObservable::FixedRecirculated,
                    value_mw: 50.0,
                    source_locator: "out[3]".to_string(),
                    evidence_ref: "fixed-evidence".to_string(),
                },
                PowerTermEvidence {
                    observable: PowerObservable::TotalRecirculated,
                    value_mw: 150.0,
                    source_locator: "out[4]".to_string(),
                    evidence_ref: "total-evidence".to_string(),
                },
                PowerTermEvidence {
                    observable: PowerObservable::NetElectric,
                    value_mw: 150.0,
                    source_locator: "out[5]".to_string(),
                    evidence_ref: "net-evidence".to_string(),
                },
            ],
        }
    }

    fn lower_bound(lower_bound_mwe: f64) -> NetExportLowerBoundReceipt {
        let profile = power_profile();
        let run = power_run();
        NetExportLowerBoundReceipt {
            receipt_id: "net-uq-receipt-42".to_string(),
            run_id: run.run_id.clone(),
            output_sha256: run.output_sha256.clone(),
            schema_ref: profile.schema_ref,
            schema_sha256: profile.schema_sha256,
            service_boundary_ref: profile.service_boundary_ref,
            unit_ref: profile.unit_ref,
            averaging_window_ref: profile.averaging_window_ref,
            sign_convention_ref: profile.sign_convention_ref,
            lower_bound_mwe,
            one_sided_coverage_level: 0.95,
            evidence_class: NetExportEvidenceClass::ModelledOutput,
            coverage_scope_ref: "single-net-export-reserve-test-v1".to_string(),
            method: NetExportBoundMethod::JointCovarianceContrast,
            method_ref: "joint-linear-contrast-v1".to_string(),
            method_evidence_sha256: sha('e'),
            method_run_id: run.run_id.clone(),
            method_output_sha256: run.output_sha256.clone(),
            method_variable_order: vec![
                "P_e_gross".to_string(),
                "P_recirc_var".to_string(),
                "P_recirc_fix".to_string(),
            ],
            distribution_assumption_ref: "calibrated-coverage-method-v1".to_string(),
            assumption_manifest_sha256: sha('c'),
            validity_domain_ref: "declared-process-regime-v1".to_string(),
            model_form_validity: ModelFormValidity::WithinDeclaredDomain,
            model_form_evidence_ref: "model-form-validity-receipt-v1".to_string(),
            model_form_evidence_sha256: sha('f'),
            covariance_receipt_ref: Some("symtropy-joint-covariance-42".to_string()),
            covariance_receipt_sha256: Some(sha('d')),
            covariance_run_id: Some(run.run_id.clone()),
            covariance_output_sha256: Some(run.output_sha256.clone()),
            covariance_variable_order: vec![
                "P_e_gross".to_string(),
                "P_recirc_var".to_string(),
                "P_recirc_fix".to_string(),
            ],
            simultaneous_allocation_ref: None,
            direct_net_analysis_ref: None,
            evidence_ref: "net-lower-bound-evidence-42".to_string(),
        }
    }

    fn point_report() -> PowerBalanceReport {
        evaluate_plant_power_balance(&power_profile(), &power_run())
    }

    fn robust_profile() -> RobustNetExportProfile {
        RobustNetExportProfile {
            required_one_sided_coverage_level: 0.95,
            numerical_tolerance_mw: 0.001,
        }
    }

    fn evaluate(receipt: Option<&NetExportLowerBoundReceipt>) -> RobustNetExportReport {
        let profile = power_profile();
        let run = power_run();
        let point = evaluate_plant_power_balance(&profile, &run);
        evaluate_robust_net_export(&profile, &run, &point, &robust_profile(), receipt)
    }

    #[test]
    fn synthetic_lower_bound_is_never_promoted_to_robust_qualification() {
        let mut receipt = lower_bound(60.0);
        receipt.evidence_class = NetExportEvidenceClass::SyntheticFixture;
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::SyntheticOnly
        );
    }

    #[test]
    fn assumption_only_lower_bound_is_not_robust_qualification() {
        let mut receipt = lower_bound(60.0);
        receipt.evidence_class = NetExportEvidenceClass::Assumption;
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::AssumptionOnly
        );
    }

    #[test]
    fn parametric_covariance_does_not_override_unknown_model_form_validity() {
        let mut receipt = lower_bound(60.0);
        receipt.model_form_validity = ModelFormValidity::Unknown;
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::ModelFormUnresolved
        );
    }

    #[test]
    fn out_of_domain_receipt_cannot_qualify_net_export() {
        let mut receipt = lower_bound(60.0);
        receipt.model_form_validity = ModelFormValidity::OutOfDomain;
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::OutOfDomain
        );
    }

    #[test]
    fn material_model_form_discrepancy_blocks_robust_qualification() {
        let mut receipt = lower_bound(60.0);
        receipt.model_form_validity = ModelFormValidity::MaterialDiscrepancy;
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::ModelFormDiscrepancy
        );
    }

    #[test]
    fn model_form_validity_must_have_exact_evidence() {
        let mut receipt = lower_bound(60.0);
        receipt.model_form_evidence_sha256 = "bad-digest".to_string();
        let report = evaluate(Some(&receipt));
        assert_eq!(report.disposition, RobustNetExportDisposition::NotEvaluable);
        assert!(
            report
                .diagnostics
                .contains(&RobustNetExportDiagnostic::InvalidModelFormEvidence)
        );
    }

    #[test]
    fn lower_bound_below_reserve_blocks_robust_qualification() {
        let receipt = lower_bound(20.0);
        let report = evaluate(Some(&receipt));
        assert_eq!(
            report.disposition,
            RobustNetExportDisposition::RobustlyBelowReserve
        );
        assert_eq!(report.robust_reserve_margin_mwe, Some(-30.0));
    }

    #[test]
    fn lower_bound_above_reserve_qualifies_declared_robust_service() {
        let receipt = lower_bound(60.0);
        let report = evaluate(Some(&receipt));
        assert_eq!(
            report.disposition,
            RobustNetExportDisposition::RobustlyAboveReserve
        );
        assert_eq!(report.robust_reserve_margin_mwe, Some(10.0));
    }

    #[test]
    fn missing_uncertainty_receipt_is_point_estimate_only() {
        let report = evaluate(None);
        assert_eq!(
            report.disposition,
            RobustNetExportDisposition::PointEstimateOnly
        );
        assert_eq!(report.one_sided_lower_bound_mwe, None);
    }

    #[test]
    fn lower_bound_must_bind_exact_run_and_output_digest() {
        let mut receipt = lower_bound(60.0);
        receipt.run_id = "different-run".to_string();
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::NotEvaluable
        );
        let mut receipt = lower_bound(60.0);
        receipt.output_sha256 = sha('d');
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::NotEvaluable
        );
    }

    #[test]
    fn lower_bound_must_bind_exact_schema_and_service_boundary() {
        let mut receipt = lower_bound(60.0);
        receipt.schema_ref = "stale-schema".to_string();
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::NotEvaluable
        );
        let mut receipt = lower_bound(60.0);
        receipt.service_boundary_ref = "gross-electric".to_string();
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::NotEvaluable
        );
    }

    #[test]
    fn unit_time_and_sign_convention_mismatches_refuse() {
        let mut receipt = lower_bound(60.0);
        receipt.unit_ref = "MW".to_string();
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::NotEvaluable
        );
        let mut receipt = lower_bound(60.0);
        receipt.averaging_window_ref = "instantaneous".to_string();
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::NotEvaluable
        );
        let mut receipt = lower_bound(60.0);
        receipt.sign_convention_ref = "net-import-positive".to_string();
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::NotEvaluable
        );
    }

    #[test]
    fn insufficient_one_sided_coverage_is_not_qualified() {
        let mut receipt = lower_bound(60.0);
        receipt.one_sided_coverage_level = 0.90;
        assert!(
            evaluate(Some(&receipt))
                .diagnostics
                .contains(&RobustNetExportDiagnostic::CoverageBelowRequirement)
        );
    }

    #[test]
    fn joint_covariance_method_requires_exact_covariance_receipt() {
        let mut receipt = lower_bound(60.0);
        receipt.covariance_receipt_ref = None;
        let report = evaluate(Some(&receipt));
        assert_eq!(report.disposition, RobustNetExportDisposition::NotEvaluable);
        assert!(
            report
                .diagnostics
                .contains(&RobustNetExportDiagnostic::MissingCovarianceReceipt)
        );
    }

    #[test]
    fn simultaneous_component_method_requires_familywise_allocation_receipt() {
        let mut receipt = lower_bound(60.0);
        receipt.method = NetExportBoundMethod::SimultaneousComponentIntervals;
        receipt.covariance_receipt_ref = None;
        let report = evaluate(Some(&receipt));
        assert_eq!(report.disposition, RobustNetExportDisposition::NotEvaluable);
        assert!(
            report
                .diagnostics
                .contains(&RobustNetExportDiagnostic::MissingSimultaneousAllocationReceipt)
        );
    }

    #[test]
    fn parametric_plus_model_form_method_requires_and_accepts_domain_evidence() {
        let mut receipt = lower_bound(60.0);
        receipt.method = NetExportBoundMethod::ParametricPlusModelFormDiscrepancy;
        receipt.method_variable_order = vec!["P_elec".to_string()];
        receipt.covariance_receipt_ref = None;
        receipt.covariance_receipt_sha256 = None;
        receipt.covariance_run_id = None;
        receipt.covariance_output_sha256 = None;
        receipt.covariance_variable_order.clear();
        receipt.direct_net_analysis_ref = None;
        receipt.simultaneous_allocation_ref = None;
        receipt.model_form_validity = ModelFormValidity::WithinDeclaredDomain;
        receipt.model_form_evidence_ref = "measured-residual-calibration-receipt".to_string();
        receipt.model_form_evidence_sha256 = sha('f');
        let report = evaluate(Some(&receipt));
        assert_eq!(
            report.disposition,
            RobustNetExportDisposition::RobustlyAboveReserve
        );
        assert!(report.evidence_refs.contains(
            &"measured-residual-calibration-receipt".to_string()
        ));
    }

    #[test]
    fn parametric_plus_model_form_method_rejects_covariance_and_direct_interval_mixins() {
        let mut receipt = lower_bound(60.0);
        receipt.method = NetExportBoundMethod::ParametricPlusModelFormDiscrepancy;
        receipt.method_variable_order = vec!["P_elec".to_string()];
        receipt.covariance_receipt_ref = None;
        receipt.covariance_receipt_sha256 = None;
        receipt.covariance_run_id = None;
        receipt.covariance_output_sha256 = None;
        receipt.covariance_variable_order.clear();
        receipt.direct_net_analysis_ref = Some("unexpected-direct-interval".to_string());
        let report = evaluate(Some(&receipt));
        assert_eq!(report.disposition, RobustNetExportDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(
            &RobustNetExportDiagnostic::UnexpectedMethodReceipt
        ));
    }

    #[test]
    fn direct_net_method_requires_direct_analysis_receipt() {
        let mut receipt = lower_bound(60.0);
        receipt.method = NetExportBoundMethod::DirectNetOutputInterval;
        receipt.covariance_receipt_ref = None;
        let report = evaluate(Some(&receipt));
        assert_eq!(report.disposition, RobustNetExportDisposition::NotEvaluable);
        assert!(
            report
                .diagnostics
                .contains(&RobustNetExportDiagnostic::MissingDirectNetAnalysisReceipt)
        );
    }

    #[test]
    fn joint_covariance_receipt_must_bind_exact_run_output_and_variable_order() {
        let mut receipt = lower_bound(60.0);
        receipt.covariance_run_id = Some("another-run".to_string());
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::NotEvaluable
        );

        let mut receipt = lower_bound(60.0);
        receipt.covariance_output_sha256 = Some(sha('f'));
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::NotEvaluable
        );

        let mut receipt = lower_bound(60.0);
        receipt.covariance_variable_order.swap(0, 1);
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::NotEvaluable
        );
    }

    #[test]
    fn simultaneous_component_interval_method_accepts_only_bound_allocation_receipt() {
        let mut receipt = lower_bound(60.0);
        receipt.method = NetExportBoundMethod::SimultaneousComponentIntervals;
        receipt.covariance_receipt_ref = None;
        receipt.covariance_receipt_sha256 = None;
        receipt.covariance_run_id = None;
        receipt.covariance_output_sha256 = None;
        receipt.covariance_variable_order.clear();
        receipt.simultaneous_allocation_ref = Some("familywise-allocation-95-v1".to_string());
        let report = evaluate(Some(&receipt));
        assert_eq!(
            report.disposition,
            RobustNetExportDisposition::RobustlyAboveReserve
        );
    }

    #[test]
    fn direct_net_interval_method_requires_bound_direct_analysis() {
        let mut receipt = lower_bound(60.0);
        receipt.method = NetExportBoundMethod::DirectNetOutputInterval;
        receipt.covariance_receipt_ref = None;
        receipt.covariance_receipt_sha256 = None;
        receipt.covariance_run_id = None;
        receipt.covariance_output_sha256 = None;
        receipt.covariance_variable_order.clear();
        receipt.method_variable_order = vec!["P_elec".to_string()];
        receipt.direct_net_analysis_ref = Some("direct-net-output-analysis-v1".to_string());
        let report = evaluate(Some(&receipt));
        assert_eq!(
            report.disposition,
            RobustNetExportDisposition::RobustlyAboveReserve
        );
    }

    #[test]
    fn method_evidence_must_bind_exact_run_output_and_declared_variables() {
        let mut receipt = lower_bound(60.0);
        receipt.method_run_id = "stale-run".to_string();
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::NotEvaluable
        );
        let mut receipt = lower_bound(60.0);
        receipt.method_output_sha256 = sha('f');
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::NotEvaluable
        );
        let mut receipt = lower_bound(60.0);
        receipt.method_variable_order.swap(0, 1);
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::NotEvaluable
        );
    }

    #[test]
    fn point_report_cannot_be_forged_or_reused_from_another_run() {
        let profile = power_profile();
        let run = power_run();
        let mut point = evaluate_plant_power_balance(&profile, &run);
        point.run_id = "different-run".to_string();
        let receipt = lower_bound(60.0);
        let report =
            evaluate_robust_net_export(&profile, &run, &point, &robust_profile(), Some(&receipt));
        assert_eq!(report.disposition, RobustNetExportDisposition::NotEvaluable);
        assert!(
            report
                .diagnostics
                .contains(&RobustNetExportDiagnostic::PointReportIdentityMismatch)
        );
    }

    #[test]
    fn coverage_scope_must_be_explicit() {
        let mut receipt = lower_bound(60.0);
        receipt.coverage_scope_ref.clear();
        let report = evaluate(Some(&receipt));
        assert_eq!(report.disposition, RobustNetExportDisposition::NotEvaluable);
        assert!(
            report
                .diagnostics
                .contains(&RobustNetExportDiagnostic::EmptyCoverageScope)
        );
    }

    #[test]
    fn lower_bound_cannot_exceed_point_estimate() {
        let receipt = lower_bound(151.0);
        let report = evaluate(Some(&receipt));
        assert_eq!(report.disposition, RobustNetExportDisposition::NotEvaluable);
        assert!(
            report
                .diagnostics
                .contains(&RobustNetExportDiagnostic::LowerBoundAbovePointEstimate)
        );
    }

    #[test]
    fn exact_reserve_boundary_obeys_inclusive_policy() {
        let receipt = lower_bound(50.0);
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::RobustlyAboveReserve
        );
        let mut profile = power_profile();
        profile.reserve_threshold_inclusive = false;
        let run = power_run();
        let point = evaluate_plant_power_balance(&profile, &run);
        let report =
            evaluate_robust_net_export(&profile, &run, &point, &robust_profile(), Some(&receipt));
        assert_eq!(
            report.disposition,
            RobustNetExportDisposition::RobustlyBelowReserve
        );
    }

    #[test]
    fn non_import_point_estimate_cannot_be_overridden_by_stale_positive_bound() {
        let profile = power_profile();
        let mut run = power_run();
        run.terms
            .iter_mut()
            .find(|term| term.observable == PowerObservable::NetElectric)
            .unwrap()
            .value_mw = -10.0;
        run.terms
            .iter_mut()
            .find(|term| term.observable == PowerObservable::GrossElectric)
            .unwrap()
            .value_mw = 140.0;
        run.terms
            .iter_mut()
            .find(|term| term.observable == PowerObservable::TotalRecirculated)
            .unwrap()
            .value_mw = 150.0;
        let point = evaluate_plant_power_balance(&profile, &run);
        let mut receipt = lower_bound(-20.0);
        receipt.run_id = run.run_id.clone();
        receipt.output_sha256 = run.output_sha256.clone();
        let report =
            evaluate_robust_net_export(&profile, &run, &point, &robust_profile(), Some(&receipt));
        assert_eq!(
            report.disposition,
            RobustNetExportDisposition::RobustlyBelowReserve
        );
    }

    #[test]
    fn missing_method_assumption_and_validity_domain_is_not_evaluable() {
        let mut receipt = lower_bound(60.0);
        receipt.method_ref.clear();
        receipt.distribution_assumption_ref.clear();
        receipt.validity_domain_ref.clear();
        let report = evaluate(Some(&receipt));
        assert_eq!(report.disposition, RobustNetExportDisposition::NotEvaluable);
        assert!(
            report
                .diagnostics
                .contains(&RobustNetExportDiagnostic::EmptyDistributionAssumption)
        );
        assert!(
            report
                .diagnostics
                .contains(&RobustNetExportDiagnostic::EmptyValidityDomain)
        );
    }

    #[test]
    fn invalid_assumption_manifest_and_coverage_profile_are_refused() {
        let mut receipt = lower_bound(60.0);
        receipt.assumption_manifest_sha256 = "bad-digest".to_string();
        assert_eq!(
            evaluate(Some(&receipt)).disposition,
            RobustNetExportDisposition::NotEvaluable
        );

        let profile = power_profile();
        let run = power_run();
        let point = point_report();
        let mut invalid = robust_profile();
        invalid.required_one_sided_coverage_level = 1.5;
        let report = evaluate_robust_net_export(&profile, &run, &point, &invalid, None);
        assert_eq!(report.disposition, RobustNetExportDisposition::NotEvaluable);
    }
}
