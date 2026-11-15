//! Plant-service net-electric power-balance qualification.
//!
//! This module distinguishes solver/physics feasibility from the service a
//! plant actually delivers. Positive fusion gain or gross generation cannot
//! override a negative or insufficient net-electric balance.

use std::collections::BTreeMap;

pub const PLANT_SERVICE_GATE_VERSION: &str = "fusion-plant-service-gate/0.1";
pub const D0FUS_3_2_0_ITER_SCHEMA_REF: &str =
    "d0fus-3.2.0-iter-run-output-165-v1";
pub const D0FUS_3_2_0_SOURCE_COMMIT: &str =
    "a682c3511dc189b2a4cdd6f4b8bc545d3cdb8924";
pub const D0FUS_3_2_0_EXPECTED_OUTPUT_LENGTH: usize = 165;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum PowerObservable {
    GrossElectric,
    VariableRecirculated,
    FixedRecirculated,
    TotalRecirculated,
    NetElectric,
}

impl PowerObservable {
    pub const REQUIRED: [Self; 5] = [
        Self::GrossElectric,
        Self::VariableRecirculated,
        Self::FixedRecirculated,
        Self::TotalRecirculated,
        Self::NetElectric,
    ];

    pub const fn id(self) -> &'static str {
        match self {
            Self::GrossElectric => "gross-electric-power",
            Self::VariableRecirculated => "variable-recirculated-electric-power",
            Self::FixedRecirculated => "fixed-recirculated-electric-power",
            Self::TotalRecirculated => "total-recirculated-electric-power",
            Self::NetElectric => "net-electric-export-power",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PowerTermEvidence {
    pub observable: PowerObservable,
    pub value_mw: f64,
    /// Exact field/report locator within the referenced solver output.
    pub source_locator: String,
    /// Evidence pointer, e.g. receipt path plus JSON pointer.
    pub evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PowerBalanceRun {
    pub run_id: String,
    pub solver_id: String,
    pub solver_revision: String,
    pub candidate_revision: String,
    pub output_sha256: String,
    pub schema_ref: String,
    pub schema_sha256: String,
    pub service_boundary_ref: String,
    pub unit_ref: String,
    pub averaging_window_ref: String,
    /// Explicit semantic rule: gross generation positive, consumption terms
    /// positive as loads, and net export positive toward the grid.
    pub sign_convention_ref: String,
    pub receipt_ref: String,
    pub terms: Vec<PowerTermEvidence>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PowerBalanceProfile {
    pub schema_ref: String,
    pub schema_sha256: String,
    pub service_boundary_ref: String,
    pub unit_ref: String,
    pub averaging_window_ref: String,
    pub sign_convention_ref: String,
    pub absolute_balance_tolerance_mw: f64,
    pub required_net_export_reserve_mw: f64,
    pub reserve_threshold_inclusive: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PowerBalanceDisposition {
    NotEvaluable,
    BalanceInconsistent,
    NetImport,
    NoNetExport,
    NetExportBelowReserve,
    NetExportQualifiedForDeclaredBoundary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PowerBalanceDiagnostic {
    EmptyRunIdentity,
    InvalidOutputDigest,
    InvalidSchemaDigest,
    EmptyReceiptReference,
    SchemaReferenceMismatch,
    SchemaDigestMismatch,
    ServiceBoundaryMismatch,
    UnitMismatch,
    AveragingWindowMismatch,
    SignConventionMismatch,
    InvalidBalanceTolerance,
    InvalidReserveRequirement,
    MissingObservable,
    DuplicateObservable,
    NonFiniteObservable,
    EmptySourceLocator,
    EmptyEvidenceReference,
    NegativeGrossGeneration,
    NegativeRecirculationLoad,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PowerBalanceReport {
    pub contract_version: &'static str,
    pub run_id: String,
    pub solver_id: String,
    pub disposition: PowerBalanceDisposition,
    pub gross_electric_mw: Option<f64>,
    pub variable_recirculated_mw: Option<f64>,
    pub fixed_recirculated_mw: Option<f64>,
    pub total_recirculated_mw: Option<f64>,
    pub reported_net_electric_mw: Option<f64>,
    pub calculated_total_recirculated_mw: Option<f64>,
    pub calculated_net_electric_mw: Option<f64>,
    pub total_recirculated_residual_mw: Option<f64>,
    pub net_electric_residual_mw: Option<f64>,
    pub maximum_absolute_balance_residual_mw: Option<f64>,
    pub normalized_balance_residual: Option<f64>,
    pub required_net_export_reserve_mw: f64,
    pub net_export_reserve_margin_mw: Option<f64>,
    pub diagnostics: Vec<PowerBalanceDiagnostic>,
    pub evidence_refs: Vec<String>,
    pub claim_ceiling: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct D0fusIterPowerValues {
    pub net_electric_mwe: f64,
    pub gross_electric_mwe: f64,
    pub variable_recirculated_mwe: f64,
    pub fixed_recirculated_mwe: f64,
    pub total_recirculated_mwe: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum D0fusSchemaError {
    WrongOutputLength { expected: usize, actual: usize },
    NonFinitePowerValue { observable: PowerObservable, index: usize },
}

/// Extract exact D0FUS 3.2.0 ITER RUN power observables from the frozen 165-value
/// return tuple. Indices are defined by the v3.2.0 return statement:
/// P_elec=25; appended tail P_e_gross=153, P_recirc_var=160,
/// P_recirc_fix=161, P_recirculated=162.
pub fn extract_d0fus_3_2_0_iter_power_values(
    values: &[f64],
) -> Result<D0fusIterPowerValues, D0fusSchemaError> {
    if values.len() != D0FUS_3_2_0_EXPECTED_OUTPUT_LENGTH {
        return Err(D0fusSchemaError::WrongOutputLength {
            expected: D0FUS_3_2_0_EXPECTED_OUTPUT_LENGTH,
            actual: values.len(),
        });
    }

    let indices = [
        (PowerObservable::NetElectric, 25),
        (PowerObservable::GrossElectric, 153),
        (PowerObservable::VariableRecirculated, 160),
        (PowerObservable::FixedRecirculated, 161),
        (PowerObservable::TotalRecirculated, 162),
    ];
    for (observable, index) in indices {
        if !values[index].is_finite() {
            return Err(D0fusSchemaError::NonFinitePowerValue { observable, index });
        }
    }

    Ok(D0fusIterPowerValues {
        net_electric_mwe: values[25],
        gross_electric_mwe: values[153],
        variable_recirculated_mwe: values[160],
        fixed_recirculated_mwe: values[161],
        total_recirculated_mwe: values[162],
    })
}

/// Evaluate two independent power-balance identities and then qualify net
/// export against an explicit reserve. Missing terms are never imputed to zero.
pub fn evaluate_plant_power_balance(
    profile: &PowerBalanceProfile,
    run: &PowerBalanceRun,
) -> PowerBalanceReport {
    let mut diagnostics = Vec::new();
    let mut term_map = BTreeMap::new();

    if run.run_id.trim().is_empty()
        || run.solver_id.trim().is_empty()
        || run.solver_revision.trim().is_empty()
        || run.candidate_revision.trim().is_empty()
    {
        diagnostics.push(PowerBalanceDiagnostic::EmptyRunIdentity);
    }
    if !is_sha256(&run.output_sha256) {
        diagnostics.push(PowerBalanceDiagnostic::InvalidOutputDigest);
    }
    if !is_sha256(&run.schema_sha256) {
        diagnostics.push(PowerBalanceDiagnostic::InvalidSchemaDigest);
    }
    if run.receipt_ref.trim().is_empty() {
        diagnostics.push(PowerBalanceDiagnostic::EmptyReceiptReference);
    }
    if run.schema_ref != profile.schema_ref {
        diagnostics.push(PowerBalanceDiagnostic::SchemaReferenceMismatch);
    }
    if run.schema_sha256 != profile.schema_sha256 {
        diagnostics.push(PowerBalanceDiagnostic::SchemaDigestMismatch);
    }
    if run.service_boundary_ref != profile.service_boundary_ref {
        diagnostics.push(PowerBalanceDiagnostic::ServiceBoundaryMismatch);
    }
    if run.unit_ref != profile.unit_ref {
        diagnostics.push(PowerBalanceDiagnostic::UnitMismatch);
    }
    if run.averaging_window_ref != profile.averaging_window_ref {
        diagnostics.push(PowerBalanceDiagnostic::AveragingWindowMismatch);
    }
    if run.sign_convention_ref != profile.sign_convention_ref {
        diagnostics.push(PowerBalanceDiagnostic::SignConventionMismatch);
    }
    if !profile.absolute_balance_tolerance_mw.is_finite()
        || profile.absolute_balance_tolerance_mw < 0.0
    {
        diagnostics.push(PowerBalanceDiagnostic::InvalidBalanceTolerance);
    }
    if !profile.required_net_export_reserve_mw.is_finite()
        || profile.required_net_export_reserve_mw < 0.0
    {
        diagnostics.push(PowerBalanceDiagnostic::InvalidReserveRequirement);
    }

    for term in &run.terms {
        if !term.value_mw.is_finite() {
            diagnostics.push(PowerBalanceDiagnostic::NonFiniteObservable);
        }
        if term.source_locator.trim().is_empty() {
            diagnostics.push(PowerBalanceDiagnostic::EmptySourceLocator);
        }
        if term.evidence_ref.trim().is_empty() {
            diagnostics.push(PowerBalanceDiagnostic::EmptyEvidenceReference);
        }
        if term_map.insert(term.observable, term).is_some() {
            diagnostics.push(PowerBalanceDiagnostic::DuplicateObservable);
        }
    }
    for observable in PowerObservable::REQUIRED {
        if !term_map.contains_key(&observable) {
            diagnostics.push(PowerBalanceDiagnostic::MissingObservable);
        }
    }

    let evidence_refs = run.terms.iter().map(|term| term.evidence_ref.clone()).collect();
    let get = |observable| term_map.get(&observable).map(|term| term.value_mw);

    let gross = get(PowerObservable::GrossElectric);
    let variable = get(PowerObservable::VariableRecirculated);
    let fixed = get(PowerObservable::FixedRecirculated);
    let total = get(PowerObservable::TotalRecirculated);
    let net = get(PowerObservable::NetElectric);

    if gross.is_some_and(|value| value < 0.0) {
        diagnostics.push(PowerBalanceDiagnostic::NegativeGrossGeneration);
    }
    if variable.is_some_and(|value| value < 0.0)
        || fixed.is_some_and(|value| value < 0.0)
        || total.is_some_and(|value| value < 0.0)
    {
        diagnostics.push(PowerBalanceDiagnostic::NegativeRecirculationLoad);
    }

    let calculated_total = variable.zip(fixed).map(|(v, f)| v + f);
    let calculated_net = gross
        .zip(variable)
        .zip(fixed)
        .map(|((g, v), f)| g - v - f);
    let total_residual = total.zip(calculated_total).map(|(reported, calculated)| {
        reported - calculated
    });
    let net_residual = net.zip(calculated_net).map(|(reported, calculated)| {
        reported - calculated
    });
    let max_residual = total_residual
        .zip(net_residual)
        .map(|(total_value, net_value)| total_value.abs().max(net_value.abs()));
    let scale = gross
        .into_iter()
        .chain(variable)
        .chain(fixed)
        .chain(total)
        .chain(net)
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let normalized_residual = max_residual.map(|value| {
        if scale == 0.0 { value } else { value / scale }
    });
    let reserve_margin = net.map(|value| value - profile.required_net_export_reserve_mw);

    let required_values_present = PowerObservable::REQUIRED
        .iter()
        .all(|observable| term_map.contains_key(observable));
    let has_non_balance_diagnostic = diagnostics.iter().any(|diagnostic| {
        !matches!(
            diagnostic,
            PowerBalanceDiagnostic::NegativeGrossGeneration
                | PowerBalanceDiagnostic::NegativeRecirculationLoad
        )
    });
    let disposition = if !required_values_present || has_non_balance_diagnostic {
        PowerBalanceDisposition::NotEvaluable
    } else if diagnostics.contains(&PowerBalanceDiagnostic::NegativeGrossGeneration)
        || diagnostics.contains(&PowerBalanceDiagnostic::NegativeRecirculationLoad)
    {
        PowerBalanceDisposition::NotEvaluable
    } else if max_residual.is_none_or(|value| value > profile.absolute_balance_tolerance_mw) {
        PowerBalanceDisposition::BalanceInconsistent
    } else if net.is_some_and(|value| value < 0.0) {
        PowerBalanceDisposition::NetImport
    } else if net.is_some_and(|value| value == 0.0) {
        PowerBalanceDisposition::NoNetExport
    } else if reserve_margin.is_some_and(|margin| {
        if profile.reserve_threshold_inclusive {
            margin < 0.0
        } else {
            margin <= 0.0
        }
    }) {
        PowerBalanceDisposition::NetExportBelowReserve
    } else {
        PowerBalanceDisposition::NetExportQualifiedForDeclaredBoundary
    };

    PowerBalanceReport {
        contract_version: PLANT_SERVICE_GATE_VERSION,
        run_id: run.run_id.clone(),
        solver_id: run.solver_id.clone(),
        disposition,
        gross_electric_mw: gross,
        variable_recirculated_mw: variable,
        fixed_recirculated_mw: fixed,
        total_recirculated_mw: total,
        reported_net_electric_mw: net,
        calculated_total_recirculated_mw: calculated_total,
        calculated_net_electric_mw: calculated_net,
        total_recirculated_residual_mw: total_residual,
        net_electric_residual_mw: net_residual,
        maximum_absolute_balance_residual_mw: max_residual,
        normalized_balance_residual: normalized_residual,
        required_net_export_reserve_mw: profile.required_net_export_reserve_mw,
        net_export_reserve_margin_mw: reserve_margin,
        diagnostics,
        evidence_refs,
        claim_ceiling: "declared case-level power-balance disposition only",
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn profile() -> PowerBalanceProfile {
        PowerBalanceProfile {
            schema_ref: D0FUS_3_2_0_ITER_SCHEMA_REF.to_string(),
            schema_sha256: sha('a'),
            service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
            unit_ref: "MWe".to_string(),
            averaging_window_ref: "iter-run-steady-state-window-v1".to_string(),
            sign_convention_ref: "gross-and-load-positive-net-export-positive-v1".to_string(),
            absolute_balance_tolerance_mw: 0.001,
            required_net_export_reserve_mw: 0.0,
            reserve_threshold_inclusive: true,
        }
    }

    fn run(
        gross: f64,
        variable: f64,
        fixed: f64,
        total: f64,
        net: f64,
    ) -> PowerBalanceRun {
        PowerBalanceRun {
            run_id: "d0fus-iter-run-v1".to_string(),
            solver_id: "D0FUS".to_string(),
            solver_revision: D0FUS_3_2_0_SOURCE_COMMIT.to_string(),
            candidate_revision: "iter-reference-deck-v1".to_string(),
            output_sha256: sha('b'),
            schema_ref: D0FUS_3_2_0_ITER_SCHEMA_REF.to_string(),
            schema_sha256: sha('a'),
            service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
            unit_ref: "MWe".to_string(),
            averaging_window_ref: "iter-run-steady-state-window-v1".to_string(),
            sign_convention_ref: "gross-and-load-positive-net-export-positive-v1".to_string(),
            receipt_ref: "workflow:43729456208:artifact:14167288485".to_string(),
            terms: vec![
                term(PowerObservable::GrossElectric, gross, "tuple[153]"),
                term(
                    PowerObservable::VariableRecirculated,
                    variable,
                    "tuple[160]",
                ),
                term(PowerObservable::FixedRecirculated, fixed, "tuple[161]"),
                term(PowerObservable::TotalRecirculated, total, "tuple[162]"),
                term(PowerObservable::NetElectric, net, "tuple[25]"),
            ],
        }
    }

    fn term(observable: PowerObservable, value_mw: f64, locator: &str) -> PowerTermEvidence {
        PowerTermEvidence {
            observable,
            value_mw,
            source_locator: locator.to_string(),
            evidence_ref: format!("output#{}", observable.id()),
        }
    }

    #[test]
    fn d0fus_iter_reference_is_net_import_despite_positive_q_and_gross_power() {
        let source = run(192.395, 144.149, 91.0951, 235.244, -42.84920863494321);
        let report = evaluate_plant_power_balance(&profile(), &source);
        assert_eq!(report.disposition, PowerBalanceDisposition::NetImport);
        assert!(report.gross_electric_mw.unwrap() > 0.0);
        assert!(report.reported_net_electric_mw.unwrap() < 0.0);
        assert_eq!(report.evidence_refs.len(), 5);
    }

    #[test]
    fn positive_net_export_above_reserve_is_qualified_for_declared_boundary() {
        let source = run(300.0, 100.0, 50.0, 150.0, 150.0);
        let report = evaluate_plant_power_balance(&profile(), &source);
        assert_eq!(
            report.disposition,
            PowerBalanceDisposition::NetExportQualifiedForDeclaredBoundary
        );
        assert_eq!(report.net_export_reserve_margin_mw, Some(150.0));
    }

    #[test]
    fn positive_net_export_below_required_reserve_is_not_qualified() {
        let mut threshold = profile();
        threshold.required_net_export_reserve_mw = 200.0;
        let source = run(300.0, 100.0, 50.0, 150.0, 150.0);
        let report = evaluate_plant_power_balance(&threshold, &source);
        assert_eq!(
            report.disposition,
            PowerBalanceDisposition::NetExportBelowReserve
        );
    }

    #[test]
    fn exact_reserve_threshold_obeys_declared_inclusive_policy() {
        let mut threshold = profile();
        threshold.required_net_export_reserve_mw = 150.0;
        let source = run(300.0, 100.0, 50.0, 150.0, 150.0);
        assert_eq!(
            evaluate_plant_power_balance(&threshold, &source).disposition,
            PowerBalanceDisposition::NetExportQualifiedForDeclaredBoundary
        );
        threshold.reserve_threshold_inclusive = false;
        assert_eq!(
            evaluate_plant_power_balance(&threshold, &source).disposition,
            PowerBalanceDisposition::NetExportBelowReserve
        );
    }

    #[test]
    fn zero_net_electricity_is_not_counted_as_export() {
        let source = run(150.0, 100.0, 50.0, 150.0, 0.0);
        assert_eq!(
            evaluate_plant_power_balance(&profile(), &source).disposition,
            PowerBalanceDisposition::NoNetExport
        );
    }

    #[test]
    fn mismatched_total_recirculation_is_balance_inconsistent() {
        let source = run(300.0, 100.0, 50.0, 149.0, 150.0);
        assert_eq!(
            evaluate_plant_power_balance(&profile(), &source).disposition,
            PowerBalanceDisposition::BalanceInconsistent
        );
    }

    #[test]
    fn mismatched_net_power_is_balance_inconsistent() {
        let source = run(300.0, 100.0, 50.0, 150.0, 149.0);
        assert_eq!(
            evaluate_plant_power_balance(&profile(), &source).disposition,
            PowerBalanceDisposition::BalanceInconsistent
        );
    }

    #[test]
    fn missing_fixed_recirculation_is_not_defaulted_to_zero() {
        let mut source = run(300.0, 100.0, 50.0, 150.0, 150.0);
        source
            .terms
            .retain(|term| term.observable != PowerObservable::FixedRecirculated);
        let report = evaluate_plant_power_balance(&profile(), &source);
        assert_eq!(report.disposition, PowerBalanceDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&PowerBalanceDiagnostic::MissingObservable));
    }

    #[test]
    fn duplicate_terms_are_rejected() {
        let mut source = run(300.0, 100.0, 50.0, 150.0, 150.0);
        source.terms.push(term(PowerObservable::NetElectric, 150.0, "duplicate"));
        assert_eq!(
            evaluate_plant_power_balance(&profile(), &source).disposition,
            PowerBalanceDisposition::NotEvaluable
        );
    }

    #[test]
    fn wrong_unit_is_not_evaluable() {
        let mut source = run(300.0, 100.0, 50.0, 150.0, 150.0);
        source.unit_ref = "MW".to_string();
        assert_eq!(
            evaluate_plant_power_balance(&profile(), &source).disposition,
            PowerBalanceDisposition::NotEvaluable
        );
    }

    #[test]
    fn wrong_service_boundary_is_not_evaluable() {
        let mut source = run(300.0, 100.0, 50.0, 150.0, 150.0);
        source.service_boundary_ref = "gross-electric-terminal-v1".to_string();
        assert_eq!(
            evaluate_plant_power_balance(&profile(), &source).disposition,
            PowerBalanceDisposition::NotEvaluable
        );
    }

    #[test]
    fn wrong_averaging_window_is_not_evaluable() {
        let mut source = run(300.0, 100.0, 50.0, 150.0, 150.0);
        source.averaging_window_ref = "instantaneous-v1".to_string();
        assert_eq!(
            evaluate_plant_power_balance(&profile(), &source).disposition,
            PowerBalanceDisposition::NotEvaluable
        );
    }

    #[test]
    fn wrong_sign_convention_is_not_evaluable() {
        let mut source = run(300.0, 100.0, 50.0, 150.0, 150.0);
        source.sign_convention_ref = "net-import-positive-v1".to_string();
        assert_eq!(
            evaluate_plant_power_balance(&profile(), &source).disposition,
            PowerBalanceDisposition::NotEvaluable
        );
    }

    #[test]
    fn stale_schema_reference_or_digest_is_not_evaluable() {
        let source = run(300.0, 100.0, 50.0, 150.0, 150.0);
        let mut changed = profile();
        changed.schema_ref = "stale-schema".to_string();
        assert_eq!(
            evaluate_plant_power_balance(&changed, &source).disposition,
            PowerBalanceDisposition::NotEvaluable
        );
        changed = profile();
        changed.schema_sha256 = sha('c');
        assert_eq!(
            evaluate_plant_power_balance(&changed, &source).disposition,
            PowerBalanceDisposition::NotEvaluable
        );
    }

    #[test]
    fn non_finite_observable_is_not_evaluable() {
        let source = run(300.0, 100.0, 50.0, 150.0, f64::NAN);
        assert_eq!(
            evaluate_plant_power_balance(&profile(), &source).disposition,
            PowerBalanceDisposition::NotEvaluable
        );
    }

    #[test]
    fn negative_recirculation_load_is_not_evaluable() {
        let source = run(300.0, -100.0, 50.0, -50.0, 350.0);
        assert_eq!(
            evaluate_plant_power_balance(&profile(), &source).disposition,
            PowerBalanceDisposition::NotEvaluable
        );
    }

    #[test]
    fn d0fus_tuple_adapter_uses_frozen_source_positions() {
        let mut values = vec![0.0; D0FUS_3_2_0_EXPECTED_OUTPUT_LENGTH];
        values[25] = -42.84920863494321;
        values[153] = 192.395;
        values[160] = 144.149;
        values[161] = 91.0951;
        values[162] = 235.244;
        let extracted = extract_d0fus_3_2_0_iter_power_values(&values).unwrap();
        assert_eq!(extracted.net_electric_mwe, values[25]);
        assert_eq!(extracted.gross_electric_mwe, values[153]);
        assert_eq!(extracted.variable_recirculated_mwe, values[160]);
        assert_eq!(extracted.fixed_recirculated_mwe, values[161]);
        assert_eq!(extracted.total_recirculated_mwe, values[162]);
    }

    #[test]
    fn d0fus_tuple_adapter_rejects_wrong_length_and_non_finite_fields() {
        assert_eq!(
            extract_d0fus_3_2_0_iter_power_values(&vec![0.0; 164]),
            Err(D0fusSchemaError::WrongOutputLength {
                expected: 165,
                actual: 164
            })
        );
        let mut values = vec![0.0; 165];
        values[160] = f64::INFINITY;
        assert_eq!(
            extract_d0fus_3_2_0_iter_power_values(&values),
            Err(D0fusSchemaError::NonFinitePowerValue {
                observable: PowerObservable::VariableRecirculated,
                index: 160
            })
        );
    }

    #[test]
    fn missing_digest_and_receipt_are_not_evaluable() {
        let mut source = run(300.0, 100.0, 50.0, 150.0, 150.0);
        source.output_sha256 = "invalid".to_string();
        source.receipt_ref.clear();
        let report = evaluate_plant_power_balance(&profile(), &source);
        assert_eq!(report.disposition, PowerBalanceDisposition::NotEvaluable);
        assert!(report.diagnostics.contains(&PowerBalanceDiagnostic::InvalidOutputDigest));
        assert!(report.diagnostics.contains(&PowerBalanceDiagnostic::EmptyReceiptReference));
    }
}
