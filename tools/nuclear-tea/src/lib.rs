//! Bounded reference calculations for port and marine energy-system economics.
//!
//! This crate is not a reactor model, grid solver, naval architecture package,
//! nuclear safety analysis tool, investment model, or project cost estimator.
//! It implements transparent arithmetic and evidence-contract checks only.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
}

impl ValidationError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl std::error::Error for ValidationError {}

#[derive(Debug, Clone, Deserialize)]
pub struct PortReferenceDataset {
    pub dataset_id: String,
    pub source_class: String,
    pub sources: Vec<SourceReference>,
    pub shore_power_demand_estimates: Vec<ShorePowerDemandRow>,
    pub estimated_spare_port_intake_capacity: Vec<SpareCapacityRow>,
    pub spare_capacity_estimation: SpareCapacityFormula,
    pub prohibited_inferences: Vec<String>,
    pub required_next_evidence: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SourceReference {
    pub source_id: String,
    pub method: Vec<String>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ShorePowerDemandRow {
    pub port: String,
    pub average_maximum_mw: f64,
    pub absolute_maximum_mw: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpareCapacityRow {
    pub rank: u32,
    pub port: String,
    pub spare_mva: f64,
    pub spare_mw: f64,
    pub connection_risk_label: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpareCapacityFormula {
    pub formula: String,
    pub assumed_power_factor: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PortDatasetReport {
    pub dataset_id: String,
    pub source_class: String,
    pub source_count: usize,
    pub shore_power_row_count: usize,
    pub spare_capacity_row_count: usize,
    pub conversion_checks: usize,
    pub validation_status: String,
}

pub fn validate_port_reference(
    dataset: &PortReferenceDataset,
) -> Result<PortDatasetReport, Vec<ValidationError>> {
    let mut errors = Vec::new();

    if dataset.dataset_id.trim().is_empty() {
        errors.push(ValidationError::new("dataset_id", "must not be empty"));
    }
    if dataset.source_class != "PUBLISHED_ESTIMATE" {
        errors.push(ValidationError::new(
            "source_class",
            "must remain PUBLISHED_ESTIMATE; do not promote estimates to observations",
        ));
    }
    if dataset.shore_power_demand_estimates.len() != 8 {
        errors.push(ValidationError::new(
            "shore_power_demand_estimates",
            "expected exactly eight rows in the frozen South African reference corpus",
        ));
    }
    if dataset.estimated_spare_port_intake_capacity.len() != 8 {
        errors.push(ValidationError::new(
            "estimated_spare_port_intake_capacity",
            "expected exactly eight rows in the frozen South African reference corpus",
        ));
    }
    if dataset.prohibited_inferences.is_empty() {
        errors.push(ValidationError::new(
            "prohibited_inferences",
            "must preserve explicit non-inference rules",
        ));
    }
    if dataset.required_next_evidence.is_empty() {
        errors.push(ValidationError::new(
            "required_next_evidence",
            "must identify evidence required before project-specific reliance",
        ));
    }

    let mut demand_ports = BTreeSet::new();
    for (index, row) in dataset.shore_power_demand_estimates.iter().enumerate() {
        let field = format!("shore_power_demand_estimates[{index}]");
        if row.port.trim().is_empty() {
            errors.push(ValidationError::new(&field, "port must not be empty"));
        }
        if !demand_ports.insert(row.port.trim().to_ascii_lowercase()) {
            errors.push(ValidationError::new(&field, "duplicate port identifier"));
        }
        if !row.average_maximum_mw.is_finite()
            || !row.absolute_maximum_mw.is_finite()
            || row.average_maximum_mw < 0.0
            || row.absolute_maximum_mw < 0.0
        {
            errors.push(ValidationError::new(
                &field,
                "demand values must be finite and nonnegative",
            ));
        } else if row.absolute_maximum_mw < row.average_maximum_mw {
            errors.push(ValidationError::new(
                &field,
                "absolute maximum must be greater than or equal to average maximum",
            ));
        }
    }

    let mut capacity_ports = BTreeSet::new();
    let mut ranks = BTreeSet::new();
    let risk_labels = ["LOW", "MEDIUM", "HIGH"];
    let mut conversion_checks = 0usize;
    for (index, row) in dataset
        .estimated_spare_port_intake_capacity
        .iter()
        .enumerate()
    {
        let field = format!("estimated_spare_port_intake_capacity[{index}]");
        if row.port.trim().is_empty() {
            errors.push(ValidationError::new(&field, "port must not be empty"));
        }
        if !capacity_ports.insert(row.port.trim().to_ascii_lowercase()) {
            errors.push(ValidationError::new(&field, "duplicate port identifier"));
        }
        if !ranks.insert(row.rank) {
            errors.push(ValidationError::new(&field, "duplicate rank"));
        }
        if !row.spare_mva.is_finite()
            || !row.spare_mw.is_finite()
            || row.spare_mva < 0.0
            || row.spare_mw < 0.0
        {
            errors.push(ValidationError::new(
                &field,
                "spare capacity values must be finite and nonnegative",
            ));
        } else if (row.spare_mw - 0.9 * row.spare_mva).abs() > 1e-9 {
            errors.push(ValidationError::new(
                &field,
                "spare MW must equal 0.9 times spare MVA under the source's stated conversion",
            ));
        } else {
            conversion_checks += 1;
        }
        if !risk_labels.contains(&row.connection_risk_label.as_str()) {
            errors.push(ValidationError::new(
                &field,
                "connection_risk_label must be LOW, MEDIUM or HIGH",
            ));
        }
    }

    const EXPECTED_FORMULA: &str =
        "0.9 * (firm_supply_mva - notified_maximum_demand_mva)";
    if dataset.spare_capacity_estimation.formula != EXPECTED_FORMULA {
        errors.push(ValidationError::new(
            "spare_capacity_estimation.formula",
            "formula does not match the frozen source methodology",
        ));
    }
    if !dataset
        .spare_capacity_estimation
        .assumed_power_factor
        .is_finite()
        || (dataset.spare_capacity_estimation.assumed_power_factor - 0.9).abs() > 1e-12
    {
        errors.push(ValidationError::new(
            "spare_capacity_estimation.assumed_power_factor",
            "must preserve the source's stated 0.9 assumption",
        ));
    }

    let source_ids: BTreeSet<&str> = dataset
        .sources
        .iter()
        .map(|source| source.source_id.as_str())
        .collect();
    for required in [
        "CSIR-PTX-SHOREPOWER-2025",
        "CSIR-PTX-PORT-CONNECTION-2025",
    ] {
        if !source_ids.contains(required) {
            errors.push(ValidationError::new(
                "sources",
                format!("missing required source record {required}"),
            ));
        }
    }
    for (index, source) in dataset.sources.iter().enumerate() {
        if source.method.is_empty() {
            errors.push(ValidationError::new(
                format!("sources[{index}].method"),
                "source method must be retained",
            ));
        }
        if source.limitations.is_empty() {
            errors.push(ValidationError::new(
                format!("sources[{index}].limitations"),
                "source limitations must be retained",
            ));
        }
    }

    if errors.is_empty() {
        Ok(PortDatasetReport {
            dataset_id: dataset.dataset_id.clone(),
            source_class: dataset.source_class.clone(),
            source_count: dataset.sources.len(),
            shore_power_row_count: dataset.shore_power_demand_estimates.len(),
            spare_capacity_row_count: dataset.estimated_spare_port_intake_capacity.len(),
            conversion_checks,
            validation_status: "STRUCTURALLY_VALIDATED_SOURCE_DATA_ONLY".to_string(),
        })
    } else {
        Err(errors)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PriceBasis {
    Real { currency: String, base_year: u16 },
    Nominal { currency: String, base_year: u16 },
}

impl PriceBasis {
    pub fn currency(&self) -> &str {
        match self {
            Self::Real { currency, .. } | Self::Nominal { currency, .. } => currency,
        }
    }

    pub fn base_year(&self) -> u16 {
        match self {
            Self::Real { base_year, .. } | Self::Nominal { base_year, .. } => *base_year,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnualFlow {
    /// Project year index. The first record must be year 0.
    pub year_index: u32,
    /// All lifecycle costs for this year, in the declared currency and price basis.
    pub lifecycle_cost: f64,
    /// Delivered service in the same year (e.g. net MWh, m3, tonne-nautical-miles).
    pub delivered_service: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicSeries {
    pub name: String,
    pub price_basis: PriceBasis,
    pub discount_basis: PriceBasis,
    /// A decimal rate (0.08 means 8%), expressed in discount_basis.
    pub discount_rate: f64,
    pub service_unit: String,
    pub delivery_boundary: String,
    pub annual_flows: Vec<AnnualFlow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LevelizedCostResult {
    pub name: String,
    pub currency: String,
    pub base_year: u16,
    pub price_basis: String,
    pub service_unit: String,
    pub delivery_boundary: String,
    pub discounted_lifecycle_cost: f64,
    pub discounted_delivered_service: f64,
    pub levelized_cost_per_service_unit: f64,
}

impl EconomicSeries {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.name.trim().is_empty() {
            return Err(ValidationError::new("name", "must not be empty"));
        }
        if self.price_basis.currency().trim().is_empty() {
            return Err(ValidationError::new("price_basis.currency", "must not be empty"));
        }
        if self.price_basis != self.discount_basis {
            return Err(ValidationError::new(
                "discount_basis",
                "must exactly match price_basis (real cash flows require a real discount rate; nominal requires nominal)",
            ));
        }
        if !self.discount_rate.is_finite() || self.discount_rate <= -1.0 {
            return Err(ValidationError::new(
                "discount_rate",
                "must be finite and greater than -100%",
            ));
        }
        if self.service_unit.trim().is_empty() {
            return Err(ValidationError::new(
                "service_unit",
                "must declare a delivered-service unit",
            ));
        }
        if self.delivery_boundary.trim().is_empty() {
            return Err(ValidationError::new(
                "delivery_boundary",
                "must declare the service delivery boundary",
            ));
        }
        if self.annual_flows.is_empty() {
            return Err(ValidationError::new(
                "annual_flows",
                "must contain at least one annual flow",
            ));
        }
        if self.annual_flows[0].year_index != 0 {
            return Err(ValidationError::new(
                "annual_flows[0].year_index",
                "first flow must be project year 0",
            ));
        }

        let mut previous_year = None;
        let mut any_service = false;
        for (index, flow) in self.annual_flows.iter().enumerate() {
            let field = format!("annual_flows[{index}]");
            if let Some(previous) = previous_year {
                if flow.year_index <= previous {
                    return Err(ValidationError::new(
                        &field,
                        "year_index values must be strictly increasing and unique",
                    ));
                }
            }
            previous_year = Some(flow.year_index);
            if !flow.lifecycle_cost.is_finite() || flow.lifecycle_cost < 0.0 {
                return Err(ValidationError::new(
                    &field,
                    "lifecycle_cost must be finite and nonnegative",
                ));
            }
            if !flow.delivered_service.is_finite() || flow.delivered_service < 0.0 {
                return Err(ValidationError::new(
                    &field,
                    "delivered_service must be finite and nonnegative",
                ));
            }
            any_service |= flow.delivered_service > 0.0;
        }
        if !any_service {
            return Err(ValidationError::new(
                "annual_flows.delivered_service",
                "discounted service denominator is zero; result is INSUFFICIENT_EVIDENCE",
            ));
        }
        Ok(())
    }

    pub fn discounted_totals(&self) -> Result<(f64, f64), ValidationError> {
        self.validate()?;
        let mut discounted_cost = 0.0;
        let mut discounted_service = 0.0;
        let discount_base = 1.0 + self.discount_rate;

        for flow in &self.annual_flows {
            let factor = discount_base.powf(f64::from(flow.year_index));
            let cost_pv = flow.lifecycle_cost / factor;
            let service_pv = flow.delivered_service / factor;
            if !cost_pv.is_finite() || !service_pv.is_finite() {
                return Err(ValidationError::new(
                    "discounted_totals",
                    "discounted values overflowed or became non-finite",
                ));
            }
            discounted_cost += cost_pv;
            discounted_service += service_pv;
        }
        if !discounted_cost.is_finite() || !discounted_service.is_finite() {
            return Err(ValidationError::new(
                "discounted_totals",
                "summed discounted values are non-finite",
            ));
        }
        if discounted_service <= 0.0 {
            return Err(ValidationError::new(
                "discounted_service",
                "must be positive; result is INSUFFICIENT_EVIDENCE",
            ));
        }
        Ok((discounted_cost, discounted_service))
    }

    pub fn levelized_cost(&self) -> Result<LevelizedCostResult, ValidationError> {
        let (discounted_lifecycle_cost, discounted_delivered_service) =
            self.discounted_totals()?;
        let levelized_cost_per_service_unit =
            discounted_lifecycle_cost / discounted_delivered_service;
        if !levelized_cost_per_service_unit.is_finite() {
            return Err(ValidationError::new(
                "levelized_cost_per_service_unit",
                "result is non-finite",
            ));
        }

        Ok(LevelizedCostResult {
            name: self.name.clone(),
            currency: self.price_basis.currency().to_string(),
            base_year: self.price_basis.base_year(),
            price_basis: match &self.price_basis {
                PriceBasis::Real { .. } => "REAL".to_string(),
                PriceBasis::Nominal { .. } => "NOMINAL".to_string(),
            },
            service_unit: self.service_unit.clone(),
            delivery_boundary: self.delivery_boundary.clone(),
            discounted_lifecycle_cost,
            discounted_delivered_service,
            levelized_cost_per_service_unit,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GateState {
    Pass,
    Fail,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateFinding {
    pub gate_id: String,
    pub mandatory: bool,
    pub state: GateState,
    pub evidence_ref: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GateDisposition {
    EligibleForEconomicComparison,
    NoGo,
    Unknown,
}

pub fn evaluate_mandatory_gates(findings: &[GateFinding]) -> GateDisposition {
    let mandatory: Vec<&GateFinding> = findings.iter().filter(|item| item.mandatory).collect();
    if mandatory.is_empty() {
        return GateDisposition::Unknown;
    }
    if mandatory.iter().any(|item| item.state == GateState::Fail) {
        return GateDisposition::NoGo;
    }
    if mandatory.iter().any(|item| item.state == GateState::Unknown) {
        return GateDisposition::Unknown;
    }
    GateDisposition::EligibleForEconomicComparison
}

/// Compare two levelized service-cost series only when their service boundary,
/// unit, currency and price basis match and all mandatory gates explicitly pass.
pub fn compare_eligible_series(
    left: &EconomicSeries,
    left_gates: &[GateFinding],
    right: &EconomicSeries,
    right_gates: &[GateFinding],
) -> Result<std::cmp::Ordering, ValidationError> {
    if evaluate_mandatory_gates(left_gates)
        != GateDisposition::EligibleForEconomicComparison
        || evaluate_mandatory_gates(right_gates)
            != GateDisposition::EligibleForEconomicComparison
    {
        return Err(ValidationError::new(
            "comparison",
            "NO_GO or UNKNOWN mandatory gate blocks economic ranking",
        ));
    }
    if left.price_basis != right.price_basis {
        return Err(ValidationError::new(
            "comparison.price_basis",
            "currency, price basis and base year must match",
        ));
    }
    if left.discount_rate.to_bits() != right.discount_rate.to_bits() {
        return Err(ValidationError::new(
            "comparison.discount_rate",
            "discount rates must match for a direct matched comparison",
        ));
    }
    if left.service_unit != right.service_unit {
        return Err(ValidationError::new(
            "comparison.service_unit",
            "delivered-service units must match",
        ));
    }
    if left.delivery_boundary != right.delivery_boundary {
        return Err(ValidationError::new(
            "comparison.delivery_boundary",
            "delivery boundaries must match",
        ));
    }

    let left_cost = left.levelized_cost()?.levelized_cost_per_service_unit;
    let right_cost = right.levelized_cost()?.levelized_cost_per_service_unit;
    Ok(left_cost.total_cmp(&right_cost))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_usd_basis() -> PriceBasis {
        PriceBasis::Real {
            currency: "USD".to_string(),
            base_year: 2025,
        }
    }

    fn exact_series(name: &str) -> EconomicSeries {
        EconomicSeries {
            name: name.to_string(),
            price_basis: real_usd_basis(),
            discount_basis: real_usd_basis(),
            discount_rate: 0.10,
            service_unit: "MWh_delivered".to_string(),
            delivery_boundary: "customer_busbar".to_string(),
            annual_flows: vec![
                AnnualFlow {
                    year_index: 0,
                    lifecycle_cost: 100.0,
                    delivered_service: 10.0,
                },
                AnnualFlow {
                    year_index: 1,
                    lifecycle_cost: 110.0,
                    delivered_service: 11.0,
                },
            ],
        }
    }

    fn passing_gates() -> Vec<GateFinding> {
        vec![
            GateFinding {
                gate_id: "legal_pathway".to_string(),
                mandatory: true,
                state: GateState::Pass,
                evidence_ref: Some("fixture://legal".to_string()),
            },
            GateFinding {
                gate_id: "insurance_pathway".to_string(),
                mandatory: true,
                state: GateState::Pass,
                evidence_ref: Some("fixture://insurance".to_string()),
            },
        ]
    }

    #[test]
    fn sa_port_reference_json_parses_and_validates() {
        let input = include_str!("../../../research/nuclear/port-reference-evidence-sa-2025.json");
        let dataset: PortReferenceDataset = serde_json::from_str(input).expect("valid frozen JSON");
        let report = validate_port_reference(&dataset).expect("valid source contract");
        assert_eq!(report.shore_power_row_count, 8);
        assert_eq!(report.spare_capacity_row_count, 8);
        assert_eq!(report.conversion_checks, 8);
        assert_eq!(report.source_class, "PUBLISHED_ESTIMATE");
    }

    #[test]
    fn port_reference_rejects_formula_drift() {
        let input = include_str!("../../../research/nuclear/port-reference-evidence-sa-2025.json");
        let mut dataset: PortReferenceDataset =
            serde_json::from_str(input).expect("valid frozen JSON");
        dataset.estimated_spare_port_intake_capacity[0].spare_mw += 0.1;
        assert!(validate_port_reference(&dataset).is_err());
    }

    #[test]
    fn discounted_cost_and_service_match_hand_calculation() {
        let series = exact_series("analytical_fixture");
        let (cost_pv, service_pv) = series.discounted_totals().expect("valid series");
        assert!((cost_pv - 200.0).abs() < 1e-12);
        assert!((service_pv - 20.0).abs() < 1e-12);
        let result = series.levelized_cost().expect("valid levelized cost");
        assert!((result.levelized_cost_per_service_unit - 10.0).abs() < 1e-12);
    }

    #[test]
    fn nominal_and_real_basis_mismatch_is_rejected() {
        let mut series = exact_series("basis_mismatch");
        series.discount_basis = PriceBasis::Nominal {
            currency: "USD".to_string(),
            base_year: 2025,
        };
        assert!(series.validate().is_err());
    }

    #[test]
    fn zero_service_denominator_is_rejected() {
        let mut series = exact_series("zero_service");
        for flow in &mut series.annual_flows {
            flow.delivered_service = 0.0;
        }
        assert!(series.validate().is_err());
    }

    #[test]
    fn negative_and_non_finite_values_are_rejected() {
        let mut negative = exact_series("negative_cost");
        negative.annual_flows[0].lifecycle_cost = -1.0;
        assert!(negative.validate().is_err());

        let mut non_finite = exact_series("non_finite_service");
        non_finite.annual_flows[0].delivered_service = f64::NAN;
        assert!(non_finite.validate().is_err());
    }

    #[test]
    fn duplicate_or_out_of_order_years_are_rejected() {
        let mut series = exact_series("duplicate_year");
        series.annual_flows[1].year_index = 0;
        assert!(series.validate().is_err());
    }

    #[test]
    fn failed_gate_overrides_unknown_gate() {
        let findings = vec![
            GateFinding {
                gate_id: "insurance".to_string(),
                mandatory: true,
                state: GateState::Unknown,
                evidence_ref: None,
            },
            GateFinding {
                gate_id: "port_access".to_string(),
                mandatory: true,
                state: GateState::Fail,
                evidence_ref: Some("fixture://not-admitted".to_string()),
            },
        ];
        assert_eq!(evaluate_mandatory_gates(&findings), GateDisposition::NoGo);
    }

    #[test]
    fn unknown_gate_blocks_economic_ranking() {
        let findings = vec![GateFinding {
            gate_id: "insurance".to_string(),
            mandatory: true,
            state: GateState::Unknown,
            evidence_ref: None,
        }];
        assert_eq!(evaluate_mandatory_gates(&findings), GateDisposition::Unknown);
    }

    #[test]
    fn empty_mandatory_gate_set_is_unknown() {
        let findings = vec![GateFinding {
            gate_id: "optional_training".to_string(),
            mandatory: false,
            state: GateState::Pass,
            evidence_ref: None,
        }];
        assert_eq!(evaluate_mandatory_gates(&findings), GateDisposition::Unknown);
    }

    #[test]
    fn comparison_requires_matching_service_scope_and_passing_gates() {
        let left = exact_series("left");
        let right = exact_series("right");
        let gates = passing_gates();
        assert_eq!(
            compare_eligible_series(&left, &gates, &right, &gates).unwrap(),
            std::cmp::Ordering::Equal
        );

        let mut mismatched = right.clone();
        mismatched.service_unit = "passenger_days".to_string();
        assert!(compare_eligible_series(&left, &gates, &mismatched, &gates).is_err());

        let unknown = vec![GateFinding {
            gate_id: "port_access".to_string(),
            mandatory: true,
            state: GateState::Unknown,
            evidence_ref: None,
        }];
        assert!(compare_eligible_series(&left, &unknown, &right, &gates).is_err());
    }

    #[test]
    fn cargo_and_passenger_service_units_remain_explicit() {
        let mut cargo = exact_series("cargo");
        cargo.service_unit = "tonne_nautical_miles".to_string();
        let mut passenger = exact_series("passenger");
        passenger.service_unit = "passenger_days".to_string();
        assert_ne!(cargo.service_unit, passenger.service_unit);
        assert!(compare_eligible_series(
            &cargo,
            &passing_gates(),
            &passenger,
            &passing_gates()
        )
        .is_err());
    }
}
