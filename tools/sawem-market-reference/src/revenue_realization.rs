//! Evidence-attributed bridge from theoretical gross revenue to realized net cash.
//!
//! Each adjustment is a signed, source-bound amount in minor ZAR:
//! positive adjustments add value and negative adjustments reduce value.
//! The bridge does not estimate missing effects. If a caller-declared required
//! category has no source-backed component, a complete reconciliation and
//! unexplained residual are withheld.
//!
//! This supports a decomposed alternative to aggregate realization haircuts.
//! It is an attribution framework, not a market revenue forecaster or
//! settlement calculator.

use std::collections::BTreeSet;

const BASIS_POINTS: i128 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RevenueComponentKind {
    ForecastRegret,
    AvailabilityLoss,
    ExecutionDeviation,
    BalancingSettlement,
    MarketFees,
    DegradationAndWarranty,
    FinancingAndCollateral,
    ContractAndCurtailment,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RevenueBridgeComponent<'a> {
    pub component_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub available_at_utc: i64,
    pub kind: RevenueComponentKind,
    /// Positive adds realized value; negative reduces it.
    pub adjustment_minor_zar: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RevenueBridgeInputs<'a> {
    pub scenario_id: &'a str,
    pub evaluation_as_of_utc: i64,
    pub theoretical_gross_revenue_minor_zar: i64,
    pub theoretical_source_ref: &'a str,
    pub theoretical_source_sha256: &'a str,
    pub theoretical_available_at_utc: i64,
    pub realized_net_cash_minor_zar: i64,
    pub realized_source_ref: &'a str,
    pub realized_source_sha256: &'a str,
    pub realized_available_at_utc: i64,
    /// The caller declares which bridge categories must be evidenced for this
    /// project/contract. An inapplicable category need not be listed.
    pub required_categories: &'a [RevenueComponentKind],
    pub components: &'a [RevenueBridgeComponent<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevenueBridgeError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    NegativeTheoreticalGrossRevenue,
    SourceNotAvailableAtEvaluation,
    NoRequiredCategories,
    DuplicateRequiredCategory,
    DuplicateComponentId,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevenueBridgeAssessment<'a> {
    pub scenario_id: &'a str,
    pub evaluation_as_of_utc: i64,
    pub theoretical_gross_revenue_minor_zar: i64,
    pub realized_net_cash_minor_zar: i64,
    pub component_count: u64,
    pub required_category_count: u64,
    pub missing_categories: Vec<RevenueComponentKind>,
    pub complete: bool,
    /// Sum of only the components actually supported by evidence.
    pub known_adjustments_sum_minor_zar: i128,
    /// Theoretical gross plus known adjustments, even when incomplete.
    pub partial_reconstructed_net_minor_zar: i128,
    /// Populated only when every caller-required category is evidenced.
    pub reconstructed_net_minor_zar: Option<i128>,
    /// Realized net cash less reconstructed net; never forced to zero.
    pub unexplained_residual_minor_zar: Option<i128>,
    /// Realized net / theoretical gross in basis points, only for a complete
    /// bridge and positive theoretical gross revenue.
    pub realized_net_to_theoretical_gross_bps: Option<i128>,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn checked_add(current: i128, value: i128) -> Result<i128, RevenueBridgeError> {
    current
        .checked_add(value)
        .ok_or(RevenueBridgeError::ArithmeticOverflow)
}

pub fn evaluate_revenue_realization_bridge<'a>(
    inputs: RevenueBridgeInputs<'a>,
) -> Result<RevenueBridgeAssessment<'a>, RevenueBridgeError> {
    if !nonempty(inputs.scenario_id)
        || !nonempty(inputs.theoretical_source_ref)
        || !nonempty(inputs.realized_source_ref)
    {
        return Err(RevenueBridgeError::InvalidEvidenceReference);
    }
    if !valid_sha256(inputs.theoretical_source_sha256)
        || !valid_sha256(inputs.realized_source_sha256)
    {
        return Err(RevenueBridgeError::InvalidSourceDigest);
    }
    if inputs.theoretical_gross_revenue_minor_zar < 0 {
        return Err(RevenueBridgeError::NegativeTheoreticalGrossRevenue);
    }
    if inputs.theoretical_available_at_utc > inputs.evaluation_as_of_utc
        || inputs.realized_available_at_utc > inputs.evaluation_as_of_utc
    {
        return Err(RevenueBridgeError::SourceNotAvailableAtEvaluation);
    }
    if inputs.required_categories.is_empty() {
        return Err(RevenueBridgeError::NoRequiredCategories);
    }

    let mut required = BTreeSet::new();
    for category in inputs.required_categories.iter().copied() {
        if !required.insert(category) {
            return Err(RevenueBridgeError::DuplicateRequiredCategory);
        }
    }

    let mut component_ids = BTreeSet::new();
    let mut evidenced_categories = BTreeSet::new();
    let mut known_adjustments = 0_i128;
    for component in inputs.components.iter().copied() {
        if !nonempty(component.component_id) || !nonempty(component.source_ref) {
            return Err(RevenueBridgeError::InvalidEvidenceReference);
        }
        if !valid_sha256(component.source_sha256) {
            return Err(RevenueBridgeError::InvalidSourceDigest);
        }
        if component.available_at_utc > inputs.evaluation_as_of_utc {
            return Err(RevenueBridgeError::SourceNotAvailableAtEvaluation);
        }
        if !component_ids.insert(component.component_id) {
            return Err(RevenueBridgeError::DuplicateComponentId);
        }
        evidenced_categories.insert(component.kind);
        known_adjustments = checked_add(
            known_adjustments,
            i128::from(component.adjustment_minor_zar),
        )?;
    }

    let missing_categories = required
        .difference(&evidenced_categories)
        .copied()
        .collect::<Vec<_>>();
    let complete = missing_categories.is_empty();
    let partial_reconstructed = checked_add(
        i128::from(inputs.theoretical_gross_revenue_minor_zar),
        known_adjustments,
    )?;
    let reconstructed_net = complete.then_some(partial_reconstructed);
    let unexplained_residual = reconstructed_net
        .map(|reconstructed| i128::from(inputs.realized_net_cash_minor_zar) - reconstructed);

    let realized_retention = if complete && inputs.theoretical_gross_revenue_minor_zar > 0 {
        Some(
            i128::from(inputs.realized_net_cash_minor_zar)
                .checked_mul(BASIS_POINTS)
                .ok_or(RevenueBridgeError::ArithmeticOverflow)?
                / i128::from(inputs.theoretical_gross_revenue_minor_zar),
        )
    } else {
        None
    };

    Ok(RevenueBridgeAssessment {
        scenario_id: inputs.scenario_id,
        evaluation_as_of_utc: inputs.evaluation_as_of_utc,
        theoretical_gross_revenue_minor_zar: inputs.theoretical_gross_revenue_minor_zar,
        realized_net_cash_minor_zar: inputs.realized_net_cash_minor_zar,
        component_count: u64::try_from(inputs.components.len())
            .map_err(|_| RevenueBridgeError::ArithmeticOverflow)?,
        required_category_count: u64::try_from(required.len())
            .map_err(|_| RevenueBridgeError::ArithmeticOverflow)?,
        missing_categories,
        complete,
        known_adjustments_sum_minor_zar: known_adjustments,
        partial_reconstructed_net_minor_zar: partial_reconstructed,
        reconstructed_net_minor_zar: reconstructed_net,
        unexplained_residual_minor_zar: unexplained_residual,
        realized_net_to_theoretical_gross_bps: realized_retention,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    const REQUIRED: [RevenueComponentKind; 7] = [
        RevenueComponentKind::ForecastRegret,
        RevenueComponentKind::AvailabilityLoss,
        RevenueComponentKind::ExecutionDeviation,
        RevenueComponentKind::BalancingSettlement,
        RevenueComponentKind::MarketFees,
        RevenueComponentKind::DegradationAndWarranty,
        RevenueComponentKind::FinancingAndCollateral,
    ];

    fn component(
        id: &'static str,
        kind: RevenueComponentKind,
        amount: i64,
    ) -> RevenueBridgeComponent<'static> {
        RevenueBridgeComponent {
            component_id: id,
            source_ref: id,
            source_sha256: SOURCE_SHA,
            available_at_utc: 90,
            kind,
            adjustment_minor_zar: amount,
        }
    }

    fn base_inputs<'a>(
        required_categories: &'a [RevenueComponentKind],
        components: &'a [RevenueBridgeComponent<'a>],
    ) -> RevenueBridgeInputs<'a> {
        RevenueBridgeInputs {
            scenario_id: "battery-001/revenue-realization",
            evaluation_as_of_utc: 100,
            theoretical_gross_revenue_minor_zar: 1_000_000,
            theoretical_source_ref: "model/perfect-information-gross",
            theoretical_source_sha256: SOURCE_SHA,
            theoretical_available_at_utc: 80,
            realized_net_cash_minor_zar: 700_000,
            realized_source_ref: "bank/settlement/net-cash",
            realized_source_sha256: SOURCE_SHA,
            realized_available_at_utc: 95,
            required_categories,
            components,
        }
    }

    #[test]
    fn complete_bridge_attributes_gaps_and_preserves_unexplained_residual() {
        let components = [
            component("forecast", RevenueComponentKind::ForecastRegret, -100_000),
            component("availability", RevenueComponentKind::AvailabilityLoss, -50_000),
            component("execution", RevenueComponentKind::ExecutionDeviation, -70_000),
            component("balancing", RevenueComponentKind::BalancingSettlement, -20_000),
            component("fees", RevenueComponentKind::MarketFees, -10_000),
            component("degradation", RevenueComponentKind::DegradationAndWarranty, -25_000),
            component("finance", RevenueComponentKind::FinancingAndCollateral, -5_000),
        ];
        let result = evaluate_revenue_realization_bridge(base_inputs(&REQUIRED, &components))
            .unwrap();
        assert!(result.complete);
        assert_eq!(result.known_adjustments_sum_minor_zar, -280_000);
        assert_eq!(result.partial_reconstructed_net_minor_zar, 720_000);
        assert_eq!(result.reconstructed_net_minor_zar, Some(720_000));
        assert_eq!(result.unexplained_residual_minor_zar, Some(-20_000));
        assert_eq!(result.realized_net_to_theoretical_gross_bps, Some(7_000));
    }

    #[test]
    fn missing_required_category_withholds_complete_reconciliation() {
        let components = [
            component("forecast", RevenueComponentKind::ForecastRegret, -100_000),
            component("fees", RevenueComponentKind::MarketFees, -10_000),
        ];
        let result = evaluate_revenue_realization_bridge(base_inputs(&REQUIRED, &components))
            .unwrap();
        assert!(!result.complete);
        assert!(result.missing_categories.contains(&RevenueComponentKind::BalancingSettlement));
        assert_eq!(result.partial_reconstructed_net_minor_zar, 890_000);
        assert_eq!(result.reconstructed_net_minor_zar, None);
        assert_eq!(result.unexplained_residual_minor_zar, None);
        assert_eq!(result.realized_net_to_theoretical_gross_bps, None);
    }

    #[test]
    fn positive_contract_adjustment_can_offset_a_negative_component() {
        let required = [
            RevenueComponentKind::ForecastRegret,
            RevenueComponentKind::ContractAndCurtailment,
        ];
        let components = [
            component("forecast", RevenueComponentKind::ForecastRegret, -50_000),
            component("contract", RevenueComponentKind::ContractAndCurtailment, 25_000),
        ];
        let result =
            evaluate_revenue_realization_bridge(base_inputs(&required, &components)).unwrap();
        assert!(result.complete);
        assert_eq!(result.known_adjustments_sum_minor_zar, -25_000);
        assert_eq!(result.reconstructed_net_minor_zar, Some(975_000));
    }

    #[test]
    fn future_known_cash_or_adjustment_evidence_is_rejected() {
        let late_component = [RevenueBridgeComponent {
            available_at_utc: 101,
            ..component("future", RevenueComponentKind::ForecastRegret, -10_000)
        }];
        assert_eq!(
            evaluate_revenue_realization_bridge(base_inputs(&REQUIRED, &late_component)),
            Err(RevenueBridgeError::SourceNotAvailableAtEvaluation)
        );
        let late_actual = RevenueBridgeInputs {
            realized_available_at_utc: 101,
            ..base_inputs(&REQUIRED, &[])
        };
        assert_eq!(
            evaluate_revenue_realization_bridge(late_actual),
            Err(RevenueBridgeError::SourceNotAvailableAtEvaluation)
        );
    }

    #[test]
    fn duplicate_component_and_required_category_ids_fail_closed() {
        let duplicate_components = [
            component("same", RevenueComponentKind::ForecastRegret, -10_000),
            component("same", RevenueComponentKind::MarketFees, -1_000),
        ];
        assert_eq!(
            evaluate_revenue_realization_bridge(base_inputs(&REQUIRED, &duplicate_components)),
            Err(RevenueBridgeError::DuplicateComponentId)
        );
        let duplicate_required = [
            RevenueComponentKind::ForecastRegret,
            RevenueComponentKind::ForecastRegret,
        ];
        assert_eq!(
            evaluate_revenue_realization_bridge(base_inputs(&duplicate_required, &[])),
            Err(RevenueBridgeError::DuplicateRequiredCategory)
        );
    }

    #[test]
    fn invalid_hashes_and_negative_theoretical_gross_fail_closed() {
        let bad_hash = RevenueBridgeInputs {
            theoretical_source_sha256: "bad",
            ..base_inputs(&REQUIRED, &[])
        };
        assert_eq!(
            evaluate_revenue_realization_bridge(bad_hash),
            Err(RevenueBridgeError::InvalidSourceDigest)
        );
        let negative_gross = RevenueBridgeInputs {
            theoretical_gross_revenue_minor_zar: -1,
            ..base_inputs(&REQUIRED, &[])
        };
        assert_eq!(
            evaluate_revenue_realization_bridge(negative_gross),
            Err(RevenueBridgeError::NegativeTheoreticalGrossRevenue)
        );
    }
}
