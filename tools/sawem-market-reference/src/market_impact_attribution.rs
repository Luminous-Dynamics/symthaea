//! Matched price-taking versus price-impact-adjusted BESS revenue attribution.
//!
//! The caller supplies source-bound reference prices and price-impact-adjusted
//! clearing prices from an external market simulation or replay. This module
//! computes the value difference with exact integer numerators and requires
//! candidate/comparator runs to share the same reference price path, market
//! impact model build, horizon and interval census. It does not solve market
//! clearing, estimate price elasticity or authenticate source archives.

const REVENUE_DENOMINATOR_MILLI_WH_PER_MWH: i128 = 1_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarketImpactInterval<'a> {
    pub interval_id: &'a str,
    pub interval_start_utc: i64,
    pub interval_end_utc: i64,
    /// Signed net energy at the grid connection point; positive export,
    /// negative import. Unit: milli-Wh.
    pub net_energy_milli_wh: i64,
    pub energy_record_id: &'a str,
    pub energy_source_ref: &'a str,
    pub energy_source_sha256: &'a str,
    pub energy_available_at_utc: i64,
    /// Reference price before the evaluated asset's own market-price impact.
    /// Unit: minor ZAR/MWh.
    pub reference_price_minor_zar_per_mwh: i64,
    pub reference_price_record_id: &'a str,
    pub reference_price_source_ref: &'a str,
    pub reference_price_sha256: &'a str,
    pub reference_price_available_at_utc: i64,
    /// Counterfactual/market-simulated clearing price with this policy's
    /// submitted net energy. Unit: minor ZAR/MWh.
    pub impact_adjusted_price_minor_zar_per_mwh: i64,
    pub impact_price_record_id: &'a str,
    pub impact_price_source_ref: &'a str,
    pub impact_price_sha256: &'a str,
    pub impact_price_available_at_utc: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarketImpactPolicyInputs<'a> {
    pub scenario_id: &'a str,
    pub evaluation_as_of_utc: i64,
    pub horizon_start_utc: i64,
    pub horizon_end_utc: i64,
    pub policy_id: &'a str,
    pub policy_sha256: &'a str,
    pub market_impact_model_id: &'a str,
    pub market_impact_model_sha256: &'a str,
    /// Frozen identity and digest of the common no-self-impact reference series.
    pub reference_price_basis_id: &'a str,
    pub reference_price_basis_sha256: &'a str,
    pub intervals: &'a [MarketImpactInterval<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketImpactError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidHorizon,
    EmptyIntervalSet,
    InvalidInterval,
    IntervalOutOfOrder,
    OverlappingIntervals,
    IncompleteHorizonCoverage,
    DuplicateIntervalId,
    DuplicateEnergyRecordId,
    DuplicateReferencePriceRecordId,
    DuplicateImpactPriceRecordId,
    FutureKnownEvidence,
    PolicyIdentityNotDistinct,
    ScenarioMismatch,
    EvaluationTimeMismatch,
    HorizonMismatch,
    MarketImpactModelMismatch,
    ReferencePriceBasisMismatch,
    IntervalCensusMismatch,
    ReferencePricePathMismatch,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarketImpactPolicyAssessment<'a> {
    pub scenario_id: &'a str,
    pub evaluation_as_of_utc: i64,
    pub policy_id: &'a str,
    pub policy_sha256: &'a str,
    pub market_impact_model_id: &'a str,
    pub market_impact_model_sha256: &'a str,
    pub interval_count: u64,
    pub horizon_duration_seconds: u64,
    pub signed_net_energy_milli_wh: i128,
    /// Exact revenue numerator; divide by 1,000,000,000 for minor ZAR.
    pub price_taker_revenue_numerator_minor_zar: i128,
    /// Exact revenue numerator at the impact-adjusted clearing price.
    pub impact_adjusted_revenue_numerator_minor_zar: i128,
    /// Impact-adjusted less price-taking revenue. Negative is self-impact loss.
    pub price_impact_adjustment_numerator_minor_zar: i128,
    pub revenue_denominator_milli_wh_per_mwh: i128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarketImpactComparisonAssessment<'a> {
    pub scenario_id: &'a str,
    pub candidate_policy_id: &'a str,
    pub comparator_policy_id: &'a str,
    pub interval_count: u64,
    pub price_taker_revenue_delta_numerator_minor_zar: i128,
    pub impact_adjusted_revenue_delta_numerator_minor_zar: i128,
    /// Difference between the two policies' own-price-impact adjustments.
    pub differential_price_impact_numerator_minor_zar: i128,
    pub revenue_denominator_milli_wh_per_mwh: i128,
    pub candidate_price_impact_adjustment_numerator_minor_zar: i128,
    pub comparator_price_impact_adjustment_numerator_minor_zar: i128,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_reference(reference: &str, digest: &str) -> Result<(), MarketImpactError> {
    if !nonempty(reference) {
        return Err(MarketImpactError::InvalidEvidenceReference);
    }
    if !valid_sha256(digest) {
        return Err(MarketImpactError::InvalidSourceDigest);
    }
    Ok(())
}

fn add_i128(left: i128, right: i128) -> Result<i128, MarketImpactError> {
    left.checked_add(right).ok_or(MarketImpactError::ArithmeticOverflow)
}

fn checked_revenue_numerator(
    energy_milli_wh: i64,
    price_minor_zar_per_mwh: i64,
) -> Result<i128, MarketImpactError> {
    i128::from(energy_milli_wh)
        .checked_mul(i128::from(price_minor_zar_per_mwh))
        .ok_or(MarketImpactError::ArithmeticOverflow)
}

/// Calculate price-taking and price-impact-adjusted revenues for one policy.
/// All intervals must exactly and contiguously cover the declared horizon.
pub fn evaluate_market_impact_policy<'a>(
    inputs: MarketImpactPolicyInputs<'a>,
) -> Result<MarketImpactPolicyAssessment<'a>, MarketImpactError> {
    if !nonempty(inputs.scenario_id)
        || !nonempty(inputs.policy_id)
        || !nonempty(inputs.market_impact_model_id)
        || !nonempty(inputs.reference_price_basis_id)
    {
        return Err(MarketImpactError::InvalidEvidenceReference);
    }
    validate_reference(inputs.policy_id, inputs.policy_sha256)?;
    validate_reference(inputs.market_impact_model_id, inputs.market_impact_model_sha256)?;
    validate_reference(inputs.reference_price_basis_id, inputs.reference_price_basis_sha256)?;
    if inputs.horizon_start_utc >= inputs.horizon_end_utc
        || inputs.evaluation_as_of_utc < inputs.horizon_end_utc
    {
        return Err(MarketImpactError::InvalidHorizon);
    }
    if inputs.intervals.is_empty() {
        return Err(MarketImpactError::EmptyIntervalSet);
    }

    let mut price_taker_numerator = 0_i128;
    let mut impact_adjusted_numerator = 0_i128;
    let mut signed_energy = 0_i128;
    let mut previous_end: Option<i64> = None;
    let mut interval_ids = std::collections::BTreeSet::new();
    let mut energy_ids = std::collections::BTreeSet::new();
    let mut reference_price_ids = std::collections::BTreeSet::new();
    let mut impact_price_ids = std::collections::BTreeSet::new();

    for interval in inputs.intervals.iter().copied() {
        if !nonempty(interval.interval_id)
            || !nonempty(interval.energy_record_id)
            || !nonempty(interval.reference_price_record_id)
            || !nonempty(interval.impact_price_record_id)
        {
            return Err(MarketImpactError::InvalidEvidenceReference);
        }
        validate_reference(interval.energy_source_ref, interval.energy_source_sha256)?;
        validate_reference(interval.reference_price_source_ref, interval.reference_price_sha256)?;
        validate_reference(interval.impact_price_source_ref, interval.impact_price_sha256)?;

        if interval.interval_start_utc >= interval.interval_end_utc
            || interval.interval_start_utc < inputs.horizon_start_utc
            || interval.interval_end_utc > inputs.horizon_end_utc
        {
            return Err(MarketImpactError::InvalidInterval);
        }
        if interval.energy_available_at_utc > inputs.evaluation_as_of_utc
            || interval.reference_price_available_at_utc > inputs.evaluation_as_of_utc
            || interval.impact_price_available_at_utc > inputs.evaluation_as_of_utc
        {
            return Err(MarketImpactError::FutureKnownEvidence);
        }
        if !interval_ids.insert(interval.interval_id) {
            return Err(MarketImpactError::DuplicateIntervalId);
        }
        if !energy_ids.insert(interval.energy_record_id) {
            return Err(MarketImpactError::DuplicateEnergyRecordId);
        }
        if !reference_price_ids.insert(interval.reference_price_record_id) {
            return Err(MarketImpactError::DuplicateReferencePriceRecordId);
        }
        if !impact_price_ids.insert(interval.impact_price_record_id) {
            return Err(MarketImpactError::DuplicateImpactPriceRecordId);
        }

        match previous_end {
            None if interval.interval_start_utc != inputs.horizon_start_utc => {
                return Err(MarketImpactError::IncompleteHorizonCoverage);
            }
            Some(previous) if interval.interval_start_utc < previous => {
                return Err(MarketImpactError::OverlappingIntervals);
            }
            Some(previous) if interval.interval_start_utc > previous => {
                return Err(MarketImpactError::IncompleteHorizonCoverage);
            }
            _ => {}
        }
        previous_end = Some(interval.interval_end_utc);

        let price_taker = checked_revenue_numerator(
            interval.net_energy_milli_wh,
            interval.reference_price_minor_zar_per_mwh,
        )?;
        let impact_adjusted = checked_revenue_numerator(
            interval.net_energy_milli_wh,
            interval.impact_adjusted_price_minor_zar_per_mwh,
        )?;
        price_taker_numerator = add_i128(price_taker_numerator, price_taker)?;
        impact_adjusted_numerator = add_i128(impact_adjusted_numerator, impact_adjusted)?;
        signed_energy = add_i128(signed_energy, i128::from(interval.net_energy_milli_wh))?;
    }

    if previous_end != Some(inputs.horizon_end_utc) {
        return Err(MarketImpactError::IncompleteHorizonCoverage);
    }
    let duration = i128::from(inputs.horizon_end_utc)
        .checked_sub(i128::from(inputs.horizon_start_utc))
        .ok_or(MarketImpactError::ArithmeticOverflow)?;
    let duration_u64 = u64::try_from(duration).map_err(|_| MarketImpactError::ArithmeticOverflow)?;
    let interval_count =
        u64::try_from(inputs.intervals.len()).map_err(|_| MarketImpactError::ArithmeticOverflow)?;
    let price_impact_adjustment = impact_adjusted_numerator
        .checked_sub(price_taker_numerator)
        .ok_or(MarketImpactError::ArithmeticOverflow)?;

    Ok(MarketImpactPolicyAssessment {
        scenario_id: inputs.scenario_id,
        evaluation_as_of_utc: inputs.evaluation_as_of_utc,
        policy_id: inputs.policy_id,
        policy_sha256: inputs.policy_sha256,
        market_impact_model_id: inputs.market_impact_model_id,
        market_impact_model_sha256: inputs.market_impact_model_sha256,
        interval_count,
        horizon_duration_seconds: duration_u64,
        signed_net_energy_milli_wh: signed_energy,
        price_taker_revenue_numerator_minor_zar: price_taker_numerator,
        impact_adjusted_revenue_numerator_minor_zar: impact_adjusted_numerator,
        price_impact_adjustment_numerator_minor_zar: price_impact_adjustment,
        revenue_denominator_milli_wh_per_mwh: REVENUE_DENOMINATOR_MILLI_WH_PER_MWH,
    })
}

/// Compare two policies only when they share the exact scenario, horizon,
/// market-impact model build and exogenous reference-price records.
pub fn compare_market_impact_policies<'a>(
    candidate_inputs: MarketImpactPolicyInputs<'a>,
    comparator_inputs: MarketImpactPolicyInputs<'a>,
) -> Result<MarketImpactComparisonAssessment<'a>, MarketImpactError> {
    if candidate_inputs.policy_id == comparator_inputs.policy_id
        && candidate_inputs.policy_sha256 == comparator_inputs.policy_sha256
    {
        return Err(MarketImpactError::PolicyIdentityNotDistinct);
    }
    if candidate_inputs.scenario_id != comparator_inputs.scenario_id {
        return Err(MarketImpactError::ScenarioMismatch);
    }
    if candidate_inputs.evaluation_as_of_utc != comparator_inputs.evaluation_as_of_utc {
        return Err(MarketImpactError::EvaluationTimeMismatch);
    }
    if candidate_inputs.horizon_start_utc != comparator_inputs.horizon_start_utc
        || candidate_inputs.horizon_end_utc != comparator_inputs.horizon_end_utc
    {
        return Err(MarketImpactError::HorizonMismatch);
    }
    if candidate_inputs.market_impact_model_id != comparator_inputs.market_impact_model_id
        || candidate_inputs.market_impact_model_sha256
            != comparator_inputs.market_impact_model_sha256
    {
        return Err(MarketImpactError::MarketImpactModelMismatch);
    }
    if candidate_inputs.reference_price_basis_id != comparator_inputs.reference_price_basis_id
        || candidate_inputs.reference_price_basis_sha256
            != comparator_inputs.reference_price_basis_sha256
    {
        return Err(MarketImpactError::ReferencePriceBasisMismatch);
    }
    if candidate_inputs.intervals.len() != comparator_inputs.intervals.len() {
        return Err(MarketImpactError::IntervalCensusMismatch);
    }

    for (candidate, comparator) in candidate_inputs
        .intervals
        .iter()
        .zip(comparator_inputs.intervals.iter())
    {
        if candidate.interval_id != comparator.interval_id
            || candidate.interval_start_utc != comparator.interval_start_utc
            || candidate.interval_end_utc != comparator.interval_end_utc
        {
            return Err(MarketImpactError::IntervalCensusMismatch);
        }
        if candidate.reference_price_minor_zar_per_mwh
            != comparator.reference_price_minor_zar_per_mwh
            || candidate.reference_price_record_id != comparator.reference_price_record_id
            || candidate.reference_price_source_ref != comparator.reference_price_source_ref
            || candidate.reference_price_sha256 != comparator.reference_price_sha256
            || candidate.reference_price_available_at_utc
                != comparator.reference_price_available_at_utc
        {
            return Err(MarketImpactError::ReferencePricePathMismatch);
        }
    }

    let candidate = evaluate_market_impact_policy(candidate_inputs)?;
    let comparator = evaluate_market_impact_policy(comparator_inputs)?;
    let price_taker_delta = candidate
        .price_taker_revenue_numerator_minor_zar
        .checked_sub(comparator.price_taker_revenue_numerator_minor_zar)
        .ok_or(MarketImpactError::ArithmeticOverflow)?;
    let impact_adjusted_delta = candidate
        .impact_adjusted_revenue_numerator_minor_zar
        .checked_sub(comparator.impact_adjusted_revenue_numerator_minor_zar)
        .ok_or(MarketImpactError::ArithmeticOverflow)?;
    let differential_impact = candidate
        .price_impact_adjustment_numerator_minor_zar
        .checked_sub(comparator.price_impact_adjustment_numerator_minor_zar)
        .ok_or(MarketImpactError::ArithmeticOverflow)?;

    Ok(MarketImpactComparisonAssessment {
        scenario_id: candidate.scenario_id,
        candidate_policy_id: candidate.policy_id,
        comparator_policy_id: comparator.policy_id,
        interval_count: candidate.interval_count,
        price_taker_revenue_delta_numerator_minor_zar: price_taker_delta,
        impact_adjusted_revenue_delta_numerator_minor_zar: impact_adjusted_delta,
        differential_price_impact_numerator_minor_zar: differential_impact,
        revenue_denominator_milli_wh_per_mwh: REVENUE_DENOMINATOR_MILLI_WH_PER_MWH,
        candidate_price_impact_adjustment_numerator_minor_zar: candidate
            .price_impact_adjustment_numerator_minor_zar,
        comparator_price_impact_adjustment_numerator_minor_zar: comparator
            .price_impact_adjustment_numerator_minor_zar,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const POLICY_SHA: &str =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn interval(
        interval_id: &'static str,
        energy_record_id: &'static str,
        impact_record_id: &'static str,
        energy_milli_wh: i64,
        reference_price: i64,
        impact_price: i64,
    ) -> MarketImpactInterval<'static> {
        MarketImpactInterval {
            interval_id,
            interval_start_utc: 0,
            interval_end_utc: 900,
            net_energy_milli_wh: energy_milli_wh,
            energy_record_id,
            energy_source_ref: energy_record_id,
            energy_source_sha256: SOURCE_SHA,
            energy_available_at_utc: 1000,
            reference_price_minor_zar_per_mwh: reference_price,
            reference_price_record_id: "reference-price/interval-1",
            reference_price_source_ref: "market/reference-series",
            reference_price_sha256: SOURCE_SHA,
            reference_price_available_at_utc: 1000,
            impact_adjusted_price_minor_zar_per_mwh: impact_price,
            impact_price_record_id: impact_record_id,
            impact_price_source_ref: impact_record_id,
            impact_price_sha256: SOURCE_SHA,
            impact_price_available_at_utc: 1000,
        }
    }

    fn inputs<'a>(
        policy_id: &'a str,
        policy_sha: &'a str,
        intervals: &'a [MarketImpactInterval<'a>],
    ) -> MarketImpactPolicyInputs<'a> {
        MarketImpactPolicyInputs {
            scenario_id: "scenario/price-impact/base",
            evaluation_as_of_utc: 2000,
            horizon_start_utc: 0,
            horizon_end_utc: 900,
            policy_id,
            policy_sha256: policy_sha,
            market_impact_model_id: "market-sim-v1",
            market_impact_model_sha256: SOURCE_SHA,
            reference_price_basis_id: "reference-market/v1",
            reference_price_basis_sha256: SOURCE_SHA,
            intervals,
        }
    }

    #[test]
    fn price_impact_adjusts_policy_revenue_with_exact_units() {
        let candidate_intervals = [interval(
            "TP-001",
            "candidate-energy/001",
            "candidate-impact/001",
            1_000_000_000,
            10_000,
            9_000,
        )];
        let result = evaluate_market_impact_policy(inputs(
            "candidate",
            POLICY_SHA,
            &candidate_intervals,
        ))
        .unwrap();
        assert_eq!(result.price_taker_revenue_numerator_minor_zar, 10_000_000_000_000);
        assert_eq!(result.impact_adjusted_revenue_numerator_minor_zar, 9_000_000_000_000);
        assert_eq!(result.price_impact_adjustment_numerator_minor_zar, -1_000_000_000_000);
        assert_eq!(result.revenue_denominator_milli_wh_per_mwh, 1_000_000_000);
    }

    #[test]
    fn paired_policy_comparison_requires_the_same_reference_price_path() {
        let candidate = [interval(
            "TP-001",
            "candidate-energy/001",
            "candidate-impact/001",
            1_000_000_000,
            10_000,
            9_000,
        )];
        let comparator = [interval(
            "TP-001",
            "comparator-energy/001",
            "comparator-impact/001",
            500_000_000,
            10_000,
            9_500,
        )];
        let result = compare_market_impact_policies(
            inputs("candidate", POLICY_SHA, &candidate),
            inputs("comparator", "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210", &comparator),
        )
        .unwrap();
        assert_eq!(
            result.price_taker_revenue_delta_numerator_minor_zar,
            5_000_000_000_000
        );
        assert_eq!(
            result.impact_adjusted_revenue_delta_numerator_minor_zar,
            4_250_000_000_000
        );
        assert_eq!(
            result.differential_price_impact_numerator_minor_zar,
            -750_000_000_000
        );
    }

    #[test]
    fn charging_at_negative_prices_uses_consistent_signed_cashflow() {
        let intervals = [interval(
            "TP-001",
            "charge-energy/001",
            "charge-impact/001",
            -1_000_000_000,
            -5_000,
            -4_000,
        )];
        let result =
            evaluate_market_impact_policy(inputs("charging-policy", POLICY_SHA, &intervals))
                .unwrap();
        assert_eq!(result.price_taker_revenue_numerator_minor_zar, 5_000_000_000_000);
        assert_eq!(result.impact_adjusted_revenue_numerator_minor_zar, 4_000_000_000_000);
        assert_eq!(result.price_impact_adjustment_numerator_minor_zar, -1_000_000_000_000);
    }

    #[test]
    fn overlapping_gap_future_known_and_duplicate_interval_data_fail_closed() {
        let overlap = [
            interval("TP-001", "energy/001", "impact/001", 1, 10, 9),
            MarketImpactInterval {
                interval_id: "TP-002",
                interval_start_utc: 800,
                interval_end_utc: 900,
                energy_record_id: "energy/002",
                energy_source_ref: "energy/002",
                impact_price_record_id: "impact/002",
                impact_price_source_ref: "impact/002",
                ..interval("TP-002", "energy/002", "impact/002", 1, 10, 9)
            },
        ];
        assert_eq!(
            evaluate_market_impact_policy(inputs("candidate", POLICY_SHA, &overlap)),
            Err(MarketImpactError::OverlappingIntervals)
        );

        let gap = [MarketImpactInterval {
            interval_start_utc: 100,
            ..interval("TP-001", "energy/003", "impact/003", 1, 10, 9)
        }];
        assert_eq!(
            evaluate_market_impact_policy(inputs("candidate", POLICY_SHA, &gap)),
            Err(MarketImpactError::IncompleteHorizonCoverage)
        );

        let future = [MarketImpactInterval {
            impact_price_available_at_utc: 2_001,
            ..interval("TP-001", "energy/004", "impact/004", 1, 10, 9)
        }];
        assert_eq!(
            evaluate_market_impact_policy(inputs("candidate", POLICY_SHA, &future)),
            Err(MarketImpactError::FutureKnownEvidence)
        );
    }

    #[test]
    fn matched_reference_price_path_requires_same_availability_vintage() {
        let candidate = [interval(
            "TP-001",
            "candidate-energy/001",
            "candidate-impact/001",
            1_000_000_000,
            10_000,
            9_000,
        )];
        let comparator = [MarketImpactInterval {
            reference_price_available_at_utc: 1_001,
            ..interval(
                "TP-001",
                "comparator-energy/001",
                "comparator-impact/001",
                500_000_000,
                10_000,
                9_500,
            )
        }];
        assert_eq!(
            compare_market_impact_policies(
                inputs("candidate", POLICY_SHA, &candidate),
                inputs(
                    "comparator",
                    "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
                    &comparator,
                ),
            ),
            Err(MarketImpactError::ReferencePricePathMismatch)
        );
    }

    #[test]
    fn same_policy_and_mismatched_simulator_builds_are_rejected() {
        let candidate = [interval(
            "TP-001",
            "candidate-energy/001",
            "candidate-impact/001",
            1_000_000_000,
            10_000,
            9_000,
        )];
        let comparator = [interval(
            "TP-001",
            "comparator-energy/001",
            "comparator-impact/001",
            1_000_000_000,
            10_000,
            9_000,
        )];
        let same_policy = inputs("candidate", POLICY_SHA, &candidate);
        assert_eq!(
            compare_market_impact_policies(same_policy, same_policy),
            Err(MarketImpactError::PolicyIdentityNotDistinct)
        );
        let different_model = MarketImpactPolicyInputs {
            market_impact_model_id: "different-build",
            ..inputs("comparator", POLICY_SHA, &comparator)
        };
        assert_eq!(
            compare_market_impact_policies(
                inputs("candidate", POLICY_SHA, &candidate),
                different_model
            ),
            Err(MarketImpactError::MarketImpactModelMismatch)
        );
    }
}
