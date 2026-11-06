//! Source-qualified warranty-envelope compliance over measured operation.
//!
//! This kernel evaluates only limits supplied by an identified warranty
//! schedule. It does not infer chemistry-specific degradation coefficients,
//! warranty exclusions, annual-cycle definitions or replacement economics.
//! Throughput accounting basis and warranty-period boundaries must be explicit
//! source-bound inputs. Missing telemetry never becomes a compliance pass.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarrantyThroughputBasis {
    ChargeOnly,
    DischargeOnly,
    ChargePlusDischarge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarrantyEnvelope<'a> {
    pub warranty_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub effective_from_utc: i64,
    pub effective_until_utc: i64,
    pub minimum_soc_bps: Option<u16>,
    pub maximum_soc_bps: Option<u16>,
    pub maximum_cell_temperature_milli_c: Option<i32>,
    pub maximum_charge_power_w: Option<u64>,
    pub maximum_discharge_power_w: Option<u64>,
    pub throughput_basis: Option<WarrantyThroughputBasis>,
    pub maximum_cumulative_throughput_wh: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarrantyPeriodBudget<'a> {
    pub period_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub maximum_throughput_wh: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarrantyOperatingSample<'a> {
    pub sample_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub warranty_period_id: &'a str,
    pub interval_start_utc: i64,
    pub interval_end_utc: i64,
    pub observed_at_utc: i64,
    pub available_at_utc: i64,
    pub minimum_soc_bps: Option<u16>,
    pub maximum_soc_bps: Option<u16>,
    pub maximum_cell_temperature_milli_c: Option<i32>,
    pub maximum_charge_power_w: Option<u64>,
    pub maximum_discharge_power_w: Option<u64>,
    pub charge_energy_wh: Option<u64>,
    pub discharge_energy_wh: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarrantyComplianceInputs<'a> {
    pub scenario_id: &'a str,
    pub evaluation_as_of_utc: i64,
    pub horizon_start_utc: i64,
    pub horizon_end_utc: i64,
    /// The calendar binds the warranty-period IDs to civil dates and timezone.
    pub period_calendar_source_ref: &'a str,
    pub period_calendar_source_sha256: &'a str,
    pub period_calendar_timezone: &'a str,
    pub envelope: WarrantyEnvelope<'a>,
    pub period_budgets: &'a [WarrantyPeriodBudget<'a>],
    pub samples: &'a [WarrantyOperatingSample<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarrantyComplianceError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidWarrantyTerms,
    InvalidHorizon,
    InvalidPeriodCalendar,
    InvalidPeriodOrdering,
    InvalidSample,
    FutureKnownSample,
    SampleOutsideWarrantyPeriod,
    SamplesOutOfOrder,
    OverlappingSamples,
    DuplicateSampleId,
    DuplicateSampleSourceReference,
    DuplicateBudgetPeriod,
    UnknownWarrantyPeriod,
    ArithmeticOverflow,
    EmptySampleSet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarrantyMetricStatus {
    NotSpecified,
    WithinSuppliedLimit,
    ExceededSuppliedLimit,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarrantyOverallStatus {
    WithinSuppliedLimits,
    ExceededSuppliedLimits,
    Indeterminate,
    NoSuppliedLimits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarrantyMetricAssessment {
    pub status: WarrantyMetricStatus,
    pub checked_sample_count: u64,
    pub missing_sample_count: u64,
    pub violation_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarrantyComplianceAssessment<'a> {
    pub scenario_id: &'a str,
    pub warranty_id: &'a str,
    pub evaluation_as_of_utc: i64,
    pub overall_status: WarrantyOverallStatus,
    pub sample_count: u64,
    pub covered_seconds: u64,
    pub uncovered_seconds: u64,
    pub state_of_charge: WarrantyMetricAssessment,
    pub cell_temperature: WarrantyMetricAssessment,
    pub charge_power: WarrantyMetricAssessment,
    pub discharge_power: WarrantyMetricAssessment,
    pub cumulative_throughput: WarrantyMetricAssessment,
    pub period_throughput: WarrantyMetricAssessment,
    pub measured_charge_energy_wh: Option<u128>,
    pub measured_discharge_energy_wh: Option<u128>,
    pub measured_warranty_throughput_wh: Option<u128>,
    pub period_budget_exceeded_count: u64,
    pub period_budget_indeterminate_count: u64,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn metric_status(
    specified: bool,
    checked: u64,
    missing: u64,
    violations: u64,
) -> WarrantyMetricStatus {
    if !specified {
        WarrantyMetricStatus::NotSpecified
    } else if violations > 0 {
        WarrantyMetricStatus::ExceededSuppliedLimit
    } else if missing > 0 {
        WarrantyMetricStatus::Indeterminate
    } else {
        WarrantyMetricStatus::WithinSuppliedLimit
    }
}

fn checked_add_u128(left: u128, right: u128) -> Result<u128, WarrantyComplianceError> {
    left.checked_add(right)
        .ok_or(WarrantyComplianceError::ArithmeticOverflow)
}

fn validate_envelope(envelope: WarrantyEnvelope<'_>) -> Result<(), WarrantyComplianceError> {
    if !nonempty(envelope.warranty_id) || !nonempty(envelope.source_ref) {
        return Err(WarrantyComplianceError::InvalidEvidenceReference);
    }
    if !valid_sha256(envelope.source_sha256) {
        return Err(WarrantyComplianceError::InvalidSourceDigest);
    }
    if envelope.effective_from_utc >= envelope.effective_until_utc {
        return Err(WarrantyComplianceError::InvalidWarrantyTerms);
    }
    if envelope.minimum_soc_bps.is_some_and(|value| value > 10_000)
        || envelope.maximum_soc_bps.is_some_and(|value| value > 10_000)
        || matches!(
            (envelope.minimum_soc_bps, envelope.maximum_soc_bps),
            (Some(minimum), Some(maximum)) if minimum > maximum
        )
        || envelope.maximum_cell_temperature_milli_c.is_some_and(|value| value < -273_150)
    {
        return Err(WarrantyComplianceError::InvalidWarrantyTerms);
    }
    if envelope.maximum_cumulative_throughput_wh.is_some()
        && envelope.throughput_basis.is_none()
    {
        return Err(WarrantyComplianceError::InvalidWarrantyTerms);
    }
    Ok(())
}

/// Evaluate measured operating samples against the exact warranty terms
/// supplied by the caller. A "within limits" result means only that the
/// supplied limits were not breached over complete, covered sample intervals.
pub fn evaluate_warranty_compliance<'a>(
    inputs: WarrantyComplianceInputs<'a>,
) -> Result<WarrantyComplianceAssessment<'a>, WarrantyComplianceError> {
    let envelope = inputs.envelope;
    if !nonempty(inputs.scenario_id)
        || !nonempty(inputs.period_calendar_source_ref)
        || !nonempty(inputs.period_calendar_timezone)
    {
        return Err(WarrantyComplianceError::InvalidEvidenceReference);
    }
    if !valid_sha256(inputs.period_calendar_source_sha256) {
        return Err(WarrantyComplianceError::InvalidSourceDigest);
    }
    validate_envelope(envelope)?;

    if inputs.horizon_start_utc >= inputs.horizon_end_utc
        || inputs.evaluation_as_of_utc < inputs.horizon_end_utc
        || inputs.horizon_start_utc < envelope.effective_from_utc
        || inputs.horizon_end_utc > envelope.effective_until_utc
    {
        return Err(WarrantyComplianceError::InvalidHorizon);
    }
    if inputs.samples.is_empty() {
        return Err(WarrantyComplianceError::EmptySampleSet);
    }
    for budget in inputs.period_budgets {
        if !nonempty(budget.period_id) || !nonempty(budget.source_ref) {
            return Err(WarrantyComplianceError::InvalidEvidenceReference);
        }
        if !valid_sha256(budget.source_sha256) {
            return Err(WarrantyComplianceError::InvalidSourceDigest);
        }
    }
    for (index, budget) in inputs.period_budgets.iter().enumerate() {
        if inputs.period_budgets[..index]
            .iter()
            .any(|previous| previous.period_id == budget.period_id)
        {
            return Err(WarrantyComplianceError::DuplicateBudgetPeriod);
        }
    }
    if !inputs.period_budgets.is_empty() && envelope.throughput_basis.is_none() {
        return Err(WarrantyComplianceError::InvalidWarrantyTerms);
    }

    let mut covered_seconds = 0_u64;
    let mut uncovered_seconds = 0_u64;
    let mut previous_end = inputs.horizon_start_utc;
    let mut previous_start = None;
    let mut ids = std::collections::BTreeSet::new();
    let mut sources = std::collections::BTreeSet::new();

    let mut soc_checked = 0_u64;
    let mut soc_missing = 0_u64;
    let mut soc_violations = 0_u64;
    let mut temperature_checked = 0_u64;
    let mut temperature_missing = 0_u64;
    let mut temperature_violations = 0_u64;
    let mut charge_power_checked = 0_u64;
    let mut charge_power_missing = 0_u64;
    let mut charge_power_violations = 0_u64;
    let mut discharge_power_checked = 0_u64;
    let mut discharge_power_missing = 0_u64;
    let mut discharge_power_violations = 0_u64;

    let mut charge_energy_sum = 0_u128;
    let mut discharge_energy_sum = 0_u128;
    let mut charge_energy_complete = true;
    let mut discharge_energy_complete = true;
    let mut throughput_complete = true;
    let mut period_energy = std::collections::BTreeMap::<&str, u128>::new();
    let mut period_energy_complete = std::collections::BTreeMap::<&str, bool>::new();

    for sample in inputs.samples.iter().copied() {
        if !nonempty(sample.sample_id)
            || !nonempty(sample.source_ref)
            || !nonempty(sample.warranty_period_id)
        {
            return Err(WarrantyComplianceError::InvalidEvidenceReference);
        }
        if !valid_sha256(sample.source_sha256) {
            return Err(WarrantyComplianceError::InvalidSourceDigest);
        }
        if sample.interval_start_utc < inputs.horizon_start_utc
            || sample.interval_end_utc > inputs.horizon_end_utc
            || sample.interval_start_utc >= sample.interval_end_utc
            || sample.observed_at_utc < sample.interval_end_utc
            || sample.observed_at_utc > sample.available_at_utc
            || sample.minimum_soc_bps.is_some_and(|value| value > 10_000)
            || sample.maximum_soc_bps.is_some_and(|value| value > 10_000)
            || matches!(
                (sample.minimum_soc_bps, sample.maximum_soc_bps),
                (Some(minimum), Some(maximum)) if minimum > maximum
            )
            || sample
                .maximum_cell_temperature_milli_c
                .is_some_and(|value| value < -273_150)
        {
            return Err(WarrantyComplianceError::InvalidSample);
        }
        if sample.available_at_utc > inputs.evaluation_as_of_utc {
            return Err(WarrantyComplianceError::FutureKnownSample);
        }
        if sample.interval_start_utc < envelope.effective_from_utc
            || sample.interval_end_utc > envelope.effective_until_utc
        {
            return Err(WarrantyComplianceError::SampleOutsideWarrantyPeriod);
        }
        if !inputs.period_budgets.is_empty()
            && !inputs
                .period_budgets
                .iter()
                .any(|budget| budget.period_id == sample.warranty_period_id)
        {
            return Err(WarrantyComplianceError::UnknownWarrantyPeriod);
        }
        if !ids.insert(sample.sample_id) {
            return Err(WarrantyComplianceError::DuplicateSampleId);
        }
        if !sources.insert(sample.source_ref) {
            return Err(WarrantyComplianceError::DuplicateSampleSourceReference);
        }
        if previous_start.is_some_and(|start| sample.interval_start_utc <= start) {
            return Err(WarrantyComplianceError::SamplesOutOfOrder);
        }
        if sample.interval_start_utc < previous_end {
            return Err(WarrantyComplianceError::OverlappingSamples);
        }
        let gap = sample
            .interval_start_utc
            .checked_sub(previous_end)
            .ok_or(WarrantyComplianceError::ArithmeticOverflow)?;
        uncovered_seconds = uncovered_seconds
            .checked_add(u64::try_from(gap).map_err(|_| WarrantyComplianceError::ArithmeticOverflow)?)
            .ok_or(WarrantyComplianceError::ArithmeticOverflow)?;
        let duration = sample
            .interval_end_utc
            .checked_sub(sample.interval_start_utc)
            .ok_or(WarrantyComplianceError::ArithmeticOverflow)?;
        let duration_u64 =
            u64::try_from(duration).map_err(|_| WarrantyComplianceError::ArithmeticOverflow)?;
        covered_seconds = covered_seconds
            .checked_add(duration_u64)
            .ok_or(WarrantyComplianceError::ArithmeticOverflow)?;

        if envelope.minimum_soc_bps.is_some() || envelope.maximum_soc_bps.is_some() {
            match (
                envelope.minimum_soc_bps,
                envelope.maximum_soc_bps,
                sample.minimum_soc_bps,
                sample.maximum_soc_bps,
            ) {
                (Some(minimum), _, Some(observed_minimum), _) => {
                    soc_checked = soc_checked.saturating_add(1);
                    if observed_minimum < minimum {
                        soc_violations = soc_violations.saturating_add(1);
                    }
                }
                (Some(_), _, None, _) => soc_missing = soc_missing.saturating_add(1),
                _ => {}
            }
            match (
                envelope.maximum_soc_bps,
                sample.maximum_soc_bps,
            ) {
                (Some(maximum), Some(observed_maximum)) => {
                    soc_checked = soc_checked.saturating_add(1);
                    if observed_maximum > maximum {
                        soc_violations = soc_violations.saturating_add(1);
                    }
                }
                (Some(_), None) => soc_missing = soc_missing.saturating_add(1),
                _ => {}
            }
        }

        if let Some(maximum) = envelope.maximum_cell_temperature_milli_c {
            match sample.maximum_cell_temperature_milli_c {
                Some(observed) => {
                    temperature_checked = temperature_checked.saturating_add(1);
                    if observed > maximum {
                        temperature_violations = temperature_violations.saturating_add(1);
                    }
                }
                None => temperature_missing = temperature_missing.saturating_add(1),
            }
        }
        if let Some(maximum) = envelope.maximum_charge_power_w {
            match sample.maximum_charge_power_w {
                Some(observed) => {
                    charge_power_checked = charge_power_checked.saturating_add(1);
                    if observed > maximum {
                        charge_power_violations = charge_power_violations.saturating_add(1);
                    }
                }
                None => charge_power_missing = charge_power_missing.saturating_add(1),
            }
        }
        if let Some(maximum) = envelope.maximum_discharge_power_w {
            match sample.maximum_discharge_power_w {
                Some(observed) => {
                    discharge_power_checked = discharge_power_checked.saturating_add(1);
                    if observed > maximum {
                        discharge_power_violations = discharge_power_violations.saturating_add(1);
                    }
                }
                None => discharge_power_missing = discharge_power_missing.saturating_add(1),
            }
        }

        if let Some(basis) = envelope.throughput_basis {
            let required_charge = matches!(
                basis,
                WarrantyThroughputBasis::ChargeOnly | WarrantyThroughputBasis::ChargePlusDischarge
            );
            let required_discharge = matches!(
                basis,
                WarrantyThroughputBasis::DischargeOnly | WarrantyThroughputBasis::ChargePlusDischarge
            );
            if sample.charge_energy_wh.is_none() {
                charge_energy_complete = false;
            }
            if sample.discharge_energy_wh.is_none() {
                discharge_energy_complete = false;
            }
            if (required_charge && sample.charge_energy_wh.is_none())
                || (required_discharge && sample.discharge_energy_wh.is_none())
            {
                throughput_complete = false;
                period_energy_complete.insert(sample.warranty_period_id, false);
            }
            let charge = sample.charge_energy_wh.unwrap_or(0);
            let discharge = sample.discharge_energy_wh.unwrap_or(0);
            charge_energy_sum = checked_add_u128(charge_energy_sum, u128::from(charge))?;
            discharge_energy_sum = checked_add_u128(discharge_energy_sum, u128::from(discharge))?;
            let selected_throughput = match basis {
                WarrantyThroughputBasis::ChargeOnly => u128::from(charge),
                WarrantyThroughputBasis::DischargeOnly => u128::from(discharge),
                WarrantyThroughputBasis::ChargePlusDischarge => {
                    u128::from(charge).checked_add(u128::from(discharge))
                        .ok_or(WarrantyComplianceError::ArithmeticOverflow)?
                }
            };
            let previous = period_energy.get(sample.warranty_period_id).copied().unwrap_or(0);
            period_energy.insert(
                sample.warranty_period_id,
                checked_add_u128(previous, selected_throughput)?,
            );
            period_energy_complete.entry(sample.warranty_period_id).or_insert(true);
        }

        previous_start = Some(sample.interval_start_utc);
        previous_end = sample.interval_end_utc;
    }

    let trailing_gap = inputs
        .horizon_end_utc
        .checked_sub(previous_end)
        .ok_or(WarrantyComplianceError::ArithmeticOverflow)?;
    uncovered_seconds = uncovered_seconds
        .checked_add(u64::try_from(trailing_gap).map_err(|_| WarrantyComplianceError::ArithmeticOverflow)?)
        .ok_or(WarrantyComplianceError::ArithmeticOverflow)?;

    let soc = WarrantyMetricAssessment {
        status: metric_status(
            envelope.minimum_soc_bps.is_some() || envelope.maximum_soc_bps.is_some(),
            soc_checked,
            soc_missing.saturating_add(u64::from(uncovered_seconds > 0)),
            soc_violations,
        ),
        checked_sample_count: soc_checked,
        missing_sample_count: soc_missing.saturating_add(u64::from(uncovered_seconds > 0)),
        violation_count: soc_violations,
    };
    let temperature = WarrantyMetricAssessment {
        status: metric_status(
            envelope.maximum_cell_temperature_milli_c.is_some(),
            temperature_checked,
            temperature_missing.saturating_add(u64::from(uncovered_seconds > 0)),
            temperature_violations,
        ),
        checked_sample_count: temperature_checked,
        missing_sample_count: temperature_missing.saturating_add(u64::from(uncovered_seconds > 0)),
        violation_count: temperature_violations,
    };
    let charge_power = WarrantyMetricAssessment {
        status: metric_status(
            envelope.maximum_charge_power_w.is_some(),
            charge_power_checked,
            charge_power_missing.saturating_add(u64::from(uncovered_seconds > 0)),
            charge_power_violations,
        ),
        checked_sample_count: charge_power_checked,
        missing_sample_count: charge_power_missing.saturating_add(u64::from(uncovered_seconds > 0)),
        violation_count: charge_power_violations,
    };
    let discharge_power = WarrantyMetricAssessment {
        status: metric_status(
            envelope.maximum_discharge_power_w.is_some(),
            discharge_power_checked,
            discharge_power_missing.saturating_add(u64::from(uncovered_seconds > 0)),
            discharge_power_violations,
        ),
        checked_sample_count: discharge_power_checked,
        missing_sample_count: discharge_power_missing.saturating_add(u64::from(uncovered_seconds > 0)),
        violation_count: discharge_power_violations,
    };

    let throughput_specified = envelope.throughput_basis.is_some();
    let measured_throughput = if throughput_specified && throughput_complete {
        Some(match envelope.throughput_basis {
            Some(WarrantyThroughputBasis::ChargeOnly) => charge_energy_sum,
            Some(WarrantyThroughputBasis::DischargeOnly) => discharge_energy_sum,
            Some(WarrantyThroughputBasis::ChargePlusDischarge) => {
                checked_add_u128(charge_energy_sum, discharge_energy_sum)?
            }
            None => 0,
        })
    } else {
        None
    };
    let cumulative_violations = match (
        envelope.maximum_cumulative_throughput_wh,
        measured_throughput,
    ) {
        (Some(maximum), Some(observed)) if observed > u128::from(maximum) => 1,
        _ => 0,
    };
    let cumulative_missing = u64::from(
        envelope.maximum_cumulative_throughput_wh.is_some()
            && (measured_throughput.is_none() || uncovered_seconds > 0),
    );
    let cumulative = WarrantyMetricAssessment {
        status: metric_status(
            envelope.maximum_cumulative_throughput_wh.is_some(),
            u64::from(measured_throughput.is_some()),
            cumulative_missing,
            cumulative_violations,
        ),
        checked_sample_count: u64::from(measured_throughput.is_some()),
        missing_sample_count: cumulative_missing,
        violation_count: cumulative_violations,
    };

    let mut period_budget_exceeded_count = 0_u64;
    let mut period_budget_indeterminate_count = 0_u64;
    for budget in inputs.period_budgets {
        let observed = period_energy.get(budget.period_id).copied();
        let complete = period_energy_complete.get(budget.period_id).copied().unwrap_or(false)
            && uncovered_seconds == 0;
        match (observed, complete) {
            (Some(value), true) if value > u128::from(budget.maximum_throughput_wh) => {
                period_budget_exceeded_count = period_budget_exceeded_count.saturating_add(1);
            }
            (Some(_), true) => {}
            _ => period_budget_indeterminate_count =
                period_budget_indeterminate_count.saturating_add(1),
        }
    }
    let period_throughput = WarrantyMetricAssessment {
        status: if inputs.period_budgets.is_empty() {
            WarrantyMetricStatus::NotSpecified
        } else if period_budget_exceeded_count > 0 {
            WarrantyMetricStatus::ExceededSuppliedLimit
        } else if period_budget_indeterminate_count > 0 {
            WarrantyMetricStatus::Indeterminate
        } else {
            WarrantyMetricStatus::WithinSuppliedLimit
        },
        checked_sample_count: u64::try_from(inputs.period_budgets.len())
            .map_err(|_| WarrantyComplianceError::ArithmeticOverflow)?
            .saturating_sub(period_budget_indeterminate_count),
        missing_sample_count: period_budget_indeterminate_count,
        violation_count: period_budget_exceeded_count,
    };

    let metrics = [soc, temperature, charge_power, discharge_power, cumulative, period_throughput];
    let has_limit = metrics
        .iter()
        .any(|metric| metric.status != WarrantyMetricStatus::NotSpecified);
    let has_breach = metrics
        .iter()
        .any(|metric| metric.status == WarrantyMetricStatus::ExceededSuppliedLimit);
    let is_indeterminate = uncovered_seconds > 0
        || metrics
            .iter()
            .any(|metric| metric.status == WarrantyMetricStatus::Indeterminate);
    let overall_status = if has_breach {
        WarrantyOverallStatus::ExceededSuppliedLimits
    } else if is_indeterminate {
        WarrantyOverallStatus::Indeterminate
    } else if !has_limit {
        WarrantyOverallStatus::NoSuppliedLimits
    } else {
        WarrantyOverallStatus::WithinSuppliedLimits
    };

    Ok(WarrantyComplianceAssessment {
        scenario_id: inputs.scenario_id,
        warranty_id: envelope.warranty_id,
        evaluation_as_of_utc: inputs.evaluation_as_of_utc,
        overall_status,
        sample_count: u64::try_from(inputs.samples.len())
            .map_err(|_| WarrantyComplianceError::ArithmeticOverflow)?,
        covered_seconds,
        uncovered_seconds,
        state_of_charge: soc,
        cell_temperature: temperature,
        charge_power,
        discharge_power,
        cumulative_throughput: cumulative,
        period_throughput,
        measured_charge_energy_wh: (throughput_specified && charge_energy_complete)
            .then_some(charge_energy_sum),
        measured_discharge_energy_wh: (throughput_specified && discharge_energy_complete)
            .then_some(discharge_energy_sum),
        measured_warranty_throughput_wh: measured_throughput,
        period_budget_exceeded_count,
        period_budget_indeterminate_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn envelope() -> WarrantyEnvelope<'static> {
        WarrantyEnvelope {
            warranty_id: "warranty/bess-001",
            source_ref: "contract/warranty/001",
            source_sha256: SOURCE_SHA,
            effective_from_utc: 0,
            effective_until_utc: 10_000,
            minimum_soc_bps: Some(1_000),
            maximum_soc_bps: Some(9_000),
            maximum_cell_temperature_milli_c: Some(45_000),
            maximum_charge_power_w: Some(50_000_000),
            maximum_discharge_power_w: Some(60_000_000),
            throughput_basis: Some(WarrantyThroughputBasis::ChargePlusDischarge),
            maximum_cumulative_throughput_wh: Some(30_000),
        }
    }

    fn sample(
        id: &'static str,
        start: i64,
        end: i64,
        charge: u64,
        discharge: u64,
    ) -> WarrantyOperatingSample<'static> {
        WarrantyOperatingSample {
            sample_id: id,
            source_ref: id,
            source_sha256: SOURCE_SHA,
            warranty_period_id: "warranty-year-1",
            interval_start_utc: start,
            interval_end_utc: end,
            observed_at_utc: end + 1,
            available_at_utc: end + 2,
            minimum_soc_bps: Some(2_000),
            maximum_soc_bps: Some(8_000),
            maximum_cell_temperature_milli_c: Some(35_000),
            maximum_charge_power_w: Some(40_000_000),
            maximum_discharge_power_w: Some(50_000_000),
            charge_energy_wh: Some(charge),
            discharge_energy_wh: Some(discharge),
        }
    }

    fn inputs<'a>(
        samples: &'a [WarrantyOperatingSample<'a>],
        budgets: &'a [WarrantyPeriodBudget<'a>],
    ) -> WarrantyComplianceInputs<'a> {
        WarrantyComplianceInputs {
            scenario_id: "battery-001/warranty-replay",
            evaluation_as_of_utc: 1_000,
            horizon_start_utc: 100,
            horizon_end_utc: 300,
            period_calendar_source_ref: "calendar/warranty/v1",
            period_calendar_source_sha256: SOURCE_SHA,
            period_calendar_timezone: "Africa/Johannesburg",
            envelope: envelope(),
            period_budgets: budgets,
            samples,
        }
    }

    #[test]
    fn within_supplied_terms_requires_complete_telemetry_and_period_budget() {
        let samples = [
            sample("sample-1", 100, 200, 10_000, 5_000),
            sample("sample-2", 200, 300, 5_000, 5_000),
        ];
        let budgets = [WarrantyPeriodBudget {
            period_id: "warranty-year-1",
            source_ref: "contract/warranty/period-1",
            source_sha256: SOURCE_SHA,
            maximum_throughput_wh: 30_000,
        }];
        let result = evaluate_warranty_compliance(inputs(&samples, &budgets)).unwrap();
        assert_eq!(result.overall_status, WarrantyOverallStatus::WithinSuppliedLimits);
        assert_eq!(result.covered_seconds, 200);
        assert_eq!(result.uncovered_seconds, 0);
        assert_eq!(result.measured_warranty_throughput_wh, Some(25_000));
        assert_eq!(result.period_budget_exceeded_count, 0);
        assert_eq!(
            result.cumulative_throughput.status,
            WarrantyMetricStatus::WithinSuppliedLimit
        );
    }

    #[test]
    fn soc_temperature_power_and_throughput_breaches_are_reported() {
        let breached = [WarrantyOperatingSample {
            minimum_soc_bps: Some(500),
            maximum_soc_bps: Some(9_500),
            maximum_cell_temperature_milli_c: Some(50_000),
            maximum_charge_power_w: Some(55_000_000),
            maximum_discharge_power_w: Some(65_000_000),
            charge_energy_wh: Some(20_000),
            discharge_energy_wh: Some(20_000),
            ..sample("breach", 100, 300, 0, 0)
        }];
        let result = evaluate_warranty_compliance(inputs(&breached, &[])).unwrap();
        assert_eq!(result.overall_status, WarrantyOverallStatus::ExceededSuppliedLimits);
        assert_eq!(result.state_of_charge.violation_count, 2);
        assert_eq!(result.cell_temperature.violation_count, 1);
        assert_eq!(result.charge_power.violation_count, 1);
        assert_eq!(result.discharge_power.violation_count, 1);
        assert_eq!(result.cumulative_throughput.violation_count, 1);
    }

    #[test]
    fn missing_required_telemetry_or_a_horizon_gap_is_indeterminate() {
        let incomplete = [WarrantyOperatingSample {
            maximum_cell_temperature_milli_c: None,
            charge_energy_wh: None,
            ..sample("incomplete", 100, 200, 10_000, 10_000)
        }];
        let budgets = [WarrantyPeriodBudget {
            period_id: "warranty-year-1",
            source_ref: "contract/warranty/period-1",
            source_sha256: SOURCE_SHA,
            maximum_throughput_wh: 100_000,
        }];
        let result = evaluate_warranty_compliance(inputs(&incomplete, &budgets)).unwrap();
        assert_eq!(result.overall_status, WarrantyOverallStatus::Indeterminate);
        assert_eq!(
            result.cell_temperature.status,
            WarrantyMetricStatus::Indeterminate
        );
        assert_eq!(
            result.cumulative_throughput.status,
            WarrantyMetricStatus::Indeterminate
        );
        assert_eq!(result.uncovered_seconds, 100);
    }

    #[test]
    fn future_known_and_pre_interval_observations_fail_closed() {
        let future = [WarrantyOperatingSample {
            available_at_utc: 1_001,
            ..sample("future", 100, 300, 10_000, 10_000)
        }];
        assert_eq!(
            evaluate_warranty_compliance(inputs(&future, &[])),
            Err(WarrantyComplianceError::FutureKnownSample)
        );

        let early = [WarrantyOperatingSample {
            observed_at_utc: 299,
            ..sample("early", 100, 300, 10_000, 10_000)
        }];
        assert_eq!(
            evaluate_warranty_compliance(inputs(&early, &[])),
            Err(WarrantyComplianceError::InvalidSample)
        );
    }

    #[test]
    fn duplicate_sources_unknown_periods_and_bad_digests_are_rejected() {
        let duplicate_source = [
            sample("one", 100, 200, 1_000, 1_000),
            WarrantyOperatingSample {
                sample_id: "two",
                interval_start_utc: 200,
                interval_end_utc: 300,
                observed_at_utc: 301,
                available_at_utc: 302,
                ..sample("two-template", 200, 300, 1_000, 1_000)
            },
        ];
        let same_source = [
            duplicate_source[0],
            WarrantyOperatingSample {
                source_ref: "one",
                ..duplicate_source[1]
            },
        ];
        assert_eq!(
            evaluate_warranty_compliance(inputs(&same_source, &[])),
            Err(WarrantyComplianceError::DuplicateSampleSourceReference)
        );

        let unknown_period = [WarrantyOperatingSample {
            warranty_period_id: "missing-period",
            ..sample("unknown", 100, 300, 1_000, 1_000)
        }];
        let budgets = [WarrantyPeriodBudget {
            period_id: "warranty-year-1",
            source_ref: "contract/warranty/period-1",
            source_sha256: SOURCE_SHA,
            maximum_throughput_wh: 100_000,
        }];
        assert_eq!(
            evaluate_warranty_compliance(inputs(&unknown_period, &budgets)),
            Err(WarrantyComplianceError::UnknownWarrantyPeriod)
        );

        let bad_digest = [WarrantyOperatingSample {
            source_sha256: "bad",
            ..sample("bad-digest", 100, 300, 1_000, 1_000)
        }];
        assert_eq!(
            evaluate_warranty_compliance(inputs(&bad_digest, &[])),
            Err(WarrantyComplianceError::InvalidSourceDigest)
        );
    }

    #[test]
    fn missing_throughput_basis_cannot_be_used_to_assert_a_budget() {
        let invalid = WarrantyEnvelope {
            throughput_basis: None,
            ..envelope()
        };
        let samples = [sample("sample", 100, 300, 1_000, 1_000)];
        let inputs = WarrantyComplianceInputs {
            envelope: invalid,
            ..inputs(&samples, &[])
        };
        assert_eq!(
            evaluate_warranty_compliance(inputs),
            Err(WarrantyComplianceError::InvalidWarrantyTerms)
        );
    }
}
