// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Time value of money, cash-flow analysis, and amortization.

use crate::error::{EconomicsError, Result, ensure_finite, ensure_rate, ensure_slice_finite};

fn checked_output(value: f64, context: &'static str) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EconomicsError::NumericalFailure { context })
    }
}

/// Future value of a present sum: `FV = PV·(1+r)^n`.
pub fn future_value(present: f64, rate: f64, periods: u32) -> Result<f64> {
    ensure_finite(present, "future-value principal")?;
    ensure_rate(rate, "future-value rate must be greater than -1")?;
    checked_output(
        present * (1.0 + rate).powi(periods as i32),
        "future-value calculation overflowed",
    )
}

/// Present value of a future sum: `PV = FV/(1+r)^n`.
pub fn present_value(future: f64, rate: f64, periods: u32) -> Result<f64> {
    ensure_finite(future, "present-value future amount")?;
    ensure_rate(rate, "present-value rate must be greater than -1")?;
    checked_output(
        future / (1.0 + rate).powi(periods as i32),
        "present-value calculation overflowed",
    )
}

/// Compound interest: `A = P·(1 + r/m)^(m·t)`.
pub fn compound_interest(
    principal: f64,
    annual_rate: f64,
    per_year: u32,
    years: f64,
) -> Result<f64> {
    ensure_finite(principal, "compound-interest principal")?;
    ensure_finite(annual_rate, "compound-interest annual rate")?;
    ensure_finite(years, "compound-interest years")?;
    if per_year == 0 {
        return Err(EconomicsError::InvalidParameter {
            context: "compounding periods per year must be positive",
        });
    }
    if years < 0.0 {
        return Err(EconomicsError::InvalidParameter {
            context: "years must be non-negative",
        });
    }
    let periods_per_year = per_year as f64;
    let periodic_rate = annual_rate / periods_per_year;
    ensure_rate(
        periodic_rate,
        "compound-interest periodic rate must be greater than -1",
    )?;
    checked_output(
        principal * (1.0 + periodic_rate).powf(periods_per_year * years),
        "compound-interest calculation overflowed",
    )
}

/// Convert a nominal annual rate into an effective annual rate.
pub fn effective_annual_rate(nominal_rate: f64, periods_per_year: u32) -> Result<f64> {
    ensure_finite(nominal_rate, "nominal annual rate")?;
    if periods_per_year == 0 {
        return Err(EconomicsError::InvalidParameter {
            context: "periods per year must be positive",
        });
    }
    let periods = periods_per_year as f64;
    let periodic_rate = nominal_rate / periods;
    ensure_rate(
        periodic_rate,
        "nominal rate implies a periodic rate at or below -100%",
    )?;
    checked_output(
        (periods * periodic_rate.ln_1p()).exp_m1(),
        "effective annual rate overflowed",
    )
}

/// Convert an effective annual rate into a nominal annual rate.
pub fn nominal_annual_rate(effective_rate: f64, periods_per_year: u32) -> Result<f64> {
    ensure_rate(
        effective_rate,
        "effective annual rate must be greater than -1",
    )?;
    if periods_per_year == 0 {
        return Err(EconomicsError::InvalidParameter {
            context: "periods per year must be positive",
        });
    }
    let periods = periods_per_year as f64;
    checked_output(
        periods * (effective_rate.ln_1p() / periods).exp_m1(),
        "nominal annual rate overflowed",
    )
}

/// Net present value of a cash-flow series (`cash_flows[0]` occurs at `t=0`).
pub fn npv(rate: f64, cash_flows: &[f64]) -> Result<f64> {
    ensure_rate(rate, "NPV discount rate must be greater than -1")?;
    if cash_flows.is_empty() {
        return Err(EconomicsError::EmptyInput {
            context: "NPV requires at least one cash flow",
        });
    }
    ensure_slice_finite(cash_flows, "NPV cash flows")?;
    npv_validated(rate, cash_flows)
}

fn npv_validated(rate: f64, cash_flows: &[f64]) -> Result<f64> {
    let growth = 1.0 + rate;
    let mut discount = 1.0;
    let mut sum = 0.0;
    let mut correction = 0.0;
    for &cash_flow in cash_flows {
        let term = cash_flow * discount;
        if !term.is_finite() {
            return Err(EconomicsError::NumericalFailure {
                context: "NPV term overflowed",
            });
        }
        // Neumaier compensated summation.
        let next = sum + term;
        if sum.abs() >= term.abs() {
            correction += (sum - next) + term;
        } else {
            correction += (term - next) + sum;
        }
        sum = next;
        discount /= growth;
    }
    checked_output(sum + correction, "NPV summation overflowed")
}

fn npv_and_derivative(rate: f64, cash_flows: &[f64]) -> Result<(f64, f64)> {
    let growth = 1.0 + rate;
    let mut discount = 1.0;
    let mut npv_sum = 0.0;
    let mut derivative = 0.0;
    for (period, &cash_flow) in cash_flows.iter().enumerate() {
        let term = cash_flow * discount;
        if !term.is_finite() {
            return Err(EconomicsError::NumericalFailure {
                context: "IRR scan overflowed",
            });
        }
        npv_sum += term;
        if period > 0 {
            derivative -= period as f64 * term / growth;
        }
        discount /= growth;
    }
    if npv_sum.is_finite() && derivative.is_finite() {
        Ok((npv_sum, derivative))
    } else {
        Err(EconomicsError::NumericalFailure {
            context: "IRR scan overflowed",
        })
    }
}

/// Configuration for bounded IRR root discovery.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IrrOptions {
    /// Lowest rate searched. Must be greater than `-1`.
    pub min_rate: f64,
    /// Highest rate searched.
    pub max_rate: f64,
    /// Number of logarithmically spaced scan intervals.
    pub scan_steps: usize,
    /// Absolute NPV tolerance after cash-flow normalization.
    pub tolerance: f64,
    /// Maximum bisection/Newton iterations per candidate.
    pub max_iterations: usize,
}

impl Default for IrrOptions {
    fn default() -> Self {
        Self {
            min_rate: -0.9999,
            max_rate: 1_000_000.0,
            scan_steps: 4096,
            tolerance: 1e-11,
            max_iterations: 200,
        }
    }
}

impl IrrOptions {
    fn validate(self) -> Result<()> {
        ensure_rate(self.min_rate, "IRR minimum rate must be greater than -1")?;
        ensure_finite(self.max_rate, "IRR maximum rate")?;
        ensure_finite(self.tolerance, "IRR tolerance")?;
        if self.max_rate <= self.min_rate {
            return Err(EconomicsError::InvalidParameter {
                context: "IRR maximum rate must exceed the minimum rate",
            });
        }
        if self.scan_steps < 16 {
            return Err(EconomicsError::InvalidParameter {
                context: "IRR scan requires at least 16 intervals",
            });
        }
        if self.tolerance <= 0.0 {
            return Err(EconomicsError::InvalidParameter {
                context: "IRR tolerance must be positive",
            });
        }
        if self.max_iterations == 0 {
            return Err(EconomicsError::InvalidParameter {
                context: "IRR iteration limit must be positive",
            });
        }
        Ok(())
    }
}

/// Classification of an IRR search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrrStatus {
    /// Every cash flow is zero, so every rate is a root.
    Indeterminate,
    /// No root was found in the configured search domain.
    NoRoot,
    /// Exactly one root was found.
    Unique,
    /// More than one root was found.
    Multiple,
}

/// Auditable result of scanning for all real IRRs in a bounded rate domain.
#[derive(Debug, Clone, PartialEq)]
pub struct IrrAnalysis {
    pub roots: Vec<f64>,
    pub cash_flow_sign_changes: usize,
    pub status: IrrStatus,
    /// False when part of the scan could not be evaluated without overflow.
    pub search_complete: bool,
    pub searched_rate_range: (f64, f64),
}

fn cash_flow_sign_changes(cash_flows: &[f64]) -> usize {
    let mut previous_sign: Option<bool> = None;
    let mut changes = 0;
    for &cash_flow in cash_flows {
        if cash_flow == 0.0 {
            continue;
        }
        let sign = cash_flow.is_sign_positive();
        if previous_sign.is_some_and(|previous| previous != sign) {
            changes += 1;
        }
        previous_sign = Some(sign);
    }
    changes
}

fn push_root(roots: &mut Vec<f64>, root: f64, tolerance: f64) {
    let duplicate = roots.iter().any(|existing| {
        (root - *existing).abs() <= tolerance * 100.0 * root.abs().max(existing.abs()).max(1.0)
    });
    if !duplicate {
        roots.push(root);
    }
}

fn bisect_value_root(
    mut lo: f64,
    mut hi: f64,
    mut f_lo: f64,
    cash_flows: &[f64],
    options: IrrOptions,
) -> Result<f64> {
    for _ in 0..options.max_iterations {
        let mid = lo + (hi - lo) * 0.5;
        let (f_mid, _) = npv_and_derivative(mid, cash_flows)?;
        if f_mid.abs() <= options.tolerance
            || (hi - lo).abs() <= options.tolerance * mid.abs().max(1.0)
        {
            return Ok(mid);
        }
        if f_lo.signum() == f_mid.signum() {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
    }
    Ok(lo + (hi - lo) * 0.5)
}

fn bisect_stationary_point(
    mut lo: f64,
    mut hi: f64,
    mut d_lo: f64,
    cash_flows: &[f64],
    options: IrrOptions,
) -> Result<f64> {
    for _ in 0..options.max_iterations {
        let mid = lo + (hi - lo) * 0.5;
        let (_, d_mid) = npv_and_derivative(mid, cash_flows)?;
        if d_mid.abs() <= options.tolerance
            || (hi - lo).abs() <= options.tolerance * mid.abs().max(1.0)
        {
            return Ok(mid);
        }
        if d_lo.signum() == d_mid.signum() {
            lo = mid;
            d_lo = d_mid;
        } else {
            hi = mid;
        }
    }
    Ok(lo + (hi - lo) * 0.5)
}

/// Discover IRR roots across a configurable bounded domain.
///
/// The scan is uniform in `ln(1 + rate)`, so it has useful resolution near
/// zero while still covering very high positive rates. Sign-changing roots are
/// refined by bisection. Stationary points are also tested, allowing repeated
/// roots that merely touch zero to be found.
pub fn irr_analysis(cash_flows: &[f64], options: IrrOptions) -> Result<IrrAnalysis> {
    options.validate()?;
    if cash_flows.is_empty() {
        return Err(EconomicsError::EmptyInput {
            context: "IRR requires at least one cash flow",
        });
    }
    ensure_slice_finite(cash_flows, "IRR cash flows")?;

    let scale = cash_flows
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let sign_changes = cash_flow_sign_changes(cash_flows);
    if scale == 0.0 {
        return Ok(IrrAnalysis {
            roots: Vec::new(),
            cash_flow_sign_changes: 0,
            status: IrrStatus::Indeterminate,
            search_complete: true,
            searched_rate_range: (options.min_rate, options.max_rate),
        });
    }

    let normalized: Vec<f64> = cash_flows.iter().map(|value| value / scale).collect();
    let min_log = options.min_rate.ln_1p();
    let max_log = options.max_rate.ln_1p();
    let mut roots = Vec::new();
    let mut search_complete = true;
    let mut previous: Option<(f64, f64, f64)> = None;

    for step in 0..=options.scan_steps {
        let fraction = step as f64 / options.scan_steps as f64;
        let rate = (min_log + (max_log - min_log) * fraction).exp_m1();
        let evaluation = npv_and_derivative(rate, &normalized);
        let (value, derivative) = match evaluation {
            Ok(result) => result,
            Err(EconomicsError::NumericalFailure { .. }) => {
                search_complete = false;
                previous = None;
                continue;
            }
            Err(error) => return Err(error),
        };

        if value.abs() <= options.tolerance {
            push_root(&mut roots, rate, options.tolerance);
        }

        if let Some((previous_rate, previous_value, previous_derivative)) = previous {
            if previous_value.signum() != value.signum() {
                let root =
                    bisect_value_root(previous_rate, rate, previous_value, &normalized, options)?;
                push_root(&mut roots, root, options.tolerance);
            }

            if previous_derivative.signum() != derivative.signum() {
                let stationary = bisect_stationary_point(
                    previous_rate,
                    rate,
                    previous_derivative,
                    &normalized,
                    options,
                )?;
                let (stationary_value, _) = npv_and_derivative(stationary, &normalized)?;
                if stationary_value.abs() <= options.tolerance * 100.0 {
                    push_root(&mut roots, stationary, options.tolerance);
                }
            }
        }
        previous = Some((rate, value, derivative));
    }

    roots.sort_by(f64::total_cmp);
    let status = match roots.len() {
        0 => IrrStatus::NoRoot,
        1 => IrrStatus::Unique,
        _ => IrrStatus::Multiple,
    };
    Ok(IrrAnalysis {
        roots,
        cash_flow_sign_changes: sign_changes,
        status,
        search_complete,
        searched_rate_range: (options.min_rate, options.max_rate),
    })
}

/// Return one unambiguous IRR.
///
/// Non-conventional cash flows with more than one sign change are intentionally
/// rejected here; call [`irr_analysis`] to inspect their discovered roots.
pub fn irr(cash_flows: &[f64]) -> Result<f64> {
    let analysis = irr_analysis(cash_flows, IrrOptions::default())?;
    match analysis.status {
        IrrStatus::Unique if analysis.cash_flow_sign_changes <= 1 => Ok(analysis.roots[0]),
        IrrStatus::NoRoot => Err(EconomicsError::NoSolution {
            context: "no IRR found in the configured search domain",
        }),
        IrrStatus::Indeterminate => Err(EconomicsError::AmbiguousSolution {
            context: "all-zero cash flows make every rate an IRR",
        }),
        IrrStatus::Unique | IrrStatus::Multiple => Err(EconomicsError::AmbiguousSolution {
            context: "cash flows require multi-root IRR analysis",
        }),
    }
}

/// Modified internal rate of return.
pub fn mirr(cash_flows: &[f64], finance_rate: f64, reinvestment_rate: f64) -> Result<f64> {
    ensure_rate(finance_rate, "MIRR finance rate must be greater than -1")?;
    ensure_rate(
        reinvestment_rate,
        "MIRR reinvestment rate must be greater than -1",
    )?;
    if cash_flows.len() < 2 {
        return Err(EconomicsError::InvalidParameter {
            context: "MIRR requires at least two periods",
        });
    }
    ensure_slice_finite(cash_flows, "MIRR cash flows")?;

    let terminal_period = cash_flows.len() - 1;
    let mut negative_present_value = 0.0;
    let mut positive_terminal_value = 0.0;
    for (period, &cash_flow) in cash_flows.iter().enumerate() {
        if cash_flow < 0.0 {
            negative_present_value += cash_flow / (1.0 + finance_rate).powi(period as i32);
        } else if cash_flow > 0.0 {
            positive_terminal_value +=
                cash_flow * (1.0 + reinvestment_rate).powi((terminal_period - period) as i32);
        }
    }
    if negative_present_value >= 0.0 || positive_terminal_value <= 0.0 {
        return Err(EconomicsError::NoSolution {
            context: "MIRR requires at least one negative and one positive cash flow",
        });
    }
    checked_output(
        (positive_terminal_value / -negative_present_value).powf(1.0 / terminal_period as f64)
            - 1.0,
        "MIRR calculation overflowed",
    )
}

/// Level payment for a fully amortizing loan: `PMT = P·r/(1-(1+r)^-n)`.
///
/// `ln_1p` and `exp_m1` avoid cancellation for very small non-zero rates.
pub fn annuity_payment(principal: f64, rate: f64, periods: u32) -> Result<f64> {
    ensure_finite(principal, "annuity principal")?;
    ensure_rate(rate, "annuity rate must be greater than -1")?;
    if principal < 0.0 {
        return Err(EconomicsError::InvalidParameter {
            context: "annuity principal must be non-negative",
        });
    }
    if periods == 0 {
        return Err(EconomicsError::InvalidParameter {
            context: "annuity periods must be positive",
        });
    }
    if rate == 0.0 {
        return Ok(principal / periods as f64);
    }
    let exponent = -(periods as f64) * rate.ln_1p();
    let denominator = -exponent.exp_m1();
    checked_output(
        principal * rate / denominator,
        "annuity calculation overflowed",
    )
}

/// One line of a fully amortizing loan schedule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmortizationPeriod {
    pub period: u32,
    pub payment: f64,
    pub interest: f64,
    pub principal: f64,
    pub remaining_balance: f64,
}

/// Generate a deterministic amortization schedule.
pub fn amortization_schedule(
    principal: f64,
    rate: f64,
    periods: u32,
) -> Result<Vec<AmortizationPeriod>> {
    let regular_payment = annuity_payment(principal, rate, periods)?;
    let mut balance = principal;
    let mut schedule = Vec::with_capacity(periods as usize);
    for period in 1..=periods {
        let interest = balance * rate;
        let (payment, principal_component, remaining_balance) = if period == periods {
            let final_payment = balance + interest;
            (final_payment, balance, 0.0)
        } else {
            let principal_component = regular_payment - interest;
            let remaining = balance - principal_component;
            (regular_payment, principal_component, remaining)
        };
        if !payment.is_finite()
            || !interest.is_finite()
            || !principal_component.is_finite()
            || !remaining_balance.is_finite()
        {
            return Err(EconomicsError::NumericalFailure {
                context: "amortization schedule overflowed",
            });
        }
        schedule.push(AmortizationPeriod {
            period,
            payment,
            interest,
            principal: principal_component,
            remaining_balance,
        });
        balance = remaining_balance;
    }
    Ok(schedule)
}


/// A dated cash flow on a UTC Unix timestamp, in seconds.
///
/// Amounts in one schedule must already share the same currency and unit.
/// This is an analytical value type, not a ledger-money representation.
/// Persistent financial records should use exact integer minor units or a
/// specified decimal representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DatedCashFlow {
    pub timestamp_seconds: i64,
    pub amount: f64,
}

impl DatedCashFlow {
    /// Construct a dated cash flow after checking that its amount is finite.
    pub fn new(timestamp_seconds: i64, amount: f64) -> Result<Self> {
        ensure_finite(amount, "dated cash-flow amount")?;
        Ok(Self {
            timestamp_seconds,
            amount,
        })
    }
}

const ACTUAL_365_FIXED_SECONDS: f64 = 365.0 * 24.0 * 60.0 * 60.0;

fn validate_dated_cash_flows(cash_flows: &[DatedCashFlow], context: &'static str) -> Result<()> {
    if cash_flows.is_empty() {
        return Err(EconomicsError::EmptyInput { context });
    }
    for flow in cash_flows {
        ensure_finite(flow.amount, "dated cash-flow amount")?;
    }
    if cash_flows
        .windows(2)
        .any(|pair| pair[0].timestamp_seconds > pair[1].timestamp_seconds)
    {
        return Err(EconomicsError::InvalidParameter {
            context: "dated cash flows must be ordered by nondecreasing UTC timestamp",
        });
    }
    Ok(())
}

fn dated_year_fraction(base_seconds: i64, timestamp_seconds: i64) -> f64 {
    let elapsed_seconds = (timestamp_seconds as i128 - base_seconds as i128) as f64;
    elapsed_seconds / ACTUAL_365_FIXED_SECONDS
}

fn neumaier_add(sum: &mut f64, correction: &mut f64, term: f64) {
    let next = *sum + term;
    if sum.abs() >= term.abs() {
        *correction += (*sum - next) + term;
    } else {
        *correction += (term - next) + *sum;
    }
    *sum = next;
}

/// Net present value for irregularly dated cash flows using Actual/365 Fixed.
///
/// The first cash-flow timestamp is the valuation date (time zero). Later
/// flows are discounted using elapsed UTC seconds divided by 365 days.
/// Inputs must be sorted by nondecreasing timestamp and share one currency
/// and unit. This function does not perform FX conversion or ledger rounding.
pub fn xnpv(rate: f64, cash_flows: &[DatedCashFlow]) -> Result<f64> {
    ensure_rate(rate, "XNPV discount rate must be greater than -1")?;
    validate_dated_cash_flows(cash_flows, "XNPV requires at least one dated cash flow")?;

    let base_seconds = cash_flows[0].timestamp_seconds;
    let log_growth = rate.ln_1p();
    let mut sum = 0.0;
    let mut correction = 0.0;

    for flow in cash_flows {
        let years = dated_year_fraction(base_seconds, flow.timestamp_seconds);
        let discount = (-years * log_growth).exp();
        let term = flow.amount * discount;
        if !term.is_finite() {
            return Err(EconomicsError::NumericalFailure {
                context: "XNPV term overflowed",
            });
        }
        neumaier_add(&mut sum, &mut correction, term);
    }

    checked_output(sum + correction, "XNPV summation overflowed")
}

fn xnpv_and_derivative(
    rate: f64,
    cash_flows: &[DatedCashFlow],
) -> Result<(f64, f64)> {
    let growth = 1.0 + rate;
    let log_growth = rate.ln_1p();
    let base_seconds = cash_flows[0].timestamp_seconds;
    let mut value_sum = 0.0;
    let mut value_correction = 0.0;
    let mut derivative_sum = 0.0;
    let mut derivative_correction = 0.0;

    for flow in cash_flows {
        let years = dated_year_fraction(base_seconds, flow.timestamp_seconds);
        let term = flow.amount * (-years * log_growth).exp();
        if !term.is_finite() {
            return Err(EconomicsError::NumericalFailure {
                context: "XIRR scan overflowed",
            });
        }
        neumaier_add(&mut value_sum, &mut value_correction, term);

        let derivative_term = -years * term / growth;
        if !derivative_term.is_finite() {
            return Err(EconomicsError::NumericalFailure {
                context: "XIRR derivative overflowed",
            });
        }
        neumaier_add(
            &mut derivative_sum,
            &mut derivative_correction,
            derivative_term,
        );
    }

    let value = value_sum + value_correction;
    let derivative = derivative_sum + derivative_correction;
    if value.is_finite() && derivative.is_finite() {
        Ok((value, derivative))
    } else {
        Err(EconomicsError::NumericalFailure {
            context: "XIRR scan overflowed",
        })
    }
}

fn bisect_xirr_value_root(
    mut lo: f64,
    mut hi: f64,
    mut f_lo: f64,
    cash_flows: &[DatedCashFlow],
    options: IrrOptions,
) -> Result<f64> {
    for _ in 0..options.max_iterations {
        let mid = lo + (hi - lo) * 0.5;
        let (f_mid, _) = xnpv_and_derivative(mid, cash_flows)?;
        if f_mid.abs() <= options.tolerance
            || (hi - lo).abs() <= options.tolerance * mid.abs().max(1.0)
        {
            return Ok(mid);
        }
        if f_lo.signum() == f_mid.signum() {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
    }
    Ok(lo + (hi - lo) * 0.5)
}

fn bisect_xirr_stationary_point(
    mut lo: f64,
    mut hi: f64,
    mut d_lo: f64,
    cash_flows: &[DatedCashFlow],
    options: IrrOptions,
) -> Result<f64> {
    for _ in 0..options.max_iterations {
        let mid = lo + (hi - lo) * 0.5;
        let (_, d_mid) = xnpv_and_derivative(mid, cash_flows)?;
        if d_mid.abs() <= options.tolerance
            || (hi - lo).abs() <= options.tolerance * mid.abs().max(1.0)
        {
            return Ok(mid);
        }
        if d_lo.signum() == d_mid.signum() {
            lo = mid;
            d_lo = d_mid;
        } else {
            hi = mid;
        }
    }
    Ok(lo + (hi - lo) * 0.5)
}

/// Discover all XIRR roots found in a bounded rate domain.
///
/// The search uses the same explicit bounds and tolerances as IrrOptions.
/// It scans uniformly in ln(1 + rate), refines sign-changing roots by
/// bisection, and checks stationary points for repeated roots. The result
/// records whether every scan point was evaluable; an incomplete scan must
/// not be reported as proof that no other root exists.
pub fn xirr_analysis(
    cash_flows: &[DatedCashFlow],
    options: IrrOptions,
) -> Result<IrrAnalysis> {
    options.validate()?;
    validate_dated_cash_flows(cash_flows, "XIRR requires at least one dated cash flow")?;

    let scale = cash_flows
        .iter()
        .map(|flow| flow.amount.abs())
        .fold(0.0_f64, f64::max);
    let amounts: Vec<f64> = cash_flows.iter().map(|flow| flow.amount).collect();
    let sign_changes = cash_flow_sign_changes(&amounts);

    if scale == 0.0 {
        return Ok(IrrAnalysis {
            roots: Vec::new(),
            cash_flow_sign_changes: 0,
            status: IrrStatus::Indeterminate,
            search_complete: true,
            searched_rate_range: (options.min_rate, options.max_rate),
        });
    }

    let normalized: Vec<DatedCashFlow> = cash_flows
        .iter()
        .map(|flow| DatedCashFlow {
            timestamp_seconds: flow.timestamp_seconds,
            amount: flow.amount / scale,
        })
        .collect();
    let min_log = options.min_rate.ln_1p();
    let max_log = options.max_rate.ln_1p();
    let mut roots = Vec::new();
    let mut search_complete = true;
    let mut previous: Option<(f64, f64, f64)> = None;

    for step in 0..=options.scan_steps {
        let fraction = step as f64 / options.scan_steps as f64;
        let rate = (min_log + (max_log - min_log) * fraction).exp_m1();
        let evaluation = xnpv_and_derivative(rate, &normalized);
        let (value, derivative) = match evaluation {
            Ok(result) => result,
            Err(EconomicsError::NumericalFailure { .. }) => {
                search_complete = false;
                previous = None;
                continue;
            }
            Err(error) => return Err(error),
        };

        if value.abs() <= options.tolerance {
            push_root(&mut roots, rate, options.tolerance);
        }

        if let Some((previous_rate, previous_value, previous_derivative)) = previous {
            if previous_value.signum() != value.signum() {
                let root = bisect_xirr_value_root(
                    previous_rate,
                    rate,
                    previous_value,
                    &normalized,
                    options,
                )?;
                push_root(&mut roots, root, options.tolerance);
            }

            if previous_derivative.signum() != derivative.signum() {
                let stationary = bisect_xirr_stationary_point(
                    previous_rate,
                    rate,
                    previous_derivative,
                    &normalized,
                    options,
                )?;
                let (stationary_value, _) = xnpv_and_derivative(stationary, &normalized)?;
                if stationary_value.abs() <= options.tolerance * 100.0 {
                    push_root(&mut roots, stationary, options.tolerance);
                }
            }
        }
        previous = Some((rate, value, derivative));
    }

    roots.sort_by(f64::total_cmp);
    let status = match roots.len() {
        0 => IrrStatus::NoRoot,
        1 => IrrStatus::Unique,
        _ => IrrStatus::Multiple,
    };
    Ok(IrrAnalysis {
        roots,
        cash_flow_sign_changes: sign_changes,
        status,
        search_complete,
        searched_rate_range: (options.min_rate, options.max_rate),
    })
}

/// Return one unambiguous XIRR using Actual/365 Fixed.
///
/// Non-conventional cash flows with multiple sign changes are rejected even
/// when this bounded search finds only one root. Use xirr_analysis to inspect
/// the roots and search-completeness status explicitly.
pub fn xirr(cash_flows: &[DatedCashFlow]) -> Result<f64> {
    let analysis = xirr_analysis(cash_flows, IrrOptions::default())?;
    if !analysis.search_complete {
        return Err(EconomicsError::NumericalFailure {
            context: "XIRR search incomplete in the configured rate domain",
        });
    }
    match analysis.status {
        IrrStatus::Unique if analysis.cash_flow_sign_changes <= 1 => Ok(analysis.roots[0]),
        IrrStatus::NoRoot => Err(EconomicsError::NoSolution {
            context: "no XIRR found in the configured search domain",
        }),
        IrrStatus::Indeterminate => Err(EconomicsError::AmbiguousSolution {
            context: "all-zero dated cash flows make every rate an XIRR",
        }),
        IrrStatus::Unique | IrrStatus::Multiple => Err(EconomicsError::AmbiguousSolution {
            context: "dated cash flows require multi-root XIRR analysis",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    fn dated_flow(day: i64, amount: f64) -> DatedCashFlow {
        DatedCashFlow::new(day * 86_400, amount).unwrap()
    }

    #[test]
    fn xnpv_matches_periodic_npv_for_annual_dates() {
        let flows = [
            dated_flow(0, -1_000.0),
            dated_flow(365, 500.0),
            dated_flow(730, 500.0),
        ];
        let dated = xnpv(0.10, &flows).unwrap();
        let periodic = npv(0.10, &[-1_000.0, 500.0, 500.0]).unwrap();
        assert!((dated - periodic).abs() < 1e-10);
    }

    #[test]
    fn xnpv_uses_actual_elapsed_days() {
        let flows = [
            dated_flow(0, -1_000.0),
            dated_flow(180, 600.0),
            dated_flow(547, 600.0),
        ];
        let expected = -1_000.0
            + 600.0 / 1.10_f64.powf(180.0 / 365.0)
            + 600.0 / 1.10_f64.powf(547.0 / 365.0);
        assert!((xnpv(0.10, &flows).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn xirr_recovers_known_rate_from_dated_cash_flows() {
        let flows = [dated_flow(0, -1_000.0), dated_flow(730, 1_210.0)];
        let rate = xirr(&flows).unwrap();
        assert!((rate - 0.10).abs() < 1e-8);
        assert!(xnpv(rate, &flows).unwrap().abs() < 1e-7);
    }

    #[test]
    fn xirr_analysis_finds_multiple_dated_roots() {
        let flows = [
            dated_flow(0, -100.0),
            dated_flow(365, 230.0),
            dated_flow(730, -132.0),
        ];
        let analysis = xirr_analysis(&flows, IrrOptions::default()).unwrap();
        assert_eq!(analysis.status, IrrStatus::Multiple);
        assert!(analysis.search_complete);
        assert_eq!(analysis.roots.len(), 2);
        assert!((analysis.roots[0] - 0.10).abs() < 1e-7);
        assert!((analysis.roots[1] - 0.20).abs() < 1e-7);
    }

    #[test]
    fn xnpv_rejects_unsorted_dates() {
        let flows = [dated_flow(365, 100.0), dated_flow(0, -100.0)];
        assert!(matches!(
            xnpv(0.10, &flows),
            Err(EconomicsError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn xnpv_rejects_non_finite_amounts() {
        let flows = [DatedCashFlow {
            timestamp_seconds: 0,
            amount: f64::NAN,
        }];
        assert!(matches!(
            xnpv(0.10, &flows),
            Err(EconomicsError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn xirr_analysis_marks_zero_cash_flows_indeterminate() {
        let flows = [dated_flow(0, 0.0), dated_flow(365, 0.0)];
        let analysis = xirr_analysis(&flows, IrrOptions::default()).unwrap();
        assert_eq!(analysis.status, IrrStatus::Indeterminate);
        assert!(analysis.search_complete);
        assert!(analysis.roots.is_empty());
    }

    #[test]
    fn fv_pv_roundtrip() {
        let value = future_value(1000.0, 0.05, 10).unwrap();
        assert!((value - 1628.894).abs() < 0.01);
        assert!((present_value(value, 0.05, 10).unwrap() - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn rate_conversions_roundtrip() {
        let effective = effective_annual_rate(0.12, 12).unwrap();
        let nominal = nominal_annual_rate(effective, 12).unwrap();
        assert!((effective - 0.1268250301).abs() < 1e-10);
        assert!((nominal - 0.12).abs() < 1e-12);
    }

    #[test]
    fn npv_known() {
        let value = npv(0.10, &[-1000.0, 500.0, 500.0, 500.0]).unwrap();
        assert!((value - 243.426).abs() < 0.01);
    }

    #[test]
    fn irr_known() {
        let flows = [-1000.0, 600.0, 600.0];
        let rate = irr(&flows).unwrap();
        assert!((rate - 0.13066).abs() < 1e-4);
        assert!(npv(rate, &flows).unwrap().abs() < 1e-6);
    }

    #[test]
    fn analysis_finds_two_irrs() {
        let analysis = irr_analysis(&[-100.0, 230.0, -132.0], IrrOptions::default()).unwrap();
        assert_eq!(analysis.status, IrrStatus::Multiple);
        assert_eq!(analysis.roots.len(), 2);
        assert!((analysis.roots[0] - 0.10).abs() < 1e-7);
        assert!((analysis.roots[1] - 0.20).abs() < 1e-7);
    }

    #[test]
    fn analysis_finds_high_irr() {
        let analysis = irr_analysis(&[-1.0, 20.0], IrrOptions::default()).unwrap();
        assert_eq!(analysis.status, IrrStatus::Unique);
        assert!((analysis.roots[0] - 19.0).abs() < 1e-7);
        assert!((irr(&[-1.0, 20.0]).unwrap() - 19.0).abs() < 1e-7);
    }

    #[test]
    fn mirr_known() {
        let rate = mirr(&[-1000.0, 400.0, 500.0, 600.0], 0.08, 0.10).unwrap();
        assert!((rate - 0.1778338119).abs() < 1e-10);
    }

    #[test]
    fn tiny_rate_annuity_is_stable() {
        let payment = annuity_payment(1200.0, 1e-12, 12).unwrap();
        assert!((payment - 100.0).abs() < 1e-8);
    }

    #[test]
    fn mortgage_schedule_closes() {
        let schedule = amortization_schedule(200_000.0, 0.04 / 12.0, 360).unwrap();
        assert_eq!(schedule.len(), 360);
        assert_eq!(schedule.last().unwrap().remaining_balance, 0.0);
        let principal_paid: f64 = schedule.iter().map(|line| line.principal).sum();
        assert!((principal_paid - 200_000.0).abs() < 1e-6);
    }
}
