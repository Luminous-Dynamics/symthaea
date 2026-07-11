// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Time value of money: future/present value, NPV, IRR, compound interest,
//! annuity (loan) payments.

/// Future value of a present sum: `FV = PV·(1+r)^n`.
pub fn future_value(present: f64, rate: f64, periods: u32) -> f64 {
    present * (1.0 + rate).powi(periods as i32)
}

/// Present value of a future sum: `PV = FV/(1+r)^n`.
pub fn present_value(future: f64, rate: f64, periods: u32) -> f64 {
    future / (1.0 + rate).powi(periods as i32)
}

/// Compound interest: `A = P·(1 + r/m)^(m·t)` (m compounding periods per year).
pub fn compound_interest(principal: f64, annual_rate: f64, per_year: u32, years: f64) -> f64 {
    let m = per_year as f64;
    principal * (1.0 + annual_rate / m).powf(m * years)
}

/// Net present value of a cash-flow series (index = period, `cash_flows[0]` at t=0).
pub fn npv(rate: f64, cash_flows: &[f64]) -> f64 {
    cash_flows
        .iter()
        .enumerate()
        .map(|(t, cf)| cf / (1.0 + rate).powi(t as i32))
        .sum()
}

/// Internal rate of return: the discount rate making NPV = 0, by bisection.
///
/// Returns `None` if NPV does not change sign over the search bracket
/// (e.g. all-positive or all-negative flows).
pub fn irr(cash_flows: &[f64]) -> Option<f64> {
    let (mut lo, mut hi) = (-0.9999_f64, 10.0_f64);
    let (mut f_lo, f_hi) = (npv(lo, cash_flows), npv(hi, cash_flows));
    if f_lo.signum() == f_hi.signum() {
        return None;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let f_mid = npv(mid, cash_flows);
        if f_mid.abs() < 1e-12 {
            return Some(mid);
        }
        if f_lo.signum() == f_mid.signum() {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// Level payment for a fully-amortizing loan: `PMT = P·r/(1-(1+r)^-n)`.
/// `rate` is per period; falls back to `principal/periods` at zero rate.
pub fn annuity_payment(principal: f64, rate: f64, periods: u32) -> f64 {
    if periods == 0 {
        return 0.0;
    }
    if rate.abs() < 1e-15 {
        return principal / periods as f64;
    }
    principal * rate / (1.0 - (1.0 + rate).powi(-(periods as i32)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fv_pv_roundtrip() {
        let fv = future_value(1000.0, 0.05, 10);
        assert!((fv - 1628.894).abs() < 0.01, "fv={fv}");
        assert!((present_value(fv, 0.05, 10) - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn npv_known() {
        // -1000 now, +500 for 3 years at 10%.
        let v = npv(0.10, &[-1000.0, 500.0, 500.0, 500.0]);
        assert!((v - 243.426).abs() < 0.01, "npv={v}");
    }

    #[test]
    fn irr_known() {
        // -1000, +600, +600 → IRR ≈ 13.07%.
        let r = irr(&[-1000.0, 600.0, 600.0]).unwrap();
        assert!((r - 0.13066).abs() < 1e-4, "irr={r}");
        assert!((npv(r, &[-1000.0, 600.0, 600.0])).abs() < 1e-6);
    }

    #[test]
    fn irr_none_when_no_sign_change() {
        assert!(irr(&[100.0, 200.0]).is_none());
    }

    #[test]
    fn mortgage_payment() {
        // $200k, 4%/yr monthly, 30 years → ~$954.83/month.
        let pmt = annuity_payment(200_000.0, 0.04 / 12.0, 360);
        assert!((pmt - 954.83).abs() < 0.01, "pmt={pmt}");
    }

    #[test]
    fn zero_rate_annuity_is_linear() {
        assert!((annuity_payment(1200.0, 0.0, 12) - 100.0).abs() < 1e-9);
    }
}
