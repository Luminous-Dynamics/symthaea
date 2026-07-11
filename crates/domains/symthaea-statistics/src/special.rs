// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Special functions underpinning the distribution CDFs: log-gamma, the error
//! function, and the regularized incomplete gamma / beta functions. These are
//! the standard Numerical-Recipes-style algorithms, checked against known
//! closed-form values.

use std::f64::consts::PI;

/// Natural log of the gamma function (Lanczos approximation, g = 7).
pub fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula for the left half-plane.
        (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// The error function (Abramowitz & Stegun 7.1.26, |error| ≤ 1.5e-7).
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

/// The complementary error function.
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// Regularized lower incomplete gamma P(a, x) = γ(a, x) / Γ(a), for x, a > 0.
pub fn gammp(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gser(a, x)
    } else {
        1.0 - gcf(a, x)
    }
}

/// Regularized upper incomplete gamma Q(a, x) = 1 − P(a, x).
pub fn gammq(a: f64, x: f64) -> f64 {
    1.0 - gammp(a, x)
}

/// Lower incomplete gamma via its series representation (converges for x < a+1).
fn gser(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..200 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    sum * (-x + a * x.ln() - gln).exp()
}

/// Upper incomplete gamma via the Lentz continued fraction (for x ≥ a+1).
fn gcf(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    let tiny = 1e-30;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..200 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - gln).exp() * h
}

/// Regularized incomplete beta I_x(a, b), for 0 ≤ x ≤ 1 and a, b > 0.
pub fn betai(a: f64, b: f64, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 || x == 1.0 {
        return x;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Continued fraction for the incomplete beta (Lentz's method).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-30;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..200 {
        let m = m as f64;
        let m2 = 2.0 * m;
        // Even step.
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step.
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_factorials() {
        // Γ(n) = (n-1)!  →  ln Γ(5) = ln 24.
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-9);
        assert!((ln_gamma(1.0)).abs() < 1e-9); // ln Γ(1) = ln 1 = 0
        // Γ(1/2) = √π.
        assert!((ln_gamma(0.5) - PI.sqrt().ln()).abs() < 1e-6);
    }

    #[test]
    fn erf_known_points() {
        assert!(erf(0.0).abs() < 1e-9);
        assert!((erf(10.0) - 1.0).abs() < 1e-9);
        assert!((erf(-1.0) + erf(1.0)).abs() < 1e-9); // odd function
        // erf(1) ≈ 0.8427.
        assert!((erf(1.0) - 0.842_700_79).abs() < 1e-6);
    }

    #[test]
    fn incomplete_gamma_closed_form() {
        // For a=1, P(1,x) = 1 - e^{-x}.
        for &x in &[0.5, 1.0, 2.0, 5.0] {
            assert!((gammp(1.0, x) - (1.0 - (-x).exp())).abs() < 1e-9, "x={x}");
        }
    }

    #[test]
    fn incomplete_beta_symmetry() {
        // I_x(a,b) = 1 - I_{1-x}(b,a).
        let v = betai(2.0, 3.0, 0.4);
        let w = 1.0 - betai(3.0, 2.0, 0.6);
        assert!((v - w).abs() < 1e-9, "{v} vs {w}");
        // I_x(1,1) = x (uniform).
        assert!((betai(1.0, 1.0, 0.3) - 0.3).abs() < 1e-9);
    }
}
