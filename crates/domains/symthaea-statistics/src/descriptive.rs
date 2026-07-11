// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Descriptive statistics over a slice of samples.

/// Arithmetic mean. `None` for an empty slice.
pub fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some(xs.iter().sum::<f64>() / xs.len() as f64)
}

/// Sample variance (Bessel-corrected, divides by n−1). Needs ≥ 2 samples.
pub fn variance(xs: &[f64]) -> Option<f64> {
    if xs.len() < 2 {
        return None;
    }
    let m = mean(xs)?;
    let ss: f64 = xs.iter().map(|x| (x - m).powi(2)).sum();
    Some(ss / (xs.len() as f64 - 1.0))
}

/// Sample standard deviation (√ of the sample variance).
pub fn std_dev(xs: &[f64]) -> Option<f64> {
    variance(xs).map(f64::sqrt)
}

/// Population variance (divides by n).
pub fn population_variance(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let m = mean(xs)?;
    let ss: f64 = xs.iter().map(|x| (x - m).powi(2)).sum();
    Some(ss / xs.len() as f64)
}

/// The `q`-quantile (0 ≤ q ≤ 1) via linear interpolation between order
/// statistics (the "type 7" / NumPy default convention).
pub fn quantile(xs: &[f64], q: f64) -> Option<f64> {
    if xs.is_empty() || !(0.0..=1.0).contains(&q) {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.len() == 1 {
        return Some(v[0]);
    }
    let pos = q * (v.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let frac = pos - lo as f64;
    if lo + 1 < v.len() {
        Some(v[lo] + frac * (v[lo + 1] - v[lo]))
    } else {
        Some(v[lo])
    }
}

/// The median (0.5-quantile).
pub fn median(xs: &[f64]) -> Option<f64> {
    quantile(xs, 0.5)
}

/// Sample covariance between paired series (Bessel-corrected). Needs equal
/// length ≥ 2.
pub fn covariance(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return None;
    }
    let (mx, my) = (mean(xs)?, mean(ys)?);
    let s: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    Some(s / (xs.len() as f64 - 1.0))
}

/// Pearson correlation coefficient in [−1, 1].
pub fn correlation(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let cov = covariance(xs, ys)?;
    let (sx, sy) = (std_dev(xs)?, std_dev(ys)?);
    if sx == 0.0 || sy == 0.0 {
        return None;
    }
    Some(cov / (sx * sy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_variance_std() {
        let xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((mean(&xs).unwrap() - 5.0).abs() < 1e-12);
        // Sample variance of this classic set is 32/7.
        assert!((variance(&xs).unwrap() - 32.0 / 7.0).abs() < 1e-12);
        assert!((std_dev(&xs).unwrap() - (32.0f64 / 7.0).sqrt()).abs() < 1e-12);
        // Population variance is 4.
        assert!((population_variance(&xs).unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn median_and_quantiles() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
        // Quartiles of 1..=5 (type 7): Q1=2, Q3=4.
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.25), Some(2.0));
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.75), Some(4.0));
    }

    #[test]
    fn correlation_perfect_and_none() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [2.0, 4.0, 6.0, 8.0]; // y = 2x
        assert!((correlation(&x, &y).unwrap() - 1.0).abs() < 1e-12);
        let z = [8.0, 6.0, 4.0, 2.0]; // negatively related
        assert!((correlation(&x, &z).unwrap() + 1.0).abs() < 1e-12);
        assert_eq!(mean(&[]), None);
    }
}
