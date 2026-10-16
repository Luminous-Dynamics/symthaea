// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validated numerical utilities for derived loudspeaker/acoustic response curves.
//!
//! A curve is supplied as (frequency_hz, level_db) pairs. This module does not ingest
//! raw observations, establish microphone calibration, or own evidence provenance;
//! those remain in FIELD. It does not claim CTA-2034, AES75, IEC 60268-21 compliance
//! or a human-preference score; those require separate versioned procedures.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseCurveError {
    TooFewPoints,
    NonFiniteInput,
    NonPositiveFrequency,
    FrequenciesNotStrictlyIncreasing,
    ExtrapolationNotAllowed,
    InvalidBand,
    NumericOverflow,
}

impl fmt::Display for ResponseCurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::TooFewPoints => "frequency response requires at least two points",
            Self::NonFiniteInput => "frequency response contains or requests a non-finite value",
            Self::NonPositiveFrequency => "frequency must be greater than zero",
            Self::FrequenciesNotStrictlyIncreasing => "frequency points must be strictly increasing",
            Self::ExtrapolationNotAllowed => "requested frequency is outside supplied curve support",
            Self::InvalidBand => "frequency band is invalid or outside curve support",
            Self::NumericOverflow => "frequency response calculation overflowed",
        };
        f.write_str(text)
    }
}
impl std::error::Error for ResponseCurveError {}

/// Validate a curve of (frequency_hz, level_db) pairs. Requires finite values,
/// positive frequencies, at least two points, and strictly increasing frequencies.
pub fn validate_response_curve(points: &[(f64, f64)]) -> Result<(), ResponseCurveError> {
    if points.len() < 2 { return Err(ResponseCurveError::TooFewPoints); }
    let mut previous = None;
    for &(frequency, level) in points {
        if !frequency.is_finite() || !level.is_finite() {
            return Err(ResponseCurveError::NonFiniteInput);
        }
        if frequency <= 0.0 { return Err(ResponseCurveError::NonPositiveFrequency); }
        if previous.is_some_and(|value| frequency <= value) {
            return Err(ResponseCurveError::FrequenciesNotStrictlyIncreasing);
        }
        previous = Some(frequency);
    }
    Ok(())
}

/// Interpolate level linearly on the log-frequency axis. This is a declared response-plot
/// convention, not a claim that the physical transfer function is linear in log frequency.
/// Extrapolation is rejected.
pub fn interpolate_log_frequency(
    points: &[(f64, f64)],
    target_frequency_hz: f64,
) -> Result<f64, ResponseCurveError> {
    validate_response_curve(points)?;
    if !target_frequency_hz.is_finite() { return Err(ResponseCurveError::NonFiniteInput); }
    if target_frequency_hz <= 0.0 { return Err(ResponseCurveError::NonPositiveFrequency); }
    if target_frequency_hz < points[0].0 || target_frequency_hz > points[points.len() - 1].0 {
        return Err(ResponseCurveError::ExtrapolationNotAllowed);
    }
    for &(frequency, level) in points {
        if target_frequency_hz == frequency { return Ok(level); }
    }
    for pair in points.windows(2) {
        let (f0, y0) = pair[0];
        let (f1, y1) = pair[1];
        if target_frequency_hz > f0 && target_frequency_hz < f1 {
            let fraction = (target_frequency_hz.ln() - f0.ln()) / (f1.ln() - f0.ln());
            let result = y0 + fraction * (y1 - y0);
            return result.is_finite().then_some(result).ok_or(ResponseCurveError::NumericOverflow);
        }
    }
    Err(ResponseCurveError::ExtrapolationNotAllowed)
}

/// Mean level in dB weighted uniformly over logarithmic frequency. Integrates trapezoids
/// in (ln(frequency), level_db) space. This is a curve-summary statistic, not a power
/// average, standardized loudspeaker score, or preference predictor.
pub fn log_frequency_mean_db(
    points: &[(f64, f64)], low_hz: f64, high_hz: f64,
) -> Result<f64, ResponseCurveError> {
    validate_response_curve(points)?;
    validate_band(points, low_hz, high_hz)?;
    let mut band = Vec::with_capacity(points.len() + 2);
    band.push((low_hz, interpolate_log_frequency(points, low_hz)?));
    band.extend(points.iter().copied().filter(|(f, _)| *f > low_hz && *f < high_hz));
    band.push((high_hz, interpolate_log_frequency(points, high_hz)?));
    let mut integral = 0.0;
    for pair in band.windows(2) {
        integral += (pair[0].1 + pair[1].1) * 0.5 * (pair[1].0.ln() - pair[0].0.ln());
        if !integral.is_finite() { return Err(ResponseCurveError::NumericOverflow); }
    }
    let result = integral / (high_hz.ln() - low_hz.ln());
    result.is_finite().then_some(result).ok_or(ResponseCurveError::NumericOverflow)
}

/// Log-frequency RMS difference between measured and target curves over a declared band.
/// Both curves must cover the band. The union of their frequency knots is integrated,
/// so different measurement grids remain comparable. Error is computed in dB space and
/// is not a CTA-2034 score, calibration certificate, or preference predictor.
pub fn log_frequency_rms_deviation_db(
    measured: &[(f64, f64)], target: &[(f64, f64)], low_hz: f64, high_hz: f64,
) -> Result<f64, ResponseCurveError> {
    validate_response_curve(measured)?;
    validate_response_curve(target)?;
    validate_band(measured, low_hz, high_hz)?;
    validate_band(target, low_hz, high_hz)?;

    let mut frequencies = Vec::with_capacity(measured.len() + target.len() + 2);
    frequencies.push(low_hz);
    frequencies.extend(measured.iter().chain(target.iter()).map(|(f, _)| *f)
        .filter(|f| *f > low_hz && *f < high_hz));
    frequencies.push(high_hz);
    frequencies.sort_by(f64::total_cmp);
    frequencies.dedup_by(|a, b| *a == *b);

    let mut deltas = Vec::with_capacity(frequencies.len());
    for frequency in frequencies {
        let delta = interpolate_log_frequency(measured, frequency)?
            - interpolate_log_frequency(target, frequency)?;
        if !delta.is_finite() { return Err(ResponseCurveError::NumericOverflow); }
        deltas.push((frequency, delta));
    }

    // Exact squared integral for a linear delta over ln(frequency):
    // delta_ln * (d0^2 + d0*d1 + d1^2) / 3.
    let mut squared_integral = 0.0;
    for pair in deltas.windows(2) {
        let (f0, d0) = pair[0];
        let (f1, d1) = pair[1];
        let term = (f1.ln() - f0.ln()) * (d0*d0 + d0*d1 + d1*d1) / 3.0;
        if !term.is_finite() { return Err(ResponseCurveError::NumericOverflow); }
        squared_integral += term;
        if !squared_integral.is_finite() { return Err(ResponseCurveError::NumericOverflow); }
    }
    let mean_square = squared_integral / (high_hz.ln() - low_hz.ln());
    if !mean_square.is_finite() || mean_square < 0.0 {
        return Err(ResponseCurveError::NumericOverflow);
    }
    let result = mean_square.sqrt();
    result.is_finite().then_some(result).ok_or(ResponseCurveError::NumericOverflow)
}

fn validate_band(points: &[(f64, f64)], low_hz: f64, high_hz: f64) -> Result<(), ResponseCurveError> {
    if !low_hz.is_finite() || !high_hz.is_finite() {
        return Err(ResponseCurveError::NonFiniteInput);
    }
    if low_hz <= 0.0 || high_hz <= 0.0 {
        return Err(ResponseCurveError::NonPositiveFrequency);
    }
    if low_hz >= high_hz || low_hz < points[0].0 || high_hz > points[points.len()-1].0 {
        return Err(ResponseCurveError::InvalidBand);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_rejects_short_unsorted_duplicate_and_invalid_values() {
        assert_eq!(validate_response_curve(&[]), Err(ResponseCurveError::TooFewPoints));
        assert_eq!(validate_response_curve(&[(100.0, 0.0)]), Err(ResponseCurveError::TooFewPoints));
        assert_eq!(validate_response_curve(&[(100.0, 0.0), (90.0, 1.0)]),
            Err(ResponseCurveError::FrequenciesNotStrictlyIncreasing));
        assert_eq!(validate_response_curve(&[(100.0, 0.0), (100.0, 1.0)]),
            Err(ResponseCurveError::FrequenciesNotStrictlyIncreasing));
        assert_eq!(validate_response_curve(&[(0.0, 0.0), (100.0, 1.0)]),
            Err(ResponseCurveError::NonPositiveFrequency));
        assert_eq!(validate_response_curve(&[(100.0, 0.0), (200.0, f64::NAN)]),
            Err(ResponseCurveError::NonFiniteInput));
    }

    #[test]
    fn interpolation_is_log_frequency_linear_and_refuses_extrapolation() {
        let curve = [(100.0, 0.0), (1_000.0, 10.0)];
        let midpoint = (100.0_f64 * 1_000.0).sqrt();
        assert!((interpolate_log_frequency(&curve, midpoint).unwrap() - 5.0).abs() < 1e-12);
        assert_eq!(interpolate_log_frequency(&curve, 100.0).unwrap(), 0.0);
        assert_eq!(interpolate_log_frequency(&curve, 50.0), Err(ResponseCurveError::ExtrapolationNotAllowed));
    }

    #[test]
    fn log_frequency_mean_matches_hand_calculation() {
        let curve = [(100.0, 0.0), (1_000.0, 10.0)];
        assert!((log_frequency_mean_db(&curve, 100.0, 1_000.0).unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn rms_deviation_matches_hand_calculation_and_unequal_grids() {
        let measured = [(100.0, 0.0), (1_000.0, 10.0)];
        let target = [(100.0, 1.0), (1_000.0, 1.0)];
        let expected = (73.0_f64 / 3.0).sqrt();
        assert!((log_frequency_rms_deviation_db(&measured, &target, 100.0, 1_000.0).unwrap() - expected).abs() < 1e-12);
        let measured_grid = [(100.0, 0.0), (300.0, 3.0), (1_000.0, 10.0)];
        let sparse_target = [(100.0, 1.0), (1_000.0, 1.0)];
        let dense_target = [(100.0, 1.0), (250.0, 1.0), (700.0, 1.0), (1_000.0, 1.0)];
        let a = log_frequency_rms_deviation_db(&measured_grid, &sparse_target, 100.0, 1_000.0).unwrap();
        let b = log_frequency_rms_deviation_db(&measured_grid, &dense_target, 100.0, 1_000.0).unwrap();
        assert!((a - b).abs() < 1e-12);
    }

    #[test]
    fn invalid_bands_fail_closed() {
        let curve = [(100.0, 0.0), (1_000.0, 10.0)];
        assert_eq!(log_frequency_mean_db(&curve, 200.0, 100.0), Err(ResponseCurveError::InvalidBand));
        assert_eq!(log_frequency_mean_db(&curve, 10.0, 1_000.0), Err(ResponseCurveError::InvalidBand));
        assert_eq!(log_frequency_rms_deviation_db(&curve, &curve, 100.0, 10_000.0), Err(ResponseCurveError::InvalidBand));
    }

    #[test]
    fn arithmetic_overflow_is_reported() {
        let measured = [(100.0, f64::MAX), (1_000.0, f64::MAX)];
        let target = [(100.0, -f64::MAX), (1_000.0, -f64::MAX)];
        assert_eq!(log_frequency_rms_deviation_db(&measured, &target, 100.0, 1_000.0),
            Err(ResponseCurveError::NumericOverflow));
    }
}
