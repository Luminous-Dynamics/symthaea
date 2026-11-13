// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finite-sweep subharmonic frequency-locking diagnostics.
//!
//! This module estimates a subharmonic spectral peak for each pulse-detuning
//! trace and measures the longest contiguous detuning interval that remains
//! frequency-locked. It deliberately does not classify a discrete
//! time-crystalline phase: trivial perfect-pi-pulse alternation, finite-size
//! metastability and initial-state-specific beating can all produce a locked
//! finite trace.

use crate::exact_floquet::analyze_subharmonic_response;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

/// One trace in a pulse-angle-detuning sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PulseSweepTrace<'a> {
    /// Pulse-angle offset from the nominal pi rotation, in radians.
    pub pulse_detuning_rad: f64,
    /// Stroboscopic observable; sample n is after n Floquet periods.
    pub signal: &'a [f64],
}

/// Configuration for finite-sweep frequency-locking analysis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrequencyLockingConfig {
    /// Number of initial stroboscopic samples excluded as transient.
    pub analysis_start: usize,
    /// Target subharmonic frequency, in cycles per drive period.
    pub target_frequency: f64,
    /// Half-width of the FFT search band around the target frequency.
    pub search_half_width: f64,
    /// Maximum frequency error for a point to be considered locked.
    pub frequency_tolerance: f64,
    /// Minimum normalized sinusoidal amplitude.
    pub minimum_peak_amplitude: f64,
    /// Minimum peak-to-median-background amplitude ratio.
    pub minimum_spectral_prominence: f64,
    /// Minimum finite-window demodulated sign coherence.
    pub minimum_sign_coherence: f64,
    /// Required number of adjacent detuning samples for a sweep-level lock.
    pub minimum_contiguous_points: usize,
}

impl Default for FrequencyLockingConfig {
    fn default() -> Self {
        Self {
            analysis_start: 20,
            target_frequency: 0.5,
            search_half_width: 0.10,
            frequency_tolerance: 0.005,
            minimum_peak_amplitude: 0.10,
            minimum_spectral_prominence: 3.0,
            minimum_sign_coherence: 0.80,
            minimum_contiguous_points: 3,
        }
    }
}

/// Invalid sweep input or failure to compute a finite spectral estimate.
#[derive(Clone, Debug, PartialEq)]
pub enum FrequencyLockingError {
    EmptySweep,
    InvalidConfiguration,
    NonFiniteDetuning { index: usize },
    NonIncreasingDetuning { index: usize },
    InvalidTrace { index: usize },
    TransformLengthOverflow,
    NonFiniteSpectrum,
}

impl fmt::Display for FrequencyLockingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySweep => write!(f, "pulse-detuning sweep is empty"),
            Self::InvalidConfiguration => write!(f, "frequency-locking configuration is invalid"),
            Self::NonFiniteDetuning { index } => {
                write!(f, "pulse detuning at index {index} is not finite")
            }
            Self::NonIncreasingDetuning { index } => write!(
                f,
                "pulse detuning at index {index} is not strictly greater than its predecessor"
            ),
            Self::InvalidTrace { index } => write!(
                f,
                "trace at sweep index {index} is too short or contains non-finite samples"
            ),
            Self::TransformLengthOverflow => write!(f, "FFT transform length overflow"),
            Self::NonFiniteSpectrum => {
                write!(f, "spectral calculation produced a non-finite value")
            }
        }
    }
}

impl Error for FrequencyLockingError {}

/// Estimated subharmonic peak for one finite stroboscopic trace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubharmonicPeak {
    /// Peak frequency in cycles per drive period.
    pub frequency: f64,
    /// FFT frequency-bin spacing after zero padding (grid spacing only).
    pub frequency_resolution: f64,
    /// Observation-limited frequency scale, approximately 1 / window duration.
    ///
    /// Zero padding makes the FFT peak grid finer but does not improve this
    /// physical resolution.
    pub observation_frequency_resolution: f64,
    /// Hann-window-normalized sinusoidal amplitude.
    pub amplitude: f64,
    /// Peak amplitude divided by the median local search-band background.
    pub spectral_prominence: f64,
    /// Number of samples in the analysis window.
    pub samples: usize,
}

/// Per-detuning measurement; this is not a phase classification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrequencyLockingPoint {
    pub pulse_detuning_rad: f64,
    pub peak: SubharmonicPeak,
    pub frequency_error: f64,
    pub demodulated_sign_coherence: f64,
    /// Whether the observation duration supports the configured frequency tolerance.
    pub frequency_resolution_sufficient: bool,
    pub is_locked: bool,
}

/// Result of testing finite-sweep frequency locking.
///
/// `observed_locking_span_rad` is the sampled span of the longest contiguous
/// locked run. It is not an asymptotic phase-boundary estimate. A noninteracting
/// perfect-pi-pulse control can pass this frequency-locking diagnostic.
#[derive(Clone, Debug, PartialEq)]
pub struct FrequencyLockingReport {
    pub points: Vec<FrequencyLockingPoint>,
    pub locked_points: usize,
    pub longest_contiguous_points: usize,
    pub observed_locking_span_rad: f64,
    pub frequency_locking_detected: bool,
    pub target_frequency: f64,
    pub frequency_tolerance: f64,
}

/// Estimate the strongest spectral peak inside a band around a target
/// subharmonic frequency.
///
/// The one-sided spectrum is computed after mean removal and Hann windowing.
/// Zero padding improves peak-grid interpolation but does not improve the
/// physical frequency resolution beyond the observation duration.
pub fn estimate_subharmonic_peak(
    signal: &[f64],
    start: usize,
    target_frequency: f64,
    search_half_width: f64,
) -> Option<SubharmonicPeak> {
    if !target_frequency.is_finite()
        || !search_half_width.is_finite()
        || target_frequency <= 0.0
        || target_frequency > 0.5
        || search_half_width <= 0.0
        || start >= signal.len()
    {
        return None;
    }

    let samples = &signal[start..];
    if samples.len() < 32 || samples.iter().any(|value| !value.is_finite()) {
        return None;
    }

    let transform_length = samples.len().checked_mul(4)?.checked_next_power_of_two()?;
    let mut spectrum = vec![Complex::new(0.0, 0.0); transform_length];
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;

    let mut window_sum = 0.0;
    for (index, value) in samples.iter().copied().enumerate() {
        let phase = 2.0 * PI * index as f64 / (samples.len() - 1) as f64;
        let window = 0.5 * (1.0 - phase.cos());
        window_sum += window;
        spectrum[index] = Complex::new((value - mean) * window, 0.0);
    }
    if !window_sum.is_finite() || window_sum <= f64::EPSILON {
        return None;
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(transform_length);
    fft.process(&mut spectrum);

    let lower_frequency = (target_frequency - search_half_width).max(0.0);
    let upper_frequency = (target_frequency + search_half_width).min(0.5);
    let minimum_bin = ((lower_frequency * transform_length as f64).ceil() as usize).max(1);
    let maximum_bin =
        ((upper_frequency * transform_length as f64).floor() as usize).min(transform_length / 2);
    if minimum_bin > maximum_bin {
        return None;
    }

    let mut magnitudes = Vec::with_capacity(maximum_bin - minimum_bin + 1);
    for bin in minimum_bin..=maximum_bin {
        let magnitude = spectrum[bin].norm();
        // Near the Nyquist boundary, the positive and negative Hann-window
        // leakage lobes overlap. Applying the ordinary factor-of-two
        // one-sided correction there double-counts a period-two component.
        // Within one observation-limited frequency scale of Nyquist, use the
        // non-doubled amplitude and accept the finite-window resolution limit.
        let frequency = bin as f64 / transform_length as f64;
        let nyquist_overlap = 0.5 - frequency <= 1.0 / samples.len() as f64;
        let one_sided_scale = if nyquist_overlap { 1.0 } else { 2.0 };
        let amplitude = one_sided_scale * magnitude / window_sum;
        if !amplitude.is_finite() {
            return None;
        }
        magnitudes.push((bin, amplitude));
    }

    let (peak_bin, amplitude) = magnitudes
        .iter()
        .copied()
        .max_by(|left, right| left.1.total_cmp(&right.1))?;

    let mut background: Vec<f64> = magnitudes
        .iter()
        .filter_map(|(bin, value)| {
            if bin.abs_diff(peak_bin) > 2 {
                Some(*value)
            } else {
                None
            }
        })
        .collect();
    background.sort_by(f64::total_cmp);
    let background_median = match background.len() {
        0 => 0.0,
        count if count % 2 == 1 => background[count / 2],
        count => (background[count / 2 - 1] + background[count / 2]) / 2.0,
    };
    let spectral_prominence = amplitude / (background_median + f64::EPSILON);
    let frequency_resolution = 1.0 / transform_length as f64;
    let frequency = peak_bin as f64 * frequency_resolution;

    if !spectral_prominence.is_finite() || !frequency.is_finite() {
        return None;
    }

    Some(SubharmonicPeak {
        frequency,
        frequency_resolution,
        observation_frequency_resolution: 1.0 / samples.len() as f64,
        amplitude,
        spectral_prominence,
        samples: samples.len(),
    })
}

/// Evaluate frequency locking over a strictly ordered pulse-detuning sweep.
///
/// Each point must pass all of the following finite-window checks:
/// - its dominant spectral peak lies within `frequency_tolerance` of the
///   target subharmonic;
/// - its normalized peak amplitude exceeds the configured floor;
/// - its spectral prominence exceeds the configured floor; and
/// - its demodulated sign coherence exceeds the configured floor.
///
/// The sweep-level flag requires a contiguous run of at least
/// `minimum_contiguous_points`. This remains a signal-level diagnostic, not a
/// discrete-time-crystal phase label. Noninteracting and metastable controls
/// must be evaluated separately.
pub fn assess_frequency_locking(
    traces: &[PulseSweepTrace<'_>],
    config: FrequencyLockingConfig,
) -> Result<FrequencyLockingReport, FrequencyLockingError> {
    if traces.is_empty() {
        return Err(FrequencyLockingError::EmptySweep);
    }
    if !config.target_frequency.is_finite()
        || config.target_frequency <= 0.0
        || config.target_frequency > 0.5
        || !config.search_half_width.is_finite()
        || config.search_half_width <= 0.0
        || !config.frequency_tolerance.is_finite()
        || config.frequency_tolerance <= 0.0
        || config.frequency_tolerance > config.search_half_width
        || !config.minimum_peak_amplitude.is_finite()
        || config.minimum_peak_amplitude < 0.0
        || !config.minimum_spectral_prominence.is_finite()
        || config.minimum_spectral_prominence < 0.0
        || !config.minimum_sign_coherence.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_sign_coherence)
        || config.minimum_contiguous_points == 0
    {
        return Err(FrequencyLockingError::InvalidConfiguration);
    }

    let mut points = Vec::with_capacity(traces.len());
    let mut previous_detuning = None;

    for (index, trace) in traces.iter().enumerate() {
        if !trace.pulse_detuning_rad.is_finite() {
            return Err(FrequencyLockingError::NonFiniteDetuning { index });
        }
        if previous_detuning.is_some_and(|previous| trace.pulse_detuning_rad <= previous) {
            return Err(FrequencyLockingError::NonIncreasingDetuning { index });
        }
        previous_detuning = Some(trace.pulse_detuning_rad);

        let peak = estimate_subharmonic_peak(
            trace.signal,
            config.analysis_start,
            config.target_frequency,
            config.search_half_width,
        )
        .ok_or(FrequencyLockingError::InvalidTrace { index })?;
        let response = analyze_subharmonic_response(trace.signal, config.analysis_start)
            .ok_or(FrequencyLockingError::InvalidTrace { index })?;
        let frequency_error = (peak.frequency - config.target_frequency).abs();
        let frequency_resolution_sufficient =
            config.frequency_tolerance >= peak.observation_frequency_resolution;
        let is_locked = frequency_resolution_sufficient
            && frequency_error <= config.frequency_tolerance
            && peak.amplitude >= config.minimum_peak_amplitude
            && peak.spectral_prominence >= config.minimum_spectral_prominence
            && response.sign_coherence >= config.minimum_sign_coherence;

        points.push(FrequencyLockingPoint {
            pulse_detuning_rad: trace.pulse_detuning_rad,
            peak,
            frequency_error,
            demodulated_sign_coherence: response.sign_coherence,
            frequency_resolution_sufficient,
            is_locked,
        });
    }

    let locked_points = points.iter().filter(|point| point.is_locked).count();
    let mut longest_contiguous_points = 0;
    let mut longest_start = 0;
    let mut current_start = None;

    for (index, point) in points.iter().enumerate() {
        if point.is_locked {
            current_start.get_or_insert(index);
        } else if let Some(start) = current_start.take() {
            let length = index - start;
            if length > longest_contiguous_points {
                longest_contiguous_points = length;
                longest_start = start;
            }
        }
    }
    if let Some(start) = current_start {
        let length = points.len() - start;
        if length > longest_contiguous_points {
            longest_contiguous_points = length;
            longest_start = start;
        }
    }

    let observed_locking_span_rad = if longest_contiguous_points >= 2 {
        let first = points[longest_start].pulse_detuning_rad;
        let last = points[longest_start + longest_contiguous_points - 1].pulse_detuning_rad;
        last - first
    } else {
        0.0
    };

    Ok(FrequencyLockingReport {
        points,
        locked_points,
        longest_contiguous_points,
        observed_locking_span_rad,
        frequency_locking_detected: longest_contiguous_points >= config.minimum_contiguous_points,
        target_frequency: config.target_frequency,
        frequency_tolerance: config.frequency_tolerance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stable_period_two_signal(samples: usize, amplitude: f64) -> Vec<f64> {
        (0..samples)
            .map(|period| {
                let oscillation = if period % 2 == 0 { 1.0 } else { -1.0 };
                0.15 + amplitude * oscillation
            })
            .collect()
    }

    fn sinusoid(samples: usize, frequency: f64, amplitude: f64) -> Vec<f64> {
        (0..samples)
            .map(|period| 0.15 + amplitude * (2.0 * PI * frequency * period as f64).cos())
            .collect()
    }

    #[test]
    fn ideal_period_two_trace_has_peak_at_half_drive_frequency() {
        let signal = stable_period_two_signal(512, 0.8);
        let peak = estimate_subharmonic_peak(&signal, 16, 0.5, 0.1).unwrap();

        assert!((peak.frequency - 0.5).abs() <= peak.frequency_resolution);
        assert!((peak.amplitude - 0.8).abs() < 1.0e-6);
        assert!(peak.spectral_prominence > 3.0);
        assert!(peak.frequency_resolution < peak.observation_frequency_resolution);
    }

    #[test]
    fn off_nyquist_amplitude_uses_the_standard_one_sided_correction() {
        let signal = sinusoid(1024, 0.42, 0.7);
        let peak = estimate_subharmonic_peak(&signal, 16, 0.42, 0.08).unwrap();

        assert!((peak.frequency - 0.42).abs() <= peak.observation_frequency_resolution);
        assert!((peak.amplitude - 0.7).abs() < 0.01);
    }

    #[test]
    fn frequency_locking_reports_a_contiguous_detuning_span() {
        let signals: Vec<Vec<f64>> = (0..5).map(|_| stable_period_two_signal(512, 0.8)).collect();
        let detunings = [-0.2, -0.1, 0.0, 0.1, 0.2];
        let traces: Vec<PulseSweepTrace<'_>> = detunings
            .iter()
            .zip(signals.iter())
            .map(|(pulse_detuning_rad, signal)| PulseSweepTrace {
                pulse_detuning_rad: *pulse_detuning_rad,
                signal,
            })
            .collect();

        let report = assess_frequency_locking(&traces, FrequencyLockingConfig::default()).unwrap();
        assert!(report.frequency_locking_detected);
        assert_eq!(report.locked_points, 5);
        assert_eq!(report.longest_contiguous_points, 5);
        assert!((report.observed_locking_span_rad - 0.4).abs() < 1.0e-12);
    }

    #[test]
    fn frequency_drift_prevents_a_finite_sweep_lock() {
        let frequencies = [0.48, 0.49, 0.5, 0.49, 0.48];
        let signals: Vec<Vec<f64>> = frequencies
            .iter()
            .map(|frequency| sinusoid(1024, *frequency, 0.8))
            .collect();
        let detunings = [-0.2, -0.1, 0.0, 0.1, 0.2];
        let traces: Vec<PulseSweepTrace<'_>> = detunings
            .iter()
            .zip(signals.iter())
            .map(|(pulse_detuning_rad, signal)| PulseSweepTrace {
                pulse_detuning_rad: *pulse_detuning_rad,
                signal,
            })
            .collect();

        let report = assess_frequency_locking(
            &traces,
            FrequencyLockingConfig {
                frequency_tolerance: 0.002,
                ..FrequencyLockingConfig::default()
            },
        )
        .unwrap();
        assert!(!report.frequency_locking_detected);
        assert!(report.longest_contiguous_points < 3);
    }

    #[test]
    fn finite_trace_cannot_claim_tolerance_below_observation_resolution() {
        let signal = stable_period_two_signal(64, 0.8);
        let traces = [-0.1, 0.0, 0.1].map(|pulse_detuning_rad| PulseSweepTrace {
            pulse_detuning_rad,
            signal: &signal,
        });
        let report = assess_frequency_locking(
            &traces,
            FrequencyLockingConfig {
                analysis_start: 0,
                frequency_tolerance: 0.005,
                ..FrequencyLockingConfig::default()
            },
        )
        .unwrap();

        assert!(
            report
                .points
                .iter()
                .all(|point| !point.frequency_resolution_sufficient)
        );
        assert!(report.points.iter().all(|point| !point.is_locked));
        assert!(!report.frequency_locking_detected);
    }

    #[test]
    fn ideal_pi_pulse_control_can_frequency_lock_without_proving_a_phase() {
        // A noninteracting perfect-pi-pulse trace alternates exactly. The
        // frequency metric should detect that signal; the result is still
        // only a finite-sweep frequency-locking report, never a DTC phase.
        let signal = stable_period_two_signal(512, 1.0);
        let traces: Vec<PulseSweepTrace<'_>> = [-0.1, 0.0, 0.1]
            .iter()
            .map(|detuning| PulseSweepTrace {
                pulse_detuning_rad: *detuning,
                signal: &signal,
            })
            .collect();
        let report = assess_frequency_locking(&traces, FrequencyLockingConfig::default()).unwrap();

        assert!(report.frequency_locking_detected);
        assert_eq!(report.locked_points, 3);
    }

    #[test]
    fn zero_signal_does_not_pass_frequency_locking_threshold() {
        let signal = vec![0.0; 512];
        let traces: Vec<PulseSweepTrace<'_>> = [-0.1, 0.0, 0.1]
            .iter()
            .map(|detuning| PulseSweepTrace {
                pulse_detuning_rad: *detuning,
                signal: &signal,
            })
            .collect();
        let report = assess_frequency_locking(&traces, FrequencyLockingConfig::default()).unwrap();

        assert_eq!(report.locked_points, 0);
        assert!(!report.frequency_locking_detected);
    }

    #[test]
    fn unordered_detuning_and_invalid_config_fail_closed() {
        let signal = stable_period_two_signal(128, 0.8);
        let unordered = [
            PulseSweepTrace {
                pulse_detuning_rad: 0.1,
                signal: &signal,
            },
            PulseSweepTrace {
                pulse_detuning_rad: 0.0,
                signal: &signal,
            },
        ];
        assert_eq!(
            assess_frequency_locking(&unordered, FrequencyLockingConfig::default()),
            Err(FrequencyLockingError::NonIncreasingDetuning { index: 1 })
        );

        assert_eq!(
            assess_frequency_locking(
                &[unordered[0]],
                FrequencyLockingConfig {
                    frequency_tolerance: 0.2,
                    search_half_width: 0.1,
                    ..FrequencyLockingConfig::default()
                }
            ),
            Err(FrequencyLockingError::InvalidConfiguration)
        );
    }

    #[test]
    fn invalid_trace_is_not_silently_skipped() {
        let invalid = [f64::NAN; 64];
        let traces = [PulseSweepTrace {
            pulse_detuning_rad: 0.0,
            signal: &invalid,
        }];
        assert_eq!(
            assess_frequency_locking(&traces, FrequencyLockingConfig::default()),
            Err(FrequencyLockingError::InvalidTrace { index: 0 })
        );
    }
}
