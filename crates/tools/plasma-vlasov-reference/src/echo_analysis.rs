//! Phase-aware prediction and measurement for two-pulse plasma echoes.
//!
//! The kinematic phase-matching relation predicts an echo mode and nominal
//! time. It does not predict a universal echo amplitude or waveform. This
//! module therefore separates pulse geometry from measurement of a sampled
//! complex electric-field mode.
//!
//! Trace measurements are resolution-limited: the reported peak is the largest
//! sampled amplitude, not a sub-sample continuous-time estimate. FWHM is
//! reported only when both half-maximum crossings are bracketed by samples.

/// Two externally prescribed spatial Fourier pulses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EchoPulsePair {
    /// Signed integer Fourier-mode index of the first pulse.
    pub first_mode: i32,
    /// Signed integer Fourier-mode index of the second pulse.
    pub second_mode: i32,
    /// Simulation time of the first pulse.
    pub first_time: f64,
    /// Simulation time of the second pulse.
    pub second_time: f64,
}

/// Kinematic phase-matching prediction for the two-pulse echo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EchoPrediction {
    /// Signed integer spatial Fourier-mode index of the echo.
    pub echo_mode: i32,
    /// Nominal echo time from the phase-matching relation.
    pub echo_time: f64,
}

/// One complex coefficient sample from the echo-mode electric-field trace.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EchoTraceSample {
    pub time: f64,
    pub real: f64,
    pub imaginary: f64,
}

impl EchoTraceSample {
    pub const fn new(time: f64, real: f64, imaginary: f64) -> Self {
        Self {
            time,
            real,
            imaginary,
        }
    }

    pub fn amplitude(self) -> f64 {
        self.real.hypot(self.imaginary)
    }

    /// Principal complex phase in radians, in [-pi, pi].
    pub fn phase(self) -> f64 {
        self.imaginary.atan2(self.real)
    }
}

/// A sampled complex spatial Fourier mode.
///
/// Samples must already be restricted to the preregistered echo measurement
/// window. The mode identity is checked against the prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EchoModeTrace<'a> {
    pub spatial_mode: i32,
    pub samples: &'a [EchoTraceSample],
}

/// Frozen signal floor used to distinguish a detected echo from a trace peak
/// that is below the measurement's declared amplitude floor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EchoMeasurementConfig {
    pub minimum_peak_amplitude: f64,
}

/// Sampled peak measurement for one predicted echo mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EchoMeasurement {
    pub echo_mode: i32,
    /// Time of the largest sampled amplitude; not a sub-sample estimate.
    pub sampled_peak_time: f64,
    pub peak_amplitude: f64,
    /// Principal phase at the sampled peak, in [-pi, pi].
    pub peak_phase_radians: f64,
    pub minimum_peak_amplitude: f64,
    /// False means the largest observed peak did not exceed the frozen floor.
    pub detected: bool,
    /// Largest adjacent sample interval around the peak. This is a resolution
    /// indicator, not a confidence interval or a certified peak-time bound.
    pub peak_sampling_interval: f64,
    /// FWHM is only present when both half-maximum crossings are bracketed.
    pub full_width_half_maximum: Option<f64>,
    pub left_half_maximum_time: Option<f64>,
    pub right_half_maximum_time: Option<f64>,
}

/// Absolute and relative tolerance for nominal echo peak-time qualification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EchoTimeTolerance {
    pub absolute: f64,
    pub relative: f64,
}

/// Invalid pulse geometry, trace data, or qualification parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EchoAnalysisError {
    ZeroPulseMode {
        pulse_index: usize,
    },
    EqualPulseModes,
    ModeDifferenceOverflow,
    NonFinitePulseTime {
        pulse_index: usize,
    },
    UnorderedPulseTimes,
    NonFutureEcho {
        echo_time: f64,
        second_pulse_time: f64,
    },
    WrongEchoMode {
        expected: i32,
        actual: i32,
    },
    TooFewTraceSamples {
        actual: usize,
    },
    NonFiniteTraceSample {
        index: usize,
    },
    NonIncreasingTraceTime {
        index: usize,
    },
    TraceDoesNotSpanEchoTime {
        echo_time: f64,
        first_sample_time: f64,
        last_sample_time: f64,
    },
    InvalidSignalFloor,
    InvalidTimeTolerance,
    EchoNotDetected {
        peak_amplitude: f64,
        minimum_peak_amplitude: f64,
    },
    EchoPeakTimeOutsideTolerance {
        observed: f64,
        predicted: f64,
        allowed: f64,
    },
}

/// Predict the signed echo mode and nominal time for two spatial Fourier pulses.
///
/// For signed integer mode indices k1 and k2:
///
/// - k_echo = k2 - k1
/// - t_echo = (k2*t2 - k1*t1) / (k2-k1)
///
/// The echo must occur strictly after the second pulse. The relation predicts
/// kinematics only; it does not supply an echo amplitude or waveform.
pub fn predict_two_pulse_echo(
    pulses: EchoPulsePair,
) -> Result<EchoPrediction, EchoAnalysisError> {
    if pulses.first_mode == 0 {
        return Err(EchoAnalysisError::ZeroPulseMode { pulse_index: 0 });
    }
    if pulses.second_mode == 0 {
        return Err(EchoAnalysisError::ZeroPulseMode { pulse_index: 1 });
    }
    if pulses.first_mode == pulses.second_mode {
        return Err(EchoAnalysisError::EqualPulseModes);
    }
    if !pulses.first_time.is_finite() {
        return Err(EchoAnalysisError::NonFinitePulseTime { pulse_index: 0 });
    }
    if !pulses.second_time.is_finite() {
        return Err(EchoAnalysisError::NonFinitePulseTime { pulse_index: 1 });
    }
    if pulses.second_time <= pulses.first_time {
        return Err(EchoAnalysisError::UnorderedPulseTimes);
    }

    let echo_mode = pulses
        .second_mode
        .checked_sub(pulses.first_mode)
        .ok_or(EchoAnalysisError::ModeDifferenceOverflow)?;
    let numerator = f64::from(pulses.second_mode) * pulses.second_time
        - f64::from(pulses.first_mode) * pulses.first_time;
    let echo_time = numerator / f64::from(echo_mode);

    if !echo_time.is_finite() || echo_time <= pulses.second_time {
        return Err(EchoAnalysisError::NonFutureEcho {
            echo_time,
            second_pulse_time: pulses.second_time,
        });
    }

    Ok(EchoPrediction {
        echo_mode,
        echo_time,
    })
}

/// Measure the largest sampled complex-mode response in a preregistered window.
///
/// The input trace must have at least three finite samples, strictly increasing
/// times, the predicted signed mode, and must span the predicted echo time.
/// The returned peak is the largest sample, not an interpolated continuous-time
/// maximum. FWHM is omitted unless both crossings are observed.
pub fn measure_echo_trace(
    prediction: EchoPrediction,
    trace: EchoModeTrace<'_>,
    config: EchoMeasurementConfig,
) -> Result<EchoMeasurement, EchoAnalysisError> {
    if trace.spatial_mode != prediction.echo_mode {
        return Err(EchoAnalysisError::WrongEchoMode {
            expected: prediction.echo_mode,
            actual: trace.spatial_mode,
        });
    }
    if trace.samples.len() < 3 {
        return Err(EchoAnalysisError::TooFewTraceSamples {
            actual: trace.samples.len(),
        });
    }
    if !config.minimum_peak_amplitude.is_finite() || config.minimum_peak_amplitude < 0.0 {
        return Err(EchoAnalysisError::InvalidSignalFloor);
    }

    for (index, sample) in trace.samples.iter().copied().enumerate() {
        if !sample.time.is_finite()
            || !sample.real.is_finite()
            || !sample.imaginary.is_finite()
            || !sample.amplitude().is_finite()
        {
            return Err(EchoAnalysisError::NonFiniteTraceSample { index });
        }
        if index > 0 && sample.time <= trace.samples[index - 1].time {
            return Err(EchoAnalysisError::NonIncreasingTraceTime { index });
        }
    }

    let first_sample_time = trace.samples[0].time;
    let last_sample_time = trace.samples[trace.samples.len() - 1].time;
    if prediction.echo_time < first_sample_time || prediction.echo_time > last_sample_time {
        return Err(EchoAnalysisError::TraceDoesNotSpanEchoTime {
            echo_time: prediction.echo_time,
            first_sample_time,
            last_sample_time,
        });
    }

    let mut peak_index = 0usize;
    let mut peak_amplitude = trace.samples[0].amplitude();
    for (index, sample) in trace.samples.iter().copied().enumerate().skip(1) {
        let amplitude = sample.amplitude();
        if amplitude > peak_amplitude {
            peak_amplitude = amplitude;
            peak_index = index;
        }
    }

    let peak = trace.samples[peak_index];
    let left_interval = if peak_index > 0 {
        peak.time - trace.samples[peak_index - 1].time
    } else {
        0.0
    };
    let right_interval = if peak_index + 1 < trace.samples.len() {
        trace.samples[peak_index + 1].time - peak.time
    } else {
        0.0
    };
    let peak_sampling_interval = left_interval.max(right_interval);

    let half_maximum = 0.5 * peak_amplitude;
    let left_crossing = find_left_half_maximum_crossing(trace.samples, peak_index, half_maximum);
    let right_crossing =
        find_right_half_maximum_crossing(trace.samples, peak_index, half_maximum);
    let full_width_half_maximum = match (left_crossing, right_crossing) {
        (Some(left), Some(right)) if right >= left => Some(right - left),
        _ => None,
    };

    Ok(EchoMeasurement {
        echo_mode: prediction.echo_mode,
        sampled_peak_time: peak.time,
        peak_amplitude,
        peak_phase_radians: peak.phase(),
        minimum_peak_amplitude: config.minimum_peak_amplitude,
        detected: peak_amplitude > config.minimum_peak_amplitude,
        peak_sampling_interval,
        full_width_half_maximum,
        left_half_maximum_time: left_crossing,
        right_half_maximum_time: right_crossing,
    })
}

/// Qualify sampled peak timing against the kinematic phase-matching prediction.
///
/// The allowed error is max(absolute_tolerance,
/// relative_tolerance * abs(predicted_time)). A below-floor trace cannot pass.
pub fn qualify_echo_peak_time(
    prediction: EchoPrediction,
    measurement: EchoMeasurement,
    tolerance: EchoTimeTolerance,
) -> Result<(), EchoAnalysisError> {
    if measurement.echo_mode != prediction.echo_mode {
        return Err(EchoAnalysisError::WrongEchoMode {
            expected: prediction.echo_mode,
            actual: measurement.echo_mode,
        });
    }
    if !tolerance.absolute.is_finite()
        || !tolerance.relative.is_finite()
        || tolerance.absolute < 0.0
        || tolerance.relative < 0.0
    {
        return Err(EchoAnalysisError::InvalidTimeTolerance);
    }
    if !measurement.detected {
        return Err(EchoAnalysisError::EchoNotDetected {
            peak_amplitude: measurement.peak_amplitude,
            minimum_peak_amplitude: measurement.minimum_peak_amplitude,
        });
    }

    let allowed = tolerance
        .absolute
        .max(tolerance.relative * prediction.echo_time.abs());
    let observed = measurement.sampled_peak_time;
    if (observed - prediction.echo_time).abs() > allowed {
        return Err(EchoAnalysisError::EchoPeakTimeOutsideTolerance {
            observed,
            predicted: prediction.echo_time,
            allowed,
        });
    }

    Ok(())
}

fn find_left_half_maximum_crossing(
    samples: &[EchoTraceSample],
    peak_index: usize,
    half_maximum: f64,
) -> Option<f64> {
    if peak_index == 0 {
        return None;
    }

    for index in (0..peak_index).rev() {
        let left = samples[index];
        let right = samples[index + 1];
        let left_amplitude = left.amplitude();
        let right_amplitude = right.amplitude();
        if left_amplitude <= half_maximum && right_amplitude >= half_maximum {
            return Some(interpolate_crossing_time(
                left.time,
                right.time,
                left_amplitude,
                right_amplitude,
                half_maximum,
            ));
        }
    }
    None
}

fn find_right_half_maximum_crossing(
    samples: &[EchoTraceSample],
    peak_index: usize,
    half_maximum: f64,
) -> Option<f64> {
    if peak_index + 1 >= samples.len() {
        return None;
    }

    for index in peak_index..(samples.len() - 1) {
        let left = samples[index];
        let right = samples[index + 1];
        let left_amplitude = left.amplitude();
        let right_amplitude = right.amplitude();
        if left_amplitude >= half_maximum && right_amplitude <= half_maximum {
            return Some(interpolate_crossing_time(
                left.time,
                right.time,
                left_amplitude,
                right_amplitude,
                half_maximum,
            ));
        }
    }
    None
}

fn interpolate_crossing_time(
    left_time: f64,
    right_time: f64,
    left_amplitude: f64,
    right_amplitude: f64,
    half_maximum: f64,
) -> f64 {
    let amplitude_span = right_amplitude - left_amplitude;
    if amplitude_span == 0.0 {
        return 0.5 * (left_time + right_time);
    }
    left_time
        + (half_maximum - left_amplitude) * (right_time - left_time) / amplitude_span
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn canonical_prediction() -> EchoPrediction {
        predict_two_pulse_echo(EchoPulsePair {
            first_mode: 1,
            second_mode: 2,
            first_time: 0.0,
            second_time: 1.0,
        })
        .unwrap()
    }

    fn phased_sample(time: f64, amplitude: f64, phase: f64) -> EchoTraceSample {
        EchoTraceSample::new(time, amplitude * phase.cos(), amplitude * phase.sin())
    }

    fn canonical_trace() -> [EchoTraceSample; 5] {
        let phase = -2.4;
        [
            phased_sample(1.0, 0.1, phase),
            phased_sample(1.5, 0.5, phase),
            phased_sample(2.0, 1.0, phase),
            phased_sample(2.5, 0.5, phase),
            phased_sample(3.0, 0.1, phase),
        ]
    }

    fn canonical_measurement() -> EchoMeasurement {
        let trace = canonical_trace();
        measure_echo_trace(
            canonical_prediction(),
            EchoModeTrace {
                spatial_mode: 1,
                samples: &trace,
            },
            EchoMeasurementConfig {
                minimum_peak_amplitude: 0.01,
            },
        )
        .unwrap()
    }

    #[test]
    fn canonical_two_pulse_geometry_predicts_mode_one_at_twice_second_pulse_time() {
        assert_eq!(
            predict_two_pulse_echo(EchoPulsePair {
                first_mode: 1,
                second_mode: 2,
                first_time: 0.0,
                second_time: 1.0,
            }),
            Ok(EchoPrediction {
                echo_mode: 1,
                echo_time: 2.0,
            })
        );
    }

    #[test]
    fn pulse_geometry_supports_nonzero_first_time_and_signed_modes() {
        let positive = predict_two_pulse_echo(EchoPulsePair {
            first_mode: 1,
            second_mode: 2,
            first_time: 0.5,
            second_time: 1.5,
        })
        .unwrap();
        assert_eq!(positive.echo_mode, 1);
        assert_eq!(positive.echo_time, 2.5);

        let negative = predict_two_pulse_echo(EchoPulsePair {
            first_mode: -1,
            second_mode: -2,
            first_time: 0.0,
            second_time: 1.0,
        })
        .unwrap();
        assert_eq!(negative.echo_mode, -1);
        assert_eq!(negative.echo_time, 2.0);
    }

    #[test]
    fn pulse_geometry_rejects_zero_and_equal_modes() {
        assert_eq!(
            predict_two_pulse_echo(EchoPulsePair {
                first_mode: 0,
                second_mode: 2,
                first_time: 0.0,
                second_time: 1.0,
            }),
            Err(EchoAnalysisError::ZeroPulseMode { pulse_index: 0 })
        );
        assert_eq!(
            predict_two_pulse_echo(EchoPulsePair {
                first_mode: 1,
                second_mode: 0,
                first_time: 0.0,
                second_time: 1.0,
            }),
            Err(EchoAnalysisError::ZeroPulseMode { pulse_index: 1 })
        );
        assert_eq!(
            predict_two_pulse_echo(EchoPulsePair {
                first_mode: 1,
                second_mode: 1,
                first_time: 0.0,
                second_time: 1.0,
            }),
            Err(EchoAnalysisError::EqualPulseModes)
        );
    }

    #[test]
    fn pulse_geometry_rejects_invalid_times_and_non_future_echoes() {
        let mut pulses = EchoPulsePair {
            first_mode: 1,
            second_mode: 2,
            first_time: 0.0,
            second_time: 1.0,
        };
        pulses.first_time = f64::NAN;
        assert_eq!(
            predict_two_pulse_echo(pulses),
            Err(EchoAnalysisError::NonFinitePulseTime { pulse_index: 0 })
        );

        pulses.first_time = 0.0;
        pulses.second_time = f64::INFINITY;
        assert_eq!(
            predict_two_pulse_echo(pulses),
            Err(EchoAnalysisError::NonFinitePulseTime { pulse_index: 1 })
        );

        pulses.second_time = -1.0;
        assert_eq!(
            predict_two_pulse_echo(pulses),
            Err(EchoAnalysisError::UnorderedPulseTimes)
        );

        pulses.second_time = 1.0;
        pulses.first_mode = 2;
        pulses.second_mode = 1;
        assert!(matches!(
            predict_two_pulse_echo(pulses),
            Err(EchoAnalysisError::NonFutureEcho { .. })
        ));
    }

    #[test]
    fn pulse_mode_difference_overflow_is_rejected() {
        assert_eq!(
            predict_two_pulse_echo(EchoPulsePair {
                first_mode: i32::MIN,
                second_mode: i32::MAX,
                first_time: 0.0,
                second_time: 1.0,
            }),
            Err(EchoAnalysisError::ModeDifferenceOverflow)
        );
    }

    #[test]
    fn complex_trace_preserves_sampled_peak_amplitude_and_phase() {
        let measurement = canonical_measurement();
        assert_eq!(measurement.echo_mode, 1);
        assert!((measurement.sampled_peak_time - 2.0).abs() < 1.0e-14);
        assert!((measurement.peak_amplitude - 1.0).abs() < 1.0e-14);
        assert!((measurement.peak_phase_radians + 2.4).abs() < 1.0e-14);
        assert!(measurement.detected);
        assert!((measurement.peak_sampling_interval - 0.5).abs() < 1.0e-14);
    }

    #[test]
    fn known_half_maximum_crossings_produce_expected_fwhm() {
        let measurement = canonical_measurement();
        assert_eq!(measurement.left_half_maximum_time, Some(1.5));
        assert_eq!(measurement.right_half_maximum_time, Some(2.5));
        assert_eq!(measurement.full_width_half_maximum, Some(1.0));
    }

    #[test]
    fn complex_phase_is_retained_in_the_negative_real_quadrant() {
        let phase = -3.0;
        let trace = [
            phased_sample(1.0, 0.1, phase),
            phased_sample(1.5, 0.5, phase),
            phased_sample(2.0, 1.0, phase),
            phased_sample(2.5, 0.5, phase),
            phased_sample(3.0, 0.1, phase),
        ];
        let measurement = measure_echo_trace(
            canonical_prediction(),
            EchoModeTrace {
                spatial_mode: 1,
                samples: &trace,
            },
            EchoMeasurementConfig {
                minimum_peak_amplitude: 0.0,
            },
        )
        .unwrap();
        assert!((measurement.peak_phase_radians + 3.0).abs() < 1.0e-14);
    }

    #[test]
    fn wrong_mode_is_rejected() {
        let trace = canonical_trace();
        assert_eq!(
            measure_echo_trace(
                canonical_prediction(),
                EchoModeTrace {
                    spatial_mode: -1,
                    samples: &trace,
                },
                EchoMeasurementConfig {
                    minimum_peak_amplitude: 0.01,
                },
            ),
            Err(EchoAnalysisError::WrongEchoMode {
                expected: 1,
                actual: -1,
            })
        );
    }

    #[test]
    fn non_monotone_duplicate_and_non_finite_samples_are_rejected() {
        let mut trace = canonical_trace();
        trace[3].time = trace[2].time;
        assert_eq!(
            measure_echo_trace(
                canonical_prediction(),
                EchoModeTrace {
                    spatial_mode: 1,
                    samples: &trace,
                },
                EchoMeasurementConfig {
                    minimum_peak_amplitude: 0.01,
                },
            ),
            Err(EchoAnalysisError::NonIncreasingTraceTime { index: 3 })
        );

        let mut trace = canonical_trace();
        trace[2].imaginary = f64::NAN;
        assert_eq!(
            measure_echo_trace(
                canonical_prediction(),
                EchoModeTrace {
                    spatial_mode: 1,
                    samples: &trace,
                },
                EchoMeasurementConfig {
                    minimum_peak_amplitude: 0.01,
                },
            ),
            Err(EchoAnalysisError::NonFiniteTraceSample { index: 2 })
        );
    }

    #[test]
    fn trace_must_span_the_predicted_echo_time() {
        let trace = [
            phased_sample(2.1, 0.1, 0.0),
            phased_sample(2.2, 0.5, 0.0),
            phased_sample(2.3, 1.0, 0.0),
        ];
        assert!(matches!(
            measure_echo_trace(
                canonical_prediction(),
                EchoModeTrace {
                    spatial_mode: 1,
                    samples: &trace,
                },
                EchoMeasurementConfig {
                    minimum_peak_amplitude: 0.01,
                },
            ),
            Err(EchoAnalysisError::TraceDoesNotSpanEchoTime { .. })
        ));
    }

    #[test]
    fn below_floor_peak_is_reported_as_not_detected() {
        let trace = [
            phased_sample(1.0, 0.01, 0.0),
            phased_sample(1.5, 0.03, 0.0),
            phased_sample(2.0, 0.05, 0.0),
            phased_sample(2.5, 0.03, 0.0),
            phased_sample(3.0, 0.01, 0.0),
        ];
        let measurement = measure_echo_trace(
            canonical_prediction(),
            EchoModeTrace {
                spatial_mode: 1,
                samples: &trace,
            },
            EchoMeasurementConfig {
                minimum_peak_amplitude: 0.1,
            },
        )
        .unwrap();
        assert!(!measurement.detected);
        assert_eq!(measurement.peak_amplitude, 0.05);
        assert_eq!(
            qualify_echo_peak_time(
                canonical_prediction(),
                measurement,
                EchoTimeTolerance {
                    absolute: 0.1,
                    relative: 0.01,
                }
            ),
            Err(EchoAnalysisError::EchoNotDetected {
                peak_amplitude: 0.05,
                minimum_peak_amplitude: 0.1,
            })
        );
    }

    #[test]
    fn absent_half_maximum_crossing_does_not_invent_a_width() {
        let trace = [
            phased_sample(1.5, 0.6, 0.0),
            phased_sample(2.0, 1.0, 0.0),
            phased_sample(2.5, 0.7, 0.0),
        ];
        let measurement = measure_echo_trace(
            canonical_prediction(),
            EchoModeTrace {
                spatial_mode: 1,
                samples: &trace,
            },
            EchoMeasurementConfig {
                minimum_peak_amplitude: 0.01,
            },
        )
        .unwrap();
        assert_eq!(measurement.full_width_half_maximum, None);
        assert_eq!(measurement.left_half_maximum_time, None);
        assert_eq!(measurement.right_half_maximum_time, None);
    }

    #[test]
    fn measurement_rejects_too_few_samples_and_invalid_signal_floor() {
        let short = [
            phased_sample(1.5, 0.5, 0.0),
            phased_sample(2.0, 1.0, 0.0),
        ];
        assert_eq!(
            measure_echo_trace(
                canonical_prediction(),
                EchoModeTrace {
                    spatial_mode: 1,
                    samples: &short,
                },
                EchoMeasurementConfig {
                    minimum_peak_amplitude: 0.01,
                },
            ),
            Err(EchoAnalysisError::TooFewTraceSamples { actual: 2 })
        );

        let trace = canonical_trace();
        for floor in [f64::NAN, f64::INFINITY, -0.1] {
            assert_eq!(
                measure_echo_trace(
                    canonical_prediction(),
                    EchoModeTrace {
                        spatial_mode: 1,
                        samples: &trace,
                    },
                    EchoMeasurementConfig {
                        minimum_peak_amplitude: floor,
                    },
                ),
                Err(EchoAnalysisError::InvalidSignalFloor)
            );
        }
    }

    #[test]
    fn peak_time_qualification_uses_maximum_of_absolute_and_relative_tolerance() {
        let measurement = canonical_measurement();
        assert_eq!(
            qualify_echo_peak_time(
                canonical_prediction(),
                measurement,
                EchoTimeTolerance {
                    absolute: 0.05,
                    relative: 0.01,
                }
            ),
            Ok(())
        );

        assert!(matches!(
            qualify_echo_peak_time(
                canonical_prediction(),
                EchoMeasurement {
                    sampled_peak_time: 2.2,
                    ..measurement
                },
                EchoTimeTolerance {
                    absolute: 0.05,
                    relative: 0.01,
                }
            ),
            Err(EchoAnalysisError::EchoPeakTimeOutsideTolerance { .. })
        ));
    }

    #[test]
    fn invalid_peak_time_tolerances_are_rejected() {
        let measurement = canonical_measurement();
        for tolerance in [
            EchoTimeTolerance {
                absolute: -0.1,
                relative: 0.01,
            },
            EchoTimeTolerance {
                absolute: 0.1,
                relative: f64::NAN,
            },
            EchoTimeTolerance {
                absolute: f64::INFINITY,
                relative: 0.0,
            },
        ] {
            assert_eq!(
                qualify_echo_peak_time(canonical_prediction(), measurement, tolerance),
                Err(EchoAnalysisError::InvalidTimeTolerance)
            );
        }
    }

    #[test]
    fn echo_phase_uses_atan2_and_remains_in_principal_interval() {
        let phases = [-PI, -2.7, -1.2, 0.0, 0.8, 2.9];
        for phase in phases {
            let sample = phased_sample(0.0, 1.0, phase);
            let observed = sample.phase();
            assert!(observed >= -PI && observed <= PI);
            assert!((observed - phase).abs() < 1.0e-14);
        }
    }
}
