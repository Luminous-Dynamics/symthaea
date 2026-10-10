// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic, hardware-independent point-reflector phantom for research tests.
//!
//! The generator creates a small synthetic RF trace from idealized round-trip
//! travel times. It assumes a homogeneous medium, constant sound speed, point
//! reflectors, no attenuation, no speckle, no transducer impulse response, and no
//! receive-chain filtering. It is useful for adapter wiring, golden tests and
//! regression checks—not for predicting real probe/image performance.
//!
//! `amplitude` is a normalized signed reflection coefficient in [-1, 1]. Each
//! echo is a Gaussian-windowed carrier centered on its analytic round-trip time.
//! The envelope width is a simple fixture parameterization, not a calibrated
//! acoustic pulse model. The Nyquist check is necessary only; the pulse envelope
//! has sidebands and a real acquisition chain may require substantially more rate.

use std::f64::consts::TAU;
use std::fmt;

/// Invalid input or bounded-resource failure in the deterministic phantom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhantomError {
    NonFinite(&'static str),
    NonPositive(&'static str),
    EmptyPhantom,
    ReflectorAmplitudeOutOfRange,
    SampleRateBelowNyquist,
    EchoOutsideTrace,
    SampleBudgetZero,
    SampleCountExceedsBudget,
    WorkBudgetZero,
    SampleReflectorProductsExceedBudget,
    WorkCountOverflow,
    DerivedValueNonFinite(&'static str),
    SerializationSizeOverflow,
    AllocationFailed,
}

impl fmt::Display for PhantomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(name) => write!(f, "{name} must be finite"),
            Self::NonPositive(name) => write!(f, "{name} must be greater than zero"),
            Self::EmptyPhantom => write!(f, "phantom must contain at least one reflector"),
            Self::ReflectorAmplitudeOutOfRange => {
                write!(f, "reflector amplitude must be within [-1, 1]")
            }
            Self::SampleRateBelowNyquist => {
                write!(f, "sample rate is below the carrier's mathematical Nyquist minimum")
            }
            Self::EchoOutsideTrace => {
                write!(f, "a reflector echo falls outside the requested acquisition window")
            }
            Self::SampleBudgetZero => write!(f, "max_samples must be greater than zero"),
            Self::SampleCountExceedsBudget => {
                write!(f, "requested trace exceeds the caller-supplied sample budget")
            }
            Self::WorkBudgetZero => {
                write!(f, "max_sample_reflector_products must be greater than zero")
            }
            Self::SampleReflectorProductsExceedBudget => {
                write!(f, "trace generation exceeds the caller-supplied work budget")
            }
            Self::WorkCountOverflow => write!(f, "sample-reflector work count overflowed"),
            Self::DerivedValueNonFinite(name) => {
                write!(f, "derived value {name} is not finite")
            }
            Self::SerializationSizeOverflow => {
                write!(f, "canonical serialization size overflowed")
            }
            Self::AllocationFailed => write!(f, "could not reserve memory for synthetic trace"),
        }
    }
}

impl std::error::Error for PhantomError {}

fn positive_finite(value: f64, name: &'static str) -> Result<(), PhantomError> {
    if !value.is_finite() {
        return Err(PhantomError::NonFinite(name));
    }
    if value <= 0.0 {
        return Err(PhantomError::NonPositive(name));
    }
    Ok(())
}

/// An idealized point reflector at a positive depth from the transducer surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointReflector {
    depth_m: f64,
    amplitude: f64,
}

impl PointReflector {
    /// Construct a reflector with depth in metres and signed normalized amplitude.
    pub fn new(depth_m: f64, amplitude: f64) -> Result<Self, PhantomError> {
        positive_finite(depth_m, "depth_m")?;
        if !amplitude.is_finite() {
            return Err(PhantomError::NonFinite("amplitude"));
        }
        if !(-1.0..=1.0).contains(&amplitude) {
            return Err(PhantomError::ReflectorAmplitudeOutOfRange);
        }
        Ok(Self { depth_m, amplitude })
    }

    pub fn depth_m(&self) -> f64 {
        self.depth_m
    }

    pub fn amplitude(&self) -> f64 {
        self.amplitude
    }
}

/// Homogeneous-medium phantom containing one or more idealized point reflectors.
#[derive(Debug, Clone, PartialEq)]
pub struct PointReflectorPhantom {
    sound_speed_m_s: f64,
    reflectors: Vec<PointReflector>,
}

impl PointReflectorPhantom {
    pub fn new(
        sound_speed_m_s: f64,
        mut reflectors: Vec<PointReflector>,
    ) -> Result<Self, PhantomError> {
        positive_finite(sound_speed_m_s, "sound_speed_m_s")?;
        if reflectors.is_empty() {
            return Err(PhantomError::EmptyPhantom);
        }

        // Canonical order keeps summation stable when callers provide the same
        // reflectors in a different order.
        reflectors.sort_by(|a, b| {
            a.depth_m
                .total_cmp(&b.depth_m)
                .then_with(|| a.amplitude.total_cmp(&b.amplitude))
        });

        Ok(Self {
            sound_speed_m_s,
            reflectors,
        })
    }

    pub fn sound_speed_m_s(&self) -> f64 {
        self.sound_speed_m_s
    }

    pub fn reflectors(&self) -> &[PointReflector] {
        &self.reflectors
    }

    /// Analytic pulse-echo round-trip delay: t = 2d/c.
    pub fn echo_time_s(&self, depth_m: f64) -> Result<f64, PhantomError> {
        positive_finite(depth_m, "depth_m")?;
        let time = 2.0 * depth_m / self.sound_speed_m_s;
        if !time.is_finite() || time <= 0.0 {
            return Err(PhantomError::DerivedValueNonFinite("echo_time_s"));
        }
        Ok(time)
    }

    /// Generate a deterministic synthetic RF trace and retain analytic echo truth.
    ///
    /// `duration_s` and `sample_rate_hz` are explicit. `max_samples` bounds the
    /// sample buffer, while `max_sample_reflector_products` bounds the nested
    /// sample-by-reflector synthesis loop. Both limits are checked before large
    /// allocations or sample generation. The sample rate must meet 2 * center
    /// frequency, a necessary but not sufficient condition for a broadband pulse.
    pub fn simulate_rf_trace(
        &self,
        center_frequency_hz: f64,
        sample_rate_hz: f64,
        pulse_cycles: f64,
        duration_s: f64,
        max_samples: usize,
        max_sample_reflector_products: usize,
    ) -> Result<SyntheticRfTrace, PhantomError> {
        positive_finite(center_frequency_hz, "center_frequency_hz")?;
        positive_finite(sample_rate_hz, "sample_rate_hz")?;
        positive_finite(pulse_cycles, "pulse_cycles")?;
        positive_finite(duration_s, "duration_s")?;
        if max_samples == 0 {
            return Err(PhantomError::SampleBudgetZero);
        }

        let minimum_sample_rate = 2.0 * center_frequency_hz;
        if !minimum_sample_rate.is_finite() {
            return Err(PhantomError::DerivedValueNonFinite(
                "minimum_sample_rate_hz",
            ));
        }
        if sample_rate_hz < minimum_sample_rate {
            return Err(PhantomError::SampleRateBelowNyquist);
        }

        let requested_samples = (duration_s * sample_rate_hz).ceil();
        if !requested_samples.is_finite() || requested_samples < 1.0 {
            return Err(PhantomError::DerivedValueNonFinite("sample_count"));
        }
        // Reject values at the float representation of usize::MAX as well:
        // rounding there cannot safely distinguish usize::MAX from overflow.
        if requested_samples >= usize::MAX as f64
            || requested_samples > max_samples as f64
        {
            return Err(PhantomError::SampleCountExceedsBudget);
        }
        let sample_count = requested_samples as usize;

        if max_sample_reflector_products == 0 {
            return Err(PhantomError::WorkBudgetZero);
        }
        let sample_reflector_products = sample_count
            .checked_mul(self.reflectors.len())
            .ok_or(PhantomError::WorkCountOverflow)?;
        if sample_reflector_products > max_sample_reflector_products {
            return Err(PhantomError::SampleReflectorProductsExceedBudget);
        }

        let sigma_s = pulse_cycles / (2.0 * center_frequency_hz);
        let phase_scale = TAU * center_frequency_hz;
        if !sigma_s.is_finite()
            || sigma_s <= 0.0
            || !phase_scale.is_finite()
            || phase_scale <= 0.0
        {
            return Err(PhantomError::DerivedValueNonFinite("pulse_shape_parameters"));
        }

        let mut expected_echoes = Vec::new();
        expected_echoes
            .try_reserve_exact(self.reflectors.len())
            .map_err(|_| PhantomError::AllocationFailed)?;
        for reflector in &self.reflectors {
            let arrival_time_s = self.echo_time_s(reflector.depth_m)?;
            if arrival_time_s >= duration_s {
                return Err(PhantomError::EchoOutsideTrace);
            }
            let fractional_sample_index = arrival_time_s * sample_rate_hz;
            if !fractional_sample_index.is_finite() {
                return Err(PhantomError::DerivedValueNonFinite(
                    "fractional_sample_index",
                ));
            }
            expected_echoes.push(ExpectedEcho {
                depth_m: reflector.depth_m,
                amplitude: reflector.amplitude,
                arrival_time_s,
                fractional_sample_index,
            });
        }

        let mut samples = Vec::new();
        samples
            .try_reserve_exact(sample_count)
            .map_err(|_| PhantomError::AllocationFailed)?;
        samples.resize(sample_count, 0.0);

        for (index, sample) in samples.iter_mut().enumerate() {
            let time_s = index as f64 / sample_rate_hz;
            let mut value = 0.0;
            for echo in &expected_echoes {
                let offset_s = time_s - echo.arrival_time_s;
                let normalized_offset = offset_s / sigma_s;
                let envelope = (-0.5 * normalized_offset * normalized_offset).exp();
                let phase = phase_scale * offset_s;
                if !phase.is_finite() {
                    return Err(PhantomError::DerivedValueNonFinite("carrier_phase"));
                }
                value += echo.amplitude * envelope * phase.cos();
                if !value.is_finite() {
                    return Err(PhantomError::DerivedValueNonFinite("rf_sample"));
                }
            }
            *sample = value;
        }

        Ok(SyntheticRfTrace {
            sound_speed_m_s: self.sound_speed_m_s,
            center_frequency_hz,
            sample_rate_hz,
            pulse_cycles,
            duration_s,
            samples,
            expected_echoes,
        })
    }
}

/// Analytic ground truth for one idealized echo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpectedEcho {
    depth_m: f64,
    amplitude: f64,
    arrival_time_s: f64,
    fractional_sample_index: f64,
}

impl ExpectedEcho {
    pub fn depth_m(&self) -> f64 {
        self.depth_m
    }

    pub fn amplitude(&self) -> f64 {
        self.amplitude
    }

    pub fn arrival_time_s(&self) -> f64 {
        self.arrival_time_s
    }

    /// Exact arrival location in sample coordinates; generally not an integer.
    pub fn fractional_sample_index(&self) -> f64 {
        self.fractional_sample_index
    }
}

/// A deterministic synthetic RF trace with retained analytic echo locations.
fn push_canonical_f64_le(bytes: &mut Vec<u8>, value: f64) {
    // IEEE-754 has distinct +0 and -0 bit patterns. They are semantically
    // equivalent for this fixture, so serialize both as positive zero.
    let normalized = if value == 0.0 { 0.0 } else { value };
    bytes.extend_from_slice(&normalized.to_le_bytes());
}

#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticRfTrace {
    sound_speed_m_s: f64,
    center_frequency_hz: f64,
    sample_rate_hz: f64,
    pulse_cycles: f64,
    duration_s: f64,
    samples: Vec<f64>,
    expected_echoes: Vec<ExpectedEcho>,
}

impl SyntheticRfTrace {
    pub fn sound_speed_m_s(&self) -> f64 {
        self.sound_speed_m_s
    }

    pub fn center_frequency_hz(&self) -> f64 {
        self.center_frequency_hz
    }

    pub fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_hz
    }

    pub fn pulse_cycles(&self) -> f64 {
        self.pulse_cycles
    }

    pub fn duration_s(&self) -> f64 {
        self.duration_s
    }

    pub fn samples(&self) -> &[f64] {
        &self.samples
    }

    pub fn expected_echoes(&self) -> &[ExpectedEcho] {
        &self.expected_echoes
    }

    /// Encode this synthetic fixture in a versioned, canonical little-endian format.
    ///
    /// The byte stream includes the format magic, generation parameters, counts,
    /// analytic echo truth and all RF samples, so a content digest commits to both
    /// observed data and the parameters needed to interpret it. This format is for
    /// synthetic research fixtures only; it is not DICOM or a real-device wire format.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, PhantomError> {
        const MAGIC: &[u8; 8] = b"SYMRF001";
        const HEADER_BYTES: usize = 8 + (5 * 8) + (2 * 8);
        const ECHO_BYTES: usize = 4 * 8;
        const SAMPLE_BYTES: usize = 8;

        let echo_bytes = self
            .expected_echoes
            .len()
            .checked_mul(ECHO_BYTES)
            .ok_or(PhantomError::SerializationSizeOverflow)?;
        let sample_bytes = self
            .samples
            .len()
            .checked_mul(SAMPLE_BYTES)
            .ok_or(PhantomError::SerializationSizeOverflow)?;
        let total_bytes = HEADER_BYTES
            .checked_add(echo_bytes)
            .and_then(|n| n.checked_add(sample_bytes))
            .ok_or(PhantomError::SerializationSizeOverflow)?;

        let echo_count = u64::try_from(self.expected_echoes.len())
            .map_err(|_| PhantomError::SerializationSizeOverflow)?;
        let sample_count = u64::try_from(self.samples.len())
            .map_err(|_| PhantomError::SerializationSizeOverflow)?;

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(total_bytes)
            .map_err(|_| PhantomError::AllocationFailed)?;
        bytes.extend_from_slice(MAGIC);
        for value in [
            self.sound_speed_m_s,
            self.center_frequency_hz,
            self.sample_rate_hz,
            self.pulse_cycles,
            self.duration_s,
        ] {
            push_canonical_f64_le(&mut bytes, value);
        }
        bytes.extend_from_slice(&sample_count.to_le_bytes());
        bytes.extend_from_slice(&echo_count.to_le_bytes());
        for echo in &self.expected_echoes {
            for value in [
                echo.depth_m,
                echo.amplitude,
                echo.arrival_time_s,
                echo.fractional_sample_index,
            ] {
                push_canonical_f64_le(&mut bytes, value);
            }
        }
        for sample in &self.samples {
            push_canonical_f64_le(&mut bytes, *sample);
        }
        debug_assert_eq!(bytes.len(), total_bytes);
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_reflector_phantom() -> PointReflectorPhantom {
        PointReflectorPhantom::new(
            1_540.0,
            vec![PointReflector::new(0.01, 0.5).unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn one_centimeter_reflector_has_known_round_trip_delay() {
        let phantom = single_reflector_phantom();
        let delay = phantom.echo_time_s(0.01).unwrap();
        assert!((delay - (2.0 * 0.01 / 1_540.0)).abs() < 1e-15);
        assert!((delay * 1_000_000.0 - 12.987_012_987).abs() < 1e-8);
    }

    #[test]
    fn synthetic_trace_is_deterministic_and_retains_analytic_truth() {
        let phantom = single_reflector_phantom();
        let first = phantom
            .simulate_rf_trace(5_000_000.0, 20_000_000.0, 2.0, 25e-6, 1_000, 10_000)
            .unwrap();
        let second = phantom
            .simulate_rf_trace(5_000_000.0, 20_000_000.0, 2.0, 25e-6, 1_000, 10_000)
            .unwrap();

        assert_eq!(first, second);
        assert_eq!(first.samples().len(), 500);
        assert_eq!(first.expected_echoes().len(), 1);
        assert!(
            (first.expected_echoes()[0].fractional_sample_index() - 259.740_259_74).abs()
                < 1e-7
        );

        let peak_index = first
            .samples()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(index, _)| index)
            .unwrap();
        assert!((peak_index as f64 - first.expected_echoes()[0].fractional_sample_index()).abs()
            <= 1.0);
    }

    #[test]
    fn canonical_bytes_bind_parameters_ground_truth_and_samples_deterministically() {
        let phantom = single_reflector_phantom();
        let trace = phantom
            .simulate_rf_trace(5_000_000.0, 20_000_000.0, 2.0, 25e-6, 1_000, 10_000)
            .unwrap();
        let first = trace.canonical_bytes().unwrap();
        let second = trace.canonical_bytes().unwrap();

        assert_eq!(&first[..8], b"SYMRF001");
        assert_eq!(first.len(), 64 + 32 + 500 * 8);
        assert_eq!(first, second);

        let changed_reflector = PointReflectorPhantom::new(
            1_540.0,
            vec![PointReflector::new(0.01, 0.4).unwrap()],
        )
        .unwrap();
        let changed_trace = changed_reflector
            .simulate_rf_trace(5_000_000.0, 20_000_000.0, 2.0, 25e-6, 1_000, 10_000)
            .unwrap();
        assert_ne!(first, changed_trace.canonical_bytes().unwrap());

        let positive_zero = PointReflectorPhantom::new(
            1_540.0,
            vec![PointReflector::new(0.01, 0.0).unwrap()],
        )
        .unwrap();
        let negative_zero = PointReflectorPhantom::new(
            1_540.0,
            vec![PointReflector::new(0.01, -0.0).unwrap()],
        )
        .unwrap();
        let positive_trace = positive_zero
            .simulate_rf_trace(5_000_000.0, 20_000_000.0, 2.0, 25e-6, 1_000, 10_000)
            .unwrap();
        let negative_trace = negative_zero
            .simulate_rf_trace(5_000_000.0, 20_000_000.0, 2.0, 25e-6, 1_000, 10_000)
            .unwrap();
        assert_eq!(
            positive_trace.canonical_bytes().unwrap(),
            negative_trace.canonical_bytes().unwrap()
        );
    }

    #[test]
    fn reflector_order_does_not_change_the_trace() {
        let a = PointReflector::new(0.01, 0.5).unwrap();
        let b = PointReflector::new(0.02, -0.25).unwrap();
        let first = PointReflectorPhantom::new(1_540.0, vec![a, b]).unwrap();
        let second = PointReflectorPhantom::new(1_540.0, vec![b, a]).unwrap();

        let trace_a = first
            .simulate_rf_trace(5_000_000.0, 20_000_000.0, 2.0, 40e-6, 1_000, 10_000)
            .unwrap();
        let trace_b = second
            .simulate_rf_trace(5_000_000.0, 20_000_000.0, 2.0, 40e-6, 1_000, 10_000)
            .unwrap();
        assert_eq!(trace_a, trace_b);
    }

    #[test]
    fn enforces_the_sample_reflector_work_budget_before_synthesis() {
        let phantom = PointReflectorPhantom::new(
            1_540.0,
            vec![
                PointReflector::new(0.01, 0.5).unwrap(),
                PointReflector::new(0.02, 0.25).unwrap(),
            ],
        )
        .unwrap();

        // 500 samples times two reflectors = 1,000 synthesis operations.
        assert_eq!(
            phantom.simulate_rf_trace(
                5_000_000.0,
                20_000_000.0,
                2.0,
                25e-6,
                1_000,
                999,
            ),
            Err(PhantomError::SampleReflectorProductsExceedBudget)
        );
        assert_eq!(
            phantom.simulate_rf_trace(
                5_000_000.0,
                20_000_000.0,
                2.0,
                25e-6,
                1_000,
                0,
            ),
            Err(PhantomError::WorkBudgetZero)
        );
    }

    #[test]
    fn rejects_invalid_reflector_and_phantom_inputs() {
        assert_eq!(
            PointReflector::new(0.0, 0.5),
            Err(PhantomError::NonPositive("depth_m"))
        );
        assert_eq!(
            PointReflector::new(0.01, 1.1),
            Err(PhantomError::ReflectorAmplitudeOutOfRange)
        );
        assert_eq!(
            PointReflectorPhantom::new(1_540.0, vec![]),
            Err(PhantomError::EmptyPhantom)
        );
        assert_eq!(
            PointReflectorPhantom::new(f64::NAN, vec![PointReflector::new(0.01, 0.5).unwrap()]),
            Err(PhantomError::NonFinite("sound_speed_m_s"))
        );
    }

    #[test]
    fn fail_closed_on_nyquist_window_budget_and_non_finite_parameters() {
        let phantom = single_reflector_phantom();
        assert_eq!(
            phantom.simulate_rf_trace(5_000_000.0, 9_000_000.0, 2.0, 25e-6, 1_000, 10_000),
            Err(PhantomError::SampleRateBelowNyquist)
        );
        assert_eq!(
            phantom.simulate_rf_trace(5_000_000.0, 20_000_000.0, 2.0, 25e-6, 100, 10_000),
            Err(PhantomError::SampleCountExceedsBudget)
        );
        assert_eq!(
            phantom.simulate_rf_trace(5_000_000.0, 20_000_000.0, 2.0, 10e-6, 1_000, 10_000),
            Err(PhantomError::EchoOutsideTrace)
        );
        assert_eq!(
            phantom.simulate_rf_trace(5_000_000.0, f64::INFINITY, 2.0, 25e-6, 1_000, 10_000),
            Err(PhantomError::NonFinite("sample_rate_hz"))
        );
    }
}
