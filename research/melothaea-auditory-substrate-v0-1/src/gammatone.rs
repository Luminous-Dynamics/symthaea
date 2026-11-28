//! Causal fourth-order complex gammatone-like auditory filterbank.
//!
//! This module is a transparent linear reference model. It is not a complete
//! cochlear model and does not implement level-dependent compression,
//! two-tone suppression, masking, loudness calibration or efferent control.

use std::error::Error;
use std::fmt;
use std::f64::consts::TAU;

/// Errors from the causal gammatone reference filterbank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GammatoneError {
    InvalidConfiguration(&'static str),
    NonFiniteSample(usize),
    SourceTooShort,
    ComputeBudgetExceeded,
    NumericalOverflow,
}

impl fmt::Display for GammatoneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(reason) => {
                write!(formatter, "invalid gammatone configuration: {reason}")
            }
            Self::NonFiniteSample(index) => {
                write!(formatter, "non-finite source sample at index {index}")
            }
            Self::SourceTooShort => write!(formatter, "source is shorter than the analysis frame"),
            Self::ComputeBudgetExceeded => write!(formatter, "gammatone reference compute budget exceeded"),
            Self::NumericalOverflow => write!(formatter, "gammatone filter produced a non-finite value"),
        }
    }
}

impl Error for GammatoneError {}

/// Explicit source-sample support. Intervals are half-open [start, end).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleSupport {
    pub start_sample: usize,
    pub end_sample: usize,
}

/// Configuration for an offline, source-reset gammatone filterbank.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammatoneConfig {
    pub sample_rate_hz: u32,
    pub frame_size: usize,
    pub hop_size: usize,
    pub band_count: usize,
    pub min_frequency_hz: f64,
    pub max_frequency_hz: f64,
}

/// Coefficients for one ERB-spaced band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammatoneBand {
    pub center_frequency_hz: f64,
    pub erb_bandwidth_hz: f64,
    pub pole_radius: f64,
    pub gain_normalization: f64,
}

/// Complex envelope of a band, with independent mean-power measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammatoneBandMeasurement {
    pub mean_envelope_real: f64,
    pub mean_envelope_imaginary: f64,
    pub mean_power: f64,
}

impl GammatoneBandMeasurement {
    /// Magnitude of the mean complex envelope.
    pub fn envelope_magnitude(self) -> f64 {
        self.mean_envelope_real.hypot(self.mean_envelope_imaginary)
    }
}

/// Local averaging support and causal IIR state support are deliberately
/// separate. The latter starts at the declared reset boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct GammatoneFrame {
    pub analysis_support: SampleSupport,
    pub causal_state_support: SampleSupport,
    pub bands: Vec<GammatoneBandMeasurement>,
}

/// Output of the offline gammatone reference filterbank.
#[derive(Debug, Clone, PartialEq)]
pub struct GammatoneFilterbank {
    pub config: GammatoneConfig,
    pub filter_order: usize,
    pub bands: Vec<GammatoneBand>,
    pub frames: Vec<GammatoneFrame>,
}

/// Glasberg-Moore ERB approximation in hertz.
pub fn erb_bandwidth_hz(center_frequency_hz: f64) -> f64 {
    24.7 * (4.37 * center_frequency_hz / 1000.0 + 1.0)
}

/// Common ERB-rate coordinate used to space the filter centres.
pub fn hz_to_erb_rate(frequency_hz: f64) -> f64 {
    21.4 * (1.0 + 0.00437 * frequency_hz).log10()
}

fn erb_rate_to_hz(erb_rate: f64) -> f64 {
    (10.0f64.powf(erb_rate / 21.4) - 1.0) / 0.00437
}

#[derive(Debug, Clone, Copy, Default)]
struct Complex {
    real: f64,
    imaginary: f64,
}

impl Complex {
    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imaginary: self.imaginary + other.imaginary,
        }
    }

    fn multiply(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imaginary * other.imaginary,
            imaginary: self.real * other.imaginary + self.imaginary * other.real,
        }
    }

    fn scale(self, factor: f64) -> Self {
        Self {
            real: self.real * factor,
            imaginary: self.imaginary * factor,
        }
    }
}

/// Compute a causal fourth-order complex gammatone-like filterbank.
///
/// Each band uses H(z) = (1 - r exp(j omega) z^-1)^-4 with
/// r = exp(-2 pi * 1.019 * ERB(fc) / fs). Four cascaded complex resonators
/// yield a cubic-exponential impulse envelope. The output is demodulated to
/// baseband. Gain normalization gives unit complex-envelope magnitude for a
/// steady-state, unit-amplitude, bin-centred sinusoid.
///
/// The IIR state is reset at sample zero. Consequently, frame-local averaging
/// support is not the whole acoustic support: every frame records the complete
/// causal source prefix that may have influenced its state.
pub fn compute_gammatone_filterbank(
    samples: &[f32],
    config: GammatoneConfig,
) -> Result<GammatoneFilterbank, GammatoneError> {
    if config.sample_rate_hz == 0 {
        return Err(GammatoneError::InvalidConfiguration("sample rate must be positive"));
    }
    if config.frame_size == 0 || config.hop_size == 0 {
        return Err(GammatoneError::InvalidConfiguration("frame size and hop size must be positive"));
    }
    if samples.len() < config.frame_size {
        return Err(GammatoneError::SourceTooShort);
    }
    if !(2..=128).contains(&config.band_count) {
        return Err(GammatoneError::InvalidConfiguration("band count must be in 2..=128"));
    }

    let nyquist = f64::from(config.sample_rate_hz) / 2.0;
    if !config.min_frequency_hz.is_finite()
        || !config.max_frequency_hz.is_finite()
        || config.min_frequency_hz <= 0.0
        || config.max_frequency_hz <= config.min_frequency_hz
        || config.max_frequency_hz > nyquist
    {
        return Err(GammatoneError::InvalidConfiguration(
            "frequency range must be finite, positive, ordered and below Nyquist",
        ));
    }
    if let Some((index, _)) = samples.iter().enumerate().find(|(_, sample)| !sample.is_finite()) {
        return Err(GammatoneError::NonFiniteSample(index));
    }

    const FILTER_ORDER: usize = 4;
    const MAX_STAGE_SAMPLE_OPERATIONS: u128 = 250_000_000;
    let estimated_operations =
        samples.len() as u128 * config.band_count as u128 * FILTER_ORDER as u128;
    if estimated_operations > MAX_STAGE_SAMPLE_OPERATIONS {
        return Err(GammatoneError::ComputeBudgetExceeded);
    }

    let erb_min = hz_to_erb_rate(config.min_frequency_hz);
    let erb_max = hz_to_erb_rate(config.max_frequency_hz);
    let erb_step = (erb_max - erb_min) / (config.band_count - 1) as f64;
    let sample_rate = f64::from(config.sample_rate_hz);
    let bands: Vec<GammatoneBand> = (0..config.band_count)
        .map(|index| {
            let center_frequency_hz = erb_rate_to_hz(erb_min + index as f64 * erb_step);
            let bandwidth_hz = erb_bandwidth_hz(center_frequency_hz);
            let pole_radius =
                (-TAU * 1.019 * bandwidth_hz / sample_rate).exp();
            GammatoneBand {
                center_frequency_hz,
                erb_bandwidth_hz: bandwidth_hz,
                pole_radius,
                gain_normalization: 2.0 * (1.0 - pole_radius).powi(FILTER_ORDER as i32),
            }
        })
        .collect();

    if bands.iter().any(|band| {
        !band.center_frequency_hz.is_finite()
            || band.center_frequency_hz <= 0.0
            || !band.erb_bandwidth_hz.is_finite()
            || band.erb_bandwidth_hz <= 0.0
            || !band.pole_radius.is_finite()
            || !(0.0..1.0).contains(&band.pole_radius)
            || !band.gain_normalization.is_finite()
            || band.gain_normalization <= f64::MIN_POSITIVE
    }) {
        return Err(GammatoneError::InvalidConfiguration("unstable or unrepresentable filter coefficients"));
    }

    let frame_starts: Vec<usize> = (0..=samples.len() - config.frame_size)
        .step_by(config.hop_size)
        .collect();
    let mut frames: Vec<GammatoneFrame> = frame_starts
        .iter()
        .map(|&start| GammatoneFrame {
            analysis_support: SampleSupport {
                start_sample: start,
                end_sample: start + config.frame_size,
            },
            causal_state_support: SampleSupport {
                start_sample: 0,
                end_sample: start + config.frame_size,
            },
            bands: Vec::with_capacity(config.band_count),
        })
        .collect();

    for band in &bands {
        let omega = TAU * band.center_frequency_hz / sample_rate;
        let pole = Complex {
            real: band.pole_radius * omega.cos(),
            imaginary: band.pole_radius * omega.sin(),
        };
        let mut states = [Complex::default(); FILTER_ORDER];
        let mut real_prefix = vec![0.0f64; samples.len() + 1];
        let mut imaginary_prefix = vec![0.0f64; samples.len() + 1];
        let mut power_prefix = vec![0.0f64; samples.len() + 1];

        for (sample_index, sample) in samples.iter().enumerate() {
            let input = Complex {
                real: f64::from(*sample),
                imaginary: 0.0,
            };
            let first = input.add(pole.multiply(states[0]));
            states[0] = first;
            let second = first.add(pole.multiply(states[1]));
            states[1] = second;
            let third = second.add(pole.multiply(states[2]));
            states[2] = third;
            let fourth = third.add(pole.multiply(states[3]));
            states[3] = fourth;

            let phase = omega * sample_index as f64;
            let demodulator = Complex {
                real: phase.cos(),
                imaginary: -phase.sin(),
            };
            let envelope = fourth
                .multiply(demodulator)
                .scale(band.gain_normalization);

            real_prefix[sample_index + 1] = real_prefix[sample_index] + envelope.real;
            imaginary_prefix[sample_index + 1] =
                imaginary_prefix[sample_index] + envelope.imaginary;
            power_prefix[sample_index + 1] =
                power_prefix[sample_index] + envelope.real * envelope.real
                    + envelope.imaginary * envelope.imaginary;
        }

        for frame in &mut frames {
            let start = frame.analysis_support.start_sample;
            let end = frame.analysis_support.end_sample;
            let count = config.frame_size as f64;
            let real = (real_prefix[end] - real_prefix[start]) / count;
            let imaginary = (imaginary_prefix[end] - imaginary_prefix[start]) / count;
            let power = (power_prefix[end] - power_prefix[start]) / count;
            if !real.is_finite() || !imaginary.is_finite() || !power.is_finite() || power < 0.0 {
                return Err(GammatoneError::NumericalOverflow);
            }
            frame.bands.push(GammatoneBandMeasurement {
                mean_envelope_real: real,
                mean_envelope_imaginary: imaginary,
                mean_power: power,
            });
        }
    }

    Ok(GammatoneFilterbank {
        config,
        filter_order: FILTER_ORDER,
        bands,
        frames,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        compute_gammatone_filterbank, erb_bandwidth_hz, hz_to_erb_rate,
        GammatoneConfig, GammatoneError,
    };

    fn tone(sample_rate: u32, frequency_hz: f64, length: usize) -> Vec<f32> {
        (0..length)
            .map(|index| {
                (std::f64::consts::TAU * frequency_hz * index as f64
                    / f64::from(sample_rate))
                    .sin() as f32
            })
            .collect()
    }

    fn config(sample_rate_hz: u32, frame_size: usize) -> GammatoneConfig {
        GammatoneConfig {
            sample_rate_hz,
            frame_size,
            hop_size: frame_size / 2,
            band_count: 24,
            min_frequency_hz: 100.0,
            max_frequency_hz: 4000.0,
        }
    }

    #[test]
    fn in_band_tone_recovers_unit_complex_envelope_and_phase() {
        let config = config(8192, 256);
        let low = hz_to_erb_rate(config.min_frequency_hz);
        let high = hz_to_erb_rate(config.max_frequency_hz);
        let target = 12usize;
        let coordinate = low + target as f64 * (high - low)
            / (config.band_count - 1) as f64;
        let center = (10.0f64.powf(coordinate / 21.4) - 1.0) / 0.00437;
        assert!(erb_bandwidth_hz(center) > 0.0);
        let signal = tone(8192, center, 8192 * 2);
        let bank = compute_gammatone_filterbank(&signal, config).unwrap();
        let measurement = bank.frames.last().unwrap().bands[target];

        assert!((measurement.envelope_magnitude() - 1.0).abs() < 2.0e-3);
        assert!((measurement.mean_power - 1.0).abs() < 2.0e-3);
        assert!(measurement.mean_envelope_imaginary < -0.99);
        assert_eq!(bank.filter_order, 4);
    }

    #[test]
    fn centre_band_attenuates_off_frequency_tone() {
        let config = config(8192, 256);
        let low = hz_to_erb_rate(config.min_frequency_hz);
        let high = hz_to_erb_rate(config.max_frequency_hz);
        let coordinate = low + 12.0 * (high - low) / 23.0;
        let center = (10.0f64.powf(coordinate / 21.4) - 1.0) / 0.00437;
        let on = compute_gammatone_filterbank(&tone(8192, center, 8192 * 2), config).unwrap();
        let off = compute_gammatone_filterbank(&tone(8192, center * 1.5, 8192 * 2), config).unwrap();
        let on_power = on.frames.last().unwrap().bands[12].mean_power;
        let off_power = off.frames.last().unwrap().bands[12].mean_power;
        assert!(off_power < on_power * 1.0e-3);
    }

    #[test]
    fn frame_window_and_causal_state_support_are_not_conflated() {
        let bank = compute_gammatone_filterbank(
            &tone(8192, 1000.0, 8192),
            config(8192, 256),
        ).unwrap();
        let frame = bank.frames.last().unwrap();
        assert_eq!(frame.causal_state_support.start_sample, 0);
        assert_eq!(
            frame.causal_state_support.end_sample,
            frame.analysis_support.end_sample
        );
        assert!(frame.analysis_support.start_sample > 0);
        assert_eq!(frame.bands.len(), bank.config.band_count);
    }

    #[test]
    fn invalid_samples_and_nyquist_bounds_reject() {
        let mut signal = tone(8192, 1000.0, 8192);
        signal[9] = f32::NAN;
        assert_eq!(
            compute_gammatone_filterbank(&signal, config(8192, 256)),
            Err(GammatoneError::NonFiniteSample(9))
        );
        assert!(matches!(
            compute_gammatone_filterbank(
                &tone(8192, 1000.0, 8192),
                GammatoneConfig {
                    max_frequency_hz: 5000.0,
                    ..config(8192, 256)
                }
            ),
            Err(GammatoneError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn work_budget_rejects_unbounded_reference_operation() {
        let signal = vec![0.0f32; 1_000_000];
        assert_eq!(
            compute_gammatone_filterbank(
                &signal,
                GammatoneConfig {
                    sample_rate_hz: 8192,
                    frame_size: 256,
                    hop_size: 128,
                    band_count: 128,
                    min_frequency_hz: 100.0,
                    max_frequency_hz: 4000.0,
                }
            ),
            Err(GammatoneError::ComputeBudgetExceeded)
        );
    }
}
