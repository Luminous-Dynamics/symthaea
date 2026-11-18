//! Deterministic integer calibration bridge from categorical log2 logits to ByteCdf.
//!
//! This module does not turn an arbitrary ContinuousHV into probabilities. An
//! upstream readout must first produce one calibrated log2 score per byte
//! symbol (Q4 fixed point), using a separate calibration split. This module
//! applies a fixed-point inverse temperature and converts those scores into a
//! positive, normalized integer CDF without floating-point math.

use crate::{ByteCdf, BYTE_SYMBOLS};

/// Maximum total frequency used by the calibrated byte CDF.
pub const CALIBRATED_CDF_TOTAL: u32 = 1 << 15;
/// One log2 unit represented in Q4.
pub const LOG2_Q4_ONE: i16 = 16;
/// Inverse temperature 1.0 represented in Q16.
pub const INVERSE_TEMPERATURE_Q16_ONE: u32 = 1 << 16;
/// Lowest supported symbol-frequency floor is 1; the upper bound is 127.
pub const MAX_MINIMUM_SYMBOL_FREQUENCY: u32 =
    (CALIBRATED_CDF_TOTAL / BYTE_SYMBOLS as u32) - 1;

// Q24 values for 2^(-fraction/16), rounded to nearest integer. The integer
// exponent is applied as a right shift, so all arithmetic is deterministic.
const EXP2_NEGATIVE_FRACTION_Q24: [u32; 16] = [
    16_777_216, 16_065_917, 15_384_775, 14_732_511, 14_107_901, 13_509_772,
    12_937_002, 12_388_516, 11_863_283, 11_360_319, 10_878_679, 10_417_458,
    9_975_792, 9_552_851, 9_147_842, 8_760_003,
];

/// Invalid fixed-point categorical-CDF calibration settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdfCalibrationError {
    /// Inverse temperature must be nonzero.
    ZeroInverseTemperature,
    /// Minimum frequency must be in 1..=127, leaving probability mass to allocate.
    InvalidMinimumFrequency,
}

impl std::fmt::Display for CdfCalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroInverseTemperature => write!(f, "inverse temperature must be nonzero"),
            Self::InvalidMinimumFrequency => {
                write!(f, "minimum symbol frequency must be in 1..=127")
            }
        }
    }
}

impl std::error::Error for CdfCalibrationError {}

/// Convert calibrated log2 scores into a deterministic, strictly positive byte CDF.
///
/// # Inputs
/// - `log2_logits_q4`: one signed Q4 log2 score for each byte symbol 0..=255.
/// - `inverse_temperature_q16`: positive Q16 scale fitted on a calibration split;
///   65,536 means 1.0, larger values sharpen the distribution.
/// - `minimum_frequency`: guaranteed integer frequency for every symbol, 1..=127.
///
/// # Guarantees
/// - Pure integer implementation; no platform-dependent floating-point exp/softmax.
/// - CDF starts at zero, is strictly increasing, and ends at 32,768.
/// - Every symbol receives at least `minimum_frequency`.
/// - Remaining frequency mass is allocated by largest remainder, with symbol ID
///   as the deterministic tie-break (lower symbol ID wins).
/// - Log2 score differences are clipped at 16 bits before exponentiation.
///
/// This function is only the final integer-normalization step. Readout training,
/// temperature fitting, held-out calibration and model/codebook cost accounting
/// are separate responsibilities of the adapter using it.
pub fn categorical_byte_cdf_from_log2_logits(
    log2_logits_q4: &[i16; BYTE_SYMBOLS],
    inverse_temperature_q16: u32,
    minimum_frequency: u32,
) -> Result<ByteCdf, CdfCalibrationError> {
    if inverse_temperature_q16 == 0 {
        return Err(CdfCalibrationError::ZeroInverseTemperature);
    }
    if !(1..=MAX_MINIMUM_SYMBOL_FREQUENCY).contains(&minimum_frequency) {
        return Err(CdfCalibrationError::InvalidMinimumFrequency);
    }

    let scaled: [i64; BYTE_SYMBOLS] = std::array::from_fn(|index| {
        (log2_logits_q4[index] as i64 * inverse_temperature_q16 as i64) / 65_536
    });
    let max_logit = *scaled.iter().max().unwrap_or(&0);

    let weights: [u32; BYTE_SYMBOLS] = std::array::from_fn(|index| {
        let delta_q4 = (max_logit - scaled[index]).clamp(0, 16 * 16) as usize;
        let integer_part = delta_q4 / 16;
        let fractional_part = delta_q4 % 16;
        (EXP2_NEGATIVE_FRACTION_Q24[fractional_part] >> integer_part).max(1)
    });

    let weight_sum: u64 = weights.iter().map(|&weight| weight as u64).sum();
    let reserved = minimum_frequency as u64 * BYTE_SYMBOLS as u64;
    let distributable = CALIBRATED_CDF_TOTAL as u64 - reserved;

    let mut frequencies = [minimum_frequency; BYTE_SYMBOLS];
    let mut remainders = [0u64; BYTE_SYMBOLS];
    let mut assigned = 0u64;

    for symbol in 0..BYTE_SYMBOLS {
        let scaled_weight = weights[symbol] as u64 * distributable;
        let extra = scaled_weight / weight_sum;
        remainders[symbol] = scaled_weight % weight_sum;
        frequencies[symbol] += extra as u32;
        assigned += extra;
    }

    let leftover = (distributable - assigned) as usize;
    let mut order: [usize; BYTE_SYMBOLS] = std::array::from_fn(|symbol| symbol);
    order.sort_unstable_by(|left, right| {
        remainders[*right]
            .cmp(&remainders[*left])
            .then_with(|| left.cmp(right))
    });
    for &symbol in order.iter().take(leftover) {
        frequencies[symbol] += 1;
    }

    let mut cdf = [0u32; BYTE_SYMBOLS + 1];
    for symbol in 0..BYTE_SYMBOLS {
        cdf[symbol + 1] = cdf[symbol] + frequencies[symbol];
    }
    Ok(cdf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frequencies(cdf: &ByteCdf) -> [u32; BYTE_SYMBOLS] {
        std::array::from_fn(|symbol| cdf[symbol + 1] - cdf[symbol])
    }

    fn assert_valid(cdf: &ByteCdf, minimum_frequency: u32) {
        assert_eq!(cdf[0], 0);
        assert_eq!(cdf[BYTE_SYMBOLS], CALIBRATED_CDF_TOTAL);
        for symbol in 0..BYTE_SYMBOLS {
            assert!(cdf[symbol + 1] > cdf[symbol]);
            assert!(cdf[symbol + 1] - cdf[symbol] >= minimum_frequency);
        }
    }

    #[test]
    fn uniform_logits_produce_uniform_positive_cdf() {
        let logits = [0i16; BYTE_SYMBOLS];
        let cdf = categorical_byte_cdf_from_log2_logits(
            &logits,
            INVERSE_TEMPERATURE_Q16_ONE,
            1,
        )
        .unwrap();
        assert_valid(&cdf, 1);
        assert!(frequencies(&cdf).iter().all(|&frequency| frequency == 128));
    }

    #[test]
    fn larger_logit_receives_at_least_as_much_frequency() {
        let mut logits = [0i16; BYTE_SYMBOLS];
        logits[42] = 16 * 4;
        let cdf = categorical_byte_cdf_from_log2_logits(
            &logits,
            INVERSE_TEMPERATURE_Q16_ONE,
            1,
        )
        .unwrap();
        let freq = frequencies(&cdf);
        assert_valid(&cdf, 1);
        assert!(freq[42] > freq[0]);
        assert!(freq.iter().all(|&frequency| frequency > 0));
    }

    #[test]
    fn higher_inverse_temperature_sharpens_distribution() {
        let mut logits = [0i16; BYTE_SYMBOLS];
        logits[7] = 16;
        let soft = categorical_byte_cdf_from_log2_logits(&logits, 1 << 16, 1).unwrap();
        let sharp = categorical_byte_cdf_from_log2_logits(&logits, 4 << 16, 1).unwrap();
        assert!(frequencies(&sharp)[7] > frequencies(&soft)[7]);
        assert_valid(&soft, 1);
        assert_valid(&sharp, 1);
    }

    #[test]
    fn extreme_logits_and_largest_frequency_floor_remain_valid() {
        let mut logits = [i16::MIN; BYTE_SYMBOLS];
        logits[255] = i16::MAX;
        for floor in [1, 2, 17, MAX_MINIMUM_SYMBOL_FREQUENCY] {
            let cdf = categorical_byte_cdf_from_log2_logits(
                &logits,
                u32::MAX,
                floor,
            )
            .unwrap();
            assert_valid(&cdf, floor);
        }
    }

    #[test]
    fn invalid_calibration_settings_are_rejected() {
        let logits = [0i16; BYTE_SYMBOLS];
        assert_eq!(
            categorical_byte_cdf_from_log2_logits(&logits, 0, 1),
            Err(CdfCalibrationError::ZeroInverseTemperature)
        );
        assert_eq!(
            categorical_byte_cdf_from_log2_logits(&logits, 1 << 16, 0),
            Err(CdfCalibrationError::InvalidMinimumFrequency)
        );
        assert_eq!(
            categorical_byte_cdf_from_log2_logits(
                &logits,
                1 << 16,
                MAX_MINIMUM_SYMBOL_FREQUENCY + 1
            ),
            Err(CdfCalibrationError::InvalidMinimumFrequency)
        );
    }
}
