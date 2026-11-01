//! Integer lookup policy for alpha = 1 - exp(-dt/tau).
//!
//! Input ratios are unsigned Q16.16. The table spans [0, 12] in steps of 1/32,
//! with linear interpolation between Q0.15 samples. Ratios >= 12 saturate to one.
//!
//! Conservative Q15 error bound over non-negative finite ratios:
//! - piecewise-linear interpolation: |f''| <= 1 and h=1/32, so M*h^2/8
//!   contributes at most 4 Q15 LSB;
//! - table rounding contributes at most 0.5 LSB;
//! - flooring dt/tau to Q16.16 contributes <0.5 LSB because |f'| <= 1;
//! - saturation above 12 contributes e^-12 * 32768 < 0.21 LSB.
//! The combined worst-case bound is strictly below 5 Q15 LSB.

use crate::liquid_state::Q15_ONE;

pub const RATIO_Q16_STEP: u32 = 2_048; // 1/32 in Q16.16.
pub const MAX_RATIO_Q16: u32 = 12 * 65_536;
pub const ALPHA_MAX_ERROR_Q15_LSB: u32 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeConstantError {
    ZeroTau,
}

const ALPHA_LUT_Q15: [u16; 385] = [
    0, 1008, 1985, 2932, 3850, 4740, 5602, 6438,
    7248, 8033, 8794, 9532, 10247, 10940, 11611, 12262,
    12893, 13505, 14097, 14672, 15229, 15768, 16291, 16798,
    17289, 17766, 18227, 18675, 19108, 19529, 19936, 20331,
    20713, 21084, 21444, 21792, 22130, 22457, 22774, 23082,
    23380, 23669, 23949, 24220, 24483, 24738, 24985, 25224,
    25456, 25681, 25899, 26111, 26316, 26514, 26707, 26893,
    27074, 27249, 27419, 27583, 27743, 27897, 28047, 28193,
    28333, 28470, 28602, 28730, 28854, 28975, 29092, 29205,
    29314, 29421, 29524, 29623, 29720, 29814, 29905, 29993,
    30078, 30161, 30241, 30319, 30394, 30467, 30538, 30607,
    30673, 30738, 30800, 30861, 30919, 30976, 31031, 31085,
    31137, 31187, 31235, 31283, 31328, 31373, 31416, 31457,
    31497, 31537, 31574, 31611, 31647, 31681, 31715, 31747,
    31778, 31809, 31838, 31867, 31895, 31922, 31948, 31973,
    31997, 32021, 32044, 32066, 32088, 32109, 32129, 32149,
    32168, 32186, 32204, 32222, 32238, 32255, 32270, 32286,
    32301, 32315, 32329, 32342, 32356, 32368, 32381, 32392,
    32404, 32415, 32426, 32437, 32447, 32457, 32466, 32476,
    32485, 32493, 32502, 32510, 32518, 32526, 32533, 32540,
    32547, 32554, 32561, 32567, 32573, 32579, 32585, 32591,
    32596, 32601, 32606, 32611, 32616, 32621, 32625, 32630,
    32634, 32638, 32642, 32646, 32650, 32653, 32657, 32660,
    32664, 32667, 32670, 32673, 32676, 32679, 32682, 32684,
    32687, 32689, 32692, 32694, 32696, 32699, 32701, 32703,
    32705, 32707, 32709, 32710, 32712, 32714, 32716, 32717,
    32719, 32720, 32722, 32723, 32725, 32726, 32727, 32728,
    32730, 32731, 32732, 32733, 32734, 32735, 32736, 32737,
    32738, 32739, 32740, 32741, 32742, 32742, 32743, 32744,
    32745, 32745, 32746, 32747, 32747, 32748, 32749, 32749,
    32750, 32750, 32751, 32751, 32752, 32752, 32753, 32753,
    32754, 32754, 32755, 32755, 32756, 32756, 32756, 32757,
    32757, 32757, 32758, 32758, 32758, 32759, 32759, 32759,
    32759, 32760, 32760, 32760, 32760, 32761, 32761, 32761,
    32761, 32762, 32762, 32762, 32762, 32762, 32762, 32763,
    32763, 32763, 32763, 32763, 32763, 32764, 32764, 32764,
    32764, 32764, 32764, 32764, 32764, 32765, 32765, 32765,
    32765, 32765, 32765, 32765, 32765, 32765, 32765, 32765,
    32766, 32766, 32766, 32766, 32766, 32766, 32766, 32766,
    32766, 32766, 32766, 32766, 32766, 32766, 32766, 32766,
    32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767,
    32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767,
    32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767,
    32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767,
    32767, 32767, 32767, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768,
    32768,
];

/// Approximate alpha from a non-negative unsigned Q16.16 dt/tau ratio.
pub fn alpha_from_ratio_q16(ratio_q16: u32) -> u16 {
    if ratio_q16 >= MAX_RATIO_Q16 {
        return Q15_ONE;
    }
    let index = (ratio_q16 / RATIO_Q16_STEP) as usize;
    let remainder = (ratio_q16 % RATIO_Q16_STEP) as u64;
    let lower = ALPHA_LUT_Q15[index] as u64;
    let upper = ALPHA_LUT_Q15[index + 1] as u64;
    let delta = upper - lower;
    let interpolated = lower
        + (delta * remainder + (RATIO_Q16_STEP as u64 / 2))
            / RATIO_Q16_STEP as u64;
    interpolated.min(Q15_ONE as u64) as u16
}

/// Compute alpha from integer microsecond durations without floating point.
///
/// The ratio is floored to Q16.16 before lookup. A zero tau is always rejected;
/// very large ratios are saturated by alpha_from_ratio_q16.
pub fn alpha_from_dt_tau_us(
    dt_us: u64,
    tau_us: u64,
) -> Result<u16, TimeConstantError> {
    if tau_us == 0 {
        return Err(TimeConstantError::ZeroTau);
    }
    let ratio_q16 = ((dt_us as u128) << 16) / tau_us as u128;
    let capped = ratio_q16.min(MAX_RATIO_Q16 as u128) as u32;
    Ok(alpha_from_ratio_q16(capped))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_and_known_time_ratios_are_bounded() {
        assert_eq!(alpha_from_ratio_q16(0), 0);
        assert_eq!(alpha_from_ratio_q16(65_536), 20_713);
        assert_eq!(alpha_from_ratio_q16(12 * 65_536), Q15_ONE);
        assert_eq!(alpha_from_dt_tau_us(0, 10), Ok(0));
        assert_eq!(alpha_from_dt_tau_us(10, 10), Ok(20_713));
        assert_eq!(alpha_from_dt_tau_us(120, 10), Ok(Q15_ONE));
        assert_eq!(alpha_from_dt_tau_us(1, 0), Err(TimeConstantError::ZeroTau));
    }

    #[test]
    fn lookup_is_monotonic_over_the_supported_domain() {
        let mut previous = 0u16;
        let mut ratio = 0u32;
        while ratio <= MAX_RATIO_Q16 {
            let alpha = alpha_from_ratio_q16(ratio);
            assert!(alpha >= previous);
            previous = alpha;
            ratio += 97;
        }
        assert_eq!(alpha_from_ratio_q16(MAX_RATIO_Q16), Q15_ONE);
    }

    #[test]
    fn integer_lookup_stays_within_five_q15_lsb_of_reference() {
        let mut ratio = 0u32;
        while ratio <= MAX_RATIO_Q16 {
            let estimated = alpha_from_ratio_q16(ratio);
            let x = ratio as f64 / 65_536.0;
            let reference = ((1.0 - (-x).exp()) * Q15_ONE as f64).round();
            let error = (estimated as f64 - reference).abs();
            assert!(
                error <= ALPHA_MAX_ERROR_Q15_LSB as f64,
                "ratio_q16={ratio} error={error}"
            );
            ratio = ratio.saturating_add(97);
        }
        assert_eq!(alpha_from_ratio_q16(u32::MAX), Q15_ONE);
    }

    #[test]
    fn large_microsecond_inputs_do_not_overflow_ratio_math() {
        assert_eq!(alpha_from_dt_tau_us(u64::MAX, 1), Ok(Q15_ONE));
        assert_eq!(alpha_from_dt_tau_us(u64::MAX, u64::MAX), Ok(20_713));
    }
}
