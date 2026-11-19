//! Deterministic prefix-only segmentation from leading-zero counts of XOR residuals.
//!
//! This is a WZS-inspired research control, not a reproduction of WZS. It emits
//! boundaries only from the decoded prefix, uses integer statistics, and enforces
//! declared minimum/maximum segment lengths. No boundary labels are transmitted.

use std::collections::VecDeque;
use std::fmt;

/// Parameters for the prefix-only LZV segmentation control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LzvConfig {
    /// Number of most recent LZ counts in the fast window.
    pub fast_window: usize,
    /// Number of earlier LZ counts in the reference window.
    pub slow_window: usize,
    /// Minimum length of every interior segment.
    pub min_segment: usize,
    /// Maximum permitted segment length.
    pub max_segment: usize,
    /// Number of observations after a detected split during which new alarms are ignored.
    pub cooldown: usize,
    /// Minimum absolute difference between fast/slow mean LZ counts.
    pub mean_shift: u8,
    /// Minimum absolute difference between fast/slow population variances.
    pub variance_shift: u16,
}

impl Default for LzvConfig {
    fn default() -> Self {
        Self {
            fast_window: 8,
            slow_window: 16,
            min_segment: 32,
            max_segment: 512,
            cooldown: 32,
            mean_shift: 8,
            variance_shift: 256,
        }
    }
}

/// Validation or execution errors for LZV segmentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LzvError {
    InvalidWindow,
    InvalidSegmentBounds,
    ZeroShiftThreshold,
}

impl fmt::Display for LzvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWindow => write!(f, "LZV windows must each be in 2..=4096"),
            Self::InvalidSegmentBounds => write!(
                f,
                "LZV segment bounds require 1 <= min_segment <= max_segment <= 1000000"
            ),
            Self::ZeroShiftThreshold => write!(f, "LZV shift thresholds must be positive"),
        }
    }
}

impl std::error::Error for LzvError {}

#[derive(Debug, Clone, Copy)]
struct WindowStats {
    sum: u128,
    sum_sq: u128,
    count: u128,
}

fn window_stats(history: &VecDeque<u8>, start: usize, count: usize) -> WindowStats {
    let mut sum = 0u128;
    let mut sum_sq = 0u128;
    for index in start..start + count {
        let value = u128::from(history[index]);
        sum += value;
        sum_sq += value * value;
    }
    WindowStats {
        sum,
        sum_sq,
        count: count as u128,
    }
}

fn changed_distribution(fast: WindowStats, slow: WindowStats, config: LzvConfig) -> bool {
    let mean_left = fast.sum * slow.count;
    let mean_right = slow.sum * fast.count;
    let mean_delta = mean_left.abs_diff(mean_right);
    let mean_shift = mean_delta
        >= u128::from(config.mean_shift) * fast.count * slow.count;

    // Population variance is represented as the exact rational:
    // (count * sum(x^2) - sum(x)^2) / count^2.
    let fast_variance_numerator =
        fast.count * fast.sum_sq - fast.sum * fast.sum;
    let slow_variance_numerator =
        slow.count * slow.sum_sq - slow.sum * slow.sum;
    let variance_left =
        fast_variance_numerator * slow.count * slow.count;
    let variance_right =
        slow_variance_numerator * fast.count * fast.count;
    let variance_delta = variance_left.abs_diff(variance_right);
    let variance_shift = variance_delta
        >= u128::from(config.variance_shift)
            * fast.count
            * fast.count
            * slow.count
            * slow.count;

    mean_shift || variance_shift
}

/// Calculate deterministic segment boundaries from exact IEEE-754 bit patterns.
///
/// The input is a sequence of raw u64 value bit patterns. LZ counts are computed
/// from consecutive XOR residuals; the function never converts values to floating
/// point. The returned vector always begins with zero and ends with the input length.
/// An empty input returns [0].
pub fn segment_lzv(values: &[u64], config: LzvConfig) -> Result<Vec<usize>, LzvError> {
    validate_config(config)?;
    let mut boundaries = vec![0usize];
    if values.is_empty() {
        return Ok(boundaries);
    }

    let history_limit = config.fast_window + config.slow_window;
    let mut history = VecDeque::<u8>::with_capacity(history_limit);
    let mut segment_start = 0usize;
    let mut cooldown_remaining = 0usize;
    let mut pending_change = false;

    for index in 1..values.len() {
        let residual = values[index] ^ values[index - 1];
        history.push_back(residual.leading_zeros() as u8);
        if history.len() > history_limit {
            let _ = history.pop_front();
        }

        let next_index = index + 1;
        let current_length = next_index - segment_start;
        let forced_max_boundary = current_length >= config.max_segment;

        if !forced_max_boundary
            && cooldown_remaining == 0
            && history.len() == history_limit
        {
            let fast = window_stats(&history, config.slow_window, config.fast_window);
            let slow = window_stats(&history, 0, config.slow_window);
            // Latch before the minimum boundary is legal: otherwise the fast
            // window can adapt and the shift can disappear before it is emitted.
            pending_change |= changed_distribution(fast, slow, config);
        }

        let change_boundary = current_length >= config.min_segment && pending_change;

        if forced_max_boundary || change_boundary {
            if next_index < values.len() {
                boundaries.push(next_index);
            }
            segment_start = next_index;
            pending_change = false;
            // Max-length splits are policy boundaries, not change alarms.
            // They must not mask a regime change immediately afterward.
            cooldown_remaining = if change_boundary {
                config.cooldown
            } else {
                0
            };
        } else if cooldown_remaining > 0 {
            cooldown_remaining -= 1;
        }
    }

    if boundaries.last().copied() != Some(values.len()) {
        boundaries.push(values.len());
    }
    Ok(boundaries)
}

fn validate_config(config: LzvConfig) -> Result<(), LzvError> {
    if !(2..=4096).contains(&config.fast_window)
        || !(2..=4096).contains(&config.slow_window)
    {
        return Err(LzvError::InvalidWindow);
    }
    if config.min_segment == 0
        || config.max_segment < config.min_segment
        || config.max_segment > 1_000_000
    {
        return Err(LzvError::InvalidSegmentBounds);
    }
    if config.mean_shift == 0 || config.variance_shift == 0 {
        return Err(LzvError::ZeroShiftThreshold);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_stream(count: usize, seed: u64) -> Vec<u64> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                state.wrapping_mul(0x2545_f491_4f6c_dd1d)
            })
            .collect()
    }

    fn check_bounds(boundaries: &[usize], count: usize, config: LzvConfig) {
        assert_eq!(boundaries.first(), Some(&0));
        assert_eq!(boundaries.last(), Some(&count));
        for pair in boundaries.windows(2) {
            let length = pair[1] - pair[0];
            assert!(length > 0);
            assert!(length <= config.max_segment);
            if pair[1] != count {
                assert!(length >= config.min_segment);
            }
        }
    }

    #[test]
    fn constant_stream_splits_only_at_maximum_length() {
        let config = LzvConfig {
            min_segment: 32,
            max_segment: 100,
            ..LzvConfig::default()
        };
        let values = vec![0x3ff0_0000_0000_0000; 350];
        assert_eq!(
            segment_lzv(&values, config).unwrap(),
            vec![0, 100, 200, 300, 350]
        );
    }

    #[test]
    fn stationary_random_stream_does_not_oversegment() {
        let config = LzvConfig {
            mean_shift: 12,
            variance_shift: 512,
            max_segment: 256,
            ..LzvConfig::default()
        };
        let values = random_stream(4096, 0x1234_5678);
        let boundaries = segment_lzv(&values, config).unwrap();
        check_bounds(&boundaries, values.len(), config);
        let max_only = values.len().div_ceil(config.max_segment);
        assert!(boundaries.len() - 1 <= max_only + 2);
    }

    #[test]
    fn abrupt_distribution_shift_is_detected_within_minimum_delay() {
        let config = LzvConfig {
            min_segment: 32,
            max_segment: 512,
            ..LzvConfig::default()
        };
        let mut values = vec![0x3ff0_0000_0000_0000; 128];
        values.extend(random_stream(128, 0x5eed));
        let boundaries = segment_lzv(&values, config).unwrap();
        check_bounds(&boundaries, values.len(), config);
        assert!(boundaries.iter().any(|boundary| (128..=160).contains(boundary)));
    }

    #[test]
    fn recurring_shift_is_detected_and_deterministic() {
        let config = LzvConfig::default();
        let mut values = vec![0u64; 512];
        values.extend(random_stream(512, 0x111));
        values.extend(vec![0u64; 512]);
        let first = segment_lzv(&values, config).unwrap();
        let second = segment_lzv(&values, config).unwrap();
        assert_eq!(first, second);
        check_bounds(&first, values.len(), config);
        assert!(first.iter().any(|boundary| (512..=544).contains(boundary)));
        assert!(first.iter().any(|boundary| (1024..=1056).contains(boundary)));
    }

    #[test]
    fn empty_and_single_value_inputs_are_defined() {
        let config = LzvConfig::default();
        assert_eq!(segment_lzv(&[], config).unwrap(), vec![0]);
        assert_eq!(segment_lzv(&[42], config).unwrap(), vec![0, 1]);
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        assert_eq!(
            segment_lzv(
                &[1, 2],
                LzvConfig {
                    fast_window: 1,
                    ..LzvConfig::default()
                }
            ),
            Err(LzvError::InvalidWindow)
        );
        assert_eq!(
            segment_lzv(
                &[1, 2],
                LzvConfig {
                    min_segment: 20,
                    max_segment: 10,
                    ..LzvConfig::default()
                }
            ),
            Err(LzvError::InvalidSegmentBounds)
        );
    }
}
