// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact causal change-point fixture subject.
//!
//! The input is a bounded noisy piecewise process over four fixed levels. True
//! changes are labeled only at the first frame of a new segment. The detection
//! tolerance window is causal: `[change_frame, change_frame + 3]` only.

use symthaea_physical_experiment::SeedPlan;
use symthaea_physical_fixtures::{
    BinaryChannel, FixtureSuite, MetadataEntry, ScalarChannel, SeedFixture,
};

/// Stable workload family.
pub const FAMILY: &str = "temporal:change-point";
/// Exact workload-definition version.
pub const VERSION: &str = "piecewise-four-level-causal-window-v1";
/// Domain separation mixed into the workload RNG state.
pub const RNG_DOMAIN_XOR: u64 = 0x4348_414e_4745_3031;
/// Minimum inclusive segment length.
pub const MIN_SEGMENT_FRAMES: usize = 64;
/// Maximum inclusive segment length.
pub const MAX_SEGMENT_FRAMES: usize = 192;
/// Number of post-change frames, in addition to the change frame, receiving tolerance credit.
pub const DETECTION_TOLERANCE_FRAMES: usize = 3;
/// Maximum frame count accepted by this generator.
pub const MAX_FRAMES: usize = 1_000_000;

const LEVELS: [f64; 4] = [-0.35, -0.15, 0.15, 0.35];
const NOISE_HALF_RANGE: f64 = 0.05;
const UNIT_53: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Exact change-point fixture-generation parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChangePointSpec {
    /// Number of causal frames generated per seed.
    pub frames: usize,
}

impl ChangePointSpec {
    /// Validate a bounded, non-empty frame count.
    pub fn validate(self) -> Result<(), ChangePointError> {
        if self.frames == 0 || self.frames > MAX_FRAMES {
            return Err(ChangePointError::InvalidFrameCount(self.frames));
        }
        Ok(())
    }
}

/// Generate the exact PHYS-006 causal change-point fixture suite.
pub fn generate_suite(
    seed_plan: &SeedPlan,
    spec: ChangePointSpec,
) -> Result<FixtureSuite, ChangePointError> {
    spec.validate()?;
    seed_plan
        .validate()
        .map_err(|error| ChangePointError::SeedPlan(error.to_string()))?;

    let records = seed_plan
        .seeds
        .iter()
        .map(|&seed| generate_seed_fixture(seed, spec.frames))
        .collect::<Result<Vec<_>, _>>()?;

    let suite = FixtureSuite {
        schema_version: 1,
        family: FAMILY.to_string(),
        version: VERSION.to_string(),
        metadata: vec![
            MetadataEntry {
                key: "change_label".to_string(),
                value: "is_change=1-only-on-first-frame-of-new-segment;frame0=0".to_string(),
            },
            MetadataEntry {
                key: "detection_window".to_string(),
                value: "causal-[change_frame,change_frame+3]".to_string(),
            },
            MetadataEntry {
                key: "discrete_sampler".to_string(),
                value: "splitmix64-unbiased-rejection-v1".to_string(),
            },
            MetadataEntry {
                key: "levels".to_string(),
                value: "-0.35,-0.15,0.15,0.35".to_string(),
            },
            MetadataEntry {
                key: "noise".to_string(),
                value: "uniform-[-0.05,0.05)".to_string(),
            },
            MetadataEntry {
                key: "retry_policy".to_string(),
                value: "none-at-fixture-level".to_string(),
            },
            MetadataEntry {
                key: "rng_domain_xor".to_string(),
                value: format!("0x{RNG_DOMAIN_XOR:016x}"),
            },
            MetadataEntry {
                key: "segment_length".to_string(),
                value: "uniform-integer-[64,192]".to_string(),
            },
            MetadataEntry {
                key: "transition".to_string(),
                value: "next-level-uniform-among-other-three".to_string(),
            },
        ],
        records,
    };
    suite
        .validate_against_seed_plan(seed_plan)
        .map_err(|error| ChangePointError::Fixture(error.to_string()))?;
    Ok(suite)
}

fn generate_seed_fixture(seed: u64, frames: usize) -> Result<SeedFixture, ChangePointError> {
    let mut rng = SplitMix64::new(seed ^ RNG_DOMAIN_XOR);
    let mut input = Vec::with_capacity(frames);
    let mut is_change = vec![0u8; frames];
    let mut within_detection_window = vec![0u8; frames];
    let mut change_points = Vec::new();

    let mut level_index = rng.next_bounded(LEVELS.len() as u64) as usize;
    let mut frame = 0usize;
    let mut first_segment = true;

    while frame < frames {
        if !first_segment {
            is_change[frame] = 1;
            change_points.push(frame);
        }

        let span = MAX_SEGMENT_FRAMES - MIN_SEGMENT_FRAMES + 1;
        let segment_frames = MIN_SEGMENT_FRAMES + rng.next_bounded(span as u64) as usize;
        let end = frame.saturating_add(segment_frames).min(frames);

        for sample_frame in frame..end {
            let noise = (rng.next_unit_f64() - 0.5) * (2.0 * NOISE_HALF_RANGE);
            let value = LEVELS[level_index] + noise;
            if !value.is_finite() {
                return Err(ChangePointError::NonFiniteInput {
                    seed,
                    frame: sample_frame,
                });
            }
            input.push(value);
        }

        frame = end;
        first_segment = false;
        if frame < frames {
            let draw = rng.next_bounded(3) as usize;
            level_index = if draw >= level_index { draw + 1 } else { draw };
        }
    }

    for change in change_points {
        for offset in 0..=DETECTION_TOLERANCE_FRAMES {
            if let Some(label) = within_detection_window.get_mut(change + offset) {
                *label = 1;
            }
        }
    }

    let fixture = SeedFixture {
        seed,
        scalar_channels: vec![ScalarChannel {
            name: "drive_voltage".to_string(),
            values: input,
        }],
        binary_channels: vec![
            BinaryChannel {
                name: "is_change".to_string(),
                values: is_change,
            },
            BinaryChannel {
                name: "within_detection_window".to_string(),
                values: within_detection_window,
            },
        ],
    };
    fixture
        .validate()
        .map_err(|error| ChangePointError::Fixture(error.to_string()))?;
    Ok(fixture)
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(state: u64) -> Self {
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// Draw exactly uniformly from `0..bound` using rejection sampling.
    ///
    /// This consumes another SplitMix64 word only when the raw value falls in
    /// the small incomplete residue interval that would otherwise create modulo
    /// bias. It is deterministic and part of the v1 workload definition.
    fn next_bounded(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let threshold = 0u64.wrapping_sub(bound) % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return value % bound;
            }
        }
    }

    fn next_unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * UNIT_53
    }
}

/// Exact change-point fixture-generation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangePointError {
    /// Frame count was zero or exceeded the bounded generator limit.
    InvalidFrameCount(usize),
    /// PHYS-002 seed plan was invalid.
    SeedPlan(String),
    /// PHYS-006 fixture validation failed.
    Fixture(String),
    /// A generated input was unexpectedly non-finite; no fixture retry is performed.
    NonFiniteInput {
        /// Seed whose stream failed.
        seed: u64,
        /// Exact frame being generated.
        frame: usize,
    },
}

impl std::fmt::Display for ChangePointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ChangePointError {}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN_INPUT_BITS: [u64; 16] = [
        0xbfbf_0ba4_bf62_5a37,
        0xbfbb_0562_630d_4a04,
        0xbfc5_8197_caf3_38e9,
        0xbfc0_c4f5_8c21_42f8,
        0xbfc4_2791_0750_4c77,
        0xbfc5_8c27_55d2_df13,
        0xbfc7_9d77_01b8_32ee,
        0xbfc4_aabc_715d_094f,
        0xbfc9_1836_0625_ba65,
        0xbfbe_1a1c_3229_f3fc,
        0xbfc3_3ede_b9af_870f,
        0xbfbe_46f5_f23d_ed7c,
        0xbfc0_2d00_0f1e_98ad,
        0xbfc5_cd9a_3c34_c753,
        0xbfc7_db01_5e1c_c602,
        0xbfbe_3ec9_9865_e14e,
    ];

    const GOLDEN_CHANGE_POINTS: [usize; 12] = [
        64, 140, 296, 375, 449, 568, 636, 752, 927, 1021, 1165, 1286,
    ];

    fn scalar<'a>(record: &'a SeedFixture, name: &str) -> &'a [f64] {
        &record
            .scalar_channels
            .iter()
            .find(|channel| channel.name == name)
            .unwrap()
            .values
    }

    fn binary<'a>(record: &'a SeedFixture, name: &str) -> &'a [u8] {
        &record
            .binary_channels
            .iter()
            .find(|channel| channel.name == name)
            .unwrap()
            .values
    }

    #[test]
    fn seed_one_prefix_and_change_points_are_golden() {
        let plan = SeedPlan { seeds: vec![1] };
        let suite = generate_suite(&plan, ChangePointSpec { frames: 3840 }).unwrap();
        let record = &suite.records[0];
        let input_bits = scalar(record, "drive_voltage")
            .iter()
            .take(16)
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(input_bits, GOLDEN_INPUT_BITS);

        let changes = binary(record, "is_change")
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (*value == 1).then_some(index))
            .collect::<Vec<_>>();
        assert_eq!(&changes[..GOLDEN_CHANGE_POINTS.len()], &GOLDEN_CHANGE_POINTS);
        assert_eq!(changes.len(), 32);
    }

    #[test]
    fn detection_window_is_causal_and_exactly_four_frames() {
        let record = generate_seed_fixture(1, 512).unwrap();
        let changes = binary(&record, "is_change");
        let windows = binary(&record, "within_detection_window");

        for (frame, &label) in changes.iter().enumerate() {
            if label == 1 {
                if frame > 0 {
                    assert_eq!(windows[frame - 1], 0);
                }
                for offset in 0..=DETECTION_TOLERANCE_FRAMES {
                    if frame + offset < windows.len() {
                        assert_eq!(windows[frame + offset], 1);
                    }
                }
                if frame + DETECTION_TOLERANCE_FRAMES + 1 < windows.len() {
                    assert_eq!(windows[frame + DETECTION_TOLERANCE_FRAMES + 1], 0);
                }
            }
        }
        assert_eq!(changes[0], 0);
    }

    #[test]
    fn drive_is_bounded_without_clipping() {
        let plan = SeedPlan {
            seeds: vec![1, 2, 3],
        };
        let suite = generate_suite(&plan, ChangePointSpec { frames: 2048 }).unwrap();
        for record in &suite.records {
            assert!(scalar(record, "drive_voltage")
                .iter()
                .all(|value| (-0.4..0.4).contains(value)));
        }
    }

    #[test]
    fn exact_channels_and_metadata_are_frozen() {
        let plan = SeedPlan { seeds: vec![5] };
        let suite = generate_suite(&plan, ChangePointSpec { frames: 256 }).unwrap();
        assert_eq!(suite.family, FAMILY);
        assert_eq!(suite.version, VERSION);
        assert_eq!(suite.records[0].scalar_channels.len(), 1);
        assert_eq!(suite.records[0].binary_channels.len(), 2);
        assert!(suite
            .metadata
            .iter()
            .any(|entry| entry.key == "retry_policy" && entry.value == "none-at-fixture-level"));
        assert!(suite.metadata.iter().any(|entry| {
            entry.key == "discrete_sampler" && entry.value == "splitmix64-unbiased-rejection-v1"
        }));
    }

    #[test]
    fn same_seed_replays_and_different_seed_changes_stream() {
        let a = generate_seed_fixture(42, 512).unwrap();
        let b = generate_seed_fixture(42, 512).unwrap();
        let c = generate_seed_fixture(43, 512).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn aurum002_fixed_seed_subject_constructs_exactly() {
        let plan = SeedPlan {
            seeds: vec![1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597],
        };
        let suite = generate_suite(&plan, ChangePointSpec { frames: 3840 }).unwrap();
        suite.validate_against_seed_plan(&plan).unwrap();
        assert_eq!(suite.records.len(), 16);
        for record in &suite.records {
            assert_eq!(scalar(record, "drive_voltage").len(), 3840);
            assert!(binary(record, "is_change")
                .iter()
                .filter(|value| **value == 1)
                .count()
                > 0);
        }
    }

    #[test]
    fn invalid_frame_counts_fail_before_generation() {
        let plan = SeedPlan { seeds: vec![1] };
        assert_eq!(
            generate_suite(&plan, ChangePointSpec { frames: 0 }),
            Err(ChangePointError::InvalidFrameCount(0))
        );
        assert_eq!(
            generate_suite(
                &plan,
                ChangePointSpec {
                    frames: MAX_FRAMES + 1,
                },
            ),
            Err(ChangePointError::InvalidFrameCount(MAX_FRAMES + 1))
        );
    }
}
