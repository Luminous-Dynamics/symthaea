// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact delayed-recall fixture subject for memory-capacity characterization.
//!
//! Frame `t` presents `drive_voltage = u[t]` and contains explicit target
//! channels for delays 1 through 32. Before sufficient history exists, delayed
//! targets are exact zero. This module does not compute R² or memory capacity.

use symthaea_physical_experiment::SeedPlan;
use symthaea_physical_fixtures::{
    FixtureSuite, MetadataEntry, ScalarChannel, SeedFixture,
};

/// Stable workload family.
pub const FAMILY: &str = "temporal:delayed-recall";
/// Exact workload-definition version.
pub const VERSION: &str = "uniform-signed-d1-32-v1";
/// First included recall delay.
pub const MIN_DELAY: usize = 1;
/// Last included recall delay.
pub const MAX_DELAY: usize = 32;
/// Domain separation mixed into the workload RNG state.
pub const RNG_DOMAIN_XOR: u64 = 0x5245_4341_4c4c_3031;
/// Maximum frame count accepted by this generator.
pub const MAX_FRAMES: usize = 1_000_000;

const UNIT_53: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Exact delayed-recall fixture-generation parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DelayedRecallSpec {
    /// Number of causal input/target frames generated per seed.
    pub frames: usize,
}

impl DelayedRecallSpec {
    /// Validate a bounded, non-empty frame count.
    pub fn validate(self) -> Result<(), DelayedRecallError> {
        if self.frames == 0 || self.frames > MAX_FRAMES {
            return Err(DelayedRecallError::InvalidFrameCount(self.frames));
        }
        Ok(())
    }
}

/// Generate the exact PHYS-006 delayed-recall fixture suite.
pub fn generate_suite(
    seed_plan: &SeedPlan,
    spec: DelayedRecallSpec,
) -> Result<FixtureSuite, DelayedRecallError> {
    spec.validate()?;
    seed_plan
        .validate()
        .map_err(|error| DelayedRecallError::SeedPlan(error.to_string()))?;

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
                key: "alignment".to_string(),
                value: "frame-t:drive_voltage=u[t];target_delay_dd=u[t-dd]".to_string(),
            },
            MetadataEntry {
                key: "delay_range".to_string(),
                value: "1..32-inclusive".to_string(),
            },
            MetadataEntry {
                key: "input_distribution".to_string(),
                value: "splitmix64-top53-uniform-[-0.5,0.5)".to_string(),
            },
            MetadataEntry {
                key: "prehistory".to_string(),
                value: "u[t<0]=0".to_string(),
            },
            MetadataEntry {
                key: "retry_policy".to_string(),
                value: "none".to_string(),
            },
            MetadataEntry {
                key: "rng_domain_xor".to_string(),
                value: format!("0x{RNG_DOMAIN_XOR:016x}"),
            },
        ],
        records,
    };
    suite
        .validate_against_seed_plan(seed_plan)
        .map_err(|error| DelayedRecallError::Fixture(error.to_string()))?;
    Ok(suite)
}

fn generate_seed_fixture(seed: u64, frames: usize) -> Result<SeedFixture, DelayedRecallError> {
    let mut rng = SplitMix64::new(seed ^ RNG_DOMAIN_XOR);
    let mut input = Vec::with_capacity(frames);
    for _ in 0..frames {
        input.push(rng.next_unit_f64() - 0.5);
    }

    let mut scalar_channels = Vec::with_capacity(1 + MAX_DELAY);
    scalar_channels.push(ScalarChannel {
        name: "drive_voltage".to_string(),
        values: input.clone(),
    });

    for delay in MIN_DELAY..=MAX_DELAY {
        let mut target = Vec::with_capacity(frames);
        for frame in 0..frames {
            target.push(if frame >= delay {
                input[frame - delay]
            } else {
                0.0
            });
        }
        scalar_channels.push(ScalarChannel {
            name: format!("target_delay_{delay:02}"),
            values: target,
        });
    }

    let fixture = SeedFixture {
        seed,
        scalar_channels,
        binary_channels: vec![],
    };
    fixture
        .validate()
        .map_err(|error| DelayedRecallError::Fixture(error.to_string()))?;
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

    fn next_unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * UNIT_53
    }
}

/// Exact delayed-recall fixture-generation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DelayedRecallError {
    /// Frame count was zero or exceeded the bounded generator limit.
    InvalidFrameCount(usize),
    /// PHYS-002 seed plan was invalid.
    SeedPlan(String),
    /// PHYS-006 fixture validation failed.
    Fixture(String),
}

impl std::fmt::Display for DelayedRecallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DelayedRecallError {}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN_INPUT_BITS: [u64; 16] = [
        0xbfcb_c2dd_3662_a490,
        0xbfc3_38da_3b73_6558,
        0xbfc2_ca7a_f649_ea30,
        0x3fa1_9ba6_54ec_5c60,
        0xbf82_b7e0_18c2_d300,
        0x3fb2_7b89_8e49_0ea8,
        0x3fd2_b40c_5ab8_997e,
        0xbf90_7266_f13b_aba0,
        0xbfc4_0380_23bb_e4f8,
        0xbfd3_f2d9_93f0_4108,
        0xbf8c_0b09_4e86_1bc0,
        0x3fd7_cc57_2b6c_ee32,
        0xbfb0_55e1_8333_2248,
        0xbfd7_8218_a8f9_34fe,
        0x3fdd_8ee5_1397_1d12,
        0x3f8d_6334_3b89_b6c0,
    ];

    fn scalar<'a>(record: &'a SeedFixture, name: &str) -> &'a [f64] {
        &record
            .scalar_channels
            .iter()
            .find(|channel| channel.name == name)
            .unwrap()
            .values
    }

    #[test]
    fn seed_one_input_prefix_is_bit_exact() {
        let plan = SeedPlan { seeds: vec![1] };
        let suite = generate_suite(&plan, DelayedRecallSpec { frames: 16 }).unwrap();
        let bits = scalar(&suite.records[0], "drive_voltage")
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(bits, GOLDEN_INPUT_BITS);
    }

    #[test]
    fn all_delays_are_explicit_and_causally_aligned() {
        let plan = SeedPlan { seeds: vec![7] };
        let suite = generate_suite(&plan, DelayedRecallSpec { frames: 96 }).unwrap();
        let record = &suite.records[0];
        assert_eq!(record.scalar_channels.len(), 33);
        let input = scalar(record, "drive_voltage");

        for delay in MIN_DELAY..=MAX_DELAY {
            let name = format!("target_delay_{delay:02}");
            let target = scalar(record, &name);
            for frame in 0..96 {
                let expected = if frame >= delay {
                    input[frame - delay]
                } else {
                    0.0
                };
                assert_eq!(target[frame].to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn inputs_are_zero_mean_domain_bounded() {
        let plan = SeedPlan {
            seeds: vec![1, 2, 3],
        };
        let suite = generate_suite(&plan, DelayedRecallSpec { frames: 1024 }).unwrap();
        for record in &suite.records {
            assert!(scalar(record, "drive_voltage")
                .iter()
                .all(|value| (-0.5..0.5).contains(value)));
        }
    }

    #[test]
    fn exact_identity_and_retry_boundary_are_explicit() {
        let plan = SeedPlan { seeds: vec![2] };
        let suite = generate_suite(&plan, DelayedRecallSpec { frames: 64 }).unwrap();
        assert_eq!(suite.family, FAMILY);
        assert_eq!(suite.version, VERSION);
        assert!(suite
            .metadata
            .iter()
            .any(|entry| entry.key == "retry_policy" && entry.value == "none"));
    }

    #[test]
    fn same_seed_replays_and_different_seed_changes_input() {
        let a = generate_seed_fixture(42, 64).unwrap();
        let b = generate_seed_fixture(42, 64).unwrap();
        let c = generate_seed_fixture(43, 64).unwrap();
        assert_eq!(a, b);
        assert_ne!(scalar(&a, "drive_voltage"), scalar(&c, "drive_voltage"));
    }

    #[test]
    fn aurum002_fixed_seed_subject_constructs_exactly() {
        let plan = SeedPlan {
            seeds: vec![1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597],
        };
        let suite = generate_suite(&plan, DelayedRecallSpec { frames: 3840 }).unwrap();
        suite.validate_against_seed_plan(&plan).unwrap();
        assert_eq!(suite.records.len(), 16);
        for record in &suite.records {
            assert_eq!(scalar(record, "drive_voltage").len(), 3840);
            assert_eq!(scalar(record, "target_delay_32").len(), 3840);
        }
    }

    #[test]
    fn invalid_frame_counts_fail_before_generation() {
        let plan = SeedPlan { seeds: vec![1] };
        assert_eq!(
            generate_suite(&plan, DelayedRecallSpec { frames: 0 }),
            Err(DelayedRecallError::InvalidFrameCount(0))
        );
        assert_eq!(
            generate_suite(
                &plan,
                DelayedRecallSpec {
                    frames: MAX_FRAMES + 1,
                },
            ),
            Err(DelayedRecallError::InvalidFrameCount(MAX_FRAMES + 1))
        );
    }
}
