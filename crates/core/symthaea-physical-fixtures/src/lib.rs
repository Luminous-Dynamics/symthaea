// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical multi-seed workload fixture suites for physical-computation research.
//!
//! This crate freezes **what inputs/targets were presented** independently from
//! the backend that consumes them. It deliberately contains no NARMA, recall,
//! change-point, substrate, learner, or result-producing implementation.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::BTreeSet;
use symthaea_physical_experiment::{Digest32, SeedPlan, WorkloadIdentity};

const FIXTURE_DOMAIN: &[u8] = b"symthaea:physical:fixture-suite:v1\0";
const MAX_RECORDS: usize = 4096;
const MAX_CHANNELS_PER_RECORD: usize = 256;
const MAX_FRAMES_PER_CHANNEL: usize = 1_000_000;
const MAX_TOTAL_VALUES: usize = 16_000_000;
const MAX_METADATA_ENTRIES: usize = 128;
const MAX_METADATA_VALUE_BYTES: usize = 4096;

/// One canonical suite-level metadata entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataEntry {
    /// Canonical metadata key.
    pub key: String,
    /// UTF-8 metadata value. Values are length-prefixed in canonical bytes.
    pub value: String,
}

/// Named finite floating-point channel for one seed record.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarChannel {
    /// Canonical channel name.
    pub name: String,
    /// Ordered channel values, one per frame.
    pub values: Vec<f64>,
}

/// Named binary channel for one seed record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryChannel {
    /// Canonical channel name.
    pub name: String,
    /// Ordered values. Every element must be exactly `0` or `1`.
    pub values: Vec<u8>,
}

/// Complete frozen workload fixture for one preregistered seed.
#[derive(Debug, Clone, PartialEq)]
pub struct SeedFixture {
    /// Exact experiment seed represented by this record.
    pub seed: u64,
    /// Named finite scalar channels.
    pub scalar_channels: Vec<ScalarChannel>,
    /// Named binary channels.
    pub binary_channels: Vec<BinaryChannel>,
}

impl SeedFixture {
    /// Validate channel names, values, frame counts, and duplicate-free schema.
    pub fn validate(&self) -> Result<usize, FixtureError> {
        let channel_count = self
            .scalar_channels
            .len()
            .checked_add(self.binary_channels.len())
            .ok_or(FixtureError::SizeOverflow)?;
        if channel_count == 0 {
            return Err(FixtureError::EmptyRecord(self.seed));
        }
        if channel_count > MAX_CHANNELS_PER_RECORD {
            return Err(FixtureError::TooManyChannels {
                seed: self.seed,
                count: channel_count,
            });
        }

        let mut names = BTreeSet::new();
        let mut frame_count = None;

        for channel in &self.scalar_channels {
            validate_token("scalar channel", &channel.name, true)?;
            if !names.insert(channel.name.as_str()) {
                return Err(FixtureError::DuplicateChannel {
                    seed: self.seed,
                    name: channel.name.clone(),
                });
            }
            validate_channel_len(self.seed, &channel.name, channel.values.len(), &mut frame_count)?;
            if channel.values.iter().any(|value| !value.is_finite()) {
                return Err(FixtureError::NonFiniteScalar {
                    seed: self.seed,
                    channel: channel.name.clone(),
                });
            }
        }

        for channel in &self.binary_channels {
            validate_token("binary channel", &channel.name, true)?;
            if !names.insert(channel.name.as_str()) {
                return Err(FixtureError::DuplicateChannel {
                    seed: self.seed,
                    name: channel.name.clone(),
                });
            }
            validate_channel_len(self.seed, &channel.name, channel.values.len(), &mut frame_count)?;
            if channel.values.iter().any(|value| !matches!(*value, 0 | 1)) {
                return Err(FixtureError::InvalidBinaryValue {
                    seed: self.seed,
                    channel: channel.name.clone(),
                });
            }
        }

        frame_count.ok_or(FixtureError::EmptyRecord(self.seed))
    }

    fn channel_schema(&self) -> Vec<(u8, String)> {
        let mut schema = self
            .scalar_channels
            .iter()
            .map(|channel| (0, channel.name.clone()))
            .chain(
                self.binary_channels
                    .iter()
                    .map(|channel| (1, channel.name.clone())),
            )
            .collect::<Vec<_>>();
        schema.sort();
        schema
    }
}

/// Frozen multi-seed workload suite whose digest becomes PHYS-002 workload identity.
#[derive(Debug, Clone, PartialEq)]
pub struct FixtureSuite {
    /// Fixture schema version. V1 is currently the only accepted value.
    pub schema_version: u16,
    /// Canonical workload family, for example `temporal:narma10`.
    pub family: String,
    /// Workload-definition/fixture version.
    pub version: String,
    /// Suite-level metadata. Entry order is intentionally non-semantic.
    pub metadata: Vec<MetadataEntry>,
    /// Ordered seed records. Record order is commitment-semantic.
    pub records: Vec<SeedFixture>,
}

impl FixtureSuite {
    /// Validate a suite without constructing any result-producing runner.
    pub fn validate(&self) -> Result<(), FixtureError> {
        if self.schema_version != 1 {
            return Err(FixtureError::UnsupportedSchema(self.schema_version));
        }
        validate_token("fixture family", &self.family, true)?;
        validate_token("fixture version", &self.version, false)?;
        validate_metadata(&self.metadata)?;

        if self.records.is_empty() {
            return Err(FixtureError::EmptySuite);
        }
        if self.records.len() > MAX_RECORDS {
            return Err(FixtureError::TooManyRecords(self.records.len()));
        }

        let mut seeds = BTreeSet::new();
        let mut expected_schema: Option<Vec<(u8, String)>> = None;
        let mut total_values = 0usize;

        for record in &self.records {
            if !seeds.insert(record.seed) {
                return Err(FixtureError::DuplicateSeed(record.seed));
            }
            let frames = record.validate()?;
            let schema = record.channel_schema();
            match &expected_schema {
                Some(expected) if expected != &schema => {
                    return Err(FixtureError::ChannelSchemaDrift(record.seed));
                }
                None => expected_schema = Some(schema),
                _ => {}
            }
            let channels = record
                .scalar_channels
                .len()
                .checked_add(record.binary_channels.len())
                .ok_or(FixtureError::SizeOverflow)?;
            total_values = total_values
                .checked_add(
                    frames
                        .checked_mul(channels)
                        .ok_or(FixtureError::SizeOverflow)?,
                )
                .ok_or(FixtureError::SizeOverflow)?;
            if total_values > MAX_TOTAL_VALUES {
                return Err(FixtureError::TooManyValues(total_values));
            }
        }
        Ok(())
    }

    /// Require exact ordered agreement with the PHYS-002 seed plan.
    pub fn validate_against_seed_plan(&self, seed_plan: &SeedPlan) -> Result<(), FixtureError> {
        self.validate()?;
        seed_plan
            .validate()
            .map_err(|error| FixtureError::Experiment(error.to_string()))?;
        let actual = self.records.iter().map(|record| record.seed).collect::<Vec<_>>();
        if actual != seed_plan.seeds {
            return Err(FixtureError::SeedPlanMismatch {
                expected: seed_plan.seeds.clone(),
                actual,
            });
        }
        Ok(())
    }

    /// Deterministic domain-separated V1 bytes.
    ///
    /// Metadata/channel insertion order is normalized by bytewise key/name sort.
    /// Seed-record order and every channel value remain commitment-semantic.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, FixtureError> {
        self.validate()?;
        let mut out = Vec::new();
        out.extend_from_slice(FIXTURE_DOMAIN);
        push_u16(&mut out, self.schema_version);
        push_str(&mut out, &self.family)?;
        push_str(&mut out, &self.version)?;

        let mut metadata = self.metadata.iter().collect::<Vec<_>>();
        metadata.sort_by(|left, right| left.key.as_bytes().cmp(right.key.as_bytes()));
        push_len_u32(&mut out, metadata.len())?;
        for entry in metadata {
            push_str(&mut out, &entry.key)?;
            push_str(&mut out, &entry.value)?;
        }

        push_len_u32(&mut out, self.records.len())?;
        for record in &self.records {
            push_u64(&mut out, record.seed);

            let mut scalars = record.scalar_channels.iter().collect::<Vec<_>>();
            scalars.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
            push_len_u32(&mut out, scalars.len())?;
            for channel in scalars {
                push_str(&mut out, &channel.name)?;
                push_len_u64(&mut out, channel.values.len())?;
                for value in &channel.values {
                    out.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }

            let mut binaries = record.binary_channels.iter().collect::<Vec<_>>();
            binaries.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
            push_len_u32(&mut out, binaries.len())?;
            for channel in binaries {
                push_str(&mut out, &channel.name)?;
                push_len_u64(&mut out, channel.values.len())?;
                out.extend_from_slice(&channel.values);
            }
        }
        Ok(out)
    }

    /// BLAKE3 identity of the exact canonical fixture suite.
    pub fn fixture_digest(&self) -> Result<Digest32, FixtureError> {
        Ok(Digest32::blake3(&self.canonical_bytes()?))
    }

    /// Construct the exact PHYS-002 workload identity for this suite.
    pub fn workload_identity(&self) -> Result<WorkloadIdentity, FixtureError> {
        let identity = WorkloadIdentity {
            family: self.family.clone(),
            version: self.version.clone(),
            fixture_digest: self.fixture_digest()?,
        };
        identity
            .validate()
            .map_err(|error| FixtureError::Experiment(error.to_string()))?;
        Ok(identity)
    }
}

/// Fail-closed fixture validation/canonicalization failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixtureError {
    /// Unsupported fixture schema version.
    UnsupportedSchema(u16),
    /// Invalid canonical token.
    InvalidToken { field: &'static str, value: String },
    /// Metadata key appeared more than once.
    DuplicateMetadataKey(String),
    /// Metadata value was empty or exceeded the bounded size.
    InvalidMetadataValue(String),
    /// Metadata count exceeded the suite bound.
    TooManyMetadataEntries(usize),
    /// Suite has no seed records.
    EmptySuite,
    /// Suite record count exceeded the bound.
    TooManyRecords(usize),
    /// Seed record contains no channels.
    EmptyRecord(u64),
    /// Seed record exceeded the channel-count bound.
    TooManyChannels { seed: u64, count: usize },
    /// Two channels in one record shared a name, including across channel types.
    DuplicateChannel { seed: u64, name: String },
    /// Channel contains zero frames.
    EmptyChannel { seed: u64, channel: String },
    /// Channel exceeded the per-channel frame bound.
    TooManyFrames { seed: u64, channel: String, count: usize },
    /// Channels inside one seed record do not share the same frame count.
    FrameCountMismatch { seed: u64, expected: usize, actual: usize },
    /// Scalar channel contains NaN or infinity.
    NonFiniteScalar { seed: u64, channel: String },
    /// Binary channel contains a value other than zero or one.
    InvalidBinaryValue { seed: u64, channel: String },
    /// Suite contains the same seed more than once.
    DuplicateSeed(u64),
    /// Channel names/types differ across seed records.
    ChannelSchemaDrift(u64),
    /// Aggregate fixture value count exceeded the suite bound.
    TooManyValues(usize),
    /// Checked size arithmetic overflowed.
    SizeOverflow,
    /// Ordered suite seed records differ from the preregistered seed plan.
    SeedPlanMismatch { expected: Vec<u64>, actual: Vec<u64> },
    /// PHYS-002 validation rejected an imported type.
    Experiment(String),
    /// Canonical length cannot be represented by the selected encoding.
    CanonicalLengthOverflow,
}

impl std::fmt::Display for FixtureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for FixtureError {}

fn validate_metadata(metadata: &[MetadataEntry]) -> Result<(), FixtureError> {
    if metadata.len() > MAX_METADATA_ENTRIES {
        return Err(FixtureError::TooManyMetadataEntries(metadata.len()));
    }
    let mut keys = BTreeSet::new();
    for entry in metadata {
        validate_token("metadata key", &entry.key, true)?;
        if !keys.insert(entry.key.as_str()) {
            return Err(FixtureError::DuplicateMetadataKey(entry.key.clone()));
        }
        if entry.value.is_empty() || entry.value.len() > MAX_METADATA_VALUE_BYTES {
            return Err(FixtureError::InvalidMetadataValue(entry.key.clone()));
        }
    }
    Ok(())
}

fn validate_channel_len(
    seed: u64,
    channel: &str,
    len: usize,
    expected: &mut Option<usize>,
) -> Result<(), FixtureError> {
    if len == 0 {
        return Err(FixtureError::EmptyChannel {
            seed,
            channel: channel.to_string(),
        });
    }
    if len > MAX_FRAMES_PER_CHANNEL {
        return Err(FixtureError::TooManyFrames {
            seed,
            channel: channel.to_string(),
            count: len,
        });
    }
    match *expected {
        Some(expected_len) if expected_len != len => Err(FixtureError::FrameCountMismatch {
            seed,
            expected: expected_len,
            actual: len,
        }),
        None => {
            *expected = Some(len);
            Ok(())
        }
        _ => Ok(()),
    }
}

fn validate_token(
    field: &'static str,
    value: &str,
    allow_colon: bool,
) -> Result<(), FixtureError> {
    let valid = !value.is_empty()
        && value.len() <= 96
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_' | b'.')
                || (allow_colon && byte == b':')
        });
    if valid {
        Ok(())
    } else {
        Err(FixtureError::InvalidToken {
            field,
            value: value.to_owned(),
        })
    }
}

fn push_str(out: &mut Vec<u8>, value: &str) -> Result<(), FixtureError> {
    push_len_u32(out, value.len())?;
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn push_len_u32(out: &mut Vec<u8>, value: usize) -> Result<(), FixtureError> {
    let value = u32::try_from(value).map_err(|_| FixtureError::CanonicalLengthOverflow)?;
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn push_len_u64(out: &mut Vec<u8>, value: usize) -> Result<(), FixtureError> {
    let value = u64::try_from(value).map_err(|_| FixtureError::CanonicalLengthOverflow)?;
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(seed: u64) -> SeedFixture {
        SeedFixture {
            seed,
            scalar_channels: vec![
                ScalarChannel {
                    name: "target".to_string(),
                    values: vec![0.1, 0.2, 0.3],
                },
                ScalarChannel {
                    name: "drive_voltage".to_string(),
                    values: vec![0.4, 0.5, 0.6],
                },
            ],
            binary_channels: vec![BinaryChannel {
                name: "is_transition".to_string(),
                values: vec![0, 1, 0],
            }],
        }
    }

    fn suite() -> FixtureSuite {
        FixtureSuite {
            schema_version: 1,
            family: "temporal:fixture-smoke".to_string(),
            version: "v1".to_string(),
            metadata: vec![
                MetadataEntry {
                    key: "generator".to_string(),
                    value: "golden-v1".to_string(),
                },
                MetadataEntry {
                    key: "precision".to_string(),
                    value: "f64".to_string(),
                },
            ],
            records: vec![record(7), record(11)],
        }
    }

    #[test]
    fn metadata_and_channel_insertion_order_are_non_semantic() {
        let a = suite();
        let mut b = a.clone();
        b.metadata.reverse();
        for record in &mut b.records {
            record.scalar_channels.reverse();
        }
        assert_eq!(a.fixture_digest().unwrap(), b.fixture_digest().unwrap());
    }

    #[test]
    fn record_order_is_semantic_and_seed_plan_is_exact_ordered() {
        let a = suite();
        let mut b = a.clone();
        b.records.reverse();
        assert_ne!(a.fixture_digest().unwrap(), b.fixture_digest().unwrap());

        let plan = SeedPlan {
            seeds: vec![7, 11],
        };
        a.validate_against_seed_plan(&plan).unwrap();
        assert!(matches!(
            b.validate_against_seed_plan(&plan),
            Err(FixtureError::SeedPlanMismatch { .. })
        ));
    }

    #[test]
    fn scalar_and_binary_mutations_change_fixture_identity() {
        let a = suite();
        let mut scalar_changed = a.clone();
        scalar_changed.records[0].scalar_channels[0].values[0] += 0.01;
        let mut binary_changed = a.clone();
        binary_changed.records[0].binary_channels[0].values[0] = 1;
        assert_ne!(a.fixture_digest().unwrap(), scalar_changed.fixture_digest().unwrap());
        assert_ne!(a.fixture_digest().unwrap(), binary_changed.fixture_digest().unwrap());
    }

    #[test]
    fn non_finite_scalar_and_non_binary_value_are_rejected() {
        let mut non_finite = suite();
        non_finite.records[0].scalar_channels[0].values[0] = f64::NAN;
        assert!(matches!(
            non_finite.validate(),
            Err(FixtureError::NonFiniteScalar { .. })
        ));

        let mut non_binary = suite();
        non_binary.records[0].binary_channels[0].values[0] = 2;
        assert!(matches!(
            non_binary.validate(),
            Err(FixtureError::InvalidBinaryValue { .. })
        ));
    }

    #[test]
    fn frame_count_and_cross_type_duplicate_names_are_rejected() {
        let mut mismatch = suite();
        mismatch.records[0].binary_channels[0].values.pop();
        assert!(matches!(
            mismatch.validate(),
            Err(FixtureError::FrameCountMismatch { .. })
        ));

        let mut duplicate = suite();
        duplicate.records[0].binary_channels[0].name = "target".to_string();
        assert!(matches!(
            duplicate.validate(),
            Err(FixtureError::DuplicateChannel { .. })
        ));
    }

    #[test]
    fn schema_drift_across_seeds_is_rejected() {
        let mut drifted = suite();
        drifted.records[1].scalar_channels[0].name = "different_target".to_string();
        assert_eq!(
            drifted.validate(),
            Err(FixtureError::ChannelSchemaDrift(11))
        );
    }

    #[test]
    fn workload_identity_uses_exact_suite_digest() {
        let fixture = suite();
        let identity = fixture.workload_identity().unwrap();
        assert_eq!(identity.family, fixture.family);
        assert_eq!(identity.version, fixture.version);
        assert_eq!(identity.fixture_digest, fixture.fixture_digest().unwrap());
    }

    #[test]
    fn duplicate_metadata_and_duplicate_seed_are_rejected() {
        let mut metadata = suite();
        metadata.metadata.push(MetadataEntry {
            key: "generator".to_string(),
            value: "other".to_string(),
        });
        assert!(matches!(
            metadata.validate(),
            Err(FixtureError::DuplicateMetadataKey(_))
        ));

        let mut seeds = suite();
        seeds.records[1].seed = 7;
        assert_eq!(seeds.validate(), Err(FixtureError::DuplicateSeed(7)));
    }
}
