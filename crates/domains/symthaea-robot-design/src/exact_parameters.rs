// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exact-by-definition designer-chosen parameter identities.
//!
//! These values describe nominal design intent. They do not describe measured
//! as-built values, calibration results, material truth, or physical evidence.

use crate::ContentDigest;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

pub const EXACT_PARAMETER_SCHEMA_ID: &str = "symthaea.robot-design.exact-parameters.v1";
pub const EXACT_PARAMETER_SCHEMA_VERSION: u32 = 1;
pub const EXACT_LENGTH_DOMAIN_SCHEMA_ID: &str = "symthaea.robot-design.exact-length-domain.v1";
pub const EXACT_LENGTH_DOMAIN_SCHEMA_VERSION: u32 = 1;

const VALUE_KIND_LENGTH_UM: u8 = 0;
const DOMAIN_KIND_EXPLICIT: u8 = 0;
const DOMAIN_KIND_STEPPED: u8 = 1;

/// Stable semantic identity for one designer-chosen parameter.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DesignParameterId(String);

impl DesignParameterId {
    pub fn new(value: impl Into<String>) -> Result<Self, ExactParameterError> {
        let value = value.into();
        if valid_semantic_id(&value) {
            Ok(Self(value))
        } else {
            Err(ExactParameterError::InvalidParameterId(value))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DesignParameterId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Exact nominal length in integer micrometres.
///
/// Zero is representable because some future exact offsets may legitimately be
/// zero. A domain/profile decides whether a particular parameter must be
/// strictly positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExactDesignLengthUmV1(u64);

impl ExactDesignLengthUmV1 {
    pub const fn from_um(value_um: u64) -> Self {
        Self(value_um)
    }

    pub const fn as_um(self) -> u64 {
        self.0
    }

    pub fn millimetres_rational(self) -> ExactRationalU64V1 {
        ExactRationalU64V1::new(self.0, 1_000).expect("fixed non-zero millimetre denominator")
    }

    pub fn metres_rational(self) -> ExactRationalU64V1 {
        ExactRationalU64V1::new(self.0, 1_000_000).expect("fixed non-zero metre denominator")
    }
}

/// Small exact rational used only for unit projection of exact design intent.
///
/// V1 deliberately does not implement ordering: derived `(numerator,
/// denominator)` ordering would be lexicographic rather than numeric rational
/// ordering and could mislead callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactRationalU64V1 {
    numerator: u64,
    denominator: u64,
}

impl ExactRationalU64V1 {
    pub fn new(numerator: u64, denominator: u64) -> Result<Self, ExactParameterError> {
        if denominator == 0 {
            return Err(ExactParameterError::ZeroDenominator);
        }
        if numerator == 0 {
            return Ok(Self {
                numerator: 0,
                denominator: 1,
            });
        }
        let divisor = gcd_u64(numerator, denominator);
        Ok(Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        })
    }

    pub const fn numerator(self) -> u64 {
        self.numerator
    }

    pub const fn denominator(self) -> u64 {
        self.denominator
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExactDesignValueV1 {
    LengthUm(ExactDesignLengthUmV1),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactDesignParameterV1 {
    pub id: DesignParameterId,
    pub value: ExactDesignValueV1,
}

impl ExactDesignParameterV1 {
    pub fn length(id: DesignParameterId, value: ExactDesignLengthUmV1) -> Self {
        Self {
            id,
            value: ExactDesignValueV1::LengthUm(value),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExactDesignParameterSetId(ContentDigest);

impl ExactDesignParameterSetId {
    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

impl fmt::Display for ExactDesignParameterSetId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Canonical set of exact designer-chosen parameters.
///
/// Parameter insertion order is non-semantic. The parameter ID and exact value
/// are semantic. Search-domain membership is deliberately not part of this ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactDesignParameterSetV1 {
    pub schema_id: String,
    pub schema_version: u32,
    pub parameters: Vec<ExactDesignParameterV1>,
}

impl ExactDesignParameterSetV1 {
    pub fn new(parameters: Vec<ExactDesignParameterV1>) -> Self {
        Self {
            schema_id: EXACT_PARAMETER_SCHEMA_ID.to_string(),
            schema_version: EXACT_PARAMETER_SCHEMA_VERSION,
            parameters,
        }
    }

    pub fn validate(&self) -> Result<(), ExactParameterError> {
        if self.schema_id != EXACT_PARAMETER_SCHEMA_ID
            || self.schema_version != EXACT_PARAMETER_SCHEMA_VERSION
        {
            return Err(ExactParameterError::UnsupportedParameterSchema {
                schema_id: self.schema_id.clone(),
                schema_version: self.schema_version,
            });
        }

        let mut ids = BTreeSet::new();
        for parameter in &self.parameters {
            if !valid_semantic_id(parameter.id.as_str()) {
                return Err(ExactParameterError::InvalidParameterId(
                    parameter.id.to_string(),
                ));
            }
            if !ids.insert(parameter.id.clone()) {
                return Err(ExactParameterError::DuplicateParameterId(
                    parameter.id.to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn canonical_transcript(&self) -> Result<Vec<u8>, ExactParameterError> {
        self.validate()?;
        let mut parameters = self.parameters.iter().collect::<Vec<_>>();
        parameters.sort_by(|left, right| left.id.cmp(&right.id));

        let mut out = Vec::new();
        put_str(&mut out, EXACT_PARAMETER_SCHEMA_ID);
        put_u32(&mut out, EXACT_PARAMETER_SCHEMA_VERSION);
        put_len(&mut out, parameters.len());

        for parameter in parameters {
            put_str(&mut out, parameter.id.as_str());
            match &parameter.value {
                ExactDesignValueV1::LengthUm(value) => {
                    put_u8(&mut out, VALUE_KIND_LENGTH_UM);
                    put_u64(&mut out, (*value).as_um());
                }
            }
        }

        Ok(out)
    }

    pub fn parameter_set_id(&self) -> Result<ExactDesignParameterSetId, ExactParameterError> {
        let transcript = self.canonical_transcript()?;
        let digest: [u8; 32] = Sha256::digest(transcript).into();
        Ok(ExactDesignParameterSetId(ContentDigest::from_bytes(digest)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExactLengthDomainKindV1 {
    Explicit {
        values: Vec<ExactDesignLengthUmV1>,
    },
    Stepped {
        lower: ExactDesignLengthUmV1,
        upper: ExactDesignLengthUmV1,
        step: ExactDesignLengthUmV1,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExactLengthDomainId(ContentDigest);

impl ExactLengthDomainId {
    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

impl fmt::Display for ExactLengthDomainId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Exact allowed values for one length-valued design parameter.
///
/// The domain identity is intentionally separate from a selected parameter-set
/// identity. Changing a bound or allowed value does not mutate the identity of
/// an already-selected exact design value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactLengthDomainV1 {
    pub schema_id: String,
    pub schema_version: u32,
    pub parameter_id: DesignParameterId,
    pub require_positive: bool,
    pub max_values: u32,
    pub kind: ExactLengthDomainKindV1,
}

impl ExactLengthDomainV1 {
    pub fn new(
        parameter_id: DesignParameterId,
        require_positive: bool,
        max_values: u32,
        kind: ExactLengthDomainKindV1,
    ) -> Self {
        Self {
            schema_id: EXACT_LENGTH_DOMAIN_SCHEMA_ID.to_string(),
            schema_version: EXACT_LENGTH_DOMAIN_SCHEMA_VERSION,
            parameter_id,
            require_positive,
            max_values,
            kind,
        }
    }

    pub fn validate(&self) -> Result<(), ExactParameterError> {
        if self.schema_id != EXACT_LENGTH_DOMAIN_SCHEMA_ID
            || self.schema_version != EXACT_LENGTH_DOMAIN_SCHEMA_VERSION
        {
            return Err(ExactParameterError::UnsupportedDomainSchema {
                schema_id: self.schema_id.clone(),
                schema_version: self.schema_version,
            });
        }
        if !valid_semantic_id(self.parameter_id.as_str()) {
            return Err(ExactParameterError::InvalidParameterId(
                self.parameter_id.to_string(),
            ));
        }
        if self.max_values == 0 {
            return Err(ExactParameterError::ZeroDomainResourceLimit);
        }

        match &self.kind {
            ExactLengthDomainKindV1::Explicit { values } => {
                if values.is_empty() {
                    return Err(ExactParameterError::EmptyExplicitDomain);
                }
                let mut unique = BTreeSet::new();
                for value in values {
                    self.validate_value(*value)?;
                    if !unique.insert(*value) {
                        return Err(ExactParameterError::DuplicateDomainValue(value.as_um()));
                    }
                }
                if values.len() > self.max_values as usize {
                    return Err(ExactParameterError::DomainTooLarge {
                        count: values.len() as u64,
                        max_values: self.max_values,
                    });
                }
            }
            ExactLengthDomainKindV1::Stepped { lower, upper, step } => {
                self.validate_value(*lower)?;
                self.validate_value(*upper)?;
                if step.as_um() == 0 {
                    return Err(ExactParameterError::ZeroStep);
                }
                if lower > upper {
                    return Err(ExactParameterError::InvalidDomainBounds {
                        lower_um: lower.as_um(),
                        upper_um: upper.as_um(),
                    });
                }
                let delta = upper.as_um() - lower.as_um();
                if delta % step.as_um() != 0 {
                    return Err(ExactParameterError::UpperBoundNotReachable);
                }
                let count = delta
                    .checked_div(step.as_um())
                    .and_then(|value| value.checked_add(1))
                    .ok_or(ExactParameterError::ArithmeticOverflow)?;
                if count > u64::from(self.max_values) {
                    return Err(ExactParameterError::DomainTooLarge {
                        count,
                        max_values: self.max_values,
                    });
                }
            }
        }

        Ok(())
    }

    pub fn enumerate(&self) -> Result<Vec<ExactDesignLengthUmV1>, ExactParameterError> {
        self.validate()?;
        match &self.kind {
            ExactLengthDomainKindV1::Explicit { values } => {
                let mut values = values.clone();
                values.sort_unstable();
                Ok(values)
            }
            ExactLengthDomainKindV1::Stepped { lower, upper, step } => {
                let count = (upper.as_um() - lower.as_um()) / step.as_um() + 1;
                let capacity =
                    usize::try_from(count).map_err(|_| ExactParameterError::ArithmeticOverflow)?;
                let mut values = Vec::with_capacity(capacity);
                let mut current = lower.as_um();
                loop {
                    values.push(ExactDesignLengthUmV1::from_um(current));
                    if current == upper.as_um() {
                        break;
                    }
                    current = current
                        .checked_add(step.as_um())
                        .ok_or(ExactParameterError::ArithmeticOverflow)?;
                }
                Ok(values)
            }
        }
    }

    pub fn contains(&self, value: ExactDesignLengthUmV1) -> Result<bool, ExactParameterError> {
        self.validate()?;
        match &self.kind {
            ExactLengthDomainKindV1::Explicit { values } => Ok(values.contains(&value)),
            ExactLengthDomainKindV1::Stepped { lower, upper, step } => {
                if value < *lower || value > *upper {
                    return Ok(false);
                }
                Ok((value.as_um() - lower.as_um()).is_multiple_of(step.as_um()))
            }
        }
    }

    pub fn canonical_transcript(&self) -> Result<Vec<u8>, ExactParameterError> {
        self.validate()?;
        let mut out = Vec::new();
        put_str(&mut out, EXACT_LENGTH_DOMAIN_SCHEMA_ID);
        put_u32(&mut out, EXACT_LENGTH_DOMAIN_SCHEMA_VERSION);
        put_str(&mut out, self.parameter_id.as_str());
        put_u8(&mut out, u8::from(self.require_positive));
        put_u32(&mut out, self.max_values);

        match &self.kind {
            ExactLengthDomainKindV1::Explicit { values } => {
                put_u8(&mut out, DOMAIN_KIND_EXPLICIT);
                let mut values = values.clone();
                values.sort_unstable();
                put_len(&mut out, values.len());
                for value in values {
                    put_u64(&mut out, value.as_um());
                }
            }
            ExactLengthDomainKindV1::Stepped { lower, upper, step } => {
                put_u8(&mut out, DOMAIN_KIND_STEPPED);
                put_u64(&mut out, lower.as_um());
                put_u64(&mut out, upper.as_um());
                put_u64(&mut out, step.as_um());
            }
        }

        Ok(out)
    }

    pub fn domain_id(&self) -> Result<ExactLengthDomainId, ExactParameterError> {
        let transcript = self.canonical_transcript()?;
        let digest: [u8; 32] = Sha256::digest(transcript).into();
        Ok(ExactLengthDomainId(ContentDigest::from_bytes(digest)))
    }

    fn validate_value(&self, value: ExactDesignLengthUmV1) -> Result<(), ExactParameterError> {
        if self.require_positive && value.as_um() == 0 {
            Err(ExactParameterError::PositiveValueRequired)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExactParameterError {
    InvalidParameterId(String),
    UnsupportedParameterSchema {
        schema_id: String,
        schema_version: u32,
    },
    UnsupportedDomainSchema {
        schema_id: String,
        schema_version: u32,
    },
    DuplicateParameterId(String),
    ZeroDenominator,
    ZeroDomainResourceLimit,
    EmptyExplicitDomain,
    DuplicateDomainValue(u64),
    PositiveValueRequired,
    ZeroStep,
    InvalidDomainBounds {
        lower_um: u64,
        upper_um: u64,
    },
    UpperBoundNotReachable,
    DomainTooLarge {
        count: u64,
        max_values: u32,
    },
    ArithmeticOverflow,
}

impl fmt::Display for ExactParameterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameterId(value) => {
                write!(formatter, "invalid design-parameter id: {value:?}")
            }
            Self::UnsupportedParameterSchema {
                schema_id,
                schema_version,
            } => write!(
                formatter,
                "unsupported exact-parameter schema {schema_id}@{schema_version}"
            ),
            Self::UnsupportedDomainSchema {
                schema_id,
                schema_version,
            } => write!(
                formatter,
                "unsupported exact-length-domain schema {schema_id}@{schema_version}"
            ),
            Self::DuplicateParameterId(id) => write!(formatter, "duplicate parameter id: {id}"),
            Self::ZeroDenominator => formatter.write_str("exact rational denominator is zero"),
            Self::ZeroDomainResourceLimit => {
                formatter.write_str("domain max_values must be greater than zero")
            }
            Self::EmptyExplicitDomain => {
                formatter.write_str("explicit exact-length domain must not be empty")
            }
            Self::DuplicateDomainValue(value) => {
                write!(formatter, "duplicate exact-length domain value: {value} um")
            }
            Self::PositiveValueRequired => {
                formatter.write_str("domain requires a strictly positive value")
            }
            Self::ZeroStep => formatter.write_str("stepped exact-length domain step is zero"),
            Self::InvalidDomainBounds { lower_um, upper_um } => write!(
                formatter,
                "invalid exact-length domain bounds: lower={lower_um} um upper={upper_um} um"
            ),
            Self::UpperBoundNotReachable => formatter
                .write_str("stepped exact-length domain upper bound is not exactly reachable"),
            Self::DomainTooLarge { count, max_values } => write!(
                formatter,
                "exact-length domain has {count} values, exceeding limit {max_values}"
            ),
            Self::ArithmeticOverflow => {
                formatter.write_str("exact design-parameter arithmetic overflow")
            }
        }
    }
}

impl std::error::Error for ExactParameterError {}

fn valid_semantic_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
        })
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn put_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_be_bytes());
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_len(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> DesignParameterId {
        DesignParameterId::new(value).unwrap()
    }

    fn length_parameter(id_value: &str, value_um: u64) -> ExactDesignParameterV1 {
        ExactDesignParameterV1::length(id(id_value), ExactDesignLengthUmV1::from_um(value_um))
    }

    fn reference_set() -> ExactDesignParameterSetV1 {
        ExactDesignParameterSetV1::new(vec![
            length_parameter("section_width", 20_000),
            length_parameter("link_length", 300_000),
            length_parameter("section_height", 6_000),
        ])
    }

    #[test]
    fn golden_parameter_set_transcript_and_digest_are_stable() {
        let set = reference_set();
        assert_eq!(
            hex(&set.canonical_transcript().unwrap()),
            "000000000000002973796d74686165612e726f626f742d64657369676e2e65786163742d706172616d65746572732e7631000000010000000000000003000000000000000b6c696e6b5f6c656e6774680000000000000493e0000000000000000e73656374696f6e5f686569676874000000000000001770000000000000000d73656374696f6e5f7769647468000000000000004e20"
        );
        assert_eq!(
            set.parameter_set_id().unwrap().to_hex(),
            "7f8721c10bdd0cd351619750e94f2e41677be266cc5ba6701e95c4444f16ba9c"
        );
    }

    #[test]
    fn parameter_insertion_order_is_nonsemantic() {
        let expected = reference_set().parameter_set_id().unwrap();
        let reordered = ExactDesignParameterSetV1::new(vec![
            length_parameter("section_height", 6_000),
            length_parameter("section_width", 20_000),
            length_parameter("link_length", 300_000),
        ]);
        assert_eq!(reordered.parameter_set_id().unwrap(), expected);
    }

    #[test]
    fn one_micrometre_change_changes_parameter_set_identity() {
        let expected = reference_set().parameter_set_id().unwrap();
        let changed = ExactDesignParameterSetV1::new(vec![
            length_parameter("section_width", 20_001),
            length_parameter("link_length", 300_000),
            length_parameter("section_height", 6_000),
        ]);
        assert_ne!(changed.parameter_set_id().unwrap(), expected);
    }

    #[test]
    fn duplicate_parameter_ids_reject() {
        let duplicate = ExactDesignParameterSetV1::new(vec![
            length_parameter("section_width", 20_000),
            length_parameter("section_width", 21_000),
        ]);
        assert!(matches!(
            duplicate.validate(),
            Err(ExactParameterError::DuplicateParameterId(_))
        ));
    }

    #[test]
    fn exact_length_unit_projection_is_rational() {
        let length = ExactDesignLengthUmV1::from_um(20_000);
        assert_eq!(
            length.millimetres_rational(),
            ExactRationalU64V1::new(20, 1).unwrap()
        );
        assert_eq!(
            length.metres_rational(),
            ExactRationalU64V1::new(1, 50).unwrap()
        );
    }

    #[test]
    fn zero_is_a_valid_primitive_but_positive_domain_rejects_it() {
        let zero = ExactDesignLengthUmV1::from_um(0);
        assert_eq!(zero.as_um(), 0);

        let domain = ExactLengthDomainV1::new(
            id("section_width"),
            true,
            2,
            ExactLengthDomainKindV1::Explicit {
                values: vec![zero, ExactDesignLengthUmV1::from_um(1)],
            },
        );
        assert_eq!(
            domain.validate(),
            Err(ExactParameterError::PositiveValueRequired)
        );
    }

    #[test]
    fn explicit_domain_order_is_nonsemantic_and_enumeration_is_sorted() {
        let left = ExactLengthDomainV1::new(
            id("section_width"),
            true,
            3,
            ExactLengthDomainKindV1::Explicit {
                values: vec![
                    ExactDesignLengthUmV1::from_um(22_000),
                    ExactDesignLengthUmV1::from_um(18_000),
                    ExactDesignLengthUmV1::from_um(20_000),
                ],
            },
        );
        let right = ExactLengthDomainV1::new(
            id("section_width"),
            true,
            3,
            ExactLengthDomainKindV1::Explicit {
                values: vec![
                    ExactDesignLengthUmV1::from_um(18_000),
                    ExactDesignLengthUmV1::from_um(20_000),
                    ExactDesignLengthUmV1::from_um(22_000),
                ],
            },
        );
        assert_eq!(left.domain_id().unwrap(), right.domain_id().unwrap());
        assert_eq!(
            left.enumerate().unwrap(),
            vec![
                ExactDesignLengthUmV1::from_um(18_000),
                ExactDesignLengthUmV1::from_um(20_000),
                ExactDesignLengthUmV1::from_um(22_000),
            ]
        );
    }

    #[test]
    fn golden_explicit_domain_transcript_and_digest_are_stable() {
        let domain = ExactLengthDomainV1::new(
            id("section_width"),
            true,
            5,
            ExactLengthDomainKindV1::Explicit {
                values: vec![
                    ExactDesignLengthUmV1::from_um(16_000),
                    ExactDesignLengthUmV1::from_um(18_000),
                    ExactDesignLengthUmV1::from_um(20_000),
                    ExactDesignLengthUmV1::from_um(22_000),
                    ExactDesignLengthUmV1::from_um(24_000),
                ],
            },
        );
        assert_eq!(
            hex(&domain.canonical_transcript().unwrap()),
            "000000000000002c73796d74686165612e726f626f742d64657369676e2e65786163742d6c656e6774682d646f6d61696e2e763100000001000000000000000d73656374696f6e5f776964746801000000050000000000000000050000000000003e8000000000000046500000000000004e2000000000000055f00000000000005dc0"
        );
        assert_eq!(
            domain.domain_id().unwrap().to_hex(),
            "62e6c2cd090bf1b954dbd1c74d93d594ef6734263e10f947303ede86406cd54f"
        );
    }

    #[test]
    fn stepped_domain_requires_exactly_reachable_upper_bound() {
        let domain = ExactLengthDomainV1::new(
            id("section_height"),
            true,
            10,
            ExactLengthDomainKindV1::Stepped {
                lower: ExactDesignLengthUmV1::from_um(1_000),
                upper: ExactDesignLengthUmV1::from_um(2_050),
                step: ExactDesignLengthUmV1::from_um(100),
            },
        );
        assert_eq!(
            domain.validate(),
            Err(ExactParameterError::UpperBoundNotReachable)
        );
    }

    #[test]
    fn stepped_domain_enumerates_completely_and_deterministically() {
        let domain = ExactLengthDomainV1::new(
            id("section_height"),
            true,
            4,
            ExactLengthDomainKindV1::Stepped {
                lower: ExactDesignLengthUmV1::from_um(1_000),
                upper: ExactDesignLengthUmV1::from_um(1_300),
                step: ExactDesignLengthUmV1::from_um(100),
            },
        );
        assert_eq!(
            domain.enumerate().unwrap(),
            vec![
                ExactDesignLengthUmV1::from_um(1_000),
                ExactDesignLengthUmV1::from_um(1_100),
                ExactDesignLengthUmV1::from_um(1_200),
                ExactDesignLengthUmV1::from_um(1_300),
            ]
        );
        assert!(
            domain
                .contains(ExactDesignLengthUmV1::from_um(1_200))
                .unwrap()
        );
        assert!(
            !domain
                .contains(ExactDesignLengthUmV1::from_um(1_250))
                .unwrap()
        );
    }

    #[test]
    fn domain_resource_limit_fails_closed() {
        let domain = ExactLengthDomainV1::new(
            id("section_width"),
            true,
            2,
            ExactLengthDomainKindV1::Explicit {
                values: vec![
                    ExactDesignLengthUmV1::from_um(1),
                    ExactDesignLengthUmV1::from_um(2),
                    ExactDesignLengthUmV1::from_um(3),
                ],
            },
        );
        assert!(matches!(
            domain.validate(),
            Err(ExactParameterError::DomainTooLarge {
                count: 3,
                max_values: 2
            })
        ));
    }

    #[test]
    fn parameter_set_identity_is_independent_of_allowed_domain() {
        let set_id = reference_set().parameter_set_id().unwrap();
        let narrow = ExactLengthDomainV1::new(
            id("section_width"),
            true,
            3,
            ExactLengthDomainKindV1::Explicit {
                values: vec![
                    ExactDesignLengthUmV1::from_um(18_000),
                    ExactDesignLengthUmV1::from_um(20_000),
                    ExactDesignLengthUmV1::from_um(22_000),
                ],
            },
        );
        let wide = ExactLengthDomainV1::new(
            id("section_width"),
            true,
            5,
            ExactLengthDomainKindV1::Explicit {
                values: vec![
                    ExactDesignLengthUmV1::from_um(16_000),
                    ExactDesignLengthUmV1::from_um(18_000),
                    ExactDesignLengthUmV1::from_um(20_000),
                    ExactDesignLengthUmV1::from_um(22_000),
                    ExactDesignLengthUmV1::from_um(24_000),
                ],
            },
        );

        assert_eq!(reference_set().parameter_set_id().unwrap(), set_id);
        assert_ne!(narrow.domain_id().unwrap(), wide.domain_id().unwrap());
    }

    #[test]
    fn invalid_parameter_id_rejects() {
        assert!(matches!(
            DesignParameterId::new("section width"),
            Err(ExactParameterError::InvalidParameterId(_))
        ));
    }

    fn hex(bytes: &[u8]) -> String {
        let mut output = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            use std::fmt::Write as _;
            let _ = write!(output, "{byte:02x}");
        }
        output
    }
}
