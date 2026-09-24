// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-by-definition design parameter semantics for ROB-DESIGN-001B0.
//!
//! This module is intentionally authority-free. It identifies designer-chosen
//! nominal intent; it does not represent measurement, uncertainty, as-built
//! state, qualification, or physical truth.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

pub const DESIGN_PARAMETER_SCHEMA_ID: &str = "symthaea.robot-design.parameters.v1";
pub const DESIGN_PARAMETER_SCHEMA_VERSION: u32 = 1;
pub const DESIGN_SEARCH_DOMAIN_SCHEMA_ID: &str = "symthaea.robot-design.search-domain.v1";
pub const DESIGN_SEARCH_DOMAIN_SCHEMA_VERSION: u32 = 1;
pub const MAX_DISCRETE_DOMAIN_VALUES: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DesignParameterError {
    InvalidParameterId(String),
    EmptyParameterSet,
    DuplicateParameterId(String),
    EmptyDomain,
    DuplicateDomainValue { parameter: String, value_um: u64 },
    InvalidSteppedDomain(&'static str),
    DomainTooLarge { parameter: String, count: usize },
    ParameterSetDoesNotMatchDomain,
}

impl fmt::Display for DesignParameterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameterId(value) => write!(formatter, "invalid design parameter id: {value}"),
            Self::EmptyParameterSet => formatter.write_str("design parameter set must not be empty"),
            Self::DuplicateParameterId(value) => {
                write!(formatter, "duplicate design parameter id: {value}")
            }
            Self::EmptyDomain => formatter.write_str("design search domain must not be empty"),
            Self::DuplicateDomainValue { parameter, value_um } => write!(
                formatter,
                "duplicate value {value_um} um in domain for parameter {parameter}"
            ),
            Self::InvalidSteppedDomain(reason) => write!(formatter, "invalid stepped domain: {reason}"),
            Self::DomainTooLarge { parameter, count } => write!(
                formatter,
                "domain for parameter {parameter} has {count} values, above limit {MAX_DISCRETE_DOMAIN_VALUES}"
            ),
            Self::ParameterSetDoesNotMatchDomain => {
                formatter.write_str("parameter set does not match search-domain parameter/value membership")
            }
        }
    }
}

impl std::error::Error for DesignParameterError {}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DesignParameterId(String);

impl DesignParameterId {
    pub fn new(value: impl Into<String>) -> Result<Self, DesignParameterError> {
        let value = value.into();
        let valid = !value.is_empty()
            && value.len() <= 128
            && value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
            });
        if valid {
            Ok(Self(value))
        } else {
            Err(DesignParameterError::InvalidParameterId(value))
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

/// Exact nominal design length in integer micrometres.
///
/// Zero is representable at this primitive layer. Application profiles such as
/// C0 may impose a strictly-positive admissibility rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DesignLengthUm(u64);

impl DesignLengthUm {
    pub const fn from_micrometres(value: u64) -> Self {
        Self(value)
    }

    pub const fn micrometres(self) -> u64 {
        self.0
    }

    /// Convenience numerical projection only; never part of semantic identity.
    pub fn to_millimetres_f64(self) -> f64 {
        self.0 as f64 / 1_000.0
    }

    /// Convenience numerical projection only; never part of semantic identity.
    pub fn to_metres_f64(self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignLengthParameterV1 {
    pub id: DesignParameterId,
    pub value: DesignLengthUm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DesignParameterSetId([u8; 32]);

impl DesignParameterSetId {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex_digest(self.0)
    }
}

impl fmt::Display for DesignParameterSetId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignParameterSetV1 {
    values: Vec<DesignLengthParameterV1>,
}

impl DesignParameterSetV1 {
    pub fn new(values: Vec<DesignLengthParameterV1>) -> Result<Self, DesignParameterError> {
        let mut result = Self { values };
        result.canonicalize()?;
        Ok(result)
    }

    pub fn values(&self) -> &[DesignLengthParameterV1] {
        &self.values
    }

    pub fn value(&self, id: &str) -> Option<DesignLengthUm> {
        self.values
            .iter()
            .find(|entry| entry.id.as_str() == id)
            .map(|entry| entry.value)
    }

    pub fn validate(&self) -> Result<(), DesignParameterError> {
        if self.values.is_empty() {
            return Err(DesignParameterError::EmptyParameterSet);
        }
        let mut ids = BTreeSet::new();
        for entry in &self.values {
            DesignParameterId::new(entry.id.as_str())?;
            if !ids.insert(entry.id.as_str()) {
                return Err(DesignParameterError::DuplicateParameterId(entry.id.to_string()));
            }
        }
        Ok(())
    }

    pub fn canonical_transcript(&self) -> Result<Vec<u8>, DesignParameterError> {
        self.validate()?;
        let mut values = self.values.iter().collect::<Vec<_>>();
        values.sort_by(|left, right| left.id.cmp(&right.id));

        let mut out = Vec::new();
        put_str(&mut out, DESIGN_PARAMETER_SCHEMA_ID);
        put_u32(&mut out, DESIGN_PARAMETER_SCHEMA_VERSION);
        put_len(&mut out, values.len());
        for entry in values {
            put_str(&mut out, entry.id.as_str());
            put_u8(&mut out, 0); // V1 quantity tag: exact length in micrometres.
            put_u64(&mut out, entry.value.micrometres());
        }
        Ok(out)
    }

    pub fn id(&self) -> Result<DesignParameterSetId, DesignParameterError> {
        let transcript = self.canonical_transcript()?;
        Ok(DesignParameterSetId(Sha256::digest(transcript).into()))
    }

    fn canonicalize(&mut self) -> Result<(), DesignParameterError> {
        self.validate()?;
        self.values.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DesignLengthDomainV1 {
    Explicit {
        id: DesignParameterId,
        values: Vec<DesignLengthUm>,
    },
    SteppedInclusive {
        id: DesignParameterId,
        lower: DesignLengthUm,
        upper: DesignLengthUm,
        step: DesignLengthUm,
    },
}

impl DesignLengthDomainV1 {
    pub fn parameter_id(&self) -> &DesignParameterId {
        match self {
            Self::Explicit { id, .. } | Self::SteppedInclusive { id, .. } => id,
        }
    }

    pub fn canonical_values(&self) -> Result<Vec<DesignLengthUm>, DesignParameterError> {
        match self {
            Self::Explicit { id, values } => {
                if values.is_empty() {
                    return Err(DesignParameterError::InvalidSteppedDomain(
                        "explicit domain must contain at least one value",
                    ));
                }
                if values.len() > MAX_DISCRETE_DOMAIN_VALUES {
                    return Err(DesignParameterError::DomainTooLarge {
                        parameter: id.to_string(),
                        count: values.len(),
                    });
                }
                let mut canonical = values.clone();
                canonical.sort_unstable();
                for pair in canonical.windows(2) {
                    if pair[0] == pair[1] {
                        return Err(DesignParameterError::DuplicateDomainValue {
                            parameter: id.to_string(),
                            value_um: pair[0].micrometres(),
                        });
                    }
                }
                Ok(canonical)
            }
            Self::SteppedInclusive {
                id,
                lower,
                upper,
                step,
            } => {
                let lower = lower.micrometres();
                let upper = upper.micrometres();
                let step = step.micrometres();
                if step == 0 {
                    return Err(DesignParameterError::InvalidSteppedDomain("step must be non-zero"));
                }
                if lower > upper {
                    return Err(DesignParameterError::InvalidSteppedDomain(
                        "lower bound exceeds upper bound",
                    ));
                }
                let span = upper - lower;
                if span % step != 0 {
                    return Err(DesignParameterError::InvalidSteppedDomain(
                        "upper bound must be exactly reachable by repeated step",
                    ));
                }
                let count_u64 = span / step + 1;
                let count = usize::try_from(count_u64).map_err(|_| {
                    DesignParameterError::DomainTooLarge {
                        parameter: id.to_string(),
                        count: usize::MAX,
                    }
                })?;
                if count > MAX_DISCRETE_DOMAIN_VALUES {
                    return Err(DesignParameterError::DomainTooLarge {
                        parameter: id.to_string(),
                        count,
                    });
                }
                let mut values = Vec::with_capacity(count);
                for index in 0..count_u64 {
                    let offset = step.checked_mul(index).ok_or(
                        DesignParameterError::InvalidSteppedDomain("step multiplication overflow"),
                    )?;
                    let value = lower.checked_add(offset).ok_or(
                        DesignParameterError::InvalidSteppedDomain("domain value overflow"),
                    )?;
                    values.push(DesignLengthUm::from_micrometres(value));
                }
                Ok(values)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DesignSearchDomainId([u8; 32]);

impl DesignSearchDomainId {
    pub const fn into_bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn to_hex(self) -> String {
        hex_digest(self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignSearchDomainV1 {
    parameters: Vec<DesignLengthDomainV1>,
}

impl DesignSearchDomainV1 {
    pub fn new(parameters: Vec<DesignLengthDomainV1>) -> Result<Self, DesignParameterError> {
        let mut result = Self { parameters };
        result.canonicalize()?;
        Ok(result)
    }

    pub fn parameters(&self) -> &[DesignLengthDomainV1] {
        &self.parameters
    }

    pub fn validate(&self) -> Result<(), DesignParameterError> {
        if self.parameters.is_empty() {
            return Err(DesignParameterError::EmptyDomain);
        }
        let mut ids = BTreeSet::new();
        for parameter in &self.parameters {
            DesignParameterId::new(parameter.parameter_id().as_str())?;
            if !ids.insert(parameter.parameter_id().as_str()) {
                return Err(DesignParameterError::DuplicateParameterId(
                    parameter.parameter_id().to_string(),
                ));
            }
            parameter.canonical_values()?;
        }
        Ok(())
    }

    pub fn canonical_transcript(&self) -> Result<Vec<u8>, DesignParameterError> {
        self.validate()?;
        let mut parameters = self.parameters.iter().collect::<Vec<_>>();
        parameters.sort_by(|left, right| left.parameter_id().cmp(right.parameter_id()));

        let mut out = Vec::new();
        put_str(&mut out, DESIGN_SEARCH_DOMAIN_SCHEMA_ID);
        put_u32(&mut out, DESIGN_SEARCH_DOMAIN_SCHEMA_VERSION);
        put_len(&mut out, parameters.len());
        for parameter in parameters {
            put_str(&mut out, parameter.parameter_id().as_str());
            let values = parameter.canonical_values()?;
            put_len(&mut out, values.len());
            for value in values {
                put_u64(&mut out, value.micrometres());
            }
        }
        Ok(out)
    }

    pub fn id(&self) -> Result<DesignSearchDomainId, DesignParameterError> {
        let transcript = self.canonical_transcript()?;
        Ok(DesignSearchDomainId(Sha256::digest(transcript).into()))
    }

    pub fn admits(&self, set: &DesignParameterSetV1) -> Result<bool, DesignParameterError> {
        self.validate()?;
        set.validate()?;
        if self.parameters.len() != set.values().len() {
            return Ok(false);
        }
        for parameter in &self.parameters {
            let Some(value) = set.value(parameter.parameter_id().as_str()) else {
                return Ok(false);
            };
            if parameter.canonical_values()?.binary_search(&value).is_err() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn canonicalize(&mut self) -> Result<(), DesignParameterError> {
        self.validate()?;
        self.parameters
            .sort_by(|left, right| left.parameter_id().cmp(right.parameter_id()));
        Ok(())
    }
}

fn hex_digest(bytes: [u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(64);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
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

    fn value(name: &str, micrometres: u64) -> DesignLengthParameterV1 {
        DesignLengthParameterV1 {
            id: id(name),
            value: DesignLengthUm::from_micrometres(micrometres),
        }
    }

    #[test]
    fn parameter_set_identity_is_order_invariant() {
        let first = DesignParameterSetV1::new(vec![
            value("section-width", 20_000),
            value("section-height", 6_000),
        ])
        .unwrap();
        let second = DesignParameterSetV1::new(vec![
            value("section-height", 6_000),
            value("section-width", 20_000),
        ])
        .unwrap();
        assert_eq!(first.id().unwrap(), second.id().unwrap());
    }

    #[test]
    fn golden_parameter_set_digest_is_stable() {
        let set = DesignParameterSetV1::new(vec![
            value("section-width", 20_000),
            value("section-height", 6_000),
        ])
        .unwrap();
        assert_eq!(
            set.id().unwrap().to_hex(),
            "c48c96ac91bea5d316ec0abf14341069002c6b88a8a8838ec0abdeda0f88a226"
        );
    }

    #[test]
    fn changed_value_changes_parameter_set_identity() {
        let baseline = DesignParameterSetV1::new(vec![value("section-width", 20_000)]).unwrap();
        let changed = DesignParameterSetV1::new(vec![value("section-width", 20_001)]).unwrap();
        assert_ne!(baseline.id().unwrap(), changed.id().unwrap());
    }

    #[test]
    fn duplicate_parameter_ids_reject() {
        let error = DesignParameterSetV1::new(vec![
            value("section-width", 20_000),
            value("section-width", 21_000),
        ])
        .unwrap_err();
        assert!(matches!(error, DesignParameterError::DuplicateParameterId(_)));
    }

    #[test]
    fn explicit_and_stepped_domains_with_same_values_have_same_identity() {
        let explicit = DesignSearchDomainV1::new(vec![DesignLengthDomainV1::Explicit {
            id: id("section-width"),
            values: vec![
                DesignLengthUm::from_micrometres(8_000),
                DesignLengthUm::from_micrometres(9_000),
                DesignLengthUm::from_micrometres(10_000),
            ],
        }])
        .unwrap();
        let stepped = DesignSearchDomainV1::new(vec![DesignLengthDomainV1::SteppedInclusive {
            id: id("section-width"),
            lower: DesignLengthUm::from_micrometres(8_000),
            upper: DesignLengthUm::from_micrometres(10_000),
            step: DesignLengthUm::from_micrometres(1_000),
        }])
        .unwrap();
        assert_eq!(explicit.id().unwrap(), stepped.id().unwrap());
    }

    #[test]
    fn parameter_set_identity_does_not_depend_on_search_domain() {
        let set = DesignParameterSetV1::new(vec![value("section-width", 20_000)]).unwrap();
        let narrow = DesignSearchDomainV1::new(vec![DesignLengthDomainV1::Explicit {
            id: id("section-width"),
            values: vec![DesignLengthUm::from_micrometres(20_000)],
        }])
        .unwrap();
        let broad = DesignSearchDomainV1::new(vec![DesignLengthDomainV1::Explicit {
            id: id("section-width"),
            values: vec![
                DesignLengthUm::from_micrometres(18_000),
                DesignLengthUm::from_micrometres(20_000),
                DesignLengthUm::from_micrometres(22_000),
            ],
        }])
        .unwrap();
        assert_ne!(narrow.id().unwrap(), broad.id().unwrap());
        assert!(narrow.admits(&set).unwrap());
        assert!(broad.admits(&set).unwrap());
        let stable_set_id = set.id().unwrap();
        assert_eq!(stable_set_id, set.id().unwrap());
    }

    #[test]
    fn invalid_stepped_domains_fail_closed() {
        let zero_step = DesignLengthDomainV1::SteppedInclusive {
            id: id("section-height"),
            lower: DesignLengthUm::from_micrometres(5_000),
            upper: DesignLengthUm::from_micrometres(7_000),
            step: DesignLengthUm::from_micrometres(0),
        };
        assert!(zero_step.canonical_values().is_err());

        let unreachable_upper = DesignLengthDomainV1::SteppedInclusive {
            id: id("section-height"),
            lower: DesignLengthUm::from_micrometres(5_000),
            upper: DesignLengthUm::from_micrometres(7_100),
            step: DesignLengthUm::from_micrometres(1_000),
        };
        assert!(unreachable_upper.canonical_values().is_err());
    }

    #[test]
    fn duplicate_explicit_domain_values_reject() {
        let domain = DesignLengthDomainV1::Explicit {
            id: id("section-height"),
            values: vec![
                DesignLengthUm::from_micrometres(6_000),
                DesignLengthUm::from_micrometres(6_000),
            ],
        };
        assert!(matches!(
            domain.canonical_values().unwrap_err(),
            DesignParameterError::DuplicateDomainValue { .. }
        ));
    }

    #[test]
    fn primitive_length_projection_is_not_identity() {
        let length = DesignLengthUm::from_micrometres(20_000);
        assert_eq!(length.micrometres(), 20_000);
        assert_eq!(length.to_millimetres_f64(), 20.0);
        assert_eq!(length.to_metres_f64(), 0.02);
    }
}
