//! Strict raw-JSON admission before study-schema deserialization.
//!
//! This module is deliberately a boundary, not a canonicalization layer:
//! exact input bytes are hashed first, encoding is fixed to UTF-8, decoded
//! object member names must be unique, and only then is the requested schema
//! type materialized.

use serde::de::{self, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

pub const DEFAULT_MAX_RAW_BYTES: usize = 16 * 1024 * 1024;
pub const DEFAULT_MAX_DEPTH: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionReason {
    ResourceLimit,
    BomForbidden,
    InvalidUtf8,
    DuplicateDecodedName,
    InvalidJson,
    TrailingData,
    UnicodeScalarPolicy,
}

#[derive(Debug)]
pub enum BoundaryResult<T> {
    Accepted {
        raw_sha256: String,
        document: T,
    },
    Rejected {
        raw_sha256: String,
        reason: RejectionReason,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct AdmissionLimits {
    pub max_raw_bytes: usize,
    pub max_depth: usize,
}

impl Default for AdmissionLimits {
    fn default() -> Self {
        Self {
            max_raw_bytes: DEFAULT_MAX_RAW_BYTES,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }
}

#[derive(Debug)]
struct GateState {
    limits: AdmissionLimits,
    depth: usize,
}

struct GateSeed<'a> {
    state: &'a mut GateState,
}

impl<'de> serde::de::DeserializeSeed<'de> for GateSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(GateVisitor { state: self.state })
    }
}

struct GateVisitor<'a> {
    state: &'a mut GateState,
}

impl<'de> Visitor<'de> for GateVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("any valid JSON value")
    }

    fn visit_bool<E>(self, _: bool) -> Result<(), E> { Ok(()) }
    fn visit_i64<E>(self, _: i64) -> Result<(), E> { Ok(()) }
    fn visit_u64<E>(self, _: u64) -> Result<(), E> { Ok(()) }
    fn visit_f64<E>(self, _: f64) -> Result<(), E> { Ok(()) }
    fn visit_str<E>(self, _: &str) -> Result<(), E> { Ok(()) }
    fn visit_string<E>(self, _: String) -> Result<(), E> { Ok(()) }
    fn visit_none<E>(self) -> Result<(), E> { Ok(()) }
    fn visit_unit<E>(self) -> Result<(), E> { Ok(()) }

    fn visit_some<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: Deserializer<'de>,
    {
        GateSeed { state: self.state }.deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut access: A) -> Result<(), A::Error>
    where
        A: SeqAccess<'de>,
    {
        self.enter::<A::Error>()?;
        while access
            .next_element_seed(GateSeed { state: self.state })?
            .is_some()
        {}
        self.leave();
        Ok(())
    }

    fn visit_map<A>(self, mut access: A) -> Result<(), A::Error>
    where
        A: MapAccess<'de>,
    {
        self.enter::<A::Error>()?;
        let mut names = BTreeSet::<String>::new();
        while let Some(name) = access.next_key::<String>()? {
            if !names.insert(name) {
                return Err(de::Error::custom("duplicate decoded object member name"));
            }
            access.next_value_seed(GateSeed { state: self.state })?;
        }
        self.leave();
        Ok(())
    }
}

impl GateVisitor<'_> {
    fn enter<E: de::Error>(&mut self) -> Result<(), E> {
        if self.state.depth >= self.state.limits.max_depth {
            return Err(E::custom("maximum JSON nesting depth exceeded"));
        }
        self.state.depth += 1;
        Ok(())
    }

    fn leave(&mut self) {
        self.state.depth -= 1;
    }
}

fn raw_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    format!("{digest:x}")
}

fn classify_gate_error(error: &serde_json::Error) -> RejectionReason {
    let message = error.to_string();
    if message.contains("duplicate decoded object member name") {
        RejectionReason::DuplicateDecodedName
    } else if message.contains("maximum JSON nesting depth exceeded") {
        RejectionReason::ResourceLimit
    } else if message.contains("surrogate") {
        RejectionReason::UnicodeScalarPolicy
    } else {
        RejectionReason::InvalidJson
    }
}

/// Admit exact raw JSON bytes before materializing the requested Rust type.
///
/// The input slice is never mutated between the gate and schema pass. The
/// gate performs no serde_json::Value construction.
pub fn admit<T: DeserializeOwned>(
    bytes: &[u8],
    limits: AdmissionLimits,
) -> BoundaryResult<T> {
    let raw_sha256 = raw_sha256(bytes);

    if bytes.len() > limits.max_raw_bytes {
        return BoundaryResult::Rejected {
            raw_sha256,
            reason: RejectionReason::ResourceLimit,
        };
    }

    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return BoundaryResult::Rejected {
            raw_sha256,
            reason: RejectionReason::BomForbidden,
        };
    }

    let text = match std::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(_) => {
            return BoundaryResult::Rejected {
                raw_sha256,
                reason: RejectionReason::InvalidUtf8,
            };
        }
    };

    let mut gate_state = GateState { limits, depth: 0 };
    let mut gate = serde_json::Deserializer::from_str(text);
    if let Err(error) = GateSeed {
        state: &mut gate_state,
    }
    .deserialize(&mut gate)
    {
        return BoundaryResult::Rejected {
            raw_sha256,
            reason: classify_gate_error(&error),
        };
    }

    if gate.end().is_err() {
        return BoundaryResult::Rejected {
            raw_sha256,
            reason: RejectionReason::TrailingData,
        };
    }

    BoundaryResult::Accepted {
        raw_sha256,
        document: text.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admit_value(bytes: &[u8]) -> BoundaryResult<serde_json::Value> {
        admit(bytes, AdmissionLimits::default())
    }

    #[test]
    fn accepts_utf8_and_hashes_exact_bytes() {
        let input = br#"{"message":"cafe"}"#;
        match admit_value(input) {
            BoundaryResult::Accepted { raw_sha256, .. } => {
                assert_eq!(
                    raw_sha256,
                    "93a4ee8700bc636139c335991cc481c8285078aa1b97d74f5fa90b3258744dd3"
                );
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn rejects_utf8_bom() {
        let input = b"\xEF\xBB\xBF{}";
        assert!(matches!(
            admit_value(input),
            BoundaryResult::Rejected { reason: RejectionReason::BomForbidden, .. }
        ));
    }

    #[test]
    fn rejects_non_utf8_encodings() {
        for input in [
            b"\xFF\xFE{\0}\0".as_slice(),
            b"\xFE\xFF\0{\0}".as_slice(),
            b"\xFF\xFE\0\0{\0\0\0}".as_slice(),
            b"\0\0\xFE\xFF\0\0\0{".as_slice(),
        ] {
            assert!(matches!(
                admit_value(input),
                BoundaryResult::Rejected { reason: RejectionReason::InvalidUtf8, .. }
            ));
        }
    }

    #[test]
    fn rejects_duplicate_literal_and_escaped_equivalent_names() {
        for input in [
            br#"{"a":1,"a":2}"#,
            br#"{"a":1,"\u0061":2}"#,
        ] {
            assert!(matches!(
                admit_value(input),
                BoundaryResult::Rejected { reason: RejectionReason::DuplicateDecodedName, .. }
            ));
        }
    }

    #[test]
    fn rejects_non_standard_numbers() {
        for input in [
            br#"NaN"#.as_slice(),
            br#"Infinity"#.as_slice(),
            br#"-Infinity"#.as_slice(),
        ] {
            assert!(matches!(
                admit_value(input),
                BoundaryResult::Rejected { reason: RejectionReason::InvalidJson, .. }
            ));
        }
    }

    #[test]
    fn accepts_trailing_json_whitespace() {
        assert!(matches!(
            admit_value(b"{ } \n\t\r"),
            BoundaryResult::Accepted { .. }
        ));
    }

    #[test]
    fn rejects_trailing_data() {
        assert!(matches!(
            admit_value(br#"{} {}"#),
            BoundaryResult::Rejected { reason: RejectionReason::TrailingData, .. }
        ));
    }

    #[test]
    fn rejects_invalid_utf8_before_parser() {
        assert!(matches!(
            admit_value(b"{\xC0\xAF}"),
            BoundaryResult::Rejected { reason: RejectionReason::InvalidUtf8, .. }
        ));
    }

    #[test]
    fn rejects_lone_surrogate_escape() {
        assert!(matches!(
            admit_value(br#"{"x":"\uDEAD"}"#),
            BoundaryResult::Rejected { reason: RejectionReason::UnicodeScalarPolicy, .. }
        ));
    }

    #[test]
    fn rejects_excessive_depth() {
        let input = b"[[[[[[[[[[0]]]]]]]]]]";
        let limits = AdmissionLimits {
            max_raw_bytes: DEFAULT_MAX_RAW_BYTES,
            max_depth: 4,
        };
        assert!(matches!(
            admit(input, limits),
            BoundaryResult::Rejected { reason: RejectionReason::ResourceLimit, .. }
        ));
    }

    #[test]
    fn rejection_does_not_materialize_schema_value() {
        #[derive(Debug, Deserialize)]
        struct Marker { marker: String }

        let result = admit::<Marker>(br#"{"marker":1,"marker":2}"#, AdmissionLimits::default());
        assert!(matches!(
            result,
            BoundaryResult::Rejected { reason: RejectionReason::DuplicateDecodedName, .. }
        ));
    }
}
