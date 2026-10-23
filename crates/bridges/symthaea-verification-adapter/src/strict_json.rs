//! Suite-neutral strict JSON/I-JSON boundary used by controller resolution and
//! cryptographic verification.
//!
//! The duplicate-name check happens during deserialization, before a generic
//! JSON object representation can collapse distinct wire members. Additional
//! Unicode and numeric interoperability checks then run recursively over the
//! decoded value.

use serde_json::{Map, Value};
use symthaea_epistemic_types::VerificationFailure;

use crate::SnapshotError;

struct StrictJsonValue;

impl<'de> serde::de::DeserializeSeed<'de> for StrictJsonValue {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonValueVisitor)
    }
}

struct StrictJsonValueVisitor;

impl<'de> serde::de::Visitor<'de> for StrictJsonValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("valid JSON value with unique object member names")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E> {
        serde_json::Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| serde::de::Error::custom("JSON number is not finite"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(Value::String(value))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(StrictJsonValue)? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut object = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if object.contains_key(&key) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate JSON object member name: {key}"
                )));
            }
            let value = map.next_value_seed(StrictJsonValue)?;
            object.insert(key, value);
        }
        Ok(Value::Object(object))
    }
}

fn reject_negative_zero_wire(bytes: &[u8]) -> Result<(), SnapshotError> {
    let mut index = 0;

    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                index += 1;
                while index < bytes.len() {
                    match bytes[index] {
                        b'\\' => {
                            // Skip the escaped byte here; malformed escapes are still
                            // rejected by serde_json after this lexical safety scan.
                            index = index.saturating_add(2);
                        }
                        b'"' => {
                            index += 1;
                            break;
                        }
                        _ => index += 1,
                    }
                }
            }
            b'-' => {
                let start = index;
                index += 1;
                while index < bytes.len()
                    && matches!(
                        bytes[index],
                        b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-'
                    )
                {
                    index += 1;
                }
                let token = &bytes[start..index];
                if is_negative_zero_number(token) {
                    return Err(SnapshotError::Verification(
                        VerificationFailure::Structural(
                            "strict JCS parsing rejected wire-level negative zero".into(),
                        ),
                    ));
                }
            }
            _ => index += 1,
        }
    }

    Ok(())
}

fn is_negative_zero_number(token: &[u8]) -> bool {
    if token.first() != Some(&b'-') {
        return false;
    }

    let mantissa_end = token
        .iter()
        .position(|byte| matches!(byte, b'e' | b'E'))
        .unwrap_or(token.len());
    let mantissa = &token[1..mantissa_end];

    let mut saw_digit = false;
    for byte in mantissa {
        match byte {
            b'0' => saw_digit = true,
            b'.' => {}
            _ => return false,
        }
    }
    saw_digit
}

pub(crate) fn parse_strict_json(bytes: &[u8]) -> Result<Value, SnapshotError> {
    reject_negative_zero_wire(bytes)?;
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = serde::de::DeserializeSeed::deserialize(StrictJsonValue, &mut deserializer)
        .map_err(|error| {
            SnapshotError::Verification(VerificationFailure::Structural(format!(
                "strict JSON parsing failed: {error}"
            )))
        })?;
    deserializer.end().map_err(|error| {
        SnapshotError::Verification(VerificationFailure::Structural(format!(
            "strict JSON parsing rejected trailing data: {error}"
        )))
    })?;
    validate_strict_ijson_value(&value)?;
    Ok(value)
}

pub(crate) fn validate_strict_ijson_value(value: &Value) -> Result<(), SnapshotError> {
    match value {
        Value::Null | Value::Bool(_) => Ok(()),
        Value::Number(number) => {
            if let Some(value) = number.as_f64() {
                if value == 0.0 && value.is_sign_negative() {
                    return Err(SnapshotError::Verification(
                        VerificationFailure::Structural(
                            "strict JCS parsing rejected programmatic negative zero".into(),
                        ),
                    ));
                }
            }
            if let Some(value) = number.as_i64() {
                if !is_binary64_integer_exact(value.unsigned_abs()) {
                    return Err(SnapshotError::Verification(
                        VerificationFailure::Structural(
                            "strict JSON numeric interoperability profile rejected an integer that cannot be represented exactly by IEEE-754 binary64"
                                .into(),
                        ),
                    ));
                }
            } else if let Some(value) = number.as_u64() {
                if !is_binary64_integer_exact(value) {
                    return Err(SnapshotError::Verification(
                        VerificationFailure::Structural(
                            "strict JSON numeric interoperability profile rejected an integer that cannot be represented exactly by IEEE-754 binary64"
                                .into(),
                        ),
                    ));
                }
            } else if number
                .as_f64()
                .map(|value| !value.is_finite())
                .unwrap_or(true)
            {
                return Err(SnapshotError::Verification(
                    VerificationFailure::Structural(
                        "strict JSON numeric interoperability profile rejected a non-finite JSON number"
                            .into(),
                    ),
                ));
            }
            Ok(())
        }
        Value::String(text) => {
            if text.chars().any(is_forbidden_ijson_code_point) {
                return Err(SnapshotError::Verification(
                    VerificationFailure::Structural(
                        "strict I-JSON parsing rejected a Unicode noncharacter".into(),
                    ),
                ));
            }
            Ok(())
        }
        Value::Array(values) => {
            for value in values {
                validate_strict_ijson_value(value)?;
            }
            Ok(())
        }
        Value::Object(object) => {
            for (key, value) in object {
                if key.chars().any(is_forbidden_ijson_code_point) {
                    return Err(SnapshotError::Verification(
                        VerificationFailure::Structural(
                            "strict I-JSON parsing rejected a Unicode noncharacter in an object member name"
                                .into(),
                        ),
                    ));
                }
                validate_strict_ijson_value(value)?;
            }
            Ok(())
        }
    }
}

fn is_binary64_integer_exact(value: u64) -> bool {
    if value == 0 {
        return true;
    }
    let significant_bits = 64 - value.leading_zeros();
    if significant_bits <= 53 {
        return true;
    }
    let discarded_bits = significant_bits - 53;
    (value & ((1u64 << discarded_bits) - 1)) == 0
}

fn is_forbidden_ijson_code_point(ch: char) -> bool {
    let code_point = ch as u32;
    (0xFDD0..=0xFDEF).contains(&code_point) || (code_point & 0xFFFF) >= 0xFFFE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wire_negative_zero_but_accepts_text_containing_negative_zero() {
        for wire in [
            br#"{"value":-0}"#,
            br#"{"value":-0.0}"#,
            br#"{"value":-0e0}"#,
            br#"{"value":-0.00E+42}"#,
        ] {
            assert!(matches!(
                parse_strict_json(wire),
                Err(SnapshotError::Verification(
                    VerificationFailure::Structural(message)
                )) if message.contains("negative zero")
            ));
        }

        let text = parse_strict_json(br#"{"value":"-0","nested":["-0"]}"#).unwrap();
        assert_eq!(text["value"], Value::String("-0".into()));
        assert_eq!(text["nested"][0], Value::String("-0".into()));
    }

    #[test]
    fn rejects_programmatic_negative_zero() {
        let number = serde_json::Number::from_f64(-0.0).unwrap();
        let value = Value::Number(number);
        assert!(matches!(
            validate_strict_ijson_value(&value),
            Err(SnapshotError::Verification(
                VerificationFailure::Structural(message)
            )) if message.contains("negative zero")
        ));
    }

    #[test]
    fn accepts_small_negative_nonzero_number() {
        let wire = br#"{"value":-0.0000000000000000001}"#;
        let value = parse_strict_json(wire).unwrap();
        assert!(value["value"].as_f64().unwrap().is_sign_negative());
        assert_ne!(value["value"].as_f64(), Some(0.0));
    }
}
