// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned admission for persisted keeper structural fingerprints.
//!
//! Atlas already writes a structural vector at keep time, but restart-time code
//! must not treat an arbitrary JSON array as compatible with the current
//! fingerprint layer layout. This module gives persisted vectors a narrow,
//! fail-closed admission boundary. It never recomposes a keeper from its recipe.

use std::fmt;

use serde_json::Value;
use symthaea_music_theory::fingerprint::STRUCT_DIMS;

/// First persisted keeper-fingerprint schema.
///
/// This versions the meaning/order of the vector independently from its length:
/// a future schema can keep `STRUCT_DIMS` unchanged while changing a layer's
/// semantics, so dimension equality alone is not sufficient compatibility proof.
pub const KEEPER_FINGERPRINT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct PersistedKeeperFingerprint {
    pub schema_version: u32,
    pub values: [f64; STRUCT_DIMS],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeeperFingerprintEvidenceError {
    PartialRecord,
    UnsupportedSchema(u32),
    InvalidDimensions(u64),
    InvalidVectorLength(usize),
    NonNumericValue(usize),
    NonFiniteValue(usize),
}

impl fmt::Display for KeeperFingerprintEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PartialRecord => write!(
                f,
                "keeper structural fingerprint record is only partially present"
            ),
            Self::UnsupportedSchema(version) => {
                write!(f, "unsupported keeper fingerprint schema version {version}")
            }
            Self::InvalidDimensions(dims) => write!(
                f,
                "keeper fingerprint declares {dims} dimensions; expected {STRUCT_DIMS}"
            ),
            Self::InvalidVectorLength(len) => write!(
                f,
                "keeper fingerprint contains {len} values; expected {STRUCT_DIMS}"
            ),
            Self::NonNumericValue(index) => {
                write!(f, "keeper fingerprint value {index} is not numeric")
            }
            Self::NonFiniteValue(index) => {
                write!(f, "keeper fingerprint value {index} is not finite")
            }
        }
    }
}

impl std::error::Error for KeeperFingerprintEvidenceError {}

/// Admit a persisted fingerprint from one keeper JSONL entry.
///
/// `Ok(None)` has one meaning only: this is a legacy/pre-schema entry with none
/// of the three fingerprint fields present. If any one field is present, all are
/// required and must validate. Callers must never turn `None` or `Err` into a
/// current-engine recomposition of the historical keeper.
pub fn fingerprint_from_keeper_entry(
    entry: &Value,
) -> Result<Option<PersistedKeeperFingerprint>, KeeperFingerprintEvidenceError> {
    let schema = entry.get("structural_fingerprint_schema");
    let dims = entry.get("structural_fingerprint_dims");
    let vector = entry.get("structural_fingerprint");

    if schema.is_none() && dims.is_none() && vector.is_none() {
        return Ok(None);
    }
    let (Some(schema), Some(dims), Some(vector)) = (schema, dims, vector) else {
        return Err(KeeperFingerprintEvidenceError::PartialRecord);
    };

    let schema = schema
        .as_u64()
        .ok_or(KeeperFingerprintEvidenceError::PartialRecord)?;
    let schema = u32::try_from(schema)
        .map_err(|_| KeeperFingerprintEvidenceError::UnsupportedSchema(u32::MAX))?;
    if schema != KEEPER_FINGERPRINT_SCHEMA_VERSION {
        return Err(KeeperFingerprintEvidenceError::UnsupportedSchema(schema));
    }

    let dims = dims
        .as_u64()
        .ok_or(KeeperFingerprintEvidenceError::PartialRecord)?;
    if dims != STRUCT_DIMS as u64 {
        return Err(KeeperFingerprintEvidenceError::InvalidDimensions(dims));
    }

    let values = vector
        .as_array()
        .ok_or(KeeperFingerprintEvidenceError::PartialRecord)?;
    if values.len() != STRUCT_DIMS {
        return Err(KeeperFingerprintEvidenceError::InvalidVectorLength(
            values.len(),
        ));
    }

    let mut exact = [0.0_f64; STRUCT_DIMS];
    for (index, value) in values.iter().enumerate() {
        let number = value
            .as_f64()
            .ok_or(KeeperFingerprintEvidenceError::NonNumericValue(index))?;
        if !number.is_finite() {
            return Err(KeeperFingerprintEvidenceError::NonFiniteValue(index));
        }
        exact[index] = number;
    }

    Ok(Some(PersistedKeeperFingerprint {
        schema_version: schema,
        values: exact,
    }))
}

/// Construct the exact JSON fields new-format keeper publication should place
/// in `keepers.jsonl`. The vector is accepted by value so callers cannot change
/// it between dimension/schema recording and serialization.
pub fn fingerprint_fields(
    values: [f64; STRUCT_DIMS],
) -> Result<Value, KeeperFingerprintEvidenceError> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(KeeperFingerprintEvidenceError::NonFiniteValue(index));
        }
    }
    Ok(serde_json::json!({
        "structural_fingerprint_schema": KEEPER_FINGERPRINT_SCHEMA_VERSION,
        "structural_fingerprint_dims": STRUCT_DIMS,
        "structural_fingerprint": values.to_vec(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector() -> [f64; STRUCT_DIMS] {
        std::array::from_fn(|index| index as f64 / STRUCT_DIMS as f64)
    }

    #[test]
    fn complete_current_schema_round_trips_exactly() {
        let expected = vector();
        let fields = fingerprint_fields(expected).unwrap();
        let admitted = fingerprint_from_keeper_entry(&fields)
            .unwrap()
            .expect("new-format record is present");
        assert_eq!(admitted.schema_version, KEEPER_FINGERPRINT_SCHEMA_VERSION);
        assert_eq!(admitted.values, expected);
    }

    #[test]
    fn total_absence_is_legacy_not_a_zero_vector() {
        assert_eq!(
            fingerprint_from_keeper_entry(&serde_json::json!({"audio_key": "legacy"})).unwrap(),
            None
        );
    }

    #[test]
    fn partial_records_fail_closed() {
        for value in [
            serde_json::json!({"structural_fingerprint_schema": 1}),
            serde_json::json!({"structural_fingerprint_dims": STRUCT_DIMS}),
            serde_json::json!({"structural_fingerprint": vector().to_vec()}),
        ] {
            assert_eq!(
                fingerprint_from_keeper_entry(&value),
                Err(KeeperFingerprintEvidenceError::PartialRecord)
            );
        }
    }

    #[test]
    fn schema_is_not_inferred_from_dimension_count() {
        let value = serde_json::json!({
            "structural_fingerprint_schema": 2,
            "structural_fingerprint_dims": STRUCT_DIMS,
            "structural_fingerprint": vector().to_vec(),
        });
        assert_eq!(
            fingerprint_from_keeper_entry(&value),
            Err(KeeperFingerprintEvidenceError::UnsupportedSchema(2))
        );
    }

    #[test]
    fn wrong_declared_or_actual_dimensions_fail_closed() {
        let expected = vector();
        let wrong_declared = serde_json::json!({
            "structural_fingerprint_schema": 1,
            "structural_fingerprint_dims": STRUCT_DIMS + 1,
            "structural_fingerprint": expected.to_vec(),
        });
        assert_eq!(
            fingerprint_from_keeper_entry(&wrong_declared),
            Err(KeeperFingerprintEvidenceError::InvalidDimensions(
                (STRUCT_DIMS + 1) as u64
            ))
        );

        let mut short = expected.to_vec();
        short.pop();
        let wrong_length = serde_json::json!({
            "structural_fingerprint_schema": 1,
            "structural_fingerprint_dims": STRUCT_DIMS,
            "structural_fingerprint": short,
        });
        assert_eq!(
            fingerprint_from_keeper_entry(&wrong_length),
            Err(KeeperFingerprintEvidenceError::InvalidVectorLength(
                STRUCT_DIMS - 1
            ))
        );
    }

    #[test]
    fn non_numeric_values_fail_closed() {
        let mut values: Vec<Value> = vector()
            .into_iter()
            .map(|value| serde_json::json!(value))
            .collect();
        values[0] = Value::String("not-a-number".to_string());
        let value = serde_json::json!({
            "structural_fingerprint_schema": 1,
            "structural_fingerprint_dims": STRUCT_DIMS,
            "structural_fingerprint": values,
        });
        assert_eq!(
            fingerprint_from_keeper_entry(&value),
            Err(KeeperFingerprintEvidenceError::NonNumericValue(0))
        );
    }

    #[test]
    fn writer_rejects_non_finite_values_before_json_serialization() {
        let mut values = vector();
        values[3] = f64::NAN;
        assert_eq!(
            fingerprint_fields(values),
            Err(KeeperFingerprintEvidenceError::NonFiniteValue(3))
        );
    }
}