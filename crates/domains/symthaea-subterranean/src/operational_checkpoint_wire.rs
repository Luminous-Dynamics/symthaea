// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lossless strict JSON ingress for operational checkpoint restore.
//!
//! Runtime authority structs intentionally remain ordinary Serde types. Restore
//! ingress has a stronger obligation: the typed checkpoint must preserve every
//! piece of wire state supplied by the caller. Otherwise the committed restore
//! source would differ from the source that was actually reviewed/transmitted.

use super::{OperationalCheckpointError, SubterraneanOperationalCheckpoint};

/// Hard bound for externally supplied operational-checkpoint JSON.
///
/// The dominant current payload is the controller projection: 6 x 16,384 f32
/// weights. Sixteen MiB leaves substantial encoding/schema headroom while
/// preventing an untrusted restore candidate from triggering unbounded duplicate
/// JSON parses before any authority decision is reached.
pub const MAX_OPERATIONAL_CHECKPOINT_WIRE_BYTES: usize = 16 * 1024 * 1024;

/// Failure to establish an exact current-schema checkpoint from external JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationalCheckpointWireError {
    /// Input exceeded the pre-parse wire bound.
    TooLarge { found: usize, max: usize },
    /// Malformed JSON, duplicate known fields, or a top-level Serde contract
    /// violation prevented typed decoding.
    Encoding,
    /// The typed checkpoint is structurally parseable but not a valid current
    /// operational source (for example an old/future schema number).
    InvalidCheckpoint(OperationalCheckpointError),
    /// Decoding discarded or synthesized JSON state. This includes unknown
    /// nested fields and skipped host-local fields supplied on the wire.
    LossyDecode,
}

impl From<OperationalCheckpointError> for OperationalCheckpointWireError {
    fn from(value: OperationalCheckpointError) -> Self {
        Self::InvalidCheckpoint(value)
    }
}

impl SubterraneanOperationalCheckpoint {
    /// Strictly decode externally supplied v4 JSON without silently losing state.
    ///
    /// Oversized input is rejected before parsing. The first typed parse then
    /// preserves Serde's duplicate-known-field and required top-level checks.
    /// After exact-schema structural validation, the raw parsed JSON value is
    /// compared with the typed checkpoint serialized back to JSON. Any unknown
    /// nested field, skipped field, or default-synthesized state makes those
    /// values differ and is rejected as [`OperationalCheckpointWireError::LossyDecode`].
    ///
    /// Comparison is over parsed JSON values rather than raw bytes, so whitespace
    /// and object-key ordering are not part of authority identity.
    pub fn from_strict_v4_json(bytes: &[u8]) -> Result<Self, OperationalCheckpointWireError> {
        if bytes.len() > MAX_OPERATIONAL_CHECKPOINT_WIRE_BYTES {
            return Err(OperationalCheckpointWireError::TooLarge {
                found: bytes.len(),
                max: MAX_OPERATIONAL_CHECKPOINT_WIRE_BYTES,
            });
        }

        let decoded: Self =
            serde_json::from_slice(bytes).map_err(|_| OperationalCheckpointWireError::Encoding)?;
        decoded.validate_source()?;

        let supplied: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|_| OperationalCheckpointWireError::Encoding)?;
        let normalized = serde_json::to_value(&decoded)
            .map_err(|_| OperationalCheckpointWireError::Encoding)?;
        if supplied != normalized {
            return Err(OperationalCheckpointWireError::LossyDecode);
        }
        Ok(decoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment::SubterraneanEmbodiment;
    use crate::update_control::{ArtifactDigest, UpdateManager};
    use crate::OPERATIONAL_CHECKPOINT_SCHEMA_VERSION;
    use symthaea_core::genesis::GenesisSeed;

    fn checkpoint() -> SubterraneanOperationalCheckpoint {
        SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("strict-v4-wire"))
            .operational_checkpoint()
    }

    fn value() -> serde_json::Value {
        serde_json::to_value(checkpoint()).expect("serialize checkpoint")
    }

    fn strict_from_value(
        value: &serde_json::Value,
    ) -> Result<SubterraneanOperationalCheckpoint, OperationalCheckpointWireError> {
        let bytes = serde_json::to_vec(value).expect("serialize test value");
        SubterraneanOperationalCheckpoint::from_strict_v4_json(&bytes)
    }

    #[test]
    fn current_v4_serialization_strict_decodes_losslessly() {
        let source = checkpoint();
        let bytes = serde_json::to_vec(&source).expect("serialize checkpoint");
        assert!(bytes.len() < MAX_OPERATIONAL_CHECKPOINT_WIRE_BYTES);
        let decoded = SubterraneanOperationalCheckpoint::from_strict_v4_json(&bytes)
            .expect("current v4 must strict-decode");
        assert_eq!(decoded.schema_version, OPERATIONAL_CHECKPOINT_SCHEMA_VERSION);
        assert_eq!(decoded.validate_source(), Ok(()));
    }

    #[test]
    fn oversized_wire_input_is_rejected_before_json_decoding() {
        let bytes = vec![b' '; MAX_OPERATIONAL_CHECKPOINT_WIRE_BYTES + 1];
        let error = SubterraneanOperationalCheckpoint::from_strict_v4_json(&bytes)
            .expect_err("oversized input must reject");
        assert_eq!(
            error,
            OperationalCheckpointWireError::TooLarge {
                found: MAX_OPERATIONAL_CHECKPOINT_WIRE_BYTES + 1,
                max: MAX_OPERATIONAL_CHECKPOINT_WIRE_BYTES,
            }
        );
    }

    #[test]
    fn insignificant_json_whitespace_is_not_authority_identity() {
        let bytes = serde_json::to_vec_pretty(&checkpoint()).expect("pretty checkpoint");
        assert!(SubterraneanOperationalCheckpoint::from_strict_v4_json(&bytes).is_ok());
    }

    #[test]
    fn unknown_top_level_field_is_rejected() {
        let mut candidate = value();
        candidate
            .as_object_mut()
            .expect("checkpoint object")
            .insert("future_authority_domain".to_string(), serde_json::json!({}));
        let error = strict_from_value(&candidate).expect_err("unknown top-level field must reject");
        assert_eq!(error, OperationalCheckpointWireError::Encoding);
    }

    #[test]
    fn unknown_nested_domain_fields_are_rejected_losslessly() {
        for domain in [
            "controller",
            "mission",
            "operator_authority",
            "degraded_supervisor",
            "sensor_fusion",
            "actuator_isolation",
            "field_envelope",
            "partition_recovery",
            "temporal",
        ] {
            let mut candidate = value();
            candidate[domain]
                .as_object_mut()
                .unwrap_or_else(|| panic!("{domain} must serialize as an object"))
                .insert(
                    "future_authority_field".to_string(),
                    serde_json::json!({"generation": 9}),
                );
            let error = strict_from_value(&candidate)
                .expect_err("unknown nested state must reject strict ingress");
            assert_eq!(
                error,
                OperationalCheckpointWireError::LossyDecode,
                "unknown nested state in {domain} must not be silently discarded"
            );
        }
    }

    #[test]
    fn unknown_field_two_levels_deep_is_rejected() {
        let mut candidate = value();
        candidate["degraded_supervisor"]["policy"]
            .as_object_mut()
            .expect("degraded policy object")
            .insert(
                "future_recovery_limit".to_string(),
                serde_json::Value::from(7),
            );
        let error = strict_from_value(&candidate).expect_err("nested policy drift must reject");
        assert_eq!(error, OperationalCheckpointWireError::LossyDecode);
    }

    #[test]
    fn unknown_update_manager_field_is_rejected_when_manager_is_present() {
        let mut source = checkpoint();
        source.update_manager = Some(
            UpdateManager::new(ArtifactDigest([1; 32]), 1).expect("valid update manager"),
        );
        let mut candidate = serde_json::to_value(source).expect("serialize checkpoint");
        candidate["update_manager"]
            .as_object_mut()
            .expect("update manager object")
            .insert(
                "future_update_authority".to_string(),
                serde_json::Value::from(true),
            );
        let error = strict_from_value(&candidate)
            .expect_err("unknown update-manager authority state must reject");
        assert_eq!(error, OperationalCheckpointWireError::LossyDecode);
    }

    #[test]
    fn injected_skipped_host_local_recovery_state_is_rejected() {
        let mut candidate = value();
        candidate["operator_authority"]
            .as_object_mut()
            .expect("operator authority object")
            .insert("issued_recovery".to_string(), serde_json::json!({}));
        let error = strict_from_value(&candidate)
            .expect_err("wire-supplied skipped recovery state must reject");
        assert_eq!(error, OperationalCheckpointWireError::LossyDecode);
    }

    #[test]
    fn missing_required_authority_state_is_rejected_before_lossless_comparison() {
        let mut candidate = value();
        candidate
            .as_object_mut()
            .expect("checkpoint object")
            .remove("temporal");
        let error = strict_from_value(&candidate).expect_err("missing temporal state must reject");
        assert_eq!(error, OperationalCheckpointWireError::Encoding);
    }

    #[test]
    fn old_and_future_schema_numbers_are_not_strict_v4_sources() {
        for schema in [
            OPERATIONAL_CHECKPOINT_SCHEMA_VERSION - 1,
            OPERATIONAL_CHECKPOINT_SCHEMA_VERSION + 1,
        ] {
            let mut candidate = value();
            candidate["schema_version"] = serde_json::Value::from(schema);
            let error = strict_from_value(&candidate).expect_err("schema drift must reject");
            assert_eq!(
                error,
                OperationalCheckpointWireError::InvalidCheckpoint(
                    OperationalCheckpointError::UnsupportedSchema {
                        found: schema,
                        expected: OPERATIONAL_CHECKPOINT_SCHEMA_VERSION,
                    }
                )
            );
        }
    }

    #[test]
    fn duplicate_known_top_level_field_is_rejected_by_direct_typed_parse() {
        let encoded = serde_json::to_string(&checkpoint()).expect("serialize checkpoint");
        let needle = format!(
            "\"schema_version\":{}",
            OPERATIONAL_CHECKPOINT_SCHEMA_VERSION
        );
        let duplicate = format!("{needle},{needle}");
        let candidate = encoded.replacen(&needle, &duplicate, 1);
        let error = SubterraneanOperationalCheckpoint::from_strict_v4_json(candidate.as_bytes())
            .expect_err("duplicate schema_version must reject");
        assert_eq!(error, OperationalCheckpointWireError::Encoding);
    }
}
