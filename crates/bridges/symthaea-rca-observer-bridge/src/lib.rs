// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Strict construction lane for RCA-observable cognitive executions.
//!
//! Ordinary [`CognitiveLoopService::new`] remains untouched and carries no RCA
//! execution-lineage claim. This bridge provides a separate opt-in lane:
//!
//! ```text
//! externally supplied source-generation identity
//!         +
//! validated CognitiveLoopConfig
//!         +
//! canonical config projection
//!         +
//! fresh live execution-lineage issuance
//!         ↓
//! construct CognitiveLoopService
//!         ↓
//! RcaObservableCognitiveLoopV1
//! ```
//!
//! The wrapper keeps both the service and live-issued lineage capability private.
//! It exposes cognitive cycles only as [`RcaCompletedCycleV1`], which binds each
//! result to the immutable execution-lineage digest and a wrapper-owned monotonic
//! cycle index. No shadow observer is invoked in RCA-002.0b.

#![deny(unsafe_code)]

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, CycleResult};
use symthaea_execution_lineage::{
    CognitiveExecutionLineageV1, ExecutionLineageError, IssuedCognitiveExecutionLineageV1,
};
use thiserror::Error;

pub const RCA_CONFIG_PROFILE_V1: &str = "rca-cognitive-loop-config-tree-v1";

/// Normative projection contract whose digest is bound into every execution lineage.
///
/// The projection first uses `serde_json::to_value` to obtain the complete config
/// value tree, then encodes that tree without relying on JSON object insertion order
/// or textual escaping:
///
/// - null: tag 0x00
/// - false/true: tags 0x01/0x02
/// - number: tag 0x03 + u64 little-endian UTF-8 length + `serde_json::Number::to_string()`
/// - string: tag 0x04 + u64 little-endian UTF-8 length + raw UTF-8
/// - array: tag 0x05 + u64 little-endian item count + recursively encoded items
/// - object: tag 0x06 + u64 little-endian entry count + entries sorted by raw UTF-8
///   key bytes; each key is u64 little-endian length + raw UTF-8, followed by the
///   recursively encoded value.
///
/// The source-generation commitment is expected to cover the repository/lockfile
/// generation that fixes serde/serde_json behavior. A future profile change requires
/// a new profile name and contract digest.
pub const RCA_CONFIG_PROFILE_CONTRACT_V1: &str = concat!(
    "rca-cognitive-loop-config-tree-v1\n",
    "input=serde_json::to_value(CognitiveLoopConfig)\n",
    "null=00\n",
    "false=01\n",
    "true=02\n",
    "number=03|u64le(utf8_len)|serde_json_Number_to_string_utf8\n",
    "string=04|u64le(utf8_len)|raw_utf8\n",
    "array=05|u64le(item_count)|recursive_items\n",
    "object=06|u64le(entry_count)|entries_sorted_by_raw_utf8_key_bytes\n",
    "object_entry=u64le(key_utf8_len)|key_raw_utf8|recursive_value\n",
    "no_whitespace_or_json_text_escaping_layer\n",
);

const PROFILE_DIGEST_DOMAIN: &[u8] = b"symthaea:rca-config-profile-contract:v1\0";

/// One completed cycle emitted by the strict RCA-observable construction lane.
///
/// Fields are private so callers cannot manufacture a lineage/cycle pairing with
/// a struct literal. This type is deliberately non-serializable in RCA-002.0b;
/// archival projection belongs to the later shadow-observation boundary.
pub struct RcaCompletedCycleV1 {
    execution_lineage_digest: String,
    cycle_index: u64,
    result: CycleResult,
}

impl RcaCompletedCycleV1 {
    pub fn execution_lineage_digest(&self) -> &str {
        &self.execution_lineage_digest
    }

    pub const fn cycle_index(&self) -> u64 {
        self.cycle_index
    }

    pub fn result(&self) -> &CycleResult {
        &self.result
    }
}

/// Cognitive loop whose execution lineage was issued before the service could run.
///
/// There is intentionally no `Deref`, `DerefMut`, `service_mut`, `into_inner`, or
/// live-issued-lineage accessor. The ordinary cognitive service cannot escape this
/// wrapper through the RCA-002.0b API and continue producing unbound cycles.
pub struct RcaObservableCognitiveLoopV1 {
    service: CognitiveLoopService,
    issued_lineage: IssuedCognitiveExecutionLineageV1,
    completed_cycles: u64,
}

impl RcaObservableCognitiveLoopV1 {
    /// Construct an opt-in RCA-observable execution.
    ///
    /// `source_generation_digest` is identity supplied by an outer build/evidence
    /// boundary. This bridge does not self-certify that generation as qualified.
    /// Lineage issuance happens before the cognitive service is constructed, and
    /// any issuance or service-construction failure returns no wrapper.
    pub fn new_shadow_observable(
        config: CognitiveLoopConfig,
        source_generation_digest: &str,
    ) -> Result<Self, RcaConstructionError> {
        config.validate().map_err(RcaConstructionError::InvalidConfig)?;

        let config_bytes = canonical_config_projection_v1(&config)?;
        let profile_digest = config_profile_contract_digest_v1();
        let genesis = config.genesis_phrase.as_deref().map(str::as_bytes);

        let issued_lineage = CognitiveExecutionLineageV1::issue(
            source_generation_digest,
            RCA_CONFIG_PROFILE_V1,
            &profile_digest,
            &config_bytes,
            genesis,
        )?;

        let service = CognitiveLoopService::new(config)
            .map_err(|error| RcaConstructionError::ServiceConstruction(error.to_string()))?;

        Ok(Self {
            service,
            issued_lineage,
            completed_cycles: 0,
        })
    }

    /// Run exactly one cycle and bind the returned result to this execution lineage.
    pub fn cycle(&mut self, input: &str) -> RcaCompletedCycleV1 {
        let result = self.service.cycle(input);
        self.completed_cycles = self
            .completed_cycles
            .checked_add(1)
            .expect("RCA execution cycle index exhausted u64");

        RcaCompletedCycleV1 {
            execution_lineage_digest: self.issued_lineage.lineage_digest().to_string(),
            cycle_index: self.completed_cycles,
            result,
        }
    }

    /// Read-only archival identity of the currently live-issued execution.
    ///
    /// This does not expose or clone the live issuance capability itself.
    pub fn archival_lineage(&self) -> &CognitiveExecutionLineageV1 {
        self.issued_lineage.lineage()
    }

    pub fn execution_lineage_digest(&self) -> &str {
        self.issued_lineage.lineage_digest()
    }

    pub const fn completed_cycles(&self) -> u64 {
        self.completed_cycles
    }
}

#[derive(Debug, Error)]
pub enum RcaConstructionError {
    #[error("invalid cognitive-loop configuration: {0}")]
    InvalidConfig(String),
    #[error("cannot project cognitive-loop configuration: {0}")]
    ConfigProjection(#[from] serde_json::Error),
    #[error("cannot issue cognitive execution lineage: {0}")]
    Lineage(#[from] ExecutionLineageError),
    #[error("cognitive-loop construction failed after lineage issuance: {0}")]
    ServiceConstruction(String),
}

/// Digest of the exact v1 config-projection semantics.
pub fn config_profile_contract_digest_v1() -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PROFILE_DIGEST_DOMAIN);
    hasher.update(&(RCA_CONFIG_PROFILE_CONTRACT_V1.len() as u64).to_le_bytes());
    hasher.update(RCA_CONFIG_PROFILE_CONTRACT_V1.as_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn canonical_config_projection_v1(
    config: &CognitiveLoopConfig,
) -> Result<Vec<u8>, RcaConstructionError> {
    let value = serde_json::to_value(config)?;
    let mut out = Vec::new();
    encode_value_v1(&value, &mut out);
    Ok(out)
}

fn encode_value_v1(value: &serde_json::Value, out: &mut Vec<u8>) {
    match value {
        serde_json::Value::Null => out.push(0x00),
        serde_json::Value::Bool(false) => out.push(0x01),
        serde_json::Value::Bool(true) => out.push(0x02),
        serde_json::Value::Number(number) => {
            out.push(0x03);
            let text = number.to_string();
            write_bytes(out, text.as_bytes());
        }
        serde_json::Value::String(text) => {
            out.push(0x04);
            write_bytes(out, text.as_bytes());
        }
        serde_json::Value::Array(items) => {
            out.push(0x05);
            write_len(out, items.len());
            for item in items {
                encode_value_v1(item, out);
            }
        }
        serde_json::Value::Object(entries) => {
            out.push(0x06);
            write_len(out, entries.len());
            let mut keys: Vec<&str> = entries.keys().map(String::as_str).collect();
            keys.sort_unstable_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
            for key in keys {
                write_bytes(out, key.as_bytes());
                encode_value_v1(&entries[key], out);
            }
        }
    }
}

fn write_len(out: &mut Vec<u8>, len: usize) {
    let len = u64::try_from(len).expect("usize length must fit into u64 on supported targets");
    out.extend_from_slice(&len.to_le_bytes());
}

fn write_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    write_len(out, bytes.len());
    out.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn encode(value: serde_json::Value) -> Vec<u8> {
        let mut out = Vec::new();
        encode_value_v1(&value, &mut out);
        out
    }

    #[test]
    fn object_key_order_does_not_change_projection() {
        let mut left = serde_json::Map::new();
        left.insert("z".into(), serde_json::json!({"b": 2, "a": 1}));
        left.insert("a".into(), serde_json::json!([3, 2, 1]));

        let mut right = serde_json::Map::new();
        right.insert("a".into(), serde_json::json!([3, 2, 1]));
        right.insert("z".into(), serde_json::json!({"a": 1, "b": 2}));

        assert_eq!(
            encode(serde_json::Value::Object(left)),
            encode(serde_json::Value::Object(right))
        );
    }

    #[test]
    fn config_changes_change_projection() {
        let left = CognitiveLoopConfig::default();
        let mut right = left.clone();
        right.learning_threshold += 0.001;

        assert_ne!(
            canonical_config_projection_v1(&left).unwrap(),
            canonical_config_projection_v1(&right).unwrap()
        );
    }

    #[test]
    fn config_projection_is_stable_for_same_config() {
        let config = CognitiveLoopConfig::default();
        assert_eq!(
            canonical_config_projection_v1(&config).unwrap(),
            canonical_config_projection_v1(&config).unwrap()
        );
    }

    #[test]
    fn profile_contract_has_strict_blake3_identity() {
        let first = config_profile_contract_digest_v1();
        let second = config_profile_contract_digest_v1();
        assert_eq!(first, second);
        assert!(first.starts_with("blake3:"));
        assert_eq!(first.len(), "blake3:".len() + 64);
    }

    #[test]
    fn separate_wrappers_get_distinct_execution_lineages() {
        let config = CognitiveLoopConfig::default();
        let first = RcaObservableCognitiveLoopV1::new_shadow_observable(config.clone(), SOURCE)
            .expect("first observable execution");
        let second = RcaObservableCognitiveLoopV1::new_shadow_observable(config, SOURCE)
            .expect("second observable execution");

        assert_eq!(
            first.archival_lineage().config_digest(),
            second.archival_lineage().config_digest()
        );
        assert_ne!(first.execution_lineage_digest(), second.execution_lineage_digest());
    }

    #[test]
    fn completed_cycles_are_lineage_bound_and_monotonic() {
        let mut observable = RcaObservableCognitiveLoopV1::new_shadow_observable(
            CognitiveLoopConfig::default(),
            SOURCE,
        )
        .expect("observable execution");
        let lineage = observable.execution_lineage_digest().to_string();

        let first = observable.cycle("first observation-bound cycle");
        assert_eq!(first.execution_lineage_digest(), lineage);
        assert_eq!(first.cycle_index(), 1);

        let second = observable.cycle("second observation-bound cycle");
        assert_eq!(second.execution_lineage_digest(), lineage);
        assert_eq!(second.cycle_index(), 2);
        assert_eq!(observable.completed_cycles(), 2);
    }

    #[test]
    fn malformed_source_identity_fails_before_wrapper_exists() {
        let error = RcaObservableCognitiveLoopV1::new_shadow_observable(
            CognitiveLoopConfig::default(),
            "git:abc123",
        )
        .err()
        .expect("malformed source identity must fail");
        assert!(matches!(
            error,
            RcaConstructionError::Lineage(ExecutionLineageError::MalformedDigest {
                field: "source_generation_digest"
            })
        ));
    }
}
