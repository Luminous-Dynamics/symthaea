// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MATH-RET-RUNTIME-002D — content-addressed materializer implementation receipt.
//!
//! 002B enforces an implementation identity before the first source fetch, but
//! the expected implementation digest is still supplied by the production
//! caller. This layer removes that remaining caller-choice seam: it loads one
//! qualified receipt plus the exact implementation artifact bytes, verifies
//! both, retains both immutable byte sequences, and derives the 002B
//! `MaterializerBinding` from the admitted evidence.

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt::{self, Write as _};
use std::fs;
use std::path::Path;
use symthaea_math_retrieval_materializer_identity::{
    MaterializerBinding, MaterializerIdentity,
};
use symthaea_math_retrieval_runtime_seam::Sha256Digest;

const VERSION: &str = "math-retrieval-materializer-implementation-v1";
const AUTHORITY: &str = "MeasurementOnly";
const ROOT_FIELDS: [&str; 18] = [
    "version",
    "receipt_id",
    "authority",
    "implementation_kind",
    "implementation_sha256",
    "artifact_size_bytes",
    "source_object_contract_sha256",
    "source_fetch_policy_sha256",
    "payload_serialization_sha256",
    "source_bundle_sha256",
    "cargo_lock_sha256",
    "rust_toolchain",
    "target_triple",
    "build_profile",
    "feature_set",
    "rustflags_sha256",
    "build_recipe_sha256",
    "build_environment_sha256",
];
const IMPLEMENTATION_KINDS: [&str; 4] = [
    "ExecutableArtifact",
    "WasiComponent",
    "SharedLibrary",
    "StaticBuildArtifact",
];

/// Exact qualification inputs known before receipt/artifact loading.
///
/// `receipt_sha256` identifies the exact qualified receipt bytes. The
/// implementation digest is intentionally absent: it is read from those exact
/// receipt bytes and independently recomputed from the supplied artifact bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializerImplementationReceiptBinding {
    pub receipt_sha256: Sha256Digest,
    pub source_object_contract_sha256: Sha256Digest,
    pub source_fetch_policy_sha256: Sha256Digest,
    pub payload_serialization_sha256: Sha256Digest,
}

/// Immutable admitted materializer implementation evidence.
#[derive(Debug, Clone)]
pub struct LoadedMaterializerImplementation {
    receipt_bytes: Box<[u8]>,
    implementation_artifact_bytes: Box<[u8]>,
    receipt_sha256: Sha256Digest,
    receipt_id: String,
    implementation_kind: String,
    identity: MaterializerIdentity,
    source_bundle_sha256: Sha256Digest,
    cargo_lock_sha256: Sha256Digest,
    rust_toolchain: String,
    target_triple: String,
    build_profile: String,
    feature_set: Vec<String>,
    rustflags_sha256: Sha256Digest,
    build_recipe_sha256: Sha256Digest,
    build_environment_sha256: Sha256Digest,
}

impl LoadedMaterializerImplementation {
    pub fn receipt_bytes(&self) -> &[u8] {
        &self.receipt_bytes
    }

    pub fn implementation_artifact_bytes(&self) -> &[u8] {
        &self.implementation_artifact_bytes
    }

    pub fn receipt_sha256(&self) -> &Sha256Digest {
        &self.receipt_sha256
    }

    pub fn receipt_id(&self) -> &str {
        &self.receipt_id
    }

    pub fn implementation_kind(&self) -> &str {
        &self.implementation_kind
    }

    pub fn identity(&self) -> &MaterializerIdentity {
        &self.identity
    }

    pub fn materializer_binding(&self) -> MaterializerBinding {
        MaterializerBinding {
            identity: self.identity.clone(),
        }
    }

    pub fn source_bundle_sha256(&self) -> &Sha256Digest {
        &self.source_bundle_sha256
    }

    pub fn cargo_lock_sha256(&self) -> &Sha256Digest {
        &self.cargo_lock_sha256
    }

    pub fn rust_toolchain(&self) -> &str {
        &self.rust_toolchain
    }

    pub fn target_triple(&self) -> &str {
        &self.target_triple
    }

    pub fn build_profile(&self) -> &str {
        &self.build_profile
    }

    pub fn feature_set(&self) -> &[String] {
        &self.feature_set
    }

    pub fn rustflags_sha256(&self) -> &Sha256Digest {
        &self.rustflags_sha256
    }

    pub fn build_recipe_sha256(&self) -> &Sha256Digest {
        &self.build_recipe_sha256
    }

    pub fn build_environment_sha256(&self) -> &Sha256Digest {
        &self.build_environment_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaterializerReceiptError {
    Io(String),
    Json(String),
    Contract(String),
    Binding(String),
}

impl fmt::Display for MaterializerReceiptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(f, "materializer receipt I/O error: {message}"),
            Self::Json(message) => write!(f, "materializer receipt JSON error: {message}"),
            Self::Contract(message) => write!(f, "materializer receipt contract error: {message}"),
            Self::Binding(message) => write!(f, "materializer receipt binding error: {message}"),
        }
    }
}

impl std::error::Error for MaterializerReceiptError {}

pub struct MaterializerImplementationReceiptLoader;

impl MaterializerImplementationReceiptLoader {
    /// Read both paths exactly once, then operate only on the loaded bytes.
    pub fn load_files(
        receipt_path: impl AsRef<Path>,
        implementation_artifact_path: impl AsRef<Path>,
        expected: &MaterializerImplementationReceiptBinding,
    ) -> Result<LoadedMaterializerImplementation, MaterializerReceiptError> {
        let receipt_bytes = fs::read(receipt_path.as_ref())
            .map_err(|error| MaterializerReceiptError::Io(error.to_string()))?;
        let implementation_artifact_bytes = fs::read(implementation_artifact_path.as_ref())
            .map_err(|error| MaterializerReceiptError::Io(error.to_string()))?;
        Self::load_bytes(&receipt_bytes, &implementation_artifact_bytes, expected)
    }

    /// Hash, parse, validate, bind, and retain exact receipt/artifact bytes.
    pub fn load_bytes(
        receipt_bytes: &[u8],
        implementation_artifact_bytes: &[u8],
        expected: &MaterializerImplementationReceiptBinding,
    ) -> Result<LoadedMaterializerImplementation, MaterializerReceiptError> {
        if implementation_artifact_bytes.is_empty() {
            return Err(MaterializerReceiptError::Contract(
                "implementation artifact must be non-empty".into(),
            ));
        }

        let actual_receipt_sha256 = sha256_digest(receipt_bytes);
        if actual_receipt_sha256 != expected.receipt_sha256 {
            return Err(MaterializerReceiptError::Binding(format!(
                "receipt SHA-256 mismatch: expected {}, got {}",
                expected.receipt_sha256, actual_receipt_sha256
            )));
        }

        let root_value: Value = serde_json::from_slice(receipt_bytes)
            .map_err(|error| MaterializerReceiptError::Json(error.to_string()))?;
        let root = root_value.as_object().ok_or_else(|| {
            MaterializerReceiptError::Contract("root must be a JSON object".into())
        })?;
        validate_exact_root_fields(root)?;

        require_string(root, "version", VERSION)?;
        require_string(root, "authority", AUTHORITY)?;

        let receipt_id = nonempty_string(root, "receipt_id")?;
        let implementation_kind = nonempty_string(root, "implementation_kind")?;
        if !IMPLEMENTATION_KINDS.contains(&implementation_kind) {
            return Err(MaterializerReceiptError::Contract(format!(
                "implementation_kind must be one of {IMPLEMENTATION_KINDS:?}"
            )));
        }

        let implementation_sha256 = parse_digest_field(root, "implementation_sha256")?;
        let actual_implementation_sha256 = sha256_digest(implementation_artifact_bytes);
        if implementation_sha256 != actual_implementation_sha256 {
            return Err(MaterializerReceiptError::Contract(format!(
                "implementation artifact SHA-256 mismatch: receipt {}, actual {}",
                implementation_sha256, actual_implementation_sha256
            )));
        }

        let artifact_size_bytes = root
            .get("artifact_size_bytes")
            .and_then(Value::as_u64)
            .filter(|size| *size > 0)
            .ok_or_else(|| {
                MaterializerReceiptError::Contract(
                    "artifact_size_bytes must be a positive integer".into(),
                )
            })?;
        if artifact_size_bytes != implementation_artifact_bytes.len() as u64 {
            return Err(MaterializerReceiptError::Contract(format!(
                "artifact_size_bytes={} but actual artifact has {} bytes",
                artifact_size_bytes,
                implementation_artifact_bytes.len()
            )));
        }

        let source_object_contract_sha256 =
            parse_digest_field(root, "source_object_contract_sha256")?;
        let source_fetch_policy_sha256 =
            parse_digest_field(root, "source_fetch_policy_sha256")?;
        let payload_serialization_sha256 =
            parse_digest_field(root, "payload_serialization_sha256")?;

        compare_binding(
            "source_object_contract_sha256",
            &source_object_contract_sha256,
            &expected.source_object_contract_sha256,
        )?;
        compare_binding(
            "source_fetch_policy_sha256",
            &source_fetch_policy_sha256,
            &expected.source_fetch_policy_sha256,
        )?;
        compare_binding(
            "payload_serialization_sha256",
            &payload_serialization_sha256,
            &expected.payload_serialization_sha256,
        )?;

        let source_bundle_sha256 = parse_digest_field(root, "source_bundle_sha256")?;
        let cargo_lock_sha256 = parse_digest_field(root, "cargo_lock_sha256")?;
        let rustflags_sha256 = parse_digest_field(root, "rustflags_sha256")?;
        let build_recipe_sha256 = parse_digest_field(root, "build_recipe_sha256")?;
        let build_environment_sha256 = parse_digest_field(root, "build_environment_sha256")?;

        let rust_toolchain = nonempty_string(root, "rust_toolchain")?.to_owned();
        let target_triple = nonempty_string(root, "target_triple")?.to_owned();
        let build_profile = nonempty_string(root, "build_profile")?.to_owned();
        let feature_set = validate_feature_set(root)?;

        let identity = MaterializerIdentity {
            source_object_contract_sha256,
            source_fetch_policy_sha256,
            payload_serialization_sha256,
            implementation_sha256,
        };

        Ok(LoadedMaterializerImplementation {
            receipt_bytes: receipt_bytes.to_vec().into_boxed_slice(),
            implementation_artifact_bytes: implementation_artifact_bytes
                .to_vec()
                .into_boxed_slice(),
            receipt_sha256: actual_receipt_sha256,
            receipt_id: receipt_id.to_owned(),
            implementation_kind: implementation_kind.to_owned(),
            identity,
            source_bundle_sha256,
            cargo_lock_sha256,
            rust_toolchain,
            target_triple,
            build_profile,
            feature_set,
            rustflags_sha256,
            build_recipe_sha256,
            build_environment_sha256,
        })
    }
}

fn validate_exact_root_fields(root: &Map<String, Value>) -> Result<(), MaterializerReceiptError> {
    let expected: BTreeSet<&str> = ROOT_FIELDS.into_iter().collect();
    let actual: BTreeSet<&str> = root.keys().map(String::as_str).collect();
    if actual != expected {
        let missing: Vec<_> = expected.difference(&actual).copied().collect();
        let extra: Vec<_> = actual.difference(&expected).copied().collect();
        return Err(MaterializerReceiptError::Contract(format!(
            "root fields differ; missing={missing:?} extra={extra:?}"
        )));
    }
    Ok(())
}

fn require_string(
    root: &Map<String, Value>,
    field: &str,
    expected: &str,
) -> Result<(), MaterializerReceiptError> {
    match root.get(field).and_then(Value::as_str) {
        Some(actual) if actual == expected => Ok(()),
        _ => Err(MaterializerReceiptError::Contract(format!(
            "{field} must equal {expected:?}"
        ))),
    }
}

fn nonempty_string<'a>(
    root: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a str, MaterializerReceiptError> {
    let value = root.get(field).and_then(Value::as_str).ok_or_else(|| {
        MaterializerReceiptError::Contract(format!("{field} must be a non-empty string"))
    })?;
    if value.trim().is_empty() {
        return Err(MaterializerReceiptError::Contract(format!(
            "{field} must be a non-empty string"
        )));
    }
    Ok(value)
}

fn parse_digest_field(
    root: &Map<String, Value>,
    field: &str,
) -> Result<Sha256Digest, MaterializerReceiptError> {
    let text = root.get(field).and_then(Value::as_str).ok_or_else(|| {
        MaterializerReceiptError::Contract(format!(
            "{field} must be sha256:<64 lowercase hex>"
        ))
    })?;
    parse_digest_text(text, field)
}

fn parse_digest_text(
    text: &str,
    where_: &str,
) -> Result<Sha256Digest, MaterializerReceiptError> {
    let bytes = text.as_bytes();
    let valid = bytes.len() == 71
        && bytes.starts_with(b"sha256:")
        && bytes[7..]
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte));
    if !valid {
        return Err(MaterializerReceiptError::Contract(format!(
            "{where_} must be sha256:<64 lowercase hex>"
        )));
    }
    Sha256Digest::parse(text.to_owned())
        .map_err(|error| MaterializerReceiptError::Contract(format!("{where_}: {error}")))
}

fn compare_binding(
    field: &str,
    actual: &Sha256Digest,
    expected: &Sha256Digest,
) -> Result<(), MaterializerReceiptError> {
    if actual != expected {
        return Err(MaterializerReceiptError::Binding(format!(
            "{field} mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn validate_feature_set(root: &Map<String, Value>) -> Result<Vec<String>, MaterializerReceiptError> {
    let array = root
        .get("feature_set")
        .and_then(Value::as_array)
        .ok_or_else(|| MaterializerReceiptError::Contract("feature_set must be an array".into()))?;
    let mut features = Vec::with_capacity(array.len());
    for (index, value) in array.iter().enumerate() {
        let feature = value.as_str().ok_or_else(|| {
            MaterializerReceiptError::Contract(format!(
                "feature_set[{index}] must be a non-empty string"
            ))
        })?;
        if feature.trim().is_empty() {
            return Err(MaterializerReceiptError::Contract(format!(
                "feature_set[{index}] must be a non-empty string"
            )));
        }
        features.push(feature.to_owned());
    }
    for pair in features.windows(2) {
        if pair[0] >= pair[1] {
            let reason = if pair[0] == pair[1] {
                "feature_set duplicates are forbidden"
            } else {
                "feature_set must be lexicographically ascending"
            };
            return Err(MaterializerReceiptError::Contract(reason.into()));
        }
    }
    Ok(features)
}

fn sha256_digest(bytes: &[u8]) -> Sha256Digest {
    let raw = Sha256::digest(bytes);
    let mut text = String::with_capacity(71);
    text.push_str("sha256:");
    for byte in raw {
        write!(&mut text, "{byte:02x}").expect("writing SHA-256 to String cannot fail");
    }
    Sha256Digest::parse(text).expect("internally generated SHA-256 must parse")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
    const ARTIFACT: &[u8] = b"materializer-artifact-v1";

    fn d(n: u8) -> String {
        format!("sha256:{n:064x}")
    }

    fn parse(n: u8) -> Sha256Digest {
        Sha256Digest::parse(d(n)).unwrap()
    }

    fn fixture(artifact: &[u8]) -> Value {
        json!({
            "version": VERSION,
            "receipt_id": "materializer-fixture-v1",
            "authority": AUTHORITY,
            "implementation_kind": "WasiComponent",
            "implementation_sha256": sha256_digest(artifact).to_string(),
            "artifact_size_bytes": artifact.len(),
            "source_object_contract_sha256": d(7),
            "source_fetch_policy_sha256": d(8),
            "payload_serialization_sha256": d(9),
            "source_bundle_sha256": d(40),
            "cargo_lock_sha256": d(41),
            "rust_toolchain": "1.96.0",
            "target_triple": "wasm32-wasip2",
            "build_profile": "release",
            "feature_set": ["canonical-source", "production"],
            "rustflags_sha256": d(42),
            "build_recipe_sha256": d(43),
            "build_environment_sha256": d(44)
        })
    }

    fn bytes(value: &Value) -> Vec<u8> {
        serde_json::to_vec(value).unwrap()
    }

    fn binding_for(receipt_bytes: &[u8]) -> MaterializerImplementationReceiptBinding {
        MaterializerImplementationReceiptBinding {
            receipt_sha256: sha256_digest(receipt_bytes),
            source_object_contract_sha256: parse(7),
            source_fetch_policy_sha256: parse(8),
            payload_serialization_sha256: parse(9),
        }
    }

    fn assert_contract_error(value: Value, artifact: &[u8]) {
        let raw = bytes(&value);
        let expected = binding_for(&raw);
        let error = MaterializerImplementationReceiptLoader::load_bytes(
            &raw,
            artifact,
            &expected,
        )
        .unwrap_err();
        assert!(matches!(error, MaterializerReceiptError::Contract(_)));
    }

    #[test]
    fn sha256_matches_nist_abc_vector() {
        assert_eq!(
            sha256_digest(b"abc").to_string(),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn exact_receipt_and_artifact_derive_materializer_binding() {
        let value = fixture(ARTIFACT);
        let raw = bytes(&value);
        let expected = binding_for(&raw);
        let loaded = MaterializerImplementationReceiptLoader::load_bytes(
            &raw,
            ARTIFACT,
            &expected,
        )
        .unwrap();

        assert_eq!(loaded.receipt_bytes(), raw);
        assert_eq!(loaded.implementation_artifact_bytes(), ARTIFACT);
        assert_eq!(loaded.receipt_sha256(), &expected.receipt_sha256);
        assert_eq!(loaded.receipt_id(), "materializer-fixture-v1");
        assert_eq!(loaded.implementation_kind(), "WasiComponent");
        assert_eq!(loaded.identity().implementation_sha256, sha256_digest(ARTIFACT));
        assert_eq!(loaded.rust_toolchain(), "1.96.0");
        assert_eq!(loaded.target_triple(), "wasm32-wasip2");
        assert_eq!(loaded.build_profile(), "release");
        assert_eq!(loaded.feature_set(), &["canonical-source".to_owned(), "production".to_owned()]);

        let binding = loaded.materializer_binding();
        assert_eq!(binding.identity, loaded.identity().clone());
        assert_eq!(binding.identity.source_object_contract_sha256, parse(7));
        assert_eq!(binding.identity.source_fetch_policy_sha256, parse(8));
        assert_eq!(binding.identity.payload_serialization_sha256, parse(9));
    }

    #[test]
    fn receipt_byte_substitution_fails_against_qualified_receipt_digest() {
        let value = fixture(ARTIFACT);
        let raw = bytes(&value);
        let expected = binding_for(&raw);
        let mut substituted = raw.clone();
        substituted.push(b'\n');

        let error = MaterializerImplementationReceiptLoader::load_bytes(
            &substituted,
            ARTIFACT,
            &expected,
        )
        .unwrap_err();
        assert!(matches!(error, MaterializerReceiptError::Binding(_)));
    }

    #[test]
    fn implementation_artifact_mutation_fails_even_when_receipt_is_unchanged() {
        let value = fixture(ARTIFACT);
        let raw = bytes(&value);
        let expected = binding_for(&raw);
        let mut mutated = ARTIFACT.to_vec();
        mutated.push(b'!');

        let error = MaterializerImplementationReceiptLoader::load_bytes(
            &raw,
            &mutated,
            &expected,
        )
        .unwrap_err();
        assert!(matches!(error, MaterializerReceiptError::Contract(_)));
    }

    #[test]
    fn structural_receipt_attacks_fail_closed_with_rebound_receipt_hash() {
        let mut extra = fixture(ARTIFACT);
        extra["unexpected"] = json!(true);
        assert_contract_error(extra, ARTIFACT);

        let mut authority = fixture(ARTIFACT);
        authority["authority"] = json!("FormalAuthority");
        assert_contract_error(authority, ARTIFACT);

        let mut kind = fixture(ARTIFACT);
        kind["implementation_kind"] = json!("RuntimeString");
        assert_contract_error(kind, ARTIFACT);

        let mut size = fixture(ARTIFACT);
        size["artifact_size_bytes"] = json!(ARTIFACT.len() + 1);
        assert_contract_error(size, ARTIFACT);

        let mut unordered = fixture(ARTIFACT);
        unordered["feature_set"] = json!(["production", "canonical-source"]);
        assert_contract_error(unordered, ARTIFACT);

        let mut duplicate = fixture(ARTIFACT);
        duplicate["feature_set"] = json!(["canonical-source", "canonical-source"]);
        assert_contract_error(duplicate, ARTIFACT);

        let mut uppercase = fixture(ARTIFACT);
        uppercase["source_bundle_sha256"] = json!(format!("sha256:{}", "A".repeat(64)));
        assert_contract_error(uppercase, ARTIFACT);

        let mut empty = fixture(ARTIFACT);
        empty["rust_toolchain"] = json!("   ");
        assert_contract_error(empty, ARTIFACT);
    }

    #[test]
    fn policy_bindings_are_checked_independently() {
        let value = fixture(ARTIFACT);
        let raw = bytes(&value);

        let mut expected = binding_for(&raw);
        expected.source_object_contract_sha256 = parse(1);
        assert!(matches!(
            MaterializerImplementationReceiptLoader::load_bytes(&raw, ARTIFACT, &expected)
                .unwrap_err(),
            MaterializerReceiptError::Binding(_)
        ));

        let mut expected = binding_for(&raw);
        expected.source_fetch_policy_sha256 = parse(2);
        assert!(matches!(
            MaterializerImplementationReceiptLoader::load_bytes(&raw, ARTIFACT, &expected)
                .unwrap_err(),
            MaterializerReceiptError::Binding(_)
        ));

        let mut expected = binding_for(&raw);
        expected.payload_serialization_sha256 = parse(3);
        assert!(matches!(
            MaterializerImplementationReceiptLoader::load_bytes(&raw, ARTIFACT, &expected)
                .unwrap_err(),
            MaterializerReceiptError::Binding(_)
        ));
    }

    #[test]
    fn implementation_digest_is_derived_from_receipt_and_artifact_not_caller_input() {
        let value = fixture(ARTIFACT);
        let raw = bytes(&value);
        let expected = binding_for(&raw);
        let loaded = MaterializerImplementationReceiptLoader::load_bytes(
            &raw,
            ARTIFACT,
            &expected,
        )
        .unwrap();

        assert_eq!(
            loaded.materializer_binding().identity.implementation_sha256,
            sha256_digest(ARTIFACT)
        );
    }

    #[test]
    fn already_loaded_receipt_and_artifact_are_immune_to_later_path_substitution() {
        let value = fixture(ARTIFACT);
        let raw = bytes(&value);
        let expected = binding_for(&raw);

        let unique = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "symthaea-math-ret-materializer-receipt-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let receipt_path = root.join("receipt.json");
        let artifact_path = root.join("materializer.bin");
        fs::write(&receipt_path, &raw).unwrap();
        fs::write(&artifact_path, ARTIFACT).unwrap();

        let loaded = MaterializerImplementationReceiptLoader::load_files(
            &receipt_path,
            &artifact_path,
            &expected,
        )
        .unwrap();

        fs::write(&receipt_path, b"{}\n").unwrap();
        fs::write(&artifact_path, b"replacement-artifact").unwrap();

        assert_eq!(loaded.receipt_bytes(), raw);
        assert_eq!(loaded.implementation_artifact_bytes(), ARTIFACT);
        assert_eq!(loaded.identity().implementation_sha256, sha256_digest(ARTIFACT));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn malformed_or_empty_inputs_fail_closed() {
        let empty_artifact = b"";
        let value = fixture(ARTIFACT);
        let raw = bytes(&value);
        let expected = binding_for(&raw);
        assert!(matches!(
            MaterializerImplementationReceiptLoader::load_bytes(
                &raw,
                empty_artifact,
                &expected,
            )
            .unwrap_err(),
            MaterializerReceiptError::Contract(_)
        ));

        let malformed = b"not-json";
        let expected = MaterializerImplementationReceiptBinding {
            receipt_sha256: sha256_digest(malformed),
            source_object_contract_sha256: parse(7),
            source_fetch_policy_sha256: parse(8),
            payload_serialization_sha256: parse(9),
        };
        assert!(matches!(
            MaterializerImplementationReceiptLoader::load_bytes(
                malformed,
                ARTIFACT,
                &expected,
            )
            .unwrap_err(),
            MaterializerReceiptError::Json(_)
        ));
    }
}
