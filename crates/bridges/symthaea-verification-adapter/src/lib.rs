//! Durable controller-document snapshot adapter for the substrate-neutral
//! verification contract.
//!
//! This adapter reads an application-owned JSON snapshot from stable storage,
//! content-addresses the complete snapshot envelope, reconstructs the exact
//! controller-document resolution facts, and emits a normal
//! `VerificationMethodResolution` receipt.
//!
//! It deliberately does not perform network I/O, DNS resolution, signature
//! verification, controller authorization, or truth assessment. Those remain
//! separate adapter responsibilities.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
mod eddsa_jcs_2022;

pub use eddsa_jcs_2022::{verify_eddsa_jcs_2022, EDDSA_JCS_2022};

use symthaea_epistemic_types::{
    ClaimControllerDocumentIdentity, ClaimControllerIdentity, ClaimVerificationMethod,
    FederationDependency,
    ControllerDocumentDereferenceAttestation, ControllerDocumentResolutionSource,
    ControllerDocumentSnapshotScope, VerificationFailure, VerificationMethodLifecycle,
    VerificationMethodResolution, VerificationRequest,
};

pub const SNAPSHOT_FILE_SCHEMA_VERSION: u16 = 1;
const SNAPSHOT_REFERENCE_DOMAIN: &str = "symthaea:controller-document-snapshot:v1";
const MAX_SNAPSHOT_ENVELOPE_BYTES: u64 = 32 * 1024 * 1024;

/// Durable JSON representation of one exact controller-document state.
///
/// The raw controller document is preserved as bytes rather than parsed and
/// reserialized by the adapter. This makes the content digest an attestation of
/// the exact bytes that were actually consumed by the resolver.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControllerDocumentSnapshotFile {
    pub schema_version: u16,
    pub controller_document_ref: String,
    pub state_at: String,
    pub resolved_at: String,
    pub response_media_type: String,
    pub document: String,
}

impl ControllerDocumentSnapshotFile {
    pub fn new(
        controller_document_ref: impl Into<String>,
        state_at: impl Into<String>,
        resolved_at: impl Into<String>,
        response_media_type: impl Into<String>,
        document: impl Into<String>,
    ) -> Result<Self, SnapshotError> {
        let snapshot = Self {
            schema_version: SNAPSHOT_FILE_SCHEMA_VERSION,
            controller_document_ref: controller_document_ref.into(),
            state_at: state_at.into(),
            resolved_at: resolved_at.into(),
            response_media_type: response_media_type.into(),
            document: document.into(),
        };
        snapshot.validate_structure()?;
        Ok(snapshot)
    }

    pub fn validate_structure(&self) -> Result<(), SnapshotError> {
        if self.schema_version != SNAPSHOT_FILE_SCHEMA_VERSION {
            return Err(SnapshotError::Malformed(
                "unsupported controller-document snapshot schema version".into(),
            ));
        }

        for (field, value) in [
            ("controller_document_ref", self.controller_document_ref.as_str()),
            ("state_at", self.state_at.as_str()),
            ("resolved_at", self.resolved_at.as_str()),
            ("response_media_type", self.response_media_type.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(SnapshotError::Malformed(format!(
                    "{field} must be non-empty"
                )));
            }
        }

        if self.document.is_empty() {
            return Err(SnapshotError::Malformed(
                "controller-document snapshot document must be non-empty".into(),
            ));
        }

        Ok(())
    }

    /// Content-address the complete persisted evidence envelope.
    ///
    /// The reference covers the schema, controller-document URL, historical
    /// state time, resolution time, media type, and exact document bytes.
    pub fn snapshot_reference(&self) -> Result<String, SnapshotError> {
        self.validate_structure()?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SNAPSHOT_REFERENCE_DOMAIN.as_bytes());
        bytes.push(0);
        append_len_prefixed(&mut bytes, self.schema_version.to_string().as_bytes());
        append_len_prefixed(
            &mut bytes,
            self.controller_document_ref.as_bytes(),
        );
        append_len_prefixed(&mut bytes, self.state_at.as_bytes());
        append_len_prefixed(&mut bytes, self.resolved_at.as_bytes());
        append_len_prefixed(&mut bytes, self.response_media_type.as_bytes());
        append_len_prefixed(&mut bytes, self.document.as_bytes());

        let digest = Sha256::digest(&bytes);
        Ok(format!("sha256:{}", hex::encode(digest)))
    }

    pub fn to_json(&self) -> Result<String, SnapshotError> {
        self.validate_structure()?;
        serde_json::to_string_pretty(self)
            .map_err(|_| SnapshotError::Malformed("snapshot serialization failed".into()))
    }
}

/// Adapter configuration.
///
/// The expected snapshot reference is mandatory. A filesystem path alone is
/// intentionally insufficient because path identity is mutable and portable
/// copies can silently replace its contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedVerificationMethodMaterial {
    Multikey { public_key_multibase: String },
    JsonWebKey { public_key_jwk: Value },
}

impl ResolvedVerificationMethodMaterial {
    pub fn method_type(&self) -> &'static str {
        match self {
            Self::Multikey { .. } => "Multikey",
            Self::JsonWebKey { .. } => "JsonWebKey",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedVerificationMethod {
    pub method: ClaimVerificationMethod,
    pub controller: ClaimControllerIdentity,
    pub material: ResolvedVerificationMethodMaterial,
    pub material_digest: String,
    pub lifecycle: VerificationMethodLifecycle,
}

impl ResolvedVerificationMethod {
    pub fn method_type(&self) -> &'static str {
        self.material.method_type()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonControllerDocumentSnapshotAdapter {
    path: PathBuf,
    expected_snapshot_reference: String,
}

impl JsonControllerDocumentSnapshotAdapter {
    pub fn new(
        path: impl Into<PathBuf>,
        expected_snapshot_reference: impl Into<String>,
    ) -> Result<Self, SnapshotError> {
        let adapter = Self {
            path: path.into(),
            expected_snapshot_reference: expected_snapshot_reference.into(),
        };
        validate_snapshot_reference(&adapter.expected_snapshot_reference)?;
        Ok(adapter)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn expected_snapshot_reference(&self) -> &str {
        &self.expected_snapshot_reference
    }

    /// Resolve a verification request entirely from the durable snapshot file.
    ///
    /// The adapter never upgrades a snapshot to a cryptographic proof. It only
    /// supplies the exact resolution facts needed by the substrate-neutral
    /// verification contract.
    pub fn resolve(
        &self,
        request: &VerificationRequest,
    ) -> Result<VerificationMethodResolution, SnapshotError> {
        request.validate_structure()?;

        let mut file = match fs::File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(SnapshotError::Unresolved(vec![
                    FederationDependency::ControllerDocument(
                        request.controller_document_ref()?,
                    ),
                ]));
            }
            Err(error) => {
                return Err(SnapshotError::Io {
                    path: self.path.clone(),
                    message: error.to_string(),
                });
            }
        };
        let mut bytes = Vec::new();
        file.take(MAX_SNAPSHOT_ENVELOPE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| SnapshotError::Io {
                path: self.path.clone(),
                message: error.to_string(),
            })?;
        if bytes.len() as u64 > MAX_SNAPSHOT_ENVELOPE_BYTES {
            return Err(SnapshotError::Malformed(
                "controller-document snapshot envelope exceeds the adapter safety limit".into(),
            ));
        }

        let snapshot: ControllerDocumentSnapshotFile =
            serde_json::from_slice(&bytes).map_err(|error| {
                SnapshotError::Malformed(format!(
                    "snapshot file is not valid JSON in the expected envelope: {error}"
                ))
            })?;
        self.resolve_snapshot(request, snapshot)
    }

    /// Resolve an already-loaded snapshot. This makes the exact same adapter
    /// semantics testable for historical registries, object stores, or database
    /// backends without coupling this contract to a filesystem API.
    pub fn resolve_snapshot(
        &self,
        request: &VerificationRequest,
        snapshot: ControllerDocumentSnapshotFile,
    ) -> Result<VerificationMethodResolution, SnapshotError> {
        self.resolve_snapshot_with_material(request, snapshot)
            .map(|(resolution, _)| resolution)
    }

    /// Resolve a durable snapshot and return the exact public verification
    /// material used to compute the material identity bound into the resolution.
    ///
    /// The returned material is public by definition; private/secret key
    /// parameters are rejected before this value is produced.
    pub fn resolve_snapshot_with_material(
        &self,
        request: &VerificationRequest,
        snapshot: ControllerDocumentSnapshotFile,
    ) -> Result<
        (
            VerificationMethodResolution,
            ResolvedVerificationMethod,
        ),
        SnapshotError,
    > {
        request.validate_structure()?;
        snapshot.validate_structure()?;

        if snapshot.document.as_bytes().len() as u64
            > request.controller_document_network_policy.max_response_bytes
        {
            return Err(SnapshotError::Verification(
                VerificationFailure::ControllerDocumentResponseTooLarge,
            ));
        }

        validate_json_media_type(&snapshot.response_media_type)?;

        let expected_document_ref = request.controller_document_ref()?;
        if snapshot.controller_document_ref != expected_document_ref {
            return Err(SnapshotError::Verification(
                VerificationFailure::ControllerDocumentMismatch {
                    expected: expected_document_ref,
                    actual: snapshot.controller_document_ref,
                },
            ));
        }

        let actual_snapshot_reference = snapshot.snapshot_reference()?;
        if actual_snapshot_reference != self.expected_snapshot_reference {
            return Err(SnapshotError::SnapshotReferenceMismatch {
                expected: self.expected_snapshot_reference.clone(),
                actual: actual_snapshot_reference,
            });
        }

        let document_bytes = snapshot.document.as_bytes();
        let actual_document_digest = sha256_hex(document_bytes);

        let document = serde_json::from_slice::<Value>(document_bytes).map_err(|error| {
            SnapshotError::Malformed(format!(
                "controller document is not valid JSON: {error}"
            ))
        })?;

        let object = document.as_object().ok_or_else(|| {
            SnapshotError::Malformed("controller document root must be a JSON object".into())
        })?;

        let document_id = required_string(object, "id")?;
        if document_id != snapshot.controller_document_ref {
            return Err(SnapshotError::Verification(
                VerificationFailure::ControllerDocumentMismatch {
                    expected: snapshot.controller_document_ref.clone(),
                    actual: document_id.to_owned(),
                },
            ));
        }

        let resolved_method = extract_verification_method(request, object)?;
        let resolved_method_controller = resolved_method.controller.clone();
        let verification_method_type = resolved_method.method_type().to_owned();
        let verification_method_material_digest =
            resolved_method.material_digest.clone();
        let lifecycle = resolved_method.lifecycle.clone();
        let relationship_methods =
            extract_relationship_methods(request, object, &snapshot.controller_document_ref)?;

        let scope = ControllerDocumentSnapshotScope::historical_at(
            snapshot.state_at.clone(),
            actual_snapshot_reference.clone(),
            snapshot.resolved_at.clone(),
        )?;

        let mut resolution = VerificationMethodResolution::from_controller_document(
            request,
            snapshot.controller_document_ref.clone(),
            ClaimControllerDocumentIdentity::new(snapshot.controller_document_ref.clone())?,
            request.verification_method.clone(),
            verification_method_type,
            verification_method_material_digest,
            resolved_method_controller,
            &relationship_methods,
            actual_document_digest.clone(),
            lifecycle,
            scope,
        )?;

        let digest_multibase = Some(sha256_multibase(&actual_document_digest)?);
        let dereference = ControllerDocumentDereferenceAttestation::from_adapter(
            request,
            snapshot.controller_document_ref.clone(),
            snapshot.response_media_type.clone(),
            document_bytes.len() as u64,
            0,
            snapshot.resolved_at.clone(),
            ControllerDocumentResolutionSource::ApplicationSnapshot,
            actual_document_digest,
            digest_multibase,
        )?;

        resolution = resolution.with_controller_document_dereference(dereference, request)?;
        resolution.validate_structure()?;
        Ok((resolution, resolved_method))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotError {
    Io { path: PathBuf, message: String },
    Unresolved(Vec<FederationDependency>),
    Malformed(String),
    SnapshotReferenceMismatch { expected: String, actual: String },
    Verification(VerificationFailure),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, message } => {
                write!(formatter, "snapshot I/O failed for {}: {message}", path.display())
            }
            Self::Unresolved(dependencies) => {
                write!(formatter, "durable snapshot dependencies unresolved: {dependencies:?}")
            }
            Self::Malformed(message) => formatter.write_str(message),
            Self::SnapshotReferenceMismatch { expected, actual } => write!(
                formatter,
                "snapshot reference mismatch: expected {expected}, observed {actual}"
            ),
            Self::Verification(error) => write!(formatter, "verification contract rejected snapshot: {error:?}"),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl From<VerificationFailure> for SnapshotError {
    fn from(error: VerificationFailure) -> Self {
        Self::Verification(error)
    }
}

fn append_len_prefixed(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value);
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn sha256_multibase(hex_digest: &str) -> Result<String, SnapshotError> {
    let digest = hex::decode(hex_digest).map_err(|_| {
        SnapshotError::Malformed(
            "internal SHA-256 digest could not be decoded for Multibase encoding".into(),
        )
    })?;
    if digest.len() != 32 {
        return Err(SnapshotError::Malformed(
            "internal SHA-256 digest must contain exactly 32 bytes".into(),
        ));
    }

    let mut multihash = Vec::with_capacity(34);
    multihash.extend_from_slice(&[0x12, 0x20]);
    multihash.extend_from_slice(&digest);
    Ok(format!("z{}", bs58::encode(multihash).into_string()))
}

pub(crate) fn verification_method_material_digest(
    method_id: &ClaimVerificationMethod,
    method_type: &str,
    object: &serde_json::Map<String, Value>,
) -> Result<String, SnapshotError> {
    let material_bytes = match method_type {
        "Multikey" => {
            let public = object
                .get("publicKeyMultibase")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| {
                    SnapshotError::Malformed(
                        "Multikey verification methods require publicKeyMultibase".into(),
                    )
                })?;

            if object.contains_key("publicKeyJwk") {
                return Err(SnapshotError::Malformed(
                    "Multikey verification methods must not use publicKeyJwk".into(),
                ));
            }

            validate_multikey_encoding(public)?;

            serde_json::to_vec(&("Multikey", public))
                .map_err(|_| SnapshotError::Malformed("verification method material serialization failed".into()))?
        }
        "JsonWebKey" => {
            let public = object.get("publicKeyJwk").ok_or_else(|| {
                SnapshotError::Malformed(
                    "JsonWebKey verification methods require publicKeyJwk".into(),
                )
            })?;

            let public_object = public.as_object().ok_or_else(|| {
                SnapshotError::Malformed(
                    "verification method publicKeyJwk must be a JSON object".into(),
                )
            })?;

            if object.contains_key("publicKeyMultibase") {
                return Err(SnapshotError::Malformed(
                    "JsonWebKey verification methods must not use publicKeyMultibase".into(),
                ));
            }

            for private_member in ["d", "p", "q", "dp", "dq", "qi", "oth"] {
                if public_object.get(private_member).is_some() {
                    return Err(SnapshotError::Malformed(
                        "public JWK verification material must not contain private key parameters"
                            .into(),
                    ));
                }
            }
            let kty = public_object.get("kty").and_then(Value::as_str).ok_or_else(|| {
                SnapshotError::Malformed(
                    "public JWK verification material requires a string kty".into(),
                )
            })?;
            if kty == "oct" {
                return Err(SnapshotError::Malformed(
                    "JsonWebKey verification material must not use a symmetric oct key".into(),
                ));
            }

            let canonical_jwk: std::collections::BTreeMap<String, Value> = public_object
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();

            serde_json::to_vec(&("JsonWebKey", canonical_jwk))
                .map_err(|_| SnapshotError::Malformed("verification method material serialization failed".into()))?
        }
        other => {
            return Err(SnapshotError::Malformed(format!(
                "unsupported verification method type: {other}"
            )));
        }
    };

    let encoded = (
        "symthaea:verification-method-material:v1",
        method_id.as_str(),
        method_type,
        material_bytes,
    );
    let bytes = serde_json::to_vec(&encoded)
        .map_err(|_| SnapshotError::Malformed("verification method material serialization failed".into()))?;
    Ok(sha256_hex(&bytes))
}

fn validate_multikey_encoding(value: &str) -> Result<(), SnapshotError> {
    let encoded = value.strip_prefix('z').ok_or_else(|| {
        SnapshotError::Malformed(
            "Multikey publicKeyMultibase must use the base58-btc multibase prefix".into(),
        )
    })?;
    let bytes = bs58::decode(encoded).into_vec().map_err(|_| {
        SnapshotError::Malformed(
            "Multikey publicKeyMultibase must contain valid base58-btc data".into(),
        )
    })?;
    if bytes.len() < 2 {
        return Err(SnapshotError::Malformed(
            "Multikey publicKeyMultibase must contain a multicodec header".into(),
        ));
    }
    Ok(())
}

fn validate_snapshot_reference(value: &str) -> Result<(), SnapshotError> {
    let digest = value.strip_prefix("sha256:").ok_or_else(|| {
        SnapshotError::Malformed(
            "snapshot reference must use the sha256:<hex-digest> format".into(),
        )
    })?;
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(SnapshotError::Malformed(
            "snapshot reference must contain a 64-character hexadecimal SHA-256 digest".into(),
        ));
    }
    Ok(())
}

fn validate_json_media_type(value: &str) -> Result<(), SnapshotError> {
    let essence = value
        .split(';')
        .next()
        .map(str::trim)
        .unwrap_or_default()
        .to_ascii_lowercase();

    match essence.as_str() {
        "application/cid" | "application/json" | "application/ld+json" => Ok(()),
        _ => Err(SnapshotError::Malformed(format!(
            "unsupported controller-document snapshot media type: {value}"
        ))),
    }
}

fn required_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
) -> Result<&'a str, SnapshotError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            SnapshotError::Malformed(format!(
                "controller document field {field} must be a non-empty string"
            ))
        })
}

fn extract_verification_method(
    request: &VerificationRequest,
    document: &serde_json::Map<String, Value>,
) -> Result<ResolvedVerificationMethod, SnapshotError> {
    let requested_id = request.verification_method.as_str();
    let mut candidates = Vec::new();

    if let Some(methods) = document
        .get("verificationMethod")
        .and_then(Value::as_array)
    {
        let mut all_ids = std::collections::HashSet::new();
        for method in methods {
            let object = method.as_object().ok_or_else(|| {
                SnapshotError::Malformed(
                    "controller document verificationMethod entries must be objects".into(),
                )
            })?;
            let id = required_string(object, "id")?.to_owned();
            if !all_ids.insert(id.clone()) {
                return Err(SnapshotError::Malformed(
                    "controller document verificationMethod identifiers must be unique".into(),
                ));
            }
            if id == requested_id {
                candidates.push(object);
            }
        }
    }

    if let Some(entries) = document
        .get(request.expected_verification_relationship.as_str())
        .and_then(Value::as_array)
    {
        for entry in entries {
            if let Some(object) = entry.as_object() {
                let id = required_string(object, "id")?;
                if id == requested_id {
                    candidates.push(object);
                }
            }
        }
    }

    if candidates.is_empty() {
        return Err(SnapshotError::Verification(
            VerificationFailure::VerificationMethodNotFound,
        ));
    }

    let mut resolved = Vec::new();
    for object in candidates {
        resolved.push(parse_verification_method_definition(
            request.verification_method.clone(),
            object,
        )?);
    }

    let first = resolved
        .first()
        .cloned()
        .expect("non-empty candidate set");

    if resolved.iter().any(|candidate| candidate != &first) {
        return Err(SnapshotError::Malformed(
            "verification method has conflicting embedded definitions".into(),
        ));
    }

    Ok(first)
}

fn parse_verification_method_definition(
    expected_method: ClaimVerificationMethod,
    object: &serde_json::Map<String, Value>,
) -> Result<ResolvedVerificationMethod, SnapshotError> {
    let id = required_string(object, "id")?.to_owned();
    let method_id = ClaimVerificationMethod::new(id.clone())
        .map_err(|error| SnapshotError::Malformed(error.to_owned()))?;
    if method_id != expected_method {
        return Err(SnapshotError::Verification(
            VerificationFailure::VerificationMethodMismatch {
                expected: expected_method,
                actual: method_id,
            },
        ));
    }

    let method_type = required_string(object, "type")?;
    let material_digest = verification_method_material_digest(
        &method_id,
        method_type,
        object,
    )?;

    let material = match method_type {
        "Multikey" => ResolvedVerificationMethodMaterial::Multikey {
            public_key_multibase: required_string(object, "publicKeyMultibase")?.to_owned(),
        },
        "JsonWebKey" => ResolvedVerificationMethodMaterial::JsonWebKey {
            public_key_jwk: object
                .get("publicKeyJwk")
                .cloned()
                .ok_or_else(|| {
                    SnapshotError::Malformed(
                        "JsonWebKey verification methods require publicKeyJwk".into(),
                    )
                })?,
        },
        other => {
            return Err(SnapshotError::Malformed(format!(
                "unsupported verification method type: {other}"
            )));
        }
    };

    let controller = required_string(object, "controller")?;
    let controller = ClaimControllerIdentity::new(controller.to_owned())
        .map_err(|error| SnapshotError::Malformed(error.to_owned()))?;
    let expires = optional_timestamp(object, "expires")?;
    let revoked = optional_timestamp(object, "revoked")?;
    let lifecycle = VerificationMethodLifecycle::new(expires.as_deref(), revoked.as_deref())?;

    Ok(ResolvedVerificationMethod {
        method: method_id,
        controller,
        material,
        material_digest,
        lifecycle,
    })
}

fn extract_relationship_methods(
    request: &VerificationRequest,
    document: &serde_json::Map<String, Value>,
    document_ref: &str,
) -> Result<Vec<ClaimVerificationMethod>, SnapshotError> {
    let relationship_name = request.expected_verification_relationship.as_str();
    let relationship = document.get(relationship_name).ok_or_else(|| {
        SnapshotError::Malformed(format!(
            "controller document is missing requested verification relationship {relationship_name}"
        ))
    })?;

    let entries = relationship.as_array().ok_or_else(|| {
        SnapshotError::Malformed(format!(
            "controller document verification relationship {relationship_name} must be an array"
        ))
    })?;

    let base = url::Url::parse(document_ref)
        .map_err(|_| SnapshotError::Malformed("controller document id must be a valid URL".into()))?;

    let mut methods = Vec::with_capacity(entries.len());
    let mut seen = std::collections::HashSet::new();

    for entry in entries {
        let (id, relationship_controller, relationship_expires, relationship_revoked) =
            match entry {
                Value::String(value) => (value.clone(), None, None, None),
                Value::Object(object) => (
                    required_string(object, "id")?.to_owned(),
                    optional_string(object, "controller")?,
                    optional_timestamp(object, "expires")?,
                    optional_timestamp(object, "revoked")?,
                ),
                _ => {
                    return Err(SnapshotError::Malformed(
                        "verification relationship entries must be strings or objects".into(),
                    ))
                }
            };

        let absolute = base
            .join(&id)
            .map_err(|_| SnapshotError::Malformed("verification relationship member id is not a valid URL".into()))?;
        let method = ClaimVerificationMethod::new(absolute.to_string())
            .map_err(|error| SnapshotError::Malformed(error.to_owned()))?;
        if !seen.insert(method.clone()) {
            return Err(SnapshotError::Malformed(
                "verification relationship members must be unique".into(),
            ));
        }

        if method == request.verification_method {
            if let Some(controller) = relationship_controller {
                let expected = document_ref;
                if controller != expected {
                    return Err(SnapshotError::Verification(
                        VerificationFailure::ControllerMismatch {
                            expected: ClaimControllerIdentity::new(expected.to_owned())
                                .map_err(|error| SnapshotError::Malformed(error.to_owned()))?,
                            actual: ClaimControllerIdentity::new(controller)
                                .map_err(|error| SnapshotError::Malformed(error.to_owned()))?,
                        },
                    ));
                }
            }
            if relationship_expires.is_some() || relationship_revoked.is_some() {
                let resolved_method = extract_verification_method(request, document)?;
                let method_controller = resolved_method.controller;
                let lifecycle = resolved_method.lifecycle;
                if let Some(expires) = relationship_expires {
                    if lifecycle.expires.as_deref() != Some(expires.as_str()) {
                        return Err(SnapshotError::Malformed(
                            "relationship member expiry conflicts with verificationMethod lifecycle".into(),
                        ));
                    }
                }
                if let Some(revoked) = relationship_revoked {
                    if lifecycle.revoked.as_deref() != Some(revoked.as_str()) {
                        return Err(SnapshotError::Malformed(
                            "relationship member revocation conflicts with verificationMethod lifecycle".into(),
                        ));
                    }
                }
                let _ = method_controller;
            }
        }

        methods.push(method);
    }

    Ok(methods)
}

fn optional_string(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<String>, SnapshotError> {
    match object.get(field) {
        None => Ok(None),
        Some(value) => value
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .map(str::to_owned)
            .ok_or_else(|| {
                SnapshotError::Malformed(format!(
                    "controller document field {field} must be a non-empty string when present"
                ))
            })
            .map(Some),
    }
}

fn optional_timestamp(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<String>, SnapshotError> {
    let value = optional_string(object, field)?;
    if let Some(value) = &value {
        chrono::DateTime::parse_from_rfc3339(value).map_err(|_| {
            SnapshotError::Malformed(format!(
                "controller document field {field} must be RFC3339 when present"
            ))
        })?;
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use symthaea_epistemic_types::{
        CanonicalAdmissionReceipt, ClaimAuthorIdentity, ClaimProofPurpose,
        FederatedClaim, ProvenanceRelation, ProvenanceRelationKind,
        ProvenanceValidationReport, ProvenanceView, VerificationRequest,
    };

    fn request() -> VerificationRequest {
        let relation = ProvenanceRelation {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let validation = ProvenanceValidationReport::from_relations(std::slice::from_ref(&relation));
        let view = ProvenanceView::from_relations(
            std::slice::from_ref(&relation),
            validation.clone(),
        )
        .unwrap();
        let receipt = CanonicalAdmissionReceipt::new(
            "admission:event-verification",
            Some("frontier:verification".into()),
            "canonical:verification",
            Some("family:verification".into()),
            view.snapshot_digest.clone(),
            view.validation.validator_version.clone(),
            view.validation.snapshot_schema_version,
        )
        .unwrap();
        let claim = FederatedClaim::new(
            "claim:verification",
            "canonical:verification",
            "family:verification",
            "author:verification",
            "statement:verification",
            view,
            receipt,
        )
        .unwrap()
        .with_authorship(
            symthaea_epistemic_types::ClaimAuthorship::new(
                ClaimAuthorIdentity::new("author:verification").unwrap(),
                ClaimProofPurpose::new("assertionMethod").unwrap(),
                Some(ClaimVerificationMethod::new(
                    "https://example.test/controller#key-1",
                ).unwrap()),
            )
            .unwrap()
            .with_verification_controller(
                ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            )
            .unwrap(),
        )
        .unwrap();

        VerificationRequest::from_claim(
            &claim,
            ClaimProofPurpose::new("assertionMethod").unwrap(),
            ClaimControllerIdentity::new("https://example.test/controller").unwrap(),
            symthaea_epistemic_types::ClaimVerificationRelationship::new("assertionMethod")
                .unwrap(),
            symthaea_epistemic_types::VerificationFreshnessContext {
                proof_created: Some("2026-10-05T00:00:00Z".into()),
                proof_expires: Some("2026-10-05T03:00:00Z".into()),
                proof_domain: Some("example.test".into()),
                proof_challenge: Some("challenge-1".into()),
                verification_time: "2026-10-05T02:00:00Z".into(),
                expected_domain: Some("example.test".into()),
                expected_challenge: Some("challenge-1".into()),
            },
        )
        .unwrap()
    }

    fn snapshot() -> ControllerDocumentSnapshotFile {
        ControllerDocumentSnapshotFile::new(
            "https://example.test/controller",
            "2026-10-05T00:00:00Z",
            "2026-10-05T02:00:00Z",
            "application/ld+json",
            r##"{
                "@context": "https://www.w3.org/ns/cid/v1",
                "id": "https://example.test/controller",
                "verificationMethod": [{
                    "id": "https://example.test/controller#key-1",
                    "type": "Multikey",
                    "controller": "https://example.test/controller",
                    "expires": "2026-10-06T00:00:00Z",
                    "publicKeyMultibase": "z6MkfFakeKeyMaterial"
                }],
                "assertionMethod": ["#key-1"]
            }"##,
        )
        .unwrap()
    }

    #[test]
    fn snapshot_reference_is_typed_as_a_sha256_anchor() {
        let snapshot = snapshot();
        let reference = snapshot.snapshot_reference().unwrap();
        assert!(JsonControllerDocumentSnapshotAdapter::new(
            "/tmp/does-not-matter",
            reference
        )
        .is_ok());
        assert!(matches!(
            JsonControllerDocumentSnapshotAdapter::new(
                "/tmp/does-not-matter",
                "not-content-addressed",
            ),
            Err(SnapshotError::Malformed(message))
                if message.contains("snapshot reference must use the sha256")
        ));
    }

    #[test]
    fn resolution_material_handoff_is_exactly_receipt_bound() {
        let request = request();
        let snapshot = snapshot();
        let reference = snapshot.snapshot_reference().unwrap();
        let adapter =
            JsonControllerDocumentSnapshotAdapter::new("/tmp/does-not-matter", reference).unwrap();

        let (resolution, material) = adapter
            .resolve_snapshot_with_material(&request, snapshot)
            .unwrap();

        assert_eq!(material.method, resolution.verification_method);
        assert_eq!(material.controller, resolution.resolved_verification_method_controller);
        assert_eq!(material.method_type(), resolution.verification_method_type);
        assert_eq!(
            material.material_digest,
            resolution.verification_method_material_digest
        );

        assert!(matches!(
            material.material,
            ResolvedVerificationMethodMaterial::Multikey { .. }
        ));
    }

    #[test]
    fn durable_snapshot_emits_historical_application_receipt() {
        let request = request();
        let snapshot = snapshot();
        let reference = snapshot.snapshot_reference().unwrap();
        let adapter = JsonControllerDocumentSnapshotAdapter::new(
            "/tmp/does-not-matter",
            reference.clone(),
        )
        .unwrap();

        let resolution = adapter.resolve_snapshot(&request, snapshot).unwrap();

        assert!(resolution.validate_structure().is_ok());
        assert_eq!(
            resolution.controller_document_ref,
            "https://example.test/controller"
        );
        assert_eq!(
            resolution.resolved_verification_method_controller.as_str(),
            "https://example.test/controller"
        );
        assert!(resolution
            .relationship_methods
            .iter()
            .any(|method| method.as_str() == "https://example.test/controller#key-1"));
        assert!(matches!(
            resolution.controller_document_snapshot_scope,
            ControllerDocumentSnapshotScope::HistoricalAt {
                snapshot_reference,
                ..
            } if snapshot_reference == reference
        ));
        let dereference = resolution.controller_document_dereference.as_ref().unwrap();
        assert_eq!(dereference.source, ControllerDocumentResolutionSource::ApplicationSnapshot);
        assert!(dereference.digest_multibase.is_some());
    }

    #[test]
    fn tampering_snapshot_bytes_fails_content_address_before_resolution() {
        let request = request();
        let mut snapshot = snapshot();
        let reference = snapshot.snapshot_reference().unwrap();
        snapshot.document = snapshot.document.replace("Multikey", "Ed25519VerificationKey2020");

        let adapter =
            JsonControllerDocumentSnapshotAdapter::new("/tmp/does-not-matter", reference).unwrap();
        assert!(matches!(
            adapter.resolve_snapshot(&request, snapshot),
            Err(SnapshotError::SnapshotReferenceMismatch { .. })
        ));
    }

    #[test]
    fn jwk_material_digest_is_independent_of_json_member_order() {
        let first: Value = serde_json::from_str(
            r##"{"kty":"OKP","crv":"Ed25519","x":"VCpo2LMLhn6iWku8MKvSLg2ZAoC-nlOyPVQaO3FxVeQ","kid":"one"}"##
        )
        .unwrap();
        let second: Value = serde_json::from_str(
            r##"{"kid":"one","x":"VCpo2LMLhn6iWku8MKvSLg2ZAoC-nlOyPVQaO3FxVeQ","kty":"OKP","crv":"Ed25519"}"##
        )
        .unwrap();

        let first_object = serde_json::json!({
            "id": "https://example.test/controller#jwk-1",
            "type": "JsonWebKey",
            "controller": "https://example.test/controller",
            "publicKeyJwk": first
        });
        let second_object = serde_json::json!({
            "id": "https://example.test/controller#jwk-1",
            "type": "JsonWebKey",
            "controller": "https://example.test/controller",
            "publicKeyJwk": second
        });

        let a = verification_method_material_digest(
            &ClaimVerificationMethod::new(
                "https://example.test/controller#jwk-1"
            )
            .unwrap(),
            "JsonWebKey",
            first_object.as_object().unwrap(),
        )
        .unwrap();
        let b = verification_method_material_digest(
            &ClaimVerificationMethod::new(
                "https://example.test/controller#jwk-1"
            )
            .unwrap(),
            "JsonWebKey",
            second_object.as_object().unwrap(),
        )
        .unwrap();

        assert_eq!(a, b);
    }

    #[test]
    fn rejects_malformed_multikey_encoding_at_resolution() {
        let request = request();
        let mut snapshot = snapshot();
        snapshot.document = snapshot.document.replace(
            r##""publicKeyMultibase": "z6MkFakeKeyMaterial""##,
            r##""publicKeyMultibase": "not-base58-multikey""##,
        );
        let reference = snapshot.snapshot_reference().unwrap();
        let adapter =
            JsonControllerDocumentSnapshotAdapter::new("/tmp/does-not-matter", reference).unwrap();

        assert!(matches!(
            adapter.resolve_snapshot(&request, snapshot),
            Err(SnapshotError::Malformed(message))
                if message.contains("base58-btc")
        ));
    }

    #[test]
    fn rejects_symmetric_jwk_as_public_verification_material() {
        let request = request();
        let mut snapshot = snapshot();
        snapshot.document = r##"{
            "id": "https://example.test/controller",
            "verificationMethod": [{
                "id": "https://example.test/controller#jwk-1",
                "type": "JsonWebKey",
                "controller": "https://example.test/controller",
                "publicKeyJwk": {
                    "kty": "oct",
                    "k": "ZmFrZS1zZWNyZXQ"
                }
            }],
            "assertionMethod": ["https://example.test/controller#jwk-1"]
        }"##.into();
        let reference = snapshot.snapshot_reference().unwrap();
        let adapter =
            JsonControllerDocumentSnapshotAdapter::new("/tmp/does-not-matter", reference).unwrap();

        assert!(matches!(
            adapter.resolve_snapshot(&request, snapshot),
            Err(SnapshotError::Malformed(message))
                if message.contains("symmetric oct key")
        ));
    }

    #[test]
    fn verification_method_definition_requires_typed_public_material_shape() {
        let request = request();
        let mut snapshot = snapshot();
        snapshot.document = snapshot.document.replace(
            r##""type": "Multikey","##,
            r##""controller": "https://example.test/controller","##,
        );
        let reference = snapshot.snapshot_reference().unwrap();
        let adapter =
            JsonControllerDocumentSnapshotAdapter::new("/tmp/does-not-matter", reference).unwrap();

        assert!(matches!(
            adapter.resolve_snapshot(&request, snapshot),
            Err(SnapshotError::Malformed(message)) if message.contains("field type")
        ));

        let request = request();
        let mut snapshot = snapshot();
        snapshot.document = snapshot.document.replace(
            r##""publicKeyMultibase": "z6MkfFakeKeyMaterial""##,
            r##""publicKeyMultibase": "z6MkfFakeKeyMaterial", "publicKeyJwk": {}"##,
        );
        let reference = snapshot.snapshot_reference().unwrap();
        let adapter =
            JsonControllerDocumentSnapshotAdapter::new("/tmp/does-not-matter", reference).unwrap();
        assert!(matches!(
            adapter.resolve_snapshot(&request, snapshot),
            Err(SnapshotError::Malformed(message))
                if message.contains("multiple public verification material formats")
        ));
    }

    #[test]
    fn embedded_relationship_method_can_supply_the_definition() {
        let request = request();
        let mut snapshot = snapshot();
        snapshot.document = r##"{
            "id": "https://example.test/controller",
            "assertionMethod": [{
                "id": "https://example.test/controller#key-1",
                "type": "Multikey",
                "controller": "https://example.test/controller",
                "publicKeyMultibase": "z6MkfFakeKeyMaterial"
            }]
        }"##.into();

        let reference = snapshot.snapshot_reference().unwrap();
        let adapter =
            JsonControllerDocumentSnapshotAdapter::new("/tmp/does-not-matter", reference).unwrap();
        let resolution = adapter.resolve_snapshot(&request, snapshot).unwrap();

        assert_eq!(resolution.verification_method_type, "Multikey");
        assert_eq!(
            resolution.resolved_verification_method_controller.as_str(),
            "https://example.test/controller"
        );
        assert!(resolution
            .verification_method_material_digest
            .len()
            == 64);
    }

    #[test]
    fn conflicting_embedded_and_top_level_method_definitions_fail_closed() {
        let request = request();
        let mut snapshot = snapshot();
        snapshot.document = r##"{
            "id": "https://example.test/controller",
            "verificationMethod": [{
                "id": "https://example.test/controller#key-1",
                "type": "Multikey",
                "controller": "https://example.test/controller",
                "publicKeyMultibase": "z6MkfFakeKeyMaterial"
            }],
            "assertionMethod": [{
                "id": "https://example.test/controller#key-1",
                "type": "Multikey",
                "controller": "https://example.test/controller",
                "publicKeyMultibase": "z6MkDifferentKeyMaterial"
            }]
        }"##.into();

        let reference = snapshot.snapshot_reference().unwrap();
        let adapter =
            JsonControllerDocumentSnapshotAdapter::new("/tmp/does-not-matter", reference).unwrap();

        assert!(matches!(
            adapter.resolve_snapshot(&request, snapshot),
            Err(SnapshotError::Malformed(message))
                if message.contains("conflicting embedded definitions")
        ));
    }

    #[test]
    fn relationship_controller_substitution_is_definitively_rejected() {
        let request = request();
        let mut snapshot = snapshot();
        snapshot.document = snapshot
            .document
            .replace(r##""assertionMethod": ["#key-1"]"##, r##""assertionMethod": [{"id": "#key-1", "controller": "https://evil.example"}]"##);
        let reference = snapshot.snapshot_reference().unwrap();
        let adapter =
            JsonControllerDocumentSnapshotAdapter::new("/tmp/does-not-matter", reference).unwrap();

        assert!(matches!(
            adapter.resolve_snapshot(&request, snapshot),
            Err(SnapshotError::Verification(VerificationFailure::ControllerMismatch { .. }))
        ));
    }

    #[test]
    fn historical_state_mismatch_is_not_coerced_into_current_state() {
        let request = request();
        let mut snapshot = snapshot();
        snapshot.state_at = "2026-10-05T00:00:01Z".into();
        let reference = snapshot.snapshot_reference().unwrap();
        let adapter =
            JsonControllerDocumentSnapshotAdapter::new("/tmp/does-not-matter", reference).unwrap();

        assert!(matches!(
            adapter.resolve_snapshot(&request, snapshot),
            Err(SnapshotError::Verification(
                VerificationFailure::HistoricalStateMismatch
            ))
        ));
    }

    #[test]
    fn payload_size_limit_applies_to_document_not_snapshot_metadata() {
        let request = request()
            .with_controller_document_network_policy(
                symthaea_epistemic_types::ControllerDocumentNetworkPolicy {
                    allowed_schemes: vec!["https".into()],
                    max_response_bytes: 32,
                    max_redirects: 0,
                    require_effective_url_match: true,
                },
            )
            .unwrap();
        let snapshot = snapshot();
        let reference = snapshot.snapshot_reference().unwrap();
        let adapter =
            JsonControllerDocumentSnapshotAdapter::new("/tmp/does-not-matter", reference).unwrap();

        assert!(matches!(
            adapter.resolve_snapshot(&request, snapshot),
            Err(SnapshotError::Verification(
                VerificationFailure::ControllerDocumentResponseTooLarge
            ))
        ));
    }

    #[test]
    fn unsupported_snapshot_media_type_fails_closed_before_resolution() {
        let request = request();
        let mut snapshot = snapshot();
        snapshot.response_media_type = "text/plain".into();
        let reference = snapshot.snapshot_reference().unwrap();
        let adapter =
            JsonControllerDocumentSnapshotAdapter::new("/tmp/does-not-matter", reference).unwrap();

        assert!(matches!(
            adapter.resolve_snapshot(&request, snapshot),
            Err(SnapshotError::Malformed(message))
                if message.contains("unsupported controller-document snapshot media type")
        ));
    }

    #[test]
    fn missing_snapshot_is_unresolved_not_invalid() {
        let request = request();
        let adapter = JsonControllerDocumentSnapshotAdapter::new(
            "/tmp/symthaea-verification-adapter-missing-snapshot.json",
            snapshot().snapshot_reference().unwrap(),
        )
        .unwrap();

        assert!(matches!(
            adapter.resolve(&request),
            Err(SnapshotError::Unresolved(dependencies))
                if dependencies == vec![FederationDependency::ControllerDocument(
                    "https://example.test/controller".into()
                )]
        ));
    }

    #[test]
    fn filesystem_path_is_anchored_to_content_reference() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("controller-snapshot.json");
        let snapshot = snapshot();
        let reference = snapshot.snapshot_reference().unwrap();
        fs::write(&path, snapshot.to_json().unwrap()).unwrap();

        let adapter =
            JsonControllerDocumentSnapshotAdapter::new(&path, reference.clone()).unwrap();
        let resolution = adapter.resolve(&request()).unwrap();

        assert_eq!(
            resolution.controller_document_snapshot_scope.resolved_at(),
            "2026-10-05T02:00:00Z"
        );
        assert_eq!(
            resolution
                .controller_document_dereference
                .as_ref()
                .unwrap()
                .resolved_at,
            "2026-10-05T02:00:00Z"
        );
        assert!(adapter.expected_snapshot_reference() == reference);
    }
}
