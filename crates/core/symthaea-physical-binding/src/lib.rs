// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical implementation capsules and exact backend-binding preflight.
//!
//! This crate is intentionally pure: it does not read the filesystem, invoke
//! tools, contact providers, or execute a backend. A result-producing runner
//! must explicitly gather the exact bytes it intends to bind and supply them
//! here before execution.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use symthaea_physical_cognition::BackendIdentity;
use symthaea_physical_experiment::{BackendBinding, Digest32};

const CAPSULE_DOMAIN: &[u8] = b"symthaea:physical:implementation-capsule:v1\0";
const MAX_CAPSULE_ENTRIES: usize = 1024;
const MAX_PATH_BYTES: usize = 4096;
const MAX_ENTRY_BYTES: usize = 64 * 1024 * 1024;
const MAX_TOTAL_BYTES: usize = 512 * 1024 * 1024;

/// One repository-relative file included in an implementation capsule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapsuleEntry {
    /// Canonical repository-relative path using `/` separators.
    pub path: String,
    /// Exact raw file bytes.
    pub content: Vec<u8>,
}

impl CapsuleEntry {
    /// Construct and validate one capsule entry.
    pub fn new(
        path: impl Into<String>,
        content: impl Into<Vec<u8>>,
    ) -> Result<Self, BindingError> {
        let entry = Self {
            path: path.into(),
            content: content.into(),
        };
        entry.validate()?;
        Ok(entry)
    }

    /// Validate path and bounded content size.
    pub fn validate(&self) -> Result<(), BindingError> {
        validate_repository_path(&self.path)?;
        if self.content.len() > MAX_ENTRY_BYTES {
            return Err(BindingError::EntryTooLarge {
                path: self.path.clone(),
                bytes: self.content.len(),
            });
        }
        Ok(())
    }
}

/// Deterministic set of exact files defining an implementation subject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplementationCapsule {
    entries: Vec<CapsuleEntry>,
}

impl ImplementationCapsule {
    /// Construct a capsule, validating and sorting entries bytewise by path.
    pub fn new(mut entries: Vec<CapsuleEntry>) -> Result<Self, BindingError> {
        if entries.is_empty() {
            return Err(BindingError::EmptyCapsule);
        }
        if entries.len() > MAX_CAPSULE_ENTRIES {
            return Err(BindingError::TooManyEntries(entries.len()));
        }

        let mut total = 0usize;
        for entry in &entries {
            entry.validate()?;
            total = total
                .checked_add(entry.content.len())
                .ok_or(BindingError::TotalSizeOverflow)?;
            if total > MAX_TOTAL_BYTES {
                return Err(BindingError::CapsuleTooLarge(total));
            }
        }

        entries.sort_by(|a, b| a.path.as_bytes().cmp(b.path.as_bytes()));
        for pair in entries.windows(2) {
            if pair[0].path == pair[1].path {
                return Err(BindingError::DuplicatePath(pair[0].path.clone()));
            }
        }

        Ok(Self { entries })
    }

    /// Canonically sorted capsule entries.
    pub fn entries(&self) -> &[CapsuleEntry] {
        &self.entries
    }

    /// Total raw content bytes across all entries.
    pub fn total_content_bytes(&self) -> usize {
        self.entries.iter().map(|entry| entry.content.len()).sum()
    }

    /// Canonical V1 bytes.
    ///
    /// Encoding is exactly: domain separator, followed by each
    /// bytewise-path-sorted entry as
    /// `u32_le(path_len) || path || u64_le(content_len) || content`.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, BindingError> {
        let mut out = Vec::new();
        out.extend_from_slice(CAPSULE_DOMAIN);
        for entry in &self.entries {
            let path_len =
                u32::try_from(entry.path.len()).map_err(|_| BindingError::LengthOverflow)?;
            let content_len =
                u64::try_from(entry.content.len()).map_err(|_| BindingError::LengthOverflow)?;
            out.extend_from_slice(&path_len.to_le_bytes());
            out.extend_from_slice(entry.path.as_bytes());
            out.extend_from_slice(&content_len.to_le_bytes());
            out.extend_from_slice(&entry.content);
        }
        Ok(out)
    }

    /// Domain-separated BLAKE3 digest through PHYS-002 `Digest32`.
    pub fn digest(&self) -> Result<Digest32, BindingError> {
        Ok(Digest32::blake3(&self.canonical_bytes()?))
    }
}

/// Optional common trait for backend configuration types.
///
/// Implementations should include their own schema/version domain separator in
/// the returned bytes. Backend crates may also use an equivalent inherent
/// canonicalizer if changing an already-frozen subject would be undesirable.
pub trait CanonicalConfiguration {
    /// Return exact behavior-affecting configuration bytes.
    fn canonical_configuration_bytes(&self) -> Result<Vec<u8>, BindingError>;

    /// BLAKE3 digest of the exact canonical configuration bytes.
    fn configuration_digest(&self) -> Result<Digest32, BindingError> {
        let bytes = self.canonical_configuration_bytes()?;
        if bytes.is_empty() {
            return Err(BindingError::EmptyConfiguration);
        }
        let digest = Digest32::blake3(&bytes);
        if digest.is_zero() {
            return Err(BindingError::ZeroConfigurationDigest);
        }
        Ok(digest)
    }
}

/// Derive an exact PHYS-002 backend binding from identity, source capsule, and
/// an independently derived configuration digest.
pub fn derive_backend_binding(
    identity: BackendIdentity,
    implementation: &ImplementationCapsule,
    configuration_digest: Digest32,
) -> Result<BackendBinding, BindingError> {
    identity
        .validate()
        .map_err(|error| BindingError::InvalidIdentity(error.to_string()))?;
    if configuration_digest.is_zero() {
        return Err(BindingError::ZeroConfigurationDigest);
    }
    let binding = BackendBinding {
        backend: identity,
        implementation_digest: implementation.digest()?,
        configuration_digest,
    };
    binding
        .validate()
        .map_err(|error| BindingError::InvalidBinding(error.to_string()))?;
    Ok(binding)
}

/// Fail closed unless expected and locally derived bindings match exactly.
pub fn verify_exact_binding(
    expected: &BackendBinding,
    actual: &BackendBinding,
) -> Result<(), BindingError> {
    expected
        .validate()
        .map_err(|error| BindingError::InvalidExpectedBinding(error.to_string()))?;
    actual
        .validate()
        .map_err(|error| BindingError::InvalidActualBinding(error.to_string()))?;

    if expected.backend != actual.backend {
        return Err(BindingError::BackendIdentityMismatch);
    }
    if expected.implementation_digest != actual.implementation_digest {
        return Err(BindingError::ImplementationDigestMismatch);
    }
    if expected.configuration_digest != actual.configuration_digest {
        return Err(BindingError::ConfigurationDigestMismatch);
    }
    Ok(())
}

/// Exact-binding validation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingError {
    /// Capsule contained no implementation files.
    EmptyCapsule,
    /// Capsule contained more entries than the bounded contract permits.
    TooManyEntries(usize),
    /// Repository-relative path is not canonical.
    InvalidPath(String),
    /// Capsule repeats the same canonical path.
    DuplicatePath(String),
    /// One file exceeds the bounded entry size.
    EntryTooLarge {
        /// Canonical path.
        path: String,
        /// Actual content bytes.
        bytes: usize,
    },
    /// Sum of entry sizes overflowed `usize`.
    TotalSizeOverflow,
    /// Capsule exceeds the bounded aggregate size.
    CapsuleTooLarge(usize),
    /// Canonical length conversion failed.
    LengthOverflow,
    /// Configuration serializer returned no bytes.
    EmptyConfiguration,
    /// Reserved all-zero configuration digest was supplied.
    ZeroConfigurationDigest,
    /// Backend identity failed PHYS-001 validation.
    InvalidIdentity(String),
    /// Newly derived PHYS-002 binding failed validation.
    InvalidBinding(String),
    /// Expected binding is malformed.
    InvalidExpectedBinding(String),
    /// Locally derived binding is malformed.
    InvalidActualBinding(String),
    /// Stable backend identity differs.
    BackendIdentityMismatch,
    /// Exact implementation capsule digest differs.
    ImplementationDigestMismatch,
    /// Exact behavior-affecting configuration digest differs.
    ConfigurationDigestMismatch,
}

impl std::fmt::Display for BindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCapsule => write!(f, "implementation capsule must not be empty"),
            Self::TooManyEntries(count) => write!(f, "too many capsule entries: {count}"),
            Self::InvalidPath(path) => write!(f, "non-canonical capsule path: {path:?}"),
            Self::DuplicatePath(path) => write!(f, "duplicate capsule path: {path:?}"),
            Self::EntryTooLarge { path, bytes } => {
                write!(f, "capsule entry {path:?} is too large: {bytes} bytes")
            }
            Self::TotalSizeOverflow => write!(f, "capsule total size overflow"),
            Self::CapsuleTooLarge(bytes) => write!(f, "capsule is too large: {bytes} bytes"),
            Self::LengthOverflow => write!(f, "canonical capsule length overflow"),
            Self::EmptyConfiguration => write!(f, "canonical configuration must not be empty"),
            Self::ZeroConfigurationDigest => {
                write!(f, "configuration digest must not be all-zero")
            }
            Self::InvalidIdentity(error) => write!(f, "invalid backend identity: {error}"),
            Self::InvalidBinding(error) => write!(f, "invalid derived binding: {error}"),
            Self::InvalidExpectedBinding(error) => {
                write!(f, "invalid expected binding: {error}")
            }
            Self::InvalidActualBinding(error) => write!(f, "invalid actual binding: {error}"),
            Self::BackendIdentityMismatch => write!(f, "backend identity mismatch"),
            Self::ImplementationDigestMismatch => write!(f, "implementation digest mismatch"),
            Self::ConfigurationDigestMismatch => write!(f, "configuration digest mismatch"),
        }
    }
}

impl std::error::Error for BindingError {}

fn validate_repository_path(path: &str) -> Result<(), BindingError> {
    let invalid = path.is_empty()
        || path.len() > MAX_PATH_BYTES
        || path.starts_with('/')
        || path.contains('\\')
        || path.contains(':')
        || path.chars().any(char::is_control)
        || path
            .split('/')
            .any(|segment| segment.is_empty() || segment == "." || segment == "..");

    if invalid {
        Err(BindingError::InvalidPath(path.to_owned()))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(path: &str, content: &[u8]) -> CapsuleEntry {
        CapsuleEntry::new(path, content).unwrap()
    }

    fn identity() -> BackendIdentity {
        BackendIdentity::new("physical:test", "v1", "test-implementation").unwrap()
    }

    #[test]
    fn insertion_order_does_not_change_capsule_identity() {
        let a =
            ImplementationCapsule::new(vec![entry("b.rs", b"b"), entry("a.rs", b"a")]).unwrap();
        let b =
            ImplementationCapsule::new(vec![entry("a.rs", b"a"), entry("b.rs", b"b")]).unwrap();
        assert_eq!(a.entries()[0].path, "a.rs");
        assert_eq!(a.digest().unwrap(), b.digest().unwrap());
    }

    #[test]
    fn path_or_content_change_changes_digest() {
        let base = ImplementationCapsule::new(vec![entry("src/lib.rs", b"alpha")]).unwrap();
        let path_changed =
            ImplementationCapsule::new(vec![entry("src/model.rs", b"alpha")]).unwrap();
        let content_changed =
            ImplementationCapsule::new(vec![entry("src/lib.rs", b"beta")]).unwrap();
        assert_ne!(base.digest().unwrap(), path_changed.digest().unwrap());
        assert_ne!(base.digest().unwrap(), content_changed.digest().unwrap());
    }

    #[test]
    fn duplicate_and_noncanonical_paths_fail_closed() {
        assert!(matches!(
            ImplementationCapsule::new(vec![entry("a", b"1"), entry("a", b"2")]),
            Err(BindingError::DuplicatePath(_))
        ));
        for path in ["", "/abs", "../escape", "a/../b", "a//b", "a\\b", "C:/x"] {
            assert!(matches!(
                CapsuleEntry::new(path, b"x"),
                Err(BindingError::InvalidPath(_))
            ));
        }
    }

    #[test]
    fn binding_preflight_distinguishes_identity_implementation_and_config_drift() {
        let capsule =
            ImplementationCapsule::new(vec![entry("src/lib.rs", b"subject")]).unwrap();
        let config = Digest32::blake3(b"config-v1");
        let expected = derive_backend_binding(identity(), &capsule, config).unwrap();
        verify_exact_binding(&expected, &expected).unwrap();

        let mut identity_drift = expected.clone();
        identity_drift.backend =
            BackendIdentity::new("physical:other", "v1", "test-implementation").unwrap();
        assert_eq!(
            verify_exact_binding(&expected, &identity_drift),
            Err(BindingError::BackendIdentityMismatch)
        );

        let other_capsule =
            ImplementationCapsule::new(vec![entry("src/lib.rs", b"changed")]).unwrap();
        let implementation_drift =
            derive_backend_binding(identity(), &other_capsule, config).unwrap();
        assert_eq!(
            verify_exact_binding(&expected, &implementation_drift),
            Err(BindingError::ImplementationDigestMismatch)
        );

        let config_drift = derive_backend_binding(
            identity(),
            &capsule,
            Digest32::blake3(b"config-v2"),
        )
        .unwrap();
        assert_eq!(
            verify_exact_binding(&expected, &config_drift),
            Err(BindingError::ConfigurationDigestMismatch)
        );
    }

    #[test]
    fn zero_config_digest_is_rejected_before_binding() {
        let capsule =
            ImplementationCapsule::new(vec![entry("src/lib.rs", b"subject")]).unwrap();
        assert_eq!(
            derive_backend_binding(identity(), &capsule, Digest32([0; 32])),
            Err(BindingError::ZeroConfigurationDigest)
        );
    }

    struct ExampleConfig(u64);

    impl CanonicalConfiguration for ExampleConfig {
        fn canonical_configuration_bytes(&self) -> Result<Vec<u8>, BindingError> {
            let mut bytes = b"example-config:v1\0".to_vec();
            bytes.extend_from_slice(&self.0.to_le_bytes());
            Ok(bytes)
        }
    }

    #[test]
    fn canonical_configuration_trait_is_deterministic_and_sensitive() {
        let a = ExampleConfig(7).configuration_digest().unwrap();
        let b = ExampleConfig(7).configuration_digest().unwrap();
        let c = ExampleConfig(8).configuration_digest().unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
