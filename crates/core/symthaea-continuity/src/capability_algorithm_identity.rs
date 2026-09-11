// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen V1 preimage specification for capability-analysis algorithm identities.
//!
//! The existing provenance module already derives stable algorithm IDs. This
//! module makes the exact byte contract behind those IDs first-class and
//! reviewable without changing either V1 identifier.
//!
//! V1 identity derivation is exactly:
//!
//! `BLAKE3-256(domain_separator || semantic_name_utf8)`.
//!
//! The V1 schema/version is encoded in the domain separator itself via the
//! exact suffix `.v1\0`. There is deliberately no separately prepended schema
//! token, length prefix, Unicode normalization step, or parameter map in V1.
//! Adding any such encoding step is an identity-layout change and therefore
//! requires a new version/domain rather than silently reinterpreting V1.
//!
//! Core theorem:
//!
//! `AlgorithmIdentityPreimage != AlgorithmExecution != ResultCorrectness != Authority`.

use crate::capability_analysis_provenance::{
    CAPABILITY_ACTIVATION_ALGORITHM_SEMANTICS_V1, CAPABILITY_COUNTERFACTUAL_ALGORITHM_SEMANTICS_V1,
    capability_activation_algorithm_id_v1, capability_counterfactual_algorithm_id_v1,
};

/// Human-readable schema name for the frozen V1 identity-preimage contract.
pub const CAPABILITY_ALGORITHM_IDENTITY_PREIMAGE_SCHEMA_V1: &str =
    "symthaea-continuity-capability-algorithm-identity-preimage-v1";

/// Digest algorithm used by the V1 identity-preimage contract.
pub const CAPABILITY_ALGORITHM_IDENTITY_DIGEST_V1: &str = "blake3-256";

/// Canonical V1 semantic-name encoding.
///
/// The semantic name contributes its exact UTF-8 bytes, with no normalization,
/// length prefix, terminator, case folding, or alternate representation.
pub const CAPABILITY_ALGORITHM_IDENTITY_SEMANTICS_ENCODING_V1: &str =
    "utf-8-exact-no-normalization-no-length-prefix";

/// Exact version bytes embedded at the end of every V1 algorithm domain.
pub const CAPABILITY_ALGORITHM_IDENTITY_DOMAIN_VERSION_SUFFIX_V1: &[u8] = b".v1\0";

/// Exact V1 domain separator for activation-algorithm identity.
pub const CAPABILITY_ACTIVATION_ALGORITHM_DOMAIN_SEPARATOR_V1: &[u8] =
    b"symthaea.continuity.capability-activation.algorithm-semantics.v1\0";

/// Exact V1 domain separator for counterfactual-algorithm identity.
pub const CAPABILITY_COUNTERFACTUAL_ALGORITHM_DOMAIN_SEPARATOR_V1: &[u8] =
    b"symthaea.continuity.capability-counterfactual.algorithm-semantics.v1\0";

/// Which capability-analysis algorithm family a V1 preimage specification names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CapabilityAlgorithmIdentityKindV1 {
    Activation,
    Counterfactual,
}

/// First-class specification of one exact V1 algorithm-identity preimage.
///
/// This is specification metadata, not a transport proof and not evidence that
/// the named algorithm was executed. It intentionally carries only immutable
/// references to the frozen V1 byte contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityAlgorithmIdentityPreimageSpecV1 {
    kind: CapabilityAlgorithmIdentityKindV1,
    domain_separator: &'static [u8],
    semantic_name: &'static str,
}

impl CapabilityAlgorithmIdentityPreimageSpecV1 {
    /// The algorithm family named by this preimage contract.
    pub const fn kind(&self) -> CapabilityAlgorithmIdentityKindV1 {
        self.kind
    }

    /// Frozen schema name describing the V1 preimage layout.
    pub const fn schema_version(&self) -> &'static str {
        CAPABILITY_ALGORITHM_IDENTITY_PREIMAGE_SCHEMA_V1
    }

    /// Digest algorithm used over the exact canonical preimage bytes.
    pub const fn digest_algorithm(&self) -> &'static str {
        CAPABILITY_ALGORITHM_IDENTITY_DIGEST_V1
    }

    /// Encoding contract for the semantic-name portion of the preimage.
    pub const fn semantic_name_encoding(&self) -> &'static str {
        CAPABILITY_ALGORITHM_IDENTITY_SEMANTICS_ENCODING_V1
    }

    /// Exact domain-separator bytes, including the terminal NUL.
    pub const fn domain_separator(&self) -> &'static [u8] {
        self.domain_separator
    }

    /// Exact semantic name whose UTF-8 bytes follow the domain separator.
    pub const fn semantic_name(&self) -> &'static str {
        self.semantic_name
    }

    /// Canonical V1 preimage bytes.
    ///
    /// Layout: `domain_separator || semantic_name.as_bytes()`.
    pub fn canonical_preimage(&self) -> Vec<u8> {
        let semantic_bytes = self.semantic_name.as_bytes();
        let mut out = Vec::with_capacity(self.domain_separator.len() + semantic_bytes.len());
        out.extend_from_slice(self.domain_separator);
        out.extend_from_slice(semantic_bytes);
        out
    }

    /// BLAKE3-256 digest of the exact canonical V1 preimage.
    pub fn digest_bytes(&self) -> [u8; 32] {
        *blake3::hash(&self.canonical_preimage()).as_bytes()
    }

    /// Whether this frozen specification reproduces the current public V1 ID.
    ///
    /// This is a compatibility check only; it proves no execution or authority.
    pub fn matches_public_v1_algorithm_id(&self) -> bool {
        match self.kind {
            CapabilityAlgorithmIdentityKindV1::Activation => {
                self.digest_bytes() == *capability_activation_algorithm_id_v1().as_bytes()
            }
            CapabilityAlgorithmIdentityKindV1::Counterfactual => {
                self.digest_bytes() == *capability_counterfactual_algorithm_id_v1().as_bytes()
            }
        }
    }
}

/// Frozen preimage specification for the V1 activation algorithm identity.
pub const fn capability_activation_algorithm_identity_preimage_spec_v1()
-> CapabilityAlgorithmIdentityPreimageSpecV1 {
    CapabilityAlgorithmIdentityPreimageSpecV1 {
        kind: CapabilityAlgorithmIdentityKindV1::Activation,
        domain_separator: CAPABILITY_ACTIVATION_ALGORITHM_DOMAIN_SEPARATOR_V1,
        semantic_name: CAPABILITY_ACTIVATION_ALGORITHM_SEMANTICS_V1,
    }
}

/// Frozen preimage specification for the V1 counterfactual algorithm identity.
pub const fn capability_counterfactual_algorithm_identity_preimage_spec_v1()
-> CapabilityAlgorithmIdentityPreimageSpecV1 {
    CapabilityAlgorithmIdentityPreimageSpecV1 {
        kind: CapabilityAlgorithmIdentityKindV1::Counterfactual,
        domain_separator: CAPABILITY_COUNTERFACTUAL_ALGORITHM_DOMAIN_SEPARATOR_V1,
        semantic_name: CAPABILITY_COUNTERFACTUAL_ALGORITHM_SEMANTICS_V1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACTIVATION_PREIMAGE_GOLDEN_V1: &[u8] = b"symthaea.continuity.capability-activation.algorithm-semantics.v1\0symthaea-continuity-capability-activation-monotone-fixed-point-v1";
    const COUNTERFACTUAL_PREIMAGE_GOLDEN_V1: &[u8] = b"symthaea.continuity.capability-counterfactual.algorithm-semantics.v1\0symthaea-continuity-capability-counterfactual-bounded-support-frontier-v1";

    const ACTIVATION_ID_GOLDEN_V1: [u8; 32] = [
        0x37, 0xc4, 0x19, 0x59, 0x4f, 0xf6, 0x86, 0xd8, 0x75, 0xb9, 0xd7, 0x4c, 0x46, 0xfe, 0x50,
        0x67, 0xaf, 0x58, 0xdd, 0xd9, 0x34, 0xd5, 0xf2, 0x2d, 0xf2, 0xaa, 0x28, 0xac, 0x0b, 0x03,
        0xc8, 0x91,
    ];

    const COUNTERFACTUAL_ID_GOLDEN_V1: [u8; 32] = [
        0x64, 0xce, 0x9f, 0x30, 0xf8, 0x49, 0xb2, 0x07, 0x35, 0xc6, 0x10, 0x88, 0x85, 0x53, 0x49,
        0x8f, 0x12, 0x35, 0x5e, 0x1b, 0x9b, 0x38, 0x41, 0x7d, 0xd3, 0x33, 0x80, 0xf1, 0xc4, 0x2b,
        0x31, 0x74,
    ];

    #[test]
    fn activation_preimage_contract_is_exactly_frozen() {
        let spec = capability_activation_algorithm_identity_preimage_spec_v1();
        assert_eq!(spec.kind(), CapabilityAlgorithmIdentityKindV1::Activation);
        assert_eq!(
            spec.schema_version(),
            CAPABILITY_ALGORITHM_IDENTITY_PREIMAGE_SCHEMA_V1
        );
        assert_eq!(
            spec.digest_algorithm(),
            CAPABILITY_ALGORITHM_IDENTITY_DIGEST_V1
        );
        assert_eq!(
            spec.semantic_name_encoding(),
            CAPABILITY_ALGORITHM_IDENTITY_SEMANTICS_ENCODING_V1
        );
        assert_eq!(spec.canonical_preimage(), ACTIVATION_PREIMAGE_GOLDEN_V1);
        assert_eq!(spec.digest_bytes(), ACTIVATION_ID_GOLDEN_V1);
        assert!(spec.matches_public_v1_algorithm_id());
    }

    #[test]
    fn counterfactual_preimage_contract_is_exactly_frozen() {
        let spec = capability_counterfactual_algorithm_identity_preimage_spec_v1();
        assert_eq!(
            spec.kind(),
            CapabilityAlgorithmIdentityKindV1::Counterfactual
        );
        assert_eq!(spec.canonical_preimage(), COUNTERFACTUAL_PREIMAGE_GOLDEN_V1);
        assert_eq!(spec.digest_bytes(), COUNTERFACTUAL_ID_GOLDEN_V1);
        assert!(spec.matches_public_v1_algorithm_id());
    }

    #[test]
    fn v1_domains_are_distinct_and_embed_the_exact_version_suffix() {
        let activation = capability_activation_algorithm_identity_preimage_spec_v1();
        let counterfactual = capability_counterfactual_algorithm_identity_preimage_spec_v1();

        assert_ne!(
            activation.domain_separator(),
            counterfactual.domain_separator()
        );
        assert!(
            activation
                .domain_separator()
                .ends_with(CAPABILITY_ALGORITHM_IDENTITY_DOMAIN_VERSION_SUFFIX_V1)
        );
        assert!(
            counterfactual
                .domain_separator()
                .ends_with(CAPABILITY_ALGORITHM_IDENTITY_DOMAIN_VERSION_SUFFIX_V1)
        );
        assert_ne!(activation.digest_bytes(), counterfactual.digest_bytes());
    }

    #[test]
    fn changing_only_the_domain_version_changes_identity() {
        let spec = capability_activation_algorithm_identity_preimage_spec_v1();
        let domain = spec.domain_separator();
        let suffix = CAPABILITY_ALGORITHM_IDENTITY_DOMAIN_VERSION_SUFFIX_V1;
        let prefix_len = domain.len() - suffix.len();

        let mut v2_preimage = domain[..prefix_len].to_vec();
        v2_preimage.extend_from_slice(b".v2\0");
        v2_preimage.extend_from_slice(spec.semantic_name().as_bytes());

        assert_ne!(*blake3::hash(&v2_preimage).as_bytes(), spec.digest_bytes());
    }

    #[test]
    fn changing_only_semantic_name_changes_identity() {
        let spec = capability_activation_algorithm_identity_preimage_spec_v1();
        let mut changed = spec.domain_separator().to_vec();
        changed.extend_from_slice(
            b"symthaea-continuity-capability-activation-monotone-fixed-point-v2",
        );

        assert_ne!(*blake3::hash(&changed).as_bytes(), spec.digest_bytes());
    }
}
