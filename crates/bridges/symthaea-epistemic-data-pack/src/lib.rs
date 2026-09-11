// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Strict declarative import boundary for Symthaea's canonical epistemic ledger.
//!
//! Third-party packs do not deserialize directly into `GlobalEpistemicLedger`.
//! This bridge owns a versioned, deny-unknown-fields wire format, validates it,
//! and only then converts it into the existing canonical domain type.

#![deny(unsafe_code)]

use serde::Deserialize;
use std::collections::BTreeSet;
use symthaea_epistemic_types::{GlobalClaim, GlobalClaimStatus, GlobalEpistemicLedger};
use symthaea_extension_core::CapabilityId;
use symthaea_extension_data_host::{DataPackValidator, ValidatedDataPack};
use thiserror::Error;

pub const GLOBAL_CLAIMS_CAPABILITY: &str = "knowledge.epistemic.global_claims";
pub const GLOBAL_CLAIMS_SCHEMA_V1: &str = "symthaea.epistemic.global_claims.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpistemicPackLimits {
    pub max_claims: usize,
    pub max_domain_bytes: usize,
    pub max_name_bytes: usize,
    pub max_proof_ref_bytes: usize,
}

impl Default for EpistemicPackLimits {
    fn default() -> Self {
        Self {
            max_claims: 10_000,
            max_domain_bytes: 128,
            max_name_bytes: 1024,
            max_proof_ref_bytes: 4096,
        }
    }
}

/// Validator/decoder for the first public epistemic claim-pack capability.
#[derive(Debug)]
pub struct GlobalClaimsPackValidator {
    capability: CapabilityId,
    limits: EpistemicPackLimits,
}

impl Default for GlobalClaimsPackValidator {
    fn default() -> Self {
        Self::new(EpistemicPackLimits::default())
    }
}

impl GlobalClaimsPackValidator {
    pub fn new(limits: EpistemicPackLimits) -> Self {
        Self {
            capability: CapabilityId::new(GLOBAL_CLAIMS_CAPABILITY),
            limits,
        }
    }

    pub const fn limits(&self) -> EpistemicPackLimits {
        self.limits
    }

    /// Decode only a pack that already passed the generic data-host admission
    /// and exact-byte binding boundary.
    pub fn decode(
        &self,
        pack: &ValidatedDataPack<'_>,
    ) -> Result<GlobalEpistemicLedger, EpistemicPackError> {
        if pack.capability() != &self.capability {
            return Err(EpistemicPackError::WrongCapability);
        }

        let wire = self.parse_and_validate(pack.payload())?;
        let ledger = GlobalEpistemicLedger {
            claims: wire
                .claims
                .into_iter()
                .map(|claim| GlobalClaim {
                    domain: claim.domain,
                    name: claim.name,
                    status: claim.status.into(),
                    // The canonical domain type currently calls this a path.
                    // The public importer treats it as opaque provenance text;
                    // this bridge never opens or resolves it as a filesystem path.
                    formal_proof_path: claim.formal_proof_ref,
                })
                .collect(),
        };

        if !ledger.audit_all() {
            return Err(EpistemicPackError::CanonicalAuditFailed);
        }
        Ok(ledger)
    }

    fn parse_and_validate(&self, payload: &[u8]) -> Result<WirePackV1, EpistemicPackError> {
        let wire: WirePackV1 = serde_json::from_slice(payload)
            .map_err(|error| EpistemicPackError::PayloadJson(error.to_string()))?;

        if wire.schema != GLOBAL_CLAIMS_SCHEMA_V1 {
            return Err(EpistemicPackError::UnsupportedSchema(wire.schema));
        }
        if wire.claims.is_empty() {
            return Err(EpistemicPackError::EmptyClaims);
        }
        if wire.claims.len() > self.limits.max_claims {
            return Err(EpistemicPackError::TooManyClaims {
                actual: wire.claims.len(),
                maximum: self.limits.max_claims,
            });
        }

        let mut identities = BTreeSet::new();
        for (index, claim) in wire.claims.iter().enumerate() {
            validate_text(
                "domain",
                index,
                &claim.domain,
                self.limits.max_domain_bytes,
            )?;
            validate_text("name", index, &claim.name, self.limits.max_name_bytes)?;

            if let Some(reference) = &claim.formal_proof_ref {
                validate_text(
                    "formal_proof_ref",
                    index,
                    reference,
                    self.limits.max_proof_ref_bytes,
                )?;
            }
            if claim.status == WireClaimStatus::Proven && claim.formal_proof_ref.is_none() {
                return Err(EpistemicPackError::ProvenClaimMissingProof { index });
            }

            let identity = (claim.domain.clone(), claim.name.clone());
            if !identities.insert(identity) {
                return Err(EpistemicPackError::DuplicateClaim {
                    index,
                    domain: claim.domain.clone(),
                    name: claim.name.clone(),
                });
            }
        }

        Ok(wire)
    }
}

impl DataPackValidator for GlobalClaimsPackValidator {
    type Error = EpistemicPackError;

    fn capability(&self) -> &CapabilityId {
        &self.capability
    }

    fn validate(&self, payload: &[u8]) -> Result<(), Self::Error> {
        self.parse_and_validate(payload).map(|_| ())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePackV1 {
    schema: String,
    claims: Vec<WireClaimV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireClaimV1 {
    domain: String,
    name: String,
    status: WireClaimStatus,
    #[serde(default)]
    formal_proof_ref: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireClaimStatus {
    Heuristic,
    Formalized,
    Proven,
}

impl From<WireClaimStatus> for GlobalClaimStatus {
    fn from(value: WireClaimStatus) -> Self {
        match value {
            WireClaimStatus::Heuristic => Self::Heuristic,
            WireClaimStatus::Formalized => Self::Formalized,
            WireClaimStatus::Proven => Self::Proven,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EpistemicPackError {
    #[error("payload JSON is invalid: {0}")]
    PayloadJson(String),
    #[error("unsupported epistemic claim-pack schema {0:?}")]
    UnsupportedSchema(String),
    #[error("claim pack must contain at least one claim")]
    EmptyClaims,
    #[error("claim pack contains {actual} claims; maximum is {maximum}")]
    TooManyClaims { actual: usize, maximum: usize },
    #[error("claim {index} has invalid {field}: {reason}")]
    InvalidText {
        field: &'static str,
        index: usize,
        reason: &'static str,
    },
    #[error("proven claim {index} is missing formal_proof_ref")]
    ProvenClaimMissingProof { index: usize },
    #[error("duplicate claim at index {index}: {domain:?} / {name:?}")]
    DuplicateClaim {
        index: usize,
        domain: String,
        name: String,
    },
    #[error("validated data pack carries the wrong semantic capability")]
    WrongCapability,
    #[error("decoded canonical ledger failed its own audit")]
    CanonicalAuditFailed,
}

fn validate_text(
    field: &'static str,
    index: usize,
    value: &str,
    max_bytes: usize,
) -> Result<(), EpistemicPackError> {
    let reason = if value.is_empty() {
        Some("value is empty")
    } else if value != value.trim() {
        Some("value has leading or trailing whitespace")
    } else if value.len() > max_bytes {
        Some("value exceeds byte limit")
    } else if value.chars().any(char::is_control) {
        Some("value contains control characters")
    } else {
        None
    };

    if let Some(reason) = reason {
        Err(EpistemicPackError::InvalidText {
            field,
            index,
            reason,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use symthaea_extension_admission::{
        AdmissionContext, AdmissionCurrentnessSource, AdmissionRecord, AdmissionSubject,
        PrincipalId, Sha256Digest, TrustLevel,
    };
    use symthaea_extension_authority::{AdmissionAuthority, ScopedAdmission};
    use symthaea_extension_core::{ExtensionManifest, PermissionSet};
    use symthaea_extension_data_host::DataPackHost;

    #[derive(Debug, Clone, Copy)]
    struct FixedCurrentness(AdmissionContext);

    impl AdmissionCurrentnessSource for FixedCurrentness {
        fn current_context(&self, subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
            assert_eq!(subject.extension().as_str(), "org.example.epistemic-ledger");
            assert_eq!(subject.extension_version(), "1.0.0");
            Some(self.0)
        }
    }

    fn current() -> FixedCurrentness {
        FixedCurrentness(AdmissionContext::active(3, 5))
    }

    fn digest(bytes: &[u8]) -> Sha256Digest {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        Sha256Digest::new(hasher.finalize().into())
    }

    fn manifest_bytes() -> &'static [u8] {
        br#"{
            "id":"org.example.epistemic-ledger",
            "name":"Example Epistemic Ledger",
            "version":"1.0.0",
            "kind":"knowledge_pack",
            "runtime":"data_only",
            "provides":[{
                "id":"knowledge.epistemic.global_claims",
                "description":"Validated global epistemic claims",
                "effect":"pure"
            }],
            "resources":{
                "memory_bytes":0,
                "fuel":0,
                "max_wall_time_ms":0,
                "max_output_bytes":0,
                "max_concurrency":0
            }
        }"#
    }

    fn valid_payload() -> &'static [u8] {
        br#"{
            "schema":"symthaea.epistemic.global_claims.v1",
            "claims":[
                {
                    "domain":"physics",
                    "name":"example heuristic",
                    "status":"heuristic"
                },
                {
                    "domain":"mathematics",
                    "name":"example theorem",
                    "status":"proven",
                    "formal_proof_ref":"proofs/example.lean"
                }
            ]
        }"#
    }

    fn record(manifest_bytes: &[u8], payload: &[u8]) -> AdmissionRecord {
        let manifest: ExtensionManifest = serde_json::from_slice(manifest_bytes).unwrap();
        AdmissionRecord::issue(
            manifest.id.clone(),
            manifest.version.clone(),
            digest(manifest_bytes),
            digest(payload),
            Sha256Digest::new([7; 32]),
            PrincipalId::new("local:test-authority").unwrap(),
            Some(PrincipalId::new("did:example:epistemic-publisher").unwrap()),
            TrustLevel::Community,
            vec![CapabilityId::new(GLOBAL_CLAIMS_CAPABILITY)],
            PermissionSet::default(),
            3,
            5,
        )
        .unwrap()
    }

    fn scoped_admission(
        authority: &AdmissionAuthority,
        manifest_bytes: &[u8],
        payload: &[u8],
    ) -> ScopedAdmission {
        let manifest: ExtensionManifest = serde_json::from_slice(manifest_bytes).unwrap();
        authority
            .activate(&record(manifest_bytes, payload), &manifest, &current())
            .unwrap()
    }

    #[test]
    fn strict_validator_rejects_unknown_fields() {
        let validator = GlobalClaimsPackValidator::default();
        let payload = br#"{
            "schema":"symthaea.epistemic.global_claims.v1",
            "claims":[{
                "domain":"physics",
                "name":"claim",
                "status":"heuristic",
                "surprise":true
            }]
        }"#;

        assert!(matches!(
            validator.validate(payload),
            Err(EpistemicPackError::PayloadJson(_))
        ));
    }

    #[test]
    fn proven_claim_requires_proof_reference() {
        let validator = GlobalClaimsPackValidator::default();
        let payload = br#"{
            "schema":"symthaea.epistemic.global_claims.v1",
            "claims":[{
                "domain":"mathematics",
                "name":"claim",
                "status":"proven"
            }]
        }"#;

        assert_eq!(
            validator.validate(payload),
            Err(EpistemicPackError::ProvenClaimMissingProof { index: 0 })
        );
    }

    #[test]
    fn duplicate_claim_identity_is_rejected() {
        let validator = GlobalClaimsPackValidator::default();
        let payload = br#"{
            "schema":"symthaea.epistemic.global_claims.v1",
            "claims":[
                {"domain":"physics","name":"same","status":"heuristic"},
                {"domain":"physics","name":"same","status":"formalized"}
            ]
        }"#;

        assert!(matches!(
            validator.validate(payload),
            Err(EpistemicPackError::DuplicateClaim { index: 1, .. })
        ));
    }

    #[test]
    fn admitted_pack_decodes_into_existing_canonical_ledger() {
        let validator = GlobalClaimsPackValidator::default();
        let authority = AdmissionAuthority::new();
        let host = DataPackHost::new(validator, authority.scope());
        let manifest = manifest_bytes();
        let payload = valid_payload();
        let admission = scoped_admission(&authority, manifest, payload);
        let source = current();
        let opened = host.open(manifest, payload, &admission, &source).unwrap();

        // Domain-specific decode occurs only after the generic host has produced
        // a validated, admitted point-of-use data handle.
        let validator = GlobalClaimsPackValidator::default();
        let ledger = validator.decode(&opened).unwrap();
        assert_eq!(ledger.claims.len(), 2);
        assert!(ledger.audit_all());
        assert_eq!(ledger.claims[1].status, GlobalClaimStatus::Proven);
        assert_eq!(
            ledger.claims[1].formal_proof_path.as_deref(),
            Some("proofs/example.lean")
        );
    }
}
