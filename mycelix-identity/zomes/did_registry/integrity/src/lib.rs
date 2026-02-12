//! DID Registry Integrity Zome
//! Defines entry types and validation for DID:mycelix identifiers
//!
//! Updated to use HDI 0.7 patterns with FlatOp validation

use hdi::prelude::*;

/// DID Document entry type
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DidDocument {
    /// The DID identifier (did:mycelix:<agent_pub_key>)
    pub id: String,
    /// Controller of this DID (usually self)
    pub controller: AgentPubKey,
    /// Verification methods (public keys)
    #[serde(rename = "verificationMethod", alias = "verification_method")]
    pub verification_method: Vec<VerificationMethod>,
    /// Authentication methods
    pub authentication: Vec<String>,
    /// Key agreement methods for encryption (W3C DID Core §5.3.3).
    ///
    /// Each entry is a DID URL fragment (e.g. "#kem-1") referencing a
    /// `VerificationMethod` with an ML-KEM public key. Recipients use this
    /// to look up the KEM key for encrypting data to this DID's owner.
    #[serde(rename = "keyAgreement", alias = "key_agreement", default, skip_serializing_if = "Vec::is_empty")]
    pub key_agreement: Vec<String>,
    /// Service endpoints
    pub service: Vec<ServiceEndpoint>,
    /// Creation timestamp
    pub created: Timestamp,
    /// Last update timestamp
    pub updated: Timestamp,
    /// Version number for updates
    pub version: u32,
}

/// Verification method for cryptographic operations
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct VerificationMethod {
    pub id: String,
    #[serde(rename = "type", alias = "type_")]
    pub type_: String,
    pub controller: String,
    #[serde(rename = "publicKeyMultibase", alias = "public_key_multibase")]
    pub public_key_multibase: String,
    /// Algorithm identifier (multicodec u16). None defaults to Ed25519 (0xed01).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<u16>,
}

/// Service endpoint for discovery
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub id: String,
    #[serde(rename = "type", alias = "type_")]
    pub type_: String,
    #[serde(rename = "serviceEndpoint", alias = "service_endpoint")]
    pub service_endpoint: String,
}

/// DID Deactivation record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DidDeactivation {
    pub did: String,
    pub reason: String,
    pub deactivated_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    DidDocument(DidDocument),
    DidDeactivation(DidDeactivation),
}

#[hdk_link_types]
pub enum LinkTypes {
    AgentToDid,
    DidToVerificationMethod,
    DidToService,
    DidHistory,
    DidToDeactivation,
}

/// Genesis self-check - called when app is installed
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern matching
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                match app_entry {
                    EntryTypes::DidDocument(did_doc) => {
                        validate_create_did_document(EntryCreationAction::Create(action), did_doc)
                    }
                    EntryTypes::DidDeactivation(deactivation) => {
                        validate_create_did_deactivation(EntryCreationAction::Create(action), deactivation)
                    }
                }
            }
            OpEntry::UpdateEntry { app_entry, action, .. } => {
                match app_entry {
                    EntryTypes::DidDocument(did_doc) => {
                        validate_update_did_document(action, did_doc)
                    }
                    EntryTypes::DidDeactivation(_) => {
                        Ok(ValidateCallbackResult::Invalid(
                            "Deactivation records cannot be updated".into(),
                        ))
                    }
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => {
            match link_type {
                LinkTypes::AgentToDid => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DidToVerificationMethod => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DidToService => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DidHistory => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DidToDeactivation => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink { .. } => {
            // Links can be deleted
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate DID document creation
fn validate_create_did_document(
    action: EntryCreationAction,
    did_doc: DidDocument,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DID format
    if !did_doc.id.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:mycelix:'".into(),
        ));
    }

    // Validate controller matches author
    let author = action.author();
    if did_doc.controller != *author {
        return Ok(ValidateCallbackResult::Invalid(
            "DID controller must be the author".into(),
        ));
    }

    // Validate at least one verification method
    if did_doc.verification_method.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must have at least one verification method".into(),
        ));
    }

    // Validate version starts at 1
    if did_doc.version != 1 {
        return Ok(ValidateCallbackResult::Invalid(
            "Initial DID version must be 1".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate DID document update
fn validate_update_did_document(
    action: Update,
    did_doc: DidDocument,
) -> ExternResult<ValidateCallbackResult> {
    // Validate author is controller
    if did_doc.controller != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Only controller can update DID".into(),
        ));
    }

    // Version validation would require fetching original - skip for now
    // More complex validation can be added later

    Ok(ValidateCallbackResult::Valid)
}

/// Validate DID deactivation creation
fn validate_create_did_deactivation(
    _action: EntryCreationAction,
    deactivation: DidDeactivation,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DID format
    if !deactivation.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:mycelix:'".into(),
        ));
    }

    // Validate reason provided
    if deactivation.reason.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Deactivation reason is required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that old snake_case MessagePack payloads deserialize through the
    /// new camelCase structs thanks to `#[serde(alias = "...")]` attributes.
    #[test]
    fn backward_compat_snake_case_msgpack_to_struct() {
        // Build a VerificationMethod map using the OLD snake_case field names.
        let old_vm = serde_json::json!({
            "id": "#key-1",
            "type_": "Ed25519VerificationKey2020",
            "controller": "did:mycelix:abc123",
            "public_key_multibase": "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
        });

        // Serialize to MessagePack (simulates data written by old code).
        let msgpack_bytes = rmp_serde::to_vec(&old_vm).expect("msgpack serialize");

        // Deserialize into the new struct — alias attributes must accept snake_case.
        let vm: VerificationMethod =
            rmp_serde::from_slice(&msgpack_bytes).expect("msgpack deserialize into VerificationMethod");

        assert_eq!(vm.id, "#key-1");
        assert_eq!(vm.type_, "Ed25519VerificationKey2020");
        assert_eq!(vm.controller, "did:mycelix:abc123");
        assert_eq!(
            vm.public_key_multibase,
            "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
        );
    }

    /// Verify that old snake_case ServiceEndpoint MessagePack deserializes correctly.
    #[test]
    fn backward_compat_snake_case_service_endpoint() {
        let old_se = serde_json::json!({
            "id": "svc-1",
            "type_": "LinkedDomains",
            "service_endpoint": "https://example.com"
        });

        let msgpack_bytes = rmp_serde::to_vec(&old_se).expect("msgpack serialize");
        let se: ServiceEndpoint =
            rmp_serde::from_slice(&msgpack_bytes).expect("msgpack deserialize into ServiceEndpoint");

        assert_eq!(se.id, "svc-1");
        assert_eq!(se.type_, "LinkedDomains");
        assert_eq!(se.service_endpoint, "https://example.com");
    }

    /// Verify that forward serialization uses camelCase keys (W3C DID Core compliant).
    #[test]
    fn forward_serialization_uses_camel_case() {
        let vm = VerificationMethod {
            id: "#key-1".into(),
            type_: "Ed25519VerificationKey2020".into(),
            controller: "did:mycelix:abc123".into(),
            public_key_multibase: "z6Mk...".into(),
            algorithm: Some(0xed01),
        };

        let json = serde_json::to_value(&vm).expect("serialize to JSON");
        // Must use camelCase, not snake_case
        assert!(json.get("publicKeyMultibase").is_some(), "expected camelCase 'publicKeyMultibase'");
        assert!(json.get("type").is_some(), "expected 'type' (renamed from type_)");
        assert!(json.get("public_key_multibase").is_none(), "snake_case key must not appear");
        assert!(json.get("type_").is_none(), "type_ must not appear");
    }

    /// Verify that the camelCase JSON round-trips through MessagePack correctly.
    #[test]
    fn camel_case_json_to_msgpack_round_trip() {
        let vm = VerificationMethod {
            id: "#key-2".into(),
            type_: "MlDsa65VerificationKey2024".into(),
            controller: "did:mycelix:def456".into(),
            public_key_multibase: "zABC...".into(),
            algorithm: Some(0x0901),
        };

        let msgpack_bytes = rmp_serde::to_vec(&vm).expect("msgpack serialize");
        let vm2: VerificationMethod =
            rmp_serde::from_slice(&msgpack_bytes).expect("msgpack deserialize");

        assert_eq!(vm, vm2);
    }
}
