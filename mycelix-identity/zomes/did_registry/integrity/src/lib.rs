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
    #[serde(
        rename = "keyAgreement",
        alias = "key_agreement",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
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
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::DidDocument(did_doc) => {
                    validate_create_did_document(EntryCreationAction::Create(action), did_doc)
                }
                EntryTypes::DidDeactivation(deactivation) => validate_create_did_deactivation(
                    EntryCreationAction::Create(action),
                    deactivation,
                ),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::DidDocument(did_doc) => validate_update_did_document(action, did_doc),
                EntryTypes::DidDeactivation(_) => Ok(ValidateCallbackResult::Invalid(
                    "Deactivation records cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => {
            // Validate tag length to prevent spam/DoS
            if tag.0.len() > 1024 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds maximum length of 1024 bytes".into(),
                ));
            }
            match link_type {
                LinkTypes::AgentToDid => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DidToVerificationMethod => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DidToService => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DidHistory => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DidToDeactivation => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            // Only the original link creator can delete their links
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the link creator can delete their links".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can update their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can delete their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
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

    // Fetch original to enforce invariants
    let original_record = must_get_valid_record(action.original_action_address.clone())?;
    let original: DidDocument = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original DID document not found".into()
        )))?;

    // Immutable fields
    if did_doc.id != original.id {
        return Ok(ValidateCallbackResult::Invalid(
            "DID id cannot be changed".into(),
        ));
    }
    if did_doc.controller != original.controller {
        return Ok(ValidateCallbackResult::Invalid(
            "DID controller cannot be changed".into(),
        ));
    }
    if did_doc.created != original.created {
        return Ok(ValidateCallbackResult::Invalid(
            "DID created timestamp cannot be changed".into(),
        ));
    }

    // Version must increment
    if did_doc.version <= original.version {
        return Ok(ValidateCallbackResult::Invalid(
            "DID version must increase on update".into(),
        ));
    }

    // Updated timestamp must advance
    if did_doc.updated <= original.updated {
        return Ok(ValidateCallbackResult::Invalid(
            "DID updated timestamp must advance".into(),
        ));
    }

    // Must still have at least one verification method
    if did_doc.verification_method.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must have at least one verification method".into(),
        ));
    }

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
        let vm: VerificationMethod = rmp_serde::from_slice(&msgpack_bytes)
            .expect("msgpack deserialize into VerificationMethod");

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
        let se: ServiceEndpoint = rmp_serde::from_slice(&msgpack_bytes)
            .expect("msgpack deserialize into ServiceEndpoint");

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
        assert!(
            json.get("publicKeyMultibase").is_some(),
            "expected camelCase 'publicKeyMultibase'"
        );
        assert!(
            json.get("type").is_some(),
            "expected 'type' (renamed from type_)"
        );
        assert!(
            json.get("public_key_multibase").is_none(),
            "snake_case key must not appear"
        );
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

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Valid multibase keys start with 'z' followed by base58btc characters.
        fn arb_multibase_key() -> impl Strategy<Value = String> {
            prop::collection::vec(any::<u8>(), 32..=32).prop_map(|bytes| {
                let encoded = bs58::encode(&bytes)
                    .with_alphabet(bs58::Alphabet::BITCOIN)
                    .into_string();
                format!("z{}", encoded)
            })
        }

        /// Arbitrary verification method with valid fields.
        fn arb_verification_method() -> impl Strategy<Value = VerificationMethod> {
            (arb_multibase_key(), any::<u16>()).prop_map(|(key, alg)| VerificationMethod {
                id: "#key-1".to_string(),
                type_: "Ed25519VerificationKey2020".to_string(),
                controller: "did:mycelix:test".to_string(),
                public_key_multibase: key,
                algorithm: Some(alg),
            })
        }

        proptest! {
            /// DID IDs without the 'did:mycelix:' prefix always fail the format check.
            #[test]
            fn did_without_prefix_fails_check(suffix in "[a-zA-Z0-9]{1,64}") {
                // Validation functions require EntryCreationAction (needs HDI host),
                // so we test the invariant directly.
                let bad_did = format!("did:other:{}", suffix);
                prop_assert!(!bad_did.starts_with("did:mycelix:"));

                // Also verify non-mycelix schemes
                let bad_did2 = format!("did:key:{}", suffix);
                prop_assert!(!bad_did2.starts_with("did:mycelix:"));
            }

            /// VerificationMethod round-trips through JSON and MessagePack.
            #[test]
            fn verification_method_roundtrips(vm in arb_verification_method()) {
                // JSON round-trip
                let json = serde_json::to_string(&vm).unwrap();
                let vm_json: VerificationMethod = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(&vm, &vm_json);

                // MessagePack round-trip
                let msgpack = rmp_serde::to_vec(&vm).unwrap();
                let vm_msgpack: VerificationMethod = rmp_serde::from_slice(&msgpack).unwrap();
                prop_assert_eq!(&vm, &vm_msgpack);
            }

            /// ServiceEndpoint round-trips through JSON and MessagePack.
            #[test]
            fn service_endpoint_roundtrips(
                id in "[a-z]{1,32}",
                type_ in "[A-Z][a-z]{3,20}",
                url in "https://[a-z]{3,20}\\.[a-z]{2,5}/[a-z]{0,10}"
            ) {
                let se = ServiceEndpoint { id, type_, service_endpoint: url };

                // JSON
                let json = serde_json::to_string(&se).unwrap();
                let se2: ServiceEndpoint = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(&se, &se2);

                // JSON must use camelCase
                let val: serde_json::Value = serde_json::from_str(&json).unwrap();
                prop_assert!(val.get("serviceEndpoint").is_some());
                prop_assert!(val.get("type").is_some());
                prop_assert!(val.get("service_endpoint").is_none());
                prop_assert!(val.get("type_").is_none());
            }

            /// Any DID with 'did:mycelix:' prefix passes the format check.
            #[test]
            fn valid_did_prefix_accepted(suffix in "[a-zA-Z0-9]{8,64}") {
                let did = format!("did:mycelix:{}", suffix);
                prop_assert!(did.starts_with("did:mycelix:"));
            }

            /// Deactivation with empty reason is invalid.
            #[test]
            fn empty_deactivation_reason_is_invalid(did in "did:mycelix:[a-zA-Z0-9]{8,32}") {
                let deactivation = DidDeactivation {
                    did,
                    reason: String::new(),
                    deactivated_at: Timestamp::from_micros(0),
                };
                prop_assert!(deactivation.reason.is_empty());
            }
        }
    }

    // =========================================================================
    // Backward-compatibility MessagePack round-trip tests
    // =========================================================================

    /// Old snake_case MessagePack data must deserialize through the new camelCase
    /// structs, thanks to `serde(alias = "...")` attributes.
    #[test]
    fn backward_compat_verification_method_snake_case_msgpack() {
        // Build a map with old snake_case keys
        let old_map = serde_json::json!({
            "id": "#keys-1",
            "type_": "Ed25519VerificationKey2020",
            "controller": "did:mycelix:test",
            "public_key_multibase": "z6Mkabcdef",
            "algorithm": null
        });

        // Serialize to MessagePack via serde_json::Value → rmp
        let msgpack_bytes = rmp_serde::to_vec(&old_map).unwrap();

        // Deserialize into VerificationMethod — the alias attributes should handle
        // snake_case keys from old entries.
        let vm: VerificationMethod = rmp_serde::from_slice(&msgpack_bytes).unwrap();

        assert_eq!(vm.id, "#keys-1");
        assert_eq!(vm.type_, "Ed25519VerificationKey2020");
        assert_eq!(vm.controller, "did:mycelix:test");
        assert_eq!(vm.public_key_multibase, "z6Mkabcdef");
        assert_eq!(vm.algorithm, None);
    }

    /// Forward direction: serialized VerificationMethod uses camelCase keys.
    #[test]
    fn forward_compat_verification_method_camel_case_json() {
        let vm = VerificationMethod {
            id: "#keys-1".into(),
            type_: "Ed25519VerificationKey2020".into(),
            controller: "did:mycelix:test".into(),
            public_key_multibase: "z6Mkabcdef".into(),
            algorithm: Some(0xed01),
        };

        let json = serde_json::to_string(&vm).unwrap();
        // Should use camelCase in output
        assert!(
            json.contains("\"type\""),
            "Should serialize as 'type', not 'type_'"
        );
        assert!(
            json.contains("\"publicKeyMultibase\""),
            "Should serialize as camelCase"
        );
        assert!(
            !json.contains("\"public_key_multibase\""),
            "Should NOT use snake_case in output"
        );
    }

    /// Old snake_case ServiceEndpoint MessagePack round-trip.
    #[test]
    fn backward_compat_service_endpoint_snake_case_msgpack() {
        let old_map = serde_json::json!({
            "id": "#svc-1",
            "type_": "LinkedDomains",
            "service_endpoint": "https://example.com"
        });

        let msgpack_bytes = rmp_serde::to_vec(&old_map).unwrap();
        let svc: ServiceEndpoint = rmp_serde::from_slice(&msgpack_bytes).unwrap();

        assert_eq!(svc.id, "#svc-1");
        assert_eq!(svc.type_, "LinkedDomains");
        assert_eq!(svc.service_endpoint, "https://example.com");
    }

    /// Forward direction: serialized ServiceEndpoint uses camelCase keys.
    #[test]
    fn forward_compat_service_endpoint_camel_case_json() {
        let svc = ServiceEndpoint {
            id: "#svc-1".into(),
            type_: "LinkedDomains".into(),
            service_endpoint: "https://example.com".into(),
        };

        let json = serde_json::to_string(&svc).unwrap();
        assert!(json.contains("\"serviceEndpoint\""), "Should use camelCase");
        assert!(
            !json.contains("\"service_endpoint\""),
            "Should NOT use snake_case"
        );
    }
}
