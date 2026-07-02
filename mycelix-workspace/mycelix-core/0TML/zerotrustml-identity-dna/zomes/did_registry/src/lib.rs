// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use ed25519_dalek::{Signature, VerifyingKey, Verifier};

//================================
// Entry Types
//================================

/// W3C DID Document stored on DHT
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DIDDocument {
    pub id: String,                    // "did:mycelix:abc123"
    pub controller: AgentPubKey,       // Holochain agent
    pub verification_methods: Vec<VerificationMethod>,
    pub authentication: Vec<String>,   // Key IDs for authentication
    pub created: i64,                  // Timestamp (microseconds)
    pub updated: i64,                  // Last update timestamp
    pub proof: Option<DIDProof>,       // Self-signature
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct VerificationMethod {
    pub id: String,                    // "did:mycelix:abc123#key-1"
    pub method_type: String,           // "Ed25519VerificationKey2020"
    pub controller: String,            // DID that controls this key
    pub public_key_multibase: String,  // Multibase-encoded public key
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct DIDProof {
    pub signature: Vec<u8>,            // Ed25519 signature
    pub verification_method: String,    // Key used to sign
    pub created: i64,                  // When proof was created
}

//================================
// Link Types
//================================

#[hdk_link_types]
pub enum LinkTypes {
    DIDResolution,       // Path -> DID Document
    DIDToDocument,       // DID string -> DID Document
}

//================================
// Entry Definitions
//================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type]
    DIDDocument(DIDDocument),
}

//================================
// Zome Functions
//================================

/// Create a new DID document on the DHT
#[hdk_extern]
pub fn create_did(input: CreateDIDInput) -> ExternResult<ActionHash> {
    // 1. Validate DID document structure
    validate_did_document(&input.document)?;

    // 2. Verify self-signature (proof)
    if let Some(ref proof) = input.document.proof {
        verify_did_proof(&input.document, proof)?;
    }

    // 3. Check caller is the controller
    let caller = agent_info()?.agent_initial_pubkey;
    if input.document.controller != caller {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Caller must be the DID controller".into()
        )));
    }

    // 4. Check DID doesn't already exist
    if let Some(_existing) = resolve_did(input.document.id.clone())? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("DID already exists: {}", input.document.id)
        )));
    }

    // 5. Create entry on DHT
    let action_hash = create_entry(&EntryTypes::DIDDocument(input.document.clone()))?;

    // 6. Create path for DID resolution (enables lookup by DID string)
    let path = Path::from(format!("did.{}", input.document.id)).typed(LinkTypes::DIDResolution)?;
    path.ensure()?;

    create_link(
        path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::DIDResolution,
        LinkTag::new("did_document")
    )?;

    debug!("DID created: {}", input.document.id);
    Ok(action_hash)
}

/// Resolve a DID to its document
#[hdk_extern]
pub fn resolve_did(did: String) -> ExternResult<Option<DIDDocument>> {
    // 1. Get path for DID
    let path = Path::from(format!("did.{}", did));

    // 2. Get links (should be one for active DID)
    let links = get_links(
        LinkQuery::try_new(
            path.path_entry_hash()?,
            LinkTypes::DIDResolution
        )?,
        GetStrategy::default()
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    // 3. Get most recent DID document (last link)
    let action_hash = links.last().expect("links verified non-empty above").target.clone().into_action_hash()
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

    let record = get(action_hash, GetOptions::default())?;

    if let Some(rec) = record {
        if let Some(entry) = rec.entry().as_option() {
            if let Entry::App(app_bytes) = entry {
                let did_doc = DIDDocument::try_from(
                    SerializedBytes::from(UnsafeBytes::from(app_bytes.bytes().to_vec()))
                ).map_err(|e| wasm_error!(e))?;
                return Ok(Some(did_doc));
            }
        }
    }

    Ok(None)
}

/// Update an existing DID document
#[hdk_extern]
pub fn update_did(input: UpdateDIDInput) -> ExternResult<ActionHash> {
    // 1. Resolve existing DID
    let existing = resolve_did(input.did.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("DID not found".into())))?;

    // 2. Verify caller is controller
    let caller = agent_info()?.agent_initial_pubkey;
    if existing.controller != caller {
        return Err(wasm_error!(WasmErrorInner::Guest("Not DID controller".into())));
    }

    // 3. Validate new document
    validate_did_document(&input.new_document)?;

    if let Some(ref proof) = input.new_document.proof {
        verify_did_proof(&input.new_document, proof)?;
    }

    // 4. Ensure DID remains the same
    if existing.id != input.new_document.id {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot change DID in update".into()
        )));
    }

    // 5. Update entry
    let action_hash = update_entry(
        input.original_action_hash,
        EntryTypes::DIDDocument(input.new_document.clone())
    )?;

    // 6. Update link to point to new action hash
    let path = Path::from(format!("did.{}", input.did));

    create_link(
        path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::DIDResolution,
        LinkTag::new("did_document")
    )?;

    debug!("DID updated: {}", input.did);
    Ok(action_hash)
}

/// Deactivate a DID (mark as inactive, don't delete)
#[hdk_extern]
pub fn deactivate_did(did: String) -> ExternResult<()> {
    // 1. Resolve existing DID
    let existing = resolve_did(did.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("DID not found".into())))?;

    // 2. Verify caller is controller
    let caller = agent_info()?.agent_initial_pubkey;
    if existing.controller != caller {
        return Err(wasm_error!(WasmErrorInner::Guest("Not DID controller".into())));
    }

    // 3. Remove links (makes DID unresolvable)
    let path = Path::from(format!("did.{}", did));
    let links = get_links(
        LinkQuery::try_new(
            path.path_entry_hash()?,
            LinkTypes::DIDResolution
        )?,
        GetStrategy::default()
    )?;

    for link in links {
        delete_link(link.create_link_hash, GetOptions::default())?;
    }

    debug!("DID deactivated: {}", did);
    Ok(())
}

//================================
// Helper Functions
//================================

fn validate_did_document(doc: &DIDDocument) -> ExternResult<()> {
    // 1. Check DID format
    if !doc.id.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format: must start with 'did:mycelix:'".into()
        )));
    }

    // 2. Check DID is not too long
    if doc.id.len() > 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID too long: max 100 characters".into()
        )));
    }

    // 3. Check verification methods exist
    if doc.verification_methods.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID must have at least one verification method".into()
        )));
    }

    // 4. Check authentication references valid keys
    for auth_ref in &doc.authentication {
        if !doc.verification_methods.iter().any(|vm| vm.id == *auth_ref) {
            return Err(wasm_error!(WasmErrorInner::Guest(
                format!("Authentication references non-existent key: {}", auth_ref)
            )));
        }
    }

    // 5. Validate each verification method
    for vm in &doc.verification_methods {
        validate_verification_method(vm, &doc.id)?;
    }

    Ok(())
}

fn validate_verification_method(vm: &VerificationMethod, did: &str) -> ExternResult<()> {
    // 1. Check ID format (should be DID#fragment)
    if !vm.id.starts_with(did) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Verification method ID must start with DID: {}", vm.id)
        )));
    }

    // 2. Check supported method type
    match vm.method_type.as_str() {
        "Ed25519VerificationKey2020" | "Ed25519VerificationKey2018" => {},
        _ => return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Unsupported verification method type: {}", vm.method_type)
        )))
    }

    // 3. Check controller is valid DID
    if !vm.controller.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid controller DID: {}", vm.controller)
        )));
    }

    // 4. Basic check on public key encoding
    if vm.public_key_multibase.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Public key cannot be empty".into()
        )));
    }

    Ok(())
}

/// Verify the Ed25519 signature in a DID proof
///
/// This function performs full cryptographic verification:
/// 1. Retrieves the verification method referenced by the proof
/// 2. Decodes the public key from multibase format
/// 3. Constructs the canonical message (DID document without proof)
/// 4. Verifies the Ed25519 signature
fn verify_did_proof(doc: &DIDDocument, proof: &DIDProof) -> ExternResult<()> {
    // 1. Get verification method
    let vm = doc.verification_methods.iter()
        .find(|v| v.id == proof.verification_method)
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Proof verification method not found".into()
        )))?;

    // 2. Check signature exists and has correct length
    if proof.signature.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proof signature cannot be empty".into()
        )));
    }

    if proof.signature.len() != 64 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid signature length: expected 64 bytes, got {}", proof.signature.len())
        )));
    }

    // 3. Check timestamp is reasonable
    if proof.created > sys_time()?.as_micros() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proof timestamp is in the future".into()
        )));
    }

    // 4. Decode public key from multibase
    let public_key = decode_multibase_public_key(&vm.public_key_multibase)?;

    // 5. Construct canonical message (DID doc without proof field)
    let canonical_message = create_canonical_message(doc)?;

    // 6. Parse the Ed25519 signature
    let sig_bytes: [u8; 64] = proof.signature.as_slice().try_into()
        .map_err(|_| wasm_error!(WasmErrorInner::Guest(
            "Invalid signature format".into()
        )))?;
    let signature = Signature::from_bytes(&sig_bytes);

    // 7. Verify the signature
    public_key.verify(canonical_message.as_bytes(), &signature)
        .map_err(|_| wasm_error!(WasmErrorInner::Guest(
            "Ed25519 signature verification failed".into()
        )))?;

    debug!("DID proof verified successfully for: {}", doc.id);
    Ok(())
}

/// Decode a public key from multibase format (base58btc prefixed with 'z')
///
/// Supports Ed25519 public keys in the following formats:
/// - z-prefixed base58btc encoding (multibase format)
/// - Raw 32-byte key (for testing)
fn decode_multibase_public_key(multibase: &str) -> ExternResult<VerifyingKey> {
    // Multibase base58btc is prefixed with 'z'
    if !multibase.starts_with('z') {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid multibase prefix: expected 'z' (base58btc), got '{}'",
                multibase.chars().next().unwrap_or('?'))
        )));
    }

    // Decode base58btc (skip the 'z' prefix)
    let decoded = bs58::decode(&multibase[1..])
        .into_vec()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(
            format!("Failed to decode base58btc: {}", e)
        )))?;

    // For Ed25519 keys in multicodec format:
    // First 2 bytes are multicodec prefix (0xed 0x01 for Ed25519)
    // Followed by 32 bytes of public key
    let key_bytes = if decoded.len() == 34 && decoded[0] == 0xed && decoded[1] == 0x01 {
        // Multicodec format: skip 2-byte prefix
        &decoded[2..]
    } else if decoded.len() == 32 {
        // Raw 32-byte key
        &decoded
    } else {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Invalid public key length: expected 32 or 34 bytes, got {}", decoded.len())
        )));
    };

    let key_array: [u8; 32] = key_bytes.try_into()
        .map_err(|_| wasm_error!(WasmErrorInner::Guest(
            "Failed to convert public key bytes to array".into()
        )))?;

    VerifyingKey::from_bytes(&key_array)
        .map_err(|_| wasm_error!(WasmErrorInner::Guest(
            "Invalid Ed25519 public key".into()
        )))
}

/// Create canonical message for signature verification
///
/// This creates a deterministic JSON representation of the DID document
/// without the proof field, which is what was originally signed.
fn create_canonical_message(doc: &DIDDocument) -> ExternResult<String> {
    // Create a copy without the proof field for signing
    #[derive(Serialize)]
    struct DIDDocumentForSigning<'a> {
        pub id: &'a str,
        pub controller: &'a AgentPubKey,
        pub verification_methods: &'a Vec<VerificationMethod>,
        pub authentication: &'a Vec<String>,
        pub created: i64,
        pub updated: i64,
    }

    let doc_for_signing = DIDDocumentForSigning {
        id: &doc.id,
        controller: &doc.controller,
        verification_methods: &doc.verification_methods,
        authentication: &doc.authentication,
        created: doc.created,
        updated: doc.updated,
    };

    serde_json::to_string(&doc_for_signing)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(
            format!("Failed to serialize DID document: {}", e)
        )))
}

//================================
// Validation Callback
//================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(StoreEntry { entry, .. }) => {
            match entry {
                Entry::App(bytes) => {
                    // Try to deserialize as DIDDocument
                    if let Ok(did_doc) = DIDDocument::try_from(
                        SerializedBytes::from(bytes.to_owned())
                    ).map_err(|e| wasm_error!(e)) {
                        // Validate DID document
                        if let Err(e) = validate_did_document(&did_doc) {
                            return Ok(ValidateCallbackResult::Invalid(
                                format!("Invalid DID document: {:?}", e)
                            ));
                        }

                        // Verify proof if present
                        if let Some(ref proof) = did_doc.proof {
                            if let Err(e) = verify_did_proof(&did_doc, proof) {
                                return Ok(ValidateCallbackResult::Invalid(
                                    format!("Invalid DID proof: {:?}", e)
                                ));
                            }
                        }

                        return Ok(ValidateCallbackResult::Valid);
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
    Ok(ValidateCallbackResult::Valid)
}

//================================
// Input Types
//================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateDIDInput {
    pub document: DIDDocument,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateDIDInput {
    pub did: String,
    pub original_action_hash: ActionHash,
    pub new_document: DIDDocument,
}
