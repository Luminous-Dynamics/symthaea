//! Sweettest integration tests for PQC (Post-Quantum Cryptography) flows
//! in the Mycelix Identity hApp.
//!
//! These tests verify that algorithm-agnostic DID operations, hybrid-signed
//! credentials, and ML-KEM key agreement work end-to-end through Holochain
//! conductors.
//!
//! Requires: DNA bundle pre-built
//!   cd mycelix-identity && cargo build --release --target wasm32-unknown-unknown
//!   hc dna pack dna/ -o dna/mycelix_identity_dna.dna
//!
//! Run: cargo test --release -- --ignored identity_pqc --test-threads=1

extern crate holochain_serialized_bytes;

use holochain::prelude::*;
use mycelix_crypto::algorithm::AlgorithmId;
use mycelix_crypto::pqc::hybrid::HybridSigner;
use mycelix_crypto::pqc::ml_kem::MlKem768KeyPair;
use mycelix_crypto::traits::Signer;
use serial_test::serial;

mod harness;
use harness::*;

// ---------------------------------------------------------------------------
// Mirror types (sweettest can't import WASM crate types)
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct VerificationMethod {
    id: String,
    #[serde(rename = "type", alias = "type_")]
    type_: String,
    controller: String,
    #[serde(rename = "publicKeyMultibase", alias = "public_key_multibase")]
    public_key_multibase: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    algorithm: Option<u16>,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct ServiceEndpoint {
    id: String,
    #[serde(rename = "type", alias = "type_")]
    type_: String,
    #[serde(rename = "serviceEndpoint", alias = "service_endpoint")]
    service_endpoint: String,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug)]
struct DidDocument {
    id: String,
    controller: AgentPubKey,
    #[serde(rename = "verificationMethod", alias = "verification_method")]
    verification_method: Vec<VerificationMethod>,
    authentication: Vec<String>,
    #[serde(rename = "keyAgreement", alias = "key_agreement", default)]
    key_agreement: Vec<String>,
    service: Vec<ServiceEndpoint>,
    created: Timestamp,
    updated: Timestamp,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug)]
struct CredentialProof {
    #[serde(rename = "type")]
    proof_type: String,
    created: String,
    #[serde(rename = "verificationMethod")]
    verification_method: String,
    #[serde(rename = "proofPurpose")]
    proof_purpose: String,
    #[serde(rename = "proofValue")]
    proof_value: String,
    #[serde(default)]
    cryptosuite: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    algorithm: Option<u16>,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug)]
struct CredentialSubject {
    id: String,
    #[serde(flatten)]
    claims: serde_json::Value,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug)]
#[serde(untagged)]
enum CredentialIssuer {
    Did(String),
    Object {
        id: String,
        #[serde(default)]
        name: Option<String>,
    },
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug)]
struct VerifiableCredential {
    #[serde(rename = "@context")]
    context: Vec<String>,
    id: String,
    #[serde(rename = "type")]
    credential_type: Vec<String>,
    issuer: CredentialIssuer,
    #[serde(rename = "validFrom")]
    valid_from: String,
    #[serde(rename = "validUntil")]
    valid_until: Option<String>,
    #[serde(rename = "credentialSubject")]
    credential_subject: CredentialSubject,
    #[serde(rename = "credentialSchema")]
    credential_schema: Option<serde_json::Value>,
    #[serde(rename = "credentialStatus")]
    credential_status: Option<serde_json::Value>,
    proof: CredentialProof,
    mycelix_schema_id: String,
    mycelix_created: Timestamp,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug)]
struct IssueCredentialWithProofInput {
    credential: VerifiableCredential,
}

// ---------------------------------------------------------------------------
// Test 1: create_did still works with default algorithm
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_create_did_with_default_algorithm() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    // Create DID
    let did_record: Record = alice
        .call_zome_fn("did_registry", "create_did", ())
        .await;

    let action_hash = did_record.action_hashed().hash.clone();
    assert!(
        !action_hash.as_ref().is_empty(),
        "DID record should have valid hash"
    );

    // Retrieve and check key_agreement is empty by default
    let retrieved: Option<Record> = alice
        .call_zome_fn(
            "did_registry",
            "get_did_document",
            alice.agent_pubkey.clone(),
        )
        .await;

    assert!(retrieved.is_some(), "Should retrieve Alice's DID document");

    // Parse the DID document to check key_agreement field
    let record = retrieved.unwrap();
    let did_doc: DidDocument = record
        .entry()
        .to_app_option()
        .expect("Failed to deserialize DID")
        .expect("DID entry is None");

    assert!(
        did_doc.key_agreement.is_empty(),
        "Default DID should have no key_agreement entries"
    );
    assert!(
        !did_doc.verification_method.is_empty(),
        "DID should have at least one verification method"
    );
}

// ---------------------------------------------------------------------------
// Test 2: add ML-KEM-768 key agreement
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_add_key_agreement_kem() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    // Create DID first
    let _: Record = alice
        .call_zome_fn("did_registry", "create_did", ())
        .await;

    // Generate ML-KEM-768 keypair off-chain
    let kem = MlKem768KeyPair::generate();
    let kem_pk = kem.public_key();
    let kem_multibase = kem_pk.to_multibase();
    let did_string = format!("did:mycelix:{}", alice.agent_pubkey);

    let method = VerificationMethod {
        id: "#kem-1".to_string(),
        type_: AlgorithmId::MlKem768
            .did_verification_method_type()
            .to_string(),
        controller: did_string,
        public_key_multibase: kem_multibase,
        algorithm: Some(AlgorithmId::MlKem768.as_u16()),
    };

    // Add key agreement
    let _: Record = alice
        .call_zome_fn("did_registry", "add_key_agreement", method)
        .await;

    // Verify key_agreement is populated
    let retrieved: Option<Record> = alice
        .call_zome_fn(
            "did_registry",
            "get_did_document",
            alice.agent_pubkey.clone(),
        )
        .await;

    let record = retrieved.expect("DID document should exist");
    let did_doc: DidDocument = record
        .entry()
        .to_app_option()
        .expect("Failed to deserialize DID")
        .expect("DID entry is None");

    assert!(
        did_doc.key_agreement.contains(&"#kem-1".to_string()),
        "key_agreement should contain #kem-1, got: {:?}",
        did_doc.key_agreement
    );

    // Verify the KEM verification method is present
    let kem_method = did_doc
        .verification_method
        .iter()
        .find(|m| m.id == "#kem-1");
    assert!(kem_method.is_some(), "KEM verification method should exist");
    assert_eq!(
        kem_method.unwrap().algorithm,
        Some(AlgorithmId::MlKem768.as_u16())
    );
}

// ---------------------------------------------------------------------------
// Test 3: issue credential with hybrid proof
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_issue_credential_with_hybrid_proof() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    // Create DID
    let _: Record = alice
        .call_zome_fn("did_registry", "create_did", ())
        .await;

    let did_string = format!("did:mycelix:{}", alice.agent_pubkey);

    // Generate hybrid keypair off-chain and sign credential
    let signer = HybridSigner::generate();
    let credential_json = serde_json::json!({
        "degree": "Computer Science",
        "institution": "Test University"
    });
    let canonical = serde_json::to_vec(&credential_json).unwrap();
    let sig = signer.sign(&canonical).unwrap();
    let alg = AlgorithmId::HybridEd25519MlDsa65;

    // Build the full VerifiableCredential with pre-signed proof
    let vc = VerifiableCredential {
        context: vec![
            "https://www.w3.org/ns/credentials/v2".into(),
            "https://w3id.org/vc/status-list/2021/v1".into(),
        ],
        id: "urn:uuid:test-hybrid-001".into(),
        credential_type: vec!["VerifiableCredential".into(), "TestCredential".into()],
        issuer: CredentialIssuer::Did(did_string.clone()),
        valid_from: "2026-02-12T00:00:00Z".into(),
        valid_until: None,
        credential_subject: CredentialSubject {
            id: "did:test:subject-001".into(),
            claims: credential_json,
        },
        credential_schema: None,
        credential_status: None,
        proof: CredentialProof {
            proof_type: "DataIntegrityProof".into(),
            created: "2026-02-12T00:00:00Z".into(),
            verification_method: format!("{}#keys-1", did_string),
            proof_purpose: "assertionMethod".into(),
            proof_value: sig.to_multibase(),
            cryptosuite: Some(alg.cryptosuite().into()),
            algorithm: Some(alg.as_u16()),
        },
        mycelix_schema_id: "mycelix:schema:test:v1".into(),
        mycelix_created: Timestamp::now(),
    };

    let input = IssueCredentialWithProofInput { credential: vc };
    let record: Record = alice
        .call_zome_fn("verifiable_credential", "issue_credential_with_proof", input)
        .await;

    assert!(
        !record.action_hashed().hash.as_ref().is_empty(),
        "Hybrid-signed credential should be stored successfully"
    );

    // Retrieve the credential by ID and verify the proof algorithm
    let retrieved: Option<Record> = alice
        .call_zome_fn(
            "verifiable_credential",
            "get_credential",
            "urn:uuid:test-hybrid-001".to_string(),
        )
        .await;
    assert!(retrieved.is_some(), "Should retrieve the issued credential");
    let cred_record = retrieved.unwrap();
    let stored_vc: VerifiableCredential = cred_record
        .entry()
        .to_app_option()
        .expect("Failed to deserialize credential")
        .expect("Credential entry is None");
    assert_eq!(
        stored_vc.proof.algorithm,
        Some(AlgorithmId::HybridEd25519MlDsa65.as_u16()),
        "Stored credential proof should have HybridEd25519MlDsa65 algorithm tag"
    );
    assert_eq!(stored_vc.proof.proof_type, "DataIntegrityProof");
    assert!(
        stored_vc.proof.cryptosuite.is_some(),
        "Hybrid proof should specify a cryptosuite"
    );
}

// ---------------------------------------------------------------------------
// Test 4: hybrid credential cross-agent DHT resolution
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_hybrid_credential_cross_agent() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    // Alice creates DID and issues hybrid-signed credential
    let _: Record = alice
        .call_zome_fn("did_registry", "create_did", ())
        .await;

    let did_string = format!("did:mycelix:{}", alice.agent_pubkey);
    let signer = HybridSigner::generate();
    let claims = serde_json::json!({"skill": "quantum computing"});
    let canonical = serde_json::to_vec(&claims).unwrap();
    let sig = signer.sign(&canonical).unwrap();
    let alg = AlgorithmId::HybridEd25519MlDsa65;

    let vc = VerifiableCredential {
        context: vec!["https://www.w3.org/ns/credentials/v2".into()],
        id: "urn:uuid:cross-agent-hybrid-001".into(),
        credential_type: vec!["VerifiableCredential".into()],
        issuer: CredentialIssuer::Did(did_string.clone()),
        valid_from: "2026-02-12T00:00:00Z".into(),
        valid_until: None,
        credential_subject: CredentialSubject {
            id: format!("did:mycelix:{}", bob.agent_pubkey),
            claims,
        },
        credential_schema: None,
        credential_status: None,
        proof: CredentialProof {
            proof_type: "DataIntegrityProof".into(),
            created: "2026-02-12T00:00:00Z".into(),
            verification_method: format!("{}#keys-1", did_string),
            proof_purpose: "assertionMethod".into(),
            proof_value: sig.to_multibase(),
            cryptosuite: Some(alg.cryptosuite().into()),
            algorithm: Some(alg.as_u16()),
        },
        mycelix_schema_id: "".into(),
        mycelix_created: Timestamp::now(),
    };

    let input = IssueCredentialWithProofInput { credential: vc };
    let cred_record: Record = alice
        .call_zome_fn("verifiable_credential", "issue_credential_with_proof", input)
        .await;

    let cred_hash = cred_record.action_hashed().hash.clone();
    assert!(!cred_hash.as_ref().is_empty());

    wait_for_dht_sync().await;

    // Bob retrieves the credential via DHT using subject link query
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);
    let retrieved: Vec<Record> = bob
        .call_zome_fn(
            "verifiable_credential",
            "get_credentials_for_subject",
            bob_did,
        )
        .await;

    assert!(
        !retrieved.is_empty(),
        "Bob should retrieve Alice's hybrid-signed credential via DHT subject link"
    );

    // Verify the first retrieved credential has a valid hash and hybrid proof
    let first = &retrieved[0];
    assert!(
        !first.action_hashed().hash.as_ref().is_empty(),
        "Retrieved credential should have a valid action hash"
    );
    let cross_vc: VerifiableCredential = first
        .entry()
        .to_app_option()
        .expect("Failed to deserialize credential")
        .expect("Credential entry is None");
    assert_eq!(
        cross_vc.proof.algorithm,
        Some(AlgorithmId::HybridEd25519MlDsa65.as_u16()),
        "Cross-agent credential should preserve HybridEd25519MlDsa65 algorithm"
    );
}

// ---------------------------------------------------------------------------
// Test 5: add PQC verification method with algorithm tag
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_add_verification_method_with_pqc_algorithm() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    let _: Record = alice
        .call_zome_fn("did_registry", "create_did", ())
        .await;

    let did_string = format!("did:mycelix:{}", alice.agent_pubkey);

    // Generate an ML-DSA-65 keypair and add as verification method
    let signer = mycelix_crypto::pqc::dilithium::MlDsa65Signer::generate();
    let pk = signer.public_key();

    let method = VerificationMethod {
        id: "#pqc-1".to_string(),
        type_: AlgorithmId::MlDsa65
            .did_verification_method_type()
            .to_string(),
        controller: did_string,
        public_key_multibase: pk.to_multibase(),
        algorithm: Some(AlgorithmId::MlDsa65.as_u16()),
    };

    let _: Record = alice
        .call_zome_fn("did_registry", "add_verification_method", method)
        .await;

    // Verify
    let retrieved: Option<Record> = alice
        .call_zome_fn(
            "did_registry",
            "get_did_document",
            alice.agent_pubkey.clone(),
        )
        .await;

    let record = retrieved.expect("DID should exist");
    let did_doc: DidDocument = record
        .entry()
        .to_app_option()
        .expect("Failed to deserialize")
        .expect("DID is None");

    let pqc_method = did_doc
        .verification_method
        .iter()
        .find(|m| m.id == "#pqc-1");
    assert!(pqc_method.is_some(), "PQC verification method should exist");
    let m = pqc_method.unwrap();
    assert_eq!(m.algorithm, Some(AlgorithmId::MlDsa65.as_u16()));
    assert_eq!(m.type_, "MlDsa65VerificationKey2024");
    assert!(
        m.public_key_multibase.starts_with('z'),
        "Key should be multibase encoded"
    );
}

// ---------------------------------------------------------------------------
// Test 6: reject unsupported algorithm for key agreement
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore]
async fn test_reject_invalid_algorithm_for_key_agreement() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    let _: Record = alice
        .call_zome_fn("did_registry", "create_did", ())
        .await;

    let did_string = format!("did:mycelix:{}", alice.agent_pubkey);

    // Try adding a signing key (Ed25519) as key agreement — should fail
    // because add_key_agreement validates that the key is a KEM algorithm
    let ed_signer = mycelix_crypto::pqc::ed25519_native::Ed25519Signer::generate();
    let ed_pk = ed_signer.public_key();

    let method = VerificationMethod {
        id: "#bad-kem".to_string(),
        type_: "Ed25519VerificationKey2020".to_string(),
        controller: did_string,
        public_key_multibase: ed_pk.to_multibase(),
        algorithm: Some(AlgorithmId::Ed25519.as_u16()),
    };

    let result: Result<Record, _> = alice
        .call_zome_fn_fallible("did_registry", "add_key_agreement", method)
        .await;

    assert!(
        result.is_err(),
        "Adding a signing key as key agreement should fail"
    );
}
