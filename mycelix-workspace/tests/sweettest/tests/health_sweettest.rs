//! Health Cluster Sweettest Integration Tests (Phase 1 — Basic CRUD)
//!
//! Tests the Tier 1 MVP zomes of the mycelix-health cluster:
//! - Patient CRUD and search
//! - Provider registration and retrieval
//! - Health records (encounters)
//! - Prescriptions
//! - Consent grant, check, and revocation
//! - Health bridge health check
//!
//! ## Prerequisites
//!
//! ```bash
//! cd mycelix-health && cargo build --release --target wasm32-unknown-unknown
//! hc dna pack mycelix-health/dna/
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo test --release -p mycelix-sweettest --test health_sweettest -- --ignored
//! ```

mod harness;

use harness::*;
use holochain::prelude::*;
use holochain::sweettest::*;
use serial_test::serial;
use std::path::PathBuf;

// ============================================================================
// Mirror types — must match actual zome integrity/coordinator struct layout
// ============================================================================

// --- Patient ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PatientInput {
    given_name: String,
    family_name: String,
    date_of_birth: String,
    gender: String,
    mrn: String,
}

// --- Provider ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ProviderInput {
    name: String,
    npi: String,
    specialty: String,
    credentials: Vec<String>,
}

// --- Encounter (Health Record) ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct EncounterInput {
    patient_id: String,
    provider_id: String,
    encounter_type: String,
    notes: String,
    date: i64,
}

// --- Prescription ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PrescriptionInput {
    patient_id: String,
    provider_id: String,
    medication: String,
    dosage: String,
    frequency: String,
}

// --- Consent ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ConsentInput {
    patient_id: String,
    grantee_did: String,
    record_types: Vec<String>,
    expires_at: Option<i64>,
}

// --- Authorization Check ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct AuthorizationCheckInput {
    patient_id: String,
    grantee_did: String,
    record_type: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct AuthorizationResult {
    authorized: bool,
}

// --- Revoke Consent ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RevokeConsentInput {
    patient_id: String,
    grantee_did: String,
}

// --- Bridge Health ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct HealthBridgeHealth {
    healthy: bool,
    domains: Vec<String>,
}

// ============================================================================
// Tests — Patient
// ============================================================================

/// Test: Create a patient with demographics and retrieve by action hash.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled health WASM + conductor"]
async fn test_patient_create_and_get() {
    let dna_path = DnaPaths::health();
    if !dna_path.exists() {
        eprintln!("Skipping: health DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "health", 1).await;
    let alice = &agents[0];

    let patient = PatientInput {
        given_name: "Alice".into(),
        family_name: "Wonderland".into(),
        date_of_birth: "1990-03-15".into(),
        gender: "female".into(),
        mrn: "MRN-001".into(),
    };

    let created: Record = alice
        .call_zome_fn("patient", "create_patient", patient)
        .await;

    let action_hash = created.action_hashed().hash.clone();

    let retrieved: Record = alice
        .call_zome_fn("patient", "get_patient", action_hash)
        .await;

    assert_eq!(
        retrieved.action_hashed().hash,
        created.action_hashed().hash,
        "Retrieved patient should match created patient action hash"
    );
}

/// Test: Create two patients and search by name.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled health WASM + conductor"]
async fn test_patient_search() {
    let dna_path = DnaPaths::health();
    if !dna_path.exists() {
        eprintln!("Skipping: health DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "health", 1).await;
    let alice = &agents[0];

    let patient_a = PatientInput {
        given_name: "Alice".into(),
        family_name: "Wonderland".into(),
        date_of_birth: "1990-03-15".into(),
        gender: "female".into(),
        mrn: "MRN-002".into(),
    };

    let patient_b = PatientInput {
        given_name: "Bob".into(),
        family_name: "Builder".into(),
        date_of_birth: "1985-07-22".into(),
        gender: "male".into(),
        mrn: "MRN-003".into(),
    };

    let _: Record = alice
        .call_zome_fn("patient", "create_patient", patient_a)
        .await;
    let _: Record = alice
        .call_zome_fn("patient", "create_patient", patient_b)
        .await;

    let results: Vec<Record> = alice
        .call_zome_fn("patient", "search_patients", "Alice".to_string())
        .await;

    assert!(
        !results.is_empty(),
        "Search for 'Alice' should return at least one result"
    );
}

// ============================================================================
// Tests — Provider
// ============================================================================

/// Test: Register a provider with credentials and retrieve.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled health WASM + conductor"]
async fn test_provider_register_and_get() {
    let dna_path = DnaPaths::health();
    if !dna_path.exists() {
        eprintln!("Skipping: health DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "health", 1).await;
    let alice = &agents[0];

    let provider = ProviderInput {
        name: "Dr. Carol Smith".into(),
        npi: "1234567890".into(),
        specialty: "Internal Medicine".into(),
        credentials: vec!["MD".into(), "Board Certified".into()],
    };

    let created: Record = alice
        .call_zome_fn("provider", "register_provider", provider)
        .await;

    let action_hash = created.action_hashed().hash.clone();

    let retrieved: Record = alice
        .call_zome_fn("provider", "get_provider", action_hash)
        .await;

    assert_eq!(
        retrieved.action_hashed().hash,
        created.action_hashed().hash,
        "Retrieved provider should match created provider action hash"
    );
}

// ============================================================================
// Tests — Health Records (Encounters)
// ============================================================================

/// Test: Create an encounter record.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled health WASM + conductor"]
async fn test_health_record_create() {
    let dna_path = DnaPaths::health();
    if !dna_path.exists() {
        eprintln!("Skipping: health DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "health", 1).await;
    let alice = &agents[0];

    let encounter = EncounterInput {
        patient_id: "patient-001".into(),
        provider_id: "provider-001".into(),
        encounter_type: "office_visit".into(),
        notes: "Annual physical examination. All vitals normal.".into(),
        date: 1709856000, // 2024-03-08T00:00:00Z
    };

    let created: Record = alice
        .call_zome_fn("records", "create_encounter", encounter)
        .await;

    let action_hash = created.action_hashed().hash.clone();

    let retrieved: Record = alice
        .call_zome_fn("records", "get_encounter", action_hash)
        .await;

    assert_eq!(
        retrieved.action_hashed().hash,
        created.action_hashed().hash,
        "Retrieved encounter should match created encounter action hash"
    );
}

// ============================================================================
// Tests — Prescriptions
// ============================================================================

/// Test: Create a prescription.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled health WASM + conductor"]
async fn test_prescription_create() {
    let dna_path = DnaPaths::health();
    if !dna_path.exists() {
        eprintln!("Skipping: health DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "health", 1).await;
    let alice = &agents[0];

    let prescription = PrescriptionInput {
        patient_id: "patient-001".into(),
        provider_id: "provider-001".into(),
        medication: "Lisinopril".into(),
        dosage: "10mg".into(),
        frequency: "once daily".into(),
    };

    let created: Record = alice
        .call_zome_fn("prescriptions", "create_prescription", prescription)
        .await;

    let action_hash = created.action_hashed().hash.clone();

    let retrieved: Record = alice
        .call_zome_fn("prescriptions", "get_prescription", action_hash)
        .await;

    assert_eq!(
        retrieved.action_hashed().hash,
        created.action_hashed().hash,
        "Retrieved prescription should match created prescription action hash"
    );
}

// ============================================================================
// Tests — Consent
// ============================================================================

/// Test: Grant consent, then verify authorization succeeds.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled health WASM + conductor"]
async fn test_consent_grant_and_check() {
    let dna_path = DnaPaths::health();
    if !dna_path.exists() {
        eprintln!("Skipping: health DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "health", 1).await;
    let alice = &agents[0];

    let consent = ConsentInput {
        patient_id: "patient-001".into(),
        grantee_did: "did:mycelix:provider-carol".into(),
        record_types: vec!["encounter".into(), "lab_result".into()],
        expires_at: None,
    };

    let _created: Record = alice
        .call_zome_fn("consent", "create_consent", consent)
        .await;

    let auth_check = AuthorizationCheckInput {
        patient_id: "patient-001".into(),
        grantee_did: "did:mycelix:provider-carol".into(),
        record_type: "encounter".into(),
    };

    let auth_result: AuthorizationResult = alice
        .call_zome_fn("consent", "check_authorization", auth_check)
        .await;

    assert!(
        auth_result.authorized,
        "Authorization should succeed after consent is granted"
    );
}

/// Test: Grant consent, then revoke, then verify authorization fails.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled health WASM + conductor"]
async fn test_consent_revoke() {
    let dna_path = DnaPaths::health();
    if !dna_path.exists() {
        eprintln!("Skipping: health DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "health", 1).await;
    let alice = &agents[0];

    // Grant consent
    let consent = ConsentInput {
        patient_id: "patient-002".into(),
        grantee_did: "did:mycelix:provider-dave".into(),
        record_types: vec!["prescription".into()],
        expires_at: None,
    };

    let _created: Record = alice
        .call_zome_fn("consent", "create_consent", consent)
        .await;

    // Revoke consent
    let revoke = RevokeConsentInput {
        patient_id: "patient-002".into(),
        grantee_did: "did:mycelix:provider-dave".into(),
    };

    let _: () = alice
        .call_zome_fn("consent", "revoke_consent", revoke)
        .await;

    // Check authorization — should fail after revocation
    let auth_check = AuthorizationCheckInput {
        patient_id: "patient-002".into(),
        grantee_did: "did:mycelix:provider-dave".into(),
        record_type: "prescription".into(),
    };

    let auth_result: AuthorizationResult = alice
        .call_zome_fn("consent", "check_authorization", auth_check)
        .await;

    assert!(
        !auth_result.authorized,
        "Authorization should fail after consent is revoked"
    );
}

// ============================================================================
// Tests — Bridge Health Check
// ============================================================================

/// Test: Health bridge reports healthy status with domain list.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled health WASM + conductor"]
async fn test_bridge_health_check() {
    let dna_path = DnaPaths::health();
    if !dna_path.exists() {
        eprintln!("Skipping: health DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "health", 1).await;
    let alice = &agents[0];

    let health: HealthBridgeHealth = alice
        .call_zome_fn("bridge", "health_check", ())
        .await;

    assert!(
        health.healthy,
        "Health bridge should report healthy status"
    );

    assert!(
        !health.domains.is_empty(),
        "Health bridge should report at least one domain"
    );
}
