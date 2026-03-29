// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

// ============================================================================
// Tier 2 Mirror Types
// ============================================================================

// --- Clinical Trials ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TrialEnrollmentInput {
    trial_id: String,
    patient_id: String,
    consent_given: bool,
    enrollment_date: i64,
}

// --- FHIR Mapping ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct FhirMappingInput {
    resource_type: String,
    source_id: String,
    fhir_json: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct FhirMappingResult {
    mapped: bool,
    fhir_id: Option<String>,
}

// --- Clinical Decision Support ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CdsRecommendationInput {
    patient_id: String,
    condition: String,
    current_medications: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CdsRecommendation {
    recommendation: String,
    severity: String,
    evidence_level: String,
}

// --- Telehealth ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TelehealthSessionInput {
    patient_id: String,
    provider_id: String,
    session_type: String,
    scheduled_at: i64,
}

// --- Nutrition ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct NutritionPlanInput {
    patient_id: String,
    dietary_restrictions: Vec<String>,
    caloric_target: u32,
    notes: String,
}

// ============================================================================
// Tests — Tier 2: Clinical Trials
// ============================================================================

/// Test: Enroll a patient in a clinical trial.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled health WASM + conductor"]
async fn test_trials_enrollment() {
    let dna_path = DnaPaths::health();
    if !dna_path.exists() {
        eprintln!("Skipping: health DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "health", 1).await;
    let alice = &agents[0];

    let enrollment = TrialEnrollmentInput {
        trial_id: "TRIAL-2026-001".into(),
        patient_id: "PAT-001".into(),
        consent_given: true,
        enrollment_date: 1711065600,
    };

    let created: Record = alice
        .call_zome_fn("trials", "enroll_patient", enrollment)
        .await;

    let action_hash = created.action_hashed().hash.clone();
    let retrieved: Record = alice
        .call_zome_fn("trials", "get_enrollment", action_hash)
        .await;

    assert_eq!(
        retrieved.action_hashed().hash,
        created.action_hashed().hash,
        "Retrieved enrollment should match created enrollment"
    );
}

// ============================================================================
// Tests — Tier 2: FHIR Mapping
// ============================================================================

/// Test: Map a patient record to FHIR format and retrieve.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled health WASM + conductor"]
async fn test_fhir_mapping_roundtrip() {
    let dna_path = DnaPaths::health();
    if !dna_path.exists() {
        eprintln!("Skipping: health DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "health", 1).await;
    let alice = &agents[0];

    let mapping = FhirMappingInput {
        resource_type: "Patient".into(),
        source_id: "PAT-001".into(),
        fhir_json: r#"{"resourceType":"Patient","name":[{"given":["Alice"],"family":"Test"}]}"#
            .into(),
    };

    let result: FhirMappingResult = alice
        .call_zome_fn("fhir_mapping", "create_mapping", mapping)
        .await;

    assert!(result.mapped, "FHIR mapping should succeed");
    assert!(
        result.fhir_id.is_some(),
        "FHIR mapping should return a FHIR ID"
    );
}

// ============================================================================
// Tests — Tier 2: Clinical Decision Support
// ============================================================================

/// Test: Get a CDS recommendation for a patient condition.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled health WASM + conductor"]
async fn test_cds_recommendation() {
    let dna_path = DnaPaths::health();
    if !dna_path.exists() {
        eprintln!("Skipping: health DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "health", 1).await;
    let alice = &agents[0];

    let input = CdsRecommendationInput {
        patient_id: "PAT-001".into(),
        condition: "hypertension".into(),
        current_medications: vec!["lisinopril".into()],
    };

    let rec: CdsRecommendation = alice
        .call_zome_fn("cds", "get_recommendation", input)
        .await;

    assert!(
        !rec.recommendation.is_empty(),
        "CDS should return a non-empty recommendation"
    );
    assert!(
        !rec.evidence_level.is_empty(),
        "CDS should specify evidence level"
    );
}

// ============================================================================
// Tests — Tier 2: Telehealth
// ============================================================================

/// Test: Create and retrieve a telehealth session.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled health WASM + conductor"]
async fn test_telehealth_session() {
    let dna_path = DnaPaths::health();
    if !dna_path.exists() {
        eprintln!("Skipping: health DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "health", 1).await;
    let alice = &agents[0];

    let session = TelehealthSessionInput {
        patient_id: "PAT-001".into(),
        provider_id: "PROV-001".into(),
        session_type: "video".into(),
        scheduled_at: 1711152000,
    };

    let created: Record = alice
        .call_zome_fn("telehealth", "create_session", session)
        .await;

    let action_hash = created.action_hashed().hash.clone();
    let retrieved: Record = alice
        .call_zome_fn("telehealth", "get_session", action_hash)
        .await;

    assert_eq!(
        retrieved.action_hashed().hash,
        created.action_hashed().hash,
        "Retrieved telehealth session should match created session"
    );
}

// ============================================================================
// Tests — Tier 2: Nutrition
// ============================================================================

/// Test: Create and retrieve a nutrition plan.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled health WASM + conductor"]
async fn test_nutrition_plan() {
    let dna_path = DnaPaths::health();
    if !dna_path.exists() {
        eprintln!("Skipping: health DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "health", 1).await;
    let alice = &agents[0];

    let plan = NutritionPlanInput {
        patient_id: "PAT-001".into(),
        dietary_restrictions: vec!["gluten-free".into(), "low-sodium".into()],
        caloric_target: 2000,
        notes: "Post-surgical recovery diet".into(),
    };

    let created: Record = alice
        .call_zome_fn("nutrition", "create_plan", plan)
        .await;

    let action_hash = created.action_hashed().hash.clone();
    let retrieved: Record = alice
        .call_zome_fn("nutrition", "get_plan", action_hash)
        .await;

    assert_eq!(
        retrieved.action_hashed().hash,
        created.action_hashed().hash,
        "Retrieved nutrition plan should match created plan"
    );
}
