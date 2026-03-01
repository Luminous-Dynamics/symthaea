//! Sweettest integration tests for Mycelix Attribution Registry.
//!
//! All tests require a running Holochain conductor and a pre-built DNA bundle.
//! Build the DNA first: `cd mycelix-attribution && hc dna pack dna/`
//! Run tests: `cd tests && cargo test -- --ignored`

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ── Mirror Types ─────────────────────────────────────────────────────
// These duplicate zome entry types to avoid importing zome crates
// (which causes __num_entry_types symbol collisions in WASM tests).

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Anchor(pub String);

// -- Registry --

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DependencyEcosystem {
    RustCrate,
    NpmPackage,
    PythonPackage,
    NixFlake,
    Other,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DependencyIdentity {
    pub id: String,
    pub name: String,
    pub ecosystem: DependencyEcosystem,
    pub maintainer_did: String,
    pub repository_url: Option<String>,
    pub license: Option<String>,
    pub description: String,
    pub version: Option<String>,
    pub registered_at: Timestamp,
    pub verified: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateDependencyInput {
    pub original_action_hash: ActionHash,
    pub dependency: DependencyIdentity,
}

// -- Usage --

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum UsageType {
    DirectDependency,
    Transitive,
    InternalTooling,
    Production,
    Research,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum UsageScale {
    Small,
    Medium,
    Large,
    Enterprise,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct UsageReceipt {
    pub id: String,
    pub dependency_id: String,
    pub user_did: String,
    pub organization: Option<String>,
    pub usage_type: UsageType,
    pub scale: Option<UsageScale>,
    pub version_range: Option<String>,
    pub context: Option<String>,
    pub attested_at: Timestamp,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct UsageAttestation {
    pub id: String,
    pub dependency_id: String,
    pub user_did: String,
    pub witness_commitment: Vec<u8>,
    pub proof_bytes: Vec<u8>,
    pub verified: bool,
    pub generated_at: Timestamp,
    pub expires_at: Option<Timestamp>,
    pub verifier_pubkey: Option<Vec<u8>>,
    pub verifier_signature: Option<Vec<u8>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerifyAttestationInput {
    pub original_action_hash: ActionHash,
    pub verifier_pubkey: Vec<u8>,
    pub verifier_signature: Vec<u8>,
}

// -- Reciprocity --

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PledgeType {
    Financial,
    Compute,
    Bandwidth,
    DeveloperTime,
    QA,
    Documentation,
    Other,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Currency {
    USD,
    EUR,
    GBP,
    BTC,
    ETH,
    Other(String),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReciprocityPledge {
    pub id: String,
    pub dependency_id: String,
    pub contributor_did: String,
    pub organization: Option<String>,
    pub pledge_type: PledgeType,
    pub amount: Option<f64>,
    pub currency: Option<Currency>,
    pub description: String,
    pub evidence_url: Option<String>,
    pub period: Option<String>,
    pub pledged_at: Timestamp,
    pub acknowledged: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StewardshipScore {
    pub dependency_id: String,
    pub usage_count: u64,
    pub pledge_count: u64,
    pub ratio: f64,
    pub weighted_score: f64,
    pub pledge_type_counts: Vec<(String, u64)>,
}

// -- Batch/Pagination/Leaderboard --

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PaginationInput {
    pub offset: u64,
    pub limit: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PaginatedDependencies {
    pub items: Vec<Record>,
    pub total: u64,
    pub offset: u64,
    pub limit: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PaginatedUsageInput {
    pub id: String,
    pub pagination: PaginationInput,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PaginatedUsage {
    pub items: Vec<Record>,
    pub total: u64,
    pub offset: u64,
    pub limit: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PaginatedPledgesInput {
    pub id: String,
    pub pagination: PaginationInput,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PaginatedPledges {
    pub items: Vec<Record>,
    pub total: u64,
    pub offset: u64,
    pub limit: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BulkRegisterResult {
    pub registered: Vec<Record>,
    pub skipped: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BulkUsageResult {
    pub recorded: u64,
    pub records: Vec<Record>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TopDependency {
    pub dependency_id: String,
    pub usage_count: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LeaderboardEntry {
    pub dependency_id: String,
    pub usage_count: u64,
    pub pledge_count: u64,
    pub weighted_score: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EcosystemStat {
    pub ecosystem: String,
    pub count: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EcosystemStatistics {
    pub total_dependencies: u64,
    pub ecosystems: Vec<EcosystemStat>,
    pub verified_count: u64,
}

// ── Helpers ──────────────────────────────────────────────────────────

fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_attribution_dna.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load attribution DNA bundle. Run `hc dna pack dna/` first.")
}

fn now_timestamp() -> Timestamp {
    Timestamp::from_micros(chrono::Utc::now().timestamp_micros())
}

/// Decode an entry from a Record into a concrete type via MessagePack.
fn decode_entry<T: serde::de::DeserializeOwned>(record: &Record) -> Option<T> {
    match record.entry().as_option()? {
        Entry::App(bytes) => {
            let sb = SerializedBytes::from(bytes.to_owned());
            rmp_serde::from_slice(sb.bytes()).ok()
        }
        _ => None,
    }
}

fn make_dependency(id: &str, ecosystem: DependencyEcosystem) -> DependencyIdentity {
    DependencyIdentity {
        id: id.to_string(),
        name: format!("Test Dep: {}", id),
        ecosystem,
        maintainer_did: "did:mycelix:maintainer001".to_string(),
        repository_url: Some("https://github.com/example/test".to_string()),
        license: Some("MIT".to_string()),
        description: "A test dependency for integration testing".to_string(),
        version: Some("1.0.0".to_string()),
        registered_at: now_timestamp(),
        verified: false,
    }
}

fn make_usage_receipt(id: &str, dep_id: &str, user_did: &str) -> UsageReceipt {
    UsageReceipt {
        id: id.to_string(),
        dependency_id: dep_id.to_string(),
        user_did: user_did.to_string(),
        organization: Some("Test Corp".to_string()),
        usage_type: UsageType::Production,
        scale: Some(UsageScale::Enterprise),
        version_range: Some(">=1.0.0".to_string()),
        context: Some("Integration test usage".to_string()),
        attested_at: now_timestamp(),
    }
}

fn make_attestation(id: &str, dep_id: &str, user_did: &str) -> UsageAttestation {
    UsageAttestation {
        id: id.to_string(),
        dependency_id: dep_id.to_string(),
        user_did: user_did.to_string(),
        witness_commitment: vec![0xAB; 32],
        proof_bytes: vec![0x01; 256],
        verified: false,
        generated_at: now_timestamp(),
        expires_at: None,
        verifier_pubkey: None,
        verifier_signature: None,
    }
}

fn make_pledge(id: &str, dep_id: &str, contributor_did: &str) -> ReciprocityPledge {
    ReciprocityPledge {
        id: id.to_string(),
        dependency_id: dep_id.to_string(),
        contributor_did: contributor_did.to_string(),
        organization: Some("Sponsor Corp".to_string()),
        pledge_type: PledgeType::Financial,
        amount: Some(5000.0),
        currency: Some(Currency::USD),
        description: "Annual sponsorship".to_string(),
        evidence_url: Some("https://opencollective.com/test".to_string()),
        period: Some("2026-Q1".to_string()),
        pledged_at: now_timestamp(),
        acknowledged: false,
    }
}

// ── Registry Tests ───────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
#[ignore] // Requires Holochain conductor
async fn test_register_and_get_dependency() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor
        .setup_app("attribution", &[dna])
        .await
        .unwrap();
    let cell = app.cells()[0].clone();

    let dep = make_dependency("crate:serde:1.0", DependencyEcosystem::RustCrate);

    // Register
    let record: Record = conductor
        .call(&cell.zome("registry"), "register_dependency", dep.clone())
        .await;
    let created = decode_entry::<DependencyIdentity>(&record).unwrap();
    assert_eq!(created.id, "crate:serde:1.0");
    assert!(!created.verified);

    // Get by ID
    let retrieved: Option<Record> = conductor
        .call(
            &cell.zome("registry"),
            "get_dependency",
            "crate:serde:1.0".to_string(),
        )
        .await;
    assert!(retrieved.is_some());
    let retrieved_dep = decode_entry::<DependencyIdentity>(&retrieved.unwrap()).unwrap();
    assert_eq!(retrieved_dep.name, dep.name);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_get_all_dependencies() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor
        .setup_app("attribution", &[dna])
        .await
        .unwrap();
    let cell = app.cells()[0].clone();

    // Register two dependencies
    let _: Record = conductor
        .call(
            &cell.zome("registry"),
            "register_dependency",
            make_dependency("crate:tokio:1.0", DependencyEcosystem::RustCrate),
        )
        .await;
    let _: Record = conductor
        .call(
            &cell.zome("registry"),
            "register_dependency",
            make_dependency("npm:react:18.0", DependencyEcosystem::NpmPackage),
        )
        .await;

    let all: Vec<Record> = conductor
        .call(&cell.zome("registry"), "get_all_dependencies", ())
        .await;
    assert_eq!(all.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_get_dependencies_by_ecosystem() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor
        .setup_app("attribution", &[dna])
        .await
        .unwrap();
    let cell = app.cells()[0].clone();

    let _: Record = conductor
        .call(
            &cell.zome("registry"),
            "register_dependency",
            make_dependency("crate:rayon:1.0", DependencyEcosystem::RustCrate),
        )
        .await;
    let _: Record = conductor
        .call(
            &cell.zome("registry"),
            "register_dependency",
            make_dependency("npm:lodash:4.0", DependencyEcosystem::NpmPackage),
        )
        .await;

    let rust_deps: Vec<Record> = conductor
        .call(
            &cell.zome("registry"),
            "get_dependencies_by_ecosystem",
            "rust_crate".to_string(),
        )
        .await;
    assert_eq!(rust_deps.len(), 1);
    let dep = decode_entry::<DependencyIdentity>(&rust_deps[0]).unwrap();
    assert_eq!(dep.id, "crate:rayon:1.0");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_verify_dependency() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor
        .setup_app("attribution", &[dna])
        .await
        .unwrap();
    let cell = app.cells()[0].clone();

    let _: Record = conductor
        .call(
            &cell.zome("registry"),
            "register_dependency",
            make_dependency("crate:hdk:0.6", DependencyEcosystem::RustCrate),
        )
        .await;

    // Verify
    let verified_record: Record = conductor
        .call(
            &cell.zome("registry"),
            "verify_dependency",
            "crate:hdk:0.6".to_string(),
        )
        .await;
    let verified = decode_entry::<DependencyIdentity>(&verified_record).unwrap();
    assert!(verified.verified);
}

// ── Usage Tests ──────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_record_and_query_usage() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor
        .setup_app("attribution", &[dna])
        .await
        .unwrap();
    let cell = app.cells()[0].clone();

    let receipt = make_usage_receipt(
        "usage-001",
        "crate:serde:1.0",
        "did:mycelix:user001",
    );

    // Record usage
    let record: Record = conductor
        .call(&cell.zome("usage"), "record_usage", receipt.clone())
        .await;
    let created = decode_entry::<UsageReceipt>(&record).unwrap();
    assert_eq!(created.id, "usage-001");

    // Query by dependency
    let dep_usage: Vec<Record> = conductor
        .call(
            &cell.zome("usage"),
            "get_dependency_usage",
            "crate:serde:1.0".to_string(),
        )
        .await;
    assert_eq!(dep_usage.len(), 1);

    // Query by user
    let user_usage: Vec<Record> = conductor
        .call(
            &cell.zome("usage"),
            "get_user_usage",
            "did:mycelix:user001".to_string(),
        )
        .await;
    assert_eq!(user_usage.len(), 1);

    // Count
    let count: u64 = conductor
        .call(
            &cell.zome("usage"),
            "get_usage_count",
            "crate:serde:1.0".to_string(),
        )
        .await;
    assert_eq!(count, 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_submit_and_verify_attestation() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor
        .setup_app("attribution", &[dna])
        .await
        .unwrap();
    let cell = app.cells()[0].clone();

    let att = make_attestation(
        "attest-001",
        "crate:serde:1.0",
        "did:mycelix:user001",
    );

    // Submit (unverified)
    let record: Record = conductor
        .call(
            &cell.zome("usage"),
            "submit_usage_attestation",
            att.clone(),
        )
        .await;
    let created = decode_entry::<UsageAttestation>(&record).unwrap();
    assert!(!created.verified);
    assert!(created.verifier_pubkey.is_none());

    let original_action_hash = record.action_address().clone();

    // Verify
    let verify_input = VerifyAttestationInput {
        original_action_hash,
        verifier_pubkey: vec![0x42; 32],
        verifier_signature: vec![0x99; 64],
    };
    let verified_record: Record = conductor
        .call(
            &cell.zome("usage"),
            "verify_usage_attestation",
            verify_input,
        )
        .await;
    let verified = decode_entry::<UsageAttestation>(&verified_record).unwrap();
    assert!(verified.verified);
    assert_eq!(verified.verifier_pubkey.unwrap(), vec![0x42; 32]);
    assert_eq!(verified.verifier_signature.unwrap(), vec![0x99; 64]);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_get_dependency_attestations() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor
        .setup_app("attribution", &[dna])
        .await
        .unwrap();
    let cell = app.cells()[0].clone();

    let _: Record = conductor
        .call(
            &cell.zome("usage"),
            "submit_usage_attestation",
            make_attestation("attest-a", "crate:hdk:0.6", "did:mycelix:user001"),
        )
        .await;
    let _: Record = conductor
        .call(
            &cell.zome("usage"),
            "submit_usage_attestation",
            make_attestation("attest-b", "crate:hdk:0.6", "did:mycelix:user002"),
        )
        .await;

    let attestations: Vec<Record> = conductor
        .call(
            &cell.zome("usage"),
            "get_dependency_attestations",
            "crate:hdk:0.6".to_string(),
        )
        .await;
    assert_eq!(attestations.len(), 2);
}

// ── Reciprocity Tests ────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_record_and_query_pledges() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor
        .setup_app("attribution", &[dna])
        .await
        .unwrap();
    let cell = app.cells()[0].clone();

    let pledge = make_pledge(
        "pledge-001",
        "crate:serde:1.0",
        "did:mycelix:corp001",
    );

    // Record pledge
    let record: Record = conductor
        .call(&cell.zome("reciprocity"), "record_pledge", pledge.clone())
        .await;
    let created = decode_entry::<ReciprocityPledge>(&record).unwrap();
    assert_eq!(created.id, "pledge-001");
    assert!(!created.acknowledged);

    // Query by dependency
    let dep_pledges: Vec<Record> = conductor
        .call(
            &cell.zome("reciprocity"),
            "get_dependency_pledges",
            "crate:serde:1.0".to_string(),
        )
        .await;
    assert_eq!(dep_pledges.len(), 1);

    // Query by contributor
    let contrib_pledges: Vec<Record> = conductor
        .call(
            &cell.zome("reciprocity"),
            "get_contributor_pledges",
            "did:mycelix:corp001".to_string(),
        )
        .await;
    assert_eq!(contrib_pledges.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_acknowledge_pledge() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor
        .setup_app("attribution", &[dna])
        .await
        .unwrap();
    let cell = app.cells()[0].clone();

    let _: Record = conductor
        .call(
            &cell.zome("reciprocity"),
            "record_pledge",
            make_pledge("pledge-ack", "crate:tokio:1.0", "did:mycelix:corp002"),
        )
        .await;

    // Acknowledge
    let ack_record: Record = conductor
        .call(
            &cell.zome("reciprocity"),
            "acknowledge_pledge",
            "pledge-ack".to_string(),
        )
        .await;
    let acked = decode_entry::<ReciprocityPledge>(&ack_record).unwrap();
    assert!(acked.acknowledged);
}

// ── Cross-Zome Tests ─────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_stewardship_score_cross_zome() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor
        .setup_app("attribution", &[dna])
        .await
        .unwrap();
    let cell = app.cells()[0].clone();

    let dep_id = "crate:crosszome:1.0";

    // Register dependency
    let _: Record = conductor
        .call(
            &cell.zome("registry"),
            "register_dependency",
            make_dependency(dep_id, DependencyEcosystem::RustCrate),
        )
        .await;

    // Record 3 usage receipts
    for i in 0..3 {
        let _: Record = conductor
            .call(
                &cell.zome("usage"),
                "record_usage",
                make_usage_receipt(
                    &format!("usage-xz-{}", i),
                    dep_id,
                    &format!("did:mycelix:user{:03}", i),
                ),
            )
            .await;
    }

    // Record 1 pledge
    let _: Record = conductor
        .call(
            &cell.zome("reciprocity"),
            "record_pledge",
            make_pledge("pledge-xz-1", dep_id, "did:mycelix:corp001"),
        )
        .await;

    // Compute stewardship score (cross-zome call: reciprocity → usage)
    let score: StewardshipScore = conductor
        .call(
            &cell.zome("reciprocity"),
            "compute_stewardship_score",
            dep_id.to_string(),
        )
        .await;

    assert_eq!(score.dependency_id, dep_id);
    assert_eq!(score.usage_count, 3);
    assert_eq!(score.pledge_count, 1);
    let expected_ratio = 1.0 / 3.0;
    assert!((score.ratio - expected_ratio).abs() < 1e-10);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_stewardship_score_zero_usage() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor
        .setup_app("attribution", &[dna])
        .await
        .unwrap();
    let cell = app.cells()[0].clone();

    // No usage, no pledges
    let score: StewardshipScore = conductor
        .call(
            &cell.zome("reciprocity"),
            "compute_stewardship_score",
            "crate:nonexistent:1.0".to_string(),
        )
        .await;
    assert_eq!(score.usage_count, 0);
    assert_eq!(score.pledge_count, 0);
    assert_eq!(score.ratio, 0.0);
}

// ── Batch Registration Tests ────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_bulk_register_dependencies() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("attribution", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    let deps = vec![
        make_dependency("crate:bulk1:1.0", DependencyEcosystem::RustCrate),
        make_dependency("crate:bulk2:2.0", DependencyEcosystem::RustCrate),
        make_dependency("npm:bulk3:3.0", DependencyEcosystem::NpmPackage),
    ];

    let result: BulkRegisterResult = conductor
        .call(&cell.zome("registry"), "bulk_register_dependencies", deps.clone())
        .await;
    assert_eq!(result.registered.len(), 3);
    assert!(result.skipped.is_empty());

    // Duplicates should be skipped
    let result2: BulkRegisterResult = conductor
        .call(&cell.zome("registry"), "bulk_register_dependencies", deps)
        .await;
    assert!(result2.registered.is_empty());
    assert_eq!(result2.skipped.len(), 3);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_bulk_record_usage() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("attribution", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    let receipts = vec![
        make_usage_receipt("bulk-u1", "crate:serde:1.0", "did:mycelix:user001"),
        make_usage_receipt("bulk-u2", "crate:serde:1.0", "did:mycelix:user002"),
        make_usage_receipt("bulk-u3", "crate:tokio:1.0", "did:mycelix:user001"),
    ];

    let result: BulkUsageResult = conductor
        .call(&cell.zome("usage"), "bulk_record_usage", receipts)
        .await;
    assert_eq!(result.recorded, 3);
    assert_eq!(result.records.len(), 3);

    // Verify counts
    let count: u64 = conductor
        .call(&cell.zome("usage"), "get_usage_count", "crate:serde:1.0".to_string())
        .await;
    assert_eq!(count, 2);
}

// ── Pagination Tests ────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_paginated_dependencies() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("attribution", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    // Register 5 dependencies
    for i in 0..5 {
        let _: Record = conductor
            .call(
                &cell.zome("registry"),
                "register_dependency",
                make_dependency(&format!("crate:page{}:1.0", i), DependencyEcosystem::RustCrate),
            )
            .await;
    }

    // Page 1: offset=0, limit=2
    let page1: PaginatedDependencies = conductor
        .call(
            &cell.zome("registry"),
            "get_all_dependencies_paginated",
            PaginationInput { offset: 0, limit: 2 },
        )
        .await;
    assert_eq!(page1.total, 5);
    assert_eq!(page1.items.len(), 2);
    assert_eq!(page1.offset, 0);
    assert_eq!(page1.limit, 2);

    // Page 3: offset=4, limit=2 — should get 1 item
    let page3: PaginatedDependencies = conductor
        .call(
            &cell.zome("registry"),
            "get_all_dependencies_paginated",
            PaginationInput { offset: 4, limit: 2 },
        )
        .await;
    assert_eq!(page3.total, 5);
    assert_eq!(page3.items.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_paginated_usage() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("attribution", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    // Record 4 usage receipts for same dep
    for i in 0..4 {
        let _: Record = conductor
            .call(
                &cell.zome("usage"),
                "record_usage",
                make_usage_receipt(
                    &format!("page-u{}", i),
                    "crate:paged:1.0",
                    &format!("did:mycelix:user{:03}", i),
                ),
            )
            .await;
    }

    let page: PaginatedUsage = conductor
        .call(
            &cell.zome("usage"),
            "get_dependency_usage_paginated",
            PaginatedUsageInput {
                id: "crate:paged:1.0".to_string(),
                pagination: PaginationInput { offset: 1, limit: 2 },
            },
        )
        .await;
    assert_eq!(page.total, 4);
    assert_eq!(page.items.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_paginated_pledges() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("attribution", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    for i in 0..3 {
        let _: Record = conductor
            .call(
                &cell.zome("reciprocity"),
                "record_pledge",
                make_pledge(
                    &format!("pledge-page{}", i),
                    "crate:paged:1.0",
                    &format!("did:mycelix:corp{:03}", i),
                ),
            )
            .await;
    }

    let page: PaginatedPledges = conductor
        .call(
            &cell.zome("reciprocity"),
            "get_dependency_pledges_paginated",
            PaginatedPledgesInput {
                id: "crate:paged:1.0".to_string(),
                pagination: PaginationInput { offset: 0, limit: 2 },
            },
        )
        .await;
    assert_eq!(page.total, 3);
    assert_eq!(page.items.len(), 2);
}

// ── Top-N & Leaderboard Tests ───────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_top_dependencies() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("attribution", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    // Register 3 deps
    for name in ["alpha", "beta", "gamma"] {
        let _: Record = conductor
            .call(
                &cell.zome("registry"),
                "register_dependency",
                make_dependency(&format!("crate:{}:1.0", name), DependencyEcosystem::RustCrate),
            )
            .await;
    }

    // alpha: 3 usages, beta: 1 usage, gamma: 0 usages
    for i in 0..3 {
        let _: Record = conductor
            .call(
                &cell.zome("usage"),
                "record_usage",
                make_usage_receipt(
                    &format!("top-alpha-{}", i),
                    "crate:alpha:1.0",
                    &format!("did:mycelix:u{}", i),
                ),
            )
            .await;
    }
    let _: Record = conductor
        .call(
            &cell.zome("usage"),
            "record_usage",
            make_usage_receipt("top-beta-0", "crate:beta:1.0", "did:mycelix:u0"),
        )
        .await;

    let top: Vec<TopDependency> = conductor
        .call(&cell.zome("usage"), "get_top_dependencies", 10u64)
        .await;

    assert_eq!(top.len(), 2); // gamma excluded (0 usage)
    assert_eq!(top[0].dependency_id, "crate:alpha:1.0");
    assert_eq!(top[0].usage_count, 3);
    assert_eq!(top[1].dependency_id, "crate:beta:1.0");
    assert_eq!(top[1].usage_count, 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_weighted_stewardship_score() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("attribution", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    let dep_id = "crate:weighted:1.0";

    // Register dependency
    let _: Record = conductor
        .call(
            &cell.zome("registry"),
            "register_dependency",
            make_dependency(dep_id, DependencyEcosystem::RustCrate),
        )
        .await;

    // 2 usage receipts
    for i in 0..2 {
        let _: Record = conductor
            .call(
                &cell.zome("usage"),
                "record_usage",
                make_usage_receipt(
                    &format!("ws-u{}", i),
                    dep_id,
                    &format!("did:mycelix:user{:03}", i),
                ),
            )
            .await;
    }

    // 1 financial pledge ($5000)
    let _: Record = conductor
        .call(
            &cell.zome("reciprocity"),
            "record_pledge",
            make_pledge("ws-pledge-1", dep_id, "did:mycelix:corp001"),
        )
        .await;

    let score: StewardshipScore = conductor
        .call(
            &cell.zome("reciprocity"),
            "compute_stewardship_score",
            dep_id.to_string(),
        )
        .await;

    assert_eq!(score.usage_count, 2);
    assert_eq!(score.pledge_count, 1);
    assert!((score.ratio - 0.5).abs() < 1e-10);
    assert!(score.weighted_score > 0.0);
    assert!(!score.pledge_type_counts.is_empty());
    // Financial pledge should be in the counts
    assert!(score.pledge_type_counts.iter().any(|(t, c)| t == "Financial" && *c == 1));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_ecosystem_statistics() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("attribution", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    // Register deps across ecosystems
    let _: Record = conductor
        .call(
            &cell.zome("registry"),
            "register_dependency",
            make_dependency("crate:eco1:1.0", DependencyEcosystem::RustCrate),
        )
        .await;
    let _: Record = conductor
        .call(
            &cell.zome("registry"),
            "register_dependency",
            make_dependency("crate:eco2:1.0", DependencyEcosystem::RustCrate),
        )
        .await;
    let _: Record = conductor
        .call(
            &cell.zome("registry"),
            "register_dependency",
            make_dependency("npm:eco3:1.0", DependencyEcosystem::NpmPackage),
        )
        .await;

    let stats: EcosystemStatistics = conductor
        .call(&cell.zome("registry"), "get_ecosystem_statistics", ())
        .await;

    assert_eq!(stats.total_dependencies, 3);
    assert_eq!(stats.verified_count, 0);
    // Should have 2 ecosystems
    assert_eq!(stats.ecosystems.len(), 2);
    let rust_count = stats.ecosystems.iter()
        .find(|e| e.ecosystem.contains("Rust"))
        .map(|e| e.count)
        .unwrap_or(0);
    assert_eq!(rust_count, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_stewardship_leaderboard() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("attribution", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    // Register 2 deps
    let _: Record = conductor
        .call(
            &cell.zome("registry"),
            "register_dependency",
            make_dependency("crate:lb1:1.0", DependencyEcosystem::RustCrate),
        )
        .await;
    let _: Record = conductor
        .call(
            &cell.zome("registry"),
            "register_dependency",
            make_dependency("crate:lb2:1.0", DependencyEcosystem::RustCrate),
        )
        .await;

    // lb1: 1 usage + 1 pledge, lb2: 1 usage + 0 pledges
    let _: Record = conductor
        .call(
            &cell.zome("usage"),
            "record_usage",
            make_usage_receipt("lb-u1", "crate:lb1:1.0", "did:mycelix:u1"),
        )
        .await;
    let _: Record = conductor
        .call(
            &cell.zome("usage"),
            "record_usage",
            make_usage_receipt("lb-u2", "crate:lb2:1.0", "did:mycelix:u2"),
        )
        .await;
    let _: Record = conductor
        .call(
            &cell.zome("reciprocity"),
            "record_pledge",
            make_pledge("lb-p1", "crate:lb1:1.0", "did:mycelix:corp1"),
        )
        .await;

    // Leaderboard should have lb1 first (higher weighted_score)
    let board: Vec<LeaderboardEntry> = conductor
        .call(
            &cell.zome("reciprocity"),
            "get_stewardship_leaderboard",
            10u64,
        )
        .await;
    assert_eq!(board.len(), 2);
    assert_eq!(board[0].dependency_id, "crate:lb1:1.0");
    assert!(board[0].weighted_score > board[1].weighted_score);

    // Under-supported should have lb2 first (lower score)
    let under: Vec<LeaderboardEntry> = conductor
        .call(
            &cell.zome("reciprocity"),
            "get_under_supported_dependencies",
            10u64,
        )
        .await;
    assert!(!under.is_empty());
    assert_eq!(under[0].dependency_id, "crate:lb2:1.0");
}
