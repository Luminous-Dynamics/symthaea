// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Mycelix Property - Sweettest Integration Tests
//!
//! Comprehensive integration tests using Holochain's sweettest framework.
//! Tests cover property registry, transfers, disputes, commons, and bridge operations.
//!
//! ## Running Tests
//!
//! ```bash
//! # Ensure the DNA bundle exists (no pre-built .dna yet - build first)
//! # hc dna pack dna/
//!
//! # Run tests (requires Holochain conductor via nix develop)
//! cargo test --test sweettest_integration -- --ignored
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types (avoids importing zome crates / duplicate symbols)
// ============================================================================

// --- registry types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Property {
    pub id: String,
    pub property_type: PropertyType,
    pub title: String,
    pub description: String,
    pub owner_did: String,
    pub co_owners: Vec<CoOwner>,
    pub geolocation: Option<GeoLocation>,
    pub address: Option<Address>,
    pub metadata: PropertyMetadata,
    pub registered: Timestamp,
    pub updated: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum PropertyType {
    Residential,
    Commercial,
    Agricultural,
    Industrial,
    Digital,
    IntellectualProperty,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CoOwner {
    pub did: String,
    pub share_percentage: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Address {
    pub street: String,
    pub city: String,
    pub state: String,
    pub country: String,
    pub postal_code: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PropertyMetadata {
    pub area_sqm: Option<f64>,
    pub year_built: Option<u32>,
    pub zoning: Option<String>,
    pub assessed_value: Option<u64>,
    pub currency: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterPropertyInput {
    pub property_type: PropertyType,
    pub title: String,
    pub description: String,
    pub owner_did: String,
    pub co_owners: Vec<CoOwner>,
    pub geolocation: Option<GeoLocation>,
    pub address: Option<Address>,
    pub metadata: PropertyMetadata,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TitleDeed {
    pub id: String,
    pub property_id: String,
    pub owner_did: String,
    pub deed_type: DeedType,
    pub issued: Timestamp,
    pub supersedes: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DeedType {
    Original,
    Transfer,
    Inheritance,
    CourtOrder,
}

// --- transfer types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Transfer {
    pub id: String,
    pub property_id: String,
    pub from_did: String,
    pub to_did: String,
    pub transfer_type: TransferType,
    pub price: Option<u64>,
    pub currency: Option<String>,
    pub conditions: Vec<TransferCondition>,
    pub status: TransferStatus,
    pub initiated: Timestamp,
    pub completed: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TransferType {
    Sale,
    Inheritance,
    Gift,
    CourtOrder,
    Exchange,
    Other(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TransferStatus {
    Initiated,
    Accepted,
    InEscrow,
    ConditionsMet,
    Completed,
    Cancelled,
    Disputed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TransferCondition {
    pub condition_type: ConditionType,
    pub description: String,
    pub satisfied: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ConditionType {
    Payment,
    Inspection,
    TitleClear,
    GovernmentApproval,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InitiateTransferInput {
    pub property_id: String,
    pub from_did: String,
    pub to_did: String,
    pub transfer_type: TransferType,
    pub price: Option<u64>,
    pub currency: Option<String>,
    pub conditions: Vec<TransferCondition>,
}

// --- disputes types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PropertyDispute {
    pub id: String,
    pub property_id: String,
    pub dispute_type: DisputeType,
    pub claimant_did: String,
    pub respondent_did: String,
    pub description: String,
    pub evidence_ids: Vec<String>,
    pub status: DisputeStatus,
    pub justice_case_id: Option<String>,
    pub filed: Timestamp,
    pub resolved: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DisputeType {
    OwnershipContest,
    BoundaryDispute,
    Encumbrance,
    Fraud,
    Inheritance,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DisputeStatus {
    Filed,
    UnderReview,
    Arbitration,
    Resolved,
    Dismissed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FileDisputeInput {
    pub property_id: String,
    pub dispute_type: DisputeType,
    pub claimant_did: String,
    pub respondent_did: String,
    pub description: String,
    pub evidence_ids: Vec<String>,
}

// --- commons types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CommonResource {
    pub id: String,
    pub name: String,
    pub resource_type: ResourceType,
    pub description: String,
    pub governance_did: String,
    pub governance_rules: GovernanceRules,
    pub created: Timestamp,
    pub updated: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ResourceType {
    Land,
    Water,
    Forest,
    Spectrum,
    Digital,
    Cultural,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GovernanceRules {
    pub max_usage_per_member: Option<u64>,
    pub usage_period_days: Option<u32>,
    pub requires_approval: bool,
    pub min_reputation_score: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateResourceInput {
    pub name: String,
    pub resource_type: ResourceType,
    pub description: String,
    pub governance_did: String,
    pub governance_rules: GovernanceRules,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GrantRightInput {
    pub resource_id: String,
    pub grantee_did: String,
    pub right_type: RightType,
    pub max_usage: Option<u64>,
    pub expires_days: Option<u32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum RightType {
    Access,
    Harvest,
    Graze,
    Build,
    Transit,
    Digital,
}

// --- bridge types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerifyOwnershipInput {
    pub property_id: String,
    pub source_happ: String,
    pub purpose: OwnershipQueryPurpose,
    pub expected_owner: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum OwnershipQueryPurpose {
    LoanCollateral,
    TransferVerification,
    DisputeResolution,
    GeneralQuery,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OwnershipResult {
    pub id: String,
    pub property_id: String,
    pub owner_did: String,
    pub ownership_type: OwnershipType,
    pub share_percentage: f64,
    pub encumbrances: Vec<EncumbranceInfo>,
    pub verified_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum OwnershipType {
    Sole,
    Joint,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EncumbranceInfo {
    pub encumbrance_type: String,
    pub holder_did: String,
    pub amount: Option<f64>,
}

// ============================================================================
// Test Utilities
// ============================================================================

fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_property.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load DNA bundle - run 'hc dna pack dna/' first")
}

fn decode_entry<T: serde::de::DeserializeOwned>(record: &Record) -> Option<T> {
    match record.entry().as_option()? {
        Entry::App(bytes) => {
            let sb = SerializedBytes::from(bytes.to_owned());
            rmp_serde::from_slice(sb.bytes()).ok()
        }
        _ => None,
    }
}

// ============================================================================
// Registry Tests
// ============================================================================

#[cfg(test)]
mod registry_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_register_and_get_property() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = RegisterPropertyInput {
            property_type: PropertyType::Residential,
            title: "Test Home".to_string(),
            description: "A test residential property".to_string(),
            owner_did: "did:mycelix:owner1".to_string(),
            co_owners: vec![],
            geolocation: Some(GeoLocation {
                latitude: 32.9483,
                longitude: -96.7298,
            }),
            address: Some(Address {
                street: "123 Test St".to_string(),
                city: "Richardson".to_string(),
                state: "TX".to_string(),
                country: "US".to_string(),
                postal_code: "75080".to_string(),
            }),
            metadata: PropertyMetadata {
                area_sqm: Some(200.0),
                year_built: Some(2020),
                zoning: Some("residential".to_string()),
                assessed_value: Some(350000),
                currency: Some("USD".to_string()),
            },
        };

        let record: Record = conductor
            .call(&cell.zome("registry"), "register_property", input)
            .await;

        let property: Property = decode_entry(&record).expect("decode property");
        assert_eq!(property.title, "Test Home");
        assert_eq!(property.owner_did, "did:mycelix:owner1");

        // Retrieve
        let retrieved: Option<Record> = conductor
            .call(&cell.zome("registry"), "get_property", property.id.clone())
            .await;

        assert!(retrieved.is_some(), "Should retrieve property");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_owner_properties() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let owner = "did:mycelix:multi-owner".to_string();

        for i in 0..2 {
            let input = RegisterPropertyInput {
                property_type: PropertyType::Commercial,
                title: format!("Property {}", i),
                description: format!("Commercial property {}", i),
                owner_did: owner.clone(),
                co_owners: vec![],
                geolocation: None,
                address: None,
                metadata: PropertyMetadata {
                    area_sqm: None,
                    year_built: None,
                    zoning: None,
                    assessed_value: None,
                    currency: None,
                },
            };
            let _: Record = conductor
                .call(&cell.zome("registry"), "register_property", input)
                .await;
        }

        let results: Vec<Record> = conductor
            .call(
                &cell.zome("registry"),
                "get_owner_properties",
                owner.clone(),
            )
            .await;

        assert!(
            results.len() >= 2,
            "Owner should have at least 2 properties"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_verify_ownership_and_clear_title() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = RegisterPropertyInput {
            property_type: PropertyType::Residential,
            title: "Clear Title Home".to_string(),
            description: "Property with clear title".to_string(),
            owner_did: "did:mycelix:clear-owner".to_string(),
            co_owners: vec![],
            geolocation: None,
            address: None,
            metadata: PropertyMetadata {
                area_sqm: None,
                year_built: None,
                zoning: None,
                assessed_value: None,
                currency: None,
            },
        };

        let record: Record = conductor
            .call(&cell.zome("registry"), "register_property", input)
            .await;
        let property: Property = decode_entry(&record).expect("decode property");

        let has_clear: bool = conductor
            .call(
                &cell.zome("registry"),
                "has_clear_title",
                property.id.clone(),
            )
            .await;

        assert!(
            has_clear,
            "Newly registered property should have clear title"
        );
    }
}

// ============================================================================
// Transfer Tests
// ============================================================================

#[cfg(test)]
mod transfer_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_initiate_and_complete_transfer() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Register property first
        let reg_input = RegisterPropertyInput {
            property_type: PropertyType::Residential,
            title: "Transferable Home".to_string(),
            description: "Property to be transferred".to_string(),
            owner_did: "did:mycelix:seller".to_string(),
            co_owners: vec![],
            geolocation: None,
            address: None,
            metadata: PropertyMetadata {
                area_sqm: Some(150.0),
                year_built: Some(2015),
                zoning: None,
                assessed_value: Some(250000),
                currency: Some("USD".to_string()),
            },
        };

        let prop_record: Record = conductor
            .call(&cell.zome("registry"), "register_property", reg_input)
            .await;
        let property: Property = decode_entry(&prop_record).expect("decode property");

        // Initiate transfer
        let transfer_input = InitiateTransferInput {
            property_id: property.id.clone(),
            from_did: "did:mycelix:seller".to_string(),
            to_did: "did:mycelix:buyer".to_string(),
            transfer_type: TransferType::Sale,
            price: Some(250000),
            currency: Some("USD".to_string()),
            conditions: vec![TransferCondition {
                condition_type: ConditionType::Payment,
                description: "Full payment received".to_string(),
                satisfied: false,
            }],
        };

        let transfer_record: Record = conductor
            .call(&cell.zome("transfer"), "initiate_transfer", transfer_input)
            .await;

        let transfer: Transfer = decode_entry(&transfer_record).expect("decode transfer");
        assert_eq!(transfer.property_id, property.id);
        assert_eq!(transfer.from_did, "did:mycelix:seller");
        assert_eq!(transfer.to_did, "did:mycelix:buyer");

        // Complete transfer
        let completed: Record = conductor
            .call(
                &cell.zome("transfer"),
                "complete_transfer",
                transfer.id.clone(),
            )
            .await;

        let completed_transfer: Transfer = decode_entry(&completed).expect("decode completed");
        assert!(
            completed_transfer.completed.is_some(),
            "Transfer should have completion time"
        );
    }
}

// ============================================================================
// Disputes Tests
// ============================================================================

#[cfg(test)]
mod disputes_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_file_and_get_dispute() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = FileDisputeInput {
            property_id: "prop:disputed-123".to_string(),
            dispute_type: DisputeType::BoundaryDispute,
            claimant_did: "did:mycelix:neighbor-a".to_string(),
            respondent_did: "did:mycelix:neighbor-b".to_string(),
            description: "Fence encroaches on my property".to_string(),
            evidence_ids: vec!["survey:2024".to_string()],
        };

        let record: Record = conductor
            .call(&cell.zome("disputes"), "file_dispute", input)
            .await;

        let dispute: PropertyDispute = decode_entry(&record).expect("decode dispute");
        assert_eq!(dispute.status, DisputeStatus::Filed);
        assert_eq!(dispute.property_id, "prop:disputed-123");

        // Get disputes for property
        let disputes: Vec<Record> = conductor
            .call(
                &cell.zome("disputes"),
                "get_property_disputes",
                "prop:disputed-123".to_string(),
            )
            .await;

        assert!(!disputes.is_empty(), "Should have disputes for property");
    }
}

// ============================================================================
// Commons Tests
// ============================================================================

#[cfg(test)]
mod commons_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_common_resource_and_grant_right() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let resource_input = CreateResourceInput {
            name: "Community Forest".to_string(),
            resource_type: ResourceType::Forest,
            description: "Shared forest for sustainable harvesting".to_string(),
            governance_did: "did:mycelix:council".to_string(),
            governance_rules: GovernanceRules {
                max_usage_per_member: Some(100),
                usage_period_days: Some(30),
                requires_approval: true,
                min_reputation_score: Some(0.5),
            },
        };

        let record: Record = conductor
            .call(
                &cell.zome("commons"),
                "create_common_resource",
                resource_input,
            )
            .await;

        let resource: CommonResource = decode_entry(&record).expect("decode resource");
        assert_eq!(resource.name, "Community Forest");

        // Grant usage right
        let right_input = GrantRightInput {
            resource_id: resource.id.clone(),
            grantee_did: "did:mycelix:member1".to_string(),
            right_type: RightType::Harvest,
            max_usage: Some(50),
            expires_days: Some(90),
        };

        let _: Record = conductor
            .call(&cell.zome("commons"), "grant_usage_right", right_input)
            .await;

        // Get resource back
        let retrieved: Option<Record> = conductor
            .call(&cell.zome("commons"), "get_resource", resource.id.clone())
            .await;

        assert!(retrieved.is_some(), "Should retrieve common resource");
    }
}

// ============================================================================
// Bridge Tests
// ============================================================================

#[cfg(test)]
mod bridge_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_verify_ownership_via_bridge() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Register property first
        let reg_input = RegisterPropertyInput {
            property_type: PropertyType::Residential,
            title: "Bridge Verify Home".to_string(),
            description: "Property for bridge verification".to_string(),
            owner_did: "did:mycelix:bridge-owner".to_string(),
            co_owners: vec![],
            geolocation: None,
            address: None,
            metadata: PropertyMetadata {
                area_sqm: None,
                year_built: None,
                zoning: None,
                assessed_value: None,
                currency: None,
            },
        };

        let prop_record: Record = conductor
            .call(&cell.zome("registry"), "register_property", reg_input)
            .await;
        let property: Property = decode_entry(&prop_record).expect("decode property");

        // Verify via bridge
        let verify_input = VerifyOwnershipInput {
            property_id: property.id.clone(),
            source_happ: "finance".to_string(),
            purpose: OwnershipQueryPurpose::LoanCollateral,
            expected_owner: Some("did:mycelix:bridge-owner".to_string()),
        };

        let result: OwnershipResult = conductor
            .call(
                &cell.zome("property_bridge"),
                "verify_ownership",
                verify_input,
            )
            .await;

        assert_eq!(result.property_id, property.id);
        assert_eq!(result.owner_did, "did:mycelix:bridge-owner");
    }
}

// ============================================================================
// Full Lifecycle Test
// ============================================================================

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_register_transfer_dispute_lifecycle() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // 1. Register property
        let reg_input = RegisterPropertyInput {
            property_type: PropertyType::Residential,
            title: "Lifecycle Home".to_string(),
            description: "Full lifecycle test property".to_string(),
            owner_did: "did:mycelix:lifecycle-owner".to_string(),
            co_owners: vec![],
            geolocation: None,
            address: None,
            metadata: PropertyMetadata {
                area_sqm: Some(180.0),
                year_built: Some(2018),
                zoning: None,
                assessed_value: Some(300000),
                currency: Some("USD".to_string()),
            },
        };

        let prop_record: Record = conductor
            .call(&cell.zome("registry"), "register_property", reg_input)
            .await;
        let property: Property = decode_entry(&prop_record).expect("decode property");

        // 2. Initiate transfer
        let transfer_input = InitiateTransferInput {
            property_id: property.id.clone(),
            from_did: "did:mycelix:lifecycle-owner".to_string(),
            to_did: "did:mycelix:new-owner".to_string(),
            transfer_type: TransferType::Sale,
            price: Some(300000),
            currency: Some("USD".to_string()),
            conditions: vec![],
        };

        let transfer_record: Record = conductor
            .call(&cell.zome("transfer"), "initiate_transfer", transfer_input)
            .await;
        let transfer: Transfer = decode_entry(&transfer_record).expect("decode transfer");

        // 3. File dispute during transfer
        let dispute_input = FileDisputeInput {
            property_id: property.id.clone(),
            dispute_type: DisputeType::OwnershipContest,
            claimant_did: "did:mycelix:heir".to_string(),
            respondent_did: "did:mycelix:lifecycle-owner".to_string(),
            description: "Claiming inheritance right".to_string(),
            evidence_ids: vec!["will:2024".to_string()],
        };

        let dispute_record: Record = conductor
            .call(&cell.zome("disputes"), "file_dispute", dispute_input)
            .await;
        let dispute: PropertyDispute = decode_entry(&dispute_record).expect("decode dispute");

        assert_eq!(dispute.status, DisputeStatus::Filed);

        // 4. Verify state
        let title_deed: Option<Record> = conductor
            .call(
                &cell.zome("registry"),
                "get_title_deed",
                property.id.clone(),
            )
            .await;

        assert!(title_deed.is_some(), "Property should have a title deed");
    }
}
