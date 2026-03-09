//! Test Fixtures for Mycelix Finance
//!
//! This module provides common test utilities, helpers, and fixtures
//! for testing the Mycelix Finance Holochain zomes.

use holochain::sweettest::{SweetConductor, SweetDnaFile, SweetAgents, SweetCell};

// Re-export zome types for tests
pub use payments_integrity::*;
pub use payments::*;

/// Standard test DID prefix
pub const TEST_DID_PREFIX: &str = "did:mycelix:test:";

/// Test currencies (only SAP and TEND are valid)
pub const TEST_CURRENCY_SAP: &str = "SAP";
pub const TEST_CURRENCY_TEND: &str = "TEND";

/// Generate a test DID
pub fn test_did(suffix: &str) -> String {
    format!("{}{}", TEST_DID_PREFIX, suffix)
}

/// Generate a unique test DID based on timestamp
pub fn unique_test_did(prefix: &str) -> String {
    let timestamp = chrono::Utc::now().timestamp_micros();
    format!("{}{}:{}", TEST_DID_PREFIX, prefix, timestamp)
}

/// Helper function to set up a conductor with default configuration
pub async fn setup_conductor() -> Result<SweetConductor, Box<dyn std::error::Error + Send + Sync>> {
    let conductor = SweetConductor::from_standard_config().await;
    Ok(conductor)
}

/// Helper function to install the Finance DNA on a conductor
pub async fn install_finance_dna(
    conductor: &mut SweetConductor,
    num_agents: usize,
) -> Result<Vec<SweetCell>, Box<dyn std::error::Error + Send + Sync>> {
    // Load the DNA from the workspace DNA bundle
    let dna_path = std::path::PathBuf::from("../dna/mycelix_finance.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path).await?;

    // Create agents using the conductor's keystore
    let keystore = conductor.keystore();
    let agents = SweetAgents::get(keystore, num_agents).await;
    let apps = conductor
        .setup_app_for_agents("mycelix-finance", &agents, [&dna])
        .await?;

    // Extract cells from installed apps
    let mut cells = Vec::new();
    for app in apps.iter() {
        cells.push(app.cells()[0].clone());
    }

    Ok(cells)
}

/// Helper to get the payments zome from a cell
pub fn payments_zome(cell: &SweetCell) -> holochain::sweettest::SweetZome {
    cell.zome("payments")
}

/// Helper to get the recognition zome from a cell
pub fn recognition_zome(cell: &SweetCell) -> holochain::sweettest::SweetZome {
    cell.zome("recognition")
}

/// Helper to get the tend zome from a cell
pub fn tend_zome(cell: &SweetCell) -> holochain::sweettest::SweetZome {
    cell.zome("tend")
}

/// Helper to get the bridge zome from a cell
pub fn bridge_zome(cell: &SweetCell) -> holochain::sweettest::SweetZome {
    cell.zome("bridge")
}

/// Helper to get the staking zome from a cell
pub fn staking_zome(cell: &SweetCell) -> holochain::sweettest::SweetZome {
    cell.zome("staking")
}

/// Helper to get the treasury zome from a cell
pub fn treasury_zome(cell: &SweetCell) -> holochain::sweettest::SweetZome {
    cell.zome("treasury")
}

/// Create a test payment input
pub fn create_test_payment_input(
    from_did: &str,
    to_did: &str,
    amount: u64,
    payment_type: PaymentType,
) -> SendPaymentInput {
    SendPaymentInput {
        from_did: from_did.to_string(),
        to_did: to_did.to_string(),
        amount,
        currency: TEST_CURRENCY_SAP.to_string(),
        payment_type,
        memo: Some("Test payment".to_string()),
    }
}

/// Create a test payment channel input
pub fn create_test_channel_input(
    party_a: &str,
    party_b: &str,
    deposit_a: u64,
    deposit_b: u64,
) -> OpenChannelInput {
    OpenChannelInput {
        party_a: party_a.to_string(),
        party_b: party_b.to_string(),
        currency: TEST_CURRENCY_SAP.to_string(),
        initial_deposit_a: deposit_a,
        initial_deposit_b: deposit_b,
    }
}

/// Helper to get current timestamp
pub fn timestamp() -> i64 {
    chrono::Utc::now().timestamp()
}

/// Test assertion helper - checks if a payment was created successfully
pub fn assert_payment_valid(payment: &Payment, from_did: &str, to_did: &str, amount: u64) {
    assert!(payment.from_did.starts_with("did:"), "From DID must be valid");
    assert!(payment.to_did.starts_with("did:"), "To DID must be valid");
    assert_eq!(payment.from_did, from_did, "From DID mismatch");
    assert_eq!(payment.to_did, to_did, "To DID mismatch");
    assert_eq!(payment.amount, amount, "Amount mismatch");
    assert!(payment.amount > 0, "Amount must be positive");
    assert!(
        payment.currency == "SAP" || payment.currency == "TEND",
        "Currency must be SAP or TEND, got: {}", payment.currency
    );
}

/// Test assertion helper - checks if a payment channel is valid
pub fn assert_channel_valid(
    channel: &PaymentChannel,
    party_a: &str,
    party_b: &str,
    balance_a: u64,
    balance_b: u64,
) {
    assert_eq!(channel.party_a, party_a, "Party A mismatch");
    assert_eq!(channel.party_b, party_b, "Party B mismatch");
    assert_eq!(channel.balance_a, balance_a, "Balance A mismatch");
    assert_eq!(channel.balance_b, balance_b, "Balance B mismatch");
    // u64 is always >= 0, no negative check needed
}

#[cfg(test)]
mod fixture_tests {
    use super::*;

    #[test]
    fn test_did_generation() {
        let did = test_did("alice");
        assert!(did.starts_with(TEST_DID_PREFIX));
        assert!(did.ends_with("alice"));
    }

    #[test]
    fn test_unique_did_generation() {
        let did1 = unique_test_did("user");
        let did2 = unique_test_did("user");
        assert_ne!(did1, did2, "Unique DIDs should be different");
    }

    #[test]
    fn test_payment_input_creation() {
        let input = create_test_payment_input(
            "did:mycelix:alice",
            "did:mycelix:bob",
            100_000_000,
            PaymentType::Direct,
        );
        assert_eq!(input.from_did, "did:mycelix:alice");
        assert_eq!(input.to_did, "did:mycelix:bob");
        assert_eq!(input.amount, 100_000_000);
        assert_eq!(input.currency, "SAP", "Default currency should be SAP");
    }

    #[test]
    fn test_currency_constants() {
        assert_eq!(TEST_CURRENCY_SAP, "SAP");
        assert_eq!(TEST_CURRENCY_TEND, "TEND");
    }
}
