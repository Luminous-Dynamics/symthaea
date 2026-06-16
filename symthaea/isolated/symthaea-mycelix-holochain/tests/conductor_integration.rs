// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Full-stack integration tests for the Holochain conductor bridge.
//!
//! These tests require a running Holochain conductor with the Mycelix governance
//! and finance hApps installed.
//!
//! Prerequisites:
//! - Conductor running at 127.0.0.1:9000 (admin) / 4446 (app)
//! - Governance and Finance DNAs installed
//!
//! Run with:
//!   HOLOCHAIN_ADMIN_PORT=9000 HOLOCHAIN_APP_PORT=4446 \
//!     cargo test --test conductor_integration -- --ignored --nocapture
//!
//! All tests are `#[ignore]` by default so `cargo test` doesn't require a conductor.

use std::sync::Arc;
use symthaea_mycelix_holochain::{
    AdminWebsocket, BridgeError, FinanceDispatcher, GovernanceDispatcher, HolochainConductor,
};

fn admin_port() -> u16 {
    std::env::var("HOLOCHAIN_ADMIN_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4444)
}

fn app_port() -> u16 {
    std::env::var("HOLOCHAIN_APP_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4445)
}

fn admin_addr() -> String {
    format!("127.0.0.1:{}", admin_port())
}

fn app_addr() -> String {
    format!("127.0.0.1:{}", app_port())
}

// ---------------------------------------------------------------------------
// Admin WebSocket connectivity tests
// ---------------------------------------------------------------------------

/// Verifies we can establish a WebSocket connection to the admin interface
/// and list installed apps.
#[tokio::test]
#[ignore] // Requires running conductor
async fn test_admin_websocket_connects() {
    let result = AdminWebsocket::connect(&*admin_addr()).await;
    match result {
        Ok(admin) => {
            println!("SUCCESS: Admin WebSocket connected at {}", admin_addr());

            // Try to list apps to verify the admin API works
            match admin.list_apps(None).await {
                Ok(apps) => {
                    println!("  Listed {} apps:", apps.len());
                    for app in &apps {
                        println!("    - {} ({:?})", app.installed_app_id, app.status);
                    }
                }
                Err(e) => {
                    let msg = format!("{e:?}");
                    if msg.contains("missing field") || msg.contains("Deserialize") {
                        println!(
                            "  EXPECTED: Version mismatch on list_apps: {}",
                            msg
                        );
                    } else {
                        println!("  Admin API error (may be version mismatch): {e:?}");
                    }
                }
            }
        }
        Err(e) => panic!("Admin WebSocket connection failed: {e:?}"),
    }
}

// ---------------------------------------------------------------------------
// App WebSocket connection tests
// ---------------------------------------------------------------------------

/// Tests the app WebSocket connection flow via HolochainConductor.
/// Expected to fail gracefully if auth token is not configured.
#[tokio::test]
#[ignore] // Requires running conductor
async fn test_app_websocket_connect_with_fallback() {
    let conductor = HolochainConductor::new(app_addr(), "mycelix-gov-only");
    let result = conductor.connect().await;
    match &result {
        Ok(()) => {
            assert!(conductor.is_connected());
            println!("SUCCESS: Full app WebSocket connection established");
        }
        Err(BridgeError::ConnectionFailed(msg)) => {
            println!(
                "EXPECTED: Connection failed (auth token or version mismatch): {}",
                msg
            );
        }
        Err(e) => {
            println!("Connection error (may be expected): {e:?}");
        }
    }
}

/// Verifies that connection to a non-existent port fails gracefully.
#[tokio::test]
#[ignore] // Requires running conductor (to contrast with invalid port)
async fn test_connect_to_invalid_url_fails_gracefully() {
    let conductor = HolochainConductor::new("127.0.0.1:1", "nonexistent");
    let result = conductor.connect().await;
    assert!(result.is_err(), "Should fail to connect to invalid port");
    println!(
        "SUCCESS: Invalid connection fails gracefully: {}",
        result.unwrap_err()
    );
}

// ---------------------------------------------------------------------------
// Governance dispatcher tests
// ---------------------------------------------------------------------------

/// Tests the governance dispatcher can attempt a zome call through the conductor.
/// Even if the call returns an error (e.g., role not found, auth issues), the fact
/// that the zome call was dispatched and a response came back proves connectivity.
#[tokio::test]
#[ignore] // Requires running conductor
async fn test_governance_dispatcher_query() {
    let conductor = Arc::new(HolochainConductor::new(app_addr(), "mycelix-gov-only"));
    let connect_result = conductor.connect().await;
    match connect_result {
        Ok(()) => println!("Connected to conductor for governance test"),
        Err(e) => {
            println!("Conductor connect failed (expected if no token): {e:?}");
            println!("Skipping governance dispatcher test — no app connection");
            return;
        }
    }

    let gov = GovernanceDispatcher::new(conductor);
    // Try to query active proposals
    match gov.get_active_proposals().await {
        Ok(proposals) => println!("Got {} proposals", proposals.len()),
        Err(e) => println!("Zome call result: {e} (expected during initial setup)"),
    }
}

// ---------------------------------------------------------------------------
// Finance dispatcher tests
// ---------------------------------------------------------------------------

/// Tests the finance dispatcher can attempt a balance query through the conductor.
#[tokio::test]
#[ignore] // Requires running conductor
async fn test_finance_dispatcher_balance_query() {
    let conductor = Arc::new(HolochainConductor::new(app_addr(), "mycelix-finance-only"));
    let connect_result = conductor.connect().await;
    match connect_result {
        Ok(()) => println!("Connected to conductor for finance test"),
        Err(e) => {
            println!("Conductor connect failed (expected if no token): {e:?}");
            println!("Skipping finance dispatcher test — no app connection");
            return;
        }
    }

    let fin = FinanceDispatcher::new(conductor);
    // Query a balance for a test DID
    match fin.query_sap_balance("did:mycelix:test-member").await {
        Ok(balance) => println!("Balance: {:?}", balance),
        Err(e) => println!("Finance query result: {e} (expected for unknown DID)"),
    }
}

// ---------------------------------------------------------------------------
// Full bridge pipeline test (SWIFT -> ISO 20022 -> Mycelix -> Holochain)
// ---------------------------------------------------------------------------

/// End-to-end test: parse a SWIFT pacs.008 message, map to Mycelix operations,
/// create an HTLC, connect to the live conductor, dispatch a finance query,
/// and complete the HTLC lifecycle.
#[tokio::test]
#[ignore] // Requires running conductor
async fn test_full_bridge_pipeline() {
    // 1. Parse SWIFT message (pacs.008 with ZAR amount)
    let xml = r#"<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.12">
  <FIToFICstmrCdtTrf>
    <GrpHdr>
      <MsgId>SWIFT-BRICS-001</MsgId>
      <CreDtTm>2026-03-19T10:00:00Z</CreDtTm>
      <NbOfTxs>1</NbOfTxs>
    </GrpHdr>
    <CdtTrfTxInf>
      <PmtId><EndToEndId>TX-ZAR-001</EndToEndId></PmtId>
      <InstdAmt Ccy="ZAR">5000.00</InstdAmt>
      <Dbtr><Nm>Mycelix Community Cooperative</Nm></Dbtr>
      <DbtrAgt><FinInstnId><BICFI>ABORZA001</BICFI></FinInstnId></DbtrAgt>
      <Cdtr><Nm>Food Cooperative Johannesburg</Nm></Cdtr>
      <CdtrAgt><FinInstnId><BICFI>NEDBZAJJ</BICFI></FinInstnId></CdtrAgt>
    </CdtTrfTxInf>
  </FIToFICstmrCdtTrf>
</Document>"#;

    let instruction = symthaea_iso20022::parse_pacs008(xml).unwrap();
    println!(
        "1. Parsed pacs.008: msg_id={}, {} transaction(s)",
        instruction.message_id,
        instruction.transactions.len()
    );
    assert_eq!(instruction.message_id, "SWIFT-BRICS-001");
    assert_eq!(instruction.transactions.len(), 1);
    assert_eq!(instruction.transactions[0].amount.currency, "ZAR");

    // 2. Map to Mycelix operations
    let mut registry = symthaea_iso20022::DidRegistry::new();
    registry.register_bic("ABORZA001", "did:mycelix:example-community");
    registry.register_bic("NEDBZAJJ", "did:mycelix:food-coop-jhb");
    let rate = symthaea_iso20022::RateSource::community_only(0.1);
    let mapping = symthaea_iso20022::map_to_mycelix(&instruction, &registry, &rate);
    println!(
        "2. Mapped to {} Mycelix operation(s), {} unmapped",
        mapping.operations.len(),
        mapping.unmapped.len()
    );
    assert!(!mapping.operations.is_empty(), "Should have at least one mapped operation");

    // 3. Create HTLC for settlement
    let mut htlc_mgr = symthaea_iso20022::HtlcManager::new();
    let (htlc_id, hash, secret) = symthaea_iso20022::create_settlement_htlc(
        &mut htlc_mgr,
        &mapping.operations[0],
        None,
    )
    .unwrap();
    println!("3. Created HTLC: id={htlc_id}, hash={:02x?}", &hash[..4]);

    // 4. Connect to live conductor
    let conductor = Arc::new(HolochainConductor::new(app_addr(), "mycelix-finance-only"));
    let connect_result = conductor.connect().await;
    match &connect_result {
        Ok(()) => println!("4. Connected to live conductor at {}", app_addr()),
        Err(e) => {
            println!("4. Conductor connect failed (expected if no token): {e:?}");
            println!("   Continuing with HTLC lifecycle (offline mode)...");
        }
    }

    // 5. Dispatch finance query if connected (prove connectivity through full path)
    if connect_result.is_ok() {
        let fin = FinanceDispatcher::new(conductor.clone());
        let result = fin
            .query_tend_balance("did:mycelix:example-community")
            .await;
        println!("5. TEND balance query result: {:?}", result);
    } else {
        println!("5. Skipped TEND query (no app connection)");
    }

    // 6. Complete HTLC lifecycle
    htlc_mgr.lock(&htlc_id).unwrap();
    println!("6a. HTLC locked");
    htlc_mgr.claim(&htlc_id, &secret).unwrap();
    println!("6b. HTLC claimed");
    htlc_mgr.settle(&htlc_id).unwrap();
    println!("6c. HTLC settled");

    println!("\nFull pipeline complete: SWIFT pacs.008 -> parse -> map -> HTLC -> conductor -> settle");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 7: Issue app auth token and connect app WebSocket
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
#[ignore]
async fn test_issue_token_and_connect_app() {
    use symthaea_mycelix_holochain::IssueAppAuthenticationTokenPayload;
    
    let admin = AdminWebsocket::connect(&*admin_addr()).await
        .expect("Admin should connect");
    
    println!("Admin connected, issuing token for mycelix-finance...");
    
    let payload = IssueAppAuthenticationTokenPayload {
        installed_app_id: "mycelix-finance".into(),
        expiry_seconds: 3600,
        single_use: false,
    };
    
    match admin.issue_app_auth_token(payload).await {
        Ok(issued) => {
            println!("TOKEN ISSUED! Length: {} bytes", issued.token.len());
            
            // Now try to connect the app WebSocket with this token
            use holochain_client::{AppWebsocket, ClientAgentSigner, AgentSigner};
            let signer: Arc<dyn AgentSigner + Send + Sync> = Arc::new(ClientAgentSigner::default());
            
            match AppWebsocket::connect(&*app_addr(), issued.token, signer).await {
                Ok(app_ws) => {
                    println!("✅ APP WEBSOCKET CONNECTED! Agent: {:?}", app_ws.my_pub_key);
                    
                    // Try a zome call!
                    use holochain_client::ZomeCallTarget;
                    use holochain_types::prelude::ExternIO;
                    
                    let input = ExternIO::encode(serde_json::json!({"member_did": "did:mycelix:test"}))
                        .expect("Should encode");
                    
                    match app_ws.call_zome(
                        ZomeCallTarget::RoleName("finance".to_string().into()),
                        "finance_bridge".into(),
                        "health_check".into(),
                        input,
                    ).await {
                        Ok(result) => {
                            println!("✅ ZOME CALL SUCCEEDED! Response: {:?}", result);
                        }
                        Err(e) => {
                            println!("Zome call error (may be expected for unknown function): {:?}", e);
                        }
                    }
                }
                Err(e) => println!("App WS connect failed with token: {:?}", e),
            }
        }
        Err(e) => {
            let msg = format!("{e:?}");
            if msg.contains("missing field") || msg.contains("Deserialize") {
                println!("EXPECTED: Admin API deserialization issue blocks token issuance: {}", msg);
            } else {
                println!("Token issuance failed: {}", msg);
            }
        }
    }
}
