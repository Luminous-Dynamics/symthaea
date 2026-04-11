// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Live conductor smoke tests for mycelix-identity.
//!
//! Requires the Holochain conductor to be running at ws://localhost:8888
//! with the mycelix_mail app installed (which includes the identity role).
//!
//! Run with:
//!   cargo test --features conductor_live -p mycelix-identity -- conductor_smoke
//!
//! Skip in CI by not enabling the `conductor_live` feature.

#[cfg(feature = "conductor_live")]
mod conductor_smoke {
    use holochain_client::{AdminWebsocket, AppWebsocket, ZomeCallTarget};
    use holochain_types::app::InstalledAppId;

    const APP_WS: &str = "ws://localhost:8888";
    const APP_ID: &str = "mycelix_mail";
    const IDENTITY_ROLE: &str = "identity";

    /// Connect to the app websocket and return it.
    async fn connect() -> AppWebsocket {
        AppWebsocket::connect(APP_WS)
            .await
            .expect("Could not connect to conductor at ws://localhost:8888 — is the conductor running?")
    }

    #[tokio::test]
    async fn test_get_my_did_returns_without_error() {
        let ws = connect().await;
        let result = ws
            .call_zome(
                ZomeCallTarget::RoleName(IDENTITY_ROLE.into()),
                "did_registry".into(),
                "get_my_did".into(),
                ExternIO::encode(()).unwrap(),
            )
            .await;
        assert!(
            result.is_ok(),
            "did_registry.get_my_did() failed: {:?}", result
        );
        // Result is Option<Record> — either None (no DID yet) or Some(record)
        println!("get_my_did result: {:?}", result.unwrap());
    }

    #[tokio::test]
    async fn test_mfa_state_readable() {
        let ws = connect().await;
        let result = ws
            .call_zome(
                ZomeCallTarget::RoleName(IDENTITY_ROLE.into()),
                "mfa".into(),
                "get_mfa_state".into(),
                ExternIO::encode("self".to_string()).unwrap(),
            )
            .await;
        assert!(
            result.is_ok(),
            "mfa.get_mfa_state() failed: {:?}", result
        );
        println!("get_mfa_state result: {:?}", result.unwrap());
    }

    #[tokio::test]
    async fn test_reputation_aggregator_readable() {
        let ws = connect().await;
        let result = ws
            .call_zome(
                ZomeCallTarget::RoleName(IDENTITY_ROLE.into()),
                "reputation_aggregator".into(),
                "get_composite_reputation".into(),
                ExternIO::encode("self".to_string()).unwrap(),
            )
            .await;
        assert!(
            result.is_ok(),
            "reputation_aggregator.get_composite_reputation() failed: {:?}", result
        );
        println!("reputation result: {:?}", result.unwrap());
    }

    #[tokio::test]
    async fn test_bridge_verify_did_returns_bool() {
        let ws = connect().await;
        // First get our DID (may be None if not yet created)
        let did_result = ws
            .call_zome(
                ZomeCallTarget::RoleName(IDENTITY_ROLE.into()),
                "did_registry".into(),
                "get_my_did".into(),
                ExternIO::encode(()).unwrap(),
            )
            .await
            .expect("get_my_did failed");

        // If we have a DID, verify it via the bridge
        if let Ok(Some(_record)) = holochain_serialized_bytes::decode::<Option<holochain_types::record::Record>>(did_result.as_bytes()) {
            let verify_result = ws
                .call_zome(
                    ZomeCallTarget::RoleName(IDENTITY_ROLE.into()),
                    "bridge".into(),
                    "verify_did".into(),
                    ExternIO::encode("test-did".to_string()).unwrap(),
                )
                .await;
            assert!(verify_result.is_ok(), "bridge.verify_did() failed: {:?}", verify_result);
        } else {
            println!("Skipping verify_did — no DID created yet");
        }
    }
}
