// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Fabrication — Printers Sweettest
//!
//! Integration tests for printer registration, discovery, compatibility,
//! availability management, and authorization enforcement.
//!
//! ## Running
//! ```bash
//! cd mycelix-workspace/happs/fabrication/tests/sweettest
//! cargo test --release --test sweettest_printers -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

use fabrication_sweettest::common::*;

// ============================================================================
// Mirror types — printers coordinator
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterPrinterInput {
    pub name: String,
    pub location: Option<GeoLocation>,
    pub printer_type: String,
    pub capabilities: PrinterCapabilities,
    pub materials_available: Vec<String>,
    pub rates: Option<serde_json::Value>,
}

/// Mirror of fabrication_common's `GeoLocation`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GeoLocation {
    pub geohash: String,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
    pub city: Option<String>,
    pub region: Option<String>,
    pub country: String,
}

/// Mirror of fabrication_common's `PrinterCapabilities`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PrinterCapabilities {
    pub build_volume: BuildVolume,
    pub layer_heights: Vec<f32>,
    pub nozzle_diameters: Vec<f32>,
    pub heated_bed: bool,
    pub enclosure: bool,
    pub multi_material: Option<u8>,
    pub max_temp_hotend: u16,
    pub max_temp_bed: u16,
    pub features: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BuildVolume {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdatePrinterInput {
    pub original_action_hash: ActionHash,
    pub name: Option<String>,
    pub location: Option<GeoLocation>,
    pub capabilities: Option<PrinterCapabilities>,
    pub materials_available: Option<Vec<String>>,
    pub rates: Option<serde_json::Value>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateAvailabilityInput {
    pub printer_hash: ActionHash,
    pub status: String,
    pub message: Option<String>,
    pub eta_available: Option<u32>,
    pub current_job: Option<ActionHash>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MyPrintersInput {
    pub pagination: Option<PaginationInput>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetAvailablePrintersInput {
    pub pagination: Option<PaginationInput>,
}

// ============================================================================
// Helper functions
// ============================================================================

fn test_capabilities() -> PrinterCapabilities {
    PrinterCapabilities {
        build_volume: BuildVolume {
            x: 220.0,
            y: 220.0,
            z: 250.0,
        },
        layer_heights: vec![0.1, 0.15, 0.2, 0.3],
        nozzle_diameters: vec![0.4],
        heated_bed: true,
        enclosure: false,
        multi_material: None,
        max_temp_hotend: 300,
        max_temp_bed: 100,
        features: vec!["AutoLeveling".to_string()],
    }
}

// ============================================================================
// Printer Registration Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_printer_register_and_get() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = RegisterPrinterInput {
        name: "Prusa MK4S".to_string(),
        location: Some(GeoLocation {
            geohash: "9vg4p8".to_string(),
            lat: Some(32.948),
            lon: Some(-96.730),
            city: Some("Richardson".to_string()),
            region: Some("TX".to_string()),
            country: "US".to_string(),
        }),
        printer_type: "FDM".to_string(),
        capabilities: test_capabilities(),
        materials_available: vec!["PLA".to_string(), "PETG".to_string()],
        rates: None,
    };

    let record: Record = conductor
        .call(
            &alice.zome("printers_coordinator"),
            "register_printer",
            input,
        )
        .await;

    assert_eq!(record.action().author(), alice.agent_pubkey());

    // Retrieve it
    let printer_hash = record.action_address().clone();
    let retrieved: Option<Record> = conductor
        .call(
            &alice.zome("printers_coordinator"),
            "get_printer",
            printer_hash,
        )
        .await;

    assert!(retrieved.is_some());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_printer_update_requires_owner() {
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let mut alice_conductor = SweetConductor::from_standard_config().await;
    let mut bob_conductor = SweetConductor::from_standard_config().await;

    let (alice,) = alice_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();
    let (bob,) = bob_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    SweetConductor::exchange_peer_info([&alice_conductor, &bob_conductor]).await;

    // Alice registers a printer
    let input = RegisterPrinterInput {
        name: "Alice's Printer".to_string(),
        location: None,
        printer_type: "FDM".to_string(),
        capabilities: test_capabilities(),
        materials_available: vec!["PLA".to_string()],
        rates: None,
    };

    let record: Record = alice_conductor
        .call(
            &alice.zome("printers_coordinator"),
            "register_printer",
            input,
        )
        .await;

    let printer_hash = record.action_address().clone();

    // Wait for gossip propagation
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Bob tries to update — should fail
    let update_input = UpdatePrinterInput {
        original_action_hash: printer_hash,
        name: Some("Bob's Hijack".to_string()),
        location: None,
        capabilities: None,
        materials_available: None,
        rates: None,
    };

    let bob_result: Result<Record, _> = bob_conductor
        .call_fallible(
            &bob.zome("printers_coordinator"),
            "update_printer",
            update_input,
        )
        .await;

    assert!(
        bob_result.is_err(),
        "Bob should not be able to update Alice's printer"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_get_my_printers_paginated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Register 3 printers
    for i in 0..3 {
        let input = RegisterPrinterInput {
            name: format!("Printer #{}", i),
            location: None,
            printer_type: "FDM".to_string(),
            capabilities: test_capabilities(),
            materials_available: vec!["PLA".to_string()],
            rates: None,
        };

        let _: Record = conductor
            .call(
                &alice.zome("printers_coordinator"),
                "register_printer",
                input,
            )
            .await;
    }

    // Get all (no pagination)
    let all: serde_json::Value = conductor
        .call(
            &alice.zome("printers_coordinator"),
            "get_my_printers",
            MyPrintersInput { pagination: None },
        )
        .await;

    let total = all.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
    assert!(total >= 3, "Should have at least 3 printers, got {}", total);

    // Get paginated (limit 2)
    let page1: serde_json::Value = conductor
        .call(
            &alice.zome("printers_coordinator"),
            "get_my_printers",
            MyPrintersInput {
                pagination: Some(PaginationInput {
                    offset: 0,
                    limit: 2,
                }),
            },
        )
        .await;

    let items = page1
        .get("items")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    assert_eq!(items, 2, "Page 1 should have 2 items");

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_printer_deactivation() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = RegisterPrinterInput {
        name: "Deactivation Test".to_string(),
        location: None,
        printer_type: "FDM".to_string(),
        capabilities: test_capabilities(),
        materials_available: vec!["PLA".to_string()],
        rates: None,
    };

    let record: Record = conductor
        .call(
            &alice.zome("printers_coordinator"),
            "register_printer",
            input,
        )
        .await;

    let printer_hash = record.action_address().clone();

    // Deactivate
    let deactivated_hash: ActionHash = conductor
        .call(
            &alice.zome("printers_coordinator"),
            "deactivate_printer",
            printer_hash,
        )
        .await;

    assert!(!deactivated_hash.get_raw_39().is_empty());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_update_availability() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = RegisterPrinterInput {
        name: "Availability Test".to_string(),
        location: None,
        printer_type: "FDM".to_string(),
        capabilities: test_capabilities(),
        materials_available: vec!["PLA".to_string()],
        rates: None,
    };

    let record: Record = conductor
        .call(
            &alice.zome("printers_coordinator"),
            "register_printer",
            input,
        )
        .await;

    let printer_hash = record.action_address().clone();

    // Update availability to Busy
    let availability_input = UpdateAvailabilityInput {
        printer_hash,
        status: "Busy".to_string(),
        message: Some("Printing large job".to_string()),
        eta_available: Some(120),
        current_job: None,
    };

    let updated: Record = conductor
        .call(
            &alice.zome("printers_coordinator"),
            "update_availability",
            availability_input,
        )
        .await;

    assert_eq!(updated.action().author(), alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
