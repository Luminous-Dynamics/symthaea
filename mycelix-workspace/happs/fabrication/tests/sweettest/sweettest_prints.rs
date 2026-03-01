//! # Fabrication — Prints & PoGF Sweettest
//!
//! Integration tests for print job lifecycle, PoGF scoring,
//! Cincinnati monitoring, and pagination.
//!
//! ## Running
//! ```bash
//! cd mycelix-workspace/happs/fabrication/tests/sweettest
//! cargo test --release --test sweettest_prints -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

use fabrication_sweettest::common::*;

// ============================================================================
// Mirror types — printers coordinator (needed for setup)
// ============================================================================

/// Mirror of printers coordinator's `RegisterPrinterInput`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterPrinterInput {
    pub name: String,
    pub location: Option<serde_json::Value>,
    pub printer_type: String,
    pub capabilities: PrinterCapabilities,
    pub materials_available: Vec<String>,
    pub rates: Option<serde_json::Value>,
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

// ============================================================================
// Mirror types — prints coordinator
// ============================================================================

/// Mirror of prints coordinator's `CreatePrintJobInput`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreatePrintJobInput {
    pub design_hash: ActionHash,
    pub printer_hash: ActionHash,
    pub settings: PrintSettings,
    pub energy_source: Option<String>,
    pub material_passport: Option<serde_json::Value>,
}

/// Mirror of fabrication_common's `PrintSettings`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PrintSettings {
    pub layer_height: f32,
    pub infill_percent: u8,
    pub material: String,
    pub supports: bool,
    pub raft: bool,
    pub print_speed: Option<u16>,
    pub temperatures: TemperatureSettings,
    pub custom_gcode: Option<String>,
}

/// Mirror of fabrication_common's `TemperatureSettings`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TemperatureSettings {
    pub hotend: u16,
    pub bed: Option<u16>,
    pub chamber: Option<u16>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateProgressInput {
    pub job_hash: ActionHash,
    pub progress_percent: u8,
    pub current_layer: Option<u32>,
    pub material_used_grams: Option<f32>,
}

/// Mirror of prints coordinator's `CompletePrintInput`.
/// `result` is a `PrintResult` enum. Simple variants serialize as strings (e.g. `"Success"`).
/// Variants with data use tagged form (e.g. `{"Failed": {"Warping": null}}`).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompletePrintInput {
    pub job_hash: ActionHash,
    pub result: serde_json::Value,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CancelPrintInput {
    pub job_hash: ActionHash,
    pub reason: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StartCincinnatiInput {
    pub job_hash: ActionHash,
    pub total_layers: u32,
    pub sampling_rate_hz: u32,
}

/// Create a design and printer, returning their hashes for print job tests.
async fn setup_design_and_printer(
    conductor: &SweetConductor,
    cell: &SweetCell,
) -> (ActionHash, ActionHash) {
    let design_input = CreateDesignInput {
        title: "Print Test Part".to_string(),
        description: "A part for print job testing".to_string(),
        category: "Parts".to_string(),
        intent_vector: None,
        parametric_schema: None,
        constraint_graph: None,
        material_compatibility: vec![],
        circularity_score: 0.5,
        embodied_energy_kwh: 1.0,
        repair_manifest: None,
        license: serde_json::json!("PublicDomain"),
        safety_class: "Class1Functional".to_string(),
    };

    let design_record: Record = conductor
        .call(
            &cell.zome("designs_coordinator"),
            "create_design",
            design_input,
        )
        .await;

    let printer_input = RegisterPrinterInput {
        name: "Test Printer".to_string(),
        location: None,
        printer_type: "FDM".to_string(),
        capabilities: PrinterCapabilities {
            build_volume: BuildVolume { x: 220.0, y: 220.0, z: 250.0 },
            layer_heights: vec![0.1, 0.15, 0.2, 0.3],
            nozzle_diameters: vec![0.4],
            heated_bed: true,
            enclosure: false,
            multi_material: None,
            max_temp_hotend: 300,
            max_temp_bed: 100,
            features: vec!["AutoLeveling".to_string()],
        },
        materials_available: vec!["PLA".to_string(), "PETG".to_string()],
        rates: None,
    };

    let printer_record: Record = conductor
        .call(
            &cell.zome("printers_coordinator"),
            "register_printer",
            printer_input,
        )
        .await;

    (
        design_record.action_address().clone(),
        printer_record.action_address().clone(),
    )
}

// ============================================================================
// Print Job Lifecycle Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_print_job_create_and_accept() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let (design_hash, printer_hash) = setup_design_and_printer(&conductor, &alice).await;

    let job_input = CreatePrintJobInput {
        design_hash,
        printer_hash,
        settings: PrintSettings {
            layer_height: 0.2,
            infill_percent: 20,
            material: "PLA".to_string(),
            supports: false,
            raft: false,
            print_speed: None,
            temperatures: TemperatureSettings {
                hotend: 210,
                bed: Some(60),
                chamber: None,
            },
            custom_gcode: None,
        },
        energy_source: Some("Solar".to_string()),
        material_passport: None,
    };

    let job_record: Record = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "create_print_job",
            job_input,
        )
        .await;

    assert_eq!(job_record.action().author(), alice.agent_pubkey());

    let job_hash = job_record.action_address().clone();

    // Accept the job
    let accepted: Record = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "accept_print_job",
            job_hash.clone(),
        )
        .await;

    assert_eq!(accepted.action().author(), alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_print_job_full_lifecycle() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let (design_hash, printer_hash) = setup_design_and_printer(&conductor, &alice).await;

    // Create job
    let job_input = CreatePrintJobInput {
        design_hash: design_hash.clone(),
        printer_hash,
        settings: PrintSettings {
            layer_height: 0.15,
            infill_percent: 30,
            material: "PETG".to_string(),
            supports: true,
            raft: false,
            print_speed: None,
            temperatures: TemperatureSettings {
                hotend: 240,
                bed: Some(80),
                chamber: None,
            },
            custom_gcode: None,
        },
        energy_source: None,
        material_passport: None,
    };

    let job_record: Record = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "create_print_job",
            job_input,
        )
        .await;

    let job_hash = job_record.action_address().clone();

    // Accept → Start → Progress → Complete
    let _: Record = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "accept_print_job",
            job_hash.clone(),
        )
        .await;

    let _: Record = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "start_print",
            job_hash.clone(),
        )
        .await;

    let _: Record = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "update_print_progress",
            UpdateProgressInput {
                job_hash: job_hash.clone(),
                progress_percent: 50,
                current_layer: Some(120),
                material_used_grams: Some(15.5),
            },
        )
        .await;

    let completed: Record = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "complete_print",
            CompletePrintInput {
                job_hash: job_hash.clone(),
                result: serde_json::json!("Success"),
            },
        )
        .await;

    assert_eq!(completed.action().author(), alice.agent_pubkey());

    // Verify statistics
    let stats: serde_json::Value = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "get_print_statistics",
            design_hash,
        )
        .await;

    let total = stats
        .get("total_prints")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert!(total >= 1, "Should have at least 1 print, got {}", total);

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_print_job_cancel() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let (design_hash, printer_hash) = setup_design_and_printer(&conductor, &alice).await;

    let job_input = CreatePrintJobInput {
        design_hash,
        printer_hash,
        settings: PrintSettings {
            layer_height: 0.2,
            infill_percent: 20,
            material: "PLA".to_string(),
            supports: false,
            raft: false,
            print_speed: None,
            temperatures: TemperatureSettings {
                hotend: 210,
                bed: Some(60),
                chamber: None,
            },
            custom_gcode: None,
        },
        energy_source: None,
        material_passport: None,
    };

    let job_record: Record = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "create_print_job",
            job_input,
        )
        .await;

    let job_hash = job_record.action_address().clone();

    // Cancel the job
    let cancelled: Record = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "cancel_print",
            CancelPrintInput {
                job_hash,
                reason: "Material out of stock".to_string(),
            },
        )
        .await;

    assert_eq!(cancelled.action().author(), alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_get_my_print_jobs_paginated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let (design_hash, printer_hash) = setup_design_and_printer(&conductor, &alice).await;

    // Create 3 print jobs
    for _ in 0..3 {
        let job_input = CreatePrintJobInput {
            design_hash: design_hash.clone(),
            printer_hash: printer_hash.clone(),
            settings: PrintSettings {
                layer_height: 0.2,
                infill_percent: 20,
                material: "PLA".to_string(),
                supports: false,
                raft: false,
                print_speed: None,
                temperatures: TemperatureSettings {
                    hotend: 210,
                    bed: Some(60),
                    chamber: None,
                },
                custom_gcode: None,
            },
            energy_source: None,
            material_passport: None,
        };

        let _: Record = conductor
            .call(
                &alice.zome("prints_coordinator"),
                "create_print_job",
                job_input,
            )
            .await;
    }

    // Get all
    let all: serde_json::Value = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "get_my_print_jobs",
            AgentPaginationInput { pagination: None },
        )
        .await;

    let total = all.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
    assert!(total >= 3, "Should have at least 3 jobs, got {}", total);

    // Get paginated
    let page: serde_json::Value = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "get_my_print_jobs",
            AgentPaginationInput {
                pagination: Some(PaginationInput {
                    offset: 0,
                    limit: 2,
                }),
            },
        )
        .await;

    let items = page
        .get("items")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    assert_eq!(items, 2, "Page should have 2 items");

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cincinnati_monitoring_start() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let (design_hash, printer_hash) = setup_design_and_printer(&conductor, &alice).await;

    // Create and start a job
    let job_input = CreatePrintJobInput {
        design_hash,
        printer_hash,
        settings: PrintSettings {
            layer_height: 0.2,
            infill_percent: 20,
            material: "PLA".to_string(),
            supports: false,
            raft: false,
            print_speed: None,
            temperatures: TemperatureSettings {
                hotend: 210,
                bed: Some(60),
                chamber: None,
            },
            custom_gcode: None,
        },
        energy_source: None,
        material_passport: None,
    };

    let job_record: Record = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "create_print_job",
            job_input,
        )
        .await;

    let job_hash = job_record.action_address().clone();

    let _: Record = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "accept_print_job",
            job_hash.clone(),
        )
        .await;

    let _: Record = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "start_print",
            job_hash.clone(),
        )
        .await;

    // Start Cincinnati monitoring
    let session: Record = conductor
        .call(
            &alice.zome("prints_coordinator"),
            "start_cincinnati_monitoring",
            StartCincinnatiInput {
                job_hash,
                total_layers: 240,
                sampling_rate_hz: 10,
            },
        )
        .await;

    assert_eq!(session.action().author(), alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
