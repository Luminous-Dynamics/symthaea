// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Planarian Cryptic Memory & Tangential Action Spaces (TAS) Simulation.
//!
//! Validates the Reactive-Analytic Hybrid infrastructure by modeling
//! bioelectric state transitions in planarian fragments.
//!
//! Scenario:
//! 1. Wild-Type (WT): Unified tissue, all cells hyperpolarized (Target State).
//! 2. Cryptic Double-Head (CDH): A hidden cluster of depolarized cells
//!    in the tail that will induce a second head upon regeneration.

use std::fs::File;
use std::io::Write;
use std::thread;
use std::time::Duration;
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_morphogenesis::{
    ActiveMorphoController, AssociativeMemory, BioelectricIngest, ConformalGeometricEngine,
    MeaPacket, MorphoMeshAdapter, MorphoTopologyConfig, MorphoTopologySupervisor, MorphoVerdict,
    TissueSnapshot,
};

const NUM_CELLS: usize = 128;
const DIM: usize = 16384;
const SEED: u64 = 42;

fn main() {
    println!("🧪 Planarian TAS Simulation (Phase 1: Organic Mesh + Telemetry Ingest)...");
    println!("Architecture: Reactive-Analytic Hybrid (500Hz Reflex + Async TDA Supervisor)");

    // 1. Setup Basis Vectors & States
    let v_hyper = ContinuousHV::random(DIM, SEED.wrapping_add(5000));
    let v_depol = ContinuousHV::random(DIM, SEED.wrapping_add(6000));

    // 2. Setup Organic Mesh (MEA Geometry)
    let mut coords = Vec::with_capacity(NUM_CELLS);
    for i in 0..NUM_CELLS {
        let x = (i % 16) as f32 + (i as f32 * 0.1).sin() * 0.2; // Add organic jitter
        let y = (i / 16) as f32 + (i as f32 * 0.1).cos() * 0.2;
        coords.push((x, y));
    }

    // Create adapter with distance-preserving graph encoding
    let adapter =
        MorphoMeshAdapter::new_from_mea(DIM, &coords, 1.5, SEED, v_hyper.clone(), v_depol.clone());
    let spatial_hvs = adapter.spatial_coordinates();

    // 3. Instantiate Morpho-Supervisor, Memory & Active Controller
    let config = MorphoTopologyConfig {
        min_persistence: 0.05,
        num_scales: 15,
        dim: DIM,
        target_beta_0: 1.0,
    };
    let supervisor = MorphoTopologySupervisor::new(config);
    let clean_up_memory = AssociativeMemory::new(DIM, vec![v_hyper.clone(), v_depol.clone()]);
    let controller = ActiveMorphoController::new(
        DIM,
        vec![
            v_hyper.clone(),
            ContinuousHV::random(DIM, 999),
            v_depol.clone().inverse(),
        ],
    );

    // 4. Simulate Wild-Type (Healthy) via Telemetry Ingest
    println!("\n--- Step 1: Ingesting WT Telemetry (Healthy Fragment) ---");
    let wt_packet = MeaPacket {
        timestamp_ms: 0,
        electrode_voltages: vec![1.0; NUM_CELLS], // All hyperpolarized (1.0)
    };
    let wt_tissue_hv = adapter.ingest_mea_packet(&wt_packet);

    // Reconstruct cell list for supervisor verification
    let wt_cells: Vec<ContinuousHV> = (0..NUM_CELLS)
        .map(|i| spatial_hvs[i].bind(&v_hyper))
        .collect();

    supervisor.submit_snapshot(TissueSnapshot {
        state_hv: wt_tissue_hv.clone(),
        cell_hvs: wt_cells,
    });

    // Wait for verdict
    let mut verdict = None;
    for _ in 0..100 {
        thread::sleep(Duration::from_millis(50));
        verdict = supervisor.poll_verdict();
        if verdict.is_some() {
            break;
        }
    }
    if let Some(MorphoVerdict::Unified { phi }) = verdict {
        println!(
            "✅ Supervisor: WT Tissue Unified (β₀ = 1.0, Φ = {:.4})",
            phi
        );
    } else {
        println!("⚠️ Supervisor verdict (WT): {:?}", verdict);
    }

    // 5. Simulate Cryptic Double-Head (CDH)
    println!("\n--- Step 2: Ingesting CDH Telemetry (Hidden Head Inducer) ---");
    let mut cdh_voltages = vec![1.0; NUM_CELLS];
    let mut actual_rogue_indices = Vec::new();
    for i in 0..NUM_CELLS {
        let x = i % 16;
        let y = i / 16;
        // Rogue cluster in bottom-right of 16x8 grid
        if x >= 12 && y >= 4 {
            cdh_voltages[i] = -1.0; // Depolarized (-1.0)
            actual_rogue_indices.push(i);
        }
    }

    let cdh_packet = MeaPacket {
        timestamp_ms: 100,
        electrode_voltages: cdh_voltages,
    };
    let cdh_tissue_hv = adapter.ingest_mea_packet(&cdh_packet);

    // Reconstruct cell list for TDA verification
    let cdh_cells: Vec<ContinuousHV> = (0..NUM_CELLS)
        .map(|i| {
            let state = if actual_rogue_indices.contains(&i) {
                &v_depol
            } else {
                &v_hyper
            };
            spatial_hvs[i].bind(state)
        })
        .collect();

    supervisor.submit_snapshot(TissueSnapshot {
        state_hv: cdh_tissue_hv.clone(),
        cell_hvs: cdh_cells.clone(),
    });

    // Wait for verdict
    let mut verdict_cdh = None;
    for _ in 0..100 {
        thread::sleep(Duration::from_millis(50));
        verdict_cdh = supervisor.poll_verdict();
        if verdict_cdh.is_some() {
            break;
        }
    }

    if let Some(v) = verdict_cdh {
        match v {
            MorphoVerdict::FragmentationAlarm {
                decoupled_voids,
                phi,
                ..
            } => {
                println!(
                    "🚨 Supervisor: FRAGMENTATION DETECTED! (β₀ - 1) = {:.2}, Φ = {:.4}",
                    decoupled_voids, phi
                );

                // 6. Localization
                println!("\n--- Step 3: Localization via Organic Mesh Unbinding ---");
                let initial_rogue = MorphoTopologySupervisor::isolate_rogue_cells(
                    &cdh_tissue_hv,
                    &spatial_hvs,
                    &v_hyper,
                    Some(&clean_up_memory),
                );
                println!("   Initial Rogue Nodes detected: {}", initial_rogue.len());

                // 7. Active Steering with Biophysical Cost
                println!("\n--- Step 4: Local Steering with Biophysical Cost (ΔE ∝ Δh²) ---");
                let mut rescued_cells = cdh_cells.clone();
                for &idx in &initial_rogue {
                    let current_cell_hv = &cdh_cells[idx];
                    let (optimal_shift, g_score) =
                        controller.select_optimal_shift(current_cell_hv, &v_hyper);

                    // Replace with steered state
                    rescued_cells[idx] = spatial_hvs[idx].bind(&optimal_shift);

                    if idx == initial_rogue[0] {
                        println!(
                            "   First Cell Rescue: G-Score = {:.4} (includes metabolic penalty)",
                            g_score
                        );
                    }
                }
                println!(
                    "   Executed EFE-optimized interventions across {} nodes.",
                    initial_rogue.len()
                );

                let rescued_refs: Vec<&ContinuousHV> = rescued_cells.iter().collect();
                let rescued_tissue_hv = ContinuousHV::bundle(&rescued_refs).normalize();

                // 8. Final Validation
                println!("\n--- Step 5: Post-Rescue Verification ---");
                let post_rescue_rogue = MorphoTopologySupervisor::isolate_rogue_cells(
                    &rescued_tissue_hv,
                    &spatial_hvs,
                    &v_hyper,
                    Some(&clean_up_memory),
                );

                println!(
                    "   Post-Rescue Rogue Nodes detected: {}",
                    post_rescue_rogue.len()
                );
                if post_rescue_rogue.len() < initial_rogue.len() {
                    println!("✅ RESCUE SUCCESSFUL: Organic mesh re-integrated.");
                }
            }
            MorphoVerdict::Unified { phi } => println!(
                "❌ Supervisor: FAILED to detect cryptic head! (Φ = {:.4})",
                phi
            ),
        }
    } else {
        println!("⏳ Supervisor timeout (CDH)");
    }

    println!("\n--- Step 6: Morphological Growth Simulation (Conformal Geometric HDC) ---");
    let cga = ConformalGeometricEngine::new(DIM, SEED + 777);

    // 1. Define original fragment (1,1) with hyperpolarized state
    println!("   Embedding cellular fragment into 5D Conformal Space...");
    let p_orig = cga.embed_point(1.0, 1.0, 0.0);
    let tissue_cga = p_orig.bind(&v_hyper);

    // 2. Create Dilator (Growth Factor = 2.0)
    println!("   Applying Dilation Operator (Growth Factor = 2.0)...");
    let dilator_hv = cga.create_dilator(2.0);

    // Applying the growth operator as a linear transformation
    let grown_tissue = tissue_cga.bind(&dilator_hv);

    // 3. Verify: The grown tissue should now be found at (2.0, 2.0)
    println!("   Verifying Spatial Recovery at Scaled Coordinates...");
    let p_new = cga.embed_point(2.0, 2.0, 0.0);

    let recovery_sim = grown_tissue.similarity(&p_new.bind(&v_hyper));
    let chance_sim = grown_tissue.similarity(&p_orig.bind(&v_hyper));

    println!("   Similarity at (2.0, 2.0): {:.4} (Target)", recovery_sim);
    println!("   Similarity at (1.0, 1.0): {:.4} (Old)", chance_sim);

    if recovery_sim > chance_sim {
        println!("✅ GROWTH VALIDATED: Conformal algebraic growth tracked tissue shift natively.");
    }

    println!("\n--- Step 7: Real-World Dataset Ingest (Hansali/McMillen CSV) ---");
    // Create a synthetic McMillen-style CSV for demonstration
    let csv_path = "mcmillen_synthetic.csv";
    {
        let mut file = File::create(csv_path).unwrap();
        writeln!(file, "X,Y,Lifetime").unwrap();
        writeln!(file, "10.0,10.0,0.85").unwrap();
        writeln!(file, "15.0,10.0,0.82").unwrap();
        writeln!(file, "20.0,10.0,-0.95").unwrap(); // Rogue cell!
    }

    println!("   Parsing McMillen & Levin (2024) Tabular Format...");
    let records = BioelectricIngest::parse_mcmillen_csv(csv_path).unwrap();
    println!("   Ingested {} records from CSV.", records.len());

    let (_mea_adapter, snapshot) = BioelectricIngest::mcmillen_to_mesh(
        DIM,
        &records,
        SEED + 888,
        v_hyper.clone(),
        v_depol.clone(),
    );

    println!(
        "   Unified Tissue HV generated from real-world schema: {:.4} norm",
        snapshot.state_hv.norm()
    );

    // Submit to supervisor for final check
    supervisor.submit_snapshot(snapshot);
    thread::sleep(Duration::from_millis(200));
    if let Some(verdict) = supervisor.poll_verdict() {
        match verdict {
            MorphoVerdict::FragmentationAlarm { phi, .. } => {
                println!(
                    "✅ DATA INGEST VALIDATED: TDA correctly flagged rogue cell in CSV dataset (Φ = {:.4}).",
                    phi
                );
            }
            _ => println!("❌ DATA INGEST FAILED: TDA did not detect rogue cell in CSV."),
        }
    }

    println!("\n🏁 Simulation Complete.");
}
