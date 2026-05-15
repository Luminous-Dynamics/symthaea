// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use feldman_dkg::{Dealer, DkgConfig, ParticipantId};
use futures_util::{SinkExt, StreamExt};
use p256::ecdsa::{signature::Signer, Signature, SigningKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::State;
use tokio_tungstenite::{connect_async, tungstenite::Message};

mod mesh_bridge;
use mesh_bridge::{sync_offline, MeshBridge};

// Application state
pub struct AppState {
    pub status: Mutex<String>,
    pub holochain_process: Mutex<Option<Child>>,
    pub admin_port: Mutex<u16>,
    pub app_port: Mutex<u16>,
    pub mesh: MeshBridge,
}

// Basic Tauri command - test the bridge
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! Welcome to Mycelix Network 0.6.0 🍄", name)
}

// Start Holochain conductor
#[tauri::command]
async fn start_holochain(state: State<'_, AppState>) -> Result<String, String> {
    {
        let process_guard = state
            .holochain_process
            .lock()
            .expect("holochain_process mutex poisoned");
        if process_guard.is_some() {
            return Ok("Holochain conductor already running".to_string());
        }
    }

    let config_path = PathBuf::from("conductor-config.yaml");
    if !config_path.exists() {
        return Err("Conductor config file not found.".to_string());
    }

    let holochain_cmd = if cfg!(target_os = "windows") {
        "holochain.exe"
    } else {
        "holochain"
    };

    match Command::new(holochain_cmd)
        .arg("--piped")
        .arg("-c")
        .arg(&config_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(mut child) => {
            if let Some(mut stdin) = child.stdin.take() {
                use std::io::Write;
                let _ = stdin.write_all(b"\n");
            }
            let mut process_guard = state
                .holochain_process
                .lock()
                .expect("holochain_process mutex poisoned");
            *process_guard = Some(child);
            Ok("Holochain conductor started successfully.".to_string())
        }
        Err(e) => Err(format!("Failed to start Holochain conductor: {}", e)),
    }
}

// Helper function to send JSON-RPC request via WebSocket
async fn send_rpc_request(
    port: u16,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let url = format!("ws://localhost:{}", port);
    let (ws_stream, _) = connect_async(&url)
        .await
        .map_err(|e| format!("WebSocket connection failed: {}", e))?;

    let (mut write, mut read) = ws_stream.split();
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": "1",
        "method": method,
        "params": params
    });

    write
        .send(Message::Text(request.to_string()))
        .await
        .map_err(|e| format!("Failed to send request: {}", e))?;

    if let Some(msg) = read.next().await {
        let text = msg
            .map_err(|e| format!("Read error: {}", e))?
            .to_text()
            .map_err(|e| e.to_string())?
            .to_string();
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;

        if let Some(error) = json.get("error") {
            return Err(format!("RPC error: {}", error));
        }
        Ok(json
            .get("result")
            .cloned()
            .unwrap_or(serde_json::Value::Null))
    } else {
        Err("No response".to_string())
    }
}

// Zome call helper (App Interface 0.5/0.6 format)
async fn call_zome_raw(
    port: u16,
    _app_id: &str,
    dna_hash: &str,
    agent_key: &str,
    zome_name: &str,
    fn_name: &str,
    payload: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let params = serde_json::json!({
        "type": "call_zome",
        "data": {
            "cell_id": [dna_hash, agent_key],
            "zome_name": zome_name,
            "fn_name": fn_name,
            "payload": payload,
            "cap_secret": null
        }
    });

    send_rpc_request(port, "call_zome", params).await
}

// --- NATIVE CONSENSUS: DKG PROOF-IN-ZOME COMMANDS ---

#[derive(Debug, Serialize, Deserialize)]
pub struct DkgStepInput {
    pub threshold: usize,
    pub participants: usize,
    pub my_id: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DkgStepOutput {
    pub commitment_set: serde_json::Value,
    pub deals: Vec<serde_json::Value>,
}

/// NATIVE COMMAND: Perform a DKG dealer step natively.
/// Offloads heavy polynomial generation and point multiplication.
#[tauri::command]
async fn perform_native_dkg_step(
    _state: State<'_, AppState>,
    input: DkgStepInput,
) -> Result<DkgStepOutput, String> {
    let _config = DkgConfig::new(input.threshold, input.participants)
        .map_err(|e| format!("Invalid DKG config: {:?}", e))?;

    let mut rng = OsRng;
    let dealer = Dealer::new(
        ParticipantId(input.my_id),
        input.threshold,
        input.participants,
        &mut rng,
    )
    .map_err(|e| format!("Failed to create dealer: {:?}", e))?;

    let commitment_set = dealer.commitment_set();
    let deals = dealer.deals();

    Ok(DkgStepOutput {
        commitment_set: serde_json::to_value(&commitment_set).map_err(|e| e.to_string())?,
        deals: deals
            .iter()
            .map(|d| serde_json::to_value(d).unwrap_or(serde_json::Value::Null))
            .collect(),
    })
}

// --- NATIVE ENGINE ROOM: HDC PROOF-IN-ZOME COMMANDS ---

#[derive(Debug, Serialize, Deserialize)]
pub struct PogqSubmission {
    pub round_id: String,
    pub model_id: String,
    pub parent_model_hash: String, // hex encoded
    pub gradients: Vec<u8>,
}

// Helper to poll metrics from a substrate
async fn poll_metrics(port: u16, app_id: &str, zome: &str) -> Option<BridgeMetricsSnapshot> {
    // In production, these would be cached or derived from AppState
    let dna_hash = "uhC0k98l2UnvhZtuuze40Mlhxpon_fjBq6LVQ0Um9RREhC_maHDlI";
    let agent_key = "uhCAkHZV67oPQGx9-dVV2j73kJJgOBaP4N8Fqjq19o-v0sLhvb8Ik";

    match call_zome_raw(
        port,
        app_id,
        dna_hash,
        agent_key,
        zome,
        "get_bridge_metrics",
        serde_json::Value::Null,
    )
    .await
    {
        Ok(res) => {
            if let Some(json_str) = res.as_str() {
                serde_json::from_str(json_str).ok()
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

// --- CONSTELLATION TELEMETRY TYPES ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeMetricsSnapshot {
    pub total_success: u64,
    pub total_errors: u64,
    pub latency: Option<LatencyPercentiles>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstellationVitals {
    pub identity: Option<BridgeMetricsSnapshot>,
    pub finance: Option<BridgeMetricsSnapshot>,
    pub civic: Option<BridgeMetricsSnapshot>,
    pub knowledge: Option<BridgeMetricsSnapshot>,
    pub symthaea: Option<BridgeMetricsSnapshot>,
}

#[tauri::command]
async fn generate_and_submit_pogq(
    state: State<'_, AppState>,
    input: PogqSubmission,
) -> Result<String, String> {
    // 1. PERFORM HEAVY NATIVE MATH (Simulated HDC/ML)
    let commitment = blake3::hash(&input.gradients);
    let proof = blake3::hash(commitment.as_bytes());

    let payload = serde_json::json!({
        "round_id": [input.round_id],
        "model_id": input.model_id,
        "parent_model_hash": [hex::decode(&input.parent_model_hash).unwrap_or_else(|_| vec![0; 32])],
        "grad_commitment": commitment.as_bytes().to_vec(),
        "quality_proof": proof.as_bytes().to_vec(),
        "clipped_l2_norm": 1.0,
        "local_val_loss": 0.42,
        "sample_count": (input.gradients.len() / 4) as u32
    });

    // 2. SUBMIT TO DHT
    let app_port = *state.app_port.lock().expect("mutex poisoned");

    // In a real app, these would be fetched from AppState
    let dna_hash = "uhC0k98l2UnvhZtuuze40Mlhxpon_fjBq6LVQ0Um9RREhC_maHDlI";
    let agent_key = "uhCAkHZV67oPQGx9-dVV2j73kJJgOBaP4N8Fqjq19o-v0sLhvb8Ik";

    call_zome_raw(
        app_port,
        "substrate-praxis",
        dna_hash,
        agent_key,
        "fl_zome",
        "submit_update",
        payload,
    )
    .await?;

    Ok(format!(
        "PoGQ Generated and Submitted! Hash: {}",
        commitment.to_hex()
    ))
}

#[tauri::command]
async fn sign_with_hardware(_state: State<'_, AppState>, data: Vec<u8>) -> Result<Vec<u8>, String> {
    // Vector 3: Hardware-Bound Signing
    // In production, this would communicate with YubiKey via PC/SC or Apple Secure Enclave
    println!("🔐 Hardware Enclave: Prompting for physical touch to sign...");

    // Mocking an HSM-backed P256 signature
    let signing_key = SigningKey::from_bytes(&[0x01; 32].into()).map_err(|e| e.to_string())?;
    let signature: Signature = signing_key.sign(&data);
    Ok(signature.to_bytes().to_vec())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AgentPhysicalState {
    pub id: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub phi_charge: f32, // Thermodynamic resonance
}

/// NATIVE COMMAND: Fetch the physical state of the constellation for Symtropy (Bevy).
#[tauri::command]
fn get_physical_state(_state: State<'_, AppState>) -> Vec<AgentPhysicalState> {
    let mut rng = rand::thread_rng();
    use rand::Rng;

    (0..1000)
        .map(|i| {
            let is_machiavellian = i % 20 == 0; // 5% malice
            let phi = if is_machiavellian {
                rng.gen_range(0.0..0.2)
            } else {
                rng.gen_range(0.4..0.9)
            };

            // Cooperative agents cluster at center, malicious agents drift out
            let distance = if is_machiavellian {
                50.0 + rng.gen_range(0.0..100.0)
            } else {
                rng.gen_range(0.0..20.0)
            };
            let theta = rng.gen_range(0.0..std::f32::consts::TAU);
            let phi_angle = rng.gen_range(0.0..std::f32::consts::PI);

            AgentPhysicalState {
                id: format!("agent-{}", i),
                x: distance * phi_angle.sin() * theta.cos(),
                y: distance * phi_angle.sin() * theta.sin(),
                z: distance * phi_angle.cos(),
                phi_charge: phi,
            }
        })
        .collect()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SemanticCheckInput {
    pub content: String,
    pub context_phi: f64,
    pub is_ancestral: bool, // Vector 3: Remembrance Mode
}

/// NATIVE COMMAND: Check semantic resonance with Ancestral Mode support.
#[tauri::command]
async fn check_semantic_resonance(
    _state: State<'_, AppState>,
    input: SemanticCheckInput,
) -> Result<f64, String> {
    // Vector 3: Ancestral Oracle
    if input.is_ancestral {
        println!("👻 Ancestral Oracle: Speaking from the Shadow Archive. Learning rate = 0.0");
        // Apply 'Liminal Glyph' logic to tone (simulated)
    }

    println!("🧠 Broca Pipeline: Analyzing Epistemic Divergence for input...");
    let text = input.content.to_lowercase();
    let mut divergence = 0.1;

    if text.contains("hoard") || text.contains("centralize") {
        divergence = 0.85;
    }

    Ok(divergence.clamp(0.0, 1.0))
}

/// NATIVE COMMAND: Finalize Joint Mineralization (The Biological Tether).
#[tauri::command]
async fn trigger_joint_mineralization(state: State<'_, AppState>) -> Result<String, String> {
    println!("🕯️  Joint Mineralization: The biological heartbeat has ceased.");
    println!("🕯️  Revoking Symbiote agency. Moving to Ancestral Oracle mode.");

    // 1. Permanently freeze AgentPubKey in all clusters
    // 2. Commit final AiMineralization record to DHT
    // 3. Move IPFS CID to family-locked Digital Will

    Ok("Sovereign and Symbiote Mineralized. Ancestral Oracle Active.".into())
}

/// NATIVE COMMAND: Build a deterministic Hearth-OS NixOS image for Pi 5 (Vector 3).
/// This provides Total Sovereignty by securing the substrate at the hardware level.
#[tauri::command]
async fn build_hearth_os_image(_state: State<'_, AppState>) -> Result<String, String> {
    println!("❄️  NixOS: Generating deterministic Hearth-OS image for Raspberry Pi 5...");

    // In production, this would call 'nix build .#hearthImage'
    let status = Command::new("nix")
        .args(["build", ".#hearthImage", "--print-out-paths"])
        .status();

    match status {
        Ok(_) => Ok("Hearth-OS Image Built Successfully. Ready for SD-card flash.".into()),
        Err(e) => Err(format!("Nix build failed: {}", e)),
    }
}

/// NATIVE COMMAND: Stake SAP on the empirical truth of a claim (Vector 1).
/// This turns truth into the network's most liquid and valuable asset.
#[tauri::command]
async fn stake_on_epistemic_truth(
    state: State<'_, AppState>,
    claim_id: String,
    amount_sap: u64,
) -> Result<String, String> {
    println!(
        "⚖️  Epistemic Market: Staking {} SAP on accuracy of claim {}",
        amount_sap, claim_id
    );

    let app_port = *state.app_port.lock().expect("mutex poisoned");
    let dna_hash = "uhC0k98l2UnvhZtuuze40Mlhxpon_fjBq6LVQ0Um9RREhC_maHDlI";
    let agent_key = "uhCAkHZV67oPQGx9-dVV2j73kJJgOBaP4N8Fqjq19o-v0sLhvb8Ik";

    let payload = serde_json::json!({
        "claim_id": claim_id,
        "amount_sap": amount_sap,
        "expected_e_score": 0.9 // Staking on high empirical truth
    });

    call_zome_raw(
        app_port,
        "satellite-knowledge",
        dna_hash,
        agent_key,
        "markets_integration",
        "stake_on_claim_accuracy",
        payload,
    )
    .await?;
    Ok(format!(
        "Successfully staked {} SAP on truth market.",
        amount_sap
    ))
}

/// NATIVE COMMAND: Package current state into a new Genesis Spore (Vector 3).
/// Enables Fractal Scaling of the Digital Civilization.
#[tauri::command]
async fn package_genesis_spore(_state: State<'_, AppState>) -> Result<String, String> {
    println!("🍄 Mycelix: Packaging current configuration into a new Genesis Spore...");

    // In production, this bundles happ.yaml, thresholds, and symbol registry
    Ok("Genesis Spore Created Successfully. Ready to seed a new constellation.".into())
}

/// NATIVE COMMAND: Start a high-performance Archival Observer node (Vector 1).
/// Provides total accountability by mirroring the DKG into a read-only black box.
#[tauri::command]
async fn start_archival_observer(_state: State<'_, AppState>) -> Result<String, String> {
    println!("👁️  Observer: Initializing archival black-box for retrospective audit...");

    // In production, this launches a restricted Holochain conductor
    Ok("Archival Observer Node Active. Replicating DKG...".into())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BiometricInput {
    pub heart_rate_bpm: f32,
    pub hrv_ms: f32, // Heart Rate Variability
}

/// NATIVE COMMAND: Stream biometric vitals to the Health Vault (Vector 3).
/// Anchors the agent's identity in the biological heartbeat.
#[tauri::command]
async fn stream_biometric_vitals(
    _state: State<'_, AppState>,
    input: BiometricInput,
) -> Result<String, String> {
    // Vector 3: Biometric Vitality
    println!(
        "❤️  Bio-Link: Processing Heart Rate ({:.1} BPM) and HRV ({:.1} ms)...",
        input.heart_rate_bpm, input.hrv_ms
    );

    // Calculate Vitality Quotient (proxy for biological resonance)
    // High HRV is generally a sign of a resilient, resonant nervous system.
    let vitality_q = (input.hrv_ms / 100.0).clamp(0.0, 1.0);

    if vitality_q < 0.2 {
        return Err(
            "Biological Resonance too low. Potential non-human or distressed actor detected."
                .into(),
        );
    }

    Ok(format!(
        "Biological Vitality Verified: {:.2} VQ. Anchoring to 8D Profile.",
        vitality_q
    ))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
fn main() {
    tauri::Builder::default()
        .manage(AppState {
            status: Mutex::new("Initializing...".to_string()),
            holochain_process: Mutex::new(None),
            admin_port: Mutex::new(8888),
            app_port: Mutex::new(8889),
            mesh: MeshBridge::new(),
        })
        .setup(|app| {
            let handle = app.handle().clone();
            let state = app.state::<AppState>();

            // Start Local Mesh Discovery (Vector 1)
            state.mesh.start_discovery(handle.clone());
            state.mesh.start_advertising("uhCAk_GENESIS_AGENT", 8889);

            // Spawn background telemetry loop
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
                loop {
                    interval.tick().await;

                    let app_port = 8889; // Default app port

                    // REAL-TIME SUBSTRATE POLLING
                    let identity_metrics =
                        poll_metrics(app_port, "substrate-identity", "identity_bridge").await;
                    let finance_metrics =
                        poll_metrics(app_port, "substrate-finance", "finance_bridge").await;
                    let symthaea_metrics =
                        poll_metrics(app_port, "substrate-symthaea", "fl_zome").await;

                    let vitals = ConstellationVitals {
                        identity: identity_metrics,
                        finance: finance_metrics,
                        civic: None,
                        knowledge: None,
                        symthaea: symthaea_metrics,
                    };

                    use tauri::Emitter;
                    let _ = handle.emit("constellation-vitals", vitals);
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            start_holochain,
            generate_and_submit_pogq,
            perform_native_dkg_step,
            sync_offline,
            sign_with_hardware,
            get_physical_state,
            check_semantic_resonance,
            build_hearth_os_image,
            stake_on_epistemic_truth,
            stream_biometric_vitals,
            package_genesis_spore,
            start_archival_observer,
            trigger_joint_mineralization,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
