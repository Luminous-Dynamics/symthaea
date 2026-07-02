// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::State;
use std::sync::Mutex;
use std::process::{Child, Command, Stdio};
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};
use tokio::sync::Mutex as AsyncMutex;
use log;

// Data structures for Holochain admin API responses
#[derive(Debug, Serialize, Deserialize)]
pub struct AppInfo {
    pub installed_app_id: String,
    pub cell_info: Vec<CellInfo>,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CellInfo {
    pub cell_id: String,
    pub dna_hash: String,
    pub agent_pub_key: String,
}

// Configuration for app interface
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub dna_hash: String,
    pub agent_key: String,
}

// Application state
pub struct AppState {
    pub status: Mutex<String>,
    pub holochain_process: Mutex<Option<Child>>,
    pub admin_port: Mutex<u16>,
    pub config: AsyncMutex<AppConfig>,
}

// Basic Tauri command - test the bridge
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! Welcome to Mycelix Network 🍄", name)
}

// Get app status
#[tauri::command]
fn get_status(state: State<AppState>) -> String {
    let status = state.status.lock().expect("status mutex poisoned");
    status.clone()
}

// Update app status
#[tauri::command]
fn set_status(state: State<AppState>, new_status: String) -> Result<String, String> {
    let mut status = state.status.lock().expect("status mutex poisoned");
    *status = new_status.clone();
    Ok(format!("Status updated to: {}", new_status))
}

// Start Holochain conductor
#[tauri::command]
async fn start_holochain(state: State<'_, AppState>) -> Result<String, String> {
    // Check if conductor is already running
    {
        let process_guard = state.holochain_process.lock().expect("holochain_process mutex poisoned");
        if process_guard.is_some() {
            return Ok("Holochain conductor already running".to_string());
        }
    }

    // Get the config file path (relative to app directory)
    let config_path = PathBuf::from("conductor-config.yaml");

    if !config_path.exists() {
        return Err("Conductor config file not found. Please create conductor-config.yaml".to_string());
    }

    // Try to find holochain binary
    // In development, it should be available via nix develop
    let holochain_cmd = if cfg!(target_os = "windows") {
        "holochain.exe"
    } else {
        "holochain"
    };

    // Start the conductor process
    // Use --piped mode to avoid interactive passphrase prompt
    match Command::new(holochain_cmd)
        .arg("--piped")
        .arg("-c")
        .arg(&config_path)
        .stdin(Stdio::piped())   // Pipe empty passphrase
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(mut child) => {
            // Write empty passphrase to stdin for --piped mode
            if let Some(mut stdin) = child.stdin.take() {
                use std::io::Write;
                let _ = stdin.write_all(b"\n");
            }

            // Store the process handle
            let mut process_guard = state.holochain_process.lock().expect("holochain_process mutex poisoned");
            *process_guard = Some(child);

            Ok(format!(
                "Holochain conductor started successfully with config: {}",
                config_path.display()
            ))
        }
        Err(e) => {
            Err(format!(
                "Failed to start Holochain conductor: {}. Make sure Holochain is installed (run 'nix develop' first)",
                e
            ))
        }
    }
}

// Stop Holochain conductor
#[tauri::command]
async fn stop_holochain(state: State<'_, AppState>) -> Result<String, String> {
    let mut process_guard = state.holochain_process.lock().expect("holochain_process mutex poisoned");

    match process_guard.take() {
        Some(mut child) => {
            match child.kill() {
                Ok(_) => Ok("Holochain conductor stopped successfully".to_string()),
                Err(e) => Err(format!("Failed to stop Holochain conductor: {}", e)),
            }
        }
        None => Ok("Holochain conductor is not running".to_string()),
    }
}

// Check if Holochain is running
#[tauri::command]
fn check_holochain_status(state: State<'_, AppState>) -> Result<String, String> {
    let mut process_guard = state.holochain_process.lock().expect("holochain_process mutex poisoned");

    match process_guard.as_mut() {
        Some(child) => {
            match child.try_wait() {
                Ok(Some(status)) => {
                    // Process has exited
                    *process_guard = None;
                    Ok(format!("Holochain conductor exited with status: {}", status))
                }
                Ok(None) => {
                    // Still running
                    Ok("Holochain conductor is running".to_string())
                }
                Err(e) => Err(format!("Error checking conductor status: {}", e)),
            }
        }
        None => Ok("Holochain conductor is not running".to_string()),
    }
}

// P2P network connection state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    pub connected_peers: Vec<String>,
    pub agent_pub_key: String,
    pub network_seed: Option<String>,
}

// Connect to P2P network via Holochain conductor
#[tauri::command]
async fn connect_to_network(state: State<'_, AppState>) -> Result<String, String> {
    let port = *state.admin_port.lock().expect("admin_port mutex poisoned");

    // Step 1: Check if conductor is running by listing apps
    let apps_result = send_admin_request(port, "list_apps", serde_json::json!({})).await;
    if let Err(e) = apps_result {
        return Err(format!("Holochain conductor not running or not reachable: {}", e));
    }

    // Step 2: Get agent info to verify network connectivity
    let agent_info_result = send_admin_request(
        port,
        "agent_info",
        serde_json::json!({ "cell_id": null })
    ).await;

    match agent_info_result {
        Ok(agent_info) => {
            // Step 3: Attach app interface if not already attached
            let attach_result = send_admin_request(
                port,
                "attach_app_interface",
                serde_json::json!({ "port": 8889, "allowed_origins": "*" })
            ).await;

            // Ignore error if interface already attached
            if let Err(e) = &attach_result {
                if !e.contains("already") {
                    log::warn!("Could not attach app interface: {}", e);
                }
            }

            // Step 4: Request network stats
            let network_stats = send_admin_request(
                port,
                "dump_network_stats",
                serde_json::json!({})
            ).await;

            let peer_count = match network_stats {
                Ok(stats) => {
                    // Extract peer count from network stats
                    stats.get("peers")
                        .and_then(|p| p.as_array())
                        .map(|arr| arr.len())
                        .unwrap_or(0)
                }
                Err(_) => 0
            };

            let network_info = NetworkInfo {
                connected_peers: agent_info.get("agent_infos")
                    .and_then(|a| a.as_array())
                    .map(|arr| arr.iter()
                        .filter_map(|v| v.get("agent").and_then(|a| a.as_str()).map(String::from))
                        .collect())
                    .unwrap_or_default(),
                agent_pub_key: agent_info.get("agent_pub_key")
                    .and_then(|k| k.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                network_seed: agent_info.get("network_seed")
                    .and_then(|s| s.as_str())
                    .map(String::from),
            };

            Ok(format!(
                "Connected to P2P network. Agent: {}..., Peers discovered: {}, Network active: true",
                &network_info.agent_pub_key[..std::cmp::min(16, network_info.agent_pub_key.len())],
                peer_count
            ))
        }
        Err(e) => {
            // Conductor running but no agent info - may need to install/enable app first
            Err(format!(
                "Holochain conductor running but P2P network not initialized. \
                 Please ensure a hApp is installed and enabled. Error: {}", e
            ))
        }
    }
}

// Helper function to send admin API request via WebSocket
async fn send_admin_request(
    port: u16,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let url = format!("ws://localhost:{}", port);

    // Connect to WebSocket
    let (ws_stream, _) = connect_async(&url)
        .await
        .map_err(|e| format!("Failed to connect to admin interface: {}", e))?;

    let (mut write, mut read) = ws_stream.split();

    // Create JSON-RPC request
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": "request-1",
        "method": method,
        "params": params
    });

    // Send request
    write
        .send(Message::Text(request.to_string()))
        .await
        .map_err(|e| format!("Failed to send request: {}", e))?;

    // Read response
    if let Some(msg) = read.next().await {
        let response = msg.map_err(|e| format!("Failed to read response: {}", e))?;

        if let Message::Text(text) = response {
            let json: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| format!("Failed to parse JSON response: {}", e))?;

            // Check for error
            if let Some(error) = json.get("error") {
                return Err(format!("Admin API error: {}", error));
            }

            // Return result
            if let Some(result) = json.get("result") {
                return Ok(result.clone());
            }

            Err("No result in response".to_string())
        } else {
            Err("Unexpected message type".to_string())
        }
    } else {
        Err("No response received".to_string())
    }
}

// Get list of installed apps
#[tauri::command]
async fn get_installed_apps(state: State<'_, AppState>) -> Result<String, String> {
    let port = *state.admin_port.lock().expect("admin_port mutex poisoned");

    let result = send_admin_request(port, "list_apps", serde_json::json!({})).await?;

    Ok(serde_json::to_string_pretty(&result)
        .unwrap_or_else(|_| "Failed to format response".to_string()))
}

// Get list of cells
#[tauri::command]
async fn get_cells(state: State<'_, AppState>) -> Result<String, String> {
    let port = *state.admin_port.lock().expect("admin_port mutex poisoned");

    let result = send_admin_request(port, "list_cell_ids", serde_json::json!({})).await?;

    Ok(serde_json::to_string_pretty(&result)
        .unwrap_or_else(|_| "Failed to format response".to_string()))
}

// Enable an app
#[tauri::command]
async fn enable_app(state: State<'_, AppState>, app_id: String) -> Result<String, String> {
    let port = *state.admin_port.lock().expect("admin_port mutex poisoned");

    let params = serde_json::json!({
        "installed_app_id": app_id
    });

    send_admin_request(port, "enable_app", params).await?;

    Ok(format!("App '{}' enabled successfully", app_id))
}

// Disable an app
#[tauri::command]
async fn disable_app(state: State<'_, AppState>, app_id: String) -> Result<String, String> {
    let port = *state.admin_port.lock().expect("admin_port mutex poisoned");

    let params = serde_json::json!({
        "installed_app_id": app_id
    });

    send_admin_request(port, "disable_app", params).await?;

    Ok(format!("App '{}' disabled successfully", app_id))
}

// Get app info with details
#[tauri::command]
async fn get_app_info(state: State<'_, AppState>, app_id: String) -> Result<String, String> {
    let port = *state.admin_port.lock().expect("admin_port mutex poisoned");

    let params = serde_json::json!({
        "installed_app_id": app_id
    });

    let result = send_admin_request(port, "get_app_info", params).await?;

    Ok(serde_json::to_string_pretty(&result)
        .unwrap_or_else(|_| "Failed to format response".to_string()))
}

// App API helper function for zome calls
async fn send_app_request(port: u16, method: &str, params: serde_json::Value) -> Result<serde_json::Value, String> {
    let url = format!("ws://localhost:{}", port);
    let (ws_stream, _) = connect_async(&url).await
        .map_err(|e| format!("Failed to connect to app interface: {}", e))?;

    let (mut write, mut read) = ws_stream.split();

    let request = serde_json::json!({
        "type": method,
        "data": params
    });

    write.send(Message::Text(request.to_string())).await
        .map_err(|e| format!("Failed to send request: {}", e))?;

    if let Some(Ok(Message::Text(response_text))) = read.next().await {
        let response: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        if let Some(error) = response.get("error") {
            return Err(format!("Request failed: {}", error));
        }

        Ok(response.get("data").unwrap_or(&serde_json::Value::Null).clone())
    } else {
        Err("No response received".to_string())
    }
}

// Generic zome call command
#[tauri::command]
async fn call_zome_function(
    state: State<'_, AppState>,
    zome_name: String,
    function_name: String,
    payload: Option<String>
) -> Result<String, String> {
    let config = state.config.lock().await;
    let app_port = 8889; // App interface port

    let payload_value: serde_json::Value = if let Some(p) = payload {
        serde_json::from_str(&p).unwrap_or_default()
    } else {
        serde_json::Value::Null
    };

    let params = serde_json::json!({
        "app_id": "mycelix-test",
        "cell_id": [
            config.dna_hash.clone(),
            config.agent_key.clone()
        ],
        "zome_name": zome_name,
        "fn_name": function_name,
        "payload": payload_value
    });

    let response = send_app_request(app_port, "call_zome", params).await?;
    Ok(serde_json::to_string_pretty(&response).unwrap_or_else(|_| response.to_string()))
}

// Convenience command: hello()
#[tauri::command]
async fn call_hello(state: State<'_, AppState>) -> Result<String, String> {
    call_zome_function(state, "hello".to_string(), "hello".to_string(), None).await
}

// Convenience command: whoami()
#[tauri::command]
async fn call_whoami(state: State<'_, AppState>) -> Result<String, String> {
    call_zome_function(state, "hello".to_string(), "whoami".to_string(), None).await
}

// Convenience command: echo(input)
#[tauri::command]
async fn call_echo(state: State<'_, AppState>, input: String) -> Result<String, String> {
    call_zome_function(state, "hello".to_string(), "echo".to_string(), Some(format!(r#"{{"input": "{}"}}"#, input))).await
}

// Convenience command: get_agent_info()
#[tauri::command]
async fn call_get_agent_info(state: State<'_, AppState>) -> Result<String, String> {
    call_zome_function(state, "hello".to_string(), "get_agent_info".to_string(), None).await
}

// Message broadcasting commands

// Send a message to the network
#[tauri::command]
async fn send_message(state: State<'_, AppState>, content: String) -> Result<String, String> {
    let payload = format!(r#"{{"content": "{}"}}"#, content);
    call_zome_function(state, "messages".to_string(), "broadcast_message".to_string(), Some(payload)).await
}

// Get all messages from the network
#[tauri::command]
async fn get_messages(state: State<'_, AppState>) -> Result<String, String> {
    call_zome_function(state, "messages".to_string(), "get_all_messages".to_string(), None).await
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
fn main() {
    tauri::Builder::default()
        .manage(AppState {
            status: Mutex::new("Initializing...".to_string()),
            holochain_process: Mutex::new(None),
            admin_port: Mutex::new(8888), // Default admin port
            config: AsyncMutex::new(AppConfig {
                dna_hash: "uhC0k98l2UnvhZtuuze40Mlhxpon_fjBq6LVQ0Um9RREhC_maHDlI".to_string(),
                agent_key: "uhCAkHZV67oPQGx9-dVV2j73kJJgOBaP4N8Fqjq19o-v0sLhvb8Ik".to_string(),
            }),
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            get_status,
            set_status,
            start_holochain,
            stop_holochain,
            check_holochain_status,
            connect_to_network,
            get_installed_apps,
            get_cells,
            enable_app,
            disable_app,
            get_app_info,
            call_zome_function,
            call_hello,
            call_whoami,
            call_echo,
            call_get_agent_info,
            send_message,
            get_messages,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}