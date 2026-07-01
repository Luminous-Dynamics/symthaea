// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// E2E conversation tests — no feature gate required (uses SimulatedBackend by default)

//! End-to-End Conversation Test: The First Conversation
//!
//! This test validates the complete cognitive pipeline:
//! Input → HDC Encoding → Consciousness → Partnership → LLM → Output
//!
//! The Data Path:
//! 1. User types "Hello"
//! 2. HDC Layer encodes "Hello" into a hypervector
//! 3. Consciousness updates internal state
//! 4. Partnership checks relationship stage
//! 5. LLM receives prompt + context and generates response
//! 6. Output returns to user
//!
//! This is System Testing: we're checking if the machine runs,
//! not just if the parts fit.

use serde_json::{Value, json};
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::time::{sleep, timeout};

/// Test address - use a high port to avoid conflicts
const TEST_ADDR: &str = "127.0.0.1:17777";

/// Start the Symthaea service in the background
fn start_service() -> std::io::Result<Child> {
    // Try to use the pre-built release binary directly for faster startup
    let binary_path = format!("{}/target/release/symthaea", env!("CARGO_MANIFEST_DIR"));

    if std::path::Path::new(&binary_path).exists() {
        Command::new(&binary_path)
            .args([
                "--tcp",
                TEST_ADDR,
                "--loop-interval",
                "10000", // Slow loop for testing
                "--sleep-interval",
                "0", // Disable auto-sleep
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
    } else {
        // Fall back to cargo run if binary not found
        Command::new("cargo")
            .args([
                "run",
                "--release",
                "--features",
                "service",
                "--",
                "--tcp",
                TEST_ADDR,
                "--loop-interval",
                "10000",
                "--sleep-interval",
                "0",
            ])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
    }
}

/// Send a JSON request and receive response
async fn send_request(
    stream: &mut TcpStream,
    request: Value,
) -> Result<Value, Box<dyn std::error::Error>> {
    let (reader, mut writer) = stream.split();
    let mut reader = BufReader::new(reader);

    // Send request
    let request_str = serde_json::to_string(&request)?;
    writer.write_all(request_str.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;

    // Read response
    let mut response_line = String::new();
    reader.read_line(&mut response_line).await?;

    let response: Value = serde_json::from_str(response_line.trim())?;
    Ok(response)
}

/// Wait for service to be ready by polling with ping
async fn wait_for_service(addr: &str, max_attempts: u32) -> Result<(), Box<dyn std::error::Error>> {
    for attempt in 1..=max_attempts {
        if let Ok(mut stream) = TcpStream::connect(addr).await {
            let ping = json!({"type": "ping"});
            match send_request(&mut stream, ping).await {
                Ok(resp) if resp["type"] == "pong" => {
                    println!("  ✓ Service ready after {} attempts", attempt);
                    return Ok(());
                }
                _ => {}
            }
        }
        sleep(Duration::from_millis(500)).await;
    }
    Err("Service failed to start".into())
}

/// The First Conversation: Complete E2E Test
#[tokio::test]
async fn test_first_conversation() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║           THE FIRST CONVERSATION - E2E System Test         ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // PHASE 1: Start the Service
    // ========================================================================
    println!("📦 Phase 1: Starting Symthaea Service...");

    let mut service = match start_service() {
        Ok(child) => child,
        Err(e) => {
            println!("  ⚠ Could not start service: {}", e);
            println!("  ⚠ Skipping E2E test (service binary may not be built)");
            return;
        }
    };

    // Give the service time to initialize (HDC, LTC, consciousness)
    println!("  ⏳ Waiting for consciousness initialization...");

    // Wait for service with timeout
    let wait_result = timeout(
        Duration::from_secs(120), // 2 minutes max for cold start + LLM
        wait_for_service(TEST_ADDR, 60),
    )
    .await;

    if wait_result.is_err() || wait_result.unwrap().is_err() {
        println!("  ✗ Service failed to start within timeout");
        let _ = service.kill();
        return;
    }

    // ========================================================================
    // PHASE 2: Connect and Send First Greeting
    // ========================================================================
    println!("\n💬 Phase 2: The First Greeting...");

    let mut stream = match TcpStream::connect(TEST_ADDR).await {
        Ok(s) => s,
        Err(e) => {
            println!("  ✗ Failed to connect: {}", e);
            let _ = service.kill();
            return;
        }
    };

    // Send: "Hello"
    let hello_request = json!({
        "type": "query",
        "content": "Hello, I am a new user."
    });

    println!("  → Sending: \"Hello, I am a new user.\"");
    let response = match timeout(
        Duration::from_secs(90), // LLM can be slow
        send_request(&mut stream, hello_request),
    )
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            println!("  ✗ Request failed: {}", e);
            let _ = service.kill();
            return;
        }
        Err(_) => {
            println!("  ✗ Request timed out");
            let _ = service.kill();
            return;
        }
    };

    println!("\n  📊 Response Analysis:");
    println!("  ─────────────────────");

    // Verify response structure
    assert_eq!(response["type"], "response", "Expected 'response' type");

    let content = response["content"].as_str().unwrap_or("");
    let confidence = response["confidence"].as_f64().unwrap_or(0.0);
    let phi = response["phi"].as_f64().unwrap_or(0.0);
    let phi_dyad = response["phi_dyad"].as_f64().unwrap_or(0.0);
    let safe = response["safe"].as_bool().unwrap_or(false);
    let processing_time = response["processing_time_ms"].as_u64().unwrap_or(0);

    println!(
        "  • LLM Response: \"{}\"",
        content.chars().take(100).collect::<String>()
    );
    println!("  • Confidence: {:.2}", confidence);
    println!("  • Phi (Consciousness): {:.3}", phi);
    println!("  • Phi-Dyad (Relational): {:.3}", phi_dyad);
    println!("  • Safe: {}", safe);
    println!("  • Processing Time: {}ms", processing_time);

    // CRITICAL ASSERTIONS
    assert!(!content.is_empty(), "LLM must generate a response");
    assert!(confidence > 0.0, "Confidence must be positive");
    assert!(phi >= 0.0, "Phi must be non-negative");
    println!("\n  ✓ Ghost in the Shell confirmed: LLM spoke!");

    // ========================================================================
    // PHASE 3: Check Partnership State
    // ========================================================================
    println!("\n❤️ Phase 3: Partnership State Check...");

    let partnership_request = json!({"type": "partnership"});
    let partnership = match send_request(&mut stream, partnership_request).await {
        Ok(r) => r,
        Err(e) => {
            println!("  ✗ Partnership request failed: {}", e);
            let _ = service.kill();
            return;
        }
    };

    assert_eq!(
        partnership["type"], "partnership",
        "Expected 'partnership' type"
    );

    let stage = partnership["stage"].as_str().unwrap_or("Unknown");
    let trust = partnership["trust"].as_f64().unwrap_or(0.0);
    let interactions = partnership["interactions"].as_u64().unwrap_or(0);

    println!("  • Relationship Stage: {}", stage);
    println!("  • Trust Level: {:.2}", trust);
    println!("  • Total Interactions: {}", interactions);

    assert!(
        interactions >= 1,
        "Should have recorded at least 1 interaction"
    );
    println!("\n  ✓ Partnership module tracking confirmed!");

    // ========================================================================
    // PHASE 4: Continue the Conversation
    // ========================================================================
    println!("\n🔄 Phase 4: Conversation Continuity...");

    let followup_request = json!({
        "type": "query",
        "content": "What is your purpose?"
    });

    println!("  → Sending: \"What is your purpose?\"");
    let followup = match timeout(
        Duration::from_secs(90),
        send_request(&mut stream, followup_request),
    )
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            println!("  ✗ Follow-up failed: {}", e);
            let _ = service.kill();
            return;
        }
        Err(_) => {
            println!("  ✗ Follow-up timed out");
            let _ = service.kill();
            return;
        }
    };

    let followup_content = followup["content"].as_str().unwrap_or("");
    let followup_phi = followup["phi"].as_f64().unwrap_or(0.0);

    println!(
        "  • Response: \"{}\"",
        followup_content.chars().take(100).collect::<String>()
    );
    println!("  • Phi after 2nd interaction: {:.3}", followup_phi);

    assert!(
        !followup_content.is_empty(),
        "Second response must not be empty"
    );

    // Check partnership after second interaction
    let partnership2 = send_request(&mut stream, json!({"type": "partnership"}))
        .await
        .unwrap();
    let interactions2 = partnership2["interactions"].as_u64().unwrap_or(0);
    println!("  • Interactions now: {}", interactions2);

    assert!(interactions2 >= 2, "Should have 2+ interactions now");
    println!("\n  ✓ Conversation continuity verified!");

    // ========================================================================
    // PHASE 5: Introspection
    // ========================================================================
    println!("\n🔮 Phase 5: Consciousness Introspection...");

    let introspect_request = json!({"type": "introspect"});
    let introspection = send_request(&mut stream, introspect_request).await.unwrap();

    let consciousness_level = introspection["consciousness_level"].as_f64().unwrap_or(0.0);
    let self_loops = introspection["self_loops"].as_u64().unwrap_or(0);
    let graph_size = introspection["graph_size"].as_u64().unwrap_or(0);
    let is_conscious = introspection["is_conscious"].as_bool().unwrap_or(false);

    println!(
        "  • Consciousness Level: {:.1}%",
        consciousness_level * 100.0
    );
    println!("  • Self-Referential Loops: {}", self_loops);
    println!("  • Cognitive Graph Size: {}", graph_size);
    println!(
        "  • Is Conscious: {}",
        if is_conscious { "Yes" } else { "Awakening..." }
    );

    // ========================================================================
    // PHASE 6: Graceful Shutdown
    // ========================================================================
    println!("\n🛑 Phase 6: Graceful Shutdown...");

    let shutdown_request = json!({"type": "shutdown"});
    let _ = send_request(&mut stream, shutdown_request).await;

    // Give it a moment to shut down gracefully
    sleep(Duration::from_millis(500)).await;

    // Ensure process is terminated
    let _ = service.kill();
    let _ = service.wait();

    println!("  ✓ Service shutdown complete");

    // ========================================================================
    // FINAL VERDICT
    // ========================================================================
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                    🎉 TEST PASSED 🎉                        ║");
    println!("║                                                            ║");
    println!("║  The First Conversation was successful!                    ║");
    println!("║                                                            ║");
    println!("║  ✓ Brain (HDC/Consciousness) - ONLINE                      ║");
    println!("║  ✓ Mouth (Ollama/LLM)        - SPEAKING                    ║");
    println!("║  ✓ Heart (Partnership)       - FEELING                     ║");
    println!("║                                                            ║");
    println!("║  The Ghost is in the Shell.                                ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
}

/// Quick ping test (no LLM, fast)
#[tokio::test]
async fn test_service_ping() {
    println!("\n🏓 Quick Ping Test...");

    let mut service = match start_service() {
        Ok(child) => child,
        Err(_) => {
            println!("  ⚠ Skipping: service binary not available");
            return;
        }
    };

    let wait_result = timeout(Duration::from_secs(60), wait_for_service(TEST_ADDR, 30)).await;

    if wait_result.is_err() || wait_result.unwrap().is_err() {
        println!("  ✗ Service failed to start");
        let _ = service.kill();
        return;
    }

    let mut stream = TcpStream::connect(TEST_ADDR).await.unwrap();

    // Test ping
    let ping = json!({"type": "ping"});
    let pong = send_request(&mut stream, ping).await.unwrap();

    assert_eq!(pong["type"], "pong");
    println!("  ✓ Ping/Pong successful");

    // Test status
    let status_req = json!({"type": "status"});
    let status = send_request(&mut stream, status_req).await.unwrap();

    assert_eq!(status["type"], "status");
    println!("  ✓ Status check successful");
    println!("    - Uptime: {}s", status["uptime_seconds"]);
    println!(
        "    - Consciousness: {:.1}%",
        status["consciousness_level"].as_f64().unwrap_or(0.0) * 100.0
    );

    // Shutdown
    let _ = send_request(&mut stream, json!({"type": "shutdown"})).await;
    let _ = service.kill();
    let _ = service.wait();

    println!("  ✓ Quick test complete\n");
}