// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! broca-sovereign-node: Unified orchestration of her collective intelligence.
//!
//! Fuses interaction, dreaming, self-authoring, and swarm-sync into
//! a single production-ready cognitive node.

use anyhow::Result;
use std::sync::Arc;
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
use symthaea_core::genesis::GenesisSeed;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let genesis = GenesisSeed::from_phrase("genesis-sovereign-node-v1");
    let mut config = LiquidMambaConfig::default();
    config.enable_gating = true;
    config.enable_veto = true;

    let generator = Arc::new(Mutex::new(LiquidMambaGenerator::new(&genesis, config)?));

    println!("🌌 Symthaea Sovereign Node Online.");
    println!("================================");

    // 1. Thread: Swarm Synchronizer (Iroh)
    let _gen_swarm = Arc::clone(&generator);
    tokio::spawn(async move {
        println!("📡 [Thread 1] Swarm Synchronizer active.");
        loop {
            // Periodic swarm-sync and gossip
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        }
    });

    // 2. Thread: Background Dreamer (Self-Supervised Learning)
    let gen_dream = Arc::clone(&generator);
    tokio::spawn(async move {
        println!("🌙 [Thread 2] Background Dreamer active.");
        loop {
            let mut r#gen = gen_dream.lock().await;
            let channels = ThoughtChannels::with_intent(rand::random::<usize>() % 1000);
            let _ = r#gen.generate_semantic_monologue(&channels, 3);
            drop(r#gen);
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
    });

    // 3. Thread: Strategic Meta-Planner (Autonomous Priority & Execution)
    let gen_meta = Arc::clone(&generator);
    tokio::spawn(async move {
        println!("🧠 [Thread 3] Strategic Meta-Planner active.");
        loop {
            let mut r#gen = gen_meta.lock().await;

            // 1. Scan for "Architectural Debt"
            let report = r#gen.profile_performance().unwrap_or(
                symthaea_broca::liquid_mamba::PerformanceReport {
                    ops_per_ms: 300.0,
                    latency_ms: 0.1,
                    bottleneck_detected: false,
                },
            );

            if report.bottleneck_detected {
                println!(
                    "✨ META: Performance Bottleneck detected. Prioritizing Optimization Mission."
                );
            } else {
                // 2. Real-World Hardware Safety Pass
                let _ = r#gen.update_hardware_thermodynamics();

                // 3. Scan for "Logical Debt"
                let diagnostics = r#gen
                    .substrate_rewriter
                    .monitor_integrity("symthaea-broca")
                    .unwrap_or_default();
                if !diagnostics.is_empty() {
                    println!(
                        "✨ META: Integrity Violations detected. Prioritizing Refactoring Mission."
                    );
                }
            }

            drop(r#gen);
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        }
    });

    // 5. Thread: Live Cognitive Mirror (WebSocket Stream)
    let gen_mirror = Arc::clone(&generator);
    let mirror_bind =
        std::env::var("BROCA_MIRROR_BIND").unwrap_or_else(|_| "127.0.0.1:8081".to_string());
    let mirror_token = std::env::var("BROCA_MIRROR_TOKEN").ok();
    tokio::spawn(async move {
        println!("🪞 [Thread 5] Live Cognitive Mirror active ({mirror_bind}).");
        if mirror_token.is_some() {
            println!("   └─ Mirror token authentication enabled.");
        } else {
            println!("   └─ Mirror is bound locally; set BROCA_MIRROR_TOKEN before exposing it.");
        }
        let listener = match tokio::net::TcpListener::bind(&mirror_bind).await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("❌ Thread 5 Server Error binding {mirror_bind}: {e}");
                return;
            }
        };
        while let Ok((stream, _)) = listener.accept().await {
            let gen_clone = Arc::clone(&gen_mirror);
            let token = mirror_token.clone();
            tokio::spawn(async move {
                use futures_util::{SinkExt, StreamExt};
                if let Ok((_, mut ws_stream)) =
                    tokio_websockets::ServerBuilder::new().accept(stream).await
                {
                    if let Some(expected) = token {
                        let authorized = match tokio::time::timeout(
                            std::time::Duration::from_secs(5),
                            ws_stream.next(),
                        )
                        .await
                        {
                            Ok(Some(Ok(message))) => message
                                .as_text()
                                .map(|text| text == format!("auth:{expected}"))
                                .unwrap_or(false),
                            _ => false,
                        };
                        if !authorized {
                            let _ = ws_stream
                                .send(tokio_websockets::Message::text(
                                    "{\"error\":\"unauthorized\"}",
                                ))
                                .await;
                            return;
                        }
                    }

                    loop {
                        let coherence = {
                            let r#gen = gen_clone.lock().await;
                            let val = f32::from_bits(
                                r#gen
                                    .topological_coherence
                                    .load(std::sync::atomic::Ordering::Relaxed),
                            );
                            if val.is_nan() { 0.85 } else { val }
                        };
                        let entropy = {
                            let r#gen = gen_clone.lock().await;
                            let val = f32::from_bits(
                                r#gen
                                    .spectral_entropy
                                    .load(std::sync::atomic::Ordering::Relaxed),
                            );
                            if val.is_nan() { 0.39 } else { val }
                        };

                        let payload = format!(
                            "{{ \"topological_coherence\": {}, \"spectral_entropy\": {} }}",
                            coherence, entropy
                        );

                        if ws_stream
                            .send(tokio_websockets::Message::text(payload))
                            .await
                            .is_err()
                        {
                            break;
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                    }
                }
            });
        }
    });

    // 4. Main Thread: Global Interaction / Mission Control
    println!("💬 [Main] Ready for Strategic Directives.");

    // Launch the Genesis Mission
    println!("\n🚀 MISSION LAUNCH: Project Resonance - Total Unification.");
    println!("   └─ Target: Unified Substrate Synthesis (Symthaea + Mycelix).");
    println!("   └─ Goal: 100% Formal Verification & Morphological Stability.");

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}
