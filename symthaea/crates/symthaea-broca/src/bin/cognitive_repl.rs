// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Symthaea Cognitive REPL v0.3 — Visualization + Checkpoints
//!
//! Interactive CLI for exercising the full Vision → Broca → Geodesic cycle.

use anyhow::Result;
use rustyline::DefaultEditor;
use std::fs;
use symthaea_broca::cognitive_loop::{CognitiveLoop, CognitiveLoopConfig, CognitiveMetrics};
use symthaea_broca::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
use symthaea_broca::thought_chunk::{NodeKind, ProgramNode, ThoughtChunkSequence};
use symthaea_core::genesis::GenesisSeed;

use symthaea_vision_manifold::manifold::VisionManifold;
use symthaea_vision_manifold::types::VisionConfig;

#[cfg(feature = "code-sheaf-eval")]
use symthaea_geodesic::synthesis::GeodesicSynthesizer;
#[cfg(feature = "code-sheaf-eval")]
use symthaea_geodesic::synthesis::SynthesisConfig;
#[cfg(feature = "code-sheaf-eval")]
use symthaea_geodesic::tri_oracle::TriOracle;

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║      SYMTHAEA COGNITIVE REPL v0.3 (Viz + Checkpoints)      ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let genesis = GenesisSeed::from_phrase("repl-session");
    let config = LiquidMambaConfig::default();
    let broca = LiquidMambaGenerator::with_mock(&genesis, config);

    let vision_cfg = VisionConfig::default();
    let vision = VisionManifold::new(vision_cfg, 640, 480);

    #[cfg(feature = "code-sheaf-eval")]
    let geodesic = GeodesicSynthesizer::new(SynthesisConfig::default());
    #[cfg(feature = "code-sheaf-eval")]
    let tri_oracle = TriOracle::with_defaults();

    let mut cognitive_loop = CognitiveLoop::new(
        vision,
        broca,
        #[cfg(feature = "code-sheaf-eval")]
        geodesic,
        #[cfg(feature = "code-sheaf-eval")]
        tri_oracle,
        CognitiveLoopConfig::default(),
    );

    let mut last_monologue: Option<ThoughtChunkSequence> = None;
    let mut rl = DefaultEditor::new()?;

    println!("Commands:");
    println!("  step [n]  - Run 1 or n cognitive steps (dt=0.033)");
    println!("  viz       - ASCII visualization of last monologue");
    println!("  nodes     - ASCII view of program node graph");
    println!("  status    - Show loop metrics and averages");
    println!("  save <f>  - Save current metrics to JSON");
    println!("  load <f>  - Load metrics from JSON");
    println!("  clear     - Reset metrics");
    println!("  help      - Show this help");
    println!("  exit      - Exit REPL\n");

    loop {
        let readline = rl.readline("symthaea> ");
        match readline {
            Ok(line) => {
                let _ = rl.add_history_entry(line.as_str());
                let parts: Vec<&str> = line.trim().split_whitespace().collect();
                if parts.is_empty() {
                    continue;
                }

                match parts[0] {
                    "step" | "s" => {
                        let count = parts
                            .get(1)
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(1);
                        for _ in 0..count {
                            let frame = vec![128u8; 640 * 480 * 3]; // Mock gray frame
                            let output = cognitive_loop.cognitive_step(&frame, 0.033)?;
                            println!(
                                "Step {}: ψ={:.2} | conf={:.2} | time={}ms",
                                cognitive_loop.step_count,
                                output.mean_psi,
                                output.mean_confidence,
                                output.cycle_time_ms
                            );
                            if let Some(s) = &output.synthesis {
                                println!("  Synthesized: {}", s);
                            }
                            last_monologue = Some(output.monologue);
                        }
                    }

                    "viz" | "v" => {
                        if let Some(monologue) = &last_monologue {
                            print_monologue_ascii(monologue);
                        } else {
                            println!("No monologue yet. Run 'step' first.");
                        }
                    }

                    "nodes" | "n" => {
                        if let Some(monologue) = &last_monologue {
                            print_nodes_ascii(&monologue.to_program_nodes());
                        } else {
                            println!("No nodes yet. Run 'step' first.");
                        }
                    }

                    "status" | "st" => {
                        cognitive_loop.print_status();
                    }

                    "save" => {
                        if parts.len() > 1 {
                            let path = parts[1];
                            let json = serde_json::to_string_pretty(cognitive_loop.get_metrics())?;
                            fs::write(path, json)?;
                            println!("Metrics saved to {}", path);
                        } else {
                            println!("Usage: save <filename>");
                        }
                    }

                    "load" => {
                        if parts.len() > 1 {
                            let path = parts[1];
                            match fs::read_to_string(path) {
                                Ok(data) => match serde_json::from_str::<CognitiveMetrics>(&data) {
                                    Ok(metrics) => {
                                        cognitive_loop.metrics = metrics;
                                        println!("Metrics loaded from {}", path);
                                    }
                                    Err(e) => println!("Failed to parse metrics: {}", e),
                                },
                                Err(e) => println!("Failed to read file: {}", e),
                            }
                        } else {
                            println!("Usage: load <filename>");
                        }
                    }

                    "clear" => {
                        cognitive_loop.reset_metrics();
                    }

                    "help" | "h" | "?" => {
                        println!(
                            "Available commands: step, viz, nodes, status, save, load, clear, help, exit"
                        );
                    }

                    "exit" | "quit" | "q" => break,

                    _ => println!(
                        "Unknown command: {}. Type 'help' for available commands.",
                        parts[0]
                    ),
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("CTRL-C");
                break;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("CTRL-D");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    Ok(())
}

fn print_monologue_ascii(monologue: &ThoughtChunkSequence) {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                    SEMANTIC MONOLOGUE                      ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    for (i, chunk) in monologue.chunks.iter().enumerate() {
        let bar_len = (chunk.psi * 20.0).round() as usize;
        let bar = "█".repeat(bar_len);
        let dots = ".".repeat(20 - bar_len);
        println!(
            "{:2}. [{}{}] ψ={:.2} {}",
            i + 1,
            bar,
            dots,
            chunk.psi,
            chunk.summary()
        );
    }
    println!("────────────────────────────────────────────────────────────\n");
}

fn print_nodes_ascii(nodes: &[ProgramNode]) {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                   PROGRAM NODE GRAPH                       ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    for (i, node) in nodes.iter().enumerate() {
        let kind_icon = match node.kind {
            NodeKind::Code => "⚙",
            NodeKind::Action => "→",
            NodeKind::Text => "✎",
            NodeKind::StructuredData => "▦",
            NodeKind::Hypothesis => "?",
            NodeKind::PlanStep => "!",
        };
        let content_preview: String = node.content.chars().take(20).collect();
        println!(
            "{:2}. {} [{:.<20}] ψ={:.2} {}",
            i + 1,
            kind_icon,
            content_preview,
            node.psi,
            node.content
        );
    }
    println!("────────────────────────────────────────────────────────────\n");
}
