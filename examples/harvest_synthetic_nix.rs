// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Synthetic Nix Harvester: Generates hundreds of training pairs by
//! permuting known templates and options.
//!
//! Unlike the golden harvester, this one trust the `NixCodeGen` engine
//! to produce correct output for valid combinations.

use std::io::Write;
use std::path::PathBuf;

use symthaea::language::nix_broca_bridge::broca_channels_for_nix_prompt;
use symthaea::language::nix_codegen::{classify_nix_intent, generate_nix};

#[derive(serde::Serialize)]
struct DistillPair {
    prompt: String,
    intent: String,
    channels: Vec<f32>,
    code: String,
    iterations: usize,
    repair_steps: usize,
    holdout: bool,
}

fn main() {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let out_path = PathBuf::from(home)
        .join(".cache")
        .join("symthaea")
        .join("synthetic-distillation-pairs.jsonl");

    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = std::fs::File::create(&out_path).unwrap();
    let mut writer = std::io::BufWriter::new(file);

    let mut prompts = Vec::new();

    // ── Services ──
    let services = [
        "postgresql",
        "nginx",
        "redis",
        "podman",
        "docker",
        "tailscale",
        "prometheus",
        "grafana",
        "openssh",
        "cups",
        "resolved",
    ];
    for s in services {
        prompts.push(format!("enable {} server", s));
        prompts.push(format!("set up {}", s));
        prompts.push(format!("configure {} service", s));
    }

    // ── Service Combinations ──
    prompts.push("set up nginx and postgresql".to_string());
    prompts.push("enable redis and docker".to_string());
    prompts.push("configure prometheus and grafana".to_string());
    prompts.push("setup openssh with tailscale".to_string());

    // ── Desktop ──
    let desktops = ["gnome", "kde plasma", "sway", "hyprland"];
    for d in desktops {
        prompts.push(format!("enable {} desktop", d));
        prompts.push(format!("set up {} window manager", d));
    }

    // ── Hardware ──
    let hw = ["nvidia", "intel", "amd"];
    for h in hw {
        prompts.push(format!("configure {} gpu drivers", h));
        prompts.push(format!("enable {} hardware acceleration", h));
    }

    // ── Dev Shells ──
    let langs = ["rust", "python", "node", "typescript"];
    for l in langs {
        prompts.push(format!("set up a {} development environment", l));
        prompts.push(format!("create a {} dev shell", l));
    }

    // ── Networking ──
    prompts.push("open port 80 in firewall".to_string());
    prompts.push("open ports 80 and 443 in firewall".to_string());
    prompts.push("enable firewall and open port 22".to_string());

    println!("Generating {} synthetic pairs...", prompts.len());

    let mut count = 0;
    for prompt in prompts {
        let lower = prompt.to_lowercase();
        let intent = classify_nix_intent(&lower);
        let channels = broca_channels_for_nix_prompt(&prompt);
        let result = generate_nix(&prompt);

        if result.code.is_empty() || result.code == "{ }" {
            continue;
        }

        let pair = DistillPair {
            prompt: prompt.clone(),
            intent: format!("{intent:?}"),
            channels: channels.to_vec(),
            code: result.code,
            iterations: 0,
            repair_steps: 0,
            holdout: false,
        };

        let line = serde_json::to_string(&pair).unwrap();
        writeln!(writer, "{}", line).unwrap();
        count += 1;
    }

    writer.flush().unwrap();
    println!(
        "✓ Successfully generated {} synthetic pairs in {}",
        count,
        out_path.display()
    );
}
