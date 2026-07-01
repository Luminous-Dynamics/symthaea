// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Demonstrates the `nix_kg` user-override mechanism end-to-end.
//!
//! The bundled knowledge graph doesn't know about WireGuard. This demo:
//!   1. Shows `set up vaultwarden VPN` classifying as Generic with defaults
//!   2. Writes a small JSON override to a tempfile
//!   3. Loads the override + defaults via `NixKg::from_path_or_default`
//!   4. Shows the same prompt now classifying as Service
//!   5. Prints the JSON shape so users can copy-paste it to
//!      `~/.cache/symthaea/nix-kg.json`
//!
//! No code recompile needed to add new services — that was the whole
//! point of refactoring the hand-coded tables into the KG module.
//!
//! Run:
//!   cargo run --release --features code_generation --example nix_kg_override_demo

use std::collections::HashMap;
use std::path::PathBuf;

use symthaea::language::nix_kg::{NixKg, NixKgFile, SCHEMA_VERSION, ServiceKeyword};

fn matches_service_with_kg(kg: &NixKg, prompt: &str) -> bool {
    // Mirrors the gate inside `classify_nix_intent` — true means this
    // prompt would route to NixIntent::Service via the KG.
    let lower = prompt.to_lowercase();
    kg.matches_service_keyword(&lower)
}

fn main() {
    let prompt = "set up vaultwarden password vault";

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Symthaea nix_kg override demo");
    println!("│ Prompt: {prompt}");
    println!("└─────────────────────────────────────────────────────────");
    println!();

    // ─── Step 1: classify with bundled defaults ───
    let default_kg = NixKg::default();
    let recognized_default = matches_service_with_kg(&default_kg, prompt);
    println!("Bundled KG service-keyword match? {recognized_default}");
    println!("  (default service keywords include: tailscale, prometheus, jellyfin, redis, …)");
    println!();

    // ─── Step 2: build an override file in a tempdir ───
    let tmp_dir = std::env::temp_dir().join(format!(
        "symthaea_nix_kg_override_demo_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let override_path: PathBuf = tmp_dir.join("nix-kg.json");

    let override_file = NixKgFile {
        version: SCHEMA_VERSION,
        option_roots: vec![],
        conflicts: vec![],
        service_keywords: vec![
            ServiceKeyword {
                keyword: "vaultwarden".to_string(),
                option_path: Some("services.vaultwarden".to_string()),
            },
            ServiceKeyword {
                keyword: "password vault".to_string(),
                option_path: Some("services.vaultwarden".to_string()),
            },
        ],
        rag_prefixes: HashMap::new(),
    };

    let json = serde_json::to_string_pretty(&override_file).expect("encode JSON");
    std::fs::write(&override_path, &json).expect("write override file");
    println!("Wrote override to: {}", override_path.display());
    println!("─── JSON shape (copy to ~/.cache/symthaea/nix-kg.json) ───");
    println!("{json}");
    println!();

    // ─── Step 3: load with the override applied ───
    let extended_kg = NixKg::from_path_or_default(&override_path);
    let recognized_extended = matches_service_with_kg(&extended_kg, prompt);
    println!("Extended KG service-keyword match? {recognized_extended}");
    println!();

    // ─── Step 4: confirm defaults still present (additive merge) ───
    let still_knows_tailscale = extended_kg.matches_service_keyword("set up tailscale vpn");
    println!(
        "Extended KG still knows tailscale? {still_knows_tailscale} (additive merge — \
         bundled entries kept)"
    );
    println!();

    // ─── Cleanup ───
    let _ = std::fs::remove_file(&override_path);
    let _ = std::fs::remove_dir(&tmp_dir);

    println!("╔═════════════════════════════════════════════════════════");
    println!("║ Result");
    println!("╠═════════════════════════════════════════════════════════");
    println!("║ Default KG recognized {prompt:?}?  {recognized_default}");
    println!("║ + override:                              {recognized_extended}");
    println!("║ Defaults preserved (tailscale)?          {still_knows_tailscale}");
    println!("╚═════════════════════════════════════════════════════════");

    if !recognized_default && recognized_extended && still_knows_tailscale {
        println!("\n✓ Override mechanism works as designed:");
        println!("  - Bundled KG didn't recognize 'vaultwarden'");
        println!("  - User-supplied JSON added it");
        println!("  - Existing entries (tailscale) survived the merge");
        println!("\nTo extend coverage in production, write to:\n  ~/.cache/symthaea/nix-kg.json");
    } else {
        eprintln!("\n✗ Unexpected outcome — check the merge logic.");
        std::process::exit(1);
    }
}
