// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Learn KG entries from an existing NixOS config tree.
//!
//! Walks `.nix` files in `/etc/nixos` (or a path passed on the CLI),
//! extracts the second-level segment of every dotted option path
//! (`services.X.Y` → `X`), cross-references against the bundled
//! `nix_kg::NixKg` defaults, and emits a starter override JSON for
//! services/programs/hardware/virtualisation that the bundled KG doesn't
//! already know about.
//!
//! Day-1 user can copy the printed JSON into
//! `~/.cache/symthaea/nix-kg.json` and the codegen pipeline will route
//! prompts that mention those services correctly without recompiling.
//!
//! The regex-based extraction is deliberately simple — it doesn't
//! understand Nix scoping, just `^|[\s{(]<root>\.<word>` patterns. False
//! positives are bounded: the user reviews the printed JSON before
//! merging.
//!
//! Usage:
//!   cargo run --release --features code_generation \
//!       --example nix_kg_learn_from_config -- /etc/nixos
//!
//! With no argument, defaults to `/etc/nixos`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use symthaea::language::nix_kg::{NixKg, NixKgFile, ServiceKeyword, SCHEMA_VERSION};

/// Roots whose second segment is a service-name candidate.
const SERVICE_LIKE_ROOTS: &[&str] = &[
    "services",
    "programs",
    "virtualisation",
    "hardware",
    "networking",
];

fn main() {
    let target_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/etc/nixos"));

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Symthaea nix_kg config learner");
    println!("│ Source: {}", target_dir.display());
    println!("└─────────────────────────────────────────────────────────");

    let nix_files: Vec<PathBuf> = match collect_nix_files(&target_dir) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("could not read {}: {e}", target_dir.display());
            std::process::exit(1);
        }
    };
    if nix_files.is_empty() {
        eprintln!("no .nix files found in {}", target_dir.display());
        std::process::exit(1);
    }
    println!("Found {} .nix files\n", nix_files.len());

    // Map: root ("services" / "hardware" / …) → set of second-segments.
    let mut by_root: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut total_paths = 0usize;
    for file in &nix_files {
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        for path in extract_option_paths(&text) {
            total_paths += 1;
            let mut parts = path.splitn(3, '.');
            let root = parts.next().unwrap_or("").to_string();
            let second = parts.next().unwrap_or("").to_string();
            if root.is_empty() || second.is_empty() {
                continue;
            }
            by_root.entry(root).or_default().insert(second);
        }
    }
    println!("Scanned {total_paths} dotted-path occurrences\n");

    // Cross-reference with bundled KG. Anything the bundled KG already
    // recognizes (as a service keyword) is omitted — we only suggest
    // ADDITIONS the user would actually benefit from.
    let bundled = NixKg::default();
    let mut new_service_keywords: Vec<ServiceKeyword> = Vec::new();
    let mut already_known: BTreeSet<String> = BTreeSet::new();

    println!("─── Per-root summary ─────────────────────────────────────");
    for (root, seconds) in &by_root {
        let mut new_count = 0usize;
        let mut bundled_count = 0usize;
        for second in seconds {
            if !SERVICE_LIKE_ROOTS.iter().any(|r| *r == root.as_str()) {
                continue;
            }
            if bundled.matches_service_keyword(&second.to_lowercase()) {
                bundled_count += 1;
                already_known.insert(second.clone());
            } else {
                new_count += 1;
                new_service_keywords.push(ServiceKeyword {
                    keyword: second.to_lowercase(),
                    option_path: Some(format!("{root}.{second}")),
                });
            }
        }
        println!(
            "  {root:>15} → {:>3} entries  ({} new, {} already in bundled KG)",
            seconds.len(),
            new_count,
            bundled_count
        );
    }
    println!();

    // De-duplicate (a service like "nginx" might appear under multiple
    // roots). Keep the FIRST suggested option_path.
    let mut seen: BTreeSet<String> = BTreeSet::new();
    new_service_keywords.retain(|s| seen.insert(s.keyword.clone()));

    if new_service_keywords.is_empty() {
        println!("✓ The bundled KG already covers every service in this config.");
        println!("  Nothing to suggest.");
        return;
    }

    let suggested = NixKgFile {
        version: SCHEMA_VERSION,
        option_roots: vec![],
        conflicts: vec![],
        service_keywords: new_service_keywords.clone(),
        rag_prefixes: Default::default(),
    };
    let json = serde_json::to_string_pretty(&suggested).expect("encode JSON");

    println!("─── Suggested ~/.cache/symthaea/nix-kg.json (additions only) ───");
    println!("{json}");
    println!();

    println!("╔═════════════════════════════════════════════════════════");
    println!(
        "║ Suggested {} new service keywords from your config",
        new_service_keywords.len()
    );
    println!(
        "║ ({} services were already in the bundled KG)",
        already_known.len()
    );
    println!("╠═════════════════════════════════════════════════════════");
    println!("║ To apply: review and merge into ~/.cache/symthaea/nix-kg.json");
    println!("║ The codegen pipeline reads this on next start (no recompile).");
    println!("╚═════════════════════════════════════════════════════════");
}

fn collect_nix_files(dir: &std::path::Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("nix") {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

/// Pull `<root>.<segment>(.<segment>)*` paths from a Nix source string.
/// Skips strings and `# ...` comments. Returns paths that start with one
/// of `SERVICE_LIKE_ROOTS`. Robust enough for `/etc/nixos/*.nix` real-world
/// files — not a full Nix parser.
fn extract_option_paths(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        // Skip "..." strings (single-line; multi-line ''...'' is rarer for
        // option-path-shaped content)
        if bytes[i] == b'"' {
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            i += 1;
            continue;
        }
        // Skip line comments
        if bytes[i] == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // Identifier start?
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric()
                    || bytes[i] == b'_'
                    || bytes[i] == b'-'
                    || bytes[i] == b'.')
            {
                i += 1;
            }
            let span = &text[start..i];
            if let Some(root) = span.split('.').next() {
                if SERVICE_LIKE_ROOTS.iter().any(|r| *r == root)
                    && span.contains('.')
                    && !span.starts_with('.')
                    && !span.ends_with('.')
                {
                    out.push(span.to_string());
                }
            }
            continue;
        }
        i += 1;
    }
    out
}
