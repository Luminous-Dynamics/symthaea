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
//!   # Auto-merge into ~/.cache/symthaea/nix-kg.json after printing a diff
//!   # and reading a y/N confirmation from stdin:
//!   cargo run --release --features code_generation \
//!       --example nix_kg_learn_from_config -- /etc/nixos --write
//!
//!   # Skip confirmation (for unattended use):
//!   cargo run --release --features code_generation \
//!       --example nix_kg_learn_from_config -- /etc/nixos --write --yes
//!
//! With no positional argument, defaults to `/etc/nixos`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use symthaea::language::nix_kg::{NixKg, NixKgFile, SCHEMA_VERSION, ServiceKeyword};

/// Roots whose second segment is a service-name candidate.
const SERVICE_LIKE_ROOTS: &[&str] = &[
    "services",
    "programs",
    "virtualisation",
    "hardware",
    "networking",
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let write_mode = args.iter().any(|a| a == "--write");
    let assume_yes = args.iter().any(|a| a == "--yes");
    let target_dir = args
        .iter()
        .find(|a| !a.starts_with("--"))
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

    let suggested_file = NixKgFile {
        version: SCHEMA_VERSION,
        option_roots: vec![],
        conflicts: vec![],
        service_keywords: new_service_keywords.clone(),
        rag_prefixes: Default::default(),
    };
    let suggested_json = serde_json::to_string_pretty(&suggested_file).expect("encode JSON");

    if !write_mode {
        println!("─── Suggested ~/.cache/symthaea/nix-kg.json (additions only) ───");
        println!("{suggested_json}");
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
        println!("║ (Or re-run with --write to merge interactively.)");
        println!("║ The codegen pipeline reads this on next start (no recompile).");
        println!("╚═════════════════════════════════════════════════════════");
        return;
    }

    // ── Write mode: merge into ~/.cache/symthaea/nix-kg.json ──────────────
    let cache_path = default_kg_cache_path();
    println!("─── --write: will merge into {} ───", cache_path.display());

    // Load existing file (if any) so we can show the diff + do a union merge.
    let existing_file = read_kg_file(&cache_path);
    let existing_json = existing_file
        .as_ref()
        .map(|f| serde_json::to_string_pretty(f).expect("re-encode existing"))
        .unwrap_or_else(|| "{}".to_string());

    // Union-merge service_keywords. Existing takes priority on conflicts
    // (if same keyword already present, keep its option_path).
    let mut merged_keywords: Vec<ServiceKeyword> = existing_file
        .as_ref()
        .map(|f| f.service_keywords.clone())
        .unwrap_or_default();
    let existing_kws: BTreeSet<String> =
        merged_keywords.iter().map(|k| k.keyword.clone()).collect();
    for new in &new_service_keywords {
        if !existing_kws.contains(&new.keyword) {
            merged_keywords.push(new.clone());
        }
    }
    let merged_file = NixKgFile {
        version: SCHEMA_VERSION,
        option_roots: existing_file
            .as_ref()
            .map(|f| f.option_roots.clone())
            .unwrap_or_default(),
        conflicts: existing_file
            .as_ref()
            .map(|f| f.conflicts.clone())
            .unwrap_or_default(),
        service_keywords: merged_keywords.clone(),
        rag_prefixes: existing_file
            .as_ref()
            .map(|f| f.rag_prefixes.clone())
            .unwrap_or_default(),
    };
    let merged_json = serde_json::to_string_pretty(&merged_file).expect("encode merged JSON");

    if existing_json == merged_json {
        println!("✓ Nothing to add — every suggestion already in cache file.");
        return;
    }

    println!("─── Diff (existing → proposed) ─────────────────────────────");
    print_line_diff(&existing_json, &merged_json);
    println!();
    println!(
        "╔═ {} new keyword(s), cache at {} will reach {} total",
        merged_keywords.len() - existing_kws.len(),
        cache_path.display(),
        merged_keywords.len()
    );

    if !assume_yes {
        use std::io::Write;
        print!("║ Merge these additions into the cache file? [y/N] ");
        std::io::stdout().flush().ok();
        let mut answer = String::new();
        if std::io::stdin().read_line(&mut answer).is_err() {
            eprintln!("✗ Failed to read confirmation — aborting.");
            std::process::exit(2);
        }
        let yes = matches!(answer.trim().to_lowercase().as_str(), "y" | "yes");
        if !yes {
            println!("║ Aborted. No file was written.");
            println!("╚═════════════════════════════════════════════════════════");
            return;
        }
    }

    if let Some(parent) = cache_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("✗ Could not create cache dir {}: {e}", parent.display());
            std::process::exit(1);
        }
    }
    if let Err(e) = std::fs::write(&cache_path, &merged_json) {
        eprintln!("✗ Could not write {}: {e}", cache_path.display());
        std::process::exit(1);
    }
    println!("║ Wrote {}", cache_path.display());
    println!("╚═════════════════════════════════════════════════════════");
}

fn default_kg_cache_path() -> PathBuf {
    // Kept in sync with `nix_kg::default_path` (private). The path is
    // documented and the loader is stable — deliberate small
    // duplication to keep the example's dependency surface minimal.
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("symthaea")
        .join("nix-kg.json")
}

/// Read an existing NixKgFile from disk. Returns None if absent, corrupt,
/// or schema-version-mismatched (same policy as the production loader).
fn read_kg_file(path: &std::path::Path) -> Option<NixKgFile> {
    let bytes = std::fs::read(path).ok()?;
    let parsed: NixKgFile = serde_json::from_slice(&bytes).ok()?;
    if parsed.version != SCHEMA_VERSION {
        return None;
    }
    Some(parsed)
}

/// Crude line-based diff: `+` lines appear only in the new version,
/// `-` lines appear only in the old. Not a full LCS-based patch, but
/// enough to eyeball a 10-keyword addition list. Good-enough for a
/// human-confirmation UX.
fn print_line_diff(old: &str, new: &str) {
    let old_lines: BTreeSet<&str> = old.lines().collect();
    let new_lines: BTreeSet<&str> = new.lines().collect();
    for line in new.lines() {
        if !old_lines.contains(line) {
            println!("  + {line}");
        }
    }
    for line in old.lines() {
        if !new_lines.contains(line) {
            println!("  - {line}");
        }
    }
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
