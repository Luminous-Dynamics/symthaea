// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Corpus scraper — extract (prompt, golden) training-pair candidates from
//! a real NixOS config tree. Session-tractable first cut of #3 in the
//! coding-AI "make this even better" list (scale to ≥200 training pairs).
//!
//! Scope contrasts:
//! - `nix_kg_learn_from_config.rs` discovers service *keywords* from
//!   `/etc/nixos` → emits KG-JSON additions. It works on dotted paths.
//! - **This example** discovers service *configurations* (the full RHS
//!   shape the user actually chose) → emits training-pair candidates
//!   suitable for the scorer. It needs the AST to capture structure.
//!
//! Output is a **review queue**, not a training corpus. Human review
//! required before these can land in `src/language/nix_eval_corpus.rs` —
//! that discipline is non-negotiable per the original P5 plan ("Never
//! auto-append").
//!
//! Usage:
//!   cargo run --features code_generation --example nix_corpus_scrape \
//!       -- --dir /etc/nixos --out /tmp/corpus-review.jsonl
//!
//! Each output line:
//!   {"prompt": "...", "golden": "...", "attrpath": "services.nginx",
//!    "source_file": "/etc/nixos/services.nix",
//!    "source_line": 42, "richness": 3}
//!
//! Richness = count of non-comment inner assignments when the RHS is an
//! attrset literal. Lets the reviewer prioritize rich blocks over
//! trivial `enable = true;` one-liners.
//!
//! Interpretation:
//! - Richness 1: trivial (`services.X.enable = true`). Low training value
//!   — the existing 26 already cover that shape.
//! - Richness ≥3: rich config. High training value — teaches the model
//!   actual option shapes (ports, paths, sub-attrs).

use rnix::{NodeOrToken, Root, SyntaxKind, SyntaxNode};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Top-level roots that produce interesting training pairs. Matches the
/// KG learner's `SERVICE_LIKE_ROOTS` for consistency.
const CORPUS_ROOTS: &[&str] = &[
    "services",
    "programs",
    "virtualisation",
    "hardware",
    "networking",
    "environment", // environment.systemPackages is useful
];

#[derive(Debug, Serialize)]
struct CorpusCandidate {
    /// Template-generated English description.
    prompt: String,
    /// Self-contained Nix block suitable for scorer comparison.
    golden: String,
    /// The attrpath this block defines (e.g. `services.nginx`).
    attrpath: String,
    /// File this block was lifted from.
    source_file: String,
    /// 1-indexed line in the source file where the block starts.
    source_line: usize,
    /// Count of inner assignments when RHS is an attrset. 1 for
    /// `services.X.enable = true;` (flat), larger for `services.X = {
    /// enable = true; port = 8080; ... }`.
    richness: usize,
}

fn parse_flag(name: &str, default: Option<String>) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    for w in args.windows(2) {
        if w[0] == name {
            return Some(w[1].clone());
        }
    }
    default
}

fn default_out_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("symthaea")
        .join("corpus-review.jsonl")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = parse_flag("--dir", Some("/etc/nixos".to_string())).unwrap();
    let out = parse_flag("--out", None)
        .map(PathBuf::from)
        .unwrap_or_else(default_out_path);

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Symthaea corpus scraper (#3 feasibility — real NixOS)");
    println!("│ Source dir: {dir}");
    println!("│ Review queue: {}", out.display());
    println!("└─────────────────────────────────────────────────────────");

    let files = collect_nix_files_recursive(Path::new(&dir))?;
    if files.is_empty() {
        eprintln!("no .nix files under {dir}");
        std::process::exit(1);
    }
    println!("Walked {} .nix files", files.len());

    let mut candidates: Vec<CorpusCandidate> = Vec::new();
    let mut parse_failures = 0usize;
    for file in &files {
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        let parsed = Root::parse(&text);
        if !parsed.errors().is_empty() {
            parse_failures += 1;
            continue;
        }
        extract_candidates(&parsed.syntax(), &text, file, &[], &mut candidates);
    }
    println!(
        "Parsed {}/{} files cleanly, {} parse-failures skipped",
        files.len() - parse_failures,
        files.len(),
        parse_failures
    );
    println!("Extracted {} candidate blocks\n", candidates.len());

    // Richness histogram — lets us eyeball training-value distribution.
    let mut richness_hist: BTreeMap<usize, usize> = BTreeMap::new();
    for c in &candidates {
        *richness_hist.entry(c.richness).or_insert(0) += 1;
    }
    println!("─── Richness distribution ────────────────────────────────");
    for (r, count) in &richness_hist {
        let bar = "█".repeat((*count).min(40));
        println!("  r={r:>2}: {count:>4} {bar}");
    }
    println!();

    // Attrpath-root histogram.
    let mut root_hist: BTreeMap<String, usize> = BTreeMap::new();
    for c in &candidates {
        let root = c.attrpath.split('.').next().unwrap_or("?").to_string();
        *root_hist.entry(root).or_insert(0) += 1;
    }
    println!("─── Root distribution ────────────────────────────────────");
    for (root, count) in &root_hist {
        println!("  {root:>15} → {count}");
    }
    println!();

    // Write JSONL.
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(&out)?;
    use std::io::Write;
    for c in &candidates {
        let line = serde_json::to_string(c)?;
        writeln!(f, "{line}")?;
    }
    println!(
        "✓ Wrote {} candidates to {}",
        candidates.len(),
        out.display()
    );

    // Print a few sample candidates (prefer rich ones) for immediate
    // eyeball review.
    let mut rich_first: Vec<&CorpusCandidate> = candidates.iter().collect();
    rich_first.sort_by(|a, b| b.richness.cmp(&a.richness));
    println!();
    println!("─── Top 3 richest samples ────────────────────────────────");
    for c in rich_first.iter().take(3) {
        println!();
        println!("  prompt:   {:?}", c.prompt);
        println!("  attrpath: {}", c.attrpath);
        println!("  richness: {}", c.richness);
        println!("  source:   {}:{}", c.source_file, c.source_line);
        println!("  golden:");
        for line in c.golden.lines() {
            println!("    {line}");
        }
    }
    println!();
    println!("╔═════════════════════════════════════════════════════════");
    println!(
        "║ Next step (manual): review {} and accept rich entries",
        out.display()
    );
    println!("║ into nix_eval_corpus.rs. Richness-1 entries are usually");
    println!("║ redundant with the existing 26 pairs — prefer r≥3.");
    println!("╚═════════════════════════════════════════════════════════");

    Ok(())
}

/// Walk `.nix` files recursively. The KG learner only scans the top
/// level; real configs are split across `services.nix`, `hardware.nix`,
/// etc. inside subdirectories.
fn collect_nix_files_recursive(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                // Skip common noise directories.
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if matches!(name, ".git" | "result" | "target" | "node_modules") {
                    continue;
                }
                stack.push(p);
            } else if p.extension().and_then(|s| s.to_str()) == Some("nix") {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Walk the AST looking for `NODE_ATTRPATH_VALUE` nodes whose attrpath
/// starts with a corpus-interesting root. Emit a CorpusCandidate per
/// match with the source range of the value node as `golden`.
fn extract_candidates(
    node: &SyntaxNode,
    full_text: &str,
    file: &Path,
    prefix: &[String],
    out: &mut Vec<CorpusCandidate>,
) {
    for child in node.children() {
        match child.kind() {
            SyntaxKind::NODE_ATTRPATH_VALUE => {
                let mut sub = child.children();
                let Some(key) = sub.next() else { continue };
                if key.kind() != SyntaxKind::NODE_ATTRPATH {
                    continue;
                }
                let Some(value) = sub.next() else { continue };
                let Some(segs) = static_attrpath(&key) else {
                    continue;
                };

                let mut full: Vec<String> = prefix.to_vec();
                full.extend(segs);
                let root = full.first().map(|s| s.as_str()).unwrap_or("");
                if !CORPUS_ROOTS.contains(&root) {
                    // Not an interesting root. Still recurse through
                    // structural nodes (let-in, attrset, etc.) in case
                    // an interesting assignment is nested inside.
                    if value.kind() == SyntaxKind::NODE_ATTR_SET {
                        extract_candidates(&value, full_text, file, &full, out);
                    }
                    continue;
                }

                // Good candidate. Build the golden block.
                let value_text = value.text().to_string();
                let attrpath = full.join(".");
                let richness = count_inner_assignments(&value);
                let golden = format_golden(&attrpath, &value_text);
                let prompt = template_prompt(&full, &value);
                let source_line = line_number_of(&child, full_text);

                out.push(CorpusCandidate {
                    prompt,
                    golden,
                    attrpath,
                    source_file: file.to_string_lossy().into_owned(),
                    source_line,
                    richness,
                });

                // Nested attrset: we already captured the enclosing block.
                // Don't recurse into it further — we'd double-count child
                // assignments as separate (redundant) candidates.
            }
            SyntaxKind::NODE_ATTR_SET
            | SyntaxKind::NODE_LET_IN
            | SyntaxKind::NODE_LAMBDA
            | SyntaxKind::NODE_APPLY
            | SyntaxKind::NODE_PAREN
            | SyntaxKind::NODE_WITH => {
                extract_candidates(&child, full_text, file, prefix, out);
            }
            _ => {}
        }
    }
}

/// Copy of nix_scorer's static_attrpath (private there). Intentional
/// small duplication — scorer is a library concern, scraper is an
/// example, and keeping example deps minimal is policy.
fn static_attrpath(key_node: &SyntaxNode) -> Option<Vec<String>> {
    let mut segs = Vec::new();
    for child in key_node.children_with_tokens() {
        match child {
            NodeOrToken::Node(n) => match n.kind() {
                SyntaxKind::NODE_IDENT => segs.push(n.text().to_string()),
                SyntaxKind::NODE_DYNAMIC | SyntaxKind::NODE_STRING => return None,
                _ => {}
            },
            NodeOrToken::Token(_) => {}
        }
    }
    if segs.is_empty() {
        None
    } else {
        Some(segs)
    }
}

/// Count `NODE_ATTRPATH_VALUE` children inside an attrset literal.
/// Returns 1 for leaf values (the whole assignment counts as one).
fn count_inner_assignments(value_node: &SyntaxNode) -> usize {
    if value_node.kind() != SyntaxKind::NODE_ATTR_SET {
        return 1;
    }
    value_node
        .children()
        .filter(|c| c.kind() == SyntaxKind::NODE_ATTRPATH_VALUE)
        .count()
        .max(1)
}

/// Build a `{ attrpath = value; }` wrapper suitable for the scorer.
/// The scorer expects full module shapes; we wrap single assignments
/// to match.
fn format_golden(attrpath: &str, value_text: &str) -> String {
    format!("{{\n  {attrpath} = {value_text};\n}}\n")
}

/// Template-generate an English prompt from the attrpath + value shape.
/// Kept intentionally simple — the human review step is the quality
/// gate, not the template. Examples:
///   `services.nginx = { enable = true; ... }` → "enable nginx"
///   `services.postgresql.enable = true`       → "enable postgresql"
///   `hardware.graphics = { enable = true; extraPackages = ... }` →
///     "configure graphics hardware"
///   `environment.systemPackages = ...`        → "install system packages"
fn template_prompt(path: &[String], value: &SyntaxNode) -> String {
    let root = path.first().map(|s| s.as_str()).unwrap_or("");
    let name = path.get(1).map(|s| s.as_str()).unwrap_or("");

    let enable_action = match root {
        "services" => "enable",
        "programs" => "enable",
        "virtualisation" => "enable",
        "networking" => "configure",
        "hardware" => "configure",
        "environment" => "install",
        _ => "configure",
    };

    // If this is a `.enable = true;` leaf, the name is at path[1]
    // (root).(name)(.enable). Pick the most descriptive segment.
    let subject = if name.is_empty() {
        root.to_string()
    } else if root == "hardware" || root == "networking" {
        format!("{} {}", name, strip_dot_enable(path))
    } else {
        name.to_string()
    };

    // Augment with shape hints when the RHS is an attrset with well-known
    // subkeys. Cheap heuristic, no NLG magic.
    let hint = shape_hint(value);
    if hint.is_empty() {
        format!("{} {}", enable_action, subject.trim())
    } else {
        format!("{} {} with {}", enable_action, subject.trim(), hint)
    }
}

fn strip_dot_enable(path: &[String]) -> String {
    path.iter()
        .skip(2)
        .filter(|s| s.as_str() != "enable")
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Inspect an attrset RHS for well-known subkeys. Returns a human-
/// readable tail like "port 8080" or "ssl" or "data directory". Empty
/// if no recognized hint. Scope deliberately narrow — add keys as real
/// configs surface them during review.
fn shape_hint(value: &SyntaxNode) -> String {
    if value.kind() != SyntaxKind::NODE_ATTR_SET {
        return String::new();
    }
    let mut hints: Vec<String> = Vec::new();
    for inner in value.children() {
        if inner.kind() != SyntaxKind::NODE_ATTRPATH_VALUE {
            continue;
        }
        let mut sub = inner.children();
        let Some(k) = sub.next() else { continue };
        let Some(v) = sub.next() else { continue };
        let key = k.text().to_string();
        let val = v.text().to_string();
        let key_lower = key.to_lowercase();

        if key_lower.contains("port") && !key_lower.contains("ports") {
            hints.push(format!("port {}", val.trim()));
        } else if key_lower == "datadir" || key_lower.contains("dataDir") {
            hints.push("custom data directory".to_string());
        } else if key_lower == "enablessl" || key_lower.contains("ssl") {
            hints.push("SSL".to_string());
        } else if key_lower.contains("package") {
            hints.push("custom package".to_string());
        }
    }
    hints.join(" and ")
}

/// 1-indexed line number where a node starts, using the node's text
/// range offset. Walks the source string up to that offset counting
/// newlines — O(offset) but we only do it per-candidate.
fn line_number_of(node: &SyntaxNode, full_text: &str) -> usize {
    let offset: usize = node.text_range().start().into();
    let slice = &full_text[..offset.min(full_text.len())];
    slice.bytes().filter(|&b| b == b'\n').count() + 1
}
