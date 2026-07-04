// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Single-shot content prefetch: reads a named set of paths wholesale and hands them
//! to the model in its first turn, instead of making it plan a sequence of
//! `read_file` calls across turns to discover the same content.
//!
//! This targets a specific weakness of smaller/local models: multi-turn tool-use
//! planning is harder for them than reading a lot of text in one pass. The model
//! keeps full tool access afterward — this only removes the *first* exploration step,
//! it doesn't remove the ability to look further.

use std::path::Path;

use crate::tools::{Sandbox, SandboxViolation};

/// Total byte budget across all prefetched files, to keep the first turn from blowing
/// the model's context window on a large directory.
const DEFAULT_MAX_BYTES: usize = 60_000;

/// Reads every path in `paths` (files read directly; directories walked, respecting
/// `.gitignore`, `.rs`/common source files only) through `sandbox`, concatenating
/// their contents with clear file headers until `max_bytes` is exhausted.
pub fn prefetch(sandbox: &Sandbox, paths: &[String], max_bytes: usize) -> String {
    let mut out = String::new();
    let mut budget = max_bytes;
    let mut truncated_files = Vec::new();

    for raw_path in paths {
        let path = raw_path.trim();
        if path.is_empty() {
            continue;
        }
        match collect_files(sandbox, path) {
            Ok(files) => {
                for (rel_path, content) in files {
                    if budget == 0 {
                        truncated_files.push(rel_path);
                        continue;
                    }
                    let take = content.len().min(budget);
                    out.push_str(&format!("=== FILE: {rel_path} ===\n"));
                    out.push_str(&content[..take]);
                    if take < content.len() {
                        out.push_str("\n... (truncated, byte budget exhausted)\n");
                    }
                    out.push_str("\n\n");
                    budget -= take;
                }
            }
            Err(e) => {
                out.push_str(&format!("=== FILE: {path} (unreadable: {e}) ===\n\n"));
            }
        }
    }

    if !truncated_files.is_empty() {
        out.push_str(&format!(
            "(byte budget exhausted before reading: {})\n",
            truncated_files.join(", ")
        ));
    }
    out
}

pub fn prefetch_default(sandbox: &Sandbox, paths: &[String]) -> String {
    prefetch(sandbox, paths, DEFAULT_MAX_BYTES)
}

fn is_source_like(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("rs" | "toml" | "md" | "py" | "js" | "ts" | "go" | "java" | "c" | "cpp" | "h" | "hpp")
    )
}

/// Returns `(relative_path_display, content)` pairs for `path` — a single file, or
/// every source-like file under a directory.
fn collect_files(sandbox: &Sandbox, path: &str) -> Result<Vec<(String, String)>, SandboxViolation> {
    let resolved = sandbox.resolve_for_prefetch(path)?;
    if resolved.is_file() {
        let preview = sandbox.read_file(path)?;
        return Ok(vec![(path.to_string(), preview.content)]);
    }
    let mut files = Vec::new();
    let walker = ignore::WalkBuilder::new(&resolved).build();
    for entry in walker.flatten() {
        let entry_path = entry.path();
        if !entry_path.is_file() || !is_source_like(entry_path) {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(entry_path) else {
            continue;
        };
        let rel = entry_path
            .strip_prefix(sandbox.root())
            .unwrap_or(entry_path);
        files.push((rel.display().to_string(), content));
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn build_sandbox() -> (tempfile::TempDir, Sandbox) {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("repo");
        fs::create_dir_all(root.join("src/sub")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("src/sub/inner.rs"), "// inner").unwrap();
        fs::write(root.join("README.md"), "# readme").unwrap();
        let sandbox = Sandbox::new(&root, vec![]).unwrap();
        (tmp, sandbox)
    }

    #[test]
    fn prefetch_single_file() {
        let (_tmp, sandbox) = build_sandbox();
        let out = prefetch(&sandbox, &["src/main.rs".to_string()], 10_000);
        assert!(out.contains("FILE: src/main.rs"));
        assert!(out.contains("fn main"));
    }

    #[test]
    fn prefetch_directory_recursive() {
        let (_tmp, sandbox) = build_sandbox();
        let out = prefetch(&sandbox, &["src".to_string()], 10_000);
        assert!(out.contains("main.rs"));
        assert!(out.contains("inner.rs"));
    }

    #[test]
    fn prefetch_respects_byte_budget() {
        let (_tmp, sandbox) = build_sandbox();
        let out = prefetch(&sandbox, &["src".to_string()], 5);
        assert!(out.contains("truncated") || out.contains("byte budget exhausted"));
    }

    #[test]
    fn prefetch_denies_sandbox_escape() {
        let (_tmp, sandbox) = build_sandbox();
        let out = prefetch(&sandbox, &["../../etc/passwd".to_string()], 10_000);
        assert!(out.contains("unreadable"));
    }

    #[test]
    fn prefetch_skips_empty_and_blank_entries() {
        let (_tmp, sandbox) = build_sandbox();
        let out = prefetch(&sandbox, &["".to_string(), "  ".to_string()], 10_000);
        assert!(out.is_empty());
    }
}
