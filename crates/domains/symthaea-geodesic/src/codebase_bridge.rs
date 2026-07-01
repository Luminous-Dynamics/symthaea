// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Codebase Bridge — index Rust files into ProgramManifold.
//!
//! Scans a directory for `.rs` files, extracts functions, builds PDGs,
//! computes topological fingerprints, and populates the manifold.
//!
//! Uses only `std::fs` for directory traversal (no external `walkdir` dependency).

use crate::ast_bridge::pdg_from_source_multi;
use crate::manifold::ProgramManifold;
use crate::topology::TopologicalFingerprint;
use std::path::Path;

/// Maximum file size to index (100 KB). Larger files are skipped to avoid
/// spending excessive time on generated code, test fixtures, or vendored deps.
const MAX_FILE_SIZE: u64 = 100 * 1024;

/// Directories to skip during recursive scanning.
const SKIP_DIRS: &[&str] = &["target", ".git", "node_modules", ".hg", "__pycache__"];

// ---------------------------------------------------------------------------
// Index result
// ---------------------------------------------------------------------------

/// Index result with detailed statistics.
#[derive(Debug)]
pub struct IndexResult {
    /// Number of `.rs` files scanned (including those with 0 functions).
    pub files_scanned: usize,
    /// Total function declarations found across all files.
    pub functions_found: usize,
    /// Functions successfully indexed into the manifold.
    pub functions_indexed: usize,
    /// New fibers created during indexing.
    pub fibers_created: usize,
    /// Errors encountered during scanning (non-fatal).
    pub errors: Vec<String>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Scan a directory for `.rs` files and index all functions into the manifold.
///
/// For each file:
/// 1. Read source text.
/// 2. Split into functions via [`pdg_from_source_multi`].
/// 3. For each function: build PDG → simplicial complex → topological
///    fingerprint → insert into manifold using the fingerprint's HDC encoding.
///
/// # Arguments
///
/// * `manifold` — The program manifold to populate.
/// * `root` — Root directory to scan recursively.
/// * `max_files` — Maximum number of files to process (0 = unlimited).
///
/// # Skipping
///
/// - Files larger than 100 KB are skipped.
/// - Directories named `target`, `.git`, `node_modules`, `.hg`, or
///   `__pycache__` are skipped.
pub fn index_directory(
    manifold: &mut ProgramManifold,
    root: &Path,
    max_files: usize,
) -> IndexResult {
    let fibers_before = manifold.fiber_count();
    let mut result = IndexResult {
        files_scanned: 0,
        functions_found: 0,
        functions_indexed: 0,
        fibers_created: 0,
        errors: Vec::new(),
    };

    // Collect .rs file paths via recursive scan.
    let mut file_paths: Vec<std::path::PathBuf> = Vec::new();
    collect_rs_files(root, &mut file_paths, &mut result.errors);

    // Sort for deterministic ordering.
    file_paths.sort();

    // Apply max_files limit.
    let limit = if max_files == 0 {
        file_paths.len()
    } else {
        max_files.min(file_paths.len())
    };

    for path in file_paths.into_iter().take(limit) {
        // Check file size.
        let metadata = match std::fs::metadata(&path) {
            Ok(m) => m,
            Err(e) => {
                result
                    .errors
                    .push(format!("{}: metadata error: {e}", path.display()));
                continue;
            }
        };

        if metadata.len() > MAX_FILE_SIZE {
            continue; // silently skip oversized files
        }

        let source = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                result
                    .errors
                    .push(format!("{}: read error: {e}", path.display()));
                continue;
            }
        };

        let indexed = index_file(manifold, &path, &source);
        result.files_scanned += 1;
        result.functions_found += indexed; // index_file returns count of found == indexed
        result.functions_indexed += indexed;
    }

    result.fibers_created = manifold.fiber_count().saturating_sub(fibers_before);
    result
}

/// Index a single Rust source file into the manifold.
///
/// Extracts all top-level function declarations, builds a PDG for each,
/// computes the topological fingerprint, and inserts into the manifold.
///
/// Returns the number of functions successfully indexed.
pub fn index_file(manifold: &mut ProgramManifold, path: &Path, source: &str) -> usize {
    let pdg_pairs = pdg_from_source_multi(source);
    let mut count = 0;

    for (name, pdg) in pdg_pairs {
        let complex = pdg.to_simplicial_complex();
        let fingerprint = TopologicalFingerprint::from_complex(&complex);

        // Use a qualified name: "path::function_name" for manifold lookups.
        let qualified = format!(
            "{}::{}",
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown"),
            name
        );

        // Insert using topology-derived encoding (same strategy as
        // bootstrap_with_topology): functions with identical Betti numbers
        // cluster into the same fiber.
        // Store function source for retrieval-based slot filling.
        // Extract the function body from the original source (approximate).
        let fn_source = extract_function_source(source, &name);
        manifold.insert_with_source(
            &qualified,
            fingerprint.hdc_encoding,
            fingerprint,
            0.5,
            fn_source,
        );
        count += 1;
    }

    count
}

/// Extract a function's source from the file source by name.
/// Finds `fn name(` and extracts until the matching closing brace.
fn extract_function_source(source: &str, name: &str) -> Option<String> {
    let pattern = format!("fn {name}(");
    let alt_pattern = format!("fn {name}<");
    let start = source
        .find(&pattern)
        .or_else(|| source.find(&alt_pattern))?;

    // Walk back to find `pub` or line start
    let line_start = source[..start].rfind('\n').map(|p| p + 1).unwrap_or(0);

    // Walk forward counting braces to find the matching close
    let mut depth = 0;
    let mut found_open = false;
    for (i, ch) in source[line_start..].char_indices() {
        match ch {
            '{' => {
                depth += 1;
                found_open = true;
            }
            '}' => {
                depth -= 1;
                if found_open && depth == 0 {
                    return Some(source[line_start..line_start + i + 1].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Internal: recursive directory scanner
// ---------------------------------------------------------------------------

/// Recursively collect `.rs` file paths under `dir`, skipping excluded directories.
fn collect_rs_files(dir: &Path, out: &mut Vec<std::path::PathBuf>, errors: &mut Vec<String>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            errors.push(format!("{}: read_dir error: {e}", dir.display()));
            return;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                errors.push(format!("{}: entry error: {e}", dir.display()));
                continue;
            }
        };

        let path = entry.path();
        let file_name = entry.file_name();
        let name_str = file_name.to_string_lossy();

        if path.is_dir() {
            // Skip excluded directories.
            if SKIP_DIRS.iter().any(|&skip| name_str == skip) {
                continue;
            }
            collect_rs_files(&path, out, errors);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Helper: create a temporary directory for testing.
    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("symthaea_geodesic_test_{name}"));
        let _ = fs::remove_dir_all(&dir); // clean up any prior run
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// Empty directory should yield 0 files and 0 functions.
    #[test]
    fn test_index_empty_directory() {
        let dir = temp_dir("empty");
        let mut manifold = ProgramManifold::new();

        let result = index_directory(&mut manifold, &dir, 0);

        assert_eq!(result.files_scanned, 0);
        assert_eq!(result.functions_found, 0);
        assert_eq!(result.functions_indexed, 0);
        assert_eq!(result.fibers_created, 0);
        assert!(result.errors.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    /// Single file with 2 functions → 2 functions indexed.
    #[test]
    fn test_index_single_file() {
        let dir = temp_dir("single");
        let source = r#"
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn multiply(a: i32, b: i32) -> i32 {
    let result = a * b;
    result
}
"#;
        fs::write(dir.join("math.rs"), source).expect("write file");

        let mut manifold = ProgramManifold::new();
        let result = index_directory(&mut manifold, &dir, 0);

        assert_eq!(result.files_scanned, 1);
        assert_eq!(result.functions_found, 2);
        assert_eq!(result.functions_indexed, 2);
        assert!(result.fibers_created > 0, "should create at least 1 fiber");
        assert_eq!(manifold.total_points(), 2);

        let _ = fs::remove_dir_all(&dir);
    }

    /// File larger than 100 KB should be skipped.
    #[test]
    fn test_index_skips_large_files() {
        let dir = temp_dir("large");

        // Create a file slightly over 100 KB.
        let big_source = format!(
            "fn big() {{\n{}\n}}",
            "    let x = 1;\n".repeat(10_000) // ~150 KB
        );
        assert!(
            big_source.len() > MAX_FILE_SIZE as usize,
            "test file should exceed 100KB, got {}",
            big_source.len()
        );
        fs::write(dir.join("big.rs"), &big_source).expect("write big file");

        // Also create a small file that should be indexed.
        fs::write(dir.join("small.rs"), "fn tiny() {\n    let x = 1;\n}\n")
            .expect("write small file");

        let mut manifold = ProgramManifold::new();
        let result = index_directory(&mut manifold, &dir, 0);

        // Only the small file should be scanned.
        assert_eq!(result.files_scanned, 1, "only small file should be scanned");
        assert_eq!(result.functions_indexed, 1, "only tiny() from small file");

        let _ = fs::remove_dir_all(&dir);
    }
}
