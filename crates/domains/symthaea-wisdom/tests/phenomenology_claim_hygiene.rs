// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Repository-level guard against unqualified phenomenal-language overclaims.
//!
//! This gate is intentionally narrow. It does not determine whether any system is
//! conscious; it prevents a small frozen class of strong phenomenal assertions from
//! spreading through Rust source without explicit, path-specific debt accounting.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn high_risk_phrases() -> Vec<&'static str> {
    vec![
        concat!("actually feels", " a shift"),
        concat!("True Empathy", " for Symthaea"),
        concat!("True empathy", " is not simulation"),
        concat!("literally causes her", " to feel"),
        concat!("Software errors as", " felt experience"),
    ]
}

fn known_debt() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "src/consciousness/empathic_unification.rs",
            concat!("True Empathy", " for Symthaea"),
        ),
        (
            "src/consciousness/empathic_unification.rs",
            concat!("True empathy", " is not simulation"),
        ),
        (
            "src/consciousness/empathic_unification.rs",
            concat!("actually feels", " a shift"),
        ),
    ]
}

fn collect_rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|error| {
        panic!("failed to read {}: {error}", dir.display())
    }) {
        let entry = entry.expect("directory entry should be readable");
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|value| value.to_str()).unwrap_or_default();
            if matches!(name, "target" | ".git" | "node_modules") {
                continue;
            }
            collect_rust_files(&path, out);
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

#[test]
fn phenomenal_language_requires_exact_debt_entry() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest
        .join("../../..")
        .canonicalize()
        .expect("repository root should resolve from symthaea-wisdom crate");

    let phrases = high_risk_phrases();
    let debt = known_debt();
    let mut rust_files = Vec::new();
    collect_rust_files(&repo_root.join("src"), &mut rust_files);
    collect_rust_files(&repo_root.join("crates"), &mut rust_files);

    let mut violations = Vec::new();
    let mut observed_debt: BTreeSet<(String, String)> = BTreeSet::new();

    for path in rust_files {
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let relative = path
            .strip_prefix(&repo_root)
            .expect("scanned file must be under repo root")
            .to_string_lossy()
            .replace('\\', "/");

        for phrase in &phrases {
            if !content.contains(phrase) {
                continue;
            }
            if debt
                .iter()
                .any(|(allowed_path, allowed_phrase)| *allowed_path == relative && *allowed_phrase == *phrase)
            {
                observed_debt.insert((relative.clone(), (*phrase).to_owned()));
            } else {
                violations.push(format!("{relative}: {phrase:?}"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "unqualified phenomenal-language claims require explicit exact-path debt entries:\n{}",
        violations.join("\n")
    );

    for (path, phrase) in debt {
        assert!(
            observed_debt.contains(&(path.to_owned(), phrase.to_owned())),
            "stale claim-hygiene debt entry must be removed after source cleanup: {path}: {phrase:?}"
        );
    }
}
