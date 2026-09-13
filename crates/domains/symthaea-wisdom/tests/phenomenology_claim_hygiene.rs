// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Repository-level guard against unqualified phenomenal-language overclaims.
//!
//! This gate is intentionally narrow. It does not determine whether any system is
//! conscious; it prevents a small frozen class of strong phenomenal assertions from
//! spreading through Rust source without explicit, path-specific debt accounting.

use std::collections::BTreeMap;
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

/// Exact legacy debt: `(relative path, exact phrase, allowed occurrence count)`.
/// Counts are frozen so an allowlisted file cannot accumulate additional copies.
fn known_debt() -> Vec<(&'static str, &'static str, usize)> {
    vec![
        (
            "src/consciousness/empathic_unification.rs",
            concat!("True Empathy", " for Symthaea"),
            1,
        ),
        (
            "src/consciousness/empathic_unification.rs",
            concat!("True empathy", " is not simulation"),
            1,
        ),
        (
            "src/consciousness/empathic_unification.rs",
            concat!("actually feels", " a shift"),
            1,
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
fn phenomenal_language_requires_exact_debt_entry_and_count() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest
        .join("../../..")
        .canonicalize()
        .expect("repository root should resolve from symthaea-wisdom crate");

    let phrases = high_risk_phrases();
    let debt = known_debt();
    let debt_map: BTreeMap<(&str, &str), usize> = debt
        .iter()
        .map(|(path, phrase, count)| ((*path, *phrase), *count))
        .collect();

    let mut rust_files = Vec::new();
    collect_rust_files(&repo_root.join("src"), &mut rust_files);
    collect_rust_files(&repo_root.join("crates"), &mut rust_files);

    let mut violations = Vec::new();
    let mut observed: BTreeMap<(String, String), usize> = BTreeMap::new();

    for path in rust_files {
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let relative = path
            .strip_prefix(&repo_root)
            .expect("scanned file must be under repo root")
            .to_string_lossy()
            .replace('\\', "/");

        for phrase in &phrases {
            let count = content.matches(phrase).count();
            if count == 0 {
                continue;
            }

            observed.insert((relative.clone(), (*phrase).to_owned()), count);
            match debt_map.get(&(relative.as_str(), *phrase)) {
                Some(expected) if *expected == count => {}
                Some(expected) => violations.push(format!(
                    "{relative}: {phrase:?} occurs {count} times; frozen debt allows {expected}"
                )),
                None => violations.push(format!(
                    "{relative}: {phrase:?} occurs {count} time(s) without an exact debt entry"
                )),
            }
        }
    }

    assert!(
        violations.is_empty(),
        "unqualified phenomenal-language claims exceed frozen debt:\n{}",
        violations.join("\n")
    );

    for (path, phrase, expected) in debt {
        let actual = observed
            .get(&(path.to_owned(), phrase.to_owned()))
            .copied()
            .unwrap_or(0);
        assert_eq!(
            actual, expected,
            "stale or changed claim-hygiene debt must be reconciled: {path}: {phrase:?}"
        );
    }
}
