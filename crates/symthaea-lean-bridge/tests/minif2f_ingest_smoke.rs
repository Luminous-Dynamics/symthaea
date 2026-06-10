// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end smoke tests for the Phase 4 miniF2F-v2 ingestion pipeline.
//!
//! Unit tests for `minif2f_ingest` (in the module itself) exercise the
//! tokenizer, parser, and translator against hand-written source
//! strings. These integration tests read actual `.lean` files from
//! `symthaea/data/benchmarks/minif2f/MiniF2F/` and run the full
//! `ingest()` pipeline on them.
//!
//! Scope: pinpoint the hand-curated fixtures from `phase3-findings.md`
//! whose shapes the current parser can handle (the 12/12 linear_real
//! category). The larger 50-problem zero-shot baseline will live in a
//! dedicated example binary that writes a CSV scorecard.

use std::fs;
use std::path::PathBuf;

use symthaea_core::hdc::fol_formula_ext::FolFormulaExt;
use symthaea_lean_bridge::minif2f_ingest::{ingest, parse_theorem};

/// Locate `symthaea/data/benchmarks/minif2f/MiniF2F/` relative to the
/// test binary's crate root. Skipped at runtime if the corpus hasn't
/// been downloaded (CI may not vendor the full miniF2F source).
fn minif2f_root() -> Option<PathBuf> {
    // CARGO_MANIFEST_DIR is `symthaea-lean-bridge/`. The corpus lives
    // two levels up at `symthaea/data/benchmarks/minif2f/MiniF2F/`.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()? // crates/
        .parent()? // symthaea/
        .join("data/benchmarks/minif2f/MiniF2F");
    root.exists().then_some(root)
}

/// Read `{Valid,Test}/<name>.lean`. Returns `None` if the file isn't in
/// the expected subdirectory of the corpus.
fn load_problem(name: &str) -> Option<String> {
    let root = minif2f_root()?;
    for sub in ["Valid", "Test"] {
        let p = root.join(sub).join(format!("{name}.lean"));
        if let Ok(s) = fs::read_to_string(&p) {
            return Some(s);
        }
    }
    None
}

#[test]
fn ingest_mathd_algebra_109_from_disk() {
    // Canonical linear_real signature, hand-translated 100%-accepted.
    let Some(src) = load_problem("mathd_algebra_109") else {
        eprintln!("miniF2F corpus not available; skipping");
        return;
    };
    let f = ingest(&src).expect("ingest should succeed on mathd_algebra_109");
    // Shape: ∀ a : ℝ, ∀ b : ℝ, h₀ → h₁ → b = 0
    assert!(
        matches!(f, FolFormulaExt::Forall(_, _, _)),
        "expected outer Forall, got {f:?}"
    );
}

#[test]
fn parse_then_translate_roundtrip_on_known_good_fixtures() {
    // Problems from phase3-findings.md's linear_real category — each
    // was hand-translated and Lake-accepted at 100%. Our automated
    // ingest should at least PARSE + TRANSLATE them cleanly; whether
    // the downstream cascade closes them is tested elsewhere.
    let names = [
        "mathd_algebra_109",
        "mathd_algebra_119",
        "mathd_algebra_126",
        "mathd_algebra_142",
    ];
    let mut parsed = 0;
    let mut translated = 0;
    let mut attempted = 0;
    for name in names {
        let Some(src) = load_problem(name) else {
            continue;
        };
        attempted += 1;
        match parse_theorem(&src) {
            Ok(t) => {
                parsed += 1;
                if let Ok(_f) = translate_or_forward(&t) {
                    translated += 1;
                }
            }
            Err(e) => {
                eprintln!("[parse] {name}: {e}");
            }
        }
    }
    if attempted == 0 {
        eprintln!("miniF2F corpus not available; skipping");
        return;
    }
    // Every fixture in the list is in the parser/translator's declared
    // scope. If any fail here, the parser or translator has regressed.
    assert_eq!(
        parsed, attempted,
        "parse rate should be 100% on known-good fixtures"
    );
    assert_eq!(
        translated, attempted,
        "translate rate should be 100% on known-good fixtures"
    );
}

/// Local shim naming the translate stage for staged measurement.
/// `translate_theorem` is `pub fn` but calling it directly in the test
/// body with the fully-qualified path inflates each line; the shim
/// keeps the scorecard loop readable.
fn translate_or_forward(
    t: &symthaea_lean_bridge::minif2f_ingest::LeanTheorem,
) -> Result<FolFormulaExt, symthaea_lean_bridge::minif2f_ingest::IngestError> {
    symthaea_lean_bridge::minif2f_ingest::translate_theorem(t)
}
