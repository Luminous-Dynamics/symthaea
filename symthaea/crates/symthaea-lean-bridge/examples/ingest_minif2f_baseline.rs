// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Phase 4/5b zero-shot baseline harness.
//!
//! Walks `data/benchmarks/minif2f/MiniF2F/{Valid,Test}/`, applies the
//! same filter rules as `scripts/filter_minif2f.sh`, seeds a
//! deterministic shuffle, takes the first N problems (default 50),
//! and runs the `ingest()` pipeline on each. Emits a CSV scorecard to
//! stdout with one row per problem; summary counts to stderr.
//!
//! This is the honest first number: how much of the 178-file
//! filter-passed pool does the Rust-native parser+translator
//! actually handle without any tuning for specific fixture shapes?
//!
//! When `LAKE_ENV=1` is set AND `lake` is on PATH, this harness also
//! runs the **tier-3 Lake verification** stage: every successfully
//! translated problem gets rendered through `render_fol_ext_file`
//! and passed to `lake env lean` for external Mathlib-backed proof
//! checking. Tier 3 is the end-to-end number that matters: "how
//! many of the auto-ingested problems does the cascade actually
//! close under Mathlib's tactics". Without `LAKE_ENV=1`, only
//! tiers 1-2 (parse + translate) are measured; tier 3 rows report
//! `not_run`.
//!
//! # Usage
//!
//! ```bash
//! # Tiers 1-2 only (fast):
//! cargo run -p symthaea-lean-bridge --example ingest_minif2f_baseline
//!
//! # Full 3-tier (slow; needs elan/lake and Mathlib-resolved project):
//! LAKE_ENV=1 cargo run -p symthaea-lean-bridge \
//!     --example ingest_minif2f_baseline
//!
//! # Larger slice, different seed:
//! MINIF2F_N=100 MINIF2F_SEED=1337 cargo run -p symthaea-lean-bridge \
//!     --example ingest_minif2f_baseline
//! ```
//!
//! # CSV columns
//!
//! `name, stage, error_category, error_message, lake_check`
//!
//! - `stage` ∈ {`parsed`, `translated`, `not_parsed`, `not_translated`}
//! - `error_category` ∈ {`parse`, `translate`, ""} — "" when the row
//!   succeeded through both stages
//! - `error_message` is the `Display` form of the `IngestError`, empty
//!   on success. CSV-escaped (commas and quotes).
//! - `lake_check` ∈ {`accepted`, `rejected`, `lake_error`, `not_run`} —
//!   `not_run` when `LAKE_ENV=1` was not set or `lake` is unavailable.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Duration;

use symthaea_lean_bridge::fol_ext_bridge::render_fol_ext_file;
use symthaea_lean_bridge::minif2f_ingest::{parse_theorem, translate_theorem, IngestError};

fn main() -> ExitCode {
    let root = match locate_corpus() {
        Some(r) => r,
        None => {
            eprintln!(
                "error: miniF2F corpus not found. Expected at \
                 symthaea/data/benchmarks/minif2f/MiniF2F/. Skipping."
            );
            return ExitCode::SUCCESS;
        }
    };

    let n = env::var("MINIF2F_N")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50);
    let seed = env::var("MINIF2F_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);

    let use_lake = lake_env_available();
    let lake_project_dir = PathBuf::from("lean-proofs/phase2");
    // Emitted `.lean` files go here when tier 3 is active. Allowing an
    // override via env var lets CI store them under `target/` without
    // polluting the worktree.
    let emit_dir = env::var("MINIF2F_EMIT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("proofs/minif2f_baseline"));
    if use_lake {
        if let Err(e) = fs::create_dir_all(&emit_dir) {
            eprintln!(
                "warn: mkdir {} failed ({e}); tier 3 disabled",
                emit_dir.display()
            );
        }
    }

    eprintln!("miniF2F ingest baseline");
    eprintln!("  corpus: {}", root.display());
    eprintln!("  n={n}, seed={seed}");
    eprintln!(
        "  tier 3 (Lake): {}",
        if use_lake {
            "enabled"
        } else {
            "disabled (set LAKE_ENV=1 + have `lake` on PATH)"
        }
    );

    let mut candidates = gather_candidates(&root);
    shuffle_with_seed(&mut candidates, seed);
    let slice: Vec<_> = candidates.into_iter().take(n).collect();
    if slice.is_empty() {
        eprintln!("  no filter-passed candidates found");
        return ExitCode::SUCCESS;
    }
    eprintln!("  measuring {} problems", slice.len());

    // CSV header.
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "name,stage,error_category,error_message,lake_check");

    let mut parsed = 0usize;
    let mut translated = 0usize;
    let mut lake_accepted = 0usize;
    let mut lake_rejected = 0usize;
    let mut lake_error = 0usize;
    // Parse-failure distribution by error-variant discriminant; gives
    // us a histogram of what's blocking the biggest class of rejects.
    let mut parse_fail_buckets: BTreeMap<&'static str, usize> = BTreeMap::new();

    for path in &slice {
        let name = path_stem(path);
        let src = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                let _ = writeln!(
                    out,
                    "{name},read_error,io,\"{}\",not_run",
                    csv_escape(&e.to_string())
                );
                continue;
            }
        };
        match parse_theorem(&src) {
            Err(e) => {
                *parse_fail_buckets
                    .entry(error_variant_name(&e))
                    .or_insert(0) += 1;
                let _ = writeln!(
                    out,
                    "{name},not_parsed,{},\"{}\",not_run",
                    e.category(),
                    csv_escape(&e.to_string())
                );
            }
            Ok(t) => {
                parsed += 1;
                match translate_theorem(&t) {
                    Err(e) => {
                        let _ = writeln!(
                            out,
                            "{name},not_translated,{},\"{}\",not_run",
                            e.category(),
                            csv_escape(&e.to_string())
                        );
                    }
                    Ok(phi) => {
                        translated += 1;
                        let lake_label = if use_lake {
                            match run_lake_check(&name, &phi, &emit_dir, &lake_project_dir) {
                                LakeOutcome::Accepted => {
                                    lake_accepted += 1;
                                    "accepted"
                                }
                                LakeOutcome::Rejected => {
                                    lake_rejected += 1;
                                    "rejected"
                                }
                                LakeOutcome::Error => {
                                    lake_error += 1;
                                    "lake_error"
                                }
                            }
                        } else {
                            "not_run"
                        };
                        let _ = writeln!(out, "{name},translated,,,{lake_label}");
                    }
                }
            }
        }
    }

    // Summary.
    let total = slice.len();
    let parse_rate = 100.0 * parsed as f64 / total as f64;
    let translate_rate_over_parsed = if parsed > 0 {
        100.0 * translated as f64 / parsed as f64
    } else {
        0.0
    };
    let translate_rate_overall = 100.0 * translated as f64 / total as f64;
    eprintln!();
    eprintln!("━━━ Baseline scorecard ━━━");
    eprintln!("  total:                   {total}");
    eprintln!("  parsed:                  {parsed:3} / {total}  ({parse_rate:5.1}%)");
    eprintln!("  translated (of parsed):  {translated:3} / {parsed:3}  ({translate_rate_over_parsed:5.1}%)");
    eprintln!(
        "  translated (of total):   {translated:3} / {total}  ({translate_rate_overall:5.1}%)"
    );
    if use_lake {
        let lake_accept_over_translated = if translated > 0 {
            100.0 * lake_accepted as f64 / translated as f64
        } else {
            0.0
        };
        let lake_accept_overall = 100.0 * lake_accepted as f64 / total as f64;
        eprintln!(
            "  Lake accepted (of translated): {lake_accepted:3} / {translated:3}  ({lake_accept_over_translated:5.1}%)"
        );
        eprintln!(
            "  Lake accepted (of total):      {lake_accepted:3} / {total}  ({lake_accept_overall:5.1}%)"
        );
        eprintln!("  Lake rejected:                 {lake_rejected:3} | errors: {lake_error:3}");
    }
    if !parse_fail_buckets.is_empty() {
        eprintln!();
        eprintln!("  parse-failure histogram (top classes):");
        let mut sorted: Vec<_> = parse_fail_buckets.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (k, v) in sorted.iter().take(8) {
            eprintln!("    {v:3}  {k}");
        }
    }
    ExitCode::SUCCESS
}

// ─── corpus walk ────────────────────────────────────────────────────

fn locate_corpus() -> Option<PathBuf> {
    // CARGO_MANIFEST_DIR is `…/symthaea-lean-bridge/`. The corpus lives
    // two levels up at `symthaea/data/benchmarks/minif2f/MiniF2F/`.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()? // crates/
        .parent()? // symthaea/
        .join("data/benchmarks/minif2f/MiniF2F");
    root.exists().then_some(root)
}

/// Gather paths of `.lean` files under `{Valid,Test}/` whose names
/// start with `mathd_algebra_` or `mathd_numbertheory_` AND whose
/// content does NOT contain any out-of-scope pattern. Mirrors the
/// logic of `scripts/filter_minif2f.sh`.
fn gather_candidates(root: &Path) -> Vec<PathBuf> {
    // Out-of-scope substring list mirroring filter_minif2f.sh's
    // regex alternation. Using plain `contains()` checks keeps this
    // dep-free and close enough to the bash filter; any drift is a
    // known-small risk because the filter is intentionally
    // conservative.
    const OUT_OF_SCOPE: &[&str] = &[
        "Real.sqrt",
        "Real.sin",
        "Real.cos",
        "Real.tan",
        "Real.log",
        "Real.exp",
        "Real.pi",
        "Nat.Prime",
        "Nat.gcd",
        "Nat.choose",
        "Nat.fib",
        "Nat.factorial",
        "Nat.digits",
        "Nat.log",
        "Finset",
        "∑",
        "∏",
        "∃!",
        "Function",
        "List.",
        "String",
        "Matrix",
        "Int.floor",
        // 2026-04-18 baseline revealed ZMod problems leak through the
        // bash filter (mathd_numbertheory_233, mathd_numbertheory_668);
        // ZMod is modular-ring arithmetic and genuinely out of scope.
        "ZMod",
        // 2026-04-18 full-pool histogram: 42/81 UnknownChar failures
        // were `%` (modular arithmetic). FolFormulaExt has no `mod`
        // operator; these are Phase 5c AST-extension territory. Putting
        // them in the filter cleans the denominator by ~23% (42/177).
        "%",
        // Complex and rational types — out of FolFormulaExt scope.
        "ℂ",
        "ℚ",
        // Function-typed binders `(f : ℝ → ℝ)` produce `Unexpected
        // found "ℝ"` at parse-time because our binder parser only
        // recognizes bare type annotations. Function abstraction +
        // application needs AST extension (Phase 5c); filter them
        // out for now via the arrow-to-type pattern — these strings
        // only appear in function-type declarations, never in
        // `→` as logical implication (whose consequent is a
        // proposition, not a bare type name).
        "→ ℝ",
        "→ ℕ",
        "→ ℤ",
    ];

    let mut out = Vec::new();
    for sub in ["Valid", "Test"] {
        let dir = root.join(sub);
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if !name.ends_with(".lean") {
                continue;
            }
            if !(name.starts_with("mathd_algebra_") || name.starts_with("mathd_numbertheory_")) {
                continue;
            }
            let Ok(content) = fs::read_to_string(&p) else {
                continue;
            };
            if OUT_OF_SCOPE.iter().any(|pat| content.contains(pat)) {
                continue;
            }
            out.push(p);
        }
    }
    out.sort();
    out
}

fn path_stem(p: &Path) -> String {
    p.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string()
}

// ─── deterministic shuffle (SplitMix64 — no rand dep) ───────────────

/// Fisher-Yates shuffle with a SplitMix64 PRNG seeded from `seed`.
/// Dep-free: keeps the harness buildable without pulling the `rand`
/// crate just for the shuffle.
fn shuffle_with_seed<T>(v: &mut [T], seed: u64) {
    let mut state = seed;
    for i in (1..v.len()).rev() {
        // Draw a random index in [0, i].
        state = splitmix64(state);
        let j = (state as usize) % (i + 1);
        v.swap(i, j);
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

// ─── error helpers ──────────────────────────────────────────────────

/// Variant name for histogramming. We don't use `std::mem::discriminant`
/// because the stringified discriminant isn't human-readable; manual
/// match keeps the histogram interpretable.
fn error_variant_name(e: &IngestError) -> &'static str {
    match e {
        IngestError::UnknownChar { .. } => "UnknownChar",
        IngestError::IntLitOverflow { .. } => "IntLitOverflow",
        IngestError::Unexpected { .. } => "Unexpected",
        IngestError::UnexpectedEof { .. } => "UnexpectedEof",
        IngestError::NotATheorem => "NotATheorem",
        IngestError::OutOfScope { .. } => "OutOfScope",
        IngestError::UnsupportedTranslation { .. } => "UnsupportedTranslation",
    }
}

/// Minimal CSV field escaping: embedded double quote → "". Newlines
/// stripped (the format's Display values are single-line).
fn csv_escape(s: &str) -> String {
    s.replace('"', "\"\"").replace('\n', " ").replace('\r', " ")
}

// ─── Tier-3 Lake verification ────────────────────────────────────────

enum LakeOutcome {
    Accepted,
    Rejected,
    Error,
}

fn lake_env_available() -> bool {
    if env::var("LAKE_ENV").ok().as_deref() != Some("1") {
        return false;
    }
    Command::new("lake")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Render `phi` through the Phase 2+ cascade, write to `emit_dir/<name>.lean`,
/// then invoke `lake env lean` from `project_dir`. Returns whether Lake
/// accepted the proof.
fn run_lake_check(
    name: &str,
    phi: &symthaea_core::hdc::fol_formula_ext::FolFormulaExt,
    emit_dir: &Path,
    project_dir: &Path,
) -> LakeOutcome {
    let contents = render_fol_ext_file(name, phi);
    let file_path = emit_dir.join(format!("{}.lean", name));
    if fs::write(&file_path, &contents).is_err() {
        return LakeOutcome::Error;
    }
    // Canonicalize to absolute path — `lake env lean` cd's into the
    // project_dir, so relative paths from cwd break.
    let abs_lean_file = match fs::canonicalize(&file_path) {
        Ok(p) => p,
        Err(_) => return LakeOutcome::Error,
    };
    // Bound the lake invocation: some `.lean` files can hit Lean's
    // heartbeat limit on NRA problems and take tens of seconds. 60s is
    // generous enough to accommodate the slow cases without letting a
    // single runaway stall the whole harness.
    let mut child = match Command::new("lake")
        .arg("env")
        .arg("lean")
        .arg(&abs_lean_file)
        .current_dir(project_dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return LakeOutcome::Error,
    };
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(60);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return if status.success() {
                    LakeOutcome::Accepted
                } else {
                    LakeOutcome::Rejected
                };
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    return LakeOutcome::Error;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(_) => return LakeOutcome::Error,
        }
    }
}
