// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! broca-exercism-bench: fixture-driven Rust coding capability evaluation.
//!
//! Each exercise creates an isolated Rust crate, writes Broca's synthesized
//! solution, and runs `cargo test`. Results are reported as measurements, not
//! as capability claims.

use anyhow::{Context, Result};
use serde::Serialize;
use std::path::PathBuf;
use std::process::Command;
use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
use symthaea_core::genesis::GenesisSeed;

#[derive(Debug, Clone)]
struct Options {
    json_out: Option<PathBuf>,
    max_attempts: usize,
    max_exercises: usize,
    keep_workdir: bool,
}

#[derive(Debug, Clone)]
struct ExerciseFixture {
    slug: &'static str,
    prompt: &'static str,
    tests: &'static str,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    schema_version: u32,
    evidence_level: &'static str,
    measured: bool,
    benchmark: &'static str,
    total_exercises: usize,
    compile_successes: usize,
    test_successes: usize,
    cases: Vec<ExerciseReport>,
}

#[derive(Debug, Serialize)]
struct ExerciseReport {
    slug: String,
    evidence_level: &'static str,
    measured: bool,
    attempts: usize,
    compiled: bool,
    tests_passed: bool,
    final_coherence: f32,
    workdir: Option<String>,
    diagnostics: Vec<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let opts = parse_args()?;

    let genesis = GenesisSeed::from_phrase("exercism-bench-v1");
    let mut config = LiquidMambaConfig::default();
    config.enable_gating = true;
    let mut generator = LiquidMambaGenerator::new(&genesis, config)?;

    let fixtures = fixtures();
    let selected_fixtures = limited_fixtures(&fixtures, opts.max_exercises);
    let mut cases = Vec::with_capacity(fixtures.len());
    for fixture in selected_fixtures {
        cases.push(run_fixture(&mut generator, fixture, &opts)?);
    }

    let report = BenchReport {
        schema_version: 1,
        evidence_level: "measured",
        measured: true,
        benchmark: "broca-exercism-rust-fixtures",
        total_exercises: cases.len(),
        compile_successes: cases.iter().filter(|case| case.compiled).count(),
        test_successes: cases.iter().filter(|case| case.tests_passed).count(),
        cases,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = opts.json_out {
        std::fs::write(path, json)?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn run_fixture(
    generator: &mut LiquidMambaGenerator,
    fixture: &ExerciseFixture,
    opts: &Options,
) -> Result<ExerciseReport> {
    println!("[Exercise] {}", fixture.slug);
    let temp = tempfile::Builder::new()
        .prefix(&format!("broca-exercism-{}-", fixture.slug))
        .tempdir()
        .context("creating isolated exercise crate")?;
    write_crate(temp.path(), fixture, "")?;

    let mut diagnostics = Vec::new();
    let mut compiled = false;
    let mut tests_passed = false;
    let mut attempts = 0usize;

    let signature = if fixture.prompt.starts_with("// Implement:") {
        Some(fixture.prompt.trim_start_matches("// Implement:").trim())
    } else {
        None
    };

    let channels = ThoughtChannels::with_intent(stable_intent_id(fixture.slug));
    for attempt in 1..=opts.max_attempts {
        attempts = attempt;
        let monologue = generator
            .generate_semantic_monologue(&channels, 3)
            .context("generating semantic monologue")?;
        let nucleus = generator.recursive_fold(&monologue);
        let synthesized_code = generator
            .synthesize_program_with_signature(&nucleus, fixture.slug, signature)
            .context("synthesizing exercise program")?;
        write_crate(temp.path(), fixture, &synthesized_code)?;

        let check = cargo(temp.path(), "check")?;
        compiled = check.success;
        diagnostics.push(format!(
            "attempt {attempt} cargo check: {}",
            if check.success { "pass" } else { "fail" }
        ));
        if !check.stderr.trim().is_empty() {
            diagnostics.push(trim_diagnostic(&check.stderr));
        }

        if !compiled {
            continue;
        }

        let test = cargo(temp.path(), "test")?;
        tests_passed = test.success;
        diagnostics.push(format!(
            "attempt {attempt} cargo test: {}",
            if test.success { "pass" } else { "fail" }
        ));
        if !test.stderr.trim().is_empty() {
            diagnostics.push(trim_diagnostic(&test.stderr));
        }
        if tests_passed {
            break;
        }
    }

    let final_coherence = f32::from_bits(
        generator
            .topological_coherence
            .load(std::sync::atomic::Ordering::Relaxed),
    );
    let workdir = if opts.keep_workdir {
        Some(temp.keep().display().to_string())
    } else {
        None
    };

    Ok(ExerciseReport {
        slug: fixture.slug.to_string(),
        evidence_level: "measured",
        measured: true,
        attempts,
        compiled,
        tests_passed,
        final_coherence,
        workdir,
        diagnostics,
    })
}

struct CargoResult {
    success: bool,
    stderr: String,
}

fn cargo(dir: &std::path::Path, subcommand: &str) -> Result<CargoResult> {
    let output = Command::new("cargo")
        .arg(subcommand)
        .arg("--quiet")
        .current_dir(dir)
        .output()
        .with_context(|| format!("running cargo {subcommand}"))?;
    Ok(CargoResult {
        success: output.status.success(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

fn write_crate(dir: &std::path::Path, fixture: &ExerciseFixture, solution: &str) -> Result<()> {
    std::fs::create_dir_all(dir.join("src"))?;
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            r#"[package]
name = "broca-exercism-{}"
version = "0.1.0"
edition = "2024"

[dependencies]
"#,
            fixture.slug.replace('-', "_")
        ),
    )?;
    std::fs::write(
        dir.join("src/lib.rs"),
        format!(
            "{}\n\n{}\n\n#[cfg(test)]\nmod tests {{\n    use super::*;\n{}\n}}\n",
            fixture.prompt, solution, fixture.tests
        ),
    )?;
    Ok(())
}

fn fixtures() -> Vec<ExerciseFixture> {
    vec![
        ExerciseFixture {
            slug: "hello-world",
            prompt: "// Implement: pub fn hello() -> &'static str",
            tests: r#"
    #[test]
    fn says_hello_world() {
        assert_eq!(hello(), "Hello, World!");
    }
"#,
        },
        ExerciseFixture {
            slug: "leap",
            prompt: "// Implement: pub fn is_leap_year(year: u64) -> bool",
            tests: r#"
    #[test]
    fn leap_rules() {
        assert!(is_leap_year(2000));
        assert!(!is_leap_year(1900));
        assert!(is_leap_year(1996));
        assert!(!is_leap_year(1997));
    }
"#,
        },
        ExerciseFixture {
            slug: "reverse-string",
            prompt: "// Implement: pub fn reverse(input: &str) -> String",
            tests: r#"
    #[test]
    fn reverses_unicode() {
        assert_eq!(reverse("robot"), "tobor");
        assert_eq!(reverse("hello, 世界"), "界世 ,olleh");
    }
"#,
        },
        ExerciseFixture {
            slug: "raindrops",
            prompt: "// Implement: pub fn raindrops(n: u32) -> String",
            tests: r#"
    #[test]
    fn raindrop_rules() {
        assert_eq!(raindrops(28), "Plong");
        assert_eq!(raindrops(30), "PlingPlang");
        assert_eq!(raindrops(34), "34");
    }
"#,
        },
    ]
}

fn limited_fixtures(fixtures: &[ExerciseFixture], max_exercises: usize) -> &[ExerciseFixture] {
    if max_exercises > 0 && max_exercises < fixtures.len() {
        &fixtures[..max_exercises]
    } else {
        fixtures
    }
}

fn stable_intent_id(slug: &str) -> usize {
    slug.bytes()
        .fold(500usize, |acc, b| acc.wrapping_add(b as usize))
        % 1000
}

fn trim_diagnostic(stderr: &str) -> String {
    const LIMIT: usize = 1600;
    let trimmed = stderr.trim();
    if trimmed.len() <= LIMIT {
        trimmed.to_string()
    } else {
        format!("{}...[truncated]", &trimmed[..LIMIT])
    }
}

fn parse_args() -> Result<Options> {
    let mut opts = Options {
        json_out: None,
        max_attempts: 3,
        max_exercises: 0,
        keep_workdir: false,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--json-out" => {
                i += 1;
                opts.json_out = Some(PathBuf::from(value(&args, i, "--json-out")?));
            }
            "--max-attempts" => {
                i += 1;
                opts.max_attempts = value(&args, i, "--max-attempts")?.parse()?;
            }
            "--max-exercises" => {
                i += 1;
                opts.max_exercises = value(&args, i, "--max-exercises")?.parse()?;
            }
            "--keep-workdir" => opts.keep_workdir = true,
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
        i += 1;
    }
    if opts.max_attempts == 0 {
        anyhow::bail!("--max-attempts must be greater than zero");
    }
    Ok(opts)
}

fn value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str> {
    args.get(index)
        .map(String::as_str)
        .with_context(|| format!("{flag} requires a value"))
}

fn print_usage() {
    eprintln!(
        "Usage: broca-exercism-bench [--json-out PATH] [--max-attempts N] [--max-exercises N] [--keep-workdir]"
    );
}
