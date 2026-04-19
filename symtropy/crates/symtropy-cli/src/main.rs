// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! `symtropy` — project scaffolding + robotics-demo launcher.
//!
//! ```text
//! symtropy new <project-name> [--template <template-name>]
//! symtropy templates        # list available templates
//! symtropy demos            # list available robotics demos
//! symtropy run <demo-name>  # launch a robotics demo
//! ```

mod templates;

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use templates::TEMPLATES;

/// Known robotics demos. Each is a standalone crate under
/// `symtropy/crates/symtropy-<name>-demo/` with its own Cargo workspace
/// (they bring Bevy 0.18 in separately and are excluded from the
/// parent symtropy workspace). The `run` subcommand shells out to
/// `cargo run --release` inside each demo's crate directory.
struct DemoEntry {
    name: &'static str,
    bin: &'static str,
    narrative: &'static str,
}

const DEMOS: &[DemoEntry] = &[
    DemoEntry {
        name: "flight",
        bin: "flight-demo",
        narrative: "Quadrotor figure-8 under wind gusts — Φ scales motor authority",
    },
    DemoEntry {
        name: "vehicle",
        bin: "vehicle-demo",
        narrative: "Autonomous car over ice patches — Stanley + sprint_floor_gain",
    },
    DemoEntry {
        name: "auv",
        bin: "auv-demo",
        narrative: "Underwater waypoint nav under rotating current",
    },
    DemoEntry {
        name: "helicopter",
        bin: "helicopter-demo",
        narrative: "SAR hover under Dryden wind gusts",
    },
    DemoEntry {
        name: "exoskeleton",
        bin: "exoskeleton-demo",
        narrative: "6-joint powered suit — Φ selects AssistanceMode",
    },
    DemoEntry {
        name: "orbital",
        bin: "orbital-demo",
        narrative: "7-joint spacecraft arm — Φ tracks mission-phase constraints",
    },
    DemoEntry {
        name: "surgical",
        bin: "surgical-demo",
        narrative: "6-DOF surgical manipulator — dual-channel cautery interlock",
    },
    DemoEntry {
        name: "humanoid",
        bin: "humanoid-demo",
        narrative: "21-DOF bipedal — DMC Humanoid benchmark",
    },
    DemoEntry {
        name: "quadruped",
        bin: "quadruped-demo",
        narrative: "12-DOF legged — Φ selects GaitType",
    },
    DemoEntry {
        name: "manipulator",
        bin: "manipulator-demo",
        narrative: "7-DOF Franka Panda — Φ-gated admittance vs ISO SSM (split screen)",
    },
    DemoEntry {
        name: "gravcraft",
        bin: "gravcraft-demo",
        narrative: "3D spacetime grid warping under metric-perturbation drive",
    },
];

const USAGE: &str = "\
symtropy — Symtropy project scaffolding + robotics-demo launcher

USAGE:
    symtropy new <project-name> [--template <name>]
    symtropy templates
    symtropy demos
    symtropy run <demo-name> [-- <extra args passed to demo>]
    symtropy calibrate <platform> [--steps N]
    symtropy --help

OPTIONS:
    --template <name>   Template to use (default: 3d-research)
    -h, --help          Show this help

TEMPLATES:
    3d-research         3D scene with one swinging pendulum + dev console
    4d-research         4D scene with hyperplane slicing + dev console
    2d-game             2D scene with a single physics-bodied sprite

DEMOS (run `symtropy demos` for the annotated list):
    flight, vehicle, auv, helicopter, exoskeleton, orbital, surgical,
    humanoid, quadruped, manipulator, gravcraft

After running `symtropy new`:
    cd <project-name>
    cargo run --release
";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let cmd = match args.next() {
        Some(c) => c,
        None => {
            eprintln!("{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    match cmd.as_str() {
        "-h" | "--help" | "help" => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        "templates" => {
            for t in TEMPLATES {
                println!("  {}  — {}", t.name, t.description);
            }
            ExitCode::SUCCESS
        }
        "demos" => {
            println!("Robotics demos ({}):", DEMOS.len());
            for d in DEMOS {
                println!("  {:<12}  {}", d.name, d.narrative);
            }
            println!();
            println!("Launch with:  symtropy run <name>");
            ExitCode::SUCCESS
        }
        "run" => match cmd_run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        "calibrate" => match cmd_calibrate(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        "new" => match cmd_new(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        other => {
            eprintln!("error: unknown subcommand `{other}`\n\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_calibrate(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    const PLATFORMS: &[&str] = &[
        "quadrotor",
        "vehicle",
        "humanoid",
        "manipulator",
        "auv",
        "helicopter",
    ];

    let platform = args
        .next()
        .ok_or_else(|| format!("missing <platform>. Available: {}", PLATFORMS.join(", ")))?;
    if !PLATFORMS.contains(&platform.as_str()) {
        return Err(format!(
            "unknown platform `{platform}`. Available: {}",
            PLATFORMS.join(", ")
        ));
    }

    let mut steps = 1000u32;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--steps" => {
                steps = args
                    .next()
                    .ok_or_else(|| "--steps requires a value".to_string())?
                    .parse()
                    .map_err(|e| format!("invalid --steps value: {e}"))?;
            }
            other => {
                return Err(format!("unexpected argument `{other}`"));
            }
        }
    }

    // Locate the repo root so we can invoke phi_trace via cargo.
    let repo = locate_repo_root()?;

    eprintln!("→ running phi_trace on {platform} with PT_PLATFORM_OBS=1, {steps} steps …");
    let output = Command::new("cargo")
        .current_dir(&repo)
        .env("PT_PLATFORM", &platform)
        .env("PT_PLATFORM_OBS", "1")
        .env("PT_STEPS", steps.to_string())
        .arg("run")
        .arg("--release")
        .arg("--quiet")
        .arg("-p")
        .arg("symtropy-robotics-bridge")
        .arg("--example")
        .arg("phi_trace")
        .output()
        .map_err(|e| format!("failed to spawn cargo: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "phi_trace failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse the distribution lines emitted by phi_trace.
    let p50 = parse_stat(&stdout, "p50")?;
    let p95 = parse_stat(&stdout, "p95")?;
    let min = parse_stat(&stdout, "min")?;
    let max = parse_stat(&stdout, "max")?;

    let suggested = (p50 * 1000.0).round() / 1000.0;
    let suggested_p95 = (p95 * 1000.0).round() / 1000.0;

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(" SPRINT_THRESHOLD calibration for `{platform}`");
    println!("════════════════════════════════════════════════════════════════════");
    println!(" observed Φ distribution ({steps} ticks, platform-aware obs):");
    println!("   min    = {min:.4}");
    println!("   max    = {max:.4}");
    println!("   p50    = {p50:.4}");
    println!("   p95    = {p95:.4}");
    println!();
    println!(" Suggested thresholds:");
    println!("   SPRINT_THRESHOLD = {suggested:.3}   (at p50 — ~50 % sprint frames)");
    println!(
        "   SPRINT_THRESHOLD = {suggested_p95:.3}   (at p95 — ~5 %  sprint frames, rare high-confidence)"
    );
    println!();
    println!(" To apply: edit `symtropy-{platform}-demo/src/plugin.rs` and set:");
    println!("   const SPRINT_THRESHOLD: f64 = {suggested:.3};");
    println!();
    Ok(())
}

fn parse_stat(stdout: &str, key: &str) -> Result<f64, String> {
    for line in stdout.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed
            .strip_prefix(key)
            .and_then(|s| s.trim_start().strip_prefix('='))
        {
            return rest
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("parse {key} failed: {e}"))
                .map(|v| v);
        }
    }
    Err(format!("could not find `{key}=` in phi_trace output"))
}

/// Locate the symtropy workspace dir (contains `symtropy/Cargo.toml` with
/// a `[workspace]` member list). Cargo commands need to run from inside
/// a workspace root, not from the monorepo root. Walks up from CWD; the
/// env var `SYMTROPY_MONOREPO_ROOT` points at the monorepo root and we
/// append `symtropy/` for the workspace.
fn locate_repo_root() -> Result<PathBuf, String> {
    if let Ok(root) = std::env::var("SYMTROPY_MONOREPO_ROOT") {
        let ws = PathBuf::from(root).join("symtropy");
        if ws.join("Cargo.toml").exists() {
            return Ok(ws);
        }
    }
    let cwd = std::env::current_dir().map_err(|e| format!("cannot read CWD: {e}"))?;
    for ancestor in cwd.ancestors() {
        // Look for the symtropy workspace Cargo.toml directly.
        if ancestor.ends_with("symtropy")
            && ancestor.join("Cargo.toml").exists()
            && ancestor.join("crates/symtropy-robotics-bridge").exists()
        {
            return Ok(ancestor.to_path_buf());
        }
        // Or a monorepo parent with symtropy/ inside.
        let ws = ancestor.join("symtropy");
        if ws.join("Cargo.toml").exists() && ws.join("crates/symtropy-robotics-bridge").exists() {
            return Ok(ws);
        }
    }
    Err(
        "could not locate symtropy workspace. Set SYMTROPY_MONOREPO_ROOT=/path/to/luminous-dynamics"
            .to_string(),
    )
}

fn cmd_run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let demo_name = args.next().ok_or_else(|| {
        let names: Vec<&str> = DEMOS.iter().map(|d| d.name).collect();
        format!("missing <demo-name>. Available: {}", names.join(", "))
    })?;

    let demo = DEMOS.iter().find(|d| d.name == demo_name).ok_or_else(|| {
        let names: Vec<&str> = DEMOS.iter().map(|d| d.name).collect();
        format!(
            "unknown demo `{demo_name}`. Available: {}",
            names.join(", ")
        )
    })?;

    // Extra args: anything after `--` is forwarded to the demo binary.
    // Anything before `--` is treated as an error (no flags defined yet).
    let mut extra: Vec<String> = Vec::new();
    let mut seen_dashdash = false;
    for a in args {
        if !seen_dashdash {
            if a == "--" {
                seen_dashdash = true;
                continue;
            }
            return Err(format!(
                "unexpected argument `{a}` (pass extras after `--`)"
            ));
        }
        extra.push(a);
    }

    let crate_dir = locate_demo_crate(demo.name)?;
    eprintln!(
        "→ cd {} && cargo run --release --bin {}",
        crate_dir.display(),
        demo.bin
    );

    let mut cmd = Command::new("cargo");
    cmd.current_dir(&crate_dir)
        .arg("run")
        .arg("--release")
        .arg("--bin")
        .arg(demo.bin);
    if !extra.is_empty() {
        cmd.arg("--");
        for a in &extra {
            cmd.arg(a);
        }
    }
    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn cargo: {e}"))?;
    if !status.success() {
        return Err(format!("demo `{demo_name}` exited with {status}"));
    }
    Ok(())
}

/// Locate the demo's standalone crate directory.
///
/// Walks up from the CLI binary's CWD looking for
/// `symtropy/crates/symtropy-<name>-demo/Cargo.toml`. Also accepts the
/// env override `SYMTROPY_MONOREPO_ROOT=/path/to/luminous-dynamics`
/// (useful when the CLI is installed outside the monorepo).
fn locate_demo_crate(name: &str) -> Result<PathBuf, String> {
    let rel = format!("symtropy/crates/symtropy-{name}-demo/Cargo.toml");

    if let Ok(root) = std::env::var("SYMTROPY_MONOREPO_ROOT") {
        let p = PathBuf::from(&root).join(&rel);
        if p.exists() {
            return Ok(p.parent().unwrap().to_path_buf());
        }
    }

    let cwd = std::env::current_dir().map_err(|e| format!("cannot read CWD: {e}"))?;
    for ancestor in cwd.ancestors() {
        let p = ancestor.join(&rel);
        if p.exists() {
            return Ok(p.parent().unwrap().to_path_buf());
        }
    }

    Err(format!(
        "could not find `{rel}` walking up from CWD. Set SYMTROPY_MONOREPO_ROOT \
         to point at the luminous-dynamics checkout."
    ))
}

// Unused but kept for future direct-path resolution.
#[allow(dead_code)]
fn demo_exists(crate_dir: &Path) -> bool {
    crate_dir.join("Cargo.toml").exists()
}

fn cmd_new(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut project_name: Option<String> = None;
    let mut template_name = "3d-research".to_string();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--template" => {
                template_name = args
                    .next()
                    .ok_or_else(|| "--template requires a value".to_string())?;
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown flag `{other}`"));
            }
            other => {
                if project_name.is_some() {
                    return Err(format!("unexpected positional argument `{other}`"));
                }
                project_name = Some(other.to_string());
            }
        }
    }

    let name = project_name.ok_or_else(|| "missing <project-name>".to_string())?;
    if !is_valid_crate_name(&name) {
        return Err(format!(
            "`{name}` is not a valid crate name (use lowercase letters, digits, dashes, underscores)"
        ));
    }

    let template = TEMPLATES
        .iter()
        .find(|t| t.name == template_name)
        .ok_or_else(|| {
            let names: Vec<&str> = TEMPLATES.iter().map(|t| t.name).collect();
            format!(
                "template `{template_name}` not found. Available: {}",
                names.join(", ")
            )
        })?;

    let target = PathBuf::from(&name);
    if target.exists() {
        return Err(format!("`{}` already exists", target.display()));
    }
    std::fs::create_dir(&target).map_err(|e| format!("create {}: {e}", target.display()))?;
    let src_dir = target.join("src");
    std::fs::create_dir(&src_dir).map_err(|e| format!("create {}: {e}", src_dir.display()))?;

    write_substituted(target.join("Cargo.toml"), template.cargo_toml, &name)?;
    write_substituted(src_dir.join("main.rs"), template.main_rs, &name)?;
    write_substituted(target.join("README.md"), template.readme_md, &name)?;
    write_substituted(target.join(".gitignore"), GITIGNORE, &name)?;

    println!("Created `{name}` from template `{}`.\n", template.name);
    println!("Next steps:");
    println!("    cd {name}");
    println!("    cargo run --release");
    println!();
    println!("Press F1 inside the demo to toggle the dev console + Φ Inspector.");

    Ok(())
}

fn write_substituted(path: PathBuf, content: &str, project_name: &str) -> Result<(), String> {
    let substituted = content.replace("{{project_name}}", project_name);
    std::fs::write(&path, substituted).map_err(|e| format!("write {}: {e}", path.display()))
}

fn is_valid_crate_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
        && !name.starts_with(|c: char| c.is_ascii_digit())
}

const GITIGNORE: &str = "/target\n/Cargo.lock\n";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_crate_names() {
        assert!(is_valid_crate_name("my-game"));
        assert!(is_valid_crate_name("my_game"));
        assert!(is_valid_crate_name("game123"));
        assert!(is_valid_crate_name("a"));
    }

    #[test]
    fn invalid_crate_names() {
        assert!(!is_valid_crate_name(""));
        assert!(!is_valid_crate_name("123game"));
        assert!(!is_valid_crate_name("My-Game"));
        assert!(!is_valid_crate_name("my game"));
        assert!(!is_valid_crate_name("my.game"));
    }

    #[test]
    fn templates_have_unique_names() {
        let mut names: Vec<&str> = TEMPLATES.iter().map(|t| t.name).collect();
        names.sort();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len, "duplicate template names");
    }

    #[test]
    fn templates_substitute_project_name() {
        for t in TEMPLATES {
            // Each template's Cargo.toml MUST contain the placeholder, otherwise
            // generated projects all have name = "{{project_name}}".
            assert!(
                t.cargo_toml.contains("{{project_name}}"),
                "template `{}` Cargo.toml missing {{{{project_name}}}}",
                t.name
            );
        }
    }

    #[test]
    fn demos_have_unique_names() {
        let mut names: Vec<&str> = DEMOS.iter().map(|d| d.name).collect();
        names.sort();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len, "duplicate demo names");
    }

    #[test]
    fn demos_have_unique_bins() {
        let mut bins: Vec<&str> = DEMOS.iter().map(|d| d.bin).collect();
        bins.sort();
        let len = bins.len();
        bins.dedup();
        assert_eq!(bins.len(), len, "duplicate demo bin names");
    }

    #[test]
    fn demo_bin_names_match_convention() {
        // Each demo's bin name should be `<short>-demo`. This keeps the
        // CLI's `cargo run --bin <bin>` guess correct.
        for d in DEMOS {
            let expected = format!("{}-demo", d.name);
            assert_eq!(d.bin, expected, "demo `{}` bin mismatch", d.name);
        }
    }
}
