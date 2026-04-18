// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! `symtropy` — project scaffolding.
//!
//! ```text
//! symtropy new <project-name> [--template <template-name>]
//! symtropy templates    # list available templates
//! ```

mod templates;

use std::path::PathBuf;
use std::process::ExitCode;

use templates::TEMPLATES;

const USAGE: &str = "\
symtropy — Symtropy project scaffolding

USAGE:
    symtropy new <project-name> [--template <name>]
    symtropy templates
    symtropy --help

OPTIONS:
    --template <name>   Template to use (default: 3d-research)
    -h, --help          Show this help

TEMPLATES:
    3d-research         3D scene with one swinging pendulum + dev console
    4d-research         4D scene with hyperplane slicing + dev console
    2d-game             2D scene with a single physics-bodied sprite

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
}
