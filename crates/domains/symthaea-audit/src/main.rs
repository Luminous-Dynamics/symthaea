// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Thin orchestration binary — see `symthaea_audit::lib` for the actual logic.

use clap::Parser;
use symthaea_audit::{agent_loop, cli::Cli, llm_client, report_prompt, tools::Sandbox};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let target = cli
        .target
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("target {:?} is not a valid directory: {e}", cli.target))?;
    if !target.is_dir() {
        anyhow::bail!("target {:?} is not a directory", target);
    }

    let allow_exec = cli.allow_exec_list();
    let run_check_enabled = !allow_exec.is_empty();
    let sandbox = Sandbox::new(&target, allow_exec)?;

    let client = llm_client::detect_backend(&cli)?;
    eprintln!("[symthaea-audit] target: {}", target.display());
    eprintln!("[symthaea-audit] llm backend: {}", client.name());
    if run_check_enabled {
        eprintln!(
            "[symthaea-audit] run_check enabled for: {:?}",
            sandbox.allow_exec()
        );
    }

    let system_prompt = report_prompt::build_system_prompt(cli.focus.as_deref(), run_check_enabled);
    let outcome = agent_loop::run(client.as_ref(), &sandbox, &system_prompt, cli.max_turns)?;

    eprintln!(
        "[symthaea-audit] done in {} turn(s){}",
        outcome.turns_used,
        if outcome.truncated_by_turn_budget {
            " (truncated)"
        } else {
            ""
        }
    );

    let out_path = cli
        .out
        .clone()
        .unwrap_or_else(|| default_report_path(&target));
    std::fs::write(&out_path, &outcome.report)?;
    eprintln!("[symthaea-audit] report written to {}", out_path.display());

    Ok(())
}

fn default_report_path(target: &std::path::Path) -> std::path::PathBuf {
    let basename = target
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("target");
    let timestamp = chrono::Utc::now().format("%Y%m%d%H%M%S");
    std::path::PathBuf::from(format!("symthaea-audit-report-{basename}-{timestamp}.md"))
}
