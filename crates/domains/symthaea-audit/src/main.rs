// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Thin orchestration binary — see `symthaea_audit::lib` for the actual logic.

use clap::Parser;
use symthaea_audit::{
    agent_loop, cli::Cli, heuristics, llm_client, prefetch, report_prompt, tools::Sandbox, verify,
};

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

    let mut initial_context = String::new();
    if !cli.no_hints {
        let hints = heuristics::find_none_arg_hints(&target);
        if !hints.is_empty() {
            eprintln!("[symthaea-audit] pre-scan found {} hint(s):", hints.len());
            for hint in &hints {
                eprintln!("  {}", hint.replace('\n', "\n  "));
            }
            initial_context.push_str(
                "## Automated pre-scan hints (unverified — confirm before relying on them)\n\n",
            );
            for hint in &hints {
                initial_context.push_str(hint);
                initial_context.push_str("\n\n");
            }
        }
    }
    let single_shot_paths = cli.single_shot_paths_list();
    if !single_shot_paths.is_empty() {
        eprintln!("[symthaea-audit] single-shot prefetch: {single_shot_paths:?}");
        initial_context.push_str("## Prefetched file contents\n\n");
        initial_context.push_str(&prefetch::prefetch_default(&sandbox, &single_shot_paths));
    }

    let system_prompt = report_prompt::build_system_prompt(cli.focus.as_deref(), run_check_enabled);
    let outcome = agent_loop::run(
        client.as_ref(),
        &sandbox,
        &system_prompt,
        cli.max_turns,
        Some(&initial_context),
    )?;

    eprintln!(
        "[symthaea-audit] done in {} turn(s){}",
        outcome.turns_used,
        if outcome.truncated_by_turn_budget {
            " (truncated)"
        } else {
            ""
        }
    );

    let mut final_report = outcome.report;
    if cli.verify {
        eprintln!("[symthaea-audit] running verification pass");
        match verify::run_verification_pass(client.as_ref(), &sandbox, &final_report, cli.max_turns)
        {
            Ok(verify_outcome) => {
                eprintln!(
                    "[symthaea-audit] verification done in {} turn(s){}",
                    verify_outcome.turns_used,
                    if verify_outcome.truncated_by_turn_budget {
                        " (truncated)"
                    } else {
                        ""
                    }
                );
                final_report.push_str("\n\n---\n\n");
                final_report.push_str(&verify_outcome.report);
            }
            Err(e) => {
                eprintln!(
                    "[symthaea-audit] verification pass failed, keeping original report: {e}"
                );
                final_report.push_str(&format!("\n\n---\n\n(verification pass failed: {e})\n"));
            }
        }
    }

    let out_path = cli
        .out
        .clone()
        .unwrap_or_else(|| default_report_path(&target));
    std::fs::write(&out_path, &final_report)?;
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
