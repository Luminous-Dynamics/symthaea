// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Thin orchestration binary — see `symthaea_audit::lib` for the actual logic.

use clap::Parser;
use symthaea_audit::{
    agent_loop, cli::Cli, diff_review, heuristics, llm_client, prefetch, report_prompt,
    tools::Sandbox, verify,
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

    if let Some(range) = cli.review_diff.as_deref() {
        eprintln!("[symthaea-audit] reviewing diff: {range}");
        let outcome = diff_review::run_diff_review(
            client.as_ref(),
            &sandbox,
            range,
            cli.focus.as_deref(),
            cli.max_turns,
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
        let final_report = maybe_verify(&cli, client.as_ref(), &sandbox, outcome.report);
        let out_path = cli
            .out
            .clone()
            .unwrap_or_else(|| default_report_path(&target));
        std::fs::write(&out_path, &final_report)?;
        eprintln!("[symthaea-audit] report written to {}", out_path.display());
        return Ok(());
    }

    let mut initial_context = String::new();
    if !cli.no_hints {
        let scan = heuristics::scan_repo(&target);
        let none_hints = heuristics::none_arg_mismatch_hints(&scan);
        let dead_fn_hints = heuristics::dead_pub_fn_hints(&scan);
        let total = none_hints.len() + dead_fn_hints.len();
        if total > 0 {
            eprintln!("[symthaea-audit] pre-scan found {total} hint(s):");
            for hint in none_hints.iter().chain(dead_fn_hints.iter()) {
                eprintln!("  {}", hint.replace('\n', "\n  "));
            }
            initial_context.push_str(
                "## Automated pre-scan hints (unverified — confirm before relying on them)\n\n",
            );
            if !none_hints.is_empty() {
                initial_context.push_str(
                    "### Functions called with None in some places, non-None in others\n\n",
                );
                for hint in &none_hints {
                    initial_context.push_str(hint);
                    initial_context.push_str("\n\n");
                }
            }
            if !dead_fn_hints.is_empty() {
                initial_context.push_str("### Public functions with no non-test usage found\n\n");
                for hint in &dead_fn_hints {
                    initial_context.push_str(hint);
                    initial_context.push_str("\n\n");
                }
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

    let final_report = maybe_verify(&cli, client.as_ref(), &sandbox, outcome.report);
    let out_path = cli
        .out
        .clone()
        .unwrap_or_else(|| default_report_path(&target));
    std::fs::write(&out_path, &final_report)?;
    eprintln!("[symthaea-audit] report written to {}", out_path.display());

    Ok(())
}

/// Runs the verification pass if `--verify` was set, appending (never replacing) its
/// output to `report`. Shared by both whole-repo audit mode and diff-review mode.
fn maybe_verify(
    cli: &Cli,
    client: &dyn llm_client::LlmClient,
    sandbox: &Sandbox,
    mut report: String,
) -> String {
    if !cli.verify {
        return report;
    }
    eprintln!("[symthaea-audit] running verification pass");
    match verify::run_verification_pass(client, sandbox, &report, cli.max_turns) {
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
            report.push_str("\n\n---\n\n");
            report.push_str(&verify_outcome.report);
        }
        Err(e) => {
            eprintln!("[symthaea-audit] verification pass failed, keeping original report: {e}");
            report.push_str(&format!("\n\n---\n\n(verification pass failed: {e})\n"));
        }
    }
    report
}

fn default_report_path(target: &std::path::Path) -> std::path::PathBuf {
    let basename = target
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("target");
    let timestamp = chrono::Utc::now().format("%Y%m%d%H%M%S");
    std::path::PathBuf::from(format!("symthaea-audit-report-{basename}-{timestamp}.md"))
}
