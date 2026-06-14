// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::metrics::ExperimentScorecard;
use crate::runner::BenchmarkRun;

pub fn claim_scope_markdown() -> &'static str {
    r#"## Claim Scope

Honest claims:

- This is an exploratory computational benchmark lab.
- The scorecards compare toy structural hypotheses against explicit null models.
- Passing a scorecard means the benchmark threshold was met for that seeded run.

Non-claims:

- This does not prove fractal time.
- This does not prove quantum consciousness.
- This does not simulate a physical quantum many-body time crystal.
- This does not compute full IIT Phi.
- This is not production scientific validation.
"#
}

pub fn scorecards_to_markdown_report(run: &BenchmarkRun) -> String {
    let mut out = String::new();

    out.push_str("# Symthaea Fractal Time Lab Benchmark Report\n\n");
    out.push_str(&format!("Version: `{}`\n\n", run.version));
    out.push_str(&format!("Epistemic status: `{}`\n\n", run.epistemic_status));
    out.push_str(&format!(
        "Configuration: `seed={}`, `trials={}`\n\n",
        run.config.seed,
        run.config.trials.max(1)
    ));

    out.push_str("## Summary\n\n");
    out.push_str(&format!(
        "- Scorecards: {}\n- All benchmark thresholds passed: `{}`\n\n",
        run.scorecards.len(),
        run.all_passed()
    ));

    out.push_str("## Scorecards\n\n");
    out.push_str(
        "| Experiment | Score | Null mean | Null std | Effect size | Threshold | Passed |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|---:|---:|\n");

    for card in &run.scorecards {
        out.push_str(&format_scorecard_row(card));
    }

    out.push('\n');

    for card in &run.scorecards {
        out.push_str(&format!("### {}\n\n", card.experiment));
        out.push_str(&format!("- Hypothesis: {}\n", card.hypothesis));
        out.push_str(&format!("- Caveat: {}\n", card.caveat));
        out.push_str(&format!("- Passed: `{}`\n\n", card.passed));
    }

    out.push_str(claim_scope_markdown());
    out
}

fn format_scorecard_row(card: &ExperimentScorecard) -> String {
    format!(
        "| {} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {} |\n",
        escape_md_table(&card.experiment),
        card.primary_score,
        card.null_mean,
        card.null_std,
        card.effect_size,
        card.threshold,
        card.passed
    )
}

fn escape_md_table(value: &str) -> String {
    value.replace('|', "\\|")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::{BenchmarkConfig, run_benchmark_run};

    #[test]
    fn test_markdown_report_contains_non_claims() {
        let run = run_benchmark_run(BenchmarkConfig {
            seed: 42,
            trials: 2,
        });
        let report = scorecards_to_markdown_report(&run);
        assert!(report.contains("does not prove fractal time"));
        assert!(report.contains("Scorecards"));
    }
}
