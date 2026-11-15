// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Characterization audit for action-dependent epistemic value in the active-core FEP.
//!
//! This is not a production fix. It records whether the current epistemic term changes
//! across candidate actions for a frozen belief/model. A passing test means the audit
//! reproduced the known action-invariance condition; it does not qualify information gain.

use std::env;

use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, ExpectedFreeEnergyComputer};

#[derive(Debug, Clone)]
struct ActionRow {
    action: usize,
    epistemic_value: f64,
    pragmatic_value: f64,
    novelty: f64,
    total: f64,
    predicted_entropy: f64,
}

#[derive(Debug, Clone)]
struct AuditReport {
    action_count: usize,
    epistemic_min: f64,
    epistemic_max: f64,
    epistemic_range: f64,
    pragmatic_range: f64,
    novelty_range: f64,
    action_dependent_epistemic: bool,
    rows: Vec<ActionRow>,
}

fn range(values: impl Iterator<Item = f64>) -> f64 {
    let collected = values.collect::<Vec<_>>();
    if collected.is_empty() {
        return 0.0;
    }
    let min = collected.iter().copied().fold(f64::INFINITY, f64::min);
    let max = collected
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    max - min
}

fn audit_current_fep() -> AuditReport {
    let agent = ActiveInferenceAgent::new(ActiveInferenceAgentConfig::default());
    let mut efe = ExpectedFreeEnergyComputer::new(agent.config.obs_dim);
    let mut rows = Vec::with_capacity(agent.config.num_actions);

    // Each action is evaluated against the same immutable belief and model.
    // Each action is unique in this sweep, so the action-novelty count is matched.
    for action in 0..agent.config.num_actions {
        let result = efe.compute(action, &agent.belief, &agent.model);
        rows.push(ActionRow {
            action,
            epistemic_value: result.epistemic,
            pragmatic_value: result.pragmatic,
            novelty: result.novelty,
            total: result.total,
            predicted_entropy: result.predicted_state.entropy(),
        });
    }

    let epistemic_min = rows
        .iter()
        .map(|row| row.epistemic_value)
        .fold(f64::INFINITY, f64::min);
    let epistemic_max = rows
        .iter()
        .map(|row| row.epistemic_value)
        .fold(f64::NEG_INFINITY, f64::max);
    let epistemic_range = epistemic_max - epistemic_min;
    let pragmatic_range = range(rows.iter().map(|row| row.pragmatic_value));
    let novelty_range = range(rows.iter().map(|row| row.novelty));

    AuditReport {
        action_count: rows.len(),
        epistemic_min,
        epistemic_max,
        epistemic_range,
        pragmatic_range,
        novelty_range,
        action_dependent_epistemic: epistemic_range > 1e-12,
        rows,
    }
}

fn valid_source_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn main() {
    let source_sha = env::var("SYMTHAEA_SOURCE_SHA")
        .expect("SYMTHAEA_SOURCE_SHA must bind this audit to an exact Git SHA");
    assert!(valid_source_sha(&source_sha), "invalid source SHA");

    let report = audit_current_fep();
    println!("receipt_version=hlb-fep-epistemic-audit-v1");
    println!("source_sha={source_sha}");
    println!("status=characterization_audit_not_a_performance_benchmark");
    println!("action_count={}", report.action_count);
    println!("epistemic_min={:.15}", report.epistemic_min);
    println!("epistemic_max={:.15}", report.epistemic_max);
    println!("epistemic_range={:.15}", report.epistemic_range);
    println!("pragmatic_range={:.15}", report.pragmatic_range);
    println!("novelty_range={:.15}", report.novelty_range);
    println!(
        "action_dependent_epistemic={}",
        report.action_dependent_epistemic
    );
    println!("integrated_hlb_executed=false");
    println!("information_gain_claim_qualified=false");
    for row in &report.rows {
        println!(
            "action={} epistemic={:.15} pragmatic={:.15} novelty={:.15} total={:.15} predicted_entropy={:.15}",
            row.action, row.epistemic_value, row.pragmatic_value, row.novelty, row.total, row.predicted_entropy
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_fep_epistemic_term_is_characterized_across_all_actions() {
        let report = audit_current_fep();
        assert!(report.action_count > 1);
        assert!(
            report.epistemic_range <= 1e-12,
            "action-invariance characterization changed: epistemic range={}",
            report.epistemic_range
        );
    }

    #[test]
    fn characterization_does_not_mistake_novelty_for_information_gain() {
        let report = audit_current_fep();
        assert!(report.novelty_range <= 1e-12);
        assert!(!report.action_dependent_epistemic);
    }

    #[test]
    fn action_scores_may_differ_without_epistemic_value_differing() {
        let report = audit_current_fep();
        assert!(report.rows.iter().all(|row| row.total.is_finite()));
        assert!(report.rows.iter().all(|row| row.predicted_entropy.is_finite()));
        // This condition is deliberately not required: the action score can differ via
        // pragmatic terms even though the epistemic term is action-independent.
        let _pragmatic_range_is_reported = report.pragmatic_range;
    }

    #[test]
    fn source_sha_shape_is_strict() {
        assert!(valid_source_sha("a3ba137d4c3900e2ef6731a4ca5065dc2c8288c3"));
        assert!(!valid_source_sha("queued"));
        assert!(!valid_source_sha("short"));
    }
}
