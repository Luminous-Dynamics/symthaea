//! Dependency-free scoring kernel for the candidate TCCIP profile.
//!
//! This consumes task utility observations from an independently defined rollout
//! evaluator. It does not simulate a causal system or establish that an
//! intervention is causally valid. See
//! docs/research/TASK_CONDITIONED_COUNTERFACTUAL_INTEGRATION_PROFILE_V0.md.
//!
//! Run with: cargo run --example task_conditioned_counterfactual_profile
//! Test with: cargo test --example task_conditioned_counterfactual_profile
//!
//! All utilities must be pre-normalized to [0, 1]. A negative TCCIP is meaningful:
//! cross-component cuts can improve task utility relative to matched controls.

use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InterventionKind {
    CrossCut,
    MatchedControl,
}

#[derive(Clone, Debug, PartialEq)]
struct InterventionOutcome {
    id: String,
    kind: InterventionKind,
    cost: u32,
    utility: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct ProfilePoint {
    max_cost: u32,
    intact_utility: f64,
    cross_cut_mean_loss: f64,
    control_mean_loss: f64,
    tccip: f64,
    worst_cross_cut_utility: f64,
    n_cross_cut: usize,
    n_matched_control: usize,
}

fn mean_loss(intact_utility: f64, outcomes: &[&InterventionOutcome]) -> f64 {
    outcomes
        .iter()
        .map(|outcome| intact_utility - outcome.utility)
        .sum::<f64>()
        / outcomes.len() as f64
}

/// Summarize the task-utility contrast at each cumulative intervention budget.
///
/// TCCIP(k) = mean[J(intact) - J(cross-cut)]
///          - mean[J(intact) - J(matched-control)].
///
/// Returns an error rather than manufacturing a zero when a budget lacks either
/// treatment or control observations. The caller is responsible for defining
/// causal interventions, matching controls, sampling rollouts, and uncertainty.
fn summarize(
    intact_utility: f64,
    outcomes: &[InterventionOutcome],
    budgets: &[u32],
) -> Result<Vec<ProfilePoint>, String> {
    if !intact_utility.is_finite() || !(0.0..=1.0).contains(&intact_utility) {
        return Err("intact utility must be finite and within [0, 1]".into());
    }
    if budgets.is_empty() {
        return Err("at least one intervention budget is required".into());
    }
    if budgets.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("budgets must be strictly increasing".into());
    }

    let mut ids = HashSet::new();
    for outcome in outcomes {
        if outcome.id.trim().is_empty() {
            return Err("intervention identifiers must be non-empty".into());
        }
        if !ids.insert(outcome.id.as_str()) {
            return Err(format!("duplicate intervention identifier: {}", outcome.id));
        }
        if !outcome.utility.is_finite() || !(0.0..=1.0).contains(&outcome.utility) {
            return Err(format!(
                "utility for {} must be finite and within [0, 1]",
                outcome.id
            ));
        }
    }

    budgets
        .iter()
        .copied()
        .map(|max_cost| {
            let eligible = outcomes
                .iter()
                .filter(|outcome| outcome.cost <= max_cost)
                .collect::<Vec<_>>();
            let cross = eligible
                .iter()
                .copied()
                .filter(|outcome| outcome.kind == InterventionKind::CrossCut)
                .collect::<Vec<_>>();
            let controls = eligible
                .iter()
                .copied()
                .filter(|outcome| outcome.kind == InterventionKind::MatchedControl)
                .collect::<Vec<_>>();

            if cross.is_empty() {
                return Err(format!(
                    "no cross-cut observations at intervention budget {max_cost}"
                ));
            }
            if controls.is_empty() {
                return Err(format!(
                    "no matched-control observations at intervention budget {max_cost}"
                ));
            }

            let cross_cut_mean_loss = mean_loss(intact_utility, &cross);
            let control_mean_loss = mean_loss(intact_utility, &controls);
            let worst_cross_cut_utility = cross
                .iter()
                .map(|outcome| outcome.utility)
                .fold(f64::INFINITY, f64::min);

            Ok(ProfilePoint {
                max_cost,
                intact_utility,
                cross_cut_mean_loss,
                control_mean_loss,
                tccip: cross_cut_mean_loss - control_mean_loss,
                worst_cross_cut_utility,
                n_cross_cut: cross.len(),
                n_matched_control: controls.len(),
            })
        })
        .collect()
}

fn main() {
    // Demonstration data only; these outcomes are illustrative inputs, not a
    // simulated or measured result. A real benchmark must retain raw rollouts.
    let outcomes = vec![
        InterventionOutcome {
            id: "cross_cut_a".into(),
            kind: InterventionKind::CrossCut,
            cost: 1,
            utility: 0.50,
        },
        InterventionOutcome {
            id: "cross_cut_b".into(),
            kind: InterventionKind::CrossCut,
            cost: 2,
            utility: 0.60,
        },
        InterventionOutcome {
            id: "matched_sham_a".into(),
            kind: InterventionKind::MatchedControl,
            cost: 1,
            utility: 1.00,
        },
        InterventionOutcome {
            id: "matched_sham_b".into(),
            kind: InterventionKind::MatchedControl,
            cost: 2,
            utility: 0.90,
        },
    ];

    match summarize(1.0, &outcomes, &[1, 2]) {
        Ok(profile) => {
            println!(
                "max_cost\tintact_utility\tcross_cut_mean_loss\tcontrol_mean_loss\ttccip\tworst_cross_cut_utility\tn_cross_cut\tn_matched_control"
            );
            for point in profile {
                println!(
                    "{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{}\t{}",
                    point.max_cost,
                    point.intact_utility,
                    point.cross_cut_mean_loss,
                    point.control_mean_loss,
                    point.tccip,
                    point.worst_cross_cut_utility,
                    point.n_cross_cut,
                    point.n_matched_control
                );
            }
        }
        Err(error) => {
            eprintln!("NOT QUALIFIED: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(id: &str, kind: InterventionKind, cost: u32, utility: f64) -> InterventionOutcome {
        InterventionOutcome {
            id: id.into(),
            kind,
            cost,
            utility,
        }
    }

    #[test]
    fn useful_cross_component_coupling_has_positive_matched_contrast() {
        let data = vec![
            outcome("cross1", InterventionKind::CrossCut, 1, 0.50),
            outcome("cross2", InterventionKind::CrossCut, 1, 0.60),
            outcome("sham1", InterventionKind::MatchedControl, 1, 1.00),
            outcome("sham2", InterventionKind::MatchedControl, 1, 0.90),
        ];
        let profile = summarize(1.0, &data, &[1]).unwrap();
        assert!((profile[0].cross_cut_mean_loss - 0.45).abs() < 1e-12);
        assert!((profile[0].control_mean_loss - 0.05).abs() < 1e-12);
        assert!((profile[0].tccip - 0.40).abs() < 1e-12);
        assert!((profile[0].worst_cross_cut_utility - 0.50).abs() < 1e-12);
    }

    #[test]
    fn irrelevant_interactions_have_zero_contrast_when_losses_match() {
        let data = vec![
            outcome("cross", InterventionKind::CrossCut, 1, 0.80),
            outcome("sham", InterventionKind::MatchedControl, 1, 0.80),
        ];
        let profile = summarize(0.90, &data, &[1]).unwrap();
        assert!(profile[0].tccip.abs() < 1e-12);
    }

    #[test]
    fn harmful_cross_component_coupling_can_produce_negative_contrast() {
        let data = vec![
            outcome("cross", InterventionKind::CrossCut, 1, 0.90),
            outcome("sham", InterventionKind::MatchedControl, 1, 0.50),
        ];
        let profile = summarize(0.50, &data, &[1]).unwrap();
        assert!((profile[0].tccip + 0.40).abs() < 1e-12);
    }

    #[test]
    fn cumulative_budgets_include_only_interventions_within_budget() {
        let data = vec![
            outcome("cross1", InterventionKind::CrossCut, 1, 0.50),
            outcome("cross2", InterventionKind::CrossCut, 2, 0.80),
            outcome("sham1", InterventionKind::MatchedControl, 1, 1.00),
            outcome("sham2", InterventionKind::MatchedControl, 2, 0.90),
        ];
        let profile = summarize(1.0, &data, &[1, 2]).unwrap();
        assert_eq!(profile[0].n_cross_cut, 1);
        assert_eq!(profile[0].n_matched_control, 1);
        assert_eq!(profile[1].n_cross_cut, 2);
        assert_eq!(profile[1].n_matched_control, 2);
    }

    #[test]
    fn missing_control_or_cross_cut_is_not_silently_zero() {
        let cross_only = vec![outcome("cross", InterventionKind::CrossCut, 1, 0.5)];
        assert!(summarize(1.0, &cross_only, &[1])
            .unwrap_err()
            .contains("no matched-control"));

        let control_only = vec![outcome("sham", InterventionKind::MatchedControl, 1, 1.0)];
        assert!(summarize(1.0, &control_only, &[1])
            .unwrap_err()
            .contains("no cross-cut"));
    }

    #[test]
    fn rejects_nonfinite_or_out_of_range_utility() {
        assert!(summarize(f64::NAN, &[], &[1]).is_err());
        let data = vec![
            outcome("cross", InterventionKind::CrossCut, 1, 1.1),
            outcome("sham", InterventionKind::MatchedControl, 1, 1.0),
        ];
        assert!(summarize(1.0, &data, &[1]).unwrap_err().contains("cross"));
    }

    #[test]
    fn rejects_duplicate_ids_and_invalid_budgets() {
        let duplicate = vec![
            outcome("same", InterventionKind::CrossCut, 1, 0.5),
            outcome("same", InterventionKind::MatchedControl, 1, 1.0),
        ];
        assert!(summarize(1.0, &duplicate, &[1])
            .unwrap_err()
            .contains("duplicate"));

        let data = vec![
            outcome("cross", InterventionKind::CrossCut, 1, 0.5),
            outcome("sham", InterventionKind::MatchedControl, 1, 1.0),
        ];
        assert!(summarize(1.0, &data, &[2, 1]).unwrap_err().contains("strictly increasing"));
    }
}
