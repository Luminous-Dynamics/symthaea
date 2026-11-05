//! Non-anticipativity audit for scenario-based battery dispatch policies.
//!
//! For two or more scenario paths that share the same information node at the
//! same decision timestamp, the implemented dispatch action must be identical.
//! The policy may react differently only after the information sets diverge.
//!
//! This is an audit of submitted scenario decisions, not a stochastic optimizer.
//! It prevents a perfect-foresight backtest from being presented as an
//! implementable forecast-informed or rolling-horizon policy.

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchAction {
    /// Positive net export, negative net import.
    pub net_power_w: i64,
    pub reserve_up_w: u64,
    pub reserve_down_w: u64,
    pub target_soc_bps: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScenarioDecision<'a> {
    pub scenario_id: &'a str,
    /// Identical node IDs represent identical information at this decision time.
    pub information_node_id: &'a str,
    pub information_set_sha256: &'a str,
    pub information_available_at_utc: i64,
    pub decision_time_utc: i64,
    pub action: DispatchAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonAnticipativityError {
    InvalidEvidenceReference,
    InvalidInformationDigest,
    InvalidTargetSoc,
    FutureInformationUsed,
    DuplicateScenarioId,
    UndeclaredScenario,
    DuplicateScenarioDecision,
    InconsistentInformationSet,
    NonAnticipativityViolation,
    ArithmeticOverflow,
    EmptyScenarioSet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonAnticipativityAssessment<'a> {
    pub decision_count: u64,
    pub scenario_count: u64,
    pub shared_information_set_count: u64,
    pub scenario_pair_comparison_count: u64,
    pub distinct_information_node_count: u64,
    pub all_shared_decisions_identical: bool,
    pub scenario_ids: &'a [&'a str],
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Verify that scenario paths with the same decision-time information node
/// choose exactly the same net-power, reserve and target-SOC action.
pub fn audit_non_anticipative_policy<'a>(
    declared_scenario_ids: &'a [&'a str],
    decisions: &[ScenarioDecision<'a>],
) -> Result<NonAnticipativityAssessment<'a>, NonAnticipativityError> {
    if declared_scenario_ids.is_empty() {
        return Err(NonAnticipativityError::EmptyScenarioSet);
    }

    let mut declared = BTreeSet::new();
    for scenario_id in declared_scenario_ids.iter().copied() {
        if !nonempty(scenario_id) {
            return Err(NonAnticipativityError::InvalidEvidenceReference);
        }
        if !declared.insert(scenario_id) {
            return Err(NonAnticipativityError::DuplicateScenarioId);
        }
    }

    let mut node_ids = BTreeSet::new();
    for (index, decision) in decisions.iter().copied().enumerate() {
        if !nonempty(decision.scenario_id) || !nonempty(decision.information_node_id) {
            return Err(NonAnticipativityError::InvalidEvidenceReference);
        }
        if !declared.contains(decision.scenario_id) {
            return Err(NonAnticipativityError::UndeclaredScenario);
        }
        if !valid_sha256(decision.information_set_sha256) {
            return Err(NonAnticipativityError::InvalidInformationDigest);
        }
        if decision.action.target_soc_bps > 10_000 {
            return Err(NonAnticipativityError::InvalidTargetSoc);
        }
        if decision.information_available_at_utc > decision.decision_time_utc {
            return Err(NonAnticipativityError::FutureInformationUsed);
        }
        node_ids.insert(decision.information_node_id);
        if decisions[..index].iter().any(|previous| {
            previous.scenario_id == decision.scenario_id
                && previous.decision_time_utc == decision.decision_time_utc
                && previous.information_node_id == decision.information_node_id
        }) {
            return Err(NonAnticipativityError::DuplicateScenarioDecision);
        }
    }

    let mut shared_information_set_count = 0_u64;
    let mut scenario_pair_comparison_count = 0_u64;

    for (index, decision) in decisions.iter().copied().enumerate() {
        let group = decisions
            .iter()
            .copied()
            .filter(|candidate| {
                candidate.decision_time_utc == decision.decision_time_utc
                    && candidate.information_node_id == decision.information_node_id
            })
            .collect::<Vec<_>>();
        if group.len() < 2 {
            continue;
        }
        if decisions[..index].iter().any(|previous| {
            previous.decision_time_utc == decision.decision_time_utc
                && previous.information_node_id == decision.information_node_id
        }) {
            continue;
        }
        shared_information_set_count = shared_information_set_count
            .checked_add(1)
            .ok_or(NonAnticipativityError::ArithmeticOverflow)?;
        let group_len = u64::try_from(group.len())
            .map_err(|_| NonAnticipativityError::ArithmeticOverflow)?;
        let pairs = group_len
            .checked_mul(group_len.saturating_sub(1))
            .and_then(|value| value.checked_div(2))
            .ok_or(NonAnticipativityError::ArithmeticOverflow)?;
        scenario_pair_comparison_count = scenario_pair_comparison_count
            .checked_add(pairs)
            .ok_or(NonAnticipativityError::ArithmeticOverflow)?;

        let reference = group[0];
        for candidate in group.iter().copied().skip(1) {
            if candidate.information_set_sha256 != reference.information_set_sha256
                || candidate.information_available_at_utc != reference.information_available_at_utc
            {
                return Err(NonAnticipativityError::InconsistentInformationSet);
            }
            if candidate.action != reference.action {
                return Err(NonAnticipativityError::NonAnticipativityViolation);
            }
        }
    }

    Ok(NonAnticipativityAssessment {
        decision_count: u64::try_from(decisions.len())
            .map_err(|_| NonAnticipativityError::ArithmeticOverflow)?,
        scenario_count: u64::try_from(declared.len())
            .map_err(|_| NonAnticipativityError::ArithmeticOverflow)?,
        shared_information_set_count,
        scenario_pair_comparison_count,
        distinct_information_node_count: u64::try_from(node_ids.len())
            .map_err(|_| NonAnticipativityError::ArithmeticOverflow)?,
        all_shared_decisions_identical: true,
        scenario_ids: declared_scenario_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const INFO_SHA_A: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const INFO_SHA_B: &str =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    const ACTION_A: DispatchAction = DispatchAction {
        net_power_w: 2_000,
        reserve_up_w: 500,
        reserve_down_w: 300,
        target_soc_bps: 7_000,
    };

    fn decision(
        scenario: &'static str,
        node: &'static str,
        digest: &'static str,
        available_at: i64,
        decision_at: i64,
        action: DispatchAction,
    ) -> ScenarioDecision<'static> {
        ScenarioDecision {
            scenario_id: scenario,
            information_node_id: node,
            information_set_sha256: digest,
            information_available_at_utc: available_at,
            decision_time_utc: decision_at,
            action,
        }
    }

    #[test]
    fn shared_information_sets_require_identical_actions() {
        let scenarios = ["price-high", "price-low", "outage"];
        let decisions = [
            decision("price-high", "root", INFO_SHA_A, 90, 100, ACTION_A),
            decision("price-low", "root", INFO_SHA_A, 90, 100, ACTION_A),
            decision("outage", "root", INFO_SHA_A, 90, 100, ACTION_A),
            decision("price-high", "after-price", INFO_SHA_B, 110, 120, ACTION_A),
            decision(
                "price-low",
                "after-price",
                INFO_SHA_B,
                110,
                120,
                DispatchAction {
                    net_power_w: 1_000,
                    ..ACTION_A
                },
            ),
        ];
        let result = audit_non_anticipative_policy(&scenarios, &decisions).unwrap();
        assert_eq!(result.scenario_count, 3);
        assert_eq!(result.decision_count, 5);
        assert_eq!(result.shared_information_set_count, 2);
        assert_eq!(result.scenario_pair_comparison_count, 4);
        assert!(result.all_shared_decisions_identical);
    }

    #[test]
    fn divergent_actions_before_information_split_are_rejected() {
        let scenarios = ["upside", "downside"];
        let decisions = [
            decision("upside", "root", INFO_SHA_A, 90, 100, ACTION_A),
            decision(
                "downside",
                "root",
                INFO_SHA_A,
                90,
                100,
                DispatchAction {
                    net_power_w: -2_000,
                    ..ACTION_A
                },
            ),
        ];
        assert_eq!(
            audit_non_anticipative_policy(&scenarios, &decisions),
            Err(NonAnticipativityError::NonAnticipativityViolation)
        );
    }

    #[test]
    fn divergent_actions_after_information_split_are_allowed() {
        let scenarios = ["upside", "downside"];
        let decisions = [
            decision("upside", "root", INFO_SHA_A, 90, 100, ACTION_A),
            decision("downside", "root", INFO_SHA_A, 90, 100, ACTION_A),
            decision("upside", "node-up", INFO_SHA_B, 110, 120, ACTION_A),
            decision(
                "downside",
                "node-down",
                INFO_SHA_A,
                110,
                120,
                DispatchAction {
                    net_power_w: -2_000,
                    ..ACTION_A
                },
            ),
        ];
        assert!(audit_non_anticipative_policy(&scenarios, &decisions).is_ok());
    }

    #[test]
    fn information_digest_and_availability_must_match_within_a_node() {
        let scenarios = ["case-a", "case-b"];
        let mismatch = [
            decision("case-a", "root", INFO_SHA_A, 90, 100, ACTION_A),
            decision("case-b", "root", INFO_SHA_B, 90, 100, ACTION_A),
        ];
        assert_eq!(
            audit_non_anticipative_policy(&scenarios, &mismatch),
            Err(NonAnticipativityError::InconsistentInformationSet)
        );

        let availability_mismatch = [
            decision("case-a", "root", INFO_SHA_A, 90, 100, ACTION_A),
            decision("case-b", "root", INFO_SHA_A, 95, 100, ACTION_A),
        ];
        assert_eq!(
            audit_non_anticipative_policy(&scenarios, &availability_mismatch),
            Err(NonAnticipativityError::InconsistentInformationSet)
        );
    }

    #[test]
    fn future_information_and_duplicate_scenario_decisions_fail_closed() {
        let scenarios = ["case-a", "case-b"];
        let future = [decision("case-a", "root", INFO_SHA_A, 101, 100, ACTION_A)];
        assert_eq!(
            audit_non_anticipative_policy(&scenarios, &future),
            Err(NonAnticipativityError::FutureInformationUsed)
        );

        let duplicate = [
            decision("case-a", "root", INFO_SHA_A, 90, 100, ACTION_A),
            decision("case-a", "root", INFO_SHA_A, 90, 100, ACTION_A),
        ];
        assert_eq!(
            audit_non_anticipative_policy(&scenarios, &duplicate),
            Err(NonAnticipativityError::DuplicateScenarioDecision)
        );
    }

    #[test]
    fn invalid_scenarios_digest_and_soc_target_fail_closed() {
        assert_eq!(
            audit_non_anticipative_policy(&[], &[]),
            Err(NonAnticipativityError::EmptyScenarioSet)
        );

        let duplicate_scenarios = ["case-a", "case-a"];
        assert_eq!(
            audit_non_anticipative_policy(&duplicate_scenarios, &[]),
            Err(NonAnticipativityError::DuplicateScenarioId)
        );

        let invalid_digest = [ScenarioDecision {
            information_set_sha256: "bad",
            ..decision("case-a", "root", INFO_SHA_A, 90, 100, ACTION_A)
        }];
        assert_eq!(
            audit_non_anticipative_policy(&["case-a"], &invalid_digest),
            Err(NonAnticipativityError::InvalidInformationDigest)
        );

        let invalid_soc = [ScenarioDecision {
            action: DispatchAction {
                target_soc_bps: 10_001,
                ..ACTION_A
            },
            ..decision("case-a", "root", INFO_SHA_A, 90, 100, ACTION_A)
        }];
        assert_eq!(
            audit_non_anticipative_policy(&["case-a"], &invalid_soc),
            Err(NonAnticipativityError::InvalidTargetSoc)
        );
    }
}
