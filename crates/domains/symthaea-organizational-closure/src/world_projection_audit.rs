//! Evaluator-only construction audit for matched observational views.
//!
//! This module compares estimator-visible columns with a separately retained
//! master-world projection, verifies the target identity Y(1)-Y(0)=tau(X), and
//! validates optional Stay/Move target transformations. Keep the oracle and
//! master projection out of every agent-facing payload.

use super::observational_views::{MatchedObservationalViews, ObservationTable};
use crate::ContentDigest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub struct MasterWorldProjection {
    /// Full evaluator-side context projection, including variables hidden
    /// from one or more observational views.
    pub context: ObservationTable,
    /// Full evaluator-side query projection, including hidden auxiliary data.
    pub query: ObservationTable,
    /// Potential outcomes and target CATE in the exact query-unit order.
    pub query_y0: Vec<f64>,
    pub query_y1: Vec<f64>,
    pub query_tau: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StayMoveTargetTruth {
    /// Stay preserves the original target CATE pointwise.
    pub stay_tau: Vec<f64>,
    /// Move changes the target by a frozen multiplicative scale.
    pub move_tau: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchedWorldOracle {
    pub scm_digest: ContentDigest,
    pub estimand_digest: ContentDigest,
    pub master: MasterWorldProjection,
    pub stay_move: Option<StayMoveTargetTruth>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldProjectionAuditPolicy {
    pub absolute_tolerance: f64,
    pub move_effect_scale: f64,
}

impl Default for WorldProjectionAuditPolicy {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1.0e-10,
            move_effect_scale: 2.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldProjectionAuditQualification {
    Qualified,
    NotQualified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldProjectionAuditFindingCode {
    InvalidTolerance,
    InvalidMoveScale,
    ScmDigestMismatch,
    EstimandDigestMismatch,
    ContextUnitIdMismatch,
    QueryUnitIdMismatch,
    MissingMasterContextColumn,
    MissingMasterQueryColumn,
    ContextProjectionMismatch,
    QueryProjectionMismatch,
    QueryTargetLengthMismatch,
    PotentialOutcomeLengthMismatch,
    NonFinitePotentialOutcome,
    PotentialEffectTargetMismatch,
    StayTargetLengthMismatch,
    MoveTargetLengthMismatch,
    NonFiniteStayMoveTarget,
    StayTargetMismatch,
    MoveTargetMismatch,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorldProjectionAuditFinding {
    pub code: WorldProjectionAuditFindingCode,
    pub regime: Option<super::observational_views::IdentificationRegime>,
    pub column: Option<String>,
    pub unit_id: Option<u64>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorldProjectionAuditReport {
    pub qualification: WorldProjectionAuditQualification,
    pub view_count: usize,
    pub context_column_projection_checks: usize,
    pub query_column_projection_checks: usize,
    pub query_target_count: usize,
    pub max_potential_effect_target_gap: Option<f64>,
    pub max_stay_target_gap: Option<f64>,
    pub max_move_target_gap: Option<f64>,
    pub findings: Vec<WorldProjectionAuditFinding>,
}

impl WorldProjectionAuditReport {
    pub fn is_qualified(&self) -> bool {
        self.qualification == WorldProjectionAuditQualification::Qualified
            && self.findings.is_empty()
    }
}

fn push_finding(
    findings: &mut Vec<WorldProjectionAuditFinding>,
    code: WorldProjectionAuditFindingCode,
    regime: Option<super::observational_views::IdentificationRegime>,
    column: Option<String>,
    unit_id: Option<u64>,
    detail: impl Into<String>,
) {
    findings.push(WorldProjectionAuditFinding {
        code,
        regime,
        column,
        unit_id,
        detail: detail.into(),
    });
}

fn validate_projection(
    view: &super::observational_views::RegimeObservationalView,
    observed: &ObservationTable,
    master: &ObservationTable,
    is_query: bool,
    findings: &mut Vec<WorldProjectionAuditFinding>,
    checks: &mut usize,
) {
    for (column, observed_rows) in &observed.columns {
        let Some(master_rows) = master.columns.get(column) else {
            push_finding(
                findings,
                if is_query {
                    WorldProjectionAuditFindingCode::MissingMasterQueryColumn
                } else {
                    WorldProjectionAuditFindingCode::MissingMasterContextColumn
                },
                Some(view.regime),
                Some(column.clone()),
                None,
                "master-world projection omits a column exposed to the estimator",
            );
            continue;
        };
        *checks = checks.saturating_add(1);
        if observed_rows != master_rows {
            let mismatch_index = observed_rows
                .iter()
                .zip(master_rows)
                .position(|(left, right)| left != right);
            let unit_id = mismatch_index.and_then(|index| {
                observed.unit_ids.get(index).copied()
            });
            push_finding(
                findings,
                if is_query {
                    WorldProjectionAuditFindingCode::QueryProjectionMismatch
                } else {
                    WorldProjectionAuditFindingCode::ContextProjectionMismatch
                },
                Some(view.regime),
                Some(column.clone()),
                unit_id,
                "estimator-visible values differ from the saved master-world projection",
            );
        }
    }
}

fn max_abs_gap(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    Some(
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max),
    )
}

/// Audit every estimator view against an evaluator-only master-world oracle.
///
/// Exact row-value projection is required for copied variables. Potential
/// outcome identities and Stay/Move target transformations use the frozen
/// absolute tolerance. This function does not validate causal assumptions;
/// it verifies that the benchmark artifact implements its declared SCM.
pub fn audit_matched_world_projection(
    matched: &MatchedObservationalViews,
    oracle: &MatchedWorldOracle,
    policy: &WorldProjectionAuditPolicy,
) -> WorldProjectionAuditReport {
    use super::observational_views::IdentificationRegime;

    let mut findings = Vec::new();
    let mut context_checks = 0_usize;
    let mut query_checks = 0_usize;
    let mut max_target_gap = None;
    let mut max_stay_gap = None;
    let mut max_move_gap = None;

    if !policy.absolute_tolerance.is_finite() || policy.absolute_tolerance < 0.0 {
        push_finding(
            &mut findings,
            WorldProjectionAuditFindingCode::InvalidTolerance,
            None,
            None,
            None,
            "absolute tolerance must be finite and non-negative",
        );
    }
    if !policy.move_effect_scale.is_finite() {
        push_finding(
            &mut findings,
            WorldProjectionAuditFindingCode::InvalidMoveScale,
            None,
            None,
            None,
            "Move effect scale must be finite",
        );
    }

    let expected_query_ids = &oracle.master.query.unit_ids;
    if oracle.master.query_y0.len() != expected_query_ids.len()
        || oracle.master.query_y1.len() != expected_query_ids.len()
        || oracle.master.query_tau.len() != expected_query_ids.len()
    {
        push_finding(
            &mut findings,
            WorldProjectionAuditFindingCode::QueryTargetLengthMismatch,
            None,
            None,
            None,
            "query Y(0), Y(1) and tau vectors must match the master query-unit count",
        );
    }

    let potential_vectors = [
        oracle.master.query_y0.as_slice(),
        oracle.master.query_y1.as_slice(),
        oracle.master.query_tau.as_slice(),
    ];
    if potential_vectors.iter().any(|values| values.iter().any(|value| !value.is_finite())) {
        push_finding(
            &mut findings,
            WorldProjectionAuditFindingCode::NonFinitePotentialOutcome,
            None,
            None,
            None,
            "potential outcomes and target effects must be finite",
        );
    }

    if oracle.master.query_y0.len() == expected_query_ids.len()
        && oracle.master.query_y1.len() == expected_query_ids.len()
        && oracle.master.query_tau.len() == expected_query_ids.len()
    {
        let gaps: Vec<f64> = oracle.master.query_y0
            .iter()
            .zip(&oracle.master.query_y1)
            .zip(&oracle.master.query_tau)
            .map(|((y0, y1), tau)| ((y1 - y0) - tau).abs())
            .collect();
        let gap = gaps.iter().copied().fold(0.0_f64, f64::max);
        max_target_gap = Some(gap);
        if gap > policy.absolute_tolerance {
            let index = gaps.iter().position(|value| *value > policy.absolute_tolerance);
            push_finding(
                &mut findings,
                WorldProjectionAuditFindingCode::PotentialEffectTargetMismatch,
                None,
                None,
                index.and_then(|i| expected_query_ids.get(i).copied()),
                "Y(1)-Y(0) does not equal the saved target CATE within tolerance",
            );
        }
    }

    for view in &matched.views {
        if view.scm_digest != oracle.scm_digest {
            push_finding(
                &mut findings,
                WorldProjectionAuditFindingCode::ScmDigestMismatch,
                Some(view.regime),
                None,
                None,
                "view SCM digest differs from the evaluator-only master-world digest",
            );
        }
        if view.estimand_digest != oracle.estimand_digest {
            push_finding(
                &mut findings,
                WorldProjectionAuditFindingCode::EstimandDigestMismatch,
                Some(view.regime),
                None,
                None,
                "view estimand digest differs from the evaluator-only target estimand",
            );
        }
        if view.context.unit_ids != oracle.master.context.unit_ids {
            push_finding(
                &mut findings,
                WorldProjectionAuditFindingCode::ContextUnitIdMismatch,
                Some(view.regime),
                None,
                None,
                "view context IDs/order differ from the master-world projection",
            );
        }
        if view.query.unit_ids != oracle.master.query.unit_ids {
            push_finding(
                &mut findings,
                WorldProjectionAuditFindingCode::QueryUnitIdMismatch,
                Some(view.regime),
                None,
                None,
                "view query IDs/order differ from the master-world projection",
            );
        }

        validate_projection(
            view,
            &view.context,
            &oracle.master.context,
            false,
            &mut findings,
            &mut context_checks,
        );
        validate_projection(
            view,
            &view.query,
            &oracle.master.query,
            true,
            &mut findings,
            &mut query_checks,
        );
    }

    if let Some(stay_move) = &oracle.stay_move {
        if stay_move.stay_tau.len() != expected_query_ids.len() {
            push_finding(
                &mut findings,
                WorldProjectionAuditFindingCode::StayTargetLengthMismatch,
                None,
                None,
                None,
                "Stay target vector must match the master query-unit count",
            );
        }
        if stay_move.move_tau.len() != expected_query_ids.len() {
            push_finding(
                &mut findings,
                WorldProjectionAuditFindingCode::MoveTargetLengthMismatch,
                None,
                None,
                None,
                "Move target vector must match the master query-unit count",
            );
        }
        if stay_move.stay_tau.iter().chain(&stay_move.move_tau).any(|value| !value.is_finite()) {
            push_finding(
                &mut findings,
                WorldProjectionAuditFindingCode::NonFiniteStayMoveTarget,
                None,
                None,
                None,
                "Stay/Move target vectors must contain only finite values",
            );
        }

        if stay_move.stay_tau.len() == expected_query_ids.len()
            && oracle.master.query_tau.len() == expected_query_ids.len()
        {
            max_stay_gap = max_abs_gap(&stay_move.stay_tau, &oracle.master.query_tau);
            if max_stay_gap.is_some_and(|gap| gap > policy.absolute_tolerance) {
                push_finding(
                    &mut findings,
                    WorldProjectionAuditFindingCode::StayTargetMismatch,
                    None,
                    None,
                    None,
                    "Stay intervention changed the target effect beyond the frozen tolerance",
                );
            }
        }
        if stay_move.move_tau.len() == expected_query_ids.len()
            && oracle.master.query_tau.len() == expected_query_ids.len()
        {
            let expected_move: Vec<f64> = oracle.master.query_tau.iter()
                .map(|tau| tau * policy.move_effect_scale)
                .collect();
            max_move_gap = max_abs_gap(&stay_move.move_tau, &expected_move);
            if max_move_gap.is_some_and(|gap| gap > policy.absolute_tolerance) {
                push_finding(
                    &mut findings,
                    WorldProjectionAuditFindingCode::MoveTargetMismatch,
                    None,
                    None,
                    None,
                    "Move intervention does not match the frozen target-effect transformation",
                );
            }
        }
    }

    // A dataset with no views is never a qualified construction.
    if matched.views.is_empty() {
        push_finding(
            &mut findings,
            WorldProjectionAuditFindingCode::ScmDigestMismatch,
            None,
            None,
            None,
            "cannot audit a master-world projection without any observational views",
        );
    }

    let qualification = if findings.is_empty() {
        WorldProjectionAuditQualification::Qualified
    } else {
        WorldProjectionAuditQualification::NotQualified
    };

    let _ = IdentificationRegime::BackDoor; // Keep the enum linked in the public audit API.

    WorldProjectionAuditReport {
        qualification,
        view_count: matched.views.len(),
        context_column_projection_checks: context_checks,
        query_column_projection_checks: query_checks,
        query_target_count: expected_query_ids.len(),
        max_potential_effect_target_gap: max_target_gap,
        max_stay_target_gap: max_stay_gap,
        max_move_target_gap: max_move_gap,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::observational_views::{
        IdentificationRegime, MatchedObservationalViews, ObservationTable,
        RegimeObservationalView,
    };
    use std::collections::BTreeMap;

    fn table(unit_ids: Vec<u64>, variables: &[&str]) -> ObservationTable {
        let columns = variables.iter().map(|name| {
            ((*name).to_owned(), unit_ids.iter().map(|id| vec![*id as f64 / 10.0]).collect())
        }).collect();
        ObservationTable::new(unit_ids, columns)
    }

    fn view(regime: IdentificationRegime, scm: ContentDigest, estimand: ContentDigest)
        -> RegimeObservationalView {
        let (extra, query_extra): (Vec<&str>, Vec<&str>) = match regime {
            IdentificationRegime::BackDoor => (vec!["U"], vec!["U"]),
            IdentificationRegime::FrontDoor => (vec!["M"], vec!["M"]),
            IdentificationRegime::InstrumentalVariable => (vec!["I"], vec!["I"]),
            IdentificationRegime::Proximal => (vec!["Z_P", "W_P"], vec!["Z_P", "W_P"]),
            IdentificationRegime::HiddenConfounding
            | IdentificationRegime::ManskiBounds
            | IdentificationRegime::SensitivityBounds => (vec![], vec![]),
            IdentificationRegime::MonotoneIvBounds => (vec!["I"], vec!["I"]),
        };
        let mut context_columns = vec!["X", "T", "Y"];
        context_columns.extend(extra);
        let mut query_columns = vec!["X"];
        query_columns.extend(query_extra);
        RegimeObservationalView::new(
            regime,
            scm,
            estimand,
            table(vec![1, 2], &context_columns),
            table(vec![3, 4], &query_columns),
        )
    }

    fn fixture() -> (MatchedObservationalViews, MatchedWorldOracle) {
        let scm = ContentDigest::of(b"master-world-v1");
        let estimand = ContentDigest::of(b"target-cate-v1");
        let matched = MatchedObservationalViews::new(vec![
            view(IdentificationRegime::BackDoor, scm, estimand),
            view(IdentificationRegime::FrontDoor, scm, estimand),
            view(IdentificationRegime::InstrumentalVariable, scm, estimand),
            view(IdentificationRegime::Proximal, scm, estimand),
        ]);

        let context = table(vec![1, 2], &["X", "T", "Y", "U", "M", "I", "Z_P", "W_P"]);
        let query = table(vec![3, 4], &["X", "U", "M", "I", "Z_P", "W_P"]);
        let master = MasterWorldProjection {
            context,
            query,
            query_y0: vec![0.2, 0.5],
            query_y1: vec![0.4, 0.2],
            query_tau: vec![0.2, -0.3],
        };
        let oracle = MatchedWorldOracle {
            scm_digest: scm,
            estimand_digest: estimand,
            master,
            stay_move: Some(StayMoveTargetTruth {
                stay_tau: vec![0.2, -0.3],
                move_tau: vec![0.4, -0.6],
            }),
        };
        (matched, oracle)
    }

    #[test]
    fn exact_master_projection_and_potential_outcome_identity_pass() {
        let (matched, oracle) = fixture();
        let report = audit_matched_world_projection(
            &matched, &oracle, &WorldProjectionAuditPolicy::default(),
        );
        assert_eq!(report.qualification, WorldProjectionAuditQualification::Qualified, "{:?}", report.findings);
        assert_eq!(report.view_count, 4);
        assert_eq!(report.query_target_count, 2);
        assert_eq!(report.max_potential_effect_target_gap, Some(0.0));
        assert_eq!(report.max_stay_target_gap, Some(0.0));
        assert_eq!(report.max_move_target_gap, Some(0.0));
    }

    #[test]
    fn altered_view_column_is_detected_against_master_projection() {
        let (mut matched, oracle) = fixture();
        matched.views[1].context.columns.get_mut("X").unwrap()[0][0] = 999.0;
        let report = audit_matched_world_projection(
            &matched, &oracle, &WorldProjectionAuditPolicy::default(),
        );
        assert_eq!(report.qualification, WorldProjectionAuditQualification::NotQualified);
        assert!(report.findings.iter().any(|finding| {
            finding.code == WorldProjectionAuditFindingCode::ContextProjectionMismatch
        }));
    }

    #[test]
    fn query_projection_mismatch_is_detected() {
        let (mut matched, oracle) = fixture();
        matched.views[2].query.columns.get_mut("I").unwrap()[1][0] = 123.0;
        let report = audit_matched_world_projection(
            &matched, &oracle, &WorldProjectionAuditPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == WorldProjectionAuditFindingCode::QueryProjectionMismatch
        }));
    }

    #[test]
    fn wrong_target_effect_identity_is_rejected() {
        let (matched, mut oracle) = fixture();
        oracle.master.query_y1[0] = 0.9;
        let report = audit_matched_world_projection(
            &matched, &oracle, &WorldProjectionAuditPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == WorldProjectionAuditFindingCode::PotentialEffectTargetMismatch
        }));
    }

    #[test]
    fn stay_target_change_is_rejected() {
        let (matched, mut oracle) = fixture();
        oracle.stay_move.as_mut().unwrap().stay_tau[0] = 0.9;
        let report = audit_matched_world_projection(
            &matched, &oracle, &WorldProjectionAuditPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == WorldProjectionAuditFindingCode::StayTargetMismatch
        }));
    }

    #[test]
    fn move_target_must_match_the_frozen_scale() {
        let (matched, mut oracle) = fixture();
        oracle.stay_move.as_mut().unwrap().move_tau[1] = -0.3;
        let report = audit_matched_world_projection(
            &matched, &oracle, &WorldProjectionAuditPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == WorldProjectionAuditFindingCode::MoveTargetMismatch
        }));
    }

    #[test]
    fn master_query_ids_and_lengths_must_match_targets() {
        let (matched, mut oracle) = fixture();
        oracle.master.query.unit_ids.reverse();
        let report = audit_matched_world_projection(
            &matched, &oracle, &WorldProjectionAuditPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == WorldProjectionAuditFindingCode::QueryUnitIdMismatch
        }));
    }

    #[test]
    fn stay_move_controls_are_optional_but_when_present_must_be_valid() {
        let (matched, mut oracle) = fixture();
        oracle.stay_move = None;
        let report = audit_matched_world_projection(
            &matched, &oracle, &WorldProjectionAuditPolicy::default(),
        );
        assert_eq!(report.qualification, WorldProjectionAuditQualification::Qualified, "{:?}", report.findings);
        assert_eq!(report.max_stay_target_gap, None);
        assert_eq!(report.max_move_target_gap, None);
    }
}
