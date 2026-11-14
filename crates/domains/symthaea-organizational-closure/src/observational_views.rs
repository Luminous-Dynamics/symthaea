//! Matched observational-view contracts for AOBench.
//!
//! This module validates that multiple estimators see different, explicitly
//! allowlisted columns from the same SCM realization and target estimand,
//! while preserving exactly the same context/query units. Evaluator-only
//! potential outcomes and target effects are forbidden from estimator views.
//!
//! This is a data-contract validator, not a causal-identification oracle. It
//! does not prove that a declared back-door/front-door/IV/proximal assumption
//! holds in the physical world; that assumption ledger is a separate input.

use crate::ContentDigest;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentificationRegime {
    BackDoor,
    FrontDoor,
    InstrumentalVariable,
    Proximal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationTable {
    /// Row identity is explicit and is never inferred from array position.
    pub unit_ids: Vec<u64>,
    /// Each key is one exposed variable. Each row vector can contain multiple
    /// coordinates (for example, a multi-dimensional X covariate).
    pub columns: BTreeMap<String, Vec<Vec<f64>>>,
}

impl ObservationTable {
    pub fn new(
        unit_ids: Vec<u64>,
        columns: BTreeMap<String, Vec<Vec<f64>>>,
    ) -> Self {
        Self { unit_ids, columns }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegimeColumnPolicy {
    pub context_columns: BTreeSet<String>,
    pub query_columns: BTreeSet<String>,
}

impl RegimeColumnPolicy {
    pub fn new(
        context_columns: impl IntoIterator<Item = String>,
        query_columns: impl IntoIterator<Item = String>,
    ) -> Self {
        Self {
            context_columns: context_columns.into_iter().collect(),
            query_columns: query_columns.into_iter().collect(),
        }
    }
}

/// One public estimator input for one matched SCM realization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegimeObservationalView {
    pub regime: IdentificationRegime,
    /// Content identity for the complete SCM realization. The validator
    /// compares identities; it does not authenticate how they were produced.
    pub scm_digest: ContentDigest,
    /// Content identity for the exact estimand and target population.
    pub estimand_digest: ContentDigest,
    pub context: ObservationTable,
    pub query: ObservationTable,
}

impl RegimeObservationalView {
    pub fn new(
        regime: IdentificationRegime,
        scm_digest: ContentDigest,
        estimand_digest: ContentDigest,
        context: ObservationTable,
        query: ObservationTable,
    ) -> Self {
        Self {
            regime,
            scm_digest,
            estimand_digest,
            context,
            query,
        }
    }
}

/// A set of matched estimator inputs derived from one hidden causal world.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchedObservationalViews {
    pub views: Vec<RegimeObservationalView>,
}

impl MatchedObservationalViews {
    pub fn new(views: Vec<RegimeObservationalView>) -> Self {
        Self { views }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationalViewPolicy {
    pub required_regimes: BTreeSet<IdentificationRegime>,
    pub column_policies: BTreeMap<IdentificationRegime, RegimeColumnPolicy>,
    /// Names are canonicalized to ASCII lowercase before checking.
    pub forbidden_columns: BTreeSet<String>,
    /// These variables must contain identical row-aligned values in every
    /// context view (the upstream CausalIDView contract uses X, T and Y).
    pub shared_context_columns: BTreeSet<String>,
    /// These variables must contain identical row-aligned values in every
    /// query view (the upstream contract uses X).
    pub shared_query_columns: BTreeSet<String>,
    pub require_identical_context_ids: bool,
    pub require_identical_query_ids: bool,
    pub require_context_query_disjointness: bool,
}

impl Default for ObservationalViewPolicy {
    fn default() -> Self {
        let column_policies = [
            (
                IdentificationRegime::BackDoor,
                RegimeColumnPolicy::new(
                    ["X", "U", "T", "Y"].into_iter().map(str::to_owned),
                    ["X", "U"].into_iter().map(str::to_owned),
                ),
            ),
            (
                IdentificationRegime::FrontDoor,
                RegimeColumnPolicy::new(
                    ["X", "M", "T", "Y"].into_iter().map(str::to_owned),
                    ["X", "M"].into_iter().map(str::to_owned),
                ),
            ),
            (
                IdentificationRegime::InstrumentalVariable,
                RegimeColumnPolicy::new(
                    ["X", "I", "T", "Y"].into_iter().map(str::to_owned),
                    ["X", "I"].into_iter().map(str::to_owned),
                ),
            ),
            (
                IdentificationRegime::Proximal,
                RegimeColumnPolicy::new(
                    ["X", "Z_P", "W_P", "T", "Y"].into_iter().map(str::to_owned),
                    ["X", "Z_P", "W_P"].into_iter().map(str::to_owned),
                ),
            ),
        ]
        .into();

        Self {
            required_regimes: [
                IdentificationRegime::BackDoor,
                IdentificationRegime::FrontDoor,
                IdentificationRegime::InstrumentalVariable,
                IdentificationRegime::Proximal,
            ]
            .into(),
            column_policies,
            forbidden_columns: [
                "tau",
                "y0",
                "y1",
                "m0",
                "m1",
                "t0",
                "t1",
                "true_effect",
                "potential_outcome",
                "potential_outcomes",
                "actual_world",
                "oracle",
                "gamma_u_x",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            shared_context_columns: ["X", "T", "Y"].into_iter().map(str::to_owned).collect(),
            shared_query_columns: ["X"].into_iter().map(str::to_owned).collect(),
            require_identical_context_ids: true,
            require_identical_query_ids: true,
            require_context_query_disjointness: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationalViewQualification {
    Qualified,
    NotQualified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationalViewFindingCode {
    EmptyViewSet,
    MissingRequiredRegime,
    DuplicateRegime,
    MissingRegimeColumnPolicy,
    ScmDigestMismatch,
    EstimandDigestMismatch,
    DuplicateContextUnitId,
    DuplicateQueryUnitId,
    ContextQueryUnitOverlap,
    ContextUnitAlignmentMismatch,
    QueryUnitAlignmentMismatch,
    MissingContextColumn,
    MissingQueryColumn,
    UnexpectedContextColumn,
    UnexpectedQueryColumn,
    ForbiddenEvaluatorColumn,
    QueryFactualTreatmentLeakage,
    QueryFactualOutcomeLeakage,
    ColumnRowCountMismatch,
    EmptyColumnVector,
    ColumnDimensionMismatch,
    NonFiniteObservedValue,
    CrossViewColumnDimensionMismatch,
    SharedContextColumnValueMismatch,
    SharedQueryColumnValueMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationalViewFinding {
    pub code: ObservationalViewFindingCode,
    pub regime: Option<IdentificationRegime>,
    pub column: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchedObservationalViewReport {
    pub qualification: ObservationalViewQualification,
    pub view_count: usize,
    pub validated_regime_count: usize,
    pub context_row_count: usize,
    pub query_row_count: usize,
    pub common_scm_digest: Option<ContentDigest>,
    pub common_estimand_digest: Option<ContentDigest>,
    pub findings: Vec<ObservationalViewFinding>,
}

impl MatchedObservationalViewReport {
    pub fn is_qualified(&self) -> bool {
        self.qualification == ObservationalViewQualification::Qualified
            && self.findings.is_empty()
    }
}

fn push_finding(
    findings: &mut Vec<ObservationalViewFinding>,
    code: ObservationalViewFindingCode,
    regime: Option<IdentificationRegime>,
    column: Option<String>,
    detail: impl Into<String>,
) {
    findings.push(ObservationalViewFinding {
        code,
        regime,
        column,
        detail: detail.into(),
    });
}

fn canonical(name: &str) -> String {
    name.to_ascii_lowercase()
}

fn duplicate_ids(ids: &[u64]) -> bool {
    let mut seen = BTreeSet::new();
    ids.iter().any(|id| !seen.insert(*id))
}

fn validate_table(
    table: &ObservationTable,
    expected_columns: &BTreeSet<String>,
    forbidden_columns: &BTreeSet<String>,
    regime: IdentificationRegime,
    is_query: bool,
    findings: &mut Vec<ObservationalViewFinding>,
) {
    let row_count = table.unit_ids.len();

    for (name, values) in &table.columns {
        let column_name = canonical(name);
        let is_forbidden = forbidden_columns.contains(&column_name);
        if is_forbidden {
            push_finding(
                findings,
                ObservationalViewFindingCode::ForbiddenEvaluatorColumn,
                Some(regime),
                Some(name.clone()),
                "estimator input contains an evaluator-only or forbidden column",
            );
        }
        if is_query && column_name == "t" {
            push_finding(
                findings,
                ObservationalViewFindingCode::QueryFactualTreatmentLeakage,
                Some(regime),
                Some(name.clone()),
                "query rows must not expose factual treatment",
            );
        }
        if is_query && column_name == "y" {
            push_finding(
                findings,
                ObservationalViewFindingCode::QueryFactualOutcomeLeakage,
                Some(regime),
                Some(name.clone()),
                "query rows must not expose factual outcome",
            );
        }

        if values.len() != row_count {
            push_finding(
                findings,
                ObservationalViewFindingCode::ColumnRowCountMismatch,
                Some(regime),
                Some(name.clone()),
                format!(
                    "column has {} rows but its table has {row_count} unit IDs",
                    values.len()
                ),
            );
        }
        let mut expected_width = None;
        for row in values {
            if row.is_empty() {
                push_finding(
                    findings,
                    ObservationalViewFindingCode::EmptyColumnVector,
                    Some(regime),
                    Some(name.clone()),
                    "each observed column row must contain at least one value",
                );
                continue;
            }
            match expected_width {
                None => expected_width = Some(row.len()),
                Some(width) if width != row.len() => push_finding(
                    findings,
                    ObservationalViewFindingCode::ColumnDimensionMismatch,
                    Some(regime),
                    Some(name.clone()),
                    "row vector dimensions differ within one column",
                ),
                _ => {}
            }
            if row.iter().any(|value| !value.is_finite()) {
                push_finding(
                    findings,
                    ObservationalViewFindingCode::NonFiniteObservedValue,
                    Some(regime),
                    Some(name.clone()),
                    "observed values must be finite",
                );
            }
        }
    }

    for expected in expected_columns {
        if !table.columns.contains_key(expected) {
            push_finding(
                findings,
                if is_query {
                    ObservationalViewFindingCode::MissingQueryColumn
                } else {
                    ObservationalViewFindingCode::MissingContextColumn
                },
                Some(regime),
                Some(expected.clone()),
                "required view column is absent",
            );
        }
    }

    for supplied in table.columns.keys() {
        if !expected_columns.contains(supplied) {
            push_finding(
                findings,
                if is_query {
                    ObservationalViewFindingCode::UnexpectedQueryColumn
                } else {
                    ObservationalViewFindingCode::UnexpectedContextColumn
                },
                Some(regime),
                Some(supplied.clone()),
                "supplied column is outside the frozen regime allowlist",
            );
        }
    }
}

/// Validate a matched set of back-door, front-door, IV and proximal views.
///
/// This checks input shape, matched-world/estimand identities, exact unit
/// alignment and estimator-visible column contracts. It does not authenticate
/// digests or prove the identification assumptions associated with a regime.
pub fn validate_matched_observational_views(
    matched: &MatchedObservationalViews,
    policy: &ObservationalViewPolicy,
) -> MatchedObservationalViewReport {
    let mut findings = Vec::new();

    if matched.views.is_empty() {
        push_finding(
            &mut findings,
            ObservationalViewFindingCode::EmptyViewSet,
            None,
            None,
            "matched observational-view set is empty",
        );
    }

    let mut by_regime: BTreeMap<IdentificationRegime, &RegimeObservationalView> =
        BTreeMap::new();
    for view in &matched.views {
        if by_regime.insert(view.regime, view).is_some() {
            push_finding(
                &mut findings,
                ObservationalViewFindingCode::DuplicateRegime,
                Some(view.regime),
                None,
                "matched view set contains more than one view for a regime",
            );
        }
    }

    for regime in &policy.required_regimes {
        if !by_regime.contains_key(regime) {
            push_finding(
                &mut findings,
                ObservationalViewFindingCode::MissingRequiredRegime,
                Some(*regime),
                None,
                "required identification regime is missing",
            );
        }
    }

    let first = matched.views.first();
    let common_scm_digest = first.map(|view| view.scm_digest);
    let common_estimand_digest = first.map(|view| view.estimand_digest);
    let common_context_ids = first.map(|view| view.context.unit_ids.as_slice());
    let common_query_ids = first.map(|view| view.query.unit_ids.as_slice());
    let mut shared_context_values: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
    let mut shared_query_values: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
    let mut common_column_dimensions: BTreeMap<String, usize> = BTreeMap::new();

    for view in &matched.views {
        if Some(view.scm_digest) != common_scm_digest {
            push_finding(
                &mut findings,
                ObservationalViewFindingCode::ScmDigestMismatch,
                Some(view.regime),
                None,
                "observational views do not share the same SCM realization digest",
            );
        }
        if Some(view.estimand_digest) != common_estimand_digest {
            push_finding(
                &mut findings,
                ObservationalViewFindingCode::EstimandDigestMismatch,
                Some(view.regime),
                None,
                "observational views do not share the same target-estimand digest",
            );
        }

        if duplicate_ids(&view.context.unit_ids) {
            push_finding(
                &mut findings,
                ObservationalViewFindingCode::DuplicateContextUnitId,
                Some(view.regime),
                None,
                "context unit IDs contain duplicates",
            );
        }
        if duplicate_ids(&view.query.unit_ids) {
            push_finding(
                &mut findings,
                ObservationalViewFindingCode::DuplicateQueryUnitId,
                Some(view.regime),
                None,
                "query unit IDs contain duplicates",
            );
        }
        if policy.require_identical_context_ids
            && common_context_ids.is_some_and(|ids| ids != view.context.unit_ids.as_slice())
        {
            push_finding(
                &mut findings,
                ObservationalViewFindingCode::ContextUnitAlignmentMismatch,
                Some(view.regime),
                None,
                "context unit IDs/order differ across observational views",
            );
        }
        if policy.require_identical_query_ids
            && common_query_ids.is_some_and(|ids| ids != view.query.unit_ids.as_slice())
        {
            push_finding(
                &mut findings,
                ObservationalViewFindingCode::QueryUnitAlignmentMismatch,
                Some(view.regime),
                None,
                "query unit IDs/order differ across observational views",
            );
        }
        if policy.require_context_query_disjointness
            && view.context.unit_ids.iter().any(|id| view.query.unit_ids.contains(id))
        {
            push_finding(
                &mut findings,
                ObservationalViewFindingCode::ContextQueryUnitOverlap,
                Some(view.regime),
                None,
                "context and query unit IDs must be disjoint",
            );
        }

        let Some(column_policy) = policy.column_policies.get(&view.regime) else {
            push_finding(
                &mut findings,
                ObservationalViewFindingCode::MissingRegimeColumnPolicy,
                Some(view.regime),
                None,
                "no frozen column policy exists for this identification regime",
            );
            continue;
        };

        validate_table(
            &view.context,
            &column_policy.context_columns,
            &policy.forbidden_columns,
            view.regime,
            false,
            &mut findings,
        );
        validate_table(
            &view.query,
            &column_policy.query_columns,
            &policy.forbidden_columns,
            view.regime,
            true,
            &mut findings,
        );

        for (table, is_query) in [(&view.context, false), (&view.query, true)] {
            for (name, rows) in &table.columns {
                if let Some(width) = rows.first().map(Vec::len) {
                    let key = canonical(name);
                    if common_column_dimensions.get(&key).is_some_and(|existing| *existing != width) {
                        push_finding(
                            &mut findings,
                            ObservationalViewFindingCode::CrossViewColumnDimensionMismatch,
                            Some(view.regime),
                            Some(name.clone()),
                            "a shared variable has different vector dimensions across views or splits",
                        );
                    } else {
                        common_column_dimensions.entry(key).or_insert(width);
                    }
                }

                let shared = if is_query {
                    policy.shared_query_columns.contains(name)
                } else {
                    policy.shared_context_columns.contains(name)
                };
                if !shared {
                    continue;
                }
                let reference = if is_query {
                    &mut shared_query_values
                } else {
                    &mut shared_context_values
                };
                if let Some(existing) = reference.get(name) {
                    if existing != rows {
                        push_finding(
                            &mut findings,
                            if is_query {
                                ObservationalViewFindingCode::SharedQueryColumnValueMismatch
                            } else {
                                ObservationalViewFindingCode::SharedContextColumnValueMismatch
                            },
                            Some(view.regime),
                            Some(name.clone()),
                            "row-aligned values of a shared column differ across observational views",
                        );
                    }
                } else {
                    reference.insert(name.clone(), rows.clone());
                }
            }
        }
    }

    let validated_regime_count = by_regime.len();
    let qualification = if findings.is_empty() {
        ObservationalViewQualification::Qualified
    } else {
        ObservationalViewQualification::NotQualified
    };

    MatchedObservationalViewReport {
        qualification,
        view_count: matched.views.len(),
        validated_regime_count,
        context_row_count: first.map_or(0, |view| view.context.unit_ids.len()),
        query_row_count: first.map_or(0, |view| view.query.unit_ids.len()),
        common_scm_digest,
        common_estimand_digest,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(unit_ids: Vec<u64>, variables: &[&str]) -> ObservationTable {
        let columns = variables
            .iter()
            .map(|variable| {
                (
                    (*variable).to_owned(),
                    unit_ids
                        .iter()
                        .map(|id| vec![*id as f64 / 10.0])
                        .collect(),
                )
            })
            .collect();
        ObservationTable::new(unit_ids, columns)
    }

    fn view(regime: IdentificationRegime, scm: ContentDigest, estimand: ContentDigest)
        -> RegimeObservationalView {
        let (extra, query_extra) = match regime {
            IdentificationRegime::BackDoor => (vec!["U"], vec!["U"]),
            IdentificationRegime::FrontDoor => (vec!["M"], vec!["M"]),
            IdentificationRegime::InstrumentalVariable => (vec!["I"], vec!["I"]),
            IdentificationRegime::Proximal => (vec!["Z_P", "W_P"], vec!["Z_P", "W_P"]),
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

    fn matched() -> MatchedObservationalViews {
        let scm = ContentDigest::of(b"same-scm-world");
        let estimand = ContentDigest::of(b"same-query-and-population");
        MatchedObservationalViews::new(vec![
            view(IdentificationRegime::BackDoor, scm, estimand),
            view(IdentificationRegime::FrontDoor, scm, estimand),
            view(IdentificationRegime::InstrumentalVariable, scm, estimand),
            view(IdentificationRegime::Proximal, scm, estimand),
        ])
    }

    #[test]
    fn four_matched_views_pass_exact_column_and_unit_contracts() {
        let report = validate_matched_observational_views(
            &matched(),
            &ObservationalViewPolicy::default(),
        );
        assert_eq!(report.qualification, ObservationalViewQualification::Qualified, "{:?}", report.findings);
        assert!(report.is_qualified());
        assert_eq!(report.view_count, 4);
        assert_eq!(report.context_row_count, 2);
        assert_eq!(report.query_row_count, 2);
    }

    #[test]
    fn evaluator_only_target_effect_leakage_is_rejected() {
        let mut matched = matched();
        matched.views[0].query.columns.insert(
            "tau".into(),
            vec![vec![0.1], vec![0.2]],
        );
        let report = validate_matched_observational_views(
            &matched,
            &ObservationalViewPolicy::default(),
        );
        assert_eq!(report.qualification, ObservationalViewQualification::NotQualified);
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::ForbiddenEvaluatorColumn
        }));
    }

    #[test]
    fn factual_query_treatment_and_outcome_are_rejected() {
        let mut matched = matched();
        matched.views[1].query.columns.insert("T".into(), vec![vec![1.0], vec![0.0]]);
        matched.views[1].query.columns.insert("Y".into(), vec![vec![0.2], vec![0.4]]);
        let report = validate_matched_observational_views(
            &matched,
            &ObservationalViewPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::QueryFactualTreatmentLeakage
        }));
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::QueryFactualOutcomeLeakage
        }));
    }

    #[test]
    fn different_query_unit_order_is_rejected() {
        let mut matched = matched();
        matched.views[2].query.unit_ids.reverse();
        for values in matched.views[2].query.columns.values_mut() {
            values.reverse();
        }
        let report = validate_matched_observational_views(
            &matched,
            &ObservationalViewPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::QueryUnitAlignmentMismatch
        }));
    }

    #[test]
    fn scm_or_estimand_digest_change_is_rejected() {
        let mut matched = matched();
        matched.views[3].scm_digest = ContentDigest::of(b"different-world");
        matched.views[2].estimand_digest = ContentDigest::of(b"different-estimand");
        let report = validate_matched_observational_views(
            &matched,
            &ObservationalViewPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::ScmDigestMismatch
        }));
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::EstimandDigestMismatch
        }));
    }

    #[test]
    fn missing_required_regime_is_rejected() {
        let mut matched = matched();
        matched.views.retain(|view| view.regime != IdentificationRegime::Proximal);
        let report = validate_matched_observational_views(
            &matched,
            &ObservationalViewPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::MissingRequiredRegime
        }));
    }

    #[test]
    fn duplicate_regime_is_rejected() {
        let mut matched = matched();
        matched.views.push(matched.views[0].clone());
        let report = validate_matched_observational_views(
            &matched,
            &ObservationalViewPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::DuplicateRegime
        }));
    }

    #[test]
    fn malformed_or_non_finite_column_values_are_rejected() {
        let mut matched = matched();
        matched.views[0].context.columns.insert("X".into(), vec![vec![f64::NAN]]);
        let report = validate_matched_observational_views(
            &matched,
            &ObservationalViewPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::ColumnRowCountMismatch
        }));
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::NonFiniteObservedValue
        }));
    }

    #[test]
    fn shared_context_values_must_remain_identical_across_views() {
        let mut matched = matched();
        matched.views[2].context.columns.get_mut("X").unwrap()[0][0] = 999.0;
        let report = validate_matched_observational_views(
            &matched,
            &ObservationalViewPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::SharedContextColumnValueMismatch
        }));
    }

    #[test]
    fn shared_query_values_must_remain_identical_across_views() {
        let mut matched = matched();
        matched.views[3].query.columns.get_mut("X").unwrap()[1][0] = 999.0;
        let report = validate_matched_observational_views(
            &matched,
            &ObservationalViewPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::SharedQueryColumnValueMismatch
        }));
    }

    #[test]
    fn shared_variable_dimensions_must_match_across_views() {
        let mut matched = matched();
        matched.views[1].query.columns.insert(
            "X".into(),
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
        );
        let report = validate_matched_observational_views(
            &matched,
            &ObservationalViewPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::CrossViewColumnDimensionMismatch
        }));
    }

    #[test]
    fn context_and_query_units_must_be_disjoint() {
        let mut matched = matched();
        matched.views[0].query.unit_ids = vec![2, 4];
        let report = validate_matched_observational_views(
            &matched,
            &ObservationalViewPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::ContextQueryUnitOverlap
        }));
    }

    #[test]
    fn unexpected_observed_column_is_rejected_even_when_not_on_forbidden_list() {
        let mut matched = matched();
        matched.views[0].query.columns.insert("surprise".into(), vec![vec![1.0], vec![2.0]]);
        let report = validate_matched_observational_views(
            &matched,
            &ObservationalViewPolicy::default(),
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == ObservationalViewFindingCode::UnexpectedQueryColumn
        }));
    }
}
