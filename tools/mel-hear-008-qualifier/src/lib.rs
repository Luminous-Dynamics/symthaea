//! Standalone, fail-closed qualifier for the MEL-HEAR-008 incidence fixture.
//!
//! This is a research contract checker, not an inference engine. Version 0
//! accepts one performer per work-performer cell and rejects ensemble rows
//! until a separate membership estimand is defined.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const TOLERANCE: f64 = 1.0e-12;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Panel {
    pub schema_version: String,
    pub study_id: String,
    pub simulation_only: bool,
    pub work_ids: Vec<String>,
    pub performer_ids: Vec<String>,
    pub cells: Vec<Cell>,
    pub estimands: Vec<Estimand>,
    pub support_audit: SupportAudit,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cell {
    pub cell_id: String,
    pub work_id: String,
    pub performer_membership: Vec<PerformerMembership>,
    #[serde(default)]
    pub session_ids: Vec<String>,
    pub status: CellStatus,
    #[serde(default)]
    pub paired_effect_mean: Option<f64>,
    #[serde(default)]
    pub repeat_count: Option<usize>,
    #[serde(default)]
    pub missingness_reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformerMembership {
    pub performer_id: String,
    pub weight: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CellStatus {
    Observed,
    MissingNotSampled,
    MissingUnknown,
    UnjoinablePerformer,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Estimand {
    pub estimand_id: String,
    pub kind: EstimandKind,
    pub target_population: String,
    pub weighting_rule: String,
    pub support_disposition: SupportDisposition,
    pub cell_weights: Vec<CellWeight>,
    #[serde(default)]
    pub target_work_weights: Vec<MarginalWeight>,
    #[serde(default)]
    pub target_performer_weights: Vec<MarginalWeight>,
    #[serde(default)]
    pub expected_point_estimate_fraction: Option<String>,
    #[serde(default)]
    pub expected_point_estimate_decimal: Option<f64>,
    #[serde(default)]
    pub disposition_reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EstimandKind {
    ObservedCellMean,
    EqualWorkMacroMean,
    EqualPerformerMacroMean,
    DeclaredTargetPopulationMean,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SupportDisposition {
    Supported,
    DescriptiveOnly,
    NotEstimable,
    ModelBasedSensitivity,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CellWeight {
    pub cell_id: String,
    pub weight: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarginalWeight {
    pub unit_id: String,
    pub weight: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupportAudit {
    pub work_count: usize,
    pub performer_count: usize,
    pub possible_single_performer_cells: usize,
    pub observed_cells: usize,
    pub complete_cartesian_support: bool,
    pub connected_components: usize,
    pub isolated_work_ids: Vec<String>,
    pub isolated_performer_ids: Vec<String>,
    pub max_work_degree: usize,
    pub max_performer_degree: usize,
    #[serde(default)]
    pub incidence_graph_digest: Option<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct AuditReport {
    pub schema: String,
    pub study_id: String,
    pub work_count: usize,
    pub performer_count: usize,
    pub potential_cells: usize,
    pub observed_cells: usize,
    pub missing_cells: usize,
    pub complete_cartesian_support: bool,
    pub connected_components: usize,
    pub isolated_work_ids: Vec<String>,
    pub isolated_performer_ids: Vec<String>,
    pub max_work_degree: usize,
    pub max_performer_degree: usize,
    pub estimands: Vec<EstimandReport>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct EstimandReport {
    pub estimand_id: String,
    pub kind: String,
    pub support_disposition: String,
    pub target_weight_sum: f64,
    pub point_estimate: Option<f64>,
    pub missing_target_cells: Vec<String>,
}

enum GraphNode {
    Work(String),
    Performer(String),
}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= TOLERANCE
}

fn validate_locator(value: &str, label: &str) -> Result<(), String> {
    let synthetic = value.strip_prefix("synthetic:").is_some_and(|suffix| {
        !suffix.is_empty()
            && suffix.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
            })
    });
    let digest = value.strip_prefix("sha256:").is_some_and(|suffix| {
        suffix.len() == 64
            && suffix
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    });
    if synthetic || digest {
        Ok(())
    } else {
        Err(format!("{label} must be a synthetic or lowercase SHA-256 locator"))
    }
}

fn unique_ids(values: &[String], label: &str) -> Result<BTreeSet<String>, String> {
    let set: BTreeSet<String> = values.iter().cloned().collect();
    if set.len() != values.len() {
        return Err(format!("{label} contains duplicate identifiers"));
    }
    for value in values {
        validate_locator(value, label)?;
    }
    Ok(set)
}

fn fraction_value(value: &str) -> Result<f64, String> {
    let (numerator, denominator) = value
        .split_once('/')
        .ok_or_else(|| "expected point-estimate fraction must use numerator/denominator".to_owned())?;
    let numerator = numerator
        .parse::<f64>()
        .map_err(|_| "invalid expected point-estimate numerator".to_owned())?;
    let denominator = denominator
        .parse::<f64>()
        .map_err(|_| "invalid expected point-estimate denominator".to_owned())?;
    if !numerator.is_finite() || !denominator.is_finite() || denominator == 0.0 {
        return Err("expected point-estimate fraction is not finite".to_owned());
    }
    Ok(numerator / denominator)
}

fn validate_marginal_weights(
    weights: &[MarginalWeight],
    allowed_ids: &BTreeSet<String>,
    label: &str,
) -> Result<(), String> {
    if weights.is_empty() {
        return Ok(());
    }
    let mut seen = BTreeSet::new();
    let mut ordered = weights.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.unit_id.cmp(&right.unit_id));
    let mut total = 0.0;
    for item in ordered {
        if !allowed_ids.contains(&item.unit_id) {
            return Err(format!("{label} references an unknown unit"));
        }
        if !seen.insert(item.unit_id.as_str()) {
            return Err(format!("{label} contains a duplicate unit"));
        }
        if !item.weight.is_finite() || item.weight <= 0.0 || item.weight > 1.0 {
            return Err(format!("{label} contains an invalid weight"));
        }
        total += item.weight;
    }
    if !approximately_equal(total, 1.0) {
        return Err(format!("{label} must sum to one"));
    }
    Ok(())
}

fn graph_support(
    work_ids: &BTreeSet<String>,
    performer_ids: &BTreeSet<String>,
    observed_cells: &[&Cell],
) -> (usize, Vec<String>, Vec<String>, usize, usize) {
    let mut work_neighbors: BTreeMap<String, BTreeSet<String>> = work_ids
        .iter()
        .map(|work| (work.clone(), BTreeSet::new()))
        .collect();
    let mut performer_neighbors: BTreeMap<String, BTreeSet<String>> = performer_ids
        .iter()
        .map(|performer| (performer.clone(), BTreeSet::new()))
        .collect();

    for cell in observed_cells {
        let performer_id = &cell.performer_membership[0].performer_id;
        work_neighbors
            .entry(cell.work_id.clone())
            .or_default()
            .insert(performer_id.clone());
        performer_neighbors
            .entry(performer_id.clone())
            .or_default()
            .insert(cell.work_id.clone());
    }

    let isolated_works = work_ids
        .iter()
        .filter(|work| work_neighbors[*work].is_empty())
        .cloned()
        .collect::<Vec<_>>();
    let isolated_performers = performer_ids
        .iter()
        .filter(|performer| performer_neighbors[*performer].is_empty())
        .cloned()
        .collect::<Vec<_>>();
    let max_work_degree = work_neighbors.values().map(BTreeSet::len).max().unwrap_or(0);
    let max_performer_degree = performer_neighbors
        .values()
        .map(BTreeSet::len)
        .max()
        .unwrap_or(0);

    let mut visited_works = BTreeSet::new();
    let mut visited_performers = BTreeSet::new();
    let mut components = 0;

    for work in work_ids {
        if visited_works.contains(work) {
            continue;
        }
        components += 1;
        let mut queue = VecDeque::from([GraphNode::Work(work.clone())]);
        visited_works.insert(work.clone());
        while let Some(node) = queue.pop_front() {
            match node {
                GraphNode::Work(current_work) => {
                    for performer in &work_neighbors[&current_work] {
                        if visited_performers.insert(performer.clone()) {
                            queue.push_back(GraphNode::Performer(performer.clone()));
                        }
                    }
                }
                GraphNode::Performer(current_performer) => {
                    for adjacent_work in &performer_neighbors[&current_performer] {
                        if visited_works.insert(adjacent_work.clone()) {
                            queue.push_back(GraphNode::Work(adjacent_work.clone()));
                        }
                    }
                }
            }
        }
    }

    for performer in performer_ids {
        if visited_performers.contains(performer) {
            continue;
        }
        components += 1;
        visited_performers.insert(performer.clone());
    }

    (
        components,
        isolated_works,
        isolated_performers,
        max_work_degree,
        max_performer_degree,
    )
}

fn validate_estimand(
    estimand: &Estimand,
    panel_cells: &[Cell],
    observed_cells: &[&Cell],
    work_ids: &BTreeSet<String>,
    performer_ids: &BTreeSet<String>,
) -> Result<EstimandReport, String> {
    if estimand.estimand_id.trim().is_empty()
        || estimand.target_population.trim().is_empty()
        || estimand.weighting_rule.trim().is_empty()
    {
        return Err("estimand identity, target population and weighting rule are required".to_owned());
    }

    validate_marginal_weights(
        &estimand.target_work_weights,
        work_ids,
        "target work weights",
    )?;
    validate_marginal_weights(
        &estimand.target_performer_weights,
        performer_ids,
        "target performer weights",
    )?;

    let cells_by_id: BTreeMap<&str, &Cell> = panel_cells
        .iter()
        .map(|cell| (cell.cell_id.as_str(), cell))
        .collect();
    let mut weights_by_id = BTreeMap::new();
    for item in &estimand.cell_weights {
        if !cells_by_id.contains_key(item.cell_id.as_str()) {
            return Err(format!("estimand {} references an unknown cell", estimand.estimand_id));
        }
        if weights_by_id.contains_key(item.cell_id.as_str()) {
            return Err(format!("estimand {} contains duplicate cell weights", estimand.estimand_id));
        }
        if !item.weight.is_finite() || item.weight <= 0.0 || item.weight > 1.0 {
            return Err(format!("estimand {} contains an invalid cell weight", estimand.estimand_id));
        }
        weights_by_id.insert(item.cell_id.as_str(), item.weight);
    }

    let target_weight_sum = weights_by_id.values().copied().sum::<f64>();
    if !approximately_equal(target_weight_sum, 1.0) {
        return Err(format!("estimand {} cell weights must sum to one", estimand.estimand_id));
    }

    let missing_target_cells = weights_by_id
        .keys()
        .filter(|cell_id| cells_by_id[*cell_id].status != CellStatus::Observed)
        .map(|cell_id| (*cell_id).to_owned())
        .collect::<Vec<_>>();

    let point_estimate = match estimand.support_disposition {
        SupportDisposition::Supported | SupportDisposition::DescriptiveOnly => {
            if !missing_target_cells.is_empty() {
                return Err(format!(
                    "estimand {} claims a numeric result while target cells are missing",
                    estimand.estimand_id
                ));
            }
            if weights_by_id.len() != observed_cells.len() {
                return Err(format!(
                    "estimand {} does not weight every observed cell exactly once",
                    estimand.estimand_id
                ));
            }
            let mut contributions = weights_by_id
                .iter()
                .map(|(cell_id, weight)| {
                    let outcome = cells_by_id[cell_id].paired_effect_mean.ok_or_else(|| {
                        format!("estimand {} references an observed cell without an outcome", estimand.estimand_id)
                    })?;
                    Ok((*cell_id, *weight, outcome))
                })
                .collect::<Result<Vec<_>, String>>()?;
            contributions.sort_by(|left, right| left.0.cmp(right.0));
            let estimate = contributions
                .iter()
                .map(|(_, weight, outcome)| weight * outcome)
                .sum::<f64>();

            let expected_decimal = estimand
                .expected_point_estimate_decimal
                .ok_or_else(|| format!("estimand {} is missing its expected point estimate", estimand.estimand_id))?;
            let expected_fraction = estimand
                .expected_point_estimate_fraction
                .as_deref()
                .ok_or_else(|| format!("estimand {} is missing its exact expected fraction", estimand.estimand_id))?;
            if !approximately_equal(estimate, expected_decimal)
                || !approximately_equal(estimate, fraction_value(expected_fraction)?)
            {
                return Err(format!("estimand {} does not reproduce its expected point estimate", estimand.estimand_id));
            }

            match estimand.kind {
                EstimandKind::ObservedCellMean => {
                    let expected_weight = 1.0 / observed_cells.len() as f64;
                    if contributions
                        .iter()
                        .any(|(_, weight, _)| !approximately_equal(*weight, expected_weight))
                    {
                        return Err("observed-cell mean must weight every observed cell equally".to_owned());
                    }
                }
                EstimandKind::EqualWorkMacroMean => {
                    let mut counts = BTreeMap::<&str, usize>::new();
                    for cell in observed_cells {
                        *counts.entry(cell.work_id.as_str()).or_default() += 1;
                    }
                    if counts.len() != work_ids.len() {
                        return Err("equal-work mean is unsupported when a work has no observed cells".to_owned());
                    }
                    for (cell_id, weight, _) in &contributions {
                        let work_id = cells_by_id[cell_id].work_id.as_str();
                        let expected_weight = 1.0 / (counts.len() * counts[work_id]) as f64;
                        if !approximately_equal(*weight, expected_weight) {
                            return Err("equal-work macro weights do not give every work equal mass".to_owned());
                        }
                    }
                }
                EstimandKind::EqualPerformerMacroMean => {
                    let mut counts = BTreeMap::<&str, usize>::new();
                    for cell in observed_cells {
                        let performer = cell.performer_membership[0].performer_id.as_str();
                        *counts.entry(performer).or_default() += 1;
                    }
                    if counts.len() != performer_ids.len() {
                        return Err("equal-performer mean is unsupported when a performer has no observed cells".to_owned());
                    }
                    for (cell_id, weight, _) in &contributions {
                        let performer = cells_by_id[cell_id].performer_membership[0].performer_id.as_str();
                        let expected_weight = 1.0 / (counts.len() * counts[performer]) as f64;
                        if !approximately_equal(*weight, expected_weight) {
                            return Err("equal-performer macro weights do not give every performer equal mass".to_owned());
                        }
                    }
                }
                EstimandKind::DeclaredTargetPopulationMean => {}
            }
            Some(estimate)
        }
        SupportDisposition::NotEstimable => {
            if missing_target_cells.is_empty() {
                return Err(format!(
                    "estimand {} is marked not-estimable without a missing target cell",
                    estimand.estimand_id
                ));
            }
            if estimand.expected_point_estimate_decimal.is_some()
                || estimand.expected_point_estimate_fraction.is_some()
            {
                return Err(format!(
                    "not-estimable estimand {} must not carry a numeric point estimate",
                    estimand.estimand_id
                ));
            }
            if weights_by_id.len() != panel_cells.len() {
                return Err(format!(
                    "not-estimable target {} must weight every declared target cell",
                    estimand.estimand_id
                ));
            }
            if estimand.disposition_reason.as_deref().is_none_or(str::is_empty) {
                return Err(format!(
                    "not-estimable estimand {} requires a disposition reason",
                    estimand.estimand_id
                ));
            }
            None
        }
        SupportDisposition::ModelBasedSensitivity => {
            if estimand.expected_point_estimate_decimal.is_some() {
                return Err("model-based sensitivity results require a separate model report".to_owned());
            }
            None
        }
    };

    Ok(EstimandReport {
        estimand_id: estimand.estimand_id.clone(),
        kind: format!("{:?}", estimand.kind),
        support_disposition: format!("{:?}", estimand.support_disposition),
        target_weight_sum,
        point_estimate,
        missing_target_cells,
    })
}

pub fn audit_panel(panel: &Panel) -> Result<AuditReport, String> {
    if panel.schema_version != "mel-hear-008-incidence-v0" {
        return Err("unsupported incidence manifest schema version".to_owned());
    }
    if !panel.simulation_only {
        return Err("v0 qualifier accepts only explicitly synthetic simulation panels".to_owned());
    }
    validate_locator(&panel.study_id, "study_id")?;

    let work_ids = unique_ids(&panel.work_ids, "work_ids")?;
    let performer_ids = unique_ids(&panel.performer_ids, "performer_ids")?;
    if work_ids.is_empty() || performer_ids.is_empty() {
        return Err("work_ids and performer_ids must both be non-empty".to_owned());
    }

    let potential_cells = work_ids
        .len()
        .checked_mul(performer_ids.len())
        .ok_or_else(|| "potential work-performer cell count overflow".to_owned())?;
    if panel.cells.len() != potential_cells {
        return Err("v0 requires an explicit record for every potential work-performer cell".to_owned());
    }

    let mut seen_cell_ids = BTreeSet::new();
    let mut seen_pairs = BTreeSet::new();
    let mut observed_cells = Vec::new();
    for cell in &panel.cells {
        validate_locator(&cell.cell_id, "cell_id")?;
        if !seen_cell_ids.insert(cell.cell_id.as_str()) {
            return Err("cell identifiers must be unique".to_owned());
        }
        if !work_ids.contains(&cell.work_id) {
            return Err("cell references an undeclared source work".to_owned());
        }
        if cell.performer_membership.len() != 1 {
            return Err("v0 rejects ensemble or set-valued performer cells until their estimand is defined".to_owned());
        }
        let membership = &cell.performer_membership[0];
        if !performer_ids.contains(&membership.performer_id) {
            return Err("cell references an undeclared performer".to_owned());
        }
        if !membership.weight.is_finite() || !approximately_equal(membership.weight, 1.0) {
            return Err("single-performer cell membership weight must equal one".to_owned());
        }
        if !seen_pairs.insert((cell.work_id.as_str(), membership.performer_id.as_str())) {
            return Err("duplicate work-performer cell; aggregate repeats in one cell record".to_owned());
        }

        match cell.status {
            CellStatus::Observed => {
                let value = cell
                    .paired_effect_mean
                    .ok_or_else(|| "observed cells require paired_effect_mean".to_owned())?;
                if !value.is_finite() || cell.repeat_count.is_none_or(|count| count == 0) {
                    return Err("observed cells require a finite outcome and positive repeat_count".to_owned());
                }
                observed_cells.push(cell);
            }
            CellStatus::MissingNotSampled
            | CellStatus::MissingUnknown
            | CellStatus::UnjoinablePerformer => {
                if cell.paired_effect_mean.is_some() || cell.repeat_count.is_some() {
                    return Err("missing cells must not carry observed outcomes or repeat counts".to_owned());
                }
                if cell.missingness_reason.as_deref().is_none_or(str::is_empty) {
                    return Err("missing cells require a non-empty missingness_reason".to_owned());
                }
            }
        }
    }

    let components = graph_support(&work_ids, &performer_ids, &observed_cells);
    let complete_cartesian_support = observed_cells.len() == potential_cells;
    let observed_pairs = seen_pairs.len();
    if observed_pairs != potential_cells {
        return Err("incidence panel does not represent every work-performer pair exactly once".to_owned());
    }

    let support = &panel.support_audit;
    if support.work_count != work_ids.len()
        || support.performer_count != performer_ids.len()
        || support.possible_single_performer_cells != potential_cells
        || support.observed_cells != observed_cells.len()
        || support.complete_cartesian_support != complete_cartesian_support
        || support.connected_components != components.0
        || support.isolated_work_ids != components.1
        || support.isolated_performer_ids != components.2
        || support.max_work_degree != components.3
        || support.max_performer_degree != components.4
    {
        return Err("declared support_audit does not match recomputed incidence support".to_owned());
    }

    let mut estimand_ids = BTreeSet::new();
    let mut estimands = panel
        .estimands
        .iter()
        .map(|estimand| {
            if !estimand_ids.insert(estimand.estimand_id.as_str()) {
                return Err("estimand identifiers must be unique".to_owned());
            }
            validate_estimand(
                estimand,
                &panel.cells,
                &observed_cells,
                &work_ids,
                &performer_ids,
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    estimands.sort_by(|left, right| left.estimand_id.cmp(&right.estimand_id));

    Ok(AuditReport {
        schema: "mel-hear-008-incidence-audit-v0".to_owned(),
        study_id: panel.study_id.clone(),
        work_count: work_ids.len(),
        performer_count: performer_ids.len(),
        potential_cells,
        observed_cells: observed_cells.len(),
        missing_cells: panel.cells.len() - observed_cells.len(),
        complete_cartesian_support,
        connected_components: components.0,
        isolated_work_ids: components.1,
        isolated_performer_ids: components.2,
        max_work_degree: components.3,
        max_performer_degree: components.4,
        estimands,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Panel {
        serde_json::from_str(include_str!(
            "../../../research/mel-hear-008/incidence-panel-v0.json"
        ))
        .expect("retained incidence fixture must parse")
    }

    fn estimate(report: &AuditReport, id: &str) -> f64 {
        report
            .estimands
            .iter()
            .find(|item| item.estimand_id == id)
            .and_then(|item| item.point_estimate)
            .expect("supported estimand must have a point estimate")
    }

    #[test]
    fn known_answer_estimands_reproduce_exact_panel_means() {
        let report = audit_panel(&fixture()).expect("fixture must qualify");
        assert_eq!(report.work_count, 3);
        assert_eq!(report.performer_count, 3);
        assert_eq!(report.potential_cells, 9);
        assert_eq!(report.observed_cells, 7);
        assert_eq!(report.missing_cells, 2);
        assert!(!report.complete_cartesian_support);
        assert_eq!(report.connected_components, 1);
        assert!((estimate(&report, "observed-cell-v0") - 4.0 / 7.0).abs() < TOLERANCE);
        assert!((estimate(&report, "equal-work-v0") - 5.0 / 9.0).abs() < TOLERANCE);
        assert!((estimate(&report, "equal-performer-v0") - 11.0 / 18.0).abs() < TOLERANCE);

        let target = report
            .estimands
            .iter()
            .find(|item| item.estimand_id == "full-crossed-target-v0")
            .expect("full target must be present");
        assert_eq!(target.support_disposition, "NotEstimable");
        assert_eq!(target.missing_target_cells.len(), 2);
        assert_eq!(target.point_estimate, None);
    }

    #[test]
    fn row_order_does_not_change_audit_result() {
        let panel = fixture();
        let expected = audit_panel(&panel).expect("original fixture must qualify");
        let mut reordered = panel.clone();
        reordered.work_ids.reverse();
        reordered.performer_ids.reverse();
        reordered.cells.reverse();
        reordered.estimands.reverse();
        for estimand in &mut reordered.estimands {
            estimand.cell_weights.reverse();
            estimand.target_work_weights.reverse();
            estimand.target_performer_weights.reverse();
        }
        let actual = audit_panel(&reordered).expect("reordered fixture must qualify");
        assert_eq!(expected, actual);
    }

    #[test]
    fn omitted_incidence_record_is_rejected() {
        let mut panel = fixture();
        panel.cells.pop();
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn duplicate_work_performer_pair_is_rejected() {
        let mut panel = fixture();
        panel.cells[0].cell_id = "synthetic:duplicate-cell".to_owned();
        panel.cells[0].work_id = panel.cells[1].work_id.clone();
        panel.cells[0].performer_membership = panel.cells[1].performer_membership.clone();
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn ensemble_membership_is_not_counted_as_independent_performers() {
        let mut panel = fixture();
        panel.cells[0].performer_membership.push(PerformerMembership {
            performer_id: "synthetic:performer-2".to_owned(),
            weight: 0.5,
        });
        panel.cells[0].performer_membership[0].weight = 0.5;
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn incorrect_recomputed_support_is_rejected() {
        let mut panel = fixture();
        panel.support_audit.connected_components = 2;
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn a_missing_target_cannot_be_promoted_to_supported() {
        let mut panel = fixture();
        let target = panel
            .estimands
            .iter_mut()
            .find(|item| item.estimand_id == "full-crossed-target-v0")
            .expect("full target must exist");
        target.support_disposition = SupportDisposition::Supported;
        assert!(audit_panel(&panel).is_err());
    }

    #[test]
    fn wrong_known_answer_is_rejected() {
        let mut panel = fixture();
        let estimand = panel
            .estimands
            .iter_mut()
            .find(|item| item.estimand_id == "equal-work-v0")
            .expect("equal-work estimand must exist");
        estimand.expected_point_estimate_decimal = Some(0.5);
        assert!(audit_panel(&panel).is_err());
    }
}
