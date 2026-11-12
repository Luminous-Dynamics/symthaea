use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitKind {
    PublishedReproduction,
    LeaveOneCellOut,
    LeaveOneDesignGroupOut,
    LeaveOneProtocolOut,
    Prospective,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellObservation {
    pub cell_id: String,
    pub design_group_id: String,
    pub protocol_group_id: String,
    pub feature_available_through_cycle: u32,
    pub prediction_cutoff_cycle: u32,
    pub feature_evidence_ref: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitManifest {
    pub split_id: String,
    pub kind: SplitKind,
    pub train_cell_ids: Vec<String>,
    pub validation_cell_ids: Vec<String>,
    pub test_cell_ids: Vec<String>,
    pub train_design_group_ids: Vec<String>,
    pub validation_design_group_ids: Vec<String>,
    pub test_design_group_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CycleLifePrediction {
    pub cell_id: String,
    pub design_group_id: String,
    pub prediction_cutoff_cycle: u32,
    pub predicted_cycle_life: f64,
    pub lower_cycle_life: f64,
    pub upper_cycle_life: f64,
    pub model_evidence_ref: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutcomeKind {
    EventObserved,
    RightCensored,
    CompetingEvent,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CycleLifeOutcome {
    pub cell_id: String,
    pub kind: OutcomeKind,
    pub observed_through_cycle: u32,
    pub event_cycle: Option<u32>,
    pub event_code: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkManifest {
    pub manifest_id: String,
    pub cutoff_cycles: Vec<u32>,
    pub cells: Vec<CellObservation>,
    pub split: SplitManifest,
    pub predictions: Vec<CycleLifePrediction>,
    pub outcomes: Vec<CycleLifeOutcome>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PointScore {
    pub event_observed_cell_count: usize,
    pub event_observed_design_group_count: usize,
    pub right_censored_cell_count: usize,
    pub competing_event_cell_count: usize,
    pub unknown_outcome_cell_count: usize,
    pub mae_cycles_pooled: f64,
    pub rmse_cycles_pooled: f64,
    pub mean_absolute_percentage_error: f64,
    pub mae_cycles_design_macro: f64,
    pub interval_covered_count: usize,
    pub interval_coverage: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AuditReport {
    pub manifest_id: String,
    pub split_id: String,
    pub train_cell_count: usize,
    pub validation_cell_count: usize,
    pub test_cell_count: usize,
    pub test_design_group_count: usize,
    pub scored_prediction_count: usize,
    pub score: PointScore,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditError(pub String);

impl std::fmt::Display for AuditError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for AuditError {}

fn require_nonempty(value: &str, label: &str) -> Result<(), AuditError> {
    if value.trim().is_empty() {
        return Err(AuditError(format!("{label} must not be empty")));
    }
    Ok(())
}

fn unique_ids<'a>(
    values: impl IntoIterator<Item = &'a str>,
    label: &str,
) -> Result<BTreeSet<String>, AuditError> {
    let mut result = BTreeSet::new();

    for value in values {
        require_nonempty(value, label)?;

        if !result.insert(value.to_string()) {
            return Err(AuditError(format!("duplicate {label}: {value}")));
        }
    }

    Ok(result)
}

fn validate_outcome(outcome: &CycleLifeOutcome) -> Result<(), AuditError> {
    if outcome.observed_through_cycle == 0 {
        return Err(AuditError(format!(
            "outcome {} has no positive observation horizon",
            outcome.cell_id
        )));
    }

    match outcome.kind {
        OutcomeKind::EventObserved => {
            let event_cycle = outcome.event_cycle.ok_or_else(|| {
                AuditError(format!(
                    "event-observed outcome {} has no event cycle",
                    outcome.cell_id
                ))
            })?;

            if event_cycle == 0 || event_cycle > outcome.observed_through_cycle {
                return Err(AuditError(format!(
                    "event cycle for {} is outside its observation horizon",
                    outcome.cell_id
                )));
            }

            if outcome.event_code.is_some() {
                return Err(AuditError(format!(
                    "ordinary event {} must not carry a competing-event code",
                    outcome.cell_id
                )));
            }
        }
        OutcomeKind::RightCensored => {
            if outcome.event_cycle.is_some() || outcome.event_code.is_some() {
                return Err(AuditError(format!(
                    "right-censored outcome {} must not be converted into an event",
                    outcome.cell_id
                )));
            }
        }
        OutcomeKind::CompetingEvent => {
            let event_cycle = outcome.event_cycle.ok_or_else(|| {
                AuditError(format!(
                    "competing event {} has no event cycle",
                    outcome.cell_id
                ))
            })?;

            if event_cycle == 0 || event_cycle > outcome.observed_through_cycle {
                return Err(AuditError(format!(
                    "competing event cycle for {} is outside its observation horizon",
                    outcome.cell_id
                )));
            }

            require_nonempty(
                outcome.event_code.as_deref().unwrap_or(""),
                "competing event code",
            )?;
        }
        OutcomeKind::Unknown => {
            if outcome.event_cycle.is_some() || outcome.event_code.is_some() {
                return Err(AuditError(format!(
                    "unknown outcome {} must not carry an event label",
                    outcome.cell_id
                )));
            }
        }
    }

    Ok(())
}

fn exact_group_ids(
    cell_ids: &BTreeSet<String>,
    cell_by_id: &BTreeMap<&str, &CellObservation>,
    label: &str,
) -> Result<BTreeSet<String>, AuditError> {
    let mut groups = BTreeSet::new();

    for cell_id in cell_ids {
        let cell = cell_by_id.get(cell_id.as_str()).ok_or_else(|| {
            AuditError(format!("{label} references unknown cell {cell_id}"))
        })?;
        groups.insert(cell.design_group_id.clone());
    }

    Ok(groups)
}

fn validate_split(
    split: &SplitManifest,
    cell_by_id: &BTreeMap<&str, &CellObservation>,
    known_cell_ids: &BTreeSet<String>,
) -> Result<(BTreeSet<String>, BTreeSet<String>, BTreeSet<String>), AuditError> {
    require_nonempty(&split.split_id, "split ID")?;

    let train = unique_ids(
        split.train_cell_ids.iter().map(String::as_str),
        "train cell ID",
    )?;
    let validation = unique_ids(
        split.validation_cell_ids.iter().map(String::as_str),
        "validation cell ID",
    )?;
    let test = unique_ids(
        split.test_cell_ids.iter().map(String::as_str),
        "test cell ID",
    )?;

    if train.is_empty() || test.is_empty() {
        return Err(AuditError(
            "split requires nonempty train and test partitions".into(),
        ));
    }

    if !train.is_disjoint(&validation)
        || !train.is_disjoint(&test)
        || !validation.is_disjoint(&test)
    {
        return Err(AuditError(
            "physical cell identity leaks across train/validation/test partitions".into(),
        ));
    }

    let assigned: BTreeSet<String> = train
        .union(&validation)
        .cloned()
        .chain(test.iter().cloned())
        .collect();

    if &assigned != known_cell_ids {
        return Err(AuditError(
            "split does not assign every declared cell exactly once".into(),
        ));
    }

    for cell_id in &assigned {
        if !cell_by_id.contains_key(cell_id.as_str()) {
            return Err(AuditError(format!("split references unknown cell {cell_id}")));
        }
    }

    let declared_train_groups = unique_ids(
        split.train_design_group_ids.iter().map(String::as_str),
        "train design group ID",
    )?;
    let declared_validation_groups = unique_ids(
        split.validation_design_group_ids.iter().map(String::as_str),
        "validation design group ID",
    )?;
    let declared_test_groups = unique_ids(
        split.test_design_group_ids.iter().map(String::as_str),
        "test design group ID",
    )?;

    let actual_train_groups = exact_group_ids(&train, cell_by_id, "train partition")?;
    let actual_validation_groups =
        exact_group_ids(&validation, cell_by_id, "validation partition")?;
    let actual_test_groups = exact_group_ids(&test, cell_by_id, "test partition")?;

    if declared_train_groups != actual_train_groups
        || declared_validation_groups != actual_validation_groups
        || declared_test_groups != actual_test_groups
    {
        return Err(AuditError(
            "declared design-group partitions do not match physical cell membership".into(),
        ));
    }

    if split.kind == SplitKind::LeaveOneDesignGroupOut
        && (!actual_train_groups.is_disjoint(&actual_validation_groups)
            || !actual_train_groups.is_disjoint(&actual_test_groups)
            || !actual_validation_groups.is_disjoint(&actual_test_groups))
    {
        return Err(AuditError(
            "design-group identity leaks across a leave-one-design-group-out split".into(),
        ));
    }

    Ok((train, validation, test))
}

fn validate_manifest(manifest: &BenchmarkManifest) -> Result<(), AuditError> {
    require_nonempty(&manifest.manifest_id, "manifest ID")?;

    if manifest.cutoff_cycles.is_empty() {
        return Err(AuditError("at least one prediction cutoff is required".into()));
    }

    let cutoff_set: BTreeSet<u32> = manifest.cutoff_cycles.iter().copied().collect();

    if cutoff_set.len() != manifest.cutoff_cycles.len()
        || cutoff_set.iter().any(|cutoff| *cutoff == 0)
    {
        return Err(AuditError(
            "prediction cutoffs must be unique positive cycle counts".into(),
        ));
    }

    if manifest.cells.is_empty() {
        return Err(AuditError("at least one cell observation is required".into()));
    }

    let cell_ids = unique_ids(
        manifest.cells.iter().map(|cell| cell.cell_id.as_str()),
        "physical cell ID",
    )?;
    let cell_by_id: BTreeMap<&str, &CellObservation> = manifest
        .cells
        .iter()
        .map(|cell| (cell.cell_id.as_str(), cell))
        .collect();

    for cell in &manifest.cells {
        require_nonempty(&cell.design_group_id, "design group ID")?;
        require_nonempty(&cell.protocol_group_id, "protocol group ID")?;
        require_nonempty(&cell.feature_evidence_ref, "feature evidence reference")?;

        if cell.feature_available_through_cycle == 0
            || cell.prediction_cutoff_cycle == 0
            || cell.prediction_cutoff_cycle > cell.feature_available_through_cycle
        {
            return Err(AuditError(format!(
                "cell {} uses features unavailable at its prediction cutoff",
                cell.cell_id
            )));
        }

        if !cutoff_set.contains(&cell.prediction_cutoff_cycle) {
            return Err(AuditError(format!(
                "cell {} uses undeclared prediction cutoff {}",
                cell.cell_id, cell.prediction_cutoff_cycle
            )));
        }
    }

    let (train, validation, test) =
        validate_split(&manifest.split, &cell_by_id, &cell_ids)?;

    let outcome_ids = unique_ids(
        manifest.outcomes.iter().map(|outcome| outcome.cell_id.as_str()),
        "outcome cell ID",
    )?;

    if outcome_ids != cell_ids {
        return Err(AuditError(
            "every physical cell must have exactly one explicit outcome state".into(),
        ));
    }

    let outcome_by_cell: BTreeMap<&str, &CycleLifeOutcome> = manifest
        .outcomes
        .iter()
        .map(|outcome| (outcome.cell_id.as_str(), outcome))
        .collect();

    for outcome in &manifest.outcomes {
        validate_outcome(outcome)?;

        let cell = cell_by_id
            .get(outcome.cell_id.as_str())
            .expect("outcome identity set matches cell identity set");

        if let Some(event_cycle) = outcome.event_cycle {
            if event_cycle <= cell.prediction_cutoff_cycle {
                return Err(AuditError(format!(
                    "cell {} prediction cutoff is not earlier than its target event",
                    cell.cell_id
                )));
            }
        }
    }

    if manifest.predictions.is_empty() {
        return Err(AuditError("at least one test prediction is required".into()));
    }

    unique_ids(
        manifest.predictions.iter().map(|item| item.cell_id.as_str()),
        "prediction cell ID",
    )?;

    let mut prediction_ids = BTreeSet::new();

    for prediction in &manifest.predictions {
        require_nonempty(&prediction.design_group_id, "prediction design group ID")?;
        require_nonempty(&prediction.model_evidence_ref, "model evidence reference")?;

        let cell = cell_by_id
            .get(prediction.cell_id.as_str())
            .ok_or_else(|| {
                AuditError(format!(
                    "prediction references unknown cell {}",
                    prediction.cell_id
                ))
            })?;

        if !test.contains(&prediction.cell_id) {
            return Err(AuditError(format!(
                "prediction for {} is not in the held-out test partition",
                prediction.cell_id
            )));
        }

        if prediction.design_group_id != cell.design_group_id {
            return Err(AuditError(format!(
                "prediction design group does not match cell {}",
                prediction.cell_id
            )));
        }

        if prediction.prediction_cutoff_cycle != cell.prediction_cutoff_cycle
            || prediction.prediction_cutoff_cycle > cell.feature_available_through_cycle
        {
            return Err(AuditError(format!(
                "prediction for {} uses a mismatched or future feature cutoff",
                prediction.cell_id
            )));
        }

        if !prediction.predicted_cycle_life.is_finite()
            || !prediction.lower_cycle_life.is_finite()
            || !prediction.upper_cycle_life.is_finite()
            || prediction.lower_cycle_life <= 0.0
            || prediction.lower_cycle_life > prediction.predicted_cycle_life
            || prediction.predicted_cycle_life > prediction.upper_cycle_life
        {
            return Err(AuditError(format!(
                "prediction for {} has invalid lifetime estimate or interval",
                prediction.cell_id
            )));
        }

        let outcome = outcome_by_cell
            .get(prediction.cell_id.as_str())
            .expect("outcome identity set matches cell identity set");

        if let Some(event_cycle) = outcome.event_cycle {
            if prediction.prediction_cutoff_cycle >= event_cycle {
                return Err(AuditError(format!(
                    "prediction for {} occurs at or after its event",
                    prediction.cell_id
                )));
            }
        }

        prediction_ids.insert(prediction.cell_id.clone());
    }

    if prediction_ids != test {
        return Err(AuditError(
            "each held-out test cell must have exactly one prediction".into(),
        ));
    }

    // Keep the explicit partition values live in this function: their non-empty
    // and exact coverage has already been checked by validate_split.
    let _partition_counts = (train.len(), validation.len(), test.len());

    Ok(())
}

/// Audit a finite design-level cycle-life transfer manifest.
///
/// The audit refuses cell/design leakage and future feature availability.
/// Ordinary point scores are computed only for observed cycle-life events;
/// censored and competing outcomes remain in separate counted strata.
pub fn audit_design_lifetime_manifest(
    manifest: &BenchmarkManifest,
) -> Result<AuditReport, AuditError> {
    validate_manifest(manifest)?;

    let cell_by_id: BTreeMap<&str, &CellObservation> = manifest
        .cells
        .iter()
        .map(|cell| (cell.cell_id.as_str(), cell))
        .collect();
    let outcome_by_cell: BTreeMap<&str, &CycleLifeOutcome> = manifest
        .outcomes
        .iter()
        .map(|outcome| (outcome.cell_id.as_str(), outcome))
        .collect();

    let mut event_errors = Vec::new();
    let mut group_errors: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut right_censored_cell_count = 0usize;
    let mut competing_event_cell_count = 0usize;
    let mut unknown_outcome_cell_count = 0usize;
    let mut interval_covered_count = 0usize;

    for prediction in &manifest.predictions {
        let outcome = outcome_by_cell
            .get(prediction.cell_id.as_str())
            .expect("manifest validation established outcomes");

        match outcome.kind {
            OutcomeKind::EventObserved => {
                let actual = outcome.event_cycle.expect("event cycle validated") as f64;
                let absolute_error = (prediction.predicted_cycle_life - actual).abs();
                event_errors.push((absolute_error, actual));
                group_errors
                    .entry(prediction.design_group_id.clone())
                    .or_default()
                    .push(absolute_error);

                if actual >= prediction.lower_cycle_life
                    && actual <= prediction.upper_cycle_life
                {
                    interval_covered_count += 1;
                }
            }
            OutcomeKind::RightCensored => right_censored_cell_count += 1,
            OutcomeKind::CompetingEvent => competing_event_cell_count += 1,
            OutcomeKind::Unknown => unknown_outcome_cell_count += 1,
        }
    }

    if event_errors.is_empty() {
        return Err(AuditError(
            "no event-observed test outcomes are available for ordinary point scoring".into(),
        ));
    }

    let count = event_errors.len() as f64;
    let mae_cycles_pooled = event_errors
        .iter()
        .map(|(error, _)| *error)
        .sum::<f64>()
        / count;
    let rmse_cycles_pooled = (event_errors
        .iter()
        .map(|(error, _)| error * error)
        .sum::<f64>()
        / count)
        .sqrt();
    let mean_absolute_percentage_error = event_errors
        .iter()
        .map(|(error, actual)| error / actual)
        .sum::<f64>()
        / count;

    let group_means: Vec<f64> = group_errors
        .values()
        .map(|errors| errors.iter().sum::<f64>() / errors.len() as f64)
        .collect();
    let mae_cycles_design_macro =
        group_means.iter().sum::<f64>() / group_means.len() as f64;
    let test_design_group_count = group_errors.len();

    let test_cell_ids = &manifest.split.test_cell_ids;
    let test_group_count = test_cell_ids
        .iter()
        .filter_map(|cell_id| cell_by_id.get(cell_id.as_str()))
        .map(|cell| cell.design_group_id.clone())
        .collect::<BTreeSet<_>>()
        .len();

    Ok(AuditReport {
        manifest_id: manifest.manifest_id.clone(),
        split_id: manifest.split.split_id.clone(),
        train_cell_count: manifest.split.train_cell_ids.len(),
        validation_cell_count: manifest.split.validation_cell_ids.len(),
        test_cell_count: manifest.split.test_cell_ids.len(),
        test_design_group_count: test_group_count,
        scored_prediction_count: event_errors.len(),
        score: PointScore {
            event_observed_cell_count: event_errors.len(),
            event_observed_design_group_count: test_design_group_count,
            right_censored_cell_count,
            competing_event_cell_count,
            unknown_outcome_cell_count,
            mae_cycles_pooled,
            rmse_cycles_pooled,
            mean_absolute_percentage_error,
            mae_cycles_design_macro,
            interval_covered_count,
            interval_coverage: interval_covered_count as f64 / count,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest() -> BenchmarkManifest {
        BenchmarkManifest {
            manifest_id: "SYNTH-DESIGN-LIFE".into(),
            cutoff_cycles: vec![5, 10, 20, 50],
            cells: vec![
                CellObservation {
                    cell_id: "A1".into(),
                    design_group_id: "DESIGN-A".into(),
                    protocol_group_id: "P1".into(),
                    feature_available_through_cycle: 50,
                    prediction_cutoff_cycle: 50,
                    feature_evidence_ref: "SYNTH-A1".into(),
                },
                CellObservation {
                    cell_id: "A2".into(),
                    design_group_id: "DESIGN-A".into(),
                    protocol_group_id: "P1".into(),
                    feature_available_through_cycle: 50,
                    prediction_cutoff_cycle: 50,
                    feature_evidence_ref: "SYNTH-A2".into(),
                },
                CellObservation {
                    cell_id: "B1".into(),
                    design_group_id: "DESIGN-B".into(),
                    protocol_group_id: "P2".into(),
                    feature_available_through_cycle: 50,
                    prediction_cutoff_cycle: 50,
                    feature_evidence_ref: "SYNTH-B1".into(),
                },
                CellObservation {
                    cell_id: "C1".into(),
                    design_group_id: "DESIGN-C".into(),
                    protocol_group_id: "P2".into(),
                    feature_available_through_cycle: 20,
                    prediction_cutoff_cycle: 20,
                    feature_evidence_ref: "SYNTH-C1".into(),
                },
            ],
            split: SplitManifest {
                split_id: "SYNTH-LODO".into(),
                kind: SplitKind::LeaveOneDesignGroupOut,
                train_cell_ids: vec!["A1".into(), "A2".into(), "C1".into()],
                validation_cell_ids: vec![],
                test_cell_ids: vec!["B1".into()],
                train_design_group_ids: vec!["DESIGN-A".into(), "DESIGN-C".into()],
                validation_design_group_ids: vec![],
                test_design_group_ids: vec!["DESIGN-B".into()],
            },
            predictions: vec![CycleLifePrediction {
                cell_id: "B1".into(),
                design_group_id: "DESIGN-B".into(),
                prediction_cutoff_cycle: 50,
                predicted_cycle_life: 900.0,
                lower_cycle_life: 820.0,
                upper_cycle_life: 980.0,
                model_evidence_ref: "SYNTH-MODEL".into(),
            }],
            outcomes: vec![
                CycleLifeOutcome {
                    cell_id: "A1".into(),
                    kind: OutcomeKind::EventObserved,
                    observed_through_cycle: 1100,
                    event_cycle: Some(1100),
                    event_code: None,
                },
                CycleLifeOutcome {
                    cell_id: "A2".into(),
                    kind: OutcomeKind::RightCensored,
                    observed_through_cycle: 850,
                    event_cycle: None,
                    event_code: None,
                },
                CycleLifeOutcome {
                    cell_id: "B1".into(),
                    kind: OutcomeKind::EventObserved,
                    observed_through_cycle: 950,
                    event_cycle: Some(950),
                    event_code: None,
                },
                CycleLifeOutcome {
                    cell_id: "C1".into(),
                    kind: OutcomeKind::CompetingEvent,
                    observed_through_cycle: 700,
                    event_cycle: Some(700),
                    event_code: Some("SYNTH-COMPETING-MODE".into()),
                },
            ],
        }
    }

    #[test]
    fn accepts_leave_one_design_group_out_manifest() {
        let report = audit_design_lifetime_manifest(&valid_manifest()).expect("valid manifest");
        assert_eq!(report.train_cell_count, 3);
        assert_eq!(report.test_cell_count, 1);
        assert_eq!(report.test_design_group_count, 1);
        assert_eq!(report.scored_prediction_count, 1);
        assert_eq!(report.score.event_observed_cell_count, 1);
        assert_eq!(report.score.right_censored_cell_count, 0);
    }

    #[test]
    fn rejects_cell_identity_leakage_between_splits() {
        let mut manifest = valid_manifest();
        manifest.split.train_cell_ids.push("B1".into());

        let error = audit_design_lifetime_manifest(&manifest).expect_err("leakage fails");
        assert!(error.0.contains("duplicate train cell ID") || error.0.contains("leaks"));
    }

    #[test]
    fn rejects_design_group_leakage() {
        let mut manifest = valid_manifest();
        manifest.split.train_cell_ids.push("B1".into());
        manifest.split.train_design_group_ids.push("DESIGN-B".into());
        manifest.split.test_cell_ids.clear();
        manifest.split.test_design_group_ids.clear();

        let error = audit_design_lifetime_manifest(&manifest).expect_err("design leakage fails");
        assert!(error.0.contains("leaks across"));
    }

    #[test]
    fn rejects_features_unavailable_at_prediction_cutoff() {
        let mut manifest = valid_manifest();
        manifest.cells[2].feature_available_through_cycle = 20;

        let error =
            audit_design_lifetime_manifest(&manifest).expect_err("future features fail");
        assert!(error.0.contains("unavailable at its prediction cutoff"));
    }

    #[test]
    fn rejects_undeclared_prediction_cutoff() {
        let mut manifest = valid_manifest();
        manifest.cells[2].prediction_cutoff_cycle = 30;
        manifest.cells[2].feature_available_through_cycle = 50;
        manifest.predictions[0].prediction_cutoff_cycle = 30;

        let error =
            audit_design_lifetime_manifest(&manifest).expect_err("undeclared cutoff fails");
        assert!(error.0.contains("undeclared prediction cutoff"));
    }

    #[test]
    fn rejects_prediction_at_or_after_observed_event() {
        let mut manifest = valid_manifest();
        manifest.cells[2].prediction_cutoff_cycle = 950;
        manifest.cells[2].feature_available_through_cycle = 950;
        manifest.predictions[0].prediction_cutoff_cycle = 950;

        let error =
            audit_design_lifetime_manifest(&manifest).expect_err("post-event prediction fails");
        assert!(error.0.contains("not earlier than its target event"));
    }

    #[test]
    fn censored_outcomes_are_not_scored_as_point_lifetimes() {
        let mut manifest = valid_manifest();
        manifest.split.test_cell_ids = vec!["A2".into()];
        manifest.split.train_cell_ids = vec!["A1".into(), "B1".into(), "C1".into()];
        manifest.split.train_design_group_ids =
            vec!["DESIGN-A".into(), "DESIGN-B".into(), "DESIGN-C".into()];
        manifest.split.test_design_group_ids = vec!["DESIGN-A".into()];
        manifest.predictions[0].cell_id = "A2".into();
        manifest.predictions[0].design_group_id = "DESIGN-A".into();
        manifest.predictions[0].prediction_cutoff_cycle = 50;

        let error =
            audit_design_lifetime_manifest(&manifest).expect_err("no event point score");
        assert!(error.0.contains("no event-observed test outcomes"));
    }

    #[test]
    fn right_censored_outcome_cannot_carry_a_pseudo_event() {
        let mut manifest = valid_manifest();
        manifest.outcomes[1].event_cycle = Some(850);

        let error =
            audit_design_lifetime_manifest(&manifest).expect_err("censor must remain censor");
        assert!(error.0.contains("must not be converted into an event"));
    }

    #[test]
    fn competing_event_requires_explicit_code() {
        let mut manifest = valid_manifest();
        manifest.outcomes[3].event_code = None;

        let error =
            audit_design_lifetime_manifest(&manifest).expect_err("competing code required");
        assert!(error.0.contains("competing event code"));
    }

    #[test]
    fn each_test_cell_must_have_exactly_one_prediction() {
        let mut manifest = valid_manifest();
        manifest.predictions.clear();

        let error =
            audit_design_lifetime_manifest(&manifest).expect_err("missing prediction fails");
        assert!(error.0.contains("at least one test prediction"));
    }

    #[test]
    fn score_is_invariant_to_prediction_and_outcome_input_order() {
        let mut manifest = valid_manifest();
        // Add a second held-out cell in the same design family with an observed
        // endpoint, demonstrating design-macro rather than per-cell weighting.
        manifest.cells.push(CellObservation {
            cell_id: "B2".into(),
            design_group_id: "DESIGN-B".into(),
            protocol_group_id: "P2".into(),
            feature_available_through_cycle: 50,
            prediction_cutoff_cycle: 50,
            feature_evidence_ref: "SYNTH-B2".into(),
        });
        manifest.split.test_cell_ids.push("B2".into());
        manifest.predictions.push(CycleLifePrediction {
            cell_id: "B2".into(),
            design_group_id: "DESIGN-B".into(),
            prediction_cutoff_cycle: 50,
            predicted_cycle_life: 1000.0,
            lower_cycle_life: 900.0,
            upper_cycle_life: 1100.0,
            model_evidence_ref: "SYNTH-MODEL".into(),
        });
        manifest.outcomes.push(CycleLifeOutcome {
            cell_id: "B2".into(),
            kind: OutcomeKind::EventObserved,
            observed_through_cycle: 1050,
            event_cycle: Some(1050),
            event_code: None,
        });

        let first = audit_design_lifetime_manifest(&manifest).expect("first order");
        manifest.predictions.reverse();
        manifest.outcomes.reverse();
        let second = audit_design_lifetime_manifest(&manifest).expect("reversed order");

        assert_eq!(first, second);
        assert_eq!(first.score.event_observed_design_group_count, 1);
    }
}
