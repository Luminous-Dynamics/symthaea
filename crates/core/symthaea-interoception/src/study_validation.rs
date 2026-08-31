use crate::{
    extract_study_blinded_metrics, ConfirmatoryHypothesisEvaluation, EvidenceRunClass,
    ExclusionDecisionReceipt, ExecutionLimits, StudyBlindedMetricReport, StudyExecutionTrace,
    StudyPreregistration, STUDY_BLINDED_METRIC_SCHEMA_VERSION,
};

/// Validate that a study-level blinded metric report is exactly reproducible from
/// the locked study, exact execution trace, and exclusion-decision receipt.
pub fn validate_study_blinded_metrics(
    study: &StudyPreregistration,
    execution: &StudyExecutionTrace,
    exclusions: &ExclusionDecisionReceipt,
    blinded: &StudyBlindedMetricReport,
    limits: ExecutionLimits,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    if let Err(study_errors) = study.validate() {
        errors.extend(study_errors);
    }
    if let Err(execution_errors) = execution.validate_against(study, limits) {
        errors.extend(
            execution_errors
                .into_iter()
                .map(|error| format!("study execution: {error}")),
        );
    }
    if let Err(exclusion_errors) = exclusions.validate_against(study, execution, limits) {
        errors.extend(
            exclusion_errors
                .into_iter()
                .map(|error| format!("exclusion receipt: {error}")),
        );
    }

    if blinded.schema_version != STUDY_BLINDED_METRIC_SCHEMA_VERSION {
        errors.push(format!(
            "study blinded metric schema version mismatch: {}",
            blinded.schema_version
        ));
    }
    if blinded.run_class != study.run_class {
        errors.push("study blinded metric run class does not match preregistration".into());
    }

    match study.sha256() {
        Ok(expected) if expected == blinded.study_preregistration_sha256 => {}
        Ok(_) => errors.push("study blinded metric preregistration digest mismatch".into()),
        Err(study_errors) => errors.extend(study_errors),
    }
    match exclusions.sha256() {
        Ok(expected) if expected == blinded.exclusion_decision_sha256 => {}
        Ok(_) => errors.push("study blinded metric exclusion digest mismatch".into()),
        Err(error) => errors.push(format!("failed to hash exclusion receipt: {error}")),
    }

    match exclusions.disposition_against(study, execution, limits) {
        Ok(expected) if expected == blinded.disposition => {}
        Ok(_) => errors.push("study blinded metric disposition does not match exclusion receipt".into()),
        Err(disposition_errors) => errors.extend(disposition_errors),
    }

    match extract_study_blinded_metrics(study, execution, exclusions, limits) {
        Ok(expected) if &expected == blinded => {}
        Ok(_) => errors.push(
            "study blinded metric report does not exactly reproduce from locked evidence".into(),
        ),
        Err(recompute_errors) => errors.extend(
            recompute_errors
                .into_iter()
                .map(|error| format!("blinded recomputation: {error}")),
        ),
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Qualified confirmatory evaluation path.
///
/// This function revalidates and recomputes the blinded metric artifact from the
/// exact execution and exclusion receipt before semantic-arm unblinding. The
/// lower-level evaluator remains crate-internal and is not exported publicly.
pub fn evaluate_confirmatory_study_bound(
    study: &StudyPreregistration,
    execution: &StudyExecutionTrace,
    exclusions: &ExclusionDecisionReceipt,
    blinded: &StudyBlindedMetricReport,
    limits: ExecutionLimits,
) -> Result<ConfirmatoryHypothesisEvaluation, Vec<String>> {
    validate_study_blinded_metrics(study, execution, exclusions, blinded, limits)?;
    if study.run_class != EvidenceRunClass::Confirmatory {
        return Err(vec![
            "exploratory studies cannot be promoted through confirmatory evaluation".into(),
        ]);
    }
    crate::study::evaluate_confirmatory_study(study, blinded)
}
